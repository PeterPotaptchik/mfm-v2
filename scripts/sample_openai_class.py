import json
import tqdm
import os
import hydra
from argparse import Namespace
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
from torch_ema import ExponentialMovingAverage
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from cleanfid import fid

from distcfm.data import get_data_module
from distcfm.utils.evaluation import load_model, set_seed
from distcfm.sde.score_fn import get_unconditional_score_fn, get_dps_score_fn, get_iwae_score_fn, get_mc_score_fn
from distcfm.utils.classifiers import load_classifier
from distcfm.sde.samplers import EulerMaruyamaSampler
from distcfm.models.model_wrapper import SDEModelWrapper
from build_fid_metrics import build_real_stats_per_class

from distcfm.utils import get_diffusion
from datetime import datetime

CIFAR_FID_CACHE = "cache"
LINEAR_CIFAR = "/vols/bitbucket/saravanan/distributional-mf/ckpts/openai_cifar10_linear.pt"
COSINE_CIFAR = "/vols/bitbucket/saravanan/distributional-mf/ckpts/openai_cifar10_cosine.pt"

def plot_samples(samples, labels, classifier_label, save_dir):
    if samples.shape[1] == 1:
         cmap = "gray"
    else:
         cmap = None
    n = samples.shape[0]
    f, axs = plt.subplots(1, n, figsize=(15, 5))
    for i, ax in enumerate(axs):
        ax.imshow(samples[i].cpu().permute(1, 2, 0).numpy(), cmap=cmap)
        ax.set_title(f"Task Label: {labels[i]}\n Classifier: {classifier_label[i]}")
        ax.axis("off")
    plt.savefig(os.path.join(save_dir, "example_generations.png"))
    plt.close()

@hydra.main(config_path="../conf/sample/", config_name="config_class_openai.yaml", version_base="1.3")
def main(cfg: DictConfig):
    # set seed
    set_seed(cfg.seed)

    dist_cfg = OmegaConf.load(cfg.dist_config)  
    print("sampling with noise_schedule:", dist_cfg.sde.noise_schedule)

    CIFAR_ARGS = Namespace(
        image_size=32,
        dataset="cifar10",
        model_name_or_path=LINEAR_CIFAR if dist_cfg.sde.noise_schedule == "linear" else COSINE_CIFAR,
        train_steps=1000,
        inference_steps=1000,
        noise_schedule=dist_cfg.sde.noise_schedule,
        device="cuda"
    )
    
    # load parameters of the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sde = instantiate(dist_cfg.sde)

    # load unconditional model
    unet, _, _, _ = get_diffusion(CIFAR_ARGS)
    rescale_time_fn = lambda t: (t * (CIFAR_ARGS.train_steps - 1)).round().long()
    uncond_model_fn = lambda x, t: unet(x, rescale_time_fn(t))
    unconditional_score_fn = get_unconditional_score_fn(uncond_model_fn, sde)
    
    # load consistency model
    dist_model = instantiate(dist_cfg.model)
    dist_model.to(device)
    dist_checkpoint = torch.load(cfg.dist_checkpoint_dir, 
                            map_location=device)
    unwrapped_dist_model = load_model(dist_model, dist_checkpoint)
    dist_model = SDEModelWrapper(unwrapped_dist_model, sde, False)

    if cfg.use_ema:
        dist_ema = ExponentialMovingAverage(dist_model.parameters(), 
                                       decay=dist_cfg.trainer.ema.decay)
        dist_ema.load_state_dict(dist_checkpoint['ema_state_dict'])
        dist_ema.to(device)
        dist_ema.store(dist_model.parameters())
        dist_ema.copy_to(dist_model.parameters())

    # data module for conditioning on noisy samples
    datamodule = get_data_module(dist_cfg)
    datamodule.setup(stage="test")

    # get sampler
    sampler = EulerMaruyamaSampler(sde, num_steps=cfg.sampling.euler_steps, t_eps=cfg.sampling.t_eps)

    # used for guidance
    model_fn, classifier_fn = load_classifier(cfg.dataset, 
                                              cfg.guidance_classifier,
                                              device)
    # used for evaluation
    model_eval_fn, _ = load_classifier(cfg.dataset, 
                                       cfg.eval_classifier, 
                                       device)
    
    # get the conditional score fn
    if cfg.conditional_score == "iwae":
        if cfg.iwae.type == "cfm_combined":
            assert cfg.unconditional_score == "mc_unconditional", "cfm_combined returns mc_unconditional as part of it's function"
        
        conditional_score_fn = get_iwae_score_fn(dist_model,
                                                 classifier_fn,
                                                 datamodule.inverse_scaler,
                                                 mc_samples=cfg.iwae.n_samples,
                                                 type=cfg.iwae.type,
                                                 sde=sde)
        save_dir = cfg.dist_dir
    elif cfg.conditional_score == "mc":
        conditional_score_fn = get_mc_score_fn(dist_model,
                                                 classifier_fn,
                                                 datamodule.inverse_scaler,
                                                 sde,
                                                 mc_samples=cfg.mc.n_samples,
                                                 type="cfm")
        save_dir = cfg.dist_dir
    elif cfg.conditional_score == "dps":
        conditional_score_fn = get_dps_score_fn(uncond_model_fn,
                                                sde,
                                                classifier_fn,
                                                datamodule.inverse_scaler,)
        save_dir = f"./outputs/{cfg.dataset}"
    else:
        raise ValueError(f"Unknown conditional score function: {cfg.conditional_score}")
    
    samples = []
    labels = []
    for label in cfg.labels_to_sample:
        l = torch.full((cfg.samples_per_label,), label, dtype=torch.long)
        labels.append(l)

    labels = torch.cat(labels, dim=0)
    labels = labels.to(device)

    gen_class_labels = []
    oracle_class_labels = []

    for i in tqdm.tqdm(range(0, labels.shape[0], cfg.sampling.batch_size), desc="Sampling"):
        label_batch = labels[i:i + cfg.sampling.batch_size]
        
        if cfg.unconditional_score != "mc_unconditional":
            joint_score_fn = lambda x, t: unconditional_score_fn(x, t) + conditional_score_fn(x, t, label_batch)
        else:
            joint_score_fn = lambda x, t: conditional_score_fn(x, t, label_batch)

        samples_batch = sampler.sample((label_batch.shape[0], *datamodule.dims),
                                        joint_score_fn,
                                        inverse_scalar_fn=datamodule.inverse_scaler,
                                        grads=True,
                                        clip=cfg.sampling.clip_each_step,)
        samples.append(samples_batch)
        with torch.inference_mode():
            gen_logits = model_fn(samples_batch) # [B, N]
            gen_probs = gen_logits.softmax(dim=-1)    # [B, N]
            gen_preds = gen_probs.argmax(dim=-1)      # [B]
            gen_class_labels.append(gen_preds)

            oracle_logits = model_eval_fn(samples_batch) # [B, N]
            oracle_probs = oracle_logits.softmax(dim=-1)    # [B, N]
            oracle_preds = oracle_probs.argmax(dim=-1)      # [B]
            oracle_class_labels.append(oracle_preds)

    gen_class_labels = torch.cat(gen_class_labels, dim=0)
    oracle_class_labels = torch.cat(oracle_class_labels, dim=0)

    gen_acc = (gen_class_labels == labels).float().mean().item()
    print("Generative classifier accuracy:", gen_acc)
    oracle_acc = (oracle_class_labels == labels).float().mean().item()
    print("Oracle classifier accuracy:", oracle_acc)

    samples = torch.cat(samples, dim=0)
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    samples_dir = os.path.join(save_dir, f"class_conditional_{time_str}")
    # make samples dir depend on time
    os.makedirs(samples_dir, exist_ok=True)

    if cfg.conditional_score == "dps":
        save_path = f"dps_{cfg.sampling.euler_steps}_{cfg.sampling.clip_each_step}"
    elif cfg.conditional_score == "iwae":
        save_path = f"iwae_{cfg.iwae.type}_{cfg.sampling.euler_steps}_{cfg.sampling.clip_each_step}_{cfg.iwae.n_samples}"
    elif cfg.conditional_score == "mc":
        save_path = f"mc_{cfg.sampling.euler_steps}_{cfg.sampling.clip_each_step}_{cfg.mc.n_samples}"

    samples_dir = os.path.join(samples_dir, save_path)
    print("saving to:", samples_dir)
    os.makedirs(samples_dir, exist_ok=True)

    # save/log
    torch.save(samples, os.path.join(samples_dir, "samples.pt"))
    torch.save(labels, os.path.join(samples_dir, "labels.pt"))
    torch.save(gen_class_labels, os.path.join(samples_dir, "gen_labels.pt"))
    torch.save(oracle_class_labels, os.path.join(samples_dir, "oracle_labels.pt"))

    # plot some example_samples: random shuffle
    idxs = torch.randperm(samples.shape[0])[:16]
    plot_samples(samples[idxs], labels[idxs], gen_class_labels[idxs], samples_dir)
    
    with open(os.path.join(samples_dir, "accuracy.txt"), "w") as f:
        f.write(f"Accuracy: {gen_acc}\n")
        f.write(f"Oracle Accuracy: {oracle_acc}\n")

    # compute class-conditional FID
    if cfg.dataset == "cifar10": 
        stats_names = build_real_stats_per_class(CIFAR_FID_CACHE)
        per_class_fid = {}
        fid_path = os.path.join(samples_dir, "fid")

        os.makedirs(fid_path, exist_ok=True)
        classes = torch.unique(labels).tolist()

        for c in classes:
            cdir = os.path.join(fid_path, f"gen_c{c}")
            os.makedirs(cdir, exist_ok=True)
            idx = (labels == c).nonzero(as_tuple=False).squeeze(1)
            samples_cls = samples[idx]
            for i, img in enumerate(samples_cls):
                save_image(img.clamp(0,1), os.path.join(cdir, f"{i:06d}.png"))

            fid_score = fid.compute_fid(cdir,
                                        dataset_name=stats_names[c],
                                        dataset_split="custom",
                                        mode="clean",
                                        dataset_res=32)
            per_class_fid[c] = fid_score
            
        with open(os.path.join(samples_dir, "per_class_fid.json"), "w") as f:
            json.dump(per_class_fid, f, indent=4)
            

if __name__ == "__main__":
    main()
