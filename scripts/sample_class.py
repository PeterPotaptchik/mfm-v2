import json
import tqdm
import os
import hydra
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
from distcfm.models.model_wrapper import SDEModelWrapper, DenoiserWrapper
from build_fid_metrics import build_real_stats_per_class

CIFAR_FID_CACHE = "cache"

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

@hydra.main(config_path="../conf/sample/", config_name="config_class.yaml", version_base="1.3")
def main(cfg: DictConfig):
    # set seed
    set_seed(cfg.seed)
    
    uncond_cfg = OmegaConf.load(cfg.uncond_config)
    dist_cfg = OmegaConf.load(cfg.dist_config)  

    # load parameters of the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sde = instantiate(uncond_cfg.sde)

    # load unconditional model
    uncond_model = instantiate(uncond_cfg.model)
    uncond_model.to(device)
    uncond_checkpoint = torch.load(cfg.uncond_checkpoint_dir, 
                            map_location=device)
    uncond_model = load_model(uncond_model, uncond_checkpoint)
    uncond_model = DenoiserWrapper(uncond_model, sde, 
                                   uncond_cfg.use_snr_parametrisation)
    
    # load consistency model
    dist_model = instantiate(dist_cfg.model)
    dist_model.to(device)
    dist_checkpoint = torch.load(cfg.dist_checkpoint_dir, 
                            map_location=device)
    unwrapped_dist_model = load_model(dist_model, dist_checkpoint)
    dist_model = SDEModelWrapper(unwrapped_dist_model, sde, False)

    print(cfg.use_ema)
    if not cfg.use_ema == True:
        # load uncond
        ema = ExponentialMovingAverage(uncond_model.parameters(), 
                                       decay=uncond_cfg.trainer.ema.decay)
        ema.load_state_dict(uncond_checkpoint['ema_state_dict'])
        ema.to(device)
        ema.store(uncond_model.parameters())
        ema.copy_to(uncond_model.parameters())
        # load dist
        dist_ema = ExponentialMovingAverage(dist_model.parameters(), 
                                       decay=dist_cfg.trainer.ema.decay)
        dist_ema.load_state_dict(dist_checkpoint['ema_state_dict'])
        dist_ema.to(device)
        dist_ema.store(dist_model.parameters())
        dist_ema.copy_to(dist_model.parameters())

    # data module for conditioning on noisy samples
    datamodule = get_data_module(uncond_cfg)
    datamodule.setup(stage="test")

    # get sampler
    sampler = EulerMaruyamaSampler(sde, num_steps=cfg.sampling.euler_steps, t_eps=cfg.sampling.t_eps)
    # get the unconditional score_fn
    unconditional_score_fn = get_unconditional_score_fn(uncond_model, sde)
    
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
        conditional_score_fn = get_iwae_score_fn(dist_model,
                                                 classifier_fn,
                                                 datamodule.inverse_scaler,
                                                 mc_samples=cfg.iwae.n_samples,
                                                 type="cfm")
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
        conditional_score_fn = get_dps_score_fn(uncond_model,
                                                sde,
                                                classifier_fn,
                                                datamodule.inverse_scaler,)
        save_dir = cfg.uncond_dir
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
        joint_score_fn = lambda x, t: unconditional_score_fn(x, t) + conditional_score_fn(x, t, label_batch)
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
    samples_dir = os.path.join(save_dir, "class_conditional")
    os.makedirs(samples_dir, exist_ok=True)

    if cfg.conditional_score == "dps":
        save_path = f"dps_{cfg.sampling.euler_steps}_{cfg.sampling.clip_each_step}"
    elif cfg.conditional_score == "iwae":
        save_path = f"iwae_{cfg.sampling.euler_steps}_{cfg.sampling.clip_each_step}_{cfg.iwae.n_samples}"
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
