import tqdm
import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
from torch_ema import ExponentialMovingAverage
import matplotlib.pyplot as plt

from distcfm.data import get_data_module
from distcfm.utils.evaluation import load_model, set_seed 
from distcfm.sde.score_fn import get_unconditional_score_fn, get_dps_score_fn, get_iwae_score_fn, get_mc_score_fn
from distcfm.utils.classifiers import load_classifier
from distcfm.sde.samplers import EulerMaruyamaSampler
from distcfm.models.model_wrapper import SDEModelWrapper, DenoiserWrapper
from distcfm.utils import get_diffusion


CIFAR_ARGS = Namespace(
    image_size=32,
    dataset="cifar10",
    model_name_or_path="/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/saved_models/openai_cifar10.pt",
    train_steps=1000,
    inference_steps=1000,
    noise_schedule="cosine", 
    device="cuda"
)

def plot_samples(samples, oracle_class_label, gen_class_label, save_dir):
    n = samples.shape[0]
    f, axs = plt.subplots(1, n, figsize=(15, 5))
    for i, ax in enumerate(axs):
        ax.imshow(samples[i].cpu().permute(1, 2, 0).numpy(), cmap="gray")
        ax.set_title(f"Oracle: {oracle_class_label[i]}\n Gen: {gen_class_label[i]}", fontdict={'fontsize': 6})
        ax.axis("off")
    plt.savefig(os.path.join(save_dir, "example_generations.png"))
    plt.close()

@hydra.main(config_path="../conf/sample/", config_name="config_multiclass.yaml", version_base="1.3")
def main(cfg: DictConfig):
    # set seed
    set_seed(cfg.seed)
    
    uncond_cfg = OmegaConf.load(cfg.uncond_config)
    dist_cfg = OmegaConf.load(cfg.dist_config)  

    # load parameters of the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sde = instantiate(uncond_cfg.sde)

    # load unconditional model
    unet, _, _, _ = get_diffusion(CIFAR_ARGS)
    rescale_time_fn = lambda t: (t * 1000).round().long()
    uncond_model_fn = lambda x, t: unet(x, rescale_time_fn(t))
    unconditional_score_fn = get_unconditional_score_fn(uncond_model_fn, sde)
    
    # load consistency model
    dist_model = instantiate(dist_cfg.model)
    dist_model.to(device)
    dist_checkpoint = torch.load(cfg.dist_checkpoint_dir, 
                            map_location=device)
    unwrapped_dist_model = load_model(dist_model, dist_checkpoint)
    dist_model = SDEModelWrapper(unwrapped_dist_model, sde, False)

    if cfg.use_ema == True:
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
    sampler = EulerMaruyamaSampler(sde, num_steps=cfg.sampling.euler_steps,)
    
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
        conditional_score_fn = get_dps_score_fn(uncond_model_fn,
                                                sde,
                                                classifier_fn,
                                                datamodule.inverse_scaler,)
        save_dir = cfg.uncond_dir
    else:
        raise ValueError(f"Unknown conditional score function: {cfg.conditional_score}")
    
    samples = []
    labels = []

    ws = torch.tensor(cfg.weights_to_sample)
    ws = ws / ws.sum()
    labels = ws.unsqueeze(0).expand(cfg.total_samples, -1).to(device)

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

    num_classes = labels.shape[1]
    gen_ratios = torch.bincount(gen_class_labels, minlength=num_classes) / len(gen_class_labels)
    gen_l1 = (gen_ratios.cpu() - ws.cpu()).abs().sum()
    print("Generative l1 accuracy:", gen_l1)

    oracle_ratios = torch.bincount(oracle_class_labels, minlength=num_classes) / len(oracle_class_labels)
    oracle_l1 = (oracle_ratios.cpu() - ws.cpu()).abs().sum()
    print("Oracle l1 accuracy:", oracle_l1)

    samples = torch.cat(samples, dim=0)
    samples_dir = os.path.join(save_dir, "multiclass_conditional")
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
    plot_samples(samples[idxs], oracle_class_labels[idxs], gen_class_labels[idxs], samples_dir)
    
    with open(os.path.join(samples_dir, "accuracy.txt"), "w") as f:
        f.write(f"Gen l1: {gen_l1}\n")
        f.write(f"Oracle l1: {oracle_l1}\n")
        f.write(f"Gen ratios: {gen_ratios.detach().cpu().tolist()}\n")
        f.write(f"Oracle ratios: {oracle_ratios.detach().cpu().tolist()}\n")

if __name__ == "__main__":
    main()