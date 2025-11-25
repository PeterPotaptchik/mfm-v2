import os
import math
import gc
import tqdm 

import torch
import lightning as pl
from lightning.pytorch.callbacks import Callback
import torchvision
import wandb
import matplotlib.pyplot as plt
from diffusers import AutoencoderKL

from distcfm.utils.evaluation import posterior_sampling_fn, plot_posterior_samples
from distcfm.utils.repa_utils import RepaModel

def broadcast_to_shape(tensor, shape):
    return tensor.view(-1, *((1,) * (len(shape) - 1)))

class TrainingModule(pl.LightningModule):
    def __init__(self, cfg, model, loss_fn, SI):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.SI = SI
        
        # Use a list to avoid registering repa_model as a submodule
        # This prevents it from being part of state_dict, so we don't need to load/save it
        self._repa_model_container = [RepaModel(cfg)]

        # Initialize VAE if using raw ImageNet images
        self._vae_container = []
        if cfg.dataset.name == "imagenet":
            print("Initializing VAE for raw ImageNet training...")
            vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
            vae.eval()
            for p in vae.parameters():
                p.requires_grad = False
            self._vae_container.append(vae)
            self.register_buffer('latents_scale', torch.tensor([0.18215, 0.18215, 0.18215, 0.18215]).view(1, 4, 1, 1))
            self.register_buffer('latents_bias', torch.tensor([0., 0., 0., 0.]).view(1, 4, 1, 1))

        if self.cfg.model.label_dim > 0 and self.cfg.trainer.class_dropout_prob > 0:
            self.register_buffer("null_class_token", torch.tensor([self.cfg.model.label_dim]))

    @property
    def repa_model(self):
        return self._repa_model_container[0]

    @property
    def vae(self):
        return self._vae_container[0] if self._vae_container else None

    def forward(self, x):
        return self.model(x)

    def setup(self, stage: str):
        if self.vae:
            self.vae.to(self.device)
        return

    def on_train_start(self):
        # Ensure repa_model is on the correct device
        self.repa_model.to(self.device)
        if self.vae:
            self.vae.to(self.device)

    def on_validation_start(self):
        # Ensure repa_model is on the correct device
        self.repa_model.to(self.device)
        if self.vae:
            self.vae.to(self.device)

    def on_load_checkpoint(self, checkpoint):
        """
        Handle checkpoint loading for backward compatibility.
        
        Old checkpoints are missing:
        1. The new proj_head weights (for REPA)
        2. Optimizer state (if saved with save_weights_only=True)
        
        We load model weights with strict=False to allow missing proj_head,
        which will be randomly initialized.
        """
        # Load model state with strict=False to allow missing keys (proj_head)
        state_dict = checkpoint.get("state_dict", {})
        
        # Check if proj_head is missing (old checkpoint)
        model_keys = set(self.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        missing_keys = model_keys - ckpt_keys
        
        proj_head_missing = any("proj_head" in k for k in missing_keys)
        
        if proj_head_missing:
            print(f"\n{'='*60}")
            print(f"Loading checkpoint with missing REPA projection head")
            print(f"  proj_head will be randomly initialized")
            print(f"{'='*60}\n")

    def training_step(self, batch, batch_idx):
        try:
            x, labels = batch
        except:
            x = batch      
            labels = None 
        
        if self.global_step == 0 and batch_idx == 0:
            print(f"DEBUG: Input batch shape (raw images): {x.shape}")

        step = self.global_step

        # Encode images if VAE is present
        repa_input = None
        if self.vae:
            repa_input = x # Raw images for REPA
            with torch.no_grad():
                # x is [-1, 1]
                # Cast x to the same dtype as VAE
                # Disable autocast to ensure VAE runs in float32
                with torch.cuda.amp.autocast(enabled=False):
                    x_vae = x.to(dtype=self.vae.dtype)
                    latents = self.vae.encode(x_vae).latent_dist.sample()
                    latents = (latents - self.latents_bias) * self.latents_scale
                    x = latents.to(dtype=x.dtype) # Cast back to original dtype (likely bf16)

        if self.cfg.model.label_dim > 0 and self.cfg.trainer.class_dropout_prob > 0:
            prob = self.cfg.trainer.class_dropout_prob
            mask = torch.bernoulli(torch.full(labels.shape, 1 - prob, device=self.device)).bool()
            labels = torch.where(mask, labels, self.null_class_token.expand_as(labels))

        losses, aux_losses = self.loss_fn(self.model, x, labels, step, repa_model=self.repa_model, repa_input=repa_input)
        for name, loss in losses.items():
            self.log(f"train/{name}", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        for name, loss in aux_losses.items():
            self.log(f"train/{name}", loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        total_loss = 0

        for name, loss in losses.items():
            if name == "distillation_loss":
                total_loss += loss * self.cfg.loss.distillation_weight
            elif name == "repa_loss":
                total_loss += loss * self.cfg.loss.repa_weight
            else:
                total_loss += loss

        self.log("train/total_loss", total_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("train/lr", current_lr, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        gate_stats_fn = getattr(self.model, "pop_gate_stats", None)
        if callable(gate_stats_fn):
            gate_stats = gate_stats_fn()
            if gate_stats is not None:
                for name, value in gate_stats.items():
                    self.log(f"train/{name}", value, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        weighting_stats_fn = getattr(self.model, "pop_weighting_stats", None)
        if callable(weighting_stats_fn):
            weighting_stats = weighting_stats_fn()
            if weighting_stats is not None:
                for name, value in weighting_stats.items():
                    self.log(f"train/{name}", value, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return total_loss

    def on_before_optimizer_step(self, optimizer,):
        if self.global_step % 10 == 0:
            total_norm = torch.norm(torch.stack([
                p.grad.detach().norm(2)
                for p in self.parameters() if p.grad is not None
            ]))
            self.log("grad_l2_norm", total_norm, on_step=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        """Run validation with EMA parameters"""
        try:
            x, labels = batch
        except:
            x = batch
            labels = None 
        
        # Encode images if VAE is present
        if self.vae:
            with torch.no_grad():
                # Disable autocast to ensure VAE runs in float32
                with torch.cuda.amp.autocast(enabled=False):
                    x_vae = x.to(dtype=self.vae.dtype)
                    latents = self.vae.encode(x_vae).latent_dist.sample()
                    latents = (latents - self.latents_bias) * self.latents_scale
                    x = latents.to(dtype=x.dtype)

        if self.cfg.model.label_dim > 0 and self.cfg.trainer.class_dropout_prob > 0:
            prob = self.cfg.trainer.class_dropout_prob
            mask = torch.bernoulli(torch.full(labels.shape, 1 - prob, device=self.device)).bool()
            labels = torch.where(mask, labels, self.null_class_token.expand_as(labels))

        ema_val_loss, ema_val_aux_losses = self.loss_fn(self.model, x, labels, self.global_step)
        for name, loss in ema_val_loss.items():
            self.log(f"val_ema/{name}", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        for name, loss in ema_val_aux_losses.items():
            self.log(f"val_ema/{name}", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        ema_total_loss = sum(ema_val_loss.values())

        self.log("val_ema/total_loss", ema_total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return ema_total_loss
            
    def configure_optimizers(self):
        # Get only trainable parameters (excludes repa_model which is in a list)
        params = [p for p in self.parameters() if p.requires_grad]
        if self.cfg.optimizer == "RAdam":
            optimizer = torch.optim.RAdam(params, lr=self.cfg.lr.val, weight_decay=self.cfg.get("weight_decay", 0.0))
        else:
            optimizer = torch.optim.Adam(params, lr=self.cfg.lr.val, weight_decay=self.cfg.get("weight_decay", 0.0))


        
        if self.cfg.lr.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=max(1, self.cfg.trainer.num_train_steps - self.cfg.lr.warmup_steps),
                eta_min=self.cfg.lr.min_lr
            )
        elif self.cfg.lr.scheduler == "constant":
            scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, 
                factor=1.0, 
                total_iters=float('inf')
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.cfg.lr.scheduler}")
        
        if self.cfg.lr.warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.cfg.lr.warmup_steps
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[self.cfg.lr.warmup_steps]
            )
    
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }


from distcfm.SI.samplers import kernel_sampler_fn

class SamplingCallback(Callback):
    """Callback for generating and saving samples during training"""
    def __init__(self, cfg, test_data, inverse_scaler, SI):
        super().__init__()
        self.cfg = cfg
        self.test_data = test_data
        self.inverse_scaler = inverse_scaler
        self.SI = SI
        # Use model input shape (latent shape) for sampling
        self.image_shape = (
            self.cfg.model.in_channels, 
            self.cfg.model.input_size, 
            self.cfg.model.input_size
        )
        # for generating x_t
        self.shared_noise = torch.randn(
            self.cfg.sampling.n_conditioning_samples, 
            *self.image_shape
        )
        # for sampling from the posterior
        self.shared_posterior_noise = torch.randn(self.cfg.sampling.n_samples_per_image*self.cfg.sampling.n_conditioning_samples,
                                                  *self.image_shape)
        self._last_sampled_step = -1 

    def _decode_if_needed(self, pl_module, samples):
        if pl_module.vae:
            # samples are scaled latents
            latents = samples
            
            # Handle 5D input [B, N, C, H, W]
            is_5d = latents.ndim == 5
            if is_5d:
                B, N, C, H, W = latents.shape
                latents = latents.view(B * N, C, H, W)

            latents = latents / pl_module.latents_scale + pl_module.latents_bias
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=False):
                    latents = latents.to(dtype=pl_module.vae.dtype)
                    images = pl_module.vae.decode(latents).sample
            
            if is_5d:
                images = images.view(B, N, *images.shape[1:])

            # images are [-1, 1]
            return self.inverse_scaler(images) # [0, 1]
        else:
            # samples are images [-1, 1]
            return self.inverse_scaler(samples) # [0, 1]
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero:
            return
        if (trainer.global_step + 1) % self.cfg.sampling.every_n_steps == 0 and (trainer.global_step != self._last_sampled_step):
            self._last_sampled_step = trainer.global_step
            was_training = pl_module.training
            pl_module.eval()
            
            # Use identity scaler for sampling, decode later
            inverse_scaler_for_sampler = lambda x: x

            for n_steps in self.cfg.sampling.n_kernel_steps:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    unconditional_samples_kernel = kernel_sampler_fn(
                        pl_module.model,
                        shape=self.image_shape,
                        shape_decoded=self.image_shape, # Return latents if VAE
                        SI=self.SI,
                        n_samples=self.cfg.sampling.n_unconditional_samples,
                        n_batch_size=self.cfg.sampling.batch_size,
                        n_steps=n_steps,
                        inverse_scaler_fn=inverse_scaler_for_sampler
                    )
                
                # Decode and inverse scale
                unconditional_samples_kernel = self._decode_if_needed(pl_module, unconditional_samples_kernel)
                
                unconditional_samples_kernel = unconditional_samples_kernel.clamp(0.0, 1.0)
                self._save_samples_unconditional(
                    pl_module, unconditional_samples_kernel, 
                    title=f"unconditional_samples_kernel_{n_steps}"
                )
                del unconditional_samples_kernel

            for t_cond in tqdm.tqdm(self.cfg.sampling.conditioning_times, desc="Sampling posteriors at different t"):
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    ode_x_t, ode_x_0 = self._sample_batch(
                        pl_module, t_cond, self.cfg.ode_sampling_cfg
                    )
                # Do not clamp here, clamp after decoding in _save_samples
                self._save_samples(
                    pl_module, ode_x_t, ode_x_0, t_cond, 
                    "ode", steps=None
                )
                del ode_x_t, ode_x_0

                for n_steps in self.cfg.consistency_sampling_cfg.steps_to_test:
                    sampling_cfg = self.cfg.consistency_sampling_cfg.copy()
                    sampling_cfg.consistency.steps = n_steps
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        cons_x_t, cons_x_0 = self._sample_batch(
                            pl_module, t_cond, sampling_cfg
                        )
                    # Do not clamp here, clamp after decoding in _save_samples
                    self._save_samples(
                        pl_module, cons_x_t, cons_x_0, t_cond,
                        "consistency", steps=n_steps
                    )
                    del cons_x_t, cons_x_0
            if was_training:
                pl_module.train()
            torch.cuda.empty_cache()
            gc.collect()

    def _sample_batch(self, pl_module, t_cond, sampling_cfg):
        x_t_list, x_1_list = [], []
        bs = self.cfg.sampling.batch_size
        m = self.cfg.sampling.n_samples_per_image
        
        # Use identity scaler for sampling, decode later
        inverse_scaler_for_sampler = lambda x: x

        for i in tqdm.tqdm(
            range(self.cfg.sampling.n_conditioning_samples // self.cfg.sampling.batch_size), 
            desc="Batches"
        ):
            x1_data = self.test_data[i*bs:(i+1)*bs].to(pl_module.device)
            
            # Encode if VAE
            if pl_module.vae:
                with torch.no_grad():
                    with torch.amp.autocast('cuda', enabled=False):
                        # Cast to VAE dtype
                        x1_data = x1_data.to(dtype=pl_module.vae.dtype)
                        latents = pl_module.vae.encode(x1_data).latent_dist.sample()
                        latents = (latents - pl_module.latents_bias) * pl_module.latents_scale
                        x1_data = latents.to(dtype=x1_data.dtype) # Cast back

            x0_data = self.shared_noise[i*bs:(i+1)*bs].to(pl_module.device)
            eps_start = self.shared_posterior_noise[i*(bs*m):(i+1)*(bs*m)].to(pl_module.device)
            t_cond_tensor = torch.full((bs,), t_cond, device=pl_module.device)

            # generate noisy input
            alpha_t, beta_t = self.SI.get_coefficients(t_cond_tensor) # Shape: [B,]
            alpha_t, beta_t = broadcast_to_shape(alpha_t, x1_data.shape), broadcast_to_shape(beta_t, x1_data.shape)
            xt_data = alpha_t * x0_data + beta_t * x1_data
            
            with torch.no_grad():
                xt, x1 = posterior_sampling_fn(
                    sampling_cfg,
                    pl_module.model,
                    xt_data,
                    t_cond_tensor,
                    n_samples_per_image=m,
                    inverse_scaler=inverse_scaler_for_sampler,
                    eps_start=eps_start,
                )
            x_t_list.append(xt)
            x_1_list.append(x1)
        
        return torch.cat(x_t_list, dim=0), torch.cat(x_1_list, dim=0)
    
    def _save_samples(self, pl_module, x_t, x_0, t_cond, sample_type, steps=None):
        save_dir = os.path.join(self.cfg.work_dir, f"samples_{pl_module.global_step}_ema")
        os.makedirs(save_dir, exist_ok=True)

        # Decode if needed
        x_t = self._decode_if_needed(pl_module, x_t)
        x_0 = self._decode_if_needed(pl_module, x_0)

        # Clamp to [0, 1]
        x_t = x_t.clamp(0.0, 1.0)
        x_0 = x_0.clamp(0.0, 1.0)

        # Generate file name
        steps_str = f"_steps_{steps}" if steps is not None else ""
        base_name = f"{sample_type}_samples_t_{t_cond}{steps_str}"
        
        # Save plot
        title = f"{sample_type.title()} Samples at t = {t_cond}"
        if steps is not None:
            title += f" with {steps} steps"

        save_dict = {
            'x_t': x_t.cpu(),
            'x_0_samples': x_0.cpu(),
            't_cond': t_cond,
        }
        if steps is not None:
            save_dict['n_steps'] = steps
                    
        fig = plot_posterior_samples(self.inverse_scaler(self.test_data[0:10].cpu().numpy()),
                                     x_t[0:10].cpu().numpy(),
                                     x_0[0:10, 0:10].cpu().numpy(),
                                     os.path.join(save_dir, f"{base_name}.png"),
                                     title)
        pl_module.logger.experiment.log({
            f"val/{base_name}_grid_ema": [wandb.Image(fig)],
            "global_step": pl_module.global_step
        })
        plt.close(fig)
        torch.save(save_dict, os.path.join(save_dir, f"{base_name}.pt"))

    def _save_samples_unconditional(self, pl_module, samples, title):
        save_dir = os.path.join(self.cfg.work_dir, f"samples_{pl_module.global_step}_ema")
        os.makedirs(save_dir, exist_ok=True)

        base_name = f"unconditional_samples_{title}"
        
        N = samples.shape[0]
        nrow = math.ceil(math.sqrt(N))
        grid = torchvision.utils.make_grid(samples, nrow=nrow, padding=2)
        
        save_dict = {
            'samples': samples.cpu(),
            'sampler': title,
        }

        pl_module.logger.experiment.log({
            f"val/unconditional_grid_ema_{title}": [wandb.Image(grid)],
            "global_step": pl_module.global_step
        })

        torchvision.utils.save_image(
            grid,
            os.path.join(save_dir, f"{base_name}.png"),
        )
        
        torch.save(save_dict, os.path.join(save_dir, f"{base_name}.pt"))