import os
import math
import gc
import tqdm 
import copy


import torch
import torch.distributed as dist

import lightning as pl
from lightning.pytorch.callbacks import Callback
import torchvision
import wandb
import matplotlib.pyplot as plt
from diffusers import AutoencoderKL

from distcfm.utils.evaluation import posterior_sampling_fn, plot_posterior_samples

from lightning.pytorch.callbacks import WeightAveraging
from torch.optim.swa_utils import get_ema_avg_fn

class EMAWeightAveraging(WeightAveraging):
    def __init__(self, decay=0.999):
        super().__init__(avg_fn=get_ema_avg_fn(decay=decay))

    def should_update(self, step_idx=None, epoch_idx=None):
        return (step_idx is not None) and (step_idx >= 100)

def broadcast_to_shape(tensor, shape):
    return tensor.view(-1, *((1,) * (len(shape) - 1)))

class TrainingModule(pl.LightningModule):
    def __init__(self, cfg, model, weighting_model, loss_fn, SI):
        super().__init__()
        self.model = model
        self._teacher_container = []
        if cfg.loss.distill_fm:
            teacher_model = copy.deepcopy(self.model)
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad = False
            self._teacher_container.append(teacher_model)

        self.weighting_model = weighting_model
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.SI = SI

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

        # Initialize EMAs for loss balancing
        self.register_buffer("distill_fm_loss_ratio_ema", torch.tensor(1.0))
        self.register_buffer("distillation_loss_ratio_ema", torch.tensor(1.0))

        self._freeze_step = self.cfg.trainer.get("freeze_main_until_step", 0)
        self._is_main_frozen = False
        if self._freeze_step > 0:
            self._freeze_main_model()

    def _freeze_main_model(self):
        print(f"Freezing main model parameters except joint attention until step {self._freeze_step}...")
        self._is_main_frozen = True
        frozen_count = 0
        unfrozen_count = 0
        for name, param in self.model.named_parameters():
            if "joint_attn" in name or "x_cond" in name:
                param.requires_grad = True
                unfrozen_count += 1
            else:
                param.requires_grad = False
                frozen_count += 1
        print(f"Frozen {frozen_count} parameters, kept {unfrozen_count} parameters trainable (joint attention).")

    def _unfreeze_main_model(self):
        print(f"Step {self.global_step}: Unfreezing main model parameters...")
        self._is_main_frozen = False
        count = 0
        for name, param in self.model.named_parameters():
            param.requires_grad = True
            count += 1
        print(f"Unfroze {count} parameters.")

    def on_train_batch_start(self, batch, batch_idx):
        if self._is_main_frozen and self.global_step >= self._freeze_step:
            self._unfreeze_main_model()

    @property
    def vae(self):
        return self._vae_container[0] if self._vae_container else None
    
    @property
    def teacher_model(self):
        return self._teacher_container[0] if self._teacher_container else None

    def forward(self, x):
        return self.model(x)

    def setup(self, stage: str):
        if self.vae:
            self.vae.to(self.device)
        if self.teacher_model:
            self.teacher_model.to(self.device)

    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print(f"Unused parameter: {name}")

    def on_train_start(self):
        if self.vae:
            self.vae.to(self.device)
        if self.teacher_model:
            self.teacher_model.to(self.device)

    def on_validation_start(self):
        if self.vae:
            self.vae.to(self.device)
        if self.teacher_model:
            self.teacher_model.to(self.device)

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

    def _get_ema_callback(self):
        for cb in self.trainer.callbacks:
            if isinstance(cb, EMAWeightAveraging):
                return cb
        self._ema_callback = None
        return None
    
    def training_step(self, batch, batch_idx):
        try:
            x, labels = batch
        except:
            x = batch      
            labels = None 
        
        if self.global_step == 0 and batch_idx == 0:
            print(f"DEBUG: Input batch shape (raw images): {x.shape}")

        step = self.global_step

        if self.vae:
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=False):
                    x_vae = x.to(dtype=self.vae.dtype)
                    latents = self.vae.encode(x_vae).latent_dist.sample()
                    latents = (latents - self.latents_bias) * self.latents_scale
                    x = latents.to(dtype=x.dtype) # Cast back to original dtype (likely bf16)

        if self.cfg.model.label_dim > 0 and self.cfg.trainer.class_dropout_prob > 0:
            prob = self.cfg.trainer.class_dropout_prob
            mask = torch.bernoulli(torch.full(labels.shape, 1 - prob, device=self.device)).bool()
            labels = torch.where(mask, labels, self.null_class_token.expand_as(labels))

        losses, aux_losses = self.loss_fn(self.model, self.weighting_model, x, labels, step, 
                                          ema_state=self._get_ema_callback(), teacher_model=self.teacher_model)
        for name, loss in losses.items():
            self.log(f"train/{name}", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        for name, loss in aux_losses.items():
            self.log(f"train/{name}", loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        total_loss = 0
        
        # Update loss ratios
        if "fm_loss" in losses and losses["fm_loss"] > 0:
            fm_val = losses["fm_loss"].detach()
            
            if "distill_fm_loss" in losses and losses["distill_fm_loss"] > 0:
                ratio = fm_val / (losses["distill_fm_loss"].detach() + 1e-8)
                self.distill_fm_loss_ratio_ema = 0.99 * self.distill_fm_loss_ratio_ema + 0.01 * ratio
                
            if "distillation_loss" in losses and losses["distillation_loss"] > 0:
                ratio = fm_val / (losses["distillation_loss"].detach() + 1e-8)
                self.distillation_loss_ratio_ema = 0.99 * self.distillation_loss_ratio_ema + 0.01 * ratio

        for name, loss in losses.items():
            if name == "distillation_loss":
                total_loss += loss * self.cfg.loss.distillation_weight * self.distillation_loss_ratio_ema
            elif name == "distill_fm_loss":
                total_loss += loss * self.cfg.loss.distill_fm_weight * self.distill_fm_loss_ratio_ema
            elif name == "fm_loss":
                total_loss += loss * self.cfg.loss.fm_weight
            else:
                total_loss += loss

        self.log("train/total_loss", total_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("train/distill_fm_loss_ratio_ema", self.distill_fm_loss_ratio_ema, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log("train/distillation_loss_ratio_ema", self.distillation_loss_ratio_ema, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("train/lr", current_lr, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        gate_stats_fn = getattr(self.model, "pop_gate_stats", None)
        if callable(gate_stats_fn):
            gate_stats = gate_stats_fn()
            if gate_stats is not None:
                for name, value in gate_stats.items():
                    self.log(f"train/{name}", value, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        weighting_stats_fn = getattr(self.weighting_model, "pop_weighting_stats", None)
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
                with torch.amp.autocast('cuda', enabled=False):
                    x_vae = x.to(dtype=self.vae.dtype)
                    latents = self.vae.encode(x_vae).latent_dist.sample()
                    latents = (latents - self.latents_bias) * self.latents_scale
                    x = latents.to(dtype=x.dtype)

        if self.cfg.model.label_dim > 0 and self.cfg.trainer.class_dropout_prob > 0:
            prob = self.cfg.trainer.class_dropout_prob
            mask = torch.bernoulli(torch.full(labels.shape, 1 - prob, device=self.device)).bool()
            labels = torch.where(mask, labels, self.null_class_token.expand_as(labels))

        ema_val_loss, ema_val_aux_losses = self.loss_fn(self.model, self.weighting_model, x, labels, self.global_step, 
                                                        ema_state=self._get_ema_callback(), teacher_model=self.teacher_model)
        for name, loss in ema_val_loss.items():
            self.log(f"val_ema/{name}", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        for name, loss in ema_val_aux_losses.items():
            self.log(f"val_ema/{name}", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        ema_total_loss = sum(ema_val_loss.values())

        self.log("val_ema/total_loss", ema_total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return ema_total_loss
            
    def configure_optimizers(self):
        # Get only trainable parameters (excludes repa_model which is in a list)
        # We include all parameters even if currently frozen (requires_grad=False)
        # so that the optimizer tracks them and can update them once unfrozen.
        params = [p for p in self.parameters()]
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
        self._last_sampled_step = -1 
    
    def setup(self, trainer, pl_module, stage=None):
        self.world_size = trainer.world_size
        self.rank = trainer.global_rank
        device = pl_module.device

        # Make unconditional samples and n_conditioning_samples divisible by world size
        self.cfg.sampling.n_unconditional_samples = (
            self.cfg.sampling.n_unconditional_samples // self.world_size
        ) * self.world_size
        self.cfg.sampling.n_conditioning_samples = (
            self.cfg.sampling.n_conditioning_samples // self.world_size
        ) * self.world_size

        assert dist.is_available() and dist.is_initialized(), "Distributed package is not available or not initialized"

        # Use CPU generator for deterministic noise across all ranks
        g = torch.Generator(device='cpu')
        g.manual_seed(self.cfg.seed + 12345) # Fixed seed

        self.shared_unconditional_noise = torch.randn(
            self.cfg.sampling.n_unconditional_samples,
            *self.image_shape,
            generator=g,
            device='cpu',
        ).to(device)

        self.shared_noise = torch.randn(
            self.cfg.sampling.n_conditioning_samples,
            *self.image_shape,
            generator=g,
            device='cpu',
        ).to(device)

        self.shared_posterior_noise = torch.randn(
            self.cfg.sampling.n_conditioning_samples * self.cfg.sampling.n_samples_per_image,
            *self.image_shape,
            generator=g,
            device='cpu',
        ).to(device)

    def _decode_if_needed(self, pl_module, samples, vae_batch_size):
        if pl_module.vae:
            # samples are scaled latents
            latents = samples
            
            # Handle 5D input [B, N, C, H, W]
            is_5d = latents.ndim == 5
            if is_5d:
                B, N, C, H, W = latents.shape
                latents = latents.view(B * N, C, H, W)

            latents = latents / pl_module.latents_scale + pl_module.latents_bias
            images = []
            
            for start in range(0, latents.shape[0], vae_batch_size):
                end = min(start + vae_batch_size, latents.shape[0])
                latents_batch = latents[start:end]
                with torch.no_grad():
                    with torch.amp.autocast('cuda', enabled=False):
                        latents_batch = latents_batch.to(dtype=pl_module.vae.dtype)
                        images_batch = pl_module.vae.decode(latents_batch).sample
                        images_batch = images_batch.to(dtype=latents.dtype) # Cast back
                images.append(images_batch)
            
            images = torch.cat(images, dim=0) 
            
            if is_5d:
                images = images.view(B, N, *images.shape[1:])

            # images are [-1, 1]
            images = self.inverse_scaler(images) # [0, 1]
        else:
            # samples are images [-1, 1]
            images = self.inverse_scaler(samples) # [0, 1]
        return images.clamp(0.0, 1.0)
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.global_step + 1) % self.cfg.sampling.every_n_steps != 0:
            return
        if (trainer.global_step == self._last_sampled_step):
            return 
    
        self._last_sampled_step = trainer.global_step
        was_training = pl_module.training
        pl_module.eval()

        self._run_distributed_sampling(pl_module, trainer) 

        if was_training:
            pl_module.train()
    
    def _gather_across_devices(self, tensor, trainer):
        """
        All-gather `tensor` across ranks and return a single concatenated tensor
        on every rank. Assumes the leading dim is batch.
        """
        if trainer.world_size == 1:
            return tensor

        gathered = trainer.strategy.all_gather(tensor)
        gathered = gathered.reshape(-1, *tensor.shape[1:])
        return gathered
    
    def _run_distributed_sampling(self, pl_module, trainer):
        device = pl_module.device

        for n_steps in self.cfg.sampling.n_kernel_steps:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                all_indices = torch.arange(self.cfg.sampling.n_unconditional_samples, device=device)
                device_indices = all_indices[self.rank::self.world_size]
                unconditional_noise_device = self.shared_unconditional_noise[device_indices]
                unconditional_samples_kernel = kernel_sampler_fn(
                    pl_module.model,
                        shape=self.image_shape,
                        shape_decoded=self.image_shape, # Return latents if VAE
                        SI=self.SI,
                        n_samples=unconditional_noise_device.shape[0],
                        n_batch_size=self.cfg.sampling.batch_size,
                        n_steps=n_steps,
                        inverse_scaler_fn=lambda x: x,
                        x0=unconditional_noise_device
                    )
            
            unconditional_samples_kernel = self._decode_if_needed(pl_module, unconditional_samples_kernel, vae_batch_size=self.cfg.sampling.vae_batch_size)
            
            unconditional_samples_kernel = self._gather_across_devices(unconditional_samples_kernel, trainer)
            if trainer.is_global_zero:
                unconditional_samples_kernel = unconditional_samples_kernel.clamp(0.0, 1.0)
                self._save_samples_unconditional(
                    pl_module, unconditional_samples_kernel, 
                    title=f"unconditional_samples_kernel_{n_steps}")

        # get device dependent test-data/corruption-noise/init-noise
        all_indices = torch.arange(self.cfg.sampling.n_conditioning_samples, device=device)
        device_indices = all_indices[self.rank::self.world_size]
        data_device = self.test_data[device_indices.cpu()].to(device)
        shared_noise_device = self.shared_noise[device_indices]
        
        all_indices = torch.arange(self.cfg.sampling.n_conditioning_samples * self.cfg.sampling.n_samples_per_image, 
                                    device=device)
        device_indices = all_indices[self.rank::self.world_size]
        shared_posterior_device = self.shared_posterior_noise[device_indices]

        for t_cond in tqdm.tqdm(self.cfg.sampling.conditioning_times, desc="Sampling posteriors at different t"):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                ode_x_t, ode_x_1 = self._sample_batch(
                    pl_module, t_cond, self.cfg.ode_sampling_cfg, 
                    data_device, shared_noise_device, shared_posterior_device
                )
                ode_x_t = self._decode_if_needed(pl_module, ode_x_t, vae_batch_size=self.cfg.sampling.vae_batch_size)
                ode_x_1 = self._decode_if_needed(pl_module, ode_x_1, vae_batch_size=self.cfg.sampling.vae_batch_size)

            ode_x_t = self._gather_across_devices(ode_x_t, trainer)
            ode_x_1 = self._gather_across_devices(ode_x_1, trainer)

            if trainer.is_global_zero:
                self._save_samples(pl_module, ode_x_t, ode_x_1, t_cond, "ode", steps=None)

            for n_steps in self.cfg.consistency_sampling_cfg.steps_to_test:
                sampling_cfg = self.cfg.consistency_sampling_cfg.copy()
                sampling_cfg.consistency.steps = n_steps
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    cons_x_t, cons_x_1 = self._sample_batch(
                        pl_module, t_cond, sampling_cfg,
                        data_device, shared_noise_device, shared_posterior_device
                    )
                    cons_x_t = self._decode_if_needed(pl_module, cons_x_t, vae_batch_size=self.cfg.sampling.vae_batch_size)
                    cons_x_1 = self._decode_if_needed(pl_module, cons_x_1, vae_batch_size=self.cfg.sampling.vae_batch_size)

                cons_x_t = self._gather_across_devices(cons_x_t, trainer)
                cons_x_1 = self._gather_across_devices(cons_x_1, trainer)

                if trainer.is_global_zero:
                    self._save_samples(pl_module, cons_x_t, cons_x_1, t_cond,"consistency", steps=n_steps)

    def _sample_batch(self, pl_module, t_cond, sampling_cfg, x1, noise_data, noise_start,):
        x_t_list, x_1_list = [], []
        N_local = x1.shape[0]
        bs = self.cfg.sampling.batch_size
        m = self.cfg.sampling.n_samples_per_image
        device = pl_module.device

        for start in tqdm.tqdm(range(0, N_local, bs), desc="Batches"):
            end = min(start + bs, N_local)
            cur_bs = end - start

            # Slice local chunks
            x1_batch = x1[start:end]
            noise_batch = noise_data[start:end]
    
            # Encode if VAE
            if pl_module.vae:
                with torch.no_grad():
                    with torch.amp.autocast('cuda', enabled=False):
                        x1_batch = x1_batch.to(dtype=pl_module.vae.dtype)
                        latents = pl_module.vae.encode(x1_batch).latent_dist.sample()
                        latents = (latents - pl_module.latents_bias) * pl_module.latents_scale
                        x1_batch = latents.to(dtype=x1_batch.dtype) # Cast back

            # generate noisy input
            t_cond_batch = torch.full((cur_bs,), t_cond, device=device)
            alpha_t, beta_t = self.SI.get_coefficients(t_cond_batch) # Shape: [B,]
            alpha_t, beta_t = broadcast_to_shape(alpha_t, x1_batch.shape), broadcast_to_shape(beta_t, x1_batch.shape)
            xt_batch = alpha_t * noise_batch + beta_t * x1_batch
            
            # get starting point for sampler
            eps_start_batch = noise_start[start*m : end*m]  # [cur_bs*m, C, H, W]

            with torch.no_grad():
                xt, x1_out = posterior_sampling_fn(
                    sampling_cfg,
                    pl_module.model,
                    xt_batch,
                    t_cond_batch,
                    n_samples_per_image=m,
                    inverse_scaler=lambda x: x,
                    eps_start=eps_start_batch,
                )
            x_t_list.append(xt)
            x_1_list.append(x1_out)
        return torch.cat(x_t_list, dim=0), torch.cat(x_1_list, dim=0)
    
    def _save_samples(self, pl_module, x_t, x_0, t_cond, sample_type, steps=None):
        save_dir = os.path.join(self.cfg.work_dir, f"samples_{pl_module.global_step}_ema")
        os.makedirs(save_dir, exist_ok=True)

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