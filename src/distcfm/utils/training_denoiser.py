import torch
import lightning as pl
import torchvision
from lightning.pytorch.callbacks import Callback
import wandb

class TrainingModuleDenoiser(pl.LightningModule):
    def __init__(self, cfg, model, loss_fn, sde, inverse_scaler,):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.inverse_scaler = inverse_scaler
        self.sde = sde

    def training_step(self, batch, batch_idx):
        try:
            x, _ = batch
        except:
            x = batch       
        step = self.global_step

        losses, aux_losses = self.loss_fn(self.model, x, step)

        for name, loss in losses.items():
            self.log(f"train/{name}", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        for name, loss in aux_losses.items():
            self.log(f"train/{name}", loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        total_loss = 0

        for name, loss in losses.items():
            total_loss += loss

        self.log("train/total_loss", total_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("train/lr", current_lr, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return total_loss

    def on_before_optimizer_step(self, optimizer,):
        if self.global_step % 10 == 0:
            total_norm = torch.norm(torch.stack([
                p.grad.detach().norm(2)
                for p in self.parameters() if p.grad is not None
            ]))
            self.log("grad_l2_norm", total_norm, on_step=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        """Run validation with both regular and EMA parameters if available"""
        try:
            x, _ = batch
        except:
            x = batch

        ema_val_loss, ema_val_aux_losses = self.loss_fn(self.model, x, self.global_step)
        for name, loss in ema_val_loss.items():
            self.log(f"val_ema/{name}", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for name, loss in ema_val_aux_losses.items():
            self.log(f"val_ema/{name}", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        ema_total_loss = sum(ema_val_loss.values())
        self.log("val_ema/total_loss", ema_total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return ema_total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr.val)
        
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


class DenoiserSamplingCallback(Callback):
    """Callback for generating and saving denoised samples during training"""
    def __init__(self, cfg, sde, inverse_scaler, sampling_eps=1e-2):
        super().__init__()
        self.cfg = cfg
        self.sde = sde
        self.inverse_scaler = inverse_scaler
        self.sampling_eps = sampling_eps
        self.n_samples = 16
        self.n_steps = 100

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.cfg.sampling.every_n_steps == 0:
            self._run_sampling(pl_module)

    def _run_sampling(self, pl_module):
        image_shape = (
            self.cfg.dataset.img_channels, 
            self.cfg.dataset.img_resolution, 
            self.cfg.dataset.img_resolution
        )

        with torch.no_grad():
            x = torch.randn(self.n_samples, *image_shape, device=pl_module.device)
            
            # Create time steps for sampling
            time_steps = torch.linspace(
                self.sde.t_max - self.sampling_eps,
                self.sampling_eps, 
                self.n_steps + 1,
                device=pl_module.device
            )

            for i in range(self.n_steps):
                t_now, t_next = time_steps[i], time_steps[i+1]
                
                alpha_t, var_t = self.sde.get_coefficients(t_now)
                sigma_t = torch.sqrt(var_t)
                predicted_noise = pl_module.model(x, t_now.expand(self.n_samples))
                x_0_pred = (x - sigma_t * predicted_noise) / alpha_t

                alpha_t_next, var_t_next = self.sde.get_coefficients(t_next)
                sigma_t_next = torch.sqrt(var_t_next)

                x = alpha_t_next * x_0_pred + sigma_t_next * predicted_noise

            generated_images = self.inverse_scaler(x.cpu())

        grid = torchvision.utils.make_grid(generated_images, nrow=4, padding=2)
        pl_module.logger.experiment.log({
            f"val/generated_grid_ema": [wandb.Image(grid)],
            "global_step": pl_module.global_step
        })
