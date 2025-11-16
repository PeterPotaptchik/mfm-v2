import torch
import os
import lightning as pl
from distcfm.utils.evaluation import posterior_sampling_fn, plot_posterior_samples
import tqdm 
from lightning.pytorch.callbacks import Callback
import matplotlib.pyplot as plt

class TrainingModuleDistributional(pl.LightningModule):
    def __init__(self, cfg, model, loss_fn, sde,):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.sde = sde

    def forward(self, x):
        return self.model(x)

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


class DistributionalSamplingCallback(Callback):
    """Callback for generating and saving distributional samples during training"""
    def __init__(self, cfg, test_data, inverse_scaler, sde):
        super().__init__()
        self.cfg = cfg
        self.test_data = test_data
        self.inverse_scaler = inverse_scaler
        self.sde = sde

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.cfg.sampling.every_n_steps == 0:
            self._run_sampling(pl_module)
    
    def _run_sampling(self, pl_module):
        for t_cond in tqdm.tqdm(self.cfg.sampling.conditioning_times, desc="Sampling"):
            x_t_samples, x_0_samples = [], []
            
            # Sample in batches
            for i in tqdm.tqdm(
                range(self.cfg.sampling.n_conditioning_samples // self.cfg.sampling.batch_size), 
                desc="Batches"
            ):
                x_batch = self.test_data[i*self.cfg.sampling.batch_size:(i+1)*self.cfg.sampling.batch_size]
                t_cond_tensor = torch.full(
                    (self.cfg.sampling.batch_size,), 
                    t_cond, 
                    device=pl_module.device
                )
                
                with torch.no_grad():
                    x_t, x_0 = posterior_sampling_fn(
                        self.cfg.dist_sampling_cfg,
                        pl_module.model,
                        self.sde,
                        x_batch,
                        t_cond_tensor,
                        inverse_scaler=self.inverse_scaler
                    )
                x_t_samples.append(x_t)
                x_0_samples.append(x_0)

            # Concatenate results
            x_t_samples = torch.cat(x_t_samples, dim=0)
            x_0_samples = torch.cat(x_0_samples, dim=0)
            
            # Save results
            save_dir = os.path.join(self.cfg.work_dir, f"samples_{pl_module.global_step}_ema")
            os.makedirs(save_dir, exist_ok=True)
            
            # Save plot
            title = f"Distributional samples at t = {t_cond}"
            fig = plot_posterior_samples(
                self.inverse_scaler(self.test_data[0:10].cpu().numpy()),
                x_t_samples[0:10].cpu().numpy(),
                x_0_samples[0:10, 0:10].cpu().numpy(),
                os.path.join(save_dir, f"dist_samples_t_{t_cond}.png"),
                title
            )
            plt.close(fig)

            # Save tensors
            torch.save({
                'x_t': x_t_samples.cpu(),
                'x_0_samples': x_0_samples.cpu(),
                't_cond': t_cond,
            }, os.path.join(save_dir, f"dist_samples_t_{t_cond}.pt"))