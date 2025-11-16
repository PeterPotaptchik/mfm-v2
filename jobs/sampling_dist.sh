model_checkpoint="/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/outputs/2025-08-12/20-05-17/checkpoints/periodic-epoch=568-step=100000.ckpt"
config="/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/outputs/2025-08-12/20-05-17/.hydra/config.yaml"

use_ema_flag=true

python scripts/sample.py \
    "++posterior_sampler=distributional_diffusion" \
    "++checkpoint_dir=\"${model_checkpoint}\"" \
    "++train_config=\"${config}\"" \
    "++use_ema=${use_ema_flag}"
