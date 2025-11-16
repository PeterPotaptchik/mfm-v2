model_checkpoint="/vols/bitbucket/saravanan/distributional-mf/outputs/2025-08-09/01-00-54/checkpoints/periodic-epoch=284-step=100000.ckpt"
config="/vols/bitbucket/saravanan/distributional-mf/outputs/2025-08-09/01-00-54/.hydra/config.yaml"
n_flow_steps=(5 10)
use_ema=(True)

for n_flow_step in "${n_flow_steps[@]}"; do
    for use_ema_flag in "${use_ema[@]}"; do
        echo "Sampling with n_flow_step=${n_flow_step} and use_ema=${use_ema_flag}"
        python scripts/sample.py \
            "++checkpoint_dir=\"${model_checkpoint}\"" \
            "++train_config=\"${config}\"" \
            "++consistency.steps=${n_flow_step}" \
            "++use_ema=${use_ema_flag}"
    done
done
