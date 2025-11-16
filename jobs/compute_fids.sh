# train_run_dir="/vols/bitbucket/saravanan/distributional-mf/outputs/2025-08-09/01-00-54"
# model_checkpoint="/vols/bitbucket/saravanan/distributional-mf/outputs/2025-08-09/01-00-54/checkpoints/periodic-epoch=284-step=100000.ckpt"
train_run_dir="/vols/bitbucket/saravanan/distributional-mf/outputs/2025-08-06/23-49-08"
model_checkpoint="/vols/bitbucket/saravanan/distributional-mf/outputs/2025-08-06/23-49-08/checkpoints/periodic-epoch=284-step=100000.ckpt"

n_flow_steps=(2)
use_ema=(True)

for n_flow_step in "${n_flow_steps[@]}"; do
    for use_ema_flag in "${use_ema[@]}"; do
        echo "Sampling with n_flow_step=${n_flow_step} and use_ema=${use_ema_flag}"
        python scripts/sample.py \
            "++train_dir=\"${train_run_dir}\"" \
            "++checkpoint_dir=\"${model_checkpoint}\"" \
            "++consistency.steps=${n_flow_step}" \
            "++use_ema=${use_ema_flag}" 
    done
done
