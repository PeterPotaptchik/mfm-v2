#!/bin/bash
#SBATCH --job-name=cifar10_sampling
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20GB
#SBATCH --time=0-0:30:00
#SBATCH --output=/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/outputs/cifar_sample-%A.out
#SBATCH --error=/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/outputs/cifar_sample-%A.err
#SBATCH --mail-type=END,FAIL

# --- Environment Setup ---
# Change to your project directory and activate the Python environment
PROJECT_SCRATCH="/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf"
cd $PROJECT_SCRATCH
source .venv/bin/activate

# --- Run Training ---
echo "Starting CIFAR-10 sampling..."


train_run_dir="/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/outputs/2025-08-08/18-00-02"
model_checkpoint="/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/outputs/2025-08-08/18-00-02/checkpoints/periodic-epoch=284-step=100000.ckpt"
n_flow_steps=(1 2 5 10)
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
