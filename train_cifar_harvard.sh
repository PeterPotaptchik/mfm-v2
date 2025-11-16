#!/bin/bash
#SBATCH --job-name=cifar10_training
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50GB
#SBATCH --time=0-7:00:00
#SBATCH --output=/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/outputs/cifar_train-%A.out
#SBATCH --error=/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/outputs/cifar_train-%A.err
#SBATCH --mail-type=END,FAIL

# --- Environment Setup ---
# Change to your project directory and activate the Python environment
PROJECT_SCRATCH="/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf"
cd $PROJECT_SCRATCH
source .venv/bin/activate

# --- Run Training ---
echo "Starting CIFAR-10 training..."

# # # Launch the training script, overriding config paths to use the correct scratch space
# srun $PROJECT_SCRATCH/.venv/bin/python scripts/train.py \
#     model=dit_cifar \
#     ++data_dir=$PROJECT_SCRATCH/data \
#     hydra.run.dir=$PROJECT_SCRATCH/outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S} \
#     wandb.name="dit 20K" \
#     ++cfg.trainer.num_warmup_steps=20000

# model_checkpoint="/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/outputs/2025-08-11/21-54-55/checkpoints/periodic-epoch=511-step=90000.ckpt"
# python scripts/train_distributional_model.py ++wandb.name=distributional_model resume_from_checkpoint="/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/outputs/2025-08-12/20-05-17/checkpoints/periodic-epoch=568-step=100000.ckpt"
# python scripts/train_distributional_model.py \
#   "wandb.name=distributional_model" \
#   "resume_from_checkpoint=\"${model_checkpoint}\"" \

# bash jobs/sampling_dist.sh

srun $PROJECT_SCRATCH/.venv/bin/python scripts/train_denoiser.py ++model.learn_loss_weighting=true ++use_snr_parametrisation=false ++sampling.every_n_steps=10000 dataset=mnist



echo "Training finished."