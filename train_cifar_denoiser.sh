#!/bin/bash
#SBATCH --job-name=cifar10_training
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50GB
#SBATCH --time=0-12:00:00
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


srun $PROJECT_SCRATCH/.venv/bin/python scripts/train_denoiser.py ++model.learn_loss_weighting=true



echo "Training finished."