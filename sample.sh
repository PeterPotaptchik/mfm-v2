#!/bin/bash
#SBATCH --job-name=imagenet
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1                  # Run a single task; torchrun will spawn GPU processes
#SBATCH --cpus-per-task=32          # Request all CPUs for the single task
#SBATCH --mem=640GB
#SBATCH --time=0-01:00:00
#SBATCH --output=/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/new-outputs/imagenet/imagenet_train-%A.out
#SBATCH --error=/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/new-outputs/imagenet/imagenet_train-%A.err
#SBATCH --mail-type=END,FAIL

# --- Environment Setup ---
set -euo pipefail # Fail fast on errors

PROJECT_SCRATCH="/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf"
cd "$PROJECT_SCRATCH"
source .venv/bin/activate

cd "/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/flash-attention/hopper"
module load cuda/12.9.1-fasrc01

export CUDA_HOME=$CUDA_HOME   # many modules already set this; check with `echo $CUDA_HOME`
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

rm -rf build/ dist/ *.egg-info flash_attn.egg-info
python setup.py install
