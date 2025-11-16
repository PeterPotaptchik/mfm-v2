#!/bin/bash
#SBATCH --job-name=deblur
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1                  # Run a single task, and let torchrun manage the GPU processes
#SBATCH --gpus-per-node=1           # Request 4 GPUs
#SBATCH --cpus-per-task=5           # Request all CPUs for the single task
#SBATCH --array=0-3%2
#SBATCH --mem=100GB
#SBATCH --time=0-2:00:00
#SBATCH --output=/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/outputs/inverse_dps/inverse_deblur-%A_%a.out
#SBATCH --error=/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/outputs/inverse_dps/inverse_deblur-%A_%a.err
#SBATCH --mail-type=END,FAIL

# --- Environment Setup ---
set -euo pipefail # Fail fast on errors

PROJECT_SCRATCH="/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf"
cd "$PROJECT_SCRATCH"
source .venv/bin/activate

# --- Set experiment parameters ---
EULER_STEPS_LIST=(100 250)
FACTOR_LIST=(0.5 1.0 2.0)

job_commands=()

# --- Loop through each parameter combination ---
for kernel_sigma in "${FACTOR_LIST[@]}"; do
    for euler_steps in "${EULER_STEPS_LIST[@]}"; do
        job_commands+=(
            "python scripts/sample_inverse.py \
            inverse_problem=deblur \
            sampling.euler_steps=$euler_steps \
            sampling.n_samples=1024 \
            sampling.batch_size=512 \
            conditional_score=dps \
            inverse_problem.kernel_sigma=$kernel_sigma"
        )
    done
done
  
command=${job_commands[$SLURM_ARRAY_TASK_ID]}
echo $command
eval $command

echo "Sampling finished."