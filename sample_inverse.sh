#!/bin/bash
#SBATCH --job-name=colourization
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1                  # Run a single task, and let torchrun manage the GPU processes
#SBATCH --gpus-per-node=1           # Request 4 GPUs
#SBATCH --cpus-per-task=5           # Request all CPUs for the single task
#SBATCH --array=0-3%1
#SBATCH --mem=100GB
#SBATCH --time=0-20:00:00
#SBATCH --output=/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/outputs/inverse/inverse_colourization-%A_%a.out
#SBATCH --error=/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/outputs/inverse/inverse_colourization-%A_%a.err
#SBATCH --mail-type=END,FAIL

# --- Environment Setup ---
set -euo pipefail # Fail fast on errors

PROJECT_SCRATCH="/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf"
cd "$PROJECT_SCRATCH"
source .venv/bin/activate

# --- Set experiment parameters ---
N_SAMPLES_LIST=(1 2 4 8)
EULER_STEPS_LIST=(100)
job_commands=()

# --- Loop through each parameter combination ---
for euler_steps in "${EULER_STEPS_LIST[@]}"; do
    for n_samples in "${N_SAMPLES_LIST[@]}"; do
        bs=$(( 512 / n_samples ))

        job_commands+=(
            "python scripts/sample_inverse.py \
            inverse_problem=colourization \
            conditional_score=mc \
            +mc.n_samples=$n_samples \
            sampling.euler_steps=$euler_steps \
            sampling.n_samples=1024 \
            sampling.batch_size=$bs"
        )
    done
done
  
command=${job_commands[$SLURM_ARRAY_TASK_ID]}
echo $command
eval $command

echo "Sampling finished."