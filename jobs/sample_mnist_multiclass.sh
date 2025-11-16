#!/bin/bash
#SBATCH --mail-user=adhithya.saravanan@stats.ox.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --output=/vols/bitbucket/saravanan/distributional-mf/slurm_outputs/slurm-%A_%a.out
#SBATCH --error=/vols/bitbucket/saravanan/distributional-mf/slurm_outputs/slurm-%A_%a.out

#SBATCH --job-name=sample_cifar
#SBATCH --cluster=swan
#SBATCH --partition=standard-rainml-gpu
#SBATCH --gres=gpu:Ampere_H100_80GB:1
#NOTSBATCH --nodelist=rainmlgpu01.cpu.stats.ox.ac.uk
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --mem=32GB
#SBATCH --ntasks=1                
#SBATCH --array=0-3%3

source /data/localhost/not-backed-up/saravanan/miniconda3/bin/activate base
env_name="distcfm"
export PATH_TO_GIT="/vols/bitbucket/saravanan/distributional-mf"    
cd $PATH_TO_GIT
conda activate $env_name

job_commands=()
n_iwaes=(1 2 4)
weights=('weights_to_sample=[1,0,0,0,0,0,0,0,0,0]')
# 'weights_to_sample=[2,0,2,0,1,0,1,0,1,1]'
euler_steps=(250)

for iwae in "${n_iwaes[@]}"; do
    for weight in "${weights[@]}"; do
        for euler_step in "${euler_steps[@]}"; do
            bs=$((1024 / iwae))
            job_commands+=("python scripts/sample_multiclass.py ++conditional_score=iwae ++total_samples=5000 ++iwae.n_samples=$iwae ++iwae.type=cfm_approximate ++sampling.batch_size=${bs} ++sampling.euler_steps=${euler_step} ${weight}")
        done
    done
done

command="${job_commands[$SLURM_ARRAY_TASK_ID]}"
eval $command