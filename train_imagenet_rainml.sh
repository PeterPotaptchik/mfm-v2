#!/bin/bash
#SBATCH --mail-user=adhithya.saravanan@stats.ox.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --output=/vols/bitbucket/saravanan/mfm-v2/slurm_outputs/slurm-%A_%a.out
#SBATCH --error=/vols/bitbucket/saravanan/mfm-v2/slurm_outputs/slurm-%A_%a.out

#SBATCH --job-name=imagenet
#SBATCH --cluster=swan
#SBATCH --partition=standard-rainml-gpu
#SBATCH --gres=gpu:Ampere_H100_80GB:2
#NOTSBATCH --nodelist=rainmlgpu01.cpu.stats.ox.ac.uk
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --ntasks-per-node=2
#SBATCH --array=0%1

source /data/localhost/not-backed-up/saravanan/miniconda3/bin/activate base
env_name="distcfm12"
export PATH_TO_GIT="/vols/bitbucket/saravanan/mfm-v2"    
cd $PATH_TO_GIT
conda activate $env_name

job_commands=()
job_commands+=("torchrun --nproc_per_node=2 --nnodes=1 scripts/train.py --config-name config_test \
        model=dit_imagenet_latent \
        dataset=imagenet_1k \
        ++data_dir=/data/localhost/not-backed-up/saravanan/datasets/imagenet_1k \
        ++model.input_size=32 \
        ++model.in_channels=4 \
        ++model.label_dim=1000 \
        ++trainer.devices=2 \
        ++trainer.num_nodes=1 \
        ++lr.val=0.0001 \
        ++lr.scheduler=constant \
        ++lr.warmup_steps=1000 \
        ++trainer.num_warmup_steps=10000 \
        ++trainer.anneal_end_step=25000 \
        ++trainer.num_train_steps=100000 \
        ++trainer.batch_size=48 \
        ++trainer.accumulate_grad_batches=2 \
        ++trainer.num_workers=12 \
        ++trainer.class_dropout_prob=0.2 \
        ++loss.distillation_type=mf \
        ++loss.model_guidance=true \
        ++compile=false \
        ++optimizer=RAdam \
        ++trainer.ema.decay=0.9995 \
        ++trainer.t_cond_warmup_steps=0 \
        ++trainer.t_cond_0_rate=0.1 \
        ++trainer.t_cond_power=1.25 \
        ++sampling.every_n_steps=10000 \
        ++trainer.checkpoint_every_n_steps=25000 \
        ++use_parametrization=False \
        ++weight_decay=0.00 \
        +init_from_sit=True \
        ++init_weights_ckpt=\"/vols/bitbucket/saravanan/mfm-v2/ckpt/dmf_xl_2_256.pt\" \
        ++resume_from_checkpoint=\"/homes/saravanan/mfm_outputs/2025-12-11/21-14-59/checkpoints/periodic-epoch_03-step_25000.ckpt\"")
        
command=${job_commands[$SLURM_ARRAY_TASK_ID]}
eval $command
