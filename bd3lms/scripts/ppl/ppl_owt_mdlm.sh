#!/bin/bash
#SBATCH -J ppl_owt_mdlm                # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32G                  # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu          # Request partition
#SBATCH --constraint="[a5000|a6000|3090|a100]"
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

srun python -u main.py \
    loader.eval_batch_size=16 \
    model=small \
    algo=mdlm \
    algo.backbone=hf_dit \
    algo.ignore_bos=false \
    data=openwebtext-split \
    model.length=1024 \
    eval.checkpoint_path=kuleshov-group/mdlm-owt \
    wandb=null \
    mode=ppl_eval > $PWD/logs/mdlm_owt.log