#!/bin/bash
#SBATCH -J recordlen_sedd                # Job name
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

SEED=$1

srun python -u -m main \
    mode=sample_eval \
    loader.eval_batch_size=1 \
    data=openwebtext-split \
    algo=sedd \
    algo.T=1000 \
    model.length=1024 \
    eval.checkpoint_path=/share/kuleshov/ma2238/textdiffusion/checkpoints/mari-owt-sedd-noeos-v4/61-1000000.ckpt \
    wandb=null \
    sampling.var_length=true \
    seed=$SEED \
    sampling.nucleus_p=0.99 \
    sampling.logdir=$PWD/sample_logs/samples_sedd_len1024
