#!/bin/bash
#SBATCH -J varlen_ar                # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=100000                  # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu          # Request partition
#SBATCH --constraint="[a5000|a6000|3090|a100]"
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

LENGTH=$1
SEED=$2

# use model trained w/o eos for variable-length generation
srun python -su -m main \
    mode=sample_eval \
    loader.eval_batch_size=1 \
    data=openwebtext-split \
    algo=ar \
    model.length=$LENGTH \
    eval.checkpoint_path=/share/kuleshov/ma2238/textdiffusion/checkpoints/mari-owt-ar-noeos-v4-1/last.ckpt \
    +wandb.offline=true \
    seed=$SEED \
    sampling.nucleus_p=0.9 \
    sampling.logdir=$PWD/sample_logs/samples_ar \
    sampling.var_length=true