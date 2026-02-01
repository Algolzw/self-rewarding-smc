#!/bin/bash
#SBATCH -J genppl_ar                # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -e watch_folder/%x_%j.err     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32G                  # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu          # Request partition
#SBATCH --constraint="[a5000|a6000|3090|a100]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

LENGTH=$1
SEED=$2

# use model trained w/o eos for variable-length generation
srun python -u -m main \
    mode=sample_eval \
    loader.eval_batch_size=1 \
    data=openwebtext-split \
    algo=ar \
    model.length=$LENGTH \
    eval.checkpoint_path=$PWD/ar_owt_noeos.ckpt \
    wandb=null \
    seed=$SEED \
    sampling.num_sample_batches=25 \
    sampling.nucleus_p=0.9 \
    sampling.logdir=$PWD/sample_logs/samples_ar_len${LENGTH} \
    sampling.kv_cache=true