#!/bin/bash
#SBATCH -J ppl_zs_owt_bd3lm              # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -e watch_folder/%x_%j.err     # error log file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32G                  # server memory requested (per node)
#SBATCH -t 96:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu         # Request partition
#SBATCH --constraint="[a5000|a6000|3090]"
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                 # Type/number of GPUs needed
#SBATCH --cpus-per-task=1              # Number of CPU cores per task
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption

datasets=("ag_news"
          "scientific_papers_pubmed"
          "scientific_papers_arxiv"
          "lambada"
          "wikitext2"
          "wikitext103"
          "ptb"
          "lm1b-gpt2")
  
BLOCK_SIZE=4

for data in "${datasets[@]}"; do
    echo "$data"
    srun python -u main.py \
        loader.eval_batch_size=16 \
        model=small \
        algo=bd3lm \
        algo.backbone=hf_dit \
        data=$data \
        +data.insert_valid_eos=False \
        model.length=1024 \
        block_size=${BLOCK_SIZE} \
        eval.checkpoint_path=kuleshov-group/bd3lm-owt-block_size${BLOCK_SIZE} \
        wandb=null \
        mode=ppl_eval \
        model.attn_backend=flex > $PWD/logs/bd3lm_${data}_block_size${BLOCK_SIZE}.log
done