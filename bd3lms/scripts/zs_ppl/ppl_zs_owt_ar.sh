#!/bin/bash
#SBATCH -J ppl_zs_owt_ar              # Job name
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

for data in "${datasets[@]}"; do
    echo "$data"
    srun python -u main.py \
        loader.eval_batch_size=16 \
        model=small \
        algo=ar \
        data=$data \
        +data.insert_valid_eos=False \
        model.length=1024 \
        eval.checkpoint_path=/share/kuleshov/ssahoo/textdiffusion/text-diffusion-exp-v4-AgBZrc-small-ar-param-ar_data-openwebtext-split_seqlen-1024_maxs-1300001_bs-512/checkpoints/last.ckpt \
        wandb=null \
        mode=ppl_eval > $PWD/logs/ar_$data.log
done