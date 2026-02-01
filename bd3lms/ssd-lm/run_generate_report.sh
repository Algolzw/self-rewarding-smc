#!/usr/bin/bash
#SBATCH -o ./generated_samples/logs/%x_%j.out  # output file (%j expands to jobID)
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -e watch_folder/%x_%j.err     # error log file (%j expands to jobID)
#SBATCH -N 1                                   # Total number of nodes requested
#SBATCH --get-user-env                         # retrieve the users login environment
#SBATCH --mem=32000                            # server memory requested (per node)
#SBATCH -t 960:00:00                           # Time limit (hh:mm:ss)
#SBATCH --partition=gpu,kuleshov                   # Request partition
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --constraint="[a5000|a6000|3090]"
#SBATCH --gres=gpu:1                 # Type/number of GPUs needed
#SBATCH --open-mode=append                     # Do not overwrite logs
#SBATCH --requeue                              # Requeue upon preemption

srun python report_genppl.py