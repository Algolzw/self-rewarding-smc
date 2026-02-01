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

# for s in $(seq 1 10); do sbatch --job-name="ssdlm_generate_seed-${s}" run_generate_text.sh ${s}; done

# source /share/kuleshov/yzs2/text-diffusion/setup_env.sh

SEED=${1}
LEN=${2} # 1025, 2050
TOTAL_T=${3} # 1000, 25

srun python generate_text.py \
  --generated_samples_outfile="${PWD}/generated_samples/samples_openwebtext-valid_bs25_l${LEN}_t${TOTAL_T}_SEED${SEED}" \
  --num_samples_to_generate=1 \
  --seed=${SEED} \
  --model_name_or_path="xhan77/ssdlm" \
  --use_slow_tokenizer=True \
  --max_seq_length=${LEN} \
  --one_hot_value=5 \
  --decoding_block_size=25 \
  --total_t=${TOTAL_T} \
  --projection_top_p=0.95