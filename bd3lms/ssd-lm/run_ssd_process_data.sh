#!/usr/bin/bash
#SBATCH -J ssdlm_tokenize             # Job name
#SBATCH -o ./%x_%j.out                # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32000                   # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=kuleshov          # Request partition
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a5000:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption
#SBATCH --mail-user=yzs2@cornell.edu  # Email
#SBATCH --mail-type=END               # Request status by email

source /share/kuleshov/yzs2/text-diffusion/setup_env.sh
trap "kill 0" EXIT

# command example: bash run_ssd_process_data.sh 2022

script_role="host"
global_seed=2022 #$1 # inline param, 2021, 2022, etc
single_device_cuda="0" # inline param, "0", "1", etc
multi_device_cuda="0" # inline param, "0,1,2,3", "0", etc
hf_cache="${HF_HOME}"
core_lm_name="roberta-large"
main_log_dir="${PWD}/logging"

interpret_dataset_name="openwebtext" # just the dataset name, please ignore the "interpret" prefix :)
interpret_dataset_config_name="none"
interpret_additional_dataset_name="none"

# change based on above datasets
interpret_dataset_tokenized_path="${main_log_dir}/openwebtext_processed_pct100_blk200"
# # setup accelerate config
accelerate_config="${main_log_dir}/gpu.yaml"
# CUDA_VISIBLE_DEVICES=${multi_device_cuda} HF_HOME=${hf_cache} accelerate config --config_file ${accelerate_config}

# data hyperparameters
interpret_raw_data_pct=100 # integer from 1pct to 100pct
global_max_seq_len=200
####

################ START ################

CUDA_VISIBLE_DEVICES=${multi_device_cuda} HF_HOME=${hf_cache} python ssd_process_data.py \
    --max_seq_length ${global_max_seq_len} \
    --dataset_name ${interpret_dataset_name} \
    --dataset_config_name ${interpret_dataset_config_name} \
    --additional_dataset_name ${interpret_additional_dataset_name} \
    --raw_data_percentage ${interpret_raw_data_pct} \
    --model_name_or_path ${core_lm_name} \
    --seed ${global_seed} \
    --use_slow_tokenizer \
    --preprocessing_num_workers 32 \
    --output_dir ${interpret_dataset_tokenized_path} \
    --tokenized_data_file_path ${interpret_dataset_tokenized_path} \
    --if_create_tokenized_data_file "yes"

