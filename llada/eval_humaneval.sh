# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=humaneval
length=256
block_length=32
steps=$((length / block_length))
model_path='GSAI-ML/LLaDA-1.5'
factor=1.0
logname=smc-b${block_length}


accelerate launch eval_llada.py --tasks ${task} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},use_cache=True,threshold=0.9,use_smc=True,show_speed=True \
--output_path evals_results/cache_parallel/humaneval-ns0-${length}-${logname} --log_samples


## NOTICE: use postprocess for humaneval
# python postprocess_code.py {the samples_xxx.jsonl file under output_path}

