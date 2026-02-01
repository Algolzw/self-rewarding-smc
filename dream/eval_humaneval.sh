# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=humaneval
length=256
block_length=32
steps=$((length / block_length))
model="Dream-org/Dream-v0-Base-7B"
logname=smc


# # prefix cache+parallel
accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},add_bos_token=true,alg=confidence_threshold,threshold=0.9,use_cache=true,temperature=1.0,use_smc=True,show_speed=True,escape_until=true \
    --tasks ${task} \
    --batch_size 1 \
    --output_path evals_results/cache_parallel/humaneval-ns0-${length}-${logname} --log_samples \
    --confirm_run_unsafe_code

# ## NOTICE: use postprocess for humaneval
python postprocess_code.py {the samples_xxx.jsonl file under output_path}
