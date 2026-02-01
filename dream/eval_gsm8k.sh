# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=gsm8k
length=256
block_length=32
num_fewshot=5
steps=$((length / block_length))
model="Dream-org/Dream-v0-Base-7B"


# prefix cache+parallel
accelerate launch --main_process_port 29601 eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},add_bos_token=true,alg=confidence_threshold,threshold=0.9,use_cache=true,temperature=1.0,use_smc=True \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 

