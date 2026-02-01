## Dream Model Evaluation Guide for paper <br><sub> Self-Rewarding Sequential Monte Carlo for Masked Diffusion Language Models</sub>

This document provides detailed instructions for evaluating the Dream model on GSM8K math problem solving and HumanEval code generation tasks.

## Environment Setup

Before running any evaluation, set the following environment variables:
```bash
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
```

## GSM8K Evaluation

GSM8K is a dataset of 8,000 grade school math problems designed to evaluate mathematical reasoning capabilities.

### Common Parameters

```bash
task=gsm8k
length=256
block_length=32
num_fewshot=5
steps=$((length / block_length))
model="Dream-org/Dream-v0-Base-7B"
```

### Evaluation Methods

1. **Baseline**
```bash
accelerate launch --main_process_port 29601 eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},add_bos_token=true,alg=confidence_threshold,threshold=0.9,use_cache=true,temperature=1.0,use_smc=False \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 
```

2. **SR-SMC**
```bash
accelerate launch --main_process_port 29601 eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},add_bos_token=true,alg=confidence_threshold,threshold=0.9,use_cache=true,temperature=1.0,use_smc=True \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 
```


### Parameter Descriptions

- `task`: Evaluation task (gsm8k)
- `length`: Generation length
- `block_length`: Block size for parallel generation
- `num_fewshot`: Number of few-shot examples
- `steps`: Number of generation steps
- `model`: Model name (Dream-v0-Base-7B)
- `use_cache`: Enable prefix cache
- `threshold`: Confidence threshold for parallel generation
- `temperature`: Temperature for diffusion sampling
- `use_smc`: Use self rewarding SMC for sampling
- `show_speed`: Display speed metrics
- `alg`: Generation algorithm (entropy or confidence_threshold)

## HumanEval Evaluation

HumanEval is a dataset of 164 Python programming problems designed to evaluate code generation capabilities.

### Common Parameters

```bash
task=humaneval
length=256
block_length=32
steps=$((length / block_length))
model="Dream-org/Dream-v0-Base-7B"
```

### Evaluation Methods

1. **Baseline**
```bash
accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},add_bos_token=true,alg=confidence_threshold,threshold=0.9,use_cache=true,temperature=1.0,use_smc=False,show_speed=True,escape_until=true \
    --tasks ${task} \
    --batch_size 1 \
    --output_path evals_results/cache_parallel/humaneval-ns0-${length}-${logname} --log_samples \
    --confirm_run_unsafe_code
```

2. **SR-SMC**
```bash
accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},add_bos_token=true,alg=confidence_threshold,threshold=0.9,use_cache=true,temperature=1.0,use_smc=True,show_speed=True,escape_until=true \
    --tasks ${task} \
    --batch_size 1 \
    --output_path evals_results/cache_parallel/humaneval-ns0-${length}-${logname} --log_samples \
    --confirm_run_unsafe_code
```

### Additional Parameters for HumanEval

- `escape_until`: Enable escape until for code generation
- `confirm_run_unsafe_code`: Confirm running unsafe code for evaluation
- `log_samples`: Log generated samples for analysis

### Post-processing

For HumanEval evaluation, post-processing is required:
```bash
python postprocess_code.py {the samples_xxx.jsonl file under output_path}
```

## Notes

1. All evaluations use the Dream-v0-Base-7B model
2. Results are saved in the `evals_results` directory
3. For HumanEval, samples are logged for postprocessing
4. Speed metrics are shown for all evaluations
5. Different optimization strategies can be combined:
6. HumanEval evaluation requires additional safety confirmations 