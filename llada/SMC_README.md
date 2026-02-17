# Self-Rewarding Sequential Monte Carlo (SMC) for LLaDA

This document provides a detailed explanation of the SMC (Sequential Monte Carlo) implementation for the LLaDA diffusion language model.

## Overview

Self-Rewarding SMC is an inference-time scaling method that leverages **trajectory-level confidence** from diffusion models as importance weights to guide generation toward globally confident, high-quality samples. The key advantage is that this method is **reward-free**, making it applicable to arbitrary pretrained models without requiring external reward models.

## Core Concepts

### 1. Particle-Based Generation

SMC maintains multiple generation trajectories (called "particles") in parallel:

```python
x = torch.full((prompt.shape[0] * num_particles, prompt.shape[1] + gen_length), 
               mask_id, dtype=torch.long).to(model.device)
```

- Each particle represents an independent generation trajectory
- Default number of particles: `num_particles=4`
- All particles start with the same prompt but evolve differently during generation

### 2. Confidence-Based Weighting

The SMC algorithm tracks the log-probability (confidence) of each token transition:

```python
logp = torch.zeros_like(x, dtype=torch.float32, device=x.device)
log_w = torch.zeros(num_particles).to(model.device)
```

**Key Variables:**
- `logp`: Stores log-probability for each token in each particle
- `log_w`: Accumulates total weight (confidence) for each particle
- Weights are updated during each token transition:
  ```python
  log_w = log_w + (x0_logp * transfer_index.float()).sum(dim=1)
  ```

### 3. Block-wise Generation

Generation happens in blocks to balance efficiency and quality:

```python
num_blocks = gen_length // block_length
steps = steps // num_blocks
```

- `gen_length`: Total length to generate (e.g., 256 tokens)
- `block_length`: Size of each block (e.g., 32 tokens)
- Each block is generated iteratively through `steps` diffusion steps

### 4. Resampling Mechanism

SMC uses **Effective Sample Size (ESS)** to determine when to resample:

```python
# Normalize weights
weights = torch.exp(log_w - log_w.max())
weights = weights / weights.sum()

# Calculate ESS
ess = 1.0 / (weights.pow(2).sum())

# Resample if ESS is too low
if ess < 0.5 * num_particles:
    k_idx = torch.multinomial(weights, num_samples=num_particles, replacement=True)
    x = x[k_idx]
    logp = logp[k_idx]
    log_w.zero_()  # Reset weights after resampling
```

**ESS Intuition:**
- ESS measures how "diverse" the particles are
- High ESS (≈ num_particles): All particles have similar weights → good diversity
- Low ESS (≪ num_particles): Few particles dominate → poor diversity
- Resampling threshold: `0.5 * num_particles`
- Resampling duplicates high-weight particles and discards low-weight ones

## Two Main Generation Functions

### 1. `generate()` - Basic SMC Generation

Standard SMC generation without caching:

```python
def generate(model, prompt, steps=128, gen_length=128, block_length=128, 
             temperature=0., remasking='low_confidence', mask_id=126336, 
             threshold=None, factor=None, num_particles=4)
```

**Process:**
1. Initialize `num_particles` copies of the prompt
2. For each block:
   - Iteratively unmask tokens using model predictions
   - Track log-probabilities for each token
   - Accumulate particle weights
   - Resample particles when ESS drops below threshold
3. Return the particle with highest total confidence

### 2. `generate_with_prefix_cache_smc()` - Optimized SMC with Caching

Optimized version using prefix caching for faster inference:

```python
def generate_with_prefix_cache_smc(model, prompt, steps=128, gen_length=128, 
                                   block_length=128, temperature=0., 
                                   remasking='low_confidence', mask_id=126336, 
                                   threshold=None, factor=None, num_particles=4)
```

**Key Optimizations:**
- Caches key-value pairs from the prompt to avoid recomputing
- Only processes new tokens in subsequent steps
- Significantly reduces computational cost (NFE - Number of Function Evaluations)

**Caching Strategy:**
```python
# First pass with caching enabled
output = model(x, use_cache=True)
past_key_values = output.past_key_values

# Subsequent passes use cached keys/values
logits = model(x[:, current_block_start:], 
               past_key_values=past_key_values, 
               use_cache=True).logits
```

## Supporting Functions

### `add_gumbel_noise()`

Adds Gumbel noise for categorical sampling:

```python
def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise
```

- `temperature=0`: Greedy (argmax) sampling
- `temperature>0`: Stochastic sampling with noise
- Uses float64 for numerical stability

### `get_num_transfer_tokens()`

Computes how many tokens to unmask at each step:

```python
def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    # ... distribute tokens evenly across steps
```

- Ensures uniform token transition schedule
- Aligns with LLaDA's linear noise schedule

### `get_transfer_index()`

Selects which tokens to unmask based on confidence:

```python
def get_transfer_index(logits, temperature, remasking, mask_index, x, 
                      num_transfer_tokens, threshold=None)
```

**Remasking Strategies:**
- `'low_confidence'`: Select tokens with highest model confidence
- `'random'`: Random selection

**Process:**
1. Apply Gumbel noise to logits
2. Get predictions: `x0 = torch.argmax(logits_with_noise, dim=-1)`
3. Calculate confidence: `x0_p = exp(log_softmax(logits))`
4. Select top-k most confident tokens to unmask

## Usage Example

### Basic Usage

```python
from generate_smc import generate_with_prefix_cache_smc
from model.modeling_llada import LLaDAModelLM
from transformers import AutoTokenizer

# Load model
model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-1.5')
tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-1.5')

# Prepare prompt
prompt_text = "What is the capital of France?"
prompt_ids = tokenizer(prompt_text)['input_ids']
prompt = torch.tensor(prompt_ids).unsqueeze(0).to(model.device)

# Generate with SMC
output, nfe = generate_with_prefix_cache_smc(
    model=model,
    prompt=prompt,
    steps=8,              # 8 steps per block
    gen_length=256,       # Generate 256 tokens
    block_length=32,      # 32 tokens per block
    temperature=1.0,      # Stochastic sampling
    remasking='low_confidence',
    num_particles=4       # Use 4 particles
)

# Decode output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Generated: {generated_text}")
print(f"NFE (Number of Function Evaluations): {nfe}")
```

### Integration with Evaluation

The SMC implementation integrates seamlessly with the evaluation harness:

```python
# In eval_llada.py
if self.use_smc:
    generated_answer, nfe = generate_with_prefix_cache_smc(
        self.model, input_ids, 
        steps=self.steps, 
        gen_length=self.gen_length,
        block_length=self.block_length, 
        temperature=self.temperature,
        threshold=self.threshold,
        num_particles=4
    )
```

## Parameter Recommendations

### For Math Tasks (GSM8K)

```bash
gen_length=256
block_length=32
steps=8  # (256 / 32 = 8 blocks)
temperature=1.0
threshold=0.9
num_particles=4
```

### For Code Generation (HumanEval)

```bash
gen_length=256
block_length=32
steps=8
temperature=1.0
threshold=0.9
num_particles=4
```

### Tuning Guidelines

**Number of Particles (`num_particles`):**
- More particles (e.g., 8-16): Better quality but slower
- Fewer particles (e.g., 2-4): Faster but may reduce quality
- Recommended: Start with 4, increase if quality is insufficient

**Temperature:**
- `0.0`: Deterministic (greedy)
- `1.0`: Standard stochastic sampling (recommended)
- `>1.0`: More diverse but potentially lower quality

**Block Length:**
- Larger blocks: Faster generation, fewer resampling opportunities
- Smaller blocks: More frequent resampling, potentially better quality
- Recommended: 32 tokens for good balance

**Resampling Frequency:**
- Currently resamples every block when `ess < 0.5 * num_particles`
- Modify the condition `if num_block % 1 == 0` to change frequency
- Example: `if num_block % 2 == 0` → resample every 2 blocks

## Performance Characteristics

### Computational Cost

**NFE (Number of Function Evaluations):**
- Without caching: `NFE ≈ num_blocks * avg_steps_per_block`
- With caching: Reduced by ~50% depending on block size
- SMC overhead: Minimal (weight computation is fast)

**Memory Usage:**
- Proportional to `num_particles`
- 4 particles ≈ 4x memory of single generation
- Consider reducing `num_particles` on memory-constrained devices

### Quality Improvements

SMC consistently improves generation quality by:
1. **Diversity**: Maintains multiple hypotheses in parallel
2. **Selection**: Automatically selects high-confidence trajectories
3. **Resampling**: Focuses computational resources on promising particles

**Typical Improvements:**
- GSM8K: +3-5% accuracy
- HumanEval: +2-4% pass@1

## Algorithm Summary

The SMC algorithm can be summarized as:

```
For each block:
    1. Initialize: Create num_particles copies
    2. Propagate: 
        For each diffusion step:
            - Predict token values with model
            - Calculate confidence scores
            - Select top-k tokens to unmask
            - Accumulate weights (log_w)
    3. Resample:
        If ESS < threshold:
            - Normalize particle weights
            - Sample particles proportional to weights
            - Reset weights
    4. Continue to next block

Finally: Select particle with highest cumulative confidence
```

## Key Differences from Standard Generation

| Aspect | Standard Generation | SMC Generation |
|--------|-------------------|----------------|
| Trajectories | Single | Multiple (particles) |
| Selection | Greedy/sampling at each step | Global confidence-based |
| Diversity | Limited | Maintained through particles |
| Resampling | None | Adaptive (ESS-based) |
| Quality | Good | Better (especially on hard tasks) |

## Advanced Topics

### Dynamic Token Selection

The `get_transfer_index_dynamic()` function provides an alternative selection strategy:

```python
def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
                               x, num_transfer_tokens, factor=1)
```

- Adjusts number of tokens to unmask based on confidence
- Factor controls aggressiveness of selection
- Can lead to adaptive step sizes

### Customizing Resampling

You can customize when resampling occurs:

```python
# Original: Resample every block when ESS < 0.5 * num_particles
if num_block % 1 == 0 and ess < 0.5 * num_particles:
    # resample...

# Example: Resample less frequently
if num_block % 2 == 0 and ess < 0.3 * num_particles:
    # resample...

# Example: Resample more aggressively  
if num_block % 1 == 0 and ess < 0.7 * num_particles:
    # resample...
```

## Troubleshooting

**Issue: Out of Memory**
- Reduce `num_particles` (e.g., from 4 to 2)
- Reduce `gen_length` or `block_length`
- Use gradient checkpointing (if available)

**Issue: Too Slow**
- Enable `use_cache=True` (should be default)
- Increase `block_length` (fewer resampling steps)
- Reduce `num_particles`

**Issue: Poor Quality**
- Increase `num_particles` (e.g., from 4 to 8)
- Adjust `temperature` (try 0.8-1.2)
- Ensure `threshold` is appropriate (0.9 works well)

## References

For more details, see:
- Paper: [Self-Rewarding Sequential Monte Carlo for Masked Diffusion Language Models](https://arxiv.org/abs/2602.01849)
- Main README: [../readme.md](../readme.md)
- Evaluation Guide: [eval.md](eval.md)
