import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    # num_transfer_tokens = torch.cumsum(num_transfer_tokens, dim=1)
    return num_transfer_tokens


@ torch.inference_mode()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None, num_particles=4):
    '''
    Self-Rewarding SMC Generation (without prefix caching).
    
    This function implements Sequential Monte Carlo (SMC) for diffusion-based text generation.
    It maintains multiple generation trajectories (particles) and uses confidence-based 
    resampling to focus on high-quality samples.
    
    Args:
        model: Mask predictor (LLaDA model).
        prompt: A tensor of shape (1, L) containing the input prompt token IDs.
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length (number of tokens to generate).
        block_length: Block length, less than or equal to gen_length. If less than gen_length, 
                     it means using semi-autoregressive remasking.
        temperature: Categorical distribution sampling temperature. 0=greedy, >0=stochastic.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        threshold: Optional confidence threshold for dynamic token selection.
        factor: Optional factor for dynamic token selection strategy.
        num_particles: The number of particles (parallel generation trajectories).
    
    Returns:
        tuple: (generated_sequence, nfe) where:
            - generated_sequence: The best particle's tokens (shape: 1 x (prompt_len + gen_length))
            - nfe: Number of function evaluations (forward passes through the model)
    '''
    # Initialize particles: Create num_particles copies of the sequence, all starting with prompt
    # and followed by mask tokens for generation
    x = torch.full((prompt.shape[0] * num_particles, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    # Track log-probabilities for each token in each particle
    logp = torch.zeros_like(x, dtype=torch.float32, device=x.device)
    # Initialize particle weights in log-space (log_w stores cumulative confidence)
    log_w = torch.zeros(num_particles).to(model.device)

    # Divide generation into blocks for efficiency
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    # Distribute diffusion steps evenly across blocks
    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 1  # Track number of function evaluations (model forward passes)
    
    # Generate each block sequentially
    for num_block in range(num_blocks):
        # Define current block boundaries
        idx_s = prompt.shape[1] + num_block * block_length
        idx_t = prompt.shape[1] + (num_block + 1) * block_length
        
        # Identify masked positions in current block
        block_mask_index = (x[:, idx_s:idx_t] == mask_id)
        # Precompute how many tokens to unmask at each diffusion step
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0

        # Iteratively unmask tokens in the current block
        while True:
            if (x[:, idx_s:idx_t] == mask_id).sum() == 0:
                break  # All tokens in block have been unmasked

            # === Propagation Step: Predict and unmask tokens ===
            mask_index = (x == mask_id)
            logits = model(x).logits  # Get model predictions for all positions
            mask_index[:, idx_t:] = 0  # Only consider current block
            
            # Select which tokens to unmask based on confidence
            x0, transfer_index, x0_logp = get_transfer_index(
                logits, temperature, remasking, mask_index, x, 
                num_transfer_tokens[:, i] if threshold is None else None, threshold)
            
            # Update tokens and their log-probabilities
            x[transfer_index] = x0[transfer_index]
            logp[transfer_index] = x0_logp[transfer_index]
            
            # === Weight Update: Accumulate confidence scores ===
            # Add log-prob of newly unmasked tokens to each particle's weight
            log_w = log_w + (x0_logp * transfer_index.float()).sum(dim=1)
            i += 1
            nfe += 1

        # === SMC Resampling: Adaptive particle resampling based on ESS ===
        if num_particles > 1:
            # Normalize weights (convert from log-space to probabilities)
            weights = torch.exp(log_w - log_w.max())
            weights = weights / weights.sum()

            # Calculate Effective Sample Size (ESS)
            # ESS measures diversity: high ESS = diverse particles, low ESS = few dominant particles
            ess = 1.0 / (weights.pow(2).sum())
            
            # Resample if particle diversity is too low
            if num_block % 1 == 0 and ess < 0.5 * num_particles:
                print(f"Normal Resampling at block {num_block}, with ess: {ess:.2f}")
                # Multinomial resampling: duplicate high-weight particles, discard low-weight ones
                k_idx = torch.multinomial(weights, num_samples=num_particles, replacement=True).squeeze(-1)
                x = x[k_idx]; logp = logp[k_idx]; 
                log_w.zero_()  # Reset weights after resampling

        tps = block_length // i  # tokens_per_step
        print(f"num_block: {num_block+1}, block length: {block_length}, diffusion steps: {i}, tokens/step: {tps}, num_particles: {num_particles}")

    # Print average confidence for each particle
    print(logp[:, prompt.shape[1]:].exp().mean(dim=1))
    
    # Select the particle with highest total confidence (sum of log-probs)
    idx = torch.argmax(logp.sum(dim=1))
    return x[idx:idx+1], nfe

@ torch.inference_mode()
def generate_with_prefix_cache_smc(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None, num_particles=4):
    '''
    Self-Rewarding SMC Generation with Prefix Caching (optimized version).
    
    This is an optimized version of the SMC generation that caches the key-value pairs
    from the prompt to avoid redundant computations. This significantly reduces the
    number of function evaluations (NFE) and speeds up generation.
    
    Args:
        model: Mask predictor (LLaDA model) that supports use_cache=True.
        prompt: A tensor of shape (1, L) containing the input prompt token IDs.
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length (number of tokens to generate).
        block_length: Block length, less than or equal to gen_length. If less than gen_length, 
                     it means using semi-autoregressive remasking.
        temperature: Categorical distribution sampling temperature. 0=greedy, >0=stochastic.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        threshold: Optional confidence threshold for dynamic token selection.
        factor: Optional factor for dynamic token selection strategy.
        num_particles: The number of particles (parallel generation trajectories).
    
    Returns:
        tuple: (generated_sequence, nfe) where:
            - generated_sequence: The best particle's tokens (shape: 1 x (prompt_len + gen_length))
            - nfe: Number of function evaluations (forward passes through the model)
    '''
    # Initialize particles: Create num_particles copies of the sequence
    x = torch.full((prompt.shape[0] * num_particles, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    # Track log-probabilities and particle weights
    logp = torch.zeros_like(x, dtype=torch.float32, device=x.device)
    log_w = torch.zeros(num_particles).to(model.device)

    # Divide generation into blocks
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0  # Track number of function evaluations
            
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        # === First forward pass with caching enabled ===
        # Cache key-value pairs from prompt and previous blocks to avoid recomputation
        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0

        # Select tokens to unmask (using either fixed or dynamic strategy)
        if factor is None:
            x0, transfer_index, x0_logp = get_transfer_index(
                output.logits, temperature, remasking, mask_index, x, 
                num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index, x0_logp = get_transfer_index_dynamic(
                output.logits, temperature, remasking, mask_index, x, None, factor)
        
        # Update tokens and accumulate weights
        x[transfer_index] = x0[transfer_index]
        logp[transfer_index] = x0_logp[transfer_index]
        log_w = log_w + (x0_logp * transfer_index.float()).sum(dim=1)

        # === Truncate cached key-values to only include prompt + previous blocks ===
        # This allows us to only process the current block in subsequent steps
        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
        
        past_key_values = new_past_key_values
        nfe += 1
        
        i = 1
        # === Subsequent diffusion steps with cached prefix ===
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break  # All tokens in block have been unmasked
            nfe += 1
            
            # Only consider masks in current block
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            # Forward pass ONLY on current block (prompt is cached)
            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            # Select tokens to unmask
            if factor is None:
                x0, transfer_index, x0_logp = get_transfer_index(
                    logits, temperature, remasking, mask_index, 
                    x[:, current_block_start:], 
                    num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index, x0_logp = get_transfer_index_dynamic(
                    logits, temperature, remasking, mask_index, 
                    x[:, current_block_start:], None, factor)
            
            # Update tokens and weights
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            logp[:, current_block_start:][transfer_index] = x0_logp[transfer_index]
            log_w = log_w + (x0_logp * transfer_index.float()).sum(dim=1)

            i += 1

        # === SMC Resampling ===
        if num_particles > 1:
            # Normalize weights
            weights = torch.exp(log_w - log_w.max())
            weights = weights / weights.sum()
            
            # Calculate Effective Sample Size (ESS)
            ess = 1.0 / (weights.pow(2).sum())
            
            # Resample if diversity is low
            if num_block % 1 == 0 and ess < 0.5 * num_particles:
                print(f"Resampling at block {num_block}, with ess: {ess:.2f}")
                k_idx = torch.multinomial(weights, num_samples=num_particles, replacement=True).squeeze(-1)
                x = x[k_idx]; logp = logp[k_idx]; log_w.zero_()


        tps = block_length // i  # tokens_per_step
        print(f"num_block: {num_block+1}, block length: {block_length}, diffusion steps: {i}, tokens/step: {tps}, num_particles: {num_particles}")
    
    # Print average confidence for each particle
    print(logp[:, prompt.shape[1]:].exp().mean(dim=1))
    
    # Select the particle with highest total confidence
    idx = torch.argmax(logp.sum(dim=1))
    return x[idx:idx+1], nfe

def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    '''
    Selects which masked tokens to unmask based on model confidence.
    
    Args:
        logits: Model predictions for all positions (shape: batch x length x vocab_size).
        temperature: Temperature for Gumbel noise sampling.
        remasking: Strategy for selecting tokens ('low_confidence' or 'random').
        mask_index: Boolean mask indicating which positions are still masked.
        x: Current sequence state.
        num_transfer_tokens: Number of tokens to unmask at this step (per particle).
        threshold: Optional confidence threshold - only unmask if confidence > threshold.
    
    Returns:
        tuple: (x0, transfer_index, x0_logp) where:
            - x0: Predicted tokens for all positions
            - transfer_index: Boolean mask of which tokens to actually unmask
            - x0_logp: Log-probabilities of the predicted tokens
    '''
    # Apply Gumbel noise and get predictions
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l
    
    # Calculate confidence scores for predictions
    if remasking == 'low_confidence':
        log_pt = F.log_softmax(logits.to(torch.float64), dim=-1)
        x0_logp = torch.squeeze(
            torch.gather(log_pt, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
        x0_p = x0_logp.exp()
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    # Only consider predictions for masked positions
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    # Select top-k most confident tokens to unmask
    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        # If threshold is set, consider all masked tokens
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    
    for j in range(confidence.shape[0]):
        # Get indices of top-k most confident predictions
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        
        # Apply confidence threshold if specified
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    
    return x0, transfer_index, x0_logp.float()

def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
    '''
    Dynamic token selection strategy that adjusts the number of tokens to unmask
    based on model confidence. More adaptive than fixed schedule.
    
    Args:
        logits: Model predictions.
        temperature: Temperature for Gumbel noise.
        remasking: Selection strategy ('low_confidence' or 'random').
        mask_index: Boolean mask of masked positions.
        x: Current sequence state.
        num_transfer_tokens: Not used in dynamic mode.
        factor: Controls aggressiveness of dynamic selection (default: 1).
    
    Returns:
        tuple: (x0, transfer_index, x0_logp) - predicted tokens, selection mask, log-probs
    '''
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
    if remasking == 'low_confidence':
        log_p = F.log_softmax(logits.to(torch.float64), dim=-1)
        x0_logp = torch.squeeze(
            torch.gather(log_p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
        x0_p = x0_logp.exp()
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    
    for j in range(confidence.shape[0]):
        ns=list(range(1,num_transfer_tokens[j]+1))
        es=[factor/(n+1) for n in ns]
        threshs=[1-e for e in es]

        # at least one token is transferred
        threshs[0]=-1
        sorted_confidence=torch.sort(confidence[j][mask_index[j]],dim=-1,descending=True)[0]
        assert len(sorted_confidence)==len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i]<threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs)-1:
            top_i+=1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index, x0_logp.float()





