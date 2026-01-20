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
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        num_particles: The number of particles
    '''
    x = torch.full((prompt.shape[0] * num_particles, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    logp = torch.zeros_like(x, dtype=torch.float32, device=x.device)
    # weight initialization in log-space
    log_w = torch.zeros(num_particles).to(model.device)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 1
    for num_block in range(num_blocks):
        idx_s = prompt.shape[1] + num_block * block_length
        idx_t = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, idx_s:idx_t] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0

        while True:
            if (x[:, idx_s:idx_t] == mask_id).sum() == 0:
                break

            # propagation
            mask_index = (x == mask_id)
            logits = model(x).logits
            mask_index[:, idx_t:] = 0
            x0, transfer_index, x0_logp = get_transfer_index(
                logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
            x[transfer_index] = x0[transfer_index]
            logp[transfer_index] = x0_logp[transfer_index]
            log_w = log_w + (x0_logp * transfer_index.float()).sum(dim=1) # weight on transfer tokens
            i += 1
            nfe += 1

        # SMC Resampling
        if num_particles > 1:
            # weighting
            weights = torch.exp(log_w - log_w.max())
            weights = weights / weights.sum()

            # resampling
            ess = 1.0 / (weights.pow(2).sum())
            if num_block % 1 == 0 and ess < 0.5 * num_particles:
                print(f"Normal Resampling at block {num_block}, with ess: {ess:.2f}")
                k_idx = torch.multinomial(weights, num_samples=num_particles, replacement=True).squeeze(-1)
                x = x[k_idx]; logp = logp[k_idx]; log_w = log_w[k_idx] #log_w.zero_()
                log_w -= log_w.mean()

        tps = block_length // i # tokens_per_step
        print(f"num_block: {num_block+1}, block length: {block_length}, diffusion steps: {i}, tokens/step: {tps}, num_particles: {num_particles}")

    print(logp[:, prompt.shape[1]:].exp().mean(dim=1))
    idx = torch.argmax(logp.sum(dim=1))
    return x[idx:idx+1], nfe

@ torch.inference_mode()
def generate_with_prefix_cache_smc(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None, num_particles=4):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0] * num_particles, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    # weight initialization in log-space
    logp = torch.zeros_like(x, dtype=torch.float32, device=x.device)
    log_w = torch.zeros(num_particles).to(model.device)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
            
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0

        if factor is None:
            x0, transfer_index, x0_logp = get_transfer_index(
                output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index, x0_logp = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]
        logp[transfer_index] = x0_logp[transfer_index]
        # weight accumulation
        log_w = log_w + (x0_logp * transfer_index.float()).sum(dim=1) # weight on transfer tokens

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
        
        past_key_values = new_past_key_values
        nfe += 1
        
        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            nfe += 1
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if factor is None:
                x0, transfer_index, x0_logp = get_transfer_index(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index, x0_logp = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], None, factor)
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            logp[:, current_block_start:][transfer_index] = x0_logp[transfer_index]
            # weighting
            log_w = log_w + (x0_logp * transfer_index.float()).sum(dim=1) # weight on transfer tokens

            i += 1

        # SMC Resampling
        if num_particles > 1:
            weights = torch.exp(log_w - log_w.max())
            weights = weights / weights.sum()
            # weights = torch.softmax(log_w, dim=0)
            # resampling
            ess = 1.0 / (weights.pow(2).sum())
            if num_block % 1 == 0 and ess < 0.5 * num_particles:
                print(f"Resampling at block {num_block}, with ess: {ess:.2f}")
                k_idx = torch.multinomial(weights, num_samples=num_particles, replacement=True).squeeze(-1)
                x = x[k_idx]; logp = logp[k_idx]; log_w = log_w[k_idx]
                log_w -= log_w.mean()
                # log_w.zero_()

        tps = block_length // i # tokens_per_step
        print(f"num_block: {num_block+1}, block length: {block_length}, diffusion steps: {i}, tokens/step: {tps}, num_particles: {num_particles}")
    print(logp[:, prompt.shape[1]:].exp().mean(dim=1))
    idx = torch.argmax(logp.sum(dim=1))
    return x[idx:idx+1], nfe

def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
    if remasking == 'low_confidence':
        log_pt = F.log_softmax(logits.to(torch.float64), dim=-1)
        x0_logp = torch.squeeze(
            torch.gather(log_pt, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
        x0_p = x0_logp.exp()
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index, x0_logp.float()

def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
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





