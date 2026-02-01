import argparse
import json
import logging
import math
import os
import time

import accelerate
import datasets
import torch
from accelerate import Accelerator
from huggingface_hub.file_download import hf_hub_download
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM
from tqdm.auto import tqdm


def get_time_variables(t, total_t, device):  # cosine schedule
    def ft(small_t, big_t, s=1e-4):
        return torch.cos((small_t / big_t + s) / (1 + s) * math.pi / 2) ** 2

    alpha_t_bar = ft(t, total_t) / ft(torch.zeros(t.shape).to(device), total_t)
    alpha_t_minus_bar = ft(t - 1, total_t) / ft(torch.zeros(t.shape).to(device), total_t)
    beta_t = 1 - (alpha_t_bar / alpha_t_minus_bar)
    beta_t_til = (1 - alpha_t_minus_bar) / (1 - alpha_t_bar) * beta_t
    alpha_t = 1 - beta_t
    return alpha_t_bar, alpha_t_minus_bar, beta_t, beta_t_til, alpha_t


def logits_sampling_projection(logits, top_p, one_hot_value):
    assert len(logits.size()) == 3
    probs = torch.nn.functional.softmax(logits, dim=-1)
    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus = cum_sum_probs < top_p
    nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
    valid_indices = nucleus.scatter(2, indices, nucleus)

    filtered_logits = logits.masked_fill(valid_indices == 0, -float('Inf'))
    m = torch.distributions.categorical.Categorical(logits=filtered_logits)
    selected = m.sample()
    return 2 * one_hot_value * torch.nn.functional.one_hot(selected, logits.size(2)) - one_hot_value


def decode(
        context_input_ids,
        model,
        model_embedding_lut,
        embedding_sum_layer,
        timestep_layer,
        tokenizer,
        vocab_size,
        device,
        total_t=1000,
        max_seq_length=50,
        one_hot_value=5,
        decoding_block_size=25,
        projection_top_p=0.95,
        verbose=False,
):
    batch_size = 1
    unit_seq_len = decoding_block_size
    if context_input_ids is not None:
        unit_context_input_ids = context_input_ids.clone()
    else:
        unit_context_input_ids = None
    history_decode_ids = None
    equivalent_score = None
    dec_depth = max_seq_length // decoding_block_size
    total_seq_time = time.time()
    avg_block_time = 0.0
    avg_decode_step_time = 0.0
    for _ in tqdm(range(dec_depth), desc='Blocks', leave=False, disable=not verbose):
        block_time = time.time()
        unit_noise = one_hot_value * torch.normal(0, 1, size=(batch_size, unit_seq_len, vocab_size)).to(device)
        xt = unit_noise

        if unit_context_input_ids is not None:
            context_inputs_embeds = model_embedding_lut(unit_context_input_ids)
        else:
            context_inputs_embeds = None

        t_range = list(range(1, total_t + 1))
        t_range.reverse()
        progress_bar = tqdm(range(len(t_range)), desc='Decoding', leave=False, disable=not verbose)

        for t in t_range:
            decode_step_time = time.time()
            selected_t = torch.FloatTensor([t]).repeat(batch_size).to(device)
            alpha_t_bar, alpha_t_minus_bar, beta_t, beta_t_til, alpha_t = get_time_variables(selected_t, total_t,
                                                                                             device)
            zt = one_hot_value * torch.normal(0, 1, size=(batch_size, unit_seq_len, vocab_size)).to(device)

            perturbed_inputs_diralpha = xt
            perturbed_inputs_simplex = torch.nn.functional.softmax(perturbed_inputs_diralpha, dim=-1)

            perturbed_inputs_embeds = embedding_sum_layer(perturbed_inputs_simplex)
            t_progress = selected_t / total_t
            timestep_embeds = timestep_layer(t_progress.view(batch_size, 1, 1).repeat(1, unit_seq_len, 1))

            diffusion_embeds = perturbed_inputs_embeds + timestep_embeds
            if context_inputs_embeds is not None:
                diffusion_embeds = torch.cat((context_inputs_embeds, diffusion_embeds), dim=1)
            outputs = model(inputs_embeds=diffusion_embeds, output_hidden_states=False)
            equivalent_score = outputs.logits
            if unit_context_input_ids is not None:
                equivalent_score = equivalent_score[:, unit_context_input_ids.size(1):].contiguous()

            projected_logits = logits_sampling_projection(equivalent_score, top_p=projection_top_p,
                                                          one_hot_value=one_hot_value)

            xt = torch.sqrt(alpha_t_minus_bar).view(-1, 1, 1) * projected_logits
            xt = xt + torch.sqrt(1 - alpha_t_minus_bar).view(-1, 1, 1) * zt

            progress_bar.update(1)
            decode_step_time = time.time() - decode_step_time
            avg_decode_step_time += decode_step_time
        simplex = equivalent_score
        real_token_ids_list = torch.argmax(simplex, dim=-1).view(batch_size, unit_seq_len)
        # if unit_context_input_ids is None:
        #     unit_context_input_ids = real_token_ids_list
        # else:
        #     unit_context_input_ids = torch.cat((unit_context_input_ids, real_token_ids_list), dim=1)
        unit_context_input_ids = real_token_ids_list  # Fixed context size of last block
        if history_decode_ids is None:
            history_decode_ids = real_token_ids_list
        else:
            history_decode_ids = torch.cat((history_decode_ids, real_token_ids_list), dim=1)
        block_time = time.time() - block_time
        avg_block_time += block_time
    avg_block_time /= dec_depth
    avg_decode_step_time /= (dec_depth * total_t)
    if context_input_ids is not None:
        init_context_input_ids = context_input_ids.clone()
        context_sequences = tokenizer.batch_decode(init_context_input_ids.detach().to('cpu'))
    else:
        init_context_input_ids = None
        context_sequences = None
    sampled_sequences = tokenizer.batch_decode(history_decode_ids.clone().detach().to('cpu'))

    return {
        # 'history_decode_ids': history_decode_ids,
        # 'init_context_input_ids': init_context_input_ids,
        'sampled_sequences': sampled_sequences[0],
        'context_sequences': context_sequences[0] if context_sequences is not None else None,
        'total_seq_time': time.time() - total_seq_time,
        'avg_block_time': avg_block_time,
        'avg_decode_step_time': avg_decode_step_time,
    }

def main(args):
    # Get model
    accelerator = Accelerator()
    accelerate.utils.set_seed(args.seed, device_specific=True)
    logger.info(f'Using device: {accelerator.device}')

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path, from_tf=False, config=config)
    model.resize_token_embeddings(len(tokenizer))
    vocab_size = model.get_input_embeddings().weight.size(0)
    hidden_size = model.get_input_embeddings().weight.size(1)

    embedding_sum_layer = torch.nn.Linear(vocab_size, hidden_size, bias=False)
    _stdict = torch.load(os.path.join(hf_hub_download(args.model_name_or_path, 'embed_sum_layer.pt')))
    _stdict = dict(
        (_k[len('module.'):], _stdict[_k]) if _k.startswith('module.') else (_k, _stdict[_k]) for _k in _stdict)
    embedding_sum_layer.load_state_dict(_stdict)

    timestep_layer = torch.nn.Linear(1, hidden_size, bias=True)
    _stdict = torch.load(os.path.join(hf_hub_download(args.model_name_or_path, 'timestep_layer.pt')))
    _stdict = dict(
        (_k[len('module.'):], _stdict[_k]) if _k.startswith('module.') else (_k, _stdict[_k]) for _k in _stdict)
    timestep_layer.load_state_dict(_stdict)
    model, embedding_sum_layer, timestep_layer = accelerator.prepare(model, embedding_sum_layer, timestep_layer)
    model.eval()
    model_embedding_lut = accelerator.unwrap_model(model).get_input_embeddings()

    # Get dataset and dataloader to have prompts
    prompt_dataset = datasets.load_dataset(args.prompt_dataset, split='test') if args.use_prompt_dataset else None

    # Generate text
    generated = []
    total_time = time.time()
    for i in tqdm(range(args.num_samples_to_generate), desc='Samples'):
        if prompt_dataset is not None:
            prompt = prompt_dataset[i]['text']
            if prompt[0] != ' ':  # prepend a space to the prompt if necessary
                prompt = f' {prompt}'  # can use ' ' or '\n\n' as prefix to the prompt
            input_ids = torch.LongTensor(tokenizer.encode(prompt, add_special_tokens=False)).to(accelerator.device)
            input_ids = input_ids[:args.decoding_block_size]
        else:
            input_ids = torch.LongTensor([tokenizer.bos_token_id]).to(accelerator.device)
        input_ids = input_ids.unsqueeze(0)

        # start sampling from SSD-LM
        decode_dict = decode(
            context_input_ids=input_ids,
            total_t=args.total_t,
            model_embedding_lut=model_embedding_lut,
            embedding_sum_layer=embedding_sum_layer,
            timestep_layer=timestep_layer,
            model=model,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            device=accelerator.device,
            max_seq_length=args.max_seq_length,
            one_hot_value=args.one_hot_value,
            decoding_block_size=args.decoding_block_size,
            projection_top_p=args.projection_top_p,
            verbose=args.verbose,
        )
        generated.append(decode_dict)
        if args.verbose:
            logger.info(f'\nSample{i+1}:\n\tContext: {decode_dict["context_sequences"]}\n\tGenerated sample: {decode_dict["sampled_sequences"]}')
    total_time = time.time() - total_time
    with open(args.generated_samples_outfile, 'w') as f:
        json.dump({
            **vars(args),
            'total_time': total_time,
            'generated_samples': [{f'sample_{i}': g} for i, g in enumerate(generated)]},
            f, indent=2)

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    # logger.setLevel(logging.INFO, format='%(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('--generated_samples_outfile', type=str, default='generated_samples.json')
    parser.add_argument('--num_samples_to_generate', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--use_prompt_dataset', type=bool, default=False)
    parser.add_argument('--prompt_dataset', type=str, default='lm1b')
    parser.add_argument('--model_name_or_path', type=str, default='xhan77/ssdlm')
    parser.add_argument('--use_slow_tokenizer', type=bool, default=True)
    parser.add_argument('--max_seq_length', type=int, default=4100)
    parser.add_argument('--one_hot_value', type=int, default=5)
    parser.add_argument('--decoding_block_size', type=int, default=25)
    parser.add_argument('--total_t', type=int, default=1000)
    parser.add_argument('--projection_top_p', type=float, default=0.95)
    parser.add_argument('--verbose', type=bool, default=True)

    opts = parser.parse_args()
    main(opts)
