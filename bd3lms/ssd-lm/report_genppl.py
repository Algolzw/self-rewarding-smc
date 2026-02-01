import json
import transformers
import sys
import glob
import os
sys.path.append('../')

import torch
from tqdm import tqdm
import metrics
from omegaconf import OmegaConf

config_dict = {
      'block_size': 25,
      'algo': 'ssd-lm',
      'training': {
          'importance_sampling': False,
          'sampling_eps': 1e-3,
      },
      'eval': {
          'perplexity_batch_size': 1,
          'gen_ppl_eval_model_name_or_path': 'gpt2-large'
      }
  }
  
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
METRIC_CONFIG = OmegaConf.create(config_dict)
METRICS = metrics.Metrics(METRIC_CONFIG)
METRICS.to('cuda')

for seqlen in {1025, 2050}:
  for total_t in {25, 1000}:
    # print all missing seeds using set difference
    all_files = glob.glob(f'/home/ma2238/bam-dlms/ssd-lm/generated_samples/samples_openwebtext-valid_bs25_l{seqlen}_t{total_t}_SEED*')
    print(f'missing seeds {set(range(1,301)) - set([int(f.split("SEED")[1]) for f in all_files])}')
    METRICS.reset()
    for seed in tqdm(range(1,5)):
        path = f'/home/ma2238/bam-dlms/ssd-lm/generated_samples/samples_openwebtext-valid_bs25_l{seqlen}_t{total_t}_SEED{seed}'
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        text = data['generated_samples'][0][f'sample_0']['sampled_sequences']
        METRICS.record_generative_perplexity(text, max_length=seqlen)
    print(f'sequence length {seqlen}, total timesteps {total_t}, num samples {len(METRICS.gen_ppls)}')
    print('generative perplexity:', torch.tensor(METRICS.gen_ppls).mean().item())
    print('entropy:', torch.tensor(METRICS.gen_entropies).mean().item())