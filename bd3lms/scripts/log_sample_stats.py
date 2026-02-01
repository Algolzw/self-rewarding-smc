import csv
import os
import numpy as np

dir_name = 'sample_logs'

# report averages
for fname in os.listdir(dir_name):
  with open(f'{dir_name}/{fname}', 'r') as f:
    fieldnames = ['gen_ppl', 'nfes', 'entropy', 'gen_lengths', 'samples', 'seed']
    reader = csv.DictReader(f, fieldnames=fieldnames)
    rows = list(reader)
    gen_ppl = [float(eval(row['gen_ppl'])) for row in rows if row['gen_ppl'] != '[nan]']
    try:
      nfes = [float(eval(row['nfes'])) for row in rows]
    except:
      nfes = [0 for row in rows]
    entropy = [float(eval(row['entropy'])) for row in rows]
    gen_lengths = [float(eval(row['gen_lengths'])) for row in rows]

    print(f'{fname}:')
    print(f'Average over {len(gen_ppl)} samples\n')
    print(f'Generative Perplexity: {sum(gen_ppl) / len(gen_ppl)}')
    print(f'NFEs: {sum(nfes) / len(nfes)}')
    print(f'Entropy: {sum(entropy) / len(entropy)}')
    print(f'Median sample length: {np.median(gen_lengths)}')
    print(f'Max sample length: {max(gen_lengths)}')
    print('\n-----------------------\n')