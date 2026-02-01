import torch
import transformers

from models.hf import BD3LMConfig
from models.hf import BD3LM

BLOCK_SIZE=16
BD3LMConfig.register_for_auto_class()
BD3LM.register_for_auto_class('AutoModelForMaskedLM')
device = 'cuda'
tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
name_or_path = f'kuleshov-group/bd3lm-owt-block_size{BLOCK_SIZE}'

config = BD3LMConfig(
    block_size=BLOCK_SIZE,
    vocab_size=tokenizer.vocab_size+1,
    model_length=1024,
    hidden_dim=768,
    cond_dim=128,
    n_blocks=12,
    n_heads=12,
    dropout=0.1,
    time_conditioning=False,
    return_dict=False
)

model = BD3LM(config)

model.config._name_or_path = name_or_path
model.config.auto_map = {
    'AutoConfig': f'{name_or_path}--configuration.BD3LMConfig',
    'AutoModelForMaskedLM': f'{name_or_path}--modeling_bd3lm.BD3LM',
}

# load ema params
# ckpt_path=f'/share/kuleshov/ma2238/textdiffusion/owt-sar4-v2/checkpoints/11-150000.ckpt'
# ckpt_path=f'/share/kuleshov/ma2238/textdiffusion/owt-sar8-v3/checkpoints/10-150000.ckpt'
ckpt_path=f'/share/kuleshov/ma2238/textdiffusion/owt-sar16-v2/checkpoints/13-150000.ckpt'

ckpt = torch.load(ckpt_path, weights_only=False)
model = model.to(device)
model.load_state_dict(ckpt['state_dict'])
ema_params = ckpt['ema']['shadow_params']
for s_param, param in zip(ema_params, model.parameters()):
    if param.requires_grad:
        param.data.copy_(s_param.data)
model.push_to_hub(name_or_path, private=True)