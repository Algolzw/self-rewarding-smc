"""BD3LM config for Hugging Face.

"""

import transformers


class BD3LMConfig(transformers.PretrainedConfig):
  """Hugging Face configuration class for BD3LM."""
  model_type = "bd3lm"

  def __init__(
    self,
    block_size: int = 1,
    vocab_size: int = 50258,
    model_length: int = 1024,
    cross_attn: bool = True,
    adaln: bool = True,
    attn_backend: str = 'flex',
    causal: bool = False,
    hidden_dim: int = 768,
    cond_dim: int = 129,
    n_blocks: int = 12,
    n_heads: int = 12,
    dropout: float = 0.1,
    time_conditioning: bool = False,
    var_min: bool = True,
    sampling_eps_min: float = 1e-3,
    sampling_eps_max: float = 0.999,
    ** kwargs):
    super().__init__(**kwargs)
    self.block_size = block_size
    self.cross_attn = cross_attn
    self.adaln = adaln
    self.attn_backend = attn_backend
    self.causal = causal
    self.vocab_size = vocab_size
    self.model_length = model_length
    self.hidden_dim = hidden_dim
    self.cond_dim = cond_dim
    self.n_blocks = n_blocks
    self.n_heads = n_heads
    self.dropout = dropout
    self.time_conditioning = time_conditioning
    self.var_min = var_min
    self.sampling_eps_min = sampling_eps_min
    self.sampling_eps_max = sampling_eps_max