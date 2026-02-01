import abc
import torch
import torch.nn as nn

def get_noise(config, noise_type=None):
  if noise_type is None:
    noise_type = config.noise.type

  if noise_type == 'loglinear':
    return LogLinearNoise()
  elif noise_type == 'square':
    return ExpNoise(2)
  elif noise_type == 'square_root':
    return ExpNoise(0.5)
  elif noise_type == 'log':
    return LogarithmicNoise()
  elif noise_type == 'cosine':
    return CosineNoise()
  else:
    raise ValueError(f'{noise_type} is not a valid noise')

class Noise(abc.ABC, nn.Module):
  """
  Baseline forward method to get the total + rate of noise at a timestep
  """
  
  def forward(self, t):
    return self.compute_loss_scaling_and_move_chance(t)
  
class CosineNoise(Noise):
  def __init__(self, eps=1e-3):
    super().__init__()
    self.eps = eps

  def compute_loss_scaling_and_move_chance(self, t):
    cos = - (1 - self.eps) * torch.cos(t * torch.pi / 2)
    sin = - (1 - self.eps) * torch.sin(t * torch.pi / 2)
    move_chance = cos + 1
    loss_scaling = sin / (move_chance + self.eps) * torch.pi / 2
    return loss_scaling, move_chance

class ExpNoise(Noise):
  def __init__(self, exp=2, eps=1e-3):
    super().__init__()
    self.eps = eps
    self.exp = exp
  
  def compute_loss_scaling_and_move_chance(self, t):
    move_chance = torch.pow(t, self.exp)
    move_chance = torch.clamp(move_chance, min=self.eps)
    loss_scaling = - (self.exp * torch.pow(t, self.exp-1)) / move_chance
    return loss_scaling, move_chance

class LogarithmicNoise(Noise):
  def __init__(self, eps=1e-3):
    super().__init__()
    self.eps = eps

  def compute_loss_scaling_and_move_chance(self, t):
    move_chance = torch.log1p(t) / torch.log(torch.tensor(2.0))
    loss_scaling = - 1 / (move_chance * torch.log(torch.tensor(2.0)) * (1 + t))
    return loss_scaling, move_chance

class LogLinearNoise(Noise):
  """Log Linear noise schedule.
  
  Built such that 1 - 1/e^(n(t)) interpolates between 0 and
  ~1 when t varies from 0 to 1. Total noise is
  -log(1 - (1 - eps) * t), so the sigma will be
  (1 - eps) * t.
  """
  def __init__(self, eps=1e-3):
    super().__init__()
    self.eps = eps
    self.sigma_max = self.total_noise(torch.tensor(1.0))
    self.sigma_min = self.eps + self.total_noise(torch.tensor(0.0))

  def rate_noise(self, t):
    return (1 - self.eps) / (1 - (1 - self.eps) * t)

  def total_noise(self, t):
    return -torch.log1p(-(1 - self.eps) * t)

  def compute_loss_scaling_and_move_chance(self, t):
    loss_scaling = - 1 / t
    return loss_scaling, t