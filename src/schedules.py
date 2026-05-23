"""
schedules.py  —  Discrete diffusion corruption schedules

Two schedules are implemented and compared in this project:

  AbsorbingSchedule  — each token is independently replaced with [MASK]
                       with probability beta(t).  The clean token is
                       "absorbed" and can never be recovered from the
                       noisy token alone; the model must predict it from
                       context. Used in MDLM and D3PM absorbing variant.

  UniformSchedule    — each token is independently replaced with a token
                       drawn uniformly from the vocabulary.  The original
                       token may still appear by chance (self-replacement),
                       making the learning signal noisier.  Used in D3PM
                       uniform variant.

Both schedules expose the same API:
  q_sample(x0, t)           →  x_t  (corrupt clean sequence at time t)
  posterior_mean(x_t, x0, t) →  logits for each position
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Literal


@dataclass
class ScheduleConfig:
    T: int   = 200          # total diffusion timesteps
    kind: Literal["absorbing", "uniform"] = "absorbing"
    # Linear beta schedule: beta linearly increases from beta_min → beta_max
    beta_min: float = 1e-4
    beta_max: float = 0.02


class AbsorbingSchedule:
    """
    Forward process:  q(x_t | x_0) = Bernoulli-mask each token.
    At time t the probability a given token has been masked is alpha(t).
    
    alpha(t) = 1 - prod_{s=1}^{t} (1 - beta(s))      (cumulative keep-prob complement)
    
    In code we work with  alpha_bar[t] = cumulative masking probability.
    """
    name = "absorbing"

    def __init__(self, cfg: ScheduleConfig, vocab_size: int, mask_id: int, pad_id: int):
        self.T          = cfg.T
        self.vocab_size = vocab_size
        self.mask_id    = mask_id
        self.pad_id     = pad_id

        # Linear beta schedule
        betas = torch.linspace(cfg.beta_min, cfg.beta_max, cfg.T)          # (T,)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)                            # (T,) keep-prob
        # alpha_bar[t] = probability token is still CLEAN at step t
        self.register_buffers(alpha_bar, betas)

    def register_buffers(self, alpha_bar, betas):
        self.alpha_bar = alpha_bar   # shape (T,)
        self.betas     = betas

    def _get_ab(self, t: torch.Tensor) -> torch.Tensor:
        """Gather alpha_bar for a batch of timesteps t (LongTensor, shape B)."""
        return self.alpha_bar.to(t.device)[t]   # (B,)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Corrupt x0 at timestep t.
        x0 : (B, L)  clean token ids
        t  : (B,)    integer timesteps in [0, T-1]
        Returns x_t : (B, L) with some tokens replaced by mask_id
        """
        B, L = x0.shape
        ab   = self._get_ab(t).view(B, 1)          # (B, 1) keep probability
        # Mask token with probability (1 - alpha_bar[t])
        noise_mask = torch.rand(B, L, device=x0.device) > ab     # True → mask
        # Don't corrupt padding positions
        pad_mask   = (x0 == self.pad_id)
        noise_mask = noise_mask & ~pad_mask

        x_t = x0.clone()
        x_t[noise_mask] = self.mask_id
        return x_t

    def masking_rate_at(self, t_frac: float) -> float:
        """Return expected masking rate at fractional time t_frac ∈ [0,1]."""
        t_idx = min(int(t_frac * self.T), self.T - 1)
        return float(1.0 - self.alpha_bar[t_idx])


class UniformSchedule:
    """
    Forward process: q(x_t | x_0).
    Each token independently:
      - stays the same with probability (1 - beta_t)
      - is replaced by a uniformly random vocabulary token with prob beta_t
      (including possibly the same token by chance)
    
    Cumulative: at time t, prob of token being corrupted = alpha_bar[t]
    When corrupted it is drawn Uniform({0, …, V-1}).
    """
    name = "uniform"

    def __init__(self, cfg: ScheduleConfig, vocab_size: int, mask_id: int, pad_id: int):
        self.T          = cfg.T
        self.vocab_size = vocab_size
        self.mask_id    = mask_id
        self.pad_id     = pad_id

        betas = torch.linspace(cfg.beta_min, cfg.beta_max, cfg.T)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.alpha_bar = alpha_bar
        self.betas     = betas

    def _get_ab(self, t: torch.Tensor) -> torch.Tensor:
        return self.alpha_bar.to(t.device)[t]

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Corrupt x0 at timestep t.
        x0 : (B, L)
        t  : (B,)
        Returns x_t : (B, L) with some tokens replaced uniformly at random.
        """
        B, L = x0.shape
        ab   = self._get_ab(t).view(B, 1)         # (B,1) keep-prob

        corrupt_mask = torch.rand(B, L, device=x0.device) > ab
        pad_mask     = (x0 == self.pad_id)
        corrupt_mask = corrupt_mask & ~pad_mask

        # Draw random replacement tokens (uniform over full vocab)
        random_tokens = torch.randint(0, self.vocab_size, (B, L), device=x0.device)

        x_t = x0.clone()
        x_t[corrupt_mask] = random_tokens[corrupt_mask]
        return x_t

    def masking_rate_at(self, t_frac: float) -> float:
        t_idx = min(int(t_frac * self.T), self.T - 1)
        return float(1.0 - self.alpha_bar[t_idx])


def build_schedule(kind: str, cfg: ScheduleConfig, vocab_size: int,
                   mask_id: int, pad_id: int):
    if kind == "absorbing":
        return AbsorbingSchedule(cfg, vocab_size, mask_id, pad_id)
    elif kind == "uniform":
        return UniformSchedule(cfg, vocab_size, mask_id, pad_id)
    else:
        raise ValueError(f"Unknown schedule: {kind!r}")
