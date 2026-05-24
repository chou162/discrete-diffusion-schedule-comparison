"""
Two corruption schedules for discrete diffusion.

Absorbing: corrupted tokens become [MASK], so the model knows exactly
which positions to predict. Uniform: corrupted tokens become random
vocabulary tokens, giving the model no explicit signal about where
corruptions are.

Both use a linear noise schedule with a 0.1 floor so there's always
some corruption signal even at low timesteps.
"""

import torch
from dataclasses import dataclass
from enum import Enum


class ScheduleType(Enum):
    ABSORBING = "absorbing"
    UNIFORM   = "uniform"


@dataclass
class CorruptionSchedule:

    schedule_type : ScheduleType
    T             : int    
    vocab_size    : int
    mask_id       : int    # only used by ABSORBING
    pad_id        : int    

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Corruption probability at timestep t, floored at 0.1."""
        
        return 0.1 + 0.9 * (t.float() / self.T)

    def corrupt(
        self,
        x0 : torch.Tensor,   
        t  : torch.Tensor,   
    ) -> torch.Tensor:
        """Corrupt x0 at noise level t, skipping padding."""
        
        B, L   = x0.shape
        device = x0.device

        a = self.alpha(t).to(device).view(B, 1).expand(B, L)

        corrupt_mask = torch.bernoulli(a).bool()

        # Never corrupt padding
        corrupt_mask = corrupt_mask & (x0 != self.pad_id)

        if self.schedule_type == ScheduleType.ABSORBING:
            return self._absorbing_corrupt(x0, corrupt_mask)
        else:
            return self._uniform_corrupt(x0, corrupt_mask, device)

    def _absorbing_corrupt(
        self,
        x0          : torch.Tensor,
        corrupt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Replace corrupted positions with [MASK]."""
        xt = x0.clone()
        xt[corrupt_mask] = self.mask_id
        return xt

    def _uniform_corrupt(
        self,
        x0          : torch.Tensor,
        corrupt_mask: torch.Tensor,
        device      : torch.device,
    ) -> torch.Tensor:
        """Replace corrupted positions with a uniformly random vocabulary token."""
        xt        = x0.clone()
        n_corrupt = corrupt_mask.sum().item()
        if n_corrupt > 0:
            random_tokens = torch.randint(
                low=0, high=self.vocab_size,
                size=(int(n_corrupt),), device=device
            )
            xt[corrupt_mask] = random_tokens
        return xt

    def should_denoise(self, xt: torch.Tensor) -> torch.Tensor:
        """Positions the model should predict (absorbing: masked only, uniform: everything)."""
        
        if self.schedule_type == ScheduleType.ABSORBING:
            return xt == self.mask_id
        else:
            return xt != self.pad_id

    def name(self) -> str:
        return self.schedule_type.value

def make_schedule(
    schedule_type : str | ScheduleType,
    T             : int,
    vocab_size    : int,
    mask_id       : int,
    pad_id        : int,
) -> CorruptionSchedule:
    if isinstance(schedule_type, str):
        schedule_type = ScheduleType(schedule_type)
    return CorruptionSchedule(
        schedule_type=schedule_type,
        T=T,
        vocab_size=vocab_size,
        mask_id=mask_id,
        pad_id=pad_id,
    )
