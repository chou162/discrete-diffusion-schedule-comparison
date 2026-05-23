"""
training/corruption.py
-----------------------
The two corruption schedules we compare:

  1. Absorbing (mask-based)
     At timestep t, each token is independently replaced with [MASK]
     with probability alpha(t) = t / T.

     This is the schedule used in MDLM (Sahoo et al., 2024) and BERT-style
     masked diffusion. The model learns to "fill in" masked positions.

     Key property: corrupted tokens carry NO information about the original.
     The model always knows *which* positions need recovery (they're [MASK]).

  2. Uniform noise
     At timestep t, each token is replaced with a uniformly random
     vocabulary token (including possibly itself) with probability alpha(t).

     This is the schedule from the original D3PM paper and SEDD. The model
     can't distinguish "already correct" from "randomly wrong" — it must
     attend to context to decide.

     Key property: corrupted tokens carry ~0 bits about the original at
     high noise levels, but the model gets no explicit "corrupted here" signal.
     This is harder to learn from and tests different inductive biases.

Both schedules use a *linear* noise schedule: alpha(t) = t / T.
You can swap in cosine or sqrt schedules by changing alpha().
"""

import torch
from dataclasses import dataclass
from enum import Enum


class ScheduleType(Enum):
    ABSORBING = "absorbing"
    UNIFORM   = "uniform"


@dataclass
class CorruptionSchedule:
    """Encapsulates a corruption schedule with forward/reverse logic."""

    schedule_type : ScheduleType
    T             : int    # total diffusion timesteps
    vocab_size    : int
    mask_id       : int    # only used by ABSORBING
    pad_id        : int    # pad positions are never corrupted

    # ── Noise schedule ──────────────────────────────────────────────────────────

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """
        Corruption probability at timestep t.
        Linear schedule: alpha(t) = t / T.

        Shape: same as t (scalar or batch).
        Range: [0, 1] — 0 = clean, 1 = fully corrupted.
        """
        return 0.1 + 0.9 * (t.float() / self.T)

    # ── Forward process: q(x_t | x_0) ──────────────────────────────────────────

    def corrupt(
        self,
        x0 : torch.Tensor,   # (B, L) clean token ids
        t  : torch.Tensor,   # (B,)   timestep per sample
    ) -> torch.Tensor:
        """
        Apply corruption to clean sequences x0 at noise level t.
        Returns x_t with the same shape as x0.

        Padding positions (x0 == pad_id) are never corrupted.
        """
        B, L   = x0.shape
        device = x0.device

        # Broadcast alpha to (B, L)
        a = self.alpha(t).to(device).view(B, 1).expand(B, L)

        # Bernoulli mask: which positions get corrupted
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

    # ── Reverse process helper ───────────────────────────────────────────────────

    def should_denoise(self, xt: torch.Tensor) -> torch.Tensor:
        """
        Boolean mask of positions the model should try to predict.

        For ABSORBING: only masked positions need prediction.
        For UNIFORM:   all non-pad positions need prediction (can't tell which
                       ones were corrupted, so predict everything).

        Shape: (B, L) bool
        """
        if self.schedule_type == ScheduleType.ABSORBING:
            return xt == self.mask_id
        else:
            return xt != self.pad_id

    def name(self) -> str:
        return self.schedule_type.value


# ── Factory ──────────────────────────────────────────────────────────────────

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
