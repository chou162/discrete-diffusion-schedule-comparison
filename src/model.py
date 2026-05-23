"""
model.py  —  Tiny Transformer denoiser for discrete diffusion

Architecture:
  Token embedding  +  sinusoidal timestep embedding  →  Transformer encoder
  →  Linear head over vocabulary  →  logits for each position

Kept intentionally small (2 layers, 4 heads, d_model=128) so it trains
in ~2 hours on a free Colab T4 GPU.  Scale up via ModelConfig if desired.
"""

import math
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size:  int   = 10_000
    seq_len:     int   = 32
    d_model:     int   = 128
    n_heads:     int   = 4
    n_layers:    int   = 2
    d_ff:        int   = 512
    dropout:     float = 0.1
    T:           int   = 200     # must match ScheduleConfig.T


# ── Sinusoidal timestep embedding ────────────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    """
    Encodes the integer timestep t into a d_model-dim vector using the
    standard sinusoidal position encoding formula (Vaswani et al. 2017).
    This gives the model a continuous signal about how noisy the input is.
    """
    def __init__(self, d_model: int, max_T: int = 1000):
        super().__init__()
        half = d_model // 2
        freqs = torch.exp(-math.log(max_T) * torch.arange(half) / (half - 1))
        self.register_buffer("freqs", freqs)          # (half,)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t : (B,)  integer timesteps
        t_f = t.float().unsqueeze(1)                  # (B, 1)
        args = t_f * self.freqs.unsqueeze(0)          # (B, half)
        emb  = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, d_model)
        return emb


# ── Main denoiser ─────────────────────────────────────────────────────────────

class DiffusionTransformer(nn.Module):
    """
    Takes (x_t, t) and predicts logits over the clean vocabulary for
    every token position.  Training objective: cross-entropy vs. x_0.

    The timestep embedding is added to every token's embedding before
    being passed to the Transformer encoder — this is the simplest
    conditioning strategy and works well at small scale.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb  = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=0)
        self.pos_emb  = nn.Embedding(cfg.seq_len,    cfg.d_model)
        self.t_emb    = SinusoidalEmbedding(cfg.d_model, max_T=cfg.T + 1)
        self.t_proj   = nn.Linear(cfg.d_model, cfg.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,        # Pre-LN: more stable training
        )
        self.encoder  = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.norm     = nn.LayerNorm(cfg.d_model)
        self.head     = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.zeros_(self.head.bias if self.head.bias is not None else
                        torch.zeros(1))   # head has no bias by design

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x_t : (B, L)   corrupted token ids
        t   : (B,)     integer diffusion timesteps
        key_padding_mask : (B, L) bool, True = ignore (pad positions)

        Returns logits : (B, L, V)
        """
        B, L = x_t.shape

        positions = torch.arange(L, device=x_t.device).unsqueeze(0)   # (1, L)

        tok  = self.tok_emb(x_t)                              # (B, L, D)
        pos  = self.pos_emb(positions)                        # (1, L, D)
        t_e  = self.t_proj(self.t_emb(t)).unsqueeze(1)        # (B, 1, D)

        h = tok + pos + t_e                                   # broadcast t over positions
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)
        logits = self.head(h)                                 # (B, L, V)
        return logits

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
