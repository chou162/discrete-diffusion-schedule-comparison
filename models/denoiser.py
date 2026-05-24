"""
Transformer denoiser for discrete diffusion.

Takes a corrupted sequence x_t and timestep t, outputs logits over
the vocabulary at each position. Trained with cross-entropy to predict
the original clean token x_0.

Timestep is encoded with sinusoidal embeddings (same idea as positional
encodings) and added to every token's representation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimestepEmbedding(nn.Module):
    """
    Maps scalar timestep t ∈ [0, T] to a d_model-dimensional vector
    using sinusoidal frequencies, then projects through a 2-layer MLP.

    This is identical in spirit to positional encodings in the original
    Transformer paper, but applied to the diffusion timestep instead of
    sequence position.
    """

    def __init__(self, d_model: int, max_period: int = 10_000):
        super().__init__()
        self.d_model    = d_model
        self.max_period = max_period

        half = d_model // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half).float() / half
        )
        self.register_buffer("freqs", freqs)  # (d_model//2,)

        # Small MLP to project sinusoidal features → d_model
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) float or long timestep indices
        returns: (B, d_model) timestep embeddings
        """
        t = t.float().unsqueeze(1)              # (B, 1)
        args = t * self.freqs.unsqueeze(0)      # (B, d_model//2)
        emb  = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, d_model)
        return self.proj(emb)


class TransformerDenoiser(nn.Module):
    """
    Transformer encoder that takes a corrupted sequence x_t and timestep t
    and outputs a logit distribution over the vocabulary for each position.

    The model is trained with cross-entropy loss to predict the original
    clean token x_0 at each position.
    """

    def __init__(
        self,
        vocab_size   : int,
        d_model      : int = 128,
        n_heads      : int = 4,
        n_layers     : int = 2,
        d_ff         : int = 256,
        max_seq_len  : int = 64,
        dropout      : float = 0.1,
        pad_id       : int = 0,
    ):
        super().__init__()

        self.d_model   = d_model
        self.pad_id    = pad_id

        # ── Embeddings ──────────────────────────────────────────────────────────
        self.token_emb    = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb      = nn.Embedding(max_seq_len, d_model)
        self.timestep_emb = SinusoidalTimestepEmbedding(d_model)

        # ── Transformer encoder ──────────────────────────────────────────────────
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = n_heads,
            dim_feedforward= d_ff,
            dropout        = dropout,
            batch_first    = True,
            norm_first     = True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # ── Output head ──────────────────────────────────────────────────────────
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie output head weights to token embedding (standard LM trick)
        self.head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize embeddings and linear layers with small normal weights."""
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight,   std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.head:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        xt: torch.Tensor,   # (B, L) corrupted token ids
        t : torch.Tensor,   # (B,)   timestep per sample
    ) -> torch.Tensor:
        """
        Returns logits of shape (B, L, vocab_size).
        """
        B, L   = xt.shape
        device = xt.device

        # Token + positional embeddings
        pos  = torch.arange(L, device=device).unsqueeze(0)   # (1, L)
        x    = self.token_emb(xt) + self.pos_emb(pos)        # (B, L, d)

        # Add timestep embedding (broadcast across sequence positions)
        t_emb = self.timestep_emb(t).unsqueeze(1)            # (B, 1, d)
        x     = x + t_emb                                    # (B, L, d)

        # Key padding mask: True = ignore (PyTorch convention)
        pad_mask = (xt == self.pad_id)                       # (B, L) bool

        # Transformer
        x = self.transformer(x, src_key_padding_mask=pad_mask)  # (B, L, d)

        # Output head
        x      = self.norm(x)
        logits = self.head(x)                                # (B, L, vocab_size)

        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
