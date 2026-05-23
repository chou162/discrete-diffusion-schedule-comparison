"""
analysis.py  —  Evaluation metrics and recovery curve analysis

This is the core research contribution of the project:
measuring *how* each corruption schedule affects the model's
ability to recover semantic content during reverse diffusion.

Key metrics:
  1. Token accuracy vs. timestep  — "recovery curve"
     At each timestep t, what fraction of corrupted tokens does the
     model correctly predict?  A schedule that front-loads recovery
     (high accuracy even at large t) gives the decoder more signal.

  2. Token entropy vs. timestep
     Average entropy of the predicted distribution at each t.
     Low entropy = confident predictions.  High entropy = uncertainty.
     Tells us when each schedule "knows" the answer.

  3. Per-position recovery
     Does the model recover function words (the, a, of) before
     content words (nouns, verbs)?  Plot mean accuracy per position.

  4. Bits-per-token estimate
     Approximation of model perplexity from held-out cross-entropy.
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import NamedTuple


class RecoveryCurve(NamedTuple):
    """Result object from compute_recovery_curve."""
    timesteps:      np.ndarray   # (K,)  t values evaluated
    accuracy:       np.ndarray   # (K,)  mean token accuracy at each t
    entropy:        np.ndarray   # (K,)  mean prediction entropy at each t
    frac_corrupted: np.ndarray   # (K,)  fraction of tokens corrupted at each t


@torch.no_grad()
def compute_recovery_curve(
    model,
    schedule,
    dataset_batch: torch.Tensor,   # (B, L) clean tokens, a single held-out batch
    pad_id:  int,
    device:  torch.device,
    n_steps: int = 40,             # evaluate at this many evenly-spaced t values
) -> RecoveryCurve:
    """
    For each of n_steps timesteps, corrupt the batch and measure
    how well the model predicts the original tokens.

    This directly answers the question:
      "At what stage of corruption does each schedule lose semantic content?"
    """
    model.eval()
    x0      = dataset_batch.to(device)          # (B, L)
    B, L    = x0.shape
    pad_mask = (x0 == pad_id)                   # (B, L)

    T        = schedule.T
    t_values = np.linspace(0, T - 1, n_steps, dtype=int)

    accuracies      = []
    entropies       = []
    frac_corrupteds = []

    for t_val in t_values:
        t_tensor = torch.full((B,), t_val, dtype=torch.long, device=device)

        # Corrupt
        x_t = schedule.q_sample(x0, t_tensor)

        # Identify corrupted positions
        corrupted = (x_t != x0) & ~pad_mask     # (B, L)
        frac_corrupted = corrupted.float().mean().item()

        # Predict
        logits = model(x_t, t_tensor, key_padding_mask=pad_mask)  # (B, L, V)
        probs  = F.softmax(logits, dim=-1)                         # (B, L, V)
        preds  = logits.argmax(dim=-1)                             # (B, L)

        # Accuracy: only at corrupted positions
        if corrupted.sum() > 0:
            correct  = (preds[corrupted] == x0[corrupted]).float().mean().item()
        else:
            correct  = 1.0

        # Entropy: at corrupted positions
        if corrupted.sum() > 0:
            p_corr   = probs[corrupted]                            # (N, V)
            ent      = -(p_corr * (p_corr + 1e-10).log()).sum(-1).mean().item()
        else:
            ent = 0.0

        accuracies.append(correct)
        entropies.append(ent)
        frac_corrupteds.append(frac_corrupted)

    return RecoveryCurve(
        timesteps      = np.array(t_values),
        accuracy       = np.array(accuracies),
        entropy        = np.array(entropies),
        frac_corrupted = np.array(frac_corrupteds),
    )


@torch.no_grad()
def compute_per_position_accuracy(
    model,
    schedule,
    dataset_batch: torch.Tensor,
    pad_id:  int,
    device:  torch.device,
    t_frac:  float = 0.5,         # evaluate at 50% noise level
) -> np.ndarray:
    """
    Returns per-position accuracy (L,) at a fixed noise level.
    Reveals whether the model recovers certain positions (e.g. function
    words at fixed positions) more reliably than others.
    """
    model.eval()
    x0 = dataset_batch.to(device)
    B, L = x0.shape
    t_val    = int(t_frac * schedule.T)
    t_tensor = torch.full((B,), t_val, dtype=torch.long, device=device)
    pad_mask = (x0 == pad_id)

    x_t    = schedule.q_sample(x0, t_tensor)
    logits = model(x_t, t_tensor, key_padding_mask=pad_mask)
    preds  = logits.argmax(dim=-1)                         # (B, L)

    correct = (preds == x0) & ~pad_mask                   # (B, L) bool
    per_pos = correct.float().mean(dim=0).cpu().numpy()   # (L,)
    return per_pos


@torch.no_grad()
def estimate_bits_per_token(
    model,
    schedule,
    loader,
    pad_id: int,
    device: torch.device,
    n_t_samples: int = 10,
) -> float:
    """
    Approximate bits-per-token by averaging cross-entropy over random
    timesteps on the validation set.  Lower = better language model.

    This is not the exact ELBO but gives a comparable scalar for ranking
    the two schedules.
    """
    model.eval()
    total_loss  = 0.0
    total_toks  = 0

    for x0 in loader:
        x0 = x0.to(device)
        B, L = x0.shape
        pad_mask = (x0 == pad_id)
        valid_toks = (~pad_mask).sum().item()

        batch_loss = 0.0
        for _ in range(n_t_samples):
            t = torch.randint(0, schedule.T, (B,), device=device)
            x_t    = schedule.q_sample(x0, t)
            logits = model(x_t, t, key_padding_mask=pad_mask)        # (B, L, V)
            loss   = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                x0.view(-1),
                ignore_index=pad_id,
                reduction="sum",
            )
            batch_loss += loss.item()

        total_loss += batch_loss / n_t_samples
        total_toks += valid_toks

    nll_per_tok  = total_loss / max(total_toks, 1)
    bits_per_tok = nll_per_tok / math.log(2)
    return bits_per_tok


def summarize_results(
    name_a:   str,
    curve_a:  RecoveryCurve,
    bpt_a:    float,
    name_b:   str,
    curve_b:  RecoveryCurve,
    bpt_b:    float,
) -> str:
    """Pretty-print a comparison summary."""
    sep = "─" * 60
    lines = [
        sep,
        f"  Schedule Comparison: {name_a}  vs.  {name_b}",
        sep,
        f"  {'Metric':<30} {name_a:>12} {name_b:>12}",
        sep,
        f"  {'Bits per token (↓ better)':<30} {bpt_a:>12.3f} {bpt_b:>12.3f}",
        f"  {'Accuracy @ t=0 (clean)':<30} {curve_a.accuracy[0]:>12.3f} {curve_b.accuracy[0]:>12.3f}",
        f"  {'Accuracy @ t=T/2 (50% noise)':<30} {curve_a.accuracy[len(curve_a.accuracy)//2]:>12.3f} {curve_b.accuracy[len(curve_b.accuracy)//2]:>12.3f}",
        f"  {'Accuracy @ t=T-1 (max noise)':<30} {curve_a.accuracy[-1]:>12.3f} {curve_b.accuracy[-1]:>12.3f}",
        f"  {'Mean entropy @ T/2':<30} {curve_a.entropy[len(curve_a.entropy)//2]:>12.3f} {curve_b.entropy[len(curve_b.entropy)//2]:>12.3f}",
        sep,
    ]
    return "\n".join(lines)
