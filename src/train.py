"""
train.py  —  Training loop for discrete diffusion denoiser

Loss: cross-entropy between predicted logits and the clean token x_0,
computed only at corrupted (non-pad) positions.

This is the "simple" or "x_0-prediction" objective used in MDLM and D3PM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Callable
import time


def compute_loss(
    model:    nn.Module,
    x0:       torch.Tensor,    # (B, L) clean tokens
    schedule,                   # AbsorbingSchedule or UniformSchedule
    pad_id:   int,
    device:   torch.device,
) -> torch.Tensor:
    """
    1. Sample a random timestep t ~ Uniform[0, T-1] per example.
    2. Corrupt x0 → x_t using the schedule's forward process.
    3. Predict clean tokens from x_t and t.
    4. Return mean cross-entropy over corrupted (non-pad) positions.
    """
    B, L = x0.shape
    x0   = x0.to(device)

    # Sample random timesteps for each example in the batch
    t = torch.randint(0, schedule.T, (B,), device=device)

    # Corrupt: apply the schedule's forward process
    x_t = schedule.q_sample(x0, t)                    # (B, L)

    # Build key_padding_mask (True = ignore pad)
    pad_mask = (x0 == pad_id)                          # (B, L)

    # Forward pass
    logits = model(x_t, t, key_padding_mask=pad_mask)  # (B, L, V)

    # Compute loss only at positions that were actually corrupted
    # (i.e. where x_t ≠ x0 and not padding)
    if schedule.name == "absorbing":
        corrupted = (x_t != x0) & ~pad_mask
    else:
        # Uniform: any position that changed from x0
        corrupted = (x_t != x0) & ~pad_mask

    if corrupted.sum() == 0:
        # Rare edge case: no tokens were corrupted (very early timesteps)
        return torch.tensor(0.0, device=device, requires_grad=True)

    logits_flat = logits[corrupted]              # (N, V)
    target_flat = x0[corrupted]                 # (N,)
    loss = F.cross_entropy(logits_flat, target_flat)
    return loss


def train_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    schedule,
    pad_id:     int,
    device:     torch.device,
    grad_clip:  float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for x0 in loader:
        optimizer.zero_grad()
        loss = compute_loss(model, x0, schedule, pad_id, device)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model:   nn.Module,
    loader:  DataLoader,
    schedule,
    pad_id:  int,
    device:  torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    for x0 in loader:
        loss = compute_loss(model, x0, schedule, pad_id, device)
        total_loss += loss.item()
        n_batches  += 1
    return total_loss / max(n_batches, 1)


def train(
    model:       nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    schedule,
    pad_id:      int,
    device:      torch.device,
    epochs:      int   = 20,
    lr:          float = 3e-4,
    grad_clip:   float = 1.0,
    patience:    int   = 5,
    verbose:     bool  = True,
) -> dict:
    """
    Full training run with early stopping.
    Returns history dict with train/valid losses per epoch.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "valid_loss": []}
    best_valid = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, schedule, pad_id, device, grad_clip)
        valid_loss = evaluate(model, valid_loader, schedule, pad_id, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)

        elapsed = time.time() - t0
        if verbose:
            print(f"  Epoch {epoch:03d} | train {train_loss:.4f} | "
                  f"valid {valid_loss:.4f} | {elapsed:.1f}s")

        if valid_loss < best_valid:
            best_valid = valid_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break

    return history
