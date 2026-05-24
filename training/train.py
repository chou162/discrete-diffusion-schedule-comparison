"""
Training loop for a single schedule + model pair.

For absorbing: loss is computed only at masked positions.
For uniform: loss is computed at all non-padding positions since
we can't tell which tokens were replaced.
"""

import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass, field

from models.denoiser import TransformerDenoiser
from training.corruption import CorruptionSchedule, ScheduleType


@dataclass
class TrainConfig:
    d_model    : int   = 128
    n_heads    : int   = 4
    n_layers   : int   = 2
    d_ff       : int   = 256
    dropout    : float = 0.1

    n_epochs   : int   = 15
    lr         : float = 3e-4
    clip_grad  : float = 1.0
    T          : int   = 100     
    max_seq_len: int   = 32

    log_every  : int   = 50      
    save_dir   : str   = "results/checkpoints"


@dataclass
class TrainResult:
    schedule_name : str
    model         : TransformerDenoiser
    train_losses  : list[float] = field(default_factory=list)
    val_losses    : list[float] = field(default_factory=list)
    epoch_times   : list[float] = field(default_factory=list)


def train_model(
    schedule     : CorruptionSchedule,
    train_loader : DataLoader,
    val_loader   : DataLoader,
    vocab_size   : int,
    pad_id       : int,
    cfg          : TrainConfig,
    device       : torch.device,
) -> TrainResult:
    """
    Train one TransformerDenoiser with the given schedule, return losses and trained model.
    """
    model = TransformerDenoiser(
        vocab_size  = vocab_size,
        d_model     = cfg.d_model,
        n_heads     = cfg.n_heads,
        n_layers    = cfg.n_layers,
        d_ff        = cfg.d_ff,
        max_seq_len = cfg.max_seq_len,
        dropout     = cfg.dropout,
        pad_id      = pad_id,
    ).to(device)

    print(f"\n[train] Schedule: {schedule.name().upper()} | "
          f"Params: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)

    total_steps  = cfg.n_epochs * len(train_loader)
    warmup_steps = max(1, total_steps // 10)
    scheduler    = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, total_steps=total_steps,
        pct_start=0.1, anneal_strategy="cos",
    )

    result = TrainResult(schedule_name=schedule.name(), model=model)
    step   = 0

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0         = time.time()

        for batch in train_loader:
            x0 = batch.to(device)                   
            B  = x0.size(0)

            t = torch.randint(1, schedule.T + 1, (B,), device=device)

            xt = schedule.corrupt(x0, t)

            logits = model(xt, t)                   

            loss_mask = _loss_mask(schedule, xt, x0, pad_id)  

            if loss_mask.sum() == 0:
                continue

            loss = _masked_cross_entropy(logits, x0, loss_mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            step       += 1

            if step % cfg.log_every == 0:
                print(f"  step {step:4d} | loss {loss.item():.4f} | "
                      f"lr {scheduler.get_last_lr()[0]:.2e}")

        val_loss = evaluate(model, schedule, val_loader, pad_id, device)
        avg_train = epoch_loss / max(1, len(train_loader))
        elapsed   = time.time() - t0

        result.train_losses.append(avg_train)
        result.val_losses.append(val_loss)
        result.epoch_times.append(elapsed)

        print(f"[epoch {epoch:2d}/{cfg.n_epochs}] "
              f"train {avg_train:.4f} | val {val_loss:.4f} | "
              f"{elapsed:.1f}s")

    os.makedirs(cfg.save_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.save_dir, f"{schedule.name()}_model.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"[train] Saved checkpoint: {ckpt_path}")

    return result


@torch.no_grad()
def evaluate(
    model      : TransformerDenoiser,
    schedule   : CorruptionSchedule,
    loader     : DataLoader,
    pad_id     : int,
    device     : torch.device,
) -> float:
    """Validation loss"""
    model.eval()
    total_loss, total_batches = 0.0, 0

    for batch in loader:
        x0 = batch.to(device)
        B  = x0.size(0)
        t  = torch.randint(1, schedule.T + 1, (B,), device=device)
        xt = schedule.corrupt(x0, t)

        logits    = model(xt, t)
        loss_mask = _loss_mask(schedule, xt, x0, pad_id)

        if loss_mask.sum() == 0:
            continue

        loss = _masked_cross_entropy(logits, x0, loss_mask)
        total_loss   += loss.item()
        total_batches += 1

    return total_loss / max(1, total_batches)


def _loss_mask(
    schedule : CorruptionSchedule,
    xt       : torch.Tensor,    
    x0       : torch.Tensor,    
    pad_id   : int,
) -> torch.Tensor:
    # absorbing: only masked positions; uniform: all non-padding positions
    
    if schedule.schedule_type == ScheduleType.ABSORBING:
        return (xt == schedule.mask_id) & (x0 != pad_id)
    else:
        return x0 != pad_id


def _masked_cross_entropy(
    logits : torch.Tensor,   
    targets: torch.Tensor,   
    mask   : torch.Tensor,   
) -> torch.Tensor:
    """Cross-entropy loss averaged over masked positions."""
    B, L, V = logits.shape
    logits_flat  = logits.view(B * L, V)
    targets_flat = targets.view(B * L)
    mask_flat    = mask.view(B * L)

    loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
    return loss[mask_flat].mean()
