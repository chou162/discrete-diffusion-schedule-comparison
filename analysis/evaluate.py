"""
Post-training analysis — three metrics:
  1. Recovery curve: fraction of tokens correctly predicted at each timestep
  2. Perplexity: exp(cross-entropy) at t = T/2
  3. Token entropy: how uncertain the model is at each noise level
"""

import os
import math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass

from models.denoiser import TransformerDenoiser
from training.corruption import CorruptionSchedule
from training.train import TrainResult



@dataclass
class AnalysisResult:
    schedule_name  : str
    timesteps      : np.ndarray         
    recovery_curve : np.ndarray         
    entropy_curve  : np.ndarray         
    perplexity     : float             


@torch.no_grad()
def analyze(
    result   : TrainResult,
    schedule : CorruptionSchedule,
    x0_batch : torch.Tensor,          
    device   : torch.device,
) -> AnalysisResult:
    """Run recovery, entropy, and perplexity analysis on a batch of clean sequences."""
  
    model = result.model.to(device)
    model.eval()

    T          = schedule.T
    timesteps  = np.arange(T, 0, -1)   
    recoveries = np.zeros(T)
    entropies  = np.zeros(T)

    x0 = x0_batch.to(device)
    B, L = x0.shape

    for i, t_val in enumerate(timesteps):
        t = torch.full((B,), t_val, dtype=torch.long, device=device)

        xt     = schedule.corrupt(x0, t)

        logits = model(xt, t)             
        probs  = torch.softmax(logits, dim=-1)
        preds  = logits.argmax(dim=-1)     

        non_pad   = (x0 != schedule.pad_id)                
        correct   = (preds == x0) & non_pad
        recovery  = correct.sum().float() / non_pad.sum().float()
        recoveries[i] = recovery.item()

        log_p   = torch.log(probs + 1e-12)
        h       = -(probs * log_p).sum(dim=-1)             
        entropy = h[non_pad].mean()
        entropies[i] = entropy.item()

    # Perplexity at t = T//2
    t_mid   = torch.full((B,), T // 2, dtype=torch.long, device=device)
    xt_mid  = schedule.corrupt(x0, t_mid)
    logits_mid = model(xt_mid, t_mid)

    non_pad_flat = (x0 != schedule.pad_id).view(-1)
    ce = torch.nn.functional.cross_entropy(
        logits_mid.view(-1, logits_mid.size(-1))[non_pad_flat],
        x0.view(-1)[non_pad_flat],
        reduction="mean",
    )
    perplexity = math.exp(ce.item())

    print(f"[analyze] {schedule.name():10s} | perplexity@t={T//2}: {perplexity:.2f}")

    return AnalysisResult(
        schedule_name  = schedule.name(),
        timesteps      = timesteps,
        recovery_curve = recoveries,
        entropy_curve  = entropies,
        perplexity     = perplexity,
    )

def plot_all(
    results       : list[AnalysisResult],
    train_results : list[TrainResult],
    save_dir      : str = "results",
) -> None:
    """Save a 2x2 comparison figure and a standalone recovery curve plot."""

    os.makedirs(save_dir, exist_ok=True)

    styles = {
        "absorbing": dict(color="#1a6faf", ls="-",  label="Absorbing (mask)"),
        "uniform":   dict(color="#c0392b", ls="--", label="Uniform noise"),
    }

    fig = plt.figure(figsize=(13, 10))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    for r in results:
        s = styles[r.schedule_name]
        noise = r.timesteps / r.timesteps.max()
        ax1.plot(noise, r.recovery_curve, color=s["color"], ls=s["ls"],
                 lw=2, label=s["label"])
    ax1.set_xlabel("Noise level (t / T)")
    ax1.set_ylabel("Fraction tokens correctly recovered")
    ax1.set_title("Recovery curve", fontweight="bold")
    ax1.set_xlim(1, 0)   
    ax1.set_ylim(-0.02, 1.05)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, lw=0.5)
    ax1.axhline(0.5, color="gray", ls=":", lw=0.8, alpha=0.6)
    ax1.text(0.98, 0.52, "50% recovery", ha="right", va="bottom",
             fontsize=7, color="gray", transform=ax1.get_xaxis_transform())

    ax2 = fig.add_subplot(gs[0, 1])
    for r in results:
        s = styles[r.schedule_name]
        noise = r.timesteps / r.timesteps.max()
        ax2.plot(noise, r.entropy_curve, color=s["color"], ls=s["ls"],
                 lw=2, label=s["label"])
    ax2.set_xlabel("Noise level (t / T)")
    ax2.set_ylabel("Mean token entropy H(p)")
    ax2.set_title("Model uncertainty over noise levels", fontweight="bold")
    ax2.set_xlim(1, 0)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, lw=0.5)

    ax3 = fig.add_subplot(gs[1, 0])
    for tr in train_results:
        s      = styles[tr.schedule_name]
        epochs = list(range(1, len(tr.train_losses) + 1))
        ax3.plot(epochs, tr.train_losses, color=s["color"], ls=s["ls"],
                 lw=2, label=f"{s['label']} (train)")
        ax3.plot(epochs, tr.val_losses,   color=s["color"], ls=":",
                 lw=1.5, alpha=0.7, label=f"{s['label']} (val)")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Cross-entropy loss")
    ax3.set_title("Training & validation loss", fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, lw=0.5)

    ax4    = fig.add_subplot(gs[1, 1])
    names  = [r.schedule_name.capitalize() for r in results]
    ppls   = [r.perplexity for r in results]
    colors = [styles[r.schedule_name]["color"] for r in results]
    bars   = ax4.bar(names, ppls, color=colors, width=0.45, edgecolor="white", lw=1.5)
    for bar, ppl in zip(bars, ppls):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{ppl:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax4.set_ylabel("Perplexity (lower = better)")
    ax4.set_title(f"Perplexity at t = T/2", fontweight="bold")
    ax4.grid(True, axis="y", alpha=0.3, lw=0.5)
    ax4.set_ylim(0, max(ppls) * 1.25)

    fig.suptitle(
        "Discrete Diffusion: Absorbing vs. Uniform Corruption Schedules",
        fontsize=14, fontweight="bold", y=0.98,
    )

    out_path = os.path.join(save_dir, "results.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved: {out_path}")

    _plot_recovery_detail(results, save_dir, styles)


def _plot_recovery_detail(
    results  : list[AnalysisResult],
    save_dir : str,
    styles   : dict,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")

    for r in results:
        s     = styles[r.schedule_name]
        noise = r.timesteps / r.timesteps.max()
        ax.plot(noise, r.recovery_curve, color=s["color"], ls=s["ls"],
                lw=2.5, label=s["label"])

        # mark where recovery crosses 50%
        cross_idx = np.argmax(r.recovery_curve >= 0.5)
        if cross_idx > 0:
            n50 = noise[cross_idx]
            ax.axvline(n50, color=s["color"], ls=":", lw=1, alpha=0.6)
            ax.text(n50 - 0.01, 0.08, f"{s['label'].split()[0]}\n50% @ {n50:.2f}",
                    ha="right", va="bottom", fontsize=8, color=s["color"])

    ax.set_xlabel("Noise level α(t) = t / T", fontsize=12)
    ax.set_ylabel("Fraction of tokens correctly recovered", fontsize=12)
    ax.set_title(
        "Recovery Trajectory: Absorbing vs. Uniform Corruption",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(1.0, 0.0)
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25, lw=0.5)

    out_path = os.path.join(save_dir, "recovery_curve_detail.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved: {out_path}")


def print_summary(results: list[AnalysisResult]) -> None:
    print("\n" + "═" * 60)
    print("  RESULTS SUMMARY")
    print("═" * 60)
    for r in results:
        def first_above(threshold):
            idx = np.argmax(r.recovery_curve >= threshold)
            return r.timesteps[idx] / r.timesteps.max() if idx > 0 else float("nan")

        print(f"\n  Schedule: {r.schedule_name.upper()}")
        print(f"    Perplexity @ t=T/2:     {r.perplexity:.2f}")
        print(f"    Noise level @ 50% rec.: {first_above(0.50):.3f}")
        print(f"    Noise level @ 80% rec.: {first_above(0.80):.3f}")
        print(f"    Max entropy:            {r.entropy_curve.max():.3f}")
        print(f"    Min entropy:            {r.entropy_curve.min():.3f}")
    print("═" * 60 + "\n")
