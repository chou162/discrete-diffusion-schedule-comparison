"""
Trains both corruption schedules and saves comparison plots to results/.

Usage:
    python run_experiment.py                      # full run
    python run_experiment.py --epochs 3           # quick test
    python run_experiment.py --schedule absorbing # one schedule only
"""

import os
import sys
import argparse
import torch

# ── make local imports work regardless of working directory ──────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset      import load_ptb
from models.denoiser   import TransformerDenoiser
from training.corruption import make_schedule, ScheduleType
from training.train    import TrainConfig, train_model
from analysis.evaluate import analyze, plot_all, print_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",    type=int,   default=15,
                   help="Training epochs per schedule")
    p.add_argument("--batch",     type=int,   default=64)
    p.add_argument("--lr",        type=float, default=3e-4)
    p.add_argument("--T",         type=int,   default=100,
                   help="Number of diffusion timesteps")
    p.add_argument("--max-len",   type=int,   default=32,
                   help="Sequence length (tokens)")
    p.add_argument("--d-model",   type=int,   default=128)
    p.add_argument("--n-layers",  type=int,   default=2)
    p.add_argument("--schedule",  type=str,   default="both",
                   choices=["absorbing", "uniform", "both"])
    p.add_argument("--no-train",  action="store_true",
                   help="Skip training, load saved checkpoints")
    p.add_argument("--save-dir",  type=str,   default="results")
    p.add_argument("--seed",      type=int,   default=42)
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] device: {device}")

    torch.manual_seed(args.seed)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, vocab = load_ptb(
        max_len    = args.max_len,
        batch_size = args.batch,
        seed       = args.seed,
    )

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = TrainConfig(
        n_epochs    = args.epochs,
        lr          = args.lr,
        T           = args.T,
        max_seq_len = args.max_len,
        d_model     = args.d_model,
        n_layers    = args.n_layers,
        save_dir    = os.path.join(args.save_dir, "checkpoints"),
    )

    # Which schedules to run
    schedule_names = (
        ["absorbing", "uniform"] if args.schedule == "both"
        else [args.schedule]
    )

    schedules = [
        make_schedule(
            schedule_type = name,
            T             = cfg.T,
            vocab_size    = len(vocab),
            mask_id       = vocab.mask_id,
            pad_id        = vocab.pad_id,
        )
        for name in schedule_names
    ]

    # ── Training ──────────────────────────────────────────────────────────────
    train_results = []
    for schedule in schedules:
        if args.no_train:
            # Load existing checkpoint
            model = TransformerDenoiser(
                vocab_size  = len(vocab),
                d_model     = cfg.d_model,
                n_heads     = 4,
                n_layers    = cfg.n_layers,
                max_seq_len = cfg.max_seq_len,
                pad_id      = vocab.pad_id,
            ).to(device)
            ckpt = os.path.join(cfg.save_dir, f"{schedule.name()}_model.pt")
            if os.path.exists(ckpt):
                model.load_state_dict(torch.load(ckpt, map_location=device))
                print(f"[main] Loaded checkpoint: {ckpt}")
            else:
                print(f"[main] No checkpoint found at {ckpt}, training instead")
                from training.train import TrainResult
                result = train_model(schedule, train_loader, val_loader,
                                     len(vocab), vocab.pad_id, cfg, device)
                train_results.append(result)
                continue

            from training.train import TrainResult
            result = TrainResult(schedule_name=schedule.name(), model=model)
        else:
            result = train_model(
                schedule, train_loader, val_loader,
                len(vocab), vocab.pad_id, cfg, device
            )
        train_results.append(result)

    # ── Analysis ──────────────────────────────────────────────────────────────
    print("\n[main] Running analysis...")

    # Grab a fixed batch of clean sequences for consistent analysis
    analysis_batch = next(iter(val_loader)).to(device)

    analysis_results = []
    for schedule, train_result in zip(schedules, train_results):
        ar = analyze(train_result, schedule, analysis_batch, device)
        analysis_results.append(ar)

    print_summary(analysis_results)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_all(analysis_results, train_results, save_dir=args.save_dir)
    print(f"\n[main] Done. Results in: {args.save_dir}/")


if __name__ == "__main__":
    main()
