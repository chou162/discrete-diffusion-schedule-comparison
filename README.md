# Discrete Diffusion: Corruption Schedule Comparison

A minimal empirical study comparing **absorbing** vs. **uniform** corruption schedules
in discrete diffusion models for text, directly inspired by work from Prof. Ruqi Zhang's
group at Purdue on discrete diffusion and diffusion LLMs.

## Research Question

> Does the choice of corruption schedule affect how quickly and reliably the model
> recovers semantic content during reverse diffusion?

## Structure

```
discrete_diffusion/
├── data/
│   └── dataset.py          # Penn Treebank loader + tokenizer
├── models/
│   └── denoiser.py         # Transformer denoising model
├── training/
│   ├── corruption.py       # Absorbing + uniform corruption schedules
│   └── train.py            # Training loop
├── analysis/
│   └── evaluate.py         # Recovery curves, perplexity, entropy analysis
├── run_experiment.py       # Main entry point — trains both models + plots results
└── README.md
```

## Quickstart (runs on free Colab T4 in ~1-2 hrs)

```bash
pip install torch datasets tqdm matplotlib numpy
python run_experiment.py
```

Results saved to `results/` — recovery curves, perplexity comparison, token entropy plots.

## Results

![Recovery curves](results/recovery_curve_detail.png)

The uniform schedule substantially outperformed absorbing under a floored linear
noise schedule — achieving 50% token recovery at noise level α=0.51 while absorbing
failed to cross that threshold (val loss 2.83 vs 4.48 at convergence). This suggests
that uniform's broader supervision signal — training on all positions rather than only
masked ones — provides more consistent gradient flow, connecting directly to training
efficiency questions in discrete diffusion LMs.

## Key Concepts

- **Absorbing schedule**: tokens are independently masked to `[MASK]` with probability t/T
- **Uniform schedule**: tokens are replaced with a uniformly random vocabulary token
- **Recovery curve**: fraction of tokens correctly restored at each reverse diffusion step
- **Token entropy**: model's uncertainty H(p) over the vocabulary at each timestep

## Connection to Lab Research

This project directly explores the corruption schedule design question raised in:
- MDLM (Sahoo et al., 2024) — masked diffusion language models
- SEDD (Lou et al., 2024) — score entropy discrete diffusion

The absorbing schedule underpins MDLM; uniform noise is the baseline alternative.
Empirically comparing their denoising trajectories builds intuition for why schedule
choice matters for inference efficiency — a core theme in Prof. Zhang's work.
