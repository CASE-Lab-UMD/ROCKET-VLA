# ROCKET on LIBERO (Simulation)

> OpenVLA-7B backbone with ROCKET multi-layer alignment on LIBERO benchmark.
> See the [main README](../README.md) for full documentation.

## Quick Start

```bash
# 1. Setup environment
conda create -n rocket python=3.10.16 -y && conda activate rocket
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
pip install -e .
pip install packaging ninja && pip install "flash-attn==2.5.5" --no-build-isolation

# 2. Download data & models (see main README for details)

# 3. Train ROCKET
bash scripts/train_rocket.sh

# 4. Evaluate
bash scripts/eval_libero.sh
```

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/train_rocket.sh` | ROCKET training (all ablations configurable) |
| `scripts/profile_layers.sh` | CKA similarity profiling between VLA and VGGT layers |
| `scripts/gradient_analysis.sh` | Gradient coherence analysis |
| `scripts/eval_libero.sh` | LIBERO evaluation |
