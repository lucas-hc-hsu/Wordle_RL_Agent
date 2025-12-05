# Training Guide

## Overview

The project uses fully vectorized PPO training for maximum GPU efficiency.

## Quick Start

```bash
source .venv/bin/activate

# Quick test run (no W&B logging)
python scripts/training/train_ppo_vectorized.py --num-envs 64 --total-timesteps 100000 --no-wandb

# Full training with default hyperparameters (achieves 96.1% win rate)
python scripts/training/train_ppo_vectorized.py

# Full training with W&B logging
python scripts/training/train_ppo_vectorized.py \
    --wandb-project "wordle-rl-agent" \
    --wandb-run-name "ppo-training"
```

## Command Line Arguments

```bash
python scripts/training/train_ppo_vectorized.py --help
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-envs` | 1024 | Number of parallel environments |
| `--total-timesteps` | 131200000 | Total training timesteps (~1000 updates) |
| `--steps-per-update` | 128 | Steps per policy update |
| `--lr` | 1e-4 | Learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--gae-lambda` | 0.95 | GAE lambda |
| `--clip-epsilon` | 0.2 | PPO clipping range |
| `--entropy-coef` | 0.01 | Entropy coefficient |
| `--value-coef` | 0.5 | Value loss coefficient |
| `--update-epochs` | 4 | PPO update epochs |
| `--mini-batch-size` | 1024 | Mini-batch size for updates |
| `--checkpoint-interval` | 1 | Updates between checkpoints |
| `--checkpoint-dir` | checkpoints | Directory to save checkpoints |
| `--test-ratio` | 0.2 | Fraction of words for testing |
| `--eval-interval` | 5 | Evaluate on test set every N updates |
| `--no-wandb` | False | Disable wandb logging |
| `--wandb-project` | wordle-rl-agent | Wandb project name |
| `--wandb-run-name` | None | Wandb run name |

## VRAM Usage Guide

| Parallel Envs | Approx. VRAM |
|---------------|--------------|
| 1024          | ~0.3 GB      |
| 4096          | ~0.6 GB      |
| 16384         | ~2.0 GB      |

## Model Checkpoints

Trained models are saved to the `checkpoints/` directory:

- `checkpoints/wordle_ppo_vectorized.pt` - Latest checkpoint
- `checkpoints/wordle_ppo_vectorized_100.pt` - Checkpoint at update 100
- `checkpoints/wordle_ppo_vectorized_200.pt` - Checkpoint at update 200
- `checkpoints/wordle_ppo_vectorized_300.pt` - Checkpoint at update 300

## Recommended Configuration

The following configuration achieves **96.1% win rate** (avg_reward = 83.36) and is the default configuration:

```bash
# RECOMMENDED - Default configuration (just run without arguments)
python scripts/training/train_ppo_vectorized.py

# Or explicitly specify parameters:
python scripts/training/train_ppo_vectorized.py \
    --num-envs 1024 \
    --mini-batch-size 1024 \
    --total-timesteps 131200000 \
    --lr 1e-4 \
    --entropy-coef 0.01 \
    --test-ratio 0.2 \
    --eval-interval 10 \
    --checkpoint-interval 100 \
    --wandb-project wordle-rl-agent \
    --wandb-run-name "ppo-training"
```

### Default Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `lr` | **1e-4** | Low learning rate for stable, gradual convergence |
| `entropy-coef` | **0.01** | Low entropy promotes exploitation over exploration |
| `num-envs` | 1024 | Maximum parallelism within constraints |
| `mini-batch-size` | 1024 | Matches num-envs for efficient updates |
| `total-timesteps` | 131.2M | ~1000 updates for thorough training |
| `clip-epsilon` | 0.2 | Default PPO clipping (balanced) |

### Why This Works

- Low learning rate (1e-4) prevents destructive updates and allows fine-grained policy improvement
- Low entropy coefficient (0.01) encourages the agent to commit to good strategies rather than exploring randomly
- Extended training (1000 updates) gives the agent enough time to fully converge

## Ineffective Configurations (Avoid These)

### Large Model with High Environment Count

```bash
# NOT RECOMMENDED - Takes too long, low GPU utilization despite high VRAM usage
python scripts/training/train_ppo_vectorized.py \
    --num-envs 16384 \
    --mini-batch-size 8192 \
    --total-timesteps 220000000 \
    --lr 3e-4
```

**Issues:**
- Model: hidden_size=1024, num_layers=3 (too large)
- ~62 seconds per update, total training time: ~2 hours for 104 updates
- GPU utilization only ~11% despite 15GB+ VRAM usage
- Excessive timesteps required for convergence

See [non_effective_hyperparameters.md](non_effective_hyperparameters.md) for more configurations to avoid.

## Training Performance

| Configuration | Updates | Win Rate | Avg Reward |
|--------------|---------|----------|------------|
| 16384 envs, large model | 104 | ~70% | - |
| **Default (1024 envs, lr=1e-4, entropy=0.01)** | **1000** | **96.1%** | **83.36** |
