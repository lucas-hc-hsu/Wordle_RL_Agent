# Weights & Biases Integration

This project uses [Weights & Biases](https://wandb.ai/) for experiment tracking and monitoring.

## Setup

### Option 1: Environment File

Create a `.env` file in the project root with your API key:

```bash
WANDB_API_KEY=your_api_key_here
```

### Option 2: Command Line Login

```bash
wandb login
```

## Running Training with W&B

```bash
source .venv/bin/activate

# With .env file
WANDB_API_KEY=$(grep WANDB_API_KEY .env | cut -d'=' -f2) python scripts/training/train_ppo_vectorized.py

# Or export first
export WANDB_API_KEY=$(grep WANDB_API_KEY .env | cut -d'=' -f2)
python scripts/training/train_ppo_vectorized.py \
    --wandb-project "wordle-rl-agent" \
    --wandb-run-name "my-experiment"
```

## Monitored Metrics

The following metrics are logged to wandb during training:

| Metric | Description |
|--------|-------------|
| `win_rate` | Percentage of games won (rolling average) |
| `avg_reward` | Average episode reward (rolling average) |
| `avg_guesses` | Average number of guesses per game |
| `loss/total` | Total training loss |
| `loss/policy` | Policy loss (PPO only) |
| `loss/value` | Value function loss (PPO only) |
| `metrics/entropy` | Policy entropy |
| `metrics/vram_gb` | VRAM usage in GB (parallel training) |
| `learning_rate` | Current learning rate |

## Disabling W&B

To run training without W&B logging:

```bash
python scripts/training/train_ppo_vectorized.py --no-wandb
```
