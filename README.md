# Wordle RL Agent

A reinforcement learning agent that learns to play Wordle using Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) algorithms.

## Project Structure

```
wordle_rl_agent/
├── model/
│   ├── dqn.py                      # Deep Q-Network model
│   └── ppo.py                      # Actor-Critic model for PPO
├── scripts/
│   ├── train.py                    # DQN training script
│   ├── train_ppo.py                # PPO training script (single environment)
│   ├── train_ppo_parallel.py       # PPO training script (parallel environments)
│   ├── train_ppo_vectorized.py     # Fully vectorized PPO (fastest, recommended)
│   ├── test_agent.py               # Agent evaluation script (parallel/sequential)
│   ├── inference.py                # Interactive Wordle player
│   └── benchmark_vram.py           # VRAM usage benchmark tool
├── utils/
│   └── word_list.py                # Curated 5-letter word list
├── checkpoints/                    # Saved model checkpoints
├── .env                            # Environment variables (WANDB_API_KEY)
└── .venv/                          # Virtual environment
```

## Installation

### Prerequisites

- Python 3.12+
- CUDA-compatible GPU (recommended)
- [uv](https://github.com/astral-sh/uv) package manager

### Setup Virtual Environment

```bash
# Create virtual environment
uv venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install torch numpy tqdm nltk wandb
```

### Verify Installation

```bash
source .venv/bin/activate
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Weights & Biases Setup

This project uses [Weights & Biases](https://wandb.ai/) for experiment tracking and monitoring.

### Setup

1. Create a `.env` file in the project root with your API key:

```bash
WANDB_API_KEY=your_api_key_here
```

2. Or login via command line:

```bash
wandb login
```

### Monitored Metrics

The following metrics are logged to wandb during training:

| Metric | Description |
|--------|-------------|
| `win_rate` | Percentage of games won (rolling average) |
| `avg_reward` | Average episode reward (rolling average) |
| `avg_guesses` | Average number of guesses per game |
| `loss/total` | Total training loss |
| `loss/policy` | Policy loss (PPO only) |
| `loss/value` | Value function loss (PPO only) |
| `metrics/entropy` | Policy entropy (PPO only) |
| `metrics/vram_gb` | VRAM usage in GB (parallel training) |
| `epsilon` | Exploration rate (DQN only) |
| `learning_rate` | Current learning rate |

## Training

### Train with Parallel PPO (Recommended)

Parallel PPO runs multiple Wordle games simultaneously for faster training and better GPU utilization.

```bash
source .venv/bin/activate

# Quick test run (no W&B logging)
python scripts/train_ppo_parallel.py --num-envs 64 --total-timesteps 100000 --no-wandb

# Full training with W&B logging
python scripts/train_ppo_parallel.py \
    --num-envs 512 \
    --total-timesteps 50000000 \
    --lr 1e-4 \
    --entropy-coef 0.02 \
    --wandb-project "wordle-rl-agent" \
    --wandb-run-name "ppo-parallel-full-training"
```

#### VRAM Usage Guide

| Parallel Envs | Approx. VRAM |
|---------------|--------------|
| 256           | ~0.1 GB      |
| 512           | ~0.2 GB      |
| 1024          | ~0.3 GB      |
| 4096          | ~0.5 GB      |
| 16384         | ~1.5 GB      |

### Train with Single-Environment PPO

```bash
source .venv/bin/activate
python scripts/train_ppo.py
```

### Train with DQN

```bash
source .venv/bin/activate
python scripts/train.py --use-replay
```

## Command Line Arguments

### Parallel PPO (`scripts/train_ppo_parallel.py`)

```bash
python scripts/train_ppo_parallel.py --help
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-envs` | 256 | Number of parallel environments |
| `--total-timesteps` | 10000000 | Total training timesteps |
| `--steps-per-update` | 128 | Steps per policy update |
| `--lr` | 3e-4 | Learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--gae-lambda` | 0.95 | GAE lambda |
| `--clip-epsilon` | 0.2 | PPO clipping range |
| `--entropy-coef` | 0.01 | Entropy coefficient |
| `--value-coef` | 0.5 | Value loss coefficient |
| `--update-epochs` | 4 | PPO update epochs |
| `--mini-batch-size` | 512 | Mini-batch size for updates |
| `--checkpoint-interval` | 100 | Updates between checkpoints |
| `--no-wandb` | False | Disable wandb logging |
| `--wandb-project` | wordle-rl-agent | Wandb project name |
| `--wandb-run-name` | None | Wandb run name |

### Single PPO (`scripts/train_ppo.py`)

```bash
python scripts/train_ppo.py --help
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--episodes` | 100000 | Number of training episodes |
| `--update-interval` | 20 | Episodes between policy updates |
| `--save-interval` | 100 | Episodes between checkpoint saves |
| `--log-interval` | 10 | Episodes between logging |
| `--lr` | 3e-4 | Learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--gae-lambda` | 0.95 | GAE lambda |
| `--clip-epsilon` | 0.2 | PPO clipping range |
| `--entropy-coef` | 0.01 | Entropy coefficient |
| `--value-coef` | 0.5 | Value loss coefficient |
| `--update-epochs` | 4 | PPO update epochs |
| `--no-wandb` | False | Disable wandb logging |
| `--wandb-project` | wordle-rl-agent | Wandb project name |
| `--wandb-run-name` | None | Wandb run name |

### DQN (`scripts/train.py`)

```bash
python scripts/train.py --help
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--episodes` | 100000 | Number of training episodes |
| `--use-replay` | False | Use experience replay buffer |
| `--lr` | 1e-3 | Learning rate |
| `--gamma` | 0.95 | Discount factor |
| `--epsilon` | 1.0 | Initial exploration rate |
| `--epsilon-min` | 0.01 | Minimum epsilon |
| `--epsilon-decay` | 0.995 | Epsilon decay rate |
| `--log-interval` | 10 | Episodes between logging |
| `--save-interval` | 100 | Episodes between checkpoint saves |
| `--no-wandb` | False | Disable wandb logging |
| `--wandb-project` | wordle-rl-agent | Wandb project name |
| `--wandb-run-name` | None | Wandb run name |

## Model Checkpoints

Trained models are saved to the `checkpoints/` directory:

- `checkpoints/wordle_ppo_vectorized.pt` - Vectorized PPO model (recommended)
- `checkpoints/wordle_ppo_parallel.pt` - Parallel PPO model
- `checkpoints/wordle_ppo_model.pt` - Single PPO model
- `checkpoints/wordle_model.pt` - DQN model

## Evaluation

Evaluate trained agents using `test_agent.py`. Supports parallel evaluation for fast benchmarking.

### Quick Evaluation

```bash
source .venv/bin/activate

# Fast parallel evaluation (default, 1000 games)
python scripts/test_agent.py --checkpoint checkpoints/wordle_ppo_vectorized.pt --deterministic

# Large-scale evaluation (10,000 games in ~0.5 seconds)
python scripts/test_agent.py --checkpoint checkpoints/wordle_ppo_vectorized.pt --num-games 10000 --deterministic
```

### Sequential Evaluation with Details

```bash
# Verbose mode - shows each guess with colored feedback
python scripts/test_agent.py --checkpoint checkpoints/wordle_ppo_vectorized.pt --num-games 10 --verbose

# Sequential mode (slower, for debugging)
python scripts/test_agent.py --checkpoint checkpoints/wordle_ppo_vectorized.pt --sequential --num-games 100
```

### Interactive Play

```bash
# Play interactively - agent suggests guesses, you provide feedback
python scripts/test_agent.py --checkpoint checkpoints/wordle_ppo_vectorized.pt --interactive
```

### Test Agent Arguments (`scripts/test_agent.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | checkpoints/wordle_ppo_vectorized.pt | Path to model checkpoint |
| `--num-games` | 1000 | Number of games to evaluate |
| `--deterministic` | False | Use deterministic action selection (greedy) |
| `--sequential` | False | Use sequential evaluation (slower) |
| `--verbose` | False | Print each game's details (implies --sequential) |
| `--interactive` | False | Play interactively with the agent |

### Sample Output

```
============================================================
EVALUATION RESULTS
============================================================
Games played: 10000
Wins: 9871
Losses: 129
Win rate: 98.71%
Average guesses (wins only): 3.51

Guess distribution:
  1:                                              2 ( 0.0%)
  2: ████████                                   826 ( 8.3%)
  3: ████████████████████████████████████████  4012 (40.1%)
  4: ███████████████████████████████           3189 (31.9%)
  5: █████████████                             1357 (13.6%)
  6: ████                                       485 ( 4.9%)
  X: █                                          129 ( 1.3%)
```

## Inference

Play Wordle interactively with a trained model:

```bash
source .venv/bin/activate
python scripts/inference.py
```

## License

MIT
