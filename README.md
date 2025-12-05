# Wordle RL Agent

A reinforcement learning agent that learns to play Wordle using Proximal Policy Optimization (PPO) algorithm.

![Wordle RL Agent Demo](assets/demo.gif)

*Training progression: Step 100 → Step 200 → Step 300 (Win rate improves as training progresses)*

The agent learns to solve Wordle puzzles through reinforcement learning, achieving **96.1% win rate** after training.

## Project Structure

```
Wordle_RL_Agent/
├── model/
│   └── ppo.py                      # Actor-Critic model for PPO
├── scripts/
│   ├── setup.sh                    # Setup script for virtual environments
│   ├── training/
│   │   └── train_ppo_vectorized.py # Fully vectorized PPO training
│   ├── rendering/
│   │   ├── render_multi_games_landscape.py
│   │   └── render_multi_games_portrait.py
│   ├── visualize_agent.py          # Visual Wordle simulation (pygame)
│   ├── inference.py                # Interactive Wordle player
│   └── play_wordle.py              # Browser-based Wordle game
├── utils/
│   └── word_list.py                # Curated 5-letter word list
├── assets/
│   ├── demo.gif                    # Demo GIF for README
│   └── videos/                     # Rendered videos and GIFs
├── docs/                           # Documentation
└── checkpoints/                    # Saved model checkpoints
```

## Play Wordle

Play Wordle in your browser with the official word list (12,966 valid guesses, 351 common target words):

```bash
source .venv/bin/activate
python scripts/play_wordle.py
```

Open http://localhost:8080 in your browser. Controls:
- **Type letters** to enter your guess
- **Enter** to submit guess
- **Backspace** to delete
- **Space** to start a new game

## Quick Start

### Prerequisites

- Python 3.12+
- CUDA-compatible GPU (recommended)
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Run setup script (creates virtual environments and installs dependencies)
./scripts/setup.sh

# Or manual setup
uv venv .venv
source .venv/bin/activate
uv pip install torch numpy tqdm nltk wandb pygame
```

### Using Pre-trained Model

The repository includes pre-trained checkpoints that achieve 96.1% win rate:

```bash
source .venv/bin/activate

# Watch the agent play (pygame)
python scripts/visualize_agent.py

# Interactive play
python scripts/inference.py
```

### Training Your Own Model (Optional)

If you want to train the model from scratch:

```bash
source .venv/bin/activate

# Quick test run (~10 seconds)
python scripts/training/train_ppo_vectorized.py --num-envs 64 --total-timesteps 100000 --no-wandb

# Full training with default hyperparameters (~2 hours, achieves 96.1% win rate)
python scripts/training/train_ppo_vectorized.py --no-wandb

# Full training with W&B logging
python scripts/training/train_ppo_vectorized.py --wandb-project "wordle-rl-agent"
```

### Video Rendering

```bash
# Render video (outputs to assets/videos/)
source .venv-manim/bin/activate
manim -pql scripts/rendering/render_multi_games_landscape.py ComparisonScene
```

## Documentation

- [Training Guide](docs/training.md) - Detailed training instructions, CLI arguments, hyperparameters
- [Visualization Guide](docs/visualization.md) - Visual simulation and video rendering
- [W&B Integration](docs/wandb.md) - Weights & Biases setup and metrics

## License

MIT
