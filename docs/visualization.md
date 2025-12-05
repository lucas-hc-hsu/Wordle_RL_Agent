# Visualization Guide

## Visual Simulation (Pygame)

Watch the RL agent play Wordle in real-time with an authentic visual interface. Supports displaying multiple agents simultaneously.

```bash
source .venv/bin/activate

# Single agent (default)
python scripts/visualize_agent.py --checkpoint checkpoints/wordle_ppo_vectorized.pt

# Multiple agents - watch 4 games at once
python scripts/visualize_agent.py --checkpoint checkpoints/wordle_ppo_vectorized.pt --num-agents 4

# 9 agents in auto-play mode
python scripts/visualize_agent.py --checkpoint checkpoints/wordle_ppo_vectorized.pt --num-agents 9 --auto-play

# Adjust speed (default 1.0x)
python scripts/visualize_agent.py --checkpoint checkpoints/wordle_ppo_vectorized.pt --auto-play --speed 2.0
```

### Controls

| Key | Action |
|-----|--------|
| `SPACE` | Start new game / Make next guess |
| `A` | Toggle auto-play mode |
| `+`/`-` | Adjust animation speed |
| `1`/`4`/`9` | Set number of agents (1, 4, or 9) |
| `R` | Reset statistics |
| `Q`/`ESC` | Quit |

### Features

- Authentic Wordle colors (green/yellow/grey)
- Multi-agent grid view (1, 4, or 9 agents)
- Win/loss indicators (green/red backgrounds)
- Real-time statistics (win rate, average guesses)
- Batched inference for multiple agents

## Interactive Inference

Play Wordle interactively with a trained model:

```bash
source .venv/bin/activate
python scripts/inference.py
```

## Video Rendering (Manim)

Create high-quality mathematical animations of Wordle games using [Manim](https://www.manim.community/) (the animation library used by 3Blue1Brown).

### Setup Video Rendering Environment

Video rendering requires a separate virtual environment due to different dependencies.

#### 1. Install System Dependencies (Ubuntu/Debian)

```bash
sudo apt-get install -y libcairo2-dev libpango1.0-dev pkg-config ffmpeg
```

#### 2. Create Manim Virtual Environment

```bash
# Create dedicated virtual environment for Manim
uv venv .venv-manim

# Activate the Manim environment
source .venv-manim/bin/activate

# Install Manim and dependencies
uv pip install manim torch numpy
```

### Render Videos

```bash
source .venv-manim/bin/activate

# Render multi-game landscape video (8 games in 2x4 grid)
manim -pql scripts/rendering/render_multi_games_landscape.py MultiAgentScene

# Render comparison video (checkpoints 100, 200, 300)
manim -pql scripts/rendering/render_multi_games_landscape.py ComparisonScene

# Render multi-game portrait video (9 games in 3x3 grid)
manim -pql scripts/rendering/render_multi_games_portrait.py MultiAgentScene

# High quality render (1080p)
manim -pqh scripts/rendering/render_multi_games_landscape.py ComparisonScene

# 4K render
manim -pqk scripts/rendering/render_multi_games_landscape.py ComparisonScene
```

### Convert to GIF

After rendering, convert videos to GIF:

```bash
# Convert landscape video to GIF
python scripts/rendering/render_multi_games_landscape.py --to-gif assets/videos/ComparisonScene.mp4

# Convert portrait video to GIF
python scripts/rendering/render_multi_games_portrait.py --to-gif assets/videos/MultiAgentScene.mp4
```

### Quality Options

| Flag | Quality | Resolution |
|------|---------|------------|
| `-pql` | Low (preview) | 480p |
| `-pqm` | Medium | 720p |
| `-pqh` | High | 1080p |
| `-pqk` | 4K | 2160p |

### Output

Videos and GIFs are saved to the `assets/videos/` directory:

```
assets/videos/
├── ComparisonScene.mp4      # Landscape video
├── ComparisonScene.gif      # Landscape GIF
├── MultiAgentScene.mp4      # Portrait video
└── MultiAgentScene.gif      # Portrait GIF
```
