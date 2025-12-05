#!/bin/bash
# Setup script for Wordle RL Agent
# This script creates virtual environments and installs all dependencies

set -e

echo "========================================"
echo "Wordle RL Agent - Setup Script"
echo "========================================"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' package manager is not installed."
    echo "Please install uv first: https://github.com/astral-sh/uv"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $PYTHON_VERSION"

# Setup main virtual environment for training
echo ""
echo "========================================"
echo "Setting up main virtual environment (.venv)"
echo "========================================"

if [ -d ".venv" ]; then
    echo ".venv already exists. Skipping creation."
else
    uv venv .venv
    echo ".venv created."
fi

echo "Installing training dependencies..."
source .venv/bin/activate
uv pip install torch numpy tqdm nltk wandb pygame
deactivate
echo "Training dependencies installed."

# Setup Manim virtual environment for video rendering
echo ""
echo "========================================"
echo "Setting up Manim virtual environment (.venv-manim)"
echo "========================================"

if [ -d ".venv-manim" ]; then
    echo ".venv-manim already exists. Skipping creation."
else
    uv venv .venv-manim
    echo ".venv-manim created."
fi

echo "Installing Manim dependencies..."
source .venv-manim/bin/activate
uv pip install manim torch numpy
deactivate
echo "Manim dependencies installed."

# Create necessary directories
echo ""
echo "========================================"
echo "Creating directories"
echo "========================================"

mkdir -p checkpoints
mkdir -p assets
echo "Directories created."

# Summary
echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To start training:"
echo "  source .venv/bin/activate"
echo "  python scripts/training/train_ppo_vectorized.py"
echo ""
echo "To render videos:"
echo "  source .venv-manim/bin/activate"
echo "  manim -pql scripts/rendering/render_wordle_video.py WordleGameScene"
echo ""
echo "Note: For Manim, you may need to install system dependencies:"
echo "  sudo apt-get install -y libcairo2-dev libpango1.0-dev pkg-config ffmpeg"
echo ""
