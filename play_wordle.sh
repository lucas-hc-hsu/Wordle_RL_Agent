#!/bin/bash
# =============================================================================
# Play Wordle - Browser-based Wordle Game
# =============================================================================
# This script automatically sets up a minimal Python environment and launches
# the Wordle game in your browser.
#
# Usage: ./play_wordle.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv-wordle"

echo "=============================================="
echo "  Wordle RL Agent - Play Wordle"
echo "=============================================="

# Check for Python 3
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python 3 is required but not found."
    echo "Please install Python 3.8+ and try again."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found Python $PYTHON_VERSION"

# Create minimal virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "Creating minimal virtual environment..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo "Virtual environment created at $VENV_DIR"
else
    echo "Using existing virtual environment at $VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# No external dependencies needed - play_wordle.py uses only standard library!
echo ""
echo "Starting Wordle game..."
echo "Open http://localhost:8080 in your browser"
echo ""
echo "Controls:"
echo "  - Type letters to enter your guess"
echo "  - Enter to submit"
echo "  - Backspace to delete"
echo "  - Space for new game"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=============================================="

# Run the game
python "$SCRIPT_DIR/scripts/play_wordle.py"
