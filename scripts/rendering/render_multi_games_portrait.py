"""
Render multiple Wordle RL agents playing simultaneously using Manim.

This script creates a grid of agents all playing Wordle at the same time,
showing how the RL model performs across multiple games.

Usage:
    # Render 9 games in 3x3 grid (default) - Portrait format (9:18)
    manim -pql scripts/rendering/render_multi_games_portrait.py MultiAgentScene

    # High quality render
    manim -pqh scripts/rendering/render_multi_games_portrait.py MultiAgentScene

Quality flags:
    -ql: Low quality (480p, 15fps) - fast preview
    -qm: Medium quality (720p, 30fps)
    -qh: High quality (1080p, 60fps)
    -qk: 4K quality (2160p, 60fps)

Output:
    Videos and GIFs are saved to: assets/videos/
    - {SceneName}.mp4 - Video file
    - {SceneName}.gif - GIF file (auto-generated)

Note: This uses 9:18 aspect ratio (width:height), standard portrait format.
      Resolution: 1080x2160 (1:2 ratio)
"""

# Configure for 9:18 aspect ratio (width:height)
# This is the standard portrait format (1:2 ratio)
from manim import config
import os
from pathlib import Path

# Set output directory to assets/videos/
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "videos"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
config.media_dir = str(OUTPUT_DIR.parent)
config.video_dir = str(OUTPUT_DIR)

config.frame_width = 9.0
config.frame_height = 18.0  # 18:9 aspect ratio
config.pixel_width = 1080
config.pixel_height = 2160

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from manim import *
import torch
import numpy as np

from model.ppo import ActorCritic
from utils.word_list import load_word_list, load_common_words


# =============================================================================
# Wordle Color Scheme
# =============================================================================
WORDLE_GREEN = "#6aaa64"   # Correct letter, correct position
WORDLE_YELLOW = "#c9b458"  # Correct letter, wrong position
WORDLE_GREY = "#787c7e"    # Letter not in word
WORDLE_DARK = "#121213"    # Background
WORDLE_OUTLINE = "#3a3a3c" # Tile outline
WORDLE_WHITE = "#ffffff"   # Text color


# =============================================================================
# RL Agent Wrapper
# =============================================================================
class WordleAgent:
    """Wrapper for the trained RL agent."""

    def __init__(self, checkpoint_path, word_list, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.word_list = word_list

        state_size = 26 * 3
        action_size = len(word_list)

        self.policy = ActorCritic(state_size, action_size, self.device).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy.eval()

        print(f"Loaded checkpoint: {checkpoint_path}")
        print(f"  Update: {checkpoint.get('update', 'N/A')}")
        print(f"  Win Rate: {checkpoint.get('win_rate', 'N/A')}")

    @torch.no_grad()
    def select_action(self, state, deterministic=True):
        """Select action based on state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_logits, _ = self.policy(state_tensor)

        if deterministic:
            action = action_logits.argmax(dim=-1).item()
        else:
            probs = torch.softmax(action_logits, dim=-1)
            action = torch.multinomial(probs, 1).item()

        return self.word_list[action]


# =============================================================================
# Wordle Game Logic
# =============================================================================
class WordleGame:
    """Simple Wordle game for generating gameplay data."""

    def __init__(self, word_list):
        self.word_list = word_list
        self.reset()

    def reset(self, target_word=None):
        if target_word is None:
            self.target_word = np.random.choice(self.word_list).upper()
        else:
            self.target_word = target_word.upper()

        self.current_guess = 0
        self.state = np.zeros(26 * 3)
        self.guesses = []
        self.feedbacks = []
        self.game_over = False
        self.won = False
        return self.state.copy()

    def get_feedback(self, guess):
        guess = guess.upper()
        target = self.target_word
        feedback = ['grey'] * 5
        target_chars = list(target)

        # First pass: find greens
        for i in range(5):
            if guess[i] == target[i]:
                feedback[i] = 'green'
                target_chars[i] = None

        # Second pass: find yellows
        for i in range(5):
            if feedback[i] != 'green' and guess[i] in target_chars:
                feedback[i] = 'yellow'
                target_chars[target_chars.index(guess[i])] = None

        return feedback

    def step(self, guess):
        if self.game_over:
            return None, True

        guess = guess.upper()
        feedback = self.get_feedback(guess)

        self.guesses.append(guess)
        self.feedbacks.append(feedback)

        # Update state
        for i, (letter, fb) in enumerate(zip(guess, feedback)):
            idx = ord(letter.lower()) - ord('a')
            if fb == 'green':
                self.state[idx * 3 + 1] = 1
            elif fb == 'yellow':
                self.state[idx * 3 + 2] = 1
            else:
                self.state[idx * 3] = 1

        self.current_guess += 1
        self.won = guess == self.target_word
        self.game_over = self.won or self.current_guess >= 6

        return feedback, self.game_over


# =============================================================================
# Manim Scene: Multiple Agents Playing Simultaneously
# =============================================================================
class MultiAgentScene(Scene):
    """Manim scene showing multiple agents playing Wordle simultaneously."""

    def setup(self):
        """Setup is called before construct."""
        self.word_list = load_word_list()
        self.common_words = load_common_words()

        # Configuration
        self.num_agents = 9  # 3x3 grid
        self.grid_cols = 3
        self.grid_rows = 3

        # Use default checkpoint
        checkpoint_path = "checkpoints/wordle_ppo_vectorized.pt"

        try:
            self.agent = WordleAgent(checkpoint_path, self.word_list)
        except FileNotFoundError:
            print(f"Checkpoint not found: {checkpoint_path}")
            self.agent = None

        # Create games with common words as targets
        self.games = [WordleGame(self.common_words) for _ in range(self.num_agents)]

        # Tile sizes for 3x3 grid - adjusted for 18:9 aspect ratio
        self.tile_size = 0.45
        self.tile_gap = 0.05
        # Grid spacing for 18:9 format
        self.grid_spacing_x = 2.8
        self.grid_spacing_y = 4.6  # Adjusted for 18:9 ratio

    def construct(self):
        """Main animation construction."""
        # Title and subtitle
        title = Text("Wordle RL Agent", font_size=52, color=WHITE, weight=BOLD)
        subtitle = Text("Trained with PPO | Rendered with Manim (3B1B)", font_size=20, color=WORDLE_GREY)
        title.to_edge(UP, buff=0.6)
        subtitle.next_to(title, DOWN, buff=0.15)
        self.play(Write(title), run_time=0.4)
        self.play(FadeIn(subtitle), run_time=0.3)

        # Create grids for all agents
        all_grids = []

        # Calculate positions for 3x3 layout - shifted down for more subtitle spacing
        grid_center_y = 0.0  # Center grids lower to increase gap from subtitle
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                grid = self.create_mini_grid()
                x = (col - 1) * self.grid_spacing_x
                y = grid_center_y + (1 - row) * self.grid_spacing_y
                grid.move_to([x, y, 0])
                all_grids.append(grid)

        # Show all grids
        self.play(
            *[Create(grid) for grid in all_grids],
            run_time=0.5
        )

        # Reset all games
        states = [game.reset() for game in self.games]

        # Track which games have shown WIN indicator
        win_indicators_shown = [False] * self.num_agents
        win_indicators = [None] * self.num_agents

        # Play all games simultaneously
        for turn in range(6):
            all_done = all(game.game_over for game in self.games)
            if all_done:
                break

            # Get guesses for all active games
            guesses = []
            for i, game in enumerate(self.games):
                if not game.game_over:
                    if self.agent:
                        guess = self.agent.select_action(states[i]).upper()
                    else:
                        guess = np.random.choice(self.word_list).upper()
                    guesses.append(guess)
                else:
                    guesses.append(None)

            # Animate all guesses simultaneously
            self.animate_all_guesses(all_grids, turn, guesses)

            # Get feedback for all games
            feedbacks = []
            for i, (game, guess) in enumerate(zip(self.games, guesses)):
                if guess and not game.game_over:
                    feedback, _ = game.step(guess)
                    states[i] = game.state.copy()
                    feedbacks.append(feedback)
                else:
                    feedbacks.append(None)

            # Animate all reveals
            self.animate_all_reveals(all_grids, turn, guesses, feedbacks)

            # Show WIN immediately for games that just won
            win_animations = []
            for i, game in enumerate(self.games):
                if game.won and not win_indicators_shown[i]:
                    win_indicator = Text("WIN", font_size=20, color=WORDLE_GREEN, weight=BOLD)
                    win_indicator.next_to(all_grids[i], DOWN, buff=0.15)
                    win_indicators[i] = win_indicator
                    win_indicators_shown[i] = True
                    win_animations.append(FadeIn(win_indicator, scale=1.2))

            if win_animations:
                self.play(*win_animations, run_time=0.4)

            self.wait(0.2)

        # Show final results (only for games that lost - show target word)
        self.show_results(all_grids, win_indicators_shown)

        self.wait(2)

    def create_mini_grid(self):
        """Create a small Wordle grid for multi-agent display."""
        tiles = VGroup()

        for row in range(6):
            row_tiles = VGroup()
            for col in range(5):
                tile = Square(
                    side_length=self.tile_size,
                    stroke_color=WORDLE_OUTLINE,
                    stroke_width=1,
                    fill_color=WORDLE_DARK,
                    fill_opacity=1
                )
                x = (col - 2) * (self.tile_size + self.tile_gap)
                y = (2.5 - row) * (self.tile_size + self.tile_gap)
                tile.move_to([x, y, 0])
                row_tiles.add(tile)
            tiles.add(row_tiles)

        return tiles

    def animate_all_guesses(self, grids, row, guesses):
        """Animate guesses for all agents simultaneously."""
        animations = []
        letter_texts = []

        for grid_idx, (grid, guess) in enumerate(zip(grids, guesses)):
            if guess is None:
                continue

            for col, letter in enumerate(guess):
                tile = grid[row][col]
                letter_text = Text(
                    letter,
                    font_size=28,  # Larger font for YouTube Shorts
                    color=WHITE,
                    weight=BOLD
                )
                letter_text.move_to(tile.get_center())
                letter_text.set_z_index(10)
                tile.letter_text = letter_text
                letter_texts.append(letter_text)
                animations.append(FadeIn(letter_text, scale=1.1))

        if animations:
            # Show all 5 letters at once (no lag) for smoother animation
            self.play(
                *animations,
                run_time=0.2
            )
            for letter_text in letter_texts:
                self.add(letter_text)

    def animate_all_reveals(self, grids, row, guesses, feedbacks):
        """Animate reveals for all agents simultaneously."""
        animations = []

        for grid_idx, (grid, guess, feedback) in enumerate(zip(grids, guesses, feedbacks)):
            if guess is None or feedback is None:
                continue

            for col, (letter, fb) in enumerate(zip(guess, feedback)):
                tile = grid[row][col]

                if fb == 'green':
                    color = WORDLE_GREEN
                elif fb == 'yellow':
                    color = WORDLE_YELLOW
                else:
                    color = WORDLE_GREY

                animations.append(
                    tile.animate.set_fill(color=color, opacity=1).set_stroke(color=color)
                )

        if animations:
            # Reveal all colors at once (no lag) for smoother animation
            self.play(
                *animations,
                run_time=0.3
            )

    def show_results(self, grids, win_indicators_shown):
        """Show results for all agents. Only show target word for lost games."""
        wins = sum(1 for game in self.games if game.won)
        total = len(self.games)

        # Add indicators only for games that lost (WIN already shown for winning games)
        indicators = []
        for i, (grid, game) in enumerate(zip(grids, self.games)):
            if not win_indicators_shown[i]:
                # Game lost - show the target word
                indicator = Text(f"{game.target_word}", font_size=16, color=WORDLE_GREY)
                indicator.next_to(grid, DOWN, buff=0.15)
                indicators.append(indicator)

        if indicators:
            self.play(
                *[FadeIn(ind) for ind in indicators],
                run_time=0.3
            )

        # Overall result - positioned at the very bottom
        result_text = Text(
            f"WIN: {wins}/{total} ({100*wins/total:.0f}%)",
            font_size=40,
            color=WORDLE_GREEN if wins > total/2 else WHITE
        )
        result_text.to_edge(DOWN, buff=1.0)
        self.play(FadeIn(result_text))


# =============================================================================
# Alternative Scene: 16 Agents (4x4 grid)
# =============================================================================
class MultiAgentScene16(Scene):
    """Manim scene showing 16 agents (4x4 grid) playing Wordle simultaneously."""

    def setup(self):
        """Setup is called before construct."""
        self.word_list = load_word_list()
        self.common_words = load_common_words()

        # Configuration for 4x4 grid
        self.num_agents = 16
        self.grid_cols = 4
        self.grid_rows = 4

        # Use default checkpoint
        checkpoint_path = "checkpoints/wordle_ppo_vectorized.pt"

        try:
            self.agent = WordleAgent(checkpoint_path, self.word_list)
        except FileNotFoundError:
            print(f"Checkpoint not found: {checkpoint_path}")
            self.agent = None

        self.games = [WordleGame(self.common_words) for _ in range(self.num_agents)]

        # Tile sizes for 4x4 grid - adjusted for 18:9 aspect ratio
        self.tile_size = 0.32
        self.tile_gap = 0.04
        # Grid spacing for 18:9 format
        self.grid_spacing_x = 2.1
        self.grid_spacing_y = 3.5  # Adjusted for 18:9 ratio

    def construct(self):
        """Main animation construction."""
        # Title and subtitle
        title = Text("Wordle RL Agent", font_size=46, color=WHITE, weight=BOLD)
        subtitle = Text("Trained with PPO | Rendered with Manim (3B1B)", font_size=16, color=WORDLE_GREY)
        title.to_edge(UP, buff=0.6)
        subtitle.next_to(title, DOWN, buff=0.12)
        self.play(Write(title), run_time=0.3)
        self.play(FadeIn(subtitle), run_time=0.25)

        all_grids = []

        # Calculate positions for 4x4 layout - centered vertically
        grid_center_y = 0.8  # Shift grids slightly up from center
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                grid = self.create_mini_grid()
                x = (col - 1.5) * self.grid_spacing_x
                y = grid_center_y + (1.5 - row) * self.grid_spacing_y
                grid.move_to([x, y, 0])
                all_grids.append(grid)

        self.play(
            *[Create(grid) for grid in all_grids],
            run_time=0.4
        )

        states = [game.reset() for game in self.games]

        # Track which games have shown WIN indicator
        win_indicators_shown = [False] * self.num_agents
        win_indicators = [None] * self.num_agents

        for turn in range(6):
            all_done = all(game.game_over for game in self.games)
            if all_done:
                break

            guesses = []
            for i, game in enumerate(self.games):
                if not game.game_over:
                    if self.agent:
                        guess = self.agent.select_action(states[i]).upper()
                    else:
                        guess = np.random.choice(self.word_list).upper()
                    guesses.append(guess)
                else:
                    guesses.append(None)

            self.animate_all_guesses(all_grids, turn, guesses)

            feedbacks = []
            for i, (game, guess) in enumerate(zip(self.games, guesses)):
                if guess and not game.game_over:
                    feedback, _ = game.step(guess)
                    states[i] = game.state.copy()
                    feedbacks.append(feedback)
                else:
                    feedbacks.append(None)

            self.animate_all_reveals(all_grids, turn, guesses, feedbacks)

            # Show WIN immediately for games that just won
            win_animations = []
            for i, game in enumerate(self.games):
                if game.won and not win_indicators_shown[i]:
                    win_indicator = Text("WIN", font_size=16, color=WORDLE_GREEN, weight=BOLD)
                    win_indicator.next_to(all_grids[i], DOWN, buff=0.1)
                    win_indicators[i] = win_indicator
                    win_indicators_shown[i] = True
                    win_animations.append(FadeIn(win_indicator, scale=1.2))

            if win_animations:
                self.play(*win_animations, run_time=0.25)

            self.wait(0.15)

        # Show final results (only for games that lost - show target word)
        self.show_results(all_grids, win_indicators_shown)
        self.wait(2)

    def create_mini_grid(self):
        tiles = VGroup()
        for row in range(6):
            row_tiles = VGroup()
            for col in range(5):
                tile = Square(
                    side_length=self.tile_size,
                    stroke_color=WORDLE_OUTLINE,
                    stroke_width=0.5,
                    fill_color=WORDLE_DARK,
                    fill_opacity=1
                )
                x = (col - 2) * (self.tile_size + self.tile_gap)
                y = (2.5 - row) * (self.tile_size + self.tile_gap)
                tile.move_to([x, y, 0])
                row_tiles.add(tile)
            tiles.add(row_tiles)
        return tiles

    def animate_all_guesses(self, grids, row, guesses):
        animations = []
        letter_texts = []

        for grid_idx, (grid, guess) in enumerate(zip(grids, guesses)):
            if guess is None:
                continue

            for col, letter in enumerate(guess):
                tile = grid[row][col]
                letter_text = Text(letter, font_size=20, color=WHITE, weight=BOLD)  # Larger font for YouTube Shorts
                letter_text.move_to(tile.get_center())
                letter_text.set_z_index(10)
                tile.letter_text = letter_text
                letter_texts.append(letter_text)
                animations.append(FadeIn(letter_text, scale=1.05))

        if animations:
            self.play(LaggedStart(*animations, lag_ratio=0.01), run_time=0.5)
            for letter_text in letter_texts:
                self.add(letter_text)

    def animate_all_reveals(self, grids, row, guesses, feedbacks):
        animations = []

        for grid_idx, (grid, guess, feedback) in enumerate(zip(grids, guesses, feedbacks)):
            if guess is None or feedback is None:
                continue

            for col, (letter, fb) in enumerate(zip(guess, feedback)):
                tile = grid[row][col]
                if fb == 'green':
                    color = WORDLE_GREEN
                elif fb == 'yellow':
                    color = WORDLE_YELLOW
                else:
                    color = WORDLE_GREY

                animations.append(
                    tile.animate.set_fill(color=color, opacity=1).set_stroke(color=color)
                )

        if animations:
            self.play(LaggedStart(*animations, lag_ratio=0.01), run_time=0.6)

    def show_results(self, grids, win_indicators_shown):
        """Show results for all agents. Only show target word for lost games."""
        wins = sum(1 for game in self.games if game.won)
        total = len(self.games)

        # Add indicators only for games that lost (WIN already shown for winning games)
        indicators = []
        for i, (grid, game) in enumerate(zip(grids, self.games)):
            if not win_indicators_shown[i]:
                # Game lost - show the target word
                indicator = Text(f"{game.target_word}", font_size=12, color=WORDLE_GREY)
                indicator.next_to(grid, DOWN, buff=0.1)
                indicators.append(indicator)

        if indicators:
            self.play(
                *[FadeIn(ind) for ind in indicators],
                run_time=0.25
            )

        # Overall result - positioned at the very bottom
        result_text = Text(
            f"WIN: {wins}/{total} ({100*wins/total:.0f}%)",
            font_size=36,
            color=WORDLE_GREEN if wins > total/2 else WHITE
        )
        result_text.to_edge(DOWN, buff=1.0)
        self.play(FadeIn(result_text))


# =============================================================================
# Comparison Scene: Two checkpoints shown sequentially (3x3 grid each)
# =============================================================================
class ComparisonScene(Scene):
    """Compare three checkpoints: show checkpoint 100, 300, then 500 (3x3 each)."""

    def setup(self):
        """Setup is called before construct."""
        self.word_list = load_word_list()
        self.common_words = load_common_words()

        # Configuration: 9 games per checkpoint (3x3 grid)
        self.num_games = 9
        self.grid_cols = 3
        self.grid_rows = 3
        self.checkpoint_dir = "checkpoints/standard_model_1024_batch4096_lr5e4"
        self.checkpoint_100 = f"{self.checkpoint_dir}/wordle_ppo_vectorized_100.pt"
        self.checkpoint_300 = f"{self.checkpoint_dir}/wordle_ppo_vectorized_300.pt"
        self.checkpoint_500 = f"{self.checkpoint_dir}/wordle_ppo_vectorized_500.pt"

        # Load all three agents
        self.agent_100 = WordleAgent(self.checkpoint_100, self.word_list)
        self.agent_300 = WordleAgent(self.checkpoint_300, self.word_list)
        self.agent_500 = WordleAgent(self.checkpoint_500, self.word_list)

        # Fixed target words for all checkpoints - ensures fair comparison
        # Selected words: checkpoint 500 wins 9/9, checkpoint 300 wins 5/9
        self.target_words = ['BUILD', 'CHECK', 'FAULT', 'FIELD', 'ABOUT', 'ABOVE', 'AGAIN', 'APPLE', 'BEACH']

        # Create games for each checkpoint
        self.games_100 = [WordleGame(self.common_words) for _ in range(self.num_games)]
        self.games_300 = [WordleGame(self.common_words) for _ in range(self.num_games)]
        self.games_500 = [WordleGame(self.common_words) for _ in range(self.num_games)]

        # Tile sizes for 3x3 grid - same as MultiAgentScene
        self.tile_size = 0.45
        self.tile_gap = 0.05
        self.grid_spacing_x = 2.8
        self.grid_spacing_y = 4.6

    def construct(self):
        """Main animation construction."""
        # =====================================================================
        # PART 1: Checkpoint 100 (3x3 grid)
        # =====================================================================
        # Title and subtitle for checkpoint 100
        title = Text("Wordle RL Agent", font_size=52, color=WHITE, weight=BOLD)
        subtitle = Text("Step 100 | Trained with PPO", font_size=20, color=WORDLE_YELLOW)
        title.to_edge(UP, buff=0.6)
        subtitle.next_to(title, DOWN, buff=0.15)
        self.play(Write(title), run_time=0.4)
        self.play(FadeIn(subtitle), run_time=0.3)

        # Create 3x3 grids for checkpoint 100
        grids_100 = []
        grid_center_y = 0.0
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                grid = self.create_mini_grid()
                x = (col - 1) * self.grid_spacing_x
                y = grid_center_y + (1 - row) * self.grid_spacing_y
                grid.move_to([x, y, 0])
                grids_100.append(grid)

        self.play(*[Create(grid) for grid in grids_100], run_time=0.5)

        # Reset games for checkpoint 100 with target words
        states_100 = []
        for i in range(self.num_games):
            states_100.append(self.games_100[i].reset(self.target_words[i]))

        win_shown_100 = [False] * self.num_games
        win_indicators_100 = []
        all_letter_texts_100 = []  # Track all letter texts for cleanup

        # Play checkpoint 100 games
        for turn in range(6):
            if all(g.game_over for g in self.games_100):
                break

            guesses = []
            for i, game in enumerate(self.games_100):
                if not game.game_over:
                    guess = self.agent_100.select_action(states_100[i]).upper()
                    guesses.append(guess)
                else:
                    guesses.append(None)

            letter_texts = self.animate_all_guesses(grids_100, turn, guesses)
            all_letter_texts_100.extend(letter_texts)

            feedbacks = []
            for i, (game, guess) in enumerate(zip(self.games_100, guesses)):
                if guess and not game.game_over:
                    feedback, _ = game.step(guess)
                    states_100[i] = game.state.copy()
                    feedbacks.append(feedback)
                else:
                    feedbacks.append(None)

            self.animate_all_reveals(grids_100, turn, guesses, feedbacks)

            # Show WIN immediately
            win_animations = []
            for i, game in enumerate(self.games_100):
                if game.won and not win_shown_100[i]:
                    # Option 1: Simple green "WIN" text
                    win_indicator = Text("WIN", font_size=20, color=WORDLE_GREEN, weight=BOLD)
                    # Option 2: Answer word + green checkmark (commented for future use)
                    # answer_text = Text(game.target_word, font_size=16, color=WHITE, weight=BOLD)
                    # checkmark = Text(" \u2713", font_size=16, color=WORDLE_GREEN, weight=BOLD)
                    # win_indicator = VGroup(answer_text, checkmark).arrange(RIGHT, buff=0.05)
                    win_indicator.next_to(grids_100[i], DOWN, buff=0.15)
                    win_shown_100[i] = True
                    win_indicators_100.append(win_indicator)
                    win_animations.append(FadeIn(win_indicator, scale=1.2))

            if win_animations:
                self.play(*win_animations, run_time=0.4)

            self.wait(0.4)

        # Show results for lost games (target words)
        loss_indicators_100 = []
        for i, game in enumerate(self.games_100):
            if not win_shown_100[i]:
                indicator = Text(f"{game.target_word}", font_size=16, color=WORDLE_GREY)
                indicator.next_to(grids_100[i], DOWN, buff=0.15)
                loss_indicators_100.append(indicator)

        if loss_indicators_100:
            self.play(*[FadeIn(ind) for ind in loss_indicators_100], run_time=0.3)

        # Show checkpoint 100 result
        wins_100 = sum(1 for g in self.games_100 if g.won)
        result_100 = Text(
            f"WIN: {wins_100}/{self.num_games} ({100*wins_100/self.num_games:.0f}%)",
            font_size=40,
            color=WORDLE_YELLOW
        )
        result_100.to_edge(DOWN, buff=1.0)
        self.play(FadeIn(result_100), run_time=0.3)

        self.wait(1.5)

        # =====================================================================
        # TRANSITION: Fade out checkpoint 100, fade in checkpoint 300
        # =====================================================================
        all_objects_100 = [subtitle, result_100] + grids_100 + win_indicators_100 + loss_indicators_100 + all_letter_texts_100
        self.play(*[FadeOut(obj) for obj in all_objects_100], run_time=0.5)

        # =====================================================================
        # PART 2: Checkpoint 300 (3x3 grid)
        # =====================================================================
        subtitle_300 = Text("Step 300 | Trained with PPO", font_size=20, color=WORDLE_GREEN)
        subtitle_300.next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(subtitle_300), run_time=0.3)

        # Create 3x3 grids for checkpoint 300
        grids_300 = []
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                grid = self.create_mini_grid()
                x = (col - 1) * self.grid_spacing_x
                y = grid_center_y + (1 - row) * self.grid_spacing_y
                grid.move_to([x, y, 0])
                grids_300.append(grid)

        self.play(*[Create(grid) for grid in grids_300], run_time=0.5)

        # Reset games for checkpoint 300 with same target words
        states_300 = []
        for i in range(self.num_games):
            states_300.append(self.games_300[i].reset(self.target_words[i]))

        win_shown_300 = [False] * self.num_games
        win_indicators_300 = []
        all_letter_texts_300 = []  # Track all letter texts for cleanup

        # Play checkpoint 300 games
        for turn in range(6):
            if all(g.game_over for g in self.games_300):
                break

            guesses = []
            for i, game in enumerate(self.games_300):
                if not game.game_over:
                    guess = self.agent_300.select_action(states_300[i]).upper()
                    guesses.append(guess)
                else:
                    guesses.append(None)

            letter_texts = self.animate_all_guesses(grids_300, turn, guesses)
            all_letter_texts_300.extend(letter_texts)

            feedbacks = []
            for i, (game, guess) in enumerate(zip(self.games_300, guesses)):
                if guess and not game.game_over:
                    feedback, _ = game.step(guess)
                    states_300[i] = game.state.copy()
                    feedbacks.append(feedback)
                else:
                    feedbacks.append(None)

            self.animate_all_reveals(grids_300, turn, guesses, feedbacks)

            # Show WIN indicators for games that just won
            win_animations = []
            for i, game in enumerate(self.games_300):
                if game.won and not win_shown_300[i]:
                    # Option 1: Simple green "WIN" text
                    win_indicator = Text("WIN", font_size=18, color=WORDLE_GREEN, weight=BOLD)
                    # Option 2: Answer word + green checkmark (commented for future use)
                    # answer_text = Text(game.target_word, font_size=16, color=WHITE, weight=BOLD)
                    # checkmark = Text(" \u2713", font_size=16, color=WORDLE_GREEN, weight=BOLD)
                    # win_indicator = VGroup(answer_text, checkmark).arrange(RIGHT, buff=0.05)
                    win_indicator.next_to(grids_300[i], DOWN, buff=0.15)
                    win_shown_300[i] = True
                    win_indicators_300.append(win_indicator)
                    win_animations.append(FadeIn(win_indicator, scale=1.2))

            if win_animations:
                self.play(*win_animations, run_time=0.4)

            self.wait(0.4)

        # Show results for lost games (target words)
        loss_indicators_300 = []
        for i, game in enumerate(self.games_300):
            if not win_shown_300[i]:
                indicator = Text(f"{game.target_word}", font_size=16, color=WORDLE_GREY)
                indicator.next_to(grids_300[i], DOWN, buff=0.15)
                loss_indicators_300.append(indicator)

        if loss_indicators_300:
            self.play(*[FadeIn(ind) for ind in loss_indicators_300], run_time=0.3)

        # Show checkpoint 300 result
        wins_300 = sum(1 for g in self.games_300 if g.won)
        result_300 = Text(
            f"WIN: {wins_300}/{self.num_games} ({100*wins_300/self.num_games:.0f}%)",
            font_size=40,
            color=WORDLE_GREEN
        )
        result_300.to_edge(DOWN, buff=1.0)
        self.play(FadeIn(result_300), run_time=0.3)

        self.wait(1.5)

        # =====================================================================
        # TRANSITION: Fade out checkpoint 300, fade in checkpoint 400
        # =====================================================================
        all_objects_300 = [subtitle_300, result_300] + grids_300 + win_indicators_300 + loss_indicators_300 + all_letter_texts_300
        self.play(*[FadeOut(obj) for obj in all_objects_300], run_time=0.5)

        # =====================================================================
        # PART 3: Checkpoint 500 (3x3 grid)
        # =====================================================================
        # New subtitle for checkpoint 500
        subtitle_500 = Text("Step 500 | Trained with PPO", font_size=20, color=WORDLE_GREEN)
        subtitle_500.next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(subtitle_500), run_time=0.3)

        # Create 3x3 grids for checkpoint 500
        grids_500 = []
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                grid = self.create_mini_grid()
                x = (col - 1) * self.grid_spacing_x
                y = grid_center_y + (1 - row) * self.grid_spacing_y
                grid.move_to([x, y, 0])
                grids_500.append(grid)

        self.play(*[Create(grid) for grid in grids_500], run_time=0.5)

        # Reset games for checkpoint 500 with SAME target words
        states_500 = []
        for i in range(self.num_games):
            states_500.append(self.games_500[i].reset(self.target_words[i]))

        win_shown_500 = [False] * self.num_games
        win_indicators_500 = []
        all_letter_texts_500 = []

        # Play checkpoint 500 games
        for turn in range(6):
            if all(g.game_over for g in self.games_500):
                break

            guesses = []
            for i, game in enumerate(self.games_500):
                if not game.game_over:
                    guess = self.agent_500.select_action(states_500[i]).upper()
                    guesses.append(guess)
                else:
                    guesses.append(None)

            letter_texts = self.animate_all_guesses(grids_500, turn, guesses)
            all_letter_texts_500.extend(letter_texts)

            feedbacks = []
            for i, (game, guess) in enumerate(zip(self.games_500, guesses)):
                if guess and not game.game_over:
                    feedback, _ = game.step(guess)
                    states_500[i] = game.state.copy()
                    feedbacks.append(feedback)
                else:
                    feedbacks.append(None)

            self.animate_all_reveals(grids_500, turn, guesses, feedbacks)

            # Show WIN immediately
            win_animations = []
            for i, game in enumerate(self.games_500):
                if game.won and not win_shown_500[i]:
                    # Option 1: Simple green "WIN" text
                    win_indicator = Text("WIN", font_size=20, color=WORDLE_GREEN, weight=BOLD)
                    # Option 2: Answer word + green checkmark (commented for future use)
                    # answer_text = Text(game.target_word, font_size=16, color=WHITE, weight=BOLD)
                    # checkmark = Text(" \u2713", font_size=16, color=WORDLE_GREEN, weight=BOLD)
                    # win_indicator = VGroup(answer_text, checkmark).arrange(RIGHT, buff=0.05)
                    win_indicator.next_to(grids_500[i], DOWN, buff=0.15)
                    win_shown_500[i] = True
                    win_indicators_500.append(win_indicator)
                    win_animations.append(FadeIn(win_indicator, scale=1.2))

            if win_animations:
                self.play(*win_animations, run_time=0.4)

            self.wait(0.4)

        # Show results for lost games (target words)
        loss_indicators_500 = []
        for i, game in enumerate(self.games_500):
            if not win_shown_500[i]:
                indicator = Text(f"{game.target_word}", font_size=16, color=WORDLE_GREY)
                indicator.next_to(grids_500[i], DOWN, buff=0.15)
                loss_indicators_500.append(indicator)

        if loss_indicators_500:
            self.play(*[FadeIn(ind) for ind in loss_indicators_500], run_time=0.3)

        # Show checkpoint 500 result
        wins_500 = sum(1 for g in self.games_500 if g.won)
        result_500 = Text(
            f"WIN: {wins_500}/{self.num_games} ({100*wins_500/self.num_games:.0f}%)",
            font_size=40,
            color=WORDLE_GREEN
        )
        result_500.to_edge(DOWN, buff=1.0)
        self.play(FadeIn(result_500), run_time=0.3)

        self.wait(2)

    def _unused_show_training_montage(self):
        """Show a fast-forwarded training montage with 7x7 grids and loss chart."""
        import random
        import math

        # Title
        title = Text("Wordle RL Agent", font_size=48, color=WHITE, weight=BOLD)
        subtitle = Text("Training with PPO", font_size=22, color=WORDLE_YELLOW)
        title.to_edge(UP, buff=0.4)
        subtitle.next_to(title, DOWN, buff=0.1)

        self.play(Write(title), run_time=0.5)
        self.play(FadeIn(subtitle), run_time=0.3)

        # Create 7x7 mini grids for training visualization
        grid_rows, grid_cols = 7, 7
        mini_tile_size = 0.12
        mini_tile_gap = 0.02
        grid_spacing_x = 0.85
        grid_spacing_y = 1.1

        def create_tiny_grid():
            """Create a tiny Wordle grid for 7x7 display."""
            tiles = VGroup()
            for row in range(6):
                row_tiles = VGroup()
                for col in range(5):
                    tile = Square(
                        side_length=mini_tile_size,
                        stroke_color=WORDLE_OUTLINE,
                        stroke_width=0.5,
                        fill_color=WORDLE_DARK,
                        fill_opacity=1
                    )
                    x = (col - 2) * (mini_tile_size + mini_tile_gap)
                    y = (2.5 - row) * (mini_tile_size + mini_tile_gap)
                    tile.move_to([x, y, 0])
                    row_tiles.add(tile)
                tiles.add(row_tiles)
            return tiles

        # Create all 49 grids
        training_grids = []
        grids_center_y = 1.5  # Move grids up to make room for chart
        for row in range(grid_rows):
            for col in range(grid_cols):
                grid = create_tiny_grid()
                x = (col - 3) * grid_spacing_x
                y = grids_center_y + (3 - row) * grid_spacing_y
                grid.move_to([x, y, 0])
                training_grids.append(grid)

        grids_group = VGroup(*training_grids)
        self.play(Create(grids_group), run_time=0.6)

        # Create loss chart area
        chart_width = 5.0
        chart_height = 2.5
        chart_center = np.array([0, -5.5, 0])

        # Chart background
        chart_bg = Rectangle(
            width=chart_width + 0.4,
            height=chart_height + 0.6,
            fill_color="#1a1a2e",
            fill_opacity=0.9,
            stroke_color=WORDLE_OUTLINE,
            stroke_width=1
        )
        chart_bg.move_to(chart_center)

        # Chart axes
        axes_origin = chart_center + np.array([-chart_width/2, -chart_height/2, 0])
        x_axis = Line(
            axes_origin,
            axes_origin + np.array([chart_width, 0, 0]),
            color=WHITE,
            stroke_width=1.5
        )
        y_axis = Line(
            axes_origin,
            axes_origin + np.array([0, chart_height, 0]),
            color=WHITE,
            stroke_width=1.5
        )

        # Chart labels
        chart_title = Text("Loss", font_size=16, color=WHITE)
        chart_title.next_to(chart_bg, UP, buff=0.1)

        x_label = Text("Step", font_size=12, color=GREY)
        x_label.next_to(x_axis, DOWN, buff=0.15)

        # Step counter
        step_counter = Text("Step: 0", font_size=28, color=WHITE)
        step_counter.next_to(grids_group, DOWN, buff=0.3)

        self.play(
            FadeIn(chart_bg),
            Create(x_axis),
            Create(y_axis),
            FadeIn(chart_title),
            FadeIn(x_label),
            FadeIn(step_counter),
            run_time=0.4
        )

        # Generate loss curve data (simulating training loss)
        training_steps = list(range(0, 101, 2))  # 0, 2, 4, ..., 100
        loss_values = []
        for step in training_steps:
            # Simulate loss: starts high, decreases with noise
            base_loss = 5.0 * math.exp(-step / 30) + 0.5
            noise = random.uniform(-0.3, 0.3) * (1 - step/100)
            loss_values.append(max(0.3, base_loss + noise))

        # Create the loss line incrementally
        loss_line_points = []
        loss_line = None
        chart_objects = [chart_bg, x_axis, y_axis, chart_title, x_label]

        # Animate training
        for step_idx, step in enumerate(training_steps):
            if step == 0:
                continue

            # Update step counter
            new_counter = Text(f"Step: {step}", font_size=28, color=WHITE)
            new_counter.move_to(step_counter.get_center())

            # Calculate loss line point
            x_pos = axes_origin[0] + (step / 100) * chart_width
            y_pos = axes_origin[1] + (1 - loss_values[step_idx] / 5.5) * chart_height
            loss_line_points.append([x_pos, y_pos, 0])

            # Randomly colorize tiles to simulate training
            tile_animations = []
            green_prob = step / 120  # Gradually increase green probability

            for grid in training_grids:
                for row in range(6):
                    for col in range(5):
                        tile = grid[row][col]
                        if random.random() < 0.15:  # 15% chance to change color
                            if random.random() < green_prob:
                                color = WORDLE_GREEN
                            elif random.random() < 0.4:
                                color = WORDLE_YELLOW
                            else:
                                color = WORDLE_GREY
                            tile_animations.append(
                                tile.animate.set_fill(color=color, opacity=1).set_stroke(color=color)
                            )

            # Create/update loss line
            animations = [Transform(step_counter, new_counter)]

            if len(loss_line_points) >= 2:
                new_loss_line = VMobject(color=WORDLE_GREEN, stroke_width=2)
                new_loss_line.set_points_as_corners(loss_line_points)

                if loss_line is None:
                    loss_line = new_loss_line
                    animations.append(Create(loss_line))
                else:
                    animations.append(Transform(loss_line, new_loss_line))

            # Add random tile animations (limit to prevent slowdown)
            if tile_animations:
                animations.extend(tile_animations[:50])

            # Play animations - speed up as training progresses
            run_time = 0.08 if step > 20 else 0.15
            self.play(*animations, run_time=run_time)

        # Brief pause at step 100
        self.wait(0.5)

        # Flash "Training Progress" message
        complete_msg = Text("Training Progress: Step 100", font_size=28, color=WORDLE_GREEN, weight=BOLD)
        complete_msg.next_to(step_counter, DOWN, buff=0.2)
        self.play(FadeIn(complete_msg, scale=1.2), run_time=0.3)
        self.wait(0.8)

        # Fade out everything
        all_objects = [title, subtitle, grids_group, step_counter, complete_msg] + chart_objects
        if loss_line:
            all_objects.append(loss_line)
        self.play(*[FadeOut(obj) for obj in all_objects], run_time=0.5)
        self.wait(0.3)

    def create_mini_grid(self):
        """Create a small Wordle grid for 3x3 display."""
        tiles = VGroup()
        for row in range(6):
            row_tiles = VGroup()
            for col in range(5):
                tile = Square(
                    side_length=self.tile_size,
                    stroke_color=WORDLE_OUTLINE,
                    stroke_width=1,
                    fill_color=WORDLE_DARK,
                    fill_opacity=1
                )
                x = (col - 2) * (self.tile_size + self.tile_gap)
                y = (2.5 - row) * (self.tile_size + self.tile_gap)
                tile.move_to([x, y, 0])
                row_tiles.add(tile)
            tiles.add(row_tiles)
        return tiles

    def animate_all_guesses(self, grids, row, guesses):
        """Animate guesses for all grids. Returns list of letter texts for cleanup."""
        animations = []
        letter_texts = []

        for grid_idx, (grid, guess) in enumerate(zip(grids, guesses)):
            if guess is None:
                continue

            for col, letter in enumerate(guess):
                tile = grid[row][col]
                letter_text = Text(letter, font_size=28, color=WHITE, weight=BOLD)
                letter_text.move_to(tile.get_center())
                letter_text.set_z_index(10)
                tile.letter_text = letter_text
                letter_texts.append(letter_text)
                animations.append(FadeIn(letter_text, scale=1.1))

        if animations:
            self.play(LaggedStart(*animations, lag_ratio=0.01), run_time=0.1)
            for letter_text in letter_texts:
                self.add(letter_text)

        return letter_texts

    def animate_all_reveals(self, grids, row, guesses, feedbacks):
        """Animate reveals for all grids."""
        animations = []

        for grid_idx, (grid, guess, feedback) in enumerate(zip(grids, guesses, feedbacks)):
            if guess is None or feedback is None:
                continue

            for col, (letter, fb) in enumerate(zip(guess, feedback)):
                tile = grid[row][col]
                if fb == 'green':
                    color = WORDLE_GREEN
                elif fb == 'yellow':
                    color = WORDLE_YELLOW
                else:
                    color = WORDLE_GREY

                animations.append(
                    tile.animate.set_fill(color=color, opacity=1).set_stroke(color=color)
                )

        if animations:
            self.play(LaggedStart(*animations, lag_ratio=0.01), run_time=0.15)


# =============================================================================
# GIF Generation Helper
# =============================================================================
def convert_to_gif(video_path: str, output_path: str = None, fps: int = 15, width: int = 400):
    """Convert video to GIF using ffmpeg."""
    import subprocess

    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return None

    if output_path is None:
        output_path = video_path.with_suffix('.gif')
    else:
        output_path = Path(output_path)

    # Use ffmpeg to convert video to GIF
    cmd = [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-vf', f'fps={fps},scale={width}:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
        '-loop', '0',
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"GIF saved to: {output_path}")
        return str(output_path)
    except subprocess.CalledProcessError as e:
        print(f"Failed to convert to GIF: {e}")
        return None
    except FileNotFoundError:
        print("ffmpeg not found. Please install ffmpeg to generate GIFs.")
        return None


# =============================================================================
# Main entry point
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Wordle RL Agent - Multi-Game Video Renderer (Portrait 9:18)")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nUsage:")
    print("  # 9 games (3x3 grid) - Portrait format:")
    print("  manim -pql scripts/rendering/render_multi_games_portrait.py MultiAgentScene")
    print()
    print("  # 16 games (4x4 grid):")
    print("  manim -pql scripts/rendering/render_multi_games_portrait.py MultiAgentScene16")
    print()
    print("  # High quality:")
    print("  manim -pqh scripts/rendering/render_multi_games_portrait.py MultiAgentScene")
    print()
    print("Available scenes:")
    print("  - MultiAgentScene: 9 games in 3x3 grid")
    print("  - MultiAgentScene16: 16 games in 4x4 grid")
    print()
    print("After rendering, convert to GIF:")
    print("  python scripts/rendering/render_multi_games_portrait.py --to-gif <video_path>")
    print("=" * 60)

    # Handle --to-gif argument
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == '--to-gif':
        video_path = sys.argv[2]
        convert_to_gif(video_path)
    elif len(sys.argv) > 1 and sys.argv[1] == '--to-gif':
        print("Usage: python scripts/rendering/render_multi_games_portrait.py --to-gif <video_path>")
