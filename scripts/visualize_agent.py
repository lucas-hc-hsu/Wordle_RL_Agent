"""
Visual Wordle Simulation Environment for RL Agent.
Displays real-time Wordle gameplay with authentic colors and animations.
Supports multiple agents playing simultaneously in a grid layout.

Usage:
    python scripts/visualize_agent.py --checkpoint checkpoints/wordle_ppo_vectorized.pt
    python scripts/visualize_agent.py --checkpoint checkpoints/wordle_ppo_vectorized.pt --num-agents 4
    python scripts/visualize_agent.py --checkpoint checkpoints/wordle_ppo_vectorized.pt --num-agents 9 --auto-play

Controls:
    SPACE - Start new game / Next guess
    A     - Toggle auto-play mode
    +/-   - Adjust speed
    1-9   - Set number of agents (1, 4, or 9)
    R     - Reset statistics
    Q/ESC - Quit

References:
    - Pygame Wordle implementation: https://github.com/baraltech/Wordle-PyGame
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pygame
import torch
import numpy as np
import argparse
import time
from collections import deque

from model.ppo import ActorCritic
from utils.word_list import load_word_list


# =============================================================================
# Color Definitions (Wordle Official Colors)
# =============================================================================
COLORS = {
    'green': "#6aaa64",       # Correct letter, correct position
    'yellow': "#c9b458",      # Correct letter, wrong position
    'grey': "#787c7e",        # Letter not in word
    'outline': "#d3d6da",     # Default box outline
    'filled': "#878a8c",      # Filled box outline
    'white': "#ffffff",
    'black': "#121213",
    'dark_grey': "#3a3a3c",
    'light_grey': "#818384",
    'win_green': "#538d4e",   # Win indicator
    'lose_red': "#b55a5a",    # Lose indicator
}

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# Pre-convert colors
GREEN = hex_to_rgb(COLORS['green'])
YELLOW = hex_to_rgb(COLORS['yellow'])
GREY = hex_to_rgb(COLORS['grey'])
OUTLINE = hex_to_rgb(COLORS['outline'])
FILLED = hex_to_rgb(COLORS['filled'])
WHITE = hex_to_rgb(COLORS['white'])
BLACK = hex_to_rgb(COLORS['black'])
DARK_GREY = hex_to_rgb(COLORS['dark_grey'])
LIGHT_GREY = hex_to_rgb(COLORS['light_grey'])
WIN_GREEN = hex_to_rgb(COLORS['win_green'])
LOSE_RED = hex_to_rgb(COLORS['lose_red'])


# =============================================================================
# Single Wordle Game Instance
# =============================================================================
class WordleGameInstance:
    """Represents a single Wordle game with visual state."""

    def __init__(self, word_list, x, y, scale=1.0):
        self.word_list = word_list
        self.x = x
        self.y = y
        self.scale = scale

        # Scaled dimensions
        self.tile_size = int(50 * scale)
        self.tile_gap = int(5 * scale)
        self.font_size = int(40 * scale)

        # Game state
        self.target_word = ""
        self.current_guess = 0
        self.state = np.zeros(26 * 3)
        self.guesses = []
        self.feedbacks = []
        self.game_over = False
        self.won = False

        # Visual state for each tile: (letter, color, revealed)
        self.tiles = [[("", BLACK, False) for _ in range(5)] for _ in range(6)]

    def reset(self, target_word=None):
        """Reset game with new target word."""
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
        self.tiles = [[("", BLACK, False) for _ in range(5)] for _ in range(6)]
        return self.state.copy()

    def get_feedback(self, guess):
        """Calculate Wordle feedback for a guess."""
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
        """Make a guess and update state."""
        if self.game_over:
            return None, True

        guess = guess.upper()
        feedback = self.get_feedback(guess)

        self.guesses.append(guess)
        self.feedbacks.append(feedback)

        # Update tiles
        for col, (letter, fb) in enumerate(zip(guess, feedback)):
            if fb == 'green':
                color = GREEN
            elif fb == 'yellow':
                color = YELLOW
            else:
                color = GREY

            self.tiles[self.current_guess][col] = (letter, color, True)

            # Update state
            idx = ord(letter.lower()) - ord('a')
            if fb == 'green':
                self.state[idx * 3 + 1] = 1
            elif fb == 'yellow':
                self.state[idx * 3 + 2] = 1
            else:
                self.state[idx * 3] = 1

        self.current_guess += 1

        # Check win/lose
        self.won = guess == self.target_word
        self.game_over = self.won or self.current_guess >= 6

        return feedback, self.game_over

    def draw(self, screen, font):
        """Draw this game instance."""
        # Draw game border/background
        total_width = 5 * (self.tile_size + self.tile_gap) - self.tile_gap + 10
        total_height = 6 * (self.tile_size + self.tile_gap) - self.tile_gap + 40

        # Background with status color
        if self.game_over:
            bg_color = WIN_GREEN if self.won else LOSE_RED
            border_rect = pygame.Rect(self.x - 5, self.y - 25, total_width, total_height)
            pygame.draw.rect(screen, bg_color, border_rect, border_radius=8)

        # Draw target word label
        label_font = pygame.font.Font(None, int(20 * self.scale))
        if self.game_over:
            label = label_font.render(f"Target: {self.target_word}", True, WHITE)
        else:
            label = label_font.render("Playing...", True, LIGHT_GREY)
        screen.blit(label, (self.x, self.y - 22))

        # Draw tiles
        for row in range(6):
            for col in range(5):
                tile_x = self.x + col * (self.tile_size + self.tile_gap)
                tile_y = self.y + row * (self.tile_size + self.tile_gap)

                letter, color, revealed = self.tiles[row][col]

                # Draw tile background
                rect = pygame.Rect(tile_x, tile_y, self.tile_size, self.tile_size)
                pygame.draw.rect(screen, color if revealed else BLACK, rect)

                # Draw border
                border_color = color if revealed else (FILLED if letter else OUTLINE)
                pygame.draw.rect(screen, border_color, rect, 2)

                # Draw letter
                if letter:
                    text = font.render(letter, True, WHITE)
                    text_rect = text.get_rect(center=(tile_x + self.tile_size // 2,
                                                       tile_y + self.tile_size // 2))
                    screen.blit(text, text_rect)


# =============================================================================
# Multi-Agent Visual Environment
# =============================================================================
class MultiAgentWordleEnv:
    """Visual Wordle environment supporting multiple agents."""

    # Layout configurations: (rows, cols, scale)
    LAYOUTS = {
        1: (1, 1, 1.0),
        4: (2, 2, 0.7),
        9: (3, 3, 0.5),
    }

    def __init__(self, word_list, num_agents=1, width=900, height=750):
        pygame.init()
        pygame.display.set_caption("Wordle RL Agent - Multi-Agent View")

        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        self.word_list = word_list
        self.num_agents = num_agents

        # Fonts (will be recreated per scale)
        self.fonts = {}

        # Create game instances
        self.games = []
        self._setup_games(num_agents)

        # Stats
        self.total_games = 0
        self.total_wins = 0
        self.total_guesses = 0

    def _setup_games(self, num_agents):
        """Setup game instances in grid layout."""
        if num_agents not in self.LAYOUTS:
            # Find closest valid layout
            valid = sorted(self.LAYOUTS.keys())
            num_agents = min(valid, key=lambda x: abs(x - num_agents))

        self.num_agents = num_agents
        rows, cols, scale = self.LAYOUTS[num_agents]

        # Calculate grid dimensions
        tile_size = int(50 * scale)
        tile_gap = int(5 * scale)
        game_width = 5 * (tile_size + tile_gap) - tile_gap + 20
        game_height = 6 * (tile_size + tile_gap) - tile_gap + 50

        # Calculate starting positions to center the grid
        total_width = cols * game_width + (cols - 1) * 20
        total_height = rows * game_height + (rows - 1) * 20

        start_x = (self.width - total_width) // 2
        # Start from top, leave space at bottom for stats panel (90px)
        start_y = 10 + (self.height - 100 - total_height) // 2

        self.games = []
        for row in range(rows):
            for col in range(cols):
                x = start_x + col * (game_width + 20)
                y = start_y + row * (game_height + 20)
                game = WordleGameInstance(self.word_list, x, y, scale)
                self.games.append(game)

        # Create font for this scale
        self.current_scale = scale
        self.fonts[scale] = pygame.font.Font(None, int(40 * scale))

    def set_num_agents(self, num_agents):
        """Change the number of visible agents."""
        if num_agents in self.LAYOUTS and num_agents != self.num_agents:
            self._setup_games(num_agents)
            self.reset_all()

    def reset_all(self):
        """Reset all game instances."""
        states = []
        for game in self.games:
            state = game.reset()
            states.append(state)
        return states

    def reset_game(self, idx):
        """Reset a specific game instance."""
        return self.games[idx].reset()

    def step(self, idx, guess):
        """Make a guess in a specific game."""
        return self.games[idx].step(guess)

    def get_states(self):
        """Get current states of all games."""
        return [game.state.copy() for game in self.games]

    def all_games_over(self):
        """Check if all games are finished."""
        return all(game.game_over for game in self.games)

    def render(self, stats=None):
        """Render all game instances."""
        self.screen.fill(BLACK)

        # Draw all games (no title in multi-agent view to save space)
        font = self.fonts.get(self.current_scale, pygame.font.Font(None, 40))
        for game in self.games:
            game.draw(self.screen, font)

        # Draw stats panel at bottom
        if stats:
            self._draw_stats(stats)

        pygame.display.flip()

    def _draw_stats(self, stats):
        """Draw statistics panel."""
        y = self.height - 90

        # Background panel
        panel_rect = pygame.Rect(10, y - 10, self.width - 20, 90)
        pygame.draw.rect(self.screen, DARK_GREY, panel_rect, border_radius=8)

        stats_font = pygame.font.Font(None, 28)
        small_font = pygame.font.Font(None, 22)

        # Stats text
        win_rate = stats.get('win_rate', 0)
        games = stats.get('games', 0)
        avg_guesses = stats.get('avg_guesses', 0)
        speed = stats.get('speed', 1.0)
        auto = stats.get('auto_play', False)

        # Line 1
        text1 = f"Total Games: {games}  |  Win Rate: {win_rate:.1f}%  |  Avg Guesses: {avg_guesses:.2f}"
        surf1 = stats_font.render(text1, True, WHITE)
        self.screen.blit(surf1, (20, y))

        # Line 2
        mode = "AUTO" if auto else "MANUAL"
        text2 = f"Speed: {speed:.1f}x  |  Mode: {mode}  |  Agents: {self.num_agents}"
        surf2 = stats_font.render(text2, True, LIGHT_GREY)
        self.screen.blit(surf2, (20, y + 25))

        # Controls hint
        text3 = "SPACE: Next | A: Auto | +/-: Speed | 1/4/9: Agents | R: Reset | Q: Quit"
        surf3 = small_font.render(text3, True, LIGHT_GREY)
        self.screen.blit(surf3, (20, y + 55))


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

    @torch.no_grad()
    def select_actions_batch(self, states, deterministic=True):
        """Select actions for multiple states."""
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        action_logits, _ = self.policy(states_tensor)

        if deterministic:
            actions = action_logits.argmax(dim=-1).cpu().numpy()
        else:
            probs = torch.softmax(action_logits, dim=-1)
            actions = torch.multinomial(probs, 1).squeeze(-1).cpu().numpy()

        return [self.word_list[a] for a in actions]


# =============================================================================
# Main Visualization Loop
# =============================================================================
def run_visualization(checkpoint_path, num_agents=1, speed=1.0, auto_play=False):
    """Run the visual Wordle simulation."""
    word_list = load_word_list()
    print(f"Loaded {len(word_list)} words")

    # Initialize agent
    agent = WordleAgent(checkpoint_path, word_list)

    # Initialize visual environment
    env = MultiAgentWordleEnv(word_list, num_agents)

    # Statistics
    stats = {
        'games': 0,
        'wins': 0,
        'total_guesses': 0,
        'win_rate': 0,
        'avg_guesses': 0,
        'speed': speed,
        'auto_play': auto_play,
    }

    running = True
    games_active = False
    waiting_for_input = True

    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    waiting_for_input = False
                elif event.key == pygame.K_a:
                    stats['auto_play'] = not stats['auto_play']
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    stats['speed'] = min(5.0, stats['speed'] + 0.5)
                elif event.key == pygame.K_MINUS:
                    stats['speed'] = max(0.1, stats['speed'] - 0.5)
                elif event.key == pygame.K_1:
                    env.set_num_agents(1)
                    games_active = False
                    waiting_for_input = True
                elif event.key == pygame.K_4:
                    env.set_num_agents(4)
                    games_active = False
                    waiting_for_input = True
                elif event.key == pygame.K_9:
                    env.set_num_agents(9)
                    games_active = False
                    waiting_for_input = True
                elif event.key == pygame.K_r:
                    stats = {
                        'games': 0, 'wins': 0, 'total_guesses': 0,
                        'win_rate': 0, 'avg_guesses': 0,
                        'speed': stats['speed'], 'auto_play': stats['auto_play']
                    }
                    games_active = False
                    waiting_for_input = True

        # Auto-play mode
        if stats['auto_play']:
            waiting_for_input = False

        # Game logic
        if not games_active and not waiting_for_input:
            # Start new games for all agents
            env.reset_all()
            games_active = True
            waiting_for_input = True

        elif games_active and not waiting_for_input:
            # Get states for all active games
            active_indices = [i for i, game in enumerate(env.games) if not game.game_over]

            if active_indices:
                # Get states and select actions
                states = [env.games[i].state for i in active_indices]
                guesses = agent.select_actions_batch(states)

                # Apply guesses
                for idx, guess in zip(active_indices, guesses):
                    env.step(idx, guess)

            # Check if all games are done
            if env.all_games_over():
                games_active = False

                # Update stats
                for game in env.games:
                    stats['games'] += 1
                    if game.won:
                        stats['wins'] += 1
                        stats['total_guesses'] += game.current_guess
                    else:
                        stats['total_guesses'] += 7

                if stats['games'] > 0:
                    stats['win_rate'] = stats['wins'] / stats['games'] * 100
                    stats['avg_guesses'] = stats['total_guesses'] / stats['games']

                # Pause before next round
                if stats['auto_play']:
                    time.sleep(1.5 / stats['speed'])

            waiting_for_input = True

        # Render
        env.render(stats)

        # Frame rate control
        if stats['auto_play'] and not waiting_for_input:
            time.sleep(0.3 / stats['speed'])
        else:
            env.clock.tick(60)

    pygame.quit()


def main():
    parser = argparse.ArgumentParser(
        description="Visual Wordle simulation for RL agent (multi-agent support)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/wordle_ppo_vectorized.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--num-agents", type=int, default=1, choices=[1, 4, 9],
        help="Number of agents to display (1, 4, or 9)"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Animation speed multiplier"
    )
    parser.add_argument(
        "--auto-play", action="store_true",
        help="Start in auto-play mode"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Wordle RL Agent Visual Simulation (Multi-Agent)")
    print("=" * 60)
    print("\nControls:")
    print("  SPACE   - Start new game / Make next guess")
    print("  A       - Toggle auto-play mode")
    print("  +/-     - Adjust speed")
    print("  1/4/9   - Set number of agents")
    print("  R       - Reset statistics")
    print("  Q/ESC   - Quit")
    print("=" * 60)

    run_visualization(
        checkpoint_path=args.checkpoint,
        num_agents=args.num_agents,
        speed=args.speed,
        auto_play=args.auto_play
    )


if __name__ == "__main__":
    main()
