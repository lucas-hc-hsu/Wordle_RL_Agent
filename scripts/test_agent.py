"""
Test script for evaluating trained Wordle RL agents.
Uses parallel/batched inference for fast evaluation.

Usage:
    python scripts/test_agent.py --checkpoint checkpoints/wordle_ppo_vectorized.pt --num-games 1000
    python scripts/test_agent.py --checkpoint checkpoints/wordle_ppo_vectorized.pt --interactive
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import argparse
from tqdm import tqdm
from collections import Counter

from model.ppo import ActorCritic
from utils.word_list import load_word_list


class ParallelWordleEvaluator:
    """
    Parallel Wordle evaluator that runs multiple games simultaneously.
    Uses batched inference for maximum speed.
    """

    def __init__(self, word_list, device, num_games=1000):
        self.word_list = word_list
        self.num_words = len(word_list)
        self.device = device
        self.num_games = num_games
        self.max_tries = 6
        self.state_size = 26 * 3

        # Pre-compute word data
        self._precompute_word_tensors()

        # Game state tensors
        self.target_indices = torch.zeros(num_games, dtype=torch.long, device=device)
        self.states = torch.zeros(num_games, self.state_size, device=device)
        self.current_tries = torch.zeros(num_games, dtype=torch.long, device=device)
        self.dones = torch.zeros(num_games, dtype=torch.bool, device=device)
        self.won = torch.zeros(num_games, dtype=torch.bool, device=device)
        self.guessed_mask = torch.zeros(num_games, self.num_words, dtype=torch.bool, device=device)

    def _precompute_word_tensors(self):
        """Pre-compute word letter tensors."""
        word_letters = []
        for word in self.word_list:
            letters = [ord(c) - ord('a') for c in word]
            word_letters.append(letters)
        self.word_letters = torch.tensor(word_letters, device=self.device, dtype=torch.long)

        # Letter presence mask
        self.word_letter_mask = torch.zeros(self.num_words, 26, dtype=torch.bool, device=self.device)
        for word_idx, word in enumerate(self.word_list):
            for c in word:
                self.word_letter_mask[word_idx, ord(c) - ord('a')] = True

    def reset(self):
        """Reset all games with random target words."""
        self.target_indices = torch.randint(
            0, self.num_words, (self.num_games,),
            device=self.device, dtype=torch.long
        )
        self.states.zero_()
        self.current_tries.zero_()
        self.dones.zero_()
        self.won.zero_()
        self.guessed_mask.zero_()

    def step(self, actions):
        """
        Take a step in all games (vectorized).

        Args:
            actions: Tensor of action indices (num_games,)
        """
        active = ~self.dones

        if not active.any():
            return

        # Get guess and target letters
        guess_letters = self.word_letters[actions]  # (num_games, 5)
        target_letters = self.word_letters[self.target_indices]  # (num_games, 5)

        # Check for repeated guesses
        batch_indices = torch.arange(self.num_games, device=self.device)
        is_repeat = self.guessed_mask[batch_indices, actions] & active

        # Mark guesses as used
        self.guessed_mask[batch_indices, actions] = True

        # Check for wins
        is_win = (actions == self.target_indices) & active & ~is_repeat
        self.won[is_win] = True
        self.dones[is_win] = True

        # Handle wrong guesses
        is_wrong = active & ~is_win & ~is_repeat
        self.current_tries[is_wrong] += 1

        # Check for game over
        game_over = is_wrong & (self.current_tries >= self.max_tries)
        self.dones[game_over] = True

        # Update states for games still playing
        still_playing = is_wrong & ~game_over
        if still_playing.any():
            self._update_states_vectorized(
                still_playing,
                guess_letters[still_playing],
                target_letters[still_playing]
            )

        # Handle repeats
        self.current_tries[is_repeat] += 1
        repeat_game_over = is_repeat & (self.current_tries >= self.max_tries)
        self.dones[repeat_game_over] = True

    def _update_states_vectorized(self, mask, guess_letters, target_letters):
        """Update states for a batch of games."""
        batch_size = guess_letters.shape[0]
        env_indices = mask.nonzero(as_tuple=True)[0]

        target_indices_batch = self.target_indices[mask]
        target_letter_mask = self.word_letter_mask[target_indices_batch]

        for pos in range(5):
            letter_idx = guess_letters[:, pos]
            target_letter = target_letters[:, pos]

            # Green (exact match)
            exact = letter_idx == target_letter
            if exact.any():
                state_idx = letter_idx[exact] * 3 + 1
                self.states[env_indices[exact], state_idx] = 1.0

            # Yellow (in word but wrong position)
            not_exact = ~exact
            batch_idx = torch.arange(batch_size, device=self.device)
            in_word = target_letter_mask[batch_idx, letter_idx] & not_exact
            if in_word.any():
                state_idx = letter_idx[in_word] * 3 + 2
                self.states[env_indices[in_word], state_idx] = 1.0

            # Gray (not in word)
            not_in_word = ~target_letter_mask[batch_idx, letter_idx] & not_exact
            if not_in_word.any():
                state_idx = letter_idx[not_in_word] * 3
                self.states[env_indices[not_in_word], state_idx] = 1.0

    def get_results(self):
        """Get evaluation results."""
        wins = self.won.sum().item()
        losses = self.num_games - wins

        # Get guess counts for wins
        win_tries = self.current_tries[self.won].cpu().numpy()

        # Count distribution
        guess_dist = Counter()
        for tries in win_tries:
            guess_dist[int(tries) + 1] += 1  # +1 because current_tries is 0-indexed
        guess_dist[7] = losses  # 7 = loss

        return {
            'wins': wins,
            'losses': losses,
            'win_rate': wins / self.num_games * 100,
            'avg_guesses_won': np.mean(win_tries + 1) if wins > 0 else 0,
            'guess_distribution': dict(sorted(guess_dist.items())),
        }


class WordleAgent:
    """Agent wrapper for inference."""

    def __init__(self, checkpoint_path, word_list, device):
        self.device = device
        self.word_list = word_list

        state_size = 26 * 3
        action_size = len(word_list)

        self.policy = ActorCritic(state_size, action_size, device).to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy.eval()

        self.checkpoint_info = {
            'update': checkpoint.get('update', checkpoint.get('episode', 'N/A')),
            'win_rate': checkpoint.get('win_rate', 'N/A'),
        }

    @torch.no_grad()
    def select_actions_batch(self, states, deterministic=False):
        """Select actions for a batch of states (parallel inference)."""
        action_logits, _ = self.policy(states)

        if deterministic:
            actions = action_logits.argmax(dim=-1)
        else:
            action_probs = torch.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(action_probs)
            actions = dist.sample()

        return actions

    def select_action(self, state, deterministic=False):
        """Select action for a single state."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_logits, _ = self.policy(state_tensor)

            if deterministic:
                action = action_logits.argmax(dim=-1).item()
            else:
                action_probs = torch.softmax(action_logits, dim=-1)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample().item()

        return self.word_list[action]


def evaluate_parallel(agent, word_list, num_games=1000, deterministic=True):
    """
    Evaluate agent using parallel games (fast).

    Args:
        agent: WordleAgent instance
        word_list: List of valid words
        num_games: Number of games to evaluate
        deterministic: Use deterministic action selection

    Returns:
        dict: Evaluation statistics
    """
    device = agent.device
    evaluator = ParallelWordleEvaluator(word_list, device, num_games)
    evaluator.reset()

    max_steps = 6 * 2  # Max 6 tries, with some buffer for edge cases

    print(f"Running {num_games} games in parallel...")

    for step in tqdm(range(max_steps), desc="Steps"):
        if evaluator.dones.all():
            break

        # Batch inference
        actions = agent.select_actions_batch(evaluator.states, deterministic=deterministic)

        # Step all games
        evaluator.step(actions)

    results = evaluator.get_results()
    results['num_games'] = num_games

    return results


def evaluate_sequential(agent, word_list, num_games=100, deterministic=False, verbose=False):
    """
    Evaluate agent sequentially (for detailed output).
    """
    wins = 0
    total_guesses = []
    guess_distribution = Counter()

    for game_idx in tqdm(range(num_games), desc="Evaluating", disable=verbose):
        target = np.random.choice(word_list)
        state = np.zeros(26 * 3)
        guessed = set()

        if verbose:
            print(f"\n{'='*50}")
            print(f"Game {game_idx + 1}: Target = {target}")
            print('='*50)

        won = False
        tries = 0

        for attempt in range(6):
            guess = agent.select_action(state, deterministic=deterministic)

            if guess in guessed:
                continue

            guessed.add(guess)
            tries += 1

            # Calculate feedback
            feedback = ['b'] * 5
            target_letters = list(target)

            for i, letter in enumerate(guess):
                if letter == target[i]:
                    feedback[i] = 'g'
                    target_letters[i] = None

            for i, letter in enumerate(guess):
                if feedback[i] != 'g' and letter in target_letters:
                    feedback[i] = 'y'
                    target_letters[target_letters.index(letter)] = None

            if verbose:
                feedback_str = ''.join(feedback)
                colored = ''
                for letter, fb in zip(guess, feedback):
                    if fb == 'g':
                        colored += f'\033[92m{letter}\033[0m'
                    elif fb == 'y':
                        colored += f'\033[93m{letter}\033[0m'
                    else:
                        colored += f'\033[90m{letter}\033[0m'
                print(f"  Guess {tries}: {guess} -> {colored} ({feedback_str})")

            # Update state
            for i, (letter, fb) in enumerate(zip(guess, feedback)):
                idx = ord(letter) - ord('a')
                if fb == 'g':
                    state[idx * 3 + 1] = 1
                elif fb == 'y':
                    state[idx * 3 + 2] = 1
                else:
                    state[idx * 3] = 1

            if guess == target:
                won = True
                break

        if won:
            wins += 1
            guess_distribution[tries] += 1
            if verbose:
                print(f"  Won in {tries} guesses!")
        else:
            guess_distribution[7] += 1
            if verbose:
                print(f"  Lost! Target was: {target}")

        total_guesses.append(tries)

    return {
        'num_games': num_games,
        'wins': wins,
        'losses': num_games - wins,
        'win_rate': wins / num_games * 100,
        'avg_guesses_won': np.mean([g for g in total_guesses if g <= 6]) if wins > 0 else 0,
        'guess_distribution': dict(sorted(guess_distribution.items())),
    }


def interactive_play(agent, word_list):
    """Play interactively with the agent."""
    print("\n" + "="*60)
    print("Interactive Wordle with AI Agent")
    print("="*60)
    print("\nThe agent will suggest guesses. You provide feedback.")
    print("Feedback format: 5 characters using g/y/b")
    print("  g = green (correct letter, correct position)")
    print("  y = yellow (correct letter, wrong position)")
    print("  b = black/gray (letter not in word)")
    print("\nType 'quit' to exit, 'new' for new game")
    print("="*60)

    while True:
        print("\n--- New Game ---")
        state = np.zeros(26 * 3)

        for attempt in range(1, 7):
            guess = agent.select_action(state, deterministic=True)
            print(f"\nAttempt {attempt}/6")
            print(f"Agent suggests: {guess.upper()}")

            feedback = input("Enter feedback (ggggg if won, or 'quit'/'new'): ").lower().strip()

            if feedback == 'quit':
                print("Goodbye!")
                return
            elif feedback == 'new':
                break
            elif feedback == 'ggggg':
                print(f"\n🎉 Won in {attempt} guesses!")
                break
            elif len(feedback) != 5 or not all(c in 'gyb' for c in feedback):
                print("Invalid feedback. Use exactly 5 characters: g, y, or b")
                continue
            else:
                for i, (letter, fb) in enumerate(zip(guess, feedback)):
                    idx = ord(letter) - ord('a')
                    if fb == 'g':
                        state[idx * 3 + 1] = 1
                    elif fb == 'y':
                        state[idx * 3 + 2] = 1
                    else:
                        state[idx * 3] = 1
        else:
            print("\n😢 Out of attempts!")

        again = input("\nPlay again? (y/n): ").lower().strip()
        if again != 'y':
            break

    print("Thanks for playing!")


def print_results(stats):
    """Print evaluation results."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Games played: {stats['num_games']}")
    print(f"Wins: {stats['wins']}")
    print(f"Losses: {stats['losses']}")
    print(f"Win rate: {stats['win_rate']:.2f}%")
    print(f"Average guesses (wins only): {stats['avg_guesses_won']:.2f}")
    print("\nGuess distribution:")

    max_count = max(stats['guess_distribution'].values()) if stats['guess_distribution'] else 1

    for guesses in range(1, 8):
        count = stats['guess_distribution'].get(guesses, 0)
        bar_len = int(count * 40 / max_count) if max_count > 0 else 0
        bar = '█' * bar_len
        label = 'X' if guesses == 7 else str(guesses)
        pct = count / stats['num_games'] * 100
        print(f"  {label}: {bar:<40} {count:>5} ({pct:>5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Test trained Wordle RL agent (with parallel evaluation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/wordle_ppo_vectorized.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--num-games", type=int, default=1000,
        help="Number of games to evaluate"
    )
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Use deterministic action selection"
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Use sequential evaluation (slower, for debugging)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print each game's details (implies --sequential)"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Play interactively with the agent"
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    word_list = load_word_list()
    print(f"Loaded {len(word_list)} words")

    print(f"Loading checkpoint: {args.checkpoint}")
    agent = WordleAgent(args.checkpoint, word_list, device)
    print(f"Checkpoint info: update={agent.checkpoint_info['update']}, "
          f"win_rate={agent.checkpoint_info['win_rate']}")

    if args.interactive:
        interactive_play(agent, word_list)
    elif args.verbose or args.sequential:
        stats = evaluate_sequential(
            agent, word_list,
            num_games=args.num_games,
            deterministic=args.deterministic,
            verbose=args.verbose
        )
        print_results(stats)
    else:
        stats = evaluate_parallel(
            agent, word_list,
            num_games=args.num_games,
            deterministic=args.deterministic
        )
        print_results(stats)


if __name__ == "__main__":
    main()
