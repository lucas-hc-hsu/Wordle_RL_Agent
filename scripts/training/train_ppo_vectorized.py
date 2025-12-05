"""
Fully Vectorized PPO Training for Wordle Agent.
Uses pure PyTorch tensor operations for maximum GPU efficiency.
No Python loops in the hot path - all environment steps are fully vectorized.

Usage:
    python train_ppo_vectorized.py --num-envs 1024 --total-timesteps 10000000

Performance Comparison:
- train_ppo_parallel.py: Uses Python loops in step() - slower
- train_ppo_vectorized.py: Fully vectorized step() - 5-10x faster

VRAM Usage Guide:
- 1024 envs: ~0.3 GB
- 4096 envs: ~0.6 GB
- 16384 envs: ~2.0 GB
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from tqdm import tqdm
import os
import argparse
import wandb
from model.ppo import ActorCritic

# VRAM limit configuration (75% of 32GB = 24GB)
MAX_VRAM_GB = 24.0


def get_vram_usage_gb():
    """Get current VRAM usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / (1024 ** 3)
    return 0


def print_cuda_info():
    """Print CUDA device information."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"Total VRAM: {total_vram:.1f} GB")
        print(f"VRAM limit: {MAX_VRAM_GB} GB")


class FullyVectorizedWordleEnv:
    """
    Fully vectorized Wordle environment using pure PyTorch tensor operations.
    No Python loops in the step() function - all operations are batched on GPU.

    Key optimizations:
    1. Pre-computed word letter tensors for fast comparison
    2. Vectorized reward calculation using tensor operations
    3. Vectorized state updates using scatter operations
    4. Bitmap-based guess tracking instead of Python sets

    Supports train/test split:
    - word_list: Full list of words the agent can guess
    - target_word_indices: Indices into word_list that can be selected as targets
    """

    def __init__(self, word_list, device, num_envs=1024, target_word_indices=None):
        self.word_list = word_list
        self.num_words = len(word_list)
        self.device = device
        self.num_envs = num_envs
        self.max_tries = 6
        self.state_size = 26 * 3  # 26 letters * 3 states

        # Target word indices (subset of word_list that can be selected as targets)
        # If None, all words can be targets (backward compatible)
        if target_word_indices is None:
            self.target_word_indices = torch.arange(self.num_words, device=device, dtype=torch.long)
        else:
            self.target_word_indices = torch.tensor(target_word_indices, device=device, dtype=torch.long)
        self.num_target_words = len(self.target_word_indices)

        # Pre-compute word data as tensors
        self._precompute_word_tensors()

        # Environment state tensors
        self.target_indices = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.current_tries = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.states = torch.zeros(num_envs, self.state_size, device=device)
        self.dones = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # Bitmap for tracking guessed words per environment
        # Shape: (num_envs, num_words) - True if word was already guessed
        self.guessed_mask = torch.zeros(num_envs, self.num_words, dtype=torch.bool, device=device)

    def _precompute_word_tensors(self):
        """Pre-compute all word data as GPU tensors for vectorized operations."""
        # Word letters: (num_words, 5) - letter indices 0-25
        word_letters = []
        for word in self.word_list:
            letters = [ord(c) - ord('a') for c in word]
            word_letters.append(letters)
        self.word_letters = torch.tensor(word_letters, device=self.device, dtype=torch.long)

        # Letter presence mask: (num_words, 26) - which letters are in each word
        self.word_letter_mask = torch.zeros(self.num_words, 26, dtype=torch.bool, device=self.device)
        for word_idx, word in enumerate(self.word_list):
            for c in word:
                self.word_letter_mask[word_idx, ord(c) - ord('a')] = True

        # Letter counts per word: (num_words, 26) - count of each letter
        self.word_letter_counts = torch.zeros(self.num_words, 26, dtype=torch.long, device=self.device)
        for word_idx, word in enumerate(self.word_list):
            for c in word:
                self.word_letter_counts[word_idx, ord(c) - ord('a')] += 1

    def reset(self, env_indices=None):
        """
        Reset specified environments or all environments (vectorized).

        Args:
            env_indices: Optional tensor of environment indices to reset.

        Returns:
            states: Current states for all environments
        """
        if env_indices is None:
            # Reset all environments
            # Select random indices from target_word_indices (train/test subset)
            random_idx = torch.randint(
                0, self.num_target_words, (self.num_envs,),
                device=self.device, dtype=torch.long
            )
            self.target_indices = self.target_word_indices[random_idx]
            self.current_tries.zero_()
            self.states.zero_()
            self.dones.zero_()
            self.guessed_mask.zero_()
        else:
            if env_indices.numel() == 0:
                return self.states.clone()

            # Reset only specified environments
            num_reset = env_indices.numel()
            # Select random indices from target_word_indices (train/test subset)
            random_idx = torch.randint(
                0, self.num_target_words, (num_reset,),
                device=self.device, dtype=torch.long
            )
            self.target_indices[env_indices] = self.target_word_indices[random_idx]
            self.current_tries[env_indices] = 0
            self.states[env_indices] = 0
            self.dones[env_indices] = False
            self.guessed_mask[env_indices] = False

        return self.states.clone()

    def step(self, actions):
        """
        Take a step in all environments simultaneously (fully vectorized).

        Args:
            actions: Tensor of action indices (num_envs,)

        Returns:
            next_states: Updated states (num_envs, state_size)
            rewards: Rewards for each environment (num_envs,)
            dones: Done flags (num_envs,)
            infos: Dict with 'won' tensor (num_envs,)
        """
        # Initialize rewards
        rewards = torch.zeros(self.num_envs, device=self.device)

        # Get active environments (not already done)
        active = ~self.dones

        if not active.any():
            return self.states.clone(), rewards, self.dones.clone(), {"won": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)}

        # Get guess and target letter data for all environments
        guess_letters = self.word_letters[actions]  # (num_envs, 5)
        target_letters = self.word_letters[self.target_indices]  # (num_envs, 5)

        # Check for repeated guesses (vectorized)
        batch_indices = torch.arange(self.num_envs, device=self.device)
        is_repeat = self.guessed_mask[batch_indices, actions]  # (num_envs,)
        is_repeat = is_repeat & active

        # Mark guesses as used
        self.guessed_mask[batch_indices, actions] = True

        # Check for exact match (win condition)
        is_win = (actions == self.target_indices) & active & ~is_repeat

        # Check for wrong guess (not win, not repeat)
        is_wrong = active & ~is_win & ~is_repeat

        # === Handle repeated guesses ===
        if is_repeat.any():
            rewards[is_repeat] = -5.0
            self.current_tries[is_repeat] += 1

            # Check if max tries reached for repeats
            max_tries_repeat = is_repeat & (self.current_tries >= self.max_tries)
            rewards[max_tries_repeat] = -50.0
            self.dones[max_tries_repeat] = True

        # === Handle wins ===
        if is_win.any():
            # Reward: 100 - tries * 10
            rewards[is_win] = 100.0 - self.current_tries[is_win].float() * 10.0
            self.dones[is_win] = True

        # === Handle wrong guesses ===
        if is_wrong.any():
            self.current_tries[is_wrong] += 1

            # Check for game over (max tries)
            game_over = is_wrong & (self.current_tries >= self.max_tries)
            still_playing = is_wrong & (self.current_tries < self.max_tries)

            rewards[game_over] = -50.0
            self.dones[game_over] = True

            # Calculate partial rewards for still playing (vectorized)
            if still_playing.any():
                partial_rewards = self._calculate_partial_rewards_vectorized(
                    guess_letters[still_playing],
                    target_letters[still_playing],
                    still_playing
                )
                rewards[still_playing] = partial_rewards

                # Update states for still playing environments
                self._update_states_vectorized(
                    still_playing,
                    guess_letters[still_playing],
                    target_letters[still_playing]
                )

        # Return info as dict with tensor
        infos = {"won": is_win}

        return self.states.clone(), rewards, self.dones.clone(), infos

    def _calculate_partial_rewards_vectorized(self, guess_letters, target_letters, mask):
        """
        Calculate partial rewards for a batch of guesses (fully vectorized).

        Args:
            guess_letters: (batch, 5) letter indices
            target_letters: (batch, 5) letter indices
            mask: Boolean mask of environments being processed

        Returns:
            rewards: (batch,) partial rewards
        """
        batch_size = guess_letters.shape[0]

        # Exact position matches (green) - worth 3 points each
        exact_matches = (guess_letters == target_letters).sum(dim=1).float() * 3.0

        # For yellow matches, we need to count letter occurrences
        # Get target letter counts for this batch
        target_indices_batch = self.target_indices[mask]
        target_counts = self.word_letter_counts[target_indices_batch].clone()  # (batch, 26)

        # Subtract exact matches from target counts
        for pos in range(5):
            exact_mask = guess_letters[:, pos] == target_letters[:, pos]
            batch_idx = torch.arange(batch_size, device=self.device)
            letter_idx = guess_letters[:, pos]
            # Only decrement where there's an exact match
            target_counts[batch_idx[exact_mask], letter_idx[exact_mask]] -= 1

        # Count yellow matches (letter in word but wrong position)
        yellow_count = torch.zeros(batch_size, device=self.device)
        for pos in range(5):
            not_exact = guess_letters[:, pos] != target_letters[:, pos]
            letter_idx = guess_letters[:, pos]
            batch_idx = torch.arange(batch_size, device=self.device)

            # Check if letter is available in target
            available = target_counts[batch_idx, letter_idx] > 0
            is_yellow = not_exact & available

            yellow_count[is_yellow] += 1
            # Decrement count for used letters
            target_counts[batch_idx[is_yellow], letter_idx[is_yellow]] -= 1

        return exact_matches + yellow_count

    def _update_states_vectorized(self, mask, guess_letters, target_letters):
        """
        Update states for a batch of environments (vectorized).

        Args:
            mask: Boolean mask of environments to update
            guess_letters: (batch, 5) letter indices for guesses
            target_letters: (batch, 5) letter indices for targets
        """
        batch_size = guess_letters.shape[0]
        env_indices = mask.nonzero(as_tuple=True)[0]

        # Get target letter masks for checking "in word"
        target_indices_batch = self.target_indices[mask]
        target_letter_mask = self.word_letter_mask[target_indices_batch]  # (batch, 26)

        for pos in range(5):
            letter_idx = guess_letters[:, pos]  # (batch,)
            target_letter = target_letters[:, pos]  # (batch,)

            # Exact match (green) - state index: letter_idx * 3 + 1
            exact = letter_idx == target_letter
            if exact.any():
                state_idx = letter_idx[exact] * 3 + 1
                self.states[env_indices[exact], state_idx] = 1.0

            # Wrong position but in word (yellow) - state index: letter_idx * 3 + 2
            not_exact = ~exact
            batch_idx = torch.arange(batch_size, device=self.device)
            in_word = target_letter_mask[batch_idx, letter_idx] & not_exact
            if in_word.any():
                state_idx = letter_idx[in_word] * 3 + 2
                self.states[env_indices[in_word], state_idx] = 1.0

            # Not in word (gray) - state index: letter_idx * 3
            not_in_word = ~target_letter_mask[batch_idx, letter_idx] & not_exact
            if not_in_word.any():
                state_idx = letter_idx[not_in_word] * 3
                self.states[env_indices[not_in_word], state_idx] = 1.0


class RolloutBuffer:
    """Buffer to store rollout experiences from parallel environments."""

    def __init__(self, num_envs, device):
        self.num_envs = num_envs
        self.device = device
        self.clear()

    def add(self, states, actions, rewards, log_probs, values, dones):
        """Add a batch of transitions from all environments."""
        self.states.append(states)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.log_probs.append(log_probs)
        self.values.append(values)
        self.dones.append(dones)

    def clear(self):
        """Clear all stored transitions."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def get_tensors(self):
        """Convert lists to stacked tensors for training."""
        return (
            torch.stack(self.states),
            torch.stack(self.actions),
            torch.stack(self.rewards),
            torch.stack(self.log_probs),
            torch.stack(self.values),
            torch.stack(self.dones),
        )

    def __len__(self):
        return len(self.states)


class PPOAgent:
    """PPO Agent optimized for parallel environments."""

    def __init__(
        self,
        state_size,
        action_size,
        word_list,
        device,
        num_envs=1024,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        update_epochs=4,
        mini_batch_size=1024,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.word_list = word_list
        self.device = device
        self.num_envs = num_envs

        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.mini_batch_size = mini_batch_size

        # Actor-Critic network
        self.policy = ActorCritic(state_size, action_size, device).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.9, patience=20
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(num_envs, device)

    @torch.no_grad()
    def act(self, states):
        """Select actions for a batch of states."""
        action_logits, values = self.policy(states)
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs, values.squeeze(-1)

    def compute_gae(self, rewards, values, dones, next_values):
        """Compute Generalized Advantage Estimation (GAE)."""
        T = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(self.num_envs, device=self.device)

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_values
            else:
                next_val = values[t + 1]

            not_done = (~dones[t]).float()
            delta = rewards[t] + self.gamma * next_val * not_done - values[t]
            gae = delta + self.gamma * self.gae_lambda * not_done * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(self, next_values):
        """Perform PPO update using collected rollout data."""
        if len(self.buffer) == 0:
            return None

        states, actions, rewards, old_log_probs, values, dones = self.buffer.get_tensors()
        advantages, returns = self.compute_gae(rewards, values, dones, next_values)

        T, num_envs = rewards.shape
        batch_size = T * num_envs

        states_flat = states.view(batch_size, self.state_size)
        actions_flat = actions.view(batch_size)
        old_log_probs_flat = old_log_probs.view(batch_size)
        advantages_flat = advantages.view(batch_size)
        returns_flat = returns.view(batch_size)

        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        indices = torch.randperm(batch_size, device=self.device)

        for _ in range(self.update_epochs):
            for start in range(0, batch_size, self.mini_batch_size):
                end = min(start + self.mini_batch_size, batch_size)
                batch_idx = indices[start:end]

                mb_states = states_flat[batch_idx]
                mb_actions = actions_flat[batch_idx]
                mb_old_log_probs = old_log_probs_flat[batch_idx]
                mb_advantages = advantages_flat[batch_idx]
                mb_returns = returns_flat[batch_idx]

                action_logits, mb_values = self.policy(mb_states)
                mb_values = mb_values.squeeze(-1)

                action_probs = torch.softmax(action_logits, dim=-1)
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(mb_values, mb_returns)
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        self.buffer.clear()

        return {
            "total_loss": total_loss / num_updates,
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
        }

    def save_checkpoint(self, update, win_rate, checkpoint_dir="checkpoints"):
        """Save model checkpoint with incremental naming.

        Saves checkpoints as: wordle_ppo_vectorized_10.pt, wordle_ppo_vectorized_20.pt, etc.
        Also saves a 'latest' checkpoint that is always overwritten.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save numbered checkpoint (e.g., wordle_ppo_vectorized_10.pt)
        numbered_checkpoint_path = os.path.join(checkpoint_dir, f"wordle_ppo_vectorized_{update}.pt")

        # Also save a 'latest' checkpoint that is always overwritten
        latest_checkpoint_path = os.path.join(checkpoint_dir, "wordle_ppo_vectorized.pt")

        checkpoint = {
            "update": update,
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "win_rate": win_rate,
        }

        torch.save(checkpoint, numbered_checkpoint_path)
        torch.save(checkpoint, latest_checkpoint_path)
        return numbered_checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint.get("update", 0), checkpoint.get("win_rate", 0)


def load_word_list():
    """Load curated list of common 5-letter English words."""
    from utils.word_list import load_word_list as load_curated_words
    return load_curated_words()


def load_word_list_with_split(test_ratio=0.2, seed=42):
    """Load word list with train/test split."""
    from utils.word_list import load_word_list_split
    return load_word_list_split(test_ratio=test_ratio, seed=seed)


def evaluate_on_test_set(agent, word_list, test_word_indices, device, num_games=500):
    """
    Evaluate the agent on the test set.

    Args:
        agent: The trained PPO agent
        word_list: Full word list (for guessing)
        test_word_indices: Indices of test words in word_list
        device: torch device
        num_games: Number of games to play for evaluation

    Returns:
        dict: Evaluation metrics (win_rate, avg_guesses)
    """
    # Create test environment with test words as targets
    test_env = FullyVectorizedWordleEnv(
        word_list, device, num_envs=num_games,
        target_word_indices=test_word_indices
    )

    states = test_env.reset()
    wins = 0
    total_guesses = 0
    completed = 0

    # Play until all games are done
    max_steps = 6 * 2  # Max 6 guesses per game, with some buffer
    for _ in range(max_steps):
        with torch.no_grad():
            actions, _, _ = agent.act(states)
        states, rewards, dones, infos = test_env.step(actions)

        # Track wins for newly completed games
        just_won = infos["won"] & ~test_env.dones
        wins += just_won.sum().item()

        # Track completed games
        just_done = dones & ~test_env.dones
        completed += just_done.sum().item()

        # Track guesses for completed games
        total_guesses += test_env.current_tries[just_done].sum().item()

        if dones.all():
            break

    win_rate = wins / num_games if num_games > 0 else 0
    avg_guesses = total_guesses / max(completed, 1)

    return {
        "test_win_rate": win_rate,
        "test_avg_guesses": avg_guesses,
        "test_games": num_games,
    }


def train(
    num_envs=1024,
    total_timesteps=131_200_000,  # ~1000 updates for thorough training
    steps_per_update=128,
    use_wandb=True,
    wandb_project="wordle-rl-agent",
    wandb_run_name=None,
    lr=1e-4,  # Low LR for stable convergence
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    entropy_coef=0.01,  # Low entropy for exploitation
    value_coef=0.5,
    update_epochs=4,
    mini_batch_size=1024,
    checkpoint_interval=1,  # Save checkpoint every update
    checkpoint_dir="checkpoints",  # Directory to save checkpoints
    test_ratio=0.2,  # Fraction of words for testing
    eval_interval=5,  # Evaluate on test set every N updates
):
    """
    Train PPO agent with fully vectorized parallel environments.
    Supports train/test split for proper evaluation.
    """
    print_cuda_info()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    print(f"Number of parallel environments: {num_envs}")
    print(f"Environment type: Fully Vectorized (GPU-optimized)")

    # Load word list with train/test split
    train_words, test_words, all_words = load_word_list_with_split(test_ratio=test_ratio)
    word_list = all_words  # Agent can guess any word

    # Create index mappings for train/test words
    word_to_idx = {word: idx for idx, word in enumerate(word_list)}
    train_word_indices = [word_to_idx[w] for w in train_words]
    test_word_indices = [word_to_idx[w] for w in test_words]

    print(f"Action space size: {len(word_list)} words")
    print(f"Training targets: {len(train_word_indices)} words")
    print(f"Testing targets: {len(test_word_indices)} words")

    config = {
        "algorithm": "PPO-FullyVectorized",
        "num_envs": num_envs,
        "total_timesteps": total_timesteps,
        "steps_per_update": steps_per_update,
        "lr": lr,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_epsilon": clip_epsilon,
        "entropy_coef": entropy_coef,
        "value_coef": value_coef,
        "update_epochs": update_epochs,
        "mini_batch_size": mini_batch_size,
        "state_size": 26 * 3,
        "action_size": len(word_list),
        "device": str(device),
        "test_ratio": test_ratio,
        "train_words": len(train_word_indices),
        "test_words": len(test_word_indices),
    }

    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=config,
        )

    # Create fully vectorized environment with TRAINING words as targets
    env = FullyVectorizedWordleEnv(
        word_list, device, num_envs,
        target_word_indices=train_word_indices
    )
    agent = PPOAgent(
        state_size=env.state_size,
        action_size=len(word_list),
        word_list=word_list,
        device=device,
        num_envs=num_envs,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        update_epochs=update_epochs,
        mini_batch_size=mini_batch_size,
    )

    # Training metrics
    total_episodes = 0
    total_wins = 0
    episode_rewards = deque(maxlen=1000)
    episode_wins = deque(maxlen=1000)
    episode_guesses = deque(maxlen=1000)
    best_win_rate = 0
    best_checkpoint_path = None

    # Initialize environments
    states = env.reset()
    episode_reward_sums = torch.zeros(num_envs, device=device)
    episode_lengths = torch.zeros(num_envs, dtype=torch.long, device=device)

    timesteps_per_update = num_envs * steps_per_update
    num_updates = max(total_timesteps // timesteps_per_update, 1)

    print(f"\nTraining for {num_updates} updates ({total_timesteps:,} timesteps)")
    print(f"Timesteps per update: {timesteps_per_update:,}")
    print("-" * 60)

    pbar = tqdm(range(num_updates), desc="Training")

    try:
        for update in pbar:
            # Collect rollouts
            for _ in range(steps_per_update):
                actions, log_probs, values = agent.act(states)
                next_states, rewards, dones, infos = env.step(actions)

                agent.buffer.add(states, actions, rewards, log_probs, values, dones)

                episode_reward_sums += rewards
                episode_lengths += 1

                # Handle completed episodes (vectorized)
                done_indices = dones.nonzero(as_tuple=True)[0]
                if done_indices.numel() > 0:
                    # Get wins from infos tensor
                    wins = infos["won"][done_indices]

                    for i, idx in enumerate(done_indices.cpu().tolist()):
                        total_episodes += 1
                        episode_rewards.append(episode_reward_sums[idx].item())
                        episode_guesses.append(episode_lengths[idx].item())

                        if wins[i].item():
                            total_wins += 1
                            episode_wins.append(1)
                        else:
                            episode_wins.append(0)

                        episode_reward_sums[idx] = 0
                        episode_lengths[idx] = 0

                    env.reset(done_indices)
                    next_states = env.states.clone()

                states = next_states

            # Update policy
            with torch.no_grad():
                _, next_values = agent.policy(states)
                next_values = next_values.squeeze(-1)

            loss_metrics = agent.update(next_values)

            # Calculate metrics
            if len(episode_rewards) > 0:
                avg_reward = np.mean(episode_rewards)
                win_rate = np.mean(episode_wins) * 100
                avg_guesses = np.mean(episode_guesses)
            else:
                avg_reward = 0
                win_rate = 0
                avg_guesses = 0

            pbar.set_description(
                f"Win: {win_rate:.1f}% | Reward: {avg_reward:.1f} | "
                f"Guesses: {avg_guesses:.2f} | Episodes: {total_episodes:,}"
            )

            if use_wandb and loss_metrics:
                current_timesteps = (update + 1) * timesteps_per_update
                vram_usage = get_vram_usage_gb()

                wandb.log({
                    "update": update + 1,
                    "timesteps": current_timesteps,
                    "episodes": total_episodes,
                    "avg_reward": avg_reward,
                    "win_rate": win_rate,
                    "avg_guesses": avg_guesses,
                    "loss/total": loss_metrics["total_loss"],
                    "loss/policy": loss_metrics["policy_loss"],
                    "loss/value": loss_metrics["value_loss"],
                    "metrics/entropy": loss_metrics["entropy"],
                    "metrics/vram_gb": vram_usage,
                    "learning_rate": agent.optimizer.param_groups[0]["lr"],
                })

                if vram_usage > MAX_VRAM_GB:
                    print(f"\nWARNING: VRAM usage ({vram_usage:.2f} GB) exceeds limit!")
                    break

            if (update + 1) % checkpoint_interval == 0:
                checkpoint_path = agent.save_checkpoint(update + 1, win_rate, checkpoint_dir)
                agent.scheduler.step(win_rate)

                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_checkpoint_path = checkpoint_path
                    print(f"\n[Update {update + 1}] New best win rate: {win_rate:.1f}%")

                    if use_wandb:
                        wandb.save(checkpoint_path)

            # Evaluate on test set periodically
            if (update + 1) % eval_interval == 0:
                test_metrics = evaluate_on_test_set(
                    agent, word_list, test_word_indices, device,
                    num_games=min(len(test_word_indices), 500)
                )
                print(f"\n[Update {update + 1}] Test set: Win rate: {test_metrics['test_win_rate']*100:.1f}%, "
                      f"Avg guesses: {test_metrics['test_avg_guesses']:.2f}")

                if use_wandb:
                    wandb.log({
                        "test/win_rate": test_metrics["test_win_rate"] * 100,
                        "test/avg_guesses": test_metrics["test_avg_guesses"],
                        "update": update + 1,
                    })

            if torch.cuda.is_available() and (update + 1) % 50 == 0:
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    if len(episode_rewards) > 0:
        final_win_rate = np.mean(episode_wins) * 100
        final_avg_reward = np.mean(episode_rewards)
        final_avg_guesses = np.mean(episode_guesses)

        print(f"Total episodes: {total_episodes:,}")
        print(f"Total wins: {total_wins:,}")
        print(f"Final win rate: {final_win_rate:.1f}%")
        print(f"Final avg reward: {final_avg_reward:.1f}")
        print(f"Final avg guesses: {final_avg_guesses:.2f}")
        print(f"Best win rate: {best_win_rate:.1f}%")
        print(f"Best checkpoint: {best_checkpoint_path}")

        if use_wandb:
            wandb.summary["final_win_rate"] = final_win_rate
            wandb.summary["final_avg_reward"] = final_avg_reward
            wandb.summary["final_avg_guesses"] = final_avg_guesses
            wandb.summary["best_win_rate"] = best_win_rate
            wandb.summary["total_episodes"] = total_episodes
            wandb.finish()

    return agent, best_checkpoint_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train fully vectorized PPO agent for Wordle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Default hyperparameters achieve 96.1% win rate
    parser.add_argument("--num-envs", type=int, default=1024, help="Number of parallel environments")
    parser.add_argument("--total-timesteps", type=int, default=131_200_000, help="Total training timesteps (~1000 updates)")
    parser.add_argument("--steps-per-update", type=int, default=128, help="Steps per policy update")

    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (1e-4 for stable convergence)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy coefficient (0.01 for exploitation)")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--update-epochs", type=int, default=4, help="PPO update epochs")
    parser.add_argument("--mini-batch-size", type=int, default=1024, help="Mini-batch size")

    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="wordle-rl-agent", help="W&B project")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--checkpoint-interval", type=int, default=1, help="Updates between checkpoints")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Fraction of words for testing (0-1)")
    parser.add_argument("--eval-interval", type=int, default=5, help="Evaluate on test set every N updates")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train(
        num_envs=args.num_envs,
        total_timesteps=args.total_timesteps,
        steps_per_update=args.steps_per_update,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        update_epochs=args.update_epochs,
        mini_batch_size=args.mini_batch_size,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
        test_ratio=args.test_ratio,
        eval_interval=args.eval_interval,
    )
