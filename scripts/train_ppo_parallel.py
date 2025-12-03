"""
Parallel PPO Training for Wordle Agent.
Runs multiple Wordle games simultaneously for faster training and better GPU utilization.

Usage:
    python train_ppo_parallel.py --num-envs 256 --total-timesteps 10000000

VRAM Usage Guide:
- 256 envs: ~0.1 GB
- 1024 envs: ~0.2 GB
- 4096 envs: ~0.5 GB
- 16384 envs: ~1.5 GB
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from tqdm import tqdm
import os
import argparse
import wandb
from model.ppo import ActorCritic

# VRAM limit configuration (50% of 32GB = 16GB)
MAX_VRAM_GB = 16.0


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


class VectorizedWordleEnv:
    """
    Vectorized Wordle environment that runs multiple games in parallel on GPU.
    All operations are batched for maximum efficiency.
    """

    def __init__(self, word_list, device, num_envs=256):
        self.word_list = word_list
        self.num_words = len(word_list)
        self.device = device
        self.num_envs = num_envs
        self.max_tries = 6
        self.state_size = 26 * 3  # 26 letters * 3 states (not in word, correct, wrong position)

        # Pre-compute word to index mapping
        self.word_to_idx = {word: idx for idx, word in enumerate(word_list)}

        # Vectorized state tracking on GPU
        self.target_indices = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.current_tries = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.states = torch.zeros(num_envs, self.state_size, device=device)
        self.dones = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # Track guessed words per environment
        self.guessed_words = [set() for _ in range(num_envs)]

        # Pre-compute letter indices for all words (5 letters per word)
        self._precompute_word_data()

    def _precompute_word_data(self):
        """Pre-compute letter information for all words for fast lookup."""
        # word_letters[i] = [letter_idx_0, letter_idx_1, ..., letter_idx_4] for word i
        word_letters = []
        for word in self.word_list:
            letters = [ord(c) - ord('a') for c in word]
            word_letters.append(letters)
        self.word_letters = torch.tensor(word_letters, device=self.device, dtype=torch.long)

    def reset(self, env_indices=None):
        """
        Reset specified environments or all environments.

        Args:
            env_indices: Optional tensor of environment indices to reset.
                        If None, resets all environments.

        Returns:
            states: Current states for all environments (num_envs, state_size)
        """
        if env_indices is None:
            env_indices = torch.arange(self.num_envs, device=self.device)
            indices_list = range(self.num_envs)
        else:
            if env_indices.numel() == 0:
                return self.states.clone()
            indices_list = env_indices.cpu().tolist()
            if not isinstance(indices_list, list):
                indices_list = [indices_list]

        # Reset each environment
        for idx in indices_list:
            target_idx = random.randint(0, self.num_words - 1)
            self.target_indices[idx] = target_idx
            self.current_tries[idx] = 0
            self.states[idx] = 0
            self.dones[idx] = False
            self.guessed_words[idx] = set()

        return self.states.clone()

    def step(self, actions):
        """
        Take a step in all environments simultaneously.

        Args:
            actions: Tensor of action indices (num_envs,)

        Returns:
            next_states: Updated states (num_envs, state_size)
            rewards: Rewards for each environment (num_envs,)
            dones: Done flags (num_envs,)
            infos: List of info dicts with 'won' key
        """
        rewards = torch.zeros(self.num_envs, device=self.device)
        infos = [{"won": False} for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            if self.dones[i]:
                continue

            action_idx = actions[i].item()
            guess = self.word_list[action_idx]
            target_idx = self.target_indices[i].item()
            target = self.word_list[target_idx]

            # Penalty for repeated guess
            if guess in self.guessed_words[i]:
                rewards[i] = -5.0
                self.current_tries[i] += 1
                if self.current_tries[i] >= self.max_tries:
                    rewards[i] = -50.0
                    self.dones[i] = True
                continue

            self.guessed_words[i].add(guess)

            # Check for win
            if guess == target:
                # Reward based on number of tries (higher reward for fewer tries)
                rewards[i] = 100.0 - self.current_tries[i].item() * 10.0
                self.dones[i] = True
                infos[i]["won"] = True
            else:
                self.current_tries[i] += 1
                if self.current_tries[i] >= self.max_tries:
                    rewards[i] = -50.0
                    self.dones[i] = True
                else:
                    # Partial reward based on letter matches
                    rewards[i] = self._calculate_partial_reward(guess, target)
                    self._update_state(i, guess, target)

        return self.states.clone(), rewards, self.dones.clone(), infos

    def _calculate_partial_reward(self, guess, target):
        """Calculate partial reward based on letter matches (green and yellow)."""
        reward = 0.0
        target_letter_count = {}
        for letter in target:
            target_letter_count[letter] = target_letter_count.get(letter, 0) + 1

        # First pass: exact position matches (green) - worth more
        for i, letter in enumerate(guess):
            if letter == target[i]:
                reward += 3.0
                target_letter_count[letter] -= 1

        # Second pass: wrong position matches (yellow)
        for i, letter in enumerate(guess):
            if letter != target[i] and target_letter_count.get(letter, 0) > 0:
                reward += 1.0
                target_letter_count[letter] -= 1

        return reward

    def _update_state(self, env_idx, guess, target):
        """Update state for a single environment based on the guess feedback."""
        for i, letter in enumerate(guess):
            letter_idx = ord(letter) - ord('a')
            if letter == target[i]:
                # Correct position (green)
                self.states[env_idx, letter_idx * 3 + 1] = 1.0
            elif letter in target:
                # Wrong position but in word (yellow)
                self.states[env_idx, letter_idx * 3 + 2] = 1.0
            else:
                # Not in word (gray)
                self.states[env_idx, letter_idx * 3] = 1.0


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
            torch.stack(self.states),      # (T, num_envs, state_size)
            torch.stack(self.actions),     # (T, num_envs)
            torch.stack(self.rewards),     # (T, num_envs)
            torch.stack(self.log_probs),   # (T, num_envs)
            torch.stack(self.values),      # (T, num_envs)
            torch.stack(self.dones),       # (T, num_envs)
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
        num_envs=256,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        update_epochs=4,
        mini_batch_size=512,
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

        # Learning rate scheduler (reduce on plateau)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.9, patience=20
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(num_envs, device)

    @torch.no_grad()
    def act(self, states):
        """
        Select actions for a batch of states.

        Args:
            states: Batch of states (num_envs, state_size)

        Returns:
            actions: Selected actions (num_envs,)
            log_probs: Log probabilities of selected actions (num_envs,)
            values: State value estimates (num_envs,)
        """
        action_logits, values = self.policy(states)

        # Sample from the action distribution
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        return actions, log_probs, values.squeeze(-1)

    def compute_gae(self, rewards, values, dones, next_values):
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: (T, num_envs)
            values: (T, num_envs)
            dones: (T, num_envs)
            next_values: (num_envs,) - value estimates for the last state

        Returns:
            advantages: (T, num_envs)
            returns: (T, num_envs)
        """
        T = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(self.num_envs, device=self.device)

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_values
            else:
                next_val = values[t + 1]

            # TD error: r + gamma * V(s') * (1 - done) - V(s)
            not_done = (~dones[t]).float()
            delta = rewards[t] + self.gamma * next_val * not_done - values[t]

            # GAE: delta + gamma * lambda * (1 - done) * gae_prev
            gae = delta + self.gamma * self.gae_lambda * not_done * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(self, next_values):
        """
        Perform PPO update using collected rollout data.

        Args:
            next_values: Value estimates for the final states (num_envs,)

        Returns:
            dict: Training metrics (losses, entropy)
        """
        if len(self.buffer) == 0:
            return None

        # Get all rollout data
        states, actions, rewards, old_log_probs, values, dones = self.buffer.get_tensors()

        # Compute advantages and returns using GAE
        advantages, returns = self.compute_gae(rewards, values, dones, next_values)

        # Flatten batch dimensions: (T, num_envs, ...) -> (T * num_envs, ...)
        T, num_envs = rewards.shape
        batch_size = T * num_envs

        states_flat = states.view(batch_size, self.state_size)
        actions_flat = actions.view(batch_size)
        old_log_probs_flat = old_log_probs.view(batch_size)
        advantages_flat = advantages.view(batch_size)
        returns_flat = returns.view(batch_size)

        # Normalize advantages
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

        # Training metrics
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        # Mini-batch training over multiple epochs
        indices = torch.randperm(batch_size, device=self.device)

        for _ in range(self.update_epochs):
            for start in range(0, batch_size, self.mini_batch_size):
                end = min(start + self.mini_batch_size, batch_size)
                batch_idx = indices[start:end]

                # Get mini-batch data
                mb_states = states_flat[batch_idx]
                mb_actions = actions_flat[batch_idx]
                mb_old_log_probs = old_log_probs_flat[batch_idx]
                mb_advantages = advantages_flat[batch_idx]
                mb_returns = returns_flat[batch_idx]

                # Forward pass
                action_logits, mb_values = self.policy(mb_states)
                mb_values = mb_values.squeeze(-1)

                # Compute new log probs and entropy
                action_probs = torch.softmax(action_logits, dim=-1)
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy()

                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (MSE)
                value_loss = nn.functional.mse_loss(mb_values, mb_returns)

                # Entropy bonus (for exploration)
                entropy_loss = -entropy.mean()

                # Combined loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        # Clear buffer after update
        self.buffer.clear()

        return {
            "total_loss": total_loss / num_updates,
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
        }

    def save_checkpoint(self, episode, win_rate, checkpoint_dir="checkpoints"):
        """Save model checkpoint."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "wordle_ppo_parallel.pt")

        checkpoint = {
            "episode": episode,
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "win_rate": win_rate,
        }

        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint.get("episode", 0), checkpoint.get("win_rate", 0)


def load_word_list():
    """Load curated list of common 5-letter English words."""
    from utils.word_list import load_word_list as load_curated_words
    return load_curated_words()


def train(
    num_envs=256,
    total_timesteps=10_000_000,
    steps_per_update=128,
    use_wandb=True,
    wandb_project="wordle-rl-agent",
    wandb_run_name=None,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    entropy_coef=0.01,
    value_coef=0.5,
    update_epochs=4,
    mini_batch_size=512,
    checkpoint_interval=100,
):
    """
    Train PPO agent with parallel environments.

    Args:
        num_envs: Number of parallel environments
        total_timesteps: Total number of environment steps
        steps_per_update: Steps to collect before each policy update
        use_wandb: Whether to log to Weights & Biases
        wandb_project: W&B project name
        wandb_run_name: W&B run name
        lr: Learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_epsilon: PPO clipping parameter
        entropy_coef: Entropy bonus coefficient
        value_coef: Value loss coefficient
        update_epochs: Number of epochs per update
        mini_batch_size: Mini-batch size for updates
        checkpoint_interval: Updates between checkpoints

    Returns:
        agent: Trained PPO agent
        best_checkpoint_path: Path to best checkpoint
    """
    print_cuda_info()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    print(f"Number of parallel environments: {num_envs}")

    # Load word list
    word_list = load_word_list()
    print(f"Action space size: {len(word_list)} words")

    # Configuration
    config = {
        "algorithm": "PPO-Parallel",
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
    }

    # Initialize W&B
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=config,
        )

    # Create environment and agent
    env = VectorizedWordleEnv(word_list, device, num_envs)
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

    # Calculate number of updates
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
                # Select actions
                actions, log_probs, values = agent.act(states)

                # Step environments
                next_states, rewards, dones, infos = env.step(actions)

                # Store transition
                agent.buffer.add(states, actions, rewards, log_probs, values, dones)

                # Track episode statistics
                episode_reward_sums += rewards
                episode_lengths += 1

                # Handle completed episodes
                done_indices = dones.nonzero(as_tuple=True)[0]
                if done_indices.numel() > 0:
                    for idx in done_indices.cpu().tolist():
                        total_episodes += 1
                        episode_rewards.append(episode_reward_sums[idx].item())
                        episode_guesses.append(episode_lengths[idx].item())

                        if infos[idx]["won"]:
                            total_wins += 1
                            episode_wins.append(1)
                        else:
                            episode_wins.append(0)

                        # Reset tracking
                        episode_reward_sums[idx] = 0
                        episode_lengths[idx] = 0

                    # Reset completed environments
                    env.reset(done_indices)
                    next_states = env.states.clone()

                states = next_states

            # Compute value estimates for final states
            with torch.no_grad():
                _, next_values = agent.policy(states)
                next_values = next_values.squeeze(-1)

            # Update policy
            loss_metrics = agent.update(next_values)

            # Calculate current metrics
            if len(episode_rewards) > 0:
                avg_reward = np.mean(episode_rewards)
                win_rate = np.mean(episode_wins) * 100
                avg_guesses = np.mean(episode_guesses)
            else:
                avg_reward = 0
                win_rate = 0
                avg_guesses = 0

            # Update progress bar
            pbar.set_description(
                f"Win: {win_rate:.1f}% | Reward: {avg_reward:.1f} | "
                f"Guesses: {avg_guesses:.2f} | Episodes: {total_episodes:,}"
            )

            # Log to W&B
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

                # VRAM safety check
                if vram_usage > MAX_VRAM_GB:
                    print(f"\nWARNING: VRAM usage ({vram_usage:.2f} GB) exceeds limit!")
                    break

            # Save checkpoint and update learning rate
            if (update + 1) % checkpoint_interval == 0:
                checkpoint_path = agent.save_checkpoint(update + 1, win_rate)
                agent.scheduler.step(win_rate)

                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_checkpoint_path = checkpoint_path
                    print(f"\n[Update {update + 1}] New best win rate: {win_rate:.1f}%")

                    if use_wandb:
                        wandb.save(checkpoint_path)

            # Clear CUDA cache periodically
            if torch.cuda.is_available() and (update + 1) % 50 == 0:
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    # Final statistics
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
        description="Train parallel PPO agent for Wordle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Environment settings
    parser.add_argument(
        "--num-envs", type=int, default=256,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=10_000_000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--steps-per-update", type=int, default=128,
        help="Steps per policy update"
    )

    # PPO hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--update-epochs", type=int, default=4, help="PPO update epochs")
    parser.add_argument("--mini-batch-size", type=int, default=512, help="Mini-batch size")

    # Logging settings
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="wordle-rl-agent", help="W&B project")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="Updates between checkpoints")

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
    )
