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

# Print CUDA information
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"CUDA memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")


class WordleEnv:
    """Wordle environment for reinforcement learning."""

    def __init__(self, word_list, device):
        self.word_list = word_list
        self.device = device
        self.target_word = None
        self.max_tries = 6
        self.current_try = 0
        self.state_size = 26 * 3  # 26 letters * 3 states (not in word, correct position, wrong position)
        self.guessed_words = set()

    def reset(self):
        """Reset the environment for a new episode."""
        self.target_word = random.choice(self.word_list)
        self.current_try = 0
        self.guessed_words = set()
        return torch.zeros(self.state_size, device=self.device)

    def step(self, action_idx):
        """
        Execute one step in the environment.

        Args:
            action_idx: Index of the word in word_list to use as guess

        Returns:
            next_state: Updated state after the guess
            reward: Reward for this action
            done: Whether the episode is finished
            info: Additional information
        """
        guess = self.word_list[action_idx].lower()
        reward = 0
        done = False
        info = {"won": False, "guess": guess, "target": self.target_word}

        # Penalty for repeated guesses
        if guess in self.guessed_words:
            reward = -5
            self.current_try += 1
            if self.current_try >= self.max_tries:
                reward = -50
                done = True
            next_state = self._get_current_state()
            return next_state, reward, done, info

        self.guessed_words.add(guess)

        if guess == self.target_word:
            # Win bonus based on number of tries
            reward = 100 - self.current_try * 10
            done = True
            info["won"] = True
        else:
            self.current_try += 1
            if self.current_try >= self.max_tries:
                reward = -50
                done = True
            else:
                reward = self._calculate_partial_reward(guess)

        next_state = self._update_state(guess)
        return next_state, reward, done, info

    def _calculate_partial_reward(self, guess):
        """Calculate reward based on letter matches."""
        reward = 0
        target_letter_count = {}
        for letter in self.target_word:
            target_letter_count[letter] = target_letter_count.get(letter, 0) + 1

        # First pass: exact matches (green)
        for i, letter in enumerate(guess):
            if letter == self.target_word[i]:
                reward += 3
                target_letter_count[letter] -= 1

        # Second pass: wrong position matches (yellow)
        for i, letter in enumerate(guess):
            if letter != self.target_word[i] and target_letter_count.get(letter, 0) > 0:
                reward += 1
                target_letter_count[letter] -= 1

        return reward

    def _update_state(self, guess):
        """Update state based on the guess."""
        state = self._get_current_state()

        for i, letter in enumerate(guess):
            idx = ord(letter) - ord("a")
            if letter == self.target_word[i]:
                state[idx * 3 + 1] = 1  # Correct position
            elif letter in self.target_word:
                state[idx * 3 + 2] = 1  # Wrong position but in word
            else:
                state[idx * 3] = 1  # Not in word

        return state

    def _get_current_state(self):
        """Get the current state tensor."""
        return torch.zeros(self.state_size, device=self.device)


class RolloutBuffer:
    """Buffer to store rollout experiences for PPO training."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def add(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def __len__(self):
        return len(self.states)


class PPOAgent:
    """PPO Agent for Wordle."""

    def __init__(
        self,
        state_size,
        action_size,
        word_list,
        device,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        update_epochs=4,
        batch_size=64,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.word_list = word_list
        self.device = device

        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size

        # Networks
        self.policy = ActorCritic(state_size, action_size, device).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.9, patience=10
        )

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Valid word indices
        self.valid_word_indices = set(range(len(word_list)))

    def get_valid_actions_mask(self):
        """Create a mask for valid actions."""
        mask = torch.zeros(self.action_size, device=self.device)
        for idx in self.valid_word_indices:
            mask[idx] = 1
        return mask

    def act(self, state):
        """Select an action using the current policy."""
        with torch.no_grad():
            mask = self.get_valid_actions_mask()
            action, log_prob, value = self.policy.act(state, mask)
        return action, log_prob, value

    def store_transition(self, state, action, reward, log_prob, value, done):
        """Store a transition in the buffer."""
        self.buffer.add(state, action, reward, log_prob, value, done)

    def compute_gae(self, next_value):
        """Compute Generalized Advantage Estimation."""
        rewards = self.buffer.rewards
        values = self.buffer.values
        dones = self.buffer.dones

        advantages = []
        gae = 0

        # Convert values to tensor if needed
        values_tensor = torch.stack(values) if isinstance(values[0], torch.Tensor) else torch.tensor(values, device=self.device)

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values_tensor[t + 1]

            current_val = values_tensor[t]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - current_val
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = advantages + values_tensor

        return advantages, returns

    def update(self, next_value):
        """Perform PPO update."""
        if len(self.buffer) == 0:
            return None

        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Prepare batch data
        states = torch.stack([s.clone() for s in self.buffer.states])
        actions = torch.tensor(self.buffer.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.stack(self.buffer.log_probs)
        mask = self.get_valid_actions_mask().unsqueeze(0).expand(len(states), -1)

        # PPO update epochs
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for _ in range(self.update_epochs):
            # Get current policy outputs
            log_probs, values, entropy = self.policy.evaluate(states, actions, mask)

            # Compute ratio
            ratio = torch.exp(log_probs - old_log_probs.detach())

            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.MSELoss()(values, returns)

            # Entropy bonus
            entropy_loss = -entropy.mean()

            # Total loss
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()

        # Clear buffer after update
        self.buffer.clear()

        # Return detailed metrics
        return {
            "total_loss": total_loss / self.update_epochs,
            "policy_loss": total_policy_loss / self.update_epochs,
            "value_loss": total_value_loss / self.update_epochs,
            "entropy": total_entropy / self.update_epochs,
        }

    def save_checkpoint(self, episode, reward, checkpoint_dir="checkpoints"):
        """Save model checkpoint."""
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        checkpoint_path = os.path.join(checkpoint_dir, "wordle_ppo_model.pt")

        checkpoint = {
            "episode": episode,
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "word_list": self.word_list,
            "state_size": self.state_size,
            "action_size": self.action_size,
        }

        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint.get("episode", 0)


def load_word_list():
    """Load curated list of common, meaningful 5-letter English words."""
    from utils.word_list import load_word_list as load_curated_words
    return load_curated_words()


def train_ppo_agent(
    episodes=100000,
    update_interval=20,
    save_interval=100,
    log_interval=10,
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
):
    """Train PPO agent to play Wordle."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load word list
    word_list = load_word_list()

    # Hyperparameters config
    config = {
        "algorithm": "PPO",
        "episodes": episodes,
        "update_interval": update_interval,
        "lr": lr,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_epsilon": clip_epsilon,
        "entropy_coef": entropy_coef,
        "value_coef": value_coef,
        "update_epochs": update_epochs,
        "state_size": 26 * 3,
        "action_size": len(word_list),
        "device": str(device),
    }

    # Initialize wandb
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=config,
        )

    # Create environment and agent
    env = WordleEnv(word_list, device)
    agent = PPOAgent(
        state_size=env.state_size,
        action_size=len(word_list),
        word_list=word_list,
        device=device,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        update_epochs=update_epochs,
        batch_size=64,
    )

    # Training metrics
    episode_rewards = deque(maxlen=100)
    episode_wins = deque(maxlen=100)
    episode_guesses = deque(maxlen=100)
    best_win_rate = 0
    best_checkpoint_path = None
    total_steps = 0

    pbar = tqdm(range(episodes), desc="Training PPO")

    try:
        for episode in pbar:
            state = env.reset()
            total_reward = 0
            done = False
            num_guesses = 0
            episode_loss = None

            while not done:
                # Select action
                action, log_prob, value = agent.act(state)

                # Take action
                next_state, reward, done, info = env.step(action)

                # Store transition
                agent.store_transition(state, action, reward, log_prob, value, done)

                state = next_state
                total_reward += reward
                num_guesses += 1
                total_steps += 1

            # Track metrics
            episode_rewards.append(total_reward)
            episode_wins.append(1 if info["won"] else 0)
            episode_guesses.append(num_guesses)

            # Update policy
            loss_metrics = None
            if (episode + 1) % update_interval == 0:
                # Get next value estimate for GAE
                with torch.no_grad():
                    _, next_value = agent.policy(state)
                    next_value = next_value.squeeze()

                loss_metrics = agent.update(next_value if not done else torch.tensor(0.0, device=device))

            # Calculate metrics
            avg_reward = np.mean(episode_rewards)
            win_rate = np.mean(episode_wins) * 100
            avg_guesses = np.mean(episode_guesses)

            # Logging
            if (episode + 1) % log_interval == 0:
                pbar.set_description(
                    f"Win: {win_rate:.1f}% | "
                    f"Avg Reward: {avg_reward:.1f} | "
                    f"Avg Guesses: {avg_guesses:.2f}"
                )

                # Log to wandb
                if use_wandb:
                    log_dict = {
                        "episode": episode + 1,
                        "total_steps": total_steps,
                        "episode_reward": total_reward,
                        "avg_reward": avg_reward,
                        "win_rate": win_rate,
                        "avg_guesses": avg_guesses,
                        "episode_guesses": num_guesses,
                        "won": 1 if info["won"] else 0,
                        "learning_rate": agent.optimizer.param_groups[0]["lr"],
                    }
                    if loss_metrics is not None:
                        log_dict["loss/total"] = loss_metrics["total_loss"]
                        log_dict["loss/policy"] = loss_metrics["policy_loss"]
                        log_dict["loss/value"] = loss_metrics["value_loss"]
                        log_dict["metrics/entropy"] = loss_metrics["entropy"]
                    wandb.log(log_dict)

            # Save checkpoint
            if (episode + 1) % save_interval == 0:
                checkpoint_path = agent.save_checkpoint(episode, total_reward)
                agent.scheduler.step(win_rate)

                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_checkpoint_path = checkpoint_path
                    print(f"\nNew best win rate: {win_rate:.1f}%")

                    # Save best model to wandb
                    if use_wandb:
                        wandb.save(checkpoint_path)

            # Clear CUDA cache periodically
            if torch.cuda.is_available() and (episode + 1) % 1000 == 0:
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # Final stats
    if len(episode_rewards) > 0:
        final_win_rate = np.mean(episode_wins) * 100
        final_avg_reward = np.mean(episode_rewards)
        final_avg_guesses = np.mean(episode_guesses)

        print(f"\nTraining completed!")
        print(f"Final win rate: {final_win_rate:.1f}%")
        print(f"Final avg reward: {final_avg_reward:.1f}")
        print(f"Final avg guesses: {final_avg_guesses:.2f}")
        print(f"Best checkpoint: {best_checkpoint_path}")

        # Log final metrics to wandb
        if use_wandb:
            wandb.summary["final_win_rate"] = final_win_rate
            wandb.summary["final_avg_reward"] = final_avg_reward
            wandb.summary["final_avg_guesses"] = final_avg_guesses
            wandb.summary["best_win_rate"] = best_win_rate
            wandb.finish()

    return agent, best_checkpoint_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train PPO agent for Wordle")
    parser.add_argument("--episodes", type=int, default=100000, help="Number of training episodes")
    parser.add_argument("--update-interval", type=int, default=20, help="Episodes between policy updates")
    parser.add_argument("--save-interval", type=int, default=100, help="Episodes between checkpoint saves")
    parser.add_argument("--log-interval", type=int, default=10, help="Episodes between logging")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--update-epochs", type=int, default=4, help="PPO update epochs")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="wordle-rl-agent", help="Wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Wandb run name")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trained_agent, best_checkpoint_path = train_ppo_agent(
        episodes=args.episodes,
        update_interval=args.update_interval,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
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
    )
