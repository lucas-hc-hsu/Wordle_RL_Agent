"""
A3C (Asynchronous Advantage Actor-Critic) Training for Wordle Agent.

A3C uses multiple parallel worker processes, each with their own environment,
that asynchronously update a shared global model. This provides:
- True parallelism via multiprocessing (not limited by Python GIL)
- Diverse exploration from multiple independent workers
- Stable training through asynchronous updates

VRAM Usage: Minimal per worker (~0.05 GB each), scales linearly with num_workers.
Recommended: 4-16 workers depending on CPU cores.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
from collections import deque
import random
import os
import argparse
import time
from datetime import datetime

# Set start method for multiprocessing
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# VRAM limit configuration (50% of 32GB = 15GB)
MAX_VRAM_GB = 15.0


class SharedAdam(optim.Adam):
    """
    Adam optimizer with shared state for A3C.
    The optimizer state is shared across all worker processes.
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # Share optimizer state across processes
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                # Share memory
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class A3CNetwork(nn.Module):
    """
    Actor-Critic network for A3C.
    Shared feature layers with separate actor (policy) and critic (value) heads.
    """

    def __init__(self, state_size, action_size):
        super(A3CNetwork, self).__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        features = self.shared(x)
        policy_logits = self.actor(features)
        value = self.critic(features)
        return policy_logits, value

    def get_action_and_value(self, state, valid_mask=None):
        """Get action, log probability, entropy, and value for a state."""
        policy_logits, value = self.forward(state)

        # Apply valid action mask if provided
        if valid_mask is not None:
            policy_logits = policy_logits.masked_fill(valid_mask == 0, float('-inf'))

        # Sample action from policy
        policy = F.softmax(policy_logits, dim=-1)
        log_policy = F.log_softmax(policy_logits, dim=-1)

        # Sample action
        action = policy.multinomial(num_samples=1).squeeze(-1)
        log_prob = log_policy.gather(1, action.unsqueeze(-1)).squeeze(-1)

        # Compute entropy
        entropy = -(policy * log_policy).sum(dim=-1)

        return action, log_prob, entropy, value.squeeze(-1)


class WordleEnv:
    """
    Single Wordle environment for A3C workers.
    Each worker has its own independent environment instance.
    """

    def __init__(self, word_list):
        self.word_list = word_list
        self.max_tries = 6
        self.state_size = 26 * 3  # 26 letters, 3 states each

        self.target_word = None
        self.current_try = 0
        self.state = None
        self.guessed_words = set()

    def reset(self):
        """Reset the environment for a new episode."""
        self.target_word = random.choice(self.word_list)
        self.current_try = 0
        self.state = np.zeros(self.state_size, dtype=np.float32)
        self.guessed_words = set()
        return self.state.copy()

    def step(self, action_idx):
        """Take a step in the environment."""
        guess = self.word_list[action_idx]
        target = self.target_word

        # Check for repeated guess
        if guess in self.guessed_words:
            self.current_try += 1
            if self.current_try >= self.max_tries:
                return self.state.copy(), -50, True, {"won": False}
            return self.state.copy(), -5, False, {"won": False}

        self.guessed_words.add(guess)

        # Check for win
        if guess == target:
            reward = 100 - self.current_try * 10
            return self.state.copy(), reward, True, {"won": True}

        # Wrong guess
        self.current_try += 1
        if self.current_try >= self.max_tries:
            return self.state.copy(), -50, True, {"won": False}

        # Calculate partial reward and update state
        reward = self._calculate_partial_reward(guess, target)
        self._update_state(guess, target)

        return self.state.copy(), reward, False, {"won": False}

    def _calculate_partial_reward(self, guess, target):
        """Calculate partial reward based on letter matches."""
        reward = 0
        target_letter_count = {}
        for letter in target:
            target_letter_count[letter] = target_letter_count.get(letter, 0) + 1

        # First pass: exact matches (green)
        for i, letter in enumerate(guess):
            if letter == target[i]:
                reward += 3
                target_letter_count[letter] -= 1

        # Second pass: wrong position matches (yellow)
        for i, letter in enumerate(guess):
            if letter != target[i] and target_letter_count.get(letter, 0) > 0:
                reward += 1
                target_letter_count[letter] -= 1

        return reward

    def _update_state(self, guess, target):
        """Update state based on guess feedback."""
        for i, letter in enumerate(guess):
            idx = ord(letter) - ord('a')
            if letter == target[i]:
                self.state[idx * 3 + 1] = 1  # Correct position
            elif letter in target:
                self.state[idx * 3 + 2] = 1  # Wrong position
            else:
                self.state[idx * 3] = 1  # Not in word


def worker(
    worker_id,
    global_model,
    optimizer,
    word_list,
    counter,
    lock,
    results_queue,
    max_episodes,
    gamma=0.99,
    entropy_coef=0.01,
    value_coef=0.5,
    max_grad_norm=40,
    t_max=20,
    device='cpu',
):
    """
    A3C worker process.
    Each worker runs its own environment and updates the global model asynchronously.
    """
    # Create local model (copy of global model)
    local_model = A3CNetwork(26 * 3, len(word_list)).to(device)
    local_model.load_state_dict(global_model.state_dict())

    # Create local environment
    env = WordleEnv(word_list)

    # Training statistics
    episode_rewards = []
    episode_wins = []
    episode_guesses = []

    while True:
        # Check if we've reached max episodes
        with lock:
            current_episode = counter.value
            if current_episode >= max_episodes:
                break
            counter.value += 1
            episode_num = current_episode

        # Sync local model with global model
        local_model.load_state_dict(global_model.state_dict())

        # Reset environment
        state = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        # Storage for computing gradients
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        entropies = []

        while not done:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            # Get action from local model
            with torch.no_grad():
                action, log_prob, entropy, value = local_model.get_action_and_value(state_tensor)

            action_idx = action.item()

            # Take step in environment
            next_state, reward, done, info = env.step(action_idx)

            # Store transition
            states.append(state_tensor)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropies.append(entropy)

            episode_reward += reward
            episode_length += 1
            state = next_state

            # Update global model every t_max steps or at episode end
            if len(states) >= t_max or done:
                # Compute returns and advantages
                if done:
                    R = 0
                else:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    with torch.no_grad():
                        _, next_value = local_model(state_tensor)
                    R = next_value.item()

                # Compute gradients
                policy_loss = 0
                value_loss = 0
                entropy_loss = 0

                returns = []
                for r in reversed(rewards):
                    R = r + gamma * R
                    returns.insert(0, R)

                returns = torch.FloatTensor(returns).to(device)
                values_tensor = torch.stack(values).squeeze()
                log_probs_tensor = torch.stack(log_probs).squeeze()
                entropies_tensor = torch.stack(entropies).squeeze()

                # Normalize returns
                if len(returns) > 1:
                    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

                # Compute advantages
                advantages = returns - values_tensor.detach()

                # Policy loss (actor)
                policy_loss = -(log_probs_tensor * advantages.detach()).mean()

                # Value loss (critic)
                value_loss = F.mse_loss(values_tensor, returns)

                # Entropy loss (for exploration)
                entropy_loss = -entropies_tensor.mean()

                # Total loss
                total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

                # Compute gradients
                optimizer.zero_grad()
                total_loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_grad_norm)

                # Copy gradients to global model and update
                for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
                    if global_param.grad is None:
                        global_param.grad = local_param.grad.clone()
                    else:
                        global_param.grad = local_param.grad.clone()

                optimizer.step()

                # Sync local model with updated global model
                local_model.load_state_dict(global_model.state_dict())

                # Clear storage
                states = []
                actions = []
                rewards = []
                values = []
                log_probs = []
                entropies = []

        # Record episode statistics
        episode_rewards.append(episode_reward)
        episode_wins.append(1 if info["won"] else 0)
        episode_guesses.append(episode_length)

        # Report results periodically
        if len(episode_rewards) >= 100:
            avg_reward = np.mean(episode_rewards[-100:])
            win_rate = np.mean(episode_wins[-100:]) * 100
            avg_guesses = np.mean(episode_guesses[-100:])
            results_queue.put({
                "worker_id": worker_id,
                "episode": episode_num,
                "avg_reward": avg_reward,
                "win_rate": win_rate,
                "avg_guesses": avg_guesses,
                "total_episodes": len(episode_rewards),
            })

    # Final report
    if len(episode_rewards) > 0:
        avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        win_rate = np.mean(episode_wins[-100:]) * 100 if len(episode_wins) >= 100 else np.mean(episode_wins) * 100
        avg_guesses = np.mean(episode_guesses[-100:]) if len(episode_guesses) >= 100 else np.mean(episode_guesses)
        results_queue.put({
            "worker_id": worker_id,
            "episode": episode_num,
            "avg_reward": avg_reward,
            "win_rate": win_rate,
            "avg_guesses": avg_guesses,
            "total_episodes": len(episode_rewards),
            "final": True,
        })


def load_word_list():
    """Load curated list of common, meaningful 5-letter English words."""
    from utils.word_list import load_word_list as load_curated_words
    return load_curated_words()


def train_a3c(
    num_workers=4,
    max_episodes=100000,
    lr=1e-4,
    gamma=0.99,
    entropy_coef=0.01,
    value_coef=0.5,
    max_grad_norm=40,
    t_max=20,
    save_interval=1000,
    log_interval=100,
    use_wandb=True,
    wandb_project="wordle-rl-agent",
    wandb_run_name=None,
    use_gpu=False,
):
    """
    Train Wordle agent using A3C algorithm.

    Args:
        num_workers: Number of parallel worker processes
        max_episodes: Total number of episodes to train
        lr: Learning rate
        gamma: Discount factor
        entropy_coef: Entropy regularization coefficient
        value_coef: Value loss coefficient
        max_grad_norm: Maximum gradient norm for clipping
        t_max: Maximum steps before updating global model
        save_interval: Episodes between checkpoint saves
        log_interval: Episodes between logging
        use_wandb: Whether to use Weights & Biases logging
        wandb_project: Wandb project name
        wandb_run_name: Wandb run name
        use_gpu: Whether to use GPU for training
    """
    # Determine device
    if use_gpu and torch.cuda.is_available():
        device = 'cuda'
        print(f"GPU enabled: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    else:
        device = 'cpu'
        if use_gpu and not torch.cuda.is_available():
            print("WARNING: GPU requested but CUDA not available, falling back to CPU")
    print(f"A3C Training Configuration:")
    print(f"  Number of workers: {num_workers}")
    print(f"  Max episodes: {max_episodes}")
    print(f"  Learning rate: {lr}")
    print(f"  Gamma: {gamma}")
    print(f"  Entropy coefficient: {entropy_coef}")
    print(f"  Value coefficient: {value_coef}")
    print(f"  t_max (steps before update): {t_max}")
    print(f"  Device: {device}")

    # Load word list
    word_list = load_word_list()
    print(f"Loaded {len(word_list)} words")

    # Config for wandb
    config = {
        "algorithm": "A3C",
        "num_workers": num_workers,
        "max_episodes": max_episodes,
        "lr": lr,
        "gamma": gamma,
        "entropy_coef": entropy_coef,
        "value_coef": value_coef,
        "max_grad_norm": max_grad_norm,
        "t_max": t_max,
        "state_size": 26 * 3,
        "action_size": len(word_list),
        "device": device,
    }

    # Initialize wandb
    if use_wandb:
        import wandb
        wandb.init(
            project=wandb_project,
            name=wandb_run_name or f"a3c-{num_workers}workers",
            config=config,
        )

    # Create global model and optimizer
    state_size = 26 * 3
    action_size = len(word_list)

    global_model = A3CNetwork(state_size, action_size).to(device)
    global_model.share_memory()  # Share model across processes

    optimizer = SharedAdam(global_model.parameters(), lr=lr)

    # Shared counter and lock for episode counting
    counter = mp.Value('i', 0)
    lock = mp.Lock()

    # Queue for results from workers
    results_queue = mp.Queue()

    # Start worker processes
    workers = []
    for worker_id in range(num_workers):
        p = mp.Process(
            target=worker,
            args=(
                worker_id,
                global_model,
                optimizer,
                word_list,
                counter,
                lock,
                results_queue,
                max_episodes,
                gamma,
                entropy_coef,
                value_coef,
                max_grad_norm,
                t_max,
                device,
            ),
        )
        p.start()
        workers.append(p)
        print(f"Started worker {worker_id}")

    # Collect results and log
    best_win_rate = 0
    best_checkpoint_path = None
    last_log_episode = 0
    last_save_episode = 0

    all_results = []
    start_time = time.time()

    try:
        while any(p.is_alive() for p in workers):
            try:
                result = results_queue.get(timeout=1)
                all_results.append(result)

                current_episode = counter.value

                # Log periodically
                if current_episode - last_log_episode >= log_interval:
                    last_log_episode = current_episode
                    elapsed = time.time() - start_time
                    eps_per_sec = current_episode / elapsed if elapsed > 0 else 0

                    # Get latest stats from results
                    recent_results = [r for r in all_results[-num_workers*10:] if "final" not in r]
                    if recent_results:
                        avg_reward = np.mean([r["avg_reward"] for r in recent_results])
                        win_rate = np.mean([r["win_rate"] for r in recent_results])
                        avg_guesses = np.mean([r["avg_guesses"] for r in recent_results])

                        print(f"Episode {current_episode}/{max_episodes} | "
                              f"Win: {win_rate:.1f}% | "
                              f"Reward: {avg_reward:.1f} | "
                              f"Guesses: {avg_guesses:.2f} | "
                              f"Speed: {eps_per_sec:.1f} eps/s")

                        if use_wandb:
                            wandb.log({
                                "episode": current_episode,
                                "avg_reward": avg_reward,
                                "win_rate": win_rate,
                                "avg_guesses": avg_guesses,
                                "episodes_per_second": eps_per_sec,
                            })

                        if win_rate > best_win_rate:
                            best_win_rate = win_rate

                # Save checkpoint periodically
                if current_episode - last_save_episode >= save_interval:
                    last_save_episode = current_episode
                    checkpoint_dir = "checkpoints"
                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)

                    checkpoint_path = os.path.join(checkpoint_dir, "wordle_a3c_model.pt")
                    torch.save({
                        "episode": current_episode,
                        "model_state_dict": global_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_win_rate": best_win_rate,
                    }, checkpoint_path)
                    best_checkpoint_path = checkpoint_path
                    print(f"Saved checkpoint to {checkpoint_path}")

            except Exception:
                # Queue empty, continue
                pass

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # Wait for all workers to finish
    for p in workers:
        p.join()

    # Final statistics
    total_time = time.time() - start_time
    final_episode = counter.value

    print(f"\nTraining completed!")
    print(f"Total episodes: {final_episode}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average speed: {final_episode / total_time:.1f} episodes/second")
    print(f"Best win rate: {best_win_rate:.1f}%")

    # Save final model
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    final_path = os.path.join(checkpoint_dir, "wordle_a3c_model_final.pt")
    torch.save({
        "episode": final_episode,
        "model_state_dict": global_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_win_rate": best_win_rate,
    }, final_path)
    print(f"Saved final model to {final_path}")

    if use_wandb:
        wandb.summary["final_episode"] = final_episode
        wandb.summary["best_win_rate"] = best_win_rate
        wandb.summary["total_time"] = total_time
        wandb.finish()

    return global_model, best_checkpoint_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train A3C agent for Wordle")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel worker processes (default: 4)")
    parser.add_argument("--max-episodes", type=int, default=100000, help="Maximum training episodes (default: 100000)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (default: 0.99)")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy coefficient (default: 0.01)")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient (default: 0.5)")
    parser.add_argument("--max-grad-norm", type=float, default=40, help="Maximum gradient norm (default: 40)")
    parser.add_argument("--t-max", type=int, default=20, help="Steps before updating global model (default: 20)")
    parser.add_argument("--save-interval", type=int, default=1000, help="Episodes between checkpoint saves (default: 1000)")
    parser.add_argument("--log-interval", type=int, default=100, help="Episodes between logging (default: 100)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="wordle-rl-agent", help="Wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training (default: CPU)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trained_model, best_checkpoint = train_a3c(
        num_workers=args.num_workers,
        max_episodes=args.max_episodes,
        lr=args.lr,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        t_max=args.t_max,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        use_gpu=args.gpu,
    )
