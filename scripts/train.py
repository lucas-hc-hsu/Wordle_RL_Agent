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
from datetime import datetime
import wandb
from model.dqn import DQN

# Print CUDA information
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"CUDA memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")


class WordleDataManager:
    def __init__(self, device):
        self.device = device
        self.word_embeddings = None
        self.precomputed_states = {}
        self.cached_actions = None

    def precompute_word_embeddings(self, word_list):
        """Precompute word embeddings and store them on GPU."""
        vocab_size = len(word_list)
        embedding_dim = 256  # Increased embedding dimension

        # Create embeddings on GPU
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim).to(self.device)

        # Create word indices tensor on GPU
        word_indices = torch.arange(vocab_size, device=self.device)

        # Compute embeddings for all words at once
        with torch.no_grad():
            self.cached_actions = self.word_embeddings(word_indices)

        print(f"Precomputed embeddings size: {self.cached_actions.shape}")
        return self.cached_actions

    def precompute_common_states(self, word_list, max_states=10000):
        """Precompute common game states and store them on GPU."""
        common_states = set()

        # Generate some common game states
        for target_word in random.sample(word_list, min(len(word_list), 1000)):
            state = torch.zeros(78, device=self.device)  # 26 * 3 state size
            common_states.add(tuple(state.tolist()))

            # Simulate some common guesses
            for guess in random.sample(word_list, 3):
                new_state = self._update_state(state, guess, target_word)
                common_states.add(tuple(new_state.tolist()))

        # Convert states to tensor and store on GPU
        for state in common_states:
            state_tensor = torch.tensor(state, device=self.device)
            self.precomputed_states[state] = state_tensor

        print(f"Precomputed {len(self.precomputed_states)} common states")

    def _update_state(self, state, guess, target):
        """Helper function to update game state."""
        new_state = state.clone()
        for i, letter in enumerate(guess):
            idx = ord(letter) - ord("a")
            if letter == target[i]:
                new_state[idx * 3 + 1] = 1
            elif letter in target:
                new_state[idx * 3 + 2] = 1
            else:
                new_state[idx * 3] = 1
        return new_state


class WordleEnv:
    def __init__(self, word_list, device, data_manager):
        self.word_list = word_list
        self.device = device
        self.data_manager = data_manager
        self.target_word = None
        self.max_tries = 6
        self.current_try = 0
        self.state_size = 26 * 3
        self.state_buffer = torch.zeros(self.state_size, pin_memory=True)
        self.reward_buffer = torch.zeros(1, pin_memory=True)

    def reset(self):
        self.target_word = random.choice(self.word_list)
        self.current_try = 0
        return torch.zeros(self.state_size, device=self.device)

    def step(self, action_idx):
        """
        Execute one step in the environment.

        Args:
            action_idx (int): Index of the word in word_list to use as guess

        Returns:
            tuple: (next_state, reward, done)
        """
        guess = self.word_list[action_idx].lower()
        reward = 0
        done = False

        if guess == self.target_word:
            reward = (
                100 - self.current_try * 10
            )  # Higher reward for earlier correct guesses
            done = True
        else:
            self.current_try += 1
            if self.current_try >= self.max_tries:
                reward = -50
                done = True
            else:
                reward = self._calculate_partial_reward(guess)

        next_state = self._update_state(guess)
        return next_state, reward, done

    def _calculate_partial_reward(self, guess):
        reward = 0
        for i, letter in enumerate(guess):
            if letter == self.target_word[i]:
                reward += 2
            elif letter in self.target_word:
                reward += 1
        return reward

    def _update_state(self, guess):
        state = torch.zeros(self.state_size, device=self.device)
        for i, letter in enumerate(guess):
            idx = ord(letter) - ord("a")
            if i < len(self.target_word) and letter == self.target_word[i]:
                state[idx * 3 + 1] = 1  # Confirmed position
            elif letter in self.target_word:
                state[idx * 3 + 2] = 1  # Confirmed existence
            else:
                state[idx * 3] = 1  # Confirmed non-existence
        return state


def calculate_optimal_batch_size():
    if not torch.cuda.is_available():
        return 32

    total_memory = torch.cuda.get_device_properties(0).total_memory
    memory_in_gb = total_memory / (1024**3)
    usable_memory = memory_in_gb * 0.9
    batch_size = int((usable_memory * 1024 * 1024) / 1024)
    batch_size = 2 ** int(np.log2(batch_size))

    print(f"Total GPU Memory: {memory_in_gb:.2f} GB")
    print(f"Usable Memory: {usable_memory:.2f} GB")
    print(f"Calculated Batch Size: {batch_size}")

    return batch_size


class WordleAgent:
    def __init__(self, state_size, action_size, word_list, device, data_manager, use_replay=False):
        self.state_size = state_size
        self.action_size = action_size
        self.word_list = word_list
        self.device = device
        self.data_manager = data_manager
        self.use_replay = use_replay  # Parameter to control whether to use replay

        if self.use_replay:
            # Only initialize memory-related buffers when using replay
            self.memory_size = 500000
            self.memory = deque(maxlen=self.memory_size)
            self.states_buffer = torch.zeros(self.memory_size, state_size, pin_memory=True)
            self.actions_buffer = torch.zeros(self.memory_size, dtype=torch.long, pin_memory=True)
            self.rewards_buffer = torch.zeros(self.memory_size, pin_memory=True)
            self.next_states_buffer = torch.zeros(self.memory_size, state_size, pin_memory=True)
            self.dones_buffer = torch.zeros(self.memory_size, pin_memory=True)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 1e-3

        self.model = DQN(state_size, action_size, device).to(device)
        self.target_model = DQN(state_size, action_size, device).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # # Calculate total training steps: episodes * max_steps_per_episode
        # total_steps = 10**5 * 6  # Assuming max 6 steps per episode

        # # OneCycleLR setup
        # self.scheduler = optim.lr_scheduler.OneCycleLR(
        #     self.optimizer,
        #     max_lr=self.learning_rate,  # Maximum learning rate
        #     total_steps=total_steps,    # Total training steps
        #     pct_start=0.3,             # Position to reach max learning rate (30%)
        #     div_factor=25.0,           # Initial learning rate = max_lr/25
        #     final_div_factor=10000.0,  # Final learning rate = max_lr/10000
        #     verbose=True
        # )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.95, patience=5
        )
        self.scaler = torch.cuda.amp.GradScaler()
        self.valid_word_indices = set(range(len(word_list)))

    def act(self, state):
        if random.random() <= self.epsilon:
            # Randomly select from valid word indices
            valid_indices = list(self.valid_word_indices)
            return random.choice(valid_indices)

        try:
            with torch.no_grad():
                # Ensure state is a tensor and on GPU
                if not isinstance(state, torch.Tensor):
                    state = torch.FloatTensor(state)
                state = state.to(self.device)

                # Ensure correct dimensions
                if len(state.shape) == 1:
                    state = state.unsqueeze(0)

                # Get model predictions
                act_values = self.model(state)

                # Create mask for valid actions
                valid_indices = torch.tensor(
                    list(self.valid_word_indices), device=self.device
                )
                valid_actions_mask = torch.zeros(self.action_size, device=self.device)
                valid_actions_mask[valid_indices] = 1

                # Expand mask to match act_values shape
                valid_actions_mask = valid_actions_mask.expand_as(act_values)

                # Apply mask
                masked_values = torch.where(
                    valid_actions_mask > 0,
                    act_values,
                    torch.tensor(float("-inf"), device=self.device),
                )

                # Ensure we only select from valid actions
                max_value, max_index = masked_values[0].max(dim=0)

                # Additional safety check
                if max_index >= self.action_size:
                    print(
                        f"Warning: Invalid action index {max_index}, falling back to random choice"
                    )
                    return random.choice(list(self.valid_word_indices))

                return max_index.item()

        except Exception as e:
            print(f"Error in act method: {e}")
            # Return random valid action when error occurs
            return random.choice(list(self.valid_word_indices))

    def remember(self, state, action, reward, next_state, done):
        if not self.use_replay:
            return  # Return immediately if not using replay
            
        try:
            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy()
            if isinstance(next_state, torch.Tensor):
                next_state = next_state.cpu().numpy()

            action = int(action)
            if action >= self.action_size:
                print(f"Warning: Invalid action {action} detected in remember")
                action = random.choice(list(self.valid_word_indices))

            self.memory.append((state, action, reward, next_state, done))
        except Exception as e:
            print(f"Error in remember method: {e}")

    def replay(self, batch_size):
        """Optimized replay method."""
        if len(self.memory) < batch_size:
            return

        try:
            indices = random.sample(range(len(self.memory)), batch_size)

            # Transfer batch data using pinned memory
            for i, idx in enumerate(indices):
                state, action, reward, next_state, done = self.memory[idx]
                self.states_buffer[i] = torch.FloatTensor(state)
                self.actions_buffer[i] = action
                self.rewards_buffer[i] = reward
                self.next_states_buffer[i] = torch.FloatTensor(next_state)
                self.dones_buffer[i] = done

            # Transfer batched data to GPU and ensure correct shapes
            states = self.states_buffer[:batch_size].to(self.device, non_blocking=True)
            actions = self.actions_buffer[:batch_size].to(
                self.device, non_blocking=True
            )
            rewards = self.rewards_buffer[:batch_size].to(
                self.device, non_blocking=True
            )
            next_states = self.next_states_buffer[:batch_size].to(
                self.device, non_blocking=True
            )
            dones = self.dones_buffer[:batch_size].to(self.device, non_blocking=True)

            # Ensure correct dimensions
            if len(states.shape) == 1:
                states = states.unsqueeze(0)
            if len(next_states.shape) == 1:
                next_states = next_states.unsqueeze(0)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                # Ensure actions are within valid range
                actions = torch.clamp(actions, 0, self.action_size - 1)

                # Calculate current Q values
                current_q_values = self.model(states)
                current_q_values = current_q_values.gather(
                    1, actions.unsqueeze(1).long()
                )

                # Use target network to calculate Q values for next states
                with torch.no_grad():
                    next_q_values = self.target_model(next_states)

                    # Create mask for valid actions
                    valid_indices = torch.tensor(
                        list(self.valid_word_indices), device=self.device
                    )
                    valid_actions_mask = torch.zeros_like(
                        next_q_values, device=self.device
                    )
                    valid_actions_mask[:, valid_indices] = 1

                    # Apply mask
                    masked_next_q_values = torch.where(
                        valid_actions_mask > 0,
                        next_q_values,
                        torch.tensor(float("-inf"), device=self.device),
                    )
                    next_q_values = masked_next_q_values.max(1)[0]

                # Ensure correct dimensions
                rewards = rewards.reshape(-1)
                dones = dones.reshape(-1)
                next_q_values = next_q_values.reshape(-1)

                # Calculate target Q values
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

                # Calculate loss
                loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

            # Perform optimization
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            return loss.item()

        except Exception as e:
            print(f"Error in replay method: {e}")
            return None

    def train_on_batch(self, state, action, reward, next_state, done):
        """Train directly on current experience without using replay."""
        try:
            # Ensure correct data format
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(self.device)
            if not isinstance(next_state, torch.Tensor):
                next_state = torch.FloatTensor(next_state).to(self.device)

            # Add batch dimension
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            if len(next_state.shape) == 1:
                next_state = next_state.unsqueeze(0)
            
            action = torch.tensor([action]).to(self.device)
            reward = torch.tensor([reward]).to(self.device)
            done = torch.tensor([float(done)]).to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                # Calculate current Q values
                current_q_values = self.model(state)
                current_q_values = current_q_values.gather(1, action.unsqueeze(1))

                # Calculate Q values for next state
                with torch.no_grad():
                    next_q_values = self.target_model(next_state)
                    valid_indices = torch.tensor(list(self.valid_word_indices), device=self.device)
                    valid_actions_mask = torch.zeros_like(next_q_values, device=self.device)
                    valid_actions_mask[:, valid_indices] = 1
                    masked_next_q_values = torch.where(
                        valid_actions_mask > 0,
                        next_q_values,
                        torch.tensor(float('-inf'), device=self.device)
                    )
                    next_q_values = masked_next_q_values.max(1)[0]

                # Calculate target Q values and loss
                target_q_values = reward + (1 - done) * self.gamma * next_q_values
                loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

            # Optimize
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            return loss.item()

        except Exception as e:
            print(f"Error in train_on_batch method: {e}")
            return None
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_checkpoint(self, episode_reward, checkpoint_dir="checkpoints"):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        checkpoint_path = os.path.join(checkpoint_dir, f"wordle_model.pt")

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "target_model_state_dict": self.target_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epsilon": self.epsilon,
            "word_list": self.word_list,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "device": self.device,
        }

        torch.save(checkpoint, checkpoint_path)
        self.scheduler.step(episode_reward)

        return checkpoint_path


def load_word_list():
    """Load curated list of common, meaningful 5-letter English words."""
    from utils.word_list import load_word_list as load_curated_words
    return load_curated_words()


def train_agent(
    use_replay=False,
    episodes=100000,
    use_wandb=True,
    wandb_project="wordle-rl-agent",
    wandb_run_name=None,
    lr=1e-3,
    gamma=0.95,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    log_interval=10,
    save_interval=100,
):
    """Train DQN agent to play Wordle."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_manager = WordleDataManager(device)
    word_list = load_word_list()

    # Hyperparameters config
    config = {
        "algorithm": "DQN",
        "episodes": episodes,
        "use_replay": use_replay,
        "lr": lr,
        "gamma": gamma,
        "epsilon": epsilon,
        "epsilon_min": epsilon_min,
        "epsilon_decay": epsilon_decay,
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

    print("Precomputing word embeddings and common states...")
    data_manager.precompute_word_embeddings(word_list)
    data_manager.precompute_common_states(word_list)

    env = WordleEnv(word_list, device, data_manager)
    state_size = env.state_size
    action_size = len(word_list)
    agent = WordleAgent(state_size, action_size, word_list, device, data_manager, use_replay)

    # Update agent hyperparameters
    agent.learning_rate = lr
    agent.gamma = gamma
    agent.epsilon = epsilon
    agent.epsilon_min = epsilon_min
    agent.epsilon_decay = epsilon_decay

    batch_size = calculate_optimal_batch_size() * 2 if use_replay else 1
    print(f"Using {'replay buffer with ' if use_replay else 'direct'} training, batch size: {batch_size}")

    pbar = tqdm(range(episodes), desc=f"Training Progress")
    best_reward = float("-inf")
    best_win_rate = 0
    best_checkpoint_path = None
    total_guesses = 0
    completed_episodes = 0
    total_wins = 0
    total_steps = 0

    # Rolling metrics
    episode_rewards = deque(maxlen=100)
    episode_wins = deque(maxlen=100)
    episode_guesses_list = deque(maxlen=100)

    try:
        for episode in pbar:
            state = env.reset()
            total_reward = 0
            losses = []
            episode_guesses = 0
            won = False

            for time in range(env.max_tries):
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                loss = None  # Initialize loss variable

                if use_replay:
                    agent.remember(state, action, reward, next_state, done)
                    if len(agent.memory) > batch_size:
                        loss = agent.replay(batch_size)
                else:
                    loss = agent.train_on_batch(state, action, reward, next_state, done)

                if loss is not None:  # Only add to list when loss is valid
                    losses.append(loss)

                state = next_state
                total_reward += reward
                episode_guesses += 1
                total_steps += 1

                if done:
                    if reward > 0:
                        total_wins += 1
                        won = True
                    break

            total_guesses += episode_guesses
            completed_episodes += 1

            # Track rolling metrics
            episode_rewards.append(total_reward)
            episode_wins.append(1 if won else 0)
            episode_guesses_list.append(episode_guesses)

            avg_guesses = np.mean(episode_guesses_list)
            win_rate = np.mean(episode_wins) * 100
            avg_reward = np.mean(episode_rewards)
            avg_loss = np.mean(losses) if losses else 0

            # Logging
            if (episode + 1) % log_interval == 0:
                pbar.set_description(
                    f"Avg Guesses: {avg_guesses:.2f} | "
                    f"Win Rate: {win_rate:.1f}% | "
                    f"Reward: {total_reward:.2f} | "
                    f"Epsilon: {agent.epsilon:.2f} | "
                    f"Loss: {avg_loss:.4f}"
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
                        "episode_guesses": episode_guesses,
                        "won": 1 if won else 0,
                        "epsilon": agent.epsilon,
                        "learning_rate": agent.optimizer.param_groups[0]["lr"],
                    }
                    if avg_loss > 0:
                        log_dict["loss/q_loss"] = avg_loss
                    wandb.log(log_dict)

            if (episode + 1) % save_interval == 0:
                agent.update_target_model()
                checkpoint_path = agent.save_checkpoint(total_reward)

                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_checkpoint_path = checkpoint_path
                    print(f"\nNew best win rate: {win_rate:.1f}%")

                    # Save best model to wandb
                    if use_wandb:
                        wandb.save(checkpoint_path)

                if total_reward > best_reward:
                    best_reward = total_reward

            if torch.cuda.is_available() and episode % 1000 == 0:
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # Final stats
    if completed_episodes > 0:
        final_win_rate = np.mean(episode_wins) * 100
        final_avg_reward = np.mean(episode_rewards)
        final_avg_guesses = np.mean(episode_guesses_list)

        print(f"\nTraining completed. Best model checkpoint: {best_checkpoint_path}")
        print(f"Final average guesses per episode: {final_avg_guesses:.2f}")
        print(f"Final win rate: {final_win_rate:.1f}%")

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
    parser = argparse.ArgumentParser(description="Train DQN agent for Wordle")
    parser.add_argument("--episodes", type=int, default=100000, help="Number of training episodes")
    parser.add_argument("--use-replay", action="store_true", help="Use experience replay buffer")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon for exploration")
    parser.add_argument("--epsilon-min", type=float, default=0.01, help="Minimum epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Epsilon decay rate")
    parser.add_argument("--log-interval", type=int, default=10, help="Episodes between logging")
    parser.add_argument("--save-interval", type=int, default=100, help="Episodes between checkpoint saves")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="wordle-rl-agent", help="Wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Wandb run name")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trained_agent, best_checkpoint_path = train_agent(
        use_replay=args.use_replay,
        episodes=args.episodes,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        lr=args.lr,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
    )
