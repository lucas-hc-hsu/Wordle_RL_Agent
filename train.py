import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from tqdm import tqdm
import os
from datetime import datetime
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
        self.use_replay = use_replay  # 新增控制是否使用 replay 的參數

        if self.use_replay:
            # 只在使用 replay 時初始化記憶相關的緩衝區
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

        # # 計算總訓練步數：episodes * max_steps_per_episode
        # total_steps = 10**5 * 6  # 假設每個 episode 最多 6 步

        # # OneCycleLR 設置
        # self.scheduler = optim.lr_scheduler.OneCycleLR(
        #     self.optimizer,
        #     max_lr=self.learning_rate,  # 最大學習率
        #     total_steps=total_steps,    # 總訓練步數
        #     pct_start=0.3,             # 達到最大學習率的位置（30%）
        #     div_factor=25.0,           # 初始學習率 = max_lr/25
        #     final_div_factor=10000.0,  # 最終學習率 = max_lr/10000
        #     verbose=True
        # )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.95, patience=5, verbose=True
        )
        self.scaler = torch.cuda.amp.GradScaler()
        self.valid_word_indices = set(range(len(word_list)))

    def act(self, state):
        if random.random() <= self.epsilon:
            # 從有效的單詞索引中隨機選擇
            valid_indices = list(self.valid_word_indices)
            return random.choice(valid_indices)

        try:
            with torch.no_grad():
                # 確保 state 是張量且在 GPU 上
                if not isinstance(state, torch.Tensor):
                    state = torch.FloatTensor(state)
                state = state.to(self.device)

                # 確保維度正確
                if len(state.shape) == 1:
                    state = state.unsqueeze(0)

                # 獲取模型預測
                act_values = self.model(state)

                # 創建有效動作的遮罩
                valid_indices = torch.tensor(
                    list(self.valid_word_indices), device=self.device
                )
                valid_actions_mask = torch.zeros(self.action_size, device=self.device)
                valid_actions_mask[valid_indices] = 1

                # 展開遮罩以匹配 act_values 的形狀
                valid_actions_mask = valid_actions_mask.expand_as(act_values)

                # 應用遮罩
                masked_values = torch.where(
                    valid_actions_mask > 0,
                    act_values,
                    torch.tensor(float("-inf"), device=self.device),
                )

                # 確保我們只從有效的動作中選擇
                max_value, max_index = masked_values[0].max(dim=0)

                # 額外的安全檢查
                if max_index >= self.action_size:
                    print(
                        f"Warning: Invalid action index {max_index}, falling back to random choice"
                    )
                    return random.choice(list(self.valid_word_indices))

                return max_index.item()

        except Exception as e:
            print(f"Error in act method: {e}")
            # 發生錯誤時返回隨機有效動作
            return random.choice(list(self.valid_word_indices))

    def remember(self, state, action, reward, next_state, done):
        if not self.use_replay:
            return  # 如果不使用 replay，直接返回
            
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
        """經過優化的回放方法"""
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

            # 確保維度正確
            if len(states.shape) == 1:
                states = states.unsqueeze(0)
            if len(next_states.shape) == 1:
                next_states = next_states.unsqueeze(0)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                # 確保 actions 在有效範圍內
                actions = torch.clamp(actions, 0, self.action_size - 1)

                # 計算當前 Q 值
                current_q_values = self.model(states)
                current_q_values = current_q_values.gather(
                    1, actions.unsqueeze(1).long()
                )

                # 使用目標網絡計算下一個狀態的 Q 值
                with torch.no_grad():
                    next_q_values = self.target_model(next_states)

                    # 創建有效動作的遮罩
                    valid_indices = torch.tensor(
                        list(self.valid_word_indices), device=self.device
                    )
                    valid_actions_mask = torch.zeros_like(
                        next_q_values, device=self.device
                    )
                    valid_actions_mask[:, valid_indices] = 1

                    # 應用遮罩
                    masked_next_q_values = torch.where(
                        valid_actions_mask > 0,
                        next_q_values,
                        torch.tensor(float("-inf"), device=self.device),
                    )
                    next_q_values = masked_next_q_values.max(1)[0]

                # 確保維度正確
                rewards = rewards.reshape(-1)
                dones = dones.reshape(-1)
                next_q_values = next_q_values.reshape(-1)

                # 計算目標 Q 值
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

                # 計算損失
                loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

            # 進行優化
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
        """直接在當前經驗上訓練，不使用 replay"""
        try:
            # 確保數據格式正確
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(self.device)
            if not isinstance(next_state, torch.Tensor):
                next_state = torch.FloatTensor(next_state).to(self.device)
            
            # 添加批次維度
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            if len(next_state.shape) == 1:
                next_state = next_state.unsqueeze(0)
            
            action = torch.tensor([action]).to(self.device)
            reward = torch.tensor([reward]).to(self.device)
            done = torch.tensor([float(done)]).to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                # 計算當前 Q 值
                current_q_values = self.model(state)
                current_q_values = current_q_values.gather(1, action.unsqueeze(1))

                # 計算下一個狀態的 Q 值
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

                # 計算目標 Q 值和損失
                target_q_values = reward + (1 - done) * self.gamma * next_q_values
                loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

            # 優化
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
    """Load and filter 5-letter words from NLTK's word corpus."""
    try:
        import nltk
        from nltk.corpus import words

        # Download required NLTK data
        nltk.download("words", quiet=True)

        # Get all words from NLTK words corpus and convert to lowercase
        word_list = set(
            word.lower() for word in words.words() if len(word) == 5 and word.isalpha()
        )

        # Convert to sorted list for consistent indexing
        word_list = sorted(list(word_list))

        print(f"Loaded {len(word_list)} valid 5-letter words")
        return word_list

    except Exception as e:
        print(f"Error loading word list: {e}")
        # Fallback to a small set of common 5-letter words
        return ["apple", "brain", "crane", "drama", "early"]


def train_agent(use_replay=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_manager = WordleDataManager(device)
    word_list = load_word_list()
    
    print("Precomputing word embeddings and common states...")
    data_manager.precompute_word_embeddings(word_list)
    data_manager.precompute_common_states(word_list)
    
    env = WordleEnv(word_list, device, data_manager)
    state_size = env.state_size
    action_size = len(word_list)
    agent = WordleAgent(state_size, action_size, word_list, device, data_manager, use_replay)
    
    batch_size = calculate_optimal_batch_size() * 2 if use_replay else 1
    print(f"Using {'replay buffer with ' if use_replay else 'direct'} training, batch size: {batch_size}")
    
    episodes = 10**5
    pbar = tqdm(range(episodes), desc=f"Training Progress")
    best_reward = float("-inf")
    best_checkpoint_path = None
    total_guesses = 0
    completed_episodes = 0
    total_wins = 0
    
    try:
        for episode in pbar:
            state = env.reset()
            total_reward = 0
            losses = []
            episode_guesses = 0
            
            for time in range(env.max_tries):
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                loss = None  # 初始化 loss 變量
                
                if use_replay:
                    agent.remember(state, action, reward, next_state, done)
                    if len(agent.memory) > batch_size:
                        loss = agent.replay(batch_size)
                else:
                    loss = agent.train_on_batch(state, action, reward, next_state, done)
                
                if loss is not None:  # 只有在有效的 loss 時才添加到列表中
                    losses.append(loss)
                
                state = next_state
                total_reward += reward
                episode_guesses += 1

                if done:
                    if reward > 0:
                        total_wins += 1
                    break
            
            total_guesses += episode_guesses
            completed_episodes += 1
            avg_guesses = total_guesses / completed_episodes
            win_rate = (total_wins / completed_episodes) * 100 if completed_episodes > 0 else 0
            
            avg_loss = np.mean(losses) if losses else 0
            pbar.set_description(
                f"Avg Guesses: {avg_guesses:.2f} | "
                f"Win Rate: {win_rate:.1f}% | "
                f"Reward: {total_reward:.2f} | "
                f"Epsilon: {agent.epsilon:.2f} | "
                f"Loss: {avg_loss:.4f}"
            )

            if episode % 100 == 0:
                agent.update_target_model()
                checkpoint_path = agent.save_checkpoint(total_reward)
                
                if total_reward > best_reward:
                    best_reward = total_reward
                    best_checkpoint_path = checkpoint_path
                
            if torch.cuda.is_available() and episode % 1000 == 0:
                torch.cuda.empty_cache()
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if completed_episodes > 0:
            print(f"Final average guesses per episode: {avg_guesses:.2f}")
            print(f"Final win rate: {win_rate:.1f}%")
    
    print(f"\nTraining completed. Best model checkpoint: {best_checkpoint_path}")
    if completed_episodes > 0:
        print(f"Final average guesses per episode: {avg_guesses:.2f}")
        print(f"Final win rate: {win_rate:.1f}%")
    return agent, best_checkpoint_path


if __name__ == "__main__":
    # 設置是否使用 replay
    USE_REPLAY = True  # 可以改為 False 來禁用 replay
    trained_agent, best_checkpoint_path = train_agent(use_replay=USE_REPLAY)
