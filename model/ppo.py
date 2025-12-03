import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO algorithm.
    The actor outputs action probabilities, and the critic outputs state values.
    """

    def __init__(self, state_size, action_size, device, hidden_size=512):
        super(ActorCritic, self).__init__()
        self.device = device

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        # Actor head (policy network)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.to(device)

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        shared_features = self.shared(state)
        action_logits = self.actor(shared_features)
        state_value = self.critic(shared_features)

        return action_logits, state_value

    def act(self, state, valid_actions_mask=None):
        """
        Select an action based on the current policy.

        Args:
            state: Current state
            valid_actions_mask: Optional mask for valid actions (1 for valid, 0 for invalid)

        Returns:
            action: Selected action
            log_prob: Log probability of the action
            state_value: Estimated state value
        """
        action_logits, state_value = self.forward(state)

        # Apply mask for valid actions if provided
        if valid_actions_mask is not None:
            if not isinstance(valid_actions_mask, torch.Tensor):
                valid_actions_mask = torch.FloatTensor(valid_actions_mask).to(self.device)
            # Set invalid actions to very negative value
            action_logits = action_logits.masked_fill(
                valid_actions_mask == 0, float("-inf")
            )

        # Create action distribution
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)

        # Sample action
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob, state_value.squeeze()

    def evaluate(self, states, actions, valid_actions_masks=None):
        """
        Evaluate actions for PPO update.

        Args:
            states: Batch of states
            actions: Batch of actions taken
            valid_actions_masks: Optional batch of masks for valid actions

        Returns:
            log_probs: Log probabilities of actions
            state_values: Estimated state values
            entropy: Policy entropy for exploration bonus
        """
        action_logits, state_values = self.forward(states)

        # Apply masks if provided
        if valid_actions_masks is not None:
            if not isinstance(valid_actions_masks, torch.Tensor):
                valid_actions_masks = torch.FloatTensor(valid_actions_masks).to(
                    self.device
                )
            action_logits = action_logits.masked_fill(
                valid_actions_masks == 0, float("-inf")
            )

        action_probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, state_values.squeeze(), entropy
