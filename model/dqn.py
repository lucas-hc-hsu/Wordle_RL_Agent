import torch.nn as nn
import torch

class DQN(nn.Module):
    def __init__(self, state_size, action_size, device):
        super(DQN, self).__init__()
        self.device = device
        
        # Define network layers
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, action_size)
        
        # Layer normalization instead of batch normalization
        self.ln1 = nn.LayerNorm(512)
        self.ln2 = nn.LayerNorm(1024)
        self.ln3 = nn.LayerNorm(2048)
        
        self.dropout = nn.Dropout(0.3)
        self.to(device)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x).to(self.device)
            
        # Handle single sample case
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        x = self.fc1(x)
        x = self.ln1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.ln2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.ln3(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        return self.fc4(x)