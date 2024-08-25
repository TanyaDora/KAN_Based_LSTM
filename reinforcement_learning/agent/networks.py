import torch.nn as nn
import torch
import torch.nn.functional as F


"""
CartPole network
"""

class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=None):
        if hidden_dims is None:
            hidden_dims = [40, 40]

        super(MLP, self).__init__()
        self.hidden_layers = nn.ModuleList()
        
        # Input layer
        self.hidden_layers.append(nn.Linear(state_dim, hidden_dims[0]))
        
        # Hidden layers
        if len(hidden_dims) > 1:
            for i in range(1, len(hidden_dims)):
                self.hidden_layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return x