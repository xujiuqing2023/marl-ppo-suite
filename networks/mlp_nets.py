import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.distributions import Categorical


def _orthogonal_init(layer, gain=1.0, bias_const=0.0):
    """Enhanced orthogonal initialization with configurable gain and bias."""
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain)
        nn.init.constant_(layer.bias, bias_const)
    elif isinstance(layer, nn.LayerNorm):
        nn.init.constant_(layer.weight, 1.0)
        nn.init.constant_(layer.bias, 0.0)

class Actor_MLP(nn.Module):
    """
    MLP Actor network for MAPPO.
    """
    def __init__(self, input_dim, action_dim, hidden_size, use_feature_normalization=False, output_gain=0.01):
        """
        Initialize the actor network.

        Args:
            input_dim (int): Dimension of the input.
            action_dim (int): Dimension of the action.
            hidden_size (int): Hidden size of the network.
            use_feature_normalization (bool): Whether to use feature normalization.
            output_gain (float): Gain for the output layer.
        """
        super(Actor_MLP, self).__init__()

        self._use_feature_normalization = use_feature_normalization
        
        if self._use_feature_normalization:
            self.layer_norm = nn.LayerNorm(input_dim)
        
        # MLP layers  
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, action_dim),
        )

        # Initialize with specific gain
        gain  = nn.init.calculate_gain('relu')

        # Initialize MLP layers
        self.apply(lambda module: _orthogonal_init(module, gain=gain))
        # Initialize the output layer
        _orthogonal_init(self.mlp[-1], gain=output_gain)
        
    def forward(self, x):
        """Forward pass of the actor network.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:    
            logits (torch.Tensor): Logits of the action distribution.
        """

        if self._use_feature_normalization:
            x = self.layer_norm(x)

        return self.mlp(x)
    
    def get_actions(self, x, available_actions=None, deterministic=False):
        """Get actions from the actor network.
        
        Args:
            x (torch.Tensor): Input tensor.
            available_actions (torch.Tensor, optional): Mask for available actions.
            deterministic (bool): Whether to use deterministic actions.
            
        Returns:    
            actions (torch.Tensor): Actions.
            action_log_probs (torch.Tensor): Log probabilities of actions.
        """
        logits = self.forward(x)
        # print(f"logits: {logits.shape}")
        # print(f"logits: {logits}")
        # print(f"available_actions: {available_actions.shape}")
        # print(f"available_actions: {available_actions}")

        # Apply mask for available actions if provided
        if available_actions is not None:
            # Set unavailable actions to have a very small probability
            logits[available_actions == 0] = -1e10

        
        if deterministic:
            actions = torch.argmax(logits, dim=-1, keepdim=True)
            action_log_probs = None
        else:
            # Convert logits to action probabilities
            action_probs = F.softmax(logits, dim=-1)
            action_dist = Categorical(action_probs) 
            actions = action_dist.sample().unsqueeze(-1) # (n_agents, 1)
            action_log_probs = action_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1) # (n_agents, 1)
            
        return actions, action_log_probs

    def evaluate_actions(self, x, actions, available_actions=None):
        """Evaluate actions for training.
        
        Args:
            x (torch.Tensor): Input tensor.
            actions (torch.Tensor): Actions.
            available_actions (torch.Tensor, optional): Mask for available actions.

        Returns:
            action_log_probs (torch.Tensor): Log probabilities of actions.
            dist_entropy (torch.Tensor): Entropy of the action distribution.
        """
        logits = self.forward(x)

        # Apply mask for available actions if provided
        if available_actions is not None:
            # Set unavailable actions to have a very small probability
            logits[available_actions == 0] = -1e10

        action_dist = Categorical(F.softmax(logits, dim=-1))
        action_log_probs = action_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1) # (n_agents, 1)
        dist_entropy = action_dist.entropy().unsqueeze(-1) # (n_agents, 1)
        return action_log_probs, dist_entropy
    

class Critic_MLP(nn.Module):
    """
    MLP Critic network for MAPPO.
    """
    def __init__(self, input_dim, hidden_size, use_feature_normalization=False):
        """
        Initialize the critic network.

        Args:
            input_dim (int): Dimension of the input.
            hidden_size (int): Hidden size of the network.
            use_feature_normalization (bool): Whether to use feature normalization.
        """
        super(Critic_MLP, self).__init__()

        self._use_feature_normalization = use_feature_normalization

        if self._use_feature_normalization:
            self.layer_norm = nn.LayerNorm(input_dim)

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )

        # Initialize with specific gain
        gain  = nn.init.calculate_gain('relu')

        # Initialize MLP layers
        self.apply(lambda module: _orthogonal_init(module, gain=gain))

        # Initialize the output layer
        _orthogonal_init(self.mlp[-1])
        
    def forward(self, x):
        """Forward pass of the critic network.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            value (torch.Tensor): Value.
        """
        if self._use_feature_normalization:
            x = self.layer_norm(x)

        return self.mlp(x)
    
