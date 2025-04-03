import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from networks.rnn_module import GRUModule

def _orthogonal_init(layer, gain=1.0, bias_const=0.0):
    """Enhanced orthogonal initialization with configurable gain and bias."""
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain)
        nn.init.constant_(layer.bias, bias_const)
    elif isinstance(layer, nn.LayerNorm):
        nn.init.constant_(layer.weight, 1.0)
        nn.init.constant_(layer.bias, 0.0)

class Actor_RNN(nn.Module):
    """
    Actor network for MAPPO.
    """
    def __init__(self, input_dim, action_dim, hidden_size, rnn_layers=1, use_feature_normalization=False, output_gain=0.01):
        """
        Initialize the actor network.

        Args:
            input_dim (int): Dimension of the input.
            action_dim (int): Dimension of the action.
            hidden_size (int): Hidden size of the network. 
            rnn_layers (int): Number of RNN layers.
            use_feature_normalization (bool): Whether to use feature normalization.
            output_gain (float): Gain for the output layer.
        """
        super(Actor_RNN, self).__init__()

        self._use_feature_normalization = use_feature_normalization

        if self._use_feature_normalization:
            self.layer_norm = nn.LayerNorm(input_dim)

        # MLP layers before RNN
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
        )

        # RNN layer (GRU)
        self.gru = GRUModule(hidden_size, hidden_size, num_layers=rnn_layers)

        # Output layer
        self.output = nn.Linear(hidden_size, action_dim)

        # Initialize with specific gain
        gain  = nn.init.calculate_gain('relu')

        # Initialize MLP layers
        self.apply(lambda module: _orthogonal_init(module, gain=gain))

        # Initialize the output layer
        _orthogonal_init(self.output, gain=output_gain)
    
    def forward(self, x, rnn_states, masks):
        """
        Forward pass of the actor network.
        Args:
            x (torch.Tensor): Input tensor (num_agents, input_dim) or (seq_len, batch_size, input_dim)
            rnn_states (torch.Tensor): RNN hidden state tensor of shape (num_agents, hidden_size) or (batch_size, hidden_size)
            masks (torch.Tensor): Mask tensor of shape (num_agents, 1) or (seq_len, batch_size, 1)

        Returns:
            logits: action logits
            rnn_states_out: updated RNN states
        """

        if self._use_feature_normalization:
            x = self.layer_norm(x)

        x = self.mlp(x)
        x, rnn_states_out = self.gru(x, rnn_states, masks)
        logits = self.output(x)  # [seq_len, batch_size, action_dim]
        
        return logits, rnn_states_out

    def get_actions(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """Get actions from the actor network.
        
        Args:
            obs: tensor of shape [n_agents, input_dim]
            rnn_states: tensor of shape [n_agents, rnn_hidden_size]
            masks: tensor of shape [n_agents, 1]
            available_actions: tensor of shape [n_agents, action_dim]
            deterministic: bool, whether to use deterministic actions

        Returns:
            actions: tensor of shape [n_agents, 1]
            action_log_probs: tensor of shape [n_agents, 1]
            next_rnn_states: tensor of shape [n_agents, rnn_hidden_size]
        """
        # Forward pass to get logits
        logits, rnn_states_out = self.forward(obs, rnn_states, masks) 

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
        
        return actions, action_log_probs, rnn_states_out


    def evaluate_actions(self, obs_seq, rnn_states, masks_seq, actions_seq, available_actions_seq):
        """Evaluate actions for training.
        
        Args:
            obs_seq: tensor of shape [seq_len, batch_size, input_dim]
            rnn_states: tensor of shape [n_agents, hidden_size] - initial hidden state
            masks_seq: tensor of shape [seq_len, batch_size, 1]
            actions_seq: tensor of shape [seq_len, batch_size, 1]
            available_actions_seq: tensor of shape [seq_len, batch_size, action_dim]
            
        Returns:
            action_log_probs: log probabilities of actions [batch_size, seq_len, 1]
            dist_entropy: entropy of action distribution [batch_size, seq_len, 1]
            rnn_states_out: updated RNN states [batch_size, hidden_size]  
        """
        logits, rnn_states_out = self.forward(obs_seq, rnn_states, masks_seq) 
        # [seq_len, batch_size, action_dim], [batch_size, hidden_size]

        # resshape for processing
        seq_len, batch_size = obs_seq.shape[:2] 

        # Flatten logits and actions for categorical operations
        flat_logits = logits.reshape(-1, logits.size(-1))  # [seq_len*batch_size, action_dim]
        # Apply mask for available actions if provided
        if available_actions_seq is not None:
            flat_available = available_actions_seq.reshape(-1, available_actions_seq.size(-1))
            # Set unavailable actions to have a very small probability
            flat_logits[flat_available == 0] = -1e10
       
        flat_actions = actions_seq.reshape(-1)  # [seq_len*batch_size, 1]
        
        # Compute probabilities and log probs
        action_probs = F.softmax(flat_logits, dim=-1)
        action_dist = Categorical(action_probs)
        action_log_probs = action_dist.log_prob(flat_actions)
        dist_entropy = action_dist.entropy()

        # Reshape back
        action_log_probs = action_log_probs.view(seq_len, batch_size, 1)
        dist_entropy = dist_entropy.view(seq_len, batch_size, 1)    

        return action_log_probs, dist_entropy, rnn_states_out


class Critic_RNN(nn.Module):
    """
    Critic network for MAPPO.
    """
    def __init__(self, input_dim, hidden_size, rnn_layers=1, use_feature_normalization=False):
        """
        Initialize the critic network.

        Args:
            input_dim (int): Dimension of the input.
            hidden_size (int): Hidden size of the network.
            rnn_layers (int): Number of RNN layers.
            use_feature_normalization (bool): Whether to use feature normalization
        """
        super(Critic_RNN, self).__init__()

        self._use_feature_normalization = use_feature_normalization

        if self._use_feature_normalization:
            self.layer_norm = nn.LayerNorm(input_dim)

        # MLP layers before RNN
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
        )
        
        # RNN layer (GRU)
        self.gru = GRUModule(hidden_size, hidden_size, num_layers=rnn_layers)

        # Output layer
        self.output = nn.Linear(hidden_size, 1)

        # Initialize with specific gains for each layer type
        gain = nn.init.calculate_gain('relu')

        # Initialize MLP layers
        self.apply(lambda module: _orthogonal_init(module, gain=gain))
        
        # Initialize the output layer
        _orthogonal_init(self.output, gain=gain)
    
    def forward(self, x, rnn_states, masks):
        """Forward pass for critic network.
        
        Args:
            x (torch.Tensor): Input tensor (num_agents, input_dim) or (seq_len, num_agents, input_dim)
            rnn_states (torch.Tensor): RNN hidden state tensor of shape (num_agents, hidden_size) or (batch_size, hidden_size)
            masks (torch.Tensor): Mask tensor of shape (num_agents, 1) or (seq_len, batch_size, 1)
            
        Returns:
            values (torch.Tensor): Value predictions, shape [n_agents, 1] or 
                                  [seq_len, batch_size, 1].
            rnn_states_out (torch.Tensor): Updated RNN states, shape [n_agents, hidden_size] or 
                                          [batch_size, hidden_size]
        """

        if self._use_feature_normalization:
            x = self.layer_norm(x)
        
        x = self.mlp(x)
        x, rnn_states_out = self.gru(x, rnn_states, masks)
        values = self.output(x)  # [seq_len, batch_size, 1]
        
        return values, rnn_states_out