import numpy as np
import torch

def _transform_data(data_np: np.ndarray, device: torch.device, sequence_first: bool = False) -> torch.Tensor:
    """
    Transform data to be used in RNN.
    
    Args:
        data_np: Input numpy array with shape [T, M, feat_dim] or [T, num_layers, M, feat_dim] for RNN states
        device: Target device for tensor
        sequence_first: If True, keeps sequence dimension first [T, M, feat_dim] -> [T, M, feat_dim]
                      If False, transposes sequence and batch [T, M, feat_dim] -> [M, T, feat_dim] -> [M*T, feat_dim]
    
    Returns:
        Transformed torch tensor
    """
    if sequence_first:
        # Keep original sequence ordering
        return torch.tensor(data_np, dtype=torch.float32).to(device)
    else:
        # Original behavior: transpose and flatten
        reshaped_data = data_np.transpose(1, 0, 2).reshape(-1, *data_np.shape[2:])
        return torch.tensor(reshaped_data, dtype=torch.float32).to(device)



class RecurrentRolloutStorage:
    """
    Rollout storage for collecting multi-agent experiences during training.
    Designed for MAPPO with n-step returns and MLP-based policies.
    """
    def __init__(self, n_steps, n_agents, obs_dim, action_dim, state_dim, hidden_size,
                 num_rnn_layers=1, device='cpu'):
        """
        Initialize rollout storage for collecting experiences.  
        
        Args:
            n_steps (int): Number of steps to collect before update (can be different from episode length)
            n_agents (int): Number of agents in the environment
            obs_dim (int): Dimension of individual agent observations
            action_dim (int): Dimension of agent actions
            state_dim (int): Dimension of global state
            hidden_size (int): Dimension of hidden state  
            num_rnn_layers (int): Number of layers in the RNN  
            device (str): Device for storage ('cpu' for numpy-based implementation)
        """
        self.n_steps = n_steps
        self.n_agents = n_agents
        self.num_rnn_layers = num_rnn_layers
        self.device = torch.device(device)
        
        # Current position in the buffer
        self.step = 0
        # (obs_0, state_0) → action_0 / action_log_prob_0 → (reward_0, obs_1, state_1, mask_1, trunc_1)
        # delta = reward_0 + gamma * value_1 * mask_1 - value_0

        # Core storage buffers - using numpy arrays for efficiency
        self.obs = np.zeros((n_steps + 1, n_agents, obs_dim), dtype=np.float32)
        self.global_state = np.zeros((n_steps + 1, state_dim), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_agents, 1), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_agents, 1), dtype=np.int64)
        self.action_log_probs = np.zeros((n_steps, n_agents, 1), dtype=np.float32)
        self.values = np.zeros((n_steps + 1, n_agents, 1), dtype=np.float32)
        self.masks = np.ones((n_steps + 1, n_agents, 1), dtype=np.float32) # 0 if episode done, 1 otherwise
        self.truncated = np.zeros((n_steps + 1, n_agents, 1), dtype=np.bool_) # 1 if episode truncated, 0 otherwise
        self.available_actions = np.zeros((n_steps+1, n_agents, action_dim), dtype=np.bool_)

        # RNN hidden states
        # Shape: [n_steps + 1, num_layers, n_agents, hidden_size]
        self.actor_rnn_states = np.zeros(
            (n_steps + 1, num_rnn_layers, n_agents, hidden_size), 
            dtype=np.float32)
        self.critic_rnn_states = np.zeros_like(self.actor_rnn_states)
        
        # Extra buffers for the algorithm
        self.returns = np.zeros((n_steps + 1, n_agents, 1), dtype=np.float32)
        self.advantages = np.zeros((n_steps, n_agents, 1), dtype=np.float32)
        
    def insert(self, obs, global_state, actions, 
        action_log_probs, values, rewards, 
        masks, truncates, available_actions,
        actor_rnn_states, critic_rnn_states):
        """
        Insert a new transition into the buffer.
        
        Args:
            obs: Agent observations [n_agents, obs_shape]
            global_state: Global state if available [state_shape]
            actions: Actions taken by agents [n_agents, action_shape]
            action_log_probs: Log probs of actions [n_agents, 1]
            values: Value predictions [n_agents, 1]
            rewards: Rewards received [n_agents]
            masks: Episode termination masks [n_agents], 0 if episode done, 1 otherwise
            truncates: Boolean array indicating if episode was truncated (e.g., due to time limit) 
                      rather than terminated [n_agents]
            available_actions: Available actions mask [n_agents, n_actions]
            actor_rnn_states: RNN states [num_layers, n_agents, hidden_size]
            critic_rnn_states: RNN states [num_layers, n_agents, hidden_size]
        """
        self.obs[self.step + 1] = obs.copy()
        self.global_state[self.step + 1] = global_state.copy()
            
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.values[self.step] = values.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        self.truncated[self.step + 1] = truncates.copy()
        self.available_actions[self.step + 1] = available_actions.copy()
        
        self.actor_rnn_states[self.step + 1] = actor_rnn_states.copy()
        self.critic_rnn_states[self.step + 1] = critic_rnn_states.copy()
        
        self.step += 1
        
    def compute_returns_and_advantages(self, next_values, gamma=0.99, lambda_=0.95, use_gae=True):
        """
        Compute returns and advantages using GAE (Generalized Advantage Estimation).
        Properly handles truncated episodes by incorporating next state values.

        Args:
            next_values: Value estimates for the next observations [n_rollout_threads, n_agents, 1]
            gamma: Discount factor
            lambda_: GAE lambda parameter for advantage weighting
            use_gae: Whether to use GAE or just n-step returns
        """
        # Set the value of the next observation
        self.values[-1] = next_values
        
        # Create arrays for storing returns and advantages
        advantages = np.zeros_like(self.rewards)
        returns = np.zeros_like(self.returns)
        
        if use_gae:
            # GAE advantage computation with vectorized operations for better performance
            gae = 0
            for step in reversed(range(self.n_steps)):       
                # For truncated episodes, we adjust rewards directly
                adjusted_rewards = self.rewards[step].copy() # [n_agents]
                
                # Identify truncated episodes (done but not terminated)
                truncated_mask = (self.masks[step + 1] == 0) & (self.truncated[step + 1] == 1) # [n_agents]
                if np.any(truncated_mask):
                    # Add bootstrapped value only for truncated episodes
                    adjusted_rewards[truncated_mask] += gamma * self.values[step + 1][truncated_mask]
                    
                # Calculate delta (TD error) with adjusted rewards
                delta = (
                    adjusted_rewards + 
                    gamma * self.values[step + 1] * self.masks[step + 1] - 
                    self.values[step]
                ) # [n_agents]
                
                # Standard GAE calculation 
                gae = delta + gamma * lambda_ * self.masks[step + 1] * gae # [n_agents]
                advantages[step] = gae # [n_agents]
                
            # Compute returns as advantages + values
            returns[:-1] = advantages + self.values[:-1] # [n_agents]
            returns[-1] = next_values # [n_agents]
            
        else:
            # N-step returns without GAE (more efficient calculation)
            returns[-1] = next_values
            for step in reversed(range(self.n_steps)):
                # Adjust rewards for truncated episodes
                adjusted_rewards = self.rewards[step].copy()
                
                # Identify truncated episodes
                truncated_mask = (self.masks[step + 1] == 0) & (self.truncated[step + 1] == 1)
                
                # For truncated episodes, add discounted bootstrapped value directly to rewards
                if np.any(truncated_mask):
                    adjusted_rewards[truncated_mask] += gamma * returns[step + 1][truncated_mask]

                # Calculate returns with proper masking
                returns[step] = adjusted_rewards + gamma * returns[step + 1] * self.masks[step + 1]
                
            # Calculate advantages
            advantages = returns[:-1] - self.values[:-1]
        
        # Store results
        self.returns = returns
        
        # Normalize advantages (helps with training stability)
        # Use stable normalization with small epsilon
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        self.advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        
        return self.advantages, self.returns

    def after_update(self):
        """Copy the last observation and masks to the beginning for the next update."""
        self.obs[0] = self.obs[-1].copy()
        self.global_state[0] = self.global_state[-1].copy()    
        self.masks[0] = self.masks[-1].copy()
        self.truncated[0] = self.truncated[-1].copy()
        self.available_actions[0] = self.available_actions[-1].copy()

        # Copy RNN states
        self.actor_rnn_states[0] = self.actor_rnn_states[-1].copy()
        self.critic_rnn_states[0] = self.critic_rnn_states[-1].copy()

        # Reset step counter
        self.step = 0


    def get_minibatches_batch_first(self, num_mini_batch, data_chunk_length = 10):
        """
        Create minibatches for training RNN using data chunks of length data_chunk_length.
        
        Args:
            num_mini_batch (int): Number of minibatches to create
            data_chunk_length (int): Length of data chunk to use for training, default is 10
        
        Returns:
            Generator yielding minibatches for training
        """
        # Calculate total steps and minibatch size
        total_steps = self.n_steps
        if total_steps < data_chunk_length:
            raise ValueError(f"n_steps ({total_steps}) must be >= data_chunk_length ({data_chunk_length})")
        
        # Calculate total possible chunks
        max_data_chunks = total_steps // data_chunk_length
        if max_data_chunks < num_mini_batch:
            num_mini_batch = max_data_chunks  # Adjust if too few chunks

        # Number of chunks per minibatch
        mini_batch_size = max(1, max_data_chunks // num_mini_batch)

        # Generate chunk start indices (ensure coverage and randomness)
        all_starts = np.arange(0, total_steps - data_chunk_length + 1, data_chunk_length)
        if len(all_starts) > max_data_chunks:
            all_starts = all_starts[:max_data_chunks]
        np.random.shuffle(all_starts)

        # Organize data chunks for mini-batches
        for i in range(0, len(all_starts), mini_batch_size):
            batch_chunk_starts = all_starts[i:i + mini_batch_size]
            if not batch_chunk_starts.size:
                continue

            # Collect sequences for each data type
            sequences = {
                'obs': [], 'global_state': [], 'actions': [], 'values': [],
                'returns': [], 'masks': [], 'action_log_probs': [],
                'advantages': [], 'available_actions': [],
                'actor_rnn_init_states': [], 'critic_rnn_init_states': []
            }

            # For each starting position, collect a sequence of length data_chunk_length
            for start_idx in batch_chunk_starts:
                end_idx = start_idx + data_chunk_length

               
                # Collect data sequences
                sequences['obs'].append(self.obs[start_idx:end_idx])
                sequences['global_state'].append(self.global_state[start_idx:end_idx])
                sequences['actions'].append(self.actions[start_idx:end_idx])
                sequences['values'].append(self.values[start_idx:end_idx])
                sequences['returns'].append(self.returns[start_idx:end_idx])
                sequences['masks'].append(self.masks[start_idx:end_idx])
                sequences['action_log_probs'].append(self.action_log_probs[start_idx:end_idx])
                sequences['advantages'].append(self.advantages[start_idx:end_idx])
                sequences['available_actions'].append(self.available_actions[start_idx:end_idx])
                
                # Get initial RNN states for the sequence
                sequences['actor_rnn_init_states'].append(self.actor_rnn_states[start_idx])
                sequences['critic_rnn_init_states'].append(self.critic_rnn_states[start_idx])

            # Stack sequences into batch tensors
            # Shape will be [batch_size, seq_length, n_agents, feat_dim] 
            # but for rnn_init_states, it will be [batch_size, num_layers, n_agents, hidden_size]
            batch = {
                key: torch.tensor(
                    np.stack(sequences[key]), 
                    dtype=torch.float32 if key not in ['actions'] else torch.int64
                ).to(self.device)
                for key in sequences
            }
                
            yield batch

    def get_minibatches_seq_first(self, num_mini_batch, data_chunk_length = 10):
        """
        Create minibatches for training RNN, flattening num_agents into total steps.
        Returns sequences with shape [seq_len, batch_size, feat_dim] and RNN states 
        with shape [num_layers, batch_size, hidden_dim].

        Args:
            num_mini_batch (int): Number of minibatches to create
            data_chunk_length (int): Length of data chunk to use for training, default is 10
        
        Returns:
            Generator yielding minibatches as tuples with the following keys:
            - obs: [seq_len, batch_size, obs_dim]
            - global_state: [seq_len, batch_size, state_dim]
            - actor_rnn_states: [num_layers, batch_size, hidden_size]
            - critic_rnn_states: [num_layers, batch_size, hidden_size]
            - actions, values, returns, etc.: [seq_len, batch_size, dim]
        """
        total_steps = self.n_steps # e.g., [T, M, feat_dim] -> T
        if total_steps < data_chunk_length:
            raise ValueError(f"n_steps ({total_steps}) must be >= data_chunk_length ({data_chunk_length})")
        
        # Calculate chunks and batch sizes
        num_agents = self.obs.shape[1]  # e.g., [T, M, feat_dim] -> M
        total_agent_steps = total_steps * num_agents  # e.g., T * M = 400 * 5 = 2000
        max_data_chunks = total_agent_steps // data_chunk_length  # e.g., 2000 // 10 = 200
        
        # Adjust mini_batch count if needed
        num_mini_batch = min(num_mini_batch, max_data_chunks) # e.g., 1
        mini_batch_size = max(1, max_data_chunks // num_mini_batch)  # e.g., 200 // 1 = 20

        # Pre-convert and flatten data, collapsing num_agents into the sequence
        data = {
            'obs': _transform_data(self.obs[:-1], self.device),
            'actions': _transform_data(self.actions, self.device),
            'values': _transform_data(self.values[:-1], self.device),
            'returns': _transform_data(self.returns[:-1], self.device),
            'masks': _transform_data(self.masks[:-1], self.device),
            'old_action_log_probs': _transform_data(self.action_log_probs, self.device),
            'advantages': _transform_data(self.advantages, self.device),
            'available_actions': _transform_data(self.available_actions[:-1], self.device),
        }

        # Handle global state specially since it needs to be repeated for each agent
        global_state = np.expand_dims(self.global_state[:-1], axis=1).repeat(self.n_agents, axis=1)
        data['global_state'] = _transform_data(global_state, self.device)
        
        # Process RNN states - maintain num_layers dimension while flattening agents
        actor_rnn_states  =  self.actor_rnn_states[:-1].transpose(2, 0, 1, 3).reshape(-1, 
                                                                                      self.actor_rnn_states.shape[1], 
                                                                                      self.actor_rnn_states.shape[-1])
        critic_rnn_states =  self.critic_rnn_states[:-1].transpose(2, 0, 1, 3).reshape(-1, 
                                                                                       self.critic_rnn_states.shape[1], 
                                                                                    self.critic_rnn_states.shape[-1])
 
        data['actor_rnn_states'] = torch.tensor(actor_rnn_states, dtype=torch.float32, device=self.device)
        data['critic_rnn_states'] = torch.tensor(critic_rnn_states, dtype=torch.float32, device=self.device)

        # Generate chunk start indices over total_agent_steps
        all_starts = np.arange(0, total_agent_steps - data_chunk_length + 1, data_chunk_length)
        if len(all_starts) > max_data_chunks:
            all_starts = all_starts[:max_data_chunks]
        np.random.shuffle(all_starts)

        # Generate minibatches
        for batch_start in range(0, len(all_starts), mini_batch_size):
            batch_chunk_starts = all_starts[batch_start:batch_start + mini_batch_size]
            if not batch_chunk_starts.size:
                continue
            
            # Collect sequences
            sequences = {key: [] for key in data.keys()}

            for start_idx in batch_chunk_starts:
                end_idx = start_idx + data_chunk_length

                # Collect sequences for each data type
                for key in data.keys():
                    if key not in ['actor_rnn_states', 'critic_rnn_states']:
                        sequences[key].append(data[key][start_idx:end_idx])

                # Get initial RNN states
                sequences['actor_rnn_states'].append(data['actor_rnn_states'][start_idx])
                sequences['critic_rnn_states'].append(data['critic_rnn_states'][start_idx])

            # Stack sequences into proper shapes (seq_len, batch_size, feat_dim) or (num_layers, batch_size, hidden_dim)
            batch = {
                key: torch.stack(sequences[key], dim=1)
                for key in sequences
            }

            # Yield the minibatch as a tuple
            yield ( 
                batch['obs'],
                batch['global_state'],
                batch['actor_rnn_states'],
                batch['critic_rnn_states'],
                batch['actions'],
                batch['values'],
                batch['returns'],
                batch['masks'],
                batch['old_action_log_probs'],
                batch['advantages'],
                batch['available_actions']
            )



    # def get_minibatches_seq_first(self, num_mini_batch, data_chunk_length = 10):
    #     """
    #     Create minibatches for training RNN, flattening num_agents into total steps.
    #     Returns sequences with shape [seq_len, batch_size, feat_dim] and RNN states 
    #     with shape [num_layers, batch_size, hidden_dim].

    #     Args:
    #         num_mini_batch (int): Number of minibatches to create
    #         data_chunk_length (int): Length of data chunk to use for training, default is 10
        
    #     Returns:
    #         Generator yielding minibatches as tuples with the following keys:
    #         - obs: [seq_len, batch_size, obs_dim]
    #         - global_state: [seq_len, batch_size, state_dim]
    #         - actor_rnn_states: [num_layers, batch_size, hidden_size]
    #         - critic_rnn_states: [num_layers, batch_size, hidden_size]
    #         - actions, values, returns, etc.: [seq_len, batch_size, dim]
    #     """
    #     total_steps = self.n_steps # e.g., [T, M, feat_dim] -> T
    #     if total_steps < data_chunk_length:
    #         raise ValueError(f"n_steps ({total_steps}) must be >= data_chunk_length ({data_chunk_length})")
        
    #     # Calculate chunks and batch sizes
    #     num_agents = self.obs.shape[1]  # e.g., [T, M, feat_dim] -> M
    #     total_agent_steps = total_steps * num_agents  # e.g., T * M = 400 * 5 = 2000
    #     max_data_chunks = total_agent_steps // data_chunk_length  # e.g., 2000 // 10 = 200
        
    #     # Adjust mini_batch count if needed
    #     num_mini_batch = min(num_mini_batch, max_data_chunks) # e.g., 1
    #     mini_batch_size = max(1, max_data_chunks // num_mini_batch)  # e.g., 200 // 1 = 20

    #     # Pre-convert and flatten data, collapsing num_agents into the sequence
    #     data = {
    #         'obs': _transform_data(self.obs[:-1], self.device),
    #         'actions': _transform_data(self.actions, self.device),
    #         'values': _transform_data(self.values[:-1], self.device),
    #         'returns': _transform_data(self.returns[:-1], self.device),
    #         'masks': _transform_data(self.masks[:-1], self.device),
    #         'old_action_log_probs': _transform_data(self.action_log_probs, self.device),
    #         'advantages': _transform_data(self.advantages, self.device),
    #         'available_actions': _transform_data(self.available_actions[:-1], self.device),
    #     }

    #     # Handle global state specially since it needs to be repeated for each agent
    #     global_state = np.expand_dims(self.global_state[:-1], axis=1).repeat(self.n_agents, axis=1)
    #     data['global_state'] = _transform_data(global_state, self.device)
        
    #     # Process RNN states - maintain num_layers dimension while flattening agents
    #     actor_rnn_states  =  self.actor_rnn_states[:-1].transpose(2, 0, 1, 3).reshape(-1, *self.actor_rnn_states.shape[2:])
    #     critic_rnn_states =  self.critic_rnn_states[:-1].transpose(2, 0, 1, 3).reshape(-1, *self.critic_rnn_states.shape[2:])
    #     data['actor_rnn_init_states'] = torch.tensor(actor_rnn_states, dtype=torch.float32, device=self.device)
    #     data['critic_rnn_init_states'] = torch.tensor(critic_rnn_states, dtype=torch.float32, device=self.device)

    #     # Generate chunk start indices over total_agent_steps
    #     all_starts = np.arange(0, total_agent_steps - data_chunk_length + 1, data_chunk_length)
    #     if len(all_starts) > max_data_chunks:
    #         all_starts = all_starts[:max_data_chunks]
    #     np.random.shuffle(all_starts)

    #     # Generate minibatches
    #     for batch_start in range(0, len(all_starts), mini_batch_size):
    #         batch_chunk_starts = all_starts[batch_start:batch_start + mini_batch_size]
    #         if not batch_chunk_starts.size:
    #             continue

    #         # Collect sequences and RNN states
    #         obs_batch = []
    #         global_state_batch = []
    #         actions_batch = []
    #         values_batch = []
    #         returns_batch = []
    #         masks_batch = []
    #         old_action_log_probs_batch = []
    #         advantages_batch = []
    #         available_actions_batch = []
    #         actor_rnn_init_states_batch = []
    #         critic_rnn_init_states_batch = []

    #         for start_idx in batch_chunk_starts:
    #             end_idx = start_idx + data_chunk_length
    #             obs_batch.append(data['obs'][start_idx:end_idx])  # [L, obs_dim]
    #             global_state_batch.append(data['global_state'][start_idx:end_idx])
    #             actions_batch.append(data['actions'][start_idx:end_idx])
    #             values_batch.append(data['values'][start_idx:end_idx])
    #             returns_batch.append(data['returns'][start_idx:end_idx])
    #             masks_batch.append(data['masks'][start_idx:end_idx])
    #             old_action_log_probs_batch.append(data['action_log_probs'][start_idx:end_idx])
    #             advantages_batch.append(data['advantages'][start_idx:end_idx])
    #             available_actions_batch.append(data['available_actions'][start_idx:end_idx])
    #             actor_rnn_init_states_batch.append(data['actor_rnn_init_states'][start_idx])
    #             critic_rnn_init_states_batch.append(data['critic_rnn_init_states'][start_idx])

    #         # Stack sequences into [seq_len, batch_size, feat_dim]
    #         L, N = data_chunk_length, mini_batch_size
    #         obs_batch = torch.stack(obs_batch, dim=1)  # [L, N, obs_dim]
    #         global_state_batch = torch.stack(global_state_batch, dim=1)  # [L, N, global_dim]
    #         actions_batch = torch.stack(actions_batch, dim=1)  # [L, N, action_dim]
    #         values_batch = torch.stack(values_batch, dim=1)  # [L, N, 1]
    #         returns_batch = torch.stack(returns_batch, dim=1)  # [L, N, 1]
    #         masks_batch = torch.stack(masks_batch, dim=1)  # [L, N, 1]
    #         old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, dim=1)  # [L, N, 1]
    #         advantages_batch = torch.stack(advantages_batch, dim=1)  # [L, N, 1]
    #         available_actions_batch = torch.stack(available_actions_batch, dim=1)  # [L, N, action_space]

    #         # Stack RNN states into [batch_size, hidden_size]
    #         actor_h0_batch = torch.stack(actor_rnn_init_states_batch, dim=1)  # [num_layers, N, hidden_size]
    #         critic_h0_batch = torch.stack(critic_rnn_init_states_batch, dim=1)  # [num_layers, N, hidden_size]

    #         # Yield the minibatch as a tuple
    #         yield (obs_batch, global_state_batch, actor_h0_batch, critic_h0_batch,
    #             actions_batch, values_batch, returns_batch, masks_batch, old_action_log_probs_batch,
    #             advantages_batch, available_actions_batch)



    # def get_minibatches_seq_first(self, num_mini_batch, data_chunk_length = 10):
    #     """
    #     Create minibatches for training RNN using data chunks of length data_chunk_length.
    #     Returns sequences with shape [seq_len, batch_size, ...] and RNN states with shape [batch_size, ...].

    #     Args:
    #         num_mini_batch (int): Number of minibatches to create
    #         data_chunk_length (int): Length of data chunk to use for training, default is 10
        
    #     Returns:
    #         Generator yielding minibatches for training
    #     """
    #     # Calculate total steps and minibatch size
    #     total_steps = self.n_steps
    #     if total_steps < data_chunk_length:
    #         raise ValueError("n_steps must be >= data_chunk_length")
        
    #     # Calculate total possible chunks
    #     max_data_chunks = total_steps // data_chunk_length
    #     if max_data_chunks < num_mini_batch:
    #         num_mini_batch = max_data_chunks  # Adjust if too few chunks

    #     # Number of chunks per minibatch (batch_size in RNN context)
    #     mini_batch_size = max(1, max_data_chunks // num_mini_batch)

    #     # Generate chunk start indices
    #     all_starts = np.arange(0, total_steps - data_chunk_length + 1, data_chunk_length)
    #     if len(all_starts) > max_data_chunks:
    #         all_starts = all_starts[:max_data_chunks]
    #     np.random.shuffle(all_starts)

    #     # Pre-convert data to tensors for efficiency
    #     data = {
    #         'obs': torch.tensor(self.obs, dtype=torch.float32, device=self.device),
    #         'global_state': torch.tensor(self.global_state, dtype=torch.float32, device=self.device),
    #         'actions': torch.tensor(self.actions, dtype=torch.int64, device=self.device),
    #         'values': torch.tensor(self.values, dtype=torch.float32, device=self.device),
    #         'returns': torch.tensor(self.returns, dtype=torch.float32, device=self.device),
    #         'masks': torch.tensor(self.masks, dtype=torch.float32, device=self.device),
    #         'action_log_probs': torch.tensor(self.action_log_probs, dtype=torch.float32, device=self.device),
    #         'advantages': torch.tensor(self.advantages, dtype=torch.float32, device=self.device),
    #         'available_actions': torch.tensor(self.available_actions, dtype=torch.bool, device=self.device),
    #         'actor_rnn_init_states': torch.tensor(self.actor_rnn_states, dtype=torch.float32, device=self.device),
    #         'critic_rnn_init_states': torch.tensor(self.critic_rnn_states, dtype=torch.float32, device=self.device),
    #     }

    #     # Organize data chunks for minibatches
    #     for i in range(0, len(all_starts), mini_batch_size):
    #         batch_chunk_starts = all_starts[i:i + mini_batch_size]
    #         if not batch_chunk_starts.size:
    #             continue

    #         # Collect sequences and RNN states
    #         obs_batch = []
    #         global_state_batch = []
    #         actions_batch = []
    #         values_batch = []
    #         returns_batch = []
    #         masks_batch = []
    #         action_log_probs_batch = []
    #         advantages_batch = []
    #         available_actions_batch = []
    #         actor_rnn_init_states_batch = []
    #         critic_rnn_init_states_batch = []

    #         for start_idx in batch_chunk_starts:
    #             end_idx = start_idx + data_chunk_length
    #             obs_batch.append(data['obs'][start_idx:end_idx])
    #             global_state_batch.append(data['global_state'][start_idx:end_idx])
    #             actions_batch.append(data['actions'][start_idx:end_idx])
    #             values_batch.append(data['values'][start_idx:end_idx])
    #             returns_batch.append(data['returns'][start_idx:end_idx])
    #             masks_batch.append(data['masks'][start_idx:end_idx])
    #             action_log_probs_batch.append(data['action_log_probs'][start_idx:end_idx])
    #             advantages_batch.append(data['advantages'][start_idx:end_idx])
    #             available_actions_batch.append(data['available_actions'][start_idx:end_idx])
    #             actor_rnn_init_states_batch.append(data['actor_rnn_init_states'][start_idx])
    #             critic_rnn_init_states_batch.append(data['critic_rnn_init_states'][start_idx])
            
    #         # Stack sequences into [seq_len, batch_size, n_agents, feat_dim]
    #         L, N = data_chunk_length, mini_batch_size
    #         obs_batch = torch.stack(obs_batch, dim=1)  # [L, N, n_agents, obs_dim]
    #         global_state_batch = torch.stack(global_state_batch, dim=1)  # [L, N, global_dim]
    #         actions_batch = torch.stack(actions_batch, dim=1)  # [L, N, n_agents, action_dim]
    #         values_batch = torch.stack(values_batch, dim=1)  # [L, N, n_agents, 1]
    #         returns_batch = torch.stack(returns_batch, dim=1)  # [L, N, n_agents, 1]
    #         masks_batch = torch.stack(masks_batch, dim=1)  # [L, N, n_agents, 1]
    #         action_log_probs_batch = torch.stack(action_log_probs_batch, dim=1)  # [L, N, n_agents, 1]
    #         advantages_batch = torch.stack(advantages_batch, dim=1)  # [L, N, n_agents, 1]
    #         available_actions_batch = torch.stack(available_actions_batch, dim=1)  # [L, N, n_agents, action_space]
            
    #         # Stack RNN states into [batch_size, n_agents, hidden_size]
    #         actor_rnn_init_states_batch = torch.stack(actor_rnn_init_states_batch, dim=0)  # [N, n_agents, hidden_size]
    #         critic_rnn_init_states_batch = torch.stack(critic_rnn_init_states_batch, dim=0)  # [N, n_agents, hidden_size]
            
    #         # Yield the minibatch as a tuple (similar to recurrent_generator)
    #         yield (obs_batch, global_state_batch, actor_rnn_init_states_batch, critic_rnn_init_states_batch,
    #            actions_batch, values_batch, returns_batch, masks_batch, action_log_probs_batch,
    #            advantages_batch, available_actions_batch)