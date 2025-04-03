import numpy as np
import torch

class RolloutStorage:
    """
    Rollout storage for collecting multi-agent experiences during training.
    Designed for MAPPO with n-step returns and MLP-based policies.
    """
    def __init__(self, n_steps, n_agents, obs_dim, action_dim, state_dim, device='cpu'):
        """
        Initialize rollout storage for collecting experiences.  
        
        Args:
            n_steps (int): Number of steps to collect before update (can be different from episode length)
            n_agents (int): Number of agents in the environment
            obs_dim (int): Dimension of individual agent observations
            action_dim (int): Dimension of agent actions
            state_dim (int): Dimension of global state
            device (str): Device for storage ('cpu' for numpy-based implementation)
        """
        self.n_steps = n_steps
        self.n_agents = n_agents
        self.device = torch.device(device)
        
        # Current position in the buffer
        self.step = 0
        # (obs_0, state_0) → action_0 / action_log_prob_0 → (reward_0, obs_1, state_1, mask_1, trunc_1)
        # delta = reward_0 + gamma * value_1 * mask_1 - value_0

        # Core storage buffers - using numpy arrays for efficiency
        self.obs = np.zeros((n_steps + 1, n_agents, obs_dim), dtype=np.float32)
        self.global_state = np.zeros((n_steps + 1, state_dim), dtype=np.float32)
        self.rewards = np.zeros((n_steps, n_agents), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_agents), dtype=np.int64)
        self.action_log_probs = np.zeros((n_steps, n_agents), dtype=np.float32)
        self.values = np.zeros((n_steps + 1, n_agents), dtype=np.float32)
        self.masks = np.ones((n_steps + 1, n_agents), dtype=np.float32) # 0 if episode done, 1 otherwise
        self.truncated = np.zeros((n_steps + 1, n_agents), dtype=np.bool_) # 1 if episode truncated, 0 otherwise
        self.available_actions = np.zeros((n_steps+1, n_agents, action_dim), dtype=np.bool_)
        
        # Extra buffers for the algorithm
        self.returns = np.zeros((n_steps + 1, n_agents), dtype=np.float32)
        self.advantages = np.zeros((n_steps, n_agents), dtype=np.float32)
        
        # Initialize available actions buffer if shape is provided
    
        
    def insert(self, obs, global_state, actions, 
        action_log_probs, values, rewards, 
        masks, truncates, available_actions):
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

        # Reset step counter
        self.step = 0
    
    def get_minibatches(self, num_mini_batch, mini_batch_size=None):
        """
        Create minibatches for training.
        
        Args:
            num_mini_batch (int): Number of minibatches to create
            mini_batch_size (int, optional): Size of each minibatch, if None will be calculated
                                            based on num_mini_batch
        
        Returns:
            Generator yielding minibatches for training
        """
        # Calculate total steps and minibatch size
        total_steps = self.n_steps
        
        if mini_batch_size is None:
            mini_batch_size = total_steps // num_mini_batch
            
        # Pre-reshape data to improve performance (only do this once)
        obs_batch = self.obs[:-1].reshape(-1, self.n_agents, *self.obs.shape[2:])
        global_state_batch = self.global_state[:-1].reshape(-1, *self.global_state.shape[1:])
        actions_batch = self.actions.reshape(-1, self.n_agents, 1) # Add explicit final dimension
        values_batch = self.values[:-1].reshape(-1, self.n_agents, 1) # Add explicit final dimension
        returns_batch = self.returns[:-1].reshape(-1, self.n_agents, 1)# Add explicit final dimension
        masks_batch = self.masks[:-1].reshape(-1, self.n_agents, 1) # Add explicit final dimension
        old_action_log_probs_batch = self.action_log_probs.reshape(-1, self.n_agents, 1)
        advantages_batch = self.advantages.reshape(-1, self.n_agents, 1) # Add explicit final dimension
        available_actions_batch = self.available_actions[:-1].reshape(-1, self.n_agents, 
                                                                        *self.available_actions.shape[2:])
        
        # Create random indices for minibatches
        batch_inds = np.random.permutation(total_steps)
        
        # Yield minibatches
        start_ind = 0
        for _ in range(num_mini_batch):
            end_ind = min(start_ind + mini_batch_size, total_steps)
            if end_ind - start_ind < 1:  # Skip empty batches
                continue
                
            batch_inds_subset = batch_inds[start_ind:end_ind]
            
            # Yield the minibatch as a tuple
            yield (
                torch.tensor(obs_batch[batch_inds_subset], dtype=torch.float32).to(self.device),
                torch.tensor(global_state_batch[batch_inds_subset], dtype=torch.float32).to(self.device),
                torch.tensor(actions_batch[batch_inds_subset], dtype=torch.int64).to(self.device),
                torch.tensor(values_batch[batch_inds_subset], dtype=torch.float32).to(self.device),
                torch.tensor(returns_batch[batch_inds_subset], dtype=torch.float32).to(self.device),
                torch.tensor(masks_batch[batch_inds_subset], dtype=torch.float32).to(self.device),
                torch.tensor(old_action_log_probs_batch[batch_inds_subset], dtype=torch.float32).to(self.device),
                torch.tensor(advantages_batch[batch_inds_subset], dtype=torch.float32).to(self.device),
                torch.tensor(available_actions_batch[batch_inds_subset], dtype=torch.bool).to(self.device) 
            )
            
            start_ind = end_ind
    
            
    def reset(self):
        """Reset the buffer counters but keep the latest observation."""
        self.step = 0 