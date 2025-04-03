import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from networks.mlp_nets import Actor_MLP, Critic_MLP
from utils.scheduler import LinearScheduler
from utils.value_normalizers import create_value_normalizer
from typing import Optional

class MAPPOAgent:
    """
    Multi-Agent Proximal Policy Optimization (MAPPO) agent implementation.

    This agent implements the MAPPO algorithm for multi-agent reinforcement learning
    with centralized training and decentralized execution.
    """
    def __init__(self, args, obs_dim, state_dim, action_dim, device=torch.device("cpu")):
        """
        Initialize the MAPPO agent.

        Args:
            args: Arguments containing training hyperparameters
            obs_dim: Observation dimension for individual agents
            state_dim: Centralized observation dimension for critic
            action_dim: Action dimension
            device (torch.device): Device to run the agent on
        """
        # Input validation
        self._validate_inputs(args, obs_dim, state_dim, action_dim)

        self.args = args
        self.device = device

         # Initialize core components
        self._init_hyperparameters()
        self._init_networks(obs_dim, state_dim, action_dim)

        if self.use_value_normalization:
            self.value_normalizer = create_value_normalizer(
                normalizer_type=self.args.value_norm_type,
                device=device
            )

        # Setup loss functions
        # if self.use_huber_loss:
        #     self.huber_loss = nn.HuberLoss(reduction="none", delta=args.huber_delta)

    def _validate_inputs(self, args, obs_dim: int, state_dim: int, action_dim: int) -> None:
        """Validate input parameters."""
        if obs_dim <= 0 or state_dim <= 0 or action_dim <= 0:
            raise ValueError("Dimensions must be positive integers")
        required_attrs = ['n_agents', 'lr', 'clip_param', 'ppo_epoch',
                         'num_mini_batch']
        missing = [attr for attr in required_attrs if not hasattr(args, attr)]
        if missing:
            raise AttributeError(f"args missing required attributes: {missing}")

    def _init_hyperparameters(self) -> None:
        """Initialize training hyperparameters."""
        self.clip_param = self.args.clip_param
        self.ppo_epoch = self.args.ppo_epoch
        self.num_mini_batch = self.args.num_mini_batch
        self.entropy_coef = self.args.entropy_coef
        self.max_grad_norm = self.args.max_grad_norm
        self.use_max_grad_norm = self.args.use_max_grad_norm
        self.use_clipped_value_loss = self.args.use_clipped_value_loss
        self.use_value_normalization = self.args.use_value_norm
        # self.target_kl = self.args.target_kl

        # Training parameters
        self.lr = self.args.lr
        self.gamma = self.args.gamma
        self.use_gae = self.args.use_gae
        self.gae_lambda = self.args.gae_lambda

    def _init_networks(self, obs_dim: int, state_dim: int, action_dim: int) -> None:
        """Initialize actor and critic networks with proper weight initialization."""
        actor_input_dim = obs_dim
        critic_input_dim = state_dim + obs_dim

        self.actor = Actor_MLP(
            actor_input_dim,
            action_dim,
            self.args.hidden_size,
            use_feature_normalization=self.args.use_feature_normalization,
            output_gain=self.args.actor_gain
        ).to(self.device)

        self.critic = Critic_MLP(
            critic_input_dim,
            self.args.hidden_size,
            use_feature_normalization=self.args.use_feature_normalization
        ).to(self.device)

        # Use a slightly higher learning rate for RNN to speed up learning
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.args.optimizer_eps
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.lr,
            eps=self.args.optimizer_eps
        )

        if self.args.use_linear_lr_decay:
            self.scheduler = LinearScheduler(
                self.lr,
                self.args.min_lr,
                self.args.max_steps
            )

    def _clip_gradients(self, network: nn.Module) -> Optional[float]:
        """Clip gradients and return gradient norm."""
        if self.use_max_grad_norm:
            return nn.utils.clip_grad_norm_(
                network.parameters(),
                self.max_grad_norm
            )
        return None

    def get_actions(
            self,
            obs: np.ndarray,
            available_actions: np.ndarray,
            deterministic: bool = False):
        """
        Get actions from the policy network.

        Args:
            obs (np.ndarray): Observation tensor
            available_actions (np.ndarray): Available actions tensor
            deterministic (bool): Whether to use deterministic actions

        Returns:
            tuple: (actions, action_log_probs)
        """
        with torch.no_grad():
            #Convert to tensors
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device) # (n_agents, n_obs)
            available_actions = torch.tensor(available_actions, dtype=torch.float32).to(self.device) # (n_agents, n_actions)

            # Process each agent's observations
            actions, action_log_probs = self.actor.get_actions(
                obs, available_actions, deterministic)  # (n_agents, 1), (n_agents, 1)

            actions = actions.cpu().numpy()
            action_log_probs = action_log_probs.cpu().numpy() if not deterministic else None

        return actions, action_log_probs

    def get_values(self, state, obs):
        """
        Get values from the critic network.

        Args:
            state (np.ndarray): State tensor
            obs (np.ndarray): Observation tensor

        Returns:
            values (np.ndarray): Values
        """
        with torch.no_grad():
            # Convert numpy arrays to torch tensors and move to device
            state = torch.tensor(state, dtype=torch.float32).to(self.device) # (n_state)
            state = state.unsqueeze(0) # (1, n_state)
            state = state.repeat(self.args.n_agents, 1) # (n_agents, n_state)
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device) #(n_agents, n_obs)

            critic_input = torch.cat((state, obs), dim=-1)

            values = self.critic(critic_input) # (n_agents, )
            values = values.cpu().numpy()

        return values

    def evaluate_actions(self, state, obs, actions, available_actions):
        """
        Evaluate actions for training.

        Args:
            state (torch.Tensor): State tensor #(batch_size, n_state)
            obs (torch.Tensor): Observation tensor #(batch_size, n_agents, n_obs)
            actions (torch.Tensor): Actions tensor #(batch_size, n_agents, )
            available_actions (torch.Tensor): Available actions tensor #(batch_size, n_agents, n_actions)

        Returns:
            tuple: (values, action_log_probs, dist_entropy)
        """
        # Convert numpy arrays to torch tensors and move to device
        state = state.unsqueeze(1) # (batch_size, 1, n_state)
        state = state.repeat(1, self.args.n_agents, 1) # (batch_size, n_agents, n_state)
        critic_input = torch.cat((state, obs), dim=-1) # (batch_size, n_agents, n_state + n_obs)


        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs, actions, available_actions)
        values = self.critic(critic_input) # (batch_size, n_agents, 1)

        return values, action_log_probs, dist_entropy

    def compute_value_loss(self, values, value_preds_batch, returns_batch):
        """
        Compute value function loss with normalization.

        Args:
            values: Current value predictions
            value_preds_batch: Old value predictions
            return_batch: Return targets
        """
        if self.use_value_normalization:
            # Normalize returns for loss computation
            returns = self.value_normalizer.normalize(returns_batch)
        else:
            returns = returns_batch

        if self.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + torch.clamp(
                values - value_preds_batch,
                -self.clip_param,
                self.clip_param
            )
            value_losses = (values - returns).pow(2)
            value_losses_clipped = (value_pred_clipped - returns).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = 0.5 * (values - returns).pow(2).mean()

        return value_loss

    def update(self, mini_batch):
        """
        Update policy using a mini-batch of experiences.

        Args:
            mini_batch (dict): Dictionary containing mini-batch data

        Returns:
            tuple: (value_loss, policy_loss, dist_entropy)
        """
        metrics = {}
        # Extract data from mini-batch
        (obs_batch, global_state_batch, actions_batch, values_batch, returns_batch, masks_batch,
            old_action_log_probs_batch, advantages_batch, available_actions_batch) = mini_batch

        # Evaluate actions
        values, action_log_probs, dist_entropy = self.evaluate_actions(
            global_state_batch, obs_batch, actions_batch, available_actions_batch)

        #  Calculate PPO ratio and KL divergence
        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
        clip_ratio = (torch.abs(ratio - 1) > self.clip_param).float().mean().item()

        # Actor Loss
        surr1 = ratio * advantages_batch
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
        policy_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = -self.entropy_coef * torch.mean(dist_entropy)
        actor_loss = policy_loss + entropy_loss

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = self._clip_gradients(self.actor)
        self.actor_optimizer.step()

        # Critic loss
        critic_loss = self.compute_value_loss(values, values_batch, returns_batch)

        # Perform optimization step
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = self._clip_gradients(self.critic)
        self.critic_optimizer.step()

        # Update metrics
        metrics.update({
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'approx_kl': approx_kl,
            'clip_ratio': clip_ratio,
            'actor_grad_norm': actor_grad_norm,
            'critic_grad_norm': critic_grad_norm
        })

        return  metrics

    def train(self, buffer):
        """
        Train the policy using experiences from the buffer.
        """
        train_info = {
            'critic_loss': 0,
            'actor_loss': 0,
            'entropy_loss': 0,
            'approx_kl': 0,
            'clip_ratio': 0,
            'actor_grad_norm': 0,
            'critic_grad_norm': 0,
        }

        # Train for ppo_epoch iterations
        for _ in range(self.ppo_epoch):

            # Generate mini-batches
            mini_batches = buffer.get_minibatches(self.num_mini_batch)

            # Update for each mini-batch
            for mini_batch in mini_batches:
                metrics = self.update(mini_batch)

                # Update training info
                for k, v in metrics.items():
                    if k in train_info:
                        train_info[k] += v

        # Calculate mean values
        num_updates = self.ppo_epoch * self.num_mini_batch
        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def update_learning_rate(self, current_step):
        """Update the learning rate based on the current step."""
        lr_now = self.scheduler.get_lr(current_step)
        for p in self.actor_optimizer.param_groups:
            p['lr'] = lr_now
        for p in self.critic_optimizer.param_groups:
            p['lr'] = lr_now

        return {
            'actor_lr': self.actor_optimizer.param_groups[0]['lr'],
            'critic_lr': self.critic_optimizer.param_groups[0]['lr']
        }

    def save(self, save_path):
        """
        Save the model checkpoint.

        Args:
            save_path (str): Path to save the model
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'args': self.args,
        }, save_path)

    def load(self, model_path):
        """
        Load a model checkpoint.

        Args:
            model_path (str): Path to the model checkpoint
        """
        checkpoint = torch.load(model_path, map_location=self.device)

        # Load network states
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])

        # Load optimizer states
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
