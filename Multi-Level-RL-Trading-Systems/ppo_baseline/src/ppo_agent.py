"""
PPO Agent Implementation for Trading System

This module implements a Proximal Policy Optimization (PPO) agent
specifically designed for cryptocurrency trading with multimodal inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO algorithm
    
    Actor: Policy network that outputs action probabilities
    Critic: Value network that estimates state values
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [128, 64]):
        super(ActorCritic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        
        # Shared feature extraction layers
        self.shared_layers = nn.ModuleList()
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            self.shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        # Actor (Policy) head
        self.actor_head = nn.Linear(prev_dim, action_dim)
        
        # Critic (Value) head  
        self.critic_head = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights using Xavier initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            
        Returns:
            action_probs: Action probabilities [batch_size, action_dim]
            state_value: State value estimate [batch_size, 1]
        """
        # Shared feature extraction
        features = state
        for layer in self.shared_layers:
            features = F.relu(layer(features))
        
        # Actor head - action probabilities
        action_logits = self.actor_head(features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Critic head - state value
        state_value = self.critic_head(features)
        
        return action_probs, state_value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, float, float]:
        """
        Get action from current state
        
        Args:
            state: Current state tensor [state_dim]
            deterministic: Whether to use deterministic action selection
            
        Returns:
            action: Selected action index
            action_prob: Probability of selected action
            state_value: Estimated state value
        """
        with torch.no_grad():
            state = state.unsqueeze(0)  # Add batch dimension
            action_probs, state_value = self.forward(state)
            
            if deterministic:
                action = torch.argmax(action_probs, dim=1).item()
            else:
                # Sample action from probability distribution
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample().item()
            
            action_prob = action_probs[0, action].item()
            state_value = state_value[0, 0].item()
            
            return action, action_prob, state_value

class PPOMemory:
    """
    Memory buffer for storing PPO training experiences
    """
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.action_probs = []
        self.state_values = []
        self.dones = []
        self.advantages = []
        self.returns = []
        
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, action_prob: float, state_value: float, done: bool):
        """Store experience in memory"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.action_probs.append(action_prob)
        self.state_values.append(state_value)
        self.dones.append(done)
    
    def compute_advantages(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Compute Generalized Advantage Estimation (GAE)"""
        advantages = []
        returns = []
        
        # Compute returns and advantages in reverse order
        next_value = 0
        next_advantage = 0
        
        for i in reversed(range(len(self.rewards))):
            if self.dones[i]:
                next_value = 0
                next_advantage = 0
            
            # Compute TD error
            delta = self.rewards[i] + gamma * next_value - self.state_values[i]
            
            # Compute advantage using GAE
            advantage = delta + gamma * gae_lambda * next_advantage
            
            # Compute return
            return_val = advantage + self.state_values[i]
            
            advantages.insert(0, advantage)
            returns.insert(0, return_val)
            
            next_value = self.state_values[i]
            next_advantage = advantage
        
        self.advantages = advantages
        self.returns = returns
    
    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of experiences for training"""
        if len(self.states) < batch_size:
            batch_size = len(self.states)
        
        indices = np.random.choice(len(self.states), batch_size, replace=False)
        
        batch = {
            'states': torch.FloatTensor([self.states[i] for i in indices]),
            'actions': torch.LongTensor([self.actions[i] for i in indices]),
            'old_action_probs': torch.FloatTensor([self.action_probs[i] for i in indices]),
            'advantages': torch.FloatTensor([self.advantages[i] for i in indices]),
            'returns': torch.FloatTensor([self.returns[i] for i in indices])
        }
        
        return batch
    
    def clear(self):
        """Clear all stored experiences"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.action_probs.clear()
        self.state_values.clear()
        self.dones.clear()
        self.advantages.clear()
        self.returns.clear()
    
    def __len__(self):
        return len(self.states)

class PPOAgent:
    """
    PPO Agent for cryptocurrency trading
    
    Implements Proximal Policy Optimization algorithm with:
    - Actor-Critic architecture
    - GAE advantage estimation
    - PPO clipping for stable training
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Initialize networks
        self.actor_critic = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=config.get('hidden_layers', [128, 64])
        )
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=config.get('learning_rate', 3e-4)
        )
        
        # Training parameters
        self.gamma = config.get('discount_factor', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.ppo_clip_ratio = config.get('ppo_clip_ratio', 0.2)
        self.value_loss_coef = config.get('value_loss_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        
        # Memory buffer
        self.memory = PPOMemory(capacity=config.get('memory_capacity', 10000))
        
        # Training statistics
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'clip_fraction': []
        }
        
        logger.info(f"PPO Agent initialized with state_dim={state_dim}, action_dim={action_dim}")
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """
        Select action for given state
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic action selection
            
        Returns:
            action: Selected action index
            action_prob: Probability of selected action
            state_value: Estimated state value
        """
        state_tensor = torch.FloatTensor(state)
        return self.actor_critic.get_action(state_tensor, deterministic)
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, action_prob: float, state_value: float, done: bool):
        """Store experience in memory buffer"""
        self.memory.push(state, action, reward, next_state, action_prob, state_value, done)
    
    def update(self, batch_size: int = 64, epochs: int = 4) -> Dict[str, float]:
        """
        Update policy and value networks using PPO
        
        Args:
            batch_size: Size of training batches
            epochs: Number of training epochs per update
            
        Returns:
            Dictionary containing training statistics
        """
        if len(self.memory) < batch_size:
            logger.warning(f"Not enough experiences for training. Need {batch_size}, have {len(self.memory)}")
            return {}
        
        # Compute advantages and returns
        self.memory.compute_advantages(self.gamma, self.gae_lambda)
        
        # Normalize advantages
        advantages = np.array(self.memory.advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.memory.advantages = advantages.tolist()
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_clip_fraction = 0
        
        # Multiple epochs of training
        for epoch in range(epochs):
            batch = self.memory.sample_batch(batch_size)
            
            # Forward pass
            action_probs, state_values = self.actor_critic(batch['states'])
            
            # Get action probabilities for taken actions
            action_probs_taken = action_probs.gather(1, batch['actions'].unsqueeze(1)).squeeze(1)
            
            # Compute probability ratio
            ratio = action_probs_taken / (batch['old_action_probs'] + 1e-8)
            
            # PPO clipping
            clipped_ratio = torch.clamp(ratio, 1 - self.ppo_clip_ratio, 1 + self.ppo_clip_ratio)
            
            # Compute surrogate losses
            surr1 = ratio * batch['advantages']
            surr2 = clipped_ratio * batch['advantages']
            
            # Policy loss
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(state_values.squeeze(), batch['returns'])
            
            # Entropy loss for exploration
            entropy_loss = -torch.mean(torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=1))
            
            # Total loss
            total_loss = (policy_loss + 
                         self.value_loss_coef * value_loss + 
                         self.entropy_coef * entropy_loss)
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            # Statistics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
            
            # Compute clip fraction
            clip_fraction = (abs(ratio - 1) > self.ppo_clip_ratio).float().mean().item()
            total_clip_fraction += clip_fraction
        
        # Average statistics over epochs
        avg_policy_loss = total_policy_loss / epochs
        avg_value_loss = total_value_loss / epochs
        avg_entropy_loss = total_entropy_loss / epochs
        avg_clip_fraction = total_clip_fraction / epochs
        
        # Store statistics
        self.training_stats['policy_loss'].append(avg_policy_loss)
        self.training_stats['value_loss'].append(avg_value_loss)
        self.training_stats['entropy_loss'].append(avg_entropy_loss)
        self.training_stats['clip_fraction'].append(avg_clip_fraction)
        
        # Clear memory after update
        self.memory.clear()
        
        logger.info(f"PPO update completed - Policy Loss: {avg_policy_loss:.4f}, "
                   f"Value Loss: {avg_value_loss:.4f}, Clip Fraction: {avg_clip_fraction:.4f}")
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy_loss': avg_entropy_loss,
            'clip_fraction': avg_clip_fraction
        }
    
    def save_model(self, filepath: str):
        """Save model weights to file"""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_stats': self.training_stats
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model weights from file"""
        checkpoint = torch.load(filepath, map_location='cpu')
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', {})
        logger.info(f"Model loaded from {filepath}")
    
    def get_training_stats(self) -> Dict[str, list]:
        """Get training statistics"""
        return self.training_stats.copy()
