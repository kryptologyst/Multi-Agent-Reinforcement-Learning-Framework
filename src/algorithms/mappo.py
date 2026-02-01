"""Multi-Agent PPO (MAPPO) implementation.

MAPPO extends PPO to multi-agent settings by sharing value functions
and using centralized training with decentralized execution (CTDE).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import gymnasium as gym


@dataclass
class MAPPOConfig:
    """Configuration for MAPPO training."""
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_centralized_value: bool = True
    use_gae: bool = True
    normalize_advantages: bool = True


class CentralizedCritic(nn.Module):
    """Centralized critic that takes global state information."""
    
    def __init__(
        self,
        global_obs_dim: int,
        hidden_dims: List[int] = [256, 128],
        activation: str = "relu",
    ):
        super().__init__()
        
        self.global_obs_dim = global_obs_dim
        
        if activation == "relu":
            activation_fn = nn.ReLU
        elif activation == "tanh":
            activation_fn = nn.Tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        layers = []
        input_dim = global_obs_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                activation_fn(),
                nn.LayerNorm(hidden_dim),
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.network(global_obs)


class ActorCritic(nn.Module):
    """Actor-critic network for individual agents."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 64],
        activation: str = "relu",
    ):
        super().__init__()
        
        if activation == "relu":
            activation_fn = nn.ReLU
        elif activation == "tanh":
            activation_fn = nn.Tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Actor network
        actor_layers = []
        input_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            actor_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                activation_fn(),
                nn.LayerNorm(hidden_dim),
            ])
            input_dim = hidden_dim
        
        actor_layers.append(nn.Linear(input_dim, action_dim))
        self.actor = nn.Sequential(*actor_layers)
        
        # Critic network (for decentralized value function)
        critic_layers = []
        input_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            critic_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                activation_fn(),
                nn.LayerNorm(hidden_dim),
            ])
            input_dim = hidden_dim
        
        critic_layers.append(nn.Linear(input_dim, 1))
        self.critic = nn.Sequential(*critic_layers)
        
    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        """Forward pass returning action distribution and value."""
        logits = self.actor(obs)
        value = self.critic(obs)
        
        dist = Categorical(logits=logits)
        return dist, value.squeeze(-1)
    
    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log probability, and value."""
        dist, value = self.forward(obs)
        
        if deterministic:
            action = torch.argmax(dist.logits, dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        return action, log_prob, value


class MAPPOTrainer:
    """Multi-Agent PPO trainer with centralized value function."""
    
    def __init__(
        self,
        env: Any,
        config: MAPPOConfig,
        device: Optional[Union[str, torch.device]] = None,
        seed: Optional[int] = None,
    ):
        self.env = env
        self.config = config
        self.device = device or torch.device("cpu")
        
        # Set seeds for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.agents = env.possible_agents
        self.num_agents = len(self.agents)
        
        # Create actor-critic networks for each agent
        self.actor_critics = {}
        for agent in self.agents:
            obs_space = env.observation_spaces[agent]
            action_space = env.action_spaces[agent]
            
            self.actor_critics[agent] = ActorCritic(
                obs_dim=obs_space.shape[0],
                action_dim=action_space.n,
            ).to(self.device)
        
        # Create centralized critic if enabled
        if config.use_centralized_value:
            global_obs_dim = sum(
                env.observation_spaces[agent].shape[0] for agent in self.agents
            )
            self.centralized_critic = CentralizedCritic(global_obs_dim).to(self.device)
        else:
            self.centralized_critic = None
        
        # Optimizers
        self.actor_optimizers = {
            agent: torch.optim.Adam(
                self.actor_critics[agent].parameters(),
                lr=config.learning_rate,
            )
            for agent in self.agents
        }
        
        if self.centralized_critic:
            self.critic_optimizer = torch.optim.Adam(
                self.centralized_critic.parameters(),
                lr=config.learning_rate,
            )
        
        # Storage for rollouts
        self.reset_storage()
    
    def reset_storage(self) -> None:
        """Reset rollout storage."""
        self.observations = {agent: [] for agent in self.agents}
        self.actions = {agent: [] for agent in self.agents}
        self.rewards = {agent: [] for agent in self.agents}
        self.values = {agent: [] for agent in self.agents}
        self.log_probs = {agent: [] for agent in self.agents}
        self.dones = []
        self.global_observations = []
    
    def collect_rollouts(self) -> None:
        """Collect rollout data."""
        self.reset_storage()
        
        obs, _ = self.env.reset()
        done = False
        step = 0
        
        while step < self.config.n_steps and not done:
            # Store observations
            for agent in self.agents:
                self.observations[agent].append(obs[agent])
            
            # Get global observation for centralized critic
            if self.centralized_critic:
                global_obs = np.concatenate([obs[agent] for agent in self.agents])
                self.global_observations.append(global_obs)
            
            # Get actions from all agents
            actions = {}
            for agent in self.agents:
                obs_tensor = torch.FloatTensor(obs[agent]).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action, log_prob, value = self.actor_critics[agent].get_action(
                        obs_tensor, deterministic=False
                    )
                
                actions[agent] = action.item()
                self.actions[agent].append(action.item())
                self.log_probs[agent].append(log_prob.item())
                self.values[agent].append(value.item())
            
            # Step environment
            next_obs, rewards, terminations, truncations, _ = self.env.step(actions)
            
            # Store rewards and done
            for agent in self.agents:
                self.rewards[agent].append(rewards[agent])
            
            done = any(terminations.values()) or any(truncations.values())
            self.dones.append(done)
            
            obs = next_obs
            step += 1
    
    def compute_advantages(self) -> Dict[str, torch.Tensor]:
        """Compute advantages using GAE."""
        advantages = {agent: [] for agent in self.agents}
        
        for agent in self.agents:
            rewards = torch.FloatTensor(self.rewards[agent])
            values = torch.FloatTensor(self.values[agent])
            dones = torch.BoolTensor(self.dones)
            
            # Compute returns and advantages
            returns = []
            advantages_agent = []
            
            next_value = 0.0
            next_advantage = 0.0
            
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_non_terminal = 1.0 - dones[t].float()
                    next_value = next_value
                else:
                    next_non_terminal = 1.0 - dones[t].float()
                    next_value = values[t + 1]
                
                delta = rewards[t] + self.config.gamma * next_value * next_non_terminal - values[t]
                next_advantage = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * next_advantage
                
                advantages_agent.insert(0, next_advantage)
                returns.insert(0, next_advantage + values[t])
            
            advantages[agent] = torch.FloatTensor(advantages_agent)
        
        return advantages
    
    def update_policies(self) -> Dict[str, float]:
        """Update all policies using PPO."""
        advantages = self.compute_advantages()
        
        # Normalize advantages
        if self.config.normalize_advantages:
            for agent in self.agents:
                advantages[agent] = (advantages[agent] - advantages[agent].mean()) / (
                    advantages[agent].std() + 1e-8
                )
        
        losses = {}
        
        for agent in self.agents:
            obs = torch.FloatTensor(self.observations[agent]).to(self.device)
            actions = torch.LongTensor(self.actions[agent]).to(self.device)
            old_log_probs = torch.FloatTensor(self.log_probs[agent]).to(self.device)
            advantages_agent = advantages[agent].to(self.device)
            
            # Compute returns
            returns = advantages_agent + torch.FloatTensor(self.values[agent]).to(self.device)
            
            # PPO update
            for epoch in range(self.config.n_epochs):
                # Get current policy outputs
                dist, values = self.actor_critics[agent](obs)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                
                # Compute ratios
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # Compute surrogate losses
                surr1 = ratio * advantages_agent
                surr2 = torch.clamp(
                    ratio, 1 - self.config.clip_range, 1 + self.config.clip_range
                ) * advantages_agent
                
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value function loss
                value_loss = F.mse_loss(values, returns)
                
                # Total loss
                total_loss = (
                    actor_loss
                    - self.config.ent_coef * entropy
                    + self.config.vf_coef * value_loss
                )
                
                # Update
                self.actor_optimizers[agent].zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor_critics[agent].parameters(), self.config.max_grad_norm
                )
                self.actor_optimizers[agent].step()
            
            losses[agent] = total_loss.item()
        
        return losses
    
    def train(self, total_timesteps: int) -> Dict[str, Any]:
        """Train the MAPPO agents."""
        training_info = {"episode_rewards": [], "losses": []}
        
        timesteps = 0
        episode = 0
        
        while timesteps < total_timesteps:
            # Collect rollouts
            self.collect_rollouts()
            
            # Update policies
            losses = self.update_policies()
            
            # Log training info
            episode_reward = sum(
                sum(self.rewards[agent]) for agent in self.agents
            )
            training_info["episode_rewards"].append(episode_reward)
            training_info["losses"].append(losses)
            
            timesteps += len(self.rewards[self.agents[0]])
            episode += 1
            
            if episode % 10 == 0:
                print(f"Episode {episode}, Timesteps {timesteps}, Reward {episode_reward:.2f}")
        
        return training_info
    
    def predict(
        self, observations: Dict[str, np.ndarray], deterministic: bool = True
    ) -> Tuple[Dict[str, int], Dict[str, Any]]:
        """Predict actions for all agents."""
        actions = {}
        infos = {}
        
        for agent in self.agents:
            if agent in observations:
                obs_tensor = torch.FloatTensor(observations[agent]).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action, _, value = self.actor_critics[agent].get_action(
                        obs_tensor, deterministic=deterministic
                    )
                
                actions[agent] = action.item()
                infos[agent] = {"value": value.item()}
        
        return actions, infos
    
    def save(self, path: str) -> None:
        """Save all models."""
        for agent in self.agents:
            torch.save(
                self.actor_critics[agent].state_dict(),
                f"{path}_{agent}_actor_critic.pth"
            )
        
        if self.centralized_critic:
            torch.save(
                self.centralized_critic.state_dict(),
                f"{path}_centralized_critic.pth"
            )
    
    def load(self, path: str) -> None:
        """Load all models."""
        for agent in self.agents:
            self.actor_critics[agent].load_state_dict(
                torch.load(f"{path}_{agent}_actor_critic.pth", map_location=self.device)
            )
        
        if self.centralized_critic:
            self.centralized_critic.load_state_dict(
                torch.load(f"{path}_centralized_critic.pth", map_location=self.device)
            )
