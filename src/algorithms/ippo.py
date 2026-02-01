"""Independent PPO (IPPO) implementation for multi-agent reinforcement learning.

IPPO treats each agent independently, training separate PPO policies
for each agent without sharing information during training.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_device


class MultiAgentFeatureExtractor(BaseFeaturesExtractor):
    """Feature extractor for multi-agent observations."""
    
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 64,
        hidden_dims: List[int] = [128, 64],
    ):
        super().__init__(observation_space, features_dim)
        
        layers = []
        input_dim = observation_space.shape[0]
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, features_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


class IPPOTrainer:
    """Independent PPO trainer for multi-agent environments.
    
    Each agent is trained with its own PPO policy independently.
    """
    
    def __init__(
        self,
        env: Any,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: Optional[Union[str, torch.device]] = None,
        seed: Optional[int] = None,
    ):
        self.env = env
        self.device = get_device(device)
        
        # Set seeds for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        # Create PPO policies for each agent
        self.policies = {}
        self.agents = env.possible_agents
        
        for agent in self.agents:
            policy_kwargs = {
                "features_extractor_class": MultiAgentFeatureExtractor,
                "features_extractor_kwargs": {"features_dim": 64},
                "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
            }
            
            self.policies[agent] = PPO(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                device=self.device,
                policy_kwargs=policy_kwargs,
                verbose=0,
            )
    
    def train(self, total_timesteps: int) -> Dict[str, Any]:
        """Train all agents independently."""
        training_info = {}
        
        for agent in self.agents:
            print(f"Training agent {agent}...")
            self.policies[agent].learn(total_timesteps=total_timesteps)
            training_info[agent] = {
                "timesteps": total_timesteps,
                "learning_rate": self.policies[agent].learning_rate,
            }
        
        return training_info
    
    def predict(
        self, observations: Dict[str, np.ndarray], deterministic: bool = True
    ) -> Tuple[Dict[str, int], Dict[str, Any]]:
        """Predict actions for all agents."""
        actions = {}
        infos = {}
        
        for agent in self.agents:
            if agent in observations:
                action, info = self.policies[agent].predict(
                    observations[agent], deterministic=deterministic
                )
                actions[agent] = action
                infos[agent] = info
        
        return actions, infos
    
    def save(self, path: str) -> None:
        """Save all policies."""
        for agent in self.agents:
            self.policies[agent].save(f"{path}_{agent}")
    
    def load(self, path: str) -> None:
        """Load all policies."""
        for agent in self.agents:
            self.policies[agent] = PPO.load(f"{path}_{agent}")
    
    def evaluate(
        self, eval_env: Any, n_eval_episodes: int = 10
    ) -> Dict[str, float]:
        """Evaluate the trained policies."""
        episode_rewards = {agent: [] for agent in self.agents}
        episode_lengths = []
        
        for episode in range(n_eval_episodes):
            obs, _ = eval_env.reset()
            episode_reward = {agent: 0.0 for agent in self.agents}
            episode_length = 0
            
            done = False
            while not done:
                actions, _ = self.predict(obs, deterministic=True)
                obs, rewards, terminations, truncations, _ = eval_env.step(actions)
                
                for agent in self.agents:
                    if agent in rewards:
                        episode_reward[agent] += rewards[agent]
                
                episode_length += 1
                done = any(terminations.values()) or any(truncations.values())
            
            for agent in self.agents:
                episode_rewards[agent].append(episode_reward[agent])
            episode_lengths.append(episode_length)
        
        # Calculate statistics
        eval_results = {}
        for agent in self.agents:
            rewards = episode_rewards[agent]
            eval_results[f"{agent}_mean_reward"] = np.mean(rewards)
            eval_results[f"{agent}_std_reward"] = np.std(rewards)
            eval_results[f"{agent}_min_reward"] = np.min(rewards)
            eval_results[f"{agent}_max_reward"] = np.max(rewards)
        
        eval_results["mean_episode_length"] = np.mean(episode_lengths)
        eval_results["std_episode_length"] = np.std(episode_lengths)
        
        return eval_results
