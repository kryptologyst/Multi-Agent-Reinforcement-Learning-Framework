"""Modern Multi-Agent Grid World Environment using PettingZoo.

This module provides a cooperative multi-agent environment where agents must
coordinate to reach goals while avoiding collisions and maximizing efficiency.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers


class MultiAgentGridWorld(ParallelEnv):
    """A cooperative multi-agent grid world environment.
    
    Agents must coordinate to reach goals while avoiding collisions.
    This environment supports both cooperative and competitive scenarios.
    
    Args:
        grid_size: Size of the square grid (default: 8)
        num_agents: Number of agents (default: 2)
        num_goals: Number of goals to reach (default: 2)
        max_steps: Maximum steps per episode (default: 100)
        collision_penalty: Penalty for agent collisions (default: -0.1)
        goal_reward: Reward for reaching a goal (default: 1.0)
        step_penalty: Small penalty per step to encourage efficiency (default: -0.01)
        seed: Random seed for reproducibility
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "name": "multi_agent_grid_world_v1"}
    
    def __init__(
        self,
        grid_size: int = 8,
        num_agents: int = 2,
        num_goals: int = 2,
        max_steps: int = 100,
        collision_penalty: float = -0.1,
        goal_reward: float = 1.0,
        step_penalty: float = -0.01,
        seed: Optional[int] = None,
    ):
        super().__init__()
        
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_goals = num_goals
        self.max_steps = max_steps
        self.collision_penalty = collision_penalty
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        
        # Agent and goal positions
        self.agent_positions: List[np.ndarray] = []
        self.goal_positions: List[np.ndarray] = []
        self.goals_reached: List[bool] = []
        
        # Episode tracking
        self.step_count = 0
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        
        # Action space: 0=up, 1=down, 2=left, 3=right, 4=stay
        self.action_spaces = {
            agent: spaces.Discrete(5) for agent in self.possible_agents
        }
        
        # Observation space: [agent_x, agent_y, goal_x, goal_y, other_agents_positions...]
        obs_size = 2 + 2 * num_goals + 2 * (num_agents - 1)
        self.observation_spaces = {
            agent: spaces.Box(
                low=0, high=grid_size - 1, shape=(obs_size,), dtype=np.float32
            )
            for agent in self.possible_agents
        }
        
        # Initialize random state
        self.np_random = np.random.RandomState(seed)
        
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        """Reset the environment to initial state."""
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
            
        # Reset episode tracking
        self.step_count = 0
        self.agents = self.possible_agents.copy()
        
        # Place agents at random positions (avoiding goals)
        self.agent_positions = []
        for _ in range(self.num_agents):
            while True:
                pos = self.np_random.randint(0, self.grid_size, size=2)
                if len(self.agent_positions) == 0 or not any(
                    np.array_equal(pos, existing) for existing in self.agent_positions
                ):
                    self.agent_positions.append(pos)
                    break
        
        # Place goals at random positions (avoiding agents)
        self.goal_positions = []
        for _ in range(self.num_goals):
            while True:
                pos = self.np_random.randint(0, self.grid_size, size=2)
                if not any(
                    np.array_equal(pos, existing) for existing in self.agent_positions
                ) and not any(
                    np.array_equal(pos, existing) for existing in self.goal_positions
                ):
                    self.goal_positions.append(pos)
                    break
        
        self.goals_reached = [False] * self.num_goals
        
        observations = self._get_observations()
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        """Execute one step in the environment."""
        self.step_count += 1
        
        # Store old positions for collision detection
        old_positions = [pos.copy() for pos in self.agent_positions]
        
        # Move agents based on actions
        for i, agent in enumerate(self.agents):
            action = actions[agent]
            if action == 0:  # Up
                self.agent_positions[i][1] = min(
                    self.agent_positions[i][1] + 1, self.grid_size - 1
                )
            elif action == 1:  # Down
                self.agent_positions[i][1] = max(
                    self.agent_positions[i][1] - 1, 0
                )
            elif action == 2:  # Left
                self.agent_positions[i][0] = max(
                    self.agent_positions[i][0] - 1, 0
                )
            elif action == 3:  # Right
                self.agent_positions[i][0] = min(
                    self.agent_positions[i][0] + 1, self.grid_size - 1
                )
            # Action 4 is stay (no movement)
        
        # Calculate rewards
        rewards = {agent: self.step_penalty for agent in self.agents}
        
        # Check for goal reaching
        for i, goal_pos in enumerate(self.goal_positions):
            if not self.goals_reached[i]:
                for j, agent_pos in enumerate(self.agent_positions):
                    if np.array_equal(agent_pos, goal_pos):
                        self.goals_reached[i] = True
                        rewards[self.agents[j]] += self.goal_reward
        
        # Check for collisions
        for i in range(len(self.agent_positions)):
            for j in range(i + 1, len(self.agent_positions)):
                if np.array_equal(self.agent_positions[i], self.agent_positions[j]):
                    rewards[self.agents[i]] += self.collision_penalty
                    rewards[self.agents[j]] += self.collision_penalty
        
        # Check termination conditions
        all_goals_reached = all(self.goals_reached)
        max_steps_reached = self.step_count >= self.max_steps
        
        terminations = {agent: all_goals_reached or max_steps_reached for agent in self.agents}
        truncations = {agent: max_steps_reached for agent in self.agents}
        
        # Remove terminated agents
        self.agents = [agent for agent in self.agents if not terminations[agent]]
        
        observations = self._get_observations()
        infos = {agent: {"goals_reached": sum(self.goals_reached)} for agent in self.agents}
        
        return observations, rewards, terminations, truncations, infos
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all agents."""
        observations = {}
        
        for i, agent in enumerate(self.agents):
            obs = []
            
            # Agent's own position
            obs.extend(self.agent_positions[i])
            
            # Goal positions
            for goal_pos in self.goal_positions:
                obs.extend(goal_pos)
            
            # Other agents' positions
            for j, other_pos in enumerate(self.agent_positions):
                if i != j:
                    obs.extend(other_pos)
            
            # Pad with zeros if needed
            while len(obs) < self.observation_spaces[agent].shape[0]:
                obs.append(0.0)
            
            observations[agent] = np.array(obs, dtype=np.float32)
        
        return observations
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment."""
        if mode == "human":
            print(f"Step {self.step_count}")
            print(f"Goals reached: {sum(self.goals_reached)}/{self.num_goals}")
            
            # Create grid visualization
            grid = np.full((self.grid_size, self.grid_size), ".")
            
            # Place goals
            for i, goal_pos in enumerate(self.goal_positions):
                if not self.goals_reached[i]:
                    grid[goal_pos[1], goal_pos[0]] = "G"
            
            # Place agents
            for i, agent_pos in enumerate(self.agent_positions):
                grid[agent_pos[1], agent_pos[0]] = str(i)
            
            # Print grid
            for row in reversed(grid):  # Reverse to show (0,0) at bottom-left
                print(" ".join(row))
            print()
            
        elif mode == "rgb_array":
            # Create RGB array representation
            rgb_array = np.zeros((self.grid_size * 32, self.grid_size * 32, 3), dtype=np.uint8)
            
            # Draw grid lines
            for i in range(self.grid_size + 1):
                rgb_array[i * 32 : i * 32 + 1, :] = [128, 128, 128]
                rgb_array[:, i * 32 : i * 32 + 1] = [128, 128, 128]
            
            # Draw goals
            for i, goal_pos in enumerate(self.goal_positions):
                if not self.goals_reached[i]:
                    y, x = goal_pos[1] * 32 + 2, goal_pos[0] * 32 + 2
                    rgb_array[y : y + 28, x : x + 28] = [0, 255, 0]  # Green
            
            # Draw agents
            colors = [[255, 0, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255]]  # Red, Blue, Yellow, Magenta
            for i, agent_pos in enumerate(self.agent_positions):
                y, x = agent_pos[1] * 32 + 4, agent_pos[0] * 32 + 4
                color = colors[i % len(colors)]
                rgb_array[y : y + 24, x : x + 24] = color
            
            return rgb_array
        
        return None
    
    def close(self) -> None:
        """Close the environment."""
        pass


def make_env(
    grid_size: int = 8,
    num_agents: int = 2,
    num_goals: int = 2,
    max_steps: int = 100,
    seed: Optional[int] = None,
) -> ParallelEnv:
    """Create a MultiAgentGridWorld environment.
    
    Args:
        grid_size: Size of the square grid
        num_agents: Number of agents
        num_goals: Number of goals to reach
        max_steps: Maximum steps per episode
        seed: Random seed for reproducibility
        
    Returns:
        Configured MultiAgentGridWorld environment
    """
    env = MultiAgentGridWorld(
        grid_size=grid_size,
        num_agents=num_agents,
        num_goals=num_goals,
        max_steps=max_steps,
        seed=seed,
    )
    return env


# Wrapper to convert to AEC format for compatibility
def make_aec_env(**kwargs) -> Any:
    """Create AEC version of the environment."""
    env = make_env(**kwargs)
    return parallel_to_aec(env)
