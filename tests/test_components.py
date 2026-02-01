"""Test suite for multi-agent reinforcement learning components."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch

from src.envs.grid_world import MultiAgentGridWorld, make_env
from src.algorithms.ippo import IPPOTrainer
from src.algorithms.mappo import MAPPOTrainer, MAPPOConfig
from src.eval.evaluator import MARLEvaluator, EvaluationConfig


class TestMultiAgentGridWorld:
    """Test cases for the multi-agent grid world environment."""
    
    def test_environment_creation(self):
        """Test environment creation and basic properties."""
        env = MultiAgentGridWorld(grid_size=5, num_agents=2, num_goals=1)
        
        assert env.grid_size == 5
        assert env.num_agents == 2
        assert env.num_goals == 1
        assert len(env.possible_agents) == 2
        assert "agent_0" in env.possible_agents
        assert "agent_1" in env.possible_agents
    
    def test_reset(self):
        """Test environment reset functionality."""
        env = MultiAgentGridWorld(grid_size=5, num_agents=2, num_goals=1, seed=42)
        
        obs, info = env.reset()
        
        assert len(obs) == 2
        assert "agent_0" in obs
        assert "agent_1" in obs
        assert len(env.agent_positions) == 2
        assert len(env.goal_positions) == 1
        assert len(env.goals_reached) == 1
        assert env.step_count == 0
    
    def test_step(self):
        """Test environment step functionality."""
        env = MultiAgentGridWorld(grid_size=5, num_agents=2, num_goals=1, seed=42)
        obs, _ = env.reset()
        
        # Test valid actions
        actions = {"agent_0": 0, "agent_1": 1}  # Up and Down
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        
        assert len(next_obs) == 2
        assert len(rewards) == 2
        assert len(terminations) == 2
        assert len(truncations) == 2
        assert len(infos) == 2
        
        # Check that agents moved
        assert env.step_count == 1
    
    def test_goal_reaching(self):
        """Test goal reaching mechanics."""
        env = MultiAgentGridWorld(grid_size=3, num_agents=1, num_goals=1, seed=42)
        obs, _ = env.reset()
        
        # Place agent at goal position
        env.agent_positions[0] = env.goal_positions[0].copy()
        
        actions = {"agent_0": 4}  # Stay action
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        
        assert rewards["agent_0"] == env.goal_reward
        assert env.goals_reached[0] == True
    
    def test_collision_detection(self):
        """Test collision detection and penalties."""
        env = MultiAgentGridWorld(grid_size=3, num_agents=2, num_goals=1, seed=42)
        obs, _ = env.reset()
        
        # Place both agents at the same position
        env.agent_positions[0] = np.array([1, 1])
        env.agent_positions[1] = np.array([1, 1])
        
        actions = {"agent_0": 4, "agent_1": 4}  # Both stay
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        
        assert rewards["agent_0"] == env.collision_penalty
        assert rewards["agent_1"] == env.collision_penalty
    
    def test_observation_space(self):
        """Test observation space properties."""
        env = MultiAgentGridWorld(grid_size=5, num_agents=2, num_goals=1)
        
        obs, _ = env.reset()
        
        for agent in env.possible_agents:
            assert agent in env.observation_spaces
            assert obs[agent].shape == env.observation_spaces[agent].shape
            assert obs[agent].dtype == np.float32
    
    def test_action_space(self):
        """Test action space properties."""
        env = MultiAgentGridWorld(grid_size=5, num_agents=2, num_goals=1)
        
        for agent in env.possible_agents:
            assert agent in env.action_spaces
            assert env.action_spaces[agent].n == 5  # 5 actions: up, down, left, right, stay


class TestIPPOTrainer:
    """Test cases for IPPO trainer."""
    
    def test_trainer_creation(self):
        """Test IPPO trainer creation."""
        env = MultiAgentGridWorld(grid_size=5, num_agents=2, num_goals=1, seed=42)
        
        trainer = IPPOTrainer(env, device="cpu", seed=42)
        
        assert len(trainer.policies) == 2
        assert "agent_0" in trainer.policies
        assert "agent_1" in trainer.policies
    
    def test_predict(self):
        """Test action prediction."""
        env = MultiAgentGridWorld(grid_size=5, num_agents=2, num_goals=1, seed=42)
        trainer = IPPOTrainer(env, device="cpu", seed=42)
        
        obs, _ = env.reset()
        actions, infos = trainer.predict(obs, deterministic=True)
        
        assert len(actions) == 2
        assert "agent_0" in actions
        assert "agent_1" in actions
        
        # Check action validity
        for agent in env.possible_agents:
            assert 0 <= actions[agent] < 5
    
    def test_evaluate(self):
        """Test policy evaluation."""
        env = MultiAgentGridWorld(grid_size=5, num_agents=2, num_goals=1, seed=42)
        trainer = IPPOTrainer(env, device="cpu", seed=42)
        
        eval_results = trainer.evaluate(env, n_eval_episodes=5)
        
        assert "agent_0_mean_reward" in eval_results
        assert "agent_1_mean_reward" in eval_results
        assert "mean_episode_length" in eval_results


class TestMAPPOTrainer:
    """Test cases for MAPPO trainer."""
    
    def test_trainer_creation(self):
        """Test MAPPO trainer creation."""
        env = MultiAgentGridWorld(grid_size=5, num_agents=2, num_goals=1, seed=42)
        config = MAPPOConfig()
        
        trainer = MAPPOTrainer(env, config, device="cpu", seed=42)
        
        assert len(trainer.actor_critics) == 2
        assert "agent_0" in trainer.actor_critics
        assert "agent_1" in trainer.actor_critics
    
    def test_predict(self):
        """Test action prediction."""
        env = MultiAgentGridWorld(grid_size=5, num_agents=2, num_goals=1, seed=42)
        config = MAPPOConfig()
        trainer = MAPPOTrainer(env, config, device="cpu", seed=42)
        
        obs, _ = env.reset()
        actions, infos = trainer.predict(obs, deterministic=True)
        
        assert len(actions) == 2
        assert "agent_0" in actions
        assert "agent_1" in actions
        
        # Check action validity
        for agent in env.possible_agents:
            assert 0 <= actions[agent] < 5
    
    def test_rollout_collection(self):
        """Test rollout collection."""
        env = MultiAgentGridWorld(grid_size=5, num_agents=2, num_goals=1, seed=42)
        config = MAPPOConfig(n_steps=10)  # Small number for testing
        trainer = MAPPOTrainer(env, config, device="cpu", seed=42)
        
        trainer.collect_rollouts()
        
        # Check that rollouts were collected
        assert len(trainer.observations["agent_0"]) > 0
        assert len(trainer.actions["agent_0"]) > 0
        assert len(trainer.rewards["agent_0"]) > 0


class TestMARLEvaluator:
    """Test cases for MARL evaluator."""
    
    def test_evaluator_creation(self):
        """Test evaluator creation."""
        env = MultiAgentGridWorld(grid_size=5, num_agents=2, num_goals=1, seed=42)
        config = EvaluationConfig(n_eval_episodes=5)
        
        evaluator = MARLEvaluator(env, config)
        
        assert evaluator.env == env
        assert evaluator.config == config
        assert len(evaluator.agents) == 2
    
    def test_policy_evaluation(self):
        """Test policy evaluation."""
        env = MultiAgentGridWorld(grid_size=5, num_agents=2, num_goals=1, seed=42)
        evaluator = MARLEvaluator(env)
        
        # Create a mock policy
        mock_policy = Mock()
        mock_policy.predict.return_value = (
            {"agent_0": 0, "agent_1": 1},
            {"agent_0": {}, "agent_1": {}}
        )
        
        results = evaluator.evaluate_policy(mock_policy, n_episodes=3)
        
        assert "agent_0_mean_reward" in results
        assert "agent_1_mean_reward" in results
        assert "social_welfare_mean" in results
        assert "efficiency_mean" in results
    
    def test_confidence_interval(self):
        """Test confidence interval calculation."""
        env = MultiAgentGridWorld(grid_size=5, num_agents=2, num_goals=1, seed=42)
        evaluator = MARLEvaluator(env)
        
        data = np.random.normal(0, 1, 100)
        ci_lower, ci_upper = evaluator._confidence_interval(data, confidence=0.95)
        
        assert ci_lower < ci_upper
        assert ci_lower < np.mean(data)
        assert ci_upper > np.mean(data)
    
    def test_gini_coefficient(self):
        """Test Gini coefficient calculation."""
        env = MultiAgentGridWorld(grid_size=5, num_agents=2, num_goals=1, seed=42)
        evaluator = MARLEvaluator(env)
        
        # Test perfect equality
        equal_values = np.array([1, 1, 1, 1])
        gini_equal = evaluator._gini_coefficient(equal_values)
        assert abs(gini_equal) < 1e-6
        
        # Test perfect inequality
        unequal_values = np.array([0, 0, 0, 1])
        gini_unequal = evaluator._gini_coefficient(unequal_values)
        assert gini_unequal > 0.5


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_training(self):
        """Test end-to-end training pipeline."""
        env = MultiAgentGridWorld(grid_size=5, num_agents=2, num_goals=1, seed=42)
        
        # Train IPPO
        trainer = IPPOTrainer(env, device="cpu", seed=42)
        
        # Mock training to avoid long execution
        with patch.object(trainer.policies["agent_0"], 'learn') as mock_learn:
            trainer.train(total_timesteps=1000)
            mock_learn.assert_called()
        
        # Test prediction
        obs, _ = env.reset()
        actions, _ = trainer.predict(obs)
        assert len(actions) == 2
    
    def test_environment_wrapper(self):
        """Test environment wrapper functionality."""
        env = make_env(grid_size=5, num_agents=2, num_goals=1, seed=42)
        
        assert hasattr(env, 'possible_agents')
        assert hasattr(env, 'action_spaces')
        assert hasattr(env, 'observation_spaces')
        
        obs, _ = env.reset()
        assert len(obs) == 2


if __name__ == "__main__":
    pytest.main([__file__])
