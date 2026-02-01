#!/usr/bin/env python3
"""Quick test script to verify the multi-agent RL framework works correctly."""

import sys
import os
import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.envs.grid_world import make_env, MultiAgentGridWorld
        print("✅ Environment imports successful")
    except ImportError as e:
        print(f"❌ Environment import failed: {e}")
        return False
    
    try:
        from src.algorithms.ippo import IPPOTrainer
        from src.algorithms.mappo import MAPPOTrainer, MAPPOConfig
        print("✅ Algorithm imports successful")
    except ImportError as e:
        print(f"❌ Algorithm import failed: {e}")
        return False
    
    try:
        from src.eval.evaluator import MARLEvaluator, EvaluationConfig
        print("✅ Evaluator imports successful")
    except ImportError as e:
        print(f"❌ Evaluator import failed: {e}")
        return False
    
    return True


def test_environment():
    """Test environment creation and basic functionality."""
    print("\nTesting environment...")
    
    try:
        from src.envs.grid_world import make_env
        
        # Create environment
        env = make_env(grid_size=5, num_agents=2, num_goals=1, seed=42)
        print("✅ Environment creation successful")
        
        # Test reset
        obs, info = env.reset()
        assert len(obs) == 2, "Should have 2 agents"
        assert "agent_0" in obs, "Should have agent_0"
        assert "agent_1" in obs, "Should have agent_1"
        print("✅ Environment reset successful")
        
        # Test step
        actions = {"agent_0": 0, "agent_1": 1}
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        assert len(rewards) == 2, "Should have rewards for 2 agents"
        print("✅ Environment step successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False


def test_algorithms():
    """Test algorithm creation and basic functionality."""
    print("\nTesting algorithms...")
    
    try:
        from src.envs.grid_world import make_env
        from src.algorithms.ippo import IPPOTrainer
        from src.algorithms.mappo import MAPPOTrainer, MAPPOConfig
        
        # Create environment
        env = make_env(grid_size=5, num_agents=2, num_goals=1, seed=42)
        
        # Test IPPO
        ippo_trainer = IPPOTrainer(env, device="cpu", seed=42)
        assert len(ippo_trainer.policies) == 2, "Should have 2 policies"
        print("✅ IPPO trainer creation successful")
        
        # Test MAPPO
        config = MAPPOConfig()
        mappo_trainer = MAPPOTrainer(env, config, device="cpu", seed=42)
        assert len(mappo_trainer.actor_critics) == 2, "Should have 2 actor-critics"
        print("✅ MAPPO trainer creation successful")
        
        # Test prediction
        obs, _ = env.reset()
        ippo_actions, _ = ippo_trainer.predict(obs, deterministic=True)
        mappo_actions, _ = mappo_trainer.predict(obs, deterministic=True)
        
        assert len(ippo_actions) == 2, "IPPO should return 2 actions"
        assert len(mappo_actions) == 2, "MAPPO should return 2 actions"
        print("✅ Algorithm prediction successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Algorithm test failed: {e}")
        return False


def test_evaluator():
    """Test evaluator functionality."""
    print("\nTesting evaluator...")
    
    try:
        from src.envs.grid_world import make_env
        from src.algorithms.ippo import IPPOTrainer
        from src.eval.evaluator import MARLEvaluator
        
        # Create environment and trainer
        env = make_env(grid_size=5, num_agents=2, num_goals=1, seed=42)
        trainer = IPPOTrainer(env, device="cpu", seed=42)
        evaluator = MARLEvaluator(env)
        
        # Test evaluation (with mock policy to avoid training)
        class MockPolicy:
            def predict(self, obs, deterministic=True):
                return {agent: env.action_spaces[agent].sample() for agent in env.possible_agents}, {}
        
        mock_policy = MockPolicy()
        results = evaluator.evaluate_policy(mock_policy, n_episodes=3)
        
        assert "social_welfare_mean" in results, "Should have social welfare metric"
        assert "efficiency_mean" in results, "Should have efficiency metric"
        print("✅ Evaluator test successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluator test failed: {e}")
        return False


def test_device_detection():
    """Test device detection."""
    print("\nTesting device detection...")
    
    try:
        from src.utils import get_device
        
        device = get_device()
        print(f"✅ Detected device: {device}")
        
        # Test with specific device
        cpu_device = get_device("cpu")
        assert str(cpu_device) == "cpu", "Should return CPU device"
        print("✅ Device specification successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Device detection test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Multi-Agent RL Framework Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_environment,
        test_algorithms,
        test_evaluator,
        test_device_detection,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The framework is ready to use.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
