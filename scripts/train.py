"""Main training script for multi-agent reinforcement learning.

This script demonstrates training multiple MARL algorithms on the grid world environment
and provides comprehensive evaluation and comparison.
"""

import argparse
import os
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

# Import our modules
from src.envs.grid_world import make_env
from src.algorithms.ippo import IPPOTrainer
from src.algorithms.mappo import MAPPOTrainer, MAPPOConfig
from src.eval.evaluator import MARLEvaluator, EvaluationConfig


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train_ippo(
    env: Any,
    config: Dict[str, Any],
    device: torch.device,
    seed: int,
) -> IPPOTrainer:
    """Train IPPO agents."""
    print("Training IPPO (Independent PPO)...")
    
    trainer = IPPOTrainer(
        env=env,
        learning_rate=config.get("learning_rate", 3e-4),
        n_steps=config.get("n_steps", 2048),
        batch_size=config.get("batch_size", 64),
        n_epochs=config.get("n_epochs", 10),
        gamma=config.get("gamma", 0.99),
        gae_lambda=config.get("gae_lambda", 0.95),
        clip_range=config.get("clip_range", 0.2),
        ent_coef=config.get("ent_coef", 0.01),
        vf_coef=config.get("vf_coef", 0.5),
        max_grad_norm=config.get("max_grad_norm", 0.5),
        device=device,
        seed=seed,
    )
    
    start_time = time.time()
    trainer.train(total_timesteps=config.get("total_timesteps", 100000))
    training_time = time.time() - start_time
    
    print(f"IPPO training completed in {training_time:.2f} seconds")
    return trainer


def train_mappo(
    env: Any,
    config: Dict[str, Any],
    device: torch.device,
    seed: int,
) -> MAPPOTrainer:
    """Train MAPPO agents."""
    print("Training MAPPO (Multi-Agent PPO)...")
    
    mappo_config = MAPPOConfig(
        learning_rate=config.get("learning_rate", 3e-4),
        n_steps=config.get("n_steps", 2048),
        batch_size=config.get("batch_size", 64),
        n_epochs=config.get("n_epochs", 10),
        gamma=config.get("gamma", 0.99),
        gae_lambda=config.get("gae_lambda", 0.95),
        clip_range=config.get("clip_range", 0.2),
        ent_coef=config.get("ent_coef", 0.01),
        vf_coef=config.get("vf_coef", 0.5),
        max_grad_norm=config.get("max_grad_norm", 0.5),
        use_centralized_value=config.get("use_centralized_value", True),
        use_gae=config.get("use_gae", True),
        normalize_advantages=config.get("normalize_advantages", True),
    )
    
    trainer = MAPPOTrainer(
        env=env,
        config=mappo_config,
        device=device,
        seed=seed,
    )
    
    start_time = time.time()
    trainer.train(total_timesteps=config.get("total_timesteps", 100000))
    training_time = time.time() - start_time
    
    print(f"MAPPO training completed in {training_time:.2f} seconds")
    return trainer


def main():
    """Main training and evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Multi-Agent RL Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file path")
    parser.add_argument("--algorithms", nargs="+", default=["ippo", "mappo"], help="Algorithms to train")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate, don't train")
    parser.add_argument("--model-path", type=str, help="Path to saved models for evaluation")
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            "env": {
                "grid_size": 8,
                "num_agents": 2,
                "num_goals": 2,
                "max_steps": 100,
            },
            "training": {
                "total_timesteps": 100000,
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
            },
            "evaluation": {
                "n_eval_episodes": 100,
                "n_eval_seeds": 5,
                "save_plots": True,
            }
        }
    
    # Set device
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Set seeds
    set_seeds(args.seed)
    
    # Create environment
    env_config = config["env"]
    env = make_env(
        grid_size=env_config["grid_size"],
        num_agents=env_config["num_agents"],
        num_goals=env_config["num_goals"],
        max_steps=env_config["max_steps"],
        seed=args.seed,
    )
    
    print(f"Environment created: {env_config['num_agents']} agents, {env_config['num_goals']} goals")
    
    # Create evaluator
    eval_config = EvaluationConfig(**config["evaluation"])
    evaluator = MARLEvaluator(env, eval_config)
    
    # Training
    trained_policies = {}
    
    if not args.eval_only:
        training_config = config["training"]
        
        for algorithm in args.algorithms:
            print(f"\n{'='*50}")
            print(f"Training {algorithm.upper()}")
            print(f"{'='*50}")
            
            if algorithm.lower() == "ippo":
                trainer = train_ippo(env, training_config, device, args.seed)
                trained_policies["IPPO"] = trainer
                
            elif algorithm.lower() == "mappo":
                trainer = train_mappo(env, training_config, device, args.seed)
                trained_policies["MAPPO"] = trainer
                
            else:
                print(f"Unknown algorithm: {algorithm}")
                continue
            
            # Save models
            os.makedirs("checkpoints", exist_ok=True)
            trainer.save(f"checkpoints/{algorithm}_{args.seed}")
            print(f"Models saved to checkpoints/{algorithm}_{args.seed}")
    
    else:
        # Load pre-trained models
        if args.model_path:
            print(f"Loading models from {args.model_path}")
            # This would need to be implemented based on the specific model format
            # For now, we'll skip this functionality
            print("Model loading not implemented yet")
            return
    
    # Evaluation
    if trained_policies:
        print(f"\n{'='*50}")
        print("EVALUATION")
        print(f"{'='*50}")
        
        # Compare policies
        comparison_results = evaluator.compare_policies(trained_policies)
        
        # Print summary
        print("\nPolicy Comparison Summary:")
        print(comparison_results["summary"].to_string(index=False))
        
        # Create leaderboard
        leaderboard = evaluator.create_leaderboard(comparison_results["individual_results"])
        print("\nLeaderboard (sorted by Social Welfare):")
        print(leaderboard.to_string(index=False))
        
        # Save results
        os.makedirs("results", exist_ok=True)
        comparison_results["summary"].to_csv(f"results/comparison_summary_{args.seed}.csv", index=False)
        leaderboard.to_csv(f"results/leaderboard_{args.seed}.csv", index=False)
        
        print(f"\nResults saved to results/ directory")
    
    # Demo evaluation
    print(f"\n{'='*50}")
    print("DEMO EVALUATION")
    print(f"{'='*50}")
    
    if trained_policies:
        # Run a few episodes with rendering
        best_policy_name = max(
            trained_policies.keys(),
            key=lambda x: evaluator.evaluate_policy(trained_policies[x])["social_welfare_mean"]
        )
        best_policy = trained_policies[best_policy_name]
        
        print(f"Running demo with best policy: {best_policy_name}")
        
        for episode in range(3):
            print(f"\nDemo Episode {episode + 1}:")
            obs, _ = env.reset()
            total_reward = {agent: 0.0 for agent in env.possible_agents}
            
            done = False
            step = 0
            while not done and step < 50:  # Limit demo steps
                actions, _ = best_policy.predict(obs, deterministic=True)
                obs, rewards, terminations, truncations, _ = env.step(actions)
                
                for agent in env.possible_agents:
                    if agent in rewards:
                        total_reward[agent] += rewards[agent]
                
                env.render(mode="human")
                step += 1
                done = any(terminations.values()) or any(truncations.values())
            
            print(f"Episode {episode + 1} completed:")
            for agent, reward in total_reward.items():
                print(f"  {agent}: {reward:.2f}")
            print(f"  Social Welfare: {sum(total_reward.values()):.2f}")
    
    print(f"\n{'='*50}")
    print("TRAINING AND EVALUATION COMPLETED")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
