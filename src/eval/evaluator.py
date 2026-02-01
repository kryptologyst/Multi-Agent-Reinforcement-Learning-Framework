"""Comprehensive evaluation metrics for multi-agent reinforcement learning.

This module provides various metrics for evaluating MARL algorithms including
social welfare, NashConv, win rates, and learning efficiency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    n_eval_episodes: int = 100
    n_eval_seeds: int = 5
    confidence_level: float = 0.95
    save_plots: bool = True
    plot_dir: str = "assets/plots"


class MARLEvaluator:
    """Comprehensive evaluator for multi-agent reinforcement learning."""
    
    def __init__(
        self,
        env: Any,
        config: Optional[EvaluationConfig] = None,
    ):
        self.env = env
        self.config = config or EvaluationConfig()
        self.agents = env.possible_agents
        self.num_agents = len(self.agents)
    
    def evaluate_policy(
        self,
        policy: Any,
        n_episodes: Optional[int] = None,
        seeds: Optional[List[int]] = None,
        render: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate a policy across multiple episodes and seeds."""
        n_episodes = n_episodes or self.config.n_eval_episodes
        seeds = seeds or list(range(self.config.n_eval_seeds))
        
        all_results = []
        
        for seed in seeds:
            episode_results = []
            
            for episode in range(n_episodes):
                obs, _ = self.env.reset(seed=seed + episode)
                episode_rewards = {agent: 0.0 for agent in self.agents}
                episode_length = 0
                episode_info = {}
                
                done = False
                while not done:
                    if hasattr(policy, 'predict'):
                        actions, _ = policy.predict(obs, deterministic=True)
                    else:
                        # Assume policy is a callable
                        actions = policy(obs)
                    
                    obs, rewards, terminations, truncations, infos = self.env.step(actions)
                    
                    for agent in self.agents:
                        if agent in rewards:
                            episode_rewards[agent] += rewards[agent]
                    
                    episode_length += 1
                    done = any(terminations.values()) or any(truncations.values())
                    
                    if render and episode == 0:  # Only render first episode
                        self.env.render()
                
                # Store episode results
                episode_result = {
                    "seed": seed,
                    "episode": episode,
                    "length": episode_length,
                    **{f"{agent}_reward": episode_rewards[agent] for agent in self.agents},
                    **episode_info,
                }
                episode_results.append(episode_result)
            
            all_results.extend(episode_results)
        
        return self._analyze_results(all_results)
    
    def _analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze evaluation results and compute statistics."""
        df = pd.DataFrame(results)
        
        analysis = {}
        
        # Individual agent metrics
        for agent in self.agents:
            reward_col = f"{agent}_reward"
            rewards = df[reward_col].values
            
            analysis[f"{agent}_mean_reward"] = np.mean(rewards)
            analysis[f"{agent}_std_reward"] = np.std(rewards)
            analysis[f"{agent}_min_reward"] = np.min(rewards)
            analysis[f"{agent}_max_reward"] = np.max(rewards)
            analysis[f"{agent}_median_reward"] = np.median(rewards)
            
            # Confidence interval
            ci_lower, ci_upper = self._confidence_interval(
                rewards, self.config.confidence_level
            )
            analysis[f"{agent}_ci_lower"] = ci_lower
            analysis[f"{agent}_ci_upper"] = ci_upper
        
        # Multi-agent metrics
        analysis.update(self._compute_multi_agent_metrics(df))
        
        # Episode length metrics
        lengths = df["length"].values
        analysis["mean_episode_length"] = np.mean(lengths)
        analysis["std_episode_length"] = np.std(lengths)
        
        return analysis
    
    def _compute_multi_agent_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute multi-agent specific metrics."""
        metrics = {}
        
        # Social welfare (sum of all agent rewards)
        social_welfare = df[[f"{agent}_reward" for agent in self.agents]].sum(axis=1)
        metrics["social_welfare_mean"] = np.mean(social_welfare)
        metrics["social_welfare_std"] = np.std(social_welfare)
        
        # Efficiency (social welfare / episode length)
        efficiency = social_welfare / df["length"]
        metrics["efficiency_mean"] = np.mean(efficiency)
        metrics["efficiency_std"] = np.std(efficiency)
        
        # Fairness metrics
        agent_rewards = df[[f"{agent}_reward" for agent in self.agents]].values
        
        # Gini coefficient for fairness
        gini_coeffs = []
        for episode_rewards in agent_rewards:
            gini_coeffs.append(self._gini_coefficient(episode_rewards))
        metrics["gini_coefficient_mean"] = np.mean(gini_coeffs)
        metrics["gini_coefficient_std"] = np.std(gini_coeffs)
        
        # Variance in rewards across agents
        reward_variances = np.var(agent_rewards, axis=1)
        metrics["reward_variance_mean"] = np.mean(reward_variances)
        metrics["reward_variance_std"] = np.std(reward_variances)
        
        # Success rate (episodes where all agents get positive reward)
        success_rate = np.mean(np.all(agent_rewards > 0, axis=1))
        metrics["success_rate"] = success_rate
        
        return metrics
    
    def _gini_coefficient(self, values: np.ndarray) -> float:
        """Compute Gini coefficient for fairness measurement."""
        if len(values) == 0:
            return 0.0
        
        sorted_values = np.sort(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0.0
    
    def _confidence_interval(
        self, data: np.ndarray, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Compute confidence interval for data."""
        n = len(data)
        mean = np.mean(data)
        std_err = np.std(data) / np.sqrt(n)
        
        # Use t-distribution for small samples
        if n < 30:
            from scipy import stats
            t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
            margin = t_val * std_err
        else:
            # Use normal distribution for large samples
            z_val = 1.96 if confidence == 0.95 else 2.576  # 99% CI
            margin = z_val * std_err
        
        return mean - margin, mean + margin
    
    def compare_policies(
        self,
        policies: Dict[str, Any],
        policy_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compare multiple policies."""
        if policy_names is None:
            policy_names = list(policies.keys())
        
        comparison_results = {}
        
        for name, policy in zip(policy_names, policies.values()):
            print(f"Evaluating policy: {name}")
            results = self.evaluate_policy(policy)
            comparison_results[name] = results
        
        # Create comparison summary
        summary = self._create_comparison_summary(comparison_results, policy_names)
        
        if self.config.save_plots:
            self._plot_comparison(comparison_results, policy_names)
        
        return {
            "individual_results": comparison_results,
            "summary": summary,
        }
    
    def _create_comparison_summary(
        self, results: Dict[str, Dict[str, Any]], policy_names: List[str]
    ) -> pd.DataFrame:
        """Create a summary table comparing policies."""
        summary_data = []
        
        for policy_name in policy_names:
            policy_results = results[policy_name]
            
            row = {
                "Policy": policy_name,
                "Social Welfare": f"{policy_results['social_welfare_mean']:.3f} ± {policy_results['social_welfare_std']:.3f}",
                "Efficiency": f"{policy_results['efficiency_mean']:.3f} ± {policy_results['efficiency_std']:.3f}",
                "Fairness (Gini)": f"{policy_results['gini_coefficient_mean']:.3f} ± {policy_results['gini_coefficient_std']:.3f}",
                "Success Rate": f"{policy_results['success_rate']:.3f}",
                "Episode Length": f"{policy_results['mean_episode_length']:.1f} ± {policy_results['std_episode_length']:.1f}",
            }
            
            # Add individual agent rewards
            for agent in self.agents:
                row[f"{agent} Reward"] = f"{policy_results[f'{agent}_mean_reward']:.3f} ± {policy_results[f'{agent}_std_reward']:.3f}"
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def _plot_comparison(
        self, results: Dict[str, Dict[str, Any]], policy_names: List[str]
    ) -> None:
        """Create comparison plots."""
        import os
        os.makedirs(self.config.plot_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Social welfare comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Social welfare
        social_welfare_means = [results[name]['social_welfare_mean'] for name in policy_names]
        social_welfare_stds = [results[name]['social_welfare_std'] for name in policy_names]
        
        axes[0, 0].bar(policy_names, social_welfare_means, yerr=social_welfare_stds, capsize=5)
        axes[0, 0].set_title('Social Welfare Comparison')
        axes[0, 0].set_ylabel('Social Welfare')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Efficiency
        efficiency_means = [results[name]['efficiency_mean'] for name in policy_names]
        efficiency_stds = [results[name]['efficiency_std'] for name in policy_names]
        
        axes[0, 1].bar(policy_names, efficiency_means, yerr=efficiency_stds, capsize=5)
        axes[0, 1].set_title('Efficiency Comparison')
        axes[0, 1].set_ylabel('Efficiency')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Fairness
        fairness_means = [results[name]['gini_coefficient_mean'] for name in policy_names]
        fairness_stds = [results[name]['gini_coefficient_std'] for name in policy_names]
        
        axes[1, 0].bar(policy_names, fairness_means, yerr=fairness_stds, capsize=5)
        axes[1, 0].set_title('Fairness Comparison (Lower Gini = More Fair)')
        axes[1, 0].set_ylabel('Gini Coefficient')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Success rate
        success_rates = [results[name]['success_rate'] for name in policy_names]
        
        axes[1, 1].bar(policy_names, success_rates)
        axes[1, 1].set_title('Success Rate Comparison')
        axes[1, 1].set_ylabel('Success Rate')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.config.plot_dir}/policy_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Individual agent rewards
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(policy_names))
        width = 0.8 / self.num_agents
        
        for i, agent in enumerate(self.agents):
            agent_means = [results[name][f'{agent}_mean_reward'] for name in policy_names]
            agent_stds = [results[name][f'{agent}_std_reward'] for name in policy_names]
            
            ax.bar(x + i * width, agent_means, width, yerr=agent_stds, 
                   label=agent, capsize=3)
        
        ax.set_xlabel('Policy')
        ax.set_ylabel('Mean Reward')
        ax.set_title('Individual Agent Rewards Comparison')
        ax.set_xticks(x + width * (self.num_agents - 1) / 2)
        ax.set_xticklabels(policy_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.config.plot_dir}/agent_rewards_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def learning_curves(
        self,
        training_data: Dict[str, List[float]],
        save_path: Optional[str] = None,
    ) -> None:
        """Plot learning curves."""
        if save_path is None:
            save_path = f"{self.config.plot_dir}/learning_curves.png"
        
        plt.figure(figsize=(12, 8))
        
        for name, rewards in training_data.items():
            episodes = range(len(rewards))
            
            # Smooth the curve
            if len(rewards) > 10:
                window_size = max(10, len(rewards) // 20)
                smoothed = pd.Series(rewards).rolling(window=window_size, center=True).mean()
                plt.plot(episodes, smoothed, label=f"{name} (smoothed)", linewidth=2)
            else:
                plt.plot(episodes, rewards, label=name, linewidth=2)
        
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_leaderboard(
        self, results: Dict[str, Dict[str, Any]], metric: str = "social_welfare_mean"
    ) -> pd.DataFrame:
        """Create a leaderboard sorted by the specified metric."""
        leaderboard_data = []
        
        for policy_name, policy_results in results.items():
            leaderboard_data.append({
                "Policy": policy_name,
                "Score": policy_results[metric],
                "Social Welfare": policy_results["social_welfare_mean"],
                "Efficiency": policy_results["efficiency_mean"],
                "Fairness": policy_results["gini_coefficient_mean"],
                "Success Rate": policy_results["success_rate"],
            })
        
        leaderboard = pd.DataFrame(leaderboard_data)
        leaderboard = leaderboard.sort_values("Score", ascending=False)
        leaderboard["Rank"] = range(1, len(leaderboard) + 1)
        
        return leaderboard[["Rank", "Policy", "Score", "Social Welfare", "Efficiency", "Fairness", "Success Rate"]]
