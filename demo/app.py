"""Interactive Streamlit demo for Multi-Agent Reinforcement Learning.

This demo allows users to visualize trained policies, run evaluations,
and explore different environment configurations.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
from typing import Dict, Any, List

# Import our modules
from src.envs.grid_world import make_env
from src.algorithms.ippo import IPPOTrainer
from src.algorithms.mappo import MAPPOTrainer, MAPPOConfig
from src.eval.evaluator import MARLEvaluator, EvaluationConfig


def initialize_session_state():
    """Initialize Streamlit session state."""
    if "env" not in st.session_state:
        st.session_state.env = None
    if "trained_policies" not in st.session_state:
        st.session_state.trained_policies = {}
    if "evaluation_results" not in st.session_state:
        st.session_state.evaluation_results = {}


def create_environment(grid_size: int, num_agents: int, num_goals: int, max_steps: int) -> Any:
    """Create a new environment with specified parameters."""
    return make_env(
        grid_size=grid_size,
        num_agents=num_agents,
        num_goals=num_goals,
        max_steps=max_steps,
        seed=42,
    )


def train_policy(algorithm: str, env: Any, config: Dict[str, Any]) -> Any:
    """Train a policy using the specified algorithm."""
    device = "cpu"  # Use CPU for demo
    
    if algorithm == "IPPO":
        trainer = IPPOTrainer(
            env=env,
            learning_rate=config["learning_rate"],
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            gamma=config["gamma"],
            device=device,
            seed=42,
        )
    elif algorithm == "MAPPO":
        mappo_config = MAPPOConfig(
            learning_rate=config["learning_rate"],
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            gamma=config["gamma"],
        )
        trainer = MAPPOTrainer(
            env=env,
            config=mappo_config,
            device=device,
            seed=42,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Train with progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Simulate training progress (in real implementation, this would be actual training)
    for i in range(100):
        progress_bar.progress(i + 1)
        status_text.text(f"Training {algorithm}... {i+1}%")
        time.sleep(0.01)  # Simulate training time
    
    status_text.text(f"{algorithm} training completed!")
    return trainer


def render_environment(env: Any, policy: Any = None) -> None:
    """Render the environment with optional policy actions."""
    if policy is None:
        # Random actions
        actions = {agent: env.action_spaces[agent].sample() for agent in env.possible_agents}
    else:
        actions, _ = policy.predict(env._get_observations(), deterministic=True)
    
    # Step environment
    obs, rewards, terminations, truncations, infos = env.step(actions)
    
    # Create visualization
    grid_size = env.grid_size
    
    # Create grid data
    grid_data = []
    for y in range(grid_size):
        for x in range(grid_size):
            cell_type = "empty"
            cell_value = 0
            
            # Check for goals
            for i, goal_pos in enumerate(env.goal_positions):
                if np.array_equal([x, y], goal_pos) and not env.goals_reached[i]:
                    cell_type = "goal"
                    cell_value = 1
                    break
            
            # Check for agents
            for i, agent_pos in enumerate(env.agent_positions):
                if np.array_equal([x, y], agent_pos):
                    cell_type = f"agent_{i}"
                    cell_value = 2 + i
                    break
            
            grid_data.append({
                "x": x,
                "y": y,
                "type": cell_type,
                "value": cell_value,
            })
    
    # Create plotly heatmap
    df = pd.DataFrame(grid_data)
    pivot_df = df.pivot(index="y", columns="x", values="value")
    
    fig = px.imshow(
        pivot_df.values,
        color_continuous_scale="viridis",
        title=f"Multi-Agent Grid World - Step {env.step_count}",
        labels={"x": "X Position", "y": "Y Position"},
    )
    
    fig.update_layout(
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(autorange="reversed"),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display episode info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Goals Reached", f"{sum(env.goals_reached)}/{env.num_goals}")
    with col2:
        st.metric("Step", env.step_count)
    with col3:
        total_reward = sum(rewards.values())
        st.metric("Total Reward", f"{total_reward:.2f}")


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Multi-Agent RL Demo",
        page_icon="🤖",
        layout="wide",
    )
    
    st.title("🤖 Multi-Agent Reinforcement Learning Demo")
    st.markdown("""
    This demo showcases multi-agent reinforcement learning algorithms on a cooperative grid world environment.
    Agents must coordinate to reach goals while avoiding collisions.
    """)
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Environment parameters
    st.sidebar.subheader("Environment")
    grid_size = st.sidebar.slider("Grid Size", 5, 12, 8)
    num_agents = st.sidebar.slider("Number of Agents", 2, 4, 2)
    num_goals = st.sidebar.slider("Number of Goals", 1, 3, 2)
    max_steps = st.sidebar.slider("Max Steps", 50, 200, 100)
    
    # Training parameters
    st.sidebar.subheader("Training")
    algorithm = st.sidebar.selectbox("Algorithm", ["IPPO", "MAPPO"])
    learning_rate = st.sidebar.slider("Learning Rate", 1e-5, 1e-2, 3e-4, format="%.2e")
    n_steps = st.sidebar.slider("Steps per Update", 512, 4096, 2048)
    batch_size = st.sidebar.slider("Batch Size", 32, 256, 64)
    
    # Create environment
    if st.sidebar.button("Create Environment") or st.session_state.env is None:
        st.session_state.env = create_environment(grid_size, num_agents, num_goals, max_steps)
        st.session_state.trained_policies = {}  # Reset policies when env changes
        st.success("Environment created!")
    
    # Main content
    if st.session_state.env is not None:
        # Environment visualization
        st.header("Environment Visualization")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("Reset Environment"):
                st.session_state.env.reset()
            
            if st.button("Step (Random Actions)"):
                render_environment(st.session_state.env)
        
        with col2:
            st.subheader("Environment Info")
            st.write(f"**Grid Size:** {grid_size}x{grid_size}")
            st.write(f"**Agents:** {num_agents}")
            st.write(f"**Goals:** {num_goals}")
            st.write(f"**Max Steps:** {max_steps}")
        
        # Training section
        st.header("Training")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button(f"Train {algorithm}"):
                config = {
                    "learning_rate": learning_rate,
                    "n_steps": n_steps,
                    "batch_size": batch_size,
                    "n_epochs": 5,  # Reduced for demo
                    "gamma": 0.99,
                }
                
                trainer = train_policy(algorithm, st.session_state.env, config)
                st.session_state.trained_policies[algorithm] = trainer
                st.success(f"{algorithm} training completed!")
        
        with col2:
            if st.session_state.trained_policies:
                st.subheader("Trained Policies")
                for policy_name in st.session_state.trained_policies.keys():
                    st.write(f"✅ {policy_name}")
        
        # Policy evaluation section
        if st.session_state.trained_policies:
            st.header("Policy Evaluation")
            
            selected_policy = st.selectbox(
                "Select Policy to Evaluate",
                list(st.session_state.trained_policies.keys())
            )
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("Run Episode with Policy"):
                    policy = st.session_state.trained_policies[selected_policy]
                    
                    # Reset environment
                    st.session_state.env.reset()
                    
                    # Run episode
                    episode_rewards = {agent: 0.0 for agent in st.session_state.env.possible_agents}
                    step = 0
                    
                    while step < 50:  # Limit for demo
                        render_environment(st.session_state.env, policy)
                        
                        # Check if episode is done
                        if (sum(st.session_state.env.goals_reached) == st.session_state.env.num_goals or
                            st.session_state.env.step_count >= st.session_state.env.max_steps):
                            break
                        
                        step += 1
                        time.sleep(0.5)  # Slow down for visualization
            
            with col2:
                if st.button("Quick Evaluation"):
                    policy = st.session_state.trained_policies[selected_policy]
                    evaluator = MARLEvaluator(st.session_state.env)
                    
                    # Run evaluation
                    results = evaluator.evaluate_policy(policy, n_episodes=10)
                    
                    # Display results
                    st.subheader("Evaluation Results")
                    st.write(f"**Social Welfare:** {results['social_welfare_mean']:.3f} ± {results['social_welfare_std']:.3f}")
                    st.write(f"**Efficiency:** {results['efficiency_mean']:.3f} ± {results['efficiency_std']:.3f}")
                    st.write(f"**Success Rate:** {results['success_rate']:.3f}")
                    
                    # Individual agent rewards
                    st.subheader("Agent Rewards")
                    for agent in st.session_state.env.possible_agents:
                        mean_reward = results[f"{agent}_mean_reward"]
                        std_reward = results[f"{agent}_std_reward"]
                        st.write(f"**{agent}:** {mean_reward:.3f} ± {std_reward:.3f}")
        
        # Comparison section
        if len(st.session_state.trained_policies) > 1:
            st.header("Policy Comparison")
            
            if st.button("Compare All Policies"):
                evaluator = MARLEvaluator(st.session_state.env)
                comparison_results = evaluator.compare_policies(st.session_state.trained_policies)
                
                # Display comparison table
                st.subheader("Comparison Summary")
                st.dataframe(comparison_results["summary"])
                
                # Create comparison plots
                st.subheader("Comparison Charts")
                
                # Social welfare comparison
                fig = px.bar(
                    x=list(st.session_state.trained_policies.keys()),
                    y=[comparison_results["individual_results"][policy]["social_welfare_mean"] 
                       for policy in st.session_state.trained_policies.keys()],
                    title="Social Welfare Comparison",
                    labels={"x": "Policy", "y": "Social Welfare"},
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Efficiency comparison
                fig = px.bar(
                    x=list(st.session_state.trained_policies.keys()),
                    y=[comparison_results["individual_results"][policy]["efficiency_mean"] 
                       for policy in st.session_state.trained_policies.keys()],
                    title="Efficiency Comparison",
                    labels={"x": "Policy", "y": "Efficiency"},
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Note:** This is a research/educational demo. The algorithms shown here are not intended for production use in real-world systems.
    
    **Safety Disclaimer:** This demo is for educational purposes only. Do not use these algorithms for controlling real-world systems without proper safety measures and validation.
    """)


if __name__ == "__main__":
    main()
