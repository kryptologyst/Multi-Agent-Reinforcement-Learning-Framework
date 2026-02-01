"""Utility functions for multi-agent reinforcement learning."""

from __future__ import annotations

import random
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Union


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set environment seeds if available
    try:
        import gymnasium as gym
        gym.utils.seeding.np_random = np.random.RandomState(seed)
    except ImportError:
        pass


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Get the best available device.
    
    Args:
        device: Preferred device or None for auto-detection
        
    Returns:
        Available torch device
    """
    if device is not None:
        return torch.device(device)
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: Reward tensor of shape (T,)
        values: Value estimates of shape (T,)
        next_values: Next value estimates of shape (T,)
        dones: Done flags of shape (T,)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        
    Returns:
        Tuple of (advantages, returns)
    """
    advantages = torch.zeros_like(rewards)
    last_advantage = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[t].float()
            next_value = next_values[t]
        else:
            next_non_terminal = 1.0 - dones[t].float()
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        advantages[t] = last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
    
    returns = advantages + values
    return advantages, returns


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize advantages to have zero mean and unit variance.
    
    Args:
        advantages: Advantage tensor
        eps: Small constant for numerical stability
        
    Returns:
        Normalized advantages
    """
    return (advantages - advantages.mean()) / (advantages.std() + eps)


def compute_returns(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """Compute discounted returns.
    
    Args:
        rewards: Reward tensor of shape (T,)
        dones: Done flags of shape (T,)
        gamma: Discount factor
        
    Returns:
        Discounted returns
    """
    returns = torch.zeros_like(rewards)
    running_return = 0
    
    for t in reversed(range(len(rewards))):
        if dones[t]:
            running_return = 0
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
    
    return returns


def create_logger(name: str, log_level: str = "INFO") -> Any:
    """Create a logger with specified name and level.
    
    Args:
        name: Logger name
        log_level: Logging level
        
    Returns:
        Logger instance
    """
    import logging
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def save_checkpoint(
    model: Any,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
) -> None:
    """Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    model: Any,
    optimizer: torch.optim.Optimizer,
    filepath: str,
) -> Dict[str, Any]:
    """Load model checkpoint.
    
    Args:
        model: Model to load state into
        optimizer: Optimizer to load state into
        filepath: Path to checkpoint file
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint


def create_summary_writer(log_dir: str) -> Any:
    """Create TensorBoard summary writer.
    
    Args:
        log_dir: Directory for logs
        
    Returns:
        SummaryWriter instance
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir)
    except ImportError:
        # Return a dummy writer if TensorBoard is not available
        class DummyWriter:
            def add_scalar(self, *args, **kwargs):
                pass
            def add_histogram(self, *args, **kwargs):
                pass
            def close(self):
                pass
        return DummyWriter()


def moving_average(data: List[float], window_size: int = 10) -> List[float]:
    """Compute moving average of data.
    
    Args:
        data: Input data
        window_size: Size of moving window
        
    Returns:
        Smoothed data
    """
    if len(data) < window_size:
        return data
    
    smoothed = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size + 1)
        smoothed.append(np.mean(data[start_idx:i+1]))
    
    return smoothed


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def print_progress_bar(
    current: int,
    total: int,
    prefix: str = "Progress",
    suffix: str = "Complete",
    length: int = 50,
) -> None:
    """Print a progress bar.
    
    Args:
        current: Current progress
        total: Total progress
        prefix: Prefix text
        suffix: Suffix text
        length: Length of progress bar
    """
    percent = 100 * (current / float(total))
    filled_length = int(length * current // total)
    bar = "█" * filled_length + "-" * (length - filled_length)
    print(f"\r{prefix} |{bar}| {percent:.1f}% {suffix}", end="", flush=True)
    
    if current == total:
        print()  # New line when complete
