"""
PPO Baseline: Static Multimodal Reinforcement Learning Trading System

A comprehensive implementation of the PPO baseline system that combines:
- Multimodal data processing (visual, technical, sentiment)
- Market regime detection using XGBoost
- PPO-based reinforcement learning for trading decisions

This package provides the foundation level (Level 0) for the multi-level
RL trading system comparison study.
"""

from .ppo_agent import PPOAgent
from .multimodal_fusion import MultimodalFusion
from .regime_detector import MarketRegimeDetector
from .data_processor import DataProcessor
from .trading_env import TradingEnvironment
from .ppo_baseline import PPOBaseline

__version__ = "1.0.0"
__author__ = "GRPO Research Team"

__all__ = [
    "PPOAgent",
    "MultimodalFusion", 
    "MarketRegimeDetector",
    "DataProcessor",
    "TradingEnvironment",
    "PPOBaseline"
]
