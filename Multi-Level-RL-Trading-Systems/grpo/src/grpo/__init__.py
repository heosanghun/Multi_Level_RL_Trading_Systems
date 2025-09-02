"""
Level 1: Basic GRPO (Group Relative Policy Optimization) Trading System

A simple GRU-based GRPO implementation for cryptocurrency trading.
"""

from .data import DataLoader
from .agent import GRPOAgent
from .trainer import GRPOTrainer
from .backtest import SimpleBacktester

__version__ = "1.0.0"
__author__ = "GRPO Research Team"

__all__ = [
    "DataLoader",
    "GRPOAgent", 
    "GRPOTrainer",
    "SimpleBacktester"
]
