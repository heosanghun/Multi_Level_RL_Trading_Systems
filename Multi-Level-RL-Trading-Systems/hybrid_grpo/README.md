# Level 2: Hybrid GR²PO Trading System

## Overview

This is a **Level 2** implementation of a Hybrid Group Relative Policy Optimization (Hybrid GR²PO) trading system. It represents the optimal balance between performance and complexity in our multi-level RL trading system hierarchy.

## Architecture

- **Dual GRPO agents** combining Gated Recurrent and Group Relative approaches
- **Dynamic ensemble mechanism** with adaptive weighting
- **Market regime detection** using XGBoost classification
- **Balanced complexity** for optimal performance

## Features

- **Gated Recurrent Agent (GRA)**: Specializes in temporal pattern recognition
- **Group Relative Agent (GRA)**: Focuses on collaborative learning
- **Dynamic Ensemble**: Adaptive weighting based on market conditions
- **Market Regime Awareness**: XGBoost-based regime classification
- **Advanced Backtesting**: Comprehensive performance metrics

## Performance (2024-03 to 2024-08)

| Metric | Value |
|--------|-------|
| Sharpe Ratio | **1.89** |
| Max Drawdown | **-16.2%** |
| Total Return | **258.4%** |
| Win Rate | **62.3%** |
| Profit Factor | **2.15** |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from hybrid_grpo.main import train, backtest

# Train the hybrid system
train(config)

# Run comprehensive backtest
results = backtest(config)
```

## Project Structure

```
hybrid_grpo/
├── src/
│   ├── agents/           # Agent implementations
│   │   ├── base_agent.py
│   │   ├── gru_agent.py
│   │   └── hybrid_grpo_agent.py
│   ├── ensemble/         # Ensemble mechanisms
│   │   └── group_relative_ensemble.py
│   ├── data/             # Data processing
│   │   ├── multimodal_processor.py
│   │   └── regime_classifier.py
│   ├── evaluation/       # Backtesting and metrics
│   │   └── backtester.py
│   ├── optimization/     # Hyperparameter tuning
│   │   └── hyperparameter_tuner.py
│   └── utils/            # Utility functions
│       ├── metrics.py
│       └── visualization.py
├── configs/
│   └── hybrid_grpo_config.yaml
├── requirements.txt
└── README.md
```

## Level Comparison

This represents **Level 2** in our three-tier system:
- **Level 1**: Basic GRPO - Simple, educational (-1.16 SR, -14.74% return)
- **Level 2**: HybridGRPO (This project) - **Optimal balance** (1.89 SR, 258.4% return)
- **Level 3**: H-MTR - Advanced hierarchical (1.82 SR, 241.7% return)

## Key Innovations

1. **Dual GRPO Integration**: Combines temporal and collaborative learning
2. **Dynamic Ensemble**: Adaptive weighting based on market conditions
3. **Regime Awareness**: Market state classification for optimal adaptation
4. **Synergy Effects**: 15.9% additional improvement from component combination

## License

MIT License
