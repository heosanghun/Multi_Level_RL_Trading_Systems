# Level 1: Basic GRPO Trading System

## Overview

This is a **Level 1** implementation of a Group Relative Policy Optimization (GRPO) trading system. It represents the foundational level of our multi-level RL trading system hierarchy.

## Architecture

- **Single GRU-based agent** with PPO optimization
- **Basic temporal pattern recognition** using Gated Recurrent Units
- **Simple architecture** suitable for educational and baseline purposes

## Features

- GRU-based policy network for temporal dependencies
- PPO algorithm implementation
- Basic backtesting framework
- YFinance data integration
- Simple performance metrics

## Performance (2024-03 to 2024-08)

| Metric | Value |
|--------|-------|
| Sharpe Ratio | -1.16 |
| Max Drawdown | -20.96% |
| Total Return | -14.74% |
| Data Points | 183 |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from grpo.main import train, backtest

# Train the agent
train(config)

# Run backtest
results = backtest(config)
```

## Project Structure

```
grpo/
├── src/grpo/
│   ├── __init__.py
│   ├── data.py          # Data loading and preprocessing
│   ├── agent.py         # GRPO agent implementation
│   ├── trainer.py       # Training logic
│   ├── backtest.py      # Backtesting framework
│   └── main.py          # Main execution script
├── configs/
│   └── config.yaml      # Configuration file
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Level Comparison

This represents **Level 1** in our three-tier system:
- **Level 1**: Basic GRPO (This project) - Simple, educational
- **Level 2**: HybridGRPO - Balanced performance/complexity
- **Level 3**: H-MTR - Advanced hierarchical approach

## License

MIT License
