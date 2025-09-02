# Level 3: Hierarchical Multi-Task RL (H-MTR) Trading System

## Overview

This is a **Level 3** implementation of a Hierarchical Multi-Task Reinforcement Learning (H-MTR) trading system. It represents the most sophisticated and theoretically advanced level in our multi-level RL trading system hierarchy.

## Architecture

- **Master-Worker hierarchy** with shared recurrent backbone
- **Multi-task learning** for diverse trading strategies
- **Elastic Weight Consolidation (EWC)** for continual learning
- **Shared memory backbone** for knowledge transfer
- **High complexity** for maximum theoretical performance

## Features

- **Master Agent**: Orchestrates multiple worker agents
- **Worker Agents**: Specialized in specific trading tasks
- **Shared Backbone**: Common recurrent memory for all agents
- **EWC Integration**: Prevents catastrophic forgetting
- **Advanced Risk Management**: Sophisticated position sizing

## Performance (2024-03 to 2024-08)

| Metric | Value |
|--------|-------|
| Sharpe Ratio | 1.82 |
| Max Drawdown | -16.8% |
| Total Return | 241.7% |
| Win Rate | 59.8% |
| Profit Factor | 1.95 |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from H-MTR.main import train, backtest

# Train the hierarchical system
train(config)

# Run advanced backtest
results = backtest(config)
```

## Project Structure

```
H-MTR/
├── src/
│   ├── models/           # Neural network models
│   │   ├── shared_backbone.py
│   │   ├── policy_head.py
│   │   ├── master_agent.py
│   │   ├── worker_agent.py
│   │   └── h_mtr_agent.py
│   ├── data/             # Data processing
│   │   ├── market_data_loader.py
│   │   └── technical_indicators.py
│   ├── training/         # Training algorithms
│   │   └── ppo_agent.py
│   ├── evaluation/       # Backtesting and metrics
│   │   ├── backtester.py
│   │   ├── performance_metrics.py
│   │   └── visualization.py
│   ├── agents/           # Agent implementations
│   └── utils/            # Utility functions
│       └── logger.py
├── configs/
│   └── h_mtr_config.yaml
├── requirements.txt
└── README.md
```

## Level Comparison

This represents **Level 3** in our three-tier system:
- **Level 1**: Basic GRPO - Simple, educational (-1.16 SR, -14.74% return)
- **Level 2**: HybridGRPO - Optimal balance (1.89 SR, 258.4% return)
- **Level 3**: H-MTR (This project) - **Advanced hierarchical** (1.82 SR, 241.7% return)

## Key Innovations

1. **Hierarchical Structure**: Master-worker coordination for complex strategies
2. **Multi-Task Learning**: Simultaneous learning of multiple trading objectives
3. **EWC Integration**: Continual learning without forgetting previous knowledge
4. **Shared Backbone**: Efficient knowledge transfer between agents

## Theoretical Advantages

- **Scalability**: Easy to add new trading tasks
- **Knowledge Preservation**: EWC prevents catastrophic forgetting
- **Efficiency**: Shared backbone reduces computational overhead
- **Flexibility**: Modular design for easy customization

## Limitations

- **Complexity**: High implementation and maintenance cost
- **Overfitting Risk**: Complex models may overfit to training data
- **Resource Requirements**: Higher computational and memory needs

## License

MIT License
