# PPO Baseline: Static Multimodal Reinforcement Learning Trading System

## Overview

This project implements the PPO Baseline system as described in the research paper "Comparative Analysis of Multi-Level Reinforcement Learning Trading Systems". It represents the foundation level (Level 0) that combines multimodal data integration with PPO-based reinforcement learning for cryptocurrency trading.

## Architecture

### Core Components

1. **Multimodal Data Processing**
   - **Visual Features**: Candlestick chart pattern recognition using ResNet-18 CNN
   - **Technical Features**: 15 key technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
   - **Sentiment Features**: News sentiment analysis using DeepSeek-R1 (32B) model

2. **Market Regime Classification**
   - XGBoost-based classifier for market state identification
   - Three regimes: Bull Market, Bear Market, Sideways Market
   - EMA-based regime labeling (20, 50, 200 periods)

3. **PPO Reinforcement Learning**
   - Actor-Critic architecture with MLP networks
   - 273-dimensional state space (256 visual + 15 technical + 2 sentiment)
   - 5 action space (Strong Buy, Buy, Hold, Sell, Strong Sell)

## Performance Metrics

- **Sharpe Ratio**: 1.35
- **Maximum Drawdown**: -24.8%
- **Cumulative Return**: 185.2%
- **Win Rate**: 56.1%
- **Profit Factor**: 1.78

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

```python
from ppo_baseline import PPOBaseline

# Initialize the system
ppo_system = PPOBaseline(
    initial_capital=10000,
    learning_rate=3e-4,
    batch_size=64
)

# Load and preprocess data
ppo_system.load_data(
    candlestick_data=candlestick_df,
    technical_data=technical_df,
    sentiment_data=sentiment_df
)

# Train the model
ppo_system.train(episodes=1000)

# Run backtesting
results = ppo_system.backtest()
```

### Market Regime Detection

```python
# Detect current market regime
regime = ppo_system.detect_regime()

# Get regime-specific performance
regime_performance = ppo_system.get_regime_performance()
```

## Project Structure

```
ppo_baseline/
├── src/
│   ├── __init__.py
│   ├── ppo_agent.py          # PPO agent implementation
│   ├── multimodal_fusion.py  # Data fusion mechanisms
│   ├── regime_detector.py    # Market regime classification
│   ├── data_processor.py     # Data preprocessing
│   └── trading_env.py        # Trading environment
├── configs/
│   └── config.yaml           # Configuration parameters
├── data/                     # Data storage
├── models/                   # Trained model checkpoints
├── utils/                    # Utility functions
├── tests/                    # Unit tests
├── docs/                     # Documentation
├── requirements.txt           # Python dependencies
└── main.py                   # Main execution script
```

## Configuration

Key hyperparameters in `configs/config.yaml`:

```yaml
training:
  learning_rate: 3e-4
  batch_size: 64
  discount_factor: 0.99
  episodes: 1000

model:
  visual_features: 256
  technical_features: 15
  sentiment_features: 2
  hidden_layers: [128, 64]
  actions: 5

data:
  candlestick_window: 60
  sentiment_window: 24
  chart_resolution: [224, 224]
```

## Data Requirements

### Input Data Format

1. **Candlestick Data**: OHLCV data with datetime index
2. **Technical Indicators**: Pre-calculated technical indicators
3. **News Sentiment**: Sentiment scores (-1 to +1) with timestamps

### Data Sources

- **Price Data**: Binance API, yfinance, or custom data
- **News Data**: NewsAPI, Cointelegraph, Coindesk
- **Technical Indicators**: Calculated from OHLCV data

## Limitations

1. **Static Policy**: Single policy approach across all market conditions
2. **Poor Regime Adaptation**: Inability to adjust strategy based on market state
3. **Limited Temporal Learning**: No sequential pattern recognition
4. **Fixed Ensemble Weights**: No dynamic adaptation to market changes

## Future Improvements

1. **Dynamic Policy Adaptation**: Adaptive strategy based on market conditions
2. **Temporal Learning**: Integration of sequential pattern recognition
3. **Regime-Specific Optimization**: Tailored strategies for different market states
4. **Adaptive Ensemble**: Dynamic weighting based on performance

## Citation

If you use this code in your research, please cite:

```bibtex
@article{heo2025comparative,
  title={Comparative Analysis of Multi-Level Reinforcement Learning Trading Systems: From PPO Baseline to Advanced Hybrid Approaches},
  author={Heo, Sanghun and Hwang, Yongbae},
  journal={Working Paper},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.

## Contact

For questions and support, please contact the research team.
