#!/usr/bin/env python3
"""
Main execution script for PPO Baseline Trading System

This script demonstrates the complete workflow:
1. Data loading and preprocessing
2. Market regime detection training
3. PPO agent training
4. Evaluation and backtesting
5. Performance analysis
"""

import os
import sys
import argparse
import logging
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from ppo_baseline import PPOBaseline
except ImportError:
    # Fallback to direct import
    from src.ppo_baseline import PPOBaseline

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ppo_baseline.log')
        ]
    )

def load_config(config_path: str = None) -> dict:
    """Load configuration from file or use defaults"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded from {config_path}")
    else:
        # Default configuration
        config = {
            'training': {
                'learning_rate': 3e-4,
                'batch_size': 64,
                'discount_factor': 0.99,
                'episodes': 1000,
                'max_steps_per_episode': 1000,
                'save_interval': 100,
                'eval_interval': 50
            },
            'model': {
                'visual_features': 256,
                'technical_features': 15,
                'sentiment_features': 2,
                'total_state_dim': 273,
                'hidden_layers': [128, 64],
                'actions': 5
            },
            'data': {
                'candlestick_window': 60,
                'chart_resolution': [224, 224],
                'sentiment_window': 24
            },
            'regime': {
                'ema_periods': [20, 50, 200],
                'xgb_params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1
                },
                'regimes': ['bull', 'bear', 'sideways']
            },
            'trading': {
                'initial_capital': 10000,
                'transaction_cost': 0.0005,
                'slippage': 0.0002,
                'max_position_size': 0.2,
                'stop_loss': 0.05,
                'take_profit': 0.10,
                'max_drawdown': 0.30
            },
            'data_sources': {
                'price_source': 'yfinance',
                'news_keywords': ['Bitcoin', 'BTC', 'cryptocurrency'],
                'sentiment_sources': ['news_api', 'cointelegraph', 'coindesk']
            },
            'logging': {
                'level': 'INFO',
                'log_file': 'logs/ppo_baseline.log'
            }
        }
        logging.info("Using default configuration")
    
    return config

def create_sample_data(days: int = 30) -> tuple:
    """Create sample data for testing"""
    logging.info(f"Creating sample data for {days} days...")
    
    # Generate sample OHLCV data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='1H')
    
    # Generate synthetic price data
    np.random.seed(42)  # For reproducibility
    
    # Start with a base price
    base_price = 50000  # BTC-like price
    prices = [base_price]
    
    for i in range(1, len(date_range)):
        # Random walk with some trend
        change = np.random.normal(0, 0.02)  # 2% volatility
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1000))  # Minimum price of 1000
    
    # Create OHLCV data
    sample_ohlcv = pd.DataFrame({
        'open': prices[:-1],
        'high': [max(p, p * (1 + abs(np.random.normal(0, 0.01)))) for p in prices[:-1]],
        'low': [min(p, p * (1 - abs(np.random.normal(0, 0.01)))) for p in prices[:-1]],
        'close': prices[1:],
        'volume': np.random.uniform(1000, 10000, len(prices) - 1)
    }, index=date_range[:-1])
    
    # Generate sample sentiment data
    sample_sentiment = pd.DataFrame({
        'timestamp': date_range[::4],  # Every 4 hours
        'sentiment_score': np.random.uniform(-1, 1, len(date_range[::4])),
        'source': np.random.choice(['news_api', 'cointelegraph', 'coindesk'], len(date_range[::4]))
    })
    
    logging.info(f"Sample data created: OHLCV {len(sample_ohlcv)} rows, Sentiment {len(sample_sentiment)} rows")
    
    return sample_ohlcv, sample_sentiment

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='PPO Baseline Trading System')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'demo'], default='demo',
                       help='Execution mode')
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--data-days', type=int, default=30, help='Days of data to generate')
    parser.add_argument('--save-model', type=str, help='Path to save trained model')
    parser.add_argument('--load-model', type=str, help='Path to load pre-trained model')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logging.info("PPO Baseline Trading System Starting...")
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Initialize system
        ppo_system = PPOBaseline(config=config)
        
        # Load or create data
        if args.load_model and os.path.exists(args.load_model):
            # Load pre-trained model
            logging.info(f"Loading pre-trained model from {args.load_model}")
            ppo_system.load_model(args.load_model)
            
            # Load sample data for evaluation
            ohlcv_data, sentiment_data = create_sample_data(args.data_days)
            ppo_system.load_data(ohlcv_data, sentiment_data)
            
        else:
            # Create sample data
            ohlcv_data, sentiment_data = create_sample_data(args.data_days)
            ppo_system.load_data(ohlcv_data, sentiment_data)
            
            if args.mode == 'train':
                # Train regime detector
                logging.info("Training market regime detector...")
                regime_performance = ppo_system.train_regime_detector()
                logging.info(f"Regime detector training completed: {regime_performance}")
                
                # Train PPO agent
                logging.info(f"Training PPO agent for {args.episodes} episodes...")
                training_history = ppo_system.train_ppo_agent(episodes=args.episodes)
                logging.info(f"PPO agent training completed. Episodes: {len(training_history)}")
                
                # Save model if requested
                if args.save_model:
                    ppo_system.save_model(args.save_model)
                    logging.info(f"Model saved to {args.save_model}")
        
        # Run evaluation
        logging.info("Running evaluation...")
        evaluation_results = ppo_system.evaluate()
        
        # Display results
        print("\n" + "="*50)
        print("PPO BASELINE EVALUATION RESULTS")
        print("="*50)
        
        print(f"Total Reward: {evaluation_results['total_reward']:.4f}")
        print(f"Total Steps: {evaluation_results['total_steps']}")
        print(f"Average Reward per Step: {evaluation_results['avg_reward_per_step']:.4f}")
        
        # Performance metrics
        perf_metrics = evaluation_results['performance_metrics']
        print(f"\nPerformance Metrics:")
        print(f"  Total Return: {perf_metrics['total_return']:.2%}")
        print(f"  Sharpe Ratio: {perf_metrics['sharpe_ratio']:.4f}")
        print(f"  Max Drawdown: {perf_metrics['max_drawdown']:.2%}")
        print(f"  Volatility: {perf_metrics['volatility']:.2%}")
        print(f"  Total Trades: {perf_metrics['total_trades']}")
        print(f"  Win Rate: {perf_metrics['win_rate']:.2%}")
        
        # Run backtesting
        logging.info("Running backtesting...")
        backtest_results = ppo_system.backtest()
        
        # Regime analysis
        if 'regime_analysis' in backtest_results:
            print(f"\nRegime Analysis:")
            for regime, stats in backtest_results['regime_analysis'].items():
                print(f"  {regime.upper()}:")
                print(f"    Count: {stats['count']}")
                print(f"    Avg Return: {stats['avg_return']:.4f}")
                print(f"    Sharpe Ratio: {stats['sharpe_ratio']:.4f}")
        
        # Plot training history if available
        if hasattr(ppo_system, 'training_history') and ppo_system.training_history:
            logging.info("Plotting training history...")
            ppo_system.plot_training_history()
        
        # System information
        system_info = ppo_system.get_system_info()
        print(f"\nSystem Information:")
        print(f"  System: {system_info['system_name']}")
        print(f"  Version: {system_info['version']}")
        print(f"  Trained: {system_info['is_trained']}")
        print(f"  OHLCV Records: {system_info['data_info']['ohlcv_records']}")
        print(f"  Sentiment Records: {system_info['data_info']['sentiment_records']}")
        
        logging.info("PPO Baseline system execution completed successfully")
        
    except Exception as e:
        logging.error(f"Error during execution: {e}")
        raise

if __name__ == "__main__":
    main()
