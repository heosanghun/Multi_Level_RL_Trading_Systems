"""
Main PPO Baseline Class

This module integrates all components to create a complete PPO Baseline trading system:
1. Data processing and multimodal fusion
2. Market regime detection
3. PPO agent training and inference
4. Trading environment management
5. Performance evaluation and backtesting
"""

import numpy as np
import pandas as pd
import torch
import yaml
import logging
import os
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

from .ppo_agent import PPOAgent
from .multimodal_fusion import MultimodalFusion
from .regime_detector import MarketRegimeDetector
from .data_processor import DataProcessor
from .trading_env import TradingEnvironment

logger = logging.getLogger(__name__)

class PPOBaseline:
    """
    Main PPO Baseline trading system
    
    Integrates all components for a complete multimodal reinforcement learning
    trading system with market regime detection.
    """
    
    def __init__(self, config_path: str = None, config: Dict[str, Any] = None):
        """
        Initialize PPO Baseline system
        
        Args:
            config_path: Path to configuration file
            config: Configuration dictionary
        """
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif config:
            self.config = config
        else:
            raise ValueError("Either config_path or config must be provided")
        
        # Set up logging
        self._setup_logging()
        
        # Initialize components
        self.data_processor = DataProcessor(self.config)
        self.multimodal_fusion = MultimodalFusion(self.config)
        self.regime_detector = MarketRegimeDetector(self.config)
        self.trading_env = TradingEnvironment(self.config)
        
        # Initialize PPO agent
        self.ppo_agent = PPOAgent(
            state_dim=self.config['model']['total_state_dim'],
            action_dim=self.config['model']['actions'],
            config=self.config['training']
        )
        
        # Data storage
        self.ohlcv_data = None
        self.sentiment_data = None
        self.processed_data = None
        
        # Training state
        self.is_trained = False
        self.training_history = []
        
        logger.info("PPO Baseline system initialized successfully")
    
    def _setup_logging(self):
        """Set up logging configuration"""
        log_level = self.config['logging']['level']
        log_file = self.config['logging']['log_file']
        
        # Create logs directory
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def load_data(self, ohlcv_data: pd.DataFrame = None, sentiment_data: pd.DataFrame = None,
                  symbol: str = 'BTC-USD', start_date: str = None, end_date: str = None):
        """
        Load trading data into the system
        
        Args:
            ohlcv_data: Pre-loaded OHLCV data
            sentiment_data: Pre-loaded sentiment data
            symbol: Trading symbol for data loading
            start_date: Start date for data loading
            end_date: End date for data loading
        """
        if ohlcv_data is not None:
            self.ohlcv_data = ohlcv_data
            logger.info(f"Using pre-loaded OHLCV data: {len(self.ohlcv_data)} records")
        else:
            # Load data from source
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            self.ohlcv_data = self.data_processor.load_ohlcv_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval='1h'
            )
            logger.info(f"Loaded OHLCV data: {len(self.ohlcv_data)} records")
        
        if sentiment_data is not None:
            self.sentiment_data = sentiment_data
            logger.info(f"Using pre-loaded sentiment data: {len(self.sentiment_data)} records")
        else:
            # Try to load sentiment data
            try:
                self.sentiment_data = self.data_processor.load_sentiment_data(
                    keywords=self.config['data_sources']['news_keywords'],
                    start_date=start_date or (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                    end_date=end_date or datetime.now().strftime('%Y-%m-%d')
                )
                logger.info(f"Loaded sentiment data: {len(self.sentiment_data)} records")
            except Exception as e:
                logger.warning(f"Could not load sentiment data: {e}")
                self.sentiment_data = pd.DataFrame()
        
        # Preprocess data
        self.ohlcv_data, self.sentiment_data = self.data_processor.preprocess_data(
            self.ohlcv_data, self.sentiment_data
        )
        
        # Load data into trading environment
        self.trading_env.load_data(self.ohlcv_data, self.sentiment_data)
        
        logger.info("Data loading completed")
    
    def train_regime_detector(self):
        """Train the market regime detection model"""
        if self.ohlcv_data is None:
            raise ValueError("No OHLCV data loaded")
        
        logger.info("Training market regime detection model...")
        
        # Train ML-based regime detector
        performance = self.regime_detector.train_ml_model(self.ohlcv_data)
        
        if performance:
            logger.info(f"Regime detector training completed. Accuracy: {performance['accuracy']:.4f}")
        else:
            logger.warning("Regime detector training failed")
        
        return performance
    
    def train_ppo_agent(self, episodes: int = None, batch_size: int = None):
        """
        Train the PPO agent
        
        Args:
            episodes: Number of training episodes
            batch_size: Training batch size
        """
        if self.ohlcv_data is None:
            raise ValueError("No data loaded")
        
        # Use config values if not specified
        episodes = episodes or self.config['training']['episodes']
        batch_size = batch_size or self.config['training']['batch_size']
        
        logger.info(f"Starting PPO agent training for {episodes} episodes...")
        
        # Training loop
        for episode in range(episodes):
            # Reset environment
            state = self.trading_env.reset()
            episode_reward = 0
            episode_steps = 0
            
            # Episode loop
            while True:
                # Select action
                action, action_prob, state_value = self.ppo_agent.select_action(state)
                
                # Execute action
                next_state, reward, done, info = self.trading_env.step(action)
                
                # Store experience
                self.ppo_agent.store_experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    action_prob=action_prob,
                    state_value=state_value,
                    done=done
                )
                
                # Update state
                state = next_state
                episode_reward += reward
                episode_steps += 1
                
                # Check if episode is done
                if done:
                    break
            
            # Update agent after episode
            if len(self.ppo_agent.memory) >= batch_size:
                update_stats = self.ppo_agent.update(batch_size=batch_size)
                self.training_history.append({
                    'episode': episode,
                    'reward': episode_reward,
                    'steps': episode_steps,
                    'update_stats': update_stats
                })
            
            # Log progress
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean([h['reward'] for h in self.training_history[-100:]])
                logger.info(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.4f}")
            
            # Save checkpoint
            if (episode + 1) % self.config['training']['save_interval'] == 0:
                self.save_model(f"models/checkpoints/ppo_baseline_episode_{episode + 1}.joblib")
        
        self.is_trained = True
        logger.info("PPO agent training completed")
        
        return self.training_history
    
    def evaluate(self, test_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Evaluate the trained system
        
        Args:
            test_data: Test data (uses training data if not provided)
            
        Returns:
            Dictionary with evaluation results
        """
        if not self.is_trained:
            raise ValueError("Agent must be trained before evaluation")
        
        # Use test data or training data
        eval_data = test_data if test_data is not None else self.ohlcv_data
        
        logger.info("Starting system evaluation...")
        
        # Create evaluation environment
        eval_env = TradingEnvironment(self.config)
        eval_env.load_data(eval_data, self.sentiment_data)
        
        # Run evaluation
        state = eval_env.reset()
        total_reward = 0
        total_steps = 0
        
        while True:
            # Select action (deterministic for evaluation)
            action, _, _ = self.ppo_agent.select_action(state, deterministic=True)
            
            # Execute action
            next_state, reward, done, info = eval_env.step(action)
            
            total_reward += reward
            total_steps += 1
            state = next_state
            
            if done:
                break
        
        # Get performance metrics
        performance_metrics = eval_env.get_performance_metrics()
        
        # Add evaluation-specific metrics
        evaluation_results = {
            'total_reward': total_reward,
            'total_steps': total_steps,
            'avg_reward_per_step': total_reward / total_steps if total_steps > 0 else 0,
            'performance_metrics': performance_metrics
        }
        
        logger.info(f"Evaluation completed. Total Reward: {total_reward:.4f}")
        
        return evaluation_results
    
    def backtest(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        Run backtesting on historical data
        
        Args:
            start_date: Start date for backtesting
            end_date: End date for backtesting
            
        Returns:
            Dictionary with backtesting results
        """
        if not self.is_trained:
            raise ValueError("Agent must be trained before backtesting")
        
        # Filter data for backtesting period
        if start_date or end_date:
            mask = pd.Series(True, index=self.ohlcv_data.index)
            
            if start_date:
                mask &= self.ohlcv_data.index >= start_date
            if end_date:
                mask &= self.ohlcv_data.index <= end_date
            
            backtest_data = self.ohlcv_data[mask]
        else:
            backtest_data = self.ohlcv_data
        
        logger.info(f"Starting backtesting on {len(backtest_data)} records...")
        
        # Run backtesting
        results = self.evaluate(backtest_data)
        
        # Add regime analysis
        regime_results = self._analyze_regime_performance(backtest_data)
        results['regime_analysis'] = regime_results
        
        logger.info("Backtesting completed")
        
        return results
    
    def _analyze_regime_performance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance by market regime"""
        if not self.regime_detector.is_trained:
            logger.warning("Regime detector not trained, skipping regime analysis")
            return {}
        
        # Detect regimes for each data point
        regimes = []
        for i in range(len(data)):
            current_data = data.iloc[:i+1]
            regime_info = self.regime_detector.detect_regime(current_data, method='ml')
            regimes.append(regime_info['regime'])
        
        # Add regime information to data
        data_with_regime = data.copy()
        data_with_regime['regime'] = regimes
        
        # Calculate returns
        data_with_regime['returns'] = data_with_regime['close'].pct_change()
        
        # Group by regime and calculate statistics
        regime_stats = {}
        for regime in ['bull', 'bear', 'sideways']:
            regime_data = data_with_regime[data_with_regime['regime'] == regime]
            
            if len(regime_data) > 0:
                regime_stats[regime] = {
                    'count': len(regime_data),
                    'avg_return': regime_data['returns'].mean(),
                    'volatility': regime_data['returns'].std(),
                    'sharpe_ratio': regime_data['returns'].mean() / regime_data['returns'].std() if regime_data['returns'].std() > 0 else 0
                }
        
        return regime_stats
    
    def predict(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Make trading prediction for current market data
        
        Args:
            current_data: Current market data
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            raise ValueError("Agent must be trained before making predictions")
        
        # Detect current regime
        regime_info = self.regime_detector.detect_regime(current_data)
        
        # Get current state
        current_state = self.multimodal_fusion.process_data(current_data, self.sentiment_data)
        
        # Get action from agent
        action, action_prob, state_value = self.ppo_agent.select_action(current_state, deterministic=True)
        
        # Get action details
        action_name = self.trading_env.action_mapping[action]
        
        prediction = {
            'action': action,
            'action_name': action_name,
            'action_probability': action_prob,
            'state_value': state_value,
            'market_regime': regime_info['regime'],
            'regime_confidence': regime_info['confidence'],
            'timestamp': datetime.now()
        }
        
        return prediction
    
    def save_model(self, filepath: str):
        """Save the complete model"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save PPO agent
        agent_path = filepath.replace('.joblib', '_agent.joblib')
        self.ppo_agent.save_model(agent_path)
        
        # Save regime detector
        regime_path = filepath.replace('.joblib', '_regime.joblib')
        self.regime_detector.save_model(regime_path)
        
        # Save system state
        system_state = {
            'config': self.config,
            'training_history': self.training_history,
            'is_trained': self.is_trained,
            'data_info': {
                'ohlcv_shape': self.ohlcv_data.shape if self.ohlcv_data is not None else None,
                'sentiment_shape': self.sentiment_data.shape if self.sentiment_data is not None else None
            }
        }
        
        import joblib
        joblib.dump(system_state, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the complete model"""
        import joblib
        
        # Load system state
        system_state = joblib.load(filepath)
        self.config = system_state['config']
        self.training_history = system_state['training_history']
        self.is_trained = system_state['is_trained']
        
        # Load PPO agent
        agent_path = filepath.replace('.joblib', '_agent.joblib')
        self.ppo_agent.load_model(agent_path)
        
        # Load regime detector
        regime_path = filepath.replace('.joblib', '_regime.joblib')
        self.regime_detector.load_model(regime_path)
        
        logger.info(f"Model loaded from {filepath}")
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history"""
        if not self.training_history:
            logger.warning("No training history to plot")
            return
        
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract data
        episodes = [h['episode'] for h in self.training_history]
        rewards = [h['reward'] for h in self.training_history]
        steps = [h['steps'] for h in self.training_history]
        
        # Plot rewards
        ax1.plot(episodes, rewards, 'b-', alpha=0.7)
        ax1.set_title('Training Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)
        
        # Plot steps
        ax2.plot(episodes, steps, 'g-', alpha=0.7)
        ax2.set_title('Episode Steps')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.grid(True, alpha=0.3)
        
        # Plot moving average rewards
        if len(rewards) > 10:
            window = min(100, len(rewards) // 10)
            moving_avg = pd.Series(rewards).rolling(window=window).mean()
            ax3.plot(episodes, rewards, 'b-', alpha=0.3, label='Raw')
            ax3.plot(episodes, moving_avg, 'r-', linewidth=2, label=f'{window}-Episode Moving Average')
            ax3.set_title('Moving Average Rewards')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Reward')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot update statistics if available
        if self.training_history and 'update_stats' in self.training_history[0]:
            policy_losses = [h['update_stats'].get('policy_loss', 0) for h in self.training_history if h.get('update_stats')]
            if policy_losses:
                ax4.plot(policy_losses, 'm-', alpha=0.7)
                ax4.set_title('Policy Loss')
                ax4.set_xlabel('Update Step')
                ax4.set_ylabel('Loss')
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        info = {
            'system_name': 'PPO Baseline',
            'version': '1.0.0',
            'is_trained': self.is_trained,
            'config_summary': {
                'model_dimensions': self.config['model'],
                'training_parameters': self.config['training'],
                'trading_parameters': self.config['trading']
            },
            'data_info': {
                'ohlcv_records': len(self.ohlcv_data) if self.ohlcv_data is not None else 0,
                'sentiment_records': len(self.sentiment_data) if self.sentiment_data is not None else 0
            },
            'training_info': {
                'total_episodes': len(self.training_history),
                'last_training_date': self.training_history[-1]['episode'] if self.training_history else None
            },
            'component_status': {
                'data_processor': 'Initialized',
                'multimodal_fusion': 'Initialized',
                'regime_detector': 'Trained' if self.regime_detector.is_trained else 'Not Trained',
                'ppo_agent': 'Trained' if self.is_trained else 'Not Trained',
                'trading_env': 'Initialized'
            }
        }
        
        return info
