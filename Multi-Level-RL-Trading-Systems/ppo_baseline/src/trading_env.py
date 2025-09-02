"""
Trading Environment Module for PPO Baseline

This module implements a gym-compatible trading environment for reinforcement learning
with realistic trading constraints and reward functions.
"""

import gym
import numpy as np
import pandas as pd
from gym import spaces
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class TradingEnvironment(gym.Env):
    """
    Gym-compatible trading environment for cryptocurrency trading
    
    Features:
    - Realistic trading constraints (transaction costs, slippage)
    - Multiple action types (Strong Buy, Buy, Hold, Sell, Strong Sell)
    - Risk management (stop loss, take profit)
    - Performance tracking and metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(TradingEnvironment, self).__init__()
        
        self.config = config
        
        # Trading parameters
        self.initial_capital = config['trading']['initial_capital']
        self.transaction_cost = config['trading']['transaction_cost']
        self.slippage = config['trading']['slippage']
        self.max_position_size = config['trading']['max_position_size']
        
        # Risk management
        self.stop_loss = config['trading']['stop_loss']
        self.take_profit = config['trading']['take_profit']
        self.max_drawdown = config['trading']['max_drawdown']
        
        # Environment state
        self.current_step = 0
        self.current_capital = self.initial_capital
        self.current_position = 0  # Current position size
        self.entry_price = 0  # Entry price for current position
        self.total_trades = 0
        self.winning_trades = 0
        
        # Performance tracking
        self.capital_history = [self.initial_capital]
        self.position_history = [0]
        self.trade_history = []
        self.reward_history = []
        
        # Data
        self.ohlcv_data = None
        self.sentiment_data = None
        self.current_state = None
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(5)  # 5 actions
        
        # State space will be set when data is loaded
        self.observation_space = None
        
        # Action mapping
        self.action_mapping = {
            0: 'strong_buy',
            1: 'buy', 
            2: 'hold',
            3: 'sell',
            4: 'strong_sell'
        }
        
        # Action multipliers for position sizing
        self.action_multipliers = {
            'strong_buy': 1.0,  # Full position
            'buy': 0.5,         # Half position
            'hold': 0.0,        # No change
            'sell': -0.5,       # Half position
            'strong_sell': -1.0 # Full position
        }
        
        logger.info(f"Trading environment initialized with capital: ${self.initial_capital:,.2f}")
    
    def load_data(self, ohlcv_data: pd.DataFrame, sentiment_data: pd.DataFrame = None):
        """
        Load trading data into the environment
        
        Args:
            ohlcv_data: OHLCV price data
            sentiment_data: Sentiment data (optional)
        """
        self.ohlcv_data = ohlcv_data.copy()
        self.sentiment_data = sentiment_data.copy() if sentiment_data is not None else None
        
        # Set observation space based on data dimensions
        # This will be set by the multimodal fusion module
        # For now, we'll use a placeholder
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(273,),  # 256 visual + 15 technical + 2 sentiment
            dtype=np.float32
        )
        
        logger.info(f"Data loaded: {len(self.ohlcv_data)} OHLCV records")
        if self.sentiment_data is not None:
            logger.info(f"Sentiment data: {len(self.sentiment_data)} records")
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state
        
        Returns:
            Initial observation
        """
        self.current_step = 0
        self.current_capital = self.initial_capital
        self.current_position = 0
        self.entry_price = 0
        self.total_trades = 0
        self.winning_trades = 0
        
        # Reset history
        self.capital_history = [self.initial_capital]
        self.position_history = [0]
        self.trade_history = []
        self.reward_history = []
        
        # Get initial state
        if self.ohlcv_data is not None and len(self.ohlcv_data) > 0:
            self.current_state = self._get_state(self.current_step)
        else:
            self.current_state = np.zeros(273)
        
        logger.info("Environment reset")
        return self.current_state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment
        
        Args:
            action: Action to take (0-4)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.ohlcv_data is None or len(self.ohlcv_data) == 0:
            raise ValueError("No data loaded in environment")
        
        if self.current_step >= len(self.ohlcv_data) - 1:
            # Episode is done
            done = True
            reward = 0
            info = self._get_info()
            return self.current_state, reward, done, info
        
        # Execute action
        reward = self._execute_action(action)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.ohlcv_data) - 1
        
        # Get new state
        if not done:
            self.current_state = self._get_state(self.current_step)
        
        # Update history
        self.capital_history.append(self.current_capital)
        self.position_history.append(self.current_position)
        self.reward_history.append(reward)
        
        # Get info
        info = self._get_info()
        
        return self.current_state, reward, done, info
    
    def _execute_action(self, action: int) -> float:
        """
        Execute the given action and return reward
        
        Args:
            action: Action index (0-4)
            
        Returns:
            Reward for the action
        """
        action_name = self.action_mapping[action]
        action_multiplier = self.action_multipliers[action_name]
        
        # Get current price
        current_price = self.ohlcv_data.iloc[self.current_step]['close']
        
        # Calculate position change
        if action_multiplier > 0:  # Buying
            # Calculate how much to buy
            max_position_value = self.current_capital * self.max_position_size
            target_position_value = max_position_value * action_multiplier
            
            # Calculate shares to buy
            shares_to_buy = target_position_value / current_price
            
            # Apply transaction costs and slippage
            total_cost = shares_to_buy * current_price * (1 + self.transaction_cost + self.slippage)
            
            if total_cost <= self.current_capital:
                # Execute buy
                self.current_position += shares_to_buy
                self.current_capital -= total_cost
                self.entry_price = current_price
                
                if action_name == 'strong_buy':
                    self.total_trades += 1
                
                logger.debug(f"Bought {shares_to_buy:.4f} shares at ${current_price:.2f}")
            else:
                # Insufficient capital
                logger.debug(f"Insufficient capital for {action_name}")
        
        elif action_multiplier < 0:  # Selling
            if self.current_position > 0:
                # Calculate how much to sell
                shares_to_sell = abs(self.current_position * action_multiplier)
                shares_to_sell = min(shares_to_sell, self.current_position)
                
                # Apply transaction costs and slippage
                sell_value = shares_to_sell * current_price * (1 - self.transaction_cost - self.slippage)
                
                # Execute sell
                self.current_position -= shares_to_sell
                self.current_capital += sell_value
                
                # Calculate profit/loss
                if self.entry_price > 0:
                    profit_loss = (current_price - self.entry_price) * shares_to_sell
                    if profit_loss > 0:
                        self.winning_trades += 1
                    
                    # Record trade
                    trade_info = {
                        'step': self.current_step,
                        'action': action_name,
                        'entry_price': self.entry_price,
                        'exit_price': current_price,
                        'shares': shares_to_sell,
                        'profit_loss': profit_loss,
                        'capital': self.current_capital
                    }
                    self.trade_history.append(trade_info)
                
                if action_name == 'strong_sell':
                    self.total_trades += 1
                
                logger.debug(f"Sold {shares_to_sell:.4f} shares at ${current_price:.2f}")
            else:
                logger.debug(f"No position to sell")
        
        # Calculate reward
        reward = self._calculate_reward()
        
        return reward
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward based on current performance
        
        Returns:
            Reward value
        """
        # Get current portfolio value
        current_price = self.ohlcv_data.iloc[self.current_step]['close']
        portfolio_value = self.current_capital + (self.current_position * current_price)
        
        # Calculate return
        if self.current_step > 0:
            prev_portfolio_value = self.capital_history[-1] + (self.position_history[-1] * 
                                                             self.ohlcv_data.iloc[self.current_step - 1]['close'])
            return_rate = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
        else:
            return_rate = 0
        
        # Risk-adjusted reward
        # Penalize large positions (risk)
        position_penalty = -abs(self.current_position * current_price / portfolio_value) * 0.1
        
        # Penalize frequent trading (transaction costs)
        trading_penalty = -len(self.trade_history) * 0.001
        
        # Combine rewards
        reward = return_rate + position_penalty + trading_penalty
        
        return reward
    
    def _get_state(self, step: int) -> np.ndarray:
        """
        Get state representation for the given step
        
        Args:
            step: Current step index
            
        Returns:
            State vector
        """
        if self.ohlcv_data is None or step >= len(self.ohlcv_data):
            return np.zeros(273)
        
        # This is a placeholder - in practice, the multimodal fusion module
        # would process the data and return the actual state vector
        # For now, we'll create a simple state representation
        
        current_data = self.ohlcv_data.iloc[step]
        
        # Simple state: [price_features, position_features, capital_features]
        price_features = np.array([
            current_data['open'],
            current_data['high'], 
            current_data['low'],
            current_data['close'],
            current_data['volume']
        ])
        
        # Normalize price features
        if price_features.max() != price_features.min():
            price_features = (price_features - price_features.min()) / (price_features.max() - price_features.min())
        
        # Position and capital features
        position_features = np.array([
            self.current_position,
            self.current_capital / self.initial_capital,
            len(self.trade_history) / 100  # Normalized trade count
        ])
        
        # Combine features (this is simplified - the actual state would be 273-dimensional)
        state = np.concatenate([price_features, position_features])
        
        # Pad to required dimension (this is just for compatibility)
        if len(state) < 273:
            state = np.pad(state, (0, 273 - len(state)), 'constant')
        else:
            state = state[:273]
        
        return state.astype(np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get current environment information"""
        if self.ohlcv_data is None or self.current_step >= len(self.ohlcv_data):
            return {}
        
        current_price = self.ohlcv_data.iloc[self.current_step]['close']
        portfolio_value = self.current_capital + (self.current_position * current_price)
        
        # Calculate metrics
        total_return = (portfolio_value - self.initial_capital) / self.initial_capital
        
        # Calculate drawdown
        peak_capital = max(self.capital_history)
        current_drawdown = (peak_capital - portfolio_value) / peak_capital if peak_capital > 0 else 0
        
        # Win rate
        win_rate = self.winning_trades / max(self.total_trades, 1)
        
        info = {
            'step': self.current_step,
            'current_price': current_price,
            'current_capital': self.current_capital,
            'current_position': self.current_position,
            'portfolio_value': portfolio_value,
            'total_return': total_return,
            'current_drawdown': current_drawdown,
            'total_trades': self.total_trades,
            'win_rate': win_rate,
            'entry_price': self.entry_price
        }
        
        return info
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get comprehensive performance metrics"""
        if not self.capital_history:
            return {}
        
        # Calculate returns
        returns = np.diff(self.capital_history) / self.capital_history[:-1]
        
        # Sharpe ratio (simplified)
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(self.capital_history)
        drawdown = (peak - self.capital_history) / peak
        max_drawdown = drawdown.max()
        
        # Total return
        total_return = (self.capital_history[-1] - self.initial_capital) / self.initial_capital
        
        # Volatility
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        
        metrics = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'final_capital': self.capital_history[-1],
            'initial_capital': self.initial_capital
        }
        
        return metrics
    
    def render(self, mode: str = 'human'):
        """Render the current state (for visualization)"""
        if mode == 'human':
            # Simple text output
            info = self._get_info()
            print(f"Step: {info.get('step', 0)}")
            print(f"Price: ${info.get('current_price', 0):.2f}")
            print(f"Capital: ${info.get('current_capital', 0):.2f}")
            print(f"Position: {info.get('current_position', 0):.4f}")
            print(f"Portfolio Value: ${info.get('portfolio_value', 0):.2f}")
            print(f"Total Return: {info.get('total_return', 0):.2%}")
        elif mode == 'rgb_array':
            # Return numpy array for plotting
            # This would be implemented for visualization
            pass
    
    def close(self):
        """Clean up resources"""
        pass
    
    def seed(self, seed: Optional[int] = None):
        """Set random seed for reproducibility"""
        np.random.seed(seed)
        return [seed]
