"""
Multimodal Fusion Module for PPO Baseline

This module handles the fusion of three types of data:
1. Visual features from candlestick charts (CNN)
2. Technical indicators (numerical)
3. News sentiment scores (numerical)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import logging
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CandlestickEncoder(nn.Module):
    """
    CNN-based encoder for candlestick chart patterns
    
    Uses ResNet-18 pre-trained on ImageNet and fine-tunes for financial charts
    """
    
    def __init__(self, output_dim: int = 256, pretrained: bool = True):
        super(CandlestickEncoder, self).__init__()
        
        # Load pre-trained ResNet-18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Remove the final classification layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Add custom output layer
        self.output_layer = nn.Linear(512, output_dim)
        
        # Freeze early layers for transfer learning
        if pretrained:
            for param in list(self.resnet.parameters())[:-20]:  # Keep last few layers trainable
                param.requires_grad = False
        
        logger.info(f"Candlestick encoder initialized with output_dim={output_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder
        
        Args:
            x: Input image tensor [batch_size, 3, 224, 224]
            
        Returns:
            Encoded features [batch_size, output_dim]
        """
        # Extract features using ResNet
        features = self.resnet(x)
        
        # Flatten features
        features = features.view(features.size(0), -1)
        
        # Project to output dimension
        output = self.output_layer(features)
        
        return output

class CandlestickVisualizer:
    """
    Utility class for creating candlestick chart images from OHLCV data
    """
    
    def __init__(self, resolution: Tuple[int, int] = (224, 224), style: str = "candlestick"):
        self.resolution = resolution
        self.style = style
        self.figsize = (resolution[0] / 100, resolution[1] / 100)  # Convert to inches
        
    def create_chart_image(self, ohlcv_data: pd.DataFrame, window: int = 60) -> Image.Image:
        """
        Create candlestick chart image from OHLCV data
        
        Args:
            ohlcv_data: DataFrame with OHLCV columns
            window: Number of periods to include in chart
            
        Returns:
            PIL Image of the candlestick chart
        """
        # Select the last 'window' periods
        data = ohlcv_data.tail(window).copy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=100)
        
        # Plot candlestick chart
        self._plot_candlesticks(ax, data)
        
        # Customize appearance
        ax.set_title('Candlestick Chart', fontsize=10, pad=10)
        ax.set_xlabel('Time', fontsize=8)
        ax.set_ylabel('Price', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Remove margins and set tight layout
        plt.tight_layout()
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        
        # Convert to PIL Image
        fig.canvas.draw()
        img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Resize to target resolution
        img = Image.fromarray(img_data)
        img = img.resize(self.resolution, Image.Resampling.LANCZOS)
        
        # Clean up
        plt.close(fig)
        
        return img
    
    def _plot_candlesticks(self, ax, data: pd.DataFrame):
        """Plot candlestick chart on given axes"""
        # Calculate candlestick properties
        opens = data['open'].values
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values
        
        # Determine colors (green for bullish, red for bearish)
        colors = ['green' if close >= open else 'red' for open, close in zip(opens, closes)]
        
        # Plot candlesticks
        for i in range(len(data)):
            # Body
            body_height = abs(closes[i] - opens[i])
            body_bottom = min(opens[i], closes[i])
            
            # Wick
            wick_top = highs[i]
            wick_bottom = lows[i]
            
            # Plot body rectangle
            rect = plt.Rectangle((i-0.3, body_bottom), 0.6, body_height, 
                               facecolor=colors[i], edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
            
            # Plot wick
            ax.plot([i, i], [wick_bottom, wick_top], color='black', linewidth=0.5)
        
        # Set x-axis labels
        if len(data) > 10:
            step = len(data) // 5
            ax.set_xticks(range(0, len(data), step))
            ax.set_xticklabels([data.index[i].strftime('%H:%M') if step > 0 else '' 
                               for i in range(0, len(data), step)], rotation=45)
        else:
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels([data.index[i].strftime('%H:%M') for i in range(len(data))], rotation=45)

class TechnicalFeatureExtractor:
    """
    Extracts and normalizes technical indicators from OHLCV data
    """
    
    def __init__(self, indicators: list = None):
        if indicators is None:
            self.indicators = [
                'SMA_20', 'SMA_50', 'SMA_200',
                'EMA_20', 'EMA_50', 'EMA_200',
                'RSI', 'MACD', 'MACD_signal', 'MACD_histogram',
                'BB_upper', 'BB_lower', 'BB_middle',
                'ATR', 'Volume_SMA'
            ]
        else:
            self.indicators = indicators
        
        logger.info(f"Technical feature extractor initialized with {len(self.indicators)} indicators")
    
    def calculate_indicators(self, ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators
        
        Args:
            ohlcv_data: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with original data and calculated indicators
        """
        data = ohlcv_data.copy()
        
        # Moving Averages
        data['SMA_20'] = data['close'].rolling(window=20).mean()
        data['SMA_50'] = data['close'].rolling(window=50).mean()
        data['SMA_200'] = data['close'].rolling(window=200).mean()
        
        data['EMA_20'] = data['close'].ewm(span=20).mean()
        data['EMA_50'] = data['close'].ewm(span=50).mean()
        data['EMA_200'] = data['close'].ewm(span=200).mean()
        
        # RSI
        data['RSI'] = self._calculate_rsi(data['close'], window=14)
        
        # MACD
        macd_data = self._calculate_macd(data['close'])
        data['MACD'] = macd_data['macd']
        data['MACD_signal'] = macd_data['signal']
        data['MACD_histogram'] = macd_data['histogram']
        
        # Bollinger Bands
        bb_data = self._calculate_bollinger_bands(data['close'])
        data['BB_upper'] = bb_data['upper']
        data['BB_lower'] = bb_data['lower']
        data['BB_middle'] = bb_data['middle']
        
        # ATR
        data['ATR'] = self._calculate_atr(data)
        
        # Volume SMA
        data['Volume_SMA'] = data['volume'].rolling(window=20).mean()
        
        return data
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD, Signal, and Histogram"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        middle = sma
        
        return {
            'upper': upper,
            'lower': lower,
            'middle': middle
        }
    
    def _calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=window).mean()
        
        return atr
    
    def normalize_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Normalize technical features to [0, 1] range
        
        Args:
            data: DataFrame with technical indicators
            
        Returns:
            Normalized features array
        """
        features = []
        
        for indicator in self.indicators:
            if indicator in data.columns:
                values = data[indicator].fillna(0).values
                
                # Handle infinite values
                values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Min-Max normalization
                if values.max() != values.min():
                    normalized = (values - values.min()) / (values.max() - values.min())
                else:
                    normalized = np.zeros_like(values)
                
                features.append(normalized)
            else:
                # If indicator not found, use zeros
                features.append(np.zeros(len(data)))
        
        return np.column_stack(features)

class SentimentFeatureExtractor:
    """
    Extracts and processes news sentiment features
    """
    
    def __init__(self, window: int = 24):
        self.window = window
        logger.info(f"Sentiment feature extractor initialized with {window}h window")
    
    def calculate_sentiment_features(self, sentiment_data: pd.DataFrame) -> np.ndarray:
        """
        Calculate sentiment features from raw sentiment scores
        
        Args:
            sentiment_data: DataFrame with timestamp and sentiment_score columns
            
        Returns:
            Array of sentiment features [rolling_mean, ewma]
        """
        if sentiment_data.empty:
            # Return neutral sentiment if no data
            return np.array([[0.0, 0.0]])
        
        # Ensure timestamp index
        if not isinstance(sentiment_data.index, pd.DatetimeIndex):
            sentiment_data = sentiment_data.set_index('timestamp')
        
        # Sort by timestamp
        sentiment_data = sentiment_data.sort_index()
        
        # Calculate rolling mean sentiment
        rolling_mean = sentiment_data['sentiment_score'].rolling(window=self.window, min_periods=1).mean()
        
        # Calculate exponential weighted moving average
        ewma = sentiment_data['sentiment_score'].ewm(span=self.window, min_periods=1).mean()
        
        # Fill NaN values with 0 (neutral sentiment)
        rolling_mean = rolling_mean.fillna(0)
        ewma = ewma.fillna(0)
        
        # Combine features
        features = np.column_stack([rolling_mean.values, ewma.values])
        
        return features

class MultimodalFusion(nn.Module):
    """
    Main multimodal fusion module that combines all three data types
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(MultimodalFusion, self).__init__()
        
        self.config = config
        
        # Extract dimensions from config
        self.visual_dim = config['model']['visual_features']
        self.technical_dim = config['model']['technical_features']
        self.sentiment_dim = config['model']['sentiment_features']
        self.total_dim = config['model']['total_state_dim']
        
        # Initialize encoders
        self.candlestick_encoder = CandlestickEncoder(output_dim=self.visual_dim)
        self.technical_extractor = TechnicalFeatureExtractor()
        self.sentiment_extractor = SentimentFeatureExtractor(
            window=config['data']['sentiment_window']
        )
        
        # Fusion layer (optional - can be used for additional processing)
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.total_dim, self.total_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        logger.info(f"Multimodal fusion module initialized with total_dim={self.total_dim}")
    
    def forward(self, visual_features: torch.Tensor, technical_features: torch.Tensor, 
                sentiment_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the fusion module
        
        Args:
            visual_features: Candlestick chart features [batch_size, visual_dim]
            technical_features: Technical indicator features [batch_size, technical_dim]
            sentiment_features: Sentiment features [batch_size, sentiment_dim]
            
        Returns:
            Fused features [batch_size, total_dim]
        """
        # Concatenate all features
        combined_features = torch.cat([visual_features, technical_features, sentiment_features], dim=1)
        
        # Apply fusion layer
        fused_features = self.fusion_layer(combined_features)
        
        return fused_features
    
    def process_data(self, ohlcv_data: pd.DataFrame, sentiment_data: pd.DataFrame = None) -> np.ndarray:
        """
        Process raw data and return fused state vector
        
        Args:
            ohlcv_data: OHLCV data for candlestick charts and technical indicators
            sentiment_data: News sentiment data (optional)
            
        Returns:
            Fused state vector [total_dim]
        """
        # 1. Process candlestick chart
        visualizer = CandlestickVisualizer(
            resolution=tuple(self.config['data']['chart_resolution'])
        )
        
        # Create chart image
        chart_image = visualizer.create_chart_image(
            ohlcv_data, 
            window=self.config['data']['candlestick_window']
        )
        
        # Convert to tensor and encode
        chart_tensor = self._image_to_tensor(chart_image)
        with torch.no_grad():
            visual_features = self.candlestick_encoder(chart_tensor)
        
        # 2. Process technical indicators
        technical_data = self.technical_extractor.calculate_indicators(ohlcv_data)
        technical_features = self.technical_extractor.normalize_features(technical_data)
        
        # 3. Process sentiment data
        if sentiment_data is not None and not sentiment_data.empty:
            sentiment_features = self.sentiment_extractor.calculate_sentiment_features(sentiment_data)
        else:
            # Use neutral sentiment if no data
            sentiment_features = np.array([[0.0, 0.0]])
        
        # 4. Combine all features
        # Take the latest values for technical and sentiment features
        latest_technical = technical_features[-1] if len(technical_features) > 0 else np.zeros(self.technical_dim)
        latest_sentiment = sentiment_features[-1] if len(sentiment_features) > 0 else np.zeros(self.sentiment_dim)
        
        # Convert visual features to numpy
        visual_np = visual_features.squeeze().cpu().numpy()
        
        # Concatenate all features
        fused_state = np.concatenate([visual_np, latest_technical, latest_sentiment])
        
        return fused_state
    
    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to PyTorch tensor"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to tensor
        image_tensor = torch.from_numpy(np.array(image)).float()
        
        # Normalize to [0, 1]
        image_tensor = image_tensor / 255.0
        
        # Add batch dimension and rearrange to [B, C, H, W]
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get dimensions of each feature type"""
        return {
            'visual': self.visual_dim,
            'technical': self.technical_dim,
            'sentiment': self.sentiment_dim,
            'total': self.total_dim
        }
