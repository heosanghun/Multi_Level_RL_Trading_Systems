"""
Market Regime Detection Module for PPO Baseline

This module implements market regime classification using:
1. Rule-based EMA classification
2. XGBoost-based machine learning classification
3. Regime-specific performance tracking
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Any, Optional
import logging
import joblib
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MarketRegimeDetector:
    """
    Market regime detection using multiple approaches
    
    Supports both rule-based and ML-based regime classification
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Extract configuration
        self.ema_periods = config['regime']['ema_periods']
        self.xgb_params = config['regime']['xgb_params']
        self.regime_labels = config['regime']['regimes']
        
        # Initialize components
        self.xgb_model = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
        # Performance tracking
        self.regime_performance = {}
        self.regime_history = []
        
        logger.info(f"Market regime detector initialized with EMA periods: {self.ema_periods}")
    
    def detect_regime_rule_based(self, ohlcv_data: pd.DataFrame) -> str:
        """
        Detect market regime using rule-based EMA classification
        
        Args:
            ohlcv_data: OHLCV data with calculated EMAs
            
        Returns:
            Regime label: 'bull', 'bear', or 'sideways'
        """
        if len(ohlcv_data) < max(self.ema_periods):
            logger.warning("Insufficient data for regime detection")
            return 'sideways'
        
        # Get latest EMA values
        latest = ohlcv_data.iloc[-1]
        
        # Check if all EMAs are available
        ema_cols = [f'EMA_{period}' for period in self.ema_periods]
        if not all(col in latest.index for col in ema_cols):
            logger.warning("EMA columns not found in data")
            return 'sideways'
        
        ema_20 = latest[f'EMA_{self.ema_periods[0]}']
        ema_50 = latest[f'EMA_{self.ema_periods[1]}']
        ema_200 = latest[f'EMA_{self.ema_periods[2]}']
        
        # Rule-based classification
        if ema_20 > ema_50 > ema_200:
            regime = 'bull'
        elif ema_20 < ema_50 < ema_200:
            regime = 'bear'
        else:
            regime = 'sideways'
        
        return regime
    
    def prepare_ml_features(self, ohlcv_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for machine learning-based regime classification
        
        Args:
            ohlcv_data: OHLCV data with technical indicators
            
        Returns:
            Features array and regime labels
        """
        # Ensure we have enough data
        if len(ohlcv_data) < max(self.ema_periods):
            logger.warning("Insufficient data for ML feature preparation")
            return np.array([]), np.array([])
        
        # Calculate EMAs if not present
        if not all(f'EMA_{period}' in ohlcv_data.columns for period in self.ema_periods):
            ohlcv_data = self._calculate_emas(ohlcv_data)
        
        # Create regime labels using rule-based method
        regime_labels = []
        for i in range(len(ohlcv_data)):
            if i < max(self.ema_periods) - 1:
                regime_labels.append('sideways')  # Not enough data
            else:
                # Use data up to current point for regime detection
                current_data = ohlcv_data.iloc[:i+1]
                regime = self.detect_regime_rule_based(current_data)
                regime_labels.append(regime)
        
        # Create feature matrix
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'SMA_20', 'SMA_50', 'SMA_200',
            'EMA_20', 'EMA_50', 'EMA_200',
            'RSI', 'MACD', 'MACD_signal', 'MACD_histogram',
            'BB_upper', 'BB_lower', 'BB_middle',
            'ATR', 'Volume_SMA'
        ]
        
        # Select available features
        available_cols = [col for col in feature_cols if col in ohlcv_data.columns]
        
        if not available_cols:
            logger.error("No features available for ML classification")
            return np.array([]), np.array([])
        
        # Extract features
        features = ohlcv_data[available_cols].values
        
        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize features
        features = self._normalize_features(features)
        
        return features, np.array(regime_labels)
    
    def _calculate_emas(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMA values for specified periods"""
        data = data.copy()
        
        for period in self.ema_periods:
            col_name = f'EMA_{period}'
            if col_name not in data.columns:
                data[col_name] = data['close'].ewm(span=period).mean()
        
        return data
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to [0, 1] range"""
        if features.size == 0:
            return features
        
        # Min-Max normalization
        for i in range(features.shape[1]):
            col = features[:, i]
            if col.max() != col.min():
                features[:, i] = (col - col.min()) / (col.max() - col.min())
            else:
                features[:, i] = 0
        
        return features
    
    def train_ml_model(self, ohlcv_data: pd.DataFrame, test_size: float = 0.2) -> Dict[str, float]:
        """
        Train XGBoost model for regime classification
        
        Args:
            ohlcv_data: OHLCV data with technical indicators
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary with training metrics
        """
        # Prepare features and labels
        features, labels = self.prepare_ml_features(ohlcv_data)
        
        if features.size == 0 or labels.size == 0:
            logger.error("No valid data for training")
            return {}
        
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, encoded_labels, test_size=test_size, random_state=42, stratify=encoded_labels
        )
        
        # Initialize and train XGBoost model
        self.xgb_model = xgb.XGBClassifier(**self.xgb_params, random_state=42)
        
        # Train the model
        self.xgb_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.xgb_model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Generate detailed report
        report = classification_report(y_test, y_pred, target_names=self.regime_labels, output_dict=True)
        
        # Store performance metrics
        self.regime_performance = {
            'accuracy': accuracy,
            'classification_report': report,
            'test_size': len(X_test),
            'train_size': len(X_train)
        }
        
        self.is_trained = True
        
        logger.info(f"ML model trained successfully. Accuracy: {accuracy:.4f}")
        
        return self.regime_performance
    
    def detect_regime_ml(self, ohlcv_data: pd.DataFrame) -> Tuple[str, float]:
        """
        Detect market regime using trained ML model
        
        Args:
            ohlcv_data: OHLCV data with technical indicators
            
        Returns:
            Tuple of (regime_label, confidence_score)
        """
        if not self.is_trained or self.xgb_model is None:
            logger.warning("ML model not trained. Using rule-based detection.")
            regime = self.detect_regime_rule_based(ohlcv_data)
            return regime, 1.0
        
        # Prepare features
        features, _ = self.prepare_ml_features(ohlcv_data)
        
        if features.size == 0:
            logger.warning("Could not prepare features for ML detection")
            return 'sideways', 0.0
        
        # Get latest feature vector
        latest_features = features[-1:].reshape(1, -1)
        
        # Make prediction
        prediction = self.xgb_model.predict(latest_features)[0]
        confidence = self.xgb_model.predict_proba(latest_features).max()
        
        # Decode label
        regime = self.label_encoder.inverse_transform([prediction])[0]
        
        return regime, confidence
    
    def detect_regime(self, ohlcv_data: pd.DataFrame, method: str = 'auto') -> Dict[str, Any]:
        """
        Detect market regime using specified method
        
        Args:
            ohlcv_data: OHLCV data
            method: 'rule_based', 'ml', or 'auto'
            
        Returns:
            Dictionary with regime information
        """
        timestamp = ohlcv_data.index[-1] if len(ohlcv_data) > 0 else datetime.now()
        
        if method == 'rule_based':
            regime = self.detect_regime_rule_based(ohlcv_data)
            confidence = 1.0
        elif method == 'ml' and self.is_trained:
            regime, confidence = self.detect_regime_ml(ohlcv_data)
        elif method == 'auto':
            # Use ML if available, otherwise fall back to rule-based
            if self.is_trained:
                regime, confidence = self.detect_regime_ml(ohlcv_data)
            else:
                regime = self.detect_regime_rule_based(ohlcv_data)
                confidence = 1.0
        else:
            logger.warning(f"Method '{method}' not available. Using rule-based detection.")
            regime = self.detect_regime_rule_based(ohlcv_data)
            confidence = 1.0
        
        # Store regime history
        regime_info = {
            'timestamp': timestamp,
            'regime': regime,
            'confidence': confidence,
            'method': method
        }
        self.regime_history.append(regime_info)
        
        return regime_info
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected regimes"""
        if not self.regime_history:
            return {}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.regime_history)
        
        # Regime distribution
        regime_counts = df['regime'].value_counts().to_dict()
        
        # Confidence statistics
        confidence_stats = df['confidence'].describe().to_dict()
        
        # Method usage
        method_counts = df['method'].value_counts().to_dict()
        
        # Recent regime changes
        recent_regimes = df.tail(10)['regime'].tolist()
        
        return {
            'regime_distribution': regime_counts,
            'confidence_statistics': confidence_stats,
            'method_usage': method_counts,
            'recent_regimes': recent_regimes,
            'total_detections': len(df)
        }
    
    def get_regime_performance(self) -> Dict[str, Any]:
        """Get performance metrics for the regime detector"""
        return self.regime_performance.copy()
    
    def save_model(self, filepath: str):
        """Save trained model to file"""
        if not self.is_trained or self.xgb_model is None:
            logger.warning("No trained model to save")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save XGBoost model
        model_path = filepath.replace('.joblib', '_xgb.json')
        self.xgb_model.save_model(model_path)
        
        # Save other components
        save_data = {
            'label_encoder': self.label_encoder,
            'regime_performance': self.regime_performance,
            'config': self.config,
            'is_trained': self.is_trained
        }
        
        joblib.dump(save_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file"""
        try:
            # Load other components
            load_data = joblib.load(filepath)
            
            self.label_encoder = load_data['label_encoder']
            self.regime_performance = load_data['regime_performance']
            self.is_trained = load_data['is_trained']
            
            # Load XGBoost model
            model_path = filepath.replace('.joblib', '_xgb.json')
            self.xgb_model = xgb.XGBClassifier()
            self.xgb_model.load_model(model_path)
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.is_trained = False
            self.xgb_model = None
    
    def plot_regime_history(self, save_path: str = None):
        """Plot regime detection history"""
        if not self.regime_history:
            logger.warning("No regime history to plot")
            return
        
        import matplotlib.pyplot as plt
        
        df = pd.DataFrame(self.regime_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot regime over time
        regime_colors = {'bull': 'green', 'bear': 'red', 'sideways': 'gray'}
        for regime in self.regime_labels:
            regime_data = df[df['regime'] == regime]
            if not regime_data.empty:
                ax1.scatter(regime_data['timestamp'], regime_data['regime'], 
                           c=regime_colors.get(regime, 'blue'), alpha=0.7, s=50)
        
        ax1.set_title('Market Regime Detection Over Time')
        ax1.set_ylabel('Regime')
        ax1.grid(True, alpha=0.3)
        
        # Plot confidence over time
        ax2.plot(df['timestamp'], df['confidence'], 'b-', alpha=0.7)
        ax2.set_title('Detection Confidence Over Time')
        ax2.set_ylabel('Confidence')
        ax2.set_xlabel('Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Regime history plot saved to {save_path}")
        
        plt.show()
