"""
Data Processor Module for PPO Baseline

This module handles data loading, preprocessing, and validation for:
1. OHLCV price data
2. News sentiment data
3. Technical indicators
4. Data quality checks and validation
"""

import pandas as pd
import numpy as np
import yfinance as yf
import ccxt
import requests
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import os
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Main data processing class for the PPO Baseline system
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_cache = {}
        
        # Initialize data sources
        self.price_source = config['data_sources']['price_source']
        self.news_sources = config['data_sources']['sentiment_sources']
        
        logger.info(f"Data processor initialized with price source: {self.price_source}")
    
    def load_ohlcv_data(self, symbol: str, start_date: str, end_date: str, 
                        interval: str = '1h') -> pd.DataFrame:
        """
        Load OHLCV data from specified source
        
        Args:
            symbol: Trading symbol (e.g., 'BTC-USD')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval ('1m', '5m', '15m', '1h', '1d')
            
        Returns:
            DataFrame with OHLCV data
        """
        if self.price_source == 'yfinance':
            return self._load_from_yfinance(symbol, start_date, end_date, interval)
        elif self.price_source == 'binance':
            return self._load_from_binance(symbol, start_date, end_date, interval)
        elif self.price_source == 'custom':
            return self._load_from_custom(symbol, start_date, end_date, interval)
        else:
            raise ValueError(f"Unsupported price source: {self.price_source}")
    
    def _load_from_yfinance(self, symbol: str, start_date: str, end_date: str, 
                           interval: str) -> pd.DataFrame:
        """Load data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            # Rename columns to standard format
            data.columns = [col.lower() for col in data.columns]
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"Missing required columns: {required_cols}")
            
            logger.info(f"Loaded {len(data)} records from Yahoo Finance for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data from Yahoo Finance: {e}")
            raise
    
    def _load_from_binance(self, symbol: str, start_date: str, end_date: str, 
                          interval: str) -> pd.DataFrame:
        """Load data from Binance API"""
        try:
            # Initialize Binance exchange
            exchange = ccxt.binance()
            
            # Convert interval to Binance format
            interval_map = {
                '1m': '1m', '5m': '5m', '15m': '15m',
                '1h': '1h', '4h': '4h', '1d': '1d'
            }
            binance_interval = interval_map.get(interval, '1h')
            
            # Convert dates to timestamps
            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
            
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(symbol, binance_interval, start_ts, end_ts)
            
            # Convert to DataFrame
            data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('timestamp', inplace=True)
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"Missing required columns: {required_cols}")
            
            logger.info(f"Loaded {len(data)} records from Binance for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data from Binance: {e}")
            raise
    
    def _load_from_custom(self, symbol: str, start_date: str, end_date: str, 
                         interval: str) -> pd.DataFrame:
        """Load data from custom source (file or database)"""
        # This is a placeholder for custom data loading
        # Users can implement their own data loading logic here
        raise NotImplementedError("Custom data loading not implemented")
    
    def load_sentiment_data(self, keywords: List[str], start_date: str, 
                           end_date: str) -> pd.DataFrame:
        """
        Load news sentiment data from specified sources
        
        Args:
            keywords: List of keywords to search for
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with sentiment data
        """
        sentiment_data = []
        
        for source in self.news_sources:
            try:
                if source == 'news_api':
                    source_data = self._load_from_news_api(keywords, start_date, end_date)
                elif source == 'cointelegraph':
                    source_data = self._load_from_cointelegraph(keywords, start_date, end_date)
                elif source == 'coindesk':
                    source_data = self._load_from_coindesk(keywords, start_date, end_date)
                else:
                    logger.warning(f"Unknown news source: {source}")
                    continue
                
                if source_data is not None and not source_data.empty:
                    sentiment_data.append(source_data)
                    
            except Exception as e:
                logger.error(f"Error loading sentiment data from {source}: {e}")
                continue
        
        if sentiment_data:
            # Combine all sources
            combined_data = pd.concat(sentiment_data, ignore_index=True)
            combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Loaded {len(combined_data)} sentiment records")
            return combined_data
        else:
            logger.warning("No sentiment data loaded from any source")
            return pd.DataFrame(columns=['timestamp', 'sentiment_score', 'source'])
    
    def _load_from_news_api(self, keywords: List[str], start_date: str, 
                           end_date: str) -> Optional[pd.DataFrame]:
        """Load sentiment data from NewsAPI"""
        try:
            api_key = os.getenv('NEWS_API_KEY')
            if not api_key:
                logger.warning("NEWS_API_KEY environment variable not set")
                return None
            
            # This is a simplified implementation
            # In practice, you would make actual API calls to NewsAPI
            logger.info("NewsAPI sentiment loading not fully implemented")
            return None
            
        except Exception as e:
            logger.error(f"Error loading from NewsAPI: {e}")
            return None
    
    def _load_from_cointelegraph(self, keywords: List[str], start_date: str, 
                                end_date: str) -> Optional[pd.DataFrame]:
        """Load sentiment data from Cointelegraph"""
        try:
            # This is a simplified implementation
            # In practice, you would scrape or use Cointelegraph's API
            logger.info("Cointelegraph sentiment loading not fully implemented")
            return None
            
        except Exception as e:
            logger.error(f"Error loading from Cointelegraph: {e}")
            return None
    
    def _load_from_coindesk(self, keywords: List[str], start_date: str, 
                           end_date: str) -> Optional[pd.DataFrame]:
        """Load sentiment data from Coindesk"""
        try:
            # This is a simplified implementation
            # In practice, you would scrape or use Coindesk's API
            logger.info("Coindesk sentiment loading not fully implemented")
            return None
            
        except Exception as e:
            logger.error(f"Error loading from Coindesk: {e}")
            return None
    
    def validate_data(self, data: pd.DataFrame, data_type: str = 'ohlcv') -> Dict[str, Any]:
        """
        Validate data quality and completeness
        
        Args:
            data: DataFrame to validate
            data_type: Type of data ('ohlcv', 'sentiment', 'technical')
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            # Check if DataFrame is empty
            if data.empty:
                validation_results['is_valid'] = False
                validation_results['errors'].append("DataFrame is empty")
                return validation_results
            
            # Check for required columns based on data type
            if data_type == 'ohlcv':
                required_cols = ['open', 'high', 'low', 'close', 'volume']
            elif data_type == 'sentiment':
                required_cols = ['timestamp', 'sentiment_score']
            elif data_type == 'technical':
                required_cols = ['close']  # At minimum, we need close prices
            else:
                required_cols = []
            
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                validation_results['is_valid'] = False
                validation_results['errors'].append(f"Missing required columns: {missing_cols}")
            
            # Check for NaN values
            nan_counts = data.isnull().sum()
            if nan_counts.sum() > 0:
                validation_results['warnings'].append(f"Found {nan_counts.sum()} NaN values")
                validation_results['statistics']['nan_counts'] = nan_counts.to_dict()
            
            # Check for infinite values
            inf_counts = np.isinf(data.select_dtypes(include=[np.number])).sum()
            if inf_counts.sum() > 0:
                validation_results['warnings'].append(f"Found {inf_counts.sum()} infinite values")
                validation_results['statistics']['inf_counts'] = inf_counts.to_dict()
            
            # Check data types
            validation_results['statistics']['dtypes'] = data.dtypes.to_dict()
            
            # Check data range
            if data_type == 'ohlcv':
                # Check for negative prices
                negative_prices = (data[['open', 'high', 'low', 'close']] < 0).sum().sum()
                if negative_prices > 0:
                    validation_results['warnings'].append(f"Found {negative_prices} negative prices")
                
                # Check for volume consistency
                negative_volume = (data['volume'] < 0).sum()
                if negative_volume > 0:
                    validation_results['warnings'].append(f"Found {negative_volume} negative volumes")
            
            elif data_type == 'sentiment':
                # Check sentiment score range
                sentiment_range = data['sentiment_score'].describe()
                if sentiment_range['min'] < -1 or sentiment_range['max'] > 1:
                    validation_results['warnings'].append("Sentiment scores outside expected range [-1, 1]")
                
                validation_results['statistics']['sentiment_range'] = sentiment_range.to_dict()
            
            # Check for duplicates
            duplicate_count = data.duplicated().sum()
            if duplicate_count > 0:
                validation_results['warnings'].append(f"Found {duplicate_count} duplicate rows")
            
            # Check timestamp consistency (if applicable)
            if 'timestamp' in data.columns:
                if not isinstance(data.index, pd.DatetimeIndex):
                    try:
                        data.index = pd.to_datetime(data.index)
                    except:
                        validation_results['warnings'].append("Could not convert index to datetime")
                
                # Check for timestamp ordering
                if data.index.is_monotonic_increasing:
                    validation_results['statistics']['timestamp_ordered'] = True
                else:
                    validation_results['warnings'].append("Timestamps are not in ascending order")
            
            # Overall statistics
            validation_results['statistics']['total_rows'] = len(data)
            validation_results['statistics']['total_columns'] = len(data.columns)
            validation_results['statistics']['memory_usage'] = data.memory_usage(deep=True).sum()
            
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Validation error: {str(e)}")
        
        return validation_results
    
    def preprocess_data(self, ohlcv_data: pd.DataFrame, 
                       sentiment_data: pd.DataFrame = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Preprocess data for training
        
        Args:
            ohlcv_data: Raw OHLCV data
            sentiment_data: Raw sentiment data (optional)
            
        Returns:
            Tuple of (preprocessed_ohlcv, preprocessed_sentiment)
        """
        # Validate OHLCV data
        ohlcv_validation = self.validate_data(ohlcv_data, 'ohlcv')
        if not ohlcv_validation['is_valid']:
            raise ValueError(f"OHLCV data validation failed: {ohlcv_validation['errors']}")
        
        # Preprocess OHLCV data
        preprocessed_ohlcv = ohlcv_data.copy()
        
        # Handle NaN values
        preprocessed_ohlcv = preprocessed_ohlcv.fillna(method='ffill').fillna(method='bfill')
        
        # Remove any remaining NaN values
        preprocessed_ohlcv = preprocessed_ohlcv.dropna()
        
        # Ensure index is datetime
        if not isinstance(preprocessed_ohlcv.index, pd.DatetimeIndex):
            preprocessed_ohlcv.index = pd.to_datetime(preprocessed_ohlcv.index)
        
        # Sort by timestamp
        preprocessed_ohlcv = preprocessed_ohlcv.sort_index()
        
        # Preprocess sentiment data if provided
        preprocessed_sentiment = None
        if sentiment_data is not None and not sentiment_data.empty:
            sentiment_validation = self.validate_data(sentiment_data, 'sentiment')
            if not sentiment_validation['is_valid']:
                logger.warning(f"Sentiment data validation failed: {sentiment_validation['errors']}")
            else:
                preprocessed_sentiment = sentiment_data.copy()
                
                # Handle NaN values
                preprocessed_sentiment = preprocessed_sentiment.fillna(0)  # Neutral sentiment
                
                # Ensure timestamp column exists and is datetime
                if 'timestamp' not in preprocessed_sentiment.columns:
                    if isinstance(preprocessed_sentiment.index, pd.DatetimeIndex):
                        preprocessed_sentiment['timestamp'] = preprocessed_sentiment.index
                    else:
                        logger.warning("No timestamp column found in sentiment data")
                        preprocessed_sentiment = None
                else:
                    preprocessed_sentiment['timestamp'] = pd.to_datetime(preprocessed_sentiment['timestamp'])
                    preprocessed_sentiment = preprocessed_sentiment.sort_values('timestamp')
        
        logger.info(f"Data preprocessing completed. OHLCV: {len(preprocessed_ohlcv)} rows")
        if preprocessed_sentiment is not None:
            logger.info(f"Sentiment: {len(preprocessed_sentiment)} rows")
        
        return preprocessed_ohlcv, preprocessed_sentiment
    
    def create_sample_data(self, days: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create sample data for testing and development
        
        Args:
            days: Number of days of sample data to generate
            
        Returns:
            Tuple of (sample_ohlcv, sample_sentiment)
        """
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
        
        logger.info(f"Generated sample data: OHLCV {len(sample_ohlcv)} rows, Sentiment {len(sample_sentiment)} rows")
        
        return sample_ohlcv, sample_sentiment
    
    def save_data(self, data: pd.DataFrame, filepath: str, format: str = 'csv'):
        """Save data to file"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            if format == 'csv':
                data.to_csv(filepath)
            elif format == 'parquet':
                data.to_parquet(filepath)
            elif format == 'json':
                data.to_json(filepath, orient='records')
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Data saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise
    
    def load_data_from_file(self, filepath: str, format: str = 'auto') -> pd.DataFrame:
        """Load data from file"""
        try:
            if format == 'auto':
                # Auto-detect format from file extension
                ext = os.path.splitext(filepath)[1].lower()
                if ext == '.csv':
                    format = 'csv'
                elif ext == '.parquet':
                    format = 'parquet'
                elif ext == '.json':
                    format = 'json'
                else:
                    raise ValueError(f"Could not auto-detect format for {filepath}")
            
            if format == 'csv':
                data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            elif format == 'parquet':
                data = pd.read_parquet(filepath)
            elif format == 'json':
                data = pd.read_json(filepath, orient='records')
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Data loaded from {filepath}: {len(data)} rows")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
