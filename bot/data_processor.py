"""
Data processing and feature engineering for the trading bot
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
try:
    import pandas_ta as ta
except ImportError:
    # Fallback if pandas_ta has compatibility issues
    ta = None

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles data processing and feature engineering"""
    
    def __init__(self, lookback_period: int = 100, feature_window: int = 20):
        self.lookback_period = lookback_period
        self.feature_window = feature_window
        self.feature_columns = []
        
    def process_ohlcv_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process OHLCV data and add technical indicators"""
        try:
            if data.empty:
                logger.warning("Empty data provided to process_ohlcv_data")
                return pd.DataFrame()
            
            df = data.copy()
            
            # Ensure proper column names
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required columns. Available: {df.columns.tolist()}")
                return pd.DataFrame()
            
            # Add basic technical indicators
            df = self._add_technical_indicators(df)
            
            # Add price-based features
            df = self._add_price_features(df)
            
            # Add volume-based features
            df = self._add_volume_features(df)
            
            # Add momentum indicators
            df = self._add_momentum_indicators(df)
            
            # Add volatility indicators
            df = self._add_volatility_indicators(df)
            
            # Clean data
            df = self._clean_data(df)
            
            logger.debug(f"Processed data shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error processing OHLCV data: {e}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        try:
            if ta is None:
                # Fallback calculations without pandas_ta
                df['sma_10'] = df['close'].rolling(10).mean()
                df['sma_20'] = df['close'].rolling(20).mean()
                df['sma_50'] = df['close'].rolling(50).mean()
                df['ema_10'] = df['close'].ewm(span=10).mean()
                df['ema_20'] = df['close'].ewm(span=20).mean()
                
                # Simple RSI calculation
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                
                # Simple Bollinger Bands
                bb_period = 20
                bb_std = 2
                df['bb_mid'] = df['close'].rolling(bb_period).mean()
                std = df['close'].rolling(bb_period).std()
                df['bb_upper'] = df['bb_mid'] + (std * bb_std)
                df['bb_lower'] = df['bb_mid'] - (std * bb_std)
                df['bb_width'] = df['bb_upper'] - df['bb_lower']
                df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
                
                return df
            
            # Moving Averages
            df['sma_10'] = ta.sma(df['close'], length=10)
            df['sma_20'] = ta.sma(df['close'], length=20)
            df['sma_50'] = ta.sma(df['close'], length=50)
            df['ema_10'] = ta.ema(df['close'], length=10)
            df['ema_20'] = ta.ema(df['close'], length=20)
            
            # RSI
            df['rsi'] = ta.rsi(df['close'], length=14)
            
            # MACD
            macd_data = ta.macd(df['close'])
            if macd_data is not None and not macd_data.empty:
                df['macd'] = macd_data.iloc[:, 0]  # MACD line
                df['macd_signal'] = macd_data.iloc[:, 1]  # Signal line
                df['macd_histogram'] = macd_data.iloc[:, 2]  # Histogram
            
            # Bollinger Bands
            bb_data = ta.bbands(df['close'], length=20)
            if bb_data is not None and not bb_data.empty:
                df['bb_upper'] = bb_data.iloc[:, 0]
                df['bb_mid'] = bb_data.iloc[:, 1]
                df['bb_lower'] = bb_data.iloc[:, 2]
                df['bb_width'] = df['bb_upper'] - df['bb_lower']
                df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Stochastic
            stoch_data = ta.stoch(df['high'], df['low'], df['close'])
            if stoch_data is not None and not stoch_data.empty:
                df['stoch_k'] = stoch_data.iloc[:, 0]
                df['stoch_d'] = stoch_data.iloc[:, 1]
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        try:
            # Price changes
            df['price_change'] = df['close'].pct_change()
            df['price_change_2'] = df['close'].pct_change(2)
            df['price_change_5'] = df['close'].pct_change(5)
            
            # High-Low range
            df['hl_range'] = (df['high'] - df['low']) / df['close']
            
            # Open-Close range
            df['oc_range'] = (df['close'] - df['open']) / df['open']
            
            # Price position in range
            df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            
            # Rolling statistics
            df['price_std_10'] = df['close'].rolling(10).std()
            df['price_mean_10'] = df['close'].rolling(10).mean()
            df['price_z_score'] = (df['close'] - df['price_mean_10']) / df['price_std_10']
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding price features: {e}")
            return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        try:
            # Volume moving averages
            df['volume_sma_10'] = df['volume'].rolling(10).mean()
            df['volume_sma_20'] = df['volume'].rolling(20).mean()
            
            # Volume ratio
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            
            # Price-Volume trend
            df['pv_trend'] = df['price_change'] * df['volume_ratio']
            
            # On Balance Volume
            if ta is not None:
                df['obv'] = ta.obv(df['close'], df['volume'])
            else:
                # Simple OBV calculation
                obv = [0]
                for i in range(1, len(df)):
                    if df['close'].iloc[i] > df['close'].iloc[i-1]:
                        obv.append(obv[-1] + df['volume'].iloc[i])
                    elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                        obv.append(obv[-1] - df['volume'].iloc[i])
                    else:
                        obv.append(obv[-1])
                df['obv'] = obv
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding volume features: {e}")
            return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators"""
        try:
            if ta is None:
                # Fallback calculations
                df['roc_5'] = df['close'].pct_change(5) * 100
                df['roc_10'] = df['close'].pct_change(10) * 100
                df['momentum'] = df['close'] - df['close'].shift(10)
                return df
            
            # Rate of Change
            df['roc_5'] = ta.roc(df['close'], length=5)
            df['roc_10'] = ta.roc(df['close'], length=10)
            
            # Williams %R
            df['williams_r'] = ta.willr(df['high'], df['low'], df['close'], length=14)
            
            # Commodity Channel Index
            df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=20)
            
            # Momentum
            df['momentum'] = ta.mom(df['close'], length=10)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding momentum indicators: {e}")
            return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators"""
        try:
            if ta is None:
                # Fallback calculations
                df['volatility'] = df['close'].rolling(20).std()
                df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(50).mean()
                return df
            
            # Average True Range
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            
            # Volatility ratio
            df['volatility'] = df['close'].rolling(20).std()
            df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(50).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding volatility indicators: {e}")
            return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data"""
        try:
            # Forward fill missing values
            df = df.fillna(method='ffill')
            
            # Drop remaining NaN values
            initial_shape = df.shape
            df = df.dropna()
            
            if df.shape[0] != initial_shape[0]:
                logger.debug(f"Dropped {initial_shape[0] - df.shape[0]} rows with NaN values")
            
            # Replace infinite values
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            return df
    
    def extract_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Extract feature DataFrame from processed data"""
        try:
            if df.empty:
                return pd.DataFrame(), []
            
            # Define feature columns (excluding OHLCV)
            exclude_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
            feature_columns = [col for col in df.columns if col not in exclude_columns]
            
            # Store feature columns for consistency
            self.feature_columns = feature_columns
            
            # Extract features as DataFrame (maintaining column names)
            features_df = df[feature_columns].copy()
            
            # Handle any remaining NaN or inf values
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            features_df = features_df.fillna(0)
            
            logger.debug(f"Extracted features shape: {features_df.shape}")
            logger.debug(f"Feature columns: {feature_columns}")
            
            return features_df, feature_columns
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return pd.DataFrame(), []
    
    def prepare_training_data(self, df: pd.DataFrame, target_window: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data with features and targets"""
        try:
            if df.empty:
                return pd.DataFrame(), pd.Series(dtype=float)
            
            # Create target (future price movement)
            df['future_return'] = df['close'].shift(-target_window) / df['close'] - 1
            
            # Create binary target (1 for up, 0 for down)
            df['target'] = (df['future_return'] > 0).astype(int)
            
            # Remove rows with no target
            df = df.dropna(subset=['target'])
            
            if df.empty:
                return pd.DataFrame(), pd.Series(dtype=float)
            
            # Extract features as DataFrame
            X, feature_columns = self.extract_features(df)
            y = df['target']
            
            logger.debug(f"Training data prepared - X shape: {X.shape}, y shape: {y.shape}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return pd.DataFrame(), pd.Series(dtype=float)
