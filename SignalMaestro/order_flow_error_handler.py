
#!/usr/bin/env python3
"""
Order Flow Error Handler
Comprehensive error handling and data validation for order flow analysis
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import traceback

class OrderFlowErrorHandler:
    """Comprehensive error handler for order flow analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_counts = {}
        self.last_errors = {}
    
    def validate_ohlcv_data(self, symbol: str, ohlcv_data: Dict[str, List]) -> bool:
        """Validate OHLCV data structure and content"""
        try:
            if not ohlcv_data or not isinstance(ohlcv_data, dict):
                self.log_error(symbol, "Invalid OHLCV data structure")
                return False
            
            for tf, data in ohlcv_data.items():
                if not isinstance(data, list):
                    self.log_error(symbol, f"Invalid data type for {tf}")
                    continue
                
                if len(data) < 20:
                    self.log_error(symbol, f"Insufficient data for {tf}: {len(data)} candles")
                    continue
                
                # Validate first candle structure
                if data and len(data[0]) < 6:
                    self.log_error(symbol, f"Invalid candle structure for {tf}")
                    continue
            
            return True
            
        except Exception as e:
            self.log_error(symbol, f"OHLCV validation error: {e}")
            return False
    
    def safe_dataframe_creation(self, symbol: str, ohlcv_data: List) -> Optional[pd.DataFrame]:
        """Safely create DataFrame with proper error handling"""
        try:
            if not ohlcv_data or not isinstance(ohlcv_data, list):
                return None
            
            # Determine column count from first row
            if not ohlcv_data[0]:
                return None
            
            col_count = len(ohlcv_data[0])
            
            if col_count == 6:
                columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            elif col_count >= 12:
                columns = [
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote', 'ignore'
                ]
            else:
                self.log_error(symbol, f"Unexpected column count: {col_count}")
                return None
            
            df = pd.DataFrame(ohlcv_data, columns=columns[:col_count])
            
            # Convert numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            if col_count >= 12:
                numeric_cols.extend(['quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote'])
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert timestamps
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            if 'close_time' in df.columns:
                df['close_time'] = pd.to_numeric(df['close_time'], errors='coerce')
            
            # Remove rows with invalid data
            df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
            
            # Validate price data
            if len(df) == 0:
                self.log_error(symbol, "No valid price data after cleaning")
                return None
            
            # Check for reasonable price values
            if (df['high'] < df['low']).any() or (df['close'] <= 0).any():
                self.log_error(symbol, "Invalid price relationships detected")
                return None
            
            return df
            
        except Exception as e:
            self.log_error(symbol, f"DataFrame creation error: {e}")
            return None
    
    def validate_indicators(self, symbol: str, indicators: Dict[str, Any]) -> bool:
        """Validate calculated indicators"""
        try:
            required_indicators = ['cvd_analysis', 'imbalance_analysis']
            
            for indicator in required_indicators:
                if indicator not in indicators:
                    self.log_error(symbol, f"Missing required indicator: {indicator}")
                    return False
            
            # Validate CVD analysis
            cvd = indicators.get('cvd_analysis', {})
            if not isinstance(cvd, dict) or 'trend' not in cvd:
                self.log_error(symbol, "Invalid CVD analysis structure")
                return False
            
            return True
            
        except Exception as e:
            self.log_error(symbol, f"Indicator validation error: {e}")
            return False
    
    def safe_calculation(self, symbol: str, calculation_name: str, calculation_func, *args, **kwargs):
        """Safely execute calculations with error handling"""
        try:
            return calculation_func(*args, **kwargs)
        except ZeroDivisionError:
            self.log_error(symbol, f"{calculation_name}: Division by zero")
            return None
        except (ValueError, TypeError) as e:
            self.log_error(symbol, f"{calculation_name}: Invalid values - {e}")
            return None
        except Exception as e:
            self.log_error(symbol, f"{calculation_name}: Unexpected error - {e}")
            return None
    
    def log_error(self, symbol: str, error_msg: str):
        """Log errors with frequency tracking"""
        error_key = f"{symbol}_{error_msg[:50]}"
        
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        self.last_errors[error_key] = datetime.now()
        
        # Only log frequently if it's a new error or hasn't occurred recently
        if self.error_counts[error_key] <= 3:
            self.logger.warning(f"Order Flow Error [{symbol}]: {error_msg}")
        elif self.error_counts[error_key] % 10 == 0:
            self.logger.warning(f"Order Flow Error [{symbol}] (x{self.error_counts[error_key]}): {error_msg}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors"""
        return {
            'total_unique_errors': len(self.error_counts),
            'total_error_count': sum(self.error_counts.values()),
            'recent_errors': dict(list(self.last_errors.items())[-10:])
        }
    
    def reset_error_tracking(self):
        """Reset error tracking counters"""
        self.error_counts.clear()
        self.last_errors.clear()
        self.logger.info("Error tracking reset")
