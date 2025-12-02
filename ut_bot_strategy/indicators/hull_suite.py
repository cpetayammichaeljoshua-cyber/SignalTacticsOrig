"""
Hull Suite Indicator - Converted from TradingView Pine Script to Python
Combines Hull Moving Averages for trend confirmation and support/resistance
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, Any

DataFrame = pd.DataFrame
Series = pd.Series


def calculate_wma(data: pd.Series, period: int) -> pd.Series:
    """Calculate Weighted Moving Average"""
    weights = np.arange(1, period + 1)
    wma_raw = data.rolling(period).apply(lambda x: (x * weights).sum() / weights.sum(), raw=False)
    wma: pd.Series = wma_raw.bfill().ffill().fillna(data)
    return wma


def calculate_hma(data: pd.Series, period: int) -> pd.Series:
    """Calculate Hull Moving Average with NaN handling"""
    half_period = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    
    wma_half = calculate_wma(data, half_period)
    wma_full = calculate_wma(data, period)
    
    raw_hma = 2 * wma_half - wma_full
    raw_hma = raw_hma.fillna(data)
    hma_raw = calculate_wma(raw_hma, sqrt_period)
    
    hma: pd.Series = hma_raw.ffill().bfill().fillna(data)
    
    return hma


class HullSuite:
    """
    Hull Suite Indicator
    
    Combines multiple Hull Moving Averages to determine trend direction
    and provide support/resistance levels
    """
    
    def __init__(self, hma_200_length: int = 200, hma_89_length: int = 89, 
                 hma_55_length: int = 55, hma_34_length: int = 34):
        """
        Initialize Hull Suite
        
        Args:
            hma_200_length: Period for long-term HMA
            hma_89_length: Period for medium HMA
            hma_55_length: Period for short-term HMA  
            hma_34_length: Period for very short-term HMA
        """
        self.hma_200_length = hma_200_length
        self.hma_89_length = hma_89_length
        self.hma_55_length = hma_55_length
        self.hma_34_length = hma_34_length
    
    def calculate(self, df: DataFrame) -> DataFrame:
        """
        Calculate Hull Suite indicators
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added Hull Suite columns
        """
        df = df.copy()
        close: pd.Series = df['close']
        
        df['hma_200'] = calculate_hma(close, self.hma_200_length)
        df['hma_89'] = calculate_hma(close, self.hma_89_length)
        df['hma_55'] = calculate_hma(close, self.hma_55_length)
        df['hma_34'] = calculate_hma(close, self.hma_34_length)
        
        # Determine color/trend based on HMA positions
        df['hull_color'] = self._get_hull_color(df)
        df['hull_trend'] = self._get_hull_trend(df)
        df['hull_support'] = df['hma_55']
        df['hull_resistance'] = df['hma_200']
        
        return df
    
    def _get_hull_color(self, df: DataFrame) -> Series:
        """
        Determine Hull Suite color (trend direction)
        Always returns GREEN or RED based on fast vs slow HMA comparison
        """
        # Fill any NaN values first
        hma_34 = df['hma_34'].bfill().ffill().fillna(df['close'])
        hma_200 = df['hma_200'].bfill().ffill().fillna(df['close'])
        close_price = df['close']
        
        # Simple, reliable logic: Compare fast HMA to slow HMA
        # GREEN if fast (34) > slow (200), RED otherwise
        # This guarantees no GRAY - always a definite direction
        color = pd.Series('green', index=df.index)
        color[hma_34 <= hma_200] = 'red'
        
        return color
    
    def _get_hull_trend(self, df: DataFrame) -> Series:
        """Get Hull trend direction: 1 = up, -1 = down (always one or other)"""
        trend = pd.Series(1, index=df.index, dtype=int)  # Default to 1 (GREEN)
        trend[df['hull_color'] == 'red'] = -1
        return trend
    
    def get_signal_strength(self, df: DataFrame) -> float:
        """
        Get Hull Suite signal strength (0-1)
        Based on HMA34 vs HMA200 separation - always returns value (never 0)
        """
        if len(df) < 1:
            return 0.5
        
        latest = df.iloc[-1]
        
        # Calculate trend strength based on HMA separation
        hma_34 = float(latest.get('hma_34', 0)) if latest.get('hma_34') is not None else 0.0
        hma_200 = float(latest.get('hma_200', 0)) if latest.get('hma_200') is not None else 0.0
        price = float(latest.get('close', 0))
        
        # Use close as fallback if HMA is 0
        if hma_200 == 0 or hma_200 != hma_200:  # Check for NaN with != comparison
            hma_200 = price
        if hma_34 == 0 or hma_34 != hma_34:
            hma_34 = price
        
        if hma_200 == 0 or price == 0:
            return 0.5  # Default to 50% strength if no data
        
        separation_pct = abs(hma_34 - hma_200) / hma_200
        strength = min(separation_pct / 0.05, 1.0)  # Normalize to 1.0
        
        # Ensure minimum 15% strength for visible signals
        return max(float(strength), 0.15)
