"""
Hull Suite Indicator - Converted from TradingView Pine Script to Python
Combines Hull Moving Averages for trend confirmation and support/resistance
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def calculate_wma(data: pd.Series, period: int) -> pd.Series:
    """Calculate Weighted Moving Average"""
    weights = np.arange(1, period + 1)
    return data.rolling(period).apply(lambda x: (x * weights).sum() / weights.sum(), raw=False)


def calculate_hma(data: pd.Series, period: int) -> pd.Series:
    """Calculate Hull Moving Average"""
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = calculate_wma(data, half_period)
    wma_full = calculate_wma(data, period)
    
    raw_hma = 2 * wma_half - wma_full
    hma = calculate_wma(raw_hma, sqrt_period)
    
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
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Hull Suite indicators
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added Hull Suite columns
        """
        df = df.copy()
        close = df['close']
        
        # Calculate Hull Moving Averages
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
    
    def _get_hull_color(self, df: pd.DataFrame) -> pd.Series:
        """
        Determine Hull Suite color (trend direction)
        Green = Uptrend, Red = Downtrend, Gray = Neutral
        """
        color = pd.Series('gray', index=df.index)
        
        # Uptrend: HMA 34 > HMA 55 > HMA 89 > HMA 200
        uptrend = (df['hma_34'] > df['hma_55']) & \
                  (df['hma_55'] > df['hma_89']) & \
                  (df['hma_89'] > df['hma_200'])
        color[uptrend] = 'green'
        
        # Downtrend: HMA 34 < HMA 55 < HMA 89 < HMA 200
        downtrend = (df['hma_34'] < df['hma_55']) & \
                    (df['hma_55'] < df['hma_89']) & \
                    (df['hma_89'] < df['hma_200'])
        color[downtrend] = 'red'
        
        return color
    
    def _get_hull_trend(self, df: pd.DataFrame) -> pd.Series:
        """Get Hull trend direction: 1 = up, -1 = down, 0 = neutral"""
        trend = pd.Series(0, index=df.index, dtype=int)
        trend[df['hull_color'] == 'green'] = 1
        trend[df['hull_color'] == 'red'] = -1
        return trend
    
    def get_signal_strength(self, df: pd.DataFrame) -> float:
        """
        Get Hull Suite signal strength (0-1)
        Higher values indicate stronger trend
        """
        if len(df) < 1:
            return 0.0
        
        latest = df.iloc[-1]
        color = latest.get('hull_color', 'gray')
        
        if color == 'gray':
            return 0.0
        
        # Calculate trend strength based on HMA separation
        hma_34 = latest.get('hma_34', 0)
        hma_200 = latest.get('hma_200', 0)
        price = latest.get('close', 0)
        
        if hma_200 == 0:
            return 0.0
        
        separation_pct = abs(hma_34 - hma_200) / hma_200
        strength = min(separation_pct / 0.05, 1.0)  # Normalize to 1.0
        
        return strength
