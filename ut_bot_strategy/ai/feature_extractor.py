"""
Feature Extractor for AI Insights

Extracts market features from OHLCV data for AI analysis including:
- Rolling statistics (volatility, momentum, trend strength)
- Price action patterns (support/resistance levels, swing points)
- Indicator values summary (UT Bot trailing stop distance, STC value/direction)
- Volume analysis
"""

import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts comprehensive market features from OHLCV data for AI analysis
    """
    
    def __init__(self, volatility_window: int = 14, momentum_window: int = 10):
        """
        Initialize Feature Extractor
        
        Args:
            volatility_window: Window for volatility calculations (default 14)
            momentum_window: Window for momentum calculations (default 10)
        """
        self.volatility_window = volatility_window
        self.momentum_window = momentum_window
        logger.info(f"FeatureExtractor initialized with volatility_window={volatility_window}, momentum_window={momentum_window}")
    
    def extract_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract all market features from OHLCV data
        
        Args:
            df: DataFrame with OHLCV data and calculated indicators
            
        Returns:
            Dictionary containing all extracted features for AI analysis
        """
        logger.debug("Extracting market features from OHLCV data")
        
        features = {}
        
        try:
            features['rolling_statistics'] = self._extract_rolling_statistics(df)
            features['price_action'] = self._extract_price_action_patterns(df)
            features['indicator_summary'] = self._extract_indicator_summary(df)
            features['volume_analysis'] = self._extract_volume_analysis(df)
            features['market_context'] = self._extract_market_context(df)
            
            logger.debug(f"Successfully extracted {len(features)} feature categories")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return self._get_fallback_features()
    
    def _extract_rolling_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract rolling statistics including volatility, momentum, and trend strength
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with rolling statistics
        """
        logger.debug("Extracting rolling statistics")
        
        close = df['close']
        high = df['high']
        low = df['low']
        
        returns = close.pct_change()
        volatility = returns.rolling(window=self.volatility_window).std() * np.sqrt(252)
        current_volatility = float(volatility.iloc[-1]) if not pd.isna(volatility.iloc[-1]) else 0.0
        
        momentum = close.pct_change(periods=self.momentum_window)
        current_momentum = float(momentum.iloc[-1]) if not pd.isna(momentum.iloc[-1]) else 0.0
        
        ema_20 = close.ewm(span=20, adjust=False).mean()
        ema_50 = close.ewm(span=50, adjust=False).mean()
        
        trend_direction = "BULLISH" if close.iloc[-1] > ema_20.iloc[-1] > ema_50.iloc[-1] else \
                         "BEARISH" if close.iloc[-1] < ema_20.iloc[-1] < ema_50.iloc[-1] else "NEUTRAL"
        
        price_range = (high - low).rolling(window=self.volatility_window).mean()
        atr_estimate = float(price_range.iloc[-1]) if not pd.isna(price_range.iloc[-1]) else 0.0
        
        trend_strength = abs(close.iloc[-1] - ema_50.iloc[-1]) / close.iloc[-1] * 100 if close.iloc[-1] > 0 else 0.0
        
        return {
            'volatility': round(current_volatility * 100, 2),
            'volatility_level': self._classify_volatility(current_volatility),
            'momentum': round(current_momentum * 100, 2),
            'momentum_direction': "POSITIVE" if current_momentum > 0 else "NEGATIVE" if current_momentum < 0 else "NEUTRAL",
            'trend_direction': trend_direction,
            'trend_strength': round(trend_strength, 2),
            'atr_estimate': round(atr_estimate, 4),
            'price_vs_ema20': round((close.iloc[-1] / ema_20.iloc[-1] - 1) * 100, 2) if ema_20.iloc[-1] > 0 else 0,
            'price_vs_ema50': round((close.iloc[-1] / ema_50.iloc[-1] - 1) * 100, 2) if ema_50.iloc[-1] > 0 else 0
        }
    
    def _extract_price_action_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract price action patterns including support/resistance and swing points
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with price action patterns
        """
        logger.debug("Extracting price action patterns")
        
        close = df['close']
        high = df['high']
        low = df['low']
        
        current_price = float(close.iloc[-1])
        
        lookback = min(20, len(df) - 1)
        recent_high = float(high.tail(lookback).max())
        recent_low = float(low.tail(lookback).min())
        
        swing_highs = self._find_swing_points(high, mode='high', lookback=5)
        swing_lows = self._find_swing_points(low, mode='low', lookback=5)
        
        resistance_levels = sorted([h for h in swing_highs if h > current_price])[:3]
        support_levels = sorted([l for l in swing_lows if l < current_price], reverse=True)[:3]
        
        nearest_resistance = resistance_levels[0] if resistance_levels else recent_high
        nearest_support = support_levels[0] if support_levels else recent_low
        
        range_size = recent_high - recent_low
        range_position = (current_price - recent_low) / range_size * 100 if range_size > 0 else 50
        
        body_size = abs(close.iloc[-1] - df['open'].iloc[-1])
        candle_range = high.iloc[-1] - low.iloc[-1]
        candle_pattern = self._identify_candle_pattern(df)
        
        return {
            'current_price': round(current_price, 4),
            'recent_high': round(recent_high, 4),
            'recent_low': round(recent_low, 4),
            'nearest_resistance': round(nearest_resistance, 4),
            'nearest_support': round(nearest_support, 4),
            'distance_to_resistance': round((nearest_resistance - current_price) / current_price * 100, 2),
            'distance_to_support': round((current_price - nearest_support) / current_price * 100, 2),
            'range_position': round(range_position, 2),
            'swing_highs_count': len(swing_highs),
            'swing_lows_count': len(swing_lows),
            'candle_pattern': candle_pattern,
            'body_ratio': round(body_size / candle_range * 100, 2) if candle_range > 0 else 0
        }
    
    def _extract_indicator_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract UT Bot and STC indicator values summary
        
        Args:
            df: DataFrame with calculated indicators
            
        Returns:
            Dictionary with indicator summary
        """
        logger.debug("Extracting indicator summary")
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        ut_bot_summary = {
            'trailing_stop': round(float(latest.get('trailing_stop', 0)), 4),
            'trailing_stop_distance': round(abs(float(latest['close']) - float(latest.get('trailing_stop', latest['close']))), 4),
            'trailing_stop_distance_pct': round(abs(float(latest['close']) - float(latest.get('trailing_stop', latest['close']))) / float(latest['close']) * 100, 2) if latest['close'] > 0 else 0,
            'position': int(latest.get('position', 0)),
            'bar_color': str(latest.get('bar_color', 'neutral')),
            'above_stop': bool(latest.get('above_stop', False)),
            'below_stop': bool(latest.get('below_stop', False)),
            'buy_signal': bool(latest.get('buy_signal', False)),
            'sell_signal': bool(latest.get('sell_signal', False)),
            'atr': round(float(latest.get('atr', 0)), 4)
        }
        
        stc_value = float(latest.get('stc', 50))
        stc_prev = float(prev.get('stc', 50))
        stc_summary = {
            'value': round(stc_value, 2),
            'previous_value': round(stc_prev, 2),
            'direction': 'UP' if stc_value > stc_prev else 'DOWN' if stc_value < stc_prev else 'FLAT',
            'color': str(latest.get('stc_color', 'neutral')),
            'slope': str(latest.get('stc_slope', 'neutral')),
            'zone': 'OVERBOUGHT' if stc_value > 75 else 'OVERSOLD' if stc_value < 25 else 'NEUTRAL',
            'above_75': bool(latest.get('stc_above_75', False)),
            'below_25': bool(latest.get('stc_below_25', False))
        }
        
        return {
            'ut_bot': ut_bot_summary,
            'stc': stc_summary
        }
    
    def _extract_volume_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract volume analysis features
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with volume analysis
        """
        logger.debug("Extracting volume analysis")
        
        if 'volume' not in df.columns or df['volume'].isna().all():
            logger.debug("Volume data not available, returning defaults")
            return {
                'available': False,
                'current_volume': 0,
                'avg_volume': 0,
                'volume_ratio': 1.0,
                'volume_trend': 'UNKNOWN'
            }
        
        volume = df['volume']
        current_volume = float(volume.iloc[-1])
        avg_volume = float(volume.rolling(window=20).mean().iloc[-1]) if len(df) >= 20 else float(volume.mean())
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        recent_vol = volume.tail(5).mean()
        older_vol = volume.tail(20).head(15).mean() if len(df) >= 20 else volume.mean()
        volume_trend = "INCREASING" if recent_vol > older_vol * 1.1 else \
                      "DECREASING" if recent_vol < older_vol * 0.9 else "STABLE"
        
        return {
            'available': True,
            'current_volume': round(current_volume, 2),
            'avg_volume': round(avg_volume, 2),
            'volume_ratio': round(volume_ratio, 2),
            'volume_trend': volume_trend,
            'is_high_volume': volume_ratio > 1.5,
            'is_low_volume': volume_ratio < 0.5
        }
    
    def _extract_market_context(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract overall market context
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with market context
        """
        logger.debug("Extracting market context")
        
        close = df['close']
        
        returns_1h = (close.iloc[-1] / close.iloc[-12] - 1) * 100 if len(df) >= 12 else 0
        returns_4h = (close.iloc[-1] / close.iloc[-48] - 1) * 100 if len(df) >= 48 else 0
        returns_24h = (close.iloc[-1] / close.iloc[-288] - 1) * 100 if len(df) >= 288 else 0
        
        daily_range = (df['high'].iloc[-1] - df['low'].iloc[-1]) / close.iloc[-1] * 100 if close.iloc[-1] > 0 else 0
        
        higher_highs = sum(1 for i in range(-10, -1) if df['high'].iloc[i] > df['high'].iloc[i-1]) if len(df) > 10 else 0
        higher_lows = sum(1 for i in range(-10, -1) if df['low'].iloc[i] > df['low'].iloc[i-1]) if len(df) > 10 else 0
        
        return {
            'returns_1h': round(returns_1h, 2),
            'returns_4h': round(returns_4h, 2),
            'returns_24h': round(returns_24h, 2),
            'daily_range_pct': round(daily_range, 2),
            'higher_highs_count': higher_highs,
            'higher_lows_count': higher_lows,
            'market_phase': self._identify_market_phase(higher_highs, higher_lows),
            'data_points': len(df)
        }
    
    def _classify_volatility(self, volatility: float) -> str:
        """Classify volatility level"""
        if volatility < 0.01:
            return "LOW"
        elif volatility < 0.03:
            return "MEDIUM"
        elif volatility < 0.05:
            return "HIGH"
        else:
            return "EXTREME"
    
    def _find_swing_points(self, series: pd.Series, mode: str = 'high', lookback: int = 5) -> List[float]:
        """
        Find swing high or low points
        
        Args:
            series: Price series (high or low)
            mode: 'high' or 'low'
            lookback: Bars to look on each side
            
        Returns:
            List of swing point prices
        """
        swing_points = []
        
        for i in range(lookback, len(series) - lookback):
            if mode == 'high':
                is_swing = all(series.iloc[i] >= series.iloc[i-j] for j in range(1, lookback + 1)) and \
                          all(series.iloc[i] >= series.iloc[i+j] for j in range(1, lookback + 1))
            else:
                is_swing = all(series.iloc[i] <= series.iloc[i-j] for j in range(1, lookback + 1)) and \
                          all(series.iloc[i] <= series.iloc[i+j] for j in range(1, lookback + 1))
            
            if is_swing:
                swing_points.append(float(series.iloc[i]))
        
        return swing_points[-5:] if len(swing_points) > 5 else swing_points
    
    def _identify_candle_pattern(self, df: pd.DataFrame) -> str:
        """Identify the latest candle pattern"""
        if len(df) < 1:
            return "UNKNOWN"
        
        latest = df.iloc[-1]
        open_price = latest['open']
        close = latest['close']
        high = latest['high']
        low = latest['low']
        
        body = abs(close - open_price)
        range_size = high - low
        upper_wick = high - max(open_price, close)
        lower_wick = min(open_price, close) - low
        
        if range_size == 0:
            return "DOJI"
        
        body_ratio = body / range_size
        
        if body_ratio < 0.1:
            return "DOJI"
        elif body_ratio > 0.8:
            return "MARUBOZU_BULLISH" if close > open_price else "MARUBOZU_BEARISH"
        elif lower_wick > body * 2 and upper_wick < body * 0.5:
            return "HAMMER" if close > open_price else "HANGING_MAN"
        elif upper_wick > body * 2 and lower_wick < body * 0.5:
            return "SHOOTING_STAR" if close < open_price else "INVERTED_HAMMER"
        else:
            return "BULLISH" if close > open_price else "BEARISH"
    
    def _identify_market_phase(self, higher_highs: int, higher_lows: int) -> str:
        """Identify the current market phase"""
        if higher_highs >= 6 and higher_lows >= 6:
            return "STRONG_UPTREND"
        elif higher_highs >= 4 and higher_lows >= 4:
            return "UPTREND"
        elif higher_highs <= 2 and higher_lows <= 2:
            return "DOWNTREND"
        elif higher_highs <= 1 and higher_lows <= 1:
            return "STRONG_DOWNTREND"
        else:
            return "RANGING"
    
    def _get_fallback_features(self) -> Dict[str, Any]:
        """Return fallback features when extraction fails"""
        return {
            'rolling_statistics': {
                'volatility': 0,
                'volatility_level': 'UNKNOWN',
                'momentum': 0,
                'momentum_direction': 'NEUTRAL',
                'trend_direction': 'NEUTRAL',
                'trend_strength': 0,
                'atr_estimate': 0,
                'price_vs_ema20': 0,
                'price_vs_ema50': 0
            },
            'price_action': {
                'current_price': 0,
                'recent_high': 0,
                'recent_low': 0,
                'nearest_resistance': 0,
                'nearest_support': 0,
                'distance_to_resistance': 0,
                'distance_to_support': 0,
                'range_position': 50,
                'swing_highs_count': 0,
                'swing_lows_count': 0,
                'candle_pattern': 'UNKNOWN',
                'body_ratio': 0
            },
            'indicator_summary': {
                'ut_bot': {
                    'trailing_stop': 0,
                    'trailing_stop_distance': 0,
                    'trailing_stop_distance_pct': 0,
                    'position': 0,
                    'bar_color': 'neutral',
                    'above_stop': False,
                    'below_stop': False,
                    'buy_signal': False,
                    'sell_signal': False,
                    'atr': 0
                },
                'stc': {
                    'value': 50,
                    'previous_value': 50,
                    'direction': 'FLAT',
                    'color': 'neutral',
                    'slope': 'neutral',
                    'zone': 'NEUTRAL',
                    'above_75': False,
                    'below_25': False
                }
            },
            'volume_analysis': {
                'available': False,
                'current_volume': 0,
                'avg_volume': 0,
                'volume_ratio': 1.0,
                'volume_trend': 'UNKNOWN'
            },
            'market_context': {
                'returns_1h': 0,
                'returns_4h': 0,
                'returns_24h': 0,
                'daily_range_pct': 0,
                'higher_highs_count': 0,
                'higher_lows_count': 0,
                'market_phase': 'UNKNOWN',
                'data_points': 0
            }
        }
