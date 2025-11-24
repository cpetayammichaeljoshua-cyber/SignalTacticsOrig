#!/usr/bin/env python3
"""
ATAS Integrated Analyzer
Comprehensive technical indicator analysis compatible with ATAS platform
Analyzes all major trading indicators for signal generation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

@dataclass
class ATASIndicatorSignal:
    """ATAS indicator signal"""
    timestamp: str
    indicator_name: str
    value: float
    signal: str  # 'BUY', 'SELL', 'NEUTRAL'
    strength: float  # 0-100
    confidence: float  # 0-100
    description: str

class ATASIntegratedAnalyzer:
    """Comprehensive technical indicator analysis using ATAS methodology"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.min_bars = 100
    
    async def analyze_all_indicators(self, ohlcv_data) -> Dict[str, Any]:
        """Analyze all ATAS indicators in parallel"""
        try:
            if isinstance(ohlcv_data, list):
                if not ohlcv_data or len(ohlcv_data) < self.min_bars:
                    return {'error': 'Insufficient data'}
                ohlcv_data = pd.DataFrame(ohlcv_data, columns=['open', 'high', 'low', 'close', 'volume'])
            elif len(ohlcv_data) < self.min_bars:
                return {'error': 'Insufficient data'}
            
            results = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'indicators': {}
            }
            
            # Analyze all indicators
            results['indicators']['moving_averages'] = await self.analyze_moving_averages(ohlcv_data)
            results['indicators']['rsi'] = await self.analyze_rsi(ohlcv_data)
            results['indicators']['macd'] = await self.analyze_macd(ohlcv_data)
            results['indicators']['bollinger_bands'] = await self.analyze_bollinger_bands(ohlcv_data)
            results['indicators']['stochastic'] = await self.analyze_stochastic(ohlcv_data)
            results['indicators']['atr'] = await self.analyze_atr(ohlcv_data)
            results['indicators']['adx'] = await self.analyze_adx(ohlcv_data)
            results['indicators']['vpt'] = await self.analyze_volume_price_trend(ohlcv_data)
            results['indicators']['obv'] = await self.analyze_obv(ohlcv_data)
            results['indicators']['accumulation_distribution'] = await self.analyze_accumulation_distribution(ohlcv_data)
            results['indicators']['keltner_channel'] = await self.analyze_keltner_channel(ohlcv_data)
            results['indicators']['pivot_points'] = await self.analyze_pivot_points(ohlcv_data)
            results['indicators']['supertrend'] = await self.analyze_supertrend(ohlcv_data)
            results['indicators']['vwap'] = await self.analyze_vwap(ohlcv_data)
            results['indicators']['ichimoku'] = await self.analyze_ichimoku_extended(ohlcv_data)
            
            # Calculate composite signal
            results['composite_signal'] = self._calculate_composite_signal(results['indicators'])
            results['overall_strength'] = self._calculate_overall_strength(results['indicators'])
            
            return results
        except Exception as e:
            self.logger.error(f"ATAS analysis error: {e}")
            return {'error': str(e)}
    
    async def analyze_moving_averages(self, df: pd.DataFrame) -> Dict[str, Any]:
        """SMA, EMA, WMA analysis"""
        try:
            close = df['close'].values
            sma_20 = np.mean(close[-20:])
            sma_50 = np.mean(close[-50:])
            sma_200 = np.mean(close[-100:])
            
            current = close[-1]
            signal = 'BUY' if current > sma_20 > sma_50 > sma_200 else ('SELL' if current < sma_20 < sma_50 < sma_200 else 'NEUTRAL')
            
            return {
                'sma_20': float(sma_20),
                'sma_50': float(sma_50),
                'sma_200': float(sma_200),
                'current_price': float(current),
                'signal': signal,
                'strength': 75.0 if signal != 'NEUTRAL' else 50.0
            }
        except Exception as e:
            self.logger.debug(f"MA analysis error: {e}")
            return {'signal': 'NEUTRAL', 'strength': 0}
    
    async def analyze_rsi(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Relative Strength Index"""
        try:
            close = df['close'].values[-50:]
            deltas = np.diff(close)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            rs = avg_gain / avg_loss if avg_loss > 0 else 0
            rsi = 100 - (100 / (1 + rs)) if rs >= 0 else 0
            
            signal = 'BUY' if rsi < 30 else ('SELL' if rsi > 70 else 'NEUTRAL')
            return {
                'rsi': float(rsi),
                'signal': signal,
                'strength': abs(rsi - 50) / 50 * 100
            }
        except Exception as e:
            self.logger.debug(f"RSI analysis error: {e}")
            return {'signal': 'NEUTRAL', 'strength': 0}
    
    async def analyze_macd(self, df: pd.DataFrame) -> Dict[str, Any]:
        """MACD Analysis"""
        try:
            close = df['close'].values
            exp1 = pd.Series(close).ewm(span=12).mean().values
            exp2 = pd.Series(close).ewm(span=26).mean().values
            macd = exp1 - exp2
            signal_line = pd.Series(macd).ewm(span=9).mean().values
            histogram = macd - signal_line
            
            current_hist = histogram[-1]
            signal = 'BUY' if current_hist > 0 and histogram[-2] < 0 else ('SELL' if current_hist < 0 and histogram[-2] > 0 else 'NEUTRAL')
            
            return {
                'macd': float(macd[-1]),
                'signal_line': float(signal_line[-1]),
                'histogram': float(current_hist),
                'signal': signal,
                'strength': min(100, abs(current_hist) * 1000)
            }
        except Exception as e:
            self.logger.debug(f"MACD analysis error: {e}")
            return {'signal': 'NEUTRAL', 'strength': 0}
    
    async def analyze_bollinger_bands(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Bollinger Bands"""
        try:
            close = df['close'].values[-20:]
            sma = np.mean(close)
            std = np.std(close)
            upper_band = sma + (2 * std)
            lower_band = sma - (2 * std)
            current = close[-1]
            
            if current < lower_band:
                signal = 'BUY'
            elif current > upper_band:
                signal = 'SELL'
            else:
                signal = 'NEUTRAL'
            
            return {
                'upper_band': float(upper_band),
                'middle_band': float(sma),
                'lower_band': float(lower_band),
                'current_price': float(current),
                'signal': signal,
                'strength': 65.0 if signal != 'NEUTRAL' else 50.0
            }
        except Exception as e:
            self.logger.debug(f"Bollinger Bands error: {e}")
            return {'signal': 'NEUTRAL', 'strength': 0}
    
    async def analyze_stochastic(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Stochastic Oscillator"""
        try:
            low = df['low'].values[-14:]
            high = df['high'].values[-14:]
            close = df['close'].values[-14:]
            
            ll = np.min(low)
            hh = np.max(high)
            k = 100 * (close[-1] - ll) / (hh - ll) if hh > ll else 50
            
            signal = 'BUY' if k < 20 else ('SELL' if k > 80 else 'NEUTRAL')
            return {
                'stoch_k': float(k),
                'signal': signal,
                'strength': abs(k - 50) / 50 * 100
            }
        except Exception as e:
            self.logger.debug(f"Stochastic error: {e}")
            return {'signal': 'NEUTRAL', 'strength': 0}
    
    async def analyze_atr(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Average True Range"""
        try:
            high = df['high'].values[-14:]
            low = df['low'].values[-14:]
            close = df['close'].values[-14:]
            
            tr = np.maximum(high - low, np.abs(high - close[:-1]), np.abs(low - close[:-1]))
            atr = np.mean(tr)
            
            return {
                'atr': float(atr),
                'atr_percent': float(atr / close[-1] * 100),
                'signal': 'NEUTRAL',
                'strength': 50.0
            }
        except Exception as e:
            self.logger.debug(f"ATR error: {e}")
            return {'signal': 'NEUTRAL', 'strength': 0}
    
    async def analyze_adx(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Average Directional Index"""
        try:
            high = df['high'].values[-14:]
            low = df['low'].values[-14:]
            
            up_move = high[1:] - high[:-1]
            down_move = low[:-1] - low[1:]
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            adx = (np.mean(plus_dm) - np.mean(minus_dm)) / (np.mean(plus_dm) + np.mean(minus_dm) + 0.001) * 100
            adx = (adx + 100) / 2  # Normalize
            
            signal = 'BUY' if adx > 60 else ('SELL' if adx < -60 else 'NEUTRAL')
            return {
                'adx': float(adx),
                'signal': signal,
                'strength': abs(adx) / 100 * 100
            }
        except Exception as e:
            self.logger.debug(f"ADX error: {e}")
            return {'signal': 'NEUTRAL', 'strength': 0}
    
    async def analyze_volume_price_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Volume Price Trend"""
        try:
            close = df['close'].values
            volume = df['volume'].values
            
            price_change = np.diff(close) / close[:-1] * 100
            vpt = np.sum(volume[1:] * price_change / 100)
            
            return {
                'vpt': float(vpt),
                'signal': 'NEUTRAL',
                'strength': 50.0
            }
        except Exception as e:
            self.logger.debug(f"VPT error: {e}")
            return {'signal': 'NEUTRAL', 'strength': 0}
    
    async def analyze_obv(self, df: pd.DataFrame) -> Dict[str, Any]:
        """On Balance Volume"""
        try:
            close = df['close'].values[-50:]
            volume = df['volume'].values[-50:]
            
            obv = np.cumsum(np.where(np.diff(close, prepend=close[0]) > 0, volume, np.where(np.diff(close, prepend=close[0]) < 0, -volume, 0)))
            
            signal = 'BUY' if obv[-1] > np.mean(obv[-20:]) else ('SELL' if obv[-1] < np.mean(obv[-20:]) else 'NEUTRAL')
            return {
                'obv': float(obv[-1]),
                'signal': signal,
                'strength': 60.0 if signal != 'NEUTRAL' else 50.0
            }
        except Exception as e:
            self.logger.debug(f"OBV error: {e}")
            return {'signal': 'NEUTRAL', 'strength': 0}
    
    async def analyze_accumulation_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Accumulation/Distribution Line"""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            volume = df['volume'].values
            
            clv = ((close - low) - (high - close)) / (high - low + 0.001)
            ad_line = np.cumsum(clv * volume)
            
            signal = 'BUY' if ad_line[-1] > np.mean(ad_line[-20:]) else ('SELL' if ad_line[-1] < np.mean(ad_line[-20:]) else 'NEUTRAL')
            return {
                'ad_line': float(ad_line[-1]),
                'signal': signal,
                'strength': 60.0 if signal != 'NEUTRAL' else 50.0
            }
        except Exception as e:
            self.logger.debug(f"A/D error: {e}")
            return {'signal': 'NEUTRAL', 'strength': 0}
    
    async def analyze_keltner_channel(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Keltner Channel"""
        try:
            close = df['close'].values[-20:]
            high = df['high'].values[-20:]
            low = df['low'].values[-20:]
            
            ema = pd.Series(close).ewm(span=10).mean().values
            tr = np.maximum(high - low, np.abs(high - ema[:-1]), np.abs(low - ema[:-1]))
            atr = np.mean(tr)
            
            upper = ema[-1] + atr
            lower = ema[-1] - atr
            
            signal = 'SELL' if close[-1] > upper else ('BUY' if close[-1] < lower else 'NEUTRAL')
            return {
                'upper': float(upper),
                'middle': float(ema[-1]),
                'lower': float(lower),
                'signal': signal,
                'strength': 65.0 if signal != 'NEUTRAL' else 50.0
            }
        except Exception as e:
            self.logger.debug(f"Keltner error: {e}")
            return {'signal': 'NEUTRAL', 'strength': 0}
    
    async def analyze_pivot_points(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Pivot Points Analysis"""
        try:
            high = df['high'].values[-1]
            low = df['low'].values[-1]
            close = df['close'].values[-1]
            
            pivot = (high + low + close) / 3
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high
            
            return {
                'pivot': float(pivot),
                'resistance_1': float(r1),
                'support_1': float(s1),
                'signal': 'NEUTRAL',
                'strength': 50.0
            }
        except Exception as e:
            self.logger.debug(f"Pivot error: {e}")
            return {'signal': 'NEUTRAL', 'strength': 0}
    
    async def analyze_supertrend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Supertrend Indicator"""
        try:
            high = df['high'].values[-20:]
            low = df['low'].values[-20:]
            close = df['close'].values[-20:]
            
            hl_avg = (high + low) / 2
            atr = np.mean(np.maximum(high - low, np.abs(high - close[:-1]), np.abs(low - close[:-1])))
            
            basic_ub = hl_avg + 3 * atr
            basic_lb = hl_avg - 3 * atr
            
            signal = 'BUY' if close[-1] > basic_ub[-1] else ('SELL' if close[-1] < basic_lb[-1] else 'NEUTRAL')
            return {
                'upper_band': float(basic_ub[-1]),
                'lower_band': float(basic_lb[-1]),
                'signal': signal,
                'strength': 70.0 if signal != 'NEUTRAL' else 50.0
            }
        except Exception as e:
            self.logger.debug(f"Supertrend error: {e}")
            return {'signal': 'NEUTRAL', 'strength': 0}
    
    async def analyze_vwap(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Volume Weighted Average Price"""
        try:
            typical_price = (df['high'].values + df['low'].values + df['close'].values) / 3
            vwap = np.sum(typical_price * df['volume'].values) / np.sum(df['volume'].values)
            current = df['close'].values[-1]
            
            signal = 'BUY' if current < vwap else ('SELL' if current > vwap else 'NEUTRAL')
            return {
                'vwap': float(vwap),
                'current_price': float(current),
                'signal': signal,
                'strength': 60.0 if signal != 'NEUTRAL' else 50.0
            }
        except Exception as e:
            self.logger.debug(f"VWAP error: {e}")
            return {'signal': 'NEUTRAL', 'strength': 0}
    
    async def analyze_ichimoku_extended(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extended Ichimoku Analysis"""
        try:
            high = df['high'].values[-52:]
            low = df['low'].values[-52:]
            close = df['close'].values[-52:]
            
            # Tenkan
            tenkan_high = np.max(high[-9:])
            tenkan_low = np.min(low[-9:])
            tenkan = (tenkan_high + tenkan_low) / 2
            
            # Kijun
            kijun_high = np.max(high[-26:])
            kijun_low = np.min(low[-26:])
            kijun = (kijun_high + kijun_low) / 2
            
            signal = 'BUY' if tenkan > kijun else ('SELL' if tenkan < kijun else 'NEUTRAL')
            return {
                'tenkan': float(tenkan),
                'kijun': float(kijun),
                'signal': signal,
                'strength': 75.0 if signal != 'NEUTRAL' else 50.0
            }
        except Exception as e:
            self.logger.debug(f"Ichimoku error: {e}")
            return {'signal': 'NEUTRAL', 'strength': 0}
    
    def _calculate_composite_signal(self, indicators: Dict[str, Any]) -> str:
        """Calculate composite signal from all indicators"""
        buy_count = 0
        sell_count = 0
        total_signals = 0
        
        for indicator_name, data in indicators.items():
            if isinstance(data, dict) and 'signal' in data:
                signal = data['signal']
                if signal == 'BUY':
                    buy_count += 1
                elif signal == 'SELL':
                    sell_count += 1
                total_signals += 1
        
        if total_signals == 0:
            return 'NEUTRAL'
        
        buy_ratio = buy_count / total_signals
        sell_ratio = sell_count / total_signals
        
        if buy_ratio > 0.6:
            return 'STRONG_BUY'
        elif buy_ratio > 0.5:
            return 'BUY'
        elif sell_ratio > 0.6:
            return 'STRONG_SELL'
        elif sell_ratio > 0.5:
            return 'SELL'
        else:
            return 'NEUTRAL'
    
    def _calculate_overall_strength(self, indicators: Dict[str, Any]) -> float:
        """Calculate overall signal strength"""
        strengths = []
        for indicator_name, data in indicators.items():
            if isinstance(data, dict) and 'strength' in data:
                strengths.append(data['strength'])
        
        return float(np.mean(strengths)) if strengths else 50.0

# Global instance
atas_analyzer = ATASIntegratedAnalyzer()
