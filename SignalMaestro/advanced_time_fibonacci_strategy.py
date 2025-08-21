#!/usr/bin/env python3
"""
Advanced Time-Based and Fibonacci Trading Strategy
Combines advanced time theory with Fibonacci retracements/extensions for maximum scalping profitability
- Time-based market session analysis
- Fibonacci golden ratios and extensions
- ML-enhanced trade validation
- Optimized for 3m-1h scalping
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import time

@dataclass
class AdvancedScalpingSignal:
    """Advanced scalping signal with time and Fibonacci analysis"""
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    signal_strength: float
    leverage: int = 50
    time_session: str = "UNKNOWN"
    fibonacci_level: float = 0.0
    time_confluence: float = 0.0
    fibonacci_confluence: float = 0.0
    ml_prediction: Dict[str, Any] = None
    optimal_entry_time: datetime = None
    session_volatility: float = 1.0
    fibonacci_extension: float = 0.0
    timestamp: datetime = None

class AdvancedTimeFibonacciStrategy:
    """Advanced strategy combining time theory and Fibonacci analysis"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Time-based trading windows (UTC)
        self.trading_sessions = {
            'ASIA_EARLY': {'start': 0, 'end': 4, 'volatility': 0.7, 'strength': 0.6},
            'ASIA_MAIN': {'start': 4, 'end': 8, 'volatility': 0.8, 'strength': 0.7},
            'LONDON_OPEN': {'start': 8, 'end': 10, 'volatility': 1.3, 'strength': 0.95},
            'LONDON_MAIN': {'start': 10, 'end': 14, 'volatility': 1.1, 'strength': 0.85},
            'NY_OVERLAP': {'start': 14, 'end': 16, 'volatility': 1.4, 'strength': 1.0},
            'NY_MAIN': {'start': 16, 'end': 20, 'volatility': 1.2, 'strength': 0.9},
            'NY_CLOSE': {'start': 20, 'end': 24, 'volatility': 0.9, 'strength': 0.75}
        }

        # Fibonacci levels for retracements and extensions
        self.fibonacci_retracements = [0.236, 0.382, 0.5, 0.618, 0.786]
        self.fibonacci_extensions = [1.272, 1.414, 1.618, 2.618, 4.236]
        self.golden_ratio = 1.618

        # Time-based confluence factors
        self.time_confluence_factors = {
            'session_strength': 0.3,
            'volatility_optimal': 0.25,
            'news_avoidance': 0.2,
            'trend_continuation': 0.15,
            'volume_confirmation': 0.1
        }

        # Fibonacci confluence factors
        self.fib_confluence_factors = {
            'retracement_accuracy': 0.35,
            'extension_projection': 0.3,
            'golden_ratio_proximity': 0.2,
            'multiple_level_confluence': 0.15
        }

        # ML enhancement parameters
        self.ml_confidence_threshold = 0.75
        self.min_signal_strength = 88  # Higher threshold for quality
        self.max_trades_per_hour = 2   # Conservative approach
        self.last_trade_times = {}

    async def analyze_symbol(self, symbol: str, ohlcv_data: Dict[str, List], ml_analyzer=None) -> Optional[AdvancedScalpingSignal]:
        """Analyze symbol with advanced time and Fibonacci theory"""
        try:
            # Rate limiting disabled - always allow trading
            pass

            # Get current time analysis
            time_analysis = self._analyze_time_confluence()

            # Skip if time conditions are poor
            if time_analysis['time_strength'] < 0.6:
                return None

            # Prepare multi-timeframe data
            timeframes = ['3m', '5m', '15m', '1h']
            tf_data = {}

            for tf in timeframes:
                if tf in ohlcv_data and len(ohlcv_data[tf]) >= 100:
                    tf_data[tf] = self._prepare_dataframe(ohlcv_data[tf])

            if len(tf_data) < 3:
                return None

            # Primary analysis on 15m timeframe
            primary_tf = '15m' if '15m' in tf_data else '5m'
            primary_df = tf_data[primary_tf]

            # Calculate Fibonacci levels
            fib_analysis = await self._calculate_fibonacci_levels(primary_df, tf_data)

            if not fib_analysis or fib_analysis['confluence_strength'] < 0.7:
                return None

            # Generate signal with time and Fibonacci confluence
            signal = await self._generate_advanced_signal(
                symbol, primary_df, fib_analysis, time_analysis, ml_analyzer
            )

            if signal and signal.signal_strength >= self.min_signal_strength:
                # Record trade time
                self.last_trade_times[symbol] = datetime.now()
                return signal

            return None

        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return None

    def _can_trade_symbol(self, symbol: str) -> bool:
        """Check if we can trade this symbol (30-minute minimum interval)"""
        if symbol not in self.last_trade_times:
            return True

        time_diff = (datetime.now() - self.last_trade_times[symbol]).total_seconds()
        return time_diff >= 1800  # 30 minutes

    def _prepare_dataframe(self, ohlcv: List[List]) -> pd.DataFrame:
        """Prepare OHLCV dataframe"""
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df.dropna()

    def _analyze_time_confluence(self) -> Dict[str, Any]:
        """Analyze current time for trading confluence"""
        now = datetime.utcnow()
        hour = now.hour
        minute = now.minute
        day_of_week = now.weekday()

        # Determine current session
        current_session = None
        session_data = None

        for session_name, session_info in self.trading_sessions.items():
            if session_info['start'] <= hour < session_info['end']:
                current_session = session_name
                session_data = session_info
                break

        if not current_session:
            return {'time_strength': 0.0, 'session': 'UNKNOWN'}

        # Calculate time-based strength
        time_strength = session_data['strength']

        # Adjust for high-impact times (market opens, closes, news times)
        if self._is_high_impact_time(hour, minute):
            time_strength *= 1.2  # Boost during high-impact times

        # Adjust for day of week (avoid Fridays after 20:00 UTC, Sundays before 22:00)
        if day_of_week == 4 and hour >= 20:  # Friday evening
            time_strength *= 0.7
        elif day_of_week == 6:  # Sunday
            time_strength *= 0.8

        # Adjust for volatility expectations
        volatility_factor = session_data['volatility']
        if 1.0 <= volatility_factor <= 1.3:  # Optimal volatility range
            time_strength *= 1.1

        return {
            'time_strength': min(time_strength, 1.0),
            'session': current_session,
            'volatility_factor': volatility_factor,
            'is_high_impact': self._is_high_impact_time(hour, minute)
        }

    def _is_high_impact_time(self, hour: int, minute: int) -> bool:
        """Check if current time is high-impact (opens, closes, news)"""
        # Major session opens/closes and common news release times
        high_impact_times = [
            (8, 30), (9, 0),   # London open
            (14, 30), (15, 0), # NY overlap
            (16, 0), (16, 30), # NY open
            (21, 0), (22, 0)   # Major closes
        ]

        return (hour, minute) in high_impact_times or minute in [0, 30]

    async def _calculate_fibonacci_levels(self, df: pd.DataFrame, tf_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate advanced Fibonacci levels and confluence"""
        try:
            # Find significant swing high and low (lookback 50 periods)
            lookback = min(50, len(df))
            recent_data = df.tail(lookback)

            swing_high = recent_data['high'].max()
            swing_low = recent_data['low'].min()
            swing_range = swing_high - swing_low

            if swing_range == 0:
                return None

            current_price = df['close'].iloc[-1]

            # Calculate Fibonacci retracement levels
            fib_levels = {}
            for level in self.fibonacci_retracements:
                fib_levels[f'retracement_{level}'] = swing_high - (swing_range * level)

            # Calculate Fibonacci extension levels
            for level in self.fibonacci_extensions:
                fib_levels[f'extension_{level}'] = swing_high + (swing_range * (level - 1))

            # Find closest Fibonacci level to current price
            closest_level = None
            min_distance = float('inf')

            for level_name, level_price in fib_levels.items():
                distance = abs(current_price - level_price) / current_price
                if distance < min_distance:
                    min_distance = distance
                    closest_level = {'name': level_name, 'price': level_price, 'distance': distance}

            # Calculate Fibonacci confluence strength
            confluence_strength = 0.0

            # Proximity to key Fibonacci level (closer = stronger)
            if closest_level and closest_level['distance'] < 0.005:  # Within 0.5%
                confluence_strength += 0.4
            elif closest_level and closest_level['distance'] < 0.01:  # Within 1%
                confluence_strength += 0.25

            # Golden ratio proximity bonus
            golden_levels = [fib_levels.get('retracement_0.618'), fib_levels.get('extension_1.618')]
            for golden_level in golden_levels:
                if golden_level:
                    golden_distance = abs(current_price - golden_level) / current_price
                    if golden_distance < 0.005:
                        confluence_strength += 0.3

            # Multiple timeframe Fibonacci confluence
            mtf_confluence = await self._check_multi_timeframe_fibonacci(current_price, tf_data)
            confluence_strength += mtf_confluence * 0.3

            # Trend direction confluence with Fibonacci
            trend_direction = self._determine_trend_direction(df)
            if trend_direction != 'sideways':
                confluence_strength += 0.2

            return {
                'swing_high': swing_high,
                'swing_low': swing_low,
                'fib_levels': fib_levels,
                'closest_level': closest_level,
                'confluence_strength': min(confluence_strength, 1.0),
                'trend_direction': trend_direction,
                'current_price': current_price
            }

        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci levels: {e}")
            return None

    async def _check_multi_timeframe_fibonacci(self, current_price: float, tf_data: Dict[str, pd.DataFrame]) -> float:
        """Check Fibonacci confluence across multiple timeframes"""
        confluence_score = 0.0
        timeframes_checked = 0

        for tf, df in tf_data.items():
            if len(df) < 30:
                continue

            try:
                # Calculate Fibonacci for this timeframe
                lookback = min(30, len(df))
                recent_data = df.tail(lookback)

                swing_high = recent_data['high'].max()
                swing_low = recent_data['low'].min()
                swing_range = swing_high - swing_low

                if swing_range == 0:
                    continue

                # Check if current price is near any Fibonacci level
                for level in self.fibonacci_retracements + self.fibonacci_extensions:
                    if level <= 1:  # Retracement
                        fib_price = swing_high - (swing_range * level)
                    else:  # Extension
                        fib_price = swing_high + (swing_range * (level - 1))

                    distance = abs(current_price - fib_price) / current_price
                    if distance < 0.01:  # Within 1%
                        confluence_score += 0.2
                        break

                timeframes_checked += 1

            except Exception:
                continue

        return confluence_score / max(timeframes_checked, 1)

    def _determine_trend_direction(self, df: pd.DataFrame) -> str:
        """Determine trend direction using EMAs"""
        try:
            if len(df) < 50:
                return 'sideways'

            # Calculate EMAs
            ema_9 = df['close'].ewm(span=9).mean()
            ema_21 = df['close'].ewm(span=21).mean()
            ema_50 = df['close'].ewm(span=50).mean()

            current_price = df['close'].iloc[-1]
            current_ema_9 = ema_9.iloc[-1]
            current_ema_21 = ema_21.iloc[-1]
            current_ema_50 = ema_50.iloc[-1]

            # Strong uptrend
            if (current_price > current_ema_9 > current_ema_21 > current_ema_50):
                return 'uptrend'
            # Strong downtrend
            elif (current_price < current_ema_9 < current_ema_21 < current_ema_50):
                return 'downtrend'
            else:
                return 'sideways'

        except Exception:
            return 'sideways'

    async def _generate_advanced_signal(self, symbol: str, df: pd.DataFrame, fib_analysis: Dict[str, Any], 
                                      time_analysis: Dict[str, Any], ml_analyzer=None) -> Optional[AdvancedScalpingSignal]:
        """Generate advanced signal combining time and Fibonacci analysis"""
        try:
            current_price = fib_analysis['current_price']
            trend_direction = fib_analysis['trend_direction']
            closest_fib = fib_analysis['closest_level']

            # Determine trade direction
            direction = None

            if trend_direction == 'uptrend' and closest_fib:
                # Look for long opportunities near Fibonacci support
                if 'retracement' in closest_fib['name'] and closest_fib['distance'] < 0.01:
                    direction = 'LONG'
            elif trend_direction == 'downtrend' and closest_fib:
                # Look for short opportunities near Fibonacci resistance
                if 'retracement' in closest_fib['name'] and closest_fib['distance'] < 0.01:
                    direction = 'SHORT'

            if not direction:
                return None

            # ML validation if available
            ml_prediction = None
            if ml_analyzer:
                signal_data = {
                    'symbol': symbol,
                    'signal_strength': 85,
                    'direction': direction,
                    'current_price': current_price,
                    'fibonacci_level': closest_fib['price'] if closest_fib else current_price,
                    'time_session': time_analysis['session'],
                    'volatility': time_analysis['volatility_factor']
                }
                ml_prediction = ml_analyzer.predict_trade_outcome(signal_data)

                # Skip if ML confidence is too low
                if ml_prediction.get('confidence', 0) < self.ml_confidence_threshold * 100:
                    return None

            # Calculate stop loss and take profits using Fibonacci
            atr = self._calculate_atr(df)

            if direction == 'LONG':
                # Stop loss below closest Fibonacci support
                stop_loss = closest_fib['price'] - (atr * 1.5) if closest_fib else current_price - (atr * 2)

                # Take profits at Fibonacci extension levels
                sl_distance = current_price - stop_loss
                tp1 = current_price + (sl_distance * 1.618)  # Golden ratio
                tp2 = current_price + (sl_distance * 2.618)
                tp3 = current_price + (sl_distance * 4.236)

            else:  # SHORT
                # Stop loss above closest Fibonacci resistance
                stop_loss = closest_fib['price'] + (atr * 1.5) if closest_fib else current_price + (atr * 2)

                # Take profits at Fibonacci extension levels
                sl_distance = stop_loss - current_price
                tp1 = current_price - (sl_distance * 1.618)  # Golden ratio
                tp2 = current_price - (sl_distance * 2.618)
                tp3 = current_price - (sl_distance * 4.236)

            # Calculate signal strength
            base_strength = 70

            # Time confluence bonus
            base_strength += time_analysis['time_strength'] * 15

            # Fibonacci confluence bonus
            base_strength += fib_analysis['confluence_strength'] * 15

            # ML prediction bonus
            if ml_prediction and ml_prediction.get('prediction') == 'favorable':
                base_strength += 10

            signal_strength = min(base_strength, 100)

            # Create advanced signal
            signal = AdvancedScalpingSignal(
                symbol=symbol,
                direction=direction,
                entry_price=current_price,
                stop_loss=stop_loss,
                tp1=tp1,
                tp2=tp2,
                tp3=tp3,
                signal_strength=signal_strength,
                leverage=50,
                time_session=time_analysis['session'],
                fibonacci_level=closest_fib['price'] if closest_fib else 0.0,
                time_confluence=time_analysis['time_strength'],
                fibonacci_confluence=fib_analysis['confluence_strength'],
                ml_prediction=ml_prediction,
                optimal_entry_time=datetime.now(),
                session_volatility=time_analysis['volatility_factor'],
                fibonacci_extension=tp3,
                timestamp=datetime.now()
            )

            return signal

        except Exception as e:
            self.logger.error(f"Error generating advanced signal: {e}")
            return None

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            if len(df) < period + 1:
                return (df['high'].iloc[-1] - df['low'].iloc[-1])

            high = df['high'].values
            low = df['low'].values
            close = df['close'].values

            tr_list = []
            for i in range(1, len(df)):
                tr1 = high[i] - low[i]
                tr2 = abs(high[i] - close[i-1])
                tr3 = abs(low[i] - close[i-1])
                tr = max(tr1, tr2, tr3)
                tr_list.append(tr)

            if len(tr_list) >= period:
                atr = sum(tr_list[-period:]) / period
            else:
                atr = sum(tr_list) / len(tr_list)

            return atr

        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return df['high'].iloc[-1] - df['low'].iloc[-1]

    def get_signal_summary(self, signal: AdvancedScalpingSignal) -> Dict[str, Any]:
        """Get comprehensive signal summary"""
        return {
            'symbol': signal.symbol,
            'direction': signal.direction,
            'strength': f"{signal.signal_strength:.1f}%",
            'entry': signal.entry_price,
            'stop_loss': signal.stop_loss,
            'take_profits': [signal.tp1, signal.tp2, signal.tp3],
            'leverage': f"{signal.leverage}x",
            'time_session': signal.time_session,
            'fibonacci_level': signal.fibonacci_level,
            'time_confluence': f"{signal.time_confluence:.1f}%",
            'fibonacci_confluence': f"{signal.fibonacci_confluence:.1f}%",
            'ml_confidence': signal.ml_prediction.get('confidence', 0) if signal.ml_prediction else 0,
            'session_volatility': signal.session_volatility,
            'optimal_entry': signal.optimal_entry_time.strftime('%H:%M:%S UTC'),
            'strategy': 'Advanced Time-Fibonacci Theory'
        }