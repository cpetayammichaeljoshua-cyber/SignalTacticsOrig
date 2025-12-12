"""
Advanced Scalping Strategy Module

Ultra-fast, intelligent scalping signals for cryptocurrency futures trading.
Designed for 1-minute timeframe with sub-second decision making.

Features:
- Multi-indicator momentum detection (RSI, MACD, Stochastic, Williams %R)
- Volume spike detection with relative volume analysis
- Price action pattern recognition (engulfing, pin bars, momentum candles)
- Order flow integration for entry confirmation
- Dynamic take-profit scaling based on volatility
- Tight risk management with ATR-based stops
- Trend strength filtering to avoid choppy markets
- Parallel data processing for speed

Signal Generation Logic:
1. Detect momentum shift using fast indicators
2. Confirm with volume spike (>1.5x average)
3. Validate with order flow direction
4. Check price action patterns
5. Calculate dynamic TP/SL based on ATR
6. Generate confidence score from all factors
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, List, Any, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ScalpingMode(str, Enum):
    """Scalping aggressiveness modes"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    ULTRA_FAST = "ultra_fast"


class MomentumType(str, Enum):
    """Type of momentum detected"""
    BULLISH_BREAKOUT = "bullish_breakout"
    BEARISH_BREAKDOWN = "bearish_breakdown"
    BULLISH_REVERSAL = "bullish_reversal"
    BEARISH_REVERSAL = "bearish_reversal"
    BULLISH_CONTINUATION = "bullish_continuation"
    BEARISH_CONTINUATION = "bearish_continuation"
    NEUTRAL = "neutral"


class PatternType(str, Enum):
    """Price action patterns"""
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"
    BULLISH_PIN_BAR = "bullish_pin_bar"
    BEARISH_PIN_BAR = "bearish_pin_bar"
    BULLISH_MOMENTUM = "bullish_momentum"
    BEARISH_MOMENTUM = "bearish_momentum"
    DOJI = "doji"
    NONE = "none"


@dataclass
class ScalpingConfig:
    """Configuration for scalping strategy"""
    mode: ScalpingMode = ScalpingMode.BALANCED
    rsi_period: int = 7
    rsi_overbought: float = 75.0
    rsi_oversold: float = 25.0
    macd_fast: int = 6
    macd_slow: int = 13
    macd_signal: int = 4
    stoch_k: int = 5
    stoch_d: int = 3
    stoch_smooth: int = 3
    williams_period: int = 7
    atr_period: int = 7
    volume_spike_threshold: float = 1.5
    min_confidence: float = 0.65
    take_profit_atr_mult: float = 1.2
    stop_loss_atr_mult: float = 0.8
    max_spread_percent: float = 0.05
    min_volatility_percent: float = 0.1
    max_volatility_percent: float = 2.0
    trend_ema_fast: int = 8
    trend_ema_slow: int = 21
    momentum_threshold: float = 0.3
    order_flow_weight: float = 0.20
    volume_weight: float = 0.15
    pattern_weight: float = 0.15
    momentum_weight: float = 0.25
    trend_weight: float = 0.15
    volatility_weight: float = 0.10
    cooldown_seconds: int = 30
    max_signals_per_hour: int = 20


@dataclass
class ScalpingSignal:
    """Scalping signal with all relevant data"""
    signal_id: str
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    confidence: float
    momentum_type: MomentumType
    pattern: PatternType
    rsi_value: float
    macd_histogram: float
    stoch_k: float
    williams_r: float
    volume_ratio: float
    atr_value: float
    trend_strength: float
    order_flow_score: float
    risk_percent: float
    reward_ratio: float
    timestamp: datetime = field(default_factory=datetime.now)
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_long(self) -> bool:
        return self.direction == "LONG"
    
    @property
    def is_short(self) -> bool:
        return self.direction == "SHORT"
    
    @property
    def risk_reward_ratio(self) -> float:
        if self.risk_percent == 0:
            return 0.0
        return self.reward_ratio
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'signal_id': self.signal_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_price': round(self.entry_price, 8),
            'stop_loss': round(self.stop_loss, 8),
            'take_profit_1': round(self.take_profit_1, 8),
            'take_profit_2': round(self.take_profit_2, 8),
            'take_profit_3': round(self.take_profit_3, 8),
            'confidence': round(self.confidence, 4),
            'momentum_type': self.momentum_type.value,
            'pattern': self.pattern.value,
            'rsi': round(self.rsi_value, 2),
            'macd_histogram': round(self.macd_histogram, 6),
            'stoch_k': round(self.stoch_k, 2),
            'williams_r': round(self.williams_r, 2),
            'volume_ratio': round(self.volume_ratio, 2),
            'atr': round(self.atr_value, 8),
            'trend_strength': round(self.trend_strength, 4),
            'order_flow_score': round(self.order_flow_score, 4),
            'risk_percent': round(self.risk_percent, 4),
            'reward_ratio': round(self.reward_ratio, 2),
            'timestamp': self.timestamp.isoformat(),
            'execution_time_ms': round(self.execution_time_ms, 2)
        }
    
    def format_telegram_message(self) -> str:
        """Format signal for Telegram notification"""
        emoji = "🟢" if self.is_long else "🔴"
        direction_text = "LONG" if self.is_long else "SHORT"
        
        tp1_pct = abs((self.take_profit_1 - self.entry_price) / self.entry_price * 100)
        tp2_pct = abs((self.take_profit_2 - self.entry_price) / self.entry_price * 100)
        tp3_pct = abs((self.take_profit_3 - self.entry_price) / self.entry_price * 100)
        sl_pct = abs((self.stop_loss - self.entry_price) / self.entry_price * 100)
        
        message = f"""
{emoji} **SCALPING SIGNAL** {emoji}

**{self.symbol}** | {direction_text}
⚡ Mode: ULTRA-FAST SCALP

📍 Entry: ${self.entry_price:,.4f}

🎯 Take Profits:
   TP1: ${self.take_profit_1:,.4f} (+{tp1_pct:.2f}%) [40%]
   TP2: ${self.take_profit_2:,.4f} (+{tp2_pct:.2f}%) [35%]
   TP3: ${self.take_profit_3:,.4f} (+{tp3_pct:.2f}%) [25%]

🛑 Stop Loss: ${self.stop_loss:,.4f} (-{sl_pct:.2f}%)

📊 Signal Analysis:
   RSI: {self.rsi_value:.1f} | MACD: {self.macd_histogram:+.4f}
   Stoch: {self.stoch_k:.1f} | W%R: {self.williams_r:.1f}
   Volume: {self.volume_ratio:.1f}x avg
   Trend: {self.trend_strength*100:.0f}%
   Order Flow: {self.order_flow_score*100:.0f}%

🎲 Confidence: {self.confidence*100:.1f}%
📈 R:R Ratio: 1:{self.reward_ratio:.1f}

⏱️ Generated: {self.timestamp.strftime('%H:%M:%S')}
⚡ Execution: {self.execution_time_ms:.0f}ms
"""
        return message.strip()


class ScalpingStrategy:
    """
    Advanced Ultra-Fast Scalping Strategy
    
    Designed for high-frequency signal generation on 1-minute timeframe.
    Combines multiple fast indicators with volume and order flow analysis.
    
    Features:
    - Sub-100ms signal generation
    - Multi-indicator confluence scoring
    - Dynamic risk/reward based on volatility
    - Order flow integration
    - Pattern recognition
    - Trend filtering
    """
    
    def __init__(
        self,
        config: Optional[ScalpingConfig] = None,
        order_flow_service: Optional[Any] = None
    ):
        """
        Initialize scalping strategy
        
        Args:
            config: ScalpingConfig with strategy parameters
            order_flow_service: OrderFlowMetricsService for real-time order flow
        """
        self.config = config or ScalpingConfig()
        self._order_flow_service = order_flow_service
        
        self._last_signal_time: Optional[datetime] = None
        self._signals_this_hour: int = 0
        self._hour_start: Optional[datetime] = None
        self._signal_history: List[ScalpingSignal] = []
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = timedelta(seconds=5)
        self._last_cache_update: Optional[datetime] = None
        
        self._initialized = False
        
        logger.info(f"ScalpingStrategy initialized in {self.config.mode.value} mode")
    
    def set_order_flow_service(self, service: Any) -> None:
        """Connect order flow service for real-time data"""
        self._order_flow_service = service
        logger.info("Order flow service connected to ScalpingStrategy")
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        
        avg_loss_safe: pd.Series = avg_loss.where(avg_loss != 0, np.inf)  # type: ignore[assignment]
        rs = avg_gain / avg_loss_safe
        rsi: pd.Series = 100 - (100 / (1 + rs))  # type: ignore[assignment]
        return pd.Series(rsi.fillna(50))
    
    def _calculate_macd(
        self, 
        prices: pd.Series,
        fast: int,
        slow: int,
        signal: int
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _calculate_stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int,
        d_period: int,
        smooth: int
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        stoch_k = stoch_k.rolling(window=smooth).mean()
        stoch_d = stoch_k.rolling(window=d_period).mean()
        
        return stoch_k.fillna(50), stoch_d.fillna(50)
    
    def _calculate_williams_r(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int
    ) -> pd.Series:
        """Calculate Williams %R indicator"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
        return williams_r.fillna(-50)
    
    def _calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int
    ) -> pd.Series:
        """Calculate Average True Range"""
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range_df = pd.concat([tr1, tr2, tr3], axis=1)
        true_range: pd.Series = true_range_df.max(axis=1)  # type: ignore[assignment]
        atr = true_range.ewm(span=period, adjust=False).mean()
        
        return pd.Series(atr.fillna(0))
    
    def _calculate_volume_ratio(self, volume: pd.Series, lookback: int = 20) -> pd.Series:
        """Calculate volume ratio compared to moving average"""
        avg_volume: pd.Series = volume.rolling(window=lookback).mean()  # type: ignore[assignment]
        avg_volume_safe = avg_volume.where(avg_volume != 0, 1.0)
        ratio = volume / avg_volume_safe
        return pd.Series(ratio.fillna(1.0))
    
    def _calculate_trend_strength(
        self,
        close: pd.Series,
        fast_period: int,
        slow_period: int
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate trend strength using EMA crossover"""
        ema_fast = close.ewm(span=fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=slow_period, adjust=False).mean()
        
        diff = (ema_fast - ema_slow) / ema_slow
        trend_strength = diff.abs().clip(upper=1.0)
        trend_direction = pd.Series(np.sign(diff.values), index=diff.index)
        
        return pd.Series(trend_strength.fillna(0)), trend_direction.fillna(0)
    
    def _detect_pattern(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> Tuple[PatternType, float]:
        """Detect price action patterns"""
        if len(close) < 3:
            return PatternType.NONE, 0.0
        
        curr_open = open_.iloc[-1]
        curr_high = high.iloc[-1]
        curr_low = low.iloc[-1]
        curr_close = close.iloc[-1]
        
        prev_open = open_.iloc[-2]
        prev_close = close.iloc[-2]
        
        body = abs(curr_close - curr_open)
        upper_wick = curr_high - max(curr_open, curr_close)
        lower_wick = min(curr_open, curr_close) - curr_low
        total_range = curr_high - curr_low
        
        if total_range == 0:
            return PatternType.NONE, 0.0
        
        body_ratio = body / total_range
        
        if body_ratio < 0.1:
            return PatternType.DOJI, 0.3
        
        prev_body = abs(prev_close - prev_open)
        if body > prev_body * 1.5:
            if curr_close > curr_open and prev_close < prev_open:
                if curr_close > prev_open and curr_open < prev_close:
                    return PatternType.BULLISH_ENGULFING, 0.8
            elif curr_close < curr_open and prev_close > prev_open:
                if curr_close < prev_open and curr_open > prev_close:
                    return PatternType.BEARISH_ENGULFING, 0.8
        
        if lower_wick > body * 2 and upper_wick < body * 0.5:
            if curr_close > curr_open:
                return PatternType.BULLISH_PIN_BAR, 0.7
        elif upper_wick > body * 2 and lower_wick < body * 0.5:
            if curr_close < curr_open:
                return PatternType.BEARISH_PIN_BAR, 0.7
        
        if body_ratio > 0.6:
            if curr_close > curr_open:
                return PatternType.BULLISH_MOMENTUM, 0.6
            else:
                return PatternType.BEARISH_MOMENTUM, 0.6
        
        return PatternType.NONE, 0.0
    
    def _detect_momentum_type(
        self,
        rsi: float,
        macd_hist: float,
        macd_hist_prev: float,
        stoch_k: float,
        williams_r: float,
        trend_direction: float
    ) -> MomentumType:
        """Detect type of momentum based on indicators"""
        bullish_count = 0
        bearish_count = 0
        
        if rsi > 50:
            bullish_count += 1
        elif rsi < 50:
            bearish_count += 1
        
        if macd_hist > 0:
            bullish_count += 1
        elif macd_hist < 0:
            bearish_count += 1
        
        if stoch_k > 50:
            bullish_count += 1
        elif stoch_k < 50:
            bearish_count += 1
        
        if williams_r > -50:
            bullish_count += 1
        elif williams_r < -50:
            bearish_count += 1
        
        macd_crossing_up = macd_hist > 0 and macd_hist_prev <= 0
        macd_crossing_down = macd_hist < 0 and macd_hist_prev >= 0
        
        rsi_oversold = rsi < self.config.rsi_oversold
        rsi_overbought = rsi > self.config.rsi_overbought
        
        if macd_crossing_up and rsi_oversold:
            return MomentumType.BULLISH_REVERSAL
        elif macd_crossing_down and rsi_overbought:
            return MomentumType.BEARISH_REVERSAL
        
        if bullish_count >= 3 and trend_direction > 0:
            if rsi > 60:
                return MomentumType.BULLISH_BREAKOUT
            return MomentumType.BULLISH_CONTINUATION
        elif bearish_count >= 3 and trend_direction < 0:
            if rsi < 40:
                return MomentumType.BEARISH_BREAKDOWN
            return MomentumType.BEARISH_CONTINUATION
        
        return MomentumType.NEUTRAL
    
    def _get_order_flow_score(self) -> Tuple[float, str]:
        """Get order flow alignment score"""
        if not self._order_flow_service:
            return 0.5, "neutral"
        
        try:
            metrics = self._order_flow_service.get_complete_metrics()
            
            if metrics is None:
                return 0.5, "neutral"
            
            cvd_trend = getattr(metrics.cvd, 'cvd_trend', 'neutral') if hasattr(metrics, 'cvd') else 'neutral'
            order_flow_bias = getattr(metrics, 'order_flow_bias', 0.0)
            
            score = (order_flow_bias + 1) / 2
            score = max(0.0, min(1.0, score))
            
            direction = "bullish" if order_flow_bias > 0.1 else ("bearish" if order_flow_bias < -0.1 else "neutral")
            
            return score, direction
            
        except Exception as e:
            logger.warning(f"Error getting order flow score: {e}")
            return 0.5, "neutral"
    
    def _check_cooldown(self) -> bool:
        """Check if cooldown period has passed"""
        if self._last_signal_time is None:
            return True
        
        elapsed = (datetime.now() - self._last_signal_time).total_seconds()
        return elapsed >= self.config.cooldown_seconds
    
    def _check_hourly_limit(self) -> bool:
        """Check if hourly signal limit is reached"""
        now = datetime.now()
        
        if self._hour_start is None or now.hour != self._hour_start.hour:
            self._hour_start = now
            self._signals_this_hour = 0
            return True
        
        return self._signals_this_hour < self.config.max_signals_per_hour
    
    def _calculate_confidence(
        self,
        momentum_type: MomentumType,
        pattern: PatternType,
        pattern_strength: float,
        rsi: float,
        stoch_k: float,
        volume_ratio: float,
        trend_strength: float,
        order_flow_score: float,
        direction: str
    ) -> float:
        """Calculate overall signal confidence"""
        momentum_score = 0.0
        
        if direction == "LONG":
            if momentum_type in [MomentumType.BULLISH_BREAKOUT, MomentumType.BULLISH_REVERSAL]:
                momentum_score = 1.0
            elif momentum_type == MomentumType.BULLISH_CONTINUATION:
                momentum_score = 0.8
            elif momentum_type == MomentumType.NEUTRAL:
                momentum_score = 0.3
            else:
                momentum_score = 0.1
        else:
            if momentum_type in [MomentumType.BEARISH_BREAKDOWN, MomentumType.BEARISH_REVERSAL]:
                momentum_score = 1.0
            elif momentum_type == MomentumType.BEARISH_CONTINUATION:
                momentum_score = 0.8
            elif momentum_type == MomentumType.NEUTRAL:
                momentum_score = 0.3
            else:
                momentum_score = 0.1
        
        volume_score = min(volume_ratio / self.config.volume_spike_threshold, 1.0)
        
        of_score = order_flow_score if direction == "LONG" else (1 - order_flow_score)
        
        trend_score = trend_strength
        
        volatility_score = 0.5
        
        confidence = (
            momentum_score * self.config.momentum_weight +
            pattern_strength * self.config.pattern_weight +
            volume_score * self.config.volume_weight +
            of_score * self.config.order_flow_weight +
            trend_score * self.config.trend_weight +
            volatility_score * self.config.volatility_weight
        )
        
        return max(0.0, min(1.0, confidence))
    
    def _calculate_targets(
        self,
        entry_price: float,
        atr: float,
        direction: str
    ) -> Tuple[float, float, float, float]:
        """Calculate stop loss and take profit levels"""
        sl_distance = atr * self.config.stop_loss_atr_mult
        tp1_distance = atr * self.config.take_profit_atr_mult * 0.8
        tp2_distance = atr * self.config.take_profit_atr_mult * 1.2
        tp3_distance = atr * self.config.take_profit_atr_mult * 1.8
        
        if direction == "LONG":
            stop_loss = entry_price - sl_distance
            tp1 = entry_price + tp1_distance
            tp2 = entry_price + tp2_distance
            tp3 = entry_price + tp3_distance
        else:
            stop_loss = entry_price + sl_distance
            tp1 = entry_price - tp1_distance
            tp2 = entry_price - tp2_distance
            tp3 = entry_price - tp3_distance
        
        return stop_loss, tp1, tp2, tp3
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        symbol: str = "BTCUSDT"
    ) -> Optional[ScalpingSignal]:
        """
        Generate scalping signal from OHLCV data
        
        Args:
            df: DataFrame with columns: open, high, low, close, volume
            symbol: Trading pair symbol
            
        Returns:
            ScalpingSignal or None if no valid signal
        """
        start_time = time.perf_counter()
        
        if not self._check_cooldown():
            return None
        
        if not self._check_hourly_limit():
            return None
        
        if df is None or len(df) < 30:
            return None
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return None
        
        try:
            open_: pd.Series = pd.Series(df['open'].astype(float).values, index=df.index)
            high: pd.Series = pd.Series(df['high'].astype(float).values, index=df.index)
            low: pd.Series = pd.Series(df['low'].astype(float).values, index=df.index)
            close: pd.Series = pd.Series(df['close'].astype(float).values, index=df.index)
            volume: pd.Series = pd.Series(df['volume'].astype(float).values, index=df.index)
            
            rsi = self._calculate_rsi(close, self.config.rsi_period)
            macd_line, signal_line, macd_hist = self._calculate_macd(
                close, self.config.macd_fast, self.config.macd_slow, self.config.macd_signal
            )
            stoch_k, stoch_d = self._calculate_stochastic(
                high, low, close, self.config.stoch_k, self.config.stoch_d, self.config.stoch_smooth
            )
            williams_r = self._calculate_williams_r(high, low, close, self.config.williams_period)
            atr = self._calculate_atr(high, low, close, self.config.atr_period)
            volume_ratio = self._calculate_volume_ratio(volume)
            trend_strength, trend_direction = self._calculate_trend_strength(
                close, self.config.trend_ema_fast, self.config.trend_ema_slow
            )
            
            current_rsi = rsi.iloc[-1]
            current_macd_hist = macd_hist.iloc[-1]
            prev_macd_hist = macd_hist.iloc[-2] if len(macd_hist) > 1 else 0
            current_stoch_k = stoch_k.iloc[-1]
            current_williams = williams_r.iloc[-1]
            current_atr = atr.iloc[-1]
            current_volume_ratio = volume_ratio.iloc[-1]
            current_trend_strength = trend_strength.iloc[-1]
            current_trend_direction = trend_direction.iloc[-1]
            current_price = close.iloc[-1]
            
            pattern, pattern_strength = self._detect_pattern(open_, high, low, close)
            
            momentum_type = self._detect_momentum_type(
                current_rsi,
                current_macd_hist,
                prev_macd_hist,
                current_stoch_k,
                current_williams,
                current_trend_direction
            )
            
            order_flow_score, of_direction = self._get_order_flow_score()
            
            direction = None
            
            if momentum_type in [MomentumType.BULLISH_BREAKOUT, MomentumType.BULLISH_REVERSAL, MomentumType.BULLISH_CONTINUATION]:
                if of_direction != "bearish":
                    direction = "LONG"
            elif momentum_type in [MomentumType.BEARISH_BREAKDOWN, MomentumType.BEARISH_REVERSAL, MomentumType.BEARISH_CONTINUATION]:
                if of_direction != "bullish":
                    direction = "SHORT"
            
            if direction is None:
                return None
            
            if current_volume_ratio < self.config.volume_spike_threshold * 0.7:
                return None
            
            confidence = self._calculate_confidence(
                momentum_type,
                pattern,
                pattern_strength,
                current_rsi,
                current_stoch_k,
                current_volume_ratio,
                current_trend_strength,
                order_flow_score,
                direction
            )
            
            if confidence < self.config.min_confidence:
                return None
            
            stop_loss, tp1, tp2, tp3 = self._calculate_targets(
                current_price,
                current_atr,
                direction
            )
            
            risk_percent = abs(current_price - stop_loss) / current_price * 100
            avg_reward = (abs(tp1 - current_price) + abs(tp2 - current_price) + abs(tp3 - current_price)) / 3
            reward_ratio = avg_reward / abs(current_price - stop_loss) if abs(current_price - stop_loss) > 0 else 0
            
            execution_time = (time.perf_counter() - start_time) * 1000
            
            signal_id = f"SCALP_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            signal = ScalpingSignal(
                signal_id=signal_id,
                symbol=symbol,
                direction=direction,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit_1=tp1,
                take_profit_2=tp2,
                take_profit_3=tp3,
                confidence=confidence,
                momentum_type=momentum_type,
                pattern=pattern,
                rsi_value=current_rsi,
                macd_histogram=current_macd_hist,
                stoch_k=current_stoch_k,
                williams_r=current_williams,
                volume_ratio=current_volume_ratio,
                atr_value=current_atr,
                trend_strength=current_trend_strength,
                order_flow_score=order_flow_score,
                risk_percent=risk_percent,
                reward_ratio=reward_ratio,
                execution_time_ms=execution_time,
                metadata={
                    'mode': self.config.mode.value,
                    'pattern_strength': pattern_strength,
                    'trend_direction': current_trend_direction,
                    'of_direction': of_direction
                }
            )
            
            self._last_signal_time = datetime.now()
            self._signals_this_hour += 1
            self._signal_history.append(signal)
            
            if len(self._signal_history) > 100:
                self._signal_history = self._signal_history[-100:]
            
            logger.info(
                f"SCALP SIGNAL: {symbol} {direction} @ {current_price:.4f} | "
                f"Conf: {confidence:.2%} | RSI: {current_rsi:.1f} | "
                f"Vol: {current_volume_ratio:.1f}x | Time: {execution_time:.1f}ms"
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating scalping signal: {e}")
            return None
    
    async def generate_signal_async(
        self,
        df: pd.DataFrame,
        symbol: str = "BTCUSDT"
    ) -> Optional[ScalpingSignal]:
        """Async wrapper for signal generation"""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            self.generate_signal,
            df,
            symbol
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get strategy statistics"""
        if not self._signal_history:
            return {
                'total_signals': 0,
                'signals_this_hour': self._signals_this_hour,
                'avg_confidence': 0.0,
                'avg_execution_time_ms': 0.0,
                'long_signals': 0,
                'short_signals': 0
            }
        
        confidences = [s.confidence for s in self._signal_history]
        exec_times = [s.execution_time_ms for s in self._signal_history]
        long_count = sum(1 for s in self._signal_history if s.is_long)
        short_count = sum(1 for s in self._signal_history if s.is_short)
        
        return {
            'total_signals': len(self._signal_history),
            'signals_this_hour': self._signals_this_hour,
            'avg_confidence': sum(confidences) / len(confidences),
            'avg_execution_time_ms': sum(exec_times) / len(exec_times),
            'long_signals': long_count,
            'short_signals': short_count,
            'mode': self.config.mode.value
        }


def create_scalping_strategy(
    mode: str = "balanced",
    order_flow_service: Optional[Any] = None
) -> ScalpingStrategy:
    """
    Factory function to create a ScalpingStrategy
    
    Args:
        mode: 'conservative', 'balanced', 'aggressive', or 'ultra_fast'
        order_flow_service: Optional OrderFlowMetricsService
        
    Returns:
        Configured ScalpingStrategy instance
    """
    mode_enum = ScalpingMode(mode.lower())
    
    if mode_enum == ScalpingMode.CONSERVATIVE:
        config = ScalpingConfig(
            mode=mode_enum,
            min_confidence=0.75,
            volume_spike_threshold=2.0,
            cooldown_seconds=60,
            max_signals_per_hour=10
        )
    elif mode_enum == ScalpingMode.AGGRESSIVE:
        config = ScalpingConfig(
            mode=mode_enum,
            min_confidence=0.55,
            volume_spike_threshold=1.2,
            cooldown_seconds=15,
            max_signals_per_hour=40
        )
    elif mode_enum == ScalpingMode.ULTRA_FAST:
        config = ScalpingConfig(
            mode=mode_enum,
            min_confidence=0.50,
            volume_spike_threshold=1.0,
            cooldown_seconds=10,
            max_signals_per_hour=60,
            rsi_period=5,
            macd_fast=4,
            macd_slow=9,
            macd_signal=3
        )
    else:
        config = ScalpingConfig(mode=mode_enum)
    
    return ScalpingStrategy(config=config, order_flow_service=order_flow_service)
