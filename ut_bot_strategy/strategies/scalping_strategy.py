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
- Numba JIT compilation for maximum performance
- LRU caching for repeated calculations

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
from functools import lru_cache
from typing import Optional, Dict, List, Any, Tuple
import numpy as np
import pandas as pd

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

logger = logging.getLogger(__name__)


@jit(nopython=True, cache=True, fastmath=True)
def _jit_calculate_rsi_core(prices: np.ndarray, period: int) -> np.ndarray:
    """JIT-compiled RSI calculation core"""
    n = len(prices)
    rsi = np.full(n, 50.0)
    
    if n < period + 1:
        return rsi
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    alpha = 1.0 / period
    avg_gain = np.sum(gains[:period]) / period
    avg_loss = np.sum(losses[:period]) / period
    
    for i in range(period, n - 1):
        avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
        avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss
        
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


@jit(nopython=True, cache=True, fastmath=True)
def _jit_calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """JIT-compiled EMA calculation"""
    n = len(prices)
    ema = np.zeros(n)
    
    if n < period:
        return ema
    
    alpha = 2.0 / (period + 1)
    ema[period - 1] = np.mean(prices[:period])
    
    for i in range(period, n):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    
    return ema


@jit(nopython=True, cache=True, fastmath=True)
def _jit_calculate_macd_core(prices: np.ndarray, fast: int, slow: int, signal: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """JIT-compiled MACD calculation"""
    ema_fast = _jit_calculate_ema(prices, fast)
    ema_slow = _jit_calculate_ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _jit_calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


@jit(nopython=True, cache=True, fastmath=True)
def _jit_calculate_stochastic_core(high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int, d_period: int, smooth: int) -> Tuple[np.ndarray, np.ndarray]:
    """JIT-compiled Stochastic calculation"""
    n = len(close)
    stoch_k_raw = np.zeros(n)
    
    for i in range(k_period - 1, n):
        highest = np.max(high[i - k_period + 1:i + 1])
        lowest = np.min(low[i - k_period + 1:i + 1])
        range_val = highest - lowest
        if range_val > 1e-10:
            stoch_k_raw[i] = 100.0 * (close[i] - lowest) / range_val
        else:
            stoch_k_raw[i] = 50.0
    
    stoch_k = np.zeros(n)
    for i in range(smooth - 1, n):
        stoch_k[i] = np.mean(stoch_k_raw[i - smooth + 1:i + 1])
    
    stoch_d = np.zeros(n)
    for i in range(d_period - 1, n):
        stoch_d[i] = np.mean(stoch_k[i - d_period + 1:i + 1])
    
    stoch_k = np.where(stoch_k == 0, 50.0, stoch_k)
    stoch_d = np.where(stoch_d == 0, 50.0, stoch_d)
    
    return stoch_k, stoch_d


@jit(nopython=True, cache=True, fastmath=True)
def _jit_calculate_williams_r_core(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """JIT-compiled Williams %R calculation"""
    n = len(close)
    williams_r = np.full(n, -50.0)
    
    for i in range(period - 1, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        range_val = highest - lowest
        if range_val > 1e-10:
            williams_r[i] = -100.0 * (highest - close[i]) / range_val
    
    return williams_r


@jit(nopython=True, cache=True, fastmath=True)
def _jit_calculate_atr_core(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """JIT-compiled ATR calculation"""
    n = len(close)
    atr = np.zeros(n)
    
    if n < 2:
        return atr
    
    true_range = np.zeros(n)
    true_range[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        true_range[i] = max(tr1, tr2, tr3)
    
    alpha = 2.0 / (period + 1)
    atr[period - 1] = np.mean(true_range[:period])
    
    for i in range(period, n):
        atr[i] = alpha * true_range[i] + (1 - alpha) * atr[i - 1]
    
    return atr


@jit(nopython=True, cache=True, fastmath=True)
def _jit_calculate_volume_ratio_core(volume: np.ndarray, lookback: int) -> np.ndarray:
    """JIT-compiled volume ratio calculation"""
    n = len(volume)
    ratio = np.ones(n)
    
    for i in range(lookback, n):
        avg_vol = np.mean(volume[i - lookback:i])
        if avg_vol > 0:
            ratio[i] = volume[i] / avg_vol
    
    return ratio


@jit(nopython=True, cache=True, fastmath=True)
def _jit_calculate_trend_strength_core(close: np.ndarray, fast_period: int, slow_period: int) -> Tuple[np.ndarray, np.ndarray]:
    """JIT-compiled trend strength calculation"""
    ema_fast = _jit_calculate_ema(close, fast_period)
    ema_slow = _jit_calculate_ema(close, slow_period)
    
    n = len(close)
    trend_strength = np.zeros(n)
    trend_direction = np.zeros(n)
    
    for i in range(slow_period, n):
        if ema_slow[i] != 0:
            diff = (ema_fast[i] - ema_slow[i]) / ema_slow[i]
            trend_strength[i] = min(abs(diff), 1.0)
            trend_direction[i] = 1.0 if diff > 0 else (-1.0 if diff < 0 else 0.0)
    
    return trend_strength, trend_direction


@jit(nopython=True, cache=True)
def _jit_calculate_confidence_core(
    momentum_score: float,
    pattern_strength: float,
    volume_score: float,
    of_score: float,
    trend_score: float,
    volatility_score: float,
    momentum_weight: float,
    pattern_weight: float,
    volume_weight: float,
    order_flow_weight: float,
    trend_weight: float,
    volatility_weight: float
) -> float:
    """JIT-compiled confidence calculation"""
    confidence = (
        momentum_score * momentum_weight +
        pattern_strength * pattern_weight +
        volume_score * volume_weight +
        of_score * order_flow_weight +
        trend_score * trend_weight +
        volatility_score * volatility_weight
    )
    return max(0.0, min(1.0, confidence))


class ScalpingMode(str, Enum):
    """Scalping aggressiveness modes"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    ULTRA_FAST = "ultra_fast"
    TURBO = "turbo"


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


MODE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "conservative": {
        "rsi_period": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "stoch_k": 14,
        "stoch_d": 3,
        "stoch_smooth": 3,
        "atr_period": 14,
        "min_confidence": 0.75,
        "volume_spike_threshold": 2.0,
        "cooldown_seconds": 60,
        "max_signals_per_hour": 10,
    },
    "balanced": {
        "rsi_period": 7,
        "macd_fast": 6,
        "macd_slow": 13,
        "macd_signal": 4,
        "stoch_k": 5,
        "stoch_d": 3,
        "stoch_smooth": 3,
        "atr_period": 7,
        "min_confidence": 0.65,
        "volume_spike_threshold": 1.5,
        "cooldown_seconds": 30,
        "max_signals_per_hour": 20,
    },
    "aggressive": {
        "rsi_period": 6,
        "macd_fast": 5,
        "macd_slow": 11,
        "macd_signal": 4,
        "stoch_k": 4,
        "stoch_d": 3,
        "stoch_smooth": 2,
        "atr_period": 6,
        "min_confidence": 0.55,
        "volume_spike_threshold": 1.2,
        "cooldown_seconds": 15,
        "max_signals_per_hour": 40,
    },
    "ultra_fast": {
        "rsi_period": 5,
        "macd_fast": 4,
        "macd_slow": 9,
        "macd_signal": 3,
        "stoch_k": 3,
        "stoch_d": 2,
        "stoch_smooth": 2,
        "atr_period": 5,
        "min_confidence": 0.50,
        "volume_spike_threshold": 1.0,
        "cooldown_seconds": 10,
        "max_signals_per_hour": 60,
    },
    "turbo": {
        "rsi_period": 5,
        "macd_fast": 4,
        "macd_slow": 9,
        "macd_signal": 3,
        "stoch_k": 3,
        "stoch_d": 2,
        "stoch_smooth": 2,
        "atr_period": 5,
        "min_confidence": 0.45,
        "volume_spike_threshold": 0.8,
        "cooldown_seconds": 10,
        "max_signals_per_hour": 40,
    },
}


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

    @classmethod
    def from_mode(cls, mode: ScalpingMode) -> 'ScalpingConfig':
        """Create config from mode preset"""
        mode_key = mode.value
        preset = MODE_CONFIGS.get(mode_key, MODE_CONFIGS["balanced"])
        return cls(
            mode=mode,
            rsi_period=preset["rsi_period"],
            macd_fast=preset["macd_fast"],
            macd_slow=preset["macd_slow"],
            macd_signal=preset["macd_signal"],
            stoch_k=preset["stoch_k"],
            stoch_d=preset["stoch_d"],
            stoch_smooth=preset["stoch_smooth"],
            atr_period=preset["atr_period"],
            min_confidence=preset["min_confidence"],
            volume_spike_threshold=preset["volume_spike_threshold"],
            cooldown_seconds=preset["cooldown_seconds"],
            max_signals_per_hour=preset["max_signals_per_hour"],
        )


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
    - Numba JIT compilation for speed
    - LRU caching for repeated calculations
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
        
        self._indicator_cache: Dict[str, Tuple[np.ndarray, datetime]] = {}
        self._indicator_cache_ttl = 2.0
        
        self._initialized = False
        
        logger.info(f"ScalpingStrategy initialized in {self.config.mode.value} mode (Numba: {NUMBA_AVAILABLE})")
    
    def set_order_flow_service(self, service: Any) -> None:
        """Connect order flow service for real-time data"""
        self._order_flow_service = service
        logger.info("Order flow service connected to ScalpingStrategy")
    
    def _get_cached_indicator(self, key: str) -> Optional[np.ndarray]:
        """Get cached indicator if still valid"""
        if key in self._indicator_cache:
            data, timestamp = self._indicator_cache[key]
            if (datetime.now() - timestamp).total_seconds() < self._indicator_cache_ttl:
                return data
        return None
    
    def _set_cached_indicator(self, key: str, data: np.ndarray) -> None:
        """Cache indicator result"""
        self._indicator_cache[key] = (data, datetime.now())
        if len(self._indicator_cache) > 50:
            oldest_key = min(self._indicator_cache, key=lambda k: self._indicator_cache[k][1])
            del self._indicator_cache[oldest_key]
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator with JIT optimization"""
        cache_key = f"rsi_{period}_{len(prices)}"
        cached = self._get_cached_indicator(cache_key)
        if cached is not None:
            return pd.Series(cached, index=prices.index)
        
        prices_arr = prices.values.astype(np.float64)
        rsi_arr = _jit_calculate_rsi_core(prices_arr, period)
        self._set_cached_indicator(cache_key, rsi_arr)
        return pd.Series(rsi_arr, index=prices.index)
    
    def _calculate_macd(
        self, 
        prices: pd.Series,
        fast: int,
        slow: int,
        signal: int
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator with JIT optimization"""
        cache_key = f"macd_{fast}_{slow}_{signal}_{len(prices)}"
        cached = self._get_cached_indicator(cache_key)
        if cached is not None:
            macd_line = pd.Series(cached[0], index=prices.index)
            signal_line = pd.Series(cached[1], index=prices.index)
            histogram = pd.Series(cached[2], index=prices.index)
            return macd_line, signal_line, histogram
        
        prices_arr = prices.values.astype(np.float64)
        macd_arr, signal_arr, hist_arr = _jit_calculate_macd_core(prices_arr, fast, slow, signal)
        self._set_cached_indicator(cache_key, np.array([macd_arr, signal_arr, hist_arr]))
        
        return (
            pd.Series(macd_arr, index=prices.index),
            pd.Series(signal_arr, index=prices.index),
            pd.Series(hist_arr, index=prices.index)
        )
    
    def _calculate_stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int,
        d_period: int,
        smooth: int
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic oscillator with JIT optimization"""
        cache_key = f"stoch_{k_period}_{d_period}_{smooth}_{len(close)}"
        cached = self._get_cached_indicator(cache_key)
        if cached is not None:
            return pd.Series(cached[0], index=close.index), pd.Series(cached[1], index=close.index)
        
        high_arr = high.values.astype(np.float64)
        low_arr = low.values.astype(np.float64)
        close_arr = close.values.astype(np.float64)
        
        stoch_k_arr, stoch_d_arr = _jit_calculate_stochastic_core(
            high_arr, low_arr, close_arr, k_period, d_period, smooth
        )
        self._set_cached_indicator(cache_key, np.array([stoch_k_arr, stoch_d_arr]))
        
        return pd.Series(stoch_k_arr, index=close.index), pd.Series(stoch_d_arr, index=close.index)
    
    def _calculate_williams_r(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int
    ) -> pd.Series:
        """Calculate Williams %R indicator with JIT optimization"""
        cache_key = f"williams_{period}_{len(close)}"
        cached = self._get_cached_indicator(cache_key)
        if cached is not None:
            return pd.Series(cached, index=close.index)
        
        high_arr = high.values.astype(np.float64)
        low_arr = low.values.astype(np.float64)
        close_arr = close.values.astype(np.float64)
        
        williams_arr = _jit_calculate_williams_r_core(high_arr, low_arr, close_arr, period)
        self._set_cached_indicator(cache_key, williams_arr)
        
        return pd.Series(williams_arr, index=close.index)
    
    def _calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int
    ) -> pd.Series:
        """Calculate Average True Range with JIT optimization"""
        cache_key = f"atr_{period}_{len(close)}"
        cached = self._get_cached_indicator(cache_key)
        if cached is not None:
            return pd.Series(cached, index=close.index)
        
        high_arr = high.values.astype(np.float64)
        low_arr = low.values.astype(np.float64)
        close_arr = close.values.astype(np.float64)
        
        atr_arr = _jit_calculate_atr_core(high_arr, low_arr, close_arr, period)
        self._set_cached_indicator(cache_key, atr_arr)
        
        return pd.Series(atr_arr, index=close.index)
    
    def _calculate_volume_ratio(self, volume: pd.Series, lookback: int = 20) -> pd.Series:
        """Calculate volume ratio with JIT optimization"""
        cache_key = f"volume_ratio_{lookback}_{len(volume)}"
        cached = self._get_cached_indicator(cache_key)
        if cached is not None:
            return pd.Series(cached, index=volume.index)
        
        volume_arr = volume.values.astype(np.float64)
        ratio_arr = _jit_calculate_volume_ratio_core(volume_arr, lookback)
        self._set_cached_indicator(cache_key, ratio_arr)
        
        return pd.Series(ratio_arr, index=volume.index)
    
    def _calculate_trend_strength(
        self,
        close: pd.Series,
        fast_period: int,
        slow_period: int
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate trend strength with JIT optimization"""
        cache_key = f"trend_{fast_period}_{slow_period}_{len(close)}"
        cached = self._get_cached_indicator(cache_key)
        if cached is not None:
            return pd.Series(cached[0], index=close.index), pd.Series(cached[1], index=close.index)
        
        close_arr = close.values.astype(np.float64)
        strength_arr, direction_arr = _jit_calculate_trend_strength_core(close_arr, fast_period, slow_period)
        self._set_cached_indicator(cache_key, np.array([strength_arr, direction_arr]))
        
        return pd.Series(strength_arr, index=close.index), pd.Series(direction_arr, index=close.index)
    
    @lru_cache(maxsize=128)
    def _detect_pattern_cached(
        self,
        curr_open: float,
        curr_high: float,
        curr_low: float,
        curr_close: float,
        prev_open: float,
        prev_close: float
    ) -> Tuple[str, float]:
        """Cached pattern detection"""
        body = abs(curr_close - curr_open)
        upper_wick = curr_high - max(curr_open, curr_close)
        lower_wick = min(curr_open, curr_close) - curr_low
        total_range = curr_high - curr_low
        
        if total_range == 0:
            return PatternType.NONE.value, 0.0
        
        body_ratio = body / total_range
        
        if body_ratio < 0.1:
            return PatternType.DOJI.value, 0.3
        
        prev_body = abs(prev_close - prev_open)
        if body > prev_body * 1.5:
            if curr_close > curr_open and prev_close < prev_open:
                if curr_close > prev_open and curr_open < prev_close:
                    return PatternType.BULLISH_ENGULFING.value, 0.8
            elif curr_close < curr_open and prev_close > prev_open:
                if curr_close < prev_open and curr_open > prev_close:
                    return PatternType.BEARISH_ENGULFING.value, 0.8
        
        if lower_wick > body * 2 and upper_wick < body * 0.5:
            if curr_close > curr_open:
                return PatternType.BULLISH_PIN_BAR.value, 0.7
        elif upper_wick > body * 2 and lower_wick < body * 0.5:
            if curr_close < curr_open:
                return PatternType.BEARISH_PIN_BAR.value, 0.7
        
        if body_ratio > 0.6:
            if curr_close > curr_open:
                return PatternType.BULLISH_MOMENTUM.value, 0.6
            else:
                return PatternType.BEARISH_MOMENTUM.value, 0.6
        
        return PatternType.NONE.value, 0.0
    
    def _detect_pattern(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> Tuple[PatternType, float]:
        """Detect price action patterns with caching"""
        if len(close) < 3:
            return PatternType.NONE, 0.0
        
        pattern_str, strength = self._detect_pattern_cached(
            round(open_.iloc[-1], 8),
            round(high.iloc[-1], 8),
            round(low.iloc[-1], 8),
            round(close.iloc[-1], 8),
            round(open_.iloc[-2], 8),
            round(close.iloc[-2], 8)
        )
        
        return PatternType(pattern_str), strength
    
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
        """Calculate overall signal confidence with JIT optimization"""
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
        
        confidence = _jit_calculate_confidence_core(
            momentum_score,
            pattern_strength,
            volume_score,
            of_score,
            trend_score,
            volatility_score,
            self.config.momentum_weight,
            self.config.pattern_weight,
            self.config.volume_weight,
            self.config.order_flow_weight,
            self.config.trend_weight,
            self.config.volatility_weight
        )
        
        return confidence
    
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
                    'of_direction': of_direction,
                    'numba_enabled': NUMBA_AVAILABLE
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
                'short_signals': 0,
                'numba_enabled': NUMBA_AVAILABLE
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
            'mode': self.config.mode.value,
            'numba_enabled': NUMBA_AVAILABLE
        }
    
    def clear_cache(self) -> None:
        """Clear indicator cache"""
        self._indicator_cache.clear()
        self._detect_pattern_cached.cache_clear()
        logger.info("Indicator cache cleared")


def create_scalping_strategy(
    mode: str = "balanced",
    order_flow_service: Optional[Any] = None
) -> ScalpingStrategy:
    """
    Factory function to create a ScalpingStrategy
    
    Args:
        mode: 'conservative', 'balanced', 'aggressive', 'ultra_fast', or 'turbo'
        order_flow_service: Optional OrderFlowMetricsService
        
    Returns:
        Configured ScalpingStrategy instance
    """
    mode_enum = ScalpingMode(mode.lower())
    config = ScalpingConfig.from_mode(mode_enum)
    
    return ScalpingStrategy(config=config, order_flow_service=order_flow_service)
