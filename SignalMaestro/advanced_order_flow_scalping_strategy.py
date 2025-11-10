#!/usr/bin/env python3
"""
Advanced Order Flow Scalping Strategy - Institutional-Grade Analysis
Uses real order book depth, trade-by-trade delta, and market microstructure
- Real Order Book Analysis with bid/ask depth
- Cumulative Volume Delta (CVD) from actual trades
- Smart Money Detection and block trade identification
- Delta Divergence detection
- Aggressive vs Passive flow analysis
- Volume Footprint patterns
- Market Microstructure (spread, depth, tick momentum)
- Ultra-fast execution targeting 60-180s holds
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import time

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False


@dataclass
class OrderFlowSignal:
    """Advanced order flow trading signal with institutional metrics"""
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    signal_strength: float
    leverage: int = 20
    margin_type: str = "cross"
    risk_reward_ratio: float = 2.5
    
    cvd_trend: str = "neutral"
    cvd_strength: float = 0.0
    delta_divergence: bool = False
    bid_ask_imbalance: float = 1.0
    order_book_pressure: str = "balanced"
    smart_money_flow: str = "neutral"
    aggressive_flow_ratio: float = 1.0
    volume_footprint_score: float = 50.0
    
    spread_quality: str = "normal"
    market_depth_score: float = 50.0
    tick_momentum_score: float = 50.0
    liquidity_zone_near: bool = False
    
    execution_urgency: str = "normal"
    expected_hold_seconds: int = 120
    signal_latency_ms: float = 0.0
    confidence_level: float = 0.0
    
    timestamp: Optional[datetime] = None


class AdvancedOrderFlowScalpingStrategy:
    """Ultimate order flow scalping strategy with institutional-grade analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self.timeframes = ['1m', '3m', '5m', '15m']
        self.max_leverage = 50
        self.min_leverage = 15
        self.margin_type = "cross"
        self.risk_percentage = 1.0
        
        self.max_trades_per_hour = 12
        self.min_trade_interval = 30
        self.last_trade_times = {}
        self.hourly_trade_counts = {}
        
        self.cvd_lookback_periods = 20
        self.imbalance_threshold = 1.5
        self.delta_divergence_threshold = 0.7
        self.smart_money_threshold = 2.0
        
        self.order_flow_weights = {
            'cvd_analysis': 0.25,
            'delta_divergence': 0.20,
            'bid_ask_imbalance': 0.18,
            'aggressive_flow': 0.15,
            'volume_footprint': 0.12,
            'smart_money_detection': 0.10
        }
        
        self.min_signal_strength = 72
        
        self.stop_loss_percentages = [0.4, 0.6, 0.9]
        self.profit_target_ratios = [1.5, 2.5, 3.5]
        
        self.cvd_history = {}
        self.delta_history = {}
        self.flow_patterns = {}
        
        self.logger.info("📊 Advanced Order Flow Scalping Strategy initialized")
    
    async def analyze_symbol(self, symbol: str, ohlcv_data: Dict[str, List], 
                           order_book_data: Optional[Dict] = None) -> Optional[OrderFlowSignal]:
        """
        Analyze symbol with advanced order flow analysis
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            ohlcv_data: Multi-timeframe OHLCV data
            order_book_data: Optional order book snapshot for depth analysis
        """
        try:
            start_time = time.time()
            
            if not self._can_trade_symbol(symbol):
                return None
            
            tf_data = {}
            for tf in self.timeframes:
                if tf in ohlcv_data and len(ohlcv_data[tf]) >= 50:
                    tf_data[tf] = self._prepare_dataframe(ohlcv_data[tf])
            
            if len(tf_data) < 2:
                return None
            
            primary_tf = '3m' if '3m' in tf_data else '1m' if '1m' in tf_data else '5m'
            primary_df = tf_data[primary_tf]
            
            trades_data = None
            if order_book_data and 'recent_trades' in order_book_data:
                trades_data = order_book_data['recent_trades']
                self.logger.debug(f"Using {len(trades_data)} trades for {symbol}")
            
            cvd_analysis = await self._analyze_cumulative_volume_delta(primary_df, symbol, trades_data)
            delta_div_analysis = await self._detect_delta_divergence(primary_df, cvd_analysis)
            imbalance_analysis = await self._analyze_bid_ask_imbalance(primary_df, order_book_data)
            aggressive_flow = await self._analyze_aggressive_vs_passive_flow(primary_df)
            volume_footprint = await self._analyze_volume_footprint(primary_df)
            smart_money = await self._detect_smart_money_flow(primary_df)
            
            liquidity_zones = await self._identify_liquidity_zones(primary_df)
            tick_momentum = await self._calculate_tick_momentum(primary_df)
            spread_analysis = await self._analyze_spread_quality(primary_df, order_book_data)
            depth_analysis = await self._analyze_market_depth(order_book_data)
            
            order_flow_score = self._calculate_order_flow_score(
                cvd_analysis, delta_div_analysis, imbalance_analysis,
                aggressive_flow, volume_footprint, smart_money
            )
            
            if order_flow_score < self.min_signal_strength:
                return None
            
            direction = self._determine_direction(
                cvd_analysis, delta_div_analysis, imbalance_analysis,
                aggressive_flow, smart_money
            )
            
            if not direction:
                return None
            
            signal = await self._generate_signal(
                symbol, direction, primary_df, tf_data,
                cvd_analysis, delta_div_analysis, imbalance_analysis,
                aggressive_flow, volume_footprint, smart_money,
                liquidity_zones, tick_momentum, spread_analysis, depth_analysis,
                order_flow_score
            )
            
            if signal:
                latency_ms = (time.time() - start_time) * 1000
                signal.signal_latency_ms = latency_ms
                self._record_trade_time(symbol)
                
                self.logger.info(
                    f"🎯 ORDER FLOW SIGNAL GENERATED: {symbol} {direction} | "
                    f"Score: {order_flow_score:.1f}% | CVD: {cvd_analysis['trend']} | "
                    f"Imbalance: {imbalance_analysis['ratio']:.2f}x | "
                    f"Latency: {latency_ms:.1f}ms"
                )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def _prepare_dataframe(self, ohlcv: List) -> pd.DataFrame:
        """Convert OHLCV list to pandas DataFrame"""
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df.dropna()
    
    async def _analyze_cumulative_volume_delta(self, df: pd.DataFrame, symbol: str,
                                              trades_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Calculate Cumulative Volume Delta (CVD)
        Uses real trade data when available, otherwise estimates from candles
        """
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            open_price = df['open'].values
            volume = df['volume'].values
            
            if len(close) < self.cvd_lookback_periods:
                return {'trend': 'neutral', 'strength': 0, 'delta_values': []}
            
            if trades_data and len(trades_data) > 0:
                self.logger.debug(f"Using {len(trades_data)} real trades for CVD calculation")
                
                buy_volume_total = 0
                sell_volume_total = 0
                
                for trade in trades_data:
                    side = trade.get('side')
                    amount = trade.get('amount', 0)
                    
                    if side == 'buy':
                        buy_volume_total += amount
                    elif side == 'sell':
                        sell_volume_total += amount
                
                real_delta = buy_volume_total - sell_volume_total
                
                delta_values = []
                for i in range(len(close)):
                    if i == len(close) - 1:
                        delta_values.append(real_delta)
                    else:
                        candle_range = high[i] - low[i]
                        if candle_range == 0:
                            delta = 0
                        else:
                            close_position = (close[i] - low[i]) / candle_range
                            if close[i] >= open_price[i]:
                                buy_volume = volume[i] * (0.5 + close_position * 0.5)
                                sell_volume = volume[i] - buy_volume
                            else:
                                sell_volume = volume[i] * (0.5 + (1 - close_position) * 0.5)
                                buy_volume = volume[i] - sell_volume
                            delta = buy_volume - sell_volume
                        delta_values.append(delta)
            else:
                delta_values = []
                for i in range(len(close)):
                    candle_range = high[i] - low[i]
                    if candle_range == 0:
                        delta = 0
                    else:
                        close_position = (close[i] - low[i]) / candle_range
                        
                        if close[i] >= open_price[i]:
                            buy_volume = volume[i] * (0.5 + close_position * 0.5)
                            sell_volume = volume[i] - buy_volume
                        else:
                            sell_volume = volume[i] * (0.5 + (1 - close_position) * 0.5)
                            buy_volume = volume[i] - sell_volume
                        
                        delta = buy_volume - sell_volume
                    
                    delta_values.append(delta)
            
            cvd = np.cumsum(delta_values)
            
            recent_cvd = cvd[-self.cvd_lookback_periods:]
            cvd_slope = np.polyfit(range(len(recent_cvd)), recent_cvd, 1)[0]
            cvd_normalized = cvd_slope / (np.mean(volume[-self.cvd_lookback_periods:]) + 1e-10)
            
            if cvd_normalized > 0.15:
                trend = 'bullish'
                strength = min(abs(cvd_normalized) * 200, 100)
            elif cvd_normalized < -0.15:
                trend = 'bearish'
                strength = min(abs(cvd_normalized) * 200, 100)
            else:
                trend = 'neutral'
                strength = 0
            
            self.cvd_history[symbol] = cvd
            
            return {
                'trend': trend,
                'strength': strength,
                'delta_values': delta_values,
                'cvd': cvd,
                'slope': cvd_normalized
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating CVD: {e}")
            return {'trend': 'neutral', 'strength': 0, 'delta_values': []}
    
    async def _detect_delta_divergence(self, df: pd.DataFrame, cvd_analysis: Dict) -> Dict[str, Any]:
        """Detect price/delta divergence patterns"""
        try:
            close = df['close'].values
            cvd = cvd_analysis.get('cvd', np.array([]))
            
            if len(close) < 10 or len(cvd) < 10:
                return {'divergence_detected': False, 'divergence_type': 'none', 'strength': 0}
            
            close_recent = close[-10:]
            cvd_recent = cvd[-10:]
            
            price_slope = np.polyfit(range(len(close_recent)), close_recent, 1)[0]
            cvd_slope = np.polyfit(range(len(cvd_recent)), cvd_recent, 1)[0]
            
            price_direction = 'up' if price_slope > 0 else 'down'
            cvd_direction = 'up' if cvd_slope > 0 else 'down'
            
            divergence_strength = 0
            divergence_type = 'none'
            divergence_detected = False
            
            if price_direction != cvd_direction:
                divergence_detected = True
                if price_direction == 'up' and cvd_direction == 'down':
                    divergence_type = 'bearish'
                    divergence_strength = min(abs(price_slope - cvd_slope) * 50, 100)
                elif price_direction == 'down' and cvd_direction == 'up':
                    divergence_type = 'bullish'
                    divergence_strength = min(abs(price_slope - cvd_slope) * 50, 100)
            
            return {
                'divergence_detected': divergence_detected,
                'divergence_type': divergence_type,
                'strength': divergence_strength
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting delta divergence: {e}")
            return {'divergence_detected': False, 'divergence_type': 'none', 'strength': 0}
    
    async def _analyze_bid_ask_imbalance(self, df: pd.DataFrame, 
                                         order_book_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze bid/ask imbalance from real order book data when available
        """
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            if len(close) < 5:
                return {'pressure': 'balanced', 'ratio': 1.0, 'strength': 0}
            
            if order_book_data and 'bids' in order_book_data and 'asks' in order_book_data:
                bids = order_book_data.get('bids', [])
                asks = order_book_data.get('asks', [])
                
                if bids and asks:
                    self.logger.debug(f"Using real order book: {len(bids)} bids, {len(asks)} asks")
                    
                    bid_volume = sum([bid[1] for bid in bids[:10]])
                    ask_volume = sum([ask[1] for ask in asks[:10]])
                    
                    total_volume = bid_volume + ask_volume
                    if total_volume > 0:
                        imbalance_ratio = bid_volume / ask_volume if ask_volume > 0 else 2.0
                    else:
                        imbalance_ratio = 1.0
                    
                else:
                    bid_volume, ask_volume, imbalance_ratio = self._estimate_imbalance_from_candles(
                        close, high, low, volume
                    )
            else:
                bid_volume, ask_volume, imbalance_ratio = self._estimate_imbalance_from_candles(
                    close, high, low, volume
                )
            
            if imbalance_ratio > self.imbalance_threshold:
                pressure = 'bullish'
                strength = min((imbalance_ratio - 1) * 50, 100)
            elif imbalance_ratio < (1 / self.imbalance_threshold):
                pressure = 'bearish'
                strength = min((1 - imbalance_ratio) * 50, 100)
            else:
                pressure = 'balanced'
                strength = 0
            
            return {
                'pressure': pressure,
                'ratio': imbalance_ratio,
                'strength': strength,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing bid/ask imbalance: {e}")
            return {'pressure': 'balanced', 'ratio': 1.0, 'strength': 0}
    
    def _estimate_imbalance_from_candles(self, close: np.array, high: np.array, 
                                         low: np.array, volume: np.array) -> Tuple[float, float, float]:
        """Estimate bid/ask imbalance from candle structure"""
        bid_volume = 0
        ask_volume = 0
        
        for i in range(-5, 0):
            candle_range = high[i] - low[i]
            if candle_range == 0:
                continue
            
            close_position = (close[i] - low[i]) / candle_range
            
            if close_position > 0.6:
                bid_volume += volume[i] * close_position
                ask_volume += volume[i] * (1 - close_position)
            elif close_position < 0.4:
                ask_volume += volume[i] * (1 - close_position)
                bid_volume += volume[i] * close_position
            else:
                bid_volume += volume[i] * 0.5
                ask_volume += volume[i] * 0.5
        
        total_volume = bid_volume + ask_volume
        if total_volume > 0:
            imbalance_ratio = bid_volume / ask_volume if ask_volume > 0 else 2.0
        else:
            imbalance_ratio = 1.0
        
        return bid_volume, ask_volume, imbalance_ratio
    
    async def _analyze_aggressive_vs_passive_flow(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze aggressive (market orders) vs passive (limit orders) flow"""
        try:
            volume = df['volume'].values
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            if len(volume) < 10:
                return {'ratio': 1.0, 'dominant_flow': 'neutral', 'strength': 0}
            
            aggressive_volume = 0
            passive_volume = 0
            
            for i in range(-10, 0):
                candle_volatility = (high[i] - low[i]) / close[i] if close[i] > 0 else 0
                
                if candle_volatility > 0.003:
                    aggressive_volume += volume[i] * 0.7
                    passive_volume += volume[i] * 0.3
                else:
                    passive_volume += volume[i] * 0.7
                    aggressive_volume += volume[i] * 0.3
            
            total = aggressive_volume + passive_volume
            ratio = aggressive_volume / passive_volume if passive_volume > 0 else 1.5
            
            if ratio > 1.3:
                dominant_flow = 'aggressive'
                strength = min((ratio - 1) * 50, 100)
            elif ratio < 0.7:
                dominant_flow = 'passive'
                strength = min((1 - ratio) * 50, 100)
            else:
                dominant_flow = 'neutral'
                strength = 0
            
            return {
                'ratio': ratio,
                'dominant_flow': dominant_flow,
                'strength': strength
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing aggressive/passive flow: {e}")
            return {'ratio': 1.0, 'dominant_flow': 'neutral', 'strength': 0}
    
    async def _analyze_volume_footprint(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume distribution at price levels"""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            volume = df['volume'].values
            
            if len(volume) < 20:
                return {'score': 50, 'pattern': 'neutral', 'strength': 0}
            
            recent_volume = volume[-20:]
            volume_ma = np.mean(recent_volume)
            volume_std = np.std(recent_volume)
            
            current_volume = volume[-1]
            volume_zscore = (current_volume - volume_ma) / (volume_std + 1e-10)
            
            upper_wick_ratio = (high[-1] - close[-1]) / (high[-1] - low[-1] + 1e-10)
            lower_wick_ratio = (close[-1] - low[-1]) / (high[-1] - low[-1] + 1e-10)
            
            score = 50
            pattern = 'neutral'
            strength = 0
            
            if volume_zscore > 1.5 and lower_wick_ratio > 0.6:
                pattern = 'bullish_absorption'
                score = min(70 + volume_zscore * 10, 100)
                strength = min(volume_zscore * 30, 100)
            elif volume_zscore > 1.5 and upper_wick_ratio > 0.6:
                pattern = 'bearish_rejection'
                score = min(30 - volume_zscore * 10, 100)
                strength = min(volume_zscore * 30, 100)
            
            return {
                'score': max(0, min(100, score)),
                'pattern': pattern,
                'strength': strength
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume footprint: {e}")
            return {'score': 50, 'pattern': 'neutral', 'strength': 0}
    
    async def _detect_smart_money_flow(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect large institutional orders and smart money activity"""
        try:
            volume = df['volume'].values
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            if len(volume) < 30:
                return {'flow': 'neutral', 'strength': 0, 'detected': False}
            
            volume_ma = np.mean(volume[-30:])
            
            large_volume_candles = []
            for i in range(-10, 0):
                if volume[i] > volume_ma * self.smart_money_threshold:
                    candle_range = high[i] - low[i]
                    close_position = (close[i] - low[i]) / (candle_range + 1e-10)
                    
                    large_volume_candles.append({
                        'index': i,
                        'volume_ratio': volume[i] / volume_ma,
                        'close_position': close_position,
                        'direction': 'buy' if close_position > 0.5 else 'sell'
                    })
            
            if not large_volume_candles:
                return {'flow': 'neutral', 'strength': 0, 'detected': False}
            
            buy_flow = sum(c['volume_ratio'] for c in large_volume_candles if c['direction'] == 'buy')
            sell_flow = sum(c['volume_ratio'] for c in large_volume_candles if c['direction'] == 'sell')
            
            if buy_flow > sell_flow * 1.5:
                flow = 'bullish'
                strength = min((buy_flow / (sell_flow + 1)) * 30, 100)
                detected = True
            elif sell_flow > buy_flow * 1.5:
                flow = 'bearish'
                strength = min((sell_flow / (buy_flow + 1)) * 30, 100)
                detected = True
            else:
                flow = 'neutral'
                strength = 0
                detected = False
            
            return {
                'flow': flow,
                'strength': strength,
                'detected': detected
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting smart money: {e}")
            return {'flow': 'neutral', 'strength': 0, 'detected': False}
    
    async def _identify_liquidity_zones(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify key liquidity zones and price levels"""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            if len(close) < 50:
                return {'zones': [], 'near_zone': False}
            
            price_levels = {}
            for i in range(-50, 0):
                level = round(close[i], -1)
                price_levels[level] = price_levels.get(level, 0) + 1
            
            liquidity_zones = sorted(
                [(level, count) for level, count in price_levels.items() if count >= 3],
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            current_price = close[-1]
            near_zone = False
            
            for zone, _ in liquidity_zones:
                if abs(current_price - zone) / current_price < 0.005:
                    near_zone = True
                    break
            
            return {
                'zones': [z[0] for z in liquidity_zones],
                'near_zone': near_zone
            }
            
        except Exception as e:
            self.logger.error(f"Error identifying liquidity zones: {e}")
            return {'zones': [], 'near_zone': False}
    
    async def _calculate_tick_momentum(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate tick-by-tick momentum"""
        try:
            close = df['close'].values
            
            if len(close) < 10:
                return {'score': 50, 'direction': 'neutral'}
            
            price_changes = np.diff(close[-10:])
            
            up_ticks = np.sum(price_changes > 0)
            down_ticks = np.sum(price_changes < 0)
            
            tick_ratio = (up_ticks + 1) / (down_ticks + 1)
            
            if tick_ratio > 1.5:
                direction = 'bullish'
                score = min(50 + (tick_ratio - 1) * 30, 100)
            elif tick_ratio < 0.67:
                direction = 'bearish'
                score = max(50 - (1 - tick_ratio) * 30, 0)
            else:
                direction = 'neutral'
                score = 50
            
            return {'score': score, 'direction': direction}
            
        except Exception as e:
            self.logger.error(f"Error calculating tick momentum: {e}")
            return {'score': 50, 'direction': 'neutral'}
    
    async def _analyze_spread_quality(self, df: pd.DataFrame, 
                                     order_book_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze bid-ask spread quality"""
        try:
            close = df['close'].values[-1]
            
            if order_book_data and 'bids' in order_book_data and 'asks' in order_book_data:
                bids = order_book_data.get('bids', [])
                asks = order_book_data.get('asks', [])
                
                if bids and asks:
                    spread = asks[0][0] - bids[0][0]
                    spread_pct = (spread / close) * 100
                    
                    if spread_pct < 0.01:
                        quality = 'excellent'
                    elif spread_pct < 0.05:
                        quality = 'good'
                    elif spread_pct < 0.1:
                        quality = 'normal'
                    else:
                        quality = 'wide'
                    
                    return {'quality': quality, 'spread_pct': spread_pct}
            
            return {'quality': 'normal', 'spread_pct': 0.05}
            
        except Exception as e:
            self.logger.error(f"Error analyzing spread: {e}")
            return {'quality': 'normal', 'spread_pct': 0.05}
    
    async def _analyze_market_depth(self, order_book_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze overall market depth"""
        try:
            if not order_book_data or 'bids' not in order_book_data or 'asks' not in order_book_data:
                return {'score': 50, 'quality': 'normal'}
            
            bids = order_book_data.get('bids', [])
            asks = order_book_data.get('asks', [])
            
            if not bids or not asks:
                return {'score': 50, 'quality': 'normal'}
            
            total_bid_volume = sum([b[1] for b in bids[:20]])
            total_ask_volume = sum([a[1] for a in asks[:20]])
            total_depth = total_bid_volume + total_ask_volume
            
            depth_levels = min(len(bids), len(asks))
            
            score = 50
            if depth_levels >= 20 and total_depth > 0:
                score = min(50 + depth_levels * 2, 100)
            
            if score >= 80:
                quality = 'deep'
            elif score >= 60:
                quality = 'good'
            else:
                quality = 'normal'
            
            return {'score': score, 'quality': quality}
            
        except Exception as e:
            self.logger.error(f"Error analyzing market depth: {e}")
            return {'score': 50, 'quality': 'normal'}
    
    def _calculate_order_flow_score(self, cvd_analysis: Dict, delta_div: Dict,
                                    imbalance: Dict, aggressive_flow: Dict,
                                    footprint: Dict, smart_money: Dict) -> float:
        """Calculate weighted order flow score"""
        try:
            score = 0
            
            score += cvd_analysis['strength'] * self.order_flow_weights['cvd_analysis']
            score += delta_div['strength'] * self.order_flow_weights['delta_divergence']
            score += imbalance['strength'] * self.order_flow_weights['bid_ask_imbalance']
            score += aggressive_flow['strength'] * self.order_flow_weights['aggressive_flow']
            score += (footprint['score'] / 100) * 100 * self.order_flow_weights['volume_footprint']
            score += smart_money['strength'] * self.order_flow_weights['smart_money_detection']
            
            return min(max(score, 0), 100)
            
        except Exception as e:
            self.logger.error(f"Error calculating order flow score: {e}")
            return 0
    
    def _determine_direction(self, cvd_analysis: Dict, delta_div: Dict,
                            imbalance: Dict, aggressive_flow: Dict,
                            smart_money: Dict) -> Optional[str]:
        """Determine trade direction from order flow signals"""
        try:
            bullish_signals = 0
            bearish_signals = 0
            
            if cvd_analysis['trend'] == 'bullish':
                bullish_signals += 1
            elif cvd_analysis['trend'] == 'bearish':
                bearish_signals += 1
            
            if delta_div['divergence_detected']:
                if delta_div['divergence_type'] == 'bullish':
                    bullish_signals += 1
                elif delta_div['divergence_type'] == 'bearish':
                    bearish_signals += 1
            
            if imbalance['pressure'] == 'bullish':
                bullish_signals += 1
            elif imbalance['pressure'] == 'bearish':
                bearish_signals += 1
            
            if smart_money['flow'] == 'bullish':
                bullish_signals += 1
            elif smart_money['flow'] == 'bearish':
                bearish_signals += 1
            
            if bullish_signals >= 3 and bullish_signals > bearish_signals:
                return 'BUY'
            elif bearish_signals >= 3 and bearish_signals > bullish_signals:
                return 'SELL'
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error determining direction: {e}")
            return None
    
    async def _generate_signal(self, symbol: str, direction: str, primary_df: pd.DataFrame,
                              tf_data: Dict, cvd_analysis: Dict, delta_div: Dict,
                              imbalance: Dict, aggressive_flow: Dict, footprint: Dict,
                              smart_money: Dict, liquidity_zones: Dict, tick_momentum: Dict,
                              spread_analysis: Dict, depth_analysis: Dict,
                              order_flow_score: float) -> Optional[OrderFlowSignal]:
        """Generate complete order flow signal"""
        try:
            close = primary_df['close'].values
            current_price = float(close[-1])
            
            atr = self._calculate_atr(primary_df)
            
            if direction == 'BUY':
                stop_loss_pct = self.stop_loss_percentages[0] / 100
                stop_loss = current_price * (1 - stop_loss_pct)
                
                tp1 = current_price * (1 + stop_loss_pct * self.profit_target_ratios[0])
                tp2 = current_price * (1 + stop_loss_pct * self.profit_target_ratios[1])
                tp3 = current_price * (1 + stop_loss_pct * self.profit_target_ratios[2])
            else:
                stop_loss_pct = self.stop_loss_percentages[0] / 100
                stop_loss = current_price * (1 + stop_loss_pct)
                
                tp1 = current_price * (1 - stop_loss_pct * self.profit_target_ratios[0])
                tp2 = current_price * (1 - stop_loss_pct * self.profit_target_ratios[1])
                tp3 = current_price * (1 - stop_loss_pct * self.profit_target_ratios[2])
            
            leverage = self._calculate_dynamic_leverage(order_flow_score)
            
            confidence = min((order_flow_score / 100) * (imbalance['strength'] / 100) * 100, 95)
            
            execution_urgency = 'high' if aggressive_flow['dominant_flow'] == 'aggressive' else 'normal'
            expected_hold = 90 if aggressive_flow['dominant_flow'] == 'aggressive' else 150
            
            signal = OrderFlowSignal(
                symbol=symbol,
                direction=direction,
                entry_price=current_price,
                stop_loss=stop_loss,
                tp1=tp1,
                tp2=tp2,
                tp3=tp3,
                signal_strength=order_flow_score,
                leverage=leverage,
                cvd_trend=cvd_analysis['trend'],
                cvd_strength=cvd_analysis['strength'],
                delta_divergence=delta_div['divergence_detected'],
                bid_ask_imbalance=imbalance['ratio'],
                order_book_pressure=imbalance['pressure'],
                smart_money_flow=smart_money['flow'],
                aggressive_flow_ratio=aggressive_flow['ratio'],
                volume_footprint_score=footprint['score'],
                spread_quality=spread_analysis['quality'],
                market_depth_score=depth_analysis['score'],
                tick_momentum_score=tick_momentum['score'],
                liquidity_zone_near=liquidity_zones['near_zone'],
                execution_urgency=execution_urgency,
                expected_hold_seconds=expected_hold,
                confidence_level=confidence,
                timestamp=datetime.now()
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return None
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            if len(close) < period + 1:
                return (high[-1] - low[-1])
            
            tr_list = []
            for i in range(-period, 0):
                tr = max(
                    high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1])
                )
                tr_list.append(tr)
            
            return np.mean(tr_list)
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return 0
    
    def _calculate_dynamic_leverage(self, signal_strength: float) -> int:
        """Calculate dynamic leverage based on signal strength"""
        if signal_strength >= 90:
            return self.max_leverage
        elif signal_strength >= 80:
            return int(self.max_leverage * 0.8)
        elif signal_strength >= 75:
            return int(self.max_leverage * 0.6)
        else:
            return self.min_leverage
    
    def _can_trade_symbol(self, symbol: str) -> bool:
        """Check if symbol can be traded based on frequency limits"""
        current_time = time.time()
        current_hour = datetime.now().strftime('%Y-%m-%d-%H')
        
        if symbol in self.last_trade_times:
            time_since_last = current_time - self.last_trade_times[symbol]
            if time_since_last < self.min_trade_interval:
                return False
        
        hour_key = f"{symbol}_{current_hour}"
        if hour_key in self.hourly_trade_counts:
            if self.hourly_trade_counts[hour_key] >= self.max_trades_per_hour:
                return False
        
        return True
    
    def _record_trade_time(self, symbol: str):
        """Record trade time for frequency limiting"""
        current_time = time.time()
        current_hour = datetime.now().strftime('%Y-%m-%d-%H')
        
        self.last_trade_times[symbol] = current_time
        
        hour_key = f"{symbol}_{current_hour}"
        self.hourly_trade_counts[hour_key] = self.hourly_trade_counts.get(hour_key, 0) + 1
    
    def get_strategy_name(self) -> str:
        """Get strategy name"""
        return "Advanced Order Flow Scalping"
    
    def get_strategy_description(self) -> str:
        """Get strategy description"""
        return "Institutional-grade order flow analysis with real-time market microstructure data"
