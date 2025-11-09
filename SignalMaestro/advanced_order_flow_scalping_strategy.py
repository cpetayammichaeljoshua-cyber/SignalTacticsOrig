
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

try:
    from order_flow_error_handler import OrderFlowErrorHandler
    ERROR_HANDLER_AVAILABLE = True
except ImportError:
    ERROR_HANDLER_AVAILABLE = False


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
    liquidity_zone_proximity: bool = False  # Added for compatibility
    
    execution_urgency: str = "normal"
    expected_hold_seconds: int = 120
    signal_latency_ms: float = 0.0
    confidence_level: float = 0.0
    
    timestamp: Optional[datetime] = None


class AdvancedOrderFlowScalpingStrategy:
    """Ultimate order flow scalping strategy with institutional-grade analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize error handler with enhanced telegram integration
        if ERROR_HANDLER_AVAILABLE:
            self.error_handler = OrderFlowErrorHandler()
        else:
            self.error_handler = None
        
        self.timeframes = ['1m', '3m', '5m', '15m']
        self.max_leverage = 75  # Increased for high-confidence signals
        self.min_leverage = 20
        self.margin_type = "cross"
        self.risk_percentage = 0.8  # Conservative risk per trade
        
        # Enhanced rate limiting for production
        self.max_trades_per_hour = 8
        self.min_trade_interval = 45  # 45 seconds minimum between trades
        self.last_trade_times = {}
        self.hourly_trade_counts = {}
        
        # Optimized parameters for better signal quality
        self.cvd_lookback_periods = 25
        self.imbalance_threshold = 1.3
        self.delta_divergence_threshold = 0.6
        self.smart_money_threshold = 1.8
        
        # Telegram integration settings
        self.telegram_enabled = True
        self.signal_quality_threshold = 85  # Higher threshold for production
        
        self.order_flow_weights = {
            'cvd_analysis': 0.25,
            'delta_divergence': 0.20,
            'bid_ask_imbalance': 0.18,
            'aggressive_flow': 0.15,
            'volume_footprint': 0.12,
            'smart_money_detection': 0.10
        }
        
        self.min_signal_strength = 82  # Raised for production quality
        
        # Enhanced Telegram signal formatting
        self.telegram_formatting_enabled = True
        self.include_technical_details = True
        
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
            
            # Validate input data with enhanced error handling
            if not ohlcv_data or not isinstance(ohlcv_data, dict):
                self.logger.warning(f"Invalid OHLCV data for {symbol}")
                return None
            
            tf_data = {}
            for tf in self.timeframes:
                if tf in ohlcv_data and isinstance(ohlcv_data[tf], list) and len(ohlcv_data[tf]) >= 50:
                    try:
                        df = self._prepare_dataframe_enhanced(ohlcv_data[tf], symbol)
                        if df is not None and len(df) >= 30:
                            tf_data[tf] = df
                    except Exception as e:
                        self.logger.warning(f"Error preparing {tf} data for {symbol}: {e}")
                        continue
            
            if len(tf_data) < 1:
                self.logger.debug(f"Insufficient timeframe data for {symbol}")
                return None
            
            # Select primary timeframe
            primary_tf = '3m' if '3m' in tf_data else '1m' if '1m' in tf_data else '5m' if '5m' in tf_data else list(tf_data.keys())[0]
            primary_df = tf_data[primary_tf]
            
            trades_data = None
            if order_book_data and 'recent_trades' in order_book_data:
                trades_data = order_book_data['recent_trades']
                self.logger.debug(f"Using {len(trades_data)} trades for {symbol}")
            
            # Enhanced order flow analysis with error handling
            try:
                cvd_analysis = await self._analyze_cumulative_volume_delta_fixed(primary_df, symbol, trades_data)
                delta_div_analysis = await self._detect_delta_divergence(primary_df, cvd_analysis)
                imbalance_analysis = await self._analyze_bid_ask_imbalance(primary_df, order_book_data)
                aggressive_flow = await self._analyze_aggressive_vs_passive_flow(primary_df)
                volume_footprint = await self._analyze_volume_footprint(primary_df)
                smart_money = await self._detect_smart_money_flow(primary_df)
                
                liquidity_zones = await self._identify_liquidity_zones(primary_df)
                tick_momentum = await self._calculate_tick_momentum(primary_df)
                spread_analysis = await self._analyze_spread_quality(primary_df, order_book_data)
                depth_analysis = await self._analyze_market_depth(order_book_data)
            except Exception as e:
                self.logger.error(f"Error in order flow analysis for {symbol}: {e}")
                return None
            
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
            
            signal = await self._generate_signal_fixed(
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
            self.logger.error(f"Error analyzing {symbol}: {str(e)[:200]}")
            import traceback
            self.logger.debug(f"Full traceback for {symbol}: {traceback.format_exc()}")
            return None
    
    def _prepare_dataframe_enhanced(self, ohlcv: List, symbol: str) -> Optional[pd.DataFrame]:
        """Enhanced DataFrame preparation with comprehensive error handling"""
        try:
            if not ohlcv or not isinstance(ohlcv, list):
                return None
            
            # Use error handler if available
            if self.error_handler:
                df = self.error_handler.safe_dataframe_creation(symbol, ohlcv)
                if df is not None:
                    return df
            
            # Enhanced fallback method with column detection
            col_count = len(ohlcv[0]) if ohlcv else 0
            
            # Dynamically determine column structure based on actual data
            if col_count == 6:
                columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            elif col_count == 8:
                columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume']
            elif col_count == 12:
                columns = [
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote', 'ignore'
                ]
            elif col_count >= 11:
                # Handle variable column count from Binance API
                columns = [
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote'
                ]
                if col_count > 11:
                    columns.extend([f'extra_col_{i}' for i in range(col_count - 11)])
            else:
                self.logger.warning(f"Unexpected column count for {symbol}: {col_count}")
                return None
            
            try:
                df = pd.DataFrame(ohlcv, columns=columns[:col_count])
            except Exception as e:
                self.logger.error(f"DataFrame creation failed for {symbol}: {e}")
                return None
            
            # Convert numeric columns safely
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            if col_count >= 8:
                numeric_cols.extend(['quote_volume'])
            if col_count >= 12:
                numeric_cols.extend(['trades', 'taker_buy_volume', 'taker_buy_quote'])
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert timestamps
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            if 'close_time' in df.columns:
                df['close_time'] = pd.to_numeric(df['close_time'], errors='coerce')
            
            # Clean invalid data
            df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
            
            # Validate data
            if len(df) == 0:
                self.logger.warning(f"No valid data after cleaning for {symbol}")
                return None
            
            # Validate price relationships
            invalid_prices = (df['high'] < df['low']) | (df['close'] <= 0) | (df['open'] <= 0)
            if invalid_prices.any():
                self.logger.warning(f"Invalid price data detected for {symbol}, cleaning...")
                df = df[~invalid_prices]
            
            if len(df) == 0:
                return None
            
            return df
            
        except Exception as e:
            self.logger.error(f"Enhanced DataFrame preparation error for {symbol}: {e}")
            return None
    
    async def _analyze_cumulative_volume_delta_fixed(self, df: pd.DataFrame, symbol: str,
                                                   trades_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Fixed CVD calculation with enhanced error handling
        """
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            open_price = df['open'].values
            volume = df['volume'].values
            
            if len(close) < self.cvd_lookback_periods:
                return {'trend': 'neutral', 'strength': 0, 'delta_values': []}
            
            # Enhanced CVD calculation with multiple fallback methods
            delta_values = []
            
            # Method 1: Use Binance taker buy volume if available
            if 'taker_buy_volume' in df.columns:
                taker_buy_volume = df['taker_buy_volume'].values
                for i in range(len(volume)):
                    try:
                        buy_volume = float(taker_buy_volume[i]) if not pd.isna(taker_buy_volume[i]) else 0
                        sell_volume = float(volume[i]) - buy_volume
                        delta = buy_volume - sell_volume
                        delta_values.append(delta)
                    except (ValueError, TypeError):
                        delta_values.append(0)
                        
                self.logger.debug(f"Using Binance taker buy volume for CVD calculation: {symbol}")
                
            # Method 2: Use real trades data
            elif trades_data and len(trades_data) > 0:
                self.logger.debug(f"Using {len(trades_data)} real trades for CVD calculation: {symbol}")
                
                buy_volume_total = 0
                sell_volume_total = 0
                
                for trade in trades_data:
                    try:
                        side = trade.get('side')
                        amount = float(trade.get('amount', 0))
                        
                        if side == 'buy':
                            buy_volume_total += amount
                        elif side == 'sell':
                            sell_volume_total += amount
                    except (ValueError, TypeError):
                        continue
                
                real_delta = buy_volume_total - sell_volume_total
                
                # Distribute real delta across candles
                for i in range(len(close)):
                    if i == len(close) - 1:
                        delta_values.append(real_delta)
                    else:
                        # Estimate based on candle structure
                        delta = self._estimate_candle_delta(
                            open_price[i], high[i], low[i], close[i], volume[i]
                        )
                        delta_values.append(delta)
            else:
                # Method 3: Estimation method
                for i in range(len(close)):
                    delta = self._estimate_candle_delta(
                        open_price[i], high[i], low[i], close[i], volume[i]
                    )
                    delta_values.append(delta)
            
            # Calculate CVD with error handling
            try:
                cvd = np.cumsum(delta_values)
                recent_cvd = cvd[-self.cvd_lookback_periods:]
                
                if len(recent_cvd) > 1:
                    cvd_slope = np.polyfit(range(len(recent_cvd)), recent_cvd, 1)[0]
                    volume_mean = np.mean(volume[-self.cvd_lookback_periods:])
                    cvd_normalized = cvd_slope / (volume_mean + 1e-10) if volume_mean > 0 else 0
                else:
                    cvd_normalized = 0
            except Exception as e:
                self.logger.warning(f"CVD calculation error for {symbol}: {e}")
                cvd_normalized = 0
                cvd = np.zeros_like(delta_values)
            
            # Determine trend and strength
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
                'slope': cvd_normalized,
                'buy_sell_ratio': np.mean([d for d in delta_values[-5:] if d != 0]) if any(delta_values[-5:]) else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating CVD for {symbol}: {e}")
            return {'trend': 'neutral', 'strength': 0, 'delta_values': [], 'cvd': np.array([]), 'slope': 0, 'buy_sell_ratio': 0}
    
    def _estimate_candle_delta(self, open_price: float, high: float, low: float, close: float, volume: float) -> float:
        """Estimate buy/sell delta from candle structure"""
        try:
            candle_range = high - low
            if candle_range == 0 or volume == 0:
                return 0
            
            close_position = (close - low) / candle_range
            
            if close >= open_price:
                # Bullish candle
                buy_volume = volume * (0.5 + close_position * 0.5)
                sell_volume = volume - buy_volume
            else:
                # Bearish candle
                sell_volume = volume * (0.5 + (1 - close_position) * 0.5)
                buy_volume = volume - sell_volume
            
            return buy_volume - sell_volume
            
        except Exception:
            return 0
    
    async def _detect_delta_divergence(self, df: pd.DataFrame, cvd_analysis: Dict) -> Dict[str, Any]:
        """Detect price/delta divergence patterns with enhanced error handling"""
        try:
            close = df['close'].values
            cvd = cvd_analysis.get('cvd', np.array([]))
            
            if len(close) < 10 or len(cvd) < 10:
                return {'divergence_detected': False, 'divergence_type': 'none', 'strength': 0}
            
            close_recent = close[-10:]
            cvd_recent = cvd[-10:] if len(cvd) >= 10 else cvd
            
            try:
                price_slope = np.polyfit(range(len(close_recent)), close_recent, 1)[0] if len(close_recent) > 1 else 0
                cvd_slope = np.polyfit(range(len(cvd_recent)), cvd_recent, 1)[0] if len(cvd_recent) > 1 else 0
            except Exception:
                return {'divergence_detected': False, 'divergence_type': 'none', 'strength': 0}
            
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
                return {'pressure': 'balanced', 'ratio': 1.0, 'strength': 0, 'bid_volume': 0, 'ask_volume': 0}
            
            if order_book_data and 'bids' in order_book_data and 'asks' in order_book_data:
                bids = order_book_data.get('bids', [])
                asks = order_book_data.get('asks', [])
                
                if bids and asks:
                    self.logger.debug(f"Using real order book: {len(bids)} bids, {len(asks)} asks")
                    
                    try:
                        bid_volume = sum([float(bid[1]) for bid in bids[:10] if len(bid) >= 2])
                        ask_volume = sum([float(ask[1]) for ask in asks[:10] if len(ask) >= 2])
                        
                        total_volume = bid_volume + ask_volume
                        if total_volume > 0:
                            imbalance_ratio = bid_volume / ask_volume if ask_volume > 0 else 2.0
                        else:
                            imbalance_ratio = 1.0
                    except (ValueError, TypeError, IndexError):
                        bid_volume, ask_volume, imbalance_ratio = self._estimate_imbalance_from_candles(
                            close, high, low, volume
                        )
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
            return {'pressure': 'balanced', 'ratio': 1.0, 'strength': 0, 'bid_volume': 0, 'ask_volume': 0}
    
    def _estimate_imbalance_from_candles(self, close: np.array, high: np.array, 
                                         low: np.array, volume: np.array) -> Tuple[float, float, float]:
        """Estimate bid/ask imbalance from candle structure with error handling"""
        bid_volume = 0
        ask_volume = 0
        
        try:
            for i in range(-min(5, len(close)), 0):
                if i >= -len(close):
                    candle_range = high[i] - low[i]
                    if candle_range == 0:
                        continue
                    
                    close_position = (close[i] - low[i]) / candle_range
                    vol = volume[i] if not pd.isna(volume[i]) else 0
                    
                    if close_position > 0.6:
                        bid_volume += vol * close_position
                        ask_volume += vol * (1 - close_position)
                    elif close_position < 0.4:
                        ask_volume += vol * (1 - close_position)
                        bid_volume += vol * close_position
                    else:
                        bid_volume += vol * 0.5
                        ask_volume += vol * 0.5
        except Exception:
            # Fallback values
            bid_volume = 1
            ask_volume = 1
        
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
            
            for i in range(-min(10, len(volume)), 0):
                if i >= -len(volume):
                    try:
                        candle_volatility = (high[i] - low[i]) / close[i] if close[i] > 0 else 0
                        vol = volume[i] if not pd.isna(volume[i]) else 0
                        
                        if candle_volatility > 0.003:
                            aggressive_volume += vol * 0.7
                            passive_volume += vol * 0.3
                        else:
                            passive_volume += vol * 0.7
                            aggressive_volume += vol * 0.3
                    except (ValueError, TypeError, ZeroDivisionError):
                        continue
            
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
            for i in range(-min(10, len(volume)), 0):
                if i >= -len(volume) and volume[i] > volume_ma * self.smart_money_threshold:
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
            for i in range(-min(50, len(close)), 0):
                if i >= -len(close):
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
                
                if bids and asks and len(bids[0]) >= 2 and len(asks[0]) >= 2:
                    try:
                        spread = float(asks[0][0]) - float(bids[0][0])
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
                    except (ValueError, TypeError, IndexError):
                        pass
            
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
            
            try:
                total_bid_volume = sum([float(b[1]) for b in bids[:20] if len(b) >= 2])
                total_ask_volume = sum([float(a[1]) for a in asks[:20] if len(a) >= 2])
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
            except (ValueError, TypeError, IndexError):
                return {'score': 50, 'quality': 'normal'}
            
        except Exception as e:
            self.logger.error(f"Error analyzing market depth: {e}")
            return {'score': 50, 'quality': 'normal'}
    
    def _calculate_order_flow_score(self, cvd_analysis: Dict, delta_div: Dict,
                                    imbalance: Dict, aggressive_flow: Dict,
                                    footprint: Dict, smart_money: Dict) -> float:
        """Calculate weighted order flow score"""
        try:
            score = 0
            
            score += cvd_analysis.get('strength', 0) * self.order_flow_weights['cvd_analysis']
            score += delta_div.get('strength', 0) * self.order_flow_weights['delta_divergence']
            score += imbalance.get('strength', 0) * self.order_flow_weights['bid_ask_imbalance']
            score += aggressive_flow.get('strength', 0) * self.order_flow_weights['aggressive_flow']
            score += (footprint.get('score', 50) / 100) * 100 * self.order_flow_weights['volume_footprint']
            score += smart_money.get('strength', 0) * self.order_flow_weights['smart_money_detection']
            
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
            
            if cvd_analysis.get('trend') == 'bullish':
                bullish_signals += 1
            elif cvd_analysis.get('trend') == 'bearish':
                bearish_signals += 1
            
            if delta_div.get('divergence_detected'):
                if delta_div.get('divergence_type') == 'bullish':
                    bullish_signals += 1
                elif delta_div.get('divergence_type') == 'bearish':
                    bearish_signals += 1
            
            if imbalance.get('pressure') == 'bullish':
                bullish_signals += 1
            elif imbalance.get('pressure') == 'bearish':
                bearish_signals += 1
            
            if smart_money.get('flow') == 'bullish':
                bullish_signals += 1
            elif smart_money.get('flow') == 'bearish':
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
    
    async def _generate_signal_fixed(self, symbol: str, direction: str, primary_df: pd.DataFrame,
                                   tf_data: Dict, cvd_analysis: Dict, delta_div: Dict,
                                   imbalance: Dict, aggressive_flow: Dict, footprint: Dict,
                                   smart_money: Dict, liquidity_zones: Dict, tick_momentum: Dict,
                                   spread_analysis: Dict, depth_analysis: Dict,
                                   order_flow_score: float) -> Optional[OrderFlowSignal]:
        """Generate complete order flow signal with enhanced error handling"""
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
            
            confidence = min((order_flow_score / 100) * (imbalance.get('strength', 0) / 100) * 100, 95)
            
            execution_urgency = 'high' if aggressive_flow.get('dominant_flow') == 'aggressive' else 'normal'
            expected_hold = 90 if aggressive_flow.get('dominant_flow') == 'aggressive' else 150
            
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
                cvd_trend=cvd_analysis.get('trend', 'neutral'),
                cvd_strength=cvd_analysis.get('strength', 0),
                delta_divergence=delta_div.get('divergence_detected', False),
                bid_ask_imbalance=imbalance.get('ratio', 1.0),
                order_book_pressure=imbalance.get('pressure', 'balanced'),
                smart_money_flow=smart_money.get('flow', 'neutral'),
                aggressive_flow_ratio=aggressive_flow.get('ratio', 1.0),
                volume_footprint_score=footprint.get('score', 50),
                spread_quality=spread_analysis.get('quality', 'normal'),
                market_depth_score=depth_analysis.get('score', 50),
                tick_momentum_score=tick_momentum.get('score', 50),
                liquidity_zone_near=liquidity_zones.get('near_zone', False),
                liquidity_zone_proximity=liquidity_zones.get('near_zone', False),
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
        """Calculate Average True Range with error handling"""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            if len(close) < period + 1:
                return (high[-1] - low[-1]) if len(high) > 0 else 0
            
            tr_list = []
            for i in range(-period, 0):
                if i >= -len(close) and i-1 >= -len(close):
                    tr = max(
                        high[i] - low[i],
                        abs(high[i] - close[i-1]),
                        abs(low[i] - close[i-1])
                    )
                    tr_list.append(tr)
            
            return np.mean(tr_list) if tr_list else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return 0
    
    def _calculate_dynamic_leverage(self, signal_strength: float) -> int:
        """Calculate dynamic leverage based on signal strength"""
        try:
            if signal_strength >= 90:
                return self.max_leverage
            elif signal_strength >= 80:
                return int(self.max_leverage * 0.8)
            elif signal_strength >= 75:
                return int(self.max_leverage * 0.6)
            else:
                return self.min_leverage
        except Exception:
            return self.min_leverage
    
    def _can_trade_symbol(self, symbol: str) -> bool:
        """Check if symbol can be traded based on frequency limits"""
        try:
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
        except Exception:
            return True
    
    def _record_trade_time(self, symbol: str):
        """Record trade time for frequency limiting"""
        try:
            current_time = time.time()
            current_hour = datetime.now().strftime('%Y-%m-%d-%H')
            
            self.last_trade_times[symbol] = current_time
            
            hour_key = f"{symbol}_{current_hour}"
            self.hourly_trade_counts[hour_key] = self.hourly_trade_counts.get(hour_key, 0) + 1
        except Exception as e:
            self.logger.error(f"Error recording trade time: {e}")
    
    def get_strategy_name(self) -> str:
        """Get strategy name"""
        return "Advanced Order Flow Scalping"
    
    def get_strategy_description(self) -> str:
        """Get strategy description"""
        return "Institutional-grade order flow analysis with real-time market microstructure data"
    
    def format_telegram_signal(self, signal: OrderFlowSignal) -> str:
        """Format signal for enhanced Telegram display"""
        try:
            symbol = signal.symbol
            direction = signal.direction
            entry = signal.entry_price
            sl = signal.stop_loss
            tp1 = signal.tp1
            tp2 = signal.tp2
            tp3 = signal.tp3
            strength = signal.signal_strength
            leverage = signal.leverage
            
            # Dynamic precision based on price
            if entry > 100:
                precision = 2
            elif entry > 1:
                precision = 4
            else:
                precision = 6
            
            # Enhanced emoji selection
            direction_emoji = "🚀" if direction == 'BUY' else "🔻"
            strength_emoji = "🔥" if strength >= 90 else "⚡" if strength >= 85 else "💎"
            urgency_emoji = "🚨" if signal.execution_urgency == 'high' else "📊"
            
            # Risk/Reward calculation
            if direction == 'BUY':
                risk_pct = abs(entry - sl) / entry * 100
                reward_pct = abs(tp1 - entry) / entry * 100
            else:
                risk_pct = abs(sl - entry) / entry * 100
                reward_pct = abs(entry - tp1) / entry * 100
            
            rr_ratio = reward_pct / risk_pct if risk_pct > 0 else 2.0
            
            # Order flow details
            order_flow_details = ""
            if hasattr(signal, 'cvd_trend') and signal.cvd_trend != 'neutral':
                cvd_emoji = "📈" if signal.cvd_trend == 'bullish' else "📉"
                order_flow_details += f"\n{cvd_emoji} CVD: {signal.cvd_trend.upper()}"
            
            if hasattr(signal, 'smart_money_flow') and signal.smart_money_flow != 'neutral':
                smart_emoji = "🐋" if signal.smart_money_flow in ['bullish', 'bearish'] else "🐟"
                order_flow_details += f"\n{smart_emoji} Smart Money: {signal.smart_money_flow.upper()}"
            
            if hasattr(signal, 'delta_divergence') and signal.delta_divergence:
                order_flow_details += f"\n⚠️ Delta Divergence Detected"
            
            # Format message
            message = f"""{urgency_emoji} {direction_emoji} <b>{symbol} - {direction}</b> {strength_emoji}

🎯 <b>ADVANCED ORDER FLOW SIGNAL</b>
⚡ Signal Strength: <b>{strength:.1f}%</b>
🔮 Leverage: <b>{leverage}x</b>
📊 Risk/Reward: <b>1:{rr_ratio:.1f}</b>

💰 <b>Entry Zone:</b> {entry:.{precision}f}
🛡️ <b>Stop Loss:</b> {sl:.{precision}f} (-{risk_pct:.1f}%)

🎯 <b>Take Profits:</b>
• TP1: {tp1:.{precision}f} (+{reward_pct:.1f}%)
• TP2: {tp2:.{precision}f}
• TP3: {tp3:.{precision}f}

📈 <b>Order Flow Analysis:</b>{order_flow_details}
🔍 Imbalance: {signal.bid_ask_imbalance:.2f}x
📊 Book Pressure: {signal.order_book_pressure.upper()}

⏰ <b>Time:</b> {datetime.now().strftime('%H:%M UTC')}
🏃‍♂️ <b>Expected Hold:</b> {signal.expected_hold_seconds//60} mins
🎯 <b>Confidence:</b> {signal.confidence_level:.0f}%

<b>#{symbol.replace('USDT', '')} #{direction} #OrderFlow #Scalping</b>

<i>⚠️ Risk Management: Use proper position sizing</i>"""

            return message.strip()
            
        except Exception as e:
            self.logger.error(f"Error formatting Telegram signal: {e}")
            return f"🚨 ORDER FLOW SIGNAL: {signal.symbol} {signal.direction} @ {signal.entry_price:.4f}"
