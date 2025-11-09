
#!/usr/bin/env python3
"""
Enhanced Order Flow Integration Module
Provides seamless integration between the Ultimate Trading Bot and Advanced Order Flow Strategy
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List
import aiohttp
import traceback

try:
    from advanced_order_flow_scalping_strategy import AdvancedOrderFlowScalpingStrategy, OrderFlowSignal
    ORDER_FLOW_AVAILABLE = True
except ImportError:
    ORDER_FLOW_AVAILABLE = False
    # Create placeholder classes
    class OrderFlowSignal:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class AdvancedOrderFlowScalpingStrategy:
        def __init__(self):
            pass
        
        async def analyze_symbol(self, symbol, ohlcv_data, order_book_data=None):
            return None

class EnhancedOrderFlowIntegration:
    """Enhanced integration layer for order flow analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.order_flow_strategy = None
        
        if ORDER_FLOW_AVAILABLE:
            try:
                self.order_flow_strategy = AdvancedOrderFlowScalpingStrategy()
                self.logger.info("✅ Advanced Order Flow Strategy initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize order flow strategy: {e}")
                self.order_flow_strategy = None
        else:
            self.logger.warning("⚠️ Order Flow Strategy not available")
    
    async def analyze_with_order_flow(self, symbol: str, ohlcv_data: Dict[str, List]) -> Optional[Dict[str, Any]]:
        """
        Enhanced order flow analysis with comprehensive market data and production optimizations
        """
        try:
            if not self.order_flow_strategy:
                return await self._fallback_order_flow_analysis(symbol, ohlcv_data)
            
            # Get real-time order book data with timeout
            order_book_data = await asyncio.wait_for(
                self._fetch_order_book_data(symbol), 
                timeout=10.0
            )
            
            # Analyze with order flow strategy
            order_flow_signal = await asyncio.wait_for(
                self.order_flow_strategy.analyze_symbol(symbol, ohlcv_data, order_book_data),
                timeout=15.0
            )
            
            if not order_flow_signal:
                # Try fallback if main analysis returns None
                return await self._fallback_order_flow_analysis(symbol, ohlcv_data)
            
            # Additional production-level validation
            if order_flow_signal.signal_strength < 82:
                self.logger.debug(f"Signal strength too low for production: {order_flow_signal.signal_strength:.1f}%")
                return None
            
            # Convert to enhanced signal format with additional metadata
            enhanced_signal_data = self._convert_to_enhanced_format(order_flow_signal)
            
            # Add production metadata
            enhanced_signal_data['production_validated'] = True
            enhanced_signal_data['analysis_timestamp'] = datetime.now()
            enhanced_signal_data['order_book_available'] = order_book_data is not None
            enhanced_signal_data['signal_version'] = '3.0'
            
            self.logger.info(
                f"🎯 HIGH-QUALITY ORDER FLOW SIGNAL: {symbol} "
                f"{order_flow_signal.direction} @ {order_flow_signal.signal_strength:.1f}% | "
                f"CVD: {order_flow_signal.cvd_trend} | "
                f"Smart: {order_flow_signal.smart_money_flow} | "
                f"Confidence: {getattr(order_flow_signal, 'confidence_level', 0):.0f}%"
            )
            
            return enhanced_signal_data
            
        except asyncio.TimeoutError:
            self.logger.warning(f"⏱️ Order flow analysis timeout for {symbol}, using fallback")
            return await self._fallback_order_flow_analysis(symbol, ohlcv_data)
        except Exception as e:
            self.logger.error(f"Error in order flow analysis for {symbol}: {e}")
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            return await self._fallback_order_flow_analysis(symbol, ohlcv_data)
    
    async def _fetch_order_book_data(self, symbol: str) -> Optional[Dict]:
        """Fetch real-time order book data from Binance with enhanced error handling"""
        try:
            url = f"https://fapi.binance.com/fapi/v1/depth"
            params = {
                'symbol': symbol,
                'limit': 20
            }
            
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                try:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            order_book = await response.json()
                            
                            # Fetch recent trades for delta calculation
                            trades_url = f"https://fapi.binance.com/fapi/v1/aggTrades"
                            trades_params = {'symbol': symbol, 'limit': 100}
                            
                            recent_trades = []
                            try:
                                async with session.get(trades_url, params=trades_params) as trades_response:
                                    if trades_response.status == 200:
                                        trades_data = await trades_response.json()
                                        recent_trades = [{
                                            'side': 'sell' if trade.get('m', False) else 'buy',
                                            'amount': float(trade.get('q', 0)),
                                            'price': float(trade.get('p', 0)),
                                            'timestamp': trade.get('T', 0)
                                        } for trade in trades_data if isinstance(trade, dict)]
                            except Exception as e:
                                self.logger.debug(f"Error fetching trades for {symbol}: {e}")
                            
                            # Validate and clean order book data
                            try:
                                cleaned_bids = []
                                for bid in order_book.get('bids', []):
                                    if isinstance(bid, list) and len(bid) >= 2:
                                        price = float(bid[0])
                                        volume = float(bid[1])
                                        if price > 0 and volume > 0:
                                            cleaned_bids.append([price, volume])
                                
                                cleaned_asks = []
                                for ask in order_book.get('asks', []):
                                    if isinstance(ask, list) and len(ask) >= 2:
                                        price = float(ask[0])
                                        volume = float(ask[1])
                                        if price > 0 and volume > 0:
                                            cleaned_asks.append([price, volume])
                                
                                return {
                                    'bids': cleaned_bids,
                                    'asks': cleaned_asks,
                                    'recent_trades': recent_trades
                                }
                            except Exception as e:
                                self.logger.debug(f"Error cleaning order book data for {symbol}: {e}")
                                return None
                        else:
                            self.logger.debug(f"Order book fetch failed for {symbol}: {response.status}")
                except asyncio.TimeoutError:
                    self.logger.debug(f"Timeout fetching order book for {symbol}")
                except Exception as e:
                    self.logger.debug(f"Network error fetching order book for {symbol}: {e}")
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Error in order book fetch for {symbol}: {e}")
            return None
    
    def _convert_to_enhanced_format(self, signal: OrderFlowSignal) -> Dict[str, Any]:
        """Convert OrderFlowSignal to enhanced format for the trading bot"""
        try:
            return {
                'symbol': getattr(signal, 'symbol', ''),
                'direction': getattr(signal, 'direction', ''),
                'entry_price': getattr(signal, 'entry_price', 0),
                'stop_loss': getattr(signal, 'stop_loss', 0),
                'tp1': getattr(signal, 'tp1', 0),
                'tp2': getattr(signal, 'tp2', 0),
                'tp3': getattr(signal, 'tp3', 0),
                'signal_strength': getattr(signal, 'signal_strength', 0),
                'leverage': getattr(signal, 'leverage', 25),
                'cvd_trend': getattr(signal, 'cvd_trend', 'neutral'),
                'cvd_strength': getattr(signal, 'cvd_strength', 0),
                'delta_divergence': getattr(signal, 'delta_divergence', False),
                'bid_ask_imbalance': getattr(signal, 'bid_ask_imbalance', 1.0),
                'order_book_pressure': getattr(signal, 'order_book_pressure', 'balanced'),
                'aggressive_flow_ratio': getattr(signal, 'aggressive_flow_ratio', 1.0),
                'smart_money_flow': getattr(signal, 'smart_money_flow', 'neutral'),
                'liquidity_zone_proximity': getattr(signal, 'liquidity_zone_proximity', 
                                                  getattr(signal, 'liquidity_zone_near', False)),
                'volume_footprint_score': getattr(signal, 'volume_footprint_score', 50),
                'spread_quality': getattr(signal, 'spread_quality', 'normal'),
                'market_depth_score': getattr(signal, 'market_depth_score', 50),
                'tick_momentum_score': getattr(signal, 'tick_momentum_score', 50),
                'execution_urgency': getattr(signal, 'execution_urgency', 'normal'),
                'expected_hold_seconds': getattr(signal, 'expected_hold_seconds', 120),
                'confidence_level': getattr(signal, 'confidence_level', 0),
                'timestamp': getattr(signal, 'timestamp', datetime.now()),
                'order_flow_enhanced': True,
                'order_flow_score': getattr(signal, 'signal_strength', 0),
                'smart_money_detected': getattr(signal, 'smart_money_flow', 'neutral') in ['bullish', 'bearish']
            }
        except Exception as e:
            self.logger.error(f"Error converting signal to enhanced format: {e}")
            return None
    
    async def _fallback_order_flow_analysis(self, symbol: str, ohlcv_data: Dict[str, List]) -> Optional[Dict[str, Any]]:
        """Fallback order flow analysis when main strategy is unavailable"""
        try:
            # Use 1m or 3m data for fallback analysis
            primary_tf = '3m' if '3m' in ohlcv_data else '1m' if '1m' in ohlcv_data else None
            if primary_tf not in ohlcv_data or not ohlcv_data[primary_tf]:
                return None
            
            # Safely create DataFrame
            try:
                raw_data = ohlcv_data[primary_tf]
                if not raw_data or len(raw_data) < 20:
                    return None
                
                # Determine column count
                col_count = len(raw_data[0]) if raw_data[0] else 0
                
                if col_count == 6:
                    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                elif col_count >= 12:
                    columns = [
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote', 'ignore'
                    ]
                else:
                    self.logger.debug(f"Unexpected column count for {symbol}: {col_count}")
                    return None
                
                df = pd.DataFrame(raw_data, columns=columns[:col_count])
                
                # Convert numeric columns
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Clean data
                df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
                
                if len(df) < 20:
                    return None
                    
            except Exception as e:
                self.logger.debug(f"Error creating DataFrame for {symbol}: {e}")
                return None
            
            # Simple order flow estimation
            current_price = float(df['close'].iloc[-1])
            
            # Volume analysis
            try:
                volume_ma = df['volume'].rolling(20).mean().iloc[-1]
                current_volume = df['volume'].iloc[-1]
                volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1.0
            except Exception:
                volume_ratio = 1.0
            
            # Price momentum
            try:
                price_change = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
            except Exception:
                price_change = 0
            
            # Simple CVD estimation
            cvd_estimate = 0
            try:
                for i in range(-10, 0):
                    if abs(i) <= len(df):
                        candle = df.iloc[i]
                        if candle['close'] > candle['open']:
                            cvd_estimate += candle['volume']
                        else:
                            cvd_estimate -= candle['volume']
            except Exception:
                cvd_estimate = 0
            
            cvd_trend = 'bullish' if cvd_estimate > 0 else 'bearish' if cvd_estimate < 0 else 'neutral'
            
            # Generate fallback signal
            if volume_ratio > 1.3 and abs(price_change) > 0.003:
                direction = 'BUY' if price_change > 0 else 'SELL'
                signal_strength = min(85, 60 + (volume_ratio - 1) * 20 + abs(price_change) * 1000)
                
                risk_pct = 0.8
                if direction == 'BUY':
                    stop_loss = current_price * (1 - risk_pct / 100)
                    tp1 = current_price * (1 + risk_pct * 0.8 / 100)
                    tp2 = current_price * (1 + risk_pct * 1.5 / 100)
                    tp3 = current_price * (1 + risk_pct * 2.2 / 100)
                else:
                    stop_loss = current_price * (1 + risk_pct / 100)
                    tp1 = current_price * (1 - risk_pct * 0.8 / 100)
                    tp2 = current_price * (1 - risk_pct * 1.5 / 100)
                    tp3 = current_price * (1 - risk_pct * 2.2 / 100)
                
                volume_ma_safe = volume_ma if volume_ma > 0 else 1
                
                return {
                    'symbol': symbol,
                    'direction': direction,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'tp1': tp1,
                    'tp2': tp2,
                    'tp3': tp3,
                    'signal_strength': signal_strength,
                    'leverage': 25,
                    'cvd_trend': cvd_trend,
                    'cvd_strength': min(100, abs(cvd_estimate) / volume_ma_safe * 10),
                    'bid_ask_imbalance': volume_ratio,
                    'order_flow_enhanced': False,
                    'order_flow_score': signal_strength * 0.7,
                    'smart_money_detected': volume_ratio > 2.0,
                    'execution_urgency': 'high' if volume_ratio > 2.0 else 'normal',
                    'confidence_level': signal_strength,
                    'timestamp': datetime.now()
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in fallback order flow analysis for {symbol}: {e}")
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def is_available(self) -> bool:
        """Check if order flow strategy is available"""
        return self.order_flow_strategy is not None
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information"""
        if self.order_flow_strategy:
            return {
                'name': self.order_flow_strategy.get_strategy_name(),
                'description': self.order_flow_strategy.get_strategy_description(),
                'available': True,
                'timeframes': getattr(self.order_flow_strategy, 'timeframes', ['1m', '3m', '5m', '15m']),
                'min_signal_strength': getattr(self.order_flow_strategy, 'min_signal_strength', 72)
            }
        else:
            return {
                'name': 'Fallback Order Flow Analysis',
                'description': 'Basic order flow estimation when advanced strategy unavailable',
                'available': False,
                'timeframes': ['1m', '3m', '5m'],
                'min_signal_strength': 75
            }
