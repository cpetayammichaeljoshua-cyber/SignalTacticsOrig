
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

try:
    from advanced_order_flow_scalping_strategy import AdvancedOrderFlowScalpingStrategy, OrderFlowSignal
    ORDER_FLOW_AVAILABLE = True
except ImportError:
    ORDER_FLOW_AVAILABLE = False

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
        Enhanced order flow analysis with comprehensive market data
        """
        try:
            if not self.order_flow_strategy:
                return await self._fallback_order_flow_analysis(symbol, ohlcv_data)
            
            # Get real-time order book data
            order_book_data = await self._fetch_order_book_data(symbol)
            
            # Analyze with order flow strategy
            order_flow_signal = await self.order_flow_strategy.analyze_symbol(
                symbol, ohlcv_data, order_book_data
            )
            
            if not order_flow_signal:
                return None
            
            # Convert to enhanced signal format
            enhanced_signal_data = self._convert_to_enhanced_format(order_flow_signal)
            
            self.logger.info(
                f"📊 Order Flow Analysis Complete: {symbol} "
                f"Signal: {order_flow_signal.direction} "
                f"Strength: {order_flow_signal.signal_strength:.1f}% "
                f"CVD: {order_flow_signal.cvd_trend} "
                f"Smart Money: {order_flow_signal.smart_money_flow}"
            )
            
            return enhanced_signal_data
            
        except Exception as e:
            self.logger.error(f"Error in order flow analysis for {symbol}: {e}")
            return await self._fallback_order_flow_analysis(symbol, ohlcv_data)
    
    async def _fetch_order_book_data(self, symbol: str) -> Optional[Dict]:
        """Fetch real-time order book data from Binance"""
        try:
            url = f"https://fapi.binance.com/fapi/v1/depth"
            params = {
                'symbol': symbol,
                'limit': 20
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        order_book = await response.json()
                        
                        # Fetch recent trades for delta calculation
                        trades_url = f"https://fapi.binance.com/fapi/v1/aggTrades"
                        trades_params = {'symbol': symbol, 'limit': 100}
                        
                        async with session.get(trades_url, params=trades_params) as trades_response:
                            recent_trades = []
                            if trades_response.status == 200:
                                trades_data = await trades_response.json()
                                recent_trades = [{
                                    'side': 'sell' if trade['m'] else 'buy',
                                    'amount': float(trade['q']),
                                    'price': float(trade['p']),
                                    'timestamp': trade['T']
                                } for trade in trades_data]
                        
                        return {
                            'bids': [[float(bid[0]), float(bid[1])] for bid in order_book.get('bids', [])],
                            'asks': [[float(ask[0]), float(ask[1])] for ask in order_book.get('asks', [])],
                            'recent_trades': recent_trades
                        }
            return None
            
        except Exception as e:
            self.logger.debug(f"Error fetching order book for {symbol}: {e}")
            return None
    
    def _convert_to_enhanced_format(self, signal: OrderFlowSignal) -> Dict[str, Any]:
        """Convert OrderFlowSignal to enhanced format for the trading bot"""
        return {
            'symbol': signal.symbol,
            'direction': signal.direction,
            'entry_price': signal.entry_price,
            'stop_loss': signal.stop_loss,
            'tp1': signal.tp1,
            'tp2': signal.tp2,
            'tp3': signal.tp3,
            'signal_strength': signal.signal_strength,
            'leverage': signal.leverage,
            'cvd_trend': signal.cvd_trend,
            'cvd_strength': signal.cvd_strength,
            'delta_divergence': signal.delta_divergence,
            'bid_ask_imbalance': signal.bid_ask_imbalance,
            'order_book_pressure': signal.order_book_pressure,
            'aggressive_flow_ratio': signal.aggressive_flow_ratio,
            'smart_money_flow': signal.smart_money_flow,
            'liquidity_zone_proximity': signal.liquidity_zone_proximity,
            'volume_footprint_score': signal.volume_footprint_score,
            'spread_quality': signal.spread_quality,
            'market_depth_score': signal.market_depth_score,
            'tick_momentum_score': signal.tick_momentum_score,
            'execution_urgency': signal.execution_urgency,
            'expected_hold_seconds': signal.expected_hold_seconds,
            'confidence_level': signal.confidence_level,
            'timestamp': signal.timestamp or datetime.now(),
            'order_flow_enhanced': True,
            'order_flow_score': signal.signal_strength,
            'smart_money_detected': signal.smart_money_flow in ['bullish', 'bearish']
        }
    
    async def _fallback_order_flow_analysis(self, symbol: str, ohlcv_data: Dict[str, List]) -> Optional[Dict[str, Any]]:
        """Fallback order flow analysis when main strategy is unavailable"""
        try:
            # Use 1m or 3m data for fallback analysis
            primary_tf = '3m' if '3m' in ohlcv_data else '1m'
            if primary_tf not in ohlcv_data:
                return None
            
            df = pd.DataFrame(ohlcv_data[primary_tf], columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            if len(df) < 20:
                return None
            
            # Simple order flow estimation
            current_price = df['close'].iloc[-1]
            
            # Volume analysis
            volume_ma = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1.0
            
            # Price momentum
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
            
            # Simple CVD estimation
            cvd_estimate = 0
            for i in range(-10, 0):
                if i >= -len(df):
                    candle = df.iloc[i]
                    if candle['close'] > candle['open']:
                        cvd_estimate += candle['volume']
                    else:
                        cvd_estimate -= candle['volume']
            
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
                    'cvd_strength': min(100, abs(cvd_estimate) / volume_ma * 10) if volume_ma > 0 else 0,
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
            self.logger.error(f"Error in fallback order flow analysis: {e}")
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
