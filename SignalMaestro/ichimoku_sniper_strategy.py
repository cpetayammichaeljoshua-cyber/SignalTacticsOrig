
#!/usr/bin/env python3
"""
Ichimoku Sniper Strategy for FXSUSDT.P
Specialized strategy for forex futures with 30-minute timeframe
Uses Ichimoku Cloud analysis with dynamic ATR-based stop loss and take profit
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

@dataclass
class IchimokuSignal:
    """Ichimoku trading signal data structure"""
    symbol: str
    action: str  # BUY or SELL
    entry_price: float
    stop_loss: float
    take_profit: float
    signal_strength: float
    confidence: float
    timeframe: str
    timestamp: datetime
    ichimoku_data: Dict[str, Any]
    atr_value: float
    risk_reward_ratio: float = 2.0

class IchimokuSniperStrategy:
    """
    Ichimoku Sniper Strategy Implementation
    - Uses 30-minute timeframe exclusively
    - Focuses on FXSUSDT.P trading
    - Implements dynamic ATR-based SL/TP with 1:2 RR ratio
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.symbol = "FXSUSDT.P"
        self.timeframe = "30m"
        
        # Ichimoku parameters
        self.tenkan_period = 9
        self.kijun_period = 26
        self.senkou_span_b_period = 52
        self.displacement = 26
        
        # ATR parameters for dynamic SL/TP
        self.atr_period = 14
        self.atr_multiplier_sl = 1.5  # Stop loss ATR multiplier
        self.atr_multiplier_tp = 3.0  # Take profit ATR multiplier (1:2 RR)
        
        # Signal strength thresholds
        self.min_signal_strength = 70.0
        self.min_confidence = 65.0
        
        self.logger.info("🎯 Ichimoku Sniper Strategy initialized for FXSUSDT.P")
    
    async def calculate_ichimoku(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Ichimoku Cloud components"""
        try:
            if len(df) < self.senkou_span_b_period + self.displacement:
                return {}
            
            # Tenkan-sen (Conversion Line)
            tenkan_high = df['high'].rolling(window=self.tenkan_period).max()
            tenkan_low = df['low'].rolling(window=self.tenkan_period).min()
            tenkan_sen = (tenkan_high + tenkan_low) / 2
            
            # Kijun-sen (Base Line)
            kijun_high = df['high'].rolling(window=self.kijun_period).max()
            kijun_low = df['low'].rolling(window=self.kijun_period).min()
            kijun_sen = (kijun_high + kijun_low) / 2
            
            # Senkou Span A (Leading Span A)
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(self.displacement)
            
            # Senkou Span B (Leading Span B)
            senkou_high = df['high'].rolling(window=self.senkou_span_b_period).max()
            senkou_low = df['low'].rolling(window=self.senkou_span_b_period).min()
            senkou_span_b = ((senkou_high + senkou_low) / 2).shift(self.displacement)
            
            # Chikou Span (Lagging Span)
            chikou_span = df['close'].shift(-self.displacement)
            
            return {
                'tenkan_sen': tenkan_sen.iloc[-1] if not tenkan_sen.empty else 0,
                'kijun_sen': kijun_sen.iloc[-1] if not kijun_sen.empty else 0,
                'senkou_span_a': senkou_span_a.iloc[-1] if not senkou_span_a.empty else 0,
                'senkou_span_b': senkou_span_b.iloc[-1] if not senkou_span_b.empty else 0,
                'chikou_span': chikou_span.iloc[-1] if not chikou_span.empty else 0,
                'current_price': df['close'].iloc[-1],
                'cloud_top': max(senkou_span_a.iloc[-1], senkou_span_b.iloc[-1]) if not senkou_span_a.empty and not senkou_span_b.empty else 0,
                'cloud_bottom': min(senkou_span_a.iloc[-1], senkou_span_b.iloc[-1]) if not senkou_span_a.empty and not senkou_span_b.empty else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Ichimoku: {e}")
            return {}
    
    async def calculate_atr(self, df: pd.DataFrame) -> float:
        """Calculate Average True Range for dynamic SL/TP"""
        try:
            if len(df) < self.atr_period:
                return 0.0
            
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift(1))
            low_close = np.abs(df['low'] - df['close'].shift(1))
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=self.atr_period).mean()
            
            return float(atr.iloc[-1]) if not atr.empty else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return 0.0
    
    async def analyze_ichimoku_signal(self, ichimoku_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Ichimoku components for trading signals"""
        try:
            signal_analysis = {
                'signal': 'HOLD',
                'strength': 0.0,
                'confidence': 0.0,
                'reasons': []
            }
            
            tenkan = ichimoku_data.get('tenkan_sen', 0)
            kijun = ichimoku_data.get('kijun_sen', 0)
            senkou_a = ichimoku_data.get('senkou_span_a', 0)
            senkou_b = ichimoku_data.get('senkou_span_b', 0)
            chikou = ichimoku_data.get('chikou_span', 0)
            current_price = ichimoku_data.get('current_price', 0)
            cloud_top = ichimoku_data.get('cloud_top', 0)
            cloud_bottom = ichimoku_data.get('cloud_bottom', 0)
            
            if not all([tenkan, kijun, current_price]):
                return signal_analysis
            
            # Signal scoring system
            bullish_score = 0
            bearish_score = 0
            
            # 1. Tenkan-Kijun Cross
            if tenkan > kijun:
                bullish_score += 25
                signal_analysis['reasons'].append("Tenkan above Kijun (bullish)")
            else:
                bearish_score += 25
                signal_analysis['reasons'].append("Tenkan below Kijun (bearish)")
            
            # 2. Price vs Cloud
            if current_price > cloud_top:
                bullish_score += 30
                signal_analysis['reasons'].append("Price above cloud (strong bullish)")
            elif current_price < cloud_bottom:
                bearish_score += 30
                signal_analysis['reasons'].append("Price below cloud (strong bearish)")
            elif cloud_bottom < current_price < cloud_top:
                # Price in cloud - neutral with slight bias
                if senkou_a > senkou_b:
                    bullish_score += 10
                    signal_analysis['reasons'].append("Price in bullish cloud")
                else:
                    bearish_score += 10
                    signal_analysis['reasons'].append("Price in bearish cloud")
            
            # 3. Chikou Span vs Price
            if chikou > current_price:
                bullish_score += 20
                signal_analysis['reasons'].append("Chikou above price (bullish momentum)")
            else:
                bearish_score += 20
                signal_analysis['reasons'].append("Chikou below price (bearish momentum)")
            
            # 4. Cloud Color (Senkou A vs Senkou B)
            if senkou_a > senkou_b:
                bullish_score += 15
                signal_analysis['reasons'].append("Bullish cloud (green)")
            else:
                bearish_score += 15
                signal_analysis['reasons'].append("Bearish cloud (red)")
            
            # 5. Price momentum vs Kijun
            if current_price > kijun:
                bullish_score += 10
                signal_analysis['reasons'].append("Price above Kijun (bullish bias)")
            else:
                bearish_score += 10
                signal_analysis['reasons'].append("Price below Kijun (bearish bias)")
            
            # Determine final signal
            total_possible = 100
            if bullish_score > bearish_score:
                signal_analysis['signal'] = 'BUY'
                signal_analysis['strength'] = (bullish_score / total_possible) * 100
                signal_analysis['confidence'] = min(((bullish_score - bearish_score) / total_possible) * 100 + 50, 95)
            elif bearish_score > bullish_score:
                signal_analysis['signal'] = 'SELL'
                signal_analysis['strength'] = (bearish_score / total_possible) * 100
                signal_analysis['confidence'] = min(((bearish_score - bullish_score) / total_possible) * 100 + 50, 95)
            else:
                signal_analysis['signal'] = 'HOLD'
                signal_analysis['strength'] = 50.0
                signal_analysis['confidence'] = 30.0
                signal_analysis['reasons'].append("Neutral signals - no clear direction")
            
            return signal_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing Ichimoku signal: {e}")
            return {'signal': 'HOLD', 'strength': 0.0, 'confidence': 0.0, 'reasons': []}
    
    async def calculate_dynamic_sl_tp(self, entry_price: float, direction: str, atr_value: float) -> Tuple[float, float]:
        """Calculate dynamic stop loss and take profit using ATR"""
        try:
            if direction.upper() == 'BUY':
                stop_loss = entry_price - (atr_value * self.atr_multiplier_sl)
                take_profit = entry_price + (atr_value * self.atr_multiplier_tp)
            else:  # SELL
                stop_loss = entry_price + (atr_value * self.atr_multiplier_sl)
                take_profit = entry_price - (atr_value * self.atr_multiplier_tp)
            
            return round(stop_loss, 5), round(take_profit, 5)
            
        except Exception as e:
            self.logger.error(f"Error calculating dynamic SL/TP: {e}")
            return entry_price, entry_price
    
    async def generate_signal(self, market_data: List[List]) -> Optional[IchimokuSignal]:
        """Generate Ichimoku Sniper trading signal"""
        try:
            if not market_data or len(market_data) < self.senkou_span_b_period + self.displacement:
                self.logger.warning("Insufficient market data for Ichimoku analysis")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(market_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Calculate Ichimoku components
            ichimoku_data = await self.calculate_ichimoku(df)
            if not ichimoku_data:
                return None
            
            # Calculate ATR for dynamic SL/TP
            atr_value = await self.calculate_atr(df)
            if atr_value <= 0:
                self.logger.warning("Invalid ATR value, skipping signal")
                return None
            
            # Analyze signal
            signal_analysis = await self.analyze_ichimoku_signal(ichimoku_data)
            
            if signal_analysis['signal'] == 'HOLD':
                return None
            
            # Check minimum thresholds
            if (signal_analysis['strength'] < self.min_signal_strength or 
                signal_analysis['confidence'] < self.min_confidence):
                self.logger.debug(f"Signal below thresholds: strength={signal_analysis['strength']:.1f}, confidence={signal_analysis['confidence']:.1f}")
                return None
            
            # Get current price and calculate SL/TP
            entry_price = float(df['close'].iloc[-1])
            stop_loss, take_profit = await self.calculate_dynamic_sl_tp(
                entry_price, signal_analysis['signal'], atr_value
            )
            
            # Create signal
            signal = IchimokuSignal(
                symbol=self.symbol,
                action=signal_analysis['signal'],
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                signal_strength=signal_analysis['strength'],
                confidence=signal_analysis['confidence'],
                timeframe=self.timeframe,
                timestamp=datetime.now(),
                ichimoku_data=ichimoku_data,
                atr_value=atr_value,
                risk_reward_ratio=2.0
            )
            
            self.logger.info(f"🎯 Ichimoku Sniper Signal: {signal.action} {self.symbol} @ {entry_price:.5f}")
            self.logger.info(f"   SL: {stop_loss:.5f} | TP: {take_profit:.5f} | Strength: {signal_analysis['strength']:.1f}%")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating Ichimoku signal: {e}")
            return None
    
    async def format_cornix_signal(self, signal: IchimokuSignal) -> str:
        """Format signal for Cornix compatibility"""
        try:
            cornix_signal = f"""
🎯 **ICHIMOKU SNIPER SIGNAL**

**Pair:** {signal.symbol}
**Direction:** {signal.action}
**Entry:** {signal.entry_price:.5f}
**Stop Loss:** {signal.stop_loss:.5f}
**Take Profit:** {signal.take_profit:.5f}

**Leverage:** Auto (Dynamic)
**Risk/Reward:** 1:{signal.risk_reward_ratio}
**Timeframe:** {signal.timeframe}
**Strength:** {signal.signal_strength:.1f}%
**Confidence:** {signal.confidence:.1f}%

**Strategy:** Ichimoku Cloud Analysis
**ATR:** {signal.atr_value:.6f}

⏰ {signal.timestamp.strftime('%H:%M:%S UTC')}
            """.strip()
            
            return cornix_signal
            
        except Exception as e:
            self.logger.error(f"Error formatting Cornix signal: {e}")
            return ""
