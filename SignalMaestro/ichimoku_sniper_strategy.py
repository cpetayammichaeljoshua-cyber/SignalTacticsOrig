
#!/usr/bin/env python3
"""
Ichimoku Sniper Strategy for FXSUSDT.P - Exact Pine Script Implementation
Based on the provided Pine Script strategy with precise parameter matching
Uses 15/30-minute timeframes with EMA 200 filter and comprehensive Ichimoku conditions
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
    risk_reward_ratio: float = 1.86  # 3.25/1.75 from Pine Script

class IchimokuSniperStrategy:
    """
    Ichimoku Sniper Strategy Implementation - Exact Pine Script Match
    - Uses 15/30-minute timeframes
    - Focuses on FXSUSDT.P trading
    - Implements exact Pine Script conditions and parameters
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.symbol = "FXSUSDT.P"
        self.timeframe = "30m"
        
        # Exact Pine Script Ichimoku parameters
        self.conversion_periods = 4      # conversionPeriods
        self.base_periods = 4           # basePeriods  
        self.lagging_span_2_periods = 46  # laggingSpan2Periods
        self.displacement = 20          # displacement
        
        # EMA 200 filter (longest)
        self.ema_period = 200
        
        # Pine Script stop loss and take profit percentages
        self.stop_loss_percent = 1.75   # percentStop
        self.take_profit_percent = 3.25 # percentTP
        
        # Signal strength thresholds
        self.min_signal_strength = 75.0
        self.min_confidence = 70.0
        
        self.logger.info(f"🎯 Ichimoku Sniper Strategy initialized (Pine Script Match)")
        self.logger.info(f"   Parameters: Conv({self.conversion_periods}), Base({self.base_periods}), LaggingB({self.lagging_span_2_periods}), Disp({self.displacement})")
        self.logger.info(f"   SL: {self.stop_loss_percent}%, TP: {self.take_profit_percent}%, EMA: {self.ema_period}")
    
    def donchian(self, data: pd.Series, length: int) -> pd.Series:
        """Donchian channel calculation - exact Pine Script implementation"""
        try:
            highest = data.rolling(window=length).max()
            lowest = data.rolling(window=length).min()
            return (highest + lowest) / 2
        except Exception as e:
            self.logger.error(f"Error in donchian calculation: {e}")
            return pd.Series([0] * len(data))
    
    async def calculate_ichimoku_pine_script(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Ichimoku components exactly as Pine Script"""
        try:
            if len(df) < max(self.lagging_span_2_periods, self.displacement, self.ema_period) + 5:
                return {}
            
            # Pine Script Ichimoku calculations
            conversion_line = self.donchian(df['high'].combine(df['low'], max), self.conversion_periods)
            base_line = self.donchian(df['high'].combine(df['low'], max), self.base_periods)
            
            # Lead Line 1 = average of conversion and base lines
            lead_line_1 = (conversion_line + base_line) / 2
            
            # Lead Line 2 = donchian of lagging span 2 periods
            lead_line_2 = self.donchian(df['high'].combine(df['low'], max), self.lagging_span_2_periods)
            
            # Lagging span = close shifted by displacement
            lagging_span = df['close'].shift(-self.displacement)
            
            # EMA 200 (longest in Pine Script)
            ema_200 = df['close'].ewm(span=self.ema_period).mean()
            
            # Current values
            current_close = df['close'].iloc[-1]
            current_conversion = conversion_line.iloc[-1] if not conversion_line.empty else 0
            current_base = base_line.iloc[-1] if not base_line.empty else 0
            current_lead1 = lead_line_1.iloc[-1] if not lead_line_1.empty else 0
            current_lead2 = lead_line_2.iloc[-1] if not lead_line_2.empty else 0
            current_ema200 = ema_200.iloc[-1] if not ema_200.empty else 0
            current_lagging = lagging_span.iloc[-1] if not lagging_span.empty else 0
            
            return {
                'conversion_line': current_conversion,
                'base_line': current_base, 
                'lead_line_1': current_lead1,
                'lead_line_2': current_lead2,
                'lagging_span': current_lagging,
                'ema_200': current_ema200,
                'current_price': current_close,
                'cloud_top': max(current_lead1, current_lead2),
                'cloud_bottom': min(current_lead1, current_lead2),
                'cloud_color': 'bullish' if current_lead1 > current_lead2 else 'bearish'
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Pine Script Ichimoku: {e}")
            return {}
    
    async def check_pine_script_conditions(self, ichimoku_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check exact Pine Script entry conditions"""
        try:
            signal_analysis = {
                'signal': 'HOLD',
                'strength': 0.0,
                'confidence': 0.0,
                'reasons': []
            }
            
            close = ichimoku_data.get('current_price', 0)
            ema_200 = ichimoku_data.get('ema_200', 0) 
            lead_line_1 = ichimoku_data.get('lead_line_1', 0)
            lead_line_2 = ichimoku_data.get('lead_line_2', 0)
            conversion_line = ichimoku_data.get('conversion_line', 0)
            base_line = ichimoku_data.get('base_line', 0)
            
            if not all([close, ema_200, lead_line_1, lead_line_2, conversion_line, base_line]):
                return signal_analysis
            
            # Exact Pine Script long entry condition
            long_entry = (close > ema_200 and 
                         close > lead_line_1 and 
                         close > lead_line_2 and 
                         close > conversion_line and 
                         close > base_line)
            
            # Exact Pine Script short entry condition  
            short_entry = (close < ema_200 and 
                          close < lead_line_1 and 
                          close < lead_line_2 and 
                          close < conversion_line and 
                          close < base_line)
            
            if long_entry:
                signal_analysis['signal'] = 'BUY'
                signal_analysis['strength'] = 85.0
                signal_analysis['confidence'] = 80.0
                signal_analysis['reasons'] = [
                    f"Price ({close:.5f}) > EMA200 ({ema_200:.5f})",
                    f"Price > Lead Line A ({lead_line_1:.5f})", 
                    f"Price > Lead Line B ({lead_line_2:.5f})",
                    f"Price > Conversion Line ({conversion_line:.5f})",
                    f"Price > Base Line ({base_line:.5f})",
                    "All Pine Script LONG conditions met"
                ]
                
            elif short_entry:
                signal_analysis['signal'] = 'SELL'
                signal_analysis['strength'] = 85.0
                signal_analysis['confidence'] = 80.0
                signal_analysis['reasons'] = [
                    f"Price ({close:.5f}) < EMA200 ({ema_200:.5f})",
                    f"Price < Lead Line A ({lead_line_1:.5f})",
                    f"Price < Lead Line B ({lead_line_2:.5f})", 
                    f"Price < Conversion Line ({conversion_line:.5f})",
                    f"Price < Base Line ({base_line:.5f})",
                    "All Pine Script SHORT conditions met"
                ]
            else:
                signal_analysis['reasons'] = ["Pine Script entry conditions not met"]
            
            return signal_analysis
            
        except Exception as e:
            self.logger.error(f"Error checking Pine Script conditions: {e}")
            return {'signal': 'HOLD', 'strength': 0.0, 'confidence': 0.0, 'reasons': []}
    
    async def calculate_pine_script_sl_tp(self, entry_price: float, direction: str) -> Tuple[float, float]:
        """Calculate SL/TP exactly as Pine Script percentages"""
        try:
            if direction.upper() == 'BUY':
                stop_loss = entry_price * (1 - self.stop_loss_percent / 100)
                take_profit = entry_price * (1 + self.take_profit_percent / 100)
            else:  # SELL
                stop_loss = entry_price * (1 + self.stop_loss_percent / 100)  
                take_profit = entry_price * (1 - self.take_profit_percent / 100)
            
            return round(stop_loss, 5), round(take_profit, 5)
            
        except Exception as e:
            self.logger.error(f"Error calculating Pine Script SL/TP: {e}")
            return entry_price, entry_price
    
    async def calculate_atr(self, df: pd.DataFrame) -> float:
        """Calculate ATR for signal metadata"""
        try:
            if len(df) < 14:
                return 0.0
            
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift(1))
            low_close = np.abs(df['low'] - df['close'].shift(1))
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=14).mean()
            
            return float(atr.iloc[-1]) if not atr.empty else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return 0.0
    
    async def generate_signal(self, market_data: List[List]) -> Optional[IchimokuSignal]:
        """Generate Ichimoku Sniper signal using exact Pine Script logic"""
        try:
            required_data_points = max(self.lagging_span_2_periods, self.displacement, self.ema_period) + 10
            if not market_data or len(market_data) < required_data_points:
                self.logger.warning(f"Insufficient data: need {required_data_points}, got {len(market_data) if market_data else 0}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(market_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Calculate Pine Script Ichimoku components
            ichimoku_data = await self.calculate_ichimoku_pine_script(df)
            if not ichimoku_data:
                self.logger.warning("Failed to calculate Ichimoku data")
                return None
            
            # Check Pine Script conditions
            signal_analysis = await self.check_pine_script_conditions(ichimoku_data)
            
            if signal_analysis['signal'] == 'HOLD':
                self.logger.debug("No Pine Script signal conditions met")
                return None
            
            # Check minimum thresholds
            if (signal_analysis['strength'] < self.min_signal_strength or 
                signal_analysis['confidence'] < self.min_confidence):
                self.logger.debug(f"Signal below thresholds: strength={signal_analysis['strength']:.1f}, confidence={signal_analysis['confidence']:.1f}")
                return None
            
            # Get entry price and calculate Pine Script SL/TP
            entry_price = float(df['close'].iloc[-1])
            stop_loss, take_profit = await self.calculate_pine_script_sl_tp(
                entry_price, signal_analysis['signal']
            )
            
            # Calculate ATR for metadata
            atr_value = await self.calculate_atr(df)
            
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
                risk_reward_ratio=self.take_profit_percent / self.stop_loss_percent
            )
            
            self.logger.info(f"🎯 Pine Script Ichimoku Signal: {signal.action} {self.symbol} @ {entry_price:.5f}")
            self.logger.info(f"   SL: {stop_loss:.5f} ({self.stop_loss_percent}%) | TP: {take_profit:.5f} ({self.take_profit_percent}%)")
            self.logger.info(f"   Conditions: {', '.join(signal_analysis['reasons'][:3])}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating Pine Script signal: {e}")
            return None
    
    async def format_cornix_signal(self, signal: IchimokuSignal) -> str:
        """Format signal for Cornix compatibility with Pine Script details"""
        try:
            # Calculate percentage differences
            if signal.action == "BUY":
                sl_percent = ((signal.entry_price - signal.stop_loss) / signal.entry_price) * 100
                tp_percent = ((signal.take_profit - signal.entry_price) / signal.entry_price) * 100
            else:
                sl_percent = ((signal.stop_loss - signal.entry_price) / signal.entry_price) * 100
                tp_percent = ((signal.entry_price - signal.take_profit) / signal.entry_price) * 100
            
            cornix_signal = f"""
🎯 **ICHIMOKU SNIPER - PINE SCRIPT MATCH**

**📊 SIGNAL DETAILS:**
• **Pair:** `{signal.symbol}`
• **Direction:** `{signal.action}`
• **Entry:** `{signal.entry_price:.5f}`
• **Stop Loss:** `{signal.stop_loss:.5f}` (-{sl_percent:.2f}%)
• **Take Profit:** `{signal.take_profit:.5f}` (+{tp_percent:.2f}%)

**⚙️ PINE SCRIPT PARAMETERS:**
• **Strategy:** Ichimoku Sniper FXSUSDT 15/30m
• **Conv/Base:** {self.conversion_periods}/{self.base_periods}
• **LaggingB/Disp:** {self.lagging_span_2_periods}/{self.displacement}
• **EMA Filter:** {self.ema_period}
• **SL/TP %:** {self.stop_loss_percent}%/{self.take_profit_percent}%

**📈 SIGNAL ANALYSIS:**
• **Strength:** `{signal.signal_strength:.1f}%`
• **Confidence:** `{signal.confidence:.1f}%`
• **R/R Ratio:** `1:{signal.risk_reward_ratio:.2f}`
• **Timeframe:** `{signal.timeframe}`

**🎯 CORNIX FORMAT:**
```
{signal.symbol} {signal.action}
Entry: {signal.entry_price:.5f}
SL: {signal.stop_loss:.5f}
TP: {signal.take_profit:.5f}
Leverage: Auto
```

**⏰ Signal Time:** `{signal.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}`
**🤖 Strategy:** `Pine Script Ichimoku Sniper v6`

*Exact Pine Script implementation with comprehensive Ichimoku analysis*
            """.strip()
            
            return cornix_signal
            
        except Exception as e:
            self.logger.error(f"Error formatting Pine Script signal: {e}")
            return ""
