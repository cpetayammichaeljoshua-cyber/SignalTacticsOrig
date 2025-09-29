
#!/usr/bin/env python3
"""
Leverage and Margin Calculator Utility
Provides standardized calculations for leverage and margin settings across all trading bots
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

class LeverageMarginCalculator:
    """Utility class for calculating optimal leverage and margin settings"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Default configuration
        self.config = {
            'min_leverage': 2,
            'max_leverage': 20,
            'default_leverage': 5,
            'default_margin_type': 'CROSS',
            'enable_cross_margin': True,
            'enable_auto_add_margin': True,
            'volatility_leverage_factor': 0.1,  # How much volatility affects leverage
            'signal_strength_factor': 0.15,    # How much signal strength affects leverage
        }
    
    def calculate_optimal_leverage(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate optimal leverage based on signal characteristics
        
        Args:
            signal_data: Dictionary containing signal information
            
        Returns:
            Dictionary with leverage and margin recommendations
        """
        try:
            # Extract signal characteristics
            signal_strength = signal_data.get('strength', 75.0)
            confidence = signal_data.get('confidence', signal_strength)
            volatility_score = signal_data.get('volatility_score', 2.0)
            timeframe = signal_data.get('timeframe', '1h')
            trade_size_usdt = signal_data.get('trade_size_usdt', 100.0)
            direction = signal_data.get('action', 'BUY').upper()
            
            # Base leverage calculation
            base_leverage = self._calculate_base_leverage(signal_strength, confidence)
            
            # Adjust for volatility
            volatility_adjustment = self._calculate_volatility_adjustment(volatility_score)
            
            # Adjust for timeframe
            timeframe_adjustment = self._calculate_timeframe_adjustment(timeframe)
            
            # Adjust for trade size
            size_adjustment = self._calculate_size_adjustment(trade_size_usdt)
            
            # Adjust for direction (shorts typically get lower leverage in volatile markets)
            direction_adjustment = self._calculate_direction_adjustment(direction, volatility_score)
            
            # Calculate final leverages
            auto_leverage = int(base_leverage * volatility_adjustment * timeframe_adjustment * 
                              size_adjustment * direction_adjustment)
            auto_leverage = max(self.config['min_leverage'], 
                               min(auto_leverage, self.config['max_leverage']))
            
            recommended_leverage = max(self.config['min_leverage'], 
                                     int(auto_leverage * 0.85))  # More conservative
            
            # Calculate margin settings
            margin_settings = self._calculate_margin_settings(auto_leverage, volatility_score)
            
            return {
                'recommended_leverage': recommended_leverage,
                'auto_leverage': auto_leverage,
                'max_safe_leverage': min(auto_leverage + 2, self.config['max_leverage']),
                'margin_type': self.config['default_margin_type'],
                'cross_margin_enabled': self.config['enable_cross_margin'],
                'leverage_mode': 'AUTO',
                'margin_settings': margin_settings,
                'leverage_rationale': self._generate_leverage_rationale(
                    base_leverage, volatility_adjustment, timeframe_adjustment,
                    size_adjustment, direction_adjustment, auto_leverage
                ),
                'risk_metrics': {
                    'volatility_impact': f"{(1 - volatility_adjustment) * 100:+.1f}%",
                    'timeframe_impact': f"{(timeframe_adjustment - 1) * 100:+.1f}%",
                    'size_impact': f"{(size_adjustment - 1) * 100:+.1f}%",
                    'direction_impact': f"{(direction_adjustment - 1) * 100:+.1f}%"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal leverage: {e}")
            return self._get_default_leverage_settings()
    
    def _calculate_base_leverage(self, signal_strength: float, confidence: float) -> float:
        """Calculate base leverage from signal strength and confidence"""
        # Combine signal strength and confidence
        combined_score = (signal_strength * 0.7) + (confidence * 0.3)
        
        # Map score to leverage (80+ = high leverage, 60- = low leverage)
        if combined_score >= 90:
            return 12.0
        elif combined_score >= 85:
            return 10.0
        elif combined_score >= 80:
            return 8.0
        elif combined_score >= 75:
            return 6.0
        elif combined_score >= 70:
            return 5.0
        elif combined_score >= 65:
            return 4.0
        else:
            return 3.0
    
    def _calculate_volatility_adjustment(self, volatility_score: float) -> float:
        """Calculate leverage adjustment based on volatility"""
        # High volatility = lower leverage
        if volatility_score <= 1.0:
            return 1.2  # Very stable - increase leverage
        elif volatility_score <= 2.0:
            return 1.0  # Normal volatility - no adjustment
        elif volatility_score <= 3.0:
            return 0.85  # Moderate volatility - reduce leverage
        elif volatility_score <= 4.0:
            return 0.7   # High volatility - significant reduction
        else:
            return 0.5   # Extreme volatility - major reduction
    
    def _calculate_timeframe_adjustment(self, timeframe: str) -> float:
        """Calculate leverage adjustment based on timeframe"""
        timeframe_multipliers = {
            '1m': 0.7,   # Very short-term - lower leverage
            '5m': 0.8,   # Short-term - reduced leverage
            '15m': 0.9,  # Short-term - slightly reduced
            '30m': 1.0,  # Standard
            '1h': 1.1,   # Medium-term - slightly higher
            '2h': 1.15,  # Medium-term - higher
            '4h': 1.2,   # Longer-term - higher leverage
            '6h': 1.2,   # Longer-term
            '12h': 1.1,  # Very long-term - moderate
            '1d': 1.0    # Daily - standard
        }
        
        return timeframe_multipliers.get(timeframe, 1.0)
    
    def _calculate_size_adjustment(self, trade_size_usdt: float) -> float:
        """Calculate leverage adjustment based on trade size"""
        # Larger trades typically use lower leverage for risk management
        if trade_size_usdt <= 50:
            return 1.0      # Small trade - no adjustment
        elif trade_size_usdt <= 100:
            return 0.95     # Medium trade - slight reduction
        elif trade_size_usdt <= 200:
            return 0.9      # Large trade - moderate reduction
        elif trade_size_usdt <= 500:
            return 0.8      # Very large trade - significant reduction
        else:
            return 0.7      # Huge trade - major reduction
    
    def _calculate_direction_adjustment(self, direction: str, volatility_score: float) -> float:
        """Calculate leverage adjustment based on trade direction and volatility"""
        # Shorts in volatile markets get lower leverage
        if direction in ['SELL', 'SHORT'] and volatility_score > 2.5:
            return 0.85
        else:
            return 1.0
    
    def _calculate_margin_settings(self, leverage: int, volatility_score: float) -> Dict[str, Any]:
        """Calculate comprehensive margin settings"""
        # Maintenance margin ratio based on leverage
        if leverage >= 15:
            maintenance_ratio = 0.10  # 10%
        elif leverage >= 10:
            maintenance_ratio = 0.075  # 7.5%
        elif leverage >= 5:
            maintenance_ratio = 0.05   # 5%
        else:
            maintenance_ratio = 0.025  # 2.5%
        
        # Adjust for volatility
        if volatility_score > 3.0:
            maintenance_ratio *= 1.2  # Increase for high volatility
        
        return {
            'type': self.config['default_margin_type'],
            'cross_enabled': self.config['enable_cross_margin'],
            'isolated_enabled': not self.config['enable_cross_margin'],
            'auto_add_margin': self.config['enable_auto_add_margin'],
            'maintenance_margin_ratio': round(maintenance_ratio, 4),
            'initial_margin_ratio': round(1.0 / leverage, 4),
            'margin_call_ratio': round(maintenance_ratio * 1.1, 4),
            'liquidation_ratio': round(maintenance_ratio * 0.9, 4)
        }
    
    def _generate_leverage_rationale(self, base_leverage: float, volatility_adj: float,
                                   timeframe_adj: float, size_adj: float, direction_adj: float,
                                   final_leverage: int) -> str:
        """Generate human-readable explanation for leverage calculation"""
        factors = []
        
        if volatility_adj < 0.9:
            factors.append(f"reduced due to high volatility ({volatility_adj:.2f}x)")
        elif volatility_adj > 1.1:
            factors.append(f"increased due to low volatility ({volatility_adj:.2f}x)")
        
        if timeframe_adj > 1.0:
            factors.append(f"increased for longer timeframe ({timeframe_adj:.2f}x)")
        elif timeframe_adj < 1.0:
            factors.append(f"reduced for shorter timeframe ({timeframe_adj:.2f}x)")
        
        if size_adj < 1.0:
            factors.append(f"reduced for large position size ({size_adj:.2f}x)")
        
        if direction_adj < 1.0:
            factors.append("reduced for short position in volatile market")
        
        if not factors:
            return f"Standard leverage calculation: {base_leverage:.1f}x → {final_leverage}x"
        
        return f"Base: {base_leverage:.1f}x, {', '.join(factors)} → Final: {final_leverage}x"
    
    def _get_default_leverage_settings(self) -> Dict[str, Any]:
        """Return default leverage settings in case of calculation error"""
        return {
            'recommended_leverage': self.config['default_leverage'],
            'auto_leverage': self.config['default_leverage'],
            'max_safe_leverage': self.config['default_leverage'] + 2,
            'margin_type': self.config['default_margin_type'],
            'cross_margin_enabled': self.config['enable_cross_margin'],
            'leverage_mode': 'AUTO',
            'margin_settings': self._calculate_margin_settings(self.config['default_leverage'], 2.0),
            'leverage_rationale': 'Using default settings due to calculation error',
            'risk_metrics': {
                'volatility_impact': '0.0%',
                'timeframe_impact': '0.0%',
                'size_impact': '0.0%',
                'direction_impact': '0.0%'
            }
        }
    
    def format_leverage_display(self, leverage_data: Dict[str, Any]) -> str:
        """Format leverage and margin information for display"""
        try:
            margin_emoji = "🔗" if leverage_data.get('cross_margin_enabled') else "🔒"
            margin_type = leverage_data.get('margin_type', 'CROSS')
            
            display = f"""⚡ **LEVERAGE & MARGIN:**
🔧 **Recommended:** `{leverage_data.get('recommended_leverage', 5)}x`
🤖 **Auto Leverage:** `{leverage_data.get('auto_leverage', 5)}x`
{margin_emoji} **Margin Type:** `{margin_type}`
🔗 **Cross Margin:** `{'✅ Enabled' if leverage_data.get('cross_margin_enabled') else '❌ Disabled'}`
📊 **Maintenance Ratio:** `{leverage_data.get('margin_settings', {}).get('maintenance_margin_ratio', 0.05)*100:.1f}%`

💡 **Rationale:** {leverage_data.get('leverage_rationale', 'Standard calculation')}"""
            
            return display
            
        except Exception as e:
            self.logger.error(f"Error formatting leverage display: {e}")
            return "⚡ **LEVERAGE:** Using default settings"
    
    def format_comprehensive_signal(self, signal_data: Dict[str, Any], price_data: Dict[str, Any]) -> str:
        """Format comprehensive trading signal with all details"""
        try:
            # Calculate leverage and margin
            leverage_info = self.calculate_optimal_leverage(signal_data)
            
            symbol = signal_data.get('symbol', 'UNKNOWN')
            action = signal_data.get('action', 'BUY').upper()
            entry = price_data.get('entry_price', 0)
            stop_loss = price_data.get('stop_loss', 0)
            take_profit = price_data.get('take_profit', 0)
            
            # Direction styling
            if action in ['BUY', 'LONG']:
                emoji = "🟢"
                direction_emoji = "📈"
            else:
                emoji = "🔴" 
                direction_emoji = "📉"
            
            # Calculate percentages
            if action == "BUY":
                sl_percent = ((entry - stop_loss) / entry) * 100 if stop_loss else 0
                tp_percent = ((take_profit - entry) / entry) * 100 if take_profit else 0
            else:
                sl_percent = ((stop_loss - entry) / entry) * 100 if stop_loss else 0
                tp_percent = ((entry - take_profit) / entry) * 100 if take_profit else 0
            
            # Risk management calculations
            risk_percentage = 2.0
            account_size = 1000.0
            risk_amount = account_size * (risk_percentage / 100)
            
            price_diff = abs(entry - stop_loss) if stop_loss else entry * 0.02
            position_size_usdt = (risk_amount / price_diff) * entry if price_diff > 0 else 100.0
            position_size_usdt = min(position_size_usdt, account_size * 0.2)
            
            quantity = round(position_size_usdt / entry, 1) if entry > 0 else 0.1
            quantity = max(0.1, quantity)
            
            comprehensive_signal = f"""
{emoji} **COMPREHENSIVE TRADING SIGNAL** {direction_emoji}

**📊 TRADE SETUP:**
• **Pair:** `{symbol}`
• **Direction:** `{action}`
• **Entry Price:** `{entry:.5f}`
• **Stop Loss:** `{stop_loss:.5f}` (-{sl_percent:.2f}%)
• **Take Profit:** `{take_profit:.5f}` (+{tp_percent:.2f}%)
• **Quantity:** `{quantity} USDT`

**⚡ DYNAMIC LEVERAGE & MARGIN:**
• **Recommended:** `{leverage_info['recommended_leverage']}x`
• **Auto Leverage:** `{leverage_info['auto_leverage']}x`
• **Max Safe:** `{leverage_info['max_safe_leverage']}x`
• **Margin Type:** `{leverage_info['margin_type']}`
• **Cross Margin:** `{'✅ Enabled' if leverage_info['cross_margin_enabled'] else '❌ Disabled'}`

**💰 RISK MANAGEMENT:**
• **Position Size:** `${position_size_usdt:.0f} USDT`
• **Risk Amount:** `${risk_amount:.0f} USDT ({risk_percentage}%)`
• **Risk/Reward:** `1:{abs(tp_percent/sl_percent):.2f}` if sl_percent > 0 else "N/A"
• **Maintenance Margin:** `{leverage_info['margin_settings']['maintenance_margin_ratio']*100:.1f}%`

**🎯 CORNIX FORMAT:**
```
#{symbol} {action}

Entry Targets:
{entry:.5f}

Take-Profit Targets:
{take_profit:.5f}

Stop Targets:  
{stop_loss:.5f}

Leverage: {leverage_info['auto_leverage']}x
```

**📈 ANALYSIS SUMMARY:**
• **Signal Strength:** `{signal_data.get('strength', 75):.1f}%`
• **Confidence Level:** `{signal_data.get('confidence', 75):.1f}%`
• **Timeframe:** `{signal_data.get('timeframe', '1h')}`
• **Strategy:** `{signal_data.get('strategy', 'Advanced Analysis')}`

**💡 Leverage Rationale:** {leverage_info.get('leverage_rationale', 'Optimized for current market conditions')}

*🤖 Dynamically Optimized | ⚡ Cross Margin Ready | 📊 Comprehensive Risk Management*
            """.strip()
            
            return comprehensive_signal
            
        except Exception as e:
            self.logger.error(f"Error formatting comprehensive signal: {e}")
            return "❌ Error formatting signal - using fallback format"

# Global instance for easy access
leverage_calculator = LeverageMarginCalculator()
