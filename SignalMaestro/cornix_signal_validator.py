
#!/usr/bin/env python3
"""
Cornix Signal Validator
Ensures signals are properly formatted for Cornix bot compatibility
"""

import logging
from typing import Dict, Any, Optional

class CornixSignalValidator:
    """Validates and formats signals for Cornix compatibility"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate signal meets Cornix requirements"""
        try:
            direction = signal.get('direction', '').upper()
            entry = float(signal.get('entry_price', 0))
            stop_loss = float(signal.get('stop_loss', 0))
            tp1 = float(signal.get('tp1', 0))
            tp2 = float(signal.get('tp2', 0))
            tp3 = float(signal.get('tp3', 0))
            
            # Check all required fields exist
            if not all([direction, entry, stop_loss, tp1, tp2, tp3]):
                self.logger.error("Missing required signal fields")
                return False
            
            # Check price relationships for BUY signals
            if direction == 'BUY':
                # For BUY: stop_loss < entry < tp1 < tp2 < tp3
                if not (stop_loss < entry < tp1 < tp2 < tp3):
                    self.logger.error(f"Invalid BUY price order: SL({stop_loss}) < Entry({entry}) < TP1({tp1}) < TP2({tp2}) < TP3({tp3})")
                    return False
            
            # Check price relationships for SELL signals
            elif direction == 'SELL':
                # For SELL: tp3 < tp2 < tp1 < entry < stop_loss
                if not (tp3 < tp2 < tp1 < entry < stop_loss):
                    self.logger.error(f"Invalid SELL price order: TP3({tp3}) < TP2({tp2}) < TP1({tp1}) < Entry({entry}) < SL({stop_loss})")
                    return False
            
            else:
                self.logger.error(f"Invalid direction: {direction}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating signal: {e}")
            return False
    
    def fix_signal_prices(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Fix signal prices to meet Cornix requirements"""
        try:
            direction = signal.get('direction', '').upper()
            entry = float(signal.get('entry_price', 0))
            
            if direction == 'BUY':
                # Calculate proper prices for BUY signal
                risk_amount = entry * 0.02  # 2% risk
                
                stop_loss = entry - risk_amount
                tp1 = entry + (risk_amount * 1.0)  # 1:1 RR
                tp2 = entry + (risk_amount * 2.0)  # 1:2 RR
                tp3 = entry + (risk_amount * 3.0)  # 1:3 RR
                
            else:  # SELL
                # Calculate proper prices for SELL signal
                risk_amount = entry * 0.02  # 2% risk
                
                stop_loss = entry + risk_amount
                tp1 = entry - (risk_amount * 1.0)  # 1:1 RR
                tp2 = entry - (risk_amount * 2.0)  # 1:2 RR
                tp3 = entry - (risk_amount * 3.0)  # 1:3 RR
            
            # Update signal with fixed prices
            fixed_signal = signal.copy()
            fixed_signal.update({
                'entry_price': entry,
                'stop_loss': stop_loss,
                'tp1': tp1,
                'tp2': tp2,
                'tp3': tp3
            })
            
            return fixed_signal
            
        except Exception as e:
            self.logger.error(f"Error fixing signal prices: {e}")
            return signal
    
    def format_for_cornix(self, signal: Dict[str, Any]) -> str:
        """Format signal message for Cornix compatibility with enhanced structure"""
        try:
            # Validate and fix signal first
            if not self.validate_signal(signal):
                signal = self.fix_signal_prices(signal)
            
            symbol = signal['symbol']
            direction = signal['direction'].upper()
            entry = signal['entry_price']
            stop_loss = signal['stop_loss']
            tp1 = signal['tp1']
            tp2 = signal['tp2']
            tp3 = signal['tp3']
            leverage = signal.get('optimal_leverage', 50)
            
            # Enhanced Cornix-compatible format that matches expected structure
            formatted_message = f"""**Channel:** SignalTactics
**Symbol:** {symbol}
**Exchanges:** Binance, BingX Spot, Bitget Spot, ByBit Spot, Huobi.pro, KuCoin, OKX

**Entry Targets:**
{entry:.6f}

**Take-Profit Targets:**
🎯 {tp1:.6f}
🎯 {tp2:.6f}
🎯 {tp3:.6f}

**Stop Targets:**
🛑 {stop_loss:.6f}

**Leverage:** {leverage}x
**Direction:** {direction}

**Trade Management:**
• TP1 Hit → SL moves to Entry
• TP2 Hit → SL moves to TP1
• TP3 Hit → Trade closes fully

**Position Distribution:** 40% TP1 | 35% TP2 | 25% TP3"""
            
            return formatted_message
            
        except Exception as e:
            self.logger.error(f"Error formatting for Cornix: {e}")
            return ""
    
    def validate_cornix_format(self, message: str) -> Dict[str, Any]:
        """Validate if message follows Cornix-compatible format"""
        try:
            validation_result = {
                'valid': False,
                'errors': [],
                'warnings': []
            }
            
            # Check for required Cornix elements
            required_elements = [
                'Channel:',
                'Symbol:',
                'Entry Targets:',
                'Take-Profit Targets:',
                'Stop Targets:',
                'Leverage:',
                'Direction:'
            ]
            
            missing_elements = []
            for element in required_elements:
                if element not in message:
                    missing_elements.append(element)
            
            if missing_elements:
                validation_result['errors'].append(f"Missing required elements: {', '.join(missing_elements)}")
            
            # Check for proper price format (6 decimal places)
            import re
            price_pattern = r'\d+\.\d{6}'
            prices_found = re.findall(price_pattern, message)
            
            if len(prices_found) < 4:  # Entry + 3 TPs + 1 SL = 5 minimum
                validation_result['warnings'].append("Insufficient price targets found")
            
            # Check for leverage format
            leverage_pattern = r'Leverage:\s*(\d+)x'
            leverage_match = re.search(leverage_pattern, message)
            
            if not leverage_match:
                validation_result['errors'].append("Leverage format not found")
            elif int(leverage_match.group(1)) > 125:
                validation_result['warnings'].append("Leverage exceeds maximum (125x)")
            
            # Check for direction
            if not any(direction in message for direction in ['Direction: BUY', 'Direction: SELL']):
                validation_result['errors'].append("Valid direction not found")
            
            # Set validation status
            validation_result['valid'] = len(validation_result['errors']) == 0
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating Cornix format: {e}")
            return {
                'valid': False,
                'errors': [f"Validation error: {str(e)}"],
                'warnings': []
            }
