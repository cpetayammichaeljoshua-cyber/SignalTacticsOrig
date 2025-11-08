"""
Signal parser for trading signals
Handles various signal formats and extracts trading parameters
"""

import re
import logging
from typing import Dict, Any, Optional, List
from decimal import Decimal

class SignalParser:
    """Parser for trading signals with multiple format support"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Common patterns for signal parsing
        self.patterns = {
            'symbol': r'([A-Z]{2,10}(?:USDT|BTC|ETH|BNB)?)',
            'action': r'\b(BUY|SELL|LONG|SHORT)\b',
            'entry': r'(?:ENTRY|ENTER|BUY|SELL)[\s:]*([0-9]+\.?[0-9]*)',
            'stop_loss': r'(?:SL|STOP\s*LOSS|STOP)[\s:]*([0-9]+\.?[0-9]*)',
            'take_profit': r'(?:TP\s*[1-3]?|TAKE\s*PROFIT\s*[1-3]?)[\s:]*([0-9]+\.?[0-9]*)',
            'leverage': r'(?:LEV|LEVERAGE)[\s:]*([0-9]+)',
            'quantity': r'(?:QTY|QUANTITY|SIZE)[\s:]*([0-9]+\.?[0-9]*)'
        }

    def parse_signal(self, signal_text: str) -> Optional[Dict[str, Any]]:
        """Parse trading signal from text"""
        try:
            if not signal_text or not isinstance(signal_text, str):
                return None

            signal_text = signal_text.upper().strip()

            # Extract basic components
            parsed = {
                'raw_text': signal_text,
                'timestamp': None
            }

            # Extract symbol
            symbol_match = re.search(self.patterns['symbol'], signal_text)
            if symbol_match:
                symbol = symbol_match.group(1)
                if not symbol.endswith('USDT') and len(symbol) <= 6:
                    symbol += 'USDT'
                parsed['symbol'] = symbol

            # Extract action
            action_match = re.search(self.patterns['action'], signal_text)
            if action_match:
                parsed['action'] = action_match.group(1)

            # Extract entry price
            entry_match = re.search(self.patterns['entry'], signal_text)
            if entry_match:
                parsed['entry_price'] = float(entry_match.group(1))

            # Extract stop loss
            sl_match = re.search(self.patterns['stop_loss'], signal_text)
            if sl_match:
                parsed['stop_loss'] = float(sl_match.group(1))

            # Extract take profits
            tp_matches = re.findall(r'(?:TP\s*[1-3]?|TAKE\s*PROFIT\s*[1-3]?)[\s:]*([0-9]+\.?[0-9]*)', signal_text)
            if tp_matches:
                for i, tp_value in enumerate(tp_matches[:3], 1):
                    parsed[f'tp{i}'] = float(tp_value)

                # If only one TP provided, use it as take_profit
                if len(tp_matches) == 1:
                    parsed['take_profit'] = float(tp_matches[0])

            # Extract leverage
            lev_match = re.search(self.patterns['leverage'], signal_text)
            if lev_match:
                parsed['leverage'] = int(lev_match.group(1))

            # Extract quantity
            qty_match = re.search(self.patterns['quantity'], signal_text)
            if qty_match:
                parsed['quantity'] = float(qty_match.group(1))

            # Validate parsed signal
            if self._validate_parsed_signal(parsed):
                self.logger.info(f"✅ Signal parsed: {parsed.get('symbol')} {parsed.get('action')}")
                return parsed
            else:
                self.logger.warning(f"⚠️ Invalid signal format: {signal_text[:100]}")
                return None

        except Exception as e:
            self.logger.error(f"❌ Error parsing signal: {e}")
            return None

    def _validate_parsed_signal(self, parsed: Dict[str, Any]) -> bool:
        """Validate parsed signal has minimum required fields"""
        try:
            # Must have symbol and action
            if not parsed.get('symbol') or not parsed.get('action'):
                return False

            # Action must be valid
            if parsed['action'] not in ['BUY', 'SELL', 'LONG', 'SHORT']:
                return False

            # If prices are provided, validate relationships
            entry = parsed.get('entry_price')
            stop_loss = parsed.get('stop_loss')
            take_profit = parsed.get('take_profit')

            if entry and stop_loss and take_profit:
                action = parsed['action']

                if action in ['BUY', 'LONG']:
                    # For long positions: SL < Entry < TP
                    if not (stop_loss < entry < take_profit):
                        return False
                else:
                    # For short positions: TP < Entry < SL
                    if not (take_profit < entry < stop_loss):
                        return False

            return True

        except Exception:
            return False

    def parse_multiple_formats(self, signal_text: str) -> Optional[Dict[str, Any]]:
        """Try parsing with multiple format approaches"""
        # Try standard format first
        result = self.parse_signal(signal_text)
        if result:
            return result

        # Try alternative patterns
        alternative_parsers = [
            self._parse_tradingview_format,
            self._parse_cornix_format,
            self._parse_simple_format
        ]

        for parser in alternative_parsers:
            try:
                result = parser(signal_text)
                if result:
                    return result
            except Exception:
                continue

        return None

    def _parse_tradingview_format(self, signal_text: str) -> Optional[Dict[str, Any]]:
        """Parse TradingView webhook format"""
        try:
            # TradingView JSON format
            import json
            if signal_text.strip().startswith('{'):
                data = json.loads(signal_text)
                return {
                    'symbol': data.get('ticker'),
                    'action': data.get('strategy', {}).get('order_action'),
                    'entry_price': data.get('strategy', {}).get('order_price'),
                    'raw_text': signal_text
                }
        except:
            pass

        return None

    def _parse_cornix_format(self, signal_text: str) -> Optional[Dict[str, Any]]:
        """Parse Cornix signal format"""
        lines = signal_text.strip().split('\n')
        parsed = {'raw_text': signal_text}

        for line in lines:
            line = line.strip().upper()

            if 'PAIR:' in line or 'SYMBOL:' in line:
                symbol = re.search(r'([A-Z]+USDT)', line)
                if symbol:
                    parsed['symbol'] = symbol.group(1)

            elif line.startswith('SIDE:') or line.startswith('DIRECTION:'):
                if 'LONG' in line or 'BUY' in line:
                    parsed['action'] = 'BUY'
                elif 'SHORT' in line or 'SELL' in line:
                    parsed['action'] = 'SELL'

        return parsed if parsed.get('symbol') and parsed.get('action') else None

    def _parse_simple_format(self, signal_text: str) -> Optional[Dict[str, Any]]:
        """Parse simple format: SYMBOL ACTION PRICE"""
        parts = signal_text.strip().upper().split()

        if len(parts) >= 2:
            parsed = {'raw_text': signal_text}

            # First part might be symbol
            if re.match(r'^[A-Z]{2,10}$', parts[0]):
                symbol = parts[0]
                if not symbol.endswith('USDT'):
                    symbol += 'USDT'
                parsed['symbol'] = symbol

            # Second part might be action
            if parts[1] in ['BUY', 'SELL', 'LONG', 'SHORT']:
                parsed['action'] = parts[1]

            # Third part might be price
            if len(parts) > 2:
                try:
                    parsed['entry_price'] = float(parts[2])
                except ValueError:
                    pass

            return parsed if parsed.get('symbol') and parsed.get('action') else None

        return None

    def extract_prices_from_text(self, text: str) -> List[float]:
        """Extract all price-like numbers from text"""
        price_pattern = r'\b\d+\.?\d*\b'
        matches = re.findall(price_pattern, text)

        prices = []
        for match in matches:
            try:
                price = float(match)
                if 0.0001 < price < 1000000:  # Reasonable price range
                    prices.append(price)
            except ValueError:
                continue

        return prices