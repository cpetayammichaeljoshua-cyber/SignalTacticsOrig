"""
Enhanced Cornix Integration with comprehensive error handling
Handles Cornix webhook forwarding with fallback mechanisms
"""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

class EnhancedCornixIntegration:
    """Enhanced Cornix integration with robust error handling"""

    def __init__(self, webhook_url: str = "", api_key: str = ""):
        self.webhook_url = webhook_url
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)

        # Fallback to logging mode if no webhook configured
        self.logging_mode = not bool(webhook_url)

        if self.logging_mode:
            self.logger.info("🔧 Cornix integration running in LOGGING mode")
        else:
            self.logger.info("🌐 Cornix integration configured with webhook")

    async def test_connection(self) -> Dict[str, Any]:
        """Test Cornix webhook connection"""
        try:
            if self.logging_mode:
                return {'success': True, 'mode': 'logging', 'message': 'Logging mode active'}

            # Test webhook with ping
            test_payload = {
                'action': 'ping',
                'timestamp': datetime.now().isoformat(),
                'source': 'trading_bot_test'
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=test_payload,
                    headers={'Content-Type': 'application/json'},
                    timeout=10
                ) as response:
                    if response.status in [200, 201, 202]:
                        return {'success': True, 'mode': 'webhook', 'status': response.status}
                    else:
                        self.logger.warning(f"Cornix webhook test returned {response.status}")
                        return {'success': False, 'status': response.status}

        except Exception as e:
            self.logger.error(f"Cornix connection test failed: {e}")
            return {'success': False, 'error': str(e)}

    async def send_initial_signal(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send initial trading signal to Cornix"""
        try:
            # Format signal for Cornix
            cornix_payload = self._format_cornix_signal(signal_data)

            if self.logging_mode:
                self.logger.info(f"📝 CORNIX LOG: Initial signal - {signal_data.get('symbol')} {signal_data.get('action')}")
                self._log_signal_details(cornix_payload)
                return {'success': True, 'mode': 'logged'}

            # Send to Cornix webhook
            return await self._send_webhook(cornix_payload, "initial_signal")

        except Exception as e:
            self.logger.error(f"Error sending initial signal: {e}")
            return {'success': False, 'error': str(e)}

    async def send_advanced_signal(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send advanced trading signal with enhanced formatting"""
        try:
            # Enhanced signal formatting
            enhanced_payload = self._format_advanced_signal(signal_data)

            if self.logging_mode:
                self.logger.info(f"📝 CORNIX LOG: Advanced signal - {signal_data.get('symbol')}")
                self._log_signal_details(enhanced_payload)
                return {'success': True, 'mode': 'logged'}

            return await self._send_webhook(enhanced_payload, "advanced_signal")

        except Exception as e:
            self.logger.error(f"Error sending advanced signal: {e}")
            return {'success': False, 'error': str(e)}

    async def update_stop_loss(self, symbol: str, new_sl: float, reason: str) -> bool:
        """Update stop loss for existing position"""
        try:
            update_payload = {
                'action': 'update_stop_loss',
                'symbol': symbol,
                'new_stop_loss': new_sl,
                'reason': reason,
                'timestamp': datetime.now().isoformat()
            }

            if self.logging_mode:
                self.logger.info(f"📝 CORNIX LOG: SL Update - {symbol} → {new_sl} ({reason})")
                return True

            result = await self._send_webhook(update_payload, "stop_loss_update")
            return result.get('success', False)

        except Exception as e:
            self.logger.error(f"Error updating stop loss: {e}")
            return False

    async def close_position(self, symbol: str, reason: str, percentage: int = 100) -> bool:
        """Close position (partial or full)"""
        try:
            close_payload = {
                'action': 'close_position',
                'symbol': symbol,
                'percentage': percentage,
                'reason': reason,
                'timestamp': datetime.now().isoformat()
            }

            if self.logging_mode:
                self.logger.info(f"📝 CORNIX LOG: Close Position - {symbol} {percentage}% ({reason})")
                return True

            result = await self._send_webhook(close_payload, "position_close")
            return result.get('success', False)

        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return False

    def _format_cornix_signal(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format signal data for Cornix"""
        return {
            'action': 'signal',
            'symbol': signal_data.get('symbol'),
            'side': signal_data.get('action', '').lower(),
            'entry': signal_data.get('entry_price'),
            'stop_loss': signal_data.get('stop_loss'),
            'take_profits': [
                signal_data.get('tp1'),
                signal_data.get('tp2'),
                signal_data.get('tp3')
            ],
            'leverage': signal_data.get('leverage', 50),
            'quantity': signal_data.get('quantity'),
            'timestamp': datetime.now().isoformat(),
            'source': 'enhanced_trading_bot'
        }

    def _format_advanced_signal(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format advanced signal with additional metadata"""
        base_signal = self._format_cornix_signal(signal_data)

        # Add advanced fields
        base_signal.update({
            'signal_strength': signal_data.get('signal_strength', 0),
            'strategy': signal_data.get('strategy', 'enhanced_scalping'),
            'timeframe': signal_data.get('timeframe', 'multi'),
            'risk_reward_ratio': signal_data.get('risk_reward_ratio', 3.0),
            'market_conditions': signal_data.get('market_conditions', {}),
            'ml_enhanced': signal_data.get('ml_enhanced', False)
        })

        return base_signal

    async def _send_webhook(self, payload: Dict[str, Any], action_type: str) -> Dict[str, Any]:
        """Send webhook request to Cornix"""
        try:
            headers = {'Content-Type': 'application/json'}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers=headers,
                    timeout=15
                ) as response:
                    if response.status in [200, 201, 202]:
                        self.logger.info(f"✅ Cornix {action_type} sent successfully")
                        return {'success': True, 'status': response.status}
                    else:
                        error_text = await response.text()
                        self.logger.error(f"❌ Cornix {action_type} failed: {response.status} - {error_text}")
                        return {'success': False, 'status': response.status, 'error': error_text}

        except asyncio.TimeoutError:
            self.logger.error(f"⏰ Cornix {action_type} timeout")
            return {'success': False, 'error': 'timeout'}
        except Exception as e:
            self.logger.error(f"❌ Cornix {action_type} error: {e}")
            return {'success': False, 'error': str(e)}

    def _log_signal_details(self, payload: Dict[str, Any]):
        """Log signal details in structured format"""
        try:
            self.logger.info("=" * 50)
            self.logger.info("📊 CORNIX SIGNAL DETAILS:")
            self.logger.info(f"Symbol: {payload.get('symbol')}")
            self.logger.info(f"Action: {payload.get('side', '').upper()}")
            self.logger.info(f"Entry: {payload.get('entry')}")
            self.logger.info(f"Stop Loss: {payload.get('stop_loss')}")

            tps = payload.get('take_profits', [])
            for i, tp in enumerate(tps, 1):
                if tp:
                    self.logger.info(f"TP{i}: {tp}")

            self.logger.info(f"Leverage: {payload.get('leverage')}x")
            self.logger.info(f"Timestamp: {payload.get('timestamp')}")
            self.logger.info("=" * 50)

        except Exception as e:
            self.logger.error(f"Error logging signal details: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get integration status"""
        return {
            'configured': bool(self.webhook_url),
            'mode': 'webhook' if self.webhook_url else 'logging',
            'webhook_url': self.webhook_url[:50] + "..." if len(self.webhook_url) > 50 else self.webhook_url,
            'api_key_configured': bool(self.api_key)
        }