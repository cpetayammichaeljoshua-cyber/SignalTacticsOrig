#!/usr/bin/env python3
"""
Advanced Telegram Signal Notifier
Sends comprehensive trading signals to Telegram with rich formatting
"""

import os
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
import aiohttp


class TelegramSignalNotifier:
    """Send formatted trading signals to Telegram"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
        
        if not self.bot_token or not self.chat_id:
            self.logger.warning("⚠️ Telegram credentials not configured")
        else:
            self.logger.info("✅ Telegram notifier initialized")
    
    async def send_signal(self, signal: Any) -> bool:
        """
        Send trading signal to Telegram
        
        Args:
            signal: HighFrequencySignal object or dict
            
        Returns:
            bool: Success status
        """
        try:
            if not self.bot_token:
                self.logger.error("❌ TELEGRAM_BOT_TOKEN not configured in environment")
                return False
                
            if not self.chat_id:
                self.logger.error("❌ TELEGRAM_CHAT_ID not configured in environment")
                return False
            
            # Format signal message
            message = self._format_signal_message(signal)
            
            # Send to Telegram
            success = await self._send_telegram_message(message)
            
            if success:
                self.logger.info(f"✅ Telegram signal sent for {self._get_symbol(signal)}")
            else:
                self.logger.error(f"❌ Failed to send Telegram signal for {self._get_symbol(signal)}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error sending Telegram signal: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    async def send_position_update(self, symbol: str, update_type: str, details: Dict[str, Any]) -> bool:
        """
        Send position update notification
        
        Args:
            symbol: Trading pair
            update_type: 'OPENED', 'CLOSED', 'TP_HIT', 'SL_HIT', 'MODIFIED'
            details: Update details
        """
        try:
            if not self.bot_token or not self.chat_id:
                return False
            
            message = self._format_position_update(symbol, update_type, details)
            return await self._send_telegram_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending position update: {e}")
            return False
    
    async def send_performance_summary(self, summary: Dict[str, Any]) -> bool:
        """Send daily/session performance summary"""
        try:
            if not self.bot_token or not self.chat_id:
                return False
            
            message = self._format_performance_summary(summary)
            return await self._send_telegram_message(message)
            
        except Exception as e:
            self.logger.error(f"Error sending performance summary: {e}")
            return False
    
    def _format_signal_message(self, signal: Any) -> str:
        """Format signal as rich Telegram message"""
        
        # Extract signal data (supports both object and dict)
        symbol = self._get_symbol(signal)
        direction = self._get_attr(signal, 'direction', 'UNKNOWN')
        entry_price = self._get_attr(signal, 'entry_price', 0)
        stop_loss = self._get_attr(signal, 'stop_loss', 0)
        tp1 = self._get_attr(signal, 'take_profit_1', 0)
        tp2 = self._get_attr(signal, 'take_profit_2', 0)
        tp3 = self._get_attr(signal, 'take_profit_3', 0)
        leverage = self._get_attr(signal, 'leverage', 0)
        signal_strength = self._get_attr(signal, 'signal_strength', 0)
        consensus_confidence = self._get_attr(signal, 'consensus_confidence', 0)
        strategies_agree = self._get_attr(signal, 'strategies_agree', 0)
        total_strategies = self._get_attr(signal, 'total_strategies', 0)
        risk_reward = self._get_attr(signal, 'risk_reward_ratio', 0)
        timeframe = self._get_attr(signal, 'timeframe', '1m')
        
        # Emoji for direction
        direction_emoji = "🟢" if direction == "LONG" else "🔴"
        
        # Build message
        lines = [
            f"{'='*40}",
            f"{direction_emoji} **HIGH-FREQUENCY SCALPING SIGNAL** {direction_emoji}",
            f"{'='*40}",
            f"",
            f"📊 **Market:** `{symbol}`",
            f"📈 **Direction:** **{direction}**",
            f"⏱️ **Timeframe:** {timeframe}",
            f"",
            f"💰 **ENTRY ZONE**",
            f"└─ Entry: `${entry_price:.4f}`",
            f"",
            f"🎯 **TAKE PROFIT TARGETS**",
            f"└─ TP1: `${tp1:.4f}` ({self._calc_pct(entry_price, tp1, direction):.2f}%)",
            f"└─ TP2: `${tp2:.4f}` ({self._calc_pct(entry_price, tp2, direction):.2f}%)",
            f"└─ TP3: `${tp3:.4f}` ({self._calc_pct(entry_price, tp3, direction):.2f}%)",
            f"",
            f"🛑 **STOP LOSS**",
            f"└─ SL: `${stop_loss:.4f}` ({self._calc_pct(entry_price, stop_loss, direction):.2f}%)",
            f"",
            f"⚡ **POSITION PARAMETERS**",
            f"└─ Leverage: **{leverage}x**",
            f"└─ Risk/Reward: **1:{risk_reward:.2f}**",
            f"",
            f"📊 **SIGNAL QUALITY**",
            f"└─ Strength: **{signal_strength:.1f}%**",
            f"└─ Consensus: **{consensus_confidence:.1f}%** ({strategies_agree}/{total_strategies} strategies)",
            f"",
            f"⏰ **Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"",
            f"{'='*40}",
            f"🤖 **High-Frequency Scalping System**",
            f"{'='*40}"
        ]
        
        return "\n".join(lines)
    
    def _format_position_update(self, symbol: str, update_type: str, details: Dict[str, Any]) -> str:
        """Format position update message"""
        
        emoji_map = {
            'OPENED': '🟢',
            'CLOSED': '⚫',
            'TP_HIT': '🎯',
            'SL_HIT': '🛑',
            'MODIFIED': '🔄',
            'PROFIT': '💰',
            'LOSS': '⚠️'
        }
        
        emoji = emoji_map.get(update_type, '📊')
        
        lines = [
            f"{emoji} **POSITION UPDATE: {update_type}** {emoji}",
            f"",
            f"📊 **Market:** `{symbol}`"
        ]
        
        # Add details
        for key, value in details.items():
            if key == 'pnl':
                pnl_emoji = "💰" if value > 0 else "⚠️"
                lines.append(f"{pnl_emoji} **PnL:** `${value:.2f}`")
            elif key == 'pnl_pct':
                lines.append(f"📈 **PnL %:** `{value:.2f}%`")
            elif key == 'price':
                lines.append(f"💵 **Price:** `${value:.4f}`")
            elif key == 'direction':
                lines.append(f"📈 **Direction:** **{value}**")
            elif key == 'reason':
                lines.append(f"💡 **Reason:** {value}")
        
        lines.append(f"⏰ **Time:** {datetime.now().strftime('%H:%M:%S UTC')}")
        
        return "\n".join(lines)
    
    def _format_performance_summary(self, summary: Dict[str, Any]) -> str:
        """Format performance summary"""
        
        lines = [
            f"📊 **PERFORMANCE SUMMARY** 📊",
            f"{'='*40}",
            f"",
            f"💼 **Trading Statistics**",
            f"└─ Total Signals: {summary.get('total_signals', 0)}",
            f"└─ Executed: {summary.get('executed', 0)}",
            f"└─ Win Rate: {summary.get('win_rate', 0):.1f}%",
            f"",
            f"💰 **Financial**",
            f"└─ Total PnL: ${summary.get('total_pnl', 0):.2f}",
            f"└─ Best Trade: ${summary.get('best_trade', 0):.2f}",
            f"└─ Worst Trade: ${summary.get('worst_trade', 0):.2f}",
            f"",
            f"⏰ **Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"{'='*40}"
        ]
        
        return "\n".join(lines)
    
    async def _send_telegram_message(self, message: str) -> bool:
        """Send message via Telegram Bot API"""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True
            }
            
            self.logger.info(f"📤 Sending to Telegram chat: {self.chat_id}")
            
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        self.logger.info("✅ Telegram message sent successfully")
                        return True
                    else:
                        error_text = await response.text()
                        self.logger.error(f"❌ Telegram API error {response.status}: {error_text}")
                        return False
            
        except aiohttp.ClientError as e:
            self.logger.error(f"❌ Network error sending to Telegram: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Unexpected error sending Telegram message: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _get_symbol(self, signal: Any) -> str:
        """Extract symbol from signal"""
        return self._get_attr(signal, 'symbol', 'UNKNOWN')
    
    def _get_attr(self, obj: Any, attr: str, default: Any = None) -> Any:
        """Get attribute from object or dict"""
        if hasattr(obj, attr):
            return getattr(obj, attr)
        elif isinstance(obj, dict):
            return obj.get(attr, default)
        return default
    
    def _calc_pct(self, entry: float, target: float, direction: str) -> float:
        """Calculate percentage change"""
        if entry == 0:
            return 0.0
        
        pct = ((target - entry) / entry) * 100
        
        # For SHORT positions, invert the sign
        if direction == "SHORT":
            pct = -pct
        
        return abs(pct)


async def test_telegram_notifier():
    """Test the Telegram notifier"""
    print("\n" + "="*80)
    print("🧪 TESTING TELEGRAM SIGNAL NOTIFIER")
    print("="*80)
    
    notifier = TelegramSignalNotifier()
    
    # Create test signal
    test_signal = {
        'symbol': 'ETH/USDT:USDT',
        'direction': 'LONG',
        'entry_price': 3500.00,
        'stop_loss': 3482.50,
        'take_profit_1': 3528.00,
        'take_profit_2': 3542.00,
        'take_profit_3': 3563.00,
        'leverage': 20,
        'signal_strength': 85.5,
        'consensus_confidence': 75.0,
        'strategies_agree': 4,
        'total_strategies': 5,
        'risk_reward_ratio': 2.4,
        'timeframe': '1m'
    }
    
    print("\n📤 Sending test signal...")
    success = await notifier.send_signal(test_signal)
    
    if success:
        print("✅ Test signal sent successfully!")
    else:
        print("❌ Failed to send test signal")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_telegram_notifier())
