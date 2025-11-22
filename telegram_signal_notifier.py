#!/usr/bin/env python3
"""
Advanced Telegram Signal Notifier
Sends comprehensive trading signals to Telegram with rich formatting
"""

import os
import logging
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional
import aiohttp


class TelegramSignalNotifier:
    """Send formatted trading signals to Telegram"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
        
        if not self.bot_token:
            self.logger.warning("⚠️ TELEGRAM_BOT_TOKEN not configured")
            self.logger.warning("💡 Set it in Replit Secrets to enable Telegram notifications")
        elif not self.chat_id:
            self.logger.warning("⚠️ TELEGRAM_CHAT_ID not configured")
            self.logger.warning("💡 Set it in Replit Secrets (your chat ID or @channelname)")
        else:
            self.logger.info("✅ Telegram notifier initialized")
            self.logger.info(f"📱 Bot token configured: {self.bot_token[:10]}...")
            self.logger.info(f"💬 Chat ID: {self.chat_id}")
    
    async def test_connection(self) -> bool:
        """Test Telegram bot connection"""
        try:
            if not self.bot_token:
                self.logger.error("❌ Cannot test - bot token not configured")
                return False
                
            url = f"https://api.telegram.org/bot{self.bot_token}/getMe"
            
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('ok'):
                            bot_info = data.get('result', {})
                            bot_name = bot_info.get('username', 'Unknown')
                            self.logger.info(f"✅ Telegram bot connected: @{bot_name}")
                            return True
                        else:
                            self.logger.error(f"❌ Bot response not OK")
                            return False
                    else:
                        error_text = await response.text()
                        self.logger.error(f"❌ Bot token test failed: {error_text}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"❌ Connection test error: {e}")
            return False
    
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
                self.logger.error("💡 Please set TELEGRAM_BOT_TOKEN in Replit Secrets")
                return False
                
            if not self.chat_id:
                self.logger.error("❌ TELEGRAM_CHAT_ID not configured in environment")
                self.logger.error("💡 Please set TELEGRAM_CHAT_ID in Replit Secrets (use your chat ID or @channelname)")
                return False
            
            # Format signal message
            message = self._format_signal_message(signal)
            
            # Log the message being sent
            self.logger.info(f"📤 Sending signal to Telegram chat: {self.chat_id}")
            self.logger.debug(f"Message preview: {message[:100]}...")
            
            # Send to Telegram with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    success = await self._send_telegram_message(message)
                    
                    if success:
                        self.logger.info(f"✅ Telegram signal sent successfully for {self._get_symbol(signal)} (attempt {attempt + 1})")
                        return True
                    else:
                        if attempt < max_retries - 1:
                            self.logger.warning(f"⚠️ Send failed, retrying... ({attempt + 1}/{max_retries})")
                            await asyncio.sleep(1)
                        else:
                            self.logger.error(f"❌ Failed to send Telegram signal after {max_retries} attempts")
                            return False
                            
                except Exception as retry_error:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"⚠️ Retry error: {retry_error}, attempting again...")
                        await asyncio.sleep(1)
                    else:
                        raise
            
            return False
            
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
        """Format signal as rich Telegram message matching professional format"""
        
        # Extract signal data (supports both object and dict)
        symbol = self._get_symbol(signal)
        direction = self._get_attr(signal, 'direction', 'UNKNOWN')
        entry_price = self._get_attr(signal, 'entry_price', 0)
        stop_loss = self._get_attr(signal, 'stop_loss', 0)
        tp1 = self._get_attr(signal, 'take_profit_1', 0)
        tp2 = self._get_attr(signal, 'take_profit_2', 0)
        tp3 = self._get_attr(signal, 'take_profit_3', 0)
        leverage = self._get_attr(signal, 'leverage', 20)
        signal_strength = self._get_attr(signal, 'signal_strength', 0)
        consensus_confidence = self._get_attr(signal, 'consensus_confidence', 0)
        strategies_agree = self._get_attr(signal, 'strategies_agree', 0)
        total_strategies = self._get_attr(signal, 'total_strategies', 0)
        risk_reward = self._get_attr(signal, 'risk_reward_ratio', 0)
        timeframe = self._get_attr(signal, 'timeframe', '1m')
        
        # Get strategy breakdown
        strategy_votes = self._get_attr(signal, 'strategy_votes', {})
        strategy_scores = self._get_attr(signal, 'strategy_scores', {})
        
        # Determine dominant strategy (highest scoring strategy that agrees with direction)
        dominant_strategy = self._get_dominant_strategy(strategy_votes, strategy_scores, direction)
        
        # Calculate ATR value (approximate from stop loss distance)
        atr_value = abs(entry_price - stop_loss) if entry_price and stop_loss else 0
        
        # Calculate SL/TP percentages
        sl_pct = self._calc_pct(entry_price, stop_loss, direction)
        tp_pct = self._calc_pct(entry_price, tp1, direction)
        
        # Format timestamp - exact format from image
        timestamp = datetime.now().strftime('%Y-%m-%d\n%H:%M:%S UTC')
        
        # Direction-specific formatting - OFFICIAL CORNIX FORMAT
        action = "Long" if direction == "LONG" else "Short"
        
        # Convert symbol format: ETH/USDT:USDT -> ETH/USDT (OFFICIAL CORNIX FORMAT)
        # Cornix expects "BTC/USDT", "ETH/USDT", etc. NOT .P suffix
        if '/' in symbol:
            # Extract base and quote: ETH/USDT:USDT -> ETH/USDT
            parts = symbol.split(':')
            cornix_symbol = parts[0] if parts else symbol
        else:
            # Construct from symbol
            base_symbol = symbol.replace('USDT', '').replace(':',  '')
            cornix_symbol = f"{base_symbol}/USDT"
        
        # Build STRICTLY FORMATTED message matching OFFICIAL CORNIX SPECIFICATION
        # Strategy details section (with formatting for readability)
        strategy_section = [
            f"🎯 *{dominant_strategy}* Multi-TF Enhanced",
            f"• Conversion/Base: 4/4 periods",
            f"• LaggingSpan2/Displacement: 46/20 periods",
            f"• EMA Filter: 200 periods",
            f"• SL/TP Percent: {sl_pct:.2f}%/{tp_pct:.2f}%",
            f"",
            f"📊 *SIGNAL ANALYSIS:*",
            f"• Strength: {signal_strength:.1f}%",
            f"• Confidence: {consensus_confidence:.1f}%",
            f"• Risk/Reward: 1:{risk_reward:.2f}",
            f"• ATR Value: {atr_value:.6f}",
            f"• Scan Mode: Multi-Timeframe Enhanced",
        ]
        
        # CORNIX SECTION - OFFICIAL SPECIFICATION FORMAT
        # Format: Pair, Direction, Entry, Targets, Stop Loss, Leverage
        cornix_section = [
            f"",
            f"🎯 CORNIX SIGNAL:",
            f"{cornix_symbol}",
            f"{action}",
            f"Leverage: {leverage}x",
            f"",
            f"Entry: {entry_price:.5f}",
        ]
        
        # Add Take Profit targets (Cornix official format: "Target 1:", "Target 2:", etc.)
        target_num = 1
        if tp1 > 0:
            cornix_section.append(f"Target {target_num}: {tp1:.5f}")
            target_num += 1
        if tp2 > 0:
            cornix_section.append(f"Target {target_num}: {tp2:.5f}")
            target_num += 1
        if tp3 > 0:
            cornix_section.append(f"Target {target_num}: {tp3:.5f}")
        
        # Add Stop Loss (Cornix official format)
        cornix_section.append(f"")
        cornix_section.append(f"Stop Loss: {stop_loss:.5f}")
        
        # Footer section
        footer_section = [
            f"",
            f"🕐 *Signal Time:* {timestamp}",
            f"🤖 *Bot:* Pine Script {dominant_strategy} v6",
            f"",
            f"Cross Margin & Auto Leverage",
            f"- Comprehensive Risk Management"
        ]
        
        # Combine all sections
        lines = strategy_section + cornix_section + footer_section
        
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
            
            # Truncate message if too long (Telegram limit is 4096 characters)
            if len(message) > 4000:
                message = message[:3900] + "\n\n... (message truncated)"
            
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True
            }
            
            self.logger.info(f"📤 Sending to Telegram chat: {self.chat_id}")
            self.logger.debug(f"URL: {url}")
            self.logger.debug(f"Message length: {len(message)} characters")
            
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    response_text = await response.text()
                    
                    if response.status == 200:
                        self.logger.info("✅ Telegram message sent successfully")
                        self.logger.debug(f"Response: {response_text}")
                        return True
                    else:
                        self.logger.error(f"❌ Telegram API error {response.status}")
                        self.logger.error(f"Response: {response_text}")
                        
                        # Parse error for better debugging
                        try:
                            error_json = json.loads(response_text)
                            error_description = error_json.get('description', 'Unknown error')
                            self.logger.error(f"Error description: {error_description}")
                            
                            # Specific error handling
                            if 'chat not found' in error_description.lower():
                                self.logger.error("💡 Chat ID is incorrect. Make sure to use your personal chat ID or @channelname")
                            elif 'bot was blocked' in error_description.lower():
                                self.logger.error("💡 Bot was blocked by the user. Unblock the bot in Telegram")
                            elif 'unauthorized' in error_description.lower():
                                self.logger.error("💡 Bot token is invalid. Check TELEGRAM_BOT_TOKEN in Replit Secrets")
                                
                        except:
                            pass
                        
                        return False
            
        except aiohttp.ClientError as e:
            self.logger.error(f"❌ Network error sending to Telegram: {e}")
            self.logger.error("💡 Check your internet connection or try again later")
            return False
        except asyncio.TimeoutError:
            self.logger.error(f"❌ Timeout sending to Telegram (30s)")
            self.logger.error("💡 Telegram API may be slow or unavailable")
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
    
    def _get_dominant_strategy(self, strategy_votes: Dict[str, str], strategy_scores: Dict[str, float], direction: str) -> str:
        """Get the dominant strategy name based on votes and scores"""
        
        # Strategy name mapping to display names
        strategy_names = {
            'ultimate_scalping': 'Ichimoku Sniper',
            'lightning_scalping': 'Lightning Scalper',
            'momentum_scalping': 'Momentum Sniper',
            'volume_breakout': 'Volume Breakout',
            'ichimoku_sniper': 'Ichimoku Sniper',
            'market_intelligence': 'Market Intelligence'
        }
        
        # Find strategies that agree with the consensus direction
        agreeing_strategies = {}
        for strategy, vote in strategy_votes.items():
            if vote == direction or (vote == 'BUY' and direction == 'LONG') or (vote == 'SELL' and direction == 'SHORT'):
                score = strategy_scores.get(strategy, 0)
                agreeing_strategies[strategy] = score
        
        # Return the highest scoring agreeing strategy
        if agreeing_strategies:
            dominant = max(agreeing_strategies, key=lambda x: agreeing_strategies[x])
            return strategy_names.get(dominant, 'Ichimoku Sniper')
        
        # Default to Ichimoku Sniper if no agreeing strategies found
        return 'Ichimoku Sniper'
    
    def _format_cornix_signal(self, signal: Any) -> str:
        """Format signal in Cornix-compatible format"""
        symbol = self._get_symbol(signal)
        direction = self._get_attr(signal, 'direction', 'UNKNOWN')
        entry_price = self._get_attr(signal, 'entry_price', 0)
        stop_loss = self._get_attr(signal, 'stop_loss', 0)
        tp1 = self._get_attr(signal, 'take_profit_1', 0)
        tp2 = self._get_attr(signal, 'take_profit_2', 0)
        tp3 = self._get_attr(signal, 'take_profit_3', 0)
        leverage = self._get_attr(signal, 'leverage', 20)
        
        action = "BUY" if direction == "LONG" else "SELL"
        
        # Cornix format
        cornix_lines = [
            f"{symbol} {action}",
            f"Entry: {entry_price:.5f}",
            f"SL: {stop_loss:.5f}",
            f"TP1: {tp1:.5f}",
            f"TP2: {tp2:.5f}",
            f"TP3: {tp3:.5f}",
            f"Leverage: {leverage}x",
            f"Margin: CROSS"
        ]
        
        return "\n".join(cornix_lines)


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
