#!/usr/bin/env python3
"""
Telegram Channel Scanner - Monitors and extracts trading signals from Telegram channels
Integrates with ZEC/USDT trading bot
"""

import asyncio
import aiohttp
import logging
import json
import re
from datetime import datetime
from typing import Dict, Any, Optional, List

class TelegramChannelScanner:
    """Scans Telegram channels for trading signals and trade information"""
    
    def __init__(self, bot_token: str, channel_id: str = "3464978276"):
        """
        Initialize channel scanner
        Args:
            bot_token: Telegram bot token
            channel_id: Channel to scan (can be numeric ID or @username)
        """
        self.bot_token = bot_token
        self.channel_id = channel_id
        self.logger = logging.getLogger(self.__class__.__name__)
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.last_message_id = 0
        self.scanned_messages = set()
        
    async def get_channel_messages(self, limit: int = 50) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch recent messages from the channel
        Args:
            limit: Number of messages to fetch
        Returns:
            List of message dictionaries or None if failed
        """
        try:
            url = f"{self.base_url}/getChatHistory"
            params = {
                "chat_id": self.channel_id,
                "limit": limit
            }
            
            async with aiohttp.ClientSession() as session:
                # Use getUpdates instead - more reliable for channels
                async with session.get(
                    f"{self.base_url}/getUpdates",
                    params={"timeout": 30}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('ok'):
                            return data.get('result', [])
            
            return None
        except Exception as e:
            self.logger.error(f"Error fetching channel messages: {e}")
            return None
    
    def extract_trade_signal(self, message_text: str) -> Optional[Dict[str, Any]]:
        """
        Extract trading information from message text
        Looks for patterns like:
        - LONG/SHORT entries
        - Price levels (entry, SL, TP)
        - Symbol references
        - Leverage mentions
        """
        if not message_text:
            return None
        
        signal = {
            'raw_text': message_text,
            'timestamp': datetime.now(),
            'direction': None,
            'entry_price': None,
            'stop_loss': None,
            'take_profit': None,
            'leverage': None,
            'symbol': 'ZECUSDT',
            'confidence': 0.75,  # Default confidence
            'source': 'telegram_channel'
        }
        
        text_lower = message_text.lower()
        
        # Detect direction
        if any(keyword in text_lower for keyword in ['long', 'buy', 'bull', 'going long', 'entry long']):
            signal['direction'] = 'LONG'
        elif any(keyword in text_lower for keyword in ['short', 'sell', 'bear', 'going short', 'entry short']):
            signal['direction'] = 'SHORT'
        
        # Extract prices using regex patterns
        price_pattern = r'(?:entry|price|@)[\s:]*([0-9]+\.?[0-9]*)'
        prices = re.findall(price_pattern, text_lower)
        
        if prices:
            signal['entry_price'] = float(prices[0])
        
        # Extract stop loss
        sl_pattern = r'(?:sl|stop.*loss|stoploss)[\s:]*([0-9]+\.?[0-9]*)'
        sl_match = re.search(sl_pattern, text_lower)
        if sl_match:
            signal['stop_loss'] = float(sl_match.group(1))
        
        # Extract take profit
        tp_pattern = r'(?:tp|take.*profit|takeprofit|target)[\s:]*([0-9]+\.?[0-9]*)'
        tp_match = re.search(tp_pattern, text_lower)
        if tp_match:
            signal['take_profit'] = float(tp_match.group(1))
        
        # Extract leverage
        lev_pattern = r'(?:leverage|lev)[\s:]*([0-9]+)x?'
        lev_match = re.search(lev_pattern, text_lower)
        if lev_match:
            signal['leverage'] = int(lev_match.group(1))
        
        # Check for symbol mentions
        symbol_pattern = r'(BTC|ETH|ZEC|BNB|SOL|XRP|ADA|DOGE)(?:/|USDT)?'
        symbol_matches = re.findall(symbol_pattern, text_lower.upper())
        if symbol_matches:
            signal['symbol'] = f"{symbol_matches[0]}USDT"
        
        # Extract confidence or strength if mentioned
        confidence_pattern = r'(?:confidence|strength|signal.*strength)[\s:]*([0-9]+)%?'
        conf_match = re.search(confidence_pattern, text_lower)
        if conf_match:
            conf_value = int(conf_match.group(1))
            signal['confidence'] = conf_value / 100 if conf_value <= 100 else conf_value / 10000
        
        # Only return signal if we found at least a direction
        if signal['direction']:
            return signal
        
        return None
    
    async def scan_for_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Scan channel for trading signals
        Args:
            limit: Number of messages to scan
        Returns:
            List of extracted trade signals
        """
        try:
            self.logger.info(f"📡 Scanning channel {self.channel_id} for trading signals...")
            
            messages = await self.get_channel_messages(limit)
            if not messages:
                self.logger.warning("No messages retrieved from channel")
                return []
            
            signals = []
            
            for update in messages:
                try:
                    # Extract message from update
                    message = update.get('message') or update.get('edited_message')
                    if not message:
                        continue
                    
                    message_id = message.get('message_id')
                    
                    # Skip if already processed
                    if message_id in self.scanned_messages:
                        continue
                    
                    # Extract text
                    text = message.get('text', '')
                    caption = message.get('caption', '')
                    full_text = f"{text} {caption}".strip()
                    
                    if not full_text:
                        continue
                    
                    # Extract trade signal
                    signal = self.extract_trade_signal(full_text)
                    if signal:
                        signal['message_id'] = message_id
                        signal['from_user'] = message.get('from', {}).get('username', 'unknown')
                        signals.append(signal)
                        self.scanned_messages.add(message_id)
                        
                        self.logger.info(f"✅ Found trade signal: {signal['direction']} {signal['symbol']} @ {signal['entry_price']}")
                
                except Exception as e:
                    self.logger.debug(f"Error processing message: {e}")
                    continue
            
            self.logger.info(f"📊 Extracted {len(signals)} trading signals from channel")
            return signals
        
        except Exception as e:
            self.logger.error(f"Error scanning channel: {e}")
            return []
    
    async def get_latest_signal(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent trading signal from channel
        Returns:
            Most recent signal or None
        """
        signals = await self.scan_for_trades(limit=10)
        if signals:
            # Return the most recent signal
            return signals[-1]
        return None
    
    async def continuous_monitor(self, callback=None, interval: int = 60):
        """
        Continuously monitor channel for new signals
        Args:
            callback: Async function to call when new signal found
            interval: Scanning interval in seconds
        """
        self.logger.info(f"📡 Starting continuous channel monitoring (every {interval}s)")
        
        try:
            while True:
                try:
                    signals = await self.scan_for_trades(limit=20)
                    
                    if signals and callback:
                        for signal in signals:
                            try:
                                await callback(signal)
                            except Exception as e:
                                self.logger.error(f"Error in callback: {e}")
                    
                    await asyncio.sleep(interval)
                
                except Exception as e:
                    self.logger.error(f"Error in monitoring cycle: {e}")
                    await asyncio.sleep(interval)
        
        except KeyboardInterrupt:
            self.logger.info("👋 Channel monitoring stopped")
    
    async def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Validate extracted signal for trading
        Returns:
            True if signal is valid for trading, False otherwise
        """
        # Check required fields
        required_fields = ['direction', 'entry_price', 'symbol']
        for field in required_fields:
            if not signal.get(field):
                self.logger.warning(f"Signal missing required field: {field}")
                return False
        
        # Validate price values
        if signal['entry_price'] <= 0:
            return False
        
        if signal.get('stop_loss') and signal['stop_loss'] <= 0:
            return False
        
        if signal.get('take_profit') and signal['take_profit'] <= 0:
            return False
        
        # Validate direction
        if signal['direction'] not in ['LONG', 'SHORT']:
            return False
        
        # Check confidence threshold (default 75%)
        if signal.get('confidence', 1.0) < 0.75:
            self.logger.warning(f"Signal confidence {signal.get('confidence', 1.0):.1%} below threshold")
            return False
        
        return True
    
    def format_signal_for_trading(self, signal: Dict[str, Any]) -> str:
        """
        Format extracted signal into trading message
        Returns:
            Formatted signal string for display/trading
        """
        msg = f"""
📊 **CHANNEL SIGNAL: {signal['direction']} {signal['symbol']}**

💰 **Entry:** {signal['entry_price']:.5f}
"""
        
        if signal.get('stop_loss'):
            msg += f"🛑 **SL:** {signal['stop_loss']:.5f}\n"
        
        if signal.get('take_profit'):
            msg += f"🎯 **TP:** {signal['take_profit']:.5f}\n"
        
        if signal.get('leverage'):
            msg += f"📈 **Leverage:** {signal['leverage']}x\n"
        
        msg += f"💪 **Confidence:** {signal.get('confidence', 0.75):.0%}\n"
        msg += f"👤 **Source:** {signal.get('from_user', 'Channel Signal')}\n"
        
        return msg.strip()
