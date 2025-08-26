
#!/usr/bin/env python3
"""
Telegram Closed Trades Scanner
Specifically designed to scan for closed/completed trades from Telegram channels
and feed the results to ML training
"""

import asyncio
import logging
import json
import re
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

class TelegramClosedTradesScanner:
    """Scanner for closed trades from Telegram channels"""
    
    def __init__(self, bot_token: str, channel_username: str):
        self.bot_token = bot_token
        self.channel_username = channel_username
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.logger = logging.getLogger(__name__)
        
        # Database for storing scanned closed trades
        self.db_path = "closed_trades_scanner.db"
        self._initialize_database()
        
        # Patterns for detecting closed trades
        self.closed_trade_patterns = {
            'closed_keywords': [
                'closed', 'tp1 hit', 'tp2 hit', 'tp3 hit', 'target reached',
                'stop loss hit', 'sl hit', 'trade closed', 'position closed',
                'profit taken', 'loss taken', 'exit', 'completed', 'finished'
            ],
            'profit_patterns': [
                r'profit[:\s]*([+-]?\d+\.?\d*)%',
                r'([+-]?\d+\.?\d*)%\s*profit',
                r'gain[:\s]*([+-]?\d+\.?\d*)%',
                r'([+-]?\d+\.?\d*)%\s*gain'
            ],
            'loss_patterns': [
                r'loss[:\s]*([+-]?\d+\.?\d*)%',
                r'([+-]?\d+\.?\d*)%\s*loss',
                r'sl\s+hit[:\s]*([+-]?\d+\.?\d*)%',
                r'stop\s+loss[:\s]*([+-]?\d+\.?\d*)%'
            ],
            'symbol_pattern': r'#?(\w+USDT?)\s+',
            'direction_pattern': r'(LONG|SHORT|BUY|SELL)',
            'entry_pattern': r'entry[:\s]*(\d+\.?\d*)',
            'exit_pattern': r'exit[:\s]*(\d+\.?\d*)',
            'leverage_pattern': r'(\d+)x'
        }
        
        self.logger.info("📊 Telegram Closed Trades Scanner initialized")
    
    def _initialize_database(self):
        """Initialize SQLite database for closed trades"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS closed_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id INTEGER UNIQUE,
                    symbol TEXT,
                    direction TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    profit_loss REAL,
                    trade_result TEXT,
                    leverage INTEGER,
                    timestamp TIMESTAMP,
                    message_text TEXT,
                    processed BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("📊 Closed trades database initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing closed trades database: {e}")
    
    async def scan_for_closed_trades(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Scan channel for closed trades in the specified time period"""
        try:
            self.logger.info(f"🔍 Scanning {self.channel_username} for closed trades (last {hours_back} hours)")
            
            closed_trades = []
            
            # Get channel messages
            messages = await self._get_channel_messages(hours_back)
            
            for message in messages:
                try:
                    closed_trade = self._parse_closed_trade_message(message)
                    if closed_trade:
                        closed_trades.append(closed_trade)
                except Exception as e:
                    self.logger.warning(f"Error parsing message {message.get('message_id', 'unknown')}: {e}")
                    continue
            
            # Store in database
            if closed_trades:
                await self._store_closed_trades(closed_trades)
            
            self.logger.info(f"📈 Found {len(closed_trades)} closed trades")
            return closed_trades
            
        except Exception as e:
            self.logger.error(f"Error scanning for closed trades: {e}")
            return []
    
    async def _get_channel_messages(self, hours_back: int) -> List[Dict]:
        """Get recent messages from the channel"""
        try:
            messages = []
            
            # Use getUpdates to get recent messages
            url = f"{self.base_url}/getUpdates"
            params = {
                'offset': -100,  # Get last 100 updates
                'allowed_updates': ['channel_post']
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        updates = data.get('result', [])
                        
                        cutoff_time = datetime.now() - timedelta(hours=hours_back)
                        
                        for update in updates:
                            if 'channel_post' in update:
                                message = update['channel_post']
                                chat = message.get('chat', {})
                                
                                # Check if it's from our target channel
                                if chat.get('username') == self.channel_username.replace('@', ''):
                                    message_time = datetime.fromtimestamp(message.get('date', 0))
                                    
                                    # Only include recent messages
                                    if message_time > cutoff_time:
                                        messages.append(message)
            
            return messages
            
        except Exception as e:
            self.logger.error(f"Error getting channel messages: {e}")
            return []
    
    def _parse_closed_trade_message(self, message: Dict) -> Optional[Dict[str, Any]]:
        """Parse a message to extract closed trade information"""
        try:
            text = message.get('text', '')
            if not text:
                return None
            
            text_lower = text.lower()
            
            # Check if message contains closed trade indicators
            if not any(keyword in text_lower for keyword in self.closed_trade_patterns['closed_keywords']):
                return None
            
            closed_trade = {
                'message_id': message.get('message_id'),
                'timestamp': datetime.fromtimestamp(message.get('date', 0)),
                'message_text': text[:500],  # Truncate for storage
                'data_source': 'telegram_channel'
            }
            
            # Extract symbol
            symbol_match = re.search(self.closed_trade_patterns['symbol_pattern'], text, re.IGNORECASE)
            if symbol_match:
                closed_trade['symbol'] = symbol_match.group(1).upper()
            
            # Extract direction
            direction_match = re.search(self.closed_trade_patterns['direction_pattern'], text, re.IGNORECASE)
            if direction_match:
                closed_trade['direction'] = direction_match.group(1).upper()
            
            # Extract profit/loss percentage
            profit_loss = None
            trade_result = None
            
            # Check for profit patterns
            for pattern in self.closed_trade_patterns['profit_patterns']:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    profit_loss = abs(float(match.group(1)))  # Make positive
                    trade_result = 'PROFIT'
                    break
            
            # Check for loss patterns if profit not found
            if profit_loss is None:
                for pattern in self.closed_trade_patterns['loss_patterns']:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        profit_loss = -abs(float(match.group(1)))  # Make negative
                        trade_result = 'LOSS'
                        break
            
            # Determine trade result from keywords if percentage not found
            if trade_result is None:
                if any(word in text_lower for word in ['tp1 hit', 'tp2 hit', 'tp3 hit', 'target reached', 'profit taken']):
                    trade_result = 'PROFIT'
                    profit_loss = 1.5  # Default positive value
                elif any(word in text_lower for word in ['stop loss hit', 'sl hit', 'loss taken']):
                    trade_result = 'LOSS'
                    profit_loss = -1.5  # Default negative value
                else:
                    trade_result = 'CLOSED'
                    profit_loss = 0.0
            
            closed_trade['profit_loss'] = profit_loss
            closed_trade['trade_result'] = trade_result
            
            # Extract entry price
            entry_match = re.search(self.closed_trade_patterns['entry_pattern'], text, re.IGNORECASE)
            if entry_match:
                closed_trade['entry_price'] = float(entry_match.group(1))
            
            # Extract exit price
            exit_match = re.search(self.closed_trade_patterns['exit_pattern'], text, re.IGNORECASE)
            if exit_match:
                closed_trade['exit_price'] = float(exit_match.group(1))
            
            # Extract leverage
            leverage_match = re.search(self.closed_trade_patterns['leverage_pattern'], text, re.IGNORECASE)
            if leverage_match:
                closed_trade['leverage'] = int(leverage_match.group(1))
            else:
                closed_trade['leverage'] = 35  # Default leverage
            
            # Only return if we have minimum required information
            if 'symbol' in closed_trade and 'trade_result' in closed_trade:
                return closed_trade
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error parsing closed trade message: {e}")
            return None
    
    async def _store_closed_trades(self, closed_trades: List[Dict[str, Any]]):
        """Store closed trades in database"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for trade in closed_trades:
                cursor.execute('''
                    INSERT OR REPLACE INTO closed_trades (
                        message_id, symbol, direction, entry_price, exit_price,
                        profit_loss, trade_result, leverage, timestamp, message_text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade.get('message_id'),
                    trade.get('symbol'),
                    trade.get('direction'),
                    trade.get('entry_price'),
                    trade.get('exit_price'),
                    trade.get('profit_loss'),
                    trade.get('trade_result'),
                    trade.get('leverage'),
                    trade.get('timestamp'),
                    trade.get('message_text')
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"💾 Stored {len(closed_trades)} closed trades")
            
        except Exception as e:
            self.logger.error(f"Error storing closed trades: {e}")
    
    async def get_unprocessed_trades(self) -> List[Dict[str, Any]]:
        """Get unprocessed closed trades for ML training"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM closed_trades 
                WHERE processed = 0 
                ORDER BY timestamp DESC
            ''')
            
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            trades = []
            for row in rows:
                trade = dict(zip(columns, row))
                trades.append(trade)
            
            conn.close()
            
            return trades
            
        except Exception as e:
            self.logger.error(f"Error getting unprocessed trades: {e}")
            return []
    
    async def mark_trades_as_processed(self, message_ids: List[int]):
        """Mark trades as processed after ML training"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for message_id in message_ids:
                cursor.execute('''
                    UPDATE closed_trades 
                    SET processed = 1 
                    WHERE message_id = ?
                ''', (message_id,))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"✅ Marked {len(message_ids)} trades as processed")
            
        except Exception as e:
            self.logger.error(f"Error marking trades as processed: {e}")
