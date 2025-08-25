
#!/usr/bin/env python3
"""
Telegram Trade Tracker for Ultimate Trading Bot
Tracks all trades from @SignalTactics channel and saves them for ML learning
"""

import asyncio
import logging
import aiohttp
import json
import sqlite3
import pandas as pd
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import os

class TelegramTradeTracker:
    """Comprehensive trade tracker for ML learning"""
    
    def __init__(self, bot_token: str, channel_username: str = "@SignalTactics"):
        self.logger = logging.getLogger(__name__)
        self.bot_token = bot_token
        self.channel_username = channel_username
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
        # Database for trade storage
        self.db_path = "ml_trade_tracking.db"
        self.trades_file = "tracked_trades.json"
        self.csv_file = "ml_training_data.csv"
        
        # Initialize database
        self._initialize_database()
        
        # Tracking parameters
        self.last_update_id = None
        self.tracking_active = True
        
        self.logger.info("📊 Telegram Trade Tracker initialized")
    
    def _initialize_database(self):
        """Initialize comprehensive trade tracking database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Main trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tracked_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id INTEGER,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL,
                    stop_loss REAL,
                    take_profit_1 REAL,
                    take_profit_2 REAL,
                    take_profit_3 REAL,
                    leverage INTEGER,
                    signal_strength REAL,
                    timestamp TIMESTAMP,
                    message_text TEXT,
                    trade_status TEXT DEFAULT 'pending',
                    outcome TEXT,
                    profit_loss REAL,
                    tp1_hit BOOLEAN DEFAULT 0,
                    tp2_hit BOOLEAN DEFAULT 0,
                    tp3_hit BOOLEAN DEFAULT 0,
                    sl_hit BOOLEAN DEFAULT 0,
                    duration_minutes REAL,
                    ml_features TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Trade updates table for tracking TP/SL hits
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id INTEGER,
                    update_type TEXT,
                    update_data TEXT,
                    timestamp TIMESTAMP,
                    message_id INTEGER,
                    FOREIGN KEY (trade_id) REFERENCES tracked_trades (id)
                )
            ''')
            
            # Channel messages archive
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS channel_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id INTEGER UNIQUE,
                    message_text TEXT,
                    timestamp TIMESTAMP,
                    is_trade_signal BOOLEAN DEFAULT 0,
                    parsed_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # ML training data export
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_training_exports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    export_type TEXT,
                    file_path TEXT,
                    records_count INTEGER,
                    export_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("📊 Trade tracking database initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing trade tracking database: {e}")
    
    async def start_tracking(self):
        """Start continuous trade tracking"""
        self.logger.info("🎯 Starting continuous trade tracking from @SignalTactics")
        
        while self.tracking_active:
            try:
                # Get channel updates
                messages = await self.get_channel_messages()
                
                if messages:
                    self.logger.info(f"📨 Processing {len(messages)} new messages")
                    
                    for message in messages:
                        try:
                            await self.process_message(message)
                        except Exception as msg_error:
                            self.logger.error(f"Error processing message: {msg_error}")
                            continue
                
                # Export data for ML training
                await self.export_ml_training_data()
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in trade tracking loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def get_channel_messages(self, limit: int = 100) -> List[Dict]:
        """Get recent messages from the channel"""
        try:
            url = f"{self.base_url}/getUpdates"
            params = {
                'timeout': 30,
                'limit': limit
            }
            
            if self.last_update_id:
                params['offset'] = self.last_update_id + 1
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        updates = data.get('result', [])
                        
                        channel_messages = []
                        for update in updates:
                            self.last_update_id = update['update_id']
                            
                            if 'channel_post' in update:
                                message = update['channel_post']
                                if message.get('chat', {}).get('username') == self.channel_username.replace('@', ''):
                                    channel_messages.append(message)
                        
                        return channel_messages
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting channel messages: {e}")
            return []
    
    async def process_message(self, message: Dict):
        """Process and analyze message for trade signals"""
        try:
            message_id = message.get('message_id')
            message_text = message.get('text', '')
            timestamp = datetime.fromtimestamp(message.get('date', 0))
            
            # Save all messages to archive
            await self.save_message_to_archive(message_id, message_text, timestamp)
            
            # Check if it's a trade signal
            if self.is_trade_signal(message_text):
                trade_data = self.parse_trade_signal(message_text)
                
                if trade_data:
                    trade_data.update({
                        'message_id': message_id,
                        'timestamp': timestamp,
                        'message_text': message_text
                    })
                    
                    await self.save_tracked_trade(trade_data)
                    self.logger.info(f"💾 Saved trade signal: {trade_data['symbol']} {trade_data['direction']}")
            
            # Check for trade updates (TP/SL hits)
            elif self.is_trade_update(message_text):
                await self.process_trade_update(message_id, message_text, timestamp)
            
        except Exception as e:
            self.logger.error(f"Error processing message {message.get('message_id')}: {e}")
    
    def is_trade_signal(self, text: str) -> bool:
        """Check if message contains a trade signal"""
        trade_indicators = [
            'BUY', 'SELL', 'LONG', 'SHORT',
            'Entry:', 'Stop Loss:', 'Take Profit:',
            'TP1:', 'TP2:', 'TP3:',
            'Leverage:', '#'
        ]
        
        text_upper = text.upper()
        return any(indicator in text_upper for indicator in trade_indicators)
    
    def is_trade_update(self, text: str) -> bool:
        """Check if message contains trade updates"""
        update_indicators = [
            'TP1 HIT', 'TP2 HIT', 'TP3 HIT',
            'STOP LOSS HIT', 'SL HIT',
            'PROFIT:', 'LOSS:', 'CLOSED',
            'TARGET 1', 'TARGET 2', 'TARGET 3'
        ]
        
        text_upper = text.upper()
        return any(indicator in text_upper for indicator in update_indicators)
    
    def parse_trade_signal(self, text: str) -> Optional[Dict]:
        """Parse trade signal from message text"""
        try:
            trade_data = {}
            
            # Extract symbol
            symbol_match = re.search(r'#?([A-Z]{3,10}USDT?)', text.upper())
            if symbol_match:
                trade_data['symbol'] = symbol_match.group(1)
                if not trade_data['symbol'].endswith('USDT'):
                    trade_data['symbol'] += 'USDT'
            
            # Extract direction
            if any(word in text.upper() for word in ['BUY', 'LONG']):
                trade_data['direction'] = 'BUY'
            elif any(word in text.upper() for word in ['SELL', 'SHORT']):
                trade_data['direction'] = 'SELL'
            
            # Extract prices
            entry_match = re.search(r'Entry:?\s*([0-9]*\.?[0-9]+)', text, re.IGNORECASE)
            if entry_match:
                trade_data['entry_price'] = float(entry_match.group(1))
            
            sl_match = re.search(r'Stop Loss:?\s*([0-9]*\.?[0-9]+)', text, re.IGNORECASE)
            if sl_match:
                trade_data['stop_loss'] = float(sl_match.group(1))
            
            # Extract take profits
            tp1_match = re.search(r'TP1:?\s*([0-9]*\.?[0-9]+)', text, re.IGNORECASE)
            if tp1_match:
                trade_data['take_profit_1'] = float(tp1_match.group(1))
            
            tp2_match = re.search(r'TP2:?\s*([0-9]*\.?[0-9]+)', text, re.IGNORECASE)
            if tp2_match:
                trade_data['take_profit_2'] = float(tp2_match.group(1))
            
            tp3_match = re.search(r'TP3:?\s*([0-9]*\.?[0-9]+)', text, re.IGNORECASE)
            if tp3_match:
                trade_data['take_profit_3'] = float(tp3_match.group(1))
            
            # Extract leverage
            leverage_match = re.search(r'Leverage:?\s*([0-9]+)x?', text, re.IGNORECASE)
            if leverage_match:
                trade_data['leverage'] = int(leverage_match.group(1))
            
            # Extract signal strength if mentioned
            strength_match = re.search(r'(?:Strength|Signal):?\s*([0-9]+)%?', text, re.IGNORECASE)
            if strength_match:
                trade_data['signal_strength'] = float(strength_match.group(1))
            
            return trade_data if 'symbol' in trade_data and 'direction' in trade_data else None
            
        except Exception as e:
            self.logger.error(f"Error parsing trade signal: {e}")
            return None
    
    async def save_message_to_archive(self, message_id: int, text: str, timestamp: datetime):
        """Save message to archive"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO channel_messages 
                (message_id, message_text, timestamp, is_trade_signal)
                VALUES (?, ?, ?, ?)
            ''', (message_id, text, timestamp, self.is_trade_signal(text)))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving message to archive: {e}")
    
    async def save_tracked_trade(self, trade_data: Dict):
        """Save tracked trade to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO tracked_trades (
                    message_id, symbol, direction, entry_price, stop_loss,
                    take_profit_1, take_profit_2, take_profit_3, leverage,
                    signal_strength, timestamp, message_text, ml_features
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data.get('message_id'),
                trade_data.get('symbol'),
                trade_data.get('direction'),
                trade_data.get('entry_price'),
                trade_data.get('stop_loss'),
                trade_data.get('take_profit_1'),
                trade_data.get('take_profit_2'),
                trade_data.get('take_profit_3'),
                trade_data.get('leverage'),
                trade_data.get('signal_strength'),
                trade_data.get('timestamp'),
                trade_data.get('message_text'),
                json.dumps(self.extract_ml_features(trade_data))
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving tracked trade: {e}")
    
    def extract_ml_features(self, trade_data: Dict) -> Dict:
        """Extract ML features from trade data"""
        try:
            features = {
                'risk_reward_ratio': 0,
                'entry_sl_distance_pct': 0,
                'tp1_distance_pct': 0,
                'tp2_distance_pct': 0,
                'tp3_distance_pct': 0,
                'hour_of_day': trade_data.get('timestamp', datetime.now()).hour,
                'day_of_week': trade_data.get('timestamp', datetime.now()).weekday(),
                'leverage': trade_data.get('leverage', 10),
                'signal_strength': trade_data.get('signal_strength', 80)
            }
            
            entry = trade_data.get('entry_price')
            sl = trade_data.get('stop_loss')
            tp1 = trade_data.get('take_profit_1')
            tp2 = trade_data.get('take_profit_2')
            tp3 = trade_data.get('take_profit_3')
            
            if entry and sl:
                features['entry_sl_distance_pct'] = abs(entry - sl) / entry * 100
                
                if tp1:
                    tp1_distance = abs(tp1 - entry)
                    sl_distance = abs(entry - sl)
                    features['risk_reward_ratio'] = tp1_distance / sl_distance if sl_distance > 0 else 0
                    features['tp1_distance_pct'] = tp1_distance / entry * 100
                
                if tp2:
                    features['tp2_distance_pct'] = abs(tp2 - entry) / entry * 100
                
                if tp3:
                    features['tp3_distance_pct'] = abs(tp3 - entry) / entry * 100
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting ML features: {e}")
            return {}
    
    async def process_trade_update(self, message_id: int, text: str, timestamp: datetime):
        """Process trade update messages (TP/SL hits)"""
        try:
            # Find related trade by looking for symbol in recent trades
            symbol_match = re.search(r'#?([A-Z]{3,10}USDT?)', text.upper())
            if not symbol_match:
                return
            
            symbol = symbol_match.group(1)
            if not symbol.endswith('USDT'):
                symbol += 'USDT'
            
            # Find the most recent pending trade for this symbol
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id FROM tracked_trades 
                WHERE symbol = ? AND trade_status = 'pending' 
                ORDER BY timestamp DESC LIMIT 1
            ''', (symbol,))
            
            trade_row = cursor.fetchone()
            if not trade_row:
                conn.close()
                return
            
            trade_id = trade_row[0]
            
            # Determine update type
            text_upper = text.upper()
            update_data = {
                'message_text': text,
                'timestamp': timestamp
            }
            
            if 'TP1' in text_upper and 'HIT' in text_upper:
                cursor.execute('UPDATE tracked_trades SET tp1_hit = 1 WHERE id = ?', (trade_id,))
                update_data['type'] = 'TP1_HIT'
            elif 'TP2' in text_upper and 'HIT' in text_upper:
                cursor.execute('UPDATE tracked_trades SET tp2_hit = 1 WHERE id = ?', (trade_id,))
                update_data['type'] = 'TP2_HIT'
            elif 'TP3' in text_upper and 'HIT' in text_upper:
                cursor.execute('UPDATE tracked_trades SET tp3_hit = 1 WHERE id = ?', (trade_id,))
                update_data['type'] = 'TP3_HIT'
            elif 'STOP LOSS' in text_upper or 'SL HIT' in text_upper:
                cursor.execute('UPDATE tracked_trades SET sl_hit = 1, trade_status = "closed" WHERE id = ?', (trade_id,))
                update_data['type'] = 'SL_HIT'
            elif 'CLOSED' in text_upper:
                cursor.execute('UPDATE tracked_trades SET trade_status = "closed" WHERE id = ?', (trade_id,))
                update_data['type'] = 'CLOSED'
            
            # Save update
            cursor.execute('''
                INSERT INTO trade_updates (trade_id, update_type, update_data, timestamp, message_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (trade_id, update_data['type'], json.dumps(update_data), timestamp, message_id))
            
            # Calculate trade outcome and profit/loss
            await self.calculate_trade_outcome(trade_id, cursor)
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"📊 Updated trade {trade_id}: {update_data['type']}")
            
        except Exception as e:
            self.logger.error(f"Error processing trade update: {e}")
    
    async def calculate_trade_outcome(self, trade_id: int, cursor):
        """Calculate final trade outcome and P&L"""
        try:
            cursor.execute('''
                SELECT entry_price, stop_loss, take_profit_1, take_profit_2, take_profit_3,
                       tp1_hit, tp2_hit, tp3_hit, sl_hit, direction, timestamp
                FROM tracked_trades WHERE id = ?
            ''', (trade_id,))
            
            trade = cursor.fetchone()
            if not trade:
                return
            
            entry, sl, tp1, tp2, tp3, tp1_hit, tp2_hit, tp3_hit, sl_hit, direction, start_time = trade
            
            profit_loss = 0
            outcome = 'unknown'
            
            if sl_hit:
                # Stop loss hit
                if direction == 'BUY':
                    profit_loss = (sl - entry) / entry * 100
                else:
                    profit_loss = (entry - sl) / entry * 100
                outcome = 'loss'
            elif tp3_hit:
                # TP3 hit (best outcome)
                if direction == 'BUY':
                    profit_loss = (tp3 - entry) / entry * 100
                else:
                    profit_loss = (entry - tp3) / entry * 100
                outcome = 'tp3_profit'
            elif tp2_hit:
                # TP2 hit
                if direction == 'BUY':
                    profit_loss = (tp2 - entry) / entry * 100
                else:
                    profit_loss = (entry - tp2) / entry * 100
                outcome = 'tp2_profit'
            elif tp1_hit:
                # TP1 hit
                if direction == 'BUY':
                    profit_loss = (tp1 - entry) / entry * 100
                else:
                    profit_loss = (entry - tp1) / entry * 100
                outcome = 'tp1_profit'
            
            # Calculate duration
            duration_minutes = 0
            if start_time:
                start_dt = datetime.fromisoformat(start_time)
                duration_minutes = (datetime.now() - start_dt).total_seconds() / 60
            
            # Update trade record
            cursor.execute('''
                UPDATE tracked_trades 
                SET outcome = ?, profit_loss = ?, duration_minutes = ?
                WHERE id = ?
            ''', (outcome, profit_loss, duration_minutes, trade_id))
            
        except Exception as e:
            self.logger.error(f"Error calculating trade outcome: {e}")
    
    async def export_ml_training_data(self):
        """Export all tracked trades to files for ML training"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Export to JSON
            query = '''
                SELECT t.*, 
                       GROUP_CONCAT(tu.update_type) as updates
                FROM tracked_trades t
                LEFT JOIN trade_updates tu ON t.id = tu.trade_id
                GROUP BY t.id
                ORDER BY t.timestamp DESC
            '''
            
            df = pd.read_sql_query(query, conn)
            
            # Save to JSON file
            trades_json = df.to_dict('records')
            with open(self.trades_file, 'w') as f:
                json.dump(trades_json, f, indent=2, default=str)
            
            # Save to CSV for easy ML processing
            df.to_csv(self.csv_file, index=False)
            
            # Log export
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO ml_training_exports (export_type, file_path, records_count)
                VALUES (?, ?, ?)
            ''', ('json_csv', f"{self.trades_file},{self.csv_file}", len(df)))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"📤 Exported {len(df)} trades to {self.trades_file} and {self.csv_file}")
            
        except Exception as e:
            self.logger.error(f"Error exporting ML training data: {e}")
    
    async def get_training_data_summary(self) -> Dict[str, Any]:
        """Get summary of tracked training data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total trades
            cursor.execute('SELECT COUNT(*) FROM tracked_trades')
            total_trades = cursor.fetchone()[0]
            
            # Completed trades
            cursor.execute('SELECT COUNT(*) FROM tracked_trades WHERE trade_status = "closed"')
            completed_trades = cursor.fetchone()[0]
            
            # Profitable trades
            cursor.execute('SELECT COUNT(*) FROM tracked_trades WHERE profit_loss > 0')
            profitable_trades = cursor.fetchone()[0]
            
            # Average profit/loss
            cursor.execute('SELECT AVG(profit_loss) FROM tracked_trades WHERE profit_loss IS NOT NULL')
            avg_pnl = cursor.fetchone()[0] or 0
            
            # Symbols tracked
            cursor.execute('SELECT COUNT(DISTINCT symbol) FROM tracked_trades')
            unique_symbols = cursor.fetchone()[0]
            
            # Recent activity
            cursor.execute('''
                SELECT COUNT(*) FROM tracked_trades 
                WHERE timestamp > datetime('now', '-24 hours')
            ''')
            recent_trades = cursor.fetchone()[0]
            
            conn.close()
            
            win_rate = (profitable_trades / completed_trades * 100) if completed_trades > 0 else 0
            
            return {
                'total_trades_tracked': total_trades,
                'completed_trades': completed_trades,
                'profitable_trades': profitable_trades,
                'win_rate': round(win_rate, 2),
                'avg_profit_loss': round(avg_pnl, 2),
                'unique_symbols': unique_symbols,
                'recent_24h_trades': recent_trades,
                'data_files': [self.trades_file, self.csv_file]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting training data summary: {e}")
            return {}
    
    def stop_tracking(self):
        """Stop trade tracking"""
        self.tracking_active = False
        self.logger.info("🛑 Trade tracking stopped")
