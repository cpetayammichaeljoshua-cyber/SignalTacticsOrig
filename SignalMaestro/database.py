"""
Database management for trading bot
Handles SQLite database operations with proper error handling and initialization
"""

import sqlite3
import asyncio
import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

class Database:
    """Database manager for trading bot with SQLite backend"""

    def __init__(self, db_path: str = "trading_bot.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

    async def initialize(self):
        """Initialize database with required tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    telegram_id INTEGER UNIQUE NOT NULL,
                    username TEXT,
                    first_name TEXT,
                    last_name TEXT,
                    is_admin BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Signals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    entry_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    quantity REAL,
                    leverage INTEGER DEFAULT 1,
                    signal_strength REAL DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    raw_signal TEXT,
                    processed_signal TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    executed_at TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')

            # Trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id INTEGER,
                    user_id INTEGER,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL,
                    exit_price REAL,
                    quantity REAL,
                    pnl REAL DEFAULT 0,
                    status TEXT DEFAULT 'open',
                    trade_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    closed_at TIMESTAMP,
                    FOREIGN KEY (signal_id) REFERENCES signals (id),
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')

            # System logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    module TEXT,
                    user_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.commit()
            conn.close()

            self.logger.info("✅ Database initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Database initialization failed: {e}")
            return False

    async def test_connection(self) -> bool:
        """Test database connection"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            conn.close()
            return True
        except Exception as e:
            self.logger.error(f"Database connection test failed: {e}")
            return False

    def save_signal_data(self, signal_data: Dict[str, Any], user_id: Optional[int] = None) -> Optional[int]:
        """Save signal data to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO signals (
                    user_id, symbol, action, entry_price, stop_loss, 
                    take_profit, quantity, leverage, signal_strength,
                    raw_signal, processed_signal, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                signal_data.get('symbol'),
                signal_data.get('action'),
                signal_data.get('entry_price'),
                signal_data.get('stop_loss'),
                signal_data.get('take_profit'),
                signal_data.get('quantity'),
                signal_data.get('leverage', 1),
                signal_data.get('signal_strength', 0),
                json.dumps(signal_data.get('raw_signal', {})),
                json.dumps(signal_data),
                'processed'
            ))

            signal_id = cursor.lastrowid
            conn.commit()
            conn.close()

            return signal_id

        except Exception as e:
            self.logger.error(f"Error saving signal data: {e}")
            return None

    async def save_system_log(self, level: str, message: str, module: str = None, user_id: int = None):
        """Save system log entry"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO system_logs (level, message, module, user_id)
                VALUES (?, ?, ?, ?)
            ''', (level, message, module, user_id))

            conn.commit()
            conn.close()

        except Exception as e:
            # Don't log database errors to avoid recursion
            pass

    def get_recent_signals(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent signals from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM signals 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (limit,))

            rows = cursor.fetchall()
            conn.close()

            # Convert to list of dictionaries
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]

        except Exception as e:
            self.logger.error(f"Error getting recent signals: {e}")
            return []

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Total signals
            cursor.execute("SELECT COUNT(*) FROM signals")
            total_signals = cursor.fetchone()[0]

            # Total trades
            cursor.execute("SELECT COUNT(*) FROM trades")
            total_trades = cursor.fetchone()[0]

            # Successful trades
            cursor.execute("SELECT COUNT(*) FROM trades WHERE pnl > 0")
            profitable_trades = cursor.fetchone()[0]

            # Total PnL
            cursor.execute("SELECT COALESCE(SUM(pnl), 0) FROM trades")
            total_pnl = cursor.fetchone()[0]

            conn.close()

            win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0

            return {
                'total_signals': total_signals,
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'losing_trades': total_trades - profitable_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl_per_trade': total_pnl / total_trades if total_trades > 0 else 0
            }

        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {
                'total_signals': 0,
                'total_trades': 0,
                'profitable_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl_per_trade': 0
            }