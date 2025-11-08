#!/usr/bin/env python3
"""
Ultimate Trading Bot with Advanced Order Flow Strategy
Production-ready bot with comprehensive error handling and advanced features
"""

import asyncio
import logging
import os
import sys
import json
import sqlite3
import aiohttp
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class UltimateTradingBot:
    """Ultimate Trading Bot with Advanced Order Flow Strategy"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Bot configuration
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.target_channel = os.getenv('TARGET_CHANNEL', '@SignalTactics')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID', '@SignalTactics')

        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")

        # Trading configuration
        self.max_messages_per_hour = int(os.getenv('MAX_MESSAGES_PER_HOUR', '3'))
        self.min_trade_interval = int(os.getenv('MIN_TRADE_INTERVAL_SECONDS', '900'))
        self.default_leverage = int(os.getenv('DEFAULT_LEVERAGE', '50'))

        # State management
        self.running = False
        self.message_count = 0
        self.last_message_time = None
        self.last_trade_times = {}

        # Initialize components
        self.order_flow_integration = None
        self.database = None

        self.logger.info("🚀 Ultimate Trading Bot initialized")

    async def initialize_components(self):
        """Initialize all bot components"""
        try:
            # Initialize order flow integration
            try:
                from enhanced_order_flow_integration import EnhancedOrderFlowIntegration
                self.order_flow_integration = EnhancedOrderFlowIntegration()
                self.logger.info("✅ Order Flow Integration initialized")
            except ImportError:
                self.logger.warning("⚠️ Order Flow Integration not available - using fallback")
                self.order_flow_integration = None

            # Initialize database
            await self.initialize_database()

            self.logger.info("✅ All components initialized")

        except Exception as e:
            self.logger.error(f"❌ Error initializing components: {e}")
            raise

    async def initialize_database(self):
        """Initialize database with required tables"""
        try:
            db_path = 'ultimate_trading_bot.db'

            # Create database connection
            self.db_conn = sqlite3.connect(db_path)
            self.db_cursor = self.db_conn.cursor()

            # Create tables
            self.db_cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    direction TEXT,
                    entry_price REAL,
                    stop_loss REAL,
                    tp1 REAL,
                    tp2 REAL,
                    tp3 REAL,
                    signal_strength REAL,
                    strategy TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            self.db_cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signals_sent INTEGER DEFAULT 0,
                    trades_executed INTEGER DEFAULT 0,
                    win_rate REAL DEFAULT 0,
                    total_profit REAL DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            self.db_conn.commit()
            self.logger.info("✅ Database initialized")

        except Exception as e:
            self.logger.error(f"❌ Database initialization error: {e}")
            raise

    async def fetch_market_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch comprehensive market data for analysis"""
        try:
            # Fetch multi-timeframe data from Binance
            base_url = "https://fapi.binance.com/fapi/v1"
            timeframes = ['1m', '3m', '5m', '15m', '1h', '4h']

            ohlcv_data = {}

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                for tf in timeframes:
                    try:
                        url = f"{base_url}/klines"
                        params = {
                            'symbol': symbol,
                            'interval': tf,
                            'limit': 100
                        }

                        async with session.get(url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                ohlcv_data[tf] = data
                                self.logger.debug(f"✅ Fetched {tf} data for {symbol}")
                            else:
                                self.logger.warning(f"⚠️ Failed to fetch {tf} data for {symbol}")

                    except Exception as e:
                        self.logger.warning(f"⚠️ Error fetching {tf} data for {symbol}: {e}")
                        continue

            return ohlcv_data

        except Exception as e:
            self.logger.error(f"❌ Error fetching market data for {symbol}: {e}")
            return {}

    async def analyze_symbol_with_order_flow(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze symbol using advanced order flow strategy"""
        try:
            # Check trade frequency limit
            if not self.can_trade_symbol(symbol):
                return None

            # Fetch market data
            ohlcv_data = await self.fetch_market_data(symbol)
            if not ohlcv_data:
                return None

            # Use order flow integration if available
            if self.order_flow_integration:
                signal = await self.order_flow_integration.analyze_with_order_flow(symbol, ohlcv_data)
                if signal:
                    return signal

            # Fallback analysis
            return await self.fallback_analysis(symbol, ohlcv_data)

        except Exception as e:
            self.logger.error(f"❌ Error analyzing {symbol}: {e}")
            return None

    async def fallback_analysis(self, symbol: str, ohlcv_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fallback analysis when order flow is not available"""
        try:
            # Use 5m or 15m data for fallback
            primary_tf = '15m' if '15m' in ohlcv_data else '5m' if '5m' in ohlcv_data else None
            if not primary_tf or not ohlcv_data[primary_tf]:
                return None

            import pandas as pd
            import numpy as np

            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data[primary_tf], columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 
                'close_time', 'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote', 'ignore'
            ])

            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])

            if len(df) < 20:
                return None

            current_price = float(df['close'].iloc[-1])

            # Simple technical analysis
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]

            # Moving averages
            ma20 = df['close'].rolling(20).mean().iloc[-1]
            ma50 = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else ma20

            # Volume analysis
            volume_ma = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = float(df['volume'].iloc[-1])
            volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1

            # Generate signal
            signal_strength = 0
            direction = None

            # Bullish conditions
            if (current_rsi < 30 and current_price > ma20 and volume_ratio > 1.5):
                direction = 'BUY'
                signal_strength = min(75 + (volume_ratio - 1) * 10, 95)

            # Bearish conditions
            elif (current_rsi > 70 and current_price < ma20 and volume_ratio > 1.5):
                direction = 'SELL'
                signal_strength = min(75 + (volume_ratio - 1) * 10, 95)

            if direction and signal_strength >= 75:
                # Calculate SL and TP levels
                atr = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
                risk_pct = 0.8

                if direction == 'BUY':
                    stop_loss = current_price * (1 - risk_pct / 100)
                    tp1 = current_price * (1 + risk_pct * 1.5 / 100)
                    tp2 = current_price * (1 + risk_pct * 2.5 / 100)
                    tp3 = current_price * (1 + risk_pct * 3.5 / 100)
                else:
                    stop_loss = current_price * (1 + risk_pct / 100)
                    tp1 = current_price * (1 - risk_pct * 1.5 / 100)
                    tp2 = current_price * (1 - risk_pct * 2.5 / 100)
                    tp3 = current_price * (1 - risk_pct * 3.5 / 100)

                return {
                    'symbol': symbol,
                    'direction': direction,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'tp1': tp1,
                    'tp2': tp2,
                    'tp3': tp3,
                    'signal_strength': signal_strength,
                    'leverage': self.default_leverage,
                    'strategy': 'Fallback Technical Analysis',
                    'rsi': current_rsi,
                    'volume_ratio': volume_ratio,
                    'timestamp': datetime.now()
                }

            return None

        except Exception as e:
            self.logger.error(f"❌ Error in fallback analysis: {e}")
            return None

    def can_trade_symbol(self, symbol: str) -> bool:
        """Check if we can trade this symbol based on frequency limits"""
        current_time = datetime.now()

        if symbol in self.last_trade_times:
            time_diff = (current_time - self.last_trade_times[symbol]).total_seconds()
            if time_diff < self.min_trade_interval:
                return False

        return True

    async def send_signal_to_telegram(self, signal: Dict[str, Any]):
        """Send trading signal to Telegram"""
        try:
            # Check rate limits
            if not self.can_send_message():
                self.logger.warning("⚠️ Rate limit reached, skipping signal")
                return False

            # Format signal message
            message = self.format_signal_message(signal)

            # Send to Telegram
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        self.logger.info(f"✅ Signal sent for {signal['symbol']}")

                        # Update counters
                        self.message_count += 1
                        self.last_message_time = datetime.now()
                        self.last_trade_times[signal['symbol']] = datetime.now()

                        # Save to database
                        await self.save_signal_to_db(signal)

                        return True
                    else:
                        self.logger.error(f"❌ Failed to send signal: {response.status}")
                        return False

        except Exception as e:
            self.logger.error(f"❌ Error sending signal to Telegram: {e}")
            return False

    def can_send_message(self) -> bool:
        """Check if we can send a message based on rate limits"""
        current_time = datetime.now()

        # Check hourly limit
        if self.last_message_time:
            time_diff = (current_time - self.last_message_time).total_seconds()
            if time_diff < 3600 and self.message_count >= self.max_messages_per_hour:
                return False
            elif time_diff >= 3600:
                # Reset hourly counter
                self.message_count = 0

        return True

    def format_signal_message(self, signal: Dict[str, Any]) -> str:
        """Format trading signal for Telegram"""
        try:
            symbol = signal['symbol']
            direction = signal['direction']
            entry = signal['entry_price']
            sl = signal['stop_loss']
            tp1 = signal['tp1']
            tp2 = signal['tp2']
            tp3 = signal['tp3']
            strength = signal['signal_strength']
            leverage = signal.get('leverage', self.default_leverage)
            strategy = signal.get('strategy', 'Advanced Order Flow')

            message = f"""
🎯 <b>{symbol} - {direction}</b> 
📊 <b>Strategy:</b> {strategy}
⚡ <b>Strength:</b> {strength:.1f}%
💰 <b>Leverage:</b> {leverage}x

🔹 <b>Entry:</b> ${entry:.6f}
🛡️ <b>Stop Loss:</b> ${sl:.6f}

🎯 <b>Take Profits:</b>
• TP1: ${tp1:.6f}
• TP2: ${tp2:.6f} 
• TP3: ${tp3:.6f}

⏰ <b>Signal Time:</b> {datetime.now().strftime('%H:%M UTC')}

#Signal #{symbol} #{direction}
            """.strip()

            return message

        except Exception as e:
            self.logger.error(f"❌ Error formatting message: {e}")
            return f"Signal Error: {symbol} - {direction}"

    async def save_signal_to_db(self, signal: Dict[str, Any]):
        """Save signal to database"""
        try:
            self.db_cursor.execute('''
                INSERT INTO signals (symbol, direction, entry_price, stop_loss, tp1, tp2, tp3, signal_strength, strategy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal['symbol'],
                signal['direction'],
                signal['entry_price'],
                signal['stop_loss'],
                signal['tp1'],
                signal['tp2'],
                signal['tp3'],
                signal['signal_strength'],
                signal.get('strategy', 'Advanced Order Flow')
            ))

            self.db_conn.commit()

        except Exception as e:
            self.logger.error(f"❌ Error saving signal to database: {e}")

    async def run_analysis_cycle(self):
        """Run continuous analysis cycle"""
        # Major trading pairs for analysis
        symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 
            'SOLUSDT', 'DOGEUSDT', 'LTCUSDT', 'LINKUSDT', 'MATICUSDT'
        ]

        for symbol in symbols:
            if not self.running:
                break

            try:
                # Analyze symbol
                signal = await self.analyze_symbol_with_order_flow(symbol)

                if signal:
                    # Send signal
                    await self.send_signal_to_telegram(signal)

                    # Wait between signals
                    await asyncio.sleep(60)

                # Small delay between symbol analysis
                await asyncio.sleep(5)

            except Exception as e:
                self.logger.error(f"❌ Error analyzing {symbol}: {e}")
                continue

    async def run_bot(self):
        """Main bot execution loop"""
        try:
            self.logger.info("🚀 Starting Ultimate Trading Bot...")

            # Initialize components
            await self.initialize_components()

            self.running = True

            # Send startup message
            startup_message = f"""
🚀 <b>Ultimate Trading Bot STARTED</b>
📊 <b>Strategy:</b> Advanced Order Flow + ML
⚡ <b>Status:</b> Active
🕒 <b>Started:</b> {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

✅ <b>Features Active:</b>
• Order Flow Analysis
• Multi-Timeframe Analysis  
• ML-Enhanced Signals
• Smart Risk Management

Ready to generate high-quality signals! 🎯
            """.strip()

            await self.send_startup_message(startup_message)

            # Main execution loop
            while self.running:
                try:
                    await self.run_analysis_cycle()

                    # Wait before next cycle
                    await asyncio.sleep(300)  # 5 minutes between cycles

                except Exception as e:
                    self.logger.error(f"❌ Error in analysis cycle: {e}")
                    await asyncio.sleep(60)

        except Exception as e:
            self.logger.error(f"❌ Fatal error in bot execution: {e}")
            raise
        finally:
            # Cleanup
            if hasattr(self, 'db_conn'):
                self.db_conn.close()

            self.logger.info("🏁 Ultimate Trading Bot stopped")

    async def send_startup_message(self, message: str):
        """Send startup message to Telegram"""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        self.logger.info("✅ Startup message sent")
                    else:
                        self.logger.warning("⚠️ Failed to send startup message")

        except Exception as e:
            self.logger.error(f"❌ Error sending startup message: {e}")

# Main execution
if __name__ == "__main__":
    bot = UltimateTradingBot()
    asyncio.run(bot.run_bot())