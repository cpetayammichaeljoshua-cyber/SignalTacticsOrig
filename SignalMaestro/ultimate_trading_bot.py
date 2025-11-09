
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultimate_trading_bot.log'),
        logging.StreamHandler()
    ]
)

class UltimateTradingBot:
    """Ultimate Trading Bot with Advanced Order Flow Strategy"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Bot configuration with validation
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.target_channel = os.getenv('TARGET_CHANNEL', '@SignalTactics')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID', '@SignalTactics')

        if not self.bot_token:
            self.logger.error("❌ TELEGRAM_BOT_TOKEN is required but not provided")
            raise ValueError("TELEGRAM_BOT_TOKEN is required")

        # Trading configuration
        self.max_messages_per_hour = int(os.getenv('MAX_MESSAGES_PER_HOUR', '3'))
        self.min_trade_interval = int(os.getenv('MIN_TRADE_INTERVAL_SECONDS', '900'))
        self.default_leverage = min(int(os.getenv('DEFAULT_LEVERAGE', '35')), 75)  # Limit leverage

        # State management
        self.running = False
        self.message_count = 0
        self.last_message_time = None
        self.last_trade_times = {}

        # Initialize components
        self.order_flow_integration = None
        self.db_conn = None
        self.db_cursor = None

        self.logger.info("🚀 Ultimate Trading Bot initialized successfully")

    async def initialize_components(self):
        """Initialize all bot components with enhanced error handling"""
        try:
            # Initialize order flow integration
            try:
                from enhanced_order_flow_integration import EnhancedOrderFlowIntegration
                self.order_flow_integration = EnhancedOrderFlowIntegration()
                self.logger.info("✅ Order Flow Integration initialized")
            except ImportError as e:
                self.logger.warning(f"⚠️ Order Flow Integration not available - using fallback: {e}")
                self.order_flow_integration = None

            # Initialize database
            await self.initialize_database()

            self.logger.info("✅ All components initialized successfully")

        except Exception as e:
            self.logger.error(f"❌ Error initializing components: {e}")
            raise

    async def initialize_database(self):
        """Initialize database with required tables and error handling"""
        try:
            db_path = 'ultimate_trading_bot.db'

            # Create database connection with error handling
            try:
                self.db_conn = sqlite3.connect(db_path, timeout=30)
                self.db_cursor = self.db_conn.cursor()
                
                # Enable WAL mode for better concurrency
                self.db_cursor.execute("PRAGMA journal_mode=WAL")
                self.db_conn.commit()
                
            except sqlite3.Error as e:
                self.logger.error(f"Database connection error: {e}")
                raise

            # Create tables with error handling
            try:
                self.db_cursor.execute('''
                    CREATE TABLE IF NOT EXISTS signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        direction TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        stop_loss REAL,
                        tp1 REAL,
                        tp2 REAL,
                        tp3 REAL,
                        signal_strength REAL,
                        strategy TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX(symbol),
                        INDEX(created_at)
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

                self.db_cursor.execute('''
                    CREATE TABLE IF NOT EXISTS bot_status (
                        id INTEGER PRIMARY KEY,
                        started_at TIMESTAMP,
                        last_signal_at TIMESTAMP,
                        total_signals INTEGER DEFAULT 0,
                        status TEXT DEFAULT 'running'
                    )
                ''')

                self.db_conn.commit()
                self.logger.info("✅ Database tables created/verified successfully")
                
            except sqlite3.Error as e:
                self.logger.error(f"Database table creation error: {e}")
                raise

        except Exception as e:
            self.logger.error(f"❌ Database initialization error: {e}")
            raise

    async def fetch_market_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch comprehensive market data for analysis with enhanced error handling"""
        try:
            # Fetch multi-timeframe data from Binance
            base_url = "https://fapi.binance.com/fapi/v1"
            timeframes = ['1m', '3m', '5m', '15m', '1h', '4h']
            ohlcv_data = {}
            successful_fetches = 0

            timeout = aiohttp.ClientTimeout(total=15)
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            
            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                # Fetch each timeframe
                tasks = []
                for tf in timeframes:
                    task = self._fetch_timeframe_data(session, base_url, symbol, tf)
                    tasks.append((tf, task))
                
                # Wait for all requests with timeout
                for tf, task in tasks:
                    try:
                        data = await asyncio.wait_for(task, timeout=10)
                        if data:
                            ohlcv_data[tf] = data
                            successful_fetches += 1
                            self.logger.debug(f"✅ Fetched {tf} data for {symbol}")
                    except asyncio.TimeoutError:
                        self.logger.warning(f"⏱️ Timeout fetching {tf} data for {symbol}")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Error fetching {tf} data for {symbol}: {e}")

            if successful_fetches == 0:
                self.logger.warning(f"❌ No market data fetched for {symbol}")
                return {}

            self.logger.debug(f"✅ Successfully fetched {successful_fetches}/{len(timeframes)} timeframes for {symbol}")
            return ohlcv_data

        except Exception as e:
            self.logger.error(f"❌ Error fetching market data for {symbol}: {e}")
            return {}

    async def _fetch_timeframe_data(self, session: aiohttp.ClientSession, base_url: str, symbol: str, timeframe: str):
        """Fetch data for a specific timeframe"""
        try:
            url = f"{base_url}/klines"
            params = {
                'symbol': symbol,
                'interval': timeframe,
                'limit': 100
            }

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    self.logger.debug(f"HTTP {response.status} for {symbol} {timeframe}")
                    return None

        except Exception as e:
            self.logger.debug(f"Error fetching {timeframe} for {symbol}: {e}")
            return None

    async def analyze_symbol_with_order_flow(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze symbol using advanced order flow strategy with enhanced error handling"""
        try:
            # Check trade frequency limit
            if not self.can_trade_symbol(symbol):
                self.logger.debug(f"⏳ Trade frequency limit reached for {symbol}")
                return None

            # Fetch market data
            ohlcv_data = await self.fetch_market_data(symbol)
            if not ohlcv_data:
                self.logger.debug(f"❌ No market data available for {symbol}")
                return None

            # Use order flow integration if available
            if self.order_flow_integration:
                try:
                    signal = await self.order_flow_integration.analyze_with_order_flow(symbol, ohlcv_data)
                    if signal:
                        return signal
                except Exception as e:
                    self.logger.warning(f"Order flow analysis failed for {symbol}: {e}")

            # Fallback analysis
            return await self.fallback_analysis(symbol, ohlcv_data)

        except Exception as e:
            self.logger.error(f"❌ Error analyzing {symbol}: {e}")
            return None

    async def fallback_analysis(self, symbol: str, ohlcv_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Enhanced fallback analysis when order flow is not available"""
        try:
            # Use 5m or 15m data for fallback
            primary_tf = None
            for tf in ['15m', '5m', '3m', '1m']:
                if tf in ohlcv_data and ohlcv_data[tf] and len(ohlcv_data[tf]) >= 50:
                    primary_tf = tf
                    break
            
            if not primary_tf:
                return None

            import pandas as pd
            import numpy as np

            # Safe DataFrame creation
            try:
                raw_data = ohlcv_data[primary_tf]
                col_count = len(raw_data[0]) if raw_data and raw_data[0] else 0
                
                if col_count >= 12:
                    columns = [
                        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
                        'close_time', 'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote', 'ignore'
                    ]
                elif col_count >= 6:
                    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                else:
                    self.logger.debug(f"Invalid column count for {symbol}: {col_count}")
                    return None
                
                df = pd.DataFrame(raw_data, columns=columns[:col_count])
                
                # Convert numeric columns safely
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Clean invalid data
                df = df.dropna(subset=numeric_cols)
                
                if len(df) < 20:
                    return None
                    
            except Exception as e:
                self.logger.debug(f"DataFrame creation failed for {symbol}: {e}")
                return None

            current_price = float(df['close'].iloc[-1])

            # Enhanced technical analysis
            try:
                # RSI calculation with error handling
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / (loss + 1e-10)  # Prevent division by zero
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

                # Moving averages with error handling
                ma20 = df['close'].rolling(20).mean().iloc[-1]
                ma50 = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else ma20

                # Volume analysis
                volume_ma = df['volume'].rolling(20).mean().iloc[-1]
                current_volume = float(df['volume'].iloc[-1])
                volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1

                # Price momentum
                price_change_5 = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
                price_change_10 = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10] if len(df) >= 10 else price_change_5
                
            except Exception as e:
                self.logger.debug(f"Technical analysis calculation error for {symbol}: {e}")
                return None

            # Enhanced signal generation with multiple conditions
            signal_strength = 0
            direction = None
            
            # Multi-condition bullish signal
            bullish_conditions = [
                current_rsi < 35,                    # Oversold
                current_price > ma20,                # Above MA20
                volume_ratio > 1.5,                  # High volume
                price_change_5 > 0.005,             # Positive momentum
                price_change_10 > 0.002             # Sustained momentum
            ]
            
            # Multi-condition bearish signal
            bearish_conditions = [
                current_rsi > 65,                    # Overbought
                current_price < ma20,                # Below MA20
                volume_ratio > 1.5,                  # High volume
                price_change_5 < -0.005,            # Negative momentum
                price_change_10 < -0.002            # Sustained momentum
            ]
            
            bullish_score = sum(bullish_conditions)
            bearish_score = sum(bearish_conditions)
            
            # Generate signal if minimum conditions are met
            if bullish_score >= 3 and bullish_score > bearish_score:
                direction = 'BUY'
                signal_strength = min(75 + bullish_score * 5 + (volume_ratio - 1) * 10, 95)
            elif bearish_score >= 3 and bearish_score > bullish_score:
                direction = 'SELL'
                signal_strength = min(75 + bearish_score * 5 + (volume_ratio - 1) * 10, 95)

            if direction and signal_strength >= 75:
                # Enhanced risk management
                try:
                    # Calculate ATR for dynamic risk sizing
                    high_low = df['high'] - df['low']
                    high_close = abs(df['high'] - df['close'].shift(1))
                    low_close = abs(df['low'] - df['close'].shift(1))
                    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    atr = true_range.rolling(14).mean().iloc[-1]
                    
                    # Dynamic risk based on volatility
                    volatility = atr / current_price
                    base_risk = 0.8  # Base 0.8%
                    adjusted_risk = max(0.5, min(1.2, base_risk * (1 + volatility * 10)))
                    
                    if direction == 'BUY':
                        stop_loss = current_price * (1 - adjusted_risk / 100)
                        tp1 = current_price * (1 + adjusted_risk * 1.2 / 100)
                        tp2 = current_price * (1 + adjusted_risk * 2.0 / 100)
                        tp3 = current_price * (1 + adjusted_risk * 3.0 / 100)
                    else:
                        stop_loss = current_price * (1 + adjusted_risk / 100)
                        tp1 = current_price * (1 - adjusted_risk * 1.2 / 100)
                        tp2 = current_price * (1 - adjusted_risk * 2.0 / 100)
                        tp3 = current_price * (1 - adjusted_risk * 3.0 / 100)
                    
                    # Dynamic leverage based on signal strength
                    base_leverage = self.default_leverage
                    if signal_strength >= 90:
                        leverage = min(base_leverage, 50)
                    elif signal_strength >= 85:
                        leverage = min(base_leverage, 40)
                    else:
                        leverage = min(base_leverage, 30)

                    return {
                        'symbol': symbol,
                        'direction': direction,
                        'entry_price': current_price,
                        'stop_loss': stop_loss,
                        'tp1': tp1,
                        'tp2': tp2,
                        'tp3': tp3,
                        'signal_strength': signal_strength,
                        'leverage': leverage,
                        'strategy': 'Enhanced Technical Analysis',
                        'rsi': current_rsi,
                        'volume_ratio': volume_ratio,
                        'price_momentum_5m': price_change_5,
                        'atr_risk': adjusted_risk,
                        'bullish_conditions': bullish_score,
                        'bearish_conditions': bearish_score,
                        'timestamp': datetime.now()
                    }
                    
                except Exception as e:
                    self.logger.debug(f"Risk calculation error for {symbol}: {e}")
                    return None

            return None

        except Exception as e:
            self.logger.error(f"❌ Error in enhanced fallback analysis for {symbol}: {e}")
            return None

    def can_trade_symbol(self, symbol: str) -> bool:
        """Check if we can trade this symbol based on frequency limits"""
        try:
            current_time = datetime.now()

            if symbol in self.last_trade_times:
                time_diff = (current_time - self.last_trade_times[symbol]).total_seconds()
                if time_diff < self.min_trade_interval:
                    return False

            return True
        except Exception:
            return True

    async def send_signal_to_telegram(self, signal: Dict[str, Any]) -> bool:
        """Send trading signal to Telegram with enhanced error handling"""
        try:
            # Check rate limits
            if not self.can_send_message():
                self.logger.warning("⚠️ Rate limit reached, skipping signal")
                return False

            # Format signal message
            message = self.format_signal_message(signal)
            if not message:
                self.logger.error("❌ Failed to format signal message")
                return False

            # Send to Telegram with retry logic
            success = False
            for attempt in range(3):
                try:
                    url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
                    payload = {
                        'chat_id': self.chat_id,
                        'text': message,
                        'parse_mode': 'HTML',
                        'disable_web_page_preview': True
                    }

                    timeout = aiohttp.ClientTimeout(total=15)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.post(url, json=payload) as response:
                            if response.status == 200:
                                self.logger.info(f"✅ Signal sent for {signal['symbol']} (attempt {attempt + 1})")
                                success = True
                                break
                            else:
                                error_text = await response.text()
                                self.logger.error(f"❌ Telegram API error {response.status}: {error_text}")
                                if response.status == 400:  # Bad request, don't retry
                                    break
                        
                except asyncio.TimeoutError:
                    self.logger.warning(f"⏱️ Timeout sending signal (attempt {attempt + 1})")
                except Exception as e:
                    self.logger.error(f"❌ Error sending signal (attempt {attempt + 1}): {e}")
                
                if not success and attempt < 2:
                    await asyncio.sleep(2)  # Wait before retry

            if success:
                # Update counters
                self.message_count += 1
                self.last_message_time = datetime.now()
                self.last_trade_times[signal['symbol']] = datetime.now()

                # Save to database
                await self.save_signal_to_db(signal)
                return True
            
            return False

        except Exception as e:
            self.logger.error(f"❌ Critical error sending signal to Telegram: {e}")
            return False

    def can_send_message(self) -> bool:
        """Check if we can send a message based on rate limits"""
        try:
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
        except Exception:
            return True

    def format_signal_message(self, signal: Dict[str, Any]) -> str:
        """Format trading signal for Telegram with enhanced formatting"""
        try:
            symbol = signal.get('symbol', '')
            direction = signal.get('direction', '')
            entry = signal.get('entry_price', 0)
            sl = signal.get('stop_loss', 0)
            tp1 = signal.get('tp1', 0)
            tp2 = signal.get('tp2', 0)
            tp3 = signal.get('tp3', 0)
            strength = signal.get('signal_strength', 0)
            leverage = signal.get('leverage', self.default_leverage)
            strategy = signal.get('strategy', 'Advanced Order Flow')

            # Determine price precision based on asset
            if 'USDT' in symbol:
                if entry > 100:
                    precision = 2
                elif entry > 1:
                    precision = 4
                else:
                    precision = 6
            else:
                precision = 6

            # Format prices with appropriate precision
            entry_str = f"{entry:.{precision}f}"
            sl_str = f"{sl:.{precision}f}"
            tp1_str = f"{tp1:.{precision}f}"
            tp2_str = f"{tp2:.{precision}f}"
            tp3_str = f"{tp3:.{precision}f}"

            # Calculate risk/reward
            try:
                if direction == 'BUY':
                    risk = abs(entry - sl) / entry * 100
                    reward1 = abs(tp1 - entry) / entry * 100
                else:
                    risk = abs(sl - entry) / entry * 100
                    reward1 = abs(entry - tp1) / entry * 100
                
                rr_ratio = reward1 / risk if risk > 0 else 0
            except:
                risk = 0.8
                rr_ratio = 2.0

            # Add order flow specific information if available
            order_flow_info = ""
            if signal.get('order_flow_enhanced'):
                cvd_trend = signal.get('cvd_trend', 'neutral')
                smart_money = signal.get('smart_money_detected', False)
                order_flow_info = f"""
🔍 <b>Order Flow Analysis:</b>
• CVD Trend: {cvd_trend.upper()}
• Smart Money: {'✅ DETECTED' if smart_money else '❌ Not Detected'}"""

            # Enhanced message with emojis and formatting
            direction_emoji = "🟢" if direction == 'BUY' else "🔴"
            strength_emoji = "🔥" if strength >= 85 else "⚡" if strength >= 75 else "💫"
            
            message = f"""{direction_emoji} <b>{symbol} - {direction}</b> {strength_emoji}

📊 <b>Strategy:</b> {strategy}
⚡ <b>Signal Strength:</b> {strength:.1f}%
💰 <b>Leverage:</b> {leverage}x
📈 <b>Risk/Reward:</b> 1:{rr_ratio:.1f}

🎯 <b>Entry:</b> {entry_str}
🛡️ <b>Stop Loss:</b> {sl_str} (-{risk:.1f}%)

🎯 <b>Take Profits:</b>
• TP1: {tp1_str}
• TP2: {tp2_str} 
• TP3: {tp3_str}{order_flow_info}

⏰ <b>Signal Time:</b> {datetime.now().strftime('%H:%M UTC')}
🏷️ #{symbol} #{direction} #Signal

<i>⚠️ Always use proper risk management</i>""".strip()

            return message

        except Exception as e:
            self.logger.error(f"❌ Error formatting message: {e}")
            return f"Signal Error: {symbol} - {direction}"

    async def save_signal_to_db(self, signal: Dict[str, Any]):
        """Save signal to database with enhanced error handling"""
        try:
            if not self.db_conn or not self.db_cursor:
                self.logger.warning("⚠️ Database not available for signal storage")
                return

            self.db_cursor.execute('''
                INSERT INTO signals (symbol, direction, entry_price, stop_loss, tp1, tp2, tp3, signal_strength, strategy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal.get('symbol', ''),
                signal.get('direction', ''),
                signal.get('entry_price', 0),
                signal.get('stop_loss', 0),
                signal.get('tp1', 0),
                signal.get('tp2', 0),
                signal.get('tp3', 0),
                signal.get('signal_strength', 0),
                signal.get('strategy', 'Advanced Order Flow')
            ))

            self.db_conn.commit()
            self.logger.debug(f"✅ Signal saved to database for {signal.get('symbol', '')}")

        except sqlite3.Error as e:
            self.logger.error(f"❌ Database error saving signal: {e}")
        except Exception as e:
            self.logger.error(f"❌ Error saving signal to database: {e}")

    async def run_analysis_cycle(self):
        """Run continuous analysis cycle with enhanced error handling"""
        # Enhanced list of major trading pairs
        symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 
            'SOLUSDT', 'DOGEUSDT', 'LTCUSDT', 'LINKUSDT', 'MATICUSDT',
            'AVAXUSDT', 'DOTUSDT', 'ATOMUSDT', 'NEARUSDT', 'FTMUSDT'
        ]

        analysis_count = 0
        signals_generated = 0

        for symbol in symbols:
            if not self.running:
                break

            try:
                analysis_count += 1
                self.logger.debug(f"🔍 Analyzing {symbol} ({analysis_count}/{len(symbols)})")
                
                # Analyze symbol
                signal = await self.analyze_symbol_with_order_flow(symbol)

                if signal:
                    signals_generated += 1
                    # Send signal
                    success = await self.send_signal_to_telegram(signal)
                    
                    if success:
                        self.logger.info(f"📤 Signal sent successfully for {symbol}")
                        # Wait between signals to avoid spam
                        await asyncio.sleep(60)
                    else:
                        self.logger.warning(f"⚠️ Failed to send signal for {symbol}")

                # Small delay between symbol analysis
                await asyncio.sleep(3)

            except Exception as e:
                self.logger.error(f"❌ Error analyzing {symbol}: {e}")
                continue

        self.logger.info(f"📊 Analysis cycle completed: {analysis_count} symbols analyzed, {signals_generated} signals generated")

    async def run_bot(self):
        """Main bot execution loop with comprehensive error handling"""
        try:
            self.logger.info("🚀 Starting Ultimate Trading Bot...")

            # Initialize components
            await self.initialize_components()

            self.running = True

            # Send startup message
            startup_message = f"""🚀 <b>Ultimate Trading Bot STARTED</b>

📊 <b>Strategy:</b> Advanced Order Flow + Enhanced TA
⚡ <b>Status:</b> Active & Monitoring Markets
🕒 <b>Started:</b> {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

✅ <b>Active Features:</b>
• 📈 Advanced Order Flow Analysis
• 🔍 Multi-Timeframe Confluence  
• 🎯 Dynamic Risk Management
• 🤖 ML-Enhanced Signal Validation
• ⚡ Ultra-Fast Market Scanning

<b>Ready to generate high-quality signals!</b> 🎯

<i>Risk Management: 0.8% per trade | Max 3 signals/hour</i>""".strip()

            await self.send_startup_message(startup_message)

            # Main execution loop
            cycle_count = 0
            while self.running:
                try:
                    cycle_count += 1
                    self.logger.info(f"🔄 Starting analysis cycle #{cycle_count}")
                    
                    start_time = datetime.now()
                    await self.run_analysis_cycle()
                    end_time = datetime.now()
                    
                    cycle_duration = (end_time - start_time).total_seconds()
                    self.logger.info(f"✅ Analysis cycle #{cycle_count} completed in {cycle_duration:.1f}s")

                    # Wait before next cycle (5 minutes)
                    self.logger.debug("⏳ Waiting 5 minutes before next cycle...")
                    await asyncio.sleep(300)

                except Exception as e:
                    self.logger.error(f"❌ Error in analysis cycle #{cycle_count}: {e}")
                    await asyncio.sleep(60)  # Shorter wait on error

        except KeyboardInterrupt:
            self.logger.info("🛑 Bot stopped by user")
        except Exception as e:
            self.logger.error(f"❌ Fatal error in bot execution: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        finally:
            # Cleanup
            if hasattr(self, 'db_conn') and self.db_conn:
                try:
                    self.db_conn.close()
                    self.logger.info("📦 Database connection closed")
                except:
                    pass

            self.logger.info("🏁 Ultimate Trading Bot stopped")

    async def send_startup_message(self, message: str):
        """Send startup message to Telegram with error handling"""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML',
                'disable_web_page_preview': True
            }

            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        self.logger.info("✅ Startup message sent successfully")
                    else:
                        error_text = await response.text()
                        self.logger.warning(f"⚠️ Failed to send startup message: {response.status} - {error_text}")

        except Exception as e:
            self.logger.error(f"❌ Error sending startup message: {e}")


# Main execution
if __name__ == "__main__":
    try:
        bot = UltimateTradingBot()
        asyncio.run(bot.run_bot())
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
