
import aiohttp
import logging
from typing import Optional, List, Dict


"""
Binance trading integration using ccxt library
Handles trade execution, portfolio management, and market data
"""

import asyncio
import logging
import ccxt.async_support as ccxt
from typing import Dict, Any, List, Optional
from decimal import Decimal, ROUND_DOWN
import time
import json
import os
from datetime import datetime
import aiosqlite

from config import Config
from technical_analysis import TechnicalAnalysis

class BinanceTrader:
    """Binance trading interface using ccxt"""
    
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self.exchange = None
        self.technical_analysis = TechnicalAnalysis()
        self.all_futures_symbols = []
        self.trade_log_file = "trade_logs.json"
        self.trade_db_file = "trade_database.db"
        
    async def initialize(self):
        """Initialize Binance exchange connection"""
        try:
            # Use testnet if API keys are empty or testnet is enabled
            use_testnet = (not self.config.BINANCE_API_KEY or 
                          not self.config.BINANCE_API_SECRET or 
                          self.config.BINANCE_TESTNET)
            
            self.exchange = ccxt.binance({
                'apiKey': self.config.BINANCE_API_KEY or 'dummy_key',
                'secret': self.config.BINANCE_API_SECRET or 'dummy_secret',
                'sandbox': use_testnet,
                'timeout': self.config.BINANCE_REQUEST_TIMEOUT * 1000,
                'rateLimit': 1200,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                }
            })
            
            if use_testnet:
                self.logger.info("Using Binance testnet (sandbox mode)")
            
            # Test connection
            await self.exchange.load_markets()
            
            # Load all futures symbols
            await self.load_all_futures_symbols()
            
            # Initialize trade tracking database
            await self.initialize_trade_database()
            
            self.logger.info("Binance futures exchange initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Binance exchange: {e}")
            raise
    
    async def load_all_futures_symbols(self):
        """Load all available futures trading symbols from Binance"""
        try:
            if not self.exchange:
                self.logger.error("Exchange not initialized")
                return
            
            markets = self.exchange.markets
            futures_symbols = []
            
            for symbol, market in markets.items():
                # Filter for futures markets that are active and USDT-settled
                if (market.get('type') == 'future' and 
                    market.get('active', False) and 
                    market.get('quote') == 'USDT' and
                    market.get('settle') == 'USDT'):
                    futures_symbols.append(symbol)
            
            # Sort symbols by volume if available (most active first)
            self.all_futures_symbols = sorted(futures_symbols)
            
            self.logger.info(f"Loaded {len(self.all_futures_symbols)} futures symbols")
            return self.all_futures_symbols
            
        except Exception as e:
            self.logger.error(f"Error loading futures symbols: {e}")
            # Fallback to common futures symbols if API fails
            self.all_futures_symbols = [
                'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT', 'SOL/USDT',
                'DOGE/USDT', 'AVAX/USDT', 'DOT/USDT', 'MATIC/USDT', 'LINK/USDT', 'LTC/USDT'
            ]
            return self.all_futures_symbols
    
    def get_futures_symbols(self) -> List[str]:
        """Get list of all available futures symbols"""
        return self.all_futures_symbols
    
    async def initialize_trade_database(self):
        """Initialize SQLite database for trade tracking"""
        try:
            async with aiosqlite.connect(self.trade_db_file) as db:
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_id TEXT UNIQUE,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        amount REAL NOT NULL,
                        entry_price REAL NOT NULL,
                        exit_price REAL,
                        pnl REAL,
                        pnl_percentage REAL,
                        signal_strength REAL,
                        entry_time TEXT NOT NULL,
                        exit_time TEXT,
                        status TEXT DEFAULT 'open',
                        technical_indicators TEXT,
                        market_conditions TEXT,
                        trade_duration REAL,
                        commission REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS trade_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        total_trades INTEGER DEFAULT 0,
                        winning_trades INTEGER DEFAULT 0,
                        losing_trades INTEGER DEFAULT 0,
                        win_rate REAL DEFAULT 0.0,
                        total_pnl REAL DEFAULT 0.0,
                        avg_win REAL DEFAULT 0.0,
                        avg_loss REAL DEFAULT 0.0,
                        max_win REAL DEFAULT 0.0,
                        max_loss REAL DEFAULT 0.0,
                        profit_factor REAL DEFAULT 0.0,
                        last_updated TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                await db.commit()
                self.logger.info("Trade database initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing trade database: {e}")
    
    async def log_trade_entry(self, trade_data: Dict[str, Any]) -> str:
        """Log trade entry to database and file"""
        try:
            trade_id = f"{trade_data['symbol']}_{int(datetime.now().timestamp())}"
            
            # Log to database
            async with aiosqlite.connect(self.trade_db_file) as db:
                await db.execute('''
                    INSERT INTO trades (
                        trade_id, symbol, side, amount, entry_price, entry_time,
                        signal_strength, technical_indicators, market_conditions
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_id,
                    trade_data['symbol'],
                    trade_data['side'],
                    trade_data['amount'],
                    trade_data['entry_price'],
                    datetime.now().isoformat(),
                    trade_data.get('signal_strength', 0),
                    json.dumps(trade_data.get('technical_indicators', {})),
                    json.dumps(trade_data.get('market_conditions', {}))
                ))
                await db.commit()
            
            # Also log to JSON file for backup
            await self._log_to_json_file('entry', trade_id, trade_data)
            
            self.logger.info(f"Trade entry logged: {trade_id}")
            return trade_id
            
        except Exception as e:
            self.logger.error(f"Error logging trade entry: {e}")
            return ""
    
    async def log_trade_exit(self, trade_id: str, exit_data: Dict[str, Any]):
        """Log trade exit and calculate PnL"""
        try:
            async with aiosqlite.connect(self.trade_db_file) as db:
                # Get trade entry data
                cursor = await db.execute(
                    'SELECT * FROM trades WHERE trade_id = ?', (trade_id,)
                )
                trade = await cursor.fetchone()
                
                if not trade:
                    self.logger.error(f"Trade {trade_id} not found for exit logging")
                    return
                
                # Calculate PnL
                entry_price = trade[5]  # entry_price column
                exit_price = exit_data['exit_price']
                amount = trade[4]  # amount column
                side = trade[3]  # side column
                
                if side.upper() == 'LONG':
                    pnl = (exit_price - entry_price) * amount
                else:  # SHORT
                    pnl = (entry_price - exit_price) * amount
                
                pnl_percentage = (pnl / (entry_price * amount)) * 100
                
                # Calculate trade duration
                entry_time = datetime.fromisoformat(trade[9])  # entry_time column
                exit_time = datetime.now()
                trade_duration = (exit_time - entry_time).total_seconds() / 60  # in minutes
                
                # Update trade record
                await db.execute('''
                    UPDATE trades SET 
                        exit_price = ?, exit_time = ?, pnl = ?, pnl_percentage = ?,
                        status = 'closed', trade_duration = ?, commission = ?
                    WHERE trade_id = ?
                ''', (
                    exit_price,
                    exit_time.isoformat(),
                    pnl,
                    pnl_percentage,
                    trade_duration,
                    exit_data.get('commission', 0),
                    trade_id
                ))
                
                await db.commit()
                
                # Update performance metrics
                await self._update_performance_metrics(trade[2], pnl)  # trade[2] is symbol
                
                # Log to JSON file
                await self._log_to_json_file('exit', trade_id, {
                    **exit_data,
                    'pnl': pnl,
                    'pnl_percentage': pnl_percentage,
                    'trade_duration': trade_duration
                })
                
                self.logger.info(f"Trade exit logged: {trade_id}, PnL: {pnl:.4f} ({pnl_percentage:.2f}%)")
                
        except Exception as e:
            self.logger.error(f"Error logging trade exit: {e}")
    
    async def _update_performance_metrics(self, symbol: str, pnl: float):
        """Update performance metrics for symbol"""
        try:
            async with aiosqlite.connect(self.trade_db_file) as db:
                # Get current performance data
                cursor = await db.execute(
                    'SELECT * FROM trade_performance WHERE symbol = ?', (symbol,)
                )
                perf = await cursor.fetchone()
                
                if perf:
                    # Update existing record
                    total_trades = perf[2] + 1
                    winning_trades = perf[3] + (1 if pnl > 0 else 0)
                    losing_trades = perf[4] + (1 if pnl <= 0 else 0)
                    total_pnl = perf[5] + pnl
                    
                    win_rate = (winning_trades / total_trades) * 100
                    
                    # Calculate averages and extremes
                    if pnl > 0:
                        avg_win = ((perf[6] * perf[3]) + pnl) / winning_trades if winning_trades > 0 else 0
                        max_win = max(perf[8], pnl)
                        avg_loss = perf[7]
                        max_loss = perf[9]
                    else:
                        avg_loss = ((perf[7] * perf[4]) + pnl) / losing_trades if losing_trades > 0 else 0
                        max_loss = min(perf[9], pnl)
                        avg_win = perf[6]
                        max_win = perf[8]
                    
                    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
                    
                    await db.execute('''
                        UPDATE trade_performance SET
                            total_trades = ?, winning_trades = ?, losing_trades = ?,
                            win_rate = ?, total_pnl = ?, avg_win = ?, avg_loss = ?,
                            max_win = ?, max_loss = ?, profit_factor = ?,
                            last_updated = CURRENT_TIMESTAMP
                        WHERE symbol = ?
                    ''', (
                        total_trades, winning_trades, losing_trades, win_rate,
                        total_pnl, avg_win, avg_loss, max_win, max_loss,
                        profit_factor, symbol
                    ))
                else:
                    # Create new record
                    await db.execute('''
                        INSERT INTO trade_performance (
                            symbol, total_trades, winning_trades, losing_trades,
                            win_rate, total_pnl, avg_win, avg_loss, max_win, max_loss
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol, 1, 1 if pnl > 0 else 0, 1 if pnl <= 0 else 0,
                        100 if pnl > 0 else 0, pnl,
                        pnl if pnl > 0 else 0, pnl if pnl <= 0 else 0,
                        pnl if pnl > 0 else 0, pnl if pnl <= 0 else 0
                    ))
                
                await db.commit()
                
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    async def _log_to_json_file(self, action: str, trade_id: str, data: Dict[str, Any]):
        """Log trade data to JSON file for backup"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'action': action,
                'trade_id': trade_id,
                'data': data
            }
            
            # Read existing logs
            if os.path.exists(self.trade_log_file):
                with open(self.trade_log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            # Append new log
            logs.append(log_entry)
            
            # Keep only last 10000 logs to prevent file from getting too large
            if len(logs) > 10000:
                logs = logs[-10000:]
            
            # Write back to file
            with open(self.trade_log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error logging to JSON file: {e}")
    
    async def get_trade_performance(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get trade performance statistics"""
        try:
            async with aiosqlite.connect(self.trade_db_file) as db:
                if symbol:
                    cursor = await db.execute(
                        'SELECT * FROM trade_performance WHERE symbol = ?', (symbol,)
                    )
                    result = await cursor.fetchone()
                    if result:
                        return {
                            'symbol': result[1],
                            'total_trades': result[2],
                            'winning_trades': result[3],
                            'losing_trades': result[4],
                            'win_rate': result[5],
                            'total_pnl': result[6],
                            'avg_win': result[7],
                            'avg_loss': result[8],
                            'max_win': result[9],
                            'max_loss': result[10],
                            'profit_factor': result[11]
                        }
                else:
                    # Get overall performance
                    cursor = await db.execute('SELECT * FROM trade_performance')
                    results = await cursor.fetchall()
                    
                    if results:
                        total_trades = sum(r[2] for r in results)
                        winning_trades = sum(r[3] for r in results)
                        total_pnl = sum(r[6] for r in results)
                        
                        return {
                            'total_trades': total_trades,
                            'winning_trades': winning_trades,
                            'losing_trades': total_trades - winning_trades,
                            'win_rate': (winning_trades / total_trades) * 100 if total_trades > 0 else 0,
                            'total_pnl': total_pnl,
                            'symbols_traded': len(results)
                        }
                
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting trade performance: {e}")
            return {}
    
    async def close(self):
        """Close exchange connection"""
        if self.exchange:
            await self.exchange.close()
    
    async def ping(self) -> bool:
        """Test exchange connectivity"""
        try:
            # Try to fetch server time first (doesn't require API auth)
            await self.exchange.fetch_time()
            self.logger.info("Binance connection successful")
            return True
        except Exception as e:
            self.logger.warning(f"Binance ping failed: {e}")
            try:
                # Fallback: try to fetch ticker for BTCUSDT (public endpoint)
                await self.exchange.fetch_ticker('BTC/USDT')
                self.logger.info("Binance public API accessible")
                return True
            except Exception as e2:
                self.logger.error(f"Binance completely inaccessible: {e2}")
                return False
    
    async def get_account_balance(self) -> Dict[str, Dict[str, float]]:
        """Get account balance for all assets"""
        try:
            balance = await self.exchange.fetch_balance()
            
            # Filter out zero balances and format response
            filtered_balance = {}
            for symbol, data in balance['total'].items():
                if data > 0:
                    filtered_balance[symbol] = {
                        'free': balance['free'].get(symbol, 0),
                        'used': balance['used'].get(symbol, 0),
                        'total': data
                    }
            
            return filtered_balance
            
        except Exception as e:
            self.logger.error(f"Error getting account balance: {e}")
            raise
    
    async def get_portfolio_value(self) -> float:
        """Calculate total portfolio value in USDT"""
        try:
            balance = await self.get_account_balance()
            total_value = 0.0
            
            for symbol, data in balance.items():
                if symbol == 'USDT':
                    total_value += data['total']
                else:
                    # Convert to USDT value
                    try:
                        ticker_symbol = f"{symbol}/USDT"
                        if ticker_symbol in self.exchange.markets:
                            ticker = await self.exchange.fetch_ticker(ticker_symbol)
                            total_value += data['total'] * ticker['last']
                    except Exception:
                        # Skip if can't get price
                        continue
            
            return total_value
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio value: {e}")
            return 0.0
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current market price for symbol"""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            price = ticker.get('last') or ticker.get('close') or ticker.get('price')
            
            if price is None:
                self.logger.warning(f"No price data available for {symbol}")
                return 0.0
                
            return float(price)
            
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
            return 0.0
    
    async def get_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> List[List]:
        """Get OHLCV market data"""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Validate the data
            if not ohlcv or len(ohlcv) == 0:
                self.logger.warning(f"No OHLCV data returned for {symbol} {timeframe}")
                return []
                
            # Check if data contains valid values
            valid_data = []
            for candle in ohlcv:
                if len(candle) >= 6 and all(x is not None for x in candle[:6]):
                    valid_data.append(candle)
            
            if len(valid_data) == 0:
                self.logger.warning(f"No valid OHLCV data for {symbol} {timeframe}")
                return []
                
            return valid_data
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol} {timeframe}: {e}")
            return []
    
    async def get_technical_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get technical analysis for symbol"""
        try:
            # Get market data
            ohlcv_1h = await self.get_market_data(symbol, '1h', 100)
            ohlcv_4h = await self.get_market_data(symbol, '4h', 100)
            ohlcv_1d = await self.get_market_data(symbol, '1d', 50)
            
            # Check if we have valid market data
            if not ohlcv_1h and not ohlcv_4h and not ohlcv_1d:
                self.logger.warning(f"No market data available for {symbol}")
                return {'symbol': symbol, 'error': 'No market data available'}
            
            # Get 24h price change with fallback
            try:
                ticker = await self.exchange.fetch_ticker(symbol)
                price_change_24h = ticker.get('percentage', 0)
                volume = ticker.get('baseVolume', 0)
            except Exception as ticker_error:
                self.logger.warning(f"Error getting ticker data for {symbol}: {ticker_error}")
                price_change_24h = 0
                volume = 0
            
            # Calculate technical indicators
            analysis = await self.technical_analysis.analyze(
                ohlcv_1h, ohlcv_4h, ohlcv_1d
            )
            
            # Only add these if analysis was successful
            if 'error' not in analysis:
                analysis['price_change_24h'] = price_change_24h
                analysis['volume'] = volume
                analysis['symbol'] = symbol
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error getting technical analysis for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    async def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get open futures positions (if using futures)"""
        try:
            # For spot trading, we'll return current holdings as "positions"
            balance = await self.get_account_balance()
            positions = []
            
            for symbol, data in balance.items():
                if symbol != 'USDT' and data['total'] > 0:
                    try:
                        ticker_symbol = f"{symbol}/USDT"
                        if ticker_symbol in self.exchange.markets:
                            ticker = await self.exchange.fetch_ticker(ticker_symbol)
                            current_price = ticker['last']
                            
                            # Estimate unrealized PnL (simplified)
                            # This would be more accurate with actual entry prices
                            estimated_value = data['total'] * current_price
                            
                            positions.append({
                                'symbol': ticker_symbol,
                                'side': 'LONG',  # Spot holdings are always long
                                'size': data['total'],
                                'entryPrice': current_price,  # Simplified
                                'markPrice': current_price,
                                'unrealizedPnl': 0,  # Would need trade history for accurate PnL
                                'value': estimated_value
                            })
                    except Exception:
                        continue
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Error getting open positions: {e}")
            return []
    
    async def execute_trade(self, signal: Dict[str, Any], user_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trade based on signal and user settings"""
        try:
            symbol = signal['symbol']
            action = signal['action'].upper()
            
            # Calculate position size
            position_size = await self._calculate_position_size(signal, user_settings)
            
            if position_size <= 0:
                return {
                    'success': False,
                    'error': 'Invalid position size calculated'
                }
            
            # Determine order type and price
            order_type = 'market'  # Default to market orders
            price = None
            
            if 'price' in signal and user_settings.get('use_limit_orders', False):
                order_type = 'limit'
                price = signal['price']
            
            # Execute the trade
            if action in ['BUY', 'LONG']:
                order = await self._execute_buy_order(symbol, position_size, order_type, price)
            elif action in ['SELL', 'SHORT']:
                order = await self._execute_sell_order(symbol, position_size, order_type, price)
            else:
                return {
                    'success': False,
                    'error': f'Unsupported action: {action}'
                }
            
            if order:
                # Set stop loss and take profit if specified
                await self._set_stop_loss_take_profit(order, signal, user_settings)
                
                return {
                    'success': True,
                    'order_id': order['id'],
                    'symbol': symbol,
                    'side': action,
                    'amount': order['amount'],
                    'price': order.get('price', order.get('average', 0)),
                    'fee': order.get('fee', {}),
                    'timestamp': order['timestamp']
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to execute order'
                }
                
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _calculate_position_size(self, signal: Dict[str, Any], user_settings: Dict[str, Any]) -> float:
        """Calculate position size based on risk management"""
        try:
            # Get account balance
            balance = await self.get_account_balance()
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            
            # Get risk percentage from user settings
            risk_percentage = user_settings.get('risk_percentage', self.config.DEFAULT_RISK_PERCENTAGE)
            
            # Calculate risk amount
            risk_amount = usdt_balance * (risk_percentage / 100)
            
            # Get current price
            symbol = signal['symbol']
            current_price = await self.get_current_price(symbol)
            
            # If specific quantity is provided in signal, use it (with limits)
            if 'quantity' in signal:
                quantity = float(signal['quantity'])
                max_quantity = risk_amount / current_price
                return min(quantity, max_quantity)
            
            # Calculate position size based on stop loss
            if 'stop_loss' in signal:
                stop_loss = float(signal['stop_loss'])
                entry_price = signal.get('price', current_price)
                
                # Calculate risk per unit
                if signal['action'].upper() in ['BUY', 'LONG']:
                    risk_per_unit = abs(entry_price - stop_loss)
                else:
                    risk_per_unit = abs(stop_loss - entry_price)
                
                if risk_per_unit > 0:
                    quantity = risk_amount / risk_per_unit
                    # Convert to base currency quantity
                    position_size = min(quantity, risk_amount / current_price)
                else:
                    position_size = risk_amount / current_price
            else:
                # No stop loss specified, use full risk amount
                position_size = risk_amount / current_price
            
            # Apply position size limits
            max_position = user_settings.get('max_position_size', self.config.MAX_POSITION_SIZE)
            min_position = user_settings.get('min_position_size', self.config.MIN_POSITION_SIZE)
            
            position_value = position_size * current_price
            
            if position_value > max_position:
                position_size = max_position / current_price
            elif position_value < min_position:
                position_size = min_position / current_price
            
            # Round to appropriate precision
            market = self.exchange.markets.get(symbol, {})
            precision = market.get('precision', {}).get('amount', 8)
            
            return float(Decimal(str(position_size)).quantize(
                Decimal('0.' + '0' * (precision - 1) + '1'),
                rounding=ROUND_DOWN
            ))
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    async def _execute_buy_order(self, symbol: str, amount: float, order_type: str, price: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Execute a buy order"""
        try:
            if order_type == 'market':
                order = await self.exchange.create_market_buy_order(symbol, amount)
            else:
                order = await self.exchange.create_limit_buy_order(symbol, amount, price)
            
            self.logger.info(f"Buy order executed: {order['id']} for {amount} {symbol}")
            return order
            
        except Exception as e:
            self.logger.error(f"Error executing buy order: {e}")
            return None
    
    async def _execute_sell_order(self, symbol: str, amount: float, order_type: str, price: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Execute a sell order"""
        try:
            if order_type == 'market':
                order = await self.exchange.create_market_sell_order(symbol, amount)
            else:
                order = await self.exchange.create_limit_sell_order(symbol, amount, price)
            
            self.logger.info(f"Sell order executed: {order['id']} for {amount} {symbol}")
            return order
            
        except Exception as e:
            self.logger.error(f"Error executing sell order: {e}")
            return None
    
    async def _set_stop_loss_take_profit(self, order: Dict[str, Any], signal: Dict[str, Any], user_settings: Dict[str, Any]):
        """Set stop loss and take profit orders"""
        try:
            symbol = order['symbol']
            amount = order['amount']
            side = 'sell' if signal['action'].upper() in ['BUY', 'LONG'] else 'buy'
            
            # Set stop loss
            if 'stop_loss' in signal:
                stop_loss_price = float(signal['stop_loss'])
                try:
                    stop_order = await self.exchange.create_order(
                        symbol=symbol,
                        type='stop_market',
                        side=side,
                        amount=amount,
                        params={'stopPrice': stop_loss_price}
                    )
                    self.logger.info(f"Stop loss set at {stop_loss_price} for order {order['id']}")
                except Exception as e:
                    self.logger.warning(f"Failed to set stop loss: {e}")
            
            # Set take profit
            if 'take_profit' in signal:
                take_profit_price = float(signal['take_profit'])
                try:
                    tp_order = await self.exchange.create_limit_order(
                        symbol=symbol,
                        side=side,
                        amount=amount,
                        price=take_profit_price
                    )
                    self.logger.info(f"Take profit set at {take_profit_price} for order {order['id']}")
                except Exception as e:
                    self.logger.warning(f"Failed to set take profit: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error setting stop loss/take profit: {e}")
    
    async def get_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Get order status"""
        try:
            order = await self.exchange.fetch_order(order_id, symbol)
            return order
            
        except Exception as e:
            self.logger.error(f"Error getting order status: {e}")
            return {}
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order"""
        try:
            await self.exchange.cancel_order(order_id, symbol)
            self.logger.info(f"Order {order_id} cancelled successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def get_trading_fees(self, symbol: str) -> Dict[str, float]:
        """Get trading fees for symbol"""
        try:
            fees = await self.exchange.fetch_trading_fee(symbol)
            return {
                'maker': fees.get('maker', 0.001),
                'taker': fees.get('taker', 0.001)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting trading fees: {e}")
            return {'maker': 0.001, 'taker': 0.001}
    
    async def get_market_summary(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get market summary for multiple symbols"""
        try:
            summaries = {}
            
            for symbol in symbols:
                try:
                    ticker = await self.exchange.fetch_ticker(symbol)
                    summaries[symbol] = {
                        'price': ticker['last'],
                        'change_24h': ticker['percentage'],
                        'volume': ticker['baseVolume'],
                        'high_24h': ticker['high'],
                        'low_24h': ticker['low']
                    }
                except Exception:
                    continue
            
            return summaries
            
        except Exception as e:
            self.logger.error(f"Error getting market summary: {e}")
            return {}
#!/usr/bin/env python3
"""
Binance Trading Integration
"""

import asyncio
import logging
import aiohttp
import json
from typing import Dict, Any, Optional, List

class BinanceTrader:
    """Binance trading interface"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.api_url = "https://api.binance.com"
        self.api_key = None
        self.api_secret = None
        
    async def test_connection(self) -> bool:
        """Test Binance API connection"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/api/v3/ping") as response:
                    return response.status == 200
        except:
            return False
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/api/v3/ticker/price?symbol={symbol}") as response:
                    if response.status == 200:
                        data = await response.json()
                        return float(data['price'])
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
        return None
    
    async def get_multi_timeframe_data(self, symbol: str, timeframes: List[str], limit: int = 100) -> Dict[str, List]:
        """Get OHLCV data for multiple timeframes"""
        try:
            data = {}
            
            for tf in timeframes:
                # Convert timeframe format
                binance_tf = self._convert_timeframe(tf)
                
                async with aiohttp.ClientSession() as session:
                    url = f"{self.api_url}/api/v3/klines?symbol={symbol}&interval={binance_tf}&limit={limit}"
                    async with session.get(url) as response:
                        if response.status == 200:
                            klines = await response.json()
                            # Convert to OHLCV format
                            ohlcv = [[
                                kline[0],  # timestamp
                                float(kline[1]),  # open
                                float(kline[2]),  # high
                                float(kline[3]),  # low
                                float(kline[4]),  # close
                                float(kline[5])   # volume
                            ] for kline in klines]
                            data[tf] = ohlcv
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error getting multi-timeframe data: {e}")
            return {}
    
    def _convert_timeframe(self, tf: str) -> str:
        """Convert timeframe to Binance format"""
        mapping = {
            '1m': '1m',
            '3m': '3m',
            '5m': '5m',
            '15m': '15m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }
        return mapping.get(tf, '5m')
    
    async def get_ohlcv_data(self, symbol: str, timeframe: str = '5m', limit: int = 100) -> List[List]:
        """Get OHLCV data for a single timeframe"""
        try:
            binance_tf = self._convert_timeframe(timeframe)
            async with aiohttp.ClientSession() as session:
                url = f"{self.api_url}/api/v3/klines?symbol={symbol}&interval={binance_tf}&limit={limit}"
                async with session.get(url) as response:
                    if response.status == 200:
                        klines = await response.json()
                        # Convert to OHLCV format
                        ohlcv = [[
                            kline[0],  # timestamp
                            float(kline[1]),  # open
                            float(kline[2]),  # high
                            float(kline[3]),  # low
                            float(kline[4]),  # close
                            float(kline[5])   # volume
                        ] for kline in klines]
                        return ohlcv
            return []
        except Exception as e:
            self.logger.error(f"Error getting OHLCV data for {symbol} {timeframe}: {e}")
            return []
    
    async def close(self):
        """Close any connections"""
        # Placeholder for closing connections if needed
        pass
