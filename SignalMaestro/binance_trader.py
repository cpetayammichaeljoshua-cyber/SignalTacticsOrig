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

from config import Config
from technical_analysis import TechnicalAnalysis

class BinanceTrader:
    """Binance trading interface using ccxt"""
    
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self.exchange = None
        self.technical_analysis = TechnicalAnalysis()
        
    async def initialize(self):
        """Initialize Binance exchange connection"""
        try:
            self.exchange = ccxt.binance({
                'apiKey': self.config.BINANCE_API_KEY,
                'secret': self.config.BINANCE_API_SECRET,
                'sandbox': False,  # Set to True for testnet
                'timeout': self.config.BINANCE_REQUEST_TIMEOUT * 1000,
                'rateLimit': 1200,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',  # 'spot', 'margin', 'future'
                }
            })
            
            # Test connection
            await self.exchange.load_markets()
            self.logger.info("Binance exchange initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Binance exchange: {e}")
            raise
    
    async def close(self):
        """Close exchange connection"""
        if self.exchange:
            await self.exchange.close()
    
    async def ping(self) -> bool:
        """Test exchange connectivity"""
        try:
            await self.exchange.fetch_status()
            return True
        except Exception as e:
            self.logger.warning(f"Binance ping failed: {e}")
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
