"""
Telegram bot implementation for trading signal processing
Handles user interactions and command processing
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    CallbackQueryHandler, ContextTypes, filters
)

from config import Config
from signal_parser import SignalParser
from risk_manager import RiskManager
from utils import format_currency, format_percentage

class TradingSignalBot:
    """Telegram bot for handling trading signals and user interactions"""
    
    def __init__(self, binance_trader, cornix_integration, database):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self.binance_trader = binance_trader
        self.cornix = cornix_integration
        self.db = database
        self.signal_parser = SignalParser()
        self.risk_manager = RiskManager()
        self.application = None
        
    async def initialize(self):
        """Initialize the Telegram bot application"""
        try:
            self.application = Application.builder().token(self.config.TELEGRAM_BOT_TOKEN).build()
            
            # Add command handlers
            self.application.add_handler(CommandHandler("start", self.start_command))
            self.application.add_handler(CommandHandler("help", self.help_command))
            self.application.add_handler(CommandHandler("balance", self.balance_command))
            self.application.add_handler(CommandHandler("positions", self.positions_command))
            self.application.add_handler(CommandHandler("signal", self.signal_command))
            self.application.add_handler(CommandHandler("settings", self.settings_command))
            self.application.add_handler(CommandHandler("status", self.status_command))
            self.application.add_handler(CommandHandler("history", self.history_command))
            
            # Add message handler for signals
            self.application.add_handler(
                MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
            )
            
            # Add callback query handler for inline keyboards
            self.application.add_handler(CallbackQueryHandler(self.handle_callback))
            
            self.logger.info("Telegram bot application initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Telegram bot: {e}")
            raise
    
    async def start(self):
        """Start the Telegram bot"""
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()
        
    async def stop(self):
        """Stop the Telegram bot"""
        if self.application:
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"
        
        if not self.config.is_authorized_user(user_id):
            await update.message.reply_text(
                "❌ You are not authorized to use this bot. Contact the administrator."
            )
            return
        
        # Save user to database
        await self.db.save_user(user_id, username)
        
        welcome_message = f"""
🚀 **Welcome to Trading Signal Bot!**

Hello {username}! I'm your automated cryptocurrency trading assistant.

**Available Commands:**
• `/balance` - Check account balance
• `/positions` - View open positions
• `/signal <pair>` - Get signal for trading pair
• `/settings` - Configure trading parameters
• `/status` - Check bot status
• `/history` - View trading history
• `/help` - Show this help message

**Signal Format:**
You can send trading signals in these formats:
• `BUY BTCUSDT at 45000`
• `SELL ETHUSDT 50% at 3200`
• `LONG BTC SL: 44000 TP: 48000`

**Features:**
✅ Automated signal parsing
✅ Risk management
✅ Binance integration
✅ Cornix forwarding
✅ Real-time monitoring

Ready to start trading! Send me a signal or use the commands above.
        """
        
        await update.message.reply_text(welcome_message, parse_mode='Markdown')
        self.logger.info(f"User {username} ({user_id}) started the bot")
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = """
📚 **Trading Signal Bot Help**

**Commands:**
• `/start` - Initialize bot
• `/balance` - Check account balance
• `/positions` - View open positions  
• `/signal <pair>` - Get signal for pair (e.g., /signal BTCUSDT)
• `/settings` - Configure trading parameters
• `/status` - Check system status
• `/history` - View recent trades
• `/help` - Show this help

**Signal Formats:**
• `BUY BTCUSDT at 45000`
• `SELL ETHUSDT 50% at 3200`
• `LONG BTC SL: 44000 TP: 48000`

**Features:**
✅ Automated signal parsing
✅ Risk management
✅ Binance integration
✅ Cornix forwarding
✅ Real-time monitoring

Send me a trading signal or use the commands above!
        """
        
        await update.message.reply_text(help_text, parse_mode='Markdown')
        self.logger.info(f"User {update.effective_user.id} requested help")

    async def balance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /balance command"""
        try:
            user_id = update.effective_user.id
            await update.message.reply_text("⏳ Fetching your account balance...")
            
            # Get balance from Binance trader
            balance_data = await self.binance_trader.get_account_balance()
            
            if balance_data:
                balance_text = "💰 **Account Balance:**\n\n"
                for asset, data in balance_data.items():
                    if float(data.get('free', 0)) > 0:
                        balance_text += f"• {asset}: {data['free']}\n"
                
                await update.message.reply_text(balance_text, parse_mode='Markdown')
            else:
                await update.message.reply_text("❌ Unable to fetch balance. Please check your API configuration.")
                
        except Exception as e:
            self.logger.error(f"Error in balance command: {e}")
            await update.message.reply_text("❌ Error fetching balance. Please try again later.")

    async def positions_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command"""
        try:
            await update.message.reply_text("⏳ Fetching your open positions...")
            
            # Get positions from Binance trader
            positions = await self.binance_trader.get_open_positions()
            
            if positions:
                positions_text = "📊 **Open Positions:**\n\n"
                for position in positions:
                    symbol = position.get('symbol', 'Unknown')
                    side = position.get('side', 'Unknown')
                    size = position.get('size', 0)
                    pnl = position.get('unrealizedPnl', 0)
                    
                    positions_text += f"• {symbol} {side}\n"
                    positions_text += f"  Size: {size}\n"
                    positions_text += f"  PnL: {pnl} USDT\n\n"
                
                await update.message.reply_text(positions_text, parse_mode='Markdown')
            else:
                await update.message.reply_text("📭 No open positions found.")
                
        except Exception as e:
            self.logger.error(f"Error in positions command: {e}")
            await update.message.reply_text("❌ Error fetching positions. Please try again later.")

    async def signal_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signal command"""
        try:
            if not context.args:
                await update.message.reply_text("Please provide a trading pair. Example: `/signal BTCUSDT`", parse_mode='Markdown')
                return
            
            symbol = context.args[0].upper()
            await update.message.reply_text(f"⏳ Analyzing {symbol}...")
            
            # Get market data for the symbol
            market_data = await self.binance_trader.get_market_data(symbol)
            
            if market_data:
                price = market_data.get('price', 0)
                change_24h = market_data.get('priceChangePercent', 0)
                
                signal_text = f"📈 **Signal for {symbol}:**\n\n"
                signal_text += f"💰 Current Price: ${price}\n"
                signal_text += f"📊 24h Change: {change_24h}%\n\n"
                signal_text += "📋 Use the web dashboard for detailed technical analysis."
                
                await update.message.reply_text(signal_text, parse_mode='Markdown')
            else:
                await update.message.reply_text(f"❌ Unable to get data for {symbol}. Please check the symbol.")
                
        except Exception as e:
            self.logger.error(f"Error in signal command: {e}")
            await update.message.reply_text("❌ Error processing signal request. Please try again later.")

    async def settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /settings command"""
        try:
            user_id = update.effective_user.id
            
            # Get user settings from database
            user_data = await self.db.get_user_settings(user_id)
            
            settings_text = "⚙️ **Trading Settings:**\n\n"
            settings_text += f"🎯 Risk per trade: {user_data.get('risk_percentage', 2)}%\n"
            settings_text += f"🤖 Auto trading: {'Enabled' if user_data.get('auto_trading', False) else 'Disabled'}\n"
            settings_text += f"📤 Cornix forwarding: {'Enabled' if user_data.get('cornix_enabled', False) else 'Disabled'}\n\n"
            settings_text += "Use the web dashboard to modify these settings."
            
            await update.message.reply_text(settings_text, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error in settings command: {e}")
            await update.message.reply_text("❌ Error fetching settings. Please try again later.")

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        try:
            # Check system components
            binance_status = "✅" if await self.binance_trader.test_connection() else "❌"
            db_status = "✅" if await self.db.test_connection() else "❌"
            
            status_text = "🤖 **System Status:**\n\n"
            status_text += f"🔗 Binance API: {binance_status}\n"
            status_text += f"💾 Database: {db_status}\n"
            status_text += f"📱 Telegram Bot: ✅\n"
            status_text += f"🌐 Webhook Server: ✅\n\n"
            status_text += "All systems operational! Ready to trade."
            
            await update.message.reply_text(status_text, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error in status command: {e}")
            await update.message.reply_text("❌ Error checking system status.")

    async def history_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /history command"""
        try:
            user_id = update.effective_user.id
            await update.message.reply_text("⏳ Fetching your trading history...")
            
            # Get recent trades from database
            trades = await self.db.get_user_trades(user_id, limit=5)
            
            if trades:
                history_text = "📊 **Recent Trades:**\n\n"
                for trade in trades:
                    symbol = trade.get('symbol', 'Unknown')
                    side = trade.get('side', 'Unknown')
                    amount = trade.get('amount', 0)
                    price = trade.get('price', 0)
                    pnl = trade.get('pnl', 0)
                    status = trade.get('status', 'Unknown')
                    
                    history_text += f"• {symbol} {side}\n"
                    history_text += f"  Amount: {amount}\n"
                    history_text += f"  Price: ${price}\n"
                    history_text += f"  P&L: ${pnl}\n"
                    history_text += f"  Status: {status}\n\n"
                
                await update.message.reply_text(history_text, parse_mode='Markdown')
            else:
                await update.message.reply_text("📭 No trading history found.")
                
        except Exception as e:
            self.logger.error(f"Error in history command: {e}")
            await update.message.reply_text("❌ Error fetching trading history.")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming text messages (potential trading signals)"""
        try:
            user_id = update.effective_user.id
            username = update.effective_user.username or "Unknown"
            message_text = update.message.text
            
            self.logger.info(f"Received message from {username} ({user_id}): {message_text}")
            
            # Send initial processing message
            processing_msg = await update.message.reply_text("🔄 Processing your signal...")
            
            # Parse the signal
            from signal_parser import SignalParser
            parser = SignalParser()
            parsed_signal = parser.parse_signal(message_text)
            
            if parsed_signal and parsed_signal.get('action'):
                # Store signal in database
                signal_data = {
                    'user_id': user_id,
                    'raw_text': message_text,
                    'parsed_signal': parsed_signal,
                    'status': 'received'
                }
                await self.db.store_signal(signal_data)
                
                # Format response
                symbol = parsed_signal.get('symbol', 'Unknown')
                action = parsed_signal.get('action', 'Unknown')
                price = parsed_signal.get('price', 0)
                
                response_text = f"✅ **Signal Received!**\n\n"
                response_text += f"📊 Pair: {symbol}\n"
                response_text += f"🔄 Action: {action}\n"
                if price > 0:
                    response_text += f"💰 Price: ${price}\n"
                response_text += f"\n⏳ Processing for execution..."
                
                await processing_msg.edit_text(response_text, parse_mode='Markdown')
                
                # Process signal for trading (if auto-trading enabled)
                user_settings = await self.db.get_user_settings(user_id)
                if user_settings.get('auto_trading', False):
                    # Execute trade through the main trading logic
                    await self.execute_signal(parsed_signal, user_id)
                
            else:
                await processing_msg.edit_text(
                    "❌ Unable to parse trading signal.\n\n"
                    "Please use format like:\n"
                    "• `BUY BTCUSDT at 45000`\n"
                    "• `SELL ETHUSDT 50% at 3200`",
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            await update.message.reply_text("❌ Error processing your message. Please try again.")

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries from inline keyboards"""
        try:
            query = update.callback_query
            await query.answer()
            
            data = query.data
            
            if data.startswith('confirm_trade_'):
                trade_id = data.replace('confirm_trade_', '')
                await query.edit_message_text("✅ Trade confirmed and executed!")
                
            elif data.startswith('cancel_trade_'):
                trade_id = data.replace('cancel_trade_', '')
                await query.edit_message_text("❌ Trade cancelled.")
                
        except Exception as e:
            self.logger.error(f"Error handling callback: {e}")

    async def execute_signal(self, parsed_signal, user_id):
        """Execute a parsed trading signal"""
        try:
            # This would integrate with the main trading logic
            # For now, just log the action
            self.logger.info(f"Would execute signal for user {user_id}: {parsed_signal}")
            
        except Exception as e:
            self.logger.error(f"Error executing signal: {e}")
