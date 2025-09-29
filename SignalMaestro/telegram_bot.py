"""
Telegram bot implementation for trading signal processing
Handles user interactions and command processing
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    CallbackQueryHandler, ContextTypes, filters
)

from config import Config
from signal_parser import SignalParser
from risk_manager import RiskManager
# from utils import format_currency, format_percentage  # Commented out due to import issues
from telegram_strategy_comparison import TelegramStrategyComparison

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
        
        # Initialize strategy comparison service
        self.strategy_comparison = TelegramStrategyComparison(self.config)
        
        # Initialize metrics manager
        self.metrics_manager = None
        
    async def initialize(self):
        """Initialize the Telegram bot application"""
        try:
            if not self.config.TELEGRAM_BOT_TOKEN:
                raise ValueError("TELEGRAM_BOT_TOKEN is not configured")
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
            
            # Add comprehensive metrics commands
            self.application.add_handler(CommandHandler("metrics", self.metrics_command))
            self.application.add_handler(CommandHandler("performance", self.performance_command))
            self.application.add_handler(CommandHandler("stats", self.stats_command))
            self.application.add_handler(CommandHandler("winrate", self.winrate_command))
            self.application.add_handler(CommandHandler("pnl", self.pnl_command))
            self.application.add_handler(CommandHandler("streaks", self.streaks_command))
            self.application.add_handler(CommandHandler("pairs", self.pairs_performance_command))
            self.application.add_handler(CommandHandler("hourly", self.hourly_performance_command))
            self.application.add_handler(CommandHandler("risk", self.risk_metrics_command))
            self.application.add_handler(CommandHandler("daily", self.daily_comparison_command))
            
            # Add strategy comparison commands
            self.application.add_handler(CommandHandler("strategies", self.strategies_command))
            self.application.add_handler(CommandHandler("compare_run", self.compare_run_command))
            self.application.add_handler(CommandHandler("compare_status", self.compare_status_command))
            self.application.add_handler(CommandHandler("compare_result", self.compare_result_command))
            self.application.add_handler(CommandHandler("compare_recent", self.compare_recent_command))
            self.application.add_handler(CommandHandler("compare_rankings", self.compare_rankings_command))
            self.application.add_handler(CommandHandler("compare_tips", self.compare_tips_command))
            self.application.add_handler(CommandHandler("compare_help", self.compare_help_command))
            
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
            if update.message:
                if update.message:
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

**📊 Performance Metrics:**
• `/metrics` - Complete performance overview
• `/performance` - Detailed performance analysis
• `/stats` - Quick statistics summary
• `/winrate` - Win rate and streak analysis
• `/pnl` - Profit & loss breakdown
• `/streaks` - Winning/losing streak details
• `/pairs` - Performance by trading pairs
• `/hourly` - Success rate by hour
• `/risk` - Risk metrics and analysis
• `/daily` - Daily performance comparison

**Strategy Comparison:**
• `/strategies` - List available strategies
• `/compare_run` - Start strategy comparison
• `/compare_recent` - Show recent comparisons
• `/compare_help` - Strategy comparison help

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
        
        if update.message:
            if update.message:
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

**📊 Performance Metrics:**
• `/metrics` - Complete performance overview
• `/performance` - Detailed performance analysis
• `/stats` - Quick statistics summary
• `/winrate` - Win rate and streak analysis
• `/pnl` - Profit & loss breakdown
• `/streaks` - Winning/losing streak details
• `/pairs` - Performance by trading pairs
• `/hourly` - Success rate by hour
• `/risk` - Risk metrics and analysis
• `/daily` - Daily performance comparison

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
        
        if update.message:
            if update.message:
                await update.message.reply_text(help_text, parse_mode='Markdown')
        self.logger.info(f"User {update.effective_user.id} requested help")

    async def balance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /balance command"""
        try:
            if not update.effective_user:
                return
            user_id = update.effective_user.id
            if update.message:
                if update.message:
                await update.message.reply_text("⏳ Fetching your account balance...")
            
            # Get balance from Binance trader
            balance_data = await self.binance_trader.get_account_balance()
            
            if balance_data:
                balance_text = "💰 **Account Balance:**\n\n"
                for asset, data in balance_data.items():
                    if float(data.get('free', 0)) > 0:
                        balance_text += f"• {asset}: {data['free']}\n"
                
                if update.message:
                    if update.message:
                await update.message.reply_text(balance_text, parse_mode='Markdown')
            else:
                if update.message:
                    if update.message:
                await update.message.reply_text("❌ Unable to fetch balance. Please check your API configuration.")
                
        except Exception as e:
            self.logger.error(f"Error in balance command: {e}")
            if update.message:
                if update.message:
                await update.message.reply_text("❌ Error fetching balance. Please try again later.")

    async def positions_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command"""
        try:
            if update.message:
                if update.message:
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
                
                if update.message:
                    if update.message:
                await update.message.reply_text(positions_text, parse_mode='Markdown')
            else:
                if update.message:
                    if update.message:
                await update.message.reply_text("📭 No open positions found.")
                
        except Exception as e:
            self.logger.error(f"Error in positions command: {e}")
            if update.message:
                if update.message:
                await update.message.reply_text("❌ Error fetching positions. Please try again later.")

    async def signal_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signal command"""
        try:
            if not context.args:
                if update.message:
                    if update.message:
                await update.message.reply_text("Please provide a trading pair. Example: `/signal BTCUSDT`", parse_mode='Markdown')
                return
            
            symbol = context.args[0].upper()
            if update.message:
                if update.message:
                await update.message.reply_text(f"⏳ Analyzing {symbol}...")
            
            # Get market data for the symbol
            market_data = await self.binance_trader.get_market_data(symbol)
            
            if market_data:
                price = market_data.get('price', 0)
                change_24h = market_data.get('priceChangePercent', 0)
                
                signal_text = f"📈 **Signal for {symbol}:**\n\n"
                signal_text += f"💰 Current Price: ${price}\n"
                signal_text += f"📊 24h Change: {change_24h}%\n\n"
                signal_text += "📊 Use `/compare_run` to backtest strategies for this symbol."
                
                if update.message:
                    if update.message:
                await update.message.reply_text(signal_text, parse_mode='Markdown')
            else:
                if update.message:
                    if update.message:
                await update.message.reply_text(f"❌ Unable to get data for {symbol}. Please check the symbol.")
                
        except Exception as e:
            self.logger.error(f"Error in signal command: {e}")
            if update.message:
                if update.message:
                await update.message.reply_text("❌ Error processing signal request. Please try again later.")

    async def settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /settings command"""
        try:
            if not update.effective_user:
                return
            user_id = update.effective_user.id
            
            # Get user settings from database
            user_data = await self.db.get_user_settings(user_id)
            
            settings_text = "⚙️ **Trading Settings:**\n\n"
            settings_text += f"🎯 Risk per trade: {user_data.get('risk_percentage', 2)}%\n"
            settings_text += f"🤖 Auto trading: {'Enabled' if user_data.get('auto_trading', False) else 'Disabled'}\n"
            settings_text += f"📤 Cornix forwarding: {'Enabled' if user_data.get('cornix_enabled', False) else 'Disabled'}\n\n"
            settings_text += "Contact your administrator to modify these settings."
            
            if update.message:
                if update.message:
                await update.message.reply_text(settings_text, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error in settings command: {e}")
            if update.message:
                if update.message:
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
            
            if update.message:
                if update.message:
                await update.message.reply_text(status_text, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error in status command: {e}")
            if update.message:
                if update.message:
                await update.message.reply_text("❌ Error checking system status.")

    async def history_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /history command"""
        try:
            if not update.effective_user:
                return
            user_id = update.effective_user.id
            if update.message:
                if update.message:
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
                
                if update.message:
                    if update.message:
                await update.message.reply_text(history_text, parse_mode='Markdown')
            else:
                if update.message:
                    if update.message:
                await update.message.reply_text("📭 No trading history found.")
                
        except Exception as e:
            self.logger.error(f"Error in history command: {e}")
            if update.message:
                if update.message:
                await update.message.reply_text("❌ Error fetching trading history.")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming text messages (potential trading signals)"""
        try:
            if not update.effective_user:
                return
            user_id = update.effective_user.id
            username = update.effective_user.username or "Unknown"
            if not update.message or not update.message.text:
                return
            message_text = update.message.text
            
            self.logger.info(f"Received message from {username} ({user_id}): {message_text}")
            
            # Send initial processing message
            if not update.message:
                return
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
            if update.message:
                if update.message:
                await update.message.reply_text("❌ Error processing your message. Please try again.")

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries from inline keyboards"""
        try:
            query = update.callback_query
            if not query:
                return
            await query.answer()
            
            data = query.data
            if not data:
                return
            
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
    
    # ======== Strategy Comparison Commands ========
    
    async def strategies_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /strategies command - list available strategies"""
        try:
            if not update.effective_user:
                return
            user_id = update.effective_user.id
            
            if not self.config.is_authorized_user(user_id):
                if update.message:
                    if update.message:
                await update.message.reply_text("❌ You are not authorized to use this bot.")
                return
            
            if update.message:
                if update.message:
                await update.message.reply_text("⏳ Loading available strategies...")
            
            strategies_text, total_count = await self.strategy_comparison.get_available_strategies()
            
            if total_count == 0:
                if update.message:
                    if update.message:
                await update.message.reply_text(strategies_text, parse_mode='Markdown')
            else:
                footer = f"\n\n📱 Use `/compare_run` to start a comparison"
                final_text = strategies_text + footer
                
                if len(final_text) > 4096:
                    # Send in chunks if too long
                    if update.message:
                        if update.message:
                await update.message.reply_text(strategies_text[:4090] + "...", parse_mode='Markdown')
                        if update.message:
                await update.message.reply_text(footer, parse_mode='Markdown')
                else:
                    if update.message:
                        if update.message:
                await update.message.reply_text(final_text, parse_mode='Markdown')
                    
        except Exception as e:
            self.logger.error(f"Error in strategies command: {e}")
            if update.message:
                if update.message:
                await update.message.reply_text("❌ Error loading strategies. Please try again later.")
    
    async def compare_run_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /compare_run command - start new strategy comparison"""
        try:
            if not update.effective_user:
                return
            user_id = update.effective_user.id
            
            if not self.config.is_authorized_user(user_id):
                if update.message:
                    if update.message:
                await update.message.reply_text("❌ You are not authorized to use this bot.")
                return
            
            # Parse arguments
            symbols = None
            days = 30
            strategies = None
            initial_capital = 10.0
            
            if context.args:
                try:
                    # Parse first argument (symbols)
                    if context.args[0].lower() != 'default':
                        symbols = [s.strip().upper() for s in context.args[0].split(',')]
                    
                    # Parse second argument (days)
                    if len(context.args) > 1:
                        days = int(context.args[1])
                    
                    # Parse third argument (strategies)  
                    if len(context.args) > 2:
                        strategies = [s.strip() for s in context.args[2].split(',')]
                    
                    # Parse fourth argument (capital)
                    if len(context.args) > 3:
                        initial_capital = float(context.args[3])
                        
                except (ValueError, IndexError) as e:
                    if update.message:
                await update.message.reply_text(
                        "❌ Invalid arguments. Usage:\n"
                        "`/compare_run [symbols] [days] [strategies] [capital]`\n\n"
                        "Examples:\n"
                        "• `/compare_run` - Use defaults\n"
                        "• `/compare_run BTCUSDT,ETHUSDT 7` - Specific pairs, 7 days\n"
                        "• `/compare_run default 30 Ultimate,Momentum 100`",
                        parse_mode='Markdown'
                    )
                    return
            
            if update.message:
                if update.message:
                await update.message.reply_text("🚀 Starting strategy comparison...")
            
            # Start comparison
            result_text, comparison_id = await self.strategy_comparison.start_comparison(
                user_id=user_id,
                symbols=symbols,
                days=days,
                strategies=strategies,
                initial_capital=initial_capital
            )
            
            if update.message:
                if update.message:
                await update.message.reply_text(result_text, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error in compare_run command: {e}")
            if update.message:
                if update.message:
                await update.message.reply_text("❌ Error starting comparison. Please try again later.")
    
    async def compare_status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /compare_status command - check comparison progress"""
        try:
            if not update.effective_user:
                return
            user_id = update.effective_user.id
            
            if not self.config.is_authorized_user(user_id):
                if update.message:
                    if update.message:
                await update.message.reply_text("❌ You are not authorized to use this bot.")
                return
            
            if not context.args:
                if update.message:
                    if update.message:
                await update.message.reply_text(
                        "❌ Please provide comparison ID.\nUsage: `/compare_status <id>`",
                        parse_mode='Markdown'
                    )
                return
            
            comparison_id = context.args[0]
            status_text = await self.strategy_comparison.get_comparison_status(comparison_id)
            
            if update.message:
                if update.message:
                await update.message.reply_text(status_text, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error in compare_status command: {e}")
            if update.message:
                if update.message:
                await update.message.reply_text("❌ Error checking status. Please try again later.")
    
    async def compare_result_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /compare_result command - show comparison results"""
        try:
            if not update.effective_user:
                return
            user_id = update.effective_user.id
            
            if not self.config.is_authorized_user(user_id):
                if update.message:
                    if update.message:
                await update.message.reply_text("❌ You are not authorized to use this bot.")
                return
            
            if not context.args:
                if update.message:
                    if update.message:
                await update.message.reply_text(
                        "❌ Please provide comparison ID.\nUsage: `/compare_result <id> [page]`",
                        parse_mode='Markdown'
                    )
                return
            
            comparison_id = context.args[0]
            page = 1
            
            if len(context.args) > 1:
                try:
                    page = int(context.args[1])
                except ValueError:
                    page = 1
            
            if update.message:
                if update.message:
                await update.message.reply_text("⏳ Loading comparison results...")
            
            # Get results
            result_text, chart_path, has_more = await self.strategy_comparison.get_comparison_result(
                comparison_id, page
            )
            
            # Send chart if available
            if chart_path and Path(chart_path).exists():
                try:
                    with open(chart_path, 'rb') as photo:
                        await update.message.reply_photo(
                            photo=photo,
                            caption=result_text,
                            parse_mode='Markdown'
                        )
                except Exception as e:
                    self.logger.warning(f"Error sending chart: {e}")
                    if update.message:
                        if update.message:
                await update.message.reply_text(result_text, parse_mode='Markdown')
            else:
                if update.message:
                    if update.message:
                await update.message.reply_text(result_text, parse_mode='Markdown')
            
            # Add pagination buttons if needed
            if has_more:
                next_page = page + 1
                if update.message:
                    if update.message:
                await update.message.reply_text(
                        f"📄 More results available. Use:\n`/compare_result {comparison_id} {next_page}`",
                        parse_mode='Markdown'
                    )
            
        except Exception as e:
            self.logger.error(f"Error in compare_result command: {e}")
            if update.message:
                if update.message:
                await update.message.reply_text("❌ Error loading results. Please try again later.")
    
    async def compare_recent_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /compare_recent command - show recent comparisons"""
        try:
            if not update.effective_user:
                return
            user_id = update.effective_user.id
            
            if not self.config.is_authorized_user(user_id):
                if update.message:
                    if update.message:
                await update.message.reply_text("❌ You are not authorized to use this bot.")
                return
            
            if update.message:
                if update.message:
                await update.message.reply_text("⏳ Loading your recent comparisons...")
            
            recent_text = await self.strategy_comparison.get_recent_comparisons(user_id)
            if update.message:
                await update.message.reply_text(recent_text, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error in compare_recent command: {e}")
            if update.message:
                await update.message.reply_text("❌ Error loading recent comparisons. Please try again later.")
    
    async def compare_rankings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /compare_rankings command - show strategy rankings by metric"""
        try:
            if not update.effective_user:
                return
            user_id = update.effective_user.id
            
            if not self.config.is_authorized_user(user_id):
                if update.message:
                    if update.message:
                await update.message.reply_text("❌ You are not authorized to use this bot.")
                return
            
            if not context.args:
                available_metrics = self.strategy_comparison.get_available_metrics()
                metrics_list = '\n'.join([f"• `{metric}`" for metric in available_metrics])
                if update.message:
                await update.message.reply_text(
                    f"❌ Please provide comparison ID and metric.\n\n"
                    f"Usage: `/compare_rankings <id> <metric>`\n\n"
                    f"Available metrics:\n{metrics_list}",
                    parse_mode='Markdown'
                )
                return
            
            comparison_id = context.args[0]
            metric = context.args[1] if len(context.args) > 1 else 'total_pnl_percentage'
            
            if update.message:
                await update.message.reply_text("⏳ Loading strategy rankings...")
            
            rankings_text = await self.strategy_comparison.get_comparison_rankings(comparison_id, metric)
            if update.message:
                await update.message.reply_text(rankings_text, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error in compare_rankings command: {e}")
            if update.message:
                await update.message.reply_text("❌ Error loading rankings. Please try again later.")
    
    async def compare_tips_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /compare_tips command - show recommendations"""
        try:
            if not update.effective_user:
                return
            user_id = update.effective_user.id
            
            if not self.config.is_authorized_user(user_id):
                if update.message:
                    if update.message:
                await update.message.reply_text("❌ You are not authorized to use this bot.")
                return
            
            if not context.args:
                if update.message:
                await update.message.reply_text(
                    "❌ Please provide comparison ID.\nUsage: `/compare_tips <id>`",
                    parse_mode='Markdown'
                )
                return
            
            comparison_id = context.args[0]
            if update.message:
                await update.message.reply_text("⏳ Loading recommendations...")
            
            recommendations_text = await self.strategy_comparison.get_comparison_recommendations(comparison_id)
            if update.message:
                await update.message.reply_text(recommendations_text, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error in compare_tips command: {e}")
            if update.message:
                await update.message.reply_text("❌ Error loading recommendations. Please try again later.")
    
    async def compare_help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /compare_help command - show strategy comparison help"""
        try:
            if not update.effective_user:
                return
            user_id = update.effective_user.id
            
            if not self.config.is_authorized_user(user_id):
                if update.message:
                    if update.message:
                await update.message.reply_text("❌ You are not authorized to use this bot.")
                return
            
            help_text = await self.strategy_comparison.get_help_text()
            if update.message:
                await update.message.reply_text(help_text, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error in compare_help command: {e}")
            if update.message:
                await update.message.reply_text("❌ Error loading help. Please try again later.")
    
    # ==================== COMPREHENSIVE METRICS COMMANDS ====================
    
    async def _ensure_metrics_manager(self):
        """Ensure metrics manager is initialized"""
        if self.metrics_manager is None:
            try:
                from trading_metrics_manager import get_global_metrics_manager
                self.metrics_manager = await get_global_metrics_manager()
            except Exception as e:
                self.logger.error(f"Error initializing metrics manager: {e}")
                return None
        return self.metrics_manager
    
    async def metrics_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /metrics command - comprehensive trading metrics"""
        try:
            if not update.effective_user:
                return
            user_id = update.effective_user.id
            
            if not self.config.is_authorized_user(user_id):
                if update.message:
                    if update.message:
                await update.message.reply_text("❌ You are not authorized to use this bot.")
                return
            
            if update.message:
                await update.message.reply_text("📊 Loading comprehensive trading metrics...")
            
            metrics_manager = await self._ensure_metrics_manager()
            if not metrics_manager:
                if update.message:
                await update.message.reply_text("❌ Metrics system not available. Please try again later.")
                return
            
            telegram_metrics = await metrics_manager.get_metrics_for_telegram()
            if update.message:
                await update.message.reply_text(telegram_metrics, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error in metrics command: {e}")
            if update.message:
                await update.message.reply_text("❌ Error loading metrics. Please try again later.")
    
    async def performance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /performance command - detailed performance analysis"""
        try:
            if not update.effective_user:
                return
            user_id = update.effective_user.id
            
            if not self.config.is_authorized_user(user_id):
                if update.message:
                    if update.message:
                await update.message.reply_text("❌ You are not authorized to use this bot.")
                return
            
            if update.message:
                await update.message.reply_text("📈 Analyzing trading performance...")
            
            metrics_manager = await self._ensure_metrics_manager()
            if not metrics_manager:
                if update.message:
                await update.message.reply_text("❌ Metrics system not available. Please try again later.")
                return
            
            metrics = await metrics_manager.calculate_comprehensive_metrics()
            
            performance_text = f"""📈 **DETAILED PERFORMANCE ANALYSIS**

🎯 **Overall Statistics**
• Total Trades: {metrics.total_trades}
• Win Rate: {metrics.win_rate_percentage:.2f}% ({metrics.winning_trades}/{metrics.total_trades})
• Total PnL: ${metrics.total_pnl:+,.2f}

💰 **Profit & Loss**
• Realized PnL: ${metrics.current_realized_pnl:+,.2f}
• Unrealized PnL: ${metrics.current_unrealized_pnl:+,.2f}
• Daily PnL: ${metrics.daily_pnl:+,.2f}

🔥 **Trade Averages**
• Avg Win: ${metrics.avg_profit_per_win:+,.2f}
• Avg Loss: ${metrics.avg_loss_per_losing_trade:+,.2f}
• Profit Factor: {(abs(metrics.avg_profit_per_win) / abs(metrics.avg_loss_per_losing_trade)):.2f} if metrics.avg_loss_per_losing_trade != 0 else "∞"

📊 **Risk Analysis**
• Sharpe Ratio: {metrics.sharpe_ratio:.3f}
• Max Drawdown: ${metrics.maximum_drawdown:+,.2f} ({metrics.maximum_drawdown_percentage:.2f}%)

⚡ **Activity**
• Today's Trades: {metrics.trades_completed_today}
• Trade Rate: {metrics.trades_per_hour:.2f}/hour

🕐 Updated: {metrics.last_updated.strftime('%H:%M:%S')}"""
            
            if update.message:
                await update.message.reply_text(performance_text, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error in performance command: {e}")
            if update.message:
                await update.message.reply_text("❌ Error loading performance data. Please try again later.")
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command - quick statistics overview"""
        try:
            if not update.effective_user:
                return
            user_id = update.effective_user.id
            
            if not self.config.is_authorized_user(user_id):
                if update.message:
                    if update.message:
                await update.message.reply_text("❌ You are not authorized to use this bot.")
                return
            
            metrics_manager = await self._ensure_metrics_manager()
            if not metrics_manager:
                if update.message:
                await update.message.reply_text("❌ Metrics system not available.")
                return
            
            summary = await metrics_manager.get_metrics_summary()
            
            stats_text = f"""📊 **QUICK STATS**

📈 Win Rate: {summary['win_rate']:.1f}% ({summary['total_trades']} trades)
💰 Total PnL: ${summary['total_pnl']:+,.2f}
📅 Daily PnL: ${summary['daily_pnl']:+,.2f}
🔥 Current Streak: {'🟢' if summary['consecutive_wins'] > 0 else '🔴' if summary['consecutive_losses'] > 0 else '⚪'} {max(summary['consecutive_wins'], summary['consecutive_losses'])}
⚡ Rate: {summary['trades_per_hour']:.1f}/hr | Today: {summary['trades_today']}
📉 Max DD: ${summary['max_drawdown']:+,.2f} ({summary['max_drawdown_pct']:.1f}%)"""
            
            if update.message:
                await update.message.reply_text(stats_text, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error in stats command: {e}")
            if update.message:
                await update.message.reply_text("❌ Error loading stats.")
    
    async def winrate_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /winrate command - detailed win rate analysis"""
        try:
            if not update.effective_user:
                return
            user_id = update.effective_user.id
            
            if not self.config.is_authorized_user(user_id):
                if update.message:
                    if update.message:
                await update.message.reply_text("❌ You are not authorized to use this bot.")
                return
            
            metrics_manager = await self._ensure_metrics_manager()
            if not metrics_manager:
                if update.message:
                await update.message.reply_text("❌ Metrics system not available.")
                return
            
            metrics = await metrics_manager.calculate_comprehensive_metrics()
            
            # Performance emoji based on win rate
            if metrics.win_rate_percentage >= 80:
                emoji = "🏆"
            elif metrics.win_rate_percentage >= 70:
                emoji = "🥇"
            elif metrics.win_rate_percentage >= 60:
                emoji = "🥈"
            elif metrics.win_rate_percentage >= 50:
                emoji = "🥉"
            else:
                emoji = "⚠️"
            
            winrate_text = f"""{emoji} **WIN RATE ANALYSIS**

📊 **Overall Win Rate:** {metrics.win_rate_percentage:.2f}%
• Winning Trades: {metrics.winning_trades}
• Losing Trades: {metrics.losing_trades}
• Total Trades: {metrics.total_trades}

🔥 **Streak Performance:**
• Current Streak: {'🟢 Wins' if metrics.current_streak_type == 'win' else '🔴 Losses' if metrics.current_streak_type == 'loss' else '⚪ None'}
• Consecutive: {metrics.consecutive_wins if metrics.current_streak_type == 'win' else metrics.consecutive_losses if metrics.current_streak_type == 'loss' else 0}
• Best Win Streak: 🏆 {metrics.best_winning_streak}
• Worst Loss Streak: 💥 {metrics.worst_losing_streak}"""
            
            # Add comparison if available
            if metrics.daily_comparison:
                daily = metrics.daily_comparison
                if 'pnl_change_percentage' in daily:
                    trend_emoji = "📈" if daily['pnl_change_percentage'] > 0 else "📉" if daily['pnl_change_percentage'] < 0 else "➡️"
                    winrate_text += f"""

📅 **Daily Comparison:**
• Performance: {trend_emoji} {daily['pnl_change_percentage']:+.1f}% vs yesterday
• Trades: {daily['today_trades']} today vs {daily['yesterday_trades']} yesterday"""
            
            if update.message:
                await update.message.reply_text(winrate_text, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error in winrate command: {e}")
            if update.message:
                await update.message.reply_text("❌ Error loading win rate data.")
    
    async def pnl_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pnl command - profit and loss analysis"""
        try:
            if not update.effective_user:
                return
            user_id = update.effective_user.id
            
            if not self.config.is_authorized_user(user_id):
                if update.message:
                    if update.message:
                await update.message.reply_text("❌ You are not authorized to use this bot.")
                return
            
            metrics_manager = await self._ensure_metrics_manager()
            if not metrics_manager:
                if update.message:
                await update.message.reply_text("❌ Metrics system not available.")
                return
            
            metrics = await metrics_manager.calculate_comprehensive_metrics()
            
            def format_pnl(value):
                return f"${value:+,.2f}" if value != 0 else "$0.00"
            
            def get_pnl_emoji(value):
                if value > 100: return "🚀"
                elif value > 50: return "💰"
                elif value > 0: return "📈"
                elif value > -50: return "📉"
                else: return "💥"
            
            total_emoji = get_pnl_emoji(metrics.total_pnl)
            daily_emoji = get_pnl_emoji(metrics.daily_pnl)
            
            pnl_text = f"""💰 **PROFIT & LOSS ANALYSIS**

{total_emoji} **Total PnL:** {format_pnl(metrics.total_pnl)}
• Realized PnL: {format_pnl(metrics.current_realized_pnl)}
• Unrealized PnL: {format_pnl(metrics.current_unrealized_pnl)}

{daily_emoji} **Daily PnL:** {format_pnl(metrics.daily_pnl)}

📊 **Trade Averages:**
• Avg Profit per Win: {format_pnl(metrics.avg_profit_per_win)}
• Avg Loss per Trade: {format_pnl(metrics.avg_loss_per_losing_trade)}
• Profit Factor: {(abs(metrics.avg_profit_per_win) / abs(metrics.avg_loss_per_losing_trade)):.2f}x if abs(metrics.avg_loss_per_losing_trade) > 0.001 else "∞"

🛡️ **Risk Metrics:**
• Sharpe Ratio: {metrics.sharpe_ratio:.3f}
• Max Drawdown: {format_pnl(metrics.maximum_drawdown)} ({metrics.maximum_drawdown_percentage:.2f}%)"""
            
            # Add daily comparison
            if metrics.daily_comparison and 'today_pnl' in metrics.daily_comparison:
                daily_comp = metrics.daily_comparison
                change_emoji = "📈" if daily_comp['pnl_change_percentage'] > 0 else "📉" if daily_comp['pnl_change_percentage'] < 0 else "➡️"
                
                pnl_text += f"""

📅 **Daily Comparison:**
• Today: {format_pnl(daily_comp['today_pnl'])}
• Yesterday: {format_pnl(daily_comp['yesterday_pnl'])}
• Change: {change_emoji} {daily_comp['pnl_change_percentage']:+.1f}%"""
            
            if update.message:
                await update.message.reply_text(pnl_text, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error in pnl command: {e}")
            if update.message:
                await update.message.reply_text("❌ Error loading PnL data.")
    
    async def streaks_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /streaks command - winning and losing streaks"""
        try:
            if not update.effective_user:
                return
            user_id = update.effective_user.id
            
            if not self.config.is_authorized_user(user_id):
                if update.message:
                    if update.message:
                await update.message.reply_text("❌ You are not authorized to use this bot.")
                return
            
            metrics_manager = await self._ensure_metrics_manager()
            if not metrics_manager:
                if update.message:
                await update.message.reply_text("❌ Metrics system not available.")
                return
            
            metrics = await metrics_manager.calculate_comprehensive_metrics()
            
            def get_streak_emoji(streak_type):
                if streak_type == "win":
                    return "🟢"
                elif streak_type == "loss":
                    return "🔴"
                else:
                    return "⚪"
            
            current_emoji = get_streak_emoji(metrics.current_streak_type)
            current_count = metrics.consecutive_wins if metrics.current_streak_type == "win" else metrics.consecutive_losses if metrics.current_streak_type == "loss" else 0
            
            streaks_text = f"""🔥 **STREAK ANALYSIS**

{current_emoji} **Current Streak:**
• Type: {metrics.current_streak_type.title() if metrics.current_streak_type != 'none' else 'None'}
• Count: {current_count}

🏆 **Record Streaks:**
• Best Winning Streak: {metrics.best_winning_streak}
• Worst Losing Streak: {metrics.worst_losing_streak}

📊 **Current Status:**
• Consecutive Wins: {'🟢 ' + str(metrics.consecutive_wins) if metrics.consecutive_wins > 0 else '⚪ 0'}
• Consecutive Losses: {'🔴 ' + str(metrics.consecutive_losses) if metrics.consecutive_losses > 0 else '⚪ 0'}

💡 **Streak Performance:**
• Win Rate: {metrics.win_rate_percentage:.1f}%
• {'Hot Streak! 🔥' if current_count >= 3 and metrics.current_streak_type == 'win' else 'Cold Streak ❄️' if current_count >= 3 and metrics.current_streak_type == 'loss' else 'Neutral 📊'}"""
            
            if update.message:
                await update.message.reply_text(streaks_text, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error in streaks command: {e}")
            if update.message:
                await update.message.reply_text("❌ Error loading streak data.")
    
    async def pairs_performance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pairs command - performance by trading pair"""
        try:
            if not update.effective_user:
                return
            user_id = update.effective_user.id
            
            if not self.config.is_authorized_user(user_id):
                if update.message:
                    if update.message:
                await update.message.reply_text("❌ You are not authorized to use this bot.")
                return
            
            metrics_manager = await self._ensure_metrics_manager()
            if not metrics_manager:
                if update.message:
                await update.message.reply_text("❌ Metrics system not available.")
                return
            
            metrics = await metrics_manager.calculate_comprehensive_metrics()
            
            if not metrics.performance_by_trading_pair:
                if update.message:
                await update.message.reply_text("📭 No trading pair data available yet.")
                return
            
            # Sort pairs by total PnL
            sorted_pairs = sorted(
                metrics.performance_by_trading_pair.items(), 
                key=lambda x: x[1]['total_pnl'], 
                reverse=True
            )
            
            pairs_text = "🏆 **TRADING PAIRS PERFORMANCE**\n\n"
            
            for i, (pair, stats) in enumerate(sorted_pairs[:10], 1):  # Top 10 pairs
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}️⃣"
                pnl_emoji = "📈" if stats['total_pnl'] > 0 else "📉" if stats['total_pnl'] < 0 else "➡️"
                
                pairs_text += f"{emoji} **{pair}**\n"
                pairs_text += f"• PnL: {pnl_emoji} ${stats['total_pnl']:+,.2f}\n"
                pairs_text += f"• Win Rate: {stats['win_rate']:.1f}% ({stats['winning_trades']}/{stats['total_trades']})\n"
                pairs_text += f"• Avg Trade: ${stats['avg_profit']:+,.2f}\n"
                pairs_text += f"• Best: ${stats['best_trade']:+,.2f} | Worst: ${stats['worst_trade']:+,.2f}\n\n"
            
            if len(sorted_pairs) > 10:
                pairs_text += f"... and {len(sorted_pairs) - 10} more pairs"
            
            if update.message:
                await update.message.reply_text(pairs_text, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error in pairs command: {e}")
            if update.message:
                await update.message.reply_text("❌ Error loading pairs data.")
    
    async def hourly_performance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /hourly command - success rate by time of day"""
        try:
            if not update.effective_user:
                return
            user_id = update.effective_user.id
            
            if not self.config.is_authorized_user(user_id):
                if update.message:
                    if update.message:
                await update.message.reply_text("❌ You are not authorized to use this bot.")
                return
            
            metrics_manager = await self._ensure_metrics_manager()
            if not metrics_manager:
                if update.message:
                await update.message.reply_text("❌ Metrics system not available.")
                return
            
            metrics = await metrics_manager.calculate_comprehensive_metrics()
            
            if not metrics.success_rate_by_time:
                if update.message:
                await update.message.reply_text("📭 No hourly performance data available yet.")
                return
            
            # Sort by success rate
            sorted_hours = sorted(
                metrics.success_rate_by_time.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            hourly_text = "🕐 **HOURLY PERFORMANCE**\n\n"
            hourly_text += "🏆 **Best Trading Hours:**\n"
            
            for i, (time_period, win_rate) in enumerate(sorted_hours[:5], 1):
                emoji = "🎯" if win_rate > 80 else "📈" if win_rate > 60 else "📊"
                hourly_text += f"{i}. {emoji} {time_period}: {win_rate:.1f}%\n"
            
            if len(sorted_hours) > 5:
                hourly_text += "\n💡 **Other Hours:**\n"
                for time_period, win_rate in sorted_hours[5:10]:
                    emoji = "📊" if win_rate > 50 else "⚠️"
                    hourly_text += f"• {emoji} {time_period}: {win_rate:.1f}%\n"
            
            # Add current hour if available
            current_hour = datetime.now().hour
            current_period = f"{current_hour:02d}:00-{current_hour:02d}:59"
            if current_period in metrics.success_rate_by_time:
                current_rate = metrics.success_rate_by_time[current_period]
                hourly_text += f"\n🕐 **Current Hour ({current_period}):** {current_rate:.1f}%"
            
            if update.message:
                await update.message.reply_text(hourly_text, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error in hourly command: {e}")
            if update.message:
                await update.message.reply_text("❌ Error loading hourly data.")
    
    async def risk_metrics_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /risk command - risk analysis metrics"""
        try:
            if not update.effective_user:
                return
            user_id = update.effective_user.id
            
            if not self.config.is_authorized_user(user_id):
                if update.message:
                    if update.message:
                await update.message.reply_text("❌ You are not authorized to use this bot.")
                return
            
            metrics_manager = await self._ensure_metrics_manager()
            if not metrics_manager:
                if update.message:
                await update.message.reply_text("❌ Metrics system not available.")
                return
            
            metrics = await metrics_manager.calculate_comprehensive_metrics()
            
            def get_sharpe_emoji(sharpe):
                if sharpe > 2: return "🎯"
                elif sharpe > 1: return "📈" 
                elif sharpe > 0: return "📊"
                else: return "⚠️"
            
            def get_drawdown_emoji(dd_pct):
                if dd_pct < 5: return "🛡️"
                elif dd_pct < 10: return "⚠️"
                elif dd_pct < 20: return "🔴"
                else: return "💥"
            
            sharpe_emoji = get_sharpe_emoji(metrics.sharpe_ratio)
            dd_emoji = get_drawdown_emoji(metrics.maximum_drawdown_percentage)
            
            risk_text = f"""🛡️ **RISK ANALYSIS**

{sharpe_emoji} **Sharpe Ratio:** {metrics.sharpe_ratio:.3f}
{'• Excellent risk-adjusted returns!' if metrics.sharpe_ratio > 2 else '• Good risk-adjusted returns' if metrics.sharpe_ratio > 1 else '• Acceptable risk-adjusted returns' if metrics.sharpe_ratio > 0 else '• Poor risk-adjusted returns'}

{dd_emoji} **Maximum Drawdown:**
• Amount: ${metrics.maximum_drawdown:+,.2f}
• Percentage: {metrics.maximum_drawdown_percentage:.2f}%

📊 **Risk Assessment:**
• Win Rate: {metrics.win_rate_percentage:.1f}%
• Avg Win: ${metrics.avg_profit_per_win:+,.2f}
• Avg Loss: ${metrics.avg_loss_per_losing_trade:+,.2f}
• Risk/Reward: 1:{abs(metrics.avg_profit_per_win) / abs(metrics.avg_loss_per_losing_trade):.2f} if abs(metrics.avg_loss_per_losing_trade) > 0.001 else "1:∞"

🎯 **Risk Status:**
{'🟢 Low Risk' if metrics.maximum_drawdown_percentage < 5 and metrics.sharpe_ratio > 1 else '🟡 Medium Risk' if metrics.maximum_drawdown_percentage < 15 and metrics.sharpe_ratio > 0 else '🔴 High Risk'}"""
            
            if update.message:
                await update.message.reply_text(risk_text, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error in risk command: {e}")
            if update.message:
                await update.message.reply_text("❌ Error loading risk metrics.")
    
    async def daily_comparison_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /daily command - daily performance comparison"""
        try:
            if not update.effective_user:
                return
            user_id = update.effective_user.id
            
            if not self.config.is_authorized_user(user_id):
                if update.message:
                    if update.message:
                await update.message.reply_text("❌ You are not authorized to use this bot.")
                return
            
            metrics_manager = await self._ensure_metrics_manager()
            if not metrics_manager:
                if update.message:
                await update.message.reply_text("❌ Metrics system not available.")
                return
            
            metrics = await metrics_manager.calculate_comprehensive_metrics()
            
            if not metrics.daily_comparison:
                if update.message:
                await update.message.reply_text("📭 No daily comparison data available yet.")
                return
            
            daily = metrics.daily_comparison
            weekly = metrics.weekly_comparison or {}
            
            def get_trend_emoji(value):
                if value > 10: return "🚀"
                elif value > 0: return "📈"
                elif value > -10: return "📉"
                else: return "💥"
            
            pnl_emoji = get_trend_emoji(daily.get('pnl_change_percentage', 0))
            trades_emoji = get_trend_emoji(daily.get('trades_change_percentage', 0))
            
            daily_text = f"""📅 **DAILY COMPARISON**

{pnl_emoji} **PnL Performance:**
• Today: ${daily.get('today_pnl', 0):+,.2f}
• Yesterday: ${daily.get('yesterday_pnl', 0):+,.2f}
• Change: {daily.get('pnl_change_percentage', 0):+.1f}%

{trades_emoji} **Trading Activity:**
• Today: {daily.get('today_trades', 0)} trades
• Yesterday: {daily.get('yesterday_trades', 0)} trades
• Change: {daily.get('trades_change_percentage', 0):+.1f}%"""
            
            if weekly:
                weekly_trend = weekly.get('performance_trend', 'unknown')
                trend_emoji = "📈" if weekly_trend == 'improving' else "📉" if weekly_trend == 'declining' else "➡️"
                
                daily_text += f"""

📊 **Weekly Overview:**
• This Week Total: ${weekly.get('total_week_pnl', 0):+,.2f}
• Daily Average: ${weekly.get('avg_daily_pnl_this_week', 0):+,.2f}
• Total Trades: {weekly.get('total_week_trades', 0)}
• Trend: {trend_emoji} {weekly_trend.title()}"""
            
            daily_text += f"\n\n🕐 Updated: {metrics.last_updated.strftime('%H:%M:%S')}"
            
            if update.message:
                await update.message.reply_text(daily_text, parse_mode='Markdown')
            
        except Exception as e:
            self.logger.error(f"Error in daily command: {e}")
            if update.message:
                await update.message.reply_text("❌ Error loading daily comparison.")
    
    # ==================== END METRICS COMMANDS ====================
