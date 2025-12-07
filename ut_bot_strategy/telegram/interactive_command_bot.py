"""
Interactive Command Bot for @TradeTactics_bot

A comprehensive Telegram bot with interactive commands for monitoring and controlling
the AI Trading Signal System.

Features:
- All trading commands (/start, /help, /status, /signals, etc.)
- Interactive inline buttons for leverage and settings
- Admin-only access with chat ID whitelist
- Rate limiting for command spam protection
- Full integration with AITradingOrchestrator components
- Professional HTML-formatted messages
"""

import os
import asyncio
import logging
import functools
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable
from collections import defaultdict

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters
)
from telegram.constants import ParseMode

logger = logging.getLogger(__name__)

LEVERAGE_SELECT, RISK_SELECT = range(2)

LEVERAGE_OPTIONS = [1, 2, 3, 5, 10, 15, 20, 25]
RISK_OPTIONS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]


def admin_only(func: Callable) -> Callable:
    """Decorator to restrict commands to admin users only"""
    @functools.wraps(func)
    async def wrapper(self, update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        if not update.effective_user or not update.effective_chat:
            return
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        
        if not self._is_authorized(user_id, chat_id):
            logger.warning(f"Unauthorized access attempt from user {user_id} in chat {chat_id}")
            
            if hasattr(self, '_first_user_auto_admin') and self._first_user_auto_admin:
                self._first_user_auto_admin = False
                self.add_admin(user_id)
                logger.info(f"Auto-authorized first user: {user_id}")
                return await func(self, update, context, *args, **kwargs)
            
            if update.message:
                await update.message.reply_text(
                    f"⛔ <b>Access Denied</b>\n\n"
                    f"Your ID: <code>{user_id}</code>\n\n"
                    f"To authorize yourself, add your ID to the "
                    f"<code>ADMIN_CHAT_IDS</code> environment variable.\n\n"
                    f"Or set <code>TELEGRAM_CHAT_ID={user_id}</code>",
                    parse_mode=ParseMode.HTML
                )
            return
        
        return await func(self, update, context, *args, **kwargs)
    return wrapper


def rate_limit(max_calls: int = 5, window_seconds: int = 60) -> Callable:
    """Decorator to rate limit command usage"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(self, update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
            if not update.effective_user:
                return
            user_id = update.effective_user.id
            command = func.__name__
            
            if not self._check_rate_limit(user_id, command, max_calls, window_seconds):
                if update.message:
                    await update.message.reply_text(
                        "⚠️ <b>Rate Limit Exceeded</b>\n\n"
                        f"Please wait before using this command again.\n"
                        f"Max {max_calls} calls per {window_seconds} seconds.",
                        parse_mode=ParseMode.HTML
                    )
                return
            
            return await func(self, update, context, *args, **kwargs)
        return wrapper
    return decorator


def error_handler(func: Callable) -> Callable:
    """Decorator for consistent error handling in command handlers"""
    @functools.wraps(func)
    async def wrapper(self, update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        try:
            return await func(self, update, context, *args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            error_message = (
                "❌ <b>Error Occurred</b>\n\n"
                f"Command: <code>/{func.__name__.replace('cmd_', '')}</code>\n"
                f"Error: <code>{str(e)[:100]}</code>\n\n"
                "Please try again or contact support."
            )
            try:
                if update.message:
                    await update.message.reply_text(error_message, parse_mode=ParseMode.HTML)
                elif update.callback_query:
                    await update.callback_query.answer("An error occurred")
                    await update.callback_query.edit_message_text(error_message, parse_mode=ParseMode.HTML)
            except Exception:
                pass
    return wrapper


class InteractiveCommandBot:
    """
    Interactive Telegram Command Bot for TradeTactics
    
    Provides comprehensive command interface for:
    - Monitoring trading system status
    - Viewing signals, positions, and performance
    - Adjusting settings like leverage and risk
    - Accessing AI insights and trade history
    """
    
    def __init__(
        self,
        bot_token: str,
        admin_chat_ids: Optional[str] = None,
        orchestrator: Optional[Any] = None
    ):
        """
        Initialize Interactive Command Bot
        
        Args:
            bot_token: Telegram Bot API token
            admin_chat_ids: Comma-separated list of authorized chat IDs (or from ADMIN_CHAT_IDS env)
            orchestrator: AITradingOrchestrator instance for data access
        """
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN', '')
        
        self.admin_chat_ids: set = set()
        
        telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
        if telegram_chat_id:
            try:
                self.admin_chat_ids.add(int(telegram_chat_id.strip()))
                logger.info(f"Added TELEGRAM_CHAT_ID {telegram_chat_id} as admin")
            except ValueError:
                logger.warning(f"Invalid TELEGRAM_CHAT_ID: {telegram_chat_id}")
        
        admin_ids_str = admin_chat_ids or os.getenv('ADMIN_CHAT_IDS', '')
        if admin_ids_str:
            for id_str in admin_ids_str.split(','):
                id_str = id_str.strip()
                if id_str:
                    try:
                        self.admin_chat_ids.add(int(id_str))
                    except ValueError:
                        logger.warning(f"Invalid chat ID in admin list: {id_str}")
        
        self.orchestrator = orchestrator
        self.application: Optional[Application] = None
        self._running = False
        
        self._rate_limit_data: Dict[str, List[datetime]] = defaultdict(list)
        
        self._settings_overrides = {}
        
        self._dynamic_admins: set = set()
        
        self._first_user_auto_admin = len(self.admin_chat_ids) == 0
        
        self._polling_task: Optional[asyncio.Task] = None
        self._update_offset: int = 0
        
        logger.info(f"InteractiveCommandBot initialized with {len(self.admin_chat_ids)} admin IDs: {self.admin_chat_ids}")
        if self._first_user_auto_admin:
            logger.info("First user auto-admin mode enabled - first user to run a command will be authorized")
    
    def _is_authorized(self, user_id: int, chat_id: int) -> bool:
        """Check if user/chat is authorized"""
        if not self.admin_chat_ids and not self._dynamic_admins:
            return True
        all_admins = self.admin_chat_ids | self._dynamic_admins
        return user_id in all_admins or chat_id in all_admins
    
    def add_admin(self, chat_id: int) -> None:
        """Dynamically add an admin chat ID"""
        self._dynamic_admins.add(chat_id)
        logger.info(f"Dynamically added admin: {chat_id}")
    
    def _check_rate_limit(self, user_id: int, command: str, max_calls: int, window_seconds: int) -> bool:
        """Check and update rate limit for a user/command"""
        key = f"{user_id}:{command}"
        now = datetime.now()
        cutoff = now - timedelta(seconds=window_seconds)
        
        self._rate_limit_data[key] = [t for t in self._rate_limit_data[key] if t > cutoff]
        
        if len(self._rate_limit_data[key]) >= max_calls:
            return False
        
        self._rate_limit_data[key].append(now)
        return True
    
    def _format_price(self, price: float) -> str:
        """Format price with appropriate decimal places"""
        if price >= 10000:
            return f"{price:,.2f}"
        elif price >= 1000:
            return f"{price:,.3f}"
        elif price >= 1:
            return f"{price:.4f}"
        elif price >= 0.01:
            return f"{price:.5f}"
        else:
            return f"{price:.8f}"
    
    def _format_timestamp(self, dt: Optional[datetime]) -> str:
        """Format datetime for display"""
        if dt is None:
            return "Never"
        return dt.strftime('%Y-%m-%d %H:%M:%S UTC')
    
    def _get_orchestrator_status(self) -> Dict[str, Any]:
        """Get status from orchestrator if available"""
        if self.orchestrator:
            try:
                return self.orchestrator.get_status()
            except Exception as e:
                logger.error(f"Error getting orchestrator status: {e}")
        return {}
    
    async def start(self) -> None:
        """Start the interactive command bot using manual polling (more robust)"""
        if self._running:
            logger.warning("Bot is already running")
            return
        
        self.application = Application.builder().token(self.bot_token).build()
        
        self.application.add_handler(CommandHandler("start", self._wrap_handler(self.cmd_start)))
        self.application.add_handler(CommandHandler("help", self._wrap_handler(self.cmd_help)))
        self.application.add_handler(CommandHandler("status", self._wrap_handler(self.cmd_status)))
        self.application.add_handler(CommandHandler("health", self._wrap_handler(self.cmd_health)))
        self.application.add_handler(CommandHandler("signals", self._wrap_handler(self.cmd_signals)))
        self.application.add_handler(CommandHandler("positions", self._wrap_handler(self.cmd_positions)))
        self.application.add_handler(CommandHandler("performance", self._wrap_handler(self.cmd_performance)))
        self.application.add_handler(CommandHandler("history", self._wrap_handler(self.cmd_history)))
        self.application.add_handler(CommandHandler("ai", self._wrap_handler(self.cmd_ai)))
        self.application.add_handler(CommandHandler("insights", self._wrap_handler(self.cmd_insights)))
        self.application.add_handler(CommandHandler("settings", self._wrap_handler(self.cmd_settings)))
        self.application.add_handler(CommandHandler("market", self._wrap_handler(self.cmd_market)))
        self.application.add_handler(CommandHandler("toggle_auto", self._wrap_handler(self.cmd_toggle_auto)))
        
        leverage_conv = ConversationHandler(
            entry_points=[CommandHandler("leverage", self._wrap_handler(self.cmd_leverage))],
            states={
                LEVERAGE_SELECT: [
                    CallbackQueryHandler(self._wrap_callback(self.leverage_callback), pattern=r"^leverage_")
                ]
            },
            fallbacks=[CommandHandler("cancel", self._wrap_handler(self.cmd_cancel))],
            per_message=False
        )
        self.application.add_handler(leverage_conv)
        
        risk_conv = ConversationHandler(
            entry_points=[CommandHandler("risk", self._wrap_handler(self.cmd_risk))],
            states={
                RISK_SELECT: [
                    CallbackQueryHandler(self._wrap_callback(self.risk_callback), pattern=r"^risk_")
                ]
            },
            fallbacks=[CommandHandler("cancel", self._wrap_handler(self.cmd_cancel))],
            per_message=False
        )
        self.application.add_handler(risk_conv)
        
        self.application.add_error_handler(self._error_callback)
        
        await self.application.initialize()
        await self.application.start()
        
        try:
            await self.application.bot.delete_webhook(drop_pending_updates=True)
        except Exception as e:
            logger.warning(f"Could not delete webhook: {e}")
        
        self._running = True
        self._polling_task = asyncio.create_task(self._manual_polling())
        
        logger.info("InteractiveCommandBot started successfully with manual polling")
    
    async def _manual_polling(self) -> None:
        """
        Manual polling loop using bot.get_updates() directly.
        
        This approach is more robust than using application.updater.start_polling()
        because it avoids internal Updater class issues in python-telegram-bot v20+.
        """
        logger.info("Starting manual polling loop...")
        
        while self._running:
            try:
                if not self.application:
                    await asyncio.sleep(1)
                    continue
                updates = await self.application.bot.get_updates(
                    offset=self._update_offset,
                    timeout=10,
                    allowed_updates=["message", "callback_query"]
                )
                
                for update in updates:
                    self._update_offset = update.update_id + 1
                    try:
                        await self.application.process_update(update)
                    except Exception as e:
                        logger.error(f"Error processing update {update.update_id}: {e}")
                        
            except asyncio.CancelledError:
                logger.info("Polling cancelled")
                break
            except Exception as e:
                logger.error(f"Polling error: {e}")
                await asyncio.sleep(2)
    
    async def stop(self) -> None:
        """Stop the interactive command bot"""
        if not self._running:
            return
        
        self._running = False
        
        if self._polling_task is not None:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass
            self._polling_task = None
        
        if self.application:
            try:
                await self.application.stop()
                await self.application.shutdown()
                logger.info("InteractiveCommandBot stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping bot: {e}")
    
    def _wrap_handler(self, handler: Callable) -> Callable:
        """Wrap a handler method to pass self"""
        async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE):
            return await handler(update, context)
        return wrapped
    
    def _wrap_callback(self, callback: Callable) -> Callable:
        """Wrap a callback method to pass self"""
        async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE):
            return await callback(update, context)
        return wrapped
    
    async def _error_callback(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Global error handler for the bot"""
        logger.error(f"Bot error: {context.error}", exc_info=context.error)
    
    @error_handler
    @rate_limit(max_calls=3, window_seconds=30)
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command - Welcome message"""
        if not update.message or not update.effective_user:
            return
        user = update.effective_user
        logger.info(f"Start command from user {user.id} ({user.username})")
        
        message = f"""
🚀 <b>Welcome to TradeTactics Bot!</b>

Hello <b>{user.first_name}</b>! I'm your AI-powered trading assistant.

━━━━━━━━━━━━━━━━━━━━━

📊 <b>What I Do:</b>
• Generate AI-enhanced trading signals
• Send Cornix-compatible notifications
• Track positions and performance
• Learn from trade outcomes

⚙️ <b>Strategy:</b>
• UT Bot Alerts + STC Indicator
• Dynamic leverage optimization
• Multi-TP level management
• AI confidence scoring

━━━━━━━━━━━━━━━━━━━━━

💡 Use /help to see all available commands.

<i>Version 2.0 | AI-Powered Trading</i>
"""
        await update.message.reply_text(message.strip(), parse_mode=ParseMode.HTML)
    
    @error_handler
    @rate_limit(max_calls=5, window_seconds=30)
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command - List all commands"""
        if not update.message or not update.effective_user:
            return
        logger.info(f"Help command from user {update.effective_user.id}")
        
        message = """
📖 <b>TradeTactics Bot Commands</b>

━━━━━━━━━━━━━━━━━━━━━

<b>📊 Status & Monitoring</b>
/status - System status overview
/health - Detailed health check
/market - Current market state

<b>📈 Trading</b>
/signals - Recent trading signals
/positions - Open positions
/performance - Performance stats
/history - Trade history

<b>🤖 AI Features</b>
/ai - AI brain status & insights
/insights - Latest AI learnings

<b>⚙️ Settings</b>
/settings - View configuration
/leverage - Set leverage level
/risk - Set risk percentage
/toggle_auto - Toggle auto trading

<b>ℹ️ General</b>
/start - Welcome message
/help - This help message
/cancel - Cancel current operation

━━━━━━━━━━━━━━━━━━━━━

<i>All commands support HTML formatting</i>
"""
        await update.message.reply_text(message.strip(), parse_mode=ParseMode.HTML)
    
    @admin_only
    @error_handler
    @rate_limit(max_calls=10, window_seconds=60)
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command - System status"""
        if not update.message or not update.effective_user:
            return
        logger.info(f"Status command from user {update.effective_user.id}")
        
        status = self._get_orchestrator_status()
        
        running = status.get('running', False)
        running_emoji = "🟢" if running else "🔴"
        running_text = "Running" if running else "Stopped"
        
        signal_count = status.get('signal_count', 0)
        error_count = status.get('error_count', 0)
        last_signal = status.get('last_signal_time')
        
        ai_active = status.get('ai_brain_active', False)
        ai_emoji = "🧠" if ai_active else "⚠️"
        ai_text = "Active (GPT-5)" if ai_active else "Fallback Mode"
        
        rate_limit_status = status.get('rate_limit_status', {})
        signals_sent = rate_limit_status.get('signals_sent_this_hour', 0)
        signals_remaining = rate_limit_status.get('signals_remaining', 6)
        
        message = f"""
📊 <b>System Status</b>

━━━━━━━━━━━━━━━━━━━━━

{running_emoji} <b>Bot Status:</b> {running_text}

<b>📈 Signal Statistics</b>
• Signals Sent: <code>{signal_count}</code>
• Last Signal: <code>{self._format_timestamp(last_signal)}</code>
• Errors: <code>{error_count}</code>

{ai_emoji} <b>AI Brain:</b> {ai_text}

<b>⏱️ Rate Limits</b>
• Signals This Hour: <code>{signals_sent}/6</code>
• Remaining: <code>{signals_remaining}</code>

━━━━━━━━━━━━━━━━━━━━━

⏰ <i>{self._format_timestamp(datetime.now())}</i>
"""
        await update.message.reply_text(message.strip(), parse_mode=ParseMode.HTML)
    
    @admin_only
    @error_handler
    @rate_limit(max_calls=5, window_seconds=60)
    async def cmd_health(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /health command - Detailed health check"""
        if not update.message or not update.effective_user:
            return
        logger.info(f"Health command from user {update.effective_user.id}")
        
        components = []
        
        if self.orchestrator:
            components.append(("Orchestrator", True, "Main orchestrator active"))
            
            if hasattr(self.orchestrator, 'data_fetcher'):
                try:
                    df = self.orchestrator.data_fetcher.fetch_historical_data(limit=5)
                    data_ok = df is not None and len(df) > 0
                    components.append(("Data Fetcher", data_ok, f"{len(df) if df is not None else 0} candles"))
                except Exception as e:
                    components.append(("Data Fetcher", False, str(e)[:30]))
            
            if hasattr(self.orchestrator, 'ai_brain'):
                ai_ok = self.orchestrator.ai_brain.ai_available
                components.append(("AI Brain", ai_ok, "GPT-5 connected" if ai_ok else "Fallback mode"))
            
            if hasattr(self.orchestrator, 'telegram_bot'):
                components.append(("Signal Bot", True, "Ready"))
            
            if hasattr(self.orchestrator, 'trade_db'):
                try:
                    await self.orchestrator.trade_db.initialize()
                    components.append(("Trade Database", True, "Connected"))
                except Exception as e:
                    components.append(("Trade Database", False, str(e)[:30]))
            
            if hasattr(self.orchestrator, 'futures_executor'):
                components.append(("Futures Executor", True, "Configured"))
        else:
            components.append(("Orchestrator", False, "Not connected"))
        
        components.append(("Command Bot", True, "Running"))
        
        health_lines = []
        all_healthy = True
        for name, status, detail in components:
            emoji = "✅" if status else "❌"
            if not status:
                all_healthy = False
            health_lines.append(f"{emoji} <b>{name}:</b> {detail}")
        
        health_text = "\n".join(health_lines)
        overall_status = "🟢 All Systems Operational" if all_healthy else "🟠 Some Issues Detected"
        
        message = f"""
🏥 <b>System Health Check</b>

━━━━━━━━━━━━━━━━━━━━━

<b>{overall_status}</b>

━━━━━━━━━━━━━━━━━━━━━

{health_text}

━━━━━━━━━━━━━━━━━━━━━

⏰ <i>Checked: {self._format_timestamp(datetime.now())}</i>
"""
        await update.message.reply_text(message.strip(), parse_mode=ParseMode.HTML)
    
    @admin_only
    @error_handler
    @rate_limit(max_calls=10, window_seconds=60)
    async def cmd_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /signals command - Show recent signals"""
        if not update.message or not update.effective_user:
            return
        logger.info(f"Signals command from user {update.effective_user.id}")
        
        signals = []
        
        if self.orchestrator and hasattr(self.orchestrator, 'telegram_bot'):
            signal_bot = self.orchestrator.telegram_bot
            if hasattr(signal_bot, '_signal_history'):
                history = list(signal_bot._signal_history.values())[-5:]
                signals = history
        
        if not signals:
            message = """
📡 <b>Recent Signals</b>

━━━━━━━━━━━━━━━━━━━━━

<i>No signals recorded yet.</i>

The system will display the last 5 signals here once trading begins.

━━━━━━━━━━━━━━━━━━━━━
"""
        else:
            signal_lines = []
            for sig in reversed(signals):
                direction_emoji = "🟢" if sig.direction == "LONG" else "🔴"
                outcome_emoji = "✅" if sig.outcome == "WIN" else "❌" if sig.outcome == "LOSS" else "⏳"
                
                signal_lines.append(
                    f"{direction_emoji} <b>{sig.symbol}</b> {sig.direction}\n"
                    f"   Entry: <code>{self._format_price(sig.entry_price)}</code>\n"
                    f"   SL: <code>{self._format_price(sig.stop_loss)}</code> | "
                    f"Leverage: <code>{sig.leverage}x</code>\n"
                    f"   {outcome_emoji} Status: {sig.outcome or 'Open'}\n"
                    f"   📅 {sig.timestamp.strftime('%m/%d %H:%M')}"
                )
            
            signals_text = "\n\n".join(signal_lines)
            message = f"""
📡 <b>Recent Signals (Last 5)</b>

━━━━━━━━━━━━━━━━━━━━━

{signals_text}

━━━━━━━━━━━━━━━━━━━━━

⏰ <i>{self._format_timestamp(datetime.now())}</i>
"""
        
        await update.message.reply_text(message.strip(), parse_mode=ParseMode.HTML)
    
    @admin_only
    @error_handler
    @rate_limit(max_calls=10, window_seconds=60)
    async def cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /positions command - Show open positions"""
        if not update.message or not update.effective_user:
            return
        logger.info(f"Positions command from user {update.effective_user.id}")
        
        positions = []
        
        if self.orchestrator and hasattr(self.orchestrator, 'futures_executor'):
            try:
                pos = await self.orchestrator.futures_executor.get_position()
                if pos:
                    positions.append(pos)
            except Exception as e:
                logger.error(f"Error fetching positions: {e}")
        
        if not positions:
            message = """
📊 <b>Current Positions</b>

━━━━━━━━━━━━━━━━━━━━━

<i>No open positions currently.</i>

Positions will appear here when trades are opened.

━━━━━━━━━━━━━━━━━━━━━
"""
        else:
            pos_lines = []
            for pos in positions:
                side = getattr(pos, 'side', 'UNKNOWN')
                side_emoji = "🟢" if side == "LONG" else "🔴"
                entry = getattr(pos, 'entry_price', 0)
                qty = getattr(pos, 'quantity', 0)
                pnl = getattr(pos, 'unrealized_pnl', 0)
                pnl_emoji = "📈" if pnl >= 0 else "📉"
                
                pos_lines.append(
                    f"{side_emoji} <b>{side}</b> Position\n"
                    f"   Entry: <code>{self._format_price(entry)}</code>\n"
                    f"   Size: <code>{qty}</code>\n"
                    f"   {pnl_emoji} P&L: <code>{pnl:+.2f}%</code>"
                )
            
            pos_text = "\n\n".join(pos_lines)
            message = f"""
📊 <b>Current Positions</b>

━━━━━━━━━━━━━━━━━━━━━

{pos_text}

━━━━━━━━━━━━━━━━━━━━━

⏰ <i>{self._format_timestamp(datetime.now())}</i>
"""
        
        await update.message.reply_text(message.strip(), parse_mode=ParseMode.HTML)
    
    @admin_only
    @error_handler
    @rate_limit(max_calls=5, window_seconds=60)
    async def cmd_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /performance command - Performance statistics"""
        if not update.message or not update.effective_user:
            return
        logger.info(f"Performance command from user {update.effective_user.id}")
        
        stats = None
        
        if self.orchestrator and hasattr(self.orchestrator, 'telegram_bot'):
            try:
                stats = self.orchestrator.telegram_bot.get_performance_stats()
            except Exception as e:
                logger.error(f"Error getting performance stats: {e}")
        
        if not stats:
            stats = {
                'total_signals': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'net_profit': 0.0,
                'total_profit': 0.0,
                'total_loss': 0.0,
                'tp1_hits': 0,
                'tp2_hits': 0,
                'tp3_hits': 0,
                'sl_hits': 0
            }
        
        total_trades = stats.get('winning_trades', 0) + stats.get('losing_trades', 0)
        win_rate = stats.get('win_rate', 0)
        net_pnl = stats.get('net_profit', 0)
        pnl_emoji = "🟢" if net_pnl >= 0 else "🔴"
        pnl_text = f"+{net_pnl:.2f}%" if net_pnl >= 0 else f"{net_pnl:.2f}%"
        
        tp1 = stats.get('tp1_hits', 0)
        tp2 = stats.get('tp2_hits', 0)
        tp3 = stats.get('tp3_hits', 0)
        sl = stats.get('sl_hits', 0)
        
        message = f"""
📈 <b>Performance Statistics</b>

━━━━━━━━━━━━━━━━━━━━━

<b>📊 Overall Performance</b>
• Total Signals: <code>{stats.get('total_signals', 0)}</code>
• Closed Trades: <code>{total_trades}</code>
• Wins: <code>{stats.get('winning_trades', 0)}</code> | Losses: <code>{stats.get('losing_trades', 0)}</code>

<b>🎯 Win Rate:</b> <code>{win_rate:.1f}%</code>

{pnl_emoji} <b>Net P&L:</b> <code>{pnl_text}</code>
• Total Profit: <code>+{stats.get('total_profit', 0):.2f}%</code>
• Total Loss: <code>-{stats.get('total_loss', 0):.2f}%</code>

━━━━━━━━━━━━━━━━━━━━━

<b>🎯 Target Hits</b>
• TP1 Hits: <code>{tp1}</code>
• TP2 Hits: <code>{tp2}</code>
• TP3 Hits: <code>{tp3}</code>
• SL Hits: <code>{sl}</code>

━━━━━━━━━━━━━━━━━━━━━

⏰ <i>{self._format_timestamp(datetime.now())}</i>
"""
        await update.message.reply_text(message.strip(), parse_mode=ParseMode.HTML)
    
    @admin_only
    @error_handler
    @rate_limit(max_calls=5, window_seconds=60)
    async def cmd_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /history command - Trade history"""
        if not update.message or not update.effective_user:
            return
        logger.info(f"History command from user {update.effective_user.id}")
        
        trades = []
        
        if self.orchestrator and hasattr(self.orchestrator, 'trade_db'):
            try:
                trades = await self.orchestrator.trade_db.get_recent_trades(limit=10)
            except Exception as e:
                logger.error(f"Error fetching trade history: {e}")
        
        if not trades:
            message = """
📜 <b>Trade History</b>

━━━━━━━━━━━━━━━━━━━━━

<i>No trade history available.</i>

Completed trades will appear here.

━━━━━━━━━━━━━━━━━━━━━
"""
        else:
            trade_lines = []
            for trade in trades[:10]:
                direction = trade.get('direction', 'UNKNOWN')
                direction_emoji = "🟢" if direction == "LONG" else "🔴"
                
                outcome = trade.get('outcome', 'OPEN')
                outcome_emoji = "✅" if outcome == "WIN" else "❌" if outcome == "LOSS" else "⏳"
                
                pnl = trade.get('profit_percent', 0) or 0
                pnl_text = f"+{pnl:.2f}%" if pnl >= 0 else f"{pnl:.2f}%"
                
                symbol = trade.get('symbol', 'UNKNOWN')
                entry = trade.get('entry_price', 0)
                exit_price = trade.get('exit_price', 0)
                leverage = trade.get('leverage', 1)
                
                trade_lines.append(
                    f"{direction_emoji} <b>{symbol}</b> {direction} @ {leverage}x\n"
                    f"   Entry: <code>{self._format_price(entry)}</code>\n"
                    f"   Exit: <code>{self._format_price(exit_price) if exit_price else 'Open'}</code>\n"
                    f"   {outcome_emoji} Result: <code>{pnl_text}</code>"
                )
            
            trades_text = "\n\n".join(trade_lines)
            message = f"""
📜 <b>Trade History (Last 10)</b>

━━━━━━━━━━━━━━━━━━━━━

{trades_text}

━━━━━━━━━━━━━━━━━━━━━

⏰ <i>{self._format_timestamp(datetime.now())}</i>
"""
        
        await update.message.reply_text(message.strip(), parse_mode=ParseMode.HTML)
    
    @admin_only
    @error_handler
    @rate_limit(max_calls=5, window_seconds=60)
    async def cmd_ai(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /ai command - AI brain status and insights"""
        if not update.message or not update.effective_user:
            return
        logger.info(f"AI command from user {update.effective_user.id}")
        
        ai_status = {
            'available': False,
            'model': 'N/A',
            'total_analyses': 0,
            'cache_size': 0
        }
        
        if self.orchestrator and hasattr(self.orchestrator, 'ai_brain'):
            ai_brain = self.orchestrator.ai_brain
            ai_status['available'] = ai_brain.ai_available
            ai_status['model'] = ai_brain.MODEL if ai_brain.ai_available else 'Fallback'
            ai_status['cache_size'] = len(ai_brain._cache) if hasattr(ai_brain, '_cache') else 0
        
        ai_emoji = "🧠" if ai_status['available'] else "⚠️"
        status_text = "Active & Learning" if ai_status['available'] else "Fallback Mode"
        
        message = f"""
🤖 <b>AI Trading Brain</b>

━━━━━━━━━━━━━━━━━━━━━

{ai_emoji} <b>Status:</b> {status_text}

<b>📊 Configuration</b>
• Model: <code>{ai_status['model']}</code>
• Cache Size: <code>{ai_status['cache_size']} entries</code>

<b>🎯 Capabilities</b>
• Signal Confidence Scoring
• Market Sentiment Analysis
• Risk Assessment
• Trade Outcome Learning
• Parameter Optimization

━━━━━━━━━━━━━━━━━━━━━

<b>💡 AI Integration</b>
The AI brain analyzes each signal for:
• Entry timing optimization
• Risk/reward assessment
• Market condition analysis
• Historical pattern matching

━━━━━━━━━━━━━━━━━━━━━

⏰ <i>{self._format_timestamp(datetime.now())}</i>
"""
        await update.message.reply_text(message.strip(), parse_mode=ParseMode.HTML)
    
    @admin_only
    @error_handler
    @rate_limit(max_calls=5, window_seconds=60)
    async def cmd_insights(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /insights command - Latest AI learning insights"""
        if not update.message or not update.effective_user:
            return
        logger.info(f"Insights command from user {update.effective_user.id}")
        
        insights = []
        
        if self.orchestrator and hasattr(self.orchestrator, 'ai_brain'):
            try:
                ai_brain = self.orchestrator.ai_brain
                if hasattr(ai_brain, 'get_learning_metrics'):
                    metrics = await ai_brain.get_learning_metrics()
                    if metrics:
                        insights = metrics
            except Exception as e:
                logger.error(f"Error fetching AI insights: {e}")
        
        if not insights:
            message = """
💡 <b>AI Learning Insights</b>

━━━━━━━━━━━━━━━━━━━━━

<i>No insights available yet.</i>

The AI brain will generate insights as it:
• Analyzes trading signals
• Learns from trade outcomes
• Identifies market patterns
• Optimizes parameters

━━━━━━━━━━━━━━━━━━━━━

<b>🔄 How It Works:</b>
1. Each signal is analyzed by GPT-5
2. Trade outcomes are recorded
3. Patterns are identified
4. Parameters are optimized

━━━━━━━━━━━━━━━━━━━━━
"""
        else:
            insight_lines = []
            for insight in insights[:5]:
                name = insight.get('metric_name', 'Unknown')
                value = insight.get('metric_value', 0)
                context_info = insight.get('context', '')
                insight_lines.append(f"• <b>{name}:</b> <code>{value:.2f}</code>\n  {context_info}")
            
            insights_text = "\n\n".join(insight_lines)
            message = f"""
💡 <b>AI Learning Insights</b>

━━━━━━━━━━━━━━━━━━━━━

{insights_text}

━━━━━━━━━━━━━━━━━━━━━

<b>🧠 Continuous Learning</b>
The AI analyzes every trade outcome to improve future signal accuracy.

━━━━━━━━━━━━━━━━━━━━━

⏰ <i>{self._format_timestamp(datetime.now())}</i>
"""
        
        await update.message.reply_text(message.strip(), parse_mode=ParseMode.HTML)
    
    @admin_only
    @error_handler
    @rate_limit(max_calls=5, window_seconds=60)
    async def cmd_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /settings command - View current configuration"""
        if not update.message or not update.effective_user:
            return
        logger.info(f"Settings command from user {update.effective_user.id}")
        
        config = None
        if self.orchestrator and hasattr(self.orchestrator, 'config'):
            config = self.orchestrator.config
        
        if config:
            symbol = config.trading.symbol
            timeframe = config.trading.timeframe
            
            ut_key = config.ut_bot.key_value
            ut_atr = config.ut_bot.atr_period
            ut_ha = "ON" if config.ut_bot.use_heikin_ashi else "OFF"
            
            stc_length = config.stc.length
            stc_fast = config.stc.fast_length
            stc_slow = config.stc.slow_length
            
            leverage_enabled = config.trading.leverage.enabled
            min_lev = config.trading.leverage.min_leverage
            max_lev = config.trading.leverage.max_leverage
            base_lev = config.trading.leverage.base_leverage
            risk_pct = config.trading.leverage.risk_per_trade_percent
            
            auto_status = "🟢 Enabled" if leverage_enabled else "🔴 Disabled"
        else:
            symbol = "ETHUSDT"
            timeframe = "5m"
            ut_key = 2.0
            ut_atr = 6
            ut_ha = "ON"
            stc_length = 80
            stc_fast = 27
            stc_slow = 50
            min_lev = 1
            max_lev = 25
            base_lev = 12
            risk_pct = 2.0
            auto_status = "⚠️ Unknown"
        
        message = f"""
⚙️ <b>Current Settings</b>

━━━━━━━━━━━━━━━━━━━━━

<b>📊 Trading Pair</b>
• Symbol: <code>{symbol}</code>
• Timeframe: <code>{timeframe}</code>

<b>📈 UT Bot Indicator</b>
• Key Value: <code>{ut_key}</code>
• ATR Period: <code>{ut_atr}</code>
• Heikin Ashi: <code>{ut_ha}</code>

<b>📉 STC Indicator</b>
• Length: <code>{stc_length}</code>
• Fast Length: <code>{stc_fast}</code>
• Slow Length: <code>{stc_slow}</code>

━━━━━━━━━━━━━━━━━━━━━

<b>⚡ Leverage Settings</b>
• Auto Trading: {auto_status}
• Range: <code>{min_lev}x - {max_lev}x</code>
• Base: <code>{base_lev}x</code>
• Risk per Trade: <code>{risk_pct}%</code>

━━━━━━━━━━━━━━━━━━━━━

💡 Use /leverage and /risk to adjust settings.

⏰ <i>{self._format_timestamp(datetime.now())}</i>
"""
        await update.message.reply_text(message.strip(), parse_mode=ParseMode.HTML)
    
    @admin_only
    @error_handler
    @rate_limit(max_calls=10, window_seconds=60)
    async def cmd_market(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /market command - Current market state"""
        if not update.message or not update.effective_user:
            return
        logger.info(f"Market command from user {update.effective_user.id}")
        
        market_data = {
            'price': 0,
            'stc_value': 0,
            'stc_color': 'neutral',
            'ut_color': 'neutral',
            'atr': 0
        }
        symbol = "ETHUSDT"
        
        if self.orchestrator:
            if hasattr(self.orchestrator, 'config'):
                symbol = self.orchestrator.config.trading.symbol
            
            if hasattr(self.orchestrator, 'data_fetcher'):
                try:
                    df = self.orchestrator.data_fetcher.fetch_historical_data(limit=50)
                    if df is not None and len(df) > 0:
                        market_data['price'] = float(df['close'].iloc[-1])
                        atr = (df['high'].iloc[-14:].max() - df['low'].iloc[-14:].min()) / 14
                        market_data['atr'] = float(atr)
                        
                        if hasattr(self.orchestrator, 'signal_engine'):
                            state = self.orchestrator.signal_engine.get_market_state(df)
                            market_data['stc_value'] = state.get('stc_value', 0)
                            market_data['stc_color'] = state.get('stc_color', 'neutral')
                            market_data['ut_color'] = state.get('ut_bar_color', 'neutral')
                except Exception as e:
                    logger.error(f"Error fetching market data: {e}")
        
        price = market_data['price']
        stc = market_data['stc_value']
        stc_color = market_data['stc_color']
        ut_color = market_data['ut_color']
        atr = market_data['atr']
        
        stc_emoji = "🟢" if stc_color == "green" else "🔴" if stc_color == "red" else "⚪"
        ut_emoji = "🟢" if ut_color == "green" else "🔴" if ut_color == "red" else "⚪"
        
        if stc > 75:
            stc_zone = "Overbought"
        elif stc < 25:
            stc_zone = "Oversold"
        else:
            stc_zone = "Neutral"
        
        message = f"""
📊 <b>Market State - {symbol}</b>

━━━━━━━━━━━━━━━━━━━━━

<b>💰 Current Price</b>
<code>{self._format_price(price)}</code>

━━━━━━━━━━━━━━━━━━━━━

<b>📈 Indicators</b>

{stc_emoji} <b>STC:</b> <code>{stc:.2f}</code>
   Zone: <code>{stc_zone}</code>
   Color: <code>{stc_color.upper()}</code>

{ut_emoji} <b>UT Bot:</b> <code>{ut_color.upper()}</code>

<b>📏 ATR:</b> <code>{atr:.4f}</code>

━━━━━━━━━━━━━━━━━━━━━

<b>🎯 Signal Conditions</b>
• LONG: UT Bot Green + STC Green ↑
• SHORT: UT Bot Red + STC Red ↓

━━━━━━━━━━━━━━━━━━━━━

⏰ <i>{self._format_timestamp(datetime.now())}</i>
"""
        await update.message.reply_text(message.strip(), parse_mode=ParseMode.HTML)
    
    @admin_only
    @error_handler
    @rate_limit(max_calls=5, window_seconds=60)
    async def cmd_leverage(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handle /leverage command - View/set leverage with inline buttons"""
        if not update.message or not update.effective_user:
            return ConversationHandler.END
        logger.info(f"Leverage command from user {update.effective_user.id}")
        
        current_leverage = 12
        if self.orchestrator and hasattr(self.orchestrator, 'config'):
            current_leverage = self.orchestrator.config.trading.leverage.base_leverage
        
        if 'leverage' in self._settings_overrides:
            current_leverage = self._settings_overrides['leverage']
        
        buttons = []
        row = []
        for lev in LEVERAGE_OPTIONS:
            emoji = "✅ " if lev == current_leverage else ""
            row.append(InlineKeyboardButton(f"{emoji}{lev}x", callback_data=f"leverage_{lev}"))
            if len(row) == 4:
                buttons.append(row)
                row = []
        if row:
            buttons.append(row)
        
        buttons.append([InlineKeyboardButton("❌ Cancel", callback_data="leverage_cancel")])
        
        keyboard = InlineKeyboardMarkup(buttons)
        
        message = f"""
⚡ <b>Leverage Settings</b>

━━━━━━━━━━━━━━━━━━━━━

<b>Current Leverage:</b> <code>{current_leverage}x</code>

Select a new leverage level below:
"""
        
        await update.message.reply_text(
            message.strip(),
            parse_mode=ParseMode.HTML,
            reply_markup=keyboard
        )
        
        return LEVERAGE_SELECT
    
    @error_handler
    async def leverage_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handle leverage selection callback"""
        if not update.callback_query or not update.effective_user:
            return ConversationHandler.END
        query = update.callback_query
        await query.answer()
        
        data = query.data
        if not data:
            return ConversationHandler.END
        
        if data == "leverage_cancel":
            await query.edit_message_text(
                "❌ Leverage change cancelled.",
                parse_mode=ParseMode.HTML
            )
            return ConversationHandler.END
        
        try:
            leverage = int(data.replace("leverage_", ""))
            
            self._settings_overrides['leverage'] = leverage
            
            if self.orchestrator and hasattr(self.orchestrator, 'config'):
                self.orchestrator.config.trading.leverage.base_leverage = leverage
            
            message = f"""
✅ <b>Leverage Updated</b>

━━━━━━━━━━━━━━━━━━━━━

<b>New Leverage:</b> <code>{leverage}x</code>

This setting will be used for future trades.

⏰ <i>{self._format_timestamp(datetime.now())}</i>
"""
            await query.edit_message_text(message.strip(), parse_mode=ParseMode.HTML)
            logger.info(f"Leverage updated to {leverage}x by user {update.effective_user.id}")
            
        except Exception as e:
            logger.error(f"Error updating leverage: {e}")
            await query.edit_message_text(
                f"❌ Error updating leverage: {str(e)}",
                parse_mode=ParseMode.HTML
            )
        
        return ConversationHandler.END
    
    @admin_only
    @error_handler
    @rate_limit(max_calls=5, window_seconds=60)
    async def cmd_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handle /risk command - View/set risk level with inline buttons"""
        if not update.message or not update.effective_user:
            return ConversationHandler.END
        logger.info(f"Risk command from user {update.effective_user.id}")
        
        current_risk = 2.0
        if self.orchestrator and hasattr(self.orchestrator, 'config'):
            current_risk = self.orchestrator.config.trading.leverage.risk_per_trade_percent
        
        if 'risk' in self._settings_overrides:
            current_risk = self._settings_overrides['risk']
        
        buttons = []
        row = []
        for risk in RISK_OPTIONS:
            emoji = "✅ " if abs(risk - current_risk) < 0.01 else ""
            row.append(InlineKeyboardButton(f"{emoji}{risk}%", callback_data=f"risk_{risk}"))
            if len(row) == 3:
                buttons.append(row)
                row = []
        if row:
            buttons.append(row)
        
        buttons.append([InlineKeyboardButton("❌ Cancel", callback_data="risk_cancel")])
        
        keyboard = InlineKeyboardMarkup(buttons)
        
        message = f"""
⚠️ <b>Risk Settings</b>

━━━━━━━━━━━━━━━━━━━━━

<b>Current Risk:</b> <code>{current_risk}%</code> per trade

Select a new risk level below:

<i>⚠️ Higher risk = higher potential gains AND losses</i>
"""
        
        await update.message.reply_text(
            message.strip(),
            parse_mode=ParseMode.HTML,
            reply_markup=keyboard
        )
        
        return RISK_SELECT
    
    @error_handler
    async def risk_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handle risk selection callback"""
        if not update.callback_query or not update.effective_user:
            return ConversationHandler.END
        query = update.callback_query
        await query.answer()
        
        data = query.data
        if not data:
            return ConversationHandler.END
        
        if data == "risk_cancel":
            await query.edit_message_text(
                "❌ Risk change cancelled.",
                parse_mode=ParseMode.HTML
            )
            return ConversationHandler.END
        
        try:
            risk = float(data.replace("risk_", ""))
            
            self._settings_overrides['risk'] = risk
            
            if self.orchestrator and hasattr(self.orchestrator, 'config'):
                self.orchestrator.config.trading.leverage.risk_per_trade_percent = risk
            
            risk_level = "Low" if risk <= 1.0 else "Medium" if risk <= 2.0 else "High"
            risk_emoji = "🟢" if risk <= 1.0 else "🟡" if risk <= 2.0 else "🔴"
            
            message = f"""
✅ <b>Risk Level Updated</b>

━━━━━━━━━━━━━━━━━━━━━

<b>New Risk:</b> <code>{risk}%</code> per trade
{risk_emoji} <b>Level:</b> {risk_level}

This setting will be used for future trades.

⏰ <i>{self._format_timestamp(datetime.now())}</i>
"""
            await query.edit_message_text(message.strip(), parse_mode=ParseMode.HTML)
            logger.info(f"Risk updated to {risk}% by user {update.effective_user.id}")
            
        except Exception as e:
            logger.error(f"Error updating risk: {e}")
            await query.edit_message_text(
                f"❌ Error updating risk: {str(e)}",
                parse_mode=ParseMode.HTML
            )
        
        return ConversationHandler.END
    
    @admin_only
    @error_handler
    @rate_limit(max_calls=3, window_seconds=60)
    async def cmd_toggle_auto(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /toggle_auto command - Toggle auto trading"""
        if not update.message or not update.effective_user:
            return
        logger.info(f"Toggle auto command from user {update.effective_user.id}")
        
        current_state = False
        new_state = True
        
        if self.orchestrator:
            current_state = getattr(self.orchestrator, 'auto_trading_enabled', False)
            new_state = not current_state
            self.orchestrator.auto_trading_enabled = new_state
            
            if hasattr(self.orchestrator, 'config'):
                self.orchestrator.config.trading.leverage.enabled = new_state
        
        status_emoji = "🟢" if new_state else "🔴"
        status_text = "ENABLED" if new_state else "DISABLED"
        action_text = "enabled" if new_state else "disabled"
        
        warning = ""
        if new_state:
            warning = "\n\n⚠️ <b>Warning:</b> Auto trading will execute real trades!"
        
        message = f"""
🔄 <b>Auto Trading Toggled</b>

━━━━━━━━━━━━━━━━━━━━━

{status_emoji} <b>Auto Trading:</b> {status_text}

Auto trading has been {action_text}.
{warning}

━━━━━━━━━━━━━━━━━━━━━

⏰ <i>{self._format_timestamp(datetime.now())}</i>
"""
        await update.message.reply_text(message.strip(), parse_mode=ParseMode.HTML)
        logger.info(f"Auto trading {'enabled' if new_state else 'disabled'} by user {update.effective_user.id}")
    
    @error_handler
    async def cmd_cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handle /cancel command - Cancel current operation"""
        if not update.message or not update.effective_user:
            return ConversationHandler.END
        logger.info(f"Cancel command from user {update.effective_user.id}")
        
        await update.message.reply_text(
            "❌ <b>Operation Cancelled</b>\n\nUse /help to see available commands.",
            parse_mode=ParseMode.HTML
        )
        return ConversationHandler.END
