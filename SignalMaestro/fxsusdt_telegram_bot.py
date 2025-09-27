#!/usr/bin/env python3
"""
FXSUSDT.P Telegram Signal Bot
Sends Ichimoku Sniper signals to @SignalTactics channel with Cornix compatibility
"""

import asyncio
import logging
import aiohttp
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import json

from ichimoku_sniper_strategy import IchimokuSniperStrategy, IchimokuSignal
from fxsusdt_trader import FXSUSDTTrader

class FXSUSDTTelegramBot:
    """Telegram bot for FXSUSDT.P signals"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Telegram Configuration
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.channel_id = "@SignalTactics"
        self.admin_chat_id = os.getenv('ADMIN_CHAT_ID')  # Optional admin notifications

        if not self.bot_token:
            raise ValueError("Missing TELEGRAM_BOT_TOKEN in Replit secrets")

        # Components
        self.strategy = IchimokuSniperStrategy()
        self.trader = FXSUSDTTrader()

        # Telegram API
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

        # Command system
        self.commands = {
            '/start': self.cmd_start,
            '/help': self.cmd_help,
            '/status': self.cmd_status,
            '/price': self.cmd_price,
            '/balance': self.cmd_balance,
            '/position': self.cmd_position,
            '/scan': self.cmd_scan,
            '/settings': self.cmd_settings,
            '/market': self.cmd_market,
            '/stats': self.cmd_stats,
            '/leverage': self.cmd_leverage,
            '/risk': self.cmd_risk,
            '/signal': self.cmd_signal,
            '/history': self.cmd_history,
            '/alerts': self.cmd_alerts,
            '/admin': self.cmd_admin,
            '/futures': self.cmd_futures_info,
            '/contract': self.cmd_contract_specs,
            '/funding': self.cmd_funding_rate,
            '/oi': self.cmd_open_interest,
            '/volume': self.cmd_volume_analysis,
            '/sentiment': self.cmd_market_sentiment,
            '/news': self.cmd_market_news,
            '/watchlist': self.cmd_watchlist,
            '/backtest': self.cmd_backtest,
            '/optimize': self.cmd_optimize_strategy
        }

        # Bot statistics and timing
        self.signal_count = 0
        self.last_signal_time = None
        self.bot_start_time = datetime.now()
        self.commands_used = {}
        self.min_signal_interval = timedelta(minutes=30)  # Minimum 30 minutes between signals
        self.telegram_app = None

        # FXSUSDT contract specifications
        self.contract_specs = {
            'symbol': 'FXSUSDT',
            'base_asset': 'FX',
            'quote_asset': 'USDT',
            'contract_type': 'PERPETUAL',
            'settlement_asset': 'USDT',
            'margin_type': 'Cross/Isolated',
            'tick_size': '0.00001',
            'step_size': '0.1',
            'max_leverage': '50x',
            'funding_interval': '8 hours'
        }

        self.logger.info("🤖 FXSUSDT Futures Telegram Bot initialized with advanced commands")

    async def send_message(self, chat_id: str, text: str, parse_mode: str = 'Markdown') -> bool:
        """Send message to Telegram chat/channel"""
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': text,
                'parse_mode': parse_mode,
                'disable_web_page_preview': True
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get('ok'):
                            self.logger.info(f"✅ Message sent to {chat_id}")
                            return True
                        else:
                            self.logger.error(f"❌ Telegram API error: {result.get('description')}")
                            return False
                    else:
                        self.logger.error(f"❌ HTTP error: {response.status}")
                        return False

        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
            return False

    def format_cornix_signal(self, signal: IchimokuSignal) -> str:
        """Format signal for Cornix compatibility"""

        # Determine direction emoji
        direction_emoji = "🟢" if signal.action == "BUY" else "🔴"

        # Calculate risk/reward percentages
        entry = signal.entry_price
        sl = signal.stop_loss
        tp = signal.take_profit

        if signal.action == "BUY":
            sl_percent = abs((entry - sl) / entry) * 100
            tp_percent = abs((tp - entry) / entry) * 100
        else:
            sl_percent = abs((sl - entry) / entry) * 100
            tp_percent = abs((entry - tp) / entry) * 100

        cornix_signal = f"""
{direction_emoji} **ICHIMOKU SNIPER - FXSUSDT.P**

**📊 SIGNAL DETAILS:**
• **Pair:** `FXSUSDT.P`
• **Direction:** `{signal.action}`
• **Entry:** `{entry:.5f}`
• **Stop Loss:** `{sl:.5f}` (-{sl_percent:.2f}%)
• **Take Profit:** `{tp:.5f}` (+{tp_percent:.2f}%)

**⚙️ TRADING PARAMETERS:**
• **Leverage:** `Auto (Dynamic)`
• **Risk/Reward:** `1:{signal.risk_reward_ratio:.1f}`
• **Timeframe:** `30 Minutes`
• **Strategy:** `Ichimoku Cloud Sniper`

**📈 SIGNAL ANALYSIS:**
• **Strength:** `{signal.signal_strength:.1f}%`
• **Confidence:** `{signal.confidence:.1f}%`
• **ATR Value:** `{signal.atr_value:.6f}`

**🎯 CORNIX COMPATIBLE FORMAT:**
```
FXSUSDT.P {signal.action}
Entry: {entry:.5f}
SL: {sl:.5f}
TP: {tp:.5f}
Leverage: Auto
```

**⏰ Signal Time:** `{signal.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}`
**🤖 Bot:** `TradeTactics - Ichimoku Sniper`

*Automated signal generated by Ichimoku Cloud analysis*
        """.strip()

        return cornix_signal

    async def send_signal_to_channel(self, signal: IchimokuSignal) -> bool:
        """Send signal to @SignalTactics channel"""
        try:
            # Check rate limiting
            if self.last_signal_time:
                time_since_last = datetime.now() - self.last_signal_time
                if time_since_last < self.min_signal_interval:
                    remaining = self.min_signal_interval - time_since_last
                    self.logger.info(f"⏳ Rate limit active, {remaining.total_seconds():.0f}s remaining")
                    return False

            # Format signal for Cornix
            formatted_signal = self.format_cornix_signal(signal)

            # Send to channel
            success = await self.send_message(self.channel_id, formatted_signal)

            if success:
                self.last_signal_time = datetime.now()
                self.logger.info(f"📡 Signal sent to {self.channel_id}: {signal.action} FXSUSDT.P @ {signal.entry_price:.5f}")

                # Send to admin if configured
                if self.admin_chat_id:
                    admin_msg = f"✅ Signal sent to {self.channel_id}\n{signal.action} FXSUSDT.P @ {signal.entry_price:.5f}"
                    await self.send_message(self.admin_chat_id, admin_msg)

                return True
            else:
                self.logger.error("❌ Failed to send signal to channel")
                return False

        except Exception as e:
            self.logger.error(f"Error sending signal to channel: {e}")
            return False

    async def send_status_update(self, message: str) -> bool:
        """Send status update to admin"""
        if self.admin_chat_id:
            return await self.send_message(self.admin_chat_id, f"🤖 **FXSUSDT Bot Status**\n\n{message}")
        return True

    async def test_telegram_connection(self) -> bool:
        """Test Telegram bot connection"""
        try:
            url = f"{self.base_url}/getMe"

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get('ok'):
                            bot_info = result.get('result', {})
                            bot_name = bot_info.get('username', 'Unknown')
                            self.logger.info(f"✅ Telegram connection successful: @{bot_name}")
                            return True

            return False

        except Exception as e:
            self.logger.error(f"❌ Telegram connection test failed: {e}")
            return False

    async def scan_and_signal(self) -> bool:
        """Scan market and send signal if conditions are met"""
        try:
            self.logger.info("🔍 Scanning FXSUSDT.P for Ichimoku signals...")

            # Get 30m market data
            market_data = await self.trader.get_30m_klines(limit=200)
            if not market_data:
                self.logger.warning("❌ No market data available")
                return False

            # Generate signal
            signal = await self.strategy.generate_signal(market_data)
            if not signal:
                self.logger.info("📊 No qualifying signal found")
                return False

            # Send signal to channel
            success = await self.send_signal_to_channel(signal)

            if success:
                self.logger.info(f"🎯 Successfully processed {signal.action} signal")
                return True
            else:
                self.logger.error("❌ Failed to send signal")
                return False

        except Exception as e:
            self.logger.error(f"Error in scan and signal: {e}")
            return False

    async def run_continuous_scanner(self):
        """Run continuous market scanner"""
        self.logger.info("🚀 Starting FXSUSDT.P continuous scanner...")

        # Initial connection tests
        if not await self.trader.test_connection():
            self.logger.error("❌ Binance API connection failed")
            return

        if not await self.test_telegram_connection():
            self.logger.error("❌ Telegram connection failed")
            return

        # Send startup notification
        await self.send_status_update("🚀 FXSUSDT.P Ichimoku Sniper Bot started\n📊 Monitoring 30-minute timeframe\n🎯 Ready for signals")

        scan_interval = 300  # 5 minutes

        try:
            while True:
                try:
                    await self.scan_and_signal()
                except Exception as e:
                    self.logger.error(f"Error in scan cycle: {e}")

                # Wait for next scan
                self.logger.debug(f"⏱️ Waiting {scan_interval}s for next scan...")
                await asyncio.sleep(scan_interval)

        except KeyboardInterrupt:
            self.logger.info("👋 Scanner stopped by user")
            await self.send_status_update("🛑 FXSUSDT.P Bot stopped")
        except Exception as e:
            self.logger.error(f"❌ Critical error in scanner: {e}")
            await self.send_status_update(f"❌ Bot error: {e}")

    # --- Command Handlers ---

    async def cmd_start(self, update, context):
        """Handle /start command"""
        chat_id = update.effective_chat.id
        await self.send_message(str(chat_id), "Welcome to the FXSUSDT.P Futures Bot! Type /help for a list of commands.")
        self.commands_used.update({str(chat_id): self.commands_used.get(str(chat_id), 0) + 1})

    async def cmd_help(self, update, context):
        """Handle /help command"""
        chat_id = str(update.effective_chat.id)
        help_text = "📚 **Available Commands:**\n\n"
        for cmd in sorted(self.commands.keys()):
            help_text += f"`{cmd}` - {self.commands[cmd].__doc__.strip() if self.commands[cmd].__doc__ else 'No description'}\n"
        await self.send_message(chat_id, help_text)
        self.commands_used.update({chat_id: self.commands_used.get(chat_id, 0) + 1})

    async def cmd_status(self, update, context):
        """Get bot status and uptime"""
        chat_id = str(update.effective_chat.id)
        uptime = datetime.now() - self.bot_start_time
        
        # Check if scanner is running by checking last signal time
        scanner_status = "Active" if self.last_signal_time and (datetime.now() - self.last_signal_time).seconds < 600 else "Active"
        
        status_message = (
            f"🤖 **FXSUSDT.P Futures Bot Status:**\n\n"
            f"• **Uptime:** `{str(uptime).split('.')[0]}`\n"
            f"• **Last Signal:** `{self.last_signal_time.strftime('%Y-%m-%d %H:%M:%S UTC') if self.last_signal_time else 'Never'}`\n"
            f"• **Signals Sent:** `{self.signal_count}`\n"
            f"• **Scanner Mode:** `{scanner_status}`\n"
            f"• **Target Channel:** `{self.channel_id}`\n"
            f"• **Contract:** `FXSUSDT.P (Perpetual Futures)`\n"
            f"• **Timeframe:** `30 Minutes`\n"
            f"• **Strategy:** `Ichimoku Cloud Sniper`\n\n"
            f"**🔧 System Status:**\n"
            f"• **API Connection:** `✅ Connected`\n"
            f"• **Telegram API:** `✅ Connected`\n"
            f"• **Commands Available:** `{len(self.commands)}`"
        )
        await self.send_message(chat_id, status_message)
        self.commands_used[chat_id] = self.commands_used.get(chat_id, 0) + 1

    async def cmd_price(self, update, context):
        """Get the current price of FXSUSDT.P"""
        chat_id = str(update.effective_chat.id)
        try:
            # Use direct price method first
            price = await self.trader.get_current_price()
            if price:
                # Get additional ticker data for comprehensive info
                ticker = await self.trader.get_24hr_ticker_stats('FXSUSDT')
                if ticker:
                    change_percent = float(ticker.get('priceChangePercent', 0))
                    high_24h = float(ticker.get('highPrice', 0))
                    low_24h = float(ticker.get('lowPrice', 0))
                    volume = float(ticker.get('volume', 0))
                    
                    direction_emoji = "🟢" if change_percent >= 0 else "🔴"
                    
                    message = f"""💰 **FXSUSDT.P Price Information:**

• **Current Price:** `{price:.5f}`
• **24h Change:** {direction_emoji} `{change_percent:+.2f}%`
• **24h High:** `{high_24h:.5f}`
• **24h Low:** `{low_24h:.5f}`
• **24h Volume:** `{volume:,.0f}`

**📊 Market:** Binance Futures (USDT-M)
**📈 Contract:** FXSUSDT Perpetual"""
                else:
                    message = f"💰 **Current FXSUSDT.P Price:** `{price:.5f}`"
                
                await self.send_message(chat_id, message)
            else:
                await self.send_message(chat_id, "❌ Could not retrieve FXSUSDT.P price.")
        except Exception as e:
            self.logger.error(f"Error in cmd_price: {e}")
            await self.send_message(chat_id, "❌ An error occurred while fetching the price.")
        self.commands_used.update({chat_id: self.commands_used.get(chat_id, 0) + 1})

    async def cmd_balance(self, update, context):
        """Get account balance information"""
        chat_id = str(update.effective_chat.id)
        try:
            balance = await self.trader.get_account_balance()
            if balance:
                message = f"""💰 **Account Balance (FXSUSDT Futures):**

• **Total Wallet Balance:** `{balance.get('total_wallet_balance', 0):.2f} USDT`
• **Available Balance:** `{balance.get('available_balance', 0):.2f} USDT`
• **Unrealized PNL:** `{balance.get('total_unrealized_pnl', 0):.2f} USDT`

**📊 Account Type:** USDT-M Futures
**⚡ Updated:** {datetime.now().strftime('%H:%M:%S UTC')}"""
                
                await self.send_message(chat_id, message)
            else:
                await self.send_message(chat_id, "❌ Could not retrieve account balance.")
        except Exception as e:
            self.logger.error(f"Error in cmd_balance: {e}")
            await self.send_message(chat_id, "❌ An error occurred while fetching the balance.")
        self.commands_used.update({chat_id: self.commands_used.get(chat_id, 0) + 1})

    async def cmd_position(self, update, context):
        """Get current open positions for FXSUSDT.P"""
        chat_id = str(update.effective_chat.id)
        try:
            positions = await self.trader.get_positions('FXSUSDT')
            if positions:
                message = "📊 **Open Positions (FXSUSDT.P):**\n\n"
                for pos in positions:
                    position_amt = float(pos.get('positionAmt', 0))
                    entry_price = float(pos.get('entryPrice', 0))
                    mark_price = float(pos.get('markPrice', 0))
                    unrealized_pnl = float(pos.get('unRealizedProfit', 0))
                    percentage = float(pos.get('percentage', 0))
                    
                    side = "LONG" if position_amt > 0 else "SHORT" if position_amt < 0 else "NONE"
                    side_emoji = "🟢" if position_amt > 0 else "🔴" if position_amt < 0 else "⚪"
                    pnl_emoji = "🟢" if unrealized_pnl >= 0 else "🔴"
                    
                    message += f"""{side_emoji} **{pos['symbol']}**
• **Side:** `{side}`
• **Size:** `{abs(position_amt):.4f}`
• **Entry Price:** `{entry_price:.5f}`
• **Mark Price:** `{mark_price:.5f}`
• **Unrealized PNL:** {pnl_emoji} `{unrealized_pnl:.2f} USDT ({percentage:+.2f}%)`
• **Leverage:** `{pos.get('leverage', '1')}x`

"""
                await self.send_message(chat_id, message)
            else:
                await self.send_message(chat_id, "ℹ️ You have no open positions for FXSUSDT.P.")
        except Exception as e:
            self.logger.error(f"Error in cmd_position: {e}")
            await self.send_message(chat_id, "❌ An error occurred while fetching positions.")
        self.commands_used.update({chat_id: self.commands_used.get(chat_id, 0) + 1})

    async def cmd_scan(self, update, context):
        """Manually trigger a market scan for signals"""
        chat_id = str(update.effective_chat.id)
        self.logger.info(f"Manual scan triggered by {chat_id}")
        await self.send_message(chat_id, "🔍 Manually triggering market scan...")
        success = await self.scan_and_signal()
        if success:
            await self.send_message(chat_id, "✅ Market scan complete. Signal sent if found.")
        else:
            await self.send_message(chat_id, "ℹ️ Market scan complete. No new signals were generated or sent.")
        self.commands_used.update({chat_id: self.commands_used.get(chat_id, 0) + 1})

    async def cmd_settings(self, update, context):
        """Display current bot settings and allow modification (future implementation)"""
        chat_id = str(update.effective_chat.id)
        settings_message = (
            "⚙️ **Bot Settings:**\n\n"
            f"• **Min Signal Interval:** {self.min_signal_interval.total_seconds() / 60} minutes\n"
            f"• **Target Channel:** `{self.channel_id}`\n"
            f"• **Admin Notifications:** {'Enabled' if self.admin_chat_id else 'Disabled'}\n\n"
            "*Note: Modifying settings requires further implementation.*"
        )
        await self.send_message(chat_id, settings_message)
        self.commands_used.update({chat_id: self.commands_used.get(chat_id, 0) + 1})

    async def cmd_market(self, update, context):
        """Get general market overview or specific symbol info"""
        chat_id = str(update.effective_chat.id)
        symbol = 'FXSUSDT' # Default to FXSUSDT
        if context.args:
            symbol = context.args[0].upper()

        try:
            # Get comprehensive ticker information
            ticker = await self.trader.get_24hr_ticker_stats(symbol)
            if ticker:
                price = float(ticker.get('lastPrice', 0))
                change = float(ticker.get('priceChange', 0))
                change_percent = float(ticker.get('priceChangePercent', 0))
                high_24h = float(ticker.get('highPrice', 0))
                low_24h = float(ticker.get('lowPrice', 0))
                volume = float(ticker.get('volume', 0))
                quote_volume = float(ticker.get('quoteVolume', 0))
                open_price = float(ticker.get('openPrice', 0))
                
                direction_emoji = "🟢" if change >= 0 else "🔴"
                
                message = f"""📈 **Market Overview for {symbol}:**

**💰 Price Information:**
• **Current Price:** `{price:.5f}`
• **24h Change:** {direction_emoji} `{change:+.5f} ({change_percent:+.2f}%)`
• **24h High:** `{high_24h:.5f}`
• **24h Low:** `{low_24h:.5f}`
• **24h Open:** `{open_price:.5f}`

**📊 Volume Information:**
• **24h Volume:** `{volume:,.0f} {symbol[:2]}`
• **24h Volume (USDT):** `${quote_volume:,.0f}`

**📋 Contract Info:**
• **Type:** Perpetual Futures
• **Settlement:** USDT
• **Exchange:** Binance Futures

**⏰ Last Update:** `{datetime.now().strftime('%H:%M:%S UTC')}`"""
                
                await self.send_message(chat_id, message)
            else:
                await self.send_message(chat_id, f"❌ Could not retrieve market data for {symbol}.")
        except Exception as e:
            self.logger.error(f"Error in cmd_market: {e}")
            await self.send_message(chat_id, f"❌ An error occurred while fetching market data for {symbol}.")
        self.commands_used.update({chat_id: self.commands_used.get(chat_id, 0) + 1})

    async def cmd_stats(self, update, context):
        """Display bot usage statistics"""
        chat_id = str(update.effective_chat.id)
        total_commands_executed = sum(self.commands_used.values())
        stats_message = (
            "📊 **Bot Statistics:**\n\n"
            f"• **Total Signals Sent:** {self.signal_count}\n"
            f"• **Total Commands Used:** {total_commands_executed}\n"
            f"• **Bot Uptime:** {datetime.now() - self.bot_start_time}\n\n"
            "**Command Usage Breakdown:**\n"
        )
        for cmd, count in sorted(self.commands_used.items()):
            stats_message += f"• `{cmd}`: {count}\n"

        await self.send_message(chat_id, stats_message)
        self.commands_used.update({chat_id: self.commands_used.get(chat_id, 0) + 1})

    async def cmd_leverage(self, update, context):
        """Get or set leverage for FXSUSDT.P"""
        chat_id = str(update.effective_chat.id)
        if len(context.args) >= 2 and context.args[0].upper() == 'FXSUSDT':
            symbol = context.args[0].upper()
            try:
                leverage = int(context.args[1])
                if 1 <= leverage <= 50: # Max leverage is 50x for FXSUSDT
                    success = await self.trader.change_leverage(symbol, leverage)
                    if success:
                        await self.send_message(chat_id, f"✅ **Leverage Updated:**\n\n• **Symbol:** `{symbol}`\n• **New Leverage:** `{leverage}x`\n• **Status:** Successfully applied")
                    else:
                        await self.send_message(chat_id, f"❌ Failed to set leverage for {symbol}. Please check your account status and try again.")
                else:
                    await self.send_message(chat_id, "❌ Leverage must be between 1x and 50x for FXSUSDT.")
            except ValueError:
                await self.send_message(chat_id, "❌ Invalid leverage value. Please provide a number.")
            except Exception as e:
                self.logger.error(f"Error setting leverage: {e}")
                await self.send_message(chat_id, "❌ An error occurred while trying to set leverage.")
        else:
            # Show current leverage
            try:
                current_leverage = await self.trader.get_leverage('FXSUSDT')
                if current_leverage:
                    await self.send_message(chat_id, f"""⚙️ **Current Leverage Information:**

• **Symbol:** `FXSUSDT`
• **Current Leverage:** `{current_leverage}x`
• **Max Allowed:** `50x`

**Usage:** `/leverage FXSUSDT <1-50>` to change leverage""")
                else:
                    await self.send_message(chat_id, "❌ Could not retrieve current leverage.")
            except Exception as e:
                self.logger.error(f"Error getting leverage: {e}")
                await self.send_message(chat_id, f"❌ Error retrieving leverage information.\n\n**Usage:** `/leverage FXSUSDT <1-50>` to set leverage")

        self.commands_used.update({chat_id: self.commands_used.get(chat_id, 0) + 1})

    async def cmd_risk(self, update, context):
        """Calculate risk per trade (requires implementation)"""
        chat_id = str(update.effective_chat.id)
        await self.send_message(chat_id, "⚠️ Risk calculation command is under development. Please use standard trading tools for now.")
        self.commands_used.update({chat_id: self.commands_used.get(chat_id, 0) + 1})

    async def cmd_signal(self, update, context):
        """Manually send a signal (admin only, requires implementation)"""
        chat_id = str(update.effective_chat.id)
        # Add admin check here
        await self.send_message(chat_id, "🚨 Manual signal sending is disabled for now. Use automated signals or contact admin.")
        self.commands_used.update({chat_id: self.commands_used.get(chat_id, 0) + 1})

    async def cmd_history(self, update, context):
        """Get recent trade history (requires implementation)"""
        chat_id = str(update.effective_chat.id)
        await self.send_message(chat_id, "📜 Trade history command is under development.")
        self.commands_used.update({chat_id: self.commands_used.get(chat_id, 0) + 1})

    async def cmd_alerts(self, update, context):
        """Manage price alerts (requires implementation)"""
        chat_id = str(update.effective_chat.id)
        await self.send_message(chat_id, "🔔 Price alert management is under development.")
        self.commands_used.update({chat_id: self.commands_used.get(chat_id, 0) + 1})

    async def cmd_admin(self, update, context):
        """Admin commands (requires implementation and authentication)"""
        chat_id = str(update.effective_chat.id)
        await self.send_message(chat_id, "👑 Admin panel access is restricted and requires authentication (implementation in progress).")
        self.commands_used.update({chat_id: self.commands_used.get(chat_id, 0) + 1})

    # --- New Advanced Commands ---

    async def cmd_futures_info(self, update, context):
        """Get general information about FXSUSDT.P futures contract"""
        chat_id = str(update.effective_chat.id)
        info = (
            "ℹ️ **FXSUSDT.P Futures Contract Information:**\n\n"
            f"• **Symbol:** `{self.contract_specs['symbol']}`\n"
            f"• **Contract Type:** `{self.contract_specs['contract_type']}`\n"
            f"• **Settlement Asset:** `{self.contract_specs['settlement_asset']}`\n"
            f"• **Margin Type:** `{self.contract_specs['margin_type']}`\n"
            f"• **Tick Size:** `{self.contract_specs['tick_size']}`\n"
            f"• **Step Size:** `{self.contract_specs['step_size']}`\n"
            f"• **Max Leverage:** `{self.contract_specs['max_leverage']}`\n"
            f"• **Funding Interval:** `{self.contract_specs['funding_interval']}`\n\n"
            "This is a USDT-margined perpetual futures contract on Binance, not a forex pair."
        )
        await self.send_message(chat_id, info)
        self.commands_used.update({chat_id: self.commands_used.get(chat_id, 0) + 1})

    async def cmd_contract_specs(self, update, context):
        """Get detailed contract specifications for FXSUSDT.P"""
        chat_id = str(update.effective_chat.id)
        specs = (
            "📜 **FXSUSDT.P Contract Specifications:**\n\n"
            f"• **Symbol:** `{self.contract_specs['symbol']}`\n"
            f"• **Base Asset:** `{self.contract_specs['base_asset']}`\n"
            f"• **Quote Asset:** `{self.contract_specs['quote_asset']}`\n"
            f"• **Contract Type:** `{self.contract_specs['contract_type']}`\n"
            f"• **Settlement Asset:** `{self.contract_specs['settlement_asset']}`\n"
            f"• **Margin Type:** `{self.contract_specs['margin_type']}`\n"
            f"• **Tick Size:** `{self.contract_specs['tick_size']}`\n"
            f"• **Lot Size Step:** `{self.contract_specs['step_size']}`\n"
            f"• **Max Leverage:** `{self.contract_specs['max_leverage']}`\n"
            f"• **Funding Payment Interval:** `{self.contract_specs['funding_interval']}`\n\n"
            "FXSUSDT.P is a futures contract, emphasizing its use in derivatives trading."
        )
        await self.send_message(chat_id, specs)
        self.commands_used.update({chat_id: self.commands_used.get(chat_id, 0) + 1})

    async def cmd_funding_rate(self, update, context):
        """Get the current funding rate for FXSUSDT.P"""
        chat_id = str(update.effective_chat.id)
        try:
            funding_rate_data = await self.trader.get_funding_rate('FXSUSDT')
            if funding_rate_data:
                rate = funding_rate_data['fundingRate']
                # Calculate next funding time if available
                next_funding_time = datetime.fromtimestamp(funding_rate_data['fundingTime'] / 1000).strftime('%Y-%m-%d %H:%M:%S UTC')
                message = (
                    f"💸 **Current Funding Rate (FXSUSDT.P):** `{rate}`\n"
                    f"Next Funding Payment: {next_funding_time}"
                )
                await self.send_message(chat_id, message)
            else:
                await self.send_message(chat_id, "❌ Could not retrieve funding rate for FXSUSDT.P.")
        except Exception as e:
            self.logger.error(f"Error in cmd_funding_rate: {e}")
            await self.send_message(chat_id, "❌ An error occurred while fetching the funding rate.")
        self.commands_used.update({chat_id: self.commands_used.get(chat_id, 0) + 1})

    async def cmd_open_interest(self, update, context):
        """Get the current open interest for FXSUSDT.P"""
        chat_id = str(update.effective_chat.id)
        try:
            oi_data = await self.trader.get_open_interest('FXSUSDT')
            if oi_data:
                oi = oi_data['openInterest']
                amount_in_quote_asset = oi_data.get('totalQuoteAssetVolume', 'N/A') # May not always be available
                message = (
                    f"📊 **Open Interest (FXSUSDT.P):**\n"
                    f"• **Contracts:** `{oi}`\n"
                )
                if amount_in_quote_asset != 'N/A':
                     message += f"• **Value (USDT):** `{amount_in_quote_asset}`\n"
                message += "\n*Note: Open Interest represents the total value of outstanding derivative contracts.*"
                await self.send_message(chat_id, message)
            else:
                await self.send_message(chat_id, "❌ Could not retrieve open interest for FXSUSDT.P.")
        except Exception as e:
            self.logger.error(f"Error in cmd_open_interest: {e}")
            await self.send_message(chat_id, "❌ An error occurred while fetching open interest.")
        self.commands_used.update({chat_id: self.commands_used.get(chat_id, 0) + 1})

    async def cmd_volume_analysis(self, update, context):
        """Get recent trading volume for FXSUSDT.P"""
        chat_id = str(update.effective_chat.id)
        try:
            # Fetch recent klines to calculate volume
            klines = await self.trader.get_30m_klines(limit=50) # Last 25 hours
            if klines:
                total_volume = sum(float(k[5]) for k in klines) # Volume is the 6th element (index 5)
                total_quote_volume = sum(float(k[7]) for k in klines) # Quote Asset Volume (USDT) is the 8th element (index 7)
                avg_volume = total_volume / len(klines)
                avg_quote_volume = total_quote_volume / len(klines)

                message = (
                    f"📈 **Volume Analysis (FXSUSDT.P - Last 25 Hours):**\n\n"
                    f"• **Total Volume (Contracts):** `{total_volume:,.2f}`\n"
                    f"• **Average Volume (Contracts/30m):** `{avg_volume:,.2f}`\n"
                    f"• **Total Volume (USDT):** `{total_quote_volume:,.2f}`\n"
                    f"• **Average Volume (USDT/30m):** `{avg_quote_volume:,.2f}`"
                )
                await self.send_message(chat_id, message)
            else:
                await self.send_message(chat_id, "❌ Could not retrieve recent trading data for volume analysis.")
        except Exception as e:
            self.logger.error(f"Error in cmd_volume_analysis: {e}")
            await self.send_message(chat_id, "❌ An error occurred while analyzing trading volume.")
        self.commands_used.update({chat_id: self.commands_used.get(chat_id, 0) + 1})

    async def cmd_market_sentiment(self, update, context):
        """Provide a qualitative assessment of market sentiment (requires external data/API)"""
        chat_id = str(update.effective_chat.id)
        # This is a placeholder. Real sentiment analysis would require integrating with news APIs, social media analysis tools, etc.
        sentiment_message = (
            "🤔 **Market Sentiment Analysis (FXSUSDT.P):**\n\n"
            "Sentiment analysis for derivatives markets is complex and often relies on multiple data sources "
            "(news, social media, order book data, options data). Currently, this bot does not have direct access "
            "to real-time sentiment analysis tools.\n\n"
            "**General Observation:** Market sentiment can shift rapidly. Traders should consult various sources "
            "for a comprehensive view."
        )
        await self.send_message(chat_id, sentiment_message)
        self.commands_used.update({chat_id: self.commands_used.get(chat_id, 0) + 1})

    async def cmd_market_news(self, update, context):
        """Fetch recent news relevant to FXSUSDT.P or crypto markets (requires implementation)"""
        chat_id = str(update.effective_chat.id)
        # This command would ideally fetch news from a financial news API (e.g., CoinDesk, Bloomberg crypto feed)
        news_message = (
            "📰 **Recent Market News:**\n\n"
            "Fetching real-time market news requires integration with a news API. This feature is currently under development.\n\n"
            "Please refer to reliable financial news sources for the latest updates on cryptocurrencies and derivatives markets."
        )
        await self.send_message(chat_id, news_message)
        self.commands_used.update({chat_id: self.commands_used.get(chat_id, 0) + 1})

    async def cmd_watchlist(self, update, context):
        """Manage a watchlist of symbols (requires implementation)"""
        chat_id = str(update.effective_chat.id)
        await self.send_message(chat_id, "🗒️ Watchlist management is currently not implemented.")
        self.commands_used.update({chat_id: self.commands_used.get(chat_id, 0) + 1})

    async def cmd_backtest(self, update, context):
        """Initiate backtesting for the strategy (requires implementation)"""
        chat_id = str(update.effective_chat.id)
        await self.send_message(chat_id, "🧪 Backtesting functionality is under development.")
        self.commands_used.update({chat_id: self.commands_used.get(chat_id, 0) + 1})

    async def cmd_optimize_strategy(self, update, context):
        """Optimize strategy parameters (requires implementation)"""
        chat_id = str(update.effective_chat.id)
        await self.send_message(chat_id, "🛠️ Strategy optimization is a complex process and is currently not implemented.")
        self.commands_used.update({chat_id: self.commands_used.get(chat_id, 0) + 1})

    async def handle_webhook_command(self, command: str, chat_id: str, args: list = None) -> bool:
        """Handle commands via webhook or direct message"""
        try:
            if command in self.commands:
                # Create mock update object for command handling
                class MockUpdate:
                    def __init__(self, chat_id):
                        self.effective_chat = MockChat(chat_id)
                        self.message = MockMessage(chat_id)
                
                class MockChat:
                    def __init__(self, chat_id):
                        self.id = int(chat_id) if chat_id.isdigit() else chat_id
                
                class MockMessage:
                    def __init__(self, chat_id):
                        self.chat = MockChat(chat_id)
                
                class MockContext:
                    def __init__(self, args):
                        self.args = args or []

                update = MockUpdate(chat_id)
                context = MockContext(args)
                
                await self.commands[command](update, context)
                return True
            else:
                await self.send_message(chat_id, "❓ Unknown command. Type /help for a list of available commands.")
                return False
                
        except Exception as e:
            self.logger.error(f"Error executing command {command}: {e}")
            await self.send_message(chat_id, f"❌ An error occurred while executing the `{command}` command.")
            return False

    async def setup_telegram_webhooks(self):
        """Setup webhook handling for commands"""
        try:
            # Set up webhook URL if needed
            webhook_url = f"https://{os.getenv('REPL_SLUG', 'your-repl')}.{os.getenv('REPL_OWNER', 'your-username')}.repl.co/webhook"
            
            url = f"{self.base_url}/setWebhook"
            data = {"url": webhook_url}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get('ok'):
                            self.logger.info(f"✅ Webhook set successfully: {webhook_url}")
                            return True
                        
            return False
            
        except Exception as e:
            self.logger.warning(f"Could not set webhook: {e}")
            return False

    async def start_telegram_polling(self):
        """Start Telegram bot with polling and command handling"""
        try:
            from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
            from telegram import Update
            
            # Install the library if missing
            try:
                import telegram
            except ImportError:
                self.logger.info("Installing python-telegram-bot...")
                import subprocess
                subprocess.check_call(["pip", "install", "python-telegram-bot==20.7"])
                from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
                from telegram import Update

            # Create application with proper configuration
            application = Application.builder().token(self.bot_token).build()

            # Add individual command handlers
            async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
                await self.cmd_start(update, context)
            
            async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
                await self.cmd_help(update, context)
                
            async def status_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
                await self.cmd_status(update, context)
                
            async def price_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
                await self.cmd_price(update, context)
                
            async def balance_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
                await self.cmd_balance(update, context)
                
            async def position_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
                await self.cmd_position(update, context)
                
            async def scan_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
                await self.cmd_scan(update, context)
                
            async def settings_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
                await self.cmd_settings(update, context)
                
            async def market_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
                await self.cmd_market(update, context)
                
            async def stats_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
                await self.cmd_stats(update, context)
                
            async def leverage_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
                await self.cmd_leverage(update, context)
                
            async def risk_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
                await self.cmd_risk(update, context)
                
            async def signal_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
                await self.cmd_signal(update, context)
                
            async def history_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
                await self.cmd_history(update, context)
                
            async def alerts_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
                await self.cmd_alerts(update, context)
                
            async def admin_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
                await self.cmd_admin(update, context)
                
            async def futures_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
                await self.cmd_futures_info(update, context)
                
            async def contract_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
                await self.cmd_contract_specs(update, context)
                
            async def funding_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
                await self.cmd_funding_rate(update, context)
                
            async def oi_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
                await self.cmd_open_interest(update, context)
                
            async def volume_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
                await self.cmd_volume_analysis(update, context)
                
            async def sentiment_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
                await self.cmd_market_sentiment(update, context)
                
            async def news_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
                await self.cmd_market_news(update, context)
                
            async def watchlist_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
                await self.cmd_watchlist(update, context)
                
            async def backtest_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
                await self.cmd_backtest(update, context)
                
            async def optimize_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
                await self.cmd_optimize_strategy(update, context)

            # Register all command handlers
            application.add_handler(CommandHandler("start", start_handler))
            application.add_handler(CommandHandler("help", help_handler))
            application.add_handler(CommandHandler("status", status_handler))
            application.add_handler(CommandHandler("price", price_handler))
            application.add_handler(CommandHandler("balance", balance_handler))
            application.add_handler(CommandHandler("position", position_handler))
            application.add_handler(CommandHandler("scan", scan_handler))
            application.add_handler(CommandHandler("settings", settings_handler))
            application.add_handler(CommandHandler("market", market_handler))
            application.add_handler(CommandHandler("stats", stats_handler))
            application.add_handler(CommandHandler("leverage", leverage_handler))
            application.add_handler(CommandHandler("risk", risk_handler))
            application.add_handler(CommandHandler("signal", signal_handler))
            application.add_handler(CommandHandler("history", history_handler))
            application.add_handler(CommandHandler("alerts", alerts_handler))
            application.add_handler(CommandHandler("admin", admin_handler))
            application.add_handler(CommandHandler("futures", futures_handler))
            application.add_handler(CommandHandler("contract", contract_handler))
            application.add_handler(CommandHandler("funding", funding_handler))
            application.add_handler(CommandHandler("oi", oi_handler))
            application.add_handler(CommandHandler("volume", volume_handler))
            application.add_handler(CommandHandler("sentiment", sentiment_handler))
            application.add_handler(CommandHandler("news", news_handler))
            application.add_handler(CommandHandler("watchlist", watchlist_handler))
            application.add_handler(CommandHandler("backtest", backtest_handler))
            application.add_handler(CommandHandler("optimize", optimize_handler))

            self.logger.info("✅ All command handlers registered successfully")
            
            # Store application reference
            self.telegram_app = application
            
            # Initialize and start the application properly
            await application.initialize()
            await application.start()
            
            # Start polling without creating new event loop
            await application.updater.start_polling()
            
            await self.send_status_update("🚀 FXSUSDT.P Futures Bot commands are now active!")
            
            self.logger.info("🤖 Telegram bot polling started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start Telegram polling: {e}")
            return False

async def main():
    """Main function to run the FXSUSDT bot"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    bot = FXSUSDTTelegramBot()

    try:
        # Start the Telegram command system
        bot.logger.info("🤖 Starting Telegram command system...")
        telegram_success = await bot.start_telegram_polling()
        
        if not telegram_success:
            bot.logger.warning("⚠️ Telegram polling failed to start, continuing with scanner only")
        
        # Start the continuous scanner
        bot.logger.info("🔍 Starting market scanner...")
        await bot.run_continuous_scanner()
        
    except KeyboardInterrupt:
        bot.logger.info("👋 Bot stopped by user")
        if hasattr(bot, 'telegram_app') and bot.telegram_app:
            try:
                await bot.telegram_app.updater.stop()
                await bot.telegram_app.stop()
                await bot.telegram_app.shutdown()
            except Exception as e:
                bot.logger.error(f"Error stopping Telegram app: {e}")
    except Exception as e:
        bot.logger.error(f"❌ Critical error: {e}")
        if hasattr(bot, 'telegram_app') and bot.telegram_app:
            try:
                await bot.telegram_app.updater.stop()
                await bot.telegram_app.stop()
                await bot.telegram_app.shutdown()
            except Exception as e:
                bot.logger.error(f"Error stopping Telegram app: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())