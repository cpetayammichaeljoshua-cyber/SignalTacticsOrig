"""
Telegram Bot for Trading Signals

Sends rich formatted trading signals to Telegram chat.
Supports:
- Buy/Sell signal notifications
- Market state updates
- Error notifications
- Performance reports
"""

import os
import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any
import aiohttp

logger = logging.getLogger(__name__)


class TelegramSignalBot:
    """
    Telegram Bot for sending trading signals
    
    Features:
    - Rich formatted signal messages
    - Async message sending
    - Retry mechanism for failed sends
    - Rate limiting compliance
    """
    
    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        """
        Initialize Telegram Bot
        
        Args:
            bot_token: Telegram Bot API token
            chat_id: Target chat ID for signals
        """
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID', '')
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self._session: Optional[aiohttp.ClientSession] = None
        self._message_queue: List[Dict] = []
        self._last_message_time: Optional[datetime] = None
        self._min_message_interval = 1.0
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self) -> None:
        """Close the aiohttp session and cleanup connectors"""
        if self._session:
            try:
                await self._session.close()
                await asyncio.sleep(0.3)
                logger.info("Telegram bot session closed")
            except Exception as e:
                logger.warning(f"Error closing session: {e}")
            finally:
                self._session = None
    
    def _format_price(self, price: float) -> str:
        """Format price with appropriate decimals"""
        if price >= 1000:
            return f"${price:,.2f}"
        elif price >= 1:
            return f"${price:.4f}"
        else:
            return f"${price:.6f}"
    
    def _format_fear_greed(self, value: int, classification: str = "") -> str:
        """
        Format Fear & Greed index with emoji
        
        Args:
            value: Fear & Greed index value (0-100)
            classification: Classification label (e.g., "Fear", "Greed")
            
        Returns:
            Formatted string with emoji
        """
        if not classification:
            if value <= 20:
                classification = "Extreme Fear"
            elif value <= 40:
                classification = "Fear"
            elif value <= 60:
                classification = "Neutral"
            elif value <= 80:
                classification = "Greed"
            else:
                classification = "Extreme Greed"
        
        if value <= 25:
            emoji = "😱"
        elif value <= 45:
            emoji = "😰"
        elif value <= 55:
            emoji = "😐"
        elif value <= 75:
            emoji = "😊"
        else:
            emoji = "🤑"
        
        return f"🎭 Fear & Greed: {value} ({classification}) {emoji}"
    
    def _format_news_sentiment(self, score: float) -> str:
        """
        Format news sentiment score with emoji and direction
        
        Args:
            score: Sentiment score (-1 to +1)
            
        Returns:
            Formatted string with emoji
        """
        if score > 0.3:
            direction = "Bullish"
            emoji = "📈"
        elif score > 0.1:
            direction = "Slightly Bullish"
            emoji = "↗️"
        elif score < -0.3:
            direction = "Bearish"
            emoji = "📉"
        elif score < -0.1:
            direction = "Slightly Bearish"
            emoji = "↘️"
        else:
            direction = "Neutral"
            emoji = "➡️"
        
        sign = "+" if score >= 0 else ""
        return f"📰 News Sentiment: {sign}{score:.2f} ({direction}) {emoji}"
    
    def _format_mtf_alignment(self, score: float, confirming_timeframes: List[str] = None) -> str:
        """
        Format multi-timeframe alignment with confirmation details
        
        Args:
            score: Alignment score (0-100)
            confirming_timeframes: List of confirming timeframes
            
        Returns:
            Formatted string with confirmation details
        """
        if confirming_timeframes and len(confirming_timeframes) > 0:
            tf_str = ", ".join(confirming_timeframes[:4])
            return f"🔄 MTF Alignment: {score:.0f}% ({tf_str} confirm)"
        return f"🔄 MTF Alignment: {score:.0f}%"
    
    def _format_market_intelligence(self, intelligence_data: Dict[str, Any]) -> str:
        """
        Format full market intelligence section
        
        Args:
            intelligence_data: Dictionary with market intelligence data:
                - fear_greed_value: int
                - fear_greed_classification: str
                - news_sentiment: float
                - market_breadth: float
                - mtf_alignment: float
                - confirming_timeframes: List[str]
                - ai_confidence: float
                
        Returns:
            Formatted market intelligence section
        """
        if not intelligence_data:
            return ""
        
        lines = ["━━━━━ MARKET INTELLIGENCE ━━━━━", ""]
        
        fear_value = intelligence_data.get('fear_greed_value', 0)
        fear_class = intelligence_data.get('fear_greed_classification', '')
        if fear_value > 0:
            lines.append(self._format_fear_greed(fear_value, fear_class))
        
        news_score = intelligence_data.get('news_sentiment')
        if news_score is not None:
            lines.append(self._format_news_sentiment(news_score))
        
        market_breadth = intelligence_data.get('market_breadth')
        if market_breadth is not None:
            breadth_emoji = "📊"
            direction = "Bullish" if market_breadth > 50 else "Bearish" if market_breadth < 50 else "Neutral"
            lines.append(f"{breadth_emoji} Market Breadth: {market_breadth:.0f}% {direction}")
        
        mtf_score = intelligence_data.get('mtf_alignment')
        confirming_tfs = intelligence_data.get('confirming_timeframes', [])
        if mtf_score is not None:
            lines.append(self._format_mtf_alignment(mtf_score, confirming_tfs))
        
        ai_confidence = intelligence_data.get('ai_confidence')
        if ai_confidence is not None:
            conf_value = ai_confidence * 100 if ai_confidence <= 1 else ai_confidence
            lines.append(f"🧠 AI Confidence: {conf_value:.0f}%")
        
        lines.append("")
        return "\n".join(lines)
    
    def _format_risk_analysis(self, signal: Dict, intelligence_data: Optional[Dict] = None) -> str:
        """
        Format risk analysis section
        
        Args:
            signal: Signal dictionary
            intelligence_data: Optional market intelligence data
            
        Returns:
            Formatted risk analysis section
        """
        lines = ["━━━━━━ RISK ANALYSIS ━━━━━━", ""]
        
        risk = signal.get('risk_percent', 0)
        rr = signal.get('risk_reward_ratio', 1.5)
        lines.append(f"📊 Risk: {risk:.2f}%")
        lines.append(f"🎲 Risk:Reward: 1:{rr}")
        
        if intelligence_data:
            order_flow = intelligence_data.get('order_flow_score')
            if order_flow is not None:
                direction = "Bullish" if order_flow > 0 else "Bearish" if order_flow < 0 else "Neutral"
                sign = "+" if order_flow >= 0 else ""
                lines.append(f"📈 Order Flow: {direction} ({sign}{order_flow:.2f})")
            
            manipulation = intelligence_data.get('manipulation_score')
            if manipulation is not None:
                warning = " ⚡" if manipulation > 0.5 else ""
                lines.append(f"⚠️ Manipulation Score: {manipulation:.2f}{warning}")
        
        lines.append("")
        return "\n".join(lines)
    
    def _format_confirmations(self, signal: Dict, intelligence_data: Optional[Dict] = None) -> str:
        """
        Format confirmation checklist section
        
        Args:
            signal: Signal dictionary
            intelligence_data: Optional market intelligence data
            
        Returns:
            Formatted confirmation section
        """
        direction = signal.get('type', 'LONG')
        lines = ["━━━━━━ CONFIRMATION ━━━━━━", ""]
        
        lines.append(f"✅ UT Bot {direction} Signal")
        stc_indicator = "Green ↑" if direction == "LONG" else "Red ↓"
        lines.append(f"✅ STC {stc_indicator}")
        
        if intelligence_data:
            fear_value = intelligence_data.get('fear_greed_value', 50)
            if fear_value > 0:
                if direction == "LONG" and fear_value < 50:
                    lines.append("✅ Fear supports LONG")
                elif direction == "SHORT" and fear_value > 50:
                    lines.append("✅ Greed supports SHORT")
                elif fear_value >= 40 and fear_value <= 60:
                    lines.append("✅ Neutral sentiment")
            
            news_sentiment = intelligence_data.get('news_sentiment')
            if news_sentiment is not None:
                if (direction == "LONG" and news_sentiment > 0.1) or \
                   (direction == "SHORT" and news_sentiment < -0.1):
                    lines.append("✅ News sentiment aligned")
            
            mtf_alignment = intelligence_data.get('mtf_alignment', 0)
            if mtf_alignment >= 70:
                lines.append("✅ Higher TF confirms")
            elif mtf_alignment >= 50:
                lines.append("⚠️ Partial TF alignment")
        else:
            lines.append("✅ All conditions met")
        
        lines.append("")
        return "\n".join(lines)
    
    def _format_signal_message(self, signal: Dict, market_intelligence: Optional[Dict] = None) -> str:
        """
        Format trading signal as Telegram message
        
        Args:
            signal: Signal dictionary from SignalEngine
            market_intelligence: Optional market intelligence data for enhanced formatting
            
        Returns:
            Formatted message string
        """
        direction = signal.get('type', 'LONG')
        emoji = "🟢" if direction == "LONG" else "🔴"
        direction_emoji = "📈" if direction == "LONG" else "📉"
        
        entry = self._format_price(signal.get('entry_price', 0))
        sl = self._format_price(signal.get('stop_loss', 0))
        tp = self._format_price(signal.get('take_profit', 0))
        
        timestamp = signal.get('timestamp', datetime.now())
        if hasattr(timestamp, 'strftime'):
            time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')
        else:
            time_str = str(timestamp)
        
        symbol = signal.get('symbol', 'ETH/USDT')
        symbol_tag = symbol.replace('/', '').upper()
        
        if market_intelligence:
            market_intel_section = self._format_market_intelligence(market_intelligence)
            risk_analysis_section = self._format_risk_analysis(signal, market_intelligence)
            confirmation_section = self._format_confirmations(signal, market_intelligence)
            
            message = f"""
{emoji} <b>UT BOT + STC SIGNAL</b> {emoji}

{direction_emoji} <b>Direction:</b> {direction}
💱 <b>Pair:</b> {symbol}
⏰ <b>Timeframe:</b> {signal.get('timeframe', '5m')}

{market_intel_section}
━━━━━━━ TRADE SETUP ━━━━━━━

💰 <b>Entry:</b> {entry}
🛑 <b>Stop Loss:</b> {sl}
🎯 <b>Take Profit:</b> {tp}

{risk_analysis_section}
{confirmation_section}
🕐 <i>{time_str}</i>

<b>#{symbol_tag} #{direction} #UTBot #STC</b>
"""
        else:
            risk = signal.get('risk_percent', 0)
            rr = signal.get('risk_reward_ratio', 1.5)
            stc_value = signal.get('stc_value', 0)
            atr = signal.get('atr', 0)
            leverage_section = self._format_leverage_section(signal)
            
            message = f"""
{emoji} <b>UT BOT + STC SIGNAL</b> {emoji}

{direction_emoji} <b>Direction:</b> {direction}
💱 <b>Pair:</b> {symbol}
⏰ <b>Timeframe:</b> {signal.get('timeframe', '5m')}

━━━━━━━━━━━━━━━━━━━━━

💰 <b>Entry Price:</b> {entry}
🛑 <b>Stop Loss:</b> {sl}
🎯 <b>Take Profit:</b> {tp}

━━━━━━━━━━━━━━━━━━━━━

📊 <b>Risk:</b> {risk:.2f}%
🎲 <b>Risk:Reward:</b> 1:{rr}

━━━━━━━━━━━━━━━━━━━━━

<b>INDICATOR VALUES:</b>
📉 <b>STC:</b> {stc_value:.2f}
📏 <b>ATR:</b> {atr:.4f}

━━━━━━━━━━━━━━━━━━━━━

{leverage_section}

━━━━━━━━━━━━━━━━━━━━━

<b>CONFIRMATION:</b>
✅ UT Bot {direction} Signal
✅ STC {"Green ↑" if direction == "LONG" else "Red ↓"}
✅ All conditions met

🕐 <i>{time_str}</i>

<b>#{symbol_tag} #{direction} #UTBot #STC</b>
"""
        return message.strip()
    
    def _format_market_state_message(self, state: Dict) -> str:
        """
        Format market state update message
        
        Args:
            state: Market state dictionary
            
        Returns:
            Formatted message string
        """
        price = self._format_price(state.get('price', 0))
        stc = state.get('stc_value', 0)
        stc_color = state.get('stc_color', 'neutral')
        stc_slope = state.get('stc_slope', 'neutral')
        ut_color = state.get('ut_bar_color', 'neutral')
        
        stc_emoji = "🟢" if stc_color == "green" else "🔴" if stc_color == "red" else "⚪"
        ut_emoji = "🟢" if ut_color == "green" else "🔴" if ut_color == "red" else "⚪"
        slope_emoji = "↗️" if stc_slope == "up" else "↘️" if stc_slope == "down" else "➡️"
        
        timestamp = state.get('timestamp', datetime.now())
        if hasattr(timestamp, 'strftime'):
            time_str = timestamp.strftime('%H:%M:%S UTC')
        else:
            time_str = str(timestamp)
        
        message = f"""
📊 <b>MARKET UPDATE</b>

💱 ETH/USDT | 5m
💰 Price: {price}

{ut_emoji} UT Bot: {ut_color.upper()}
{stc_emoji} STC: {stc:.1f} {slope_emoji}

🕐 {time_str}
"""
        return message.strip()
    
    def _format_error_message(self, error: str, context: str = "") -> str:
        """Format error notification message"""
        return f"""
⚠️ <b>BOT ERROR</b>

{context}
<code>{error}</code>

🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
    
    def _format_startup_message(self) -> str:
        """Format bot startup message"""
        return f"""
🚀 <b>UT BOT + STC SIGNAL BOT STARTED</b>

📊 Strategy: UT Bot Alerts + STC Indicator
💱 Pair: ETH/USDT
⏰ Timeframe: 5 minutes
🎲 Risk:Reward: 1:1.5

<b>Settings:</b>
• UT Bot Key: 2.0, ATR Period: 6
• STC Length: 80, Fast: 27, Slow: 50
• Swing Lookback: 5 bars

✅ Bot is now monitoring for signals...

🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
    
    def _format_shutdown_message(self, reason: str = "Manual shutdown") -> str:
        """Format bot shutdown message"""
        return f"""
🛑 <b>BOT STOPPED</b>

Reason: {reason}

🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
    
    async def send_message(self, text: str, parse_mode: str = "HTML", 
                          disable_notification: bool = False,
                          retry_count: int = 3) -> bool:
        """
        Send message to Telegram
        
        Args:
            text: Message text
            parse_mode: Telegram parse mode (HTML or Markdown)
            disable_notification: Send without notification sound
            retry_count: Number of retries on failure
            
        Returns:
            True if sent successfully
        """
        if not self.bot_token or not self.chat_id:
            logger.error("Telegram bot token or chat ID not configured")
            return False
        
        now = datetime.now()
        if self._last_message_time:
            elapsed = (now - self._last_message_time).total_seconds()
            if elapsed < self._min_message_interval:
                await asyncio.sleep(self._min_message_interval - elapsed)
        
        session = await self._get_session()
        url = f"{self.base_url}/sendMessage"
        
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_notification": disable_notification
        }
        
        for attempt in range(retry_count):
            try:
                timeout = aiohttp.ClientTimeout(total=10)
                async with session.post(url, json=payload, timeout=timeout) as response:
                    if response.status == 200:
                        self._last_message_time = datetime.now()
                        logger.info("Message sent successfully")
                        return True
                    else:
                        error_text = await response.text()
                        logger.warning(f"Telegram API error (attempt {attempt+1}): {error_text}")
                        
            except asyncio.TimeoutError:
                logger.warning(f"Timeout sending message (attempt {attempt+1})")
            except Exception as e:
                logger.error(f"Error sending message (attempt {attempt+1}): {e}")
            
            if attempt < retry_count - 1:
                await asyncio.sleep(2 ** attempt)
        
        logger.error("Failed to send message after all retries")
        return False
    
    def _format_leverage_section(self, signal: Dict) -> str:
        """Format leverage and margin details from signal"""
        leverage_config = signal.get('leverage_config', {})
        recommended_lev = leverage_config.get('base_leverage', 5)
        auto_lev = signal.get('recommended_leverage', recommended_lev)
        margin_type = leverage_config.get('margin_type', 'CROSS')
        auto_margin = leverage_config.get('auto_add_margin', True)
        
        return f"""
⚡ <b>LEVERAGE & MARGIN:</b>
• Recommended: {recommended_lev}x
• Auto Leverage: {auto_lev}x
• Margin Type: {margin_type}
• Cross Margin: {'✅ Enabled' if margin_type == 'CROSS' else '❌ Disabled'}
• Auto Add Margin: {'✅ Active' if auto_margin else '❌ Inactive'}"""
    
    def _format_trade_execution(self, trade_info: Dict) -> str:
        """
        Format trade execution details
        
        Args:
            trade_info: Trade execution info from orchestrator
            
        Returns:
            Formatted trade execution section
        """
        if not trade_info:
            return ""
        
        trade_result = trade_info.get('trade_result', {})
        leverage_result = trade_info.get('leverage_result')
        balance = trade_info.get('balance', 0)
        
        if not trade_result.get('success'):
            return f"""
━━━━━━━━━━━━━━━━━━━━━

⚠️ <b>AUTO-TRADE FAILED</b>
{trade_result.get('message', 'Unknown error')}
"""
        
        leverage = leverage_result.leverage if leverage_result else 0
        position_size = leverage_result.position_size if leverage_result else 0
        position_value = leverage_result.position_value if leverage_result else 0
        margin = leverage_result.margin_required if leverage_result else 0
        confidence = leverage_result.confidence if leverage_result else 0
        reason = leverage_result.reason if leverage_result else ""
        
        entry_order = trade_result.get('entry', {})
        sl_order = trade_result.get('stop_loss', {})
        tp_order = trade_result.get('take_profit', {})
        
        return f"""
━━━━━━━━━━━━━━━━━━━━━

🤖 <b>AUTO-TRADE EXECUTED</b>

⚡ <b>Leverage:</b> {leverage}x
📊 <b>Position Size:</b> {position_size:.4f} ETH
💵 <b>Position Value:</b> ${position_value:,.2f}
💰 <b>Margin Used:</b> ${margin:,.2f}
📈 <b>Confidence:</b> {confidence*100:.1f}%

<b>LEVERAGE CALCULATION:</b>
{reason}

<b>ORDERS:</b>
✅ Entry: {'Filled' if entry_order.success else 'Failed'}
🛑 Stop Loss: {'Set' if sl_order.success else 'Failed'}
🎯 Take Profit: {'Set' if tp_order.success else 'Failed'}

💳 <b>Account Balance:</b> ${balance:,.2f}
"""
    
    async def send_signal(self, signal: Dict, trade_info: Optional[Dict] = None,
                          market_intelligence: Optional[Dict] = None) -> bool:
        """
        Send trading signal to Telegram
        
        Args:
            signal: Signal dictionary from SignalEngine
            trade_info: Optional trade execution info
            market_intelligence: Optional market intelligence data for enhanced formatting
                - fear_greed_value: int (0-100)
                - fear_greed_classification: str
                - news_sentiment: float (-1 to +1)
                - market_breadth: float (0-100)
                - mtf_alignment: float (0-100)
                - confirming_timeframes: List[str]
                - ai_confidence: float (0-1)
                - order_flow_score: float (-1 to +1)
                - manipulation_score: float (0-1)
            
        Returns:
            True if sent successfully
        """
        message = self._format_signal_message(signal, market_intelligence)
        
        if trade_info:
            message += self._format_trade_execution(trade_info)
        
        return await self.send_message(message, disable_notification=False)
    
    async def send_market_update(self, state: Dict) -> bool:
        """
        Send market state update
        
        Args:
            state: Market state dictionary
            
        Returns:
            True if sent successfully
        """
        message = self._format_market_state_message(state)
        return await self.send_message(message, disable_notification=True)
    
    async def send_error(self, error: str, context: str = "") -> bool:
        """
        Send error notification
        
        Args:
            error: Error message
            context: Additional context
            
        Returns:
            True if sent successfully
        """
        message = self._format_error_message(error, context)
        return await self.send_message(message)
    
    async def send_startup_notification(self) -> bool:
        """Send bot startup notification"""
        message = self._format_startup_message()
        return await self.send_message(message)
    
    async def send_shutdown_notification(self, reason: str = "Manual shutdown") -> bool:
        """Send bot shutdown notification"""
        message = self._format_shutdown_message(reason)
        return await self.send_message(message)
    
    async def send_market_intelligence_update(self, intelligence_data: Dict[str, Any],
                                               symbol: str = "ETH/USDT") -> bool:
        """
        Send periodic market intelligence summary
        
        Args:
            intelligence_data: Market intelligence data:
                - fear_greed_value: int (0-100)
                - fear_greed_classification: str
                - news_sentiment: float (-1 to +1)
                - market_breadth: float (0-100)
                - mtf_alignment: float (0-100)
                - confirming_timeframes: List[str]
                - ai_confidence: float (0-1)
                - order_flow_score: float (-1 to +1)
                - manipulation_score: float (0-1)
                - market_trend: str ("bullish", "bearish", "neutral")
                - volatility: float
            symbol: Trading pair symbol
            
        Returns:
            True if sent successfully
        """
        if not intelligence_data:
            return False
        
        fear_value = intelligence_data.get('fear_greed_value', 0)
        fear_class = intelligence_data.get('fear_greed_classification', '')
        news_score = intelligence_data.get('news_sentiment', 0)
        market_breadth = intelligence_data.get('market_breadth', 50)
        mtf_alignment = intelligence_data.get('mtf_alignment', 0)
        confirming_tfs = intelligence_data.get('confirming_timeframes', [])
        ai_confidence = intelligence_data.get('ai_confidence', 0)
        order_flow = intelligence_data.get('order_flow_score', 0)
        manipulation = intelligence_data.get('manipulation_score', 0)
        market_trend = intelligence_data.get('market_trend', 'neutral')
        volatility = intelligence_data.get('volatility', 0)
        
        trend_emoji = "📈" if market_trend == "bullish" else "📉" if market_trend == "bearish" else "➡️"
        trend_str = market_trend.upper()
        
        conf_value = ai_confidence * 100 if ai_confidence <= 1 else ai_confidence
        
        overall_emoji = "🟢" if market_breadth > 60 and news_score > 0.1 else \
                       "🔴" if market_breadth < 40 and news_score < -0.1 else "🟡"
        
        vol_status = "High ⚡" if volatility > 2 else "Normal 📊" if volatility > 0.5 else "Low 😴"
        
        fear_line = self._format_fear_greed(fear_value, fear_class) if fear_value > 0 else ""
        news_line = self._format_news_sentiment(news_score) if news_score != 0 else ""
        mtf_line = self._format_mtf_alignment(mtf_alignment, confirming_tfs) if mtf_alignment > 0 else ""
        
        order_flow_dir = "Bullish" if order_flow > 0 else "Bearish" if order_flow < 0 else "Neutral"
        order_flow_sign = "+" if order_flow >= 0 else ""
        
        manipulation_warning = " ⚡ CAUTION" if manipulation > 0.5 else ""
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        
        message = f"""
{overall_emoji} <b>MARKET INTELLIGENCE UPDATE</b> {overall_emoji}

💱 <b>Symbol:</b> {symbol}
{trend_emoji} <b>Market Trend:</b> {trend_str}

━━━━━ SENTIMENT ANALYSIS ━━━━━

{fear_line}
{news_line}
📊 Market Breadth: {market_breadth:.0f}% {"Bullish" if market_breadth > 50 else "Bearish" if market_breadth < 50 else "Neutral"}

━━━━━ TECHNICAL CONTEXT ━━━━━

{mtf_line}
🧠 AI Confidence: {conf_value:.0f}%
📉 Volatility: {vol_status}

━━━━━━ FLOW ANALYSIS ━━━━━━

📈 Order Flow: {order_flow_dir} ({order_flow_sign}{order_flow:.2f})
⚠️ Manipulation Score: {manipulation:.2f}{manipulation_warning}

━━━━━━━━━━━━━━━━━━━━━

🕐 <i>{timestamp}</i>

<b>#MarketIntelligence #CryptoAnalysis</b>
"""
        message = "\n".join([line for line in message.split("\n") if line.strip()])
        
        return await self.send_message(message.strip(), disable_notification=True)
    
    def send_signal_sync(self, signal: Dict) -> bool:
        """Synchronous wrapper for send_signal"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = asyncio.ensure_future(self.send_signal(signal))
                return True
            else:
                return loop.run_until_complete(self.send_signal(signal))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.send_signal(signal))
            finally:
                loop.close()
    
    def send_message_sync(self, text: str) -> bool:
        """Synchronous wrapper for send_message"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = asyncio.ensure_future(self.send_message(text))
                return True
            else:
                return loop.run_until_complete(self.send_message(text))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.send_message(text))
            finally:
                loop.close()
