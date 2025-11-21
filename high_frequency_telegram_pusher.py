
#!/usr/bin/env python3
"""
High-Frequency Scalping Signal Pusher to Telegram
Dynamically pushes all high-frequency scalping signals to @SignalTactics
"""

import asyncio
import logging
import os
import sys
import aiohttp
from datetime import datetime
from typing import Dict, Any, Optional

# Add project paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from high_frequency_scalping_orchestrator import HighFrequencyScalpingOrchestrator, HighFrequencySignal

class HighFrequencyTelegramPusher:
    """Push high-frequency signals to Telegram with advanced formatting"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # Telegram configuration
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN not found in environment")
        
        self.channel_id = "@SignalTactics"
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        
        # Signal tracking
        self.signals_sent = 0
        self.last_signal_time = None
        
        self.logger.info("✅ High-Frequency Telegram Pusher initialized")
    
    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - HF_TELEGRAM - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('high_frequency_signals.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    async def push_signal_to_telegram(self, signal: HighFrequencySignal) -> bool:
        """Push high-frequency signal to Telegram channel"""
        try:
            # Format signal message
            message = self._format_signal_message(signal)
            
            # Send to Telegram
            success = await self._send_telegram_message(message)
            
            if success:
                self.signals_sent += 1
                self.last_signal_time = datetime.now()
                self.logger.info(f"📡 Signal #{self.signals_sent} pushed: {signal.symbol} {signal.direction}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error pushing signal to Telegram: {e}")
            return False
    
    def _format_signal_message(self, signal: HighFrequencySignal) -> str:
        """Format high-frequency signal for Telegram with Cornix compatibility"""
        
        direction_emoji = "🟢" if signal.direction == "LONG" else "🔴"
        
        # Calculate percentages
        entry = signal.entry_price
        sl = signal.stop_loss
        
        if signal.direction == "LONG":
            sl_percent = ((entry - sl) / entry) * 100
            tp1_percent = ((signal.take_profit_1 - entry) / entry) * 100
            tp2_percent = ((signal.take_profit_2 - entry) / entry) * 100
            tp3_percent = ((signal.take_profit_3 - entry) / entry) * 100
        else:
            sl_percent = ((sl - entry) / entry) * 100
            tp1_percent = ((entry - signal.take_profit_1) / entry) * 100
            tp2_percent = ((entry - signal.take_profit_2) / entry) * 100
            tp3_percent = ((entry - signal.take_profit_3) / entry) * 100
        
        # Strategy consensus breakdown
        strategy_info = ""
        for strategy_name, vote in signal.strategy_votes.items():
            emoji = "✅" if vote == signal.direction else "❌"
            score = signal.strategy_scores.get(strategy_name, 0)
            strategy_info += f"  • {emoji} {strategy_name}: {score:.1f}%\n"
        
        message = f"""
{direction_emoji} **HIGH-FREQUENCY SCALPING SIGNAL**

**📊 TRADE SETUP:**
• **Pair:** `{signal.symbol}`
• **Direction:** `{signal.direction}`
• **Entry:** `${entry:.6f}`
• **Timeframe:** `{signal.timeframe}` ⚡

**🎯 TARGETS:**
• **TP1:** `${signal.take_profit_1:.6f}` (+{tp1_percent:.2f}%)
• **TP2:** `${signal.take_profit_2:.6f}` (+{tp2_percent:.2f}%)
• **TP3:** `${signal.take_profit_3:.6f}` (+{tp3_percent:.2f}%)
• **Stop Loss:** `${sl:.6f}` (-{sl_percent:.2f}%)

**⚡ POSITION PARAMETERS:**
• **Leverage:** `{signal.leverage}x CROSS`
• **Position Size:** `${signal.position_size_usdt:.2f} USDT`
• **Risk/Reward:** `1:{signal.risk_reward_ratio:.2f}`

**🤖 MULTI-STRATEGY CONSENSUS:**
• **Signal Strength:** `{signal.signal_strength:.1f}%`
• **Consensus:** `{signal.consensus_confidence:.1f}%`
• **Strategies Agree:** `{signal.strategies_agree}/{signal.total_strategies}`

**📈 STRATEGY BREAKDOWN:**
{strategy_info}
**⚙️ MARKET CONTEXT:**
• **Volatility:** `{signal.market_volatility.upper()}`
• **Momentum:** `{signal.momentum_phase.title()}`
• **Volume:** `{signal.volume_profile.upper()}`
• **Expected Duration:** `{signal.expected_duration_seconds}s`
• **Signal Latency:** `{signal.signal_latency_ms:.1f}ms`

**🎯 CORNIX FORMAT:**
```
{signal.symbol}
{signal.direction}
Entry: {entry:.6f}
TP1: {signal.take_profit_1:.6f}
TP2: {signal.take_profit_2:.6f}
TP3: {signal.take_profit_3:.6f}
SL: {sl:.6f}
Leverage: {signal.leverage}x
```

**⏰ Signal Time:** `{signal.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}`
**🤖 Engine:** High-Frequency Multi-Strategy Orchestrator

*Ultra-fast scalping with 6+ strategy consensus*
"""
        
        return message.strip()
    
    async def _send_telegram_message(self, message: str) -> bool:
        """Send message to Telegram channel"""
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': self.channel_id,
                'text': message,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        return True
                    else:
                        error = await response.text()
                        self.logger.error(f"Telegram API error: {error}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"Error sending to Telegram: {e}")
            return False
    
    async def send_status_update(self, message: str) -> bool:
        """Send status update to channel"""
        try:
            return await self._send_telegram_message(f"🤖 **System Status**\n\n{message}")
        except Exception as e:
            self.logger.error(f"Error sending status: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pusher statistics"""
        return {
            'signals_sent': self.signals_sent,
            'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
            'channel_id': self.channel_id,
            'status': 'active'
        }


async def main():
    """Test the Telegram pusher"""
    pusher = HighFrequencyTelegramPusher()
    
    # Send startup message
    await pusher.send_status_update(
        "🚀 High-Frequency Scalping Bot ONLINE\n"
        "⚡ Multi-Strategy Engine Active\n"
        "📊 Monitoring all timeframes\n"
        "🎯 Ready for ultra-fast signals"
    )
    
    print("✅ High-Frequency Telegram Pusher ready!")
    print(f"📡 Channel: {pusher.channel_id}")
    print("⚡ Waiting for signals from orchestrator...")


if __name__ == "__main__":
    asyncio.run(main())
