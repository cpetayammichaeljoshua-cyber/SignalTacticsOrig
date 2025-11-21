
#!/usr/bin/env python3
"""
Quick Telegram Connection Test
Verifies bot token and chat ID are working correctly
"""

import asyncio
import os
import logging
from telegram_signal_notifier import TelegramSignalNotifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    print("\n" + "="*60)
    print("🧪 TELEGRAM CONNECTION TEST")
    print("="*60 + "\n")
    
    # Check environment variables
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
    chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
    
    print("📋 Configuration Check:")
    print(f"   Bot Token: {'✅ Set' if bot_token else '❌ Missing'}")
    if bot_token:
        print(f"              {bot_token[:10]}...{bot_token[-5:]}")
    print(f"   Chat ID:   {'✅ Set' if chat_id else '❌ Missing'}")
    if chat_id:
        print(f"              {chat_id}")
    print()
    
    if not bot_token or not chat_id:
        print("❌ Missing Telegram credentials!")
        print("\n💡 How to fix:")
        print("   1. Go to Replit Secrets (🔒 icon in left sidebar)")
        print("   2. Add TELEGRAM_BOT_TOKEN (get from @BotFather)")
        print("   3. Add TELEGRAM_CHAT_ID (your chat ID or @channelname)")
        return
    
    # Initialize notifier
    print("🔧 Initializing Telegram notifier...")
    notifier = TelegramSignalNotifier()
    print()
    
    # Test bot connection
    print("🔌 Testing bot connection...")
    connection_ok = await notifier.test_connection()
    print()
    
    if not connection_ok:
        print("❌ Bot connection failed!")
        print("\n💡 Possible issues:")
        print("   - Invalid bot token")
        print("   - Network connectivity problem")
        return
    
    # Test sending a message
    print("📤 Testing message sending...")
    test_signal = {
        'symbol': 'TEST/USDT:USDT',
        'direction': 'LONG',
        'entry_price': 1.0000,
        'stop_loss': 0.9800,
        'take_profit_1': 1.0200,
        'take_profit_2': 1.0400,
        'take_profit_3': 1.0600,
        'leverage': 10,
        'signal_strength': 85.5,
        'consensus_confidence': 75.0,
        'strategies_agree': 4,
        'total_strategies': 5,
        'risk_reward_ratio': 2.0,
        'timeframe': '5m'
    }
    
    send_ok = await notifier.send_signal(test_signal)
    print()
    
    if send_ok:
        print("✅ TEST SUCCESSFUL!")
        print("   Check your Telegram for the test signal message")
    else:
        print("❌ TEST FAILED!")
        print("\n💡 Possible issues:")
        print("   - Invalid chat ID")
        print("   - Bot not started (send /start to bot)")
        print("   - Bot blocked by user")
        print("   - Chat doesn't exist")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    asyncio.run(main())
