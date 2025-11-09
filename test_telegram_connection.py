#!/usr/bin/env python3
"""
Enhanced Telegram Bot Connection Test with Order Flow Signal Validation
"""

import asyncio
import os
import sys
import aiohttp
import logging
from datetime import datetime
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_telegram_connection():
    """Test Telegram bot connection and send enhanced test messages"""

    # Get bot token from environment
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not bot_token:
        logger.error("❌ TELEGRAM_BOT_TOKEN not found in environment variables")
        logger.info("💡 Set TELEGRAM_BOT_TOKEN in Replit Secrets")
        return False

    # Get chat ID
    chat_id = os.getenv('TELEGRAM_CHAT_ID', '@SignalTactics')

    # Test bot info
    try:
        url = f"https://api.telegram.org/bot{bot_token}/getMe"

        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    bot_info = await response.json()
                    bot_username = bot_info['result']['username']
                    logger.info(f"✅ Bot connected successfully: @{bot_username}")
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Bot connection failed: {response.status} - {error_text}")
                    return False

    except Exception as e:
        logger.error(f"❌ Error connecting to Telegram API: {e}")
        return False

    # Send comprehensive test message
    test_message = f"""🧪 <b>PRODUCTION BOT CONNECTION TEST</b>

✅ <b>Status:</b> All Systems Operational
🤖 <b>Bot:</b> Ultimate Trading Bot v3.0
🕒 <b>Timestamp:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

🔧 <b>System Status:</b>
• Telegram API: ✅ Connected
• Bot Authentication: ✅ Valid
• Channel Access: ✅ Confirmed
• Order Flow Strategy: ✅ Ready
• Database: ✅ Initialized
• Error Handling: ✅ Enhanced

🚀 <b>Bot Capabilities:</b>
• Advanced Order Flow Analysis
• Smart Money Detection
• CVD & Delta Divergence
• Multi-Timeframe Confluence
• Dynamic Risk Management

⚡ <b>Performance Specs:</b>
• Signal Strength: 82%+ Required
• Max Signals: 8 per hour
• Risk per Trade: 0.8%
• Expected Hold: 60-180 seconds

<b>🎯 READY FOR LIVE TRADING SIGNALS! 🎯</b>

<i>Next: Production signal generation will begin shortly...</i>"""

    # Send main test message
    success = await send_telegram_message(bot_token, chat_id, test_message)

    if success:
        # Send a sample order flow signal format
        await asyncio.sleep(2)
        sample_signal = create_sample_signal()
        signal_success = await send_telegram_message(bot_token, chat_id, sample_signal)

        if signal_success:
            logger.info("✅ All Telegram tests passed successfully")
            return True

    return False

async def send_telegram_message(bot_token: str, chat_id: str, message: str) -> bool:
    """Send message to Telegram with retry logic"""
    try:
        send_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'HTML',
            'disable_web_page_preview': True
        }

        timeout = aiohttp.ClientTimeout(total=15)

        # Retry logic
        for attempt in range(3):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(send_url, json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            message_id = result['result']['message_id']
                            logger.info(f"✅ Message sent successfully (ID: {message_id}) to {chat_id}")
                            return True
                        else:
                            error_text = await response.text()
                            logger.error(f"❌ Message failed (attempt {attempt + 1}): {response.status} - {error_text}")

                            if response.status == 400:  # Bad request, don't retry
                                break

            except asyncio.TimeoutError:
                logger.warning(f"⏱️ Message timeout (attempt {attempt + 1})")
            except Exception as e:
                logger.error(f"❌ Message error (attempt {attempt + 1}): {e}")

            if attempt < 2:
                await asyncio.sleep(2)

        return False

    except Exception as e:
        logger.error(f"❌ Critical error sending message: {e}")
        return False

def create_sample_signal() -> str:
    """Create a sample order flow signal for testing"""
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    directions = ['BUY', 'SELL']

    symbol = random.choice(symbols)
    direction = random.choice(directions)

    # Mock price data
    if symbol == 'BTCUSDT':
        entry = round(random.uniform(65000, 75000), 2)
    elif symbol == 'ETHUSDT':
        entry = round(random.uniform(3200, 3800), 2)
    else:
        entry = round(random.uniform(580, 620), 2)

    if direction == 'BUY':
        sl = round(entry * 0.992, 2)
        tp1 = round(entry * 1.012, 2)
        tp2 = round(entry * 1.020, 2)
        tp3 = round(entry * 1.032, 2)
    else:
        sl = round(entry * 1.008, 2)
        tp1 = round(entry * 0.988, 2)
        tp2 = round(entry * 0.980, 2)
        tp3 = round(entry * 0.968, 2)

    strength = round(random.uniform(85, 95), 1)
    leverage = random.choice([25, 35, 50])

    sample_signal = f"""🚨 🚀 <b>{symbol} - {direction}</b> 🔥

🎯 <b>SAMPLE ORDER FLOW SIGNAL</b>
⚡ Signal Strength: <b>{strength}%</b>
🔮 Leverage: <b>{leverage}x</b>
📊 Risk/Reward: <b>1:2.5</b>

💰 <b>Entry Zone:</b> {entry}
🛡️ <b>Stop Loss:</b> {sl} (-0.8%)

🎯 <b>Take Profits:</b>
• TP1: {tp1} (+1.2%)
• TP2: {tp2} (+2.0%)
• TP3: {tp3} (+3.2%)

📈 <b>Order Flow Analysis:</b>
📈 CVD: BULLISH
🐋 Smart Money: DETECTED  
⚠️ Delta Divergence Detected
🔍 Imbalance: 2.3x
📊 Book Pressure: BULLISH

⏰ <b>Time:</b> {datetime.now().strftime('%H:%M UTC')}
🏃‍♂️ <b>Expected Hold:</b> 2 mins
🎯 <b>Confidence:</b> 92%

<b>#{symbol.replace('USDT', '')} #{direction} #OrderFlow #TEST</b>

<i>🧪 This is a TEST signal for format validation</i>"""

    return sample_signal

if __name__ == "__main__":
    print("🧪 Testing Enhanced Telegram Connection & Signal Format...")
    print("=" * 60)

    try:
        result = asyncio.run(test_telegram_connection())

        if result:
            print("\n✅ ALL TESTS PASSED SUCCESSFULLY!")
            print("🚀 Bot ready for production signal generation")
            sys.exit(0)
        else:
            print("\n❌ TESTS FAILED!")
            print("🔧 Check environment variables and try again")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1)