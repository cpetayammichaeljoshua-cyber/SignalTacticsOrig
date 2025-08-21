#!/usr/bin/env python3
"""
Startup Script for Enhanced Perfect Scalping Bot V3
Ultimate trading system with all profitable indicators
"""

import asyncio
import logging
import os
import sys
import traceback
import warnings
from pathlib import Path

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# Add SignalMaestro to path
signal_maestro_path = str(Path(__file__).parent / "SignalMaestro")
if signal_maestro_path not in sys.path:
    sys.path.insert(0, signal_maestro_path)

# Import the bot class with error handling
try:
    from enhanced_perfect_scalping_bot_v3 import EnhancedPerfectScalpingBotV3
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔧 Trying alternative import...")
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from SignalMaestro.enhanced_perfect_scalping_bot_v3 import EnhancedPerfectScalpingBotV3
    except ImportError as e2:
        print(f"❌ Alternative import failed: {e2}")
        print("🛠️ Creating minimal bot implementation...")
        
        class EnhancedPerfectScalpingBotV3:
            def __init__(self):
                self.logger = logging.getLogger('MinimalBot')
                
            async def start(self):
                self.logger.info("🚀 Minimal bot started - please check imports")
                while True:
                    await asyncio.sleep(60)
                    
            async def stop(self):
                self.logger.info("🛑 Minimal bot stopped")
                pass

def setup_startup_logging():
    """Setup logging for startup"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('BotV3Startup')

class Config:
    def __init__(self):
        self.TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
        self.TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')


async def main():
    """Main function to start Enhanced Perfect Scalping Bot V3"""
    startup_logger = setup_startup_logging()

    startup_logger.info("🚀 Starting Enhanced Perfect Scalping Bot V3...")
    startup_logger.info("📊 Ultimate Scalping Strategy with All Profitable Indicators")
    startup_logger.info("⚡ Features: 1SL+3TP Auto-Management, 50x Cross Margin, ML Learning")

    # Validate configuration with better defaults
    config = Config()

    if not config.TELEGRAM_BOT_TOKEN:
        startup_logger.error("❌ Missing required environment variable: TELEGRAM_BOT_TOKEN")
        startup_logger.error("Please set TELEGRAM_BOT_TOKEN in the Secrets tab")
        return

    # Set default chat ID if not provided
    if not config.TELEGRAM_CHAT_ID:
        config.TELEGRAM_CHAT_ID = '@TradeTactics_bot'
        startup_logger.info(f"📱 Using default chat ID: {config.TELEGRAM_CHAT_ID}")

    startup_logger.info(f"🤖 Bot: @TradeTactics_bot")
    startup_logger.info(f"📢 Channel: @SignalTactics")
    startup_logger.info(f"💬 Admin Chat: {config.TELEGRAM_CHAT_ID}")

    # Initialize and start the bot
    bot = EnhancedPerfectScalpingBotV3()

    try:
        await bot.start()
    except KeyboardInterrupt:
        startup_logger.info("🛑 Bot stopped by user")
        await bot.stop()
    except Exception as e:
        startup_logger.error(f"❌ Fatal error: {e}")
        traceback.print_exc()
        await bot.stop()
    finally:
        startup_logger.info("✅ Ultimate Scalping Bot V3 shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())