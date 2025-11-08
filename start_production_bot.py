#!/usr/bin/env python3
"""
Production Trading Bot Startup Script
Handles initialization, validation, and robust error recovery
"""

import asyncio
import logging
import os
import sys
import signal
import time
from pathlib import Path
from datetime import datetime

# Ensure SignalMaestro is in path
sys.path.insert(0, str(Path(__file__).parent / "SignalMaestro"))

# Setup logging first
def setup_logging():
    """Setup comprehensive logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'production_bot.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Reduce noise from external libraries
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

    return logging.getLogger(__name__)

async def validate_environment():
    """Validate environment and configuration"""
    logger = logging.getLogger(__name__)

    try:
        # Check required environment variables
        required_vars = ['TELEGRAM_BOT_TOKEN']
        missing_vars = []

        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            logger.error(f"❌ Missing required environment variables: {missing_vars}")
            return False

        logger.info("✅ Environment validation passed")
        return True

    except Exception as e:
        logger.error(f"❌ Environment validation failed: {e}")
        return False

async def initialize_components():
    """Initialize all bot components"""
    logger = logging.getLogger(__name__)

    try:
        logger.info("🚀 Initializing bot components...")

        # Create required directories
        directories = ["logs", "ml_models", "data", "backups"]
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)

        logger.info("✅ All components initialized")
        return True

    except Exception as e:
        logger.error(f"❌ Component initialization failed: {e}")
        return False

async def start_trading_bot():
    """Start the main trading bot"""
    logger = logging.getLogger(__name__)

    try:
        logger.info("🤖 Starting Ultimate Trading Bot...")

        # Import the main bot
        from ultimate_trading_bot import UltimateTradingBot

        # Create and configure bot
        bot = UltimateTradingBot()

        # Run the bot
        await bot.start()

        return True

    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
        return True
    except Exception as e:
        logger.error(f"❌ Trading bot error: {e}")
        return False

async def main():
    """Main startup function with comprehensive error handling"""
    logger = setup_logging()

    print("=" * 60)
    print("🚀 PRODUCTION TRADING BOT STARTUP")
    print("=" * 60)
    print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Working Directory: {os.getcwd()}")
    print(f"🐍 Python Version: {sys.version}")
    print("=" * 60)

    restart_count = 0
    max_restarts = 5

    while restart_count < max_restarts:
        try:
            # Validate environment
            if not await validate_environment():
                logger.error("❌ Environment validation failed")
                return False

            # Initialize components
            if not await initialize_components():
                logger.error("❌ Component initialization failed")
                return False

            # Start trading bot
            logger.info("🎯 All systems ready - Starting trading bot")
            success = await start_trading_bot()

            if success:
                logger.info("✅ Bot completed successfully")
                break
            else:
                raise Exception("Bot returned error status")

        except KeyboardInterrupt:
            logger.info("🛑 Manual shutdown requested")
            break
        except Exception as e:
            restart_count += 1
            logger.error(f"❌ Bot crashed (attempt {restart_count}/{max_restarts}): {e}")

            if restart_count < max_restarts:
                wait_time = min(30 * restart_count, 300)  # Max 5 minutes
                logger.info(f"⏳ Restarting in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                logger.error("❌ Maximum restart attempts reached")
                return False

    logger.info("🏁 Production bot startup completed")
    return True

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger = logging.getLogger(__name__)
    logger.info(f"📢 Received signal {signum}, shutting down gracefully...")
    sys.exit(0)

if __name__ == "__main__":
    # Setup signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Run the main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)