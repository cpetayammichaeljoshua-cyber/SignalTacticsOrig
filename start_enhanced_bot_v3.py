
#!/usr/bin/env python3
"""
Startup Script for Enhanced Perfect Scalping Bot V3
Ultimate trading system with all profitable indicators
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add SignalMaestro to path
sys.path.append(str(Path(__file__).parent / "SignalMaestro"))

from SignalMaestro.enhanced_perfect_scalping_bot_v3 import EnhancedPerfectScalpingBotV3

def setup_logging():
    """Setup logging for startup"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger('BotV3Startup')
    return logger

async def main():
    """Main startup function"""
    logger = setup_logging()
    
    logger.info("🚀 Starting Enhanced Perfect Scalping Bot V3...")
    logger.info("📊 Ultimate Scalping Strategy with All Profitable Indicators")
    logger.info("⚡ Features: 1SL+3TP Auto-Management, 50x Cross Margin, ML Learning")
    
    # Check environment variables
    required_env_vars = ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"❌ Missing environment variables: {missing_vars}")
        logger.error("Please set them in the Secrets tab")
        return
    
    # Initialize and start bot
    bot = EnhancedPerfectScalpingBotV3()
    
    try:
        logger.info("🔧 Initializing ultimate trading system...")
        await bot.start()
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Bot stopped by user")
        await bot.stop()
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        await bot.stop()
        
    finally:
        logger.info("✅ Ultimate Scalping Bot V3 shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())
