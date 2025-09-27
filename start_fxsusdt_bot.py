
#!/usr/bin/env python3
"""
FXSUSDT.P Ichimoku Sniper Bot Runner
Main entry point for the FXSUSDT.P trading bot
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add SignalMaestro to path
sys.path.append(str(Path(__file__).parent / "SignalMaestro"))

from SignalMaestro.fxsusdt_telegram_bot import FXSUSDTTelegramBot

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('fxsusdt_bot.log'),
            logging.StreamHandler()
        ]
    )

async def main():
    """Main function"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting FXSUSDT.P Ichimoku Sniper Bot")
    logger.info("📊 Strategy: Ichimoku Cloud Analysis")
    logger.info("⏰ Timeframe: 30 Minutes")
    logger.info("🎯 Target: @SignalTactics")
    logger.info("🤖 Bot: TradeTactics")
    
    # Check required environment variables
    required_secrets = ['TELEGRAM_BOT_TOKEN', 'BINANCE_API_KEY', 'BINANCE_API_SECRET']
    missing_secrets = [secret for secret in required_secrets if not os.getenv(secret)]
    
    if missing_secrets:
        logger.error(f"❌ Missing required secrets: {', '.join(missing_secrets)}")
        logger.error("Please add these to your Replit secrets:")
        for secret in missing_secrets:
            logger.error(f"   {secret}")
        return
    
    try:
        # Create and run bot
        bot = FXSUSDTTelegramBot()
        await bot.run_continuous_scanner()
        
    except KeyboardInterrupt:
        logger.info("👋 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Critical error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
