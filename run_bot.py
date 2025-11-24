#!/usr/bin/env python3
"""
Simple Bot Launcher - Handles imports properly
"""
import asyncio
import sys
import os

# Add SignalMaestro to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SignalMaestro'))

from fxsusdt_telegram_bot import FXSUSDTTelegramBot
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def main():
    """Run the bot"""
    logger.info("🚀 Starting FXSUSDT Trading Bot...")
    
    bot = FXSUSDTTelegramBot()
    logger.info("✅ Bot initialized")
    
    try:
        await bot.start_telegram_polling()
        await bot.run_continuous_scanner()
    except KeyboardInterrupt:
        logger.info("👋 Bot stopped")

if __name__ == "__main__":
    asyncio.run(main())
