#!/usr/bin/env python3
"""
Ultimate Cryptocurrency Trading Bot
Main entry point for the automated trading system
"""

import asyncio
import signal
import sys
import os
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from bot.trading_bot import TradingBot
from web.dashboard import create_dashboard_app
from utils.logger import setup_logging
from utils.config import Config

class BotLauncher:
    def __init__(self):
        self.bot = None
        self.web_app = None
        self.shutdown_event = asyncio.Event()
        
    async def start(self):
        """Start the trading bot and web dashboard"""
        setup_logging()
        logger = logging.getLogger(__name__)
        
        try:
            logger.info("🚀 Starting Ultimate Trading Bot launcher...")
            
            # Initialize configuration
            config = Config()
            
            # Start trading bot
            self.bot = TradingBot(config)
            bot_task = asyncio.create_task(self.bot.start())
            
            # Start web dashboard
            self.web_app = create_dashboard_app(self.bot)
            web_task = asyncio.create_task(self.web_app.start())
            
            logger.info("✅ All services started successfully")
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
            # Graceful shutdown
            logger.info("🛑 Initiating graceful shutdown...")
            
            if self.bot:
                await self.bot.stop()
            
            if self.web_app:
                await self.web_app.stop()
            
            # Cancel tasks
            bot_task.cancel()
            web_task.cancel()
            
            try:
                await asyncio.gather(bot_task, web_task, return_exceptions=True)
            except Exception as e:
                logger.error(f"Error during task cleanup: {e}")
            
            logger.info("✅ Ultimate Trading Bot launcher shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Fatal error in bot launcher: {e}")
            raise
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger = logging.getLogger(__name__)
        logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()

async def main():
    """Main entry point"""
    launcher = BotLauncher()
    
    # Setup signal handlers
    signal.signal(signal.SIGTERM, launcher.signal_handler)
    signal.signal(signal.SIGINT, launcher.signal_handler)
    
    try:
        await launcher.start()
    except KeyboardInterrupt:
        print("🛑 Bot stopped manually")
    except Exception as e:
        print(f"❌ Bot crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
