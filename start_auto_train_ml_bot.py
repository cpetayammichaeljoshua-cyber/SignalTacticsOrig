
#!/usr/bin/env python3
"""
Auto-Training ML Bot Startup Script
Launches the most advanced self-learning trading bot with OpenAI integration
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add SignalMaestro to path
sys.path.insert(0, str(Path(__file__).parent / "SignalMaestro"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_train_ml_bot.log'),
        logging.StreamHandler()
    ]
)

async def main():
    """Main function to run auto-training ML bot"""
    logger = logging.getLogger(__name__)
    
    print("🚀 Auto-Training ML Bot Starting")
    print("🧠 Machine learning auto-training enabled")
    print("🤖 OpenAI integration active")
    print("📊 Continuous model improvement")
    print("⚡ Fastest learning system")
    
    try:
        # Import the ultimate trading bot
        from SignalMaestro.ultimate_trading_bot import UltimateTradingBot
        
        # Initialize bot
        bot = UltimateTradingBot()
        
        # Start the bot
        logger.info("🎯 Starting auto-training ML bot...")
        await bot.run()
        
    except KeyboardInterrupt:
        logger.info("🛑 Manual shutdown requested")
    except Exception as e:
        logger.error(f"❌ Bot error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
