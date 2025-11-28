"""
UT Bot + STC Trading Signal Bot - Main Entry Point

This bot monitors ETH/USDT on a 5-minute timeframe and sends
trading signals to Telegram when conditions are met.

Strategy:
- Combines UT Bot Alerts and STC indicators
- LONG: UT Bot BUY + STC green + pointing up + below 75
- SHORT: UT Bot SELL + STC red + pointing down + above 25
- Stop loss at recent swing high/low
- Take profit at 1.5x risk
"""

import asyncio
import sys
import logging

from ut_bot_strategy import TradingOrchestrator, load_config
from ut_bot_strategy.orchestrator import setup_signal_handlers

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ut_bot_signals.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)


def print_banner():
    """Print startup banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║      UT BOT + STC AUTO-LEVERAGE TRADING BOT                 ║
║                                                              ║
║  Strategy: UT Bot Alerts + Schaff Trend Cycle               ║
║  Pair: ETH/USDT | Timeframe: 5 minutes                      ║
║  Risk:Reward: 1:1.5 | Auto-Trading: ENABLED                 ║
║                                                              ║
║  Indicator Settings:                                         ║
║  - UT Bot: Key=2, ATR=6, Heikin Ashi=ON                     ║
║  - STC: Length=80, Fast=27, Slow=50                         ║
║                                                              ║
║  Leverage: Dynamic 1x-20x based on volatility               ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


async def run_bot():
    """Run the trading bot"""
    print_banner()
    
    logger.info("Loading configuration...")
    config = load_config()
    
    if not config.validate():
        logger.error("Configuration validation failed!")
        logger.error("Please ensure all required environment variables are set:")
        logger.error("  - BINANCE_API_KEY")
        logger.error("  - BINANCE_API_SECRET")
        logger.error("  - TELEGRAM_BOT_TOKEN")
        logger.error("  - TELEGRAM_CHAT_ID")
        sys.exit(1)
    
    logger.info("Configuration loaded successfully")
    logger.info(f"Trading pair: {config.trading.symbol}")
    logger.info(f"Timeframe: {config.trading.timeframe}")
    logger.info(f"Check interval: {config.bot.check_interval_seconds} seconds")
    
    orchestrator = TradingOrchestrator(config)
    setup_signal_handlers(orchestrator)
    
    logger.info("Starting bot...")
    
    try:
        await orchestrator.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        if orchestrator.running:
            await orchestrator.shutdown("Application exit")


def main():
    """Main entry point"""
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
