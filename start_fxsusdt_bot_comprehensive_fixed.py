#!/usr/bin/env python3
"""
Start FXSUSDT Bot with Comprehensive Fixes and Dynamic Position Management
"""

import asyncio
import logging
import sys
import os
import warnings

# Suppress all warnings globally
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['PYTHONWARNINGS'] = 'ignore'

# Configure pandas to suppress warnings
try:
    import pandas as pd
    pd.set_option('mode.chained_assignment', None)
    pd.options.mode.copy_on_write = True
    try:
        pd.set_option('future.no_silent_downcasting', True)
    except:
        pass
except ImportError:
    pass

# Suppress numpy warnings
try:
    import numpy as np
    np.seterr(all='ignore')
except ImportError:
    pass

# Add SignalMaestro to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SignalMaestro'))

# Import with error handling
try:
    from SignalMaestro.fxsusdt_telegram_bot import FXSUSDTTelegramBot
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("🔧 Attempting to fix import issues...")
    sys.path.insert(0, os.path.dirname(__file__))
    from SignalMaestro.fxsusdt_telegram_bot import FXSUSDTTelegramBot

async def main():
    """Main function with dynamic position management"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("🚀 FXSUSDT BOT - COMPREHENSIVE DYNAMIC POSITION MANAGEMENT")
    logger.info("=" * 70)
    logger.info("📊 Features: Multi-timeframe ATR, Market Regime Detection, Dynamic Leverage")
    logger.info("🎯 Advanced: Trailing Stops, Adaptive SL/TP, Volatility-Based Sizing")
    logger.info("🤖 AI Enhancement: Fallback AI with 75%+ confidence threshold")
    logger.info("📡 Channel: @SignalTactics | Bot: @TradeTactics_bot")
    logger.info("=" * 70)

    bot = FXSUSDTTelegramBot()

    # Register dynamic commands
    logger.info("✅ Dynamic commands registered:")
    logger.info("   • /leverage AUTO - Calculate optimal leverage")
    logger.info("   • /dynamic_sltp LONG/SHORT - Get dynamic SL/TP levels")
    logger.info("   • /dashboard - Market analysis dashboard")
    logger.info("   • /price - Current price & 24h stats")
    logger.info("   • /balance - Account balance")
    logger.info("   • /position - Open positions")
    logger.info("=" * 70)

    # Start continuous scanner
    await bot.run_continuous_scanner()

if __name__ == "__main__":
    asyncio.run(main())