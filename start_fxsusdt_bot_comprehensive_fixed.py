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
os.environ['PYTHONWARNINGS'] = 'ignore'

# Configure pandas to suppress warnings
try:
    import pandas as pd
    pd.set_option('mode.chained_assignment', None)
    pd.options.mode.copy_on_write = True
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

from SignalMaestro.fxsusdt_telegram_bot import FXSUSDTTelegramBot

async def main():
    """Main function with dynamic position management"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)

    logger.info("🚀 Starting FXSUSDT Bot with Dynamic Position Management")
    logger.info("📊 Features: Multi-timeframe ATR, Market Regime Detection, Dynamic Leverage")
    logger.info("🎯 Advanced: Trailing Stops, Adaptive SL/TP, Volatility-Based Sizing")

    bot = FXSUSDTTelegramBot()

    # Register dynamic commands
    logger.info("✅ Dynamic SL/TP and Leverage commands registered")
    logger.info("💡 Commands available:")
    logger.info("   • /leverage AUTO - Calculate optimal leverage")
    logger.info("   • /dynamic_sltp LONG/SHORT - Get dynamic SL/TP levels")
    logger.info("   • /dashboard - Market analysis dashboard")

    # Start continuous scanner
    await bot.run_continuous_scanner()

if __name__ == "__main__":
    asyncio.run(main())