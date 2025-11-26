
#!/usr/bin/env python3
"""
Ultimate Trading Bot Launcher - Production Deployment
Integrates all enhancements: Pure Python tape analysis, enhanced AI intelligence,
improved market microstructure, adaptive thresholds, comprehensive error handling.
"""

import os
import sys
import asyncio
import signal
import warnings
import logging
from pathlib import Path

# Suppress all warnings globally
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['PYTHONWARNINGS'] = 'ignore'

# Add SignalMaestro to path
sys.path.insert(0, str(Path(__file__).parent / "SignalMaestro"))
sys.path.insert(0, os.path.dirname(__file__))

# Import with comprehensive error handling
try:
    from SignalMaestro.fxsusdt_telegram_bot import FXSUSDTTelegramBot
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("🔧 Attempting to fix import issues...")
    from fxsusdt_telegram_bot import FXSUSDTTelegramBot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    """Main ultimate bot with all enhancements integrated"""
    logger = logging.getLogger(__name__)
    
    # Print startup banner
    logger.info("=" * 90)
    logger.info("🚀 ULTIMATE FXSUSDT TRADING BOT - PRODUCTION DEPLOYMENT")
    logger.info("=" * 90)
    logger.info("📊 Symbol: FXSUSDT (Perpetual Futures)")
    logger.info("⏱️  Primary Timeframe: 1m Scalping + 5m Confirmation")
    logger.info("🎯 Strategy: Ichimoku Sniper + AI Enhancement")
    logger.info("")
    
    logger.info("✅ ENHANCEMENTS INTEGRATED:")
    logger.info("   ✓ Pure Python Tape Analysis (No NumPy)")
    logger.info("   ✓ Adaptive AI Thresholds (72%+)")
    logger.info("   ✓ Smart Divergence Handling")
    logger.info("   ✓ Enhanced Pattern Detection")
    logger.info("   ✓ Comprehensive Error Handling")
    logger.info("   ✓ Multi-Level TP Allocation (45/35/20)")
    logger.info("   ✓ 1M Scalping Optimization")
    logger.info("   ✓ Dynamic Leverage Control (5-50x)")
    logger.info("")
    
    logger.info("📊 SL/TP CONFIGURATION (1M Optimized):")
    logger.info("   • Stop Loss: 0.45%")
    logger.info("   • Take Profit: 1.05%")
    logger.info("   • TP Allocation: 45% / 35% / 20%")
    logger.info("")
    
    logger.info("⚡ EXECUTION SPEED (4X Faster):")
    logger.info("   • Scan Interval: 20-30s")
    logger.info("   • Signal Interval: 45s minimum")
    logger.info("   • Estimated Signals/Hour: 15-25+")
    logger.info("=" * 90)
    
    # Initialize bot
    logger.info("🔧 Initializing Ultimate Trading Bot...")
    try:
        bot = FXSUSDTTelegramBot()
        logger.info("✅ Bot components initialized successfully")
    except Exception as e:
        logger.error(f"❌ Critical - Failed to initialize bot: {e}")
        raise
    
    logger.info("=" * 90)
    logger.info("✅ ALL SYSTEMS ONLINE - STARTING CONTINUOUS SCANNER")
    logger.info("=" * 90)
    
    try:
        await bot.run_continuous_scanner()
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error in scanner: {e}")
        raise


def main_launcher():
    """Main launcher with auto-restart capability"""
    restart_count = 0
    max_restarts = 100
    
    print("🚀 Ultimate Trading Bot Launcher - Production Ready")
    print("🔧 Integrated with all enhancements")
    print("🌐 Starting with auto-restart protection...\n")
    
    # Check for required environment variables
    required_vars = ['TELEGRAM_BOT_TOKEN', 'BINANCE_API_KEY', 'BINANCE_API_SECRET']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("Please set these in the Secrets tab in Replit")
        return
    
    while restart_count < max_restarts:
        try:
            print(f"\n🎯 Starting Ultimate Trading Bot (attempt #{restart_count + 1})")
            
            # Run the bot
            asyncio.run(main())
            
            restart_count += 1
            print(f"🔄 Auto-restart #{restart_count}/{max_restarts} in 15 seconds...")
            
            # Progressive restart delay
            import time
            if restart_count <= 5:
                delay = 15
            elif restart_count <= 10:
                delay = 30
            else:
                delay = 60
            
            time.sleep(delay)
            
        except KeyboardInterrupt:
            print("\n🛑 Manual shutdown requested")
            break
        except Exception as e:
            restart_count += 1
            print(f"💥 Critical error #{restart_count}: {e}")
            print(f"🔄 Restarting in 30 seconds...")
            import time
            time.sleep(30)
    
    if restart_count >= max_restarts:
        print(f"⚠️ Maximum restart limit reached ({max_restarts})")
    
    print("✅ Ultimate Trading Bot launcher shutdown complete")

if __name__ == "__main__":
    main_launcher()
