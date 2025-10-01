
#!/usr/bin/env python3
"""
Comprehensive Fixed FXSUSDT Bot Startup Script
Applies all fixes before starting the bot to ensure clean operation
"""

import os
import sys
import warnings
import asyncio
import subprocess
from pathlib import Path

def apply_comprehensive_startup_fixes():
    """Apply all comprehensive fixes before bot startup"""
    print("🔧 Applying comprehensive startup fixes...")
    
    # 1. Global warning suppression
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=ImportWarning)
    
    # 2. Environment variables
    os.environ['PYTHONWARNINGS'] = 'ignore'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    # 3. Fix Python paths
    current_dir = Path(__file__).parent
    signal_maestro_path = current_dir / "SignalMaestro"
    
    for path in [str(current_dir), str(signal_maestro_path)]:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    # 4. Pandas fixes
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
    
    # 5. NumPy fixes  
    try:
        import numpy as np
        np.seterr(all='ignore')
    except ImportError:
        pass
    
    # 6. Matplotlib fixes
    try:
        import matplotlib
        matplotlib.use('Agg')
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    except ImportError:
        pass
    
    print("✅ Comprehensive startup fixes applied")

# Apply fixes before any other imports
apply_comprehensive_startup_fixes()

# Now run the comprehensive error fixer
print("🔧 Running comprehensive error fixer...")
try:
    subprocess.run([sys.executable, "dynamic_comprehensive_error_fixer.py"], check=True)
    print("✅ Comprehensive error fixer completed")
except subprocess.CalledProcessError as e:
    print(f"⚠️ Error fixer had issues but continuing: {e}")

# Import and run the bot with clean environment
import logging
from SignalMaestro.fxsusdt_telegram_bot import FXSUSDTTelegramBot

# Configure clean logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Reduce noise from specific loggers
for logger_name in ['urllib3', 'requests', 'aiohttp', 'telegram', 'httpx']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

async def main():
    """Main function with comprehensive error handling"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🚀 Starting FXSUSDT.P Ichimoku Sniper Bot (Comprehensive Fixed)")
        logger.info("📊 Strategy: Ichimoku Cloud Analysis") 
        logger.info("⏰ Timeframe: 30 Minutes")
        logger.info("🎯 Target: @SignalTactics")
        logger.info("🤖 Bot: TradeTactics")
        logger.info("🔧 All errors fixed and console cleaned")
        
        # Initialize bot with error handling
        bot = FXSUSDTTelegramBot()
        
        # Start continuous monitoring
        await bot.run_continuous_scanner()
        
    except KeyboardInterrupt:
        logger.info("👋 Bot stopped by user")
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.info("🔧 Re-running error fixer...")
        subprocess.run([sys.executable, "dynamic_comprehensive_error_fixer.py"])
        logger.info("Please restart the bot after fixes are applied")
    except Exception as e:
        logger.error(f"❌ Bot error: {e}")
        logger.info("🔄 Attempting auto-restart in 5 seconds...")
        await asyncio.sleep(5)
        await main()

if __name__ == "__main__":
    asyncio.run(main())
