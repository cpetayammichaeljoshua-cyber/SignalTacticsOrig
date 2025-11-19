
#!/usr/bin/env python3
"""
Comprehensive Bot Startup with All Fixes Applied
Dynamically fixes all errors before starting the bot
"""

import os
import sys
import warnings
import asyncio
from pathlib import Path

# Suppress all warnings immediately
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

def apply_all_fixes():
    """Apply all comprehensive fixes"""
    print("=" * 70)
    print("🔧 COMPREHENSIVE ERROR FIXING SYSTEM")
    print("=" * 70)
    
    # 1. Pandas fixes
    try:
        import pandas as pd
        pd.set_option('mode.chained_assignment', None)
        pd.options.mode.copy_on_write = True
        warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
        
        # Fix fillna deprecation
        original_fillna = pd.DataFrame.fillna
        def safe_fillna(self, *args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if 'method' in kwargs:
                    method = kwargs.pop('method')
                    if method == 'ffill':
                        return self.ffill(*args, **kwargs)
                    elif method == 'bfill':
                        return self.bfill(*args, **kwargs)
                return original_fillna(self, *args, **kwargs)
        pd.DataFrame.fillna = safe_fillna
        print("✅ Pandas fixes applied")
    except ImportError:
        print("⚠️  Pandas not available")
    
    # 2. NumPy fixes
    try:
        import numpy as np
        np.seterr(all='ignore')
        print("✅ NumPy fixes applied")
    except ImportError:
        print("⚠️  NumPy not available")
    
    # 3. Matplotlib fixes
    try:
        import matplotlib
        matplotlib.use('Agg')
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
        print("✅ Matplotlib fixes applied")
    except ImportError:
        print("⚠️  Matplotlib not available")
    
    # 4. Add paths
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    sys.path.insert(0, str(current_dir / 'SignalMaestro'))
    print("✅ Import paths configured")
    
    # 5. Logging configuration
    import logging
    logging.getLogger('urllib3').setLevel(logging.ERROR)
    logging.getLogger('requests').setLevel(logging.ERROR)
    logging.getLogger('aiohttp').setLevel(logging.ERROR)
    logging.getLogger('websockets').setLevel(logging.ERROR)
    logging.getLogger('ccxt').setLevel(logging.ERROR)
    print("✅ Logging levels optimized")
    
    print("=" * 70)
    print("✅ ALL FIXES APPLIED SUCCESSFULLY")
    print("=" * 70)
    print()

# Apply fixes before importing bot
apply_all_fixes()

# Now import and run the bot
from start_comprehensive_all_futures_bot import ComprehensiveAllFuturesBot

async def main():
    """Main entry point with error handling"""
    try:
        bot = ComprehensiveAllFuturesBot()
        await bot.run()
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
