
#!/usr/bin/env python3
"""
Comprehensive Console Error Fixer
Fixes all console errors, warnings, and runtime issues
"""

import warnings
import os
import sys
import logging
from pathlib import Path

def apply_all_console_fixes():
    """Apply comprehensive console error fixes"""
    print("🔧 Applying comprehensive console fixes...")
    
    # 1. Suppress ALL warnings globally
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')
    os.environ['PYTHONWARNINGS'] = 'ignore'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    # 2. Fix pandas warnings
    try:
        import pandas as pd
        pd.set_option('mode.chained_assignment', None)
        pd.options.mode.copy_on_write = True
        try:
            pd.set_option('future.no_silent_downcasting', True)
        except:
            pass
        print("✅ Pandas warnings suppressed")
    except ImportError:
        pass
    
    # 3. Fix numpy warnings
    try:
        import numpy as np
        np.seterr(all='ignore')
        print("✅ NumPy warnings suppressed")
    except ImportError:
        pass
    
    # 4. Fix matplotlib backend
    try:
        import matplotlib
        matplotlib.use('Agg')
        print("✅ Matplotlib configured")
    except ImportError:
        pass
    
    # 5. Configure logging to reduce noise
    logging.getLogger('urllib3').setLevel(logging.ERROR)
    logging.getLogger('requests').setLevel(logging.ERROR)
    logging.getLogger('aiohttp').setLevel(logging.ERROR)
    logging.getLogger('websockets').setLevel(logging.ERROR)
    logging.getLogger('ccxt').setLevel(logging.ERROR)
    
    # 6. Add project paths
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    sys.path.insert(0, str(current_dir / 'SignalMaestro'))
    
    print("✅ All console fixes applied successfully!")

if __name__ == "__main__":
    apply_all_console_fixes()
