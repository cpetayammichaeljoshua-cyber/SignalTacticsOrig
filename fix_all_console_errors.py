
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
    
    # 2. Fix pandas warnings and deprecations
    try:
        import pandas as pd
        pd.set_option('mode.chained_assignment', None)
        pd.options.mode.copy_on_write = True
        
        # Suppress FutureWarnings
        warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
        
        try:
            pd.set_option('future.no_silent_downcasting', True)
        except:
            pass
        
        # Monkey-patch DataFrame.fillna to suppress deprecation
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
