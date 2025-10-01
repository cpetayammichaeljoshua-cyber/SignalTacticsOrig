
#!/usr/bin/env python3
"""
Comprehensive Import Fixer
Fixes all import issues and missing modules dynamically
"""

import os
import sys
import logging
import warnings
from pathlib import Path

# Suppress warnings at startup
warnings.filterwarnings('ignore')

def fix_all_imports():
    """Fix all import issues comprehensively"""
    print("🔧 Fixing all import issues...")
    
    # Add current directory and SignalMaestro to Python path
    current_dir = Path(__file__).parent.absolute()
    signal_maestro_dir = current_dir / "SignalMaestro"
    
    paths_to_add = [
        str(current_dir),
        str(signal_maestro_dir),
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
            print(f"✅ Added {path} to Python path")
    
    # Create missing __init__.py files
    init_dirs = [
        "SignalMaestro",
        "utils", 
        "ml_models",
        "data",
        "logs",
        "bot"
    ]
    
    for dir_name in init_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            init_file = dir_path / "__init__.py"
            if not init_file.exists():
                init_file.write_text("# Auto-generated __init__.py\n")
                print(f"✅ Created {init_file}")
    
    # Fix common import issues by creating symbolic links or fixing relative imports
    common_fixes = {
        "ichimoku_sniper_strategy.py": "SignalMaestro/ichimoku_sniper_strategy.py",
        "fxsusdt_trader.py": "SignalMaestro/fxsusdt_trader.py",
        "fxsusdt_telegram_bot.py": "SignalMaestro/fxsusdt_telegram_bot.py"
    }
    
    for target, source in common_fixes.items():
        target_path = Path(target)
        source_path = Path(source)
        
        if source_path.exists() and not target_path.exists():
            try:
                # Create a copy for import compatibility
                import shutil
                shutil.copy2(source_path, target_path)
                print(f"✅ Created import alias: {target}")
            except Exception as e:
                print(f"⚠️ Could not create alias for {target}: {e}")
    
    print("✅ All import fixes applied")

if __name__ == "__main__":
    fix_all_imports()
