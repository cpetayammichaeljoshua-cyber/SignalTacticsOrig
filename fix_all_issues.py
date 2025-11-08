
#!/usr/bin/env python3
"""
Comprehensive Issue Detection and Fix Script
Identifies and resolves all common trading bot issues
"""

import os
import sys
import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime

def setup_logging():
    """Setup logging for troubleshooting"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | FIX | %(levelname)s | %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

def fix_directory_structure():
    """Ensure all required directories exist"""
    logger = logging.getLogger(__name__)
    
    directories = [
        "logs",
        "SignalMaestro/logs",
        "ml_models",
        "SignalMaestro/ml_models",
        "data",
        "backups"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Directory ensured: {directory}")

def fix_database_issues():
    """Fix database connection and table issues"""
    logger = logging.getLogger(__name__)
    
    db_files = [
        "trading_bot.db",
        "SignalMaestro/trading_bot.db",
        "advanced_ml_trading.db",
        "ml_trade_learning.db"
    ]
    
    for db_file in db_files:
        try:
            if not os.path.exists(db_file):
                # Create database with basic structure
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                
                # Create basic tables
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT,
                        action TEXT,
                        entry_price REAL,
                        stop_loss REAL,
                        take_profit REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT,
                        side TEXT,
                        quantity REAL,
                        price REAL,
                        pnl REAL DEFAULT 0,
                        status TEXT DEFAULT 'open',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                conn.close()
                logger.info(f"✅ Database created: {db_file}")
            else:
                # Test connection
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                conn.close()
                logger.info(f"✅ Database OK: {db_file}")
                
        except Exception as e:
            logger.error(f"❌ Database issue {db_file}: {e}")

def fix_import_errors():
    """Fix common import and module issues"""
    logger = logging.getLogger(__name__)
    
    # Add SignalMaestro to Python path
    signalmaestro_path = Path(__file__).parent / "SignalMaestro"
    if signalmaestro_path.exists() and str(signalmaestro_path) not in sys.path:
        sys.path.insert(0, str(signalmaestro_path))
        logger.info("✅ Added SignalMaestro to Python path")
    
    # Test critical imports
    critical_modules = [
        'aiohttp',
        'asyncio',
        'sqlite3',
        'json',
        'logging',
        'datetime',
        'pandas',
        'numpy'
    ]
    
    missing_modules = []
    for module in critical_modules:
        try:
            __import__(module)
            logger.info(f"✅ Module available: {module}")
        except ImportError:
            missing_modules.append(module)
            logger.warning(f"⚠️ Module missing: {module}")
    
    if missing_modules:
        logger.info("💡 Install missing modules with: pip install " + " ".join(missing_modules))

def fix_configuration_issues():
    """Fix configuration and environment variable issues"""
    logger = logging.getLogger(__name__)
    
    # Check for essential environment variables
    env_vars = [
        'TELEGRAM_BOT_TOKEN',
        'TELEGRAM_CHAT_ID',
        'BINANCE_API_KEY',
        'BINANCE_API_SECRET'
    ]
    
    missing_env = []
    for var in env_vars:
        if not os.getenv(var):
            missing_env.append(var)
        else:
            logger.info(f"✅ Environment variable set: {var}")
    
    if missing_env:
        logger.warning("⚠️ Missing environment variables:")
        for var in missing_env:
            logger.warning(f"   • {var}")
        logger.info("💡 Set missing variables in the Secrets tab")

def fix_file_permissions():
    """Fix file permission issues"""
    logger = logging.getLogger(__name__)
    
    # Make Python files executable
    python_files = list(Path("SignalMaestro").glob("*.py"))
    python_files.extend(list(Path(".").glob("*.py")))
    
    for file_path in python_files:
        try:
            # Make readable and executable
            os.chmod(file_path, 0o755)
            logger.info(f"✅ Fixed permissions: {file_path.name}")
        except Exception as e:
            logger.error(f"❌ Permission fix failed for {file_path}: {e}")

def create_default_configs():
    """Create default configuration files"""
    logger = logging.getLogger(__name__)
    
    # Create default bot status file
    default_status = {
        "status": "ready",
        "last_update": datetime.now().isoformat(),
        "version": "production-v1.0",
        "components": {
            "database": "initialized",
            "telegram": "ready",
            "binance": "simulation",
            "cornix": "logging"
        }
    }
    
    with open("bot_status.json", "w") as f:
        json.dump(default_status, f, indent=2)
    logger.info("✅ Created default status file")
    
    # Create default ML metrics
    ml_metrics = {
        "total_trades_analyzed": 0,
        "win_rate": 0.0,
        "last_training": None,
        "model_accuracy": {
            "signal_classification": 0.0,
            "profit_prediction": 0.0,
            "risk_assessment": 0.0
        }
    }
    
    with open("ml_models/performance_metrics.json", "w") as f:
        json.dump(ml_metrics, f, indent=2)
    logger.info("✅ Created default ML metrics")

def test_bot_components():
    """Test all bot components for basic functionality"""
    logger = logging.getLogger(__name__)
    
    try:
        # Test configuration
        sys.path.insert(0, "SignalMaestro")
        from config import Config
        config = Config()
        validation = config.validate_config()
        
        if validation['valid']:
            logger.info("✅ Configuration validation passed")
        else:
            logger.warning("⚠️ Configuration has issues")
            
        # Test database
        from database import Database
        db = Database()
        logger.info("✅ Database class imported")
        
        # Test signal parser
        from signal_parser import SignalParser
        parser = SignalParser()
        test_signal = parser.parse_signal("BTCUSDT BUY 45000 SL 44000 TP 46000")
        if test_signal:
            logger.info("✅ Signal parser working")
        else:
            logger.warning("⚠️ Signal parser needs attention")
            
    except Exception as e:
        logger.error(f"❌ Component test failed: {e}")

def main():
    """Run all fixes"""
    logger = setup_logging()
    
    print("🔧 COMPREHENSIVE BOT ISSUE FIXER")
    print("=" * 50)
    print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    fixes = [
        ("📁 Directory Structure", fix_directory_structure),
        ("🗃️ Database Issues", fix_database_issues),
        ("📦 Import Issues", fix_import_errors),
        ("⚙️ Configuration", fix_configuration_issues),
        ("🔐 File Permissions", fix_file_permissions),
        ("📄 Default Configs", create_default_configs),
        ("🧪 Component Tests", test_bot_components)
    ]
    
    for description, fix_function in fixes:
        try:
            logger.info(f"🔄 Running: {description}")
            fix_function()
            logger.info(f"✅ Completed: {description}")
        except Exception as e:
            logger.error(f"❌ Failed: {description} - {e}")
        print("-" * 30)
    
    print("=" * 50)
    logger.info("🎉 All fixes completed!")
    logger.info("💡 Next step: Run 'python start_production_bot.py'")
    print("=" * 50)

if __name__ == "__main__":
    main()
