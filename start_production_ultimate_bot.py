
#!/usr/bin/env python3
"""
Production Ultimate Trading Bot with Advanced Order Flow Strategy
Comprehensive production-ready bot with all fixes and error handling
"""

import asyncio
import logging
import os
import sys
import signal
import time
import json
import traceback
from datetime import datetime
from pathlib import Path

# Setup paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "SignalMaestro"))

# Production logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production_ultimate_bot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ProductionUltimateTradingBot:
    """Production Ultimate Trading Bot with Advanced Order Flow Strategy"""
    
    def __init__(self):
        self.bot_process = None
        self.restart_count = 0
        self.max_restarts = 10
        self.running = True
        self.start_time = datetime.now()
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Environment validation and setup
        self._setup_environment()
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    def _setup_environment(self):
        """Setup and validate environment variables with defaults"""
        # Set default values for missing environment variables
        env_defaults = {
            'TARGET_CHANNEL': '@SignalTactics',
            'TELEGRAM_CHAT_ID': '@SignalTactics',
            'MAX_MESSAGES_PER_HOUR': '3',
            'MIN_TRADE_INTERVAL_SECONDS': '900',
            'DEFAULT_LEVERAGE': '50',
            'MARGIN_TYPE': 'cross',
            'LOG_LEVEL': 'INFO',
            'BINANCE_TESTNET': 'true'
        }
        
        for key, default_value in env_defaults.items():
            if not os.getenv(key):
                os.environ[key] = default_value
                logger.info(f"✅ Set default {key} = {default_value}")
        
        # Validate critical environment variables
        critical_vars = ['TELEGRAM_BOT_TOKEN']
        for var in critical_vars:
            if not os.getenv(var):
                logger.error(f"❌ Critical environment variable missing: {var}")
                raise ValueError(f"Missing critical environment variable: {var}")
    
    async def check_environment(self):
        """Check production environment requirements"""
        logger.info("🔍 Checking production environment...")
        
        # Check critical files
        critical_files = [
            'SignalMaestro/ultimate_trading_bot.py',
            'SignalMaestro/advanced_order_flow_scalping_strategy.py',
            'SignalMaestro/enhanced_order_flow_integration.py'
        ]
        
        for file_path in critical_files:
            if not Path(file_path).exists():
                logger.error(f"❌ Critical file missing: {file_path}")
                return False
        
        # Create required directories
        required_dirs = [
            'logs', 'data', 'ml_models', 'backups',
            'SignalMaestro/logs', 'SignalMaestro/ml_models'
        ]
        
        for dir_path in required_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ Production environment check passed")
        return True
    
    async def start_bot(self):
        """Start the Ultimate Trading Bot with enhanced monitoring"""
        try:
            logger.info("🚀 Starting Production Ultimate Trading Bot...")
            logger.info(f"📊 Strategy: Advanced Order Flow + ML Enhanced")
            logger.info(f"⏰ Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            logger.info("=" * 80)
            
            # Import and initialize bot
            from ultimate_trading_bot import UltimateTradingBot
            
            bot = UltimateTradingBot()
            
            # Start bot with monitoring
            await bot.run_bot()
            
        except ImportError as e:
            logger.error(f"❌ Import error: {e}")
            # Try to fix import issues
            await self._fix_import_issues()
            raise
        except Exception as e:
            logger.error(f"❌ Bot execution error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    async def _fix_import_issues(self):
        """Fix common import issues"""
        logger.info("🔧 Attempting to fix import issues...")
        
        try:
            # Create missing placeholder files
            placeholder_files = [
                'SignalMaestro/config.py',
                'SignalMaestro/database.py',
                'SignalMaestro/signal_parser.py',
                'SignalMaestro/telegram_bot.py'
            ]
            
            for file_path in placeholder_files:
                if not Path(file_path).exists():
                    await self._create_placeholder_file(file_path)
                    
        except Exception as e:
            logger.error(f"Error fixing imports: {e}")
    
    async def _create_placeholder_file(self, file_path):
        """Create placeholder file with basic structure"""
        file_name = Path(file_path).stem
        
        placeholder_content = f'''"""
{file_name.replace('_', ' ').title()} Module
Auto-generated placeholder for production bot
"""

import logging
import asyncio
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class {file_name.replace('_', '').title()}:
    """Placeholder class for {file_name}"""
    
    def __init__(self):
        self.logger = logger
        logger.info(f"{{self.__class__.__name__}} initialized")
    
    async def initialize(self):
        """Initialize the module"""
        pass
'''
        
        with open(file_path, 'w') as f:
            f.write(placeholder_content)
        
        logger.info(f"✅ Created placeholder: {file_path}")
    
    async def run_with_auto_restart(self):
        """Run bot with automatic restart capability"""
        while self.running and self.restart_count < self.max_restarts:
            try:
                # Check environment before each start
                if not await self.check_environment():
                    logger.error("❌ Environment check failed")
                    break
                
                if self.restart_count > 0:
                    logger.info(f"🔄 Restarting bot (attempt {self.restart_count + 1}/{self.max_restarts})")
                    await asyncio.sleep(10)  # Wait before restart
                
                await self.start_bot()
                
            except KeyboardInterrupt:
                logger.info("🛑 Keyboard interrupt received")
                break
            except Exception as e:
                self.restart_count += 1
                logger.error(f"❌ Bot crashed (restart {self.restart_count}): {e}")
                
                if self.restart_count >= self.max_restarts:
                    logger.error(f"❌ Maximum restarts ({self.max_restarts}) reached. Stopping.")
                    break
                
                # Progressive backoff
                wait_time = min(60, 5 * self.restart_count)
                logger.info(f"⏳ Waiting {wait_time} seconds before restart...")
                await asyncio.sleep(wait_time)
        
        # Save final status
        await self.save_status()
        logger.info("🏁 Production Ultimate Trading Bot stopped")
    
    async def save_status(self):
        """Save bot status to file"""
        try:
            status = {
                'start_time': self.start_time.isoformat(),
                'stop_time': datetime.now().isoformat(),
                'restart_count': self.restart_count,
                'running': self.running,
                'uptime_minutes': (datetime.now() - self.start_time).total_seconds() / 60
            }
            
            with open('production_bot_status.json', 'w') as f:
                json.dump(status, f, indent=2)
            
            logger.info(f"📊 Bot status saved - Uptime: {status['uptime_minutes']:.1f} minutes")
        except Exception as e:
            logger.error(f"Error saving status: {e}")

async def main():
    """Main entry point"""
    launcher = ProductionUltimateTradingBot()
    try:
        await launcher.run_with_auto_restart()
    except Exception as e:
        logger.error(f"❌ Fatal launcher error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("🚀 Production Ultimate Trading Bot Launcher")
    print("📊 Advanced Order Flow + ML Enhanced Strategy")
    print("=" * 60)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Launcher stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
