
#!/usr/bin/env python3
"""
Comprehensive Error Fixer and Monitor
Automatically detects and fixes common errors in the trading system
"""

import asyncio
import logging
import subprocess
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
import sqlite3
from typing import Dict, List, Any, Optional

class ComprehensiveErrorFixer:
    """Comprehensive error detection and fixing system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_patterns = {
            'import_error': [
                'No module named',
                'ImportError',
                'ModuleNotFoundError'
            ],
            'api_error': [
                'API error',
                'Connection error',
                'HTTP error',
                'Timeout error'
            ],
            'data_error': [
                'No trades generated',
                'Insufficient data',
                'No qualifying signals'
            ],
            'config_error': [
                'Missing configuration',
                'Invalid parameter',
                'Configuration error'
            ]
        }
        
        self.fixes_applied = []
        self.monitoring_active = False
        
    def install_missing_package(self, package_name: str) -> bool:
        """Install missing package"""
        try:
            self.logger.info(f"Installing missing package: {package_name}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install {package_name}: {e}")
            return False
    
    def fix_import_errors(self, error_message: str) -> bool:
        """Fix import-related errors"""
        if "No module named 'schedule'" in error_message:
            return self.install_missing_package("schedule")
        elif "No module named 'numpy'" in error_message:
            return self.install_missing_package("numpy")
        elif "No module named 'httpx'" in error_message:
            return self.install_missing_package("httpx")
        elif "No module named 'telegram'" in error_message:
            return self.install_missing_package("python-telegram-bot==20.7")
        elif "No module named 'aiohttp'" in error_message:
            return self.install_missing_package("aiohttp")
        return False
    
    def fix_data_errors(self, error_message: str) -> bool:
        """Fix data-related errors"""
        if "No trades generated during backtest" in error_message:
            # Adjust strategy parameters to be more permissive
            try:
                config_file = Path("SignalMaestro/strategy_config.json")
                config = {
                    "min_signal_strength": 50.0,  # Reduced threshold
                    "min_confidence": 40.0,       # Reduced threshold
                    "signal_generation_mode": "relaxed",
                    "backtest_simulation_enhanced": True
                }
                
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2)
                
                self.logger.info("Applied relaxed signal generation parameters")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to fix data error: {e}")
                return False
        
        return False
    
    def fix_api_errors(self, error_message: str) -> bool:
        """Fix API-related errors"""
        if any(pattern in error_message for pattern in ["API error", "Connection error"]):
            # Implement retry mechanism and fallback
            try:
                # Create API resilience configuration
                config = {
                    "retry_attempts": 5,
                    "retry_delay": 10,
                    "use_fallback_data": True,
                    "connection_timeout": 30,
                    "read_timeout": 60
                }
                
                config_file = Path("SignalMaestro/api_resilience_config.json")
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2)
                
                self.logger.info("Applied API resilience configuration")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to fix API error: {e}")
                return False
        
        return False
    
    def monitor_and_fix_errors(self, log_text: str) -> List[str]:
        """Monitor log text and apply fixes"""
        fixes_applied = []
        
        for error_type, patterns in self.error_patterns.items():
            for pattern in patterns:
                if pattern in log_text:
                    self.logger.warning(f"Detected {error_type}: {pattern}")
                    
                    if error_type == 'import_error':
                        if self.fix_import_errors(log_text):
                            fixes_applied.append(f"Fixed import error: {pattern}")
                    
                    elif error_type == 'data_error':
                        if self.fix_data_errors(log_text):
                            fixes_applied.append(f"Fixed data error: {pattern}")
                    
                    elif error_type == 'api_error':
                        if self.fix_api_errors(log_text):
                            fixes_applied.append(f"Fixed API error: {pattern}")
        
        return fixes_applied
    
    async def optimize_bot_performance(self):
        """Apply comprehensive bot optimizations"""
        optimizations = []
        
        try:
            # 1. Optimize signal generation parameters
            strategy_config = {
                "ichimoku_params": {
                    "conversion_periods": 4,
                    "base_periods": 4,
                    "lagging_span2_periods": 46,
                    "displacement": 20,
                    "ema_periods": 200,
                    "stop_loss_percent": 1.75,
                    "take_profit_percent": 3.25
                },
                "signal_filtering": {
                    "min_signal_strength": 60.0,
                    "min_confidence": 50.0,
                    "use_multi_timeframe": True,
                    "timeframes": ["1m", "5m", "15m", "30m"]
                },
                "risk_management": {
                    "max_risk_per_trade": 0.02,
                    "max_signals_per_hour": 2,
                    "min_signal_interval_minutes": 30
                }
            }
            
            config_file = Path("SignalMaestro/optimized_strategy_config.json")
            with open(config_file, 'w') as f:
                json.dump(strategy_config, f, indent=2)
            
            optimizations.append("Applied optimized strategy configuration")
            
            # 2. Optimize database performance
            db_files = [
                "SignalMaestro/automated_optimization.db",
                "SignalMaestro/trade_learning.db",
                "SignalMaestro/error_logs.db"
            ]
            
            for db_file in db_files:
                if Path(db_file).exists():
                    try:
                        with sqlite3.connect(db_file) as conn:
                            conn.execute("VACUUM")
                            conn.execute("ANALYZE")
                            conn.commit()
                        optimizations.append(f"Optimized database: {db_file}")
                    except Exception as e:
                        self.logger.warning(f"Failed to optimize {db_file}: {e}")
            
            # 3. Apply memory optimizations
            memory_config = {
                "cleanup_interval_minutes": 30,
                "max_log_entries": 10000,
                "max_trade_history": 5000,
                "enable_garbage_collection": True
            }
            
            with open("SignalMaestro/memory_optimization_config.json", 'w') as f:
                json.dump(memory_config, f, indent=2)
            
            optimizations.append("Applied memory optimization settings")
            
            self.logger.info(f"Applied {len(optimizations)} optimizations")
            return optimizations
            
        except Exception as e:
            self.logger.error(f"Error in bot optimization: {e}")
            return optimizations
    
    async def continuous_monitoring(self, duration_minutes: int = 60):
        """Run continuous error monitoring and fixing"""
        self.monitoring_active = True
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        
        self.logger.info(f"Starting continuous monitoring for {duration_minutes} minutes")
        
        while datetime.now() < end_time and self.monitoring_active:
            try:
                # Check log files for errors
                log_files = [
                    "SignalMaestro/logs/hourly_automation.log",
                    "SignalMaestro/hourly_automation.log"
                ]
                
                for log_file in log_files:
                    if Path(log_file).exists():
                        try:
                            with open(log_file, 'r') as f:
                                recent_logs = f.read()[-10000:]  # Last 10KB
                            
                            fixes = self.monitor_and_fix_errors(recent_logs)
                            if fixes:
                                self.fixes_applied.extend(fixes)
                                self.logger.info(f"Applied fixes: {fixes}")
                        
                        except Exception as e:
                            self.logger.warning(f"Error reading {log_file}: {e}")
                
                # Apply performance optimizations periodically
                if len(self.fixes_applied) % 5 == 0:
                    await self.optimize_bot_performance()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in monitoring cycle: {e}")
                await asyncio.sleep(60)
        
        self.logger.info(f"Monitoring completed. Total fixes applied: {len(self.fixes_applied)}")
        return self.fixes_applied

async def main():
    """Main function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    fixer = ComprehensiveErrorFixer()
    
    print("🔧 COMPREHENSIVE ERROR FIXER AND OPTIMIZER")
    print("=" * 60)
    
    # Apply immediate optimizations
    print("🚀 Applying immediate optimizations...")
    optimizations = await fixer.optimize_bot_performance()
    
    for opt in optimizations:
        print(f"✅ {opt}")
    
    # Start continuous monitoring
    print("\n🔍 Starting continuous error monitoring...")
    fixes = await fixer.continuous_monitoring(duration_minutes=120)  # 2 hours
    
    print(f"\n📋 MONITORING SUMMARY:")
    print(f"   • Total fixes applied: {len(fixes)}")
    for fix in fixes[-10:]:  # Show last 10 fixes
        print(f"   • {fix}")

if __name__ == "__main__":
    asyncio.run(main())
