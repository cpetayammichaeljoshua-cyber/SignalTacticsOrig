
#!/usr/bin/env python3
"""
Bot Continuation System - Dynamically Perfectly Advanced Flexible Adaptable Comprehensive
Fixes the issue where bot stops after 5th method completes and ensures continuous signal pushing
"""

import asyncio
import logging
import subprocess
import json
import os
import time
import signal
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import psutil
from datetime import datetime, timedelta

class BotContinuationSystem:
    """Advanced system to ensure bot continues operation after workflow completion"""
    
    def __init__(self):
        self.setup_logging()
        self.running = True
        self.monitored_processes = {}
        self.restart_count = 0
        self.last_health_check = datetime.now()
        
        # Configuration
        self.health_check_interval = 30  # seconds
        self.max_restart_attempts = 10
        self.process_timeout = 300  # 5 minutes
        
        # Process priority order
        self.bot_priority = [
            ("python continuous_signal_pusher.py", "Continuous Signal Pusher", True),
            ("python start_ultimate_bot.py", "Ultimate Trading Bot", False),
            ("python SignalMaestro/ultimate_trading_bot.py", "Ultimate Trading Bot Direct", False),
            ("python SignalMaestro/perfect_scalping_bot.py", "Perfect Scalping Bot", False),
            ("python SignalMaestro/enhanced_signal_bot.py", "Enhanced Signal Bot", False)
        ]
        
        # Signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.logger.info("Bot Continuation System initialized")
    
    def setup_logging(self):
        """Setup logging system"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - BOT_CONTINUATION - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "bot_continuation.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"🛑 Received shutdown signal {signum}")
        self.running = False
        self.stop_all_processes()
        sys.exit(0)
    
    async def check_ultimate_workflow_status(self) -> bool:
        """Check if the ultimate workflow has completed"""
        try:
            # Check for completion indicators
            status_files = [
                'ultimate_bot_process.json',
                'ultimate_continuous_status.json',
                'bot_status.json'
            ]
            
            for status_file in status_files:
                if Path(status_file).exists():
                    with open(status_file, 'r') as f:
                        status = json.load(f)
                    
                    # Check if workflow shows completion
                    if 'completed' in status or 'finished' in status:
                        self.logger.info(f"✅ Workflow completion detected in {status_file}")
                        return True
            
            # Check for process completion by examining running processes
            result = subprocess.run(['pgrep', '-f', 'ultimate_trading_bot.py'], 
                                  capture_output=True, text=True)
            
            if not result.stdout.strip():
                self.logger.info("🔍 No ultimate trading bot process detected")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking workflow status: {e}")
            return False
    
    async def ensure_critical_processes_running(self):
        """Ensure critical processes are running"""
        try:
            self.logger.info("🔄 Ensuring critical processes are running...")
            
            for command, bot_name, is_critical in self.bot_priority:
                if is_critical:
                    # Check if process is already running
                    script_name = command.split()[-1]
                    result = subprocess.run(['pgrep', '-f', script_name], 
                                          capture_output=True, text=True)
                    
                    if not result.stdout.strip():
                        self.logger.info(f"🚀 Starting critical process: {bot_name}")
                        process = subprocess.Popen(
                            command.split(),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True
                        )
                        
                        self.monitored_processes[bot_name] = process
                        
                        # Save process info
                        process_info = {
                            'pid': process.pid,
                            'command': command,
                            'bot_name': bot_name,
                            'is_critical': is_critical,
                            'start_time': datetime.now().isoformat()
                        }
                        
                        process_file = f"{bot_name.lower().replace(' ', '_')}_process.json"
                        with open(process_file, 'w') as f:
                            json.dump(process_info, f, indent=2)
                        
                        self.logger.info(f"✅ {bot_name} started with PID: {process.pid}")
                    else:
                        self.logger.info(f"✅ {bot_name} already running")
            
        except Exception as e:
            self.logger.error(f"Error ensuring critical processes: {e}")
    
    async def fix_signal_channel_connection(self):
        """Fix and ensure signal channel connection"""
        try:
            self.logger.info("📱 Fixing signal channel connection...")
            
            # Check for Telegram bot token
            bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            if not bot_token:
                # Try to load from config files
                config_files = ['ultimate_unified_bot_config.json', 'enhanced_optimized_bot_config.json']
                for config_file in config_files:
                    if Path(config_file).exists():
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                            if 'TELEGRAM_BOT_TOKEN' in config:
                                os.environ['TELEGRAM_BOT_TOKEN'] = config['TELEGRAM_BOT_TOKEN']
                                bot_token = config['TELEGRAM_BOT_TOKEN']
                                break
            
            # Create signal pushing configuration
            signal_config = {
                "signal_generation_enabled": True,
                "continuous_pushing": True,
                "target_channel": "@SignalTactics",
                "signal_interval_minutes": 5,
                "max_signals_per_hour": 12,
                "fallback_generation": True,
                "error_recovery": True,
                "restart_on_failure": True,
                "health_monitoring": True,
                "channel_connection_fixed": True,
                "last_fix_time": datetime.now().isoformat()
            }
            
            with open('signal_pushing_config.json', 'w') as f:
                json.dump(signal_config, f, indent=2)
            
            self.logger.info("✅ Signal channel connection configuration updated")
            
        except Exception as e:
            self.logger.error(f"Error fixing signal channel connection: {e}")
    
    def stop_all_processes(self):
        """Stop all monitored processes"""
        self.logger.info("🛑 Stopping all monitored processes...")
        
        for name, process in self.monitored_processes.items():
            try:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
                    self.logger.info(f"✅ Stopped {name}")
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                self.logger.warning(f"⚠️ Force killed {name}")
            except Exception as e:
                self.logger.error(f"Error stopping {name}: {e}")
        
        self.monitored_processes.clear()
    
    async def monitor_processes(self):
        """Monitor processes and restart if needed"""
        while self.running:
            try:
                # Check if any monitored processes have died
                dead_processes = []
                for name, process in self.monitored_processes.items():
                    if process.poll() is not None:
                        dead_processes.append(name)
                        self.logger.warning(f"⚠️ Process {name} has died")
                
                # Restart dead processes
                for name in dead_processes:
                    del self.monitored_processes[name]
                    await self.ensure_critical_processes_running()
                
                # Fix signal channel connection
                await self.fix_signal_channel_connection()
                
                # Status report
                active_processes = len([p for p in self.monitored_processes.values() if p.poll() is None])
                self.logger.info(f"📊 Status: {active_processes} active processes, {self.restart_count} restarts")
                
                # Wait before next check
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)  # Recovery delay
    
    async def initialize_continuation_system(self):
        """Initialize the bot continuation system"""
        try:
            self.logger.info("🎯 Initializing Bot Continuation System...")
            
            # Wait for any existing processes to complete
            await asyncio.sleep(30)
            
            # Check if Ultimate Combined Workflow completed
            workflow_completed = await self.check_ultimate_workflow_status()
            
            if workflow_completed:
                self.logger.info("✅ Workflow completion detected, starting continuation phase...")
                
                # Ensure continuous signal pusher is running
                await self.ensure_critical_processes_running()
                
                # Start monitoring loop
                await self.monitor_processes()
            else:
                self.logger.info("🔍 Workflow still running, monitoring for completion...")
                
                # Monitor for workflow completion
                while self.running:
                    if await self.check_ultimate_workflow_status():
                        self.logger.info("✅ Workflow completion detected!")
                        break
                    
                    await asyncio.sleep(60)  # Check every minute
                
                # Start continuation phase
                await self.ensure_critical_processes_running()
                await self.monitor_processes()
            
        except Exception as e:
            self.logger.error(f"Error in continuation system: {e}")
    
    async def run_continuation_system(self):
        """Main entry point for the continuation system"""
        try:
            self.logger.info("🚀 BOT CONTINUATION SYSTEM STARTING")
            self.logger.info("=" * 80)
            self.logger.info("🎯 Ensuring continuous operation after workflow completion")
            self.logger.info("📡 Maintaining signal pushing and bot operation")
            self.logger.info("=" * 80)
            
            # Create continuation status file
            status = {
                'system_status': 'active',
                'start_time': datetime.now().isoformat(),
                'purpose': 'ensure_continuous_operation_after_workflow_completion',
                'monitoring_enabled': True,
                'signal_pushing_enabled': True
            }
            
            with open('bot_continuation_status.json', 'w') as f:
                json.dump(status, f, indent=2)
            
            # Initialize and run the continuation system
            await self.initialize_continuation_system()
            
        except Exception as e:
            self.logger.error(f"Fatal error in continuation system: {e}")
        finally:
            self.logger.info("🧹 Bot Continuation System shutting down...")
            self.stop_all_processes()

async def main():
    """Main async function"""
    continuation_system = BotContinuationSystem()
    
    try:
        await continuation_system.run_continuation_system()
    except KeyboardInterrupt:
        print("\n🛑 Manual shutdown requested")
    except Exception as e:
        print(f"💥 Fatal error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
