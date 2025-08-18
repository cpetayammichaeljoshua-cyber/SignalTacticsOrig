
#!/usr/bin/env python3
"""
Process Manager for Perfect Scalping Bot
Provides process monitoring, health checks, and management capabilities
"""

import os
import sys
import time
import json
import signal
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

class BotProcessManager:
    """Process manager for the Perfect Scalping Bot"""
    
    def __init__(self):
        self.pid_file = Path("perfect_scalping_bot.pid")
        self.status_file = Path("bot_status.json")
        self.log_file = Path("perfect_scalping_bot.log")
        
    def is_running(self) -> bool:
        """Check if bot process is running"""
        if not self.pid_file.exists():
            return False
            
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # Check if process exists
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError, ValueError):
            # Process doesn't exist, clean up stale PID file
            if self.pid_file.exists():
                self.pid_file.unlink()
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed bot status"""
        status = {
            'running': self.is_running(),
            'pid': None,
            'uptime': None,
            'restart_count': 0,
            'last_update': None
        }
        
        # Get PID if running
        if status['running'] and self.pid_file.exists():
            try:
                with open(self.pid_file, 'r') as f:
                    status['pid'] = int(f.read().strip())
            except:
                pass
        
        # Get status from status file
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    file_status = json.load(f)
                    status.update(file_status)
                    
                    # Calculate uptime
                    if 'start_time' in file_status:
                        start_time = datetime.fromisoformat(file_status['start_time'])
                        uptime = datetime.now() - start_time
                        status['uptime_seconds'] = uptime.total_seconds()
                        status['uptime_formatted'] = f"{uptime.days}d {uptime.seconds//3600}h {(uptime.seconds%3600)//60}m"
            except:
                pass
        
        return status
    
    def start_bot(self, background: bool = True) -> bool:
        """Start the bot process"""
        if self.is_running():
            print("❌ Bot is already running")
            return False
        
        try:
            if background:
                # Start as background process
                process = subprocess.Popen([
                    sys.executable, 
                    "SignalMaestro/perfect_scalping_bot.py"
                ], 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
                )
                print(f"✅ Bot started in background with PID {process.pid}")
            else:
                # Start in foreground
                subprocess.run([sys.executable, "SignalMaestro/perfect_scalping_bot.py"])
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start bot: {e}")
            return False
    
    def stop_bot(self, force: bool = False) -> bool:
        """Stop the bot process"""
        if not self.is_running():
            print("❌ Bot is not running")
            return False
        
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            if force:
                os.kill(pid, signal.SIGKILL)
                print(f"💥 Bot forcefully stopped (PID {pid})")
            else:
                os.kill(pid, signal.SIGTERM)
                print(f"🛑 Stop signal sent to bot (PID {pid})")
                
                # Wait for graceful shutdown
                time.sleep(5)
                if self.is_running():
                    print("⚠️ Bot didn't stop gracefully, forcing...")
                    os.kill(pid, signal.SIGKILL)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to stop bot: {e}")
            return False
    
    def restart_bot(self) -> bool:
        """Restart the bot process"""
        print("🔄 Restarting bot...")
        
        if self.is_running():
            if not self.stop_bot():
                return False
            time.sleep(2)
        
        return self.start_bot()
    
    def get_logs(self, lines: int = 50) -> str:
        """Get recent log entries"""
        if not self.log_file.exists():
            return "No log file found"
        
        try:
            with open(self.log_file, 'r') as f:
                all_lines = f.readlines()
                recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                return ''.join(recent_lines)
        except Exception as e:
            return f"Error reading logs: {e}"
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        status = self.get_status()
        health = {
            'healthy': False,
            'issues': [],
            'recommendations': []
        }
        
        if not status['running']:
            health['issues'].append("Bot is not running")
            health['recommendations'].append("Start the bot using: python process_manager.py start")
        else:
            health['healthy'] = True
            
            # Check uptime
            if status.get('uptime_seconds', 0) < 300:  # Less than 5 minutes
                health['issues'].append("Bot was recently restarted")
            
            # Check restart count
            restart_count = status.get('restart_count', 0)
            if restart_count > 10:
                health['issues'].append(f"High restart count: {restart_count}")
                health['recommendations'].append("Check logs for recurring errors")
            
            # Check log file size
            if self.log_file.exists():
                log_size_mb = self.log_file.stat().st_size / (1024 * 1024)
                if log_size_mb > 100:  # More than 100MB
                    health['issues'].append(f"Large log file: {log_size_mb:.1f}MB")
                    health['recommendations'].append("Consider log rotation")
        
        if health['issues']:
            health['healthy'] = False
        
        return health

def main():
    """CLI interface for bot process management"""
    manager = BotProcessManager()
    
    if len(sys.argv) < 2:
        print("""
🤖 Perfect Scalping Bot Process Manager

Usage:
  python process_manager.py <command>

Commands:
  start     - Start the bot
  stop      - Stop the bot gracefully
  restart   - Restart the bot
  status    - Show bot status
  health    - Perform health check
  logs      - Show recent logs
  force-stop - Force stop the bot

Examples:
  python process_manager.py start
  python process_manager.py status
  python process_manager.py logs
        """)
        return
    
    command = sys.argv[1].lower()
    
    if command == 'start':
        manager.start_bot()
    elif command == 'stop':
        manager.stop_bot()
    elif command == 'restart':
        manager.restart_bot()
    elif command == 'force-stop':
        manager.stop_bot(force=True)
    elif command == 'status':
        status = manager.get_status()
        print("\n📊 Bot Status:")
        print(f"Running: {'✅ Yes' if status['running'] else '❌ No'}")
        if status['pid']:
            print(f"PID: {status['pid']}")
        if status.get('uptime_formatted'):
            print(f"Uptime: {status['uptime_formatted']}")
        if status.get('restart_count'):
            print(f"Restart Count: {status['restart_count']}")
        if status.get('last_update'):
            print(f"Last Update: {status['last_update']}")
    elif command == 'health':
        health = manager.health_check()
        print(f"\n🏥 Health Check: {'✅ Healthy' if health['healthy'] else '⚠️ Issues Found'}")
        if health['issues']:
            print("\n⚠️ Issues:")
            for issue in health['issues']:
                print(f"  • {issue}")
        if health['recommendations']:
            print("\n💡 Recommendations:")
            for rec in health['recommendations']:
                print(f"  • {rec}")
    elif command == 'logs':
        lines = 50
        if len(sys.argv) > 2:
            try:
                lines = int(sys.argv[2])
            except ValueError:
                pass
        print(f"\n📋 Recent Logs ({lines} lines):")
        print("=" * 50)
        print(manager.get_logs(lines))
    else:
        print(f"❌ Unknown command: {command}")

if __name__ == "__main__":
    main()
