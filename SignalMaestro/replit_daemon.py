#!/usr/bin/env python3
"""
Replit Daemon System for Perfect Scalping Bot
Optimized for Replit's infrastructure with auto-restart and monitoring
"""

import os
import sys
import time
import signal
import asyncio
import logging
import subprocess
import json
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import aiohttp
import psutil

class ReplitDaemon:
    """Replit-optimized daemon for indefinite bot operation"""

    def __init__(self, script_path: str = "SignalMaestro/perfect_scalping_bot.py"):
        self.script_path = script_path
        self.process = None
        self.running = False
        self.restart_count = 0
        self.max_restarts = 99999  # Virtually unlimited
        self.restart_delay = 10
        self.health_check_interval = 15  # More frequent checks
        self.max_memory_mb = 512  # Replit memory limit

        # Replit-specific settings
        self.replit_keep_alive_port = 5000 # Changed port for flask server
        self.keep_alive_server = None

        # Status tracking
        self.status_file = Path("bot_daemon_status.json")
        self.log_file = Path("daemon.log")
        self.pid_file = Path("daemon.pid")

        # Statistics
        self.stats = {
            'start_time': datetime.now().isoformat(),
            'total_restarts': 0,
            'last_restart': None,
            'uptime_total': 0,
            'health_checks': 0,
            'last_health_check': None
        }
        self.last_restart = None # Added for consistency with new monitoring

        self._setup_logging()
        self._setup_signal_handlers()
        self._write_pid_file()

    def _setup_logging(self):
        """Setup logging optimized for Replit"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - DAEMON - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, mode='a'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"🛑 Received signal {signum}, shutting down daemon...")
            self.running = False
            if self.process:
                self.stop_bot()
            if self.keep_alive_server:
                self.stop_keep_alive_server()
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    def _write_pid_file(self):
        """Write daemon PID file"""
        try:
            with open(self.pid_file, 'w') as f:
                f.write(str(os.getpid()))
            self.logger.info(f"📝 Daemon PID: {os.getpid()}")
        except Exception as e:
            self.logger.error(f"Could not write PID file: {e}")

    def _update_status(self, status: str, details: Dict[str, Any] = None):
        """Update status file for monitoring"""
        try:
            status_data = {
                'status': status,
                'timestamp': datetime.now().isoformat(),
                'daemon_pid': os.getpid(),
                'bot_pid': self.process.pid if self.process else None,
                'restart_count': self.restart_count,
                'uptime_seconds': (datetime.now() - datetime.fromisoformat(self.stats['start_time'])).total_seconds(),
                'stats': self.stats,
                'replit_optimized': True
            }
            if details:
                status_data.update(details)

            with open(self.status_file, 'w') as f:
                json.dump(status_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Could not update status: {e}")
            
    def _update_health_status(self, health_details: Dict[str, Any]):
        """Update the status file with detailed health information."""
        try:
            status_data = {
                'status': 'running' if self.is_bot_running() else 'stopped',
                'timestamp': datetime.now().isoformat(),
                'daemon_pid': os.getpid(),
                'bot_pid': self.process.pid if self.process else None,
                'restart_count': self.restart_count,
                'uptime_seconds': self.get_uptime(),
                'health': health_details,
                'stats': self.stats,
                'replit_optimized': True
            }
            with open(self.status_file, 'w') as f:
                json.dump(status_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Could not update health status: {e}")

    def start_keep_alive_server(self):
        """Start keep-alive HTTP server for Replit with comprehensive monitoring"""
        try:
            try:
                from flask import Flask, jsonify, request
            except ImportError:
                self.logger.error("Flask not available - installing...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "flask"])
                from flask import Flask, jsonify, request
            app = Flask(__name__)

            @app.route('/')
            def health_check():
                health_status = {
                    'status': 'healthy' if self.is_bot_running() else 'unhealthy',
                    'timestamp': datetime.now().isoformat(),
                    'bot_running': self.is_bot_running(),
                    'uptime': self.get_uptime(),
                    'restart_count': self.restart_count,
                    'last_restart': self.last_restart.isoformat() if self.last_restart else None,
                    'memory_usage': self._get_memory_usage(),
                    'server_type': 'replit_deployment'
                }
                return jsonify(health_status)

            @app.route('/stats')
            def stats():
                return jsonify(self.get_stats())

            @app.route('/force-restart', methods=['POST'])
            def force_restart():
                """Emergency restart endpoint"""
                try:
                    self.restart_bot()
                    return jsonify({'status': 'restart_initiated', 'timestamp': datetime.now().isoformat()})
                except Exception as e:
                    return jsonify({'status': 'restart_failed', 'error': str(e)}), 500

            @app.route('/logs')
            def get_logs():
                """Get recent log entries"""
                try:
                    with open('daemon.log', 'r') as f:
                        lines = f.readlines()[-100:]  # Last 100 lines
                    return jsonify({'logs': lines, 'count': len(lines)})
                except Exception as e:
                    return jsonify({'logs': [], 'error': str(e)})

            @app.route('/deploy-status')
            def deploy_status():
                """Deployment readiness check"""
                return jsonify({
                    'ready': True,
                    'environment': 'replit',
                    'deployment_time': datetime.now().isoformat(),
                    'bot_status': 'running' if self.is_bot_running() else 'stopped',
                    'auto_restart': True,
                    'keep_alive': True
                })

            # Configure for production deployment
            import threading
            server_thread = threading.Thread(
                target=lambda: app.run(
                    host='0.0.0.0', 
                    port=self.replit_keep_alive_port, 
                    debug=False,
                    threaded=True,
                    use_reloader=False
                ),
                daemon=True
            )
            server_thread.start()
            self.logger.info(f"🌐 Production keep-alive server started on http://0.0.0.0:{self.replit_keep_alive_port}")
            self.logger.info("🚀 Ready for Replit deployment with auto-scaling")

        except Exception as e:
            self.logger.error(f"Failed to start keep-alive server: {e}")

    def stop_keep_alive_server(self):
        """Stop keep-alive server"""
        if self.keep_alive_server:
            try:
                # Flask development server cannot be cleanly stopped this way.
                # For production, a proper WSGI server like Gunicorn is recommended.
                # This is a placeholder; in a real scenario, you'd need a way to signal the Flask app to exit.
                self.logger.warning("Stopping Flask development server is not fully supported; server may continue running.")
            except Exception as e:
                self.logger.error(f"Error stopping keep-alive server: {e}")

    def start_bot(self) -> bool:
        """Start the trading bot"""
        try:
            if self.is_bot_running():
                self.logger.warning("Bot already running")
                return True

            self.logger.info(f"🚀 Starting bot (attempt #{self.restart_count + 1})")

            # Enhanced environment for Replit
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            env['REPLIT_DAEMON'] = '1'

            # Start bot process
            self.process = subprocess.Popen([
                sys.executable, self.script_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            bufsize=1,
            universal_newlines=True
            )

            # Give the bot a moment to start
            time.sleep(5)

            if self.is_bot_running():
                self.restart_count += 1
                self.stats['total_restarts'] += 1
                self.last_restart = datetime.now() # Update last restart time
                self.stats['last_restart'] = self.last_restart.isoformat()

                self.logger.info(f"✅ Bot started successfully (PID: {self.process.pid})")
                self._update_status('running', {'bot_pid': self.process.pid})

                # Start output monitoring
                threading.Thread(target=self._monitor_bot_output, daemon=True).start()

                return True
            else:
                self.logger.error("❌ Bot failed to start")
                return False

        except Exception as e:
            self.logger.error(f"❌ Error starting bot: {e}")
            return False

    def stop_bot(self, force: bool = False) -> bool:
        """Stop the trading bot"""
        if not self.is_bot_running():
            return True

        try:
            self.logger.info(f"🛑 Stopping bot (PID: {self.process.pid})")

            if force:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL) # Use killpg for process group
                self.logger.info("💥 Bot forcefully killed")
            else:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM) # Use killpg for process group
                try:
                    self.process.wait(timeout=10)
                    self.logger.info("✅ Bot stopped gracefully")
                except subprocess.TimeoutExpired:
                    self.logger.warning("⚠️ Graceful shutdown timeout, forcing...")
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)

            self.process = None
            self._update_status('stopped')
            return True

        except Exception as e:
            self.logger.error(f"Error stopping bot: {e}")
            return False

    def is_bot_running(self) -> bool:
        """Check if bot is running"""
        if not self.process:
            return False
        try:
            # Check if the process is still running using poll()
            return self.process.poll() is None
        except Exception as e:
            self.logger.error(f"Error checking bot running status: {e}")
            return False

    def restart_bot(self) -> bool:
        """Restart the bot"""
        self.logger.info("🔄 Restarting bot...")

        if self.is_bot_running():
            self.stop_bot()
            time.sleep(self.restart_delay) # Wait before starting again

        return self.start_bot()
        
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage of the bot process in MB."""
        if not self.process or not self.is_bot_running():
            return None
        try:
            process = psutil.Process(self.process.pid)
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Resident Set Size in MB
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return None # Process might have died between checks
        except Exception as e:
            self.logger.error(f"Error getting memory usage: {e}")
            return None

    def get_uptime(self) -> float:
        """Calculate the total uptime of the bot in seconds."""
        if self.stats['start_time']:
            start = datetime.fromisoformat(self.stats['start_time'])
            return (datetime.now() - start).total_seconds()
        return 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Return collected statistics."""
        self.stats['current_uptime'] = self.get_uptime()
        return self.stats

    def _monitor_bot_output(self):
        """Monitor bot output for health"""
        try:
            last_output_time = datetime.now()
            output_timeout = 300  # 5 minutes

            while self.is_bot_running() and self.running:
                if self.process.stdout:
                    line = self.process.stdout.readline()
                    if line:
                        last_output_time = datetime.now()
                        # Log important messages
                        log_line = line.strip()
                        if any(keyword in log_line.lower() for keyword in ['error', 'exception', 'failed', 'critical']):
                            self.logger.warning(f"Bot: {log_line}")
                        elif any(keyword in log_line.lower() for keyword in ['signal', 'trade', 'profit', 'started', 'running']):
                            self.logger.info(f"Bot: {log_line}")
                        # Optionally log all lines:
                        # else:
                        #     self.logger.debug(f"Bot: {log_line}")

                # Check for output timeout (bot might be frozen)
                if (datetime.now() - last_output_time).total_seconds() > output_timeout:
                    self.logger.warning(f"⚠️ No bot output for {output_timeout} seconds, considering restart.")
                    # No automatic restart here, relies on main loop's health check.
                    
                time.sleep(1)

        except Exception as e:
            self.logger.error(f"Error monitoring bot output: {e}")

    def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        health = {
            'healthy': False,
            'checks': {},
            'issues': [],
            'replit_optimized': True
        }

        self.stats['health_checks'] += 1
        self.stats['last_health_check'] = datetime.now().isoformat()

        # Check if bot is running
        bot_running = self.is_bot_running()
        health['checks']['bot_running'] = bot_running

        if not bot_running:
            health['issues'].append("Bot process is not running")
            return health

        try:
            # Memory check (Replit-specific)
            memory_mb = self._get_memory_usage()
            if memory_mb is not None:
                health['checks']['memory_mb'] = memory_mb
                health['checks']['memory_ok'] = memory_mb < self.max_memory_mb
                if memory_mb > self.max_memory_mb:
                    health['issues'].append(f"High memory usage: {memory_mb:.1f}MB (limit {self.max_memory_mb}MB)")
            else:
                health['checks']['memory_check'] = 'unavailable'

            # Overall health
            health['healthy'] = bot_running and len(health['issues']) == 0

        except Exception as e:
            health['issues'].append(f"Health check error: {e}")

        return health

    def _emergency_recovery(self):
        """Placeholder for emergency recovery actions."""
        self.logger.critical("Entering emergency recovery mode. Consider manual intervention.")
        # In a real-world scenario, this might involve:
        # - Sending alerts to an external monitoring service.
        # - Attempting to clean up resources.
        # - Gracefully shutting down to prevent further issues.
        pass

    def monitor_bot(self):
        """Enhanced monitoring with production-grade health checks and alerting"""
        error_count = 0
        max_errors = 10  # Increased for production
        health_check_interval = 15  # More frequent checks
        consecutive_failures = 0

        self.logger.info("🔍 Starting production-grade monitoring system...")

        while self.running:
            try:
                # Comprehensive health check
                bot_running = self.is_bot_running()
                memory_usage = self._get_memory_usage()
                uptime = self.get_uptime()

                if not bot_running:
                    consecutive_failures += 1
                    self.logger.warning(f"⚠️ Bot process died (failure #{consecutive_failures}), restarting...")

                    # Implement exponential backoff for restarts
                    backoff_delay = min(60, 2 ** consecutive_failures) # Cap delay at 60 seconds
                    time.sleep(backoff_delay)

                    if self.restart_bot():
                        self.logger.info("✅ Bot restarted successfully")
                        consecutive_failures = 0
                        error_count = 0
                    else:
                        error_count += 1
                        self.logger.error(f"❌ Failed to restart bot (attempt {error_count}/{max_errors})")

                        if error_count >= max_errors:
                            self.logger.critical("💥 Maximum restart attempts reached - entering emergency mode")
                            self._emergency_recovery()
                            break
                else:
                    consecutive_failures = 0 # Reset on success

                    # Check memory usage
                    if memory_usage is not None:
                        if memory_usage > 800:  # MB threshold for warning
                            self.logger.warning(f"⚠️ High memory usage: {memory_usage:.1f}MB - considering restart")
                            if memory_usage > 1000:  # Force restart at 1GB
                                self.logger.warning("🔄 Forcing restart due to high memory usage")
                                self.restart_bot()
                        
                        # Log healthy status periodically to confirm operation
                        if int(time.time()) % 300 == 0:  # Log every 5 minutes if healthy
                            self.logger.info(f"💚 System healthy - Uptime: {uptime:.0f}s, Memory: {memory_usage:.1f}MB")

                # Update status for external monitoring (e.g., via /status endpoint)
                self._update_health_status({
                    'bot_running': bot_running,
                    'memory_usage_mb': memory_usage,
                    'uptime_seconds': uptime,
                    'consecutive_failures': consecutive_failures,
                    'daemon_error_count': error_count # Count of daemon's failed restart attempts
                })

                time.sleep(health_check_interval)

            except KeyboardInterrupt:
                self.logger.info("🛑 Monitor interrupted")
                break
            except Exception as e:
                self.logger.error(f"Monitor error: {e}")
                time.sleep(10) # Wait before retrying monitor loop

    async def daemon_loop(self):
        """Main daemon loop"""
        self.logger.info("🔍 Starting Replit daemon loop...")

        # Start keep-alive server
        self.start_keep_alive_server()

        # Start the monitoring thread
        monitor_thread = threading.Thread(target=self.monitor_bot, daemon=True)
        monitor_thread.start()

        # Initial bot start (handled by monitor_bot now)
        if not self.start_bot(): # Initial start attempt outside monitor
            self.logger.error("❌ Failed to start bot initially")
            # The monitor will attempt to restart it.

        while self.running:
            # Keep the main loop alive; monitoring is handled by a separate thread.
            # We can use sleep or other methods to keep the daemon process running.
            await asyncio.sleep(60) # Sleep for a minute, allowing monitor and server to run

    def start_daemon(self):
        """Start the daemon"""
        self.logger.info("🤖 Starting Replit Daemon for Perfect Scalping Bot")
        self.logger.info(f"📁 Bot script: {self.script_path}")
        self.logger.info(f"🆔 Daemon PID: {os.getpid()}")
        self.logger.info(f"🌐 Keep-alive port: {self.replit_keep_alive_port}")

        self.running = True

        try:
            # Run the daemon loop asynchronously
            asyncio.run(self.daemon_loop())
        except KeyboardInterrupt:
            self.logger.info("🛑 Daemon interrupted by user")
        finally:
            self._cleanup()

        return True

    def _cleanup(self):
        """Cleanup on shutdown"""
        self.logger.info("🧹 Cleaning up daemon...")

        if self.is_bot_running():
            self.stop_bot()

        if self.keep_alive_server:
            # Attempt to stop the keep-alive server (Flask dev server is tricky)
            self.stop_keep_alive_server()

        # Remove PID file
        if self.pid_file.exists():
            try:
                self.pid_file.unlink()
            except OSError as e:
                self.logger.error(f"Error removing PID file {self.pid_file}: {e}")

        # Update status to stopped
        self._update_status('stopped')
        self.logger.info("✅ Daemon cleanup complete")

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("""
🤖 Replit Daemon for Perfect Scalping Bot

Usage:
  python replit_daemon.py <command>

Commands:
  start     - Start daemon and bot
  stop      - Stop daemon and bot  
  restart   - Restart everything
  status    - Show status
  health    - Health check
        """)
        return

    command = sys.argv[1].lower()
    daemon = ReplitDaemon()

    if command == 'start':
        print("🚀 Starting Replit Daemon...")
        daemon.start_daemon()

    elif command == 'stop':
        # Send SIGTERM to the daemon process if it exists
        if daemon.pid_file.exists():
            try:
                with open(daemon.pid_file, 'r') as f:
                    pid = int(f.read().strip())
                os.kill(pid, signal.SIGTERM)
                print("🛑 Stop signal sent to daemon")
            except (FileNotFoundError, ProcessLookupError):
                print("❌ Daemon not running or PID file is stale")
            except Exception as e:
                print(f"Error sending stop signal: {e}")
        else:
            print("❌ No daemon PID file found, assuming daemon is not running.")

    elif command == 'restart':
        # Send SIGTERM to the daemon process if it exists
        if daemon.pid_file.exists():
            try:
                with open(daemon.pid_file, 'r') as f:
                    pid = int(f.read().strip())
                os.kill(pid, signal.SIGTERM)
                print("🛑 Stop signal sent to daemon for restart...")
                time.sleep(3) # Give it a moment to shut down
            except (FileNotFoundError, ProcessLookupError):
                print("❌ Daemon not running for restart, starting new instance.")
            except Exception as e:
                print(f"Error sending stop signal for restart: {e}")
        
        # Start new daemon instance
        daemon.start_daemon()

    elif command == 'status':
        if daemon.status_file.exists():
            try:
                with open(daemon.status_file, 'r') as f:
                    status = json.load(f)
                print(f"\n📊 Replit Daemon Status:")
                print(f"Status: {status.get('status', 'unknown')}")
                print(f"Daemon PID: {status.get('daemon_pid', 'N/A')}")
                print(f"Bot PID: {status.get('bot_pid', 'N/A')}")
                print(f"Restart Count: {status.get('restart_count', 0)}")
                uptime_sec = status.get('uptime_seconds')
                if uptime_sec is not None:
                    uptime = timedelta(seconds=uptime_sec)
                    print(f"Uptime: {uptime}")
                
                health_info = status.get('health')
                if health_info:
                    print(f"Bot Running: {health_info.get('bot_running', 'N/A')}")
                    print(f"Memory Usage: {health_info.get('memory_usage_mb', 'N/A')} MB")
                    print(f"Consecutive Failures: {health_info.get('consecutive_failures', 'N/A')}")
                    print(f"Daemon Errors: {health_info.get('daemon_error_count', 'N/A')}")

            except Exception as e:
                print(f"Error reading status file: {e}")
        else:
            print("No status file found. The daemon might not be running.")

    elif command == 'health':
        # Create a temporary daemon instance to run health check
        temp_daemon = ReplitDaemon()
        print("Performing health check...")
        health = temp_daemon.health_check()
        print(f"\n🏥 Health: {'✅ Healthy' if health['healthy'] else '⚠️ Issues Found'}")

        print("\n--- Checks ---")
        for check, result in health.get('checks', {}).items():
            icon = '✅' if result else '❌' if result is False else ' '
            print(f"  {icon} {check}: {result}")

        if health.get('issues'):
            print("\n--- Issues ---")
            for issue in health['issues']:
                print(f"  • {issue}")
        else:
            print("\nNo specific issues detected.")

    else:
        print(f"❌ Unknown command: {command}")

if __name__ == "__main__":
    main()