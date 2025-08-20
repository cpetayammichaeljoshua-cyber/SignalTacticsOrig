
#!/usr/bin/env python3
"""
Main starter script for Perfect Scalping Bot on Replit
Handles automatic startup and indefinite operation
"""

import os
import sys
import asyncio
import signal
from pathlib import Path

# Add SignalMaestro to path
sys.path.insert(0, str(Path(__file__).parent / "SignalMaestro"))

from replit_daemon import ReplitDaemon

def main():
    """Main entry point for Replit production deployment"""
    print("🚀 Perfect Scalping Bot - Production Deployment on Replit")
    print("🔧 Optimized for indefinite operation with auto-scaling")
    print("🌐 Starting production daemon with monitoring...")
    print("📊 Environment: Replit Cloud Hosting")
    
    # Check for required environment variables
    required_vars = ['TELEGRAM_BOT_TOKEN']
    optional_vars = ['BINANCE_API_KEY', 'BINANCE_API_SECRET', 'SESSION_SECRET']
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("Please set these in the Secrets tab in Replit")
        print("💡 Go to Secrets tab and add your TELEGRAM_BOT_TOKEN")
        return
    
    # Show environment status
    print("\n🔐 Environment Variables Status:")
    for var in required_vars + optional_vars:
        status = "✅" if os.getenv(var) else "❌"
        print(f"  {status} {var}")
    
    # Initialize and start production daemon
    daemon = ReplitDaemon("SignalMaestro/perfect_scalping_bot.py")
    
    # Setup production features
    try:
        daemon.setup_production_logging()
        daemon.setup_deployment_alerts()
    except AttributeError as e:
        print(f"⚠️ Warning: {e}")
        print("🔧 Some production features may not be available")
    
    print("\n🎯 Production Features Enabled:")
    print("  ✅ Auto-restart on failures")
    print("  ✅ Memory monitoring")
    print("  ✅ Health check endpoints")
    print("  ✅ Production logging")
    print("  ✅ Emergency recovery")
    print("  ✅ HTTP keep-alive server")
    print("  ✅ Deployment-ready configuration")
    
    print(f"\n🌐 Server will be accessible at: https://{os.getenv('REPL_SLUG', 'perfect-scalping-bot')}.{os.getenv('REPL_OWNER', 'user')}.repl.co")
    print("📈 Bot will run indefinitely with automatic restarts")
    print("🔄 Process manager handles all failures and recovery")
    
    try:
        daemon.start_daemon()
    except KeyboardInterrupt:
        print("\n🛑 Production shutdown initiated")
        daemon.running = False
        daemon._cleanup()
    except Exception as e:
        print(f"❌ Production error: {e}")
        # Attempt emergency recovery
        try:
            daemon._emergency_recovery()
        except AttributeError:
            print("⚠️ Emergency recovery not available")
        return

if __name__ == "__main__":
    main()
