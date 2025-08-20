
#!/usr/bin/env python3
"""
Enhanced Perfect Scalping Bot Launcher
Starts the bot with proper error handling and recovery
"""

import asyncio
import sys
import os
import signal
import logging
from datetime import datetime

# Add SignalMaestro to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SignalMaestro'))

from SignalMaestro.enhanced_perfect_scalping_bot import EnhancedPerfectScalpingBot

class BotLauncher:
    """Enhanced bot launcher with auto-restart capabilities"""
    
    def __init__(self):
        self.bot = None
        self.running = True
        self.restart_count = 0
        self.max_restarts = 100
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - LAUNCHER - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('enhanced_bot_launcher.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"🛑 Received signal {signum}, shutting down...")
        self.running = False
        if self.bot:
            asyncio.create_task(self.bot.stop())
    
    async def run_bot(self):
        """Run bot with auto-restart capability"""
        while self.running and self.restart_count < self.max_restarts:
            try:
                self.logger.info(f"🚀 Starting Enhanced Perfect Scalping Bot (attempt #{self.restart_count + 1})")
                
                # Create new bot instance
                self.bot = EnhancedPerfectScalpingBot()
                
                # Start bot
                await self.bot.start()
                
            except KeyboardInterrupt:
                self.logger.info("🛑 Bot stopped by user")
                break
                
            except Exception as e:
                self.logger.error(f"❌ Bot crashed: {e}")
                
                if self.bot:
                    try:
                        await self.bot.stop()
                    except:
                        pass
                
                self.restart_count += 1
                
                if self.restart_count < self.max_restarts:
                    self.logger.info(f"🔄 Restarting in 5 seconds... ({self.restart_count}/{self.max_restarts})")
                    await asyncio.sleep(5)
                else:
                    self.logger.error("❌ Maximum restart limit reached")
                    break
        
        # Final cleanup
        if self.bot:
            try:
                await self.bot.stop()
            except:
                pass
        
        self.logger.info("✅ Bot launcher shutdown complete")

def main():
    """Main launcher function"""
    print("""
🤖 Enhanced Perfect Scalping Bot v2.0
=====================================
🎯 Advanced SL/TP Management
⚡ 3-Level Take Profit System
🔄 Auto SL Movement
🌐 Cornix Integration
📊 Rate-Limited Responses (3/hr)
🚀 Auto-Restart Capability

Starting bot...
    """)
    
    launcher = BotLauncher()
    
    try:
        asyncio.run(launcher.run_bot())
    except Exception as e:
        print(f"❌ Launcher error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
