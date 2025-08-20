
#!/usr/bin/env python3
"""
Uptime Service for Trading Bot
Includes a minimal web server and external ping service integration
"""

import asyncio
import logging
import aiohttp
from aiohttp import web
import time
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import json
import threading

class UptimeService:
    """Uptime monitoring service with web server"""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.logger = logging.getLogger(__name__)
        self.app = None
        self.runner = None
        self.start_time = datetime.now()
        self.ping_count = 0
        self.last_ping = None
        self.external_ping_urls = [
            "https://kaffeine.herokuapp.com/",  # Kaffeine service
            "https://uptimerobot.com/",         # UptimeRobot (if configured)
        ]
        self.replit_url = None
        self._setup_replit_url()
        
    def _setup_replit_url(self):
        """Setup Replit URL for external pinging"""
        repl_name = os.getenv('REPL_SLUG', 'trading-bot')
        repl_owner = os.getenv('REPL_OWNER', 'user')
        self.replit_url = f"https://{repl_name}.{repl_owner}.repl.co"
        self.logger.info(f"Replit URL configured: {self.replit_url}")
    
    def create_app(self):
        """Create the web application"""
        app = web.Application()
        
        # Health check endpoint
        app.router.add_get('/', self.health_check)
        app.router.add_get('/ping', self.ping_handler)
        app.router.add_get('/health', self.health_check)
        app.router.add_get('/status', self.status_handler)
        app.router.add_get('/uptime', self.uptime_handler)
        
        # Keep-alive endpoint for external services
        app.router.add_get('/keepalive', self.keepalive_handler)
        
        return app
    
    async def health_check(self, request):
        """Basic health check endpoint"""
        return web.json_response({
            'status': 'healthy',
            'service': 'trading_bot_uptime',
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
        })
    
    async def ping_handler(self, request):
        """Handle ping requests"""
        self.ping_count += 1
        self.last_ping = datetime.now()
        
        return web.json_response({
            'pong': True,
            'ping_count': self.ping_count,
            'timestamp': self.last_ping.isoformat(),
            'message': 'Bot is alive'
        })
    
    async def status_handler(self, request):
        """Detailed status information"""
        uptime_seconds = (datetime.now() - self.start_time).total_seconds()
        
        status = {
            'service': 'Trading Bot Uptime Service',
            'status': 'running',
            'start_time': self.start_time.isoformat(),
            'uptime_seconds': uptime_seconds,
            'uptime_human': self._format_uptime(uptime_seconds),
            'ping_count': self.ping_count,
            'last_ping': self.last_ping.isoformat() if self.last_ping else None,
            'replit_url': self.replit_url,
            'port': self.port
        }
        
        return web.json_response(status)
    
    async def uptime_handler(self, request):
        """Simple uptime endpoint"""
        uptime_seconds = (datetime.now() - self.start_time).total_seconds()
        return web.Response(text=f"Uptime: {self._format_uptime(uptime_seconds)}")
    
    async def keepalive_handler(self, request):
        """Keep-alive endpoint for external monitoring services"""
        self.ping_count += 1
        self.last_ping = datetime.now()
        
        return web.Response(text="OK", status=200)
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human-readable format"""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
    
    async def start_server(self):
        """Start the web server"""
        try:
            self.app = self.create_app()
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            
            site = web.TCPSite(self.runner, '0.0.0.0', self.port)
            await site.start()
            
            self.logger.info(f"🌐 Uptime service started on port {self.port}")
            self.logger.info(f"📡 Health check: {self.replit_url}/health")
            self.logger.info(f"🏓 Ping endpoint: {self.replit_url}/ping")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start uptime service: {e}")
            return False
    
    async def stop_server(self):
        """Stop the web server"""
        if self.runner:
            await self.runner.cleanup()
            self.logger.info("🛑 Uptime service stopped")
    
    async def self_ping_loop(self):
        """Internal ping loop to keep the service active"""
        while True:
            try:
                await asyncio.sleep(240)  # Ping every 4 minutes
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://localhost:{self.port}/ping") as response:
                        if response.status == 200:
                            self.logger.debug("✅ Self-ping successful")
                        else:
                            self.logger.warning(f"⚠️ Self-ping failed: {response.status}")
                            
            except Exception as e:
                self.logger.error(f"Self-ping error: {e}")
    
    async def external_ping_setup(self):
        """Setup external ping services"""
        if not self.replit_url:
            self.logger.warning("No Replit URL configured for external pings")
            return
        
        # Log instructions for manual setup
        self.logger.info("🔧 External Ping Setup Instructions:")
        self.logger.info(f"📌 Add this URL to Kaffeine: {self.replit_url}/keepalive")
        self.logger.info("📌 Visit: https://kaffeine.herokuapp.com/")
        self.logger.info("📌 Or use UptimeRobot, Pingdom, or similar services")
        
        # Try to automatically ping Kaffeine (if accessible)
        await self._try_kaffeine_setup()
    
    async def _try_kaffeine_setup(self):
        """Try to setup Kaffeine automatically"""
        try:
            kaffeine_url = "https://kaffeine.herokuapp.com/"
            payload = {
                'url': f"{self.replit_url}/keepalive",
                'interval': 5  # 5 minutes
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(kaffeine_url, json=payload) as response:
                    if response.status == 200:
                        self.logger.info("✅ Kaffeine auto-setup successful")
                    else:
                        self.logger.info("ℹ️ Kaffeine auto-setup not available, manual setup required")
                        
        except Exception as e:
            self.logger.debug(f"Kaffeine auto-setup failed: {e}")
    
    async def run(self):
        """Main run method"""
        try:
            # Start the web server
            if await self.start_server():
                # Setup external pings
                await self.external_ping_setup()
                
                # Start self-ping loop
                asyncio.create_task(self.self_ping_loop())
                
                self.logger.info("🚀 Uptime service fully operational")
                
                # Keep running
                while True:
                    await asyncio.sleep(60)
                    
        except KeyboardInterrupt:
            self.logger.info("🛑 Uptime service shutdown requested")
        finally:
            await self.stop_server()

async def main():
    """Main function"""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s'
    )
    
    # Use port 8080 to avoid conflicts with main bot on 5000
    uptime_service = UptimeService(port=8080)
    await uptime_service.run()

if __name__ == "__main__":
    asyncio.run(main())
