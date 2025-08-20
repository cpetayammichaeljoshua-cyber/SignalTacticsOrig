
#!/usr/bin/env python3
"""
Keep-Alive Script for Replit Deployment
Runs a simple web server to respond to uptime pings
"""

import asyncio
import logging
from aiohttp import web
import os
from datetime import datetime

async def health_check(request):
    """Health check endpoint"""
    return web.json_response({
        'status': 'alive',
        'timestamp': datetime.now().isoformat(),
        'service': 'trading_bot_keepalive'
    })

async def ping_handler(request):
    """Ping response"""
    return web.json_response({'pong': True})

async def main():
    """Main keep-alive server"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    app = web.Application()
    app.router.add_get('/', health_check)
    app.router.add_get('/ping', ping_handler)
    app.router.add_get('/health', health_check)
    app.router.add_get('/keepalive', ping_handler)
    
    # Use port 3000 for keep-alive (different from main bot)
    port = int(os.getenv('KEEP_ALIVE_PORT', 3000))
    
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    
    logger.info(f"🌐 Keep-alive server running on port {port}")
    
    # Keep running
    while True:
        await asyncio.sleep(3600)  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
