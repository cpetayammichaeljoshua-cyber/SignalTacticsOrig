

#!/usr/bin/env python3
"""
Production Monitoring Script for Replit Deployment
Provides external monitoring and alerting capabilities
"""

import asyncio
import aiohttp
import json
import time
import logging
from datetime import datetime
import os

class ProductionMonitor:
    """External monitoring for production deployment"""
    
    def __init__(self):
        self.deployment_url = f"https://{os.getenv('REPL_SLUG', 'perfect-scalping-bot')}.{os.getenv('REPL_OWNER', 'user')}.repl.co"
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.alert_chat_id = os.getenv('ALERT_CHAT_ID')
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - MONITOR - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    async def check_deployment_health(self):
        """Check if deployment is healthy"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.deployment_url}/", timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'healthy': data.get('status') == 'healthy',
                            'bot_running': data.get('bot_running', False),
                            'uptime': data.get('uptime', 0),
                            'memory_usage': data.get('memory_usage', 0),
                            'timestamp': data.get('timestamp')
                        }
                    else:
                        return {'healthy': False, 'error': f'HTTP {response.status}'}
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    async def send_alert(self, message):
        """Send alert via Telegram"""
        if not self.telegram_token or not self.alert_chat_id:
            self.logger.warning("Telegram alerts not configured")
            return
            
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                'chat_id': self.alert_chat_id,
                'text': f"🚨 PRODUCTION ALERT\n\n{message}",
                'parse_mode': 'Markdown'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        self.logger.info("Alert sent successfully")
                    else:
                        self.logger.error(f"Failed to send alert: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Error sending alert: {e}")
    
    async def force_restart_deployment(self):
        """Force restart the deployment"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.deployment_url}/force-restart") as response:
                    if response.status == 200:
                        self.logger.info("Forced restart initiated")
                        await self.send_alert("Forced restart initiated due to health check failure")
                        return True
                    else:
                        self.logger.error(f"Failed to force restart: {response.status}")
                        return False
        except Exception as e:
            self.logger.error(f"Error forcing restart: {e}")
            return False
    
    async def monitor_loop(self):
        """Main monitoring loop"""
        consecutive_failures = 0
        max_failures = 3
        check_interval = 60  # Check every minute
        
        self.logger.info("🔍 Starting production monitoring loop...")
        
        while True:
            try:
                health = await self.check_deployment_health()
                
                if health['healthy']:
                    consecutive_failures = 0
                    if health.get('bot_running'):
                        self.logger.info(f"✅ Deployment healthy - Uptime: {health.get('uptime', 0)}s")
                    else:
                        self.logger.warning("⚠️ Deployment healthy but bot not running")
                else:
                    consecutive_failures += 1
                    error_msg = health.get('error', 'Unknown error')
                    self.logger.error(f"❌ Health check failed ({consecutive_failures}/{max_failures}): {error_msg}")
                    
                    if consecutive_failures >= max_failures:
                        self.logger.critical("🚨 Multiple consecutive failures - attempting force restart")
                        await self.send_alert(f"Deployment unhealthy for {consecutive_failures} checks. Error: {error_msg}")
                        
                        # Attempt force restart
                        if await self.force_restart_deployment():
                            consecutive_failures = 0
                        else:
                            await self.send_alert("❌ Failed to force restart deployment - manual intervention required")
                
                await asyncio.sleep(check_interval)
                
            except KeyboardInterrupt:
                self.logger.info("🛑 Monitoring stopped")
                break
            except Exception as e:
                self.logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(10)

async def main():
    """Main entry point for production monitoring"""
    monitor = ProductionMonitor()
    await monitor.monitor_loop()

if __name__ == "__main__":
    asyncio.run(main())
