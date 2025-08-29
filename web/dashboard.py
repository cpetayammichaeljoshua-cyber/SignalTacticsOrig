"""
Web dashboard for monitoring the trading bot
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from aiohttp import web, WSMsgType
import aiohttp_cors
from pathlib import Path

logger = logging.getLogger(__name__)

class TradingBotDashboard:
    """Web dashboard for the trading bot"""
    
    def __init__(self, trading_bot):
        self.trading_bot = trading_bot
        self.app = None
        self.websockets = set()
        self.runner = None
        self.site = None
        
    async def start(self):
        """Start the web dashboard"""
        try:
            logger.info("🌐 Starting web dashboard...")
            
            # Create aiohttp application
            self.app = web.Application()
            
            # Setup CORS
            cors = aiohttp_cors.setup(self.app, defaults={
                "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*",
                    allow_methods="*"
                )
            })
            
            # Add routes
            self._setup_routes(cors)
            
            # Start server
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            
            self.site = web.TCPSite(self.runner, '0.0.0.0', 5000)
            await self.site.start()
            
            logger.info("✅ Web dashboard started on http://0.0.0.0:5000")
            
            # Start background tasks
            asyncio.create_task(self._websocket_broadcaster())
            
        except Exception as e:
            logger.error(f"❌ Error starting web dashboard: {e}")
            raise
    
    async def stop(self):
        """Stop the web dashboard"""
        try:
            logger.info("🛑 Stopping web dashboard...")
            
            # Close all websockets
            for ws in self.websockets.copy():
                await ws.close()
            
            # Stop server
            if self.site:
                await self.site.stop()
            if self.runner:
                await self.runner.cleanup()
            
            logger.info("✅ Web dashboard stopped")
            
        except Exception as e:
            logger.error(f"❌ Error stopping web dashboard: {e}")
    
    def _setup_routes(self, cors):
        """Setup web routes"""
        # API routes first
        self.app.router.add_get('/api/status', self._handle_api_status)
        self.app.router.add_get('/api/positions', self._handle_api_positions)
        self.app.router.add_get('/api/trades', self._handle_api_trades)
        self.app.router.add_get('/api/performance', self._handle_api_performance)
        self.app.router.add_get('/api/signals', self._handle_api_signals)
        self.app.router.add_get('/ws', self._handle_websocket)
        
        # Dashboard route
        self.app.router.add_get('/', self._handle_dashboard)
        
        # Static files
        static_path = Path(__file__).parent / 'static'
        self.app.router.add_static('/static', static_path, name='static')
        
        # Add CORS to all routes
        for route in list(self.app.router.routes()):
            cors.add(route)
    
    async def _handle_dashboard(self, request):
        """Serve the main dashboard page"""
        try:
            template_path = Path(__file__).parent / 'templates' / 'dashboard.html'
            if template_path.exists():
                with open(template_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                return web.Response(text=html_content, content_type='text/html')
            else:
                return web.Response(text="Dashboard template not found", status=404)
        except Exception as e:
            logger.error(f"Error serving dashboard: {e}")
            return web.Response(text="Error loading dashboard", status=500)
    
    async def _handle_api_status(self, request):
        """API endpoint for bot status"""
        try:
            status = self.trading_bot.get_status()
            return web.json_response(status)
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def _handle_api_positions(self, request):
        """API endpoint for current positions"""
        try:
            positions = self.trading_bot.current_positions
            return web.json_response(positions)
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def _handle_api_trades(self, request):
        """API endpoint for trade history"""
        try:
            limit = int(request.query.get('limit', 50))
            trades = self.trading_bot.get_trade_history(limit)
            
            # Convert datetime objects to ISO format for JSON serialization
            serializable_trades = []
            for trade in trades:
                trade_copy = trade.copy()
                if 'timestamp' in trade_copy and isinstance(trade_copy['timestamp'], datetime):
                    trade_copy['timestamp'] = trade_copy['timestamp'].isoformat()
                serializable_trades.append(trade_copy)
            
            return web.json_response(serializable_trades)
        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def _handle_api_performance(self, request):
        """API endpoint for performance data"""
        try:
            performance = self.trading_bot.get_performance_history()
            
            # Convert datetime objects for JSON serialization
            serializable_performance = []
            for entry in performance:
                entry_copy = entry.copy()
                if 'timestamp' in entry_copy and isinstance(entry_copy['timestamp'], datetime):
                    entry_copy['timestamp'] = entry_copy['timestamp'].isoformat()
                serializable_performance.append(entry_copy)
            
            return web.json_response(serializable_performance)
        except Exception as e:
            logger.error(f"Error getting performance: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def _handle_api_signals(self, request):
        """API endpoint for signal statistics"""
        try:
            signals = self.trading_bot.get_signal_statistics()
            
            # Convert datetime objects for JSON serialization
            serializable_signals = {}
            for symbol, stats in signals.items():
                stats_copy = stats.copy()
                if 'last_signal_time' in stats_copy and isinstance(stats_copy['last_signal_time'], datetime):
                    stats_copy['last_signal_time'] = stats_copy['last_signal_time'].isoformat()
                serializable_signals[symbol] = stats_copy
            
            return web.json_response(serializable_signals)
        except Exception as e:
            logger.error(f"Error getting signals: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def _handle_websocket(self, request):
        """WebSocket handler for real-time updates"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.websockets.add(ws)
        logger.debug("WebSocket client connected")
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self._handle_websocket_message(ws, data)
                    except json.JSONDecodeError:
                        await ws.send_str(json.dumps({'error': 'Invalid JSON'}))
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f'WebSocket error: {ws.exception()}')
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            self.websockets.discard(ws)
            logger.debug("WebSocket client disconnected")
        
        return ws
    
    async def _handle_websocket_message(self, ws, data):
        """Handle incoming WebSocket messages"""
        try:
            message_type = data.get('type')
            
            if message_type == 'subscribe':
                # Client subscribing to updates
                await ws.send_str(json.dumps({
                    'type': 'subscription_confirmed',
                    'timestamp': datetime.now().isoformat()
                }))
            elif message_type == 'get_status':
                # Client requesting current status
                status = self.trading_bot.get_status()
                await ws.send_str(json.dumps({
                    'type': 'status_update',
                    'data': status,
                    'timestamp': datetime.now().isoformat()
                }))
            
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            await ws.send_str(json.dumps({'error': str(e)}))
    
    async def _websocket_broadcaster(self):
        """Broadcast updates to all connected WebSocket clients"""
        try:
            while True:
                if self.websockets:
                    try:
                        # Get current status
                        status = self.trading_bot.get_status()
                        
                        # Broadcast to all clients
                        message = json.dumps({
                            'type': 'status_update',
                            'data': status,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        # Send to all connected clients
                        disconnected = set()
                        for ws in self.websockets.copy():
                            try:
                                await ws.send_str(message)
                            except Exception as e:
                                logger.debug(f"WebSocket send failed: {e}")
                                disconnected.add(ws)
                        
                        # Remove disconnected clients
                        self.websockets -= disconnected
                        
                    except Exception as e:
                        logger.error(f"Error broadcasting to WebSockets: {e}")
                
                # Wait before next broadcast
                await asyncio.sleep(5)  # Update every 5 seconds
                
        except Exception as e:
            logger.error(f"Error in WebSocket broadcaster: {e}")

def create_dashboard_app(trading_bot):
    """Create and return dashboard application"""
    return TradingBotDashboard(trading_bot)
