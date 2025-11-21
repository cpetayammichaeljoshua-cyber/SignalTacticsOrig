#!/usr/bin/env python3
"""
ATAS Platform Integration Module
Provides HTTP API bridge to export trading signals to ATAS platform
Since ATAS uses C# .NET, this module creates a REST API that ATAS can consume
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import aiohttp
from aiohttp import web
import os


@dataclass
class ATASSignal:
    """Signal formatted for ATAS platform"""
    timestamp: str
    symbol: str
    direction: str  # BUY or SELL
    entry_price: float
    stop_loss: float
    take_profit: float
    leverage: int
    signal_strength: float
    timeframe: str
    strategy_name: str
    metadata: Dict[str, Any]


class ATASPlatformIntegration:
    """
    Integration bridge between Python trading bot and ATAS platform
    Provides REST API endpoints that ATAS C# strategies can call
    """
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8888):
        self.logger = logging.getLogger(__name__)
        self.host = host
        self.port = port
        
        # Signal storage
        self.active_signals: List[ATASSignal] = []
        self.signal_history: List[ATASSignal] = []
        self.max_active_signals = 50
        self.max_history = 500
        
        # API statistics
        self.total_signals_exported = 0
        self.total_api_requests = 0
        
        # Web app
        self.app = web.Application()
        self._setup_routes()
        
        self.logger.info(f"✅ ATAS Integration initialized on {host}:{port}")
    
    def _setup_routes(self):
        """Configure API routes"""
        self.app.router.add_get('/', self.handle_index)
        self.app.router.add_get('/api/signals', self.handle_get_signals)
        self.app.router.add_get('/api/signals/latest', self.handle_get_latest_signal)
        self.app.router.add_get('/api/signals/{symbol}', self.handle_get_symbol_signals)
        self.app.router.add_post('/api/signals/acknowledge', self.handle_acknowledge_signal)
        self.app.router.add_get('/api/health', self.handle_health)
        self.app.router.add_get('/api/stats', self.handle_stats)
    
    async def handle_index(self, request):
        """API documentation endpoint"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ATAS Platform Integration API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f4f4f4; }
                .container { background: white; padding: 30px; border-radius: 8px; }
                h1 { color: #2c3e50; }
                .endpoint { background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }
                code { background: #34495e; color: #ecf0f1; padding: 3px 6px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔌 ATAS Platform Integration API</h1>
                <p>REST API for exporting Python trading signals to ATAS C# strategies</p>
                
                <h2>Available Endpoints:</h2>
                
                <div class="endpoint">
                    <h3>GET /api/signals</h3>
                    <p>Retrieve all active trading signals</p>
                    <code>curl http://localhost:8888/api/signals</code>
                </div>
                
                <div class="endpoint">
                    <h3>GET /api/signals/latest</h3>
                    <p>Get the most recent signal</p>
                    <code>curl http://localhost:8888/api/signals/latest</code>
                </div>
                
                <div class="endpoint">
                    <h3>GET /api/signals/{symbol}</h3>
                    <p>Get signals for specific symbol (e.g., ETHUSDT)</p>
                    <code>curl http://localhost:8888/api/signals/ETHUSDT</code>
                </div>
                
                <div class="endpoint">
                    <h3>GET /api/health</h3>
                    <p>Health check endpoint</p>
                    <code>curl http://localhost:8888/api/health</code>
                </div>
                
                <div class="endpoint">
                    <h3>GET /api/stats</h3>
                    <p>Get API statistics</p>
                    <code>curl http://localhost:8888/api/stats</code>
                </div>
                
                <h2>ATAS C# Integration Example:</h2>
                <pre><code>
// In your ATAS C# strategy:
using System.Net.Http;
using Newtonsoft.Json;

public class ATASSignalImporter : ChartStrategy
{
    private readonly HttpClient _httpClient = new HttpClient();
    private const string API_URL = "http://localhost:8888/api/signals/latest";
    
    protected override void OnCalculate(int bar, decimal value)
    {
        if (bar % 100 == 0) // Check every 100 bars
        {
            var signal = FetchLatestSignal();
            if (signal != null && signal.Direction == "BUY")
            {
                // Execute buy order in ATAS
                TradingManager.PlaceMarketOrder(OrderSide.Buy, GetQuantity());
            }
        }
    }
    
    private Signal FetchLatestSignal()
    {
        var response = _httpClient.GetAsync(API_URL).Result;
        if (response.IsSuccessStatusCode)
        {
            var json = response.Content.ReadAsStringAsync().Result;
            return JsonConvert.DeserializeObject&lt;Signal&gt;(json);
        }
        return null;
    }
}
                </code></pre>
            </div>
        </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')
    
    async def handle_get_signals(self, request):
        """Get all active signals"""
        self.total_api_requests += 1
        
        signals_data = [asdict(signal) for signal in self.active_signals]
        
        return web.json_response({
            'status': 'success',
            'count': len(signals_data),
            'signals': signals_data,
            'timestamp': datetime.now().isoformat()
        })
    
    async def handle_get_latest_signal(self, request):
        """Get most recent signal"""
        self.total_api_requests += 1
        
        if not self.active_signals:
            return web.json_response({
                'status': 'no_signals',
                'message': 'No active signals available',
                'timestamp': datetime.now().isoformat()
            }, status=404)
        
        latest = self.active_signals[-1]
        
        return web.json_response({
            'status': 'success',
            'signal': asdict(latest),
            'timestamp': datetime.now().isoformat()
        })
    
    async def handle_get_symbol_signals(self, request):
        """Get signals for specific symbol"""
        self.total_api_requests += 1
        
        symbol = request.match_info['symbol'].upper()
        
        # Filter signals by symbol
        symbol_signals = [
            signal for signal in self.active_signals
            if symbol in signal.symbol.replace('/', '').replace(':USDT', '')
        ]
        
        signals_data = [asdict(signal) for signal in symbol_signals]
        
        return web.json_response({
            'status': 'success',
            'symbol': symbol,
            'count': len(signals_data),
            'signals': signals_data,
            'timestamp': datetime.now().isoformat()
        })
    
    async def handle_acknowledge_signal(self, request):
        """Acknowledge signal has been processed by ATAS"""
        self.total_api_requests += 1
        
        try:
            data = await request.json()
            signal_timestamp = data.get('timestamp')
            
            # Move to history
            self.active_signals = [
                s for s in self.active_signals 
                if s.timestamp != signal_timestamp
            ]
            
            return web.json_response({
                'status': 'success',
                'message': 'Signal acknowledged',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return web.json_response({
                'status': 'error',
                'message': str(e)
            }, status=400)
    
    async def handle_health(self, request):
        """Health check"""
        self.total_api_requests += 1
        
        return web.json_response({
            'status': 'healthy',
            'service': 'ATAS Platform Integration',
            'active_signals': len(self.active_signals),
            'uptime': 'running',
            'timestamp': datetime.now().isoformat()
        })
    
    async def handle_stats(self, request):
        """API statistics"""
        self.total_api_requests += 1
        
        return web.json_response({
            'status': 'success',
            'statistics': {
                'total_signals_exported': self.total_signals_exported,
                'active_signals': len(self.active_signals),
                'historical_signals': len(self.signal_history),
                'total_api_requests': self.total_api_requests
            },
            'timestamp': datetime.now().isoformat()
        })
    
    async def export_signal(self, signal: Any):
        """
        Export trading signal to ATAS platform
        
        Args:
            signal: HighFrequencySignal or dict
        """
        try:
            # Convert to ATAS format
            atas_signal = self._convert_to_atas_signal(signal)
            
            # Add to active signals
            self.active_signals.append(atas_signal)
            
            # Maintain max limit
            if len(self.active_signals) > self.max_active_signals:
                # Move oldest to history
                old_signal = self.active_signals.pop(0)
                self.signal_history.append(old_signal)
            
            # Maintain history limit
            if len(self.signal_history) > self.max_history:
                self.signal_history.pop(0)
            
            self.total_signals_exported += 1
            
            self.logger.info(f"📤 Exported signal to ATAS: {atas_signal.symbol} {atas_signal.direction}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting signal to ATAS: {e}")
            return False
    
    def _convert_to_atas_signal(self, signal: Any) -> ATASSignal:
        """Convert Python signal to ATAS format"""
        
        # Extract data (supports both object and dict)
        symbol = self._get_attr(signal, 'symbol', 'UNKNOWN')
        direction = self._get_attr(signal, 'direction', 'LONG')
        
        # Convert symbol format (ETH/USDT:USDT -> ETHUSDT)
        clean_symbol = symbol.replace('/USDT:USDT', '').replace('/', '')
        
        # Convert direction
        atas_direction = 'BUY' if direction == 'LONG' else 'SELL'
        
        # Get primary take profit (TP2)
        take_profit = self._get_attr(signal, 'take_profit_2', 
                                     self._get_attr(signal, 'take_profit_1', 0))
        
        return ATASSignal(
            timestamp=datetime.now().isoformat(),
            symbol=clean_symbol,
            direction=atas_direction,
            entry_price=self._get_attr(signal, 'entry_price', 0),
            stop_loss=self._get_attr(signal, 'stop_loss', 0),
            take_profit=take_profit,
            leverage=self._get_attr(signal, 'leverage', 10),
            signal_strength=self._get_attr(signal, 'signal_strength', 0),
            timeframe=self._get_attr(signal, 'timeframe', '1m'),
            strategy_name='HighFrequencyScalping',
            metadata={
                'consensus_confidence': self._get_attr(signal, 'consensus_confidence', 0),
                'strategies_agree': self._get_attr(signal, 'strategies_agree', 0),
                'risk_reward_ratio': self._get_attr(signal, 'risk_reward_ratio', 0),
                'tp1': self._get_attr(signal, 'take_profit_1', 0),
                'tp2': self._get_attr(signal, 'take_profit_2', 0),
                'tp3': self._get_attr(signal, 'take_profit_3', 0)
            }
        )
    
    def _get_attr(self, obj: Any, attr: str, default: Any = None) -> Any:
        """Get attribute from object or dict"""
        if hasattr(obj, attr):
            return getattr(obj, attr)
        elif isinstance(obj, dict):
            return obj.get(attr, default)
        return default
    
    async def start_server(self):
        """Start the API server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        self.logger.info(f"🚀 ATAS Integration API started on http://{self.host}:{self.port}")
        self.logger.info(f"📖 API Documentation: http://{self.host}:{self.port}/")
        
        # Keep running
        while True:
            await asyncio.sleep(3600)


async def demo_atas_integration():
    """Demo ATAS integration"""
    print("\n" + "="*80)
    print("🔌 ATAS PLATFORM INTEGRATION DEMO")
    print("="*80)
    
    integration = ATASPlatformIntegration(port=8888)
    
    # Start server in background
    server_task = asyncio.create_task(integration.start_server())
    
    # Wait a bit for server to start
    await asyncio.sleep(2)
    
    # Simulate exporting some signals
    test_signals = [
        {
            'symbol': 'ETH/USDT:USDT',
            'direction': 'LONG',
            'entry_price': 3500.00,
            'stop_loss': 3482.50,
            'take_profit_1': 3528.00,
            'take_profit_2': 3542.00,
            'take_profit_3': 3563.00,
            'leverage': 20,
            'signal_strength': 85.5,
            'consensus_confidence': 75.0,
            'timeframe': '1m'
        },
        {
            'symbol': 'BTC/USDT:USDT',
            'direction': 'SHORT',
            'entry_price': 50000.00,
            'stop_loss': 50500.00,
            'take_profit_1': 49500.00,
            'take_profit_2': 49000.00,
            'take_profit_3': 48500.00,
            'leverage': 15,
            'signal_strength': 78.3,
            'consensus_confidence': 70.0,
            'timeframe': '3m'
        }
    ]
    
    for signal in test_signals:
        await integration.export_signal(signal)
        print(f"✅ Exported signal: {signal['symbol']} {signal['direction']}")
    
    print(f"\n📊 API Stats:")
    print(f"   Active Signals: {len(integration.active_signals)}")
    print(f"   Total Exported: {integration.total_signals_exported}")
    
    print(f"\n🌐 API running at: http://localhost:8888")
    print(f"📖 Documentation: http://localhost:8888/")
    print(f"🔍 Test endpoint: http://localhost:8888/api/signals")
    
    # Keep running
    await server_task


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_atas_integration())
