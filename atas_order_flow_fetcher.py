#!/usr/bin/env python3
"""
ATAS Order Flow Data Fetcher
Fetches real-time order flow analysis from ATAS platform:
- Cumulative Volume Delta (CVD)
- Buy/Sell volume distribution
- Order flow imbalances
- Large order detection
- Absorption zones
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import aiohttp


class ATASOrderFlowFetcher:
    """
    Fetches order flow data from ATAS platform
    Provides real-time buy/sell pressure analysis
    """
    
    def __init__(self, atas_host: str = "localhost", atas_port: int = 8888):
        self.logger = logging.getLogger(__name__)
        self.atas_url = f"http://{atas_host}:{atas_port}"
        self.cache = {}
        self.cache_ttl = 5  # 5 seconds cache
        
        self.logger.info(f"✅ ATAS Order Flow Fetcher initialized: {self.atas_url}")
    
    async def fetch_order_flow(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch order flow data for symbol from ATAS
        
        Args:
            symbol: Trading pair (e.g., 'ETH/USDT:USDT')
            
        Returns:
            Order flow data dictionary
        """
        try:
            # Check cache first
            cache_key = f"flow_{symbol}"
            if cache_key in self.cache:
                cached_data, cached_time = self.cache[cache_key]
                if (datetime.now() - cached_time).total_seconds() < self.cache_ttl:
                    return cached_data
            
            # Fetch from ATAS
            url = f"{self.atas_url}/api/orderflow/{symbol}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Cache the result
                        self.cache[cache_key] = (data, datetime.now())
                        
                        return data
                    else:
                        self.logger.warning(f"ATAS API error: {response.status}")
                        return None
            
        except asyncio.TimeoutError:
            self.logger.warning(f"ATAS API timeout for {symbol}")
            return None
        except Exception as e:
            self.logger.debug(f"Order flow fetch error: {e}")
            return None
    
    async def fetch_volume_profile(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch volume profile data from ATAS"""
        try:
            url = f"{self.atas_url}/api/volumeprofile/{symbol}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        return await response.json()
                    return None
        except Exception as e:
            self.logger.debug(f"Volume profile fetch error: {e}")
            return None
    
    async def fetch_large_orders(self, symbol: str) -> Optional[List[Dict[str, Any]]]:
        """Fetch large orders (whale activity) from ATAS"""
        try:
            url = f"{self.atas_url}/api/largeorders/{symbol}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        return await response.json()
                    return None
        except Exception as e:
            self.logger.debug(f"Large orders fetch error: {e}")
            return None
    
    async def fetch_market_depth(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch market depth (order book) from ATAS"""
        try:
            url = f"{self.atas_url}/api/depth/{symbol}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        return await response.json()
                    return None
        except Exception as e:
            self.logger.debug(f"Market depth fetch error: {e}")
            return None
    
    async def get_comprehensive_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive order flow analysis
        
        Returns:
            Complete analysis with buy/sell pressure, delta, and recommendations
        """
        try:
            # Fetch all data in parallel
            flow_task = self.fetch_order_flow(symbol)
            profile_task = self.fetch_volume_profile(symbol)
            orders_task = self.fetch_large_orders(symbol)
            depth_task = self.fetch_market_depth(symbol)
            
            flow_data, profile_data, orders_data, depth_data = await asyncio.gather(
                flow_task, profile_task, orders_task, depth_task,
                return_exceptions=True
            )
            
            # Build comprehensive analysis
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'order_flow': self._parse_order_flow(flow_data) if flow_data and not isinstance(flow_data, Exception) else {},
                'volume_profile': self._parse_volume_profile(profile_data) if profile_data and not isinstance(profile_data, Exception) else {},
                'large_orders': self._parse_large_orders(orders_data) if orders_data and not isinstance(orders_data, Exception) else {},
                'market_depth': self._parse_market_depth(depth_data) if depth_data and not isinstance(depth_data, Exception) else {},
                'recommendation': 'NEUTRAL'
            }
            
            # Generate recommendation
            analysis['recommendation'] = self._generate_recommendation(analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis error: {e}")
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'recommendation': 'NEUTRAL'
            }
    
    def _parse_order_flow(self, data: Dict) -> Dict:
        """Parse order flow data"""
        if not data:
            return {}
        
        return {
            'buy_volume': data.get('buy_volume', 0),
            'sell_volume': data.get('sell_volume', 0),
            'delta': data.get('delta', 0),
            'cvd': data.get('cvd', 0),
            'buy_pressure': data.get('buy_pressure', 50.0),
            'sell_pressure': data.get('sell_pressure', 50.0)
        }
    
    def _parse_volume_profile(self, data: Dict) -> Dict:
        """Parse volume profile data"""
        if not data:
            return {}
        
        return {
            'poc': data.get('poc', 0),  # Point of Control
            'value_area_high': data.get('vah', 0),
            'value_area_low': data.get('val', 0),
            'high_volume_nodes': data.get('hvn', []),
            'low_volume_nodes': data.get('lvn', [])
        }
    
    def _parse_large_orders(self, data: List) -> Dict:
        """Parse large orders data"""
        if not data:
            return {'count': 0, 'buy_count': 0, 'sell_count': 0}
        
        buy_count = sum(1 for order in data if order.get('side') == 'buy')
        sell_count = sum(1 for order in data if order.get('side') == 'sell')
        
        return {
            'count': len(data),
            'buy_count': buy_count,
            'sell_count': sell_count,
            'buy_dominance': (buy_count / len(data) * 100) if data else 50.0
        }
    
    def _parse_market_depth(self, data: Dict) -> Dict:
        """Parse market depth data"""
        if not data:
            return {}
        
        bids = data.get('bids', [])
        asks = data.get('asks', [])
        
        bid_volume = sum(b[1] for b in bids[:20]) if bids else 0
        ask_volume = sum(a[1] for a in asks[:20]) if asks else 0
        total = bid_volume + ask_volume
        
        return {
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'bid_dominance': (bid_volume / total * 100) if total > 0 else 50.0,
            'imbalance': (bid_volume - ask_volume) / total * 100 if total > 0 else 0
        }
    
    def _generate_recommendation(self, analysis: Dict) -> str:
        """
        Generate trading recommendation based on comprehensive analysis
        
        Returns:
            'BULLISH', 'BEARISH', or 'NEUTRAL'
        """
        bullish_score = 0
        bearish_score = 0
        
        # Order flow signals
        flow = analysis.get('order_flow', {})
        if flow.get('buy_pressure', 50) > 60:
            bullish_score += 2
        if flow.get('sell_pressure', 50) > 60:
            bearish_score += 2
        
        if flow.get('delta', 0) > 0:
            bullish_score += 1
        elif flow.get('delta', 0) < 0:
            bearish_score += 1
        
        # Large orders
        orders = analysis.get('large_orders', {})
        if orders.get('buy_dominance', 50) > 60:
            bullish_score += 1
        elif orders.get('buy_dominance', 50) < 40:
            bearish_score += 1
        
        # Market depth
        depth = analysis.get('market_depth', {})
        if depth.get('bid_dominance', 50) > 55:
            bullish_score += 1
        elif depth.get('bid_dominance', 50) < 45:
            bearish_score += 1
        
        # Decision
        if bullish_score >= bearish_score + 2:
            return 'BULLISH'
        elif bearish_score >= bullish_score + 2:
            return 'BEARISH'
        else:
            return 'NEUTRAL'


async def demo_atas_fetcher():
    """Demo the ATAS order flow fetcher"""
    print("\n" + "="*80)
    print("📊 ATAS ORDER FLOW FETCHER DEMO")
    print("="*80)
    
    fetcher = ATASOrderFlowFetcher()
    
    symbol = "ETH/USDT:USDT"
    
    print(f"\n🔍 Fetching order flow data for {symbol}...")
    analysis = await fetcher.get_comprehensive_analysis(symbol)
    
    print(f"\n📊 Analysis Result:")
    print(f"   Symbol: {analysis['symbol']}")
    print(f"   Recommendation: {analysis['recommendation']}")
    
    if analysis.get('order_flow'):
        flow = analysis['order_flow']
        print(f"\n📈 Order Flow:")
        print(f"   Buy Pressure: {flow.get('buy_pressure', 0):.1f}%")
        print(f"   Sell Pressure: {flow.get('sell_pressure', 0):.1f}%")
        print(f"   Delta: {flow.get('delta', 0)}")
    
    if analysis.get('large_orders'):
        orders = analysis['large_orders']
        print(f"\n🐋 Large Orders:")
        print(f"   Total: {orders.get('count', 0)}")
        print(f"   Buy Dominance: {orders.get('buy_dominance', 0):.1f}%")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_atas_fetcher())
