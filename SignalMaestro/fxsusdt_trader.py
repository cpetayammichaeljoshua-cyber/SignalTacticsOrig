
#!/usr/bin/env python3
"""
FXSUSDT.P Futures Trader
Specialized for forex futures trading with API secrets management
"""

import asyncio
import logging
import aiohttp
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import hmac
import hashlib
import time

class FXSUSDTTrader:
    """Binance Futures trader specifically for FXSUSDT.P"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # API Configuration - Using Replit Secrets
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        self.testnet = os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'
        
        # API URLs
        if self.testnet:
            self.base_url = "https://testnet.binancefuture.com"
        else:
            self.base_url = "https://fapi.binance.com"
        
        # Trading parameters
        self.symbol = "FXSUSDT"  # Binance symbol format
        self.timeframe = "30m"
        
        # Validate API credentials
        if not self.api_key or not self.api_secret:
            self.logger.warning("⚠️ API credentials not found in secrets")
            raise ValueError("Missing BINANCE_API_KEY or BINANCE_API_SECRET in Replit secrets")
        
        self.logger.info(f"✅ FXSUSDT Trader initialized ({'Testnet' if self.testnet else 'Mainnet'})")
    
    def _generate_signature(self, query_string: str) -> str:
        """Generate HMAC SHA256 signature for authenticated requests"""
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    async def get_current_price(self) -> Optional[float]:
        """Get current FXSUSDT price"""
        try:
            url = f"{self.base_url}/fapi/v1/ticker/price"
            params = {"symbol": self.symbol}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        price = float(data['price'])
                        self.logger.debug(f"💰 {self.symbol} current price: {price:.5f}")
                        return price
                    else:
                        self.logger.error(f"Failed to get price: {response.status}")
                        return None
                        
        except Exception as e:
            self.logger.error(f"Error getting current price: {e}")
            return None
    
    async def get_klines(self, interval: str, limit: int = 100) -> List[List]:
        """Get kline data for any timeframe"""
        try:
            url = f"{self.base_url}/fapi/v1/klines"
            params = {
                "symbol": self.symbol,
                "interval": interval,
                "limit": limit
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Convert to OHLCV format
                        ohlcv_data = []
                        for kline in data:
                            ohlcv_data.append([
                                int(kline[0]),      # timestamp
                                float(kline[1]),    # open
                                float(kline[2]),    # high
                                float(kline[3]),    # low
                                float(kline[4]),    # close
                                float(kline[5])     # volume
                            ])
                        
                        self.logger.debug(f"📊 Retrieved {len(ohlcv_data)} {interval} candles for {self.symbol}")
                        return ohlcv_data
                    else:
                        self.logger.error(f"Failed to get {interval} klines: {response.status}")
                        return []
                        
        except Exception as e:
            self.logger.error(f"Error getting {interval} kline data: {e}")
            return []
    
    async def get_30m_klines(self, limit: int = 100) -> List[List]:
        """Get 30-minute kline data for FXSUSDT"""
        return await self.get_klines(self.timeframe, limit)
                
    
    async def get_account_balance(self) -> Dict[str, Any]:
        """Get futures account balance"""
        try:
            endpoint = "/fapi/v2/account"
            timestamp = int(time.time() * 1000)
            query_string = f"timestamp={timestamp}"
            signature = self._generate_signature(query_string)
            
            url = f"{self.base_url}{endpoint}?{query_string}&signature={signature}"
            
            headers = {
                'X-MBX-APIKEY': self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Extract USDT balance
                        usdt_balance = 0.0
                        for asset in data.get('assets', []):
                            if asset['asset'] == 'USDT':
                                usdt_balance = float(asset['availableBalance'])
                                break
                        
                        return {
                            'total_wallet_balance': float(data.get('totalWalletBalance', 0)),
                            'available_balance': usdt_balance,
                            'total_unrealized_pnl': float(data.get('totalUnrealizedProfit', 0))
                        }
                    else:
                        self.logger.error(f"Failed to get account: {response.status}")
                        return {}
                        
        except Exception as e:
            self.logger.error(f"Error getting account balance: {e}")
            return {}
    
    async def get_position_info(self) -> Dict[str, Any]:
        """Get current FXSUSDT position information"""
        try:
            endpoint = "/fapi/v2/positionRisk"
            timestamp = int(time.time() * 1000)
            query_string = f"symbol={self.symbol}&timestamp={timestamp}"
            signature = self._generate_signature(query_string)
            
            url = f"{self.base_url}{endpoint}?{query_string}&signature={signature}"
            
            headers = {
                'X-MBX-APIKEY': self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data and len(data) > 0:
                            position = data[0]  # FXSUSDT position
                            return {
                                'symbol': position.get('symbol'),
                                'position_amount': float(position.get('positionAmt', 0)),
                                'entry_price': float(position.get('entryPrice', 0)),
                                'mark_price': float(position.get('markPrice', 0)),
                                'unrealized_pnl': float(position.get('unRealizedProfit', 0)),
                                'leverage': float(position.get('leverage', 1))
                            }
                        else:
                            return {'position_amount': 0}
                            
                    else:
                        self.logger.error(f"Failed to get position: {response.status}")
                        return {}
                        
        except Exception as e:
            self.logger.error(f"Error getting position info: {e}")
            return {}
    
    async def get_symbol_ticker(self, symbol: str = None) -> Optional[Dict[str, Any]]:
        """Get symbol ticker information"""
        try:
            symbol = symbol or self.symbol
            url = f"{self.base_url}/fapi/v1/ticker/24hr"
            params = {"symbol": symbol}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.logger.debug(f"📊 Retrieved ticker for {symbol}")
                        return data
                    else:
                        self.logger.error(f"Failed to get ticker: {response.status}")
                        return None
                        
        except Exception as e:
            self.logger.error(f"Error getting ticker: {e}")
            return None

    async def get_funding_rate(self, symbol: str = None) -> Optional[Dict[str, Any]]:
        """Get current funding rate for symbol"""
        try:
            symbol = symbol or self.symbol
            url = f"{self.base_url}/fapi/v1/fundingRate"
            params = {"symbol": symbol, "limit": 1}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data:
                            self.logger.debug(f"📊 Retrieved funding rate for {symbol}")
                            return data[0]  # Return latest funding rate
                        return None
                    else:
                        self.logger.error(f"Failed to get funding rate: {response.status}")
                        return None
                        
        except Exception as e:
            self.logger.error(f"Error getting funding rate: {e}")
            return None

    async def get_open_interest(self, symbol: str = None) -> Optional[Dict[str, Any]]:
        """Get open interest for symbol"""
        try:
            symbol = symbol or self.symbol
            url = f"{self.base_url}/fapi/v1/openInterest"
            params = {"symbol": symbol}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.logger.debug(f"📊 Retrieved open interest for {symbol}")
                        return data
                    else:
                        self.logger.error(f"Failed to get open interest: {response.status}")
                        return None
                        
        except Exception as e:
            self.logger.error(f"Error getting open interest: {e}")
            return None

    async def get_positions(self, symbol: str = None) -> List[Dict[str, Any]]:
        """Get current positions for symbol"""
        try:
            symbol = symbol or self.symbol
            endpoint = "/fapi/v2/positionRisk"
            timestamp = int(time.time() * 1000)
            
            if symbol:
                query_string = f"symbol={symbol}&timestamp={timestamp}"
            else:
                query_string = f"timestamp={timestamp}"
                
            signature = self._generate_signature(query_string)
            url = f"{self.base_url}{endpoint}?{query_string}&signature={signature}"
            
            headers = {
                'X-MBX-APIKEY': self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Filter for non-zero positions
                        active_positions = [pos for pos in data if float(pos.get('positionAmt', 0)) != 0]
                        self.logger.debug(f"📊 Retrieved {len(active_positions)} active positions")
                        return active_positions
                    else:
                        self.logger.error(f"Failed to get positions: {response.status}")
                        return []
                        
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []

    async def get_account_info(self) -> Dict[str, Any]:
        """Get detailed account information"""
        try:
            endpoint = "/fapi/v2/account"
            timestamp = int(time.time() * 1000)
            query_string = f"timestamp={timestamp}"
            signature = self._generate_signature(query_string)
            
            url = f"{self.base_url}{endpoint}?{query_string}&signature={signature}"
            
            headers = {
                'X-MBX-APIKEY': self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.logger.debug("📊 Retrieved account info")
                        return data
                    else:
                        self.logger.error(f"Failed to get account info: {response.status}")
                        return {}
                        
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {}

    async def get_exchange_info(self, symbol: str = None) -> Dict[str, Any]:
        """Get exchange information for symbol"""
        try:
            url = f"{self.base_url}/fapi/v1/exchangeInfo"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if symbol:
                            # Find specific symbol info
                            for sym_info in data.get('symbols', []):
                                if sym_info.get('symbol') == symbol:
                                    return sym_info
                            return {}
                        return data
                    else:
                        self.logger.error(f"Failed to get exchange info: {response.status}")
                        return {}
                        
        except Exception as e:
            self.logger.error(f"Error getting exchange info: {e}")
            return {}

    async def get_24hr_ticker_stats(self, symbol: str = None) -> Dict[str, Any]:
        """Get 24hr ticker statistics"""
        try:
            symbol = symbol or self.symbol
            url = f"{self.base_url}/fapi/v1/ticker/24hr"
            params = {"symbol": symbol}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.logger.debug(f"📊 Retrieved 24hr stats for {symbol}")
                        return data
                    else:
                        self.logger.error(f"Failed to get 24hr stats: {response.status}")
                        return {}
                        
        except Exception as e:
            self.logger.error(f"Error getting 24hr stats: {e}")
            return {}

    async def change_leverage(self, symbol: str, leverage: int) -> bool:
        """Change leverage for symbol"""
        try:
            endpoint = "/fapi/v1/leverage"
            timestamp = int(time.time() * 1000)
            query_string = f"symbol={symbol}&leverage={leverage}&timestamp={timestamp}"
            signature = self._generate_signature(query_string)
            
            url = f"{self.base_url}{endpoint}"
            
            headers = {
                'X-MBX-APIKEY': self.api_key
            }
            
            data = {
                'symbol': symbol,
                'leverage': leverage,
                'timestamp': timestamp,
                'signature': signature
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.logger.info(f"✅ Leverage changed to {leverage}x for {symbol}")
                        return True
                    else:
                        self.logger.error(f"Failed to change leverage: {response.status}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"Error changing leverage: {e}")
            return False

    async def get_leverage(self, symbol: str = None) -> Optional[int]:
        """Get current leverage for symbol"""
        try:
            symbol = symbol or self.symbol
            position_info = await self.get_position_info()
            
            if position_info and 'leverage' in position_info:
                return int(float(position_info['leverage']))
            
            # Fallback to account info
            account_info = await self.get_account_info()
            for position in account_info.get('positions', []):
                if position.get('symbol') == symbol:
                    return int(float(position.get('leverage', 1)))
            
            return 1  # Default leverage
            
        except Exception as e:
            self.logger.error(f"Error getting leverage: {e}")
            return None

    async def get_trade_history(self, symbol: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent trade history for symbol"""
        try:
            symbol = symbol or self.symbol
            endpoint = "/fapi/v1/userTrades"
            timestamp = int(time.time() * 1000)
            query_string = f"symbol={symbol}&limit={limit}&timestamp={timestamp}"
            signature = self._generate_signature(query_string)
            
            url = f"{self.base_url}{endpoint}?{query_string}&signature={signature}"
            
            headers = {
                'X-MBX-APIKEY': self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.logger.debug(f"📊 Retrieved {len(data)} trades for {symbol}")
                        return data
                    else:
                        self.logger.error(f"Failed to get trade history: {response.status}")
                        return []
                        
        except Exception as e:
            self.logger.error(f"Error getting trade history: {e}")
            return []

    async def get_market_data(self, symbol: str, timeframe: str, limit: int = 500) -> List[List]:
        """Get market data (alias for get_klines for compatibility)"""
        return await self.get_klines(timeframe, limit)

    async def get_order_book(self, symbol: str = None, limit: int = 20) -> Optional[Dict[str, Any]]:
        """Get order book (depth) data for market microstructure analysis"""
        try:
            symbol = symbol or self.symbol
            url = f"{self.base_url}/fapi/v1/depth"
            params = {"symbol": symbol, "limit": limit}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.logger.debug(f"📊 Order book retrieved for {symbol}")
                        return data
                    else:
                        self.logger.error(f"Failed to get order book: {response.status}")
                        return None
                        
        except Exception as e:
            self.logger.error(f"Error getting order book: {e}")
            return None

    async def get_recent_trades(self, symbol: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent market trades for time & sales tape analysis"""
        try:
            symbol = symbol or self.symbol
            url = f"{self.base_url}/fapi/v1/trades"
            params = {"symbol": symbol, "limit": limit}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.logger.debug(f"📈 Retrieved {len(data)} recent trades for {symbol}")
                        return data
                    else:
                        self.logger.error(f"Failed to get recent trades: {response.status}")
                        return []
                        
        except Exception as e:
            self.logger.error(f"Error getting recent trades: {e}")
            return []

    async def test_connection(self) -> bool:
        """Test API connection and credentials"""
        try:
            # Test public endpoint first
            price = await self.get_current_price()
            if price is None:
                return False
            
            # Test authenticated endpoint
            account = await self.get_account_balance()
            if not account:
                return False
            
            self.logger.info("✅ API connection test successful")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ API connection test failed: {e}")
            return False
