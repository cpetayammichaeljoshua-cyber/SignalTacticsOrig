#!/usr/bin/env python3
"""
Async Market Data Fetcher with Caching
Efficiently fetches and caches market data to prevent redundant API calls
"""

import asyncio
import aiohttp
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import os

from SignalMaestro.market_data_contracts import MarketSnapshot, OrderBookSnapshot

class AsyncMarketDataFetcher:
    """
    Fetches market data asynchronously with intelligent caching
    Prevents redundant API calls and shares data across analyzers
    """
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Binance API credentials
        self.api_key = api_key or os.getenv('BINANCE_API_KEY', '')
        self.api_secret = api_secret or os.getenv('BINANCE_SECRET_KEY', '')
        
        # Base URLs
        self.base_url = 'https://fapi.binance.com'  # Futures
        self.spot_url = 'https://api.binance.com'
        
        # Cache
        self.cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        self.cache_ttl = timedelta(seconds=10)  # 10 second TTL
        
        # Rate limiting
        self.request_count = 0
        self.last_request_time = datetime.now()
        self.max_requests_per_minute = 1200
        
    async def fetch_market_snapshot(self, symbol: str, 
                                    timeframe: str = '30m',
                                    limit: int = 500,
                                    include_orderbook: bool = True,
                                    include_trades: bool = True,
                                    correlated_symbols: Optional[List[str]] = None) -> MarketSnapshot:
        """
        Fetch complete market snapshot
        
        Args:
            symbol: Trading symbol (e.g., 'FXSUSDT')
            timeframe: Candlestick timeframe
            limit: Number of candles to fetch
            include_orderbook: Whether to fetch order book
            include_trades: Whether to fetch recent trades
            correlated_symbols: List of symbols to fetch for correlation analysis
            
        Returns:
            MarketSnapshot with all requested data
        """
        # Fetch all data concurrently
        tasks = []
        
        # OHLCV data
        tasks.append(self._fetch_ohlcv(symbol, timeframe, limit))
        
        # Current price
        tasks.append(self._fetch_ticker(symbol))
        
        # Optional data
        if include_orderbook:
            tasks.append(self._fetch_orderbook(symbol))
        else:
            tasks.append(asyncio.sleep(0))  # Placeholder
        
        if include_trades:
            tasks.append(self._fetch_recent_trades(symbol))
        else:
            tasks.append(asyncio.sleep(0))  # Placeholder
        
        # Funding rate
        tasks.append(self._fetch_funding_rate(symbol))
        
        # Open interest
        tasks.append(self._fetch_open_interest(symbol))
        
        # Correlated symbols
        if correlated_symbols:
            for corr_symbol in correlated_symbols:
                tasks.append(self._fetch_ohlcv(corr_symbol, timeframe, limit))
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Parse results
        ohlcv_df = results[0] if not isinstance(results[0], Exception) else pd.DataFrame()
        ticker = results[1] if not isinstance(results[1], Exception) else {}
        orderbook = results[2] if not isinstance(results[2], Exception) and include_orderbook else None
        trades = results[3] if not isinstance(results[3], Exception) and include_trades else None
        funding_rate = results[4] if not isinstance(results[4], Exception) else None
        open_interest = results[5] if not isinstance(results[5], Exception) else None
        
        # Parse correlated symbols data
        correlated_data = {}
        if correlated_symbols:
            for i, corr_symbol in enumerate(correlated_symbols):
                corr_df = results[6 + i]
                if not isinstance(corr_df, Exception):
                    correlated_data[corr_symbol] = corr_df
        
        # Extract price data
        current_price = ticker.get('lastPrice', 0.0) if ticker else 0.0
        volume_24h = ticker.get('quoteVolume', 0.0) if ticker else 0.0
        price_change_24h = ticker.get('priceChangePercent', 0.0) if ticker else 0.0
        
        # Create snapshot
        return MarketSnapshot(
            symbol=symbol,
            timestamp=datetime.now(),
            ohlcv_df=ohlcv_df,
            current_price=float(current_price),
            bids=orderbook.bids if orderbook else None,
            asks=orderbook.asks if orderbook else None,
            recent_trades=trades,
            funding_rate=funding_rate,
            open_interest=open_interest,
            volume_24h=float(volume_24h) if volume_24h else None,
            price_change_24h=float(price_change_24h) if price_change_24h else None,
            correlated_symbols=correlated_data if correlated_data else None
        )
    
    async def _fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Fetch OHLCV data"""
        cache_key = f"ohlcv_{symbol}_{timeframe}_{limit}"
        
        # Check cache
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        # Fetch from API
        url = f"{self.base_url}/fapi/v1/klines"
        params = {
            'symbol': symbol,
            'interval': timeframe,
            'limit': limit
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Convert to DataFrame
                        df = pd.DataFrame(data, columns=[
                            'timestamp', 'open', 'high', 'low', 'close', 'volume',
                            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                            'taker_buy_quote', 'ignore'
                        ])
                        
                        # Convert types
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            df[col] = df[col].astype(float)
                        
                        # Cache result
                        self._cache_result(cache_key, df)
                        
                        return df
                    else:
                        self.logger.warning(f"Failed to fetch OHLCV for {symbol}: {response.status}")
                        return pd.DataFrame()
                        
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return pd.DataFrame()
    
    async def _fetch_ticker(self, symbol: str) -> Dict:
        """Fetch 24h ticker data"""
        cache_key = f"ticker_{symbol}"
        
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        url = f"{self.base_url}/fapi/v1/ticker/24hr"
        params = {'symbol': symbol}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        self._cache_result(cache_key, data)
                        return data
                    return {}
        except Exception as e:
            self.logger.error(f"Error fetching ticker for {symbol}: {e}")
            return {}
    
    async def _fetch_orderbook(self, symbol: str, depth: int = 20) -> Optional[OrderBookSnapshot]:
        """Fetch order book"""
        cache_key = f"orderbook_{symbol}_{depth}"
        
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        url = f"{self.base_url}/fapi/v1/depth"
        params = {'symbol': symbol, 'limit': depth}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        orderbook = OrderBookSnapshot(
                            symbol=symbol,
                            timestamp=datetime.now(),
                            bids=[(float(p), float(v)) for p, v in data.get('bids', [])],
                            asks=[(float(p), float(v)) for p, v in data.get('asks', [])]
                        )
                        
                        self._cache_result(cache_key, orderbook)
                        return orderbook
                    return None
        except Exception as e:
            self.logger.error(f"Error fetching orderbook for {symbol}: {e}")
            return None
    
    async def _fetch_recent_trades(self, symbol: str, limit: int = 100) -> Optional[List[Dict]]:
        """Fetch recent trades"""
        cache_key = f"trades_{symbol}_{limit}"
        
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        url = f"{self.base_url}/fapi/v1/trades"
        params = {'symbol': symbol, 'limit': limit}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        trades = [
                            {
                                'price': float(t['price']),
                                'volume': float(t['qty']),
                                'is_buyer_maker': t['isBuyerMaker'],
                                'timestamp': t['time']
                            }
                            for t in data
                        ]
                        
                        self._cache_result(cache_key, trades)
                        return trades
                    return None
        except Exception as e:
            self.logger.error(f"Error fetching trades for {symbol}: {e}")
            return None
    
    async def _fetch_funding_rate(self, symbol: str) -> Optional[float]:
        """Fetch current funding rate"""
        cache_key = f"funding_{symbol}"
        
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        url = f"{self.base_url}/fapi/v1/premiumIndex"
        params = {'symbol': symbol}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        funding_rate = float(data.get('lastFundingRate', 0))
                        self._cache_result(cache_key, funding_rate)
                        return funding_rate
                    return None
        except Exception as e:
            self.logger.error(f"Error fetching funding rate for {symbol}: {e}")
            return None
    
    async def _fetch_open_interest(self, symbol: str) -> Optional[float]:
        """Fetch current open interest"""
        cache_key = f"oi_{symbol}"
        
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        url = f"{self.base_url}/fapi/v1/openInterest"
        params = {'symbol': symbol}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        oi = float(data.get('openInterest', 0))
                        self._cache_result(cache_key, oi)
                        return oi
                    return None
        except Exception as e:
            self.logger.error(f"Error fetching open interest for {symbol}: {e}")
            return None
    
    def _is_cached(self, key: str) -> bool:
        """Check if key is in cache and not expired"""
        if key not in self.cache:
            return False
        
        timestamp = self.cache_timestamps.get(key)
        if not timestamp:
            return False
        
        age = datetime.now() - timestamp
        return age < self.cache_ttl
    
    def _cache_result(self, key: str, value: Any):
        """Cache a result"""
        self.cache[key] = value
        self.cache_timestamps[key] = datetime.now()
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        self.cache_timestamps.clear()
