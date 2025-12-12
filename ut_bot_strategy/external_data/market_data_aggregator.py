"""
Market Data Aggregator - CoinGecko Integration

Fetches market data from CoinGecko API.
Free tier: 30 calls/minute without API key.
Pro tier: Higher limits with API key.

API Documentation: https://www.coingecko.com/en/api/documentation
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class TrendingCoin:
    """Trending coin data"""
    id: str
    symbol: str
    name: str
    market_cap_rank: Optional[int]
    thumb: str
    score: int
    price_btc: Optional[float] = None


@dataclass
class CoinMarketData:
    """Market data for a specific coin"""
    id: str
    symbol: str
    name: str
    current_price: float
    market_cap: float
    market_cap_rank: int
    total_volume: float
    high_24h: float
    low_24h: float
    price_change_24h: float
    price_change_percentage_24h: float
    price_change_percentage_7d: Optional[float] = None
    price_change_percentage_30d: Optional[float] = None
    circulating_supply: Optional[float] = None
    total_supply: Optional[float] = None
    ath: Optional[float] = None
    ath_change_percentage: Optional[float] = None
    last_updated: Optional[datetime] = None


@dataclass
class GlobalMarketData:
    """Global cryptocurrency market data"""
    total_market_cap_usd: float
    total_volume_24h_usd: float
    btc_dominance: float
    eth_dominance: float
    active_cryptocurrencies: int
    markets: int
    market_cap_change_percentage_24h: float
    updated_at: datetime


class MarketDataAggregator:
    """
    CoinGecko Market Data Aggregator
    
    Features:
    - Get trending coins
    - Get market stats for top cryptos
    - Get global market cap data
    - Rate limiting (30 calls/min free tier)
    - Caching with 2 minute expiry
    - Optional API key support via COINGECKO_API_KEY env var
    """
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    PRO_BASE_URL = "https://pro-api.coingecko.com/api/v3"
    CACHE_TTL_SECONDS = 120  # 2 minutes
    RATE_LIMIT_CALLS = 30
    RATE_LIMIT_WINDOW = 60  # seconds
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('COINGECKO_API_KEY', '')
        self._session: Optional[aiohttp.ClientSession] = None
        
        self._trending_cache: Optional[List[TrendingCoin]] = None
        self._trending_cache_time: Optional[datetime] = None
        
        self._market_cache: Dict[str, CoinMarketData] = {}
        self._market_cache_time: Optional[datetime] = None
        
        self._global_cache: Optional[GlobalMarketData] = None
        self._global_cache_time: Optional[datetime] = None
        
        self._call_times: List[datetime] = []
    
    @property
    def base_url(self) -> str:
        """Get appropriate base URL based on API key presence"""
        return self.PRO_BASE_URL if self.api_key else self.BASE_URL
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=15)
            headers = {}
            if self.api_key:
                headers['x-cg-pro-api-key'] = self.api_key
            self._session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self._session
    
    async def close(self) -> None:
        """Close the aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    async def _rate_limit_check(self) -> None:
        """Check and wait for rate limiting if necessary"""
        now = datetime.now()
        window_start = now - timedelta(seconds=self.RATE_LIMIT_WINDOW)
        self._call_times = [t for t in self._call_times if t > window_start]
        
        if len(self._call_times) >= self.RATE_LIMIT_CALLS:
            oldest_call = min(self._call_times)
            wait_time = (oldest_call + timedelta(seconds=self.RATE_LIMIT_WINDOW) - now).total_seconds()
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time + 0.1)
        
        self._call_times.append(datetime.now())
    
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make rate-limited request to CoinGecko API"""
        await self._rate_limit_check()
        
        try:
            session = await self._get_session()
            url = f"{self.base_url}{endpoint}"
            
            async with session.get(url, params=params) as response:
                if response.status == 429:
                    logger.warning("CoinGecko rate limit exceeded")
                    retry_after = int(response.headers.get('Retry-After', 60))
                    await asyncio.sleep(retry_after)
                    return await self._make_request(endpoint, params)
                
                if response.status != 200:
                    logger.warning(f"CoinGecko API returned status {response.status}")
                    return None
                
                return await response.json()
                
        except asyncio.TimeoutError:
            logger.warning("CoinGecko API timeout")
            return None
        except aiohttp.ClientError as e:
            logger.warning(f"CoinGecko API client error: {e}")
            return None
        except Exception as e:
            logger.error(f"CoinGecko API unexpected error: {e}")
            return None
    
    def _is_cache_valid(self, cache_time: Optional[datetime]) -> bool:
        """Check if cache is still valid"""
        if cache_time is None:
            return False
        return (datetime.now() - cache_time).total_seconds() < self.CACHE_TTL_SECONDS
    
    async def get_trending_coins(self, force_refresh: bool = False) -> Optional[List[TrendingCoin]]:
        """
        Get trending coins on CoinGecko
        
        Args:
            force_refresh: Bypass cache and fetch fresh data
            
        Returns:
            List of TrendingCoin or None if API unavailable
        """
        if not force_refresh and self._is_cache_valid(self._trending_cache_time):
            logger.debug("Returning cached trending coins")
            return self._trending_cache
        
        data = await self._make_request("/search/trending")
        if not data:
            return self._trending_cache
        
        try:
            coins = data.get('coins', [])
            self._trending_cache = [
                TrendingCoin(
                    id=coin['item']['id'],
                    symbol=coin['item']['symbol'],
                    name=coin['item']['name'],
                    market_cap_rank=coin['item'].get('market_cap_rank'),
                    thumb=coin['item'].get('thumb', ''),
                    score=coin['item'].get('score', 0),
                    price_btc=coin['item'].get('price_btc')
                )
                for coin in coins
            ]
            self._trending_cache_time = datetime.now()
            
            logger.info(f"Fetched {len(self._trending_cache)} trending coins")
            return self._trending_cache
            
        except (KeyError, TypeError) as e:
            logger.error(f"Error parsing trending coins: {e}")
            return self._trending_cache
    
    async def get_market_data(
        self, 
        coin_ids: Optional[List[str]] = None,
        vs_currency: str = 'usd',
        per_page: int = 100,
        page: int = 1,
        force_refresh: bool = False
    ) -> Optional[List[CoinMarketData]]:
        """
        Get market data for coins
        
        Args:
            coin_ids: Specific coin IDs to fetch (None for top coins)
            vs_currency: Currency for prices (default 'usd')
            per_page: Number of results per page (max 250)
            page: Page number
            force_refresh: Bypass cache and fetch fresh data
            
        Returns:
            List of CoinMarketData or None if API unavailable
        """
        if not force_refresh and self._is_cache_valid(self._market_cache_time):
            if coin_ids:
                cached: List[CoinMarketData] = [
                    self._market_cache[cid] for cid in coin_ids 
                    if cid in self._market_cache and self._market_cache[cid] is not None
                ]
                if len(cached) == len(coin_ids):
                    logger.debug("Returning cached market data")
                    return cached
            elif self._market_cache:
                logger.debug("Returning cached market data")
                return list(self._market_cache.values())[:per_page]
        
        params = {
            'vs_currency': vs_currency,
            'order': 'market_cap_desc',
            'per_page': min(per_page, 250),
            'page': page,
            'sparkline': 'false',
            'price_change_percentage': '7d,30d'
        }
        
        if coin_ids:
            params['ids'] = ','.join(coin_ids)
        
        data = await self._make_request("/coins/markets", params)
        if not data:
            if coin_ids:
                cached_fallback: List[CoinMarketData] = [
                    self._market_cache[cid] for cid in coin_ids 
                    if cid in self._market_cache and self._market_cache[cid] is not None
                ]
                return cached_fallback if cached_fallback else None
            return list(self._market_cache.values()) if self._market_cache else None
        
        try:
            result = []
            for coin in data:
                coin_data = CoinMarketData(
                    id=coin['id'],
                    symbol=coin['symbol'],
                    name=coin['name'],
                    current_price=float(coin.get('current_price', 0) or 0),
                    market_cap=float(coin.get('market_cap', 0) or 0),
                    market_cap_rank=int(coin.get('market_cap_rank', 0) or 0),
                    total_volume=float(coin.get('total_volume', 0) or 0),
                    high_24h=float(coin.get('high_24h', 0) or 0),
                    low_24h=float(coin.get('low_24h', 0) or 0),
                    price_change_24h=float(coin.get('price_change_24h', 0) or 0),
                    price_change_percentage_24h=float(coin.get('price_change_percentage_24h', 0) or 0),
                    price_change_percentage_7d=coin.get('price_change_percentage_7d_in_currency'),
                    price_change_percentage_30d=coin.get('price_change_percentage_30d_in_currency'),
                    circulating_supply=coin.get('circulating_supply'),
                    total_supply=coin.get('total_supply'),
                    ath=coin.get('ath'),
                    ath_change_percentage=coin.get('ath_change_percentage'),
                    last_updated=datetime.fromisoformat(coin['last_updated'].replace('Z', '+00:00')) if coin.get('last_updated') else None
                )
                result.append(coin_data)
                self._market_cache[coin['id']] = coin_data
            
            self._market_cache_time = datetime.now()
            logger.info(f"Fetched market data for {len(result)} coins")
            return result
            
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Error parsing market data: {e}")
            return list(self._market_cache.values()) if self._market_cache else None
    
    async def get_global_market_data(self, force_refresh: bool = False) -> Optional[GlobalMarketData]:
        """
        Get global cryptocurrency market data
        
        Args:
            force_refresh: Bypass cache and fetch fresh data
            
        Returns:
            GlobalMarketData or None if API unavailable
        """
        if not force_refresh and self._is_cache_valid(self._global_cache_time):
            logger.debug("Returning cached global market data")
            return self._global_cache
        
        data = await self._make_request("/global")
        if not data or 'data' not in data:
            return self._global_cache
        
        try:
            global_data = data['data']
            self._global_cache = GlobalMarketData(
                total_market_cap_usd=float(global_data.get('total_market_cap', {}).get('usd', 0)),
                total_volume_24h_usd=float(global_data.get('total_volume', {}).get('usd', 0)),
                btc_dominance=float(global_data.get('market_cap_percentage', {}).get('btc', 0)),
                eth_dominance=float(global_data.get('market_cap_percentage', {}).get('eth', 0)),
                active_cryptocurrencies=int(global_data.get('active_cryptocurrencies', 0)),
                markets=int(global_data.get('markets', 0)),
                market_cap_change_percentage_24h=float(global_data.get('market_cap_change_percentage_24h_usd', 0)),
                updated_at=datetime.fromtimestamp(global_data.get('updated_at', 0))
            )
            self._global_cache_time = datetime.now()
            
            logger.info(f"Global market cap: ${self._global_cache.total_market_cap_usd:,.0f}")
            return self._global_cache
            
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Error parsing global market data: {e}")
            return self._global_cache
    
    async def get_eth_market_data(self, force_refresh: bool = False) -> Optional[CoinMarketData]:
        """Get market data specifically for Ethereum"""
        result = await self.get_market_data(coin_ids=['ethereum'], force_refresh=force_refresh)
        return result[0] if result else None
    
    async def get_btc_market_data(self, force_refresh: bool = False) -> Optional[CoinMarketData]:
        """Get market data specifically for Bitcoin"""
        result = await self.get_market_data(coin_ids=['bitcoin'], force_refresh=force_refresh)
        return result[0] if result else None
    
    async def get_top_coins(self, limit: int = 10, force_refresh: bool = False) -> Optional[List[CoinMarketData]]:
        """Get top coins by market cap"""
        result = await self.get_market_data(per_page=limit, force_refresh=force_refresh)
        return result[:limit] if result else None
    
    def get_cached_coin(self, coin_id: str) -> Optional[CoinMarketData]:
        """Get cached coin data without making API call"""
        return self._market_cache.get(coin_id)
    
    def get_cached_global(self) -> Optional[GlobalMarketData]:
        """Get cached global data without making API call"""
        return self._global_cache
