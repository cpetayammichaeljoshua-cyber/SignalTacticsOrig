"""
Dynamic Futures Pairs Fetcher

Fetches ALL liquid USDT-M perpetual futures pairs from Binance public API.
No API key required - uses public endpoints.

API Documentation: https://binance-docs.github.io/apidocs/futures/en/
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class FuturesPair:
    """Binance USDT-M futures pair data"""
    symbol: str
    base_asset: str
    quote_asset: str
    volume_24h_usd: float
    price: float
    price_change_24h: float
    price_change_percent_24h: float
    high_24h: float
    low_24h: float
    trades_count_24h: int
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'base_asset': self.base_asset,
            'quote_asset': self.quote_asset,
            'volume_24h_usd': self.volume_24h_usd,
            'price': self.price,
            'price_change_24h': self.price_change_24h,
            'price_change_percent_24h': self.price_change_percent_24h,
            'high_24h': self.high_24h,
            'low_24h': self.low_24h,
            'trades_count_24h': self.trades_count_24h,
            'timestamp': self.timestamp.isoformat()
        }


class DynamicPairsFetcher:
    """
    Dynamic Futures Pairs Fetcher for Binance USDT-M Perpetuals
    
    Features:
    - Fetches ALL liquid USDT-M perpetual futures pairs
    - Filters by minimum 24h volume, trading status, and contract type
    - Sorts by 24h volume descending
    - Caches results for 1 hour
    - No API key required (public endpoints)
    
    Public API Endpoints Used:
    - GET /fapi/v1/exchangeInfo - Contract specifications
    - GET /fapi/v1/ticker/24hr - 24h price and volume data
    """
    
    BASE_URL = "https://fapi.binance.com"
    CACHE_TTL_SECONDS = 3600  # 1 hour cache
    MIN_VOLUME_USD = 10_000_000  # $10M minimum 24h volume
    
    def __init__(
        self,
        min_volume_usd: float = 10_000_000,
        cache_ttl_seconds: int = 3600
    ):
        """
        Initialize Dynamic Pairs Fetcher
        
        Args:
            min_volume_usd: Minimum 24h volume in USD (default $10M)
            cache_ttl_seconds: Cache TTL in seconds (default 1 hour)
        """
        self.min_volume_usd = min_volume_usd
        self.cache_ttl_seconds = cache_ttl_seconds
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._pairs_cache: List[FuturesPair] = []
        self._cache_time: Optional[datetime] = None
        self._exchange_info_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"DynamicPairsFetcher initialized (min_volume=${min_volume_usd:,.0f})")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self) -> None:
        """Close the aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        if not self._pairs_cache or self._cache_time is None:
            return False
        return (datetime.now() - self._cache_time).total_seconds() < self.cache_ttl_seconds
    
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Any]:
        """Make request to Binance API"""
        try:
            session = await self._get_session()
            url = f"{self.BASE_URL}{endpoint}"
            
            async with session.get(url, params=params) as response:
                if response.status == 429:
                    logger.warning("Binance rate limit exceeded")
                    return None
                
                if response.status != 200:
                    text = await response.text()
                    logger.warning(f"Binance API returned status {response.status}: {text}")
                    return None
                
                return await response.json()
                
        except asyncio.TimeoutError:
            logger.warning(f"Binance API timeout for {endpoint}")
            return None
        except aiohttp.ClientError as e:
            logger.warning(f"Binance API client error: {e}")
            return None
        except Exception as e:
            logger.error(f"Binance API unexpected error: {e}")
            return None
    
    async def _fetch_exchange_info(self) -> Dict[str, Dict[str, Any]]:
        """Fetch exchange info for all symbols"""
        data = await self._make_request("/fapi/v1/exchangeInfo")
        
        if not data or 'symbols' not in data:
            logger.warning("Failed to fetch exchange info")
            return self._exchange_info_cache
        
        result = {}
        for symbol_info in data['symbols']:
            symbol = symbol_info.get('symbol', '')
            if (symbol_info.get('contractType') == 'PERPETUAL' and
                symbol_info.get('status') == 'TRADING' and
                symbol_info.get('quoteAsset') == 'USDT'):
                result[symbol] = {
                    'symbol': symbol,
                    'base_asset': symbol_info.get('baseAsset', ''),
                    'quote_asset': symbol_info.get('quoteAsset', ''),
                    'contract_type': symbol_info.get('contractType', ''),
                    'status': symbol_info.get('status', '')
                }
        
        self._exchange_info_cache = result
        logger.info(f"Fetched exchange info for {len(result)} USDT-M perpetual symbols")
        return result
    
    async def _fetch_ticker_data(self) -> Dict[str, Dict[str, Any]]:
        """Fetch 24h ticker data for all symbols"""
        data = await self._make_request("/fapi/v1/ticker/24hr")
        
        if not data:
            logger.warning("Failed to fetch ticker data")
            return {}
        
        result = {}
        for ticker in data:
            symbol = ticker.get('symbol', '')
            result[symbol] = {
                'price': float(ticker.get('lastPrice', 0)),
                'volume_24h_usd': float(ticker.get('quoteVolume', 0)),
                'price_change_24h': float(ticker.get('priceChange', 0)),
                'price_change_percent_24h': float(ticker.get('priceChangePercent', 0)),
                'high_24h': float(ticker.get('highPrice', 0)),
                'low_24h': float(ticker.get('lowPrice', 0)),
                'trades_count_24h': int(ticker.get('count', 0))
            }
        
        logger.debug(f"Fetched ticker data for {len(result)} symbols")
        return result
    
    async def refresh_pairs(self) -> List[FuturesPair]:
        """
        Force refresh the pairs list from Binance API
        
        Returns:
            List of FuturesPair sorted by 24h volume descending
        """
        logger.info("Refreshing futures pairs from Binance API...")
        
        exchange_info, ticker_data = await asyncio.gather(
            self._fetch_exchange_info(),
            self._fetch_ticker_data()
        )
        
        if not exchange_info or not ticker_data:
            logger.warning("Failed to fetch data, returning cached pairs")
            return self._pairs_cache
        
        pairs = []
        for symbol, info in exchange_info.items():
            ticker = ticker_data.get(symbol)
            if not ticker:
                continue
            
            volume_24h_usd = ticker['volume_24h_usd']
            
            if volume_24h_usd < self.min_volume_usd:
                continue
            
            pair = FuturesPair(
                symbol=symbol,
                base_asset=info['base_asset'],
                quote_asset=info['quote_asset'],
                volume_24h_usd=volume_24h_usd,
                price=ticker['price'],
                price_change_24h=ticker['price_change_24h'],
                price_change_percent_24h=ticker['price_change_percent_24h'],
                high_24h=ticker['high_24h'],
                low_24h=ticker['low_24h'],
                trades_count_24h=ticker['trades_count_24h']
            )
            pairs.append(pair)
        
        pairs.sort(key=lambda p: p.volume_24h_usd, reverse=True)
        
        self._pairs_cache = pairs
        self._cache_time = datetime.now()
        
        logger.info(f"Fetched {len(pairs)} liquid USDT-M perpetual pairs (volume >= ${self.min_volume_usd:,.0f})")
        return pairs
    
    async def get_all_pairs(self, force_refresh: bool = False) -> List[FuturesPair]:
        """
        Get all liquid USDT-M perpetual futures pairs
        
        Args:
            force_refresh: Bypass cache and fetch fresh data
            
        Returns:
            List of FuturesPair sorted by 24h volume descending
        """
        if not force_refresh and self._is_cache_valid():
            logger.debug(f"Returning cached pairs ({len(self._pairs_cache)} pairs)")
            return self._pairs_cache
        
        return await self.refresh_pairs()
    
    async def get_top_pairs(self, n: int = 20, force_refresh: bool = False) -> List[FuturesPair]:
        """
        Get top N liquid USDT-M perpetual futures pairs by volume
        
        Args:
            n: Number of top pairs to return
            force_refresh: Bypass cache and fetch fresh data
            
        Returns:
            List of top N FuturesPair sorted by 24h volume descending
        """
        pairs = await self.get_all_pairs(force_refresh=force_refresh)
        return pairs[:n]
    
    async def get_pair_symbols(self, force_refresh: bool = False) -> List[str]:
        """
        Get just the symbol names of all liquid pairs
        
        Args:
            force_refresh: Bypass cache and fetch fresh data
            
        Returns:
            List of symbol strings (e.g., ['BTCUSDT', 'ETHUSDT', ...])
        """
        pairs = await self.get_all_pairs(force_refresh=force_refresh)
        return [p.symbol for p in pairs]
    
    async def get_top_pair_symbols(self, n: int = 20, force_refresh: bool = False) -> List[str]:
        """
        Get just the symbol names of top N pairs
        
        Args:
            n: Number of top pairs to return
            force_refresh: Bypass cache and fetch fresh data
            
        Returns:
            List of symbol strings
        """
        pairs = await self.get_top_pairs(n=n, force_refresh=force_refresh)
        return [p.symbol for p in pairs]
    
    def get_cache_age_seconds(self) -> Optional[float]:
        """Get the age of the cache in seconds, or None if no cache"""
        if self._cache_time is None:
            return None
        return (datetime.now() - self._cache_time).total_seconds()
    
    def get_cache_remaining_seconds(self) -> Optional[float]:
        """Get remaining cache TTL in seconds, or None if expired/no cache"""
        age = self.get_cache_age_seconds()
        if age is None:
            return None
        remaining = self.cache_ttl_seconds - age
        return remaining if remaining > 0 else None
    
    async def get_pair_by_symbol(self, symbol: str, force_refresh: bool = False) -> Optional[FuturesPair]:
        """
        Get a specific pair by symbol
        
        Args:
            symbol: Symbol to search for (e.g., 'BTCUSDT')
            force_refresh: Bypass cache and fetch fresh data
            
        Returns:
            FuturesPair if found, None otherwise
        """
        pairs = await self.get_all_pairs(force_refresh=force_refresh)
        symbol_upper = symbol.upper()
        for pair in pairs:
            if pair.symbol == symbol_upper:
                return pair
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current fetcher status"""
        return {
            'pairs_count': len(self._pairs_cache),
            'min_volume_usd': self.min_volume_usd,
            'cache_ttl_seconds': self.cache_ttl_seconds,
            'cache_age_seconds': self.get_cache_age_seconds(),
            'cache_remaining_seconds': self.get_cache_remaining_seconds(),
            'cache_valid': self._is_cache_valid(),
            'last_update': self._cache_time.isoformat() if self._cache_time else None
        }


async def create_pairs_fetcher(
    min_volume_usd: float = 10_000_000,
    cache_ttl_seconds: int = 3600,
    auto_fetch: bool = True
) -> DynamicPairsFetcher:
    """
    Factory function to create and optionally initialize a DynamicPairsFetcher
    
    Args:
        min_volume_usd: Minimum 24h volume in USD
        cache_ttl_seconds: Cache TTL in seconds
        auto_fetch: Whether to fetch pairs immediately
        
    Returns:
        Initialized DynamicPairsFetcher instance
    """
    fetcher = DynamicPairsFetcher(
        min_volume_usd=min_volume_usd,
        cache_ttl_seconds=cache_ttl_seconds
    )
    
    if auto_fetch:
        await fetcher.refresh_pairs()
    
    return fetcher
