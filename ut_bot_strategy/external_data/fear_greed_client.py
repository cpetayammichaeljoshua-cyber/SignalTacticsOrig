"""
Fear & Greed Index Client

Fetches Fear & Greed Index data from Alternative.me API.
Completely free, no API key required.

API Documentation: https://alternative.me/crypto/fear-and-greed-index/
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class FearGreedData:
    """Current Fear & Greed Index data"""
    value: int
    value_classification: str
    timestamp: datetime
    time_until_update: Optional[int] = None
    
    @property
    def is_extreme_fear(self) -> bool:
        return self.value <= 25
    
    @property
    def is_fear(self) -> bool:
        return 25 < self.value <= 45
    
    @property
    def is_neutral(self) -> bool:
        return 45 < self.value <= 55
    
    @property
    def is_greed(self) -> bool:
        return 55 < self.value <= 75
    
    @property
    def is_extreme_greed(self) -> bool:
        return self.value > 75


@dataclass
class FearGreedHistoryEntry:
    """Historical Fear & Greed Index entry"""
    value: int
    value_classification: str
    timestamp: datetime


class FearGreedClient:
    """
    Client for Alternative.me Fear & Greed Index API
    
    Features:
    - Fetch current fear/greed value (0-100)
    - Historical data support
    - Caching (5 minute refresh)
    - Graceful degradation if API fails
    """
    
    BASE_URL = "https://api.alternative.me/fng/"
    CACHE_TTL_SECONDS = 300  # 5 minutes
    
    def __init__(self):
        self._cache: Optional[FearGreedData] = None
        self._cache_time: Optional[datetime] = None
        self._history_cache: Optional[List[FearGreedHistoryEntry]] = None
        self._history_cache_time: Optional[datetime] = None
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=10)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self) -> None:
        """Close the aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    def _is_cache_valid(self) -> bool:
        """Check if current cache is still valid"""
        if self._cache is None or self._cache_time is None:
            return False
        return (datetime.now() - self._cache_time).total_seconds() < self.CACHE_TTL_SECONDS
    
    def _is_history_cache_valid(self) -> bool:
        """Check if history cache is still valid"""
        if self._history_cache is None or self._history_cache_time is None:
            return False
        return (datetime.now() - self._history_cache_time).total_seconds() < self.CACHE_TTL_SECONDS
    
    async def get_current(self, force_refresh: bool = False) -> Optional[FearGreedData]:
        """
        Get current Fear & Greed Index value
        
        Args:
            force_refresh: Bypass cache and fetch fresh data
            
        Returns:
            FearGreedData or None if API unavailable
        """
        if not force_refresh and self._is_cache_valid():
            logger.debug("Returning cached Fear & Greed data")
            return self._cache
        
        try:
            session = await self._get_session()
            async with session.get(self.BASE_URL) as response:
                if response.status != 200:
                    logger.warning(f"Fear & Greed API returned status {response.status}")
                    return self._cache  # Return stale cache if available
                
                data = await response.json()
                
                if data.get('metadata', {}).get('error'):
                    logger.warning(f"Fear & Greed API error: {data['metadata']['error']}")
                    return self._cache
                
                if not data.get('data'):
                    logger.warning("Fear & Greed API returned empty data")
                    return self._cache
                
                entry = data['data'][0]
                self._cache = FearGreedData(
                    value=int(entry['value']),
                    value_classification=entry['value_classification'],
                    timestamp=datetime.fromtimestamp(int(entry['timestamp'])),
                    time_until_update=int(entry.get('time_until_update', 0)) if entry.get('time_until_update') else None
                )
                self._cache_time = datetime.now()
                
                logger.info(f"Fear & Greed Index: {self._cache.value} ({self._cache.value_classification})")
                return self._cache
                
        except asyncio.TimeoutError:
            logger.warning("Fear & Greed API timeout")
            return self._cache
        except aiohttp.ClientError as e:
            logger.warning(f"Fear & Greed API client error: {e}")
            return self._cache
        except Exception as e:
            logger.error(f"Fear & Greed API unexpected error: {e}")
            return self._cache
    
    async def get_history(self, limit: int = 30, force_refresh: bool = False) -> Optional[List[FearGreedHistoryEntry]]:
        """
        Get historical Fear & Greed Index data
        
        Args:
            limit: Number of days to fetch (max 365)
            force_refresh: Bypass cache and fetch fresh data
            
        Returns:
            List of FearGreedHistoryEntry or None if API unavailable
        """
        if not force_refresh and self._is_history_cache_valid():
            logger.debug("Returning cached Fear & Greed history")
            return self._history_cache[:limit] if self._history_cache else None
        
        try:
            session = await self._get_session()
            url = f"{self.BASE_URL}?limit={min(limit, 365)}"
            
            async with session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Fear & Greed history API returned status {response.status}")
                    return self._history_cache
                
                data = await response.json()
                
                if data.get('metadata', {}).get('error'):
                    logger.warning(f"Fear & Greed history API error: {data['metadata']['error']}")
                    return self._history_cache
                
                if not data.get('data'):
                    logger.warning("Fear & Greed history API returned empty data")
                    return self._history_cache
                
                self._history_cache = [
                    FearGreedHistoryEntry(
                        value=int(entry['value']),
                        value_classification=entry['value_classification'],
                        timestamp=datetime.fromtimestamp(int(entry['timestamp']))
                    )
                    for entry in data['data']
                ]
                self._history_cache_time = datetime.now()
                
                logger.info(f"Fetched {len(self._history_cache)} Fear & Greed history entries")
                return self._history_cache
                
        except asyncio.TimeoutError:
            logger.warning("Fear & Greed history API timeout")
            return self._history_cache
        except aiohttp.ClientError as e:
            logger.warning(f"Fear & Greed history API client error: {e}")
            return self._history_cache
        except Exception as e:
            logger.error(f"Fear & Greed history API unexpected error: {e}")
            return self._history_cache
    
    async def get_average(self, days: int = 7) -> Optional[float]:
        """
        Get average Fear & Greed value over specified days
        
        Args:
            days: Number of days to average
            
        Returns:
            Average value or None if data unavailable
        """
        history = await self.get_history(limit=days)
        if not history:
            return None
        
        values = [entry.value for entry in history[:days]]
        if not values:
            return None
        
        return sum(values) / len(values)
    
    async def get_trend(self, days: int = 7) -> Optional[str]:
        """
        Determine trend direction over specified days
        
        Args:
            days: Number of days to analyze
            
        Returns:
            'improving', 'declining', or 'stable'
        """
        history = await self.get_history(limit=days)
        if not history or len(history) < 2:
            return None
        
        recent = history[0].value
        older = history[-1].value if len(history) >= days else history[-1].value
        
        diff = recent - older
        if diff > 5:
            return 'improving'
        elif diff < -5:
            return 'declining'
        else:
            return 'stable'
    
    def get_cached_value(self) -> Optional[int]:
        """Get cached value without making API call"""
        return self._cache.value if self._cache else None
    
    def get_cached_classification(self) -> Optional[str]:
        """Get cached classification without making API call"""
        return self._cache.value_classification if self._cache else None
