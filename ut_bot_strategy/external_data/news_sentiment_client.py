"""
News Sentiment Client - CryptoPanic Integration

Fetches crypto news with sentiment analysis from CryptoPanic API.
API key optional but recommended for better rate limits.

API Documentation: https://cryptopanic.com/developers/api/
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Literal
from enum import Enum
import aiohttp

logger = logging.getLogger(__name__)


class NewsFilter(str, Enum):
    """News filter types"""
    HOT = 'hot'
    RISING = 'rising'
    BULLISH = 'bullish'
    BEARISH = 'bearish'
    IMPORTANT = 'important'
    SAVED = 'saved'
    LOL = 'lol'


class NewsKind(str, Enum):
    """News content types"""
    NEWS = 'news'
    MEDIA = 'media'


@dataclass
class NewsItem:
    """Individual news item"""
    id: int
    title: str
    url: str
    source: str
    published_at: datetime
    currencies: List[str]
    kind: str
    domain: str
    votes: Dict[str, int] = field(default_factory=dict)
    
    @property
    def is_bullish(self) -> bool:
        positive = self.votes.get('positive', 0)
        negative = self.votes.get('negative', 0)
        return positive > negative
    
    @property
    def is_bearish(self) -> bool:
        positive = self.votes.get('positive', 0)
        negative = self.votes.get('negative', 0)
        return negative > positive
    
    @property
    def sentiment_score(self) -> float:
        """Calculate sentiment score from -1 (bearish) to 1 (bullish)"""
        positive = self.votes.get('positive', 0)
        negative = self.votes.get('negative', 0)
        total = positive + negative
        if total == 0:
            return 0.0
        return (positive - negative) / total
    
    @property
    def is_important(self) -> bool:
        return self.votes.get('important', 0) > 0


@dataclass
class NewsSentimentSummary:
    """Summary of news sentiment analysis"""
    total_news: int
    bullish_count: int
    bearish_count: int
    neutral_count: int
    average_sentiment: float
    most_mentioned_currencies: List[str]
    latest_news_time: Optional[datetime]
    
    @property
    def sentiment_label(self) -> str:
        if self.average_sentiment > 0.2:
            return 'bullish'
        elif self.average_sentiment < -0.2:
            return 'bearish'
        return 'neutral'


class NewsSentimentClient:
    """
    CryptoPanic News Sentiment Client
    
    Features:
    - Fetch crypto news with sentiment labels
    - Filter by currency (ETH, BTC)
    - Filter by type (bullish, bearish, hot, important)
    - Caching with 3 minute expiry
    - API key via CRYPTOPANIC_API_KEY env var (optional, graceful degradation)
    """
    
    BASE_URL = "https://cryptopanic.com/api/v1"
    CACHE_TTL_SECONDS = 180  # 3 minutes
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('CRYPTOPANIC_API_KEY', '')
        self._session: Optional[aiohttp.ClientSession] = None
        
        self._news_cache: Dict[str, List[NewsItem]] = {}
        self._news_cache_time: Dict[str, datetime] = {}
        
        self._summary_cache: Optional[NewsSentimentSummary] = None
        self._summary_cache_time: Optional[datetime] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=15)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self) -> None:
        """Close the aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    def _get_cache_key(
        self, 
        currencies: Optional[List[str]] = None,
        filter_type: Optional[str] = None,
        kind: Optional[str] = None
    ) -> str:
        """Generate cache key from parameters"""
        parts = []
        if currencies:
            parts.append(f"cur:{','.join(sorted(currencies))}")
        if filter_type:
            parts.append(f"filter:{filter_type}")
        if kind:
            parts.append(f"kind:{kind}")
        return '|'.join(parts) if parts else 'default'
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache is still valid for given key"""
        cache_time = self._news_cache_time.get(cache_key)
        if cache_time is None:
            return False
        return (datetime.now() - cache_time).total_seconds() < self.CACHE_TTL_SECONDS
    
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make request to CryptoPanic API"""
        if not self.api_key:
            logger.warning("CryptoPanic API key not set, using limited public access")
        
        try:
            session = await self._get_session()
            url = f"{self.BASE_URL}{endpoint}"
            
            if params is None:
                params = {}
            
            if self.api_key:
                params['auth_token'] = self.api_key
            
            params['public'] = 'true'
            
            async with session.get(url, params=params) as response:
                if response.status == 401:
                    logger.warning("CryptoPanic API authentication failed")
                    return None
                
                if response.status == 429:
                    logger.warning("CryptoPanic rate limit exceeded")
                    return None
                
                if response.status != 200:
                    logger.warning(f"CryptoPanic API returned status {response.status}")
                    return None
                
                return await response.json()
                
        except asyncio.TimeoutError:
            logger.warning("CryptoPanic API timeout")
            return None
        except aiohttp.ClientError as e:
            logger.warning(f"CryptoPanic API client error: {e}")
            return None
        except Exception as e:
            logger.error(f"CryptoPanic API unexpected error: {e}")
            return None
    
    async def get_news(
        self,
        currencies: Optional[List[str]] = None,
        filter_type: Optional[str] = None,
        kind: Optional[str] = None,
        limit: int = 50,
        force_refresh: bool = False
    ) -> Optional[List[NewsItem]]:
        """
        Fetch crypto news
        
        Args:
            currencies: Filter by currencies (e.g., ['ETH', 'BTC'])
            filter_type: Filter type ('hot', 'rising', 'bullish', 'bearish', 'important')
            kind: Content kind ('news', 'media')
            limit: Maximum number of news items
            force_refresh: Bypass cache
            
        Returns:
            List of NewsItem or None if API unavailable
        """
        cache_key = self._get_cache_key(currencies, filter_type, kind)
        
        if not force_refresh and self._is_cache_valid(cache_key):
            logger.debug(f"Returning cached news for {cache_key}")
            cached = self._news_cache.get(cache_key, [])
            return cached[:limit]
        
        params = {}
        
        if currencies:
            params['currencies'] = ','.join(currencies)
        
        if filter_type:
            params['filter'] = filter_type
        
        if kind:
            params['kind'] = kind
        
        data = await self._make_request("/posts/", params)
        
        if not data:
            cached = self._news_cache.get(cache_key)
            return cached[:limit] if cached else None
        
        try:
            results = data.get('results', [])
            news_items = []
            
            for item in results:
                currencies_list = [c['code'] for c in item.get('currencies', [])]
                
                votes = {}
                if item.get('votes'):
                    votes = {
                        'positive': item['votes'].get('positive', 0),
                        'negative': item['votes'].get('negative', 0),
                        'important': item['votes'].get('important', 0),
                        'liked': item['votes'].get('liked', 0),
                        'disliked': item['votes'].get('disliked', 0),
                        'lol': item['votes'].get('lol', 0),
                        'toxic': item['votes'].get('toxic', 0),
                        'saved': item['votes'].get('saved', 0),
                        'comments': item['votes'].get('comments', 0)
                    }
                
                published_at = datetime.fromisoformat(
                    item['published_at'].replace('Z', '+00:00')
                ) if item.get('published_at') else datetime.now()
                
                news_item = NewsItem(
                    id=item.get('id', 0),
                    title=item.get('title', ''),
                    url=item.get('url', ''),
                    source=item.get('source', {}).get('title', 'Unknown'),
                    published_at=published_at,
                    currencies=currencies_list,
                    kind=item.get('kind', 'news'),
                    domain=item.get('domain', ''),
                    votes=votes
                )
                news_items.append(news_item)
            
            self._news_cache[cache_key] = news_items
            self._news_cache_time[cache_key] = datetime.now()
            
            logger.info(f"Fetched {len(news_items)} news items")
            return news_items[:limit]
            
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Error parsing news data: {e}")
            cached = self._news_cache.get(cache_key)
            return cached[:limit] if cached else None
    
    async def get_eth_news(self, limit: int = 20, force_refresh: bool = False) -> Optional[List[NewsItem]]:
        """Get Ethereum-specific news"""
        return await self.get_news(currencies=['ETH'], limit=limit, force_refresh=force_refresh)
    
    async def get_btc_news(self, limit: int = 20, force_refresh: bool = False) -> Optional[List[NewsItem]]:
        """Get Bitcoin-specific news"""
        return await self.get_news(currencies=['BTC'], limit=limit, force_refresh=force_refresh)
    
    async def get_bullish_news(
        self, 
        currencies: Optional[List[str]] = None,
        limit: int = 20,
        force_refresh: bool = False
    ) -> Optional[List[NewsItem]]:
        """Get bullish news"""
        return await self.get_news(
            currencies=currencies, 
            filter_type='bullish', 
            limit=limit, 
            force_refresh=force_refresh
        )
    
    async def get_bearish_news(
        self,
        currencies: Optional[List[str]] = None,
        limit: int = 20,
        force_refresh: bool = False
    ) -> Optional[List[NewsItem]]:
        """Get bearish news"""
        return await self.get_news(
            currencies=currencies,
            filter_type='bearish',
            limit=limit,
            force_refresh=force_refresh
        )
    
    async def get_hot_news(
        self,
        currencies: Optional[List[str]] = None,
        limit: int = 20,
        force_refresh: bool = False
    ) -> Optional[List[NewsItem]]:
        """Get trending/hot news"""
        return await self.get_news(
            currencies=currencies,
            filter_type='hot',
            limit=limit,
            force_refresh=force_refresh
        )
    
    async def get_important_news(
        self,
        currencies: Optional[List[str]] = None,
        limit: int = 20,
        force_refresh: bool = False
    ) -> Optional[List[NewsItem]]:
        """Get important news"""
        return await self.get_news(
            currencies=currencies,
            filter_type='important',
            limit=limit,
            force_refresh=force_refresh
        )
    
    async def get_sentiment_summary(
        self,
        currencies: Optional[List[str]] = None,
        force_refresh: bool = False
    ) -> Optional[NewsSentimentSummary]:
        """
        Get sentiment summary for recent news
        
        Args:
            currencies: Filter by currencies
            force_refresh: Bypass cache
            
        Returns:
            NewsSentimentSummary or None
        """
        cache_key = self._get_cache_key(currencies)
        summary_cache_key = f"summary:{cache_key}"
        
        if not force_refresh and self._summary_cache and self._summary_cache_time:
            if (datetime.now() - self._summary_cache_time).total_seconds() < self.CACHE_TTL_SECONDS:
                return self._summary_cache
        
        news = await self.get_news(currencies=currencies, limit=100, force_refresh=force_refresh)
        if not news:
            return None
        
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        sentiment_scores = []
        currency_counts: Dict[str, int] = {}
        latest_time: Optional[datetime] = None
        
        for item in news:
            score = item.sentiment_score
            sentiment_scores.append(score)
            
            if score > 0.1:
                bullish_count += 1
            elif score < -0.1:
                bearish_count += 1
            else:
                neutral_count += 1
            
            for currency in item.currencies:
                currency_counts[currency] = currency_counts.get(currency, 0) + 1
            
            if latest_time is None or item.published_at > latest_time:
                latest_time = item.published_at
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
        
        sorted_currencies = sorted(currency_counts.items(), key=lambda x: x[1], reverse=True)
        top_currencies = [c[0] for c in sorted_currencies[:5]]
        
        self._summary_cache = NewsSentimentSummary(
            total_news=len(news),
            bullish_count=bullish_count,
            bearish_count=bearish_count,
            neutral_count=neutral_count,
            average_sentiment=avg_sentiment,
            most_mentioned_currencies=top_currencies,
            latest_news_time=latest_time
        )
        self._summary_cache_time = datetime.now()
        
        logger.info(f"Sentiment summary: {self._summary_cache.sentiment_label} (avg: {avg_sentiment:.2f})")
        return self._summary_cache
    
    def get_cached_news(self, currencies: Optional[List[str]] = None) -> Optional[List[NewsItem]]:
        """Get cached news without making API call"""
        cache_key = self._get_cache_key(currencies)
        return self._news_cache.get(cache_key)
    
    def get_cached_summary(self) -> Optional[NewsSentimentSummary]:
        """Get cached summary without making API call"""
        return self._summary_cache
