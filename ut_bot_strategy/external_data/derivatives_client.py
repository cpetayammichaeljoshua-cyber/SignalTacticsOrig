"""
Binance Futures Derivatives Data Client

Fetches FREE market structure data from Binance Futures public APIs.
No API key required for these endpoints.

API Documentation: https://binance-docs.github.io/apidocs/futures/en/
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class DerivativesData:
    """Aggregated derivatives market data"""
    funding_rate: float  # Current funding rate
    funding_rate_8h_avg: float  # 8-hour average
    funding_rate_trend: str  # "rising", "falling", "stable"
    open_interest: float  # Current OI in contracts
    open_interest_value: float  # OI in USD
    oi_change_24h: float  # Percentage change
    long_short_ratio: float  # Global ratio
    top_trader_long_ratio: float  # Top traders long %
    taker_buy_ratio: float  # Taker buy volume ratio
    market_sentiment: str  # "extremely_bullish", "bullish", "neutral", "bearish", "extremely_bearish"
    derivatives_score: float  # -1 to +1 composite score


@dataclass
class FundingRateEntry:
    """Individual funding rate entry"""
    symbol: str
    funding_rate: float
    funding_time: datetime
    mark_price: Optional[float] = None


@dataclass
class OpenInterestData:
    """Open interest data"""
    symbol: str
    open_interest: float
    open_interest_value: float
    timestamp: datetime


@dataclass
class LongShortRatioData:
    """Long/short ratio data"""
    symbol: str
    long_short_ratio: float
    long_account: float
    short_account: float
    timestamp: datetime


@dataclass
class TakerVolumeData:
    """Taker buy/sell volume data"""
    symbol: str
    buy_sell_ratio: float
    buy_volume: float
    sell_volume: float
    timestamp: datetime


class BinanceDerivativesClient:
    """
    Client for Binance Futures FREE public API endpoints
    
    Features:
    - Funding rate data (current and historical)
    - Open interest data
    - Long/short ratio (global and top traders)
    - Taker buy/sell volume ratio
    - Caching with configurable TTL
    - Graceful error handling
    """
    
    BASE_URL = "https://fapi.binance.com"
    CACHE_TTL_FUNDING = 60  # 1 minute for funding rates
    CACHE_TTL_DEFAULT = 300  # 5 minutes for other data
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Caches with timestamps
        self._funding_cache: Dict[str, tuple[List[FundingRateEntry], datetime]] = {}
        self._oi_cache: Dict[str, tuple[OpenInterestData, datetime]] = {}
        self._oi_hist_cache: Dict[str, tuple[List[OpenInterestData], datetime]] = {}
        self._ls_ratio_cache: Dict[str, tuple[LongShortRatioData, datetime]] = {}
        self._top_ls_ratio_cache: Dict[str, tuple[LongShortRatioData, datetime]] = {}
        self._top_pos_ratio_cache: Dict[str, tuple[LongShortRatioData, datetime]] = {}
        self._taker_volume_cache: Dict[str, tuple[List[TakerVolumeData], datetime]] = {}
    
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
    
    def _is_cache_valid(self, cache_time: Optional[datetime], ttl: int) -> bool:
        """Check if cache is still valid"""
        if cache_time is None:
            return False
        return (datetime.now() - cache_time).total_seconds() < ttl
    
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
    
    async def get_funding_rate(
        self, 
        symbol: str, 
        limit: int = 100,
        force_refresh: bool = False
    ) -> Optional[List[FundingRateEntry]]:
        """
        Get historical funding rates for a symbol
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            limit: Number of entries to fetch (max 1000)
            force_refresh: Bypass cache
            
        Returns:
            List of FundingRateEntry or None if unavailable
        """
        cache_key = f"{symbol}_{limit}"
        cached = self._funding_cache.get(cache_key)
        
        if not force_refresh and cached:
            data, cache_time = cached
            if self._is_cache_valid(cache_time, self.CACHE_TTL_FUNDING):
                logger.debug(f"Returning cached funding rate for {symbol}")
                return data
        
        params = {'symbol': symbol.upper(), 'limit': min(limit, 1000)}
        data = await self._make_request("/fapi/v1/fundingRate", params)
        
        if not data:
            return cached[0] if cached else None
        
        try:
            result = [
                FundingRateEntry(
                    symbol=entry['symbol'],
                    funding_rate=float(entry['fundingRate']),
                    funding_time=datetime.fromtimestamp(int(entry['fundingTime']) / 1000),
                    mark_price=float(entry.get('markPrice', 0)) if entry.get('markPrice') else None
                )
                for entry in data
            ]
            
            self._funding_cache[cache_key] = (result, datetime.now())
            logger.info(f"Fetched {len(result)} funding rate entries for {symbol}")
            return result
            
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Error parsing funding rate data: {e}")
            return cached[0] if cached else None
    
    async def get_current_funding_rate(self, symbol: str) -> Optional[float]:
        """Get the most recent funding rate for a symbol"""
        rates = await self.get_funding_rate(symbol, limit=1)
        return rates[0].funding_rate if rates else None
    
    async def get_premium_index(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get premium index and funding rate info
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dict with premium index data or None
        """
        params = {'symbol': symbol.upper()}
        data = await self._make_request("/fapi/v1/premiumIndex", params)
        
        if not data:
            return None
        
        try:
            return {
                'symbol': data['symbol'],
                'mark_price': float(data['markPrice']),
                'index_price': float(data['indexPrice']),
                'estimated_settle_price': float(data.get('estimatedSettlePrice', 0)),
                'last_funding_rate': float(data['lastFundingRate']),
                'next_funding_time': datetime.fromtimestamp(int(data['nextFundingTime']) / 1000),
                'interest_rate': float(data.get('interestRate', 0))
            }
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Error parsing premium index: {e}")
            return None
    
    async def get_open_interest(
        self, 
        symbol: str,
        force_refresh: bool = False
    ) -> Optional[OpenInterestData]:
        """
        Get current open interest for a symbol
        
        Args:
            symbol: Trading pair symbol
            force_refresh: Bypass cache
            
        Returns:
            OpenInterestData or None
        """
        cached = self._oi_cache.get(symbol)
        
        if not force_refresh and cached:
            data, cache_time = cached
            if self._is_cache_valid(cache_time, self.CACHE_TTL_DEFAULT):
                logger.debug(f"Returning cached open interest for {symbol}")
                return data
        
        params = {'symbol': symbol.upper()}
        data = await self._make_request("/fapi/v1/openInterest", params)
        
        if not data:
            return cached[0] if cached else None
        
        try:
            # Get current price for USD value calculation
            ticker = await self._make_request("/fapi/v1/ticker/price", {'symbol': symbol.upper()})
            price = float(ticker['price']) if ticker else 0
            
            oi = float(data['openInterest'])
            result = OpenInterestData(
                symbol=data['symbol'],
                open_interest=oi,
                open_interest_value=oi * price,
                timestamp=datetime.now()
            )
            
            self._oi_cache[symbol] = (result, datetime.now())
            logger.info(f"Open interest for {symbol}: {oi:,.0f} contracts (${result.open_interest_value:,.0f})")
            return result
            
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Error parsing open interest: {e}")
            return cached[0] if cached else None
    
    async def get_open_interest_history(
        self,
        symbol: str,
        period: str = "5m",
        limit: int = 30,
        force_refresh: bool = False
    ) -> Optional[List[OpenInterestData]]:
        """
        Get historical open interest statistics
        
        Args:
            symbol: Trading pair symbol
            period: Time period - "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"
            limit: Number of entries (max 500)
            force_refresh: Bypass cache
            
        Returns:
            List of OpenInterestData or None
        """
        cache_key = f"{symbol}_{period}_{limit}"
        cached = self._oi_hist_cache.get(cache_key)
        
        if not force_refresh and cached:
            data, cache_time = cached
            if self._is_cache_valid(cache_time, self.CACHE_TTL_DEFAULT):
                logger.debug(f"Returning cached OI history for {symbol}")
                return data
        
        params = {
            'symbol': symbol.upper(),
            'period': period,
            'limit': min(limit, 500)
        }
        data = await self._make_request("/futures/data/openInterestHist", params)
        
        if not data:
            return cached[0] if cached else None
        
        try:
            result = [
                OpenInterestData(
                    symbol=entry['symbol'],
                    open_interest=float(entry['sumOpenInterest']),
                    open_interest_value=float(entry['sumOpenInterestValue']),
                    timestamp=datetime.fromtimestamp(int(entry['timestamp']) / 1000)
                )
                for entry in data
            ]
            
            self._oi_hist_cache[cache_key] = (result, datetime.now())
            logger.info(f"Fetched {len(result)} OI history entries for {symbol}")
            return result
            
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Error parsing OI history: {e}")
            return cached[0] if cached else None
    
    async def get_long_short_ratio(
        self,
        symbol: str,
        period: str = "5m",
        limit: int = 1,
        force_refresh: bool = False
    ) -> Optional[LongShortRatioData]:
        """
        Get global long/short account ratio
        
        Args:
            symbol: Trading pair symbol
            period: Time period - "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"
            limit: Number of entries
            force_refresh: Bypass cache
            
        Returns:
            LongShortRatioData or None
        """
        cache_key = f"{symbol}_{period}"
        cached = self._ls_ratio_cache.get(cache_key)
        
        if not force_refresh and cached:
            data, cache_time = cached
            if self._is_cache_valid(cache_time, self.CACHE_TTL_DEFAULT):
                logger.debug(f"Returning cached L/S ratio for {symbol}")
                return data
        
        params = {
            'symbol': symbol.upper(),
            'period': period,
            'limit': limit
        }
        data = await self._make_request("/futures/data/globalLongShortAccountRatio", params)
        
        if not data or len(data) == 0:
            return cached[0] if cached else None
        
        try:
            entry = data[0]
            result = LongShortRatioData(
                symbol=entry['symbol'],
                long_short_ratio=float(entry['longShortRatio']),
                long_account=float(entry['longAccount']),
                short_account=float(entry['shortAccount']),
                timestamp=datetime.fromtimestamp(int(entry['timestamp']) / 1000)
            )
            
            self._ls_ratio_cache[cache_key] = (result, datetime.now())
            logger.info(f"L/S ratio for {symbol}: {result.long_short_ratio:.3f} (Long: {result.long_account:.1%}, Short: {result.short_account:.1%})")
            return result
            
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Error parsing L/S ratio: {e}")
            return cached[0] if cached else None
    
    async def get_top_trader_long_short_ratio(
        self,
        symbol: str,
        period: str = "5m",
        force_refresh: bool = False
    ) -> Optional[LongShortRatioData]:
        """
        Get top trader long/short account ratio
        
        Args:
            symbol: Trading pair symbol
            period: Time period
            force_refresh: Bypass cache
            
        Returns:
            LongShortRatioData or None
        """
        cache_key = f"{symbol}_{period}"
        cached = self._top_ls_ratio_cache.get(cache_key)
        
        if not force_refresh and cached:
            data, cache_time = cached
            if self._is_cache_valid(cache_time, self.CACHE_TTL_DEFAULT):
                logger.debug(f"Returning cached top trader L/S ratio for {symbol}")
                return data
        
        params = {
            'symbol': symbol.upper(),
            'period': period,
            'limit': 1
        }
        data = await self._make_request("/futures/data/topLongShortAccountRatio", params)
        
        if not data or len(data) == 0:
            return cached[0] if cached else None
        
        try:
            entry = data[0]
            result = LongShortRatioData(
                symbol=entry['symbol'],
                long_short_ratio=float(entry['longShortRatio']),
                long_account=float(entry['longAccount']),
                short_account=float(entry['shortAccount']),
                timestamp=datetime.fromtimestamp(int(entry['timestamp']) / 1000)
            )
            
            self._top_ls_ratio_cache[cache_key] = (result, datetime.now())
            logger.info(f"Top trader L/S ratio for {symbol}: {result.long_short_ratio:.3f}")
            return result
            
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Error parsing top trader L/S ratio: {e}")
            return cached[0] if cached else None
    
    async def get_top_trader_position_ratio(
        self,
        symbol: str,
        period: str = "5m",
        force_refresh: bool = False
    ) -> Optional[LongShortRatioData]:
        """
        Get top trader position long/short ratio
        
        Args:
            symbol: Trading pair symbol
            period: Time period
            force_refresh: Bypass cache
            
        Returns:
            LongShortRatioData or None
        """
        cache_key = f"{symbol}_{period}"
        cached = self._top_pos_ratio_cache.get(cache_key)
        
        if not force_refresh and cached:
            data, cache_time = cached
            if self._is_cache_valid(cache_time, self.CACHE_TTL_DEFAULT):
                logger.debug(f"Returning cached top trader position ratio for {symbol}")
                return data
        
        params = {
            'symbol': symbol.upper(),
            'period': period,
            'limit': 1
        }
        data = await self._make_request("/futures/data/topLongShortPositionRatio", params)
        
        if not data or len(data) == 0:
            return cached[0] if cached else None
        
        try:
            entry = data[0]
            result = LongShortRatioData(
                symbol=entry['symbol'],
                long_short_ratio=float(entry['longShortRatio']),
                long_account=float(entry['longAccount']),
                short_account=float(entry['shortAccount']),
                timestamp=datetime.fromtimestamp(int(entry['timestamp']) / 1000)
            )
            
            self._top_pos_ratio_cache[cache_key] = (result, datetime.now())
            logger.info(f"Top trader position ratio for {symbol}: {result.long_short_ratio:.3f}")
            return result
            
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Error parsing top trader position ratio: {e}")
            return cached[0] if cached else None
    
    async def get_taker_buy_sell_volume(
        self,
        symbol: str,
        period: str = "5m",
        limit: int = 30,
        force_refresh: bool = False
    ) -> Optional[List[TakerVolumeData]]:
        """
        Get taker buy/sell volume ratio
        
        Args:
            symbol: Trading pair symbol
            period: Time period - "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"
            limit: Number of entries (max 500)
            force_refresh: Bypass cache
            
        Returns:
            List of TakerVolumeData or None
        """
        cache_key = f"{symbol}_{period}_{limit}"
        cached = self._taker_volume_cache.get(cache_key)
        
        if not force_refresh and cached:
            data, cache_time = cached
            if self._is_cache_valid(cache_time, self.CACHE_TTL_DEFAULT):
                logger.debug(f"Returning cached taker volume for {symbol}")
                return data
        
        params = {
            'symbol': symbol.upper(),
            'period': period,
            'limit': min(limit, 500)
        }
        data = await self._make_request("/futures/data/takerlongshortRatio", params)
        
        if not data:
            return cached[0] if cached else None
        
        try:
            result = [
                TakerVolumeData(
                    symbol=entry['symbol'] if 'symbol' in entry else symbol.upper(),
                    buy_sell_ratio=float(entry['buySellRatio']),
                    buy_volume=float(entry['buyVol']),
                    sell_volume=float(entry['sellVol']),
                    timestamp=datetime.fromtimestamp(int(entry['timestamp']) / 1000)
                )
                for entry in data
            ]
            
            self._taker_volume_cache[cache_key] = (result, datetime.now())
            logger.info(f"Fetched {len(result)} taker volume entries for {symbol}")
            return result
            
        except (KeyError, TypeError, ValueError) as e:
            logger.error(f"Error parsing taker volume: {e}")
            return cached[0] if cached else None
    
    def _calculate_funding_trend(self, funding_rates: List[FundingRateEntry]) -> str:
        """Calculate funding rate trend from historical data"""
        if not funding_rates or len(funding_rates) < 3:
            return "stable"
        
        # Compare recent average to older average
        recent = funding_rates[:3]
        older = funding_rates[-3:] if len(funding_rates) >= 6 else funding_rates[3:]
        
        recent_avg = sum(r.funding_rate for r in recent) / len(recent)
        older_avg = sum(r.funding_rate for r in older) / len(older)
        
        diff = recent_avg - older_avg
        if diff > 0.0001:  # Threshold for rising
            return "rising"
        elif diff < -0.0001:  # Threshold for falling
            return "falling"
        return "stable"
    
    def _calculate_sentiment(self, derivatives_score: float) -> str:
        """Determine market sentiment from derivatives score"""
        if derivatives_score >= 0.6:
            return "extremely_bullish"
        elif derivatives_score >= 0.2:
            return "bullish"
        elif derivatives_score >= -0.2:
            return "neutral"
        elif derivatives_score >= -0.6:
            return "bearish"
        else:
            return "extremely_bearish"
    
    def _calculate_derivatives_score(
        self,
        funding_rate: float,
        long_short_ratio: float,
        oi_change_24h: float,
        taker_buy_ratio: float
    ) -> float:
        """
        Calculate composite derivatives score from -1 to +1
        
        Scoring logic:
        - Positive funding = bearish (crowded long)
        - High L/S ratio = bearish (contrarian)
        - Rising OI = context dependent
        - High taker buy ratio = bullish
        """
        score = 0.0
        
        # Funding rate component (-0.3 to +0.3)
        # Positive funding means longs pay shorts -> crowded long -> bearish
        # Negative funding means shorts pay longs -> crowded short -> bullish
        funding_score = -funding_rate * 3000  # Scale typical funding rates
        funding_score = max(-0.3, min(0.3, funding_score))
        score += funding_score
        
        # Long/Short ratio component (-0.25 to +0.25)
        # High L/S ratio (>1.5) = too many longs -> bearish
        # Low L/S ratio (<0.7) = too many shorts -> bullish
        ls_score = -(long_short_ratio - 1.0) * 0.5
        ls_score = max(-0.25, min(0.25, ls_score))
        score += ls_score
        
        # OI change component (-0.2 to +0.2)
        # Rising OI with positive indicators = bullish momentum
        oi_score = oi_change_24h * 0.02
        oi_score = max(-0.2, min(0.2, oi_score))
        score += oi_score
        
        # Taker buy ratio component (-0.25 to +0.25)
        # High taker buy ratio (>1.0) = aggressive buying -> bullish
        # Low taker buy ratio (<1.0) = aggressive selling -> bearish
        taker_score = (taker_buy_ratio - 1.0) * 0.5
        taker_score = max(-0.25, min(0.25, taker_score))
        score += taker_score
        
        return max(-1.0, min(1.0, score))
    
    async def get_derivatives_intelligence(
        self,
        symbol: str,
        force_refresh: bool = False
    ) -> Optional[DerivativesData]:
        """
        Get comprehensive derivatives market intelligence
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            force_refresh: Bypass all caches
            
        Returns:
            DerivativesData with aggregated metrics or None if unavailable
        """
        logger.info(f"Fetching derivatives intelligence for {symbol}")
        
        try:
            # Fetch all data concurrently
            funding_task = self.get_funding_rate(symbol, limit=24, force_refresh=force_refresh)
            oi_task = self.get_open_interest(symbol, force_refresh=force_refresh)
            oi_hist_task = self.get_open_interest_history(symbol, period="1h", limit=24, force_refresh=force_refresh)
            ls_ratio_task = self.get_long_short_ratio(symbol, force_refresh=force_refresh)
            top_ls_task = self.get_top_trader_long_short_ratio(symbol, force_refresh=force_refresh)
            taker_task = self.get_taker_buy_sell_volume(symbol, period="5m", limit=12, force_refresh=force_refresh)
            
            results = await asyncio.gather(
                funding_task, oi_task, oi_hist_task,
                ls_ratio_task, top_ls_task, taker_task,
                return_exceptions=True
            )
            
            funding_rates: Optional[List[FundingRateEntry]] = results[0] if not isinstance(results[0], BaseException) else None
            oi_data: Optional[OpenInterestData] = results[1] if not isinstance(results[1], BaseException) else None
            oi_hist: Optional[List[OpenInterestData]] = results[2] if not isinstance(results[2], BaseException) else None
            ls_ratio: Optional[LongShortRatioData] = results[3] if not isinstance(results[3], BaseException) else None
            top_ls: Optional[LongShortRatioData] = results[4] if not isinstance(results[4], BaseException) else None
            taker_data: Optional[List[TakerVolumeData]] = results[5] if not isinstance(results[5], BaseException) else None
            
            # Calculate current funding rate
            current_funding = funding_rates[0].funding_rate if funding_rates and len(funding_rates) > 0 else 0.0
            
            # Calculate 8-hour average (3 funding periods)
            if funding_rates and len(funding_rates) >= 3:
                funding_8h_avg = sum(r.funding_rate for r in funding_rates[:3]) / 3
            else:
                funding_8h_avg = current_funding
            
            # Calculate funding trend
            funding_trend = self._calculate_funding_trend(funding_rates) if funding_rates else "stable"
            
            # Get open interest data
            current_oi = oi_data.open_interest if oi_data else 0.0
            current_oi_value = oi_data.open_interest_value if oi_data else 0.0
            
            # Calculate 24h OI change
            oi_change_24h = 0.0
            if oi_hist and len(oi_hist) >= 2:
                latest_oi = oi_hist[0].open_interest_value
                oldest_oi = oi_hist[-1].open_interest_value
                if oldest_oi > 0:
                    oi_change_24h = ((latest_oi - oldest_oi) / oldest_oi) * 100
            
            # Get long/short ratio
            current_ls_ratio = ls_ratio.long_short_ratio if ls_ratio else 1.0
            
            # Get top trader long ratio
            top_long_ratio = top_ls.long_account if top_ls else 0.5
            
            # Calculate taker buy ratio (average of recent periods)
            if taker_data and len(taker_data) > 0:
                taker_buy_ratio = sum(t.buy_sell_ratio for t in taker_data) / len(taker_data)
            else:
                taker_buy_ratio = 1.0
            
            # Calculate composite score
            derivatives_score = self._calculate_derivatives_score(
                funding_rate=current_funding,
                long_short_ratio=current_ls_ratio,
                oi_change_24h=oi_change_24h,
                taker_buy_ratio=taker_buy_ratio
            )
            
            # Determine sentiment
            sentiment = self._calculate_sentiment(derivatives_score)
            
            result = DerivativesData(
                funding_rate=current_funding,
                funding_rate_8h_avg=funding_8h_avg,
                funding_rate_trend=funding_trend,
                open_interest=current_oi,
                open_interest_value=current_oi_value,
                oi_change_24h=oi_change_24h,
                long_short_ratio=current_ls_ratio,
                top_trader_long_ratio=top_long_ratio,
                taker_buy_ratio=taker_buy_ratio,
                market_sentiment=sentiment,
                derivatives_score=derivatives_score
            )
            
            logger.info(
                f"Derivatives intelligence for {symbol}: "
                f"Score={derivatives_score:.3f}, Sentiment={sentiment}, "
                f"Funding={current_funding:.6f}, L/S={current_ls_ratio:.3f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting derivatives intelligence for {symbol}: {e}")
            return None
    
    def get_cached_derivatives_summary(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get a summary of cached derivatives data without making API calls"""
        summary = {}
        
        # Check funding cache
        for key, (data, _) in self._funding_cache.items():
            if key.startswith(symbol):
                summary['funding_rate'] = data[0].funding_rate if data else None
                break
        
        # Check OI cache
        if symbol in self._oi_cache:
            data, _ = self._oi_cache[symbol]
            summary['open_interest'] = data.open_interest
            summary['open_interest_value'] = data.open_interest_value
        
        # Check L/S ratio cache
        for key, (data, _) in self._ls_ratio_cache.items():
            if key.startswith(symbol):
                summary['long_short_ratio'] = data.long_short_ratio
                break
        
        return summary if summary else None
