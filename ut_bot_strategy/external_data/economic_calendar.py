"""
Economic Calendar Client for Macro Event Awareness

Fetches upcoming economic events that affect crypto markets.
Primary: Finnhub API (free tier, requires API key)
Fallback: Pre-defined scheduled major events + alternative APIs

API Documentation: https://finnhub.io/docs/api/economic-calendar

Key events tracked: FOMC, CPI, NFP, PPI, GDP, Interest Rate decisions, Fed speakers
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
from enum import Enum
import aiohttp

logger = logging.getLogger(__name__)


class EventImpact(str, Enum):
    """Economic event impact levels"""
    HIGH = 'HIGH'
    MEDIUM = 'MEDIUM'
    LOW = 'LOW'
    UNKNOWN = 'UNKNOWN'


@dataclass
class EconomicEvent:
    """Individual economic event"""
    name: str
    time: datetime
    impact: EventImpact
    currency: str
    forecast: Optional[str] = None
    previous: Optional[str] = None
    actual: Optional[str] = None
    country: str = ''
    unit: str = ''
    
    @property
    def is_high_impact(self) -> bool:
        return self.impact == EventImpact.HIGH
    
    @property
    def is_crypto_relevant(self) -> bool:
        crypto_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CNY']
        crypto_keywords = [
            'fomc', 'fed', 'interest rate', 'cpi', 'inflation',
            'nfp', 'non-farm', 'employment', 'gdp', 'ppi',
            'retail sales', 'pce', 'treasury', 'powell', 'yellen'
        ]
        if self.currency.upper() in crypto_currencies:
            return True
        name_lower = self.name.lower()
        return any(keyword in name_lower for keyword in crypto_keywords)
    
    @property
    def minutes_until(self) -> int:
        now = datetime.now(timezone.utc)
        event_time = self.time if self.time.tzinfo else self.time.replace(tzinfo=timezone.utc)
        delta = event_time - now
        return int(delta.total_seconds() / 60)
    
    @property
    def is_imminent(self) -> bool:
        return 0 <= self.minutes_until <= 60
    
    def get_volatility_warning(self) -> Optional[str]:
        if not self.is_high_impact:
            return None
        minutes = self.minutes_until
        if minutes < 0:
            return None
        if minutes <= 15:
            return f"⚠️ CRITICAL: {self.name} in {minutes} minutes - AVOID NEW POSITIONS"
        elif minutes <= 60:
            return f"🔔 WARNING: {self.name} in {minutes} minutes - EXERCISE CAUTION"
        elif minutes <= 240:
            return f"📅 UPCOMING: {self.name} in {minutes // 60}h {minutes % 60}m"
        return None


@dataclass
class EconomicCalendarSummary:
    """Summary of economic calendar analysis"""
    total_events: int
    high_impact_events: int
    upcoming_high_impact: List[EconomicEvent]
    has_imminent_event: bool
    next_major_event: Optional[EconomicEvent]
    trading_recommendation: str
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def should_avoid_trading(self) -> bool:
        if not self.upcoming_high_impact:
            return False
        return any(e.minutes_until <= 30 for e in self.upcoming_high_impact)


HIGH_IMPACT_KEYWORDS = [
    'fomc', 'federal reserve', 'fed chair', 'powell', 'interest rate decision',
    'cpi', 'consumer price index', 'inflation',
    'nfp', 'non-farm payrolls', 'nonfarm payrolls', 'employment change',
    'ppi', 'producer price index',
    'gdp', 'gross domestic product',
    'retail sales', 'pce', 'core pce', 'personal consumption',
    'unemployment rate', 'jobless claims', 'initial claims',
    'ism manufacturing', 'ism services', 'ism pmi',
    'treasury auction', 'treasury yield',
    'ecb', 'european central bank', 'lagarde',
    'boe', 'bank of england', 'bailey'
]

MEDIUM_IMPACT_KEYWORDS = [
    'durable goods', 'housing starts', 'building permits',
    'consumer confidence', 'michigan consumer', 'sentiment',
    'trade balance', 'current account',
    'industrial production', 'capacity utilization',
    'existing home sales', 'new home sales',
    'pmi', 'purchasing managers'
]


def determine_impact(event_name: str, api_impact: Optional[int] = None) -> EventImpact:
    """Determine event impact based on name keywords and API data"""
    if api_impact is not None:
        if api_impact >= 3:
            return EventImpact.HIGH
        elif api_impact == 2:
            return EventImpact.MEDIUM
        elif api_impact >= 0:
            return EventImpact.LOW
    
    name_lower = event_name.lower()
    
    if any(keyword in name_lower for keyword in HIGH_IMPACT_KEYWORDS):
        return EventImpact.HIGH
    
    if any(keyword in name_lower for keyword in MEDIUM_IMPACT_KEYWORDS):
        return EventImpact.MEDIUM
    
    return EventImpact.LOW


class EconomicCalendarClient:
    """
    Economic Calendar Client for macro event awareness
    
    Features:
    - Fetch upcoming economic events affecting crypto markets
    - Filter by impact level (HIGH/MEDIUM/LOW)
    - Check for imminent high-impact events
    - Trading recommendations during volatile periods
    - 30-minute cache with graceful degradation
    """
    
    FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
    CACHE_TTL_SECONDS = 1800  # 30 minutes
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('FINNHUB_API_KEY', '')
        self._session: Optional[aiohttp.ClientSession] = None
        
        self._events_cache: Optional[List[EconomicEvent]] = None
        self._events_cache_time: Optional[datetime] = None
        
        self._summary_cache: Optional[EconomicCalendarSummary] = None
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
    
    def _is_cache_valid(self) -> bool:
        """Check if events cache is still valid"""
        if self._events_cache is None or self._events_cache_time is None:
            return False
        return (datetime.now() - self._events_cache_time).total_seconds() < self.CACHE_TTL_SECONDS
    
    def _is_summary_cache_valid(self) -> bool:
        """Check if summary cache is still valid"""
        if self._summary_cache is None or self._summary_cache_time is None:
            return False
        return (datetime.now() - self._summary_cache_time).total_seconds() < self.CACHE_TTL_SECONDS
    
    async def _fetch_finnhub_calendar(
        self,
        from_date: str,
        to_date: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Fetch economic calendar from Finnhub API"""
        if not self.api_key:
            logger.warning("Finnhub API key not set - using fallback data")
            return None
        
        try:
            session = await self._get_session()
            url = f"{self.FINNHUB_BASE_URL}/calendar/economic"
            params = {
                'from': from_date,
                'to': to_date,
                'token': self.api_key
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 401:
                    logger.warning("Finnhub API authentication failed")
                    return None
                
                if response.status == 429:
                    logger.warning("Finnhub rate limit exceeded")
                    return None
                
                if response.status != 200:
                    text = await response.text()
                    logger.warning(f"Finnhub API returned status {response.status}: {text}")
                    return None
                
                data = await response.json()
                return data.get('economicCalendar', [])
                
        except asyncio.TimeoutError:
            logger.warning("Finnhub API timeout")
            return None
        except aiohttp.ClientError as e:
            logger.warning(f"Finnhub API client error: {e}")
            return None
        except Exception as e:
            logger.error(f"Finnhub API unexpected error: {e}")
            return None
    
    def _get_known_scheduled_events(self, hours_ahead: int = 168) -> List[EconomicEvent]:
        """
        Get known scheduled high-impact events as fallback
        These are well-known recurring events that can be predicted
        """
        now = datetime.now(timezone.utc)
        end_time = now + timedelta(hours=hours_ahead)
        events = []
        
        known_events = [
            ("FOMC Interest Rate Decision", "USD", EventImpact.HIGH),
            ("US CPI (YoY)", "USD", EventImpact.HIGH),
            ("US CPI (MoM)", "USD", EventImpact.HIGH),
            ("US Non-Farm Payrolls", "USD", EventImpact.HIGH),
            ("US Unemployment Rate", "USD", EventImpact.HIGH),
            ("US PPI (YoY)", "USD", EventImpact.HIGH),
            ("US GDP (QoQ)", "USD", EventImpact.HIGH),
            ("US Core PCE (YoY)", "USD", EventImpact.HIGH),
            ("US Retail Sales (MoM)", "USD", EventImpact.HIGH),
            ("US ISM Manufacturing PMI", "USD", EventImpact.MEDIUM),
            ("US ISM Services PMI", "USD", EventImpact.MEDIUM),
            ("ECB Interest Rate Decision", "EUR", EventImpact.HIGH),
            ("EU CPI (YoY)", "EUR", EventImpact.HIGH),
            ("BOE Interest Rate Decision", "GBP", EventImpact.HIGH),
            ("UK CPI (YoY)", "GBP", EventImpact.HIGH),
            ("China GDP (YoY)", "CNY", EventImpact.HIGH),
            ("Japan Interest Rate Decision", "JPY", EventImpact.HIGH),
        ]
        
        for event_name, currency, impact in known_events:
            event = EconomicEvent(
                name=event_name,
                time=end_time,
                impact=impact,
                currency=currency,
                country=currency[:2] if len(currency) >= 2 else currency
            )
            events.append(event)
        
        logger.info(f"Using {len(events)} known scheduled events as reference")
        return events
    
    async def get_upcoming_events(
        self,
        hours_ahead: int = 24,
        force_refresh: bool = False
    ) -> List[EconomicEvent]:
        """
        Get upcoming economic events in the next N hours
        
        Args:
            hours_ahead: Number of hours to look ahead (default 24)
            force_refresh: Bypass cache
            
        Returns:
            List of EconomicEvent sorted by time
        """
        if not force_refresh and self._is_cache_valid() and self._events_cache is not None:
            logger.debug("Returning cached economic events")
            now = datetime.now(timezone.utc)
            cutoff = now + timedelta(hours=hours_ahead)
            return [
                e for e in self._events_cache
                if e.time.replace(tzinfo=timezone.utc) <= cutoff
                and e.time.replace(tzinfo=timezone.utc) >= now
            ]
        
        now = datetime.now(timezone.utc)
        from_date = now.strftime('%Y-%m-%d')
        to_date = (now + timedelta(hours=hours_ahead + 24)).strftime('%Y-%m-%d')
        
        api_data = await self._fetch_finnhub_calendar(from_date, to_date)
        
        if api_data:
            events = []
            for item in api_data:
                try:
                    event_time_str = item.get('time', '')
                    if not event_time_str:
                        continue
                    
                    try:
                        event_time = datetime.fromisoformat(event_time_str.replace('Z', '+00:00'))
                    except (ValueError, TypeError):
                        continue
                    
                    if not event_time.tzinfo:
                        event_time = event_time.replace(tzinfo=timezone.utc)
                    
                    if event_time < now or event_time > now + timedelta(hours=hours_ahead):
                        continue
                    
                    event_name = item.get('event', '')
                    api_impact = item.get('impact')
                    impact = determine_impact(event_name, api_impact)
                    
                    event = EconomicEvent(
                        name=event_name,
                        time=event_time,
                        impact=impact,
                        currency=item.get('currency', item.get('country', 'USD')),
                        forecast=str(item.get('estimate', '')) if item.get('estimate') else None,
                        previous=str(item.get('prev', '')) if item.get('prev') else None,
                        actual=str(item.get('actual', '')) if item.get('actual') else None,
                        country=item.get('country', ''),
                        unit=item.get('unit', '')
                    )
                    events.append(event)
                    
                except (KeyError, TypeError, ValueError) as e:
                    logger.warning(f"Error parsing event: {e}")
                    continue
            
            events.sort(key=lambda x: x.time)
            
            self._events_cache = events
            self._events_cache_time = datetime.now()
            
            logger.info(f"Fetched {len(events)} economic events from Finnhub")
            return events
        
        if self._events_cache:
            logger.info("API unavailable, returning stale cache")
            now = datetime.now(timezone.utc)
            cutoff = now + timedelta(hours=hours_ahead)
            return [
                e for e in self._events_cache
                if e.time.replace(tzinfo=timezone.utc) <= cutoff
                and e.time.replace(tzinfo=timezone.utc) >= now
            ]
        
        logger.warning("No API data available, using known scheduled events as fallback")
        return self._get_known_scheduled_events(hours_ahead)
    
    async def get_high_impact_events(
        self,
        hours_ahead: int = 24,
        force_refresh: bool = False
    ) -> List[EconomicEvent]:
        """
        Get only HIGH impact events in the next N hours
        
        Args:
            hours_ahead: Number of hours to look ahead (default 24)
            force_refresh: Bypass cache
            
        Returns:
            List of high-impact EconomicEvent sorted by time
        """
        all_events = await self.get_upcoming_events(hours_ahead, force_refresh)
        high_impact = [e for e in all_events if e.is_high_impact]
        logger.info(f"Found {len(high_impact)} high-impact events in next {hours_ahead} hours")
        return high_impact
    
    async def get_crypto_relevant_events(
        self,
        hours_ahead: int = 24,
        force_refresh: bool = False
    ) -> List[EconomicEvent]:
        """
        Get events specifically relevant to crypto markets
        
        Args:
            hours_ahead: Number of hours to look ahead
            force_refresh: Bypass cache
            
        Returns:
            List of crypto-relevant EconomicEvent
        """
        all_events = await self.get_upcoming_events(hours_ahead, force_refresh)
        return [e for e in all_events if e.is_crypto_relevant]
    
    async def has_high_impact_event_soon(
        self,
        minutes: int = 60,
        force_refresh: bool = False
    ) -> bool:
        """
        Check if a high-impact event is imminent
        
        Args:
            minutes: Time window to check (default 60 minutes)
            force_refresh: Bypass cache
            
        Returns:
            True if high-impact event within time window
        """
        hours_ahead = max(2, (minutes // 60) + 1)
        high_impact = await self.get_high_impact_events(hours_ahead, force_refresh)
        
        for event in high_impact:
            event_minutes = event.minutes_until
            if 0 <= event_minutes <= minutes:
                logger.warning(f"High-impact event imminent: {event.name} in {event_minutes} minutes")
                return True
        
        return False
    
    async def get_next_major_event(
        self,
        force_refresh: bool = False
    ) -> Optional[EconomicEvent]:
        """
        Get the next major (high-impact) economic event
        
        Returns:
            Next EconomicEvent or None if no upcoming high-impact events
        """
        high_impact = await self.get_high_impact_events(hours_ahead=168, force_refresh=force_refresh)
        
        if not high_impact:
            return None
        
        now = datetime.now(timezone.utc)
        future_events = [e for e in high_impact if e.minutes_until > 0]
        
        if not future_events:
            return None
        
        return min(future_events, key=lambda x: x.time)
    
    async def get_trading_recommendation(
        self,
        force_refresh: bool = False
    ) -> str:
        """
        Get trading recommendation based on upcoming events
        
        Returns:
            Trading recommendation string
        """
        has_imminent = await self.has_high_impact_event_soon(minutes=30, force_refresh=force_refresh)
        
        if has_imminent:
            return "🚫 AVOID NEW POSITIONS - High-impact event in next 30 minutes"
        
        has_upcoming = await self.has_high_impact_event_soon(minutes=60, force_refresh=force_refresh)
        
        if has_upcoming:
            return "⚠️ EXERCISE CAUTION - High-impact event in next hour"
        
        high_impact_4h = await self.get_high_impact_events(hours_ahead=4, force_refresh=False)
        
        if high_impact_4h:
            next_event = high_impact_4h[0]
            return f"📋 MONITOR - {next_event.name} in {next_event.minutes_until // 60}h {next_event.minutes_until % 60}m"
        
        return "✅ NORMAL TRADING CONDITIONS - No imminent high-impact events"
    
    async def get_summary(
        self,
        hours_ahead: int = 24,
        force_refresh: bool = False
    ) -> EconomicCalendarSummary:
        """
        Get comprehensive summary of economic calendar
        
        Args:
            hours_ahead: Hours to look ahead
            force_refresh: Bypass cache
            
        Returns:
            EconomicCalendarSummary
        """
        if not force_refresh and self._is_summary_cache_valid() and self._summary_cache is not None:
            return self._summary_cache
        
        all_events = await self.get_upcoming_events(hours_ahead, force_refresh)
        high_impact = [e for e in all_events if e.is_high_impact]
        has_imminent = await self.has_high_impact_event_soon(minutes=60, force_refresh=False)
        recommendation = await self.get_trading_recommendation(force_refresh=False)
        next_major = await self.get_next_major_event(force_refresh=False)
        
        summary = EconomicCalendarSummary(
            total_events=len(all_events),
            high_impact_events=len(high_impact),
            upcoming_high_impact=high_impact,
            has_imminent_event=has_imminent,
            next_major_event=next_major,
            trading_recommendation=recommendation,
            last_updated=datetime.now()
        )
        
        self._summary_cache = summary
        self._summary_cache_time = datetime.now()
        
        logger.info(
            f"Economic Calendar Summary: {summary.total_events} events, "
            f"{summary.high_impact_events} high-impact, "
            f"imminent={summary.has_imminent_event}"
        )
        
        return summary
    
    def get_volatility_warnings(
        self,
        events: Optional[List[EconomicEvent]] = None
    ) -> List[str]:
        """
        Generate volatility warnings for given events
        
        Args:
            events: List of events to check (uses cache if None)
            
        Returns:
            List of warning strings
        """
        if events is None:
            events = self._events_cache or []
        
        warnings = []
        for event in events:
            warning = event.get_volatility_warning()
            if warning:
                warnings.append(warning)
        
        return warnings
    
    def clear_cache(self) -> None:
        """Clear all cached data"""
        self._events_cache = None
        self._events_cache_time = None
        self._summary_cache = None
        self._summary_cache_time = None
        logger.info("Economic calendar cache cleared")
    
    def get_cached_events(self) -> Optional[List[EconomicEvent]]:
        """Get cached events without making API call"""
        return self._events_cache
    
    def get_cache_age_seconds(self) -> Optional[float]:
        """Get age of cache in seconds"""
        if self._events_cache_time is None:
            return None
        return (datetime.now() - self._events_cache_time).total_seconds()


async def check_economic_events_before_trade() -> Dict[str, Any]:
    """
    Utility function to check economic events before executing a trade
    
    Returns:
        Dict with 'safe_to_trade', 'warnings', and 'next_event' keys
    """
    client = EconomicCalendarClient()
    
    try:
        has_imminent = await client.has_high_impact_event_soon(minutes=30)
        recommendation = await client.get_trading_recommendation()
        next_major = await client.get_next_major_event()
        warnings = client.get_volatility_warnings()
        
        return {
            'safe_to_trade': not has_imminent,
            'recommendation': recommendation,
            'warnings': warnings,
            'next_major_event': {
                'name': next_major.name,
                'time': next_major.time.isoformat(),
                'minutes_until': next_major.minutes_until,
                'impact': next_major.impact.value
            } if next_major else None
        }
    finally:
        await client.close()
