"""
External Intelligence Manager

Coordinates all market intelligence sources with reliability scoring,
multi-source alignment detection, and graceful degradation.

Features:
- Per-source reliability tracking (uptime, response time, data freshness)
- Multi-source alignment detection (confidence boost when sources agree)
- Graceful degradation when sources fail
- Configurable refresh intervals per source
- Health status reporting
- Clean interface for SignalEngine consumption
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Literal
from enum import Enum
from collections import deque

from .fear_greed_client import FearGreedClient, FearGreedData
from .market_data_aggregator import MarketDataAggregator, GlobalMarketData
from .news_sentiment_client import NewsSentimentClient, NewsSentimentSummary
from .derivatives_client import BinanceDerivativesClient, DerivativesData
from .whale_tracker import WhaleTracker, WhaleMetrics
from .economic_calendar import EconomicCalendarClient, EconomicCalendarSummary
from .liquidation_monitor import LiquidationMonitor, LiquidationMetrics

logger = logging.getLogger(__name__)


class SourceDirection(str, Enum):
    """Direction signal from a data source"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    UNAVAILABLE = "unavailable"


class SourceHealth(str, Enum):
    """Health status of a data source"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


@dataclass
class SourceReliability:
    """Reliability metrics for a single data source"""
    name: str
    success_count: int = 0
    failure_count: int = 0
    total_response_time_ms: float = 0.0
    last_success_time: Optional[datetime] = None
    last_failure_time: Optional[datetime] = None
    last_error: Optional[str] = None
    consecutive_failures: int = 0
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 1.0
        return self.success_count / total
    
    @property
    def avg_response_time_ms(self) -> float:
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    
    @property
    def health(self) -> SourceHealth:
        if self.consecutive_failures >= 5:
            return SourceHealth.OFFLINE
        if self.consecutive_failures >= 3:
            return SourceHealth.UNHEALTHY
        if self.success_rate < 0.8:
            return SourceHealth.DEGRADED
        return SourceHealth.HEALTHY
    
    @property
    def reliability_weight(self) -> float:
        """Calculate reliability weight for weighted calculations (0.0 to 1.0)"""
        if self.health == SourceHealth.OFFLINE:
            return 0.0
        if self.health == SourceHealth.UNHEALTHY:
            return 0.3
        if self.health == SourceHealth.DEGRADED:
            return 0.7
        return 1.0
    
    def record_success(self, response_time_ms: float) -> None:
        """Record a successful API call"""
        self.success_count += 1
        self.consecutive_failures = 0
        self.last_success_time = datetime.now()
        self.response_times.append(response_time_ms)
        self.total_response_time_ms += response_time_ms
    
    def record_failure(self, error: str) -> None:
        """Record a failed API call"""
        self.failure_count += 1
        self.consecutive_failures += 1
        self.last_failure_time = datetime.now()
        self.last_error = error
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'success_rate': round(self.success_rate, 4),
            'avg_response_time_ms': round(self.avg_response_time_ms, 2),
            'consecutive_failures': self.consecutive_failures,
            'health': self.health.value,
            'reliability_weight': round(self.reliability_weight, 2),
            'last_success': self.last_success_time.isoformat() if self.last_success_time else None,
            'last_failure': self.last_failure_time.isoformat() if self.last_failure_time else None,
            'last_error': self.last_error
        }


@dataclass
class SourceSignal:
    """Signal from a single data source"""
    source_name: str
    direction: SourceDirection
    strength: float  # 0.0 to 1.0
    raw_data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedIntelligence:
    """Aggregated intelligence from all sources"""
    timestamp: datetime
    overall_direction: SourceDirection
    overall_confidence: float  # 0.0 to 1.0
    alignment_count: int
    alignment_boost: float
    conflict_penalty: float
    signals: Dict[str, SourceSignal]
    reliability_scores: Dict[str, float]
    health_status: Dict[str, str]
    warnings: List[str]
    should_avoid_trading: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'overall_direction': self.overall_direction.value,
            'overall_confidence': round(self.overall_confidence, 4),
            'alignment_count': self.alignment_count,
            'alignment_boost': round(self.alignment_boost, 4),
            'conflict_penalty': round(self.conflict_penalty, 4),
            'signals': {k: {
                'direction': v.direction.value,
                'strength': round(v.strength, 4),
                'timestamp': v.timestamp.isoformat()
            } for k, v in self.signals.items()},
            'reliability_scores': {k: round(v, 4) for k, v in self.reliability_scores.items()},
            'health_status': self.health_status,
            'warnings': self.warnings,
            'should_avoid_trading': self.should_avoid_trading
        }


@dataclass 
class SourceConfig:
    """Configuration for a data source"""
    name: str
    refresh_interval_seconds: int = 300
    timeout_seconds: float = 15.0
    enabled: bool = True
    weight: float = 1.0


class ExternalIntelligenceManager:
    """
    Central coordinator for all external market intelligence sources.
    
    Features:
    - Coordinates Fear & Greed, CoinGecko, News Sentiment, Derivatives,
      Whale Tracker, Economic Calendar, and Liquidation Monitor
    - Per-source reliability tracking
    - Multi-source alignment detection with confidence adjustments
    - Graceful degradation when sources fail
    - Configurable refresh intervals
    - Health status reporting
    """
    
    SOURCE_FEAR_GREED = "fear_greed"
    SOURCE_MARKET_DATA = "market_data"
    SOURCE_NEWS_SENTIMENT = "news_sentiment"
    SOURCE_DERIVATIVES = "derivatives"
    SOURCE_WHALE_TRACKER = "whale_tracker"
    SOURCE_ECONOMIC_CALENDAR = "economic_calendar"
    SOURCE_LIQUIDATION = "liquidation"
    
    ALIGNMENT_BOOST_3 = 0.15
    ALIGNMENT_BOOST_4 = 0.25
    CONFLICT_PENALTY = 0.10
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        source_configs: Optional[Dict[str, SourceConfig]] = None,
        enable_websocket_sources: bool = False
    ):
        """
        Initialize the External Intelligence Manager.
        
        Args:
            symbol: Primary trading symbol for relevant data sources
            source_configs: Custom configuration per source
            enable_websocket_sources: Enable WebSocket-based sources (Whale, Liquidation)
        """
        self.symbol = symbol.upper()
        self.enable_websocket_sources = enable_websocket_sources
        
        self.source_configs: Dict[str, SourceConfig] = {
            self.SOURCE_FEAR_GREED: SourceConfig(
                name=self.SOURCE_FEAR_GREED,
                refresh_interval_seconds=300,
                weight=1.0
            ),
            self.SOURCE_MARKET_DATA: SourceConfig(
                name=self.SOURCE_MARKET_DATA,
                refresh_interval_seconds=120,
                weight=0.8
            ),
            self.SOURCE_NEWS_SENTIMENT: SourceConfig(
                name=self.SOURCE_NEWS_SENTIMENT,
                refresh_interval_seconds=180,
                weight=0.9
            ),
            self.SOURCE_DERIVATIVES: SourceConfig(
                name=self.SOURCE_DERIVATIVES,
                refresh_interval_seconds=60,
                weight=1.2
            ),
            self.SOURCE_WHALE_TRACKER: SourceConfig(
                name=self.SOURCE_WHALE_TRACKER,
                refresh_interval_seconds=30,
                weight=1.1,
                enabled=enable_websocket_sources
            ),
            self.SOURCE_ECONOMIC_CALENDAR: SourceConfig(
                name=self.SOURCE_ECONOMIC_CALENDAR,
                refresh_interval_seconds=1800,
                weight=0.7
            ),
            self.SOURCE_LIQUIDATION: SourceConfig(
                name=self.SOURCE_LIQUIDATION,
                refresh_interval_seconds=30,
                weight=1.0,
                enabled=enable_websocket_sources
            )
        }
        
        if source_configs:
            for name, config in source_configs.items():
                if name in self.source_configs:
                    self.source_configs[name] = config
        
        self.fear_greed_client = FearGreedClient()
        self.market_data_client = MarketDataAggregator()
        self.news_sentiment_client = NewsSentimentClient()
        self.derivatives_client = BinanceDerivativesClient()
        self.economic_calendar_client = EconomicCalendarClient()
        
        self.whale_tracker: Optional[WhaleTracker] = None
        self.liquidation_monitor: Optional[LiquidationMonitor] = None
        
        if enable_websocket_sources:
            self.whale_tracker = WhaleTracker(symbol=symbol)
            self.liquidation_monitor = LiquidationMonitor()
        
        self.reliability: Dict[str, SourceReliability] = {
            name: SourceReliability(name=name)
            for name in self.source_configs.keys()
        }
        
        self._signal_cache: Dict[str, SourceSignal] = {}
        self._last_fetch: Dict[str, datetime] = {}
        self._running = False
        self._update_task: Optional[asyncio.Task] = None
        
        logger.info(
            f"ExternalIntelligenceManager initialized for {symbol}, "
            f"WebSocket sources: {'enabled' if enable_websocket_sources else 'disabled'}"
        )
    
    async def start(self) -> None:
        """Start the intelligence manager and any WebSocket sources"""
        if self._running:
            logger.warning("IntelligenceManager is already running")
            return
        
        self._running = True
        logger.info("Starting ExternalIntelligenceManager...")
        
        if self.whale_tracker and self.source_configs[self.SOURCE_WHALE_TRACKER].enabled:
            try:
                await self.whale_tracker.start()
                logger.info("WhaleTracker started")
            except Exception as e:
                logger.error(f"Failed to start WhaleTracker: {e}")
                self.reliability[self.SOURCE_WHALE_TRACKER].record_failure(str(e))
        
        if self.liquidation_monitor and self.source_configs[self.SOURCE_LIQUIDATION].enabled:
            try:
                await self.liquidation_monitor.start()
                logger.info("LiquidationMonitor started")
            except Exception as e:
                logger.error(f"Failed to start LiquidationMonitor: {e}")
                self.reliability[self.SOURCE_LIQUIDATION].record_failure(str(e))
        
        await self.refresh_all()
        
        self._update_task = asyncio.create_task(self._background_refresh_loop())
        logger.info("ExternalIntelligenceManager started successfully")
    
    async def stop(self) -> None:
        """Stop the intelligence manager and cleanup"""
        if not self._running:
            return
        
        self._running = False
        logger.info("Stopping ExternalIntelligenceManager...")
        
        if self._update_task and not self._update_task.done():
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        if self.whale_tracker:
            await self.whale_tracker.stop()
        
        if self.liquidation_monitor:
            await self.liquidation_monitor.stop()
        
        await self.fear_greed_client.close()
        await self.market_data_client.close()
        await self.news_sentiment_client.close()
        await self.derivatives_client.close()
        await self.economic_calendar_client.close()
        
        logger.info("ExternalIntelligenceManager stopped")
    
    async def _background_refresh_loop(self) -> None:
        """Background task to refresh data sources based on their intervals"""
        while self._running:
            try:
                await asyncio.sleep(10)
                
                now = datetime.now()
                refresh_tasks = []
                
                for source_name, config in self.source_configs.items():
                    if not config.enabled:
                        continue
                    
                    last_fetch = self._last_fetch.get(source_name)
                    if last_fetch is None:
                        continue
                    
                    elapsed = (now - last_fetch).total_seconds()
                    if elapsed >= config.refresh_interval_seconds:
                        refresh_tasks.append(self._fetch_source(source_name))
                
                if refresh_tasks:
                    await asyncio.gather(*refresh_tasks, return_exceptions=True)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background refresh error: {e}")
    
    async def refresh_all(self, force: bool = False) -> Dict[str, bool]:
        """
        Refresh all enabled data sources.
        
        Args:
            force: Force refresh even if cache is valid
            
        Returns:
            Dict mapping source names to success status
        """
        results = {}
        tasks = []
        
        for source_name, config in self.source_configs.items():
            if config.enabled:
                tasks.append(self._fetch_source(source_name, force=force))
        
        if tasks:
            fetch_results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, source_name in enumerate([n for n, c in self.source_configs.items() if c.enabled]):
                result = fetch_results[i]
                if isinstance(result, Exception):
                    results[source_name] = False
                else:
                    results[source_name] = result
        
        return results
    
    async def _fetch_source(self, source_name: str, force: bool = False) -> bool:
        """Fetch data from a specific source with timing and error tracking"""
        config = self.source_configs.get(source_name)
        if not config or not config.enabled:
            return False
        
        start_time = time.time()
        
        try:
            signal = await self._fetch_signal_from_source(source_name)
            
            elapsed_ms = (time.time() - start_time) * 1000
            self.reliability[source_name].record_success(elapsed_ms)
            
            if signal:
                self._signal_cache[source_name] = signal
                self._last_fetch[source_name] = datetime.now()
                logger.debug(f"Fetched {source_name}: {signal.direction.value} (strength={signal.strength:.2f})")
                return True
            
            return False
            
        except asyncio.TimeoutError:
            self.reliability[source_name].record_failure("Timeout")
            logger.warning(f"Timeout fetching {source_name}")
            return False
        except Exception as e:
            self.reliability[source_name].record_failure(str(e))
            logger.error(f"Error fetching {source_name}: {e}")
            return False
    
    async def _fetch_signal_from_source(self, source_name: str) -> Optional[SourceSignal]:
        """Fetch and convert data from a source to a SourceSignal"""
        
        if source_name == self.SOURCE_FEAR_GREED:
            return await self._fetch_fear_greed()
        elif source_name == self.SOURCE_MARKET_DATA:
            return await self._fetch_market_data()
        elif source_name == self.SOURCE_NEWS_SENTIMENT:
            return await self._fetch_news_sentiment()
        elif source_name == self.SOURCE_DERIVATIVES:
            return await self._fetch_derivatives()
        elif source_name == self.SOURCE_WHALE_TRACKER:
            return self._get_whale_signal()
        elif source_name == self.SOURCE_ECONOMIC_CALENDAR:
            return await self._fetch_economic_calendar()
        elif source_name == self.SOURCE_LIQUIDATION:
            return self._get_liquidation_signal()
        
        return None
    
    async def _fetch_fear_greed(self) -> Optional[SourceSignal]:
        """Fetch Fear & Greed Index signal"""
        data = await self.fear_greed_client.get_current()
        if not data:
            return None
        
        direction = SourceDirection.NEUTRAL
        strength = 0.5
        
        if data.is_extreme_fear:
            direction = SourceDirection.BULLISH
            strength = 1.0 - (data.value / 100.0)
        elif data.is_fear:
            direction = SourceDirection.BULLISH
            strength = 0.6
        elif data.is_greed:
            direction = SourceDirection.BEARISH
            strength = 0.6
        elif data.is_extreme_greed:
            direction = SourceDirection.BEARISH
            strength = data.value / 100.0
        
        return SourceSignal(
            source_name=self.SOURCE_FEAR_GREED,
            direction=direction,
            strength=strength,
            raw_data=data,
            metadata={
                'value': data.value,
                'classification': data.value_classification
            }
        )
    
    async def _fetch_market_data(self) -> Optional[SourceSignal]:
        """Fetch global market data signal"""
        data = await self.market_data_client.get_global_market_data()
        if not data:
            return None
        
        direction = SourceDirection.NEUTRAL
        strength = 0.5
        
        change_24h = data.market_cap_change_percentage_24h
        
        if change_24h > 3.0:
            direction = SourceDirection.BULLISH
            strength = min(1.0, 0.5 + (change_24h / 10.0))
        elif change_24h > 1.0:
            direction = SourceDirection.BULLISH
            strength = 0.6
        elif change_24h < -3.0:
            direction = SourceDirection.BEARISH
            strength = min(1.0, 0.5 + (abs(change_24h) / 10.0))
        elif change_24h < -1.0:
            direction = SourceDirection.BEARISH
            strength = 0.6
        
        return SourceSignal(
            source_name=self.SOURCE_MARKET_DATA,
            direction=direction,
            strength=strength,
            raw_data=data,
            metadata={
                'market_cap_change_24h': change_24h,
                'btc_dominance': data.btc_dominance
            }
        )
    
    async def _fetch_news_sentiment(self) -> Optional[SourceSignal]:
        """Fetch news sentiment signal"""
        summary = await self.news_sentiment_client.get_sentiment_summary()
        if not summary:
            return None
        
        direction = SourceDirection.NEUTRAL
        strength = 0.5
        
        avg_sentiment = summary.average_sentiment
        
        if avg_sentiment > 0.4:
            direction = SourceDirection.BULLISH
            strength = min(1.0, 0.5 + avg_sentiment)
        elif avg_sentiment > 0.1:
            direction = SourceDirection.BULLISH
            strength = 0.6
        elif avg_sentiment < -0.4:
            direction = SourceDirection.BEARISH
            strength = min(1.0, 0.5 + abs(avg_sentiment))
        elif avg_sentiment < -0.1:
            direction = SourceDirection.BEARISH
            strength = 0.6
        
        return SourceSignal(
            source_name=self.SOURCE_NEWS_SENTIMENT,
            direction=direction,
            strength=strength,
            raw_data=summary,
            metadata={
                'average_sentiment': avg_sentiment,
                'sentiment_label': summary.sentiment_label,
                'total_news': summary.total_news
            }
        )
    
    async def _fetch_derivatives(self) -> Optional[SourceSignal]:
        """Fetch derivatives data signal"""
        data = await self.derivatives_client.get_derivatives_intelligence(self.symbol)
        if not data:
            return None
        
        direction = SourceDirection.NEUTRAL
        strength = 0.5
        
        score = data.derivatives_score
        
        if score > 0.4:
            direction = SourceDirection.BULLISH
            strength = min(1.0, 0.5 + score)
        elif score > 0.1:
            direction = SourceDirection.BULLISH
            strength = 0.6
        elif score < -0.4:
            direction = SourceDirection.BEARISH
            strength = min(1.0, 0.5 + abs(score))
        elif score < -0.1:
            direction = SourceDirection.BEARISH
            strength = 0.6
        
        return SourceSignal(
            source_name=self.SOURCE_DERIVATIVES,
            direction=direction,
            strength=strength,
            raw_data=data,
            metadata={
                'derivatives_score': score,
                'market_sentiment': data.market_sentiment,
                'funding_rate': data.funding_rate,
                'long_short_ratio': data.long_short_ratio
            }
        )
    
    def _get_whale_signal(self) -> Optional[SourceSignal]:
        """Get whale tracker signal (from WebSocket data)"""
        if not self.whale_tracker:
            return None
        
        metrics = self.whale_tracker.get_current_metrics()
        if not metrics:
            return None
        
        direction = SourceDirection.NEUTRAL
        strength = 0.5
        
        bias = metrics.whale_bias
        smart_direction = metrics.smart_money_direction
        
        if smart_direction in ("ACCUMULATING", "BULLISH"):
            direction = SourceDirection.BULLISH
            strength = min(1.0, 0.6 + abs(bias))
        elif smart_direction == "SLIGHTLY_BULLISH":
            direction = SourceDirection.BULLISH
            strength = 0.55
        elif smart_direction in ("DISTRIBUTING", "BEARISH"):
            direction = SourceDirection.BEARISH
            strength = min(1.0, 0.6 + abs(bias))
        elif smart_direction == "SLIGHTLY_BEARISH":
            direction = SourceDirection.BEARISH
            strength = 0.55
        
        return SourceSignal(
            source_name=self.SOURCE_WHALE_TRACKER,
            direction=direction,
            strength=strength,
            raw_data=metrics,
            metadata={
                'whale_bias': bias,
                'smart_money_direction': smart_direction,
                'net_whale_flow': metrics.net_whale_flow
            }
        )
    
    async def _fetch_economic_calendar(self) -> Optional[SourceSignal]:
        """Fetch economic calendar signal"""
        summary = await self.economic_calendar_client.get_summary()
        if not summary:
            return None
        
        direction = SourceDirection.NEUTRAL
        strength = 0.5
        
        if summary.has_imminent_event:
            direction = SourceDirection.NEUTRAL
            strength = 0.2
        elif summary.should_avoid_trading:
            direction = SourceDirection.NEUTRAL
            strength = 0.3
        
        return SourceSignal(
            source_name=self.SOURCE_ECONOMIC_CALENDAR,
            direction=direction,
            strength=strength,
            raw_data=summary,
            metadata={
                'has_imminent_event': summary.has_imminent_event,
                'should_avoid_trading': summary.should_avoid_trading,
                'high_impact_events': summary.high_impact_events,
                'trading_recommendation': summary.trading_recommendation
            }
        )
    
    def _get_liquidation_signal(self) -> Optional[SourceSignal]:
        """Get liquidation monitor signal (from WebSocket data)"""
        if not self.liquidation_monitor:
            return None
        
        metrics = self.liquidation_monitor.get_metrics()
        if not metrics:
            return None
        
        direction = SourceDirection.NEUTRAL
        strength = 0.5
        
        bias = metrics.signal_bias
        imbalance = metrics.liquidation_imbalance
        
        if bias == "bullish":
            direction = SourceDirection.BULLISH
            strength = min(1.0, 0.6 + abs(imbalance))
        elif bias == "bearish":
            direction = SourceDirection.BEARISH
            strength = min(1.0, 0.6 + abs(imbalance))
        
        if metrics.liquidation_intensity == "extreme":
            strength *= 0.5
        elif metrics.liquidation_intensity == "high":
            strength *= 0.7
        
        return SourceSignal(
            source_name=self.SOURCE_LIQUIDATION,
            direction=direction,
            strength=strength,
            raw_data=metrics,
            metadata={
                'signal_bias': bias,
                'liquidation_imbalance': imbalance,
                'liquidation_intensity': metrics.liquidation_intensity,
                'total_liquidations_usd': metrics.total_liquidations_usd
            }
        )
    
    def _calculate_alignment(
        self,
        signals: Dict[str, SourceSignal]
    ) -> tuple[int, int, int]:
        """
        Calculate alignment between signals.
        
        Returns:
            Tuple of (bullish_count, bearish_count, neutral_count)
        """
        bullish = 0
        bearish = 0
        neutral = 0
        
        for signal in signals.values():
            if signal.direction == SourceDirection.BULLISH:
                bullish += 1
            elif signal.direction == SourceDirection.BEARISH:
                bearish += 1
            else:
                neutral += 1
        
        return bullish, bearish, neutral
    
    def _calculate_weighted_direction(
        self,
        signals: Dict[str, SourceSignal]
    ) -> tuple[SourceDirection, float]:
        """
        Calculate weighted overall direction and confidence.
        
        Returns:
            Tuple of (direction, base_confidence)
        """
        bullish_weight = 0.0
        bearish_weight = 0.0
        total_weight = 0.0
        
        for source_name, signal in signals.items():
            config = self.source_configs.get(source_name)
            reliability = self.reliability.get(source_name)
            
            if not config or not reliability:
                continue
            
            source_weight = config.weight * reliability.reliability_weight * signal.strength
            total_weight += source_weight
            
            if signal.direction == SourceDirection.BULLISH:
                bullish_weight += source_weight
            elif signal.direction == SourceDirection.BEARISH:
                bearish_weight += source_weight
        
        if total_weight == 0:
            return SourceDirection.NEUTRAL, 0.5
        
        if bullish_weight > bearish_weight * 1.2:
            direction = SourceDirection.BULLISH
            confidence = bullish_weight / total_weight
        elif bearish_weight > bullish_weight * 1.2:
            direction = SourceDirection.BEARISH
            confidence = bearish_weight / total_weight
        else:
            direction = SourceDirection.NEUTRAL
            confidence = 0.5
        
        return direction, min(1.0, confidence)
    
    def get_aggregated_intelligence(self) -> AggregatedIntelligence:
        """
        Get aggregated intelligence from all sources.
        
        Returns:
            AggregatedIntelligence with overall direction, confidence, and alignment data
        """
        available_signals = {
            name: signal for name, signal in self._signal_cache.items()
            if signal and self.source_configs.get(name, SourceConfig(name=name)).enabled
        }
        
        if not available_signals:
            return AggregatedIntelligence(
                timestamp=datetime.now(),
                overall_direction=SourceDirection.UNAVAILABLE,
                overall_confidence=0.0,
                alignment_count=0,
                alignment_boost=0.0,
                conflict_penalty=0.0,
                signals={},
                reliability_scores={},
                health_status={n: r.health.value for n, r in self.reliability.items()},
                warnings=["No data sources available"],
                should_avoid_trading=True
            )
        
        bullish_count, bearish_count, neutral_count = self._calculate_alignment(available_signals)
        
        direction, base_confidence = self._calculate_weighted_direction(available_signals)
        
        alignment_boost = 0.0
        conflict_penalty = 0.0
        
        max_alignment = max(bullish_count, bearish_count)
        
        if max_alignment >= 4:
            alignment_boost = self.ALIGNMENT_BOOST_4
            logger.info(f"Strong alignment detected: {max_alignment} sources agree")
        elif max_alignment >= 3:
            alignment_boost = self.ALIGNMENT_BOOST_3
            logger.info(f"Good alignment detected: {max_alignment} sources agree")
        
        if bullish_count >= 2 and bearish_count >= 2:
            conflict_penalty = self.CONFLICT_PENALTY
            logger.info(f"Source conflict detected: {bullish_count} bullish vs {bearish_count} bearish")
        
        final_confidence = min(1.0, max(0.0, base_confidence + alignment_boost - conflict_penalty))
        
        warnings = []
        should_avoid_trading = False
        
        calendar_signal = available_signals.get(self.SOURCE_ECONOMIC_CALENDAR)
        if calendar_signal and calendar_signal.raw_data:
            summary = calendar_signal.raw_data
            if hasattr(summary, 'should_avoid_trading') and summary.should_avoid_trading:
                should_avoid_trading = True
                warnings.append("High-impact economic event imminent")
            if hasattr(summary, 'trading_recommendation'):
                warnings.append(summary.trading_recommendation)
        
        for name, reliability in self.reliability.items():
            if reliability.health in (SourceHealth.UNHEALTHY, SourceHealth.OFFLINE):
                warnings.append(f"Source {name} is {reliability.health.value}")
        
        liq_signal = available_signals.get(self.SOURCE_LIQUIDATION)
        if liq_signal and liq_signal.metadata.get('liquidation_intensity') == 'extreme':
            warnings.append("Extreme liquidation activity detected")
        
        return AggregatedIntelligence(
            timestamp=datetime.now(),
            overall_direction=direction,
            overall_confidence=final_confidence,
            alignment_count=max_alignment,
            alignment_boost=alignment_boost,
            conflict_penalty=conflict_penalty,
            signals=available_signals,
            reliability_scores={
                name: rel.reliability_weight
                for name, rel in self.reliability.items()
            },
            health_status={n: r.health.value for n, r in self.reliability.items()},
            warnings=warnings,
            should_avoid_trading=should_avoid_trading
        )
    
    def get_signal_for_engine(self) -> Dict[str, Any]:
        """
        Get a simplified signal for SignalEngine consumption.
        
        Returns:
            Dict with direction_bias, confidence_adjustment, and metadata
        """
        intel = self.get_aggregated_intelligence()
        
        direction_bias = 0.0
        if intel.overall_direction == SourceDirection.BULLISH:
            direction_bias = intel.overall_confidence
        elif intel.overall_direction == SourceDirection.BEARISH:
            direction_bias = -intel.overall_confidence
        
        confidence_adjustment = intel.alignment_boost - intel.conflict_penalty
        
        return {
            'direction_bias': direction_bias,
            'confidence_adjustment': confidence_adjustment,
            'overall_direction': intel.overall_direction.value,
            'overall_confidence': intel.overall_confidence,
            'alignment_count': intel.alignment_count,
            'should_avoid_trading': intel.should_avoid_trading,
            'warnings': intel.warnings,
            'source_signals': {
                name: {
                    'direction': sig.direction.value,
                    'strength': sig.strength
                }
                for name, sig in intel.signals.items()
            },
            'timestamp': intel.timestamp.isoformat()
        }
    
    def get_health_report(self) -> Dict[str, Any]:
        """
        Get comprehensive health report for all sources.
        
        Returns:
            Dict with health metrics for each source and overall status
        """
        sources_report = {
            name: reliability.to_dict()
            for name, reliability in self.reliability.items()
        }
        
        healthy_count = sum(
            1 for r in self.reliability.values()
            if r.health == SourceHealth.HEALTHY
        )
        degraded_count = sum(
            1 for r in self.reliability.values()
            if r.health == SourceHealth.DEGRADED
        )
        unhealthy_count = sum(
            1 for r in self.reliability.values()
            if r.health in (SourceHealth.UNHEALTHY, SourceHealth.OFFLINE)
        )
        
        total_sources = len(self.reliability)
        
        if unhealthy_count == 0 and degraded_count == 0:
            overall_status = "healthy"
        elif unhealthy_count >= total_sources // 2:
            overall_status = "critical"
        elif unhealthy_count > 0:
            overall_status = "degraded"
        elif degraded_count > 0:
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        return {
            'overall_status': overall_status,
            'healthy_sources': healthy_count,
            'degraded_sources': degraded_count,
            'unhealthy_sources': unhealthy_count,
            'total_sources': total_sources,
            'sources': sources_report,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_cached_signal(self, source_name: str) -> Optional[SourceSignal]:
        """Get cached signal for a specific source"""
        return self._signal_cache.get(source_name)
    
    def is_source_healthy(self, source_name: str) -> bool:
        """Check if a specific source is healthy"""
        reliability = self.reliability.get(source_name)
        if not reliability:
            return False
        return reliability.health in (SourceHealth.HEALTHY, SourceHealth.DEGRADED)
    
    def get_available_sources(self) -> List[str]:
        """Get list of available/enabled source names"""
        return [
            name for name, config in self.source_configs.items()
            if config.enabled and self.is_source_healthy(name)
        ]
    
    @property
    def is_running(self) -> bool:
        """Check if manager is running"""
        return self._running


async def create_intelligence_manager(
    symbol: str = "BTCUSDT",
    enable_websocket_sources: bool = False,
    auto_start: bool = True
) -> ExternalIntelligenceManager:
    """
    Factory function to create and optionally start an ExternalIntelligenceManager.
    
    Args:
        symbol: Primary trading symbol
        enable_websocket_sources: Enable WebSocket sources (Whale, Liquidation)
        auto_start: Start the manager immediately
        
    Returns:
        ExternalIntelligenceManager instance
    """
    manager = ExternalIntelligenceManager(
        symbol=symbol,
        enable_websocket_sources=enable_websocket_sources
    )
    
    if auto_start:
        await manager.start()
    
    return manager
