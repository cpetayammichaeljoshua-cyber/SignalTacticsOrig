"""
External Data Package for Market Intelligence

Provides API clients for fetching external market data:
- Fear & Greed Index (Alternative.me)
- Market Data Aggregator (CoinGecko)
- News Sentiment (CryptoPanic)
- Binance Derivatives Data (Funding, OI, L/S Ratio)
- Binance Liquidation Monitor (Real-time WebSocket)
- Dynamic Futures Pairs Fetcher (Binance public API)
- Whale Tracker (Large trade detection and metrics)
- Economic Calendar (Finnhub API for macro event awareness)
"""

from ut_bot_strategy.external_data.fear_greed_client import (
    FearGreedClient,
    FearGreedData,
    FearGreedHistoryEntry
)
from ut_bot_strategy.external_data.market_data_aggregator import (
    MarketDataAggregator,
    TrendingCoin,
    CoinMarketData,
    GlobalMarketData
)
from ut_bot_strategy.external_data.news_sentiment_client import (
    NewsSentimentClient,
    NewsItem,
    NewsSentimentSummary
)
from ut_bot_strategy.external_data.derivatives_client import (
    BinanceDerivativesClient,
    DerivativesData,
    FundingRateEntry,
    OpenInterestData,
    LongShortRatioData,
    TakerVolumeData
)
from ut_bot_strategy.external_data.liquidation_monitor import (
    LiquidationMonitor,
    LiquidationEvent,
    LiquidationMetrics,
    create_liquidation_monitor
)
from ut_bot_strategy.external_data.dynamic_pairs_fetcher import (
    DynamicPairsFetcher,
    FuturesPair,
    create_pairs_fetcher
)
from ut_bot_strategy.external_data.whale_tracker import (
    WhaleTracker,
    WhaleTrade,
    WhaleMetrics,
    create_whale_tracker_from_order_flow,
    create_standalone_whale_tracker
)
from ut_bot_strategy.external_data.economic_calendar import (
    EconomicCalendarClient,
    EconomicEvent,
    EconomicCalendarSummary,
    EventImpact,
    check_economic_events_before_trade
)
from ut_bot_strategy.external_data.intelligence_manager import (
    ExternalIntelligenceManager,
    AggregatedIntelligence,
    SourceSignal,
    SourceReliability,
    SourceDirection,
    SourceHealth,
    SourceConfig,
    create_intelligence_manager
)

__all__ = [
    'FearGreedClient',
    'FearGreedData',
    'FearGreedHistoryEntry',
    'MarketDataAggregator',
    'TrendingCoin',
    'CoinMarketData',
    'GlobalMarketData',
    'NewsSentimentClient',
    'NewsItem',
    'NewsSentimentSummary',
    'BinanceDerivativesClient',
    'DerivativesData',
    'FundingRateEntry',
    'OpenInterestData',
    'LongShortRatioData',
    'TakerVolumeData',
    'LiquidationMonitor',
    'LiquidationEvent',
    'LiquidationMetrics',
    'create_liquidation_monitor',
    'DynamicPairsFetcher',
    'FuturesPair',
    'create_pairs_fetcher',
    'WhaleTracker',
    'WhaleTrade',
    'WhaleMetrics',
    'create_whale_tracker_from_order_flow',
    'create_standalone_whale_tracker',
    'EconomicCalendarClient',
    'EconomicEvent',
    'EconomicCalendarSummary',
    'EventImpact',
    'check_economic_events_before_trade',
    'ExternalIntelligenceManager',
    'AggregatedIntelligence',
    'SourceSignal',
    'SourceReliability',
    'SourceDirection',
    'SourceHealth',
    'SourceConfig',
    'create_intelligence_manager'
]
