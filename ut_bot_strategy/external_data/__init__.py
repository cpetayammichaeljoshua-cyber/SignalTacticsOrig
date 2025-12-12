"""
External Data Package for Market Intelligence

Provides API clients for fetching external market data:
- Fear & Greed Index (Alternative.me)
- Market Data Aggregator (CoinGecko)
- News Sentiment (CryptoPanic)
- Binance Derivatives Data (Funding, OI, L/S Ratio)
- Binance Liquidation Monitor (Real-time WebSocket)
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
    'create_liquidation_monitor'
]
