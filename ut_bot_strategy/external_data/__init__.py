"""
External Data Package for Market Intelligence

Provides API clients for fetching external market data:
- Fear & Greed Index (Alternative.me)
- Market Data Aggregator (CoinGecko)
- News Sentiment (CryptoPanic)
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
    'NewsSentimentSummary'
]
