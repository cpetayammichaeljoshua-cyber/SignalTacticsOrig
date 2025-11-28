"""
Data Fetching Module

Contains data fetchers for various exchanges:
- BinanceDataFetcher: Fetch ETH/USDT data from Binance
"""

from .binance_fetcher import BinanceDataFetcher

__all__ = ['BinanceDataFetcher']
