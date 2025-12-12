"""
Multi-Asset Scanner for Binance USDT-M Futures

Scans top tradeable USDT-M futures pairs for trading signals using
the UT Bot + STC strategy. Fetches data in parallel, runs signal
engine on each symbol, and ranks opportunities by confidence.
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd

from ..data.binance_fetcher import BinanceDataFetcher
from ..engine.signal_engine import SignalEngine
from ..external_data.fear_greed_client import FearGreedClient, FearGreedData
from ..external_data.news_sentiment_client import NewsSentimentClient, NewsSentimentSummary
from ..external_data.market_data_aggregator import MarketDataAggregator

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """Result from scanning a single asset for trading signals"""
    symbol: str
    signal_type: str  # 'LONG', 'SHORT', or 'NONE'
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    volume_24h: float
    volatility: float
    fear_greed_alignment: bool
    news_sentiment_score: float
    market_breadth_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    risk_percent: float = 0.0
    reward_percent: float = 0.0
    order_flow_bias: float = 0.0
    manipulation_score: float = 0.0
    signal_details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_signal(self) -> bool:
        return self.signal_type in ('LONG', 'SHORT')
    
    @property
    def composite_score(self) -> float:
        """Calculate composite opportunity score (0-100)"""
        if not self.has_signal:
            return 0.0
        
        base_score = self.confidence * 40
        
        volume_score = min(self.volume_24h / 1e9, 1.0) * 10
        
        volatility_score = min(self.volatility / 5.0, 1.0) * 10 if self.volatility > 0.5 else 0
        
        sentiment_score = 0
        if self.fear_greed_alignment:
            sentiment_score += 10
        if abs(self.news_sentiment_score) > 0.2:
            if (self.signal_type == 'LONG' and self.news_sentiment_score > 0) or \
               (self.signal_type == 'SHORT' and self.news_sentiment_score < 0):
                sentiment_score += 10
        
        breadth_score = self.market_breadth_score * 10
        
        manipulation_penalty = self.manipulation_score * 10
        
        total = base_score + volume_score + volatility_score + sentiment_score + breadth_score - manipulation_penalty
        return max(0.0, min(100.0, total))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type,
            'confidence': self.confidence,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'volume_24h': self.volume_24h,
            'volatility': self.volatility,
            'fear_greed_alignment': self.fear_greed_alignment,
            'news_sentiment_score': self.news_sentiment_score,
            'market_breadth_score': self.market_breadth_score,
            'timestamp': self.timestamp.isoformat(),
            'risk_percent': self.risk_percent,
            'reward_percent': self.reward_percent,
            'composite_score': self.composite_score,
            'order_flow_bias': self.order_flow_bias,
            'manipulation_score': self.manipulation_score
        }


@dataclass
class ScannerConfig:
    """Configuration for the multi-asset scanner"""
    timeframe: str = '5m'
    candles_to_fetch: int = 200
    min_confidence: float = 0.5
    max_concurrent_scans: int = 5
    scan_interval_seconds: int = 60
    enable_fear_greed: bool = True
    enable_news_sentiment: bool = True
    enable_market_data: bool = True


class MultiAssetScanner:
    """
    Multi-Asset Scanner for Binance USDT-M Futures
    
    Scans top tradeable futures pairs for trading signals using
    the UT Bot + STC strategy with order flow analysis integration.
    
    Features:
    - Parallel data fetching using asyncio
    - Signal generation for each symbol
    - Integration with fear/greed index
    - News sentiment analysis
    - Market breadth scoring
    - Opportunity ranking
    """
    
    DEFAULT_WATCHLIST = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
        'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT',
        'MATICUSDT', 'LTCUSDT', 'SHIBUSDT', 'UNIUSDT', 'ATOMUSDT',
        'XLMUSDT', 'ETCUSDT', 'FILUSDT', 'APTUSDT', 'NEARUSDT'
    ]
    
    SYMBOL_TO_CURRENCY = {
        'BTCUSDT': 'BTC',
        'ETHUSDT': 'ETH',
        'BNBUSDT': 'BNB',
        'SOLUSDT': 'SOL',
        'XRPUSDT': 'XRP',
        'DOGEUSDT': 'DOGE',
        'ADAUSDT': 'ADA',
        'AVAXUSDT': 'AVAX',
        'LINKUSDT': 'LINK',
        'DOTUSDT': 'DOT',
        'MATICUSDT': 'MATIC',
        'LTCUSDT': 'LTC',
        'SHIBUSDT': 'SHIB',
        'UNIUSDT': 'UNI',
        'ATOMUSDT': 'ATOM',
        'XLMUSDT': 'XLM',
        'ETCUSDT': 'ETC',
        'FILUSDT': 'FIL',
        'APTUSDT': 'APT',
        'NEARUSDT': 'NEAR'
    }
    
    def __init__(
        self,
        config: Optional[ScannerConfig] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        watchlist: Optional[List[str]] = None
    ):
        """
        Initialize Multi-Asset Scanner
        
        Args:
            config: Scanner configuration
            api_key: Binance API key
            api_secret: Binance API secret
            watchlist: Custom watchlist of symbols
        """
        self.config = config or ScannerConfig()
        self.api_key = api_key or os.getenv('BINANCE_API_KEY', '')
        self.api_secret = api_secret or os.getenv('BINANCE_API_SECRET', '')
        self.watchlist = list(watchlist) if watchlist else list(self.DEFAULT_WATCHLIST)
        
        self._fetchers: Dict[str, BinanceDataFetcher] = {}
        self._signal_engines: Dict[str, SignalEngine] = {}
        
        self._fear_greed_client: Optional[FearGreedClient] = None
        self._news_client: Optional[NewsSentimentClient] = None
        self._market_data_client: Optional[MarketDataAggregator] = None
        
        self._last_scan_results: Dict[str, ScanResult] = {}
        self._scan_history: List[Dict[str, Any]] = []
        self._last_scan_time: Optional[datetime] = None
        
        self._fear_greed_data: Optional[FearGreedData] = None
        self._news_sentiment: Optional[NewsSentimentSummary] = None
        self._market_breadth: float = 0.5
        
        self._initialized = False
        
        logger.info(f"MultiAssetScanner initialized with {len(self.watchlist)} symbols")
    
    async def initialize(self) -> None:
        """Initialize all clients and prepare for scanning"""
        if self._initialized:
            return
        
        for symbol in self.watchlist:
            self._fetchers[symbol] = BinanceDataFetcher(
                api_key=self.api_key,
                api_secret=self.api_secret,
                symbol=symbol,
                interval=self.config.timeframe
            )
            self._signal_engines[symbol] = SignalEngine(
                order_flow_enabled=True,
                manipulation_filter_enabled=True
            )
        
        if self.config.enable_fear_greed:
            self._fear_greed_client = FearGreedClient()
        
        if self.config.enable_news_sentiment:
            self._news_client = NewsSentimentClient()
        
        if self.config.enable_market_data:
            self._market_data_client = MarketDataAggregator()
        
        self._initialized = True
        logger.info("MultiAssetScanner fully initialized")
    
    async def close(self) -> None:
        """Close all clients and release resources"""
        close_tasks = []
        
        for fetcher in self._fetchers.values():
            close_tasks.append(fetcher.close())
        
        if self._fear_greed_client:
            close_tasks.append(self._fear_greed_client.close())
        
        if self._news_client:
            close_tasks.append(self._news_client.close())
        
        if self._market_data_client:
            close_tasks.append(self._market_data_client.close())
        
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        
        self._initialized = False
        logger.info("MultiAssetScanner closed")
    
    def add_symbol(self, symbol: str) -> bool:
        """
        Add a symbol to the watchlist
        
        Args:
            symbol: Symbol to add (e.g., 'ARBUSDT')
            
        Returns:
            True if added, False if already exists
        """
        symbol = symbol.upper()
        if symbol in self.watchlist:
            logger.warning(f"Symbol {symbol} already in watchlist")
            return False
        
        self.watchlist.append(symbol)
        
        if self._initialized:
            self._fetchers[symbol] = BinanceDataFetcher(
                api_key=self.api_key,
                api_secret=self.api_secret,
                symbol=symbol,
                interval=self.config.timeframe
            )
            self._signal_engines[symbol] = SignalEngine(
                order_flow_enabled=True,
                manipulation_filter_enabled=True
            )
        
        logger.info(f"Added {symbol} to watchlist")
        return True
    
    def remove_symbol(self, symbol: str) -> bool:
        """
        Remove a symbol from the watchlist
        
        Args:
            symbol: Symbol to remove
            
        Returns:
            True if removed, False if not found
        """
        symbol = symbol.upper()
        if symbol not in self.watchlist:
            logger.warning(f"Symbol {symbol} not in watchlist")
            return False
        
        self.watchlist.remove(symbol)
        
        if symbol in self._fetchers:
            del self._fetchers[symbol]
        if symbol in self._signal_engines:
            del self._signal_engines[symbol]
        if symbol in self._last_scan_results:
            del self._last_scan_results[symbol]
        
        logger.info(f"Removed {symbol} from watchlist")
        return True
    
    async def _fetch_external_data(self) -> None:
        """Fetch fear/greed, news sentiment, and market data"""
        tasks = []
        
        if self._fear_greed_client:
            tasks.append(self._fear_greed_client.get_current())
        
        if self._news_client:
            tasks.append(self._news_client.get_sentiment_summary())
        
        if self._market_data_client:
            tasks.append(self._market_data_client.get_global_market_data())
        
        if not tasks:
            return
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        idx = 0
        if self._fear_greed_client:
            result = results[idx]
            if not isinstance(result, BaseException) and result is not None:
                self._fear_greed_data = result
            idx += 1
        
        if self._news_client:
            result = results[idx]
            if not isinstance(result, BaseException) and result is not None:
                self._news_sentiment = result
            idx += 1
        
        if self._market_data_client and idx < len(results):
            result = results[idx]
            if not isinstance(result, BaseException) and result is not None:
                global_data = result
                if hasattr(global_data, 'market_cap_change_percentage_24h'):
                    change = global_data.market_cap_change_percentage_24h
                    self._market_breadth = 0.5 + (change / 20)
                    self._market_breadth = max(0.0, min(1.0, self._market_breadth))
    
    def _check_fear_greed_alignment(self, signal_type: str) -> bool:
        """Check if signal aligns with fear/greed sentiment"""
        if not self._fear_greed_data:
            return True
        
        value = self._fear_greed_data.value
        
        if signal_type == 'LONG':
            return value < 75
        elif signal_type == 'SHORT':
            return value > 25
        
        return True
    
    def _get_news_sentiment_score(self, symbol: str) -> float:
        """Get news sentiment score for a symbol"""
        if not self._news_sentiment:
            return 0.0
        
        return self._news_sentiment.average_sentiment
    
    async def get_symbol_data(self, symbol: str, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Get OHLCV data for a specific symbol
        
        Args:
            symbol: Trading pair symbol
            force_refresh: Force fresh data fetch
            
        Returns:
            DataFrame with OHLCV data or None
        """
        symbol = symbol.upper()
        
        if symbol not in self._fetchers:
            if not self._initialized:
                await self.initialize()
            
            if symbol not in self._fetchers:
                self._fetchers[symbol] = BinanceDataFetcher(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    symbol=symbol,
                    interval=self.config.timeframe
                )
        
        try:
            return self._fetchers[symbol].get_cached_or_fresh_data(
                limit=self.config.candles_to_fetch,
                force_refresh=force_refresh
            )
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    async def _scan_symbol(self, symbol: str) -> Optional[ScanResult]:
        """
        Scan a single symbol for trading signals
        
        Args:
            symbol: Symbol to scan
            
        Returns:
            ScanResult or None if scan failed
        """
        try:
            df = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._fetchers[symbol].get_cached_or_fresh_data(
                    limit=self.config.candles_to_fetch,
                    force_refresh=True
                )
            )
            
            if df is None or len(df) < 50:
                logger.warning(f"Insufficient data for {symbol}")
                return None
            
            engine = self._signal_engines[symbol]
            df_with_indicators = engine.calculate_indicators(df)
            
            signal = engine.generate_signal(df_with_indicators)
            
            if signal is None or signal.get('type') == 'NONE':
                return ScanResult(
                    symbol=symbol,
                    signal_type='NONE',
                    confidence=0.0,
                    entry_price=float(df['close'].iloc[-1]),
                    stop_loss=0.0,
                    take_profit=0.0,
                    volume_24h=float(df['volume'].tail(288).sum()) if len(df) >= 288 else float(df['volume'].sum()),
                    volatility=self._calculate_volatility(df),
                    fear_greed_alignment=True,
                    news_sentiment_score=self._get_news_sentiment_score(symbol),
                    market_breadth_score=self._market_breadth
                )
            
            signal_type = signal.get('type', 'NONE')
            fear_greed_aligned = self._check_fear_greed_alignment(signal_type)
            news_sentiment = self._get_news_sentiment_score(symbol)
            
            volume_24h = float(df['volume'].tail(288).sum()) if len(df) >= 288 else float(df['volume'].sum())
            volatility = self._calculate_volatility(df)
            
            confidence = signal.get('enhanced_confidence', signal.get('confidence', 0.7))
            
            result = ScanResult(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                entry_price=signal.get('entry_price', float(df['close'].iloc[-1])),
                stop_loss=signal.get('stop_loss', 0.0),
                take_profit=signal.get('take_profit', 0.0),
                volume_24h=volume_24h,
                volatility=volatility,
                fear_greed_alignment=fear_greed_aligned,
                news_sentiment_score=news_sentiment,
                market_breadth_score=self._market_breadth,
                risk_percent=signal.get('risk_percent', 0.0),
                reward_percent=signal.get('reward_percent', 0.0),
                order_flow_bias=signal.get('order_flow_bias', 0.0),
                manipulation_score=signal.get('manipulation_score', 0.0),
                signal_details=signal
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")
            return None
    
    def _calculate_volatility(self, df: pd.DataFrame, period: int = 20) -> float:
        """Calculate volatility as percentage"""
        if len(df) < period:
            return 0.0
        
        returns = df['close'].pct_change().dropna().tail(period)
        if len(returns) == 0:
            return 0.0
        
        return float(returns.std() * 100 * (252 ** 0.5))
    
    async def scan_all(self) -> List[ScanResult]:
        """
        Scan all symbols in watchlist and return ranked opportunities
        
        Returns:
            List of ScanResult sorted by composite score
        """
        if not self._initialized:
            await self.initialize()
        
        await self._fetch_external_data()
        
        semaphore = asyncio.Semaphore(self.config.max_concurrent_scans)
        
        async def scan_with_limit(symbol: str) -> Optional[ScanResult]:
            async with semaphore:
                return await self._scan_symbol(symbol)
        
        tasks = [scan_with_limit(symbol) for symbol in self.watchlist]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        scan_results = []
        for result in results:
            if isinstance(result, ScanResult):
                scan_results.append(result)
                self._last_scan_results[result.symbol] = result
            elif isinstance(result, Exception):
                logger.error(f"Scan task failed: {result}")
        
        scan_results.sort(key=lambda x: x.composite_score, reverse=True)
        
        self._last_scan_time = datetime.now()
        self._scan_history.append({
            'timestamp': self._last_scan_time,
            'total_scanned': len(scan_results),
            'signals_found': sum(1 for r in scan_results if r.has_signal),
            'top_opportunity': scan_results[0].symbol if scan_results else None
        })
        
        if len(self._scan_history) > 100:
            self._scan_history = self._scan_history[-100:]
        
        logger.info(f"Scan complete: {len(scan_results)} symbols, "
                   f"{sum(1 for r in scan_results if r.has_signal)} signals found")
        
        return scan_results
    
    async def get_top_opportunities(self, n: int = 5) -> List[ScanResult]:
        """
        Get top N trading opportunities
        
        Args:
            n: Number of top opportunities to return
            
        Returns:
            List of top N ScanResult with signals
        """
        results = await self.scan_all()
        
        opportunities = [r for r in results if r.has_signal and r.confidence >= self.config.min_confidence]
        
        return opportunities[:n]
    
    def get_last_scan_results(self) -> Dict[str, ScanResult]:
        """Get results from the last scan"""
        return self._last_scan_results.copy()
    
    def get_scan_history(self) -> List[Dict[str, Any]]:
        """Get scan history"""
        return self._scan_history.copy()
    
    def get_watchlist(self) -> List[str]:
        """Get current watchlist"""
        return self.watchlist.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """Get scanner status"""
        return {
            'initialized': self._initialized,
            'watchlist_count': len(self.watchlist),
            'last_scan_time': self._last_scan_time.isoformat() if self._last_scan_time else None,
            'fear_greed_value': self._fear_greed_data.value if self._fear_greed_data else None,
            'fear_greed_classification': self._fear_greed_data.value_classification if self._fear_greed_data else None,
            'news_sentiment': self._news_sentiment.sentiment_label if self._news_sentiment else None,
            'market_breadth': self._market_breadth,
            'total_scans': len(self._scan_history),
            'config': {
                'timeframe': self.config.timeframe,
                'min_confidence': self.config.min_confidence,
                'max_concurrent_scans': self.config.max_concurrent_scans
            }
        }
