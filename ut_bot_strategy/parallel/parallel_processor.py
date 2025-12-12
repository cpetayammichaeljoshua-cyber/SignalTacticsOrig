"""
Enhanced Parallel Processing Module for Trading Signal Bot

Provides high-performance parallel processing capabilities for:
- Batch task processing with rate limiting and progress tracking
- Parallel market scanning of 100+ trading pairs
- Concurrent multi-API data fetching with fallback mechanisms

Features:
- Configurable concurrency limits using asyncio.Semaphore
- Dynamic batch sizing based on system load
- Priority queue for high-volume trading pairs
- Comprehensive error handling with partial results
- Performance metrics and cache management
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import (
    Optional, List, Dict, Any, Callable, Coroutine, 
    TypeVar, Generic, Tuple, Set
)
from collections import deque
from enum import Enum
import heapq

from ..external_data.dynamic_pairs_fetcher import DynamicPairsFetcher, FuturesPair
from ..external_data.fear_greed_client import FearGreedClient, FearGreedData
from ..external_data.market_data_aggregator import MarketDataAggregator, GlobalMarketData
from ..external_data.derivatives_client import BinanceDerivativesClient, DerivativesData
from ..data.binance_fetcher import BinanceDataFetcher
from ..engine.signal_engine import SignalEngine

logger = logging.getLogger(__name__)

MAX_CONCURRENT_SCANS: int = 25
MAX_CONCURRENT_API_CALLS: int = 10
BATCH_SIZE: int = 50
SCAN_TIMEOUT_SECONDS: int = 30

T = TypeVar('T')
R = TypeVar('R')


class TaskStatus(str, Enum):
    """Status of a task in batch processing"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class TaskResult(Generic[T]):
    """Result of a single task execution"""
    task_id: str
    status: TaskStatus
    result: Optional[T] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def success(self) -> bool:
        return self.status == TaskStatus.COMPLETED
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'status': self.status.value,
            'result': self.result,
            'error': self.error,
            'execution_time_ms': round(self.execution_time_ms, 2),
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class BatchMetrics:
    """Performance metrics for batch processing"""
    total_tasks: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_timeout: int = 0
    total_time_ms: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def avg_time_per_task(self) -> float:
        if self.tasks_completed == 0:
            return 0.0
        return self.total_time_ms / self.tasks_completed
    
    @property
    def success_rate(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.tasks_completed / self.total_tasks
    
    @property
    def throughput_per_second(self) -> float:
        if self.total_time_ms == 0:
            return 0.0
        return (self.tasks_completed / self.total_time_ms) * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_tasks': self.total_tasks,
            'tasks_completed': self.tasks_completed,
            'tasks_failed': self.tasks_failed,
            'tasks_timeout': self.tasks_timeout,
            'avg_time_per_task_ms': round(self.avg_time_per_task, 2),
            'success_rate': round(self.success_rate, 4),
            'throughput_per_second': round(self.throughput_per_second, 2),
            'total_time_ms': round(self.total_time_ms, 2),
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None
        }


@dataclass
class ScanResultEntry:
    """Result from scanning a single trading pair"""
    symbol: str
    signal_type: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    volume_24h: float
    priority_score: float
    scan_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None
    
    @property
    def has_signal(self) -> bool:
        return self.signal_type in ('LONG', 'SHORT')
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type,
            'confidence': round(self.confidence, 4),
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'volume_24h': self.volume_24h,
            'priority_score': round(self.priority_score, 4),
            'scan_time_ms': round(self.scan_time_ms, 2),
            'timestamp': self.timestamp.isoformat(),
            'error': self.error
        }


@dataclass
class APISourceResult:
    """Result from fetching data from an API source"""
    source_name: str
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    response_time_ms: float = 0.0
    from_cache: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


class ParallelBatchProcessor:
    """
    High-performance batch processor with configurable concurrency.
    
    Features:
    - Configurable concurrency limit via max_workers (default 20)
    - asyncio.Semaphore for rate limiting
    - Batch processing with progress tracking
    - Error handling with partial results
    - Performance metrics (tasks_completed, avg_time_per_task)
    
    Example:
        processor = ParallelBatchProcessor(max_workers=20)
        
        async def fetch_data(symbol: str) -> dict:
            return await api.get_data(symbol)
        
        tasks = [('BTC', fetch_data, ('BTCUSDT',)) for _ in range(100)]
        results = await processor.process_batch(tasks)
    """
    
    def __init__(
        self,
        max_workers: int = 20,
        task_timeout_seconds: float = 30.0,
        enable_progress_tracking: bool = True
    ):
        """
        Initialize the batch processor.
        
        Args:
            max_workers: Maximum concurrent tasks (default 20)
            task_timeout_seconds: Timeout per task in seconds (default 30)
            enable_progress_tracking: Enable progress callbacks (default True)
        """
        self.max_workers = max_workers
        self.task_timeout_seconds = task_timeout_seconds
        self.enable_progress_tracking = enable_progress_tracking
        
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._metrics = BatchMetrics()
        self._progress_callback: Optional[Callable[[int, int], None]] = None
        self._task_times: deque = deque(maxlen=1000)
        
        logger.info(f"ParallelBatchProcessor initialized (max_workers={max_workers})")
    
    def set_progress_callback(self, callback: Callable[[int, int], None]) -> None:
        """
        Set a callback for progress updates.
        
        Args:
            callback: Function(completed, total) called on each task completion
        """
        self._progress_callback = callback
    
    async def _execute_task(
        self,
        task_id: str,
        coro: Coroutine[Any, Any, T],
        completed_count: List[int],
        total: int
    ) -> TaskResult[T]:
        """Execute a single task with semaphore, timeout, and error handling."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_workers)
        
        start_time = time.perf_counter()
        
        async with self._semaphore:
            try:
                result = await asyncio.wait_for(
                    coro,
                    timeout=self.task_timeout_seconds
                )
                execution_time = (time.perf_counter() - start_time) * 1000
                self._task_times.append(execution_time)
                
                completed_count[0] += 1
                if self._progress_callback and self.enable_progress_tracking:
                    self._progress_callback(completed_count[0], total)
                
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED,
                    result=result,
                    execution_time_ms=execution_time
                )
                
            except asyncio.TimeoutError:
                execution_time = (time.perf_counter() - start_time) * 1000
                completed_count[0] += 1
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.TIMEOUT,
                    error=f"Task timed out after {self.task_timeout_seconds}s",
                    execution_time_ms=execution_time
                )
                
            except Exception as e:
                execution_time = (time.perf_counter() - start_time) * 1000
                completed_count[0] += 1
                logger.warning(f"Task {task_id} failed: {e}")
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    error=str(e),
                    execution_time_ms=execution_time
                )
    
    async def process_batch(
        self,
        tasks: List[Tuple[str, Coroutine[Any, Any, T]]]
    ) -> Tuple[List[TaskResult[T]], BatchMetrics]:
        """
        Process a batch of tasks in parallel with rate limiting.
        
        Args:
            tasks: List of (task_id, coroutine) tuples
        
        Returns:
            Tuple of (list of TaskResults, BatchMetrics)
        """
        if not tasks:
            return [], BatchMetrics()
        
        self._semaphore = asyncio.Semaphore(self.max_workers)
        self._metrics = BatchMetrics(
            total_tasks=len(tasks),
            start_time=datetime.now()
        )
        
        start_time = time.perf_counter()
        completed_count = [0]
        
        coroutines = [
            self._execute_task(task_id, coro, completed_count, len(tasks))
            for task_id, coro in tasks
        ]
        
        results = await asyncio.gather(*coroutines, return_exceptions=False)
        
        end_time = time.perf_counter()
        self._metrics.end_time = datetime.now()
        self._metrics.total_time_ms = (end_time - start_time) * 1000
        
        for result in results:
            if result.status == TaskStatus.COMPLETED:
                self._metrics.tasks_completed += 1
            elif result.status == TaskStatus.TIMEOUT:
                self._metrics.tasks_timeout += 1
            else:
                self._metrics.tasks_failed += 1
        
        logger.info(
            f"Batch completed: {self._metrics.tasks_completed}/{self._metrics.total_tasks} "
            f"success, avg {self._metrics.avg_time_per_task:.2f}ms/task"
        )
        
        return results, self._metrics
    
    def get_metrics(self) -> BatchMetrics:
        """Get the current batch processing metrics."""
        return self._metrics
    
    @property
    def avg_task_time_ms(self) -> float:
        """Get average task execution time from recent history."""
        if not self._task_times:
            return 0.0
        return sum(self._task_times) / len(self._task_times)


@dataclass
class PriorityPair:
    """Trading pair with priority for queue ordering"""
    symbol: str
    volume_24h: float
    priority: float
    
    def __lt__(self, other: 'PriorityPair') -> bool:
        return self.priority > other.priority


class ParallelMarketScanner:
    """
    High-performance market scanner for 100+ trading pairs.
    
    Features:
    - Scan 100+ pairs simultaneously using asyncio.gather()
    - Dynamic batch sizing based on system load
    - Priority queue for high-volume pairs
    - Integration with DynamicPairsFetcher for symbol lists
    - Parallel signal generation for each pair
    - Results aggregation with ranking
    
    Example:
        scanner = ParallelMarketScanner()
        await scanner.initialize()
        
        results = await scanner.scan_all_pairs()
        top_signals = scanner.get_top_signals(limit=10)
    """
    
    def __init__(
        self,
        max_concurrent_scans: int = MAX_CONCURRENT_SCANS,
        batch_size: int = BATCH_SIZE,
        scan_timeout_seconds: int = SCAN_TIMEOUT_SECONDS,
        min_volume_usd: float = 10_000_000
    ):
        """
        Initialize the market scanner.
        
        Args:
            max_concurrent_scans: Maximum concurrent pair scans (default 25)
            batch_size: Pairs per batch for dynamic sizing (default 50)
            scan_timeout_seconds: Timeout per scan (default 30)
            min_volume_usd: Minimum 24h volume filter (default $10M)
        """
        self.max_concurrent_scans = max_concurrent_scans
        self.batch_size = batch_size
        self.scan_timeout_seconds = scan_timeout_seconds
        self.min_volume_usd = min_volume_usd
        
        self._pairs_fetcher: Optional[DynamicPairsFetcher] = None
        self._batch_processor: Optional[ParallelBatchProcessor] = None
        self._signal_engines: Dict[str, SignalEngine] = {}
        self._data_fetchers: Dict[str, BinanceDataFetcher] = {}
        
        self._priority_queue: List[PriorityPair] = []
        self._scan_results: Dict[str, ScanResultEntry] = {}
        self._last_scan_time: Optional[datetime] = None
        
        self._initialized = False
        
        logger.info(f"ParallelMarketScanner initialized (max_scans={max_concurrent_scans})")
    
    async def initialize(self) -> None:
        """Initialize the scanner and fetch available pairs."""
        if self._initialized:
            return
        
        self._pairs_fetcher = DynamicPairsFetcher(min_volume_usd=self.min_volume_usd)
        self._batch_processor = ParallelBatchProcessor(
            max_workers=self.max_concurrent_scans,
            task_timeout_seconds=self.scan_timeout_seconds
        )
        
        await self._refresh_pairs()
        self._initialized = True
        logger.info("ParallelMarketScanner initialized")
    
    async def _refresh_pairs(self) -> None:
        """Refresh the list of tradeable pairs and update priority queue."""
        if self._pairs_fetcher is None:
            return
        
        pairs = await self._pairs_fetcher.get_all_pairs()
        
        self._priority_queue = []
        for pair in pairs:
            priority = self._calculate_priority(pair)
            heapq.heappush(
                self._priority_queue,
                PriorityPair(symbol=pair.symbol, volume_24h=pair.volume_24h_usd, priority=priority)
            )
        
        logger.info(f"Refreshed {len(self._priority_queue)} pairs in priority queue")
    
    def _calculate_priority(self, pair: FuturesPair) -> float:
        """
        Calculate priority score for a trading pair.
        
        Higher volume and volatility = higher priority.
        """
        volume_score = min(pair.volume_24h_usd / 1e9, 10.0)
        volatility_score = abs(pair.price_change_percent_24h) / 10.0
        trades_score = min(pair.trades_count_24h / 1_000_000, 5.0)
        
        return volume_score * 0.5 + volatility_score * 0.3 + trades_score * 0.2
    
    def _get_dynamic_batch_size(self) -> int:
        """
        Calculate dynamic batch size based on current load.
        
        Reduces batch size if previous scans were slow.
        """
        if self._batch_processor is None:
            return self.batch_size
        
        avg_time = self._batch_processor.avg_task_time_ms
        if avg_time == 0:
            return self.batch_size
        
        if avg_time > 5000:
            return max(10, self.batch_size // 2)
        elif avg_time > 2000:
            return max(25, int(self.batch_size * 0.75))
        elif avg_time < 500:
            return min(100, int(self.batch_size * 1.5))
        
        return self.batch_size
    
    def _get_or_create_engine(self, symbol: str) -> SignalEngine:
        """Get or create a SignalEngine for a symbol."""
        if symbol not in self._signal_engines:
            self._signal_engines[symbol] = SignalEngine()
        return self._signal_engines[symbol]
    
    def _get_or_create_fetcher(self, symbol: str) -> BinanceDataFetcher:
        """Get or create a BinanceDataFetcher for a symbol."""
        if symbol not in self._data_fetchers:
            self._data_fetchers[symbol] = BinanceDataFetcher(symbol=symbol)
        return self._data_fetchers[symbol]
    
    async def _scan_single_pair(self, symbol: str, volume_24h: float) -> ScanResultEntry:
        """Scan a single trading pair for signals."""
        start_time = time.perf_counter()
        
        try:
            fetcher = self._get_or_create_fetcher(symbol)
            engine = self._get_or_create_engine(symbol)
            
            df = fetcher.fetch_historical_data(limit=200)
            
            if df is None or len(df) < 50:
                return ScanResultEntry(
                    symbol=symbol,
                    signal_type='NONE',
                    confidence=0.0,
                    entry_price=0.0,
                    stop_loss=0.0,
                    take_profit=0.0,
                    volume_24h=volume_24h,
                    priority_score=0.0,
                    scan_time_ms=(time.perf_counter() - start_time) * 1000,
                    error="Insufficient data"
                )
            
            signal = engine.generate_signal(df)
            
            scan_time = (time.perf_counter() - start_time) * 1000
            
            if signal is None:
                signal = {}
            
            return ScanResultEntry(
                symbol=symbol,
                signal_type=signal.get('type', signal.get('signal_type', 'NONE')),
                confidence=signal.get('confidence', 0.0),
                entry_price=signal.get('entry_price', 0.0),
                stop_loss=signal.get('stop_loss', 0.0),
                take_profit=signal.get('take_profit', 0.0),
                volume_24h=volume_24h,
                priority_score=self._calculate_signal_priority(signal, volume_24h),
                scan_time_ms=scan_time
            )
            
        except Exception as e:
            scan_time = (time.perf_counter() - start_time) * 1000
            logger.warning(f"Error scanning {symbol}: {e}")
            return ScanResultEntry(
                symbol=symbol,
                signal_type='NONE',
                confidence=0.0,
                entry_price=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                volume_24h=volume_24h,
                priority_score=0.0,
                scan_time_ms=scan_time,
                error=str(e)
            )
    
    def _calculate_signal_priority(self, signal: Dict[str, Any], volume_24h: float) -> float:
        """Calculate priority score for ranking signals."""
        if signal.get('signal_type') == 'NONE':
            return 0.0
        
        confidence = signal.get('confidence', 0.0)
        volume_score = min(volume_24h / 1e9, 1.0)
        
        return confidence * 0.7 + volume_score * 0.3
    
    async def scan_all_pairs(
        self,
        limit: Optional[int] = None,
        symbols: Optional[List[str]] = None
    ) -> Tuple[List[ScanResultEntry], BatchMetrics]:
        """
        Scan all pairs or a subset for trading signals.
        
        Args:
            limit: Maximum pairs to scan (None for all)
            symbols: Specific symbols to scan (None for priority queue)
        
        Returns:
            Tuple of (list of ScanResultEntry, BatchMetrics)
        """
        if not self._initialized:
            await self.initialize()
        
        if symbols:
            pairs_to_scan = [(s, 0.0) for s in symbols]
        else:
            sorted_queue = sorted(self._priority_queue, reverse=True)
            if limit:
                sorted_queue = sorted_queue[:limit]
            pairs_to_scan = [(p.symbol, p.volume_24h) for p in sorted_queue]
        
        dynamic_batch_size = self._get_dynamic_batch_size()
        logger.info(f"Scanning {len(pairs_to_scan)} pairs (batch_size={dynamic_batch_size})")
        
        tasks = [
            (symbol, self._scan_single_pair(symbol, volume))
            for symbol, volume in pairs_to_scan
        ]
        
        if self._batch_processor is None:
            raise RuntimeError("Batch processor not initialized")
        
        results, metrics = await self._batch_processor.process_batch(tasks)
        
        self._scan_results.clear()
        scan_entries = []
        
        for result in results:
            if result.success and result.result:
                entry = result.result
                self._scan_results[entry.symbol] = entry
                scan_entries.append(entry)
        
        scan_entries.sort(key=lambda x: x.priority_score, reverse=True)
        
        self._last_scan_time = datetime.now()
        
        signals_found = sum(1 for e in scan_entries if e.has_signal)
        logger.info(f"Scan complete: {signals_found} signals from {len(scan_entries)} pairs")
        
        return scan_entries, metrics
    
    def get_top_signals(self, limit: int = 10, signal_type: Optional[str] = None) -> List[ScanResultEntry]:
        """
        Get top-ranked signals from the last scan.
        
        Args:
            limit: Maximum signals to return
            signal_type: Filter by 'LONG' or 'SHORT' (None for all)
        
        Returns:
            List of top ScanResultEntry sorted by priority
        """
        results = list(self._scan_results.values())
        
        results = [r for r in results if r.has_signal]
        
        if signal_type:
            results = [r for r in results if r.signal_type == signal_type]
        
        results.sort(key=lambda x: x.priority_score, reverse=True)
        
        return results[:limit]
    
    def get_scan_metrics(self) -> Dict[str, Any]:
        """Get metrics from the last scan."""
        if self._batch_processor is None:
            return {}
        
        metrics = self._batch_processor.get_metrics()
        return {
            'total_pairs_scanned': metrics.total_tasks,
            'successful_scans': metrics.tasks_completed,
            'failed_scans': metrics.tasks_failed,
            'timeout_scans': metrics.tasks_timeout,
            'avg_scan_time_ms': metrics.avg_time_per_task,
            'total_scan_time_ms': metrics.total_time_ms,
            'signals_found': sum(1 for r in self._scan_results.values() if r.has_signal),
            'last_scan_time': self._last_scan_time.isoformat() if self._last_scan_time else None
        }
    
    async def close(self) -> None:
        """Clean up resources."""
        if self._pairs_fetcher:
            await self._pairs_fetcher.close()
        
        self._signal_engines.clear()
        self._data_fetchers.clear()
        self._initialized = False


class ParallelDataFetcher:
    """
    Concurrent multi-API data fetcher with fallback mechanisms.
    
    Features:
    - Concurrent fetching from multiple APIs (Fear/Greed, CoinGecko, Derivatives)
    - Timeout handling per source
    - Fallback mechanisms when sources fail
    - Cache management with configurable TTL
    
    Example:
        fetcher = ParallelDataFetcher()
        await fetcher.initialize()
        
        data = await fetcher.fetch_all_sources(symbol='BTCUSDT')
        fear_greed = data.get('fear_greed')
    """
    
    SOURCE_FEAR_GREED = "fear_greed"
    SOURCE_COINGECKO = "coingecko"
    SOURCE_DERIVATIVES = "derivatives"
    
    def __init__(
        self,
        max_concurrent_calls: int = MAX_CONCURRENT_API_CALLS,
        default_timeout_seconds: float = 15.0,
        cache_ttl_seconds: int = 300
    ):
        """
        Initialize the data fetcher.
        
        Args:
            max_concurrent_calls: Maximum concurrent API calls (default 10)
            default_timeout_seconds: Default timeout per source (default 15)
            cache_ttl_seconds: Cache TTL in seconds (default 300)
        """
        self.max_concurrent_calls = max_concurrent_calls
        self.default_timeout_seconds = default_timeout_seconds
        self.cache_ttl_seconds = cache_ttl_seconds
        
        self._semaphore: Optional[asyncio.Semaphore] = None
        
        self._fear_greed_client: Optional[FearGreedClient] = None
        self._coingecko_client: Optional[MarketDataAggregator] = None
        self._derivatives_client: Optional[BinanceDerivativesClient] = None
        
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._source_health: Dict[str, Dict[str, Any]] = {}
        self._source_timeouts: Dict[str, float] = {
            self.SOURCE_FEAR_GREED: 10.0,
            self.SOURCE_COINGECKO: 15.0,
            self.SOURCE_DERIVATIVES: 10.0,
        }
        
        self._initialized = False
        
        logger.info(f"ParallelDataFetcher initialized (max_calls={max_concurrent_calls})")
    
    async def initialize(self) -> None:
        """Initialize all API clients."""
        if self._initialized:
            return
        
        self._semaphore = asyncio.Semaphore(self.max_concurrent_calls)
        
        self._fear_greed_client = FearGreedClient()
        self._coingecko_client = MarketDataAggregator()
        self._derivatives_client = BinanceDerivativesClient()
        
        for source in [self.SOURCE_FEAR_GREED, self.SOURCE_COINGECKO, self.SOURCE_DERIVATIVES]:
            self._source_health[source] = {
                'success_count': 0,
                'failure_count': 0,
                'consecutive_failures': 0,
                'last_success': None,
                'last_error': None
            }
        
        self._initialized = True
        logger.info("ParallelDataFetcher initialized with all clients")
    
    def set_source_timeout(self, source_name: str, timeout_seconds: float) -> None:
        """
        Set custom timeout for a specific source.
        
        Args:
            source_name: Name of the source (fear_greed, coingecko, derivatives)
            timeout_seconds: Timeout in seconds
        """
        self._source_timeouts[source_name] = timeout_seconds
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if a cache entry is still valid."""
        if cache_key not in self._cache:
            return False
        _, cache_time = self._cache[cache_key]
        return (datetime.now() - cache_time).total_seconds() < self.cache_ttl_seconds
    
    def _get_cached(self, cache_key: str) -> Optional[Any]:
        """Get data from cache if valid."""
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key][0]
        return None
    
    def _set_cached(self, cache_key: str, data: Any) -> None:
        """Store data in cache."""
        self._cache[cache_key] = (data, datetime.now())
    
    def _record_success(self, source_name: str) -> None:
        """Record a successful API call."""
        health = self._source_health.get(source_name, {})
        health['success_count'] = health.get('success_count', 0) + 1
        health['consecutive_failures'] = 0
        health['last_success'] = datetime.now()
        self._source_health[source_name] = health
    
    def _record_failure(self, source_name: str, error: str) -> None:
        """Record a failed API call."""
        health = self._source_health.get(source_name, {})
        health['failure_count'] = health.get('failure_count', 0) + 1
        health['consecutive_failures'] = health.get('consecutive_failures', 0) + 1
        health['last_error'] = error
        self._source_health[source_name] = health
    
    def _is_source_healthy(self, source_name: str) -> bool:
        """Check if a source is healthy enough to use."""
        health = self._source_health.get(source_name, {})
        return health.get('consecutive_failures', 0) < 5
    
    async def _fetch_with_timeout(
        self,
        source_name: str,
        fetch_coro: Coroutine[Any, Any, Any],
        cache_key: str
    ) -> APISourceResult:
        """Fetch data from a source with timeout and error handling."""
        cached = self._get_cached(cache_key)
        if cached is not None:
            return APISourceResult(
                source_name=source_name,
                success=True,
                data=cached,
                from_cache=True
            )
        
        if not self._is_source_healthy(source_name):
            return APISourceResult(
                source_name=source_name,
                success=False,
                error="Source marked unhealthy due to consecutive failures"
            )
        
        timeout = self._source_timeouts.get(source_name, self.default_timeout_seconds)
        start_time = time.perf_counter()
        
        async with self._semaphore:
            try:
                result = await asyncio.wait_for(fetch_coro, timeout=timeout)
                response_time = (time.perf_counter() - start_time) * 1000
                
                if result is not None:
                    self._set_cached(cache_key, result)
                    self._record_success(source_name)
                    return APISourceResult(
                        source_name=source_name,
                        success=True,
                        data=result,
                        response_time_ms=response_time
                    )
                else:
                    self._record_failure(source_name, "No data returned")
                    return APISourceResult(
                        source_name=source_name,
                        success=False,
                        error="No data returned",
                        response_time_ms=response_time
                    )
                    
            except asyncio.TimeoutError:
                response_time = (time.perf_counter() - start_time) * 1000
                error = f"Timeout after {timeout}s"
                self._record_failure(source_name, error)
                return APISourceResult(
                    source_name=source_name,
                    success=False,
                    error=error,
                    response_time_ms=response_time
                )
                
            except Exception as e:
                response_time = (time.perf_counter() - start_time) * 1000
                error = str(e)
                self._record_failure(source_name, error)
                logger.warning(f"Error fetching from {source_name}: {error}")
                return APISourceResult(
                    source_name=source_name,
                    success=False,
                    error=error,
                    response_time_ms=response_time
                )
    
    async def fetch_fear_greed(self) -> APISourceResult:
        """Fetch Fear & Greed Index data."""
        if self._fear_greed_client is None:
            return APISourceResult(
                source_name=self.SOURCE_FEAR_GREED,
                success=False,
                error="Client not initialized"
            )
        
        return await self._fetch_with_timeout(
            self.SOURCE_FEAR_GREED,
            self._fear_greed_client.get_current(),
            f"fear_greed_current"
        )
    
    async def fetch_coingecko_global(self) -> APISourceResult:
        """Fetch global market data from CoinGecko."""
        if self._coingecko_client is None:
            return APISourceResult(
                source_name=self.SOURCE_COINGECKO,
                success=False,
                error="Client not initialized"
            )
        
        return await self._fetch_with_timeout(
            self.SOURCE_COINGECKO,
            self._coingecko_client.get_global_market_data(),
            f"coingecko_global"
        )
    
    async def fetch_derivatives(self, symbol: str = "BTCUSDT") -> APISourceResult:
        """Fetch derivatives data from Binance."""
        if self._derivatives_client is None:
            return APISourceResult(
                source_name=self.SOURCE_DERIVATIVES,
                success=False,
                error="Client not initialized"
            )
        
        return await self._fetch_with_timeout(
            self.SOURCE_DERIVATIVES,
            self._derivatives_client.get_derivatives_intelligence(symbol),
            f"derivatives_{symbol}"
        )
    
    async def fetch_all_sources(
        self,
        symbol: str = "BTCUSDT",
        include_sources: Optional[List[str]] = None
    ) -> Dict[str, APISourceResult]:
        """
        Fetch data from all sources concurrently.
        
        Args:
            symbol: Trading symbol for derivatives data
            include_sources: List of sources to include (None for all)
        
        Returns:
            Dictionary mapping source name to APISourceResult
        """
        if not self._initialized:
            await self.initialize()
        
        sources = include_sources or [
            self.SOURCE_FEAR_GREED,
            self.SOURCE_COINGECKO,
            self.SOURCE_DERIVATIVES
        ]
        
        tasks = {}
        if self.SOURCE_FEAR_GREED in sources:
            tasks[self.SOURCE_FEAR_GREED] = self.fetch_fear_greed()
        if self.SOURCE_COINGECKO in sources:
            tasks[self.SOURCE_COINGECKO] = self.fetch_coingecko_global()
        if self.SOURCE_DERIVATIVES in sources:
            tasks[self.SOURCE_DERIVATIVES] = self.fetch_derivatives(symbol)
        
        results = await asyncio.gather(*tasks.values(), return_exceptions=False)
        
        return dict(zip(tasks.keys(), results))
    
    def get_fallback_value(self, source_name: str) -> Any:
        """
        Get a fallback value for a source that failed.
        
        Returns cached data if available, or a neutral default.
        """
        for key, (data, _) in self._cache.items():
            if key.startswith(source_name):
                return data
        
        if source_name == self.SOURCE_FEAR_GREED:
            return FearGreedData(
                value=50,
                value_classification="Neutral",
                timestamp=datetime.now()
            )
        elif source_name == self.SOURCE_COINGECKO:
            return None
        elif source_name == self.SOURCE_DERIVATIVES:
            return None
        
        return None
    
    def get_source_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all sources."""
        return {
            source: {
                **health,
                'is_healthy': self._is_source_healthy(source),
                'timeout_seconds': self._source_timeouts.get(source, self.default_timeout_seconds)
            }
            for source, health in self._source_health.items()
        }
    
    def clear_cache(self, source_name: Optional[str] = None) -> None:
        """
        Clear the cache.
        
        Args:
            source_name: Clear only this source's cache (None for all)
        """
        if source_name:
            keys_to_remove = [k for k in self._cache if k.startswith(source_name)]
            for key in keys_to_remove:
                del self._cache[key]
        else:
            self._cache.clear()
    
    async def close(self) -> None:
        """Clean up resources and close all clients."""
        if self._fear_greed_client:
            await self._fear_greed_client.close()
        if self._coingecko_client:
            await self._coingecko_client.close()
        if self._derivatives_client:
            await self._derivatives_client.close()
        
        self._cache.clear()
        self._initialized = False
        logger.info("ParallelDataFetcher closed")
