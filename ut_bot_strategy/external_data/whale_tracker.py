"""
Whale Tracker for Binance Futures

Tracks large trades ($100K+ USD) and aggregates whale activity metrics.
Can integrate with existing OrderFlowStream or operate standalone.

Features:
- Real-time whale trade detection via WebSocket
- Aggregated metrics over configurable time windows
- Buy/sell imbalance tracking
- Accumulation/distribution scoring
- Smart money flow direction detection
"""

import asyncio
import logging
import json
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Callable, Any, Deque
from collections import deque

import aiohttp

AIOHTTP_AVAILABLE = True

logger = logging.getLogger(__name__)


@dataclass
class WhaleTrade:
    """Individual whale trade data"""
    timestamp: float
    symbol: str
    price: float
    quantity: float
    notional_value: float
    is_buy: bool
    trade_id: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'symbol': self.symbol,
            'price': self.price,
            'quantity': self.quantity,
            'notional_value': self.notional_value,
            'is_buy': self.is_buy,
            'side': 'BUY' if self.is_buy else 'SELL',
            'trade_id': self.trade_id
        }


@dataclass
class WhaleMetrics:
    """Aggregated whale activity metrics"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    symbol: str = ""
    window_minutes: int = 15
    whale_buy_volume: float = 0.0
    whale_sell_volume: float = 0.0
    whale_buy_count: int = 0
    whale_sell_count: int = 0
    net_whale_flow: float = 0.0
    whale_bias: float = 0.0
    accumulation_score: float = 0.0
    distribution_score: float = 0.0
    smart_money_direction: str = "NEUTRAL"
    largest_buy: float = 0.0
    largest_sell: float = 0.0
    avg_whale_trade_size: float = 0.0
    
    @property
    def total_whale_volume(self) -> float:
        return self.whale_buy_volume + self.whale_sell_volume
    
    @property
    def total_whale_count(self) -> int:
        return self.whale_buy_count + self.whale_sell_count
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'window_minutes': self.window_minutes,
            'whale_buy_volume': round(self.whale_buy_volume, 2),
            'whale_sell_volume': round(self.whale_sell_volume, 2),
            'whale_buy_count': self.whale_buy_count,
            'whale_sell_count': self.whale_sell_count,
            'total_whale_volume': round(self.total_whale_volume, 2),
            'total_whale_count': self.total_whale_count,
            'net_whale_flow': round(self.net_whale_flow, 2),
            'whale_bias': round(self.whale_bias, 4),
            'accumulation_score': round(self.accumulation_score, 4),
            'distribution_score': round(self.distribution_score, 4),
            'smart_money_direction': self.smart_money_direction,
            'largest_buy': round(self.largest_buy, 2),
            'largest_sell': round(self.largest_sell, 2),
            'avg_whale_trade_size': round(self.avg_whale_trade_size, 2)
        }


class WhaleTracker:
    """
    Real-time Whale Trade Tracker for Binance Futures
    
    Tracks large trades ($100K+ USD value) and computes whale activity metrics
    including buy/sell imbalance, accumulation/distribution scores, and
    smart money flow direction.
    
    Can operate standalone or integrate with OrderFlowStream.
    """
    
    BINANCE_FUTURES_WS_BASE = "wss://fstream.binance.com/ws"
    DEFAULT_WHALE_THRESHOLD = 100_000.0  # $100K USD
    DEFAULT_WINDOW_MINUTES = 15
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        whale_threshold_usd: float = 100_000.0,
        window_minutes: int = 15,
        on_whale_trade_callback: Optional[Callable[[WhaleTrade], None]] = None,
        on_metrics_callback: Optional[Callable[[WhaleMetrics], None]] = None
    ):
        """
        Initialize Whale Tracker
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            whale_threshold_usd: Minimum USD value for whale trades (default $100K)
            window_minutes: Rolling window for metrics in minutes (default 15)
            on_whale_trade_callback: Callback for each whale trade detected
            on_metrics_callback: Callback for metrics updates
        """
        self.symbol = symbol.upper()
        self.symbol_lower = symbol.lower()
        self.whale_threshold_usd = whale_threshold_usd
        self.window_minutes = window_minutes
        self.window_seconds = window_minutes * 60
        
        self.on_whale_trade_callback = on_whale_trade_callback
        self.on_metrics_callback = on_metrics_callback
        
        self._lock = threading.RLock()
        
        self._whale_trades: Deque[WhaleTrade] = deque(maxlen=10000)
        self._recent_prices: Deque[float] = deque(maxlen=1000)
        
        self._running = False
        self._ws_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0
        
        logger.info(
            f"WhaleTracker initialized for {self.symbol} "
            f"(threshold=${whale_threshold_usd:,.0f}, window={window_minutes}min)"
        )
    
    async def start(self) -> None:
        """Start the WebSocket connection and metrics computation"""
        if self._running:
            logger.warning("WhaleTracker is already running")
            return
        
        if not AIOHTTP_AVAILABLE:
            logger.error("aiohttp not available, cannot start WhaleTracker")
            return
        
        self._running = True
        logger.info(f"Starting WhaleTracker for {self.symbol}")
        
        try:
            self._ws_task = asyncio.create_task(self._run_trade_stream())
            self._metrics_task = asyncio.create_task(self._run_metrics_updater())
            logger.info("WhaleTracker started successfully")
        except Exception as e:
            logger.error(f"Error starting WhaleTracker: {e}")
            self._running = False
            raise
    
    async def stop(self) -> None:
        """Stop the WebSocket connection and metrics computation"""
        if not self._running:
            return
        
        self._running = False
        logger.info("Stopping WhaleTracker...")
        
        for task in [self._ws_task, self._metrics_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("WhaleTracker stopped")
    
    async def _run_trade_stream(self) -> None:
        """Run the trade stream with auto-reconnection"""
        while self._running:
            try:
                await self._connect_trade_stream()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Trade stream error: {e}")
                if self._running:
                    await asyncio.sleep(self._reconnect_delay)
                    self._reconnect_delay = min(
                        self._reconnect_delay * 2,
                        self._max_reconnect_delay
                    )
    
    async def _connect_trade_stream(self) -> None:
        """Connect to the aggTrade WebSocket stream"""
        stream_url = f"{self.BINANCE_FUTURES_WS_BASE}/{self.symbol_lower}@aggTrade"
        
        logger.info(f"Connecting to trade stream: {stream_url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(stream_url, heartbeat=30) as ws:
                self._reconnect_delay = 1.0
                logger.info(f"Trade stream connected for {self.symbol}")
                
                async for msg in ws:
                    if not self._running:
                        break
                    
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            data = json.loads(msg.data)
                            await self._process_trade(data)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse trade message: {e}")
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"WebSocket error: {ws.exception()}")
                        break
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        logger.warning("WebSocket closed")
                        break
    
    async def _process_trade(self, data: Dict) -> None:
        """Process incoming aggTrade data and detect whale trades"""
        try:
            price = float(data.get('p', 0))
            quantity = float(data.get('q', 0))
            notional_value = price * quantity
            is_buyer_maker = data.get('m', False)
            
            with self._lock:
                self._recent_prices.append(price)
            
            if notional_value >= self.whale_threshold_usd:
                whale_trade = WhaleTrade(
                    timestamp=float(data.get('T', time.time() * 1000)) / 1000.0,
                    symbol=self.symbol,
                    price=price,
                    quantity=quantity,
                    notional_value=notional_value,
                    is_buy=not is_buyer_maker,
                    trade_id=int(data.get('a', 0))
                )
                
                with self._lock:
                    self._whale_trades.append(whale_trade)
                
                logger.info(
                    f"🐋 Whale {('BUY' if whale_trade.is_buy else 'SELL')} detected: "
                    f"{self.symbol} {quantity:.4f} @ ${price:,.2f} = ${notional_value:,.0f}"
                )
                
                if self.on_whale_trade_callback:
                    try:
                        self.on_whale_trade_callback(whale_trade)
                    except Exception as e:
                        logger.warning(f"Whale trade callback error: {e}")
                        
        except Exception as e:
            logger.error(f"Error processing trade: {e}")
    
    async def _run_metrics_updater(self) -> None:
        """Periodically compute and emit metrics"""
        while self._running:
            try:
                await asyncio.sleep(5.0)
                
                metrics = self.get_current_metrics()
                
                if self.on_metrics_callback:
                    try:
                        self.on_metrics_callback(metrics)
                    except Exception as e:
                        logger.warning(f"Metrics callback error: {e}")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics updater error: {e}")
    
    def get_current_metrics(self) -> WhaleMetrics:
        """
        Get current whale activity metrics (thread-safe)
        
        Returns:
            WhaleMetrics dataclass with aggregated whale data
        """
        with self._lock:
            return self._compute_metrics()
    
    def _compute_metrics(self) -> WhaleMetrics:
        """Compute whale metrics (must be called with lock held)"""
        now = time.time()
        cutoff_time = now - self.window_seconds
        
        window_trades = [t for t in self._whale_trades if t.timestamp >= cutoff_time]
        
        if not window_trades:
            return WhaleMetrics(
                symbol=self.symbol,
                window_minutes=self.window_minutes
            )
        
        buy_trades = [t for t in window_trades if t.is_buy]
        sell_trades = [t for t in window_trades if not t.is_buy]
        
        whale_buy_volume = sum(t.notional_value for t in buy_trades)
        whale_sell_volume = sum(t.notional_value for t in sell_trades)
        whale_buy_count = len(buy_trades)
        whale_sell_count = len(sell_trades)
        
        net_whale_flow = whale_buy_volume - whale_sell_volume
        
        total_volume = whale_buy_volume + whale_sell_volume
        if total_volume > 0:
            whale_bias = (whale_buy_volume - whale_sell_volume) / total_volume
        else:
            whale_bias = 0.0
        
        whale_bias = max(-1.0, min(1.0, whale_bias))
        
        accumulation_score, distribution_score = self._compute_acc_dist_scores(
            buy_trades, sell_trades, window_trades
        )
        
        smart_money_direction = self._determine_smart_money_direction(
            whale_bias, accumulation_score, distribution_score
        )
        
        largest_buy = max((t.notional_value for t in buy_trades), default=0.0)
        largest_sell = max((t.notional_value for t in sell_trades), default=0.0)
        
        total_count = len(window_trades)
        avg_whale_trade_size = total_volume / total_count if total_count > 0 else 0.0
        
        return WhaleMetrics(
            timestamp=datetime.utcnow(),
            symbol=self.symbol,
            window_minutes=self.window_minutes,
            whale_buy_volume=whale_buy_volume,
            whale_sell_volume=whale_sell_volume,
            whale_buy_count=whale_buy_count,
            whale_sell_count=whale_sell_count,
            net_whale_flow=net_whale_flow,
            whale_bias=whale_bias,
            accumulation_score=accumulation_score,
            distribution_score=distribution_score,
            smart_money_direction=smart_money_direction,
            largest_buy=largest_buy,
            largest_sell=largest_sell,
            avg_whale_trade_size=avg_whale_trade_size
        )
    
    def _compute_acc_dist_scores(
        self,
        buy_trades: List[WhaleTrade],
        sell_trades: List[WhaleTrade],
        all_trades: List[WhaleTrade]
    ) -> tuple[float, float]:
        """
        Compute accumulation and distribution scores
        
        Accumulation: Large buys near lows, suggesting smart money buying dips
        Distribution: Large sells near highs, suggesting smart money selling tops
        
        Returns:
            Tuple of (accumulation_score, distribution_score) from 0 to 1
        """
        if len(self._recent_prices) < 10 or not all_trades:
            return 0.0, 0.0
        
        prices = list(self._recent_prices)
        price_high = max(prices)
        price_low = min(prices)
        price_range = price_high - price_low
        
        if price_range == 0:
            return 0.0, 0.0
        
        accumulation_score = 0.0
        for trade in buy_trades:
            price_position = (trade.price - price_low) / price_range
            near_low_score = 1.0 - price_position
            size_weight = min(trade.notional_value / 500_000, 2.0)
            accumulation_score += near_low_score * size_weight
        
        if buy_trades:
            accumulation_score = min(1.0, accumulation_score / len(buy_trades))
        
        distribution_score = 0.0
        for trade in sell_trades:
            price_position = (trade.price - price_low) / price_range
            near_high_score = price_position
            size_weight = min(trade.notional_value / 500_000, 2.0)
            distribution_score += near_high_score * size_weight
        
        if sell_trades:
            distribution_score = min(1.0, distribution_score / len(sell_trades))
        
        return accumulation_score, distribution_score
    
    def _determine_smart_money_direction(
        self,
        whale_bias: float,
        accumulation_score: float,
        distribution_score: float
    ) -> str:
        """
        Determine smart money flow direction based on metrics
        
        Returns:
            'ACCUMULATING', 'DISTRIBUTING', 'BULLISH', 'BEARISH', or 'NEUTRAL'
        """
        if accumulation_score > 0.6 and whale_bias > 0.3:
            return "ACCUMULATING"
        
        if distribution_score > 0.6 and whale_bias < -0.3:
            return "DISTRIBUTING"
        
        if whale_bias > 0.5:
            return "BULLISH"
        elif whale_bias < -0.5:
            return "BEARISH"
        elif whale_bias > 0.2:
            return "SLIGHTLY_BULLISH"
        elif whale_bias < -0.2:
            return "SLIGHTLY_BEARISH"
        
        return "NEUTRAL"
    
    def get_whale_trades(self, minutes: Optional[int] = None) -> List[WhaleTrade]:
        """
        Get whale trades from the last N minutes (thread-safe)
        
        Args:
            minutes: Number of minutes to look back (default: window_minutes)
            
        Returns:
            List of WhaleTrade objects
        """
        minutes = minutes or self.window_minutes
        cutoff = time.time() - (minutes * 60)
        
        with self._lock:
            return [t for t in self._whale_trades if t.timestamp >= cutoff]
    
    def get_recent_large_trades(self, min_value_usd: float = 500_000, minutes: int = 5) -> List[WhaleTrade]:
        """
        Get very large trades above a custom threshold
        
        Args:
            min_value_usd: Minimum USD value filter
            minutes: Number of minutes to look back
            
        Returns:
            List of WhaleTrade objects
        """
        cutoff = time.time() - (minutes * 60)
        
        with self._lock:
            return [
                t for t in self._whale_trades
                if t.timestamp >= cutoff and t.notional_value >= min_value_usd
            ]
    
    def process_external_trade(
        self,
        price: float,
        quantity: float,
        is_buy: bool,
        trade_id: int = 0,
        timestamp: Optional[float] = None
    ) -> Optional[WhaleTrade]:
        """
        Process an externally provided trade (e.g., from OrderFlowStream)
        
        Args:
            price: Trade price
            quantity: Trade quantity
            is_buy: Whether it's a buy (taker buy)
            trade_id: Trade ID
            timestamp: Trade timestamp (default: now)
            
        Returns:
            WhaleTrade if it meets threshold, None otherwise
        """
        notional_value = price * quantity
        
        if notional_value < self.whale_threshold_usd:
            return None
        
        whale_trade = WhaleTrade(
            timestamp=timestamp or time.time(),
            symbol=self.symbol,
            price=price,
            quantity=quantity,
            notional_value=notional_value,
            is_buy=is_buy,
            trade_id=trade_id
        )
        
        with self._lock:
            self._whale_trades.append(whale_trade)
            self._recent_prices.append(price)
        
        logger.info(
            f"🐋 External whale {('BUY' if is_buy else 'SELL')}: "
            f"{self.symbol} {quantity:.4f} @ ${price:,.2f} = ${notional_value:,.0f}"
        )
        
        if self.on_whale_trade_callback:
            try:
                self.on_whale_trade_callback(whale_trade)
            except Exception as e:
                logger.warning(f"Whale trade callback error: {e}")
        
        return whale_trade
    
    def get_status(self) -> Dict[str, Any]:
        """Get current tracker status"""
        with self._lock:
            whale_count = len(self._whale_trades)
            recent_count = len([
                t for t in self._whale_trades
                if t.timestamp >= time.time() - self.window_seconds
            ])
        
        return {
            'symbol': self.symbol,
            'running': self._running,
            'whale_threshold_usd': self.whale_threshold_usd,
            'window_minutes': self.window_minutes,
            'total_whale_trades_tracked': whale_count,
            'whale_trades_in_window': recent_count
        }


def create_whale_tracker_from_order_flow(
    order_flow_stream: Any,
    whale_threshold_usd: float = 100_000.0,
    window_minutes: int = 15,
    on_whale_trade_callback: Optional[Callable[[WhaleTrade], None]] = None,
    on_metrics_callback: Optional[Callable[[WhaleMetrics], None]] = None
) -> WhaleTracker:
    """
    Create a WhaleTracker that integrates with an existing OrderFlowStream
    
    Args:
        order_flow_stream: OrderFlowStream instance to integrate with
        whale_threshold_usd: Minimum USD value for whale trades
        window_minutes: Rolling window for metrics
        on_whale_trade_callback: Callback for whale trades
        on_metrics_callback: Callback for metrics
        
    Returns:
        WhaleTracker instance with OrderFlowStream integration
    """
    tracker = WhaleTracker(
        symbol=order_flow_stream.symbol,
        whale_threshold_usd=whale_threshold_usd,
        window_minutes=window_minutes,
        on_whale_trade_callback=on_whale_trade_callback,
        on_metrics_callback=on_metrics_callback
    )
    
    original_callback = order_flow_stream.on_trade_callback
    
    def integrated_trade_callback(trade_data):
        if original_callback:
            original_callback(trade_data)
        
        tracker.process_external_trade(
            price=trade_data.price,
            quantity=trade_data.quantity,
            is_buy=not trade_data.is_buyer_maker,
            trade_id=trade_data.trade_id,
            timestamp=trade_data.timestamp
        )
    
    order_flow_stream.on_trade_callback = integrated_trade_callback
    
    logger.info(f"WhaleTracker integrated with OrderFlowStream for {tracker.symbol}")
    return tracker


async def create_standalone_whale_tracker(
    symbol: str = "BTCUSDT",
    whale_threshold_usd: float = 100_000.0,
    window_minutes: int = 15,
    on_whale_trade_callback: Optional[Callable[[WhaleTrade], None]] = None,
    on_metrics_callback: Optional[Callable[[WhaleMetrics], None]] = None,
    auto_start: bool = True
) -> WhaleTracker:
    """
    Factory function to create a standalone WhaleTracker
    
    Args:
        symbol: Trading pair symbol
        whale_threshold_usd: Minimum USD value for whale trades
        window_minutes: Rolling window for metrics
        on_whale_trade_callback: Callback for whale trades
        on_metrics_callback: Callback for metrics
        auto_start: Whether to start the tracker immediately
        
    Returns:
        WhaleTracker instance
    """
    tracker = WhaleTracker(
        symbol=symbol,
        whale_threshold_usd=whale_threshold_usd,
        window_minutes=window_minutes,
        on_whale_trade_callback=on_whale_trade_callback,
        on_metrics_callback=on_metrics_callback
    )
    
    if auto_start:
        await tracker.start()
    
    return tracker
