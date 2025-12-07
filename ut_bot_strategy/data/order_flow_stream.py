"""
Order Flow Stream for Real-time Binance WebSocket Data

Provides real-time order flow analysis including:
- Aggregated trades (@aggTrade) tracking
- Order book depth (@depth20) monitoring
- Cumulative Volume Delta (CVD) calculation
- Large order detection
- Order book imbalance ratio
- Absorption detection
- Tape speed metrics
"""

import os
import asyncio
import logging
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Callable, Any, Deque
from collections import deque
from dataclasses import dataclass, field

AIOHTTP_AVAILABLE = False
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None

CCXT_AVAILABLE = False
try:
    import ccxt.pro as ccxtpro
    CCXT_AVAILABLE = True
except ImportError:
    try:
        import ccxt
        CCXT_AVAILABLE = False
    except ImportError:
        ccxt = None

logger = logging.getLogger(__name__)


@dataclass
class TradeData:
    """Individual trade data"""
    timestamp: float
    price: float
    quantity: float
    is_buyer_maker: bool
    trade_id: int
    notional_value: float = 0.0
    
    def __post_init__(self):
        self.notional_value = self.price * self.quantity


@dataclass
class DepthData:
    """Order book depth snapshot"""
    timestamp: float
    bids: List[List[float]]
    asks: List[List[float]]
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    
    def __post_init__(self):
        self.bid_volume = sum(float(b[1]) for b in self.bids) if self.bids else 0.0
        self.ask_volume = sum(float(a[1]) for a in self.asks) if self.asks else 0.0


@dataclass
class OrderFlowMetrics:
    """Current order flow metrics snapshot"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    cumulative_delta: float = 0.0
    delta_change: float = 0.0
    bid_ask_imbalance: float = 0.0
    large_order_count: int = 0
    large_buy_count: int = 0
    large_sell_count: int = 0
    absorption_score: float = 0.0
    tape_speed: float = 0.0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    total_volume: float = 0.0
    vwap: float = 0.0
    last_price: float = 0.0
    best_bid: float = 0.0
    best_ask: float = 0.0
    spread: float = 0.0
    spread_percent: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cumulative_delta': round(self.cumulative_delta, 4),
            'delta_change': round(self.delta_change, 4),
            'bid_ask_imbalance': round(self.bid_ask_imbalance, 4),
            'large_order_count': self.large_order_count,
            'large_buy_count': self.large_buy_count,
            'large_sell_count': self.large_sell_count,
            'absorption_score': round(self.absorption_score, 4),
            'tape_speed': round(self.tape_speed, 2),
            'buy_volume': round(self.buy_volume, 4),
            'sell_volume': round(self.sell_volume, 4),
            'total_volume': round(self.total_volume, 4),
            'vwap': round(self.vwap, 2),
            'last_price': round(self.last_price, 2),
            'best_bid': round(self.best_bid, 2),
            'best_ask': round(self.best_ask, 2),
            'spread': round(self.spread, 4),
            'spread_percent': round(self.spread_percent, 6)
        }


class OrderFlowStream:
    """
    Real-time Order Flow Analysis Stream
    
    Connects to Binance WebSocket streams for aggregated trades and depth data.
    Computes real-time order flow metrics including CVD, imbalance, and absorption.
    
    Thread-safe for concurrent access to metrics.
    Stores rolling 5-minute windows of data.
    """
    
    BINANCE_WS_BASE = "wss://stream.binance.com:9443/ws"
    BINANCE_FUTURES_WS_BASE = "wss://fstream.binance.com/ws"
    
    LARGE_ORDER_THRESHOLD = 10000.0
    ROLLING_WINDOW_SECONDS = 300
    ABSORPTION_PRICE_THRESHOLD = 0.001
    ABSORPTION_VOLUME_THRESHOLD = 50000.0
    
    def __init__(
        self,
        symbol: str = "ETHUSDT",
        large_order_threshold: float = 10000.0,
        rolling_window_seconds: int = 300,
        on_trade_callback: Optional[Callable[[TradeData], None]] = None,
        on_depth_callback: Optional[Callable[[DepthData], None]] = None,
        on_metrics_callback: Optional[Callable[[OrderFlowMetrics], None]] = None,
        use_futures: bool = False
    ):
        """
        Initialize Order Flow Stream
        
        Args:
            symbol: Trading pair symbol (default 'ETHUSDT')
            large_order_threshold: Notional value threshold for large orders (default $10,000)
            rolling_window_seconds: Rolling window for metrics in seconds (default 300 = 5 min)
            on_trade_callback: Callback function for each trade
            on_depth_callback: Callback function for depth updates
            on_metrics_callback: Callback function for metrics updates
            use_futures: Use futures WebSocket instead of spot
        """
        self.symbol = symbol.upper()
        self.symbol_lower = symbol.lower()
        self.large_order_threshold = large_order_threshold
        self.rolling_window_seconds = rolling_window_seconds
        self.use_futures = use_futures
        
        self.on_trade_callback = on_trade_callback
        self.on_depth_callback = on_depth_callback
        self.on_metrics_callback = on_metrics_callback
        
        self._lock = threading.RLock()
        
        self._trades: Deque[TradeData] = deque(maxlen=50000)
        self._large_trades: Deque[TradeData] = deque(maxlen=1000)
        self._depth_snapshots: Deque[DepthData] = deque(maxlen=1000)
        
        self._cumulative_delta: float = 0.0
        self._prev_delta: float = 0.0
        self._buy_volume: float = 0.0
        self._sell_volume: float = 0.0
        self._total_notional: float = 0.0
        self._total_volume: float = 0.0
        
        self._last_depth: Optional[DepthData] = None
        self._last_price: float = 0.0
        self._price_at_absorption_start: float = 0.0
        self._absorption_volume: float = 0.0
        
        self._trade_count_window: Deque[float] = deque(maxlen=10000)
        
        self._running = False
        self._ws_session: Optional[aiohttp.ClientSession] = None
        self._trade_ws: Optional[Any] = None
        self._depth_ws: Optional[Any] = None
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0
        
        self._metrics_task: Optional[asyncio.Task] = None
        self._trade_task: Optional[asyncio.Task] = None
        self._depth_task: Optional[asyncio.Task] = None
        
        logger.info(f"OrderFlowStream initialized for {self.symbol}")
    
    async def start(self) -> None:
        """Start the WebSocket connections and metrics computation"""
        if self._running:
            logger.warning("OrderFlowStream is already running")
            return
        
        self._running = True
        logger.info(f"Starting OrderFlowStream for {self.symbol}")
        
        try:
            self._trade_task = asyncio.create_task(self._run_trade_stream())
            self._depth_task = asyncio.create_task(self._run_depth_stream())
            self._metrics_task = asyncio.create_task(self._run_metrics_updater())
            
            logger.info("OrderFlowStream started successfully")
        except Exception as e:
            logger.error(f"Error starting OrderFlowStream: {e}")
            self._running = False
            raise
    
    async def stop(self) -> None:
        """Stop all WebSocket connections and tasks"""
        if not self._running:
            return
        
        self._running = False
        logger.info("Stopping OrderFlowStream...")
        
        for task in [self._trade_task, self._depth_task, self._metrics_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        if self._ws_session and not self._ws_session.closed:
            await self._ws_session.close()
        
        logger.info("OrderFlowStream stopped")
    
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
    
    async def _run_depth_stream(self) -> None:
        """Run the depth stream with auto-reconnection"""
        while self._running:
            try:
                await self._connect_depth_stream()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Depth stream error: {e}")
                if self._running:
                    await asyncio.sleep(self._reconnect_delay)
                    self._reconnect_delay = min(
                        self._reconnect_delay * 2,
                        self._max_reconnect_delay
                    )
    
    async def _connect_trade_stream(self) -> None:
        """Connect to the aggTrade WebSocket stream"""
        if not AIOHTTP_AVAILABLE:
            logger.error("aiohttp not available for WebSocket connection")
            return
        
        base_url = self.BINANCE_FUTURES_WS_BASE if self.use_futures else self.BINANCE_WS_BASE
        stream_url = f"{base_url}/{self.symbol_lower}@aggTrade"
        
        logger.info(f"Connecting to trade stream: {stream_url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(stream_url, heartbeat=30) as ws:
                self._trade_ws = ws
                self._reconnect_delay = 1.0
                logger.info("Trade stream connected")
                
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
                        logger.error(f"Trade WebSocket error: {ws.exception()}")
                        break
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        logger.warning("Trade WebSocket closed")
                        break
    
    async def _connect_depth_stream(self) -> None:
        """Connect to the depth20 WebSocket stream"""
        if not AIOHTTP_AVAILABLE:
            logger.error("aiohttp not available for WebSocket connection")
            return
        
        base_url = self.BINANCE_FUTURES_WS_BASE if self.use_futures else self.BINANCE_WS_BASE
        stream_url = f"{base_url}/{self.symbol_lower}@depth20@100ms"
        
        logger.info(f"Connecting to depth stream: {stream_url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(stream_url, heartbeat=30) as ws:
                self._depth_ws = ws
                logger.info("Depth stream connected")
                
                async for msg in ws:
                    if not self._running:
                        break
                    
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            data = json.loads(msg.data)
                            await self._process_depth(data)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse depth message: {e}")
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"Depth WebSocket error: {ws.exception()}")
                        break
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        logger.warning("Depth WebSocket closed")
                        break
    
    async def _process_trade(self, data: Dict) -> None:
        """Process incoming aggTrade data"""
        try:
            trade = TradeData(
                timestamp=float(data.get('T', time.time() * 1000)) / 1000.0,
                price=float(data.get('p', 0)),
                quantity=float(data.get('q', 0)),
                is_buyer_maker=data.get('m', False),
                trade_id=int(data.get('a', 0))
            )
            
            with self._lock:
                self._trades.append(trade)
                self._trade_count_window.append(trade.timestamp)
                self._last_price = trade.price
                
                if trade.is_buyer_maker:
                    self._sell_volume += trade.quantity
                    self._cumulative_delta -= trade.quantity
                else:
                    self._buy_volume += trade.quantity
                    self._cumulative_delta += trade.quantity
                
                self._total_volume += trade.quantity
                self._total_notional += trade.notional_value
                
                if trade.notional_value >= self.large_order_threshold:
                    self._large_trades.append(trade)
                    logger.debug(
                        f"Large order detected: {trade.quantity:.4f} @ ${trade.price:.2f} "
                        f"(${trade.notional_value:.2f}) - {'SELL' if trade.is_buyer_maker else 'BUY'}"
                    )
                
                self._update_absorption_tracking(trade)
            
            if self.on_trade_callback:
                try:
                    self.on_trade_callback(trade)
                except Exception as e:
                    logger.warning(f"Trade callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing trade: {e}")
    
    async def _process_depth(self, data: Dict) -> None:
        """Process incoming depth data"""
        try:
            depth = DepthData(
                timestamp=time.time(),
                bids=[[float(b[0]), float(b[1])] for b in data.get('bids', [])],
                asks=[[float(a[0]), float(a[1])] for a in data.get('asks', [])]
            )
            
            with self._lock:
                self._depth_snapshots.append(depth)
                self._last_depth = depth
            
            if self.on_depth_callback:
                try:
                    self.on_depth_callback(depth)
                except Exception as e:
                    logger.warning(f"Depth callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing depth: {e}")
    
    def _update_absorption_tracking(self, trade: TradeData) -> None:
        """Track absorption patterns - large volume without price movement"""
        if self._price_at_absorption_start == 0:
            self._price_at_absorption_start = trade.price
            self._absorption_volume = 0.0
        
        self._absorption_volume += trade.notional_value
        
        price_change_percent = abs(trade.price - self._price_at_absorption_start) / self._price_at_absorption_start
        
        if price_change_percent > self.ABSORPTION_PRICE_THRESHOLD:
            self._price_at_absorption_start = trade.price
            self._absorption_volume = 0.0
    
    async def _run_metrics_updater(self) -> None:
        """Periodically update and emit metrics"""
        while self._running:
            try:
                await asyncio.sleep(1.0)
                
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
    
    def get_current_metrics(self) -> OrderFlowMetrics:
        """
        Get current order flow metrics snapshot (thread-safe)
        
        Returns:
            OrderFlowMetrics dataclass with all computed metrics
        """
        with self._lock:
            return self._compute_metrics()
    
    def _compute_metrics(self) -> OrderFlowMetrics:
        """Compute all order flow metrics (must be called with lock held)"""
        now = time.time()
        cutoff_time = now - self.rolling_window_seconds
        
        window_trades = [t for t in self._trades if t.timestamp >= cutoff_time]
        window_large_trades = [t for t in self._large_trades if t.timestamp >= cutoff_time]
        
        window_buy_volume = sum(t.quantity for t in window_trades if not t.is_buyer_maker)
        window_sell_volume = sum(t.quantity for t in window_trades if t.is_buyer_maker)
        window_total_volume = window_buy_volume + window_sell_volume
        
        window_delta = window_buy_volume - window_sell_volume
        delta_change = window_delta - self._prev_delta
        self._prev_delta = window_delta
        
        imbalance = 0.0
        best_bid = 0.0
        best_ask = 0.0
        spread = 0.0
        spread_percent = 0.0
        
        if self._last_depth:
            total_book_volume = self._last_depth.bid_volume + self._last_depth.ask_volume
            if total_book_volume > 0:
                imbalance = (self._last_depth.bid_volume - self._last_depth.ask_volume) / total_book_volume
            
            if self._last_depth.bids:
                best_bid = self._last_depth.bids[0][0]
            if self._last_depth.asks:
                best_ask = self._last_depth.asks[0][0]
            
            if best_bid > 0 and best_ask > 0:
                spread = best_ask - best_bid
                mid_price = (best_bid + best_ask) / 2
                spread_percent = spread / mid_price if mid_price > 0 else 0
        
        large_buy_count = sum(1 for t in window_large_trades if not t.is_buyer_maker)
        large_sell_count = sum(1 for t in window_large_trades if t.is_buyer_maker)
        
        absorption_score = self._compute_absorption_score(window_trades)
        
        recent_trades = [t for t in self._trade_count_window if t >= now - 1.0]
        tape_speed = len(recent_trades)
        
        vwap = 0.0
        if window_trades:
            total_pv = sum(t.price * t.quantity for t in window_trades)
            total_v = sum(t.quantity for t in window_trades)
            if total_v > 0:
                vwap = total_pv / total_v
        
        return OrderFlowMetrics(
            timestamp=datetime.utcnow(),
            cumulative_delta=self._cumulative_delta,
            delta_change=delta_change,
            bid_ask_imbalance=imbalance,
            large_order_count=len(window_large_trades),
            large_buy_count=large_buy_count,
            large_sell_count=large_sell_count,
            absorption_score=absorption_score,
            tape_speed=tape_speed,
            buy_volume=window_buy_volume,
            sell_volume=window_sell_volume,
            total_volume=window_total_volume,
            vwap=vwap,
            last_price=self._last_price,
            best_bid=best_bid,
            best_ask=best_ask,
            spread=spread,
            spread_percent=spread_percent
        )
    
    def _compute_absorption_score(self, trades: List[TradeData]) -> float:
        """
        Compute absorption score (0-1)
        
        Absorption occurs when large volume is traded without significant price movement.
        Higher score indicates stronger absorption (institutional activity).
        """
        if not trades or len(trades) < 10:
            return 0.0
        
        total_notional = sum(t.notional_value for t in trades)
        if total_notional < self.ABSORPTION_VOLUME_THRESHOLD * 0.1:
            return 0.0
        
        price_start = trades[0].price
        price_end = trades[-1].price
        price_high = max(t.price for t in trades)
        price_low = min(t.price for t in trades)
        
        if price_start == 0:
            return 0.0
        
        price_range = (price_high - price_low) / price_start
        net_change = abs(price_end - price_start) / price_start
        
        volume_factor = min(1.0, total_notional / self.ABSORPTION_VOLUME_THRESHOLD)
        
        if price_range < 0.0001:
            price_stability = 1.0
        else:
            price_stability = max(0, 1.0 - (price_range / 0.01))
        
        if net_change < 0.0001:
            reversion_factor = 1.0
        else:
            reversion_factor = max(0, 1.0 - (net_change / price_range)) if price_range > 0 else 0.5
        
        absorption_score = volume_factor * 0.4 + price_stability * 0.35 + reversion_factor * 0.25
        
        return min(1.0, max(0.0, absorption_score))
    
    def get_rolling_trades(self, seconds: int = 60) -> List[TradeData]:
        """
        Get trades from the last N seconds (thread-safe)
        
        Args:
            seconds: Number of seconds to look back
            
        Returns:
            List of TradeData objects
        """
        cutoff = time.time() - seconds
        with self._lock:
            return [t for t in self._trades if t.timestamp >= cutoff]
    
    def get_large_orders(self, seconds: int = 300) -> List[TradeData]:
        """
        Get large orders from the last N seconds (thread-safe)
        
        Args:
            seconds: Number of seconds to look back
            
        Returns:
            List of large TradeData objects
        """
        cutoff = time.time() - seconds
        with self._lock:
            return [t for t in self._large_trades if t.timestamp >= cutoff]
    
    def get_order_book_snapshot(self) -> Optional[DepthData]:
        """
        Get the latest order book snapshot (thread-safe)
        
        Returns:
            DepthData object or None
        """
        with self._lock:
            return self._last_depth
    
    def get_cvd_history(self, seconds: int = 300, interval: int = 10) -> List[Dict[str, float]]:
        """
        Get CVD history at specified intervals
        
        Args:
            seconds: Total seconds of history
            interval: Interval between data points in seconds
            
        Returns:
            List of dicts with timestamp and cumulative_delta
        """
        now = time.time()
        cutoff = now - seconds
        
        with self._lock:
            trades = [t for t in self._trades if t.timestamp >= cutoff]
        
        if not trades:
            return []
        
        history = []
        running_delta = 0.0
        current_interval_start = cutoff
        
        for trade in sorted(trades, key=lambda x: x.timestamp):
            while trade.timestamp >= current_interval_start + interval:
                if history or running_delta != 0:
                    history.append({
                        'timestamp': current_interval_start + interval,
                        'cumulative_delta': running_delta
                    })
                current_interval_start += interval
            
            if trade.is_buyer_maker:
                running_delta -= trade.quantity
            else:
                running_delta += trade.quantity
        
        history.append({
            'timestamp': now,
            'cumulative_delta': running_delta
        })
        
        return history
    
    def get_tape_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive tape analysis
        
        Returns:
            Dictionary with tape analysis metrics
        """
        with self._lock:
            now = time.time()
            
            trades_1m = [t for t in self._trades if t.timestamp >= now - 60]
            trades_5m = [t for t in self._trades if t.timestamp >= now - 300]
            
            analysis = {
                'trades_1m_count': len(trades_1m),
                'trades_5m_count': len(trades_5m),
                'avg_trade_size_1m': 0.0,
                'avg_trade_size_5m': 0.0,
                'buy_pressure_1m': 0.0,
                'buy_pressure_5m': 0.0,
                'large_trade_ratio_1m': 0.0,
                'large_trade_ratio_5m': 0.0
            }
            
            if trades_1m:
                analysis['avg_trade_size_1m'] = sum(t.notional_value for t in trades_1m) / len(trades_1m)
                buy_count = sum(1 for t in trades_1m if not t.is_buyer_maker)
                analysis['buy_pressure_1m'] = buy_count / len(trades_1m)
                large_count = sum(1 for t in trades_1m if t.notional_value >= self.large_order_threshold)
                analysis['large_trade_ratio_1m'] = large_count / len(trades_1m)
            
            if trades_5m:
                analysis['avg_trade_size_5m'] = sum(t.notional_value for t in trades_5m) / len(trades_5m)
                buy_count = sum(1 for t in trades_5m if not t.is_buyer_maker)
                analysis['buy_pressure_5m'] = buy_count / len(trades_5m)
                large_count = sum(1 for t in trades_5m if t.notional_value >= self.large_order_threshold)
                analysis['large_trade_ratio_5m'] = large_count / len(trades_5m)
            
            return analysis
    
    def reset_cumulative_delta(self) -> None:
        """Reset cumulative delta to zero (thread-safe)"""
        with self._lock:
            self._cumulative_delta = 0.0
            self._prev_delta = 0.0
            self._buy_volume = 0.0
            self._sell_volume = 0.0
            logger.info("Cumulative delta reset to zero")
    
    def clear_history(self) -> None:
        """Clear all historical data (thread-safe)"""
        with self._lock:
            self._trades.clear()
            self._large_trades.clear()
            self._depth_snapshots.clear()
            self._trade_count_window.clear()
            self._cumulative_delta = 0.0
            self._prev_delta = 0.0
            self._buy_volume = 0.0
            self._sell_volume = 0.0
            self._total_volume = 0.0
            self._total_notional = 0.0
            logger.info("Order flow history cleared")
    
    @property
    def is_running(self) -> bool:
        """Check if stream is running"""
        return self._running
    
    @property
    def symbol_info(self) -> str:
        """Get formatted symbol info"""
        return f"{self.symbol} ({'Futures' if self.use_futures else 'Spot'})"
    
    async def close(self) -> None:
        """Close all connections and cleanup resources"""
        await self.stop()
        logger.info("OrderFlowStream closed")


async def create_order_flow_stream(
    symbol: str = "ETHUSDT",
    on_trade: Optional[Callable[[TradeData], None]] = None,
    on_depth: Optional[Callable[[DepthData], None]] = None,
    on_metrics: Optional[Callable[[OrderFlowMetrics], None]] = None,
    **kwargs
) -> OrderFlowStream:
    """
    Factory function to create and start an OrderFlowStream
    
    Args:
        symbol: Trading pair symbol
        on_trade: Trade callback function
        on_depth: Depth callback function
        on_metrics: Metrics callback function
        **kwargs: Additional arguments for OrderFlowStream
        
    Returns:
        Started OrderFlowStream instance
    """
    stream = OrderFlowStream(
        symbol=symbol,
        on_trade_callback=on_trade,
        on_depth_callback=on_depth,
        on_metrics_callback=on_metrics,
        **kwargs
    )
    await stream.start()
    return stream


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    def on_trade(trade: TradeData):
        side = "SELL" if trade.is_buyer_maker else "BUY"
        print(f"Trade: {side} {trade.quantity:.4f} @ ${trade.price:.2f} (${trade.notional_value:.2f})")
    
    def on_metrics(metrics: OrderFlowMetrics):
        print(f"\nMetrics Update:")
        print(f"  CVD: {metrics.cumulative_delta:.4f}")
        print(f"  Delta Change: {metrics.delta_change:.4f}")
        print(f"  Imbalance: {metrics.bid_ask_imbalance:.4f}")
        print(f"  Large Orders: {metrics.large_order_count}")
        print(f"  Tape Speed: {metrics.tape_speed:.1f} trades/sec")
        print(f"  Absorption: {metrics.absorption_score:.2f}")
        print(f"  Spread: {metrics.spread:.4f} ({metrics.spread_percent*100:.4f}%)")
    
    async def main():
        stream = await create_order_flow_stream(
            symbol="ETHUSDT",
            on_metrics=on_metrics,
            large_order_threshold=10000.0
        )
        
        try:
            print("Order Flow Stream running. Press Ctrl+C to stop.")
            while True:
                await asyncio.sleep(60)
                analysis = stream.get_tape_analysis()
                print(f"\nTape Analysis: {json.dumps(analysis, indent=2)}")
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            await stream.close()
    
    asyncio.run(main())
