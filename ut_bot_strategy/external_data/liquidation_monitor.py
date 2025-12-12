"""
Liquidation Monitor for Real-time Binance Futures Liquidation Data

Streams real-time liquidation data from Binance's free WebSocket stream.
Tracks rolling 15-minute liquidation totals, large liquidations, and imbalance metrics.
"""

import asyncio
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Deque, Callable

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    websockets = None
    WEBSOCKETS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class LiquidationEvent:
    """Individual liquidation event data"""
    symbol: str
    side: str
    price: float
    quantity: float
    value_usd: float
    timestamp: datetime
    is_large: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'price': round(self.price, 4),
            'quantity': round(self.quantity, 6),
            'value_usd': round(self.value_usd, 2),
            'timestamp': self.timestamp.isoformat(),
            'is_large': self.is_large
        }


@dataclass
class LiquidationMetrics:
    """Current liquidation metrics snapshot"""
    long_liquidations_usd: float
    short_liquidations_usd: float
    total_liquidations_usd: float
    liquidation_imbalance: float
    large_liquidation_count: int
    liquidation_intensity: str
    recent_large_liquidations: List[LiquidationEvent]
    signal_bias: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'long_liquidations_usd': round(self.long_liquidations_usd, 2),
            'short_liquidations_usd': round(self.short_liquidations_usd, 2),
            'total_liquidations_usd': round(self.total_liquidations_usd, 2),
            'liquidation_imbalance': round(self.liquidation_imbalance, 4),
            'large_liquidation_count': self.large_liquidation_count,
            'liquidation_intensity': self.liquidation_intensity,
            'recent_large_liquidations': [e.to_dict() for e in self.recent_large_liquidations],
            'signal_bias': self.signal_bias,
            'timestamp': self.timestamp.isoformat()
        }


class LiquidationMonitor:
    """
    Real-time Liquidation Monitor for Binance Futures
    
    Connects to Binance's free WebSocket stream for liquidation data.
    Tracks rolling 15-minute windows of liquidation events.
    Provides metrics including imbalance, intensity, and large liquidation tracking.
    """
    
    WEBSOCKET_URL = "wss://fstream.binance.com/ws/!forceOrder@arr"
    LARGE_LIQUIDATION_THRESHOLD = 100000.0
    ROLLING_WINDOW_SECONDS = 900
    HISTORICAL_AVG_15M = 5000000.0
    HIGH_INTENSITY_THRESHOLD = 10000000.0
    EXTREME_INTENSITY_THRESHOLD = 25000000.0
    
    def __init__(
        self,
        large_threshold: float = 100000.0,
        rolling_window_seconds: int = 900,
        on_liquidation_callback: Optional[Callable[[LiquidationEvent], None]] = None,
        on_metrics_callback: Optional[Callable[[LiquidationMetrics], None]] = None
    ):
        """
        Initialize Liquidation Monitor
        
        Args:
            large_threshold: USD threshold for large liquidations (default $100,000)
            rolling_window_seconds: Rolling window in seconds (default 900 = 15 min)
            on_liquidation_callback: Callback for each liquidation event
            on_metrics_callback: Callback for metrics updates
        """
        self.large_threshold = large_threshold
        self.rolling_window_seconds = rolling_window_seconds
        self.on_liquidation_callback = on_liquidation_callback
        self.on_metrics_callback = on_metrics_callback
        
        self._lock = threading.RLock()
        
        self._all_liquidations: Deque[LiquidationEvent] = deque(maxlen=100000)
        self._large_liquidations: Deque[LiquidationEvent] = deque(maxlen=10000)
        self._symbol_liquidations: Dict[str, Deque[LiquidationEvent]] = {}
        
        self._running = False
        self._ws = None
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0
        self._ws_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        
        self._total_long_volume = 0.0
        self._total_short_volume = 0.0
        self._event_count = 0
        
        logger.info("LiquidationMonitor initialized")
    
    async def start(self) -> None:
        """Start the WebSocket connection and metrics computation"""
        if not WEBSOCKETS_AVAILABLE:
            logger.error("websockets library not available")
            raise ImportError("websockets library is required but not installed")
        
        if self._running:
            logger.warning("LiquidationMonitor is already running")
            return
        
        self._running = True
        logger.info("Starting LiquidationMonitor")
        
        try:
            self._ws_task = asyncio.create_task(self._run_websocket())
            self._metrics_task = asyncio.create_task(self._run_metrics_updater())
            logger.info("LiquidationMonitor started successfully")
        except Exception as e:
            logger.error(f"Error starting LiquidationMonitor: {e}")
            self._running = False
            raise
    
    async def stop(self) -> None:
        """Stop the WebSocket connection and cleanup"""
        if not self._running:
            return
        
        self._running = False
        logger.info("Stopping LiquidationMonitor...")
        
        for task in [self._ws_task, self._metrics_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        if self._ws:
            await self._ws.close()
            self._ws = None
        
        logger.info("LiquidationMonitor stopped")
    
    async def _run_websocket(self) -> None:
        """Run WebSocket connection with auto-reconnection"""
        while self._running:
            try:
                await self._connect_websocket()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if self._running:
                    logger.info(f"Reconnecting in {self._reconnect_delay} seconds...")
                    await asyncio.sleep(self._reconnect_delay)
                    self._reconnect_delay = min(
                        self._reconnect_delay * 2,
                        self._max_reconnect_delay
                    )
    
    async def _connect_websocket(self) -> None:
        """Connect to Binance liquidation WebSocket stream"""
        if not WEBSOCKETS_AVAILABLE or websockets is None:
            logger.error("websockets library not available")
            return
            
        logger.info(f"Connecting to {self.WEBSOCKET_URL}")
        
        async with websockets.connect(
            self.WEBSOCKET_URL,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5
        ) as ws:
            self._ws = ws
            self._reconnect_delay = 1.0
            logger.info("Liquidation WebSocket connected")
            
            async for message in ws:
                if not self._running:
                    break
                
                try:
                    data = json.loads(message)
                    await self._process_liquidation(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse message: {e}")
                except Exception as e:
                    logger.error(f"Error processing liquidation: {e}")
    
    async def _process_liquidation(self, data: Dict) -> None:
        """Process incoming liquidation event"""
        try:
            order = data.get('o', {})
            
            symbol = order.get('s', '')
            side = order.get('S', '')
            price = float(order.get('p', 0))
            quantity = float(order.get('q', 0))
            timestamp_ms = int(order.get('T', time.time() * 1000))
            
            value_usd = price * quantity
            is_large = value_usd >= self.large_threshold
            
            event = LiquidationEvent(
                symbol=symbol,
                side=side,
                price=price,
                quantity=quantity,
                value_usd=value_usd,
                timestamp=datetime.utcfromtimestamp(timestamp_ms / 1000.0),
                is_large=is_large
            )
            
            with self._lock:
                self._all_liquidations.append(event)
                self._event_count += 1
                
                if symbol not in self._symbol_liquidations:
                    self._symbol_liquidations[symbol] = deque(maxlen=10000)
                self._symbol_liquidations[symbol].append(event)
                
                if is_large:
                    self._large_liquidations.append(event)
                    logger.info(
                        f"Large liquidation: {symbol} {side} "
                        f"{quantity:.4f} @ ${price:.2f} (${value_usd:,.2f})"
                    )
                
                if side == "SELL":
                    self._total_long_volume += value_usd
                else:
                    self._total_short_volume += value_usd
            
            if self.on_liquidation_callback:
                try:
                    self.on_liquidation_callback(event)
                except Exception as e:
                    logger.warning(f"Liquidation callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing liquidation data: {e}")
    
    async def _run_metrics_updater(self) -> None:
        """Periodically update and emit metrics"""
        while self._running:
            try:
                await asyncio.sleep(5.0)
                
                if self.on_metrics_callback:
                    metrics = self.get_metrics()
                    try:
                        self.on_metrics_callback(metrics)
                    except Exception as e:
                        logger.warning(f"Metrics callback error: {e}")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics updater error: {e}")
    
    def get_metrics(self) -> LiquidationMetrics:
        """
        Get current liquidation metrics (thread-safe)
        
        Returns:
            LiquidationMetrics dataclass with all computed metrics
        """
        with self._lock:
            return self._compute_metrics()
    
    def _compute_metrics(self) -> LiquidationMetrics:
        """Compute all liquidation metrics (must be called with lock held)"""
        now = datetime.utcnow()
        cutoff_time = now - timedelta(seconds=self.rolling_window_seconds)
        
        window_liquidations = [
            e for e in self._all_liquidations
            if e.timestamp >= cutoff_time
        ]
        
        long_liquidations_usd = sum(
            e.value_usd for e in window_liquidations
            if e.side == "SELL"
        )
        short_liquidations_usd = sum(
            e.value_usd for e in window_liquidations
            if e.side == "BUY"
        )
        total_liquidations_usd = long_liquidations_usd + short_liquidations_usd
        
        if total_liquidations_usd > 0:
            liquidation_imbalance = (long_liquidations_usd - short_liquidations_usd) / total_liquidations_usd
        else:
            liquidation_imbalance = 0.0
        
        window_large = [
            e for e in self._large_liquidations
            if e.timestamp >= cutoff_time
        ]
        large_liquidation_count = len(window_large)
        
        intensity_ratio = total_liquidations_usd / self.HISTORICAL_AVG_15M
        if intensity_ratio >= 5.0:
            liquidation_intensity = "extreme"
        elif intensity_ratio >= 2.0:
            liquidation_intensity = "high"
        elif intensity_ratio >= 0.5:
            liquidation_intensity = "medium"
        else:
            liquidation_intensity = "low"
        
        if short_liquidations_usd > long_liquidations_usd * 1.5:
            signal_bias = "bullish"
        elif long_liquidations_usd > short_liquidations_usd * 1.5:
            signal_bias = "bearish"
        else:
            signal_bias = "neutral"
        
        recent_large = sorted(
            window_large,
            key=lambda x: x.value_usd,
            reverse=True
        )[:10]
        
        return LiquidationMetrics(
            long_liquidations_usd=long_liquidations_usd,
            short_liquidations_usd=short_liquidations_usd,
            total_liquidations_usd=total_liquidations_usd,
            liquidation_imbalance=liquidation_imbalance,
            large_liquidation_count=large_liquidation_count,
            liquidation_intensity=liquidation_intensity,
            recent_large_liquidations=recent_large,
            signal_bias=signal_bias,
            timestamp=now
        )
    
    def get_symbol_metrics(self, symbol: str) -> LiquidationMetrics:
        """
        Get metrics for a specific symbol (thread-safe)
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            LiquidationMetrics for the specified symbol
        """
        symbol = symbol.upper()
        
        with self._lock:
            now = datetime.utcnow()
            cutoff_time = now - timedelta(seconds=self.rolling_window_seconds)
            
            if symbol not in self._symbol_liquidations:
                return LiquidationMetrics(
                    long_liquidations_usd=0.0,
                    short_liquidations_usd=0.0,
                    total_liquidations_usd=0.0,
                    liquidation_imbalance=0.0,
                    large_liquidation_count=0,
                    liquidation_intensity="low",
                    recent_large_liquidations=[],
                    signal_bias="neutral",
                    timestamp=now
                )
            
            symbol_events = [
                e for e in self._symbol_liquidations[symbol]
                if e.timestamp >= cutoff_time
            ]
            
            long_liquidations_usd = sum(
                e.value_usd for e in symbol_events
                if e.side == "SELL"
            )
            short_liquidations_usd = sum(
                e.value_usd for e in symbol_events
                if e.side == "BUY"
            )
            total_liquidations_usd = long_liquidations_usd + short_liquidations_usd
            
            if total_liquidations_usd > 0:
                liquidation_imbalance = (long_liquidations_usd - short_liquidations_usd) / total_liquidations_usd
            else:
                liquidation_imbalance = 0.0
            
            large_events = [e for e in symbol_events if e.is_large]
            large_liquidation_count = len(large_events)
            
            symbol_avg = self.HISTORICAL_AVG_15M / 100
            intensity_ratio = total_liquidations_usd / symbol_avg if symbol_avg > 0 else 0
            if intensity_ratio >= 5.0:
                liquidation_intensity = "extreme"
            elif intensity_ratio >= 2.0:
                liquidation_intensity = "high"
            elif intensity_ratio >= 0.5:
                liquidation_intensity = "medium"
            else:
                liquidation_intensity = "low"
            
            if short_liquidations_usd > long_liquidations_usd * 1.5:
                signal_bias = "bullish"
            elif long_liquidations_usd > short_liquidations_usd * 1.5:
                signal_bias = "bearish"
            else:
                signal_bias = "neutral"
            
            recent_large = sorted(
                large_events,
                key=lambda x: x.value_usd,
                reverse=True
            )[:10]
            
            return LiquidationMetrics(
                long_liquidations_usd=long_liquidations_usd,
                short_liquidations_usd=short_liquidations_usd,
                total_liquidations_usd=total_liquidations_usd,
                liquidation_imbalance=liquidation_imbalance,
                large_liquidation_count=large_liquidation_count,
                liquidation_intensity=liquidation_intensity,
                recent_large_liquidations=recent_large,
                signal_bias=signal_bias,
                timestamp=now
            )
    
    def get_recent_liquidations(self, seconds: int = 60, symbol: Optional[str] = None) -> List[LiquidationEvent]:
        """
        Get recent liquidation events (thread-safe)
        
        Args:
            seconds: Number of seconds to look back
            symbol: Optional symbol filter
            
        Returns:
            List of LiquidationEvent objects
        """
        cutoff_time = datetime.utcnow() - timedelta(seconds=seconds)
        
        with self._lock:
            if symbol:
                symbol = symbol.upper()
                if symbol not in self._symbol_liquidations:
                    return []
                return [
                    e for e in self._symbol_liquidations[symbol]
                    if e.timestamp >= cutoff_time
                ]
            else:
                return [
                    e for e in self._all_liquidations
                    if e.timestamp >= cutoff_time
                ]
    
    def get_large_liquidations(self, seconds: int = 900) -> List[LiquidationEvent]:
        """
        Get large liquidation events (thread-safe)
        
        Args:
            seconds: Number of seconds to look back
            
        Returns:
            List of large LiquidationEvent objects
        """
        cutoff_time = datetime.utcnow() - timedelta(seconds=seconds)
        
        with self._lock:
            return [
                e for e in self._large_liquidations
                if e.timestamp >= cutoff_time
            ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics (thread-safe)
        
        Returns:
            Dictionary with statistics
        """
        with self._lock:
            return {
                'total_events_processed': self._event_count,
                'total_long_volume_usd': round(self._total_long_volume, 2),
                'total_short_volume_usd': round(self._total_short_volume, 2),
                'total_volume_usd': round(self._total_long_volume + self._total_short_volume, 2),
                'large_liquidations_stored': len(self._large_liquidations),
                'symbols_tracked': len(self._symbol_liquidations),
                'is_running': self._running
            }
    
    def clear_history(self) -> None:
        """Clear all historical data (thread-safe)"""
        with self._lock:
            self._all_liquidations.clear()
            self._large_liquidations.clear()
            self._symbol_liquidations.clear()
            self._total_long_volume = 0.0
            self._total_short_volume = 0.0
            self._event_count = 0
            logger.info("Liquidation history cleared")
    
    @property
    def is_running(self) -> bool:
        """Check if monitor is running"""
        return self._running


async def create_liquidation_monitor(
    on_liquidation: Optional[Callable[[LiquidationEvent], None]] = None,
    on_metrics: Optional[Callable[[LiquidationMetrics], None]] = None,
    **kwargs
) -> LiquidationMonitor:
    """
    Factory function to create and start a LiquidationMonitor
    
    Args:
        on_liquidation: Liquidation event callback
        on_metrics: Metrics callback
        **kwargs: Additional arguments for LiquidationMonitor
        
    Returns:
        Started LiquidationMonitor instance
    """
    monitor = LiquidationMonitor(
        on_liquidation_callback=on_liquidation,
        on_metrics_callback=on_metrics,
        **kwargs
    )
    await monitor.start()
    return monitor


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    def on_liquidation(event: LiquidationEvent):
        side_type = "LONG" if event.side == "SELL" else "SHORT"
        print(
            f"Liquidation: {event.symbol} {side_type} "
            f"{event.quantity:.4f} @ ${event.price:.2f} "
            f"(${event.value_usd:,.2f})"
            f"{' [LARGE]' if event.is_large else ''}"
        )
    
    def on_metrics(metrics: LiquidationMetrics):
        print(f"\n{'='*60}")
        print(f"Metrics Update ({metrics.timestamp.strftime('%H:%M:%S')})")
        print(f"  Long Liquidations (15m):  ${metrics.long_liquidations_usd:,.2f}")
        print(f"  Short Liquidations (15m): ${metrics.short_liquidations_usd:,.2f}")
        print(f"  Total Liquidations (15m): ${metrics.total_liquidations_usd:,.2f}")
        print(f"  Imbalance: {metrics.liquidation_imbalance:+.4f}")
        print(f"  Intensity: {metrics.liquidation_intensity}")
        print(f"  Large Count: {metrics.large_liquidation_count}")
        print(f"  Signal Bias: {metrics.signal_bias}")
        print(f"{'='*60}\n")
    
    async def main():
        monitor = await create_liquidation_monitor(
            on_liquidation=on_liquidation,
            on_metrics=on_metrics
        )
        
        try:
            print("Liquidation Monitor running. Press Ctrl+C to stop.")
            while True:
                await asyncio.sleep(60)
                stats = monitor.get_statistics()
                print(f"\nStats: {json.dumps(stats, indent=2)}")
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            await monitor.stop()
    
    asyncio.run(main())
