"""
Footprint and Tape Analyzer Module for Advanced Order Flow Analysis

Provides professional-grade footprint chart analysis and tape reading capabilities:
- FootprintBar: Aggregates per-price-level bid/ask volume for candles
- TapeAnalyzer: Analyzes Time & Sales data for trading signals

Based on TML settings (ES-FOOTPRINT):
- bidColor: FF0000 (red for bids)
- askColor: 19B4CD (cyan for asks)  
- dFilt: 99 (99th percentile filter for delta)
- imbDelta1Enabled: true (imbalance detection on)
"""

import logging
import time
import threading
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Any, Deque
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum

try:
    from ..data.order_flow_stream import TradeData, DepthData, OrderFlowMetrics, OrderFlowStream
except ImportError:
    TradeData = None
    DepthData = None
    OrderFlowMetrics = None
    OrderFlowStream = None

logger = logging.getLogger(__name__)


class ImbalanceType(Enum):
    """Type of imbalance detected"""
    BID_IMBALANCE = "bid_imbalance"
    ASK_IMBALANCE = "ask_imbalance"
    NEUTRAL = "neutral"


class AuctionState(Enum):
    """Auction state for finished/unfinished detection"""
    FINISHED = "finished"
    UNFINISHED = "unfinished"
    UNKNOWN = "unknown"


@dataclass
class PriceLevel:
    """Per-price level volume data for footprint analysis"""
    price: float
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    trade_count: int = 0
    bid_count: int = 0
    ask_count: int = 0
    
    @property
    def delta(self) -> float:
        """Volume delta at this level (ask - bid)"""
        return self.ask_volume - self.bid_volume
    
    @property
    def total_volume(self) -> float:
        """Total volume at this level"""
        return self.bid_volume + self.ask_volume
    
    @property
    def imbalance_ratio(self) -> float:
        """Bid/Ask imbalance ratio (>1 = bid heavy, <1 = ask heavy)"""
        if self.ask_volume > 0:
            return self.bid_volume / self.ask_volume
        elif self.bid_volume > 0:
            return float('inf')
        return 1.0
    
    def get_imbalance_type(self, threshold: float = 2.0) -> ImbalanceType:
        """Check if level has significant imbalance"""
        if self.ask_volume > 0 and self.bid_volume / self.ask_volume >= threshold:
            return ImbalanceType.BID_IMBALANCE
        elif self.bid_volume > 0 and self.ask_volume / self.bid_volume >= threshold:
            return ImbalanceType.ASK_IMBALANCE
        return ImbalanceType.NEUTRAL
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'price': self.price,
            'bid_volume': round(self.bid_volume, 4),
            'ask_volume': round(self.ask_volume, 4),
            'delta': round(self.delta, 4),
            'total_volume': round(self.total_volume, 4),
            'trade_count': self.trade_count,
            'imbalance_ratio': round(self.imbalance_ratio, 2) if self.imbalance_ratio != float('inf') else 999.99
        }


@dataclass
class ImbalanceLevel:
    """Detected imbalance at a price level"""
    price: float
    imbalance_type: ImbalanceType
    ratio: float
    bid_volume: float
    ask_volume: float
    delta: float
    strength: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'price': self.price,
            'type': self.imbalance_type.value,
            'ratio': round(self.ratio, 2),
            'bid_volume': round(self.bid_volume, 4),
            'ask_volume': round(self.ask_volume, 4),
            'delta': round(self.delta, 4),
            'strength': round(self.strength, 2)
        }


@dataclass
class StackedImbalance:
    """Consecutive imbalances at adjacent price levels - strong signal"""
    start_price: float
    end_price: float
    imbalance_type: ImbalanceType
    level_count: int
    total_delta: float
    avg_ratio: float
    prices: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'start_price': self.start_price,
            'end_price': self.end_price,
            'type': self.imbalance_type.value,
            'level_count': self.level_count,
            'total_delta': round(self.total_delta, 4),
            'avg_ratio': round(self.avg_ratio, 2),
            'prices': self.prices
        }


@dataclass
class AbsorptionZone:
    """Zone where orders were absorbed without price movement"""
    price: float
    absorbed_volume: float
    direction: str
    price_stability: float
    trade_count: int
    time_duration: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'price': self.price,
            'absorbed_volume': round(self.absorbed_volume, 4),
            'direction': self.direction,
            'price_stability': round(self.price_stability, 4),
            'trade_count': self.trade_count,
            'time_duration': round(self.time_duration, 2)
        }


@dataclass
class SweepEvent:
    """Detected sweep - aggressive market orders clearing multiple levels"""
    timestamp: float
    direction: str
    start_price: float
    end_price: float
    levels_cleared: int
    total_volume: float
    notional_value: float
    trade_ids: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': datetime.fromtimestamp(self.timestamp).isoformat(),
            'direction': self.direction,
            'start_price': self.start_price,
            'end_price': self.end_price,
            'levels_cleared': self.levels_cleared,
            'total_volume': round(self.total_volume, 4),
            'notional_value': round(self.notional_value, 2),
            'price_range': abs(self.end_price - self.start_price)
        }


@dataclass
class LargePrint:
    """Large print detection (>$50k for crypto)"""
    timestamp: float
    price: float
    quantity: float
    notional_value: float
    direction: str
    trade_id: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': datetime.fromtimestamp(self.timestamp).isoformat(),
            'price': self.price,
            'quantity': round(self.quantity, 4),
            'notional_value': round(self.notional_value, 2),
            'direction': self.direction,
            'trade_id': self.trade_id
        }


@dataclass 
class DeltaSpike:
    """Sudden volume imbalance spike"""
    timestamp: float
    delta_value: float
    delta_percentile: float
    direction: str
    trade_count: int
    duration_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': datetime.fromtimestamp(self.timestamp).isoformat(),
            'delta_value': round(self.delta_value, 4),
            'delta_percentile': round(self.delta_percentile, 2),
            'direction': self.direction,
            'trade_count': self.trade_count,
            'duration_seconds': round(self.duration_seconds, 2)
        }


class FootprintBar:
    """
    Footprint Bar - Aggregates per-price-level bid/ask volume for a candle
    
    Features:
    - Per-price bid volume, ask volume, delta per level
    - POC (Point of Control): Price with highest volume
    - High Volume Nodes (HVN) and Low Volume Nodes (LVN)
    - Delta histogram with totals
    - Imbalance detection: When bid/ask ratio exceeds threshold (2:1)
    - Value area (70% of volume)
    
    Based on TML settings with dFilt=99 for 99th percentile filtering
    """
    
    VALUE_AREA_PERCENT = 0.70
    DEFAULT_IMBALANCE_THRESHOLD = 2.0
    DELTA_FILTER_PERCENTILE = 99
    
    def __init__(
        self,
        open_time: float,
        close_time: float,
        tick_size: float = 0.01,
        imbalance_threshold: float = 2.0
    ):
        """
        Initialize FootprintBar
        
        Args:
            open_time: Bar open timestamp
            close_time: Bar close timestamp
            tick_size: Price tick size for level aggregation
            imbalance_threshold: Ratio threshold for imbalance detection (default 2:1)
        """
        self.open_time = open_time
        self.close_time = close_time
        self.tick_size = tick_size
        self.imbalance_threshold = imbalance_threshold
        
        self._levels: Dict[float, PriceLevel] = {}
        self._trades: List[Any] = []
        
        self.open_price: float = 0.0
        self.high_price: float = 0.0
        self.low_price: float = float('inf')
        self.close_price: float = 0.0
        
        self.total_bid_volume: float = 0.0
        self.total_ask_volume: float = 0.0
        self.total_trades: int = 0
        
        self._poc_price: Optional[float] = None
        self._value_area: Optional[Tuple[float, float]] = None
        self._hvn_levels: List[float] = []
        self._lvn_levels: List[float] = []
        
        self._lock = threading.Lock()
    
    def _round_price(self, price: float) -> float:
        """Round price to tick size"""
        return round(price / self.tick_size) * self.tick_size
    
    def add_trade(self, trade: Any) -> None:
        """
        Add a trade to the footprint bar
        
        Args:
            trade: TradeData object from OrderFlowStream
        """
        with self._lock:
            price = self._round_price(trade.price)
            
            if price not in self._levels:
                self._levels[price] = PriceLevel(price=price)
            
            level = self._levels[price]
            level.trade_count += 1
            
            if trade.is_buyer_maker:
                level.bid_volume += trade.quantity
                level.bid_count += 1
                self.total_bid_volume += trade.quantity
            else:
                level.ask_volume += trade.quantity
                level.ask_count += 1
                self.total_ask_volume += trade.quantity
            
            self.total_trades += 1
            self._trades.append(trade)
            
            if self.open_price == 0:
                self.open_price = trade.price
            self.close_price = trade.price
            self.high_price = max(self.high_price, trade.price)
            self.low_price = min(self.low_price, trade.price)
            
            self._poc_price = None
            self._value_area = None
            self._hvn_levels = []
            self._lvn_levels = []
    
    def add_trades(self, trades: List[Any]) -> None:
        """Add multiple trades at once"""
        for trade in trades:
            self.add_trade(trade)
    
    @property
    def levels(self) -> Dict[float, PriceLevel]:
        """Get all price levels"""
        with self._lock:
            return dict(self._levels)
    
    @property
    def sorted_levels(self) -> List[PriceLevel]:
        """Get price levels sorted by price (ascending)"""
        with self._lock:
            return sorted(self._levels.values(), key=lambda x: x.price)
    
    @property
    def total_volume(self) -> float:
        """Total volume in bar"""
        return self.total_bid_volume + self.total_ask_volume
    
    @property
    def total_delta(self) -> float:
        """Total delta (ask volume - bid volume)"""
        return self.total_ask_volume - self.total_bid_volume
    
    @property
    def delta_percent(self) -> float:
        """Delta as percentage of total volume"""
        if self.total_volume > 0:
            return (self.total_delta / self.total_volume) * 100
        return 0.0
    
    def get_poc(self) -> Optional[float]:
        """
        Get Point of Control - price level with highest volume
        
        Returns:
            POC price or None if no data
        """
        with self._lock:
            if self._poc_price is not None:
                return self._poc_price
            
            if not self._levels:
                return None
            
            max_vol = 0
            poc = None
            for price, level in self._levels.items():
                if level.total_volume > max_vol:
                    max_vol = level.total_volume
                    poc = price
            
            self._poc_price = poc
            return poc
    
    def get_value_area(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate Value Area (70% of volume)
        
        Returns:
            Tuple of (VAL, VAH) or (None, None) if insufficient data
        """
        with self._lock:
            if self._value_area is not None:
                return self._value_area
            
            if not self._levels:
                return (None, None)
            
            sorted_levels = sorted(self._levels.values(), key=lambda x: x.price)
            total_vol = sum(l.total_volume for l in sorted_levels)
            target_vol = total_vol * self.VALUE_AREA_PERCENT
            
            poc = self.get_poc()
            if poc is None:
                return (None, None)
            
            poc_idx = next((i for i, l in enumerate(sorted_levels) if l.price == poc), 0)
            
            val_idx = poc_idx
            vah_idx = poc_idx
            current_vol = sorted_levels[poc_idx].total_volume if sorted_levels else 0
            
            while current_vol < target_vol:
                vol_below = 0
                vol_above = 0
                
                if val_idx > 0:
                    vol_below = sorted_levels[val_idx - 1].total_volume
                if vah_idx < len(sorted_levels) - 1:
                    vol_above = sorted_levels[vah_idx + 1].total_volume
                
                if vol_below == 0 and vol_above == 0:
                    break
                
                if vol_below >= vol_above:
                    val_idx -= 1
                    current_vol += vol_below
                else:
                    vah_idx += 1
                    current_vol += vol_above
            
            val = sorted_levels[val_idx].price if val_idx < len(sorted_levels) else None
            vah = sorted_levels[vah_idx].price if vah_idx < len(sorted_levels) else None
            
            self._value_area = (val, vah)
            return (val, vah)
    
    def get_hvn_levels(self, percentile: float = 75.0) -> List[float]:
        """
        Get High Volume Nodes - levels with volume above percentile
        
        Args:
            percentile: Volume percentile threshold (default 75th)
            
        Returns:
            List of HVN prices
        """
        with self._lock:
            if self._hvn_levels:
                return self._hvn_levels
            
            if not self._levels:
                return []
            
            volumes = [l.total_volume for l in self._levels.values()]
            if not volumes:
                return []
            
            threshold = np.percentile(volumes, percentile)
            self._hvn_levels = [
                price for price, level in self._levels.items()
                if level.total_volume >= threshold
            ]
            return self._hvn_levels
    
    def get_lvn_levels(self, percentile: float = 25.0) -> List[float]:
        """
        Get Low Volume Nodes - levels with volume below percentile
        
        Args:
            percentile: Volume percentile threshold (default 25th)
            
        Returns:
            List of LVN prices
        """
        with self._lock:
            if self._lvn_levels:
                return self._lvn_levels
            
            if not self._levels:
                return []
            
            volumes = [l.total_volume for l in self._levels.values()]
            if not volumes:
                return []
            
            threshold = np.percentile(volumes, percentile)
            self._lvn_levels = [
                price for price, level in self._levels.items()
                if level.total_volume <= threshold and level.total_volume > 0
            ]
            return self._lvn_levels
    
    def get_delta_histogram(self) -> Dict[float, float]:
        """
        Get delta values per price level
        
        Returns:
            Dictionary of {price: delta}
        """
        with self._lock:
            return {price: level.delta for price, level in self._levels.items()}
    
    def get_filtered_delta_levels(self, percentile: float = 99.0) -> List[Tuple[float, float]]:
        """
        Get delta levels filtered by percentile (based on dFilt=99 from TML)
        
        Args:
            percentile: Delta percentile filter (default 99th)
            
        Returns:
            List of (price, delta) tuples above percentile
        """
        with self._lock:
            if not self._levels:
                return []
            
            deltas = [abs(level.delta) for level in self._levels.values()]
            if not deltas:
                return []
            
            threshold = np.percentile(deltas, percentile)
            return [
                (price, level.delta)
                for price, level in self._levels.items()
                if abs(level.delta) >= threshold
            ]
    
    def detect_imbalances(self, threshold: Optional[float] = None) -> List[ImbalanceLevel]:
        """
        Detect imbalances at each price level
        
        Args:
            threshold: Ratio threshold for imbalance (default 2.0 = 2:1)
            
        Returns:
            List of ImbalanceLevel objects
        """
        threshold = threshold or self.imbalance_threshold
        imbalances = []
        
        with self._lock:
            max_volume = max((l.total_volume for l in self._levels.values()), default=1)
            
            for price, level in self._levels.items():
                imb_type = level.get_imbalance_type(threshold)
                if imb_type != ImbalanceType.NEUTRAL:
                    strength = level.total_volume / max_volume if max_volume > 0 else 0
                    imbalances.append(ImbalanceLevel(
                        price=price,
                        imbalance_type=imb_type,
                        ratio=level.imbalance_ratio if level.imbalance_ratio != float('inf') else 999.99,
                        bid_volume=level.bid_volume,
                        ask_volume=level.ask_volume,
                        delta=level.delta,
                        strength=strength
                    ))
        
        return sorted(imbalances, key=lambda x: x.price)
    
    def detect_stacked_imbalances(self, min_levels: int = 3, threshold: float = 2.0) -> List[StackedImbalance]:
        """
        Detect stacked imbalances - consecutive imbalances at adjacent prices
        These are strong signals for institutional activity
        
        Args:
            min_levels: Minimum consecutive levels for stacked imbalance (default 3)
            threshold: Imbalance ratio threshold (default 2.0)
            
        Returns:
            List of StackedImbalance objects
        """
        imbalances = self.detect_imbalances(threshold)
        if len(imbalances) < min_levels:
            return []
        
        sorted_imb = sorted(imbalances, key=lambda x: x.price)
        stacked = []
        current_stack: List[ImbalanceLevel] = []
        current_type: Optional[ImbalanceType] = None
        
        for imb in sorted_imb:
            if not current_stack:
                current_stack = [imb]
                current_type = imb.imbalance_type
                continue
            
            last_price = current_stack[-1].price
            price_gap = abs(imb.price - last_price)
            
            if imb.imbalance_type == current_type and price_gap <= self.tick_size * 2:
                current_stack.append(imb)
            else:
                if len(current_stack) >= min_levels and current_type is not None:
                    total_delta = sum(i.delta for i in current_stack)
                    avg_ratio = sum(i.ratio for i in current_stack) / len(current_stack)
                    stacked.append(StackedImbalance(
                        start_price=current_stack[0].price,
                        end_price=current_stack[-1].price,
                        imbalance_type=current_type,
                        level_count=len(current_stack),
                        total_delta=total_delta,
                        avg_ratio=avg_ratio,
                        prices=[i.price for i in current_stack]
                    ))
                current_stack = [imb]
                current_type = imb.imbalance_type
        
        if len(current_stack) >= min_levels and current_type is not None:
            total_delta = sum(i.delta for i in current_stack)
            avg_ratio = sum(i.ratio for i in current_stack) / len(current_stack)
            stacked.append(StackedImbalance(
                start_price=current_stack[0].price,
                end_price=current_stack[-1].price,
                imbalance_type=current_type,
                level_count=len(current_stack),
                total_delta=total_delta,
                avg_ratio=avg_ratio,
                prices=[i.price for i in current_stack]
            ))
        
        return stacked
    
    def get_auction_state(self, threshold_percent: float = 0.1) -> Tuple[AuctionState, AuctionState]:
        """
        Detect finished/unfinished auction at high and low
        
        Finished auction: Price rejected, unlikely to return
        Unfinished auction: Price accepted, likely to be revisited
        
        Args:
            threshold_percent: Volume threshold as percent of max level volume
            
        Returns:
            Tuple of (high_state, low_state)
        """
        with self._lock:
            if not self._levels:
                return (AuctionState.UNKNOWN, AuctionState.UNKNOWN)
            
            sorted_levels = self.sorted_levels
            if len(sorted_levels) < 3:
                return (AuctionState.UNKNOWN, AuctionState.UNKNOWN)
            
            max_volume = max(l.total_volume for l in sorted_levels)
            threshold = max_volume * threshold_percent
            
            high_level = sorted_levels[-1]
            high_state = AuctionState.FINISHED if high_level.total_volume < threshold else AuctionState.UNFINISHED
            
            low_level = sorted_levels[0]
            low_state = AuctionState.FINISHED if low_level.total_volume < threshold else AuctionState.UNFINISHED
            
            return (high_state, low_state)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert footprint bar to dictionary"""
        val, vah = self.get_value_area()
        high_auction, low_auction = self.get_auction_state()
        
        return {
            'open_time': datetime.fromtimestamp(self.open_time).isoformat(),
            'close_time': datetime.fromtimestamp(self.close_time).isoformat(),
            'ohlc': {
                'open': self.open_price,
                'high': self.high_price,
                'low': self.low_price if self.low_price != float('inf') else 0,
                'close': self.close_price
            },
            'volume': {
                'total': round(self.total_volume, 4),
                'bid': round(self.total_bid_volume, 4),
                'ask': round(self.total_ask_volume, 4),
                'delta': round(self.total_delta, 4),
                'delta_percent': round(self.delta_percent, 2)
            },
            'poc': self.get_poc(),
            'value_area': {'val': val, 'vah': vah},
            'hvn_levels': self.get_hvn_levels(),
            'lvn_levels': self.get_lvn_levels(),
            'auction_state': {
                'high': high_auction.value,
                'low': low_auction.value
            },
            'total_trades': self.total_trades,
            'levels': [level.to_dict() for level in self.sorted_levels]
        }


class TapeAnalyzer:
    """
    Tape Analyzer - Analyzes Time & Sales data for trading signals
    
    Features:
    - Tape speed (trades/second)
    - Large prints detection (>$50k for crypto)
    - Delta spikes (sudden volume imbalance)
    - Sweeps detection (aggressive market orders clearing levels)
    - Stacked imbalances analysis
    - Finished/unfinished auction detection
    - Absorption zone detection
    
    Uses OrderFlowStream metrics as input data source
    """
    
    LARGE_PRINT_THRESHOLD = 50000.0
    SWEEP_MIN_LEVELS = 3
    SWEEP_TIME_WINDOW = 1.0
    DELTA_SPIKE_PERCENTILE = 95.0
    ABSORPTION_STABILITY_THRESHOLD = 0.001
    
    def __init__(
        self,
        order_flow_stream: Optional[Any] = None,
        large_print_threshold: float = 50000.0,
        tick_size: float = 0.01,
        rolling_window_seconds: int = 300,
        imbalance_threshold: float = 2.0
    ):
        """
        Initialize TapeAnalyzer
        
        Args:
            order_flow_stream: OrderFlowStream instance for data
            large_print_threshold: Notional value for large print detection (default $50k)
            tick_size: Price tick size for level aggregation
            rolling_window_seconds: Rolling window for analysis (default 5 min)
            imbalance_threshold: Ratio for imbalance detection (default 2:1)
        """
        self.order_flow_stream = order_flow_stream
        self.large_print_threshold = large_print_threshold
        self.tick_size = tick_size
        self.rolling_window_seconds = rolling_window_seconds
        self.imbalance_threshold = imbalance_threshold
        
        self._trades: Deque[Any] = deque(maxlen=100000)
        self._large_prints: Deque[LargePrint] = deque(maxlen=1000)
        self._sweeps: Deque[SweepEvent] = deque(maxlen=500)
        self._delta_history: Deque[Tuple[float, float]] = deque(maxlen=10000)
        self._delta_spikes: Deque[DeltaSpike] = deque(maxlen=500)
        self._absorption_zones: Deque[AbsorptionZone] = deque(maxlen=200)
        
        self._current_footprint: Optional[FootprintBar] = None
        self._footprint_history: Deque[FootprintBar] = deque(maxlen=1000)
        
        self._lock = threading.RLock()
        
        self._running_delta: float = 0.0
        self._tape_speed: float = 0.0
        self._last_speed_calc: float = time.time()
        self._speed_trade_count: int = 0
        
        logger.info(f"TapeAnalyzer initialized with large_print_threshold=${large_print_threshold}")
    
    def process_trade(self, trade: Any) -> None:
        """
        Process a single trade from the stream
        
        Args:
            trade: TradeData object from OrderFlowStream
        """
        with self._lock:
            self._trades.append(trade)
            
            if trade.is_buyer_maker:
                self._running_delta -= trade.quantity
            else:
                self._running_delta += trade.quantity
            
            self._delta_history.append((trade.timestamp, self._running_delta))
            
            self._speed_trade_count += 1
            now = time.time()
            if now - self._last_speed_calc >= 1.0:
                self._tape_speed = self._speed_trade_count / (now - self._last_speed_calc)
                self._speed_trade_count = 0
                self._last_speed_calc = now
            
            if trade.notional_value >= self.large_print_threshold:
                self._detect_large_print(trade)
            
            self._detect_sweep(trade)
            self._detect_delta_spike()
            self._detect_absorption(trade)
            
            if self._current_footprint:
                self._current_footprint.add_trade(trade)
    
    def process_trades(self, trades: List[Any]) -> None:
        """Process multiple trades"""
        for trade in trades:
            self.process_trade(trade)
    
    def _detect_large_print(self, trade: Any) -> None:
        """Detect and record large print"""
        direction = "SELL" if trade.is_buyer_maker else "BUY"
        large_print = LargePrint(
            timestamp=trade.timestamp,
            price=trade.price,
            quantity=trade.quantity,
            notional_value=trade.notional_value,
            direction=direction,
            trade_id=trade.trade_id
        )
        self._large_prints.append(large_print)
        logger.debug(f"Large print detected: ${trade.notional_value:.2f} {direction} @ {trade.price}")
    
    def _detect_sweep(self, trade: Any) -> None:
        """Detect sweep events - aggressive orders clearing multiple levels"""
        now = trade.timestamp
        recent_trades = [
            t for t in self._trades 
            if now - t.timestamp <= self.SWEEP_TIME_WINDOW
        ]
        
        if len(recent_trades) < self.SWEEP_MIN_LEVELS:
            return
        
        same_direction = [
            t for t in recent_trades
            if t.is_buyer_maker == trade.is_buyer_maker
        ]
        
        if len(same_direction) < self.SWEEP_MIN_LEVELS:
            return
        
        prices = set(self._round_price(t.price) for t in same_direction)
        if len(prices) < self.SWEEP_MIN_LEVELS:
            return
        
        sorted_prices = sorted(prices)
        consecutive_count = 1
        max_consecutive = 1
        
        for i in range(1, len(sorted_prices)):
            if sorted_prices[i] - sorted_prices[i-1] <= self.tick_size * 2:
                consecutive_count += 1
                max_consecutive = max(max_consecutive, consecutive_count)
            else:
                consecutive_count = 1
        
        if max_consecutive >= self.SWEEP_MIN_LEVELS:
            if self._sweeps and now - self._sweeps[-1].timestamp < 0.5:
                return
            
            direction = "DOWN" if trade.is_buyer_maker else "UP"
            total_volume = sum(t.quantity for t in same_direction)
            total_notional = sum(t.notional_value for t in same_direction)
            
            sweep = SweepEvent(
                timestamp=now,
                direction=direction,
                start_price=min(t.price for t in same_direction),
                end_price=max(t.price for t in same_direction),
                levels_cleared=max_consecutive,
                total_volume=total_volume,
                notional_value=total_notional,
                trade_ids=[t.trade_id for t in same_direction]
            )
            self._sweeps.append(sweep)
            logger.debug(f"Sweep detected: {direction} clearing {max_consecutive} levels")
    
    def _detect_delta_spike(self) -> None:
        """Detect sudden delta spikes"""
        if len(self._delta_history) < 100:
            return
        
        deltas = [d[1] for d in list(self._delta_history)[-100:]]
        delta_changes = [deltas[i] - deltas[i-1] for i in range(1, len(deltas))]
        
        if not delta_changes:
            return
        
        abs_changes = [abs(c) for c in delta_changes]
        threshold = np.percentile(abs_changes, self.DELTA_SPIKE_PERCENTILE)
        
        current_change = delta_changes[-1] if delta_changes else 0
        
        if abs(current_change) >= threshold:
            if self._delta_spikes and time.time() - self._delta_spikes[-1].timestamp < 2.0:
                return
            
            direction = "BUY" if current_change > 0 else "SELL"
            percentile = (sum(1 for c in abs_changes if abs(current_change) > c) / len(abs_changes)) * 100
            
            spike = DeltaSpike(
                timestamp=time.time(),
                delta_value=current_change,
                delta_percentile=percentile,
                direction=direction,
                trade_count=len(list(self._delta_history)[-10:]),
                duration_seconds=1.0
            )
            self._delta_spikes.append(spike)
            logger.debug(f"Delta spike detected: {current_change:.4f} ({direction})")
    
    def _detect_absorption(self, trade: Any) -> None:
        """Detect absorption zones where volume is absorbed without price movement"""
        now = trade.timestamp
        window_trades = [
            t for t in self._trades
            if now - t.timestamp <= 5.0
        ]
        
        if len(window_trades) < 20:
            return
        
        prices = [t.price for t in window_trades]
        price_range = max(prices) - min(prices)
        avg_price = sum(prices) / len(prices)
        
        if avg_price == 0:
            return
        
        stability = 1.0 - (price_range / avg_price) if avg_price > 0 else 0
        
        if stability >= (1.0 - self.ABSORPTION_STABILITY_THRESHOLD):
            total_volume = sum(t.quantity for t in window_trades)
            buy_vol = sum(t.quantity for t in window_trades if not t.is_buyer_maker)
            sell_vol = sum(t.quantity for t in window_trades if t.is_buyer_maker)
            
            direction = "BUY_ABSORPTION" if sell_vol > buy_vol * 1.5 else (
                "SELL_ABSORPTION" if buy_vol > sell_vol * 1.5 else "NEUTRAL"
            )
            
            if direction != "NEUTRAL" and total_volume > 0:
                if self._absorption_zones and now - self._absorption_zones[-1].time_duration < 3.0:
                    return
                
                zone = AbsorptionZone(
                    price=avg_price,
                    absorbed_volume=total_volume,
                    direction=direction,
                    price_stability=stability,
                    trade_count=len(window_trades),
                    time_duration=5.0
                )
                self._absorption_zones.append(zone)
                logger.debug(f"Absorption zone detected: {direction} @ {avg_price:.2f}")
    
    def _round_price(self, price: float) -> float:
        """Round price to tick size"""
        return round(price / self.tick_size) * self.tick_size
    
    def start_footprint_bar(self, open_time: float, close_time: float) -> FootprintBar:
        """
        Start a new footprint bar for the given time period
        
        Args:
            open_time: Bar open timestamp
            close_time: Bar close timestamp
            
        Returns:
            New FootprintBar instance
        """
        with self._lock:
            if self._current_footprint:
                self._footprint_history.append(self._current_footprint)
            
            self._current_footprint = FootprintBar(
                open_time=open_time,
                close_time=close_time,
                tick_size=self.tick_size,
                imbalance_threshold=self.imbalance_threshold
            )
            return self._current_footprint
    
    def get_current_footprint(self) -> Optional[FootprintBar]:
        """Get current footprint bar"""
        with self._lock:
            return self._current_footprint
    
    def get_footprint_analysis(
        self,
        trades: Optional[List[Any]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get complete footprint analysis for a period
        
        Args:
            trades: List of trades to analyze (uses stream data if None)
            start_time: Start timestamp (uses rolling window if None)
            end_time: End timestamp (uses current time if None)
            
        Returns:
            Dictionary with complete footprint analysis
        """
        with self._lock:
            now = time.time()
            
            if trades is None:
                start_time = start_time or (now - self.rolling_window_seconds)
                end_time = end_time or now
                trades = [
                    t for t in self._trades
                    if start_time <= t.timestamp <= end_time
                ]
            else:
                if trades:
                    start_time = start_time or trades[0].timestamp
                    end_time = end_time or trades[-1].timestamp
                else:
                    start_time = now - self.rolling_window_seconds
                    end_time = now
            
            footprint = FootprintBar(
                open_time=start_time,
                close_time=end_time,
                tick_size=self.tick_size,
                imbalance_threshold=self.imbalance_threshold
            )
            footprint.add_trades(trades)
            
            imbalances = footprint.detect_imbalances()
            stacked = footprint.detect_stacked_imbalances()
            
            return {
                'footprint': footprint.to_dict(),
                'imbalances': [i.to_dict() for i in imbalances],
                'stacked_imbalances': [s.to_dict() for s in stacked],
                'summary': {
                    'total_trades': len(trades),
                    'imbalance_count': len(imbalances),
                    'stacked_count': len(stacked),
                    'poc': footprint.get_poc(),
                    'delta': footprint.total_delta,
                    'delta_percent': footprint.delta_percent
                }
            }
    
    def detect_imbalances(self, threshold: float = 2.0) -> List[Dict[str, Any]]:
        """
        Detect current imbalances from recent trading activity
        
        Args:
            threshold: Bid/Ask ratio threshold (default 2:1)
            
        Returns:
            List of imbalance dictionaries
        """
        with self._lock:
            if self._current_footprint:
                imbalances = self._current_footprint.detect_imbalances(threshold)
                return [i.to_dict() for i in imbalances]
            
            now = time.time()
            recent_trades = [
                t for t in self._trades
                if now - t.timestamp <= 60
            ]
            
            if not recent_trades:
                return []
            
            footprint = FootprintBar(
                open_time=now - 60,
                close_time=now,
                tick_size=self.tick_size,
                imbalance_threshold=threshold
            )
            footprint.add_trades(recent_trades)
            
            return [i.to_dict() for i in footprint.detect_imbalances(threshold)]
    
    def detect_stacked_imbalances(self, min_levels: int = 3) -> List[Dict[str, Any]]:
        """
        Detect stacked imbalances (consecutive imbalances at adjacent prices)
        Strong signal for institutional activity
        
        Args:
            min_levels: Minimum consecutive levels required
            
        Returns:
            List of stacked imbalance dictionaries
        """
        with self._lock:
            if self._current_footprint:
                stacked = self._current_footprint.detect_stacked_imbalances(min_levels)
                return [s.to_dict() for s in stacked]
            
            now = time.time()
            recent_trades = [
                t for t in self._trades
                if now - t.timestamp <= 60
            ]
            
            if not recent_trades:
                return []
            
            footprint = FootprintBar(
                open_time=now - 60,
                close_time=now,
                tick_size=self.tick_size,
                imbalance_threshold=self.imbalance_threshold
            )
            footprint.add_trades(recent_trades)
            
            return [s.to_dict() for s in footprint.detect_stacked_imbalances(min_levels)]
    
    def get_absorption_zones(self, lookback_seconds: int = 300) -> List[Dict[str, Any]]:
        """
        Get detected absorption zones
        
        Args:
            lookback_seconds: How far back to look
            
        Returns:
            List of absorption zone dictionaries
        """
        with self._lock:
            cutoff = time.time() - lookback_seconds
            recent = [z for z in self._absorption_zones if z.time_duration >= cutoff - lookback_seconds]
            return [z.to_dict() for z in recent]
    
    def get_large_prints(self, lookback_seconds: int = 300) -> List[Dict[str, Any]]:
        """
        Get large prints from recent activity
        
        Args:
            lookback_seconds: How far back to look
            
        Returns:
            List of large print dictionaries
        """
        with self._lock:
            cutoff = time.time() - lookback_seconds
            recent = [lp for lp in self._large_prints if lp.timestamp >= cutoff]
            return [lp.to_dict() for lp in recent]
    
    def get_sweeps(self, lookback_seconds: int = 300) -> List[Dict[str, Any]]:
        """
        Get detected sweep events
        
        Args:
            lookback_seconds: How far back to look
            
        Returns:
            List of sweep dictionaries
        """
        with self._lock:
            cutoff = time.time() - lookback_seconds
            recent = [s for s in self._sweeps if s.timestamp >= cutoff]
            return [s.to_dict() for s in recent]
    
    def get_delta_spikes(self, lookback_seconds: int = 300) -> List[Dict[str, Any]]:
        """
        Get detected delta spikes
        
        Args:
            lookback_seconds: How far back to look
            
        Returns:
            List of delta spike dictionaries
        """
        with self._lock:
            cutoff = time.time() - lookback_seconds
            recent = [d for d in self._delta_spikes if d.timestamp >= cutoff]
            return [d.to_dict() for d in recent]
    
    def get_tape_speed(self) -> float:
        """
        Get current tape speed (trades per second)
        
        Returns:
            Trades per second
        """
        with self._lock:
            return self._tape_speed
    
    def get_tape_signal(self) -> float:
        """
        Get overall tape bias signal
        
        Returns:
            Float from -1 (bearish) to +1 (bullish)
        """
        with self._lock:
            signal = 0.0
            weights_sum = 0.0
            
            now = time.time()
            recent_trades = [t for t in self._trades if now - t.timestamp <= 60]
            
            if recent_trades:
                buy_vol = sum(t.quantity for t in recent_trades if not t.is_buyer_maker)
                sell_vol = sum(t.quantity for t in recent_trades if t.is_buyer_maker)
                total_vol = buy_vol + sell_vol
                
                if total_vol > 0:
                    delta_signal = (buy_vol - sell_vol) / total_vol
                    signal += delta_signal * 0.3
                    weights_sum += 0.3
            
            recent_large = [lp for lp in self._large_prints if now - lp.timestamp <= 300]
            if recent_large:
                buy_large = sum(1 for lp in recent_large if lp.direction == "BUY")
                sell_large = sum(1 for lp in recent_large if lp.direction == "SELL")
                total_large = buy_large + sell_large
                
                if total_large > 0:
                    large_signal = (buy_large - sell_large) / total_large
                    signal += large_signal * 0.25
                    weights_sum += 0.25
            
            recent_sweeps = [s for s in self._sweeps if now - s.timestamp <= 300]
            if recent_sweeps:
                up_sweeps = sum(1 for s in recent_sweeps if s.direction == "UP")
                down_sweeps = sum(1 for s in recent_sweeps if s.direction == "DOWN")
                total_sweeps = up_sweeps + down_sweeps
                
                if total_sweeps > 0:
                    sweep_signal = (up_sweeps - down_sweeps) / total_sweeps
                    signal += sweep_signal * 0.25
                    weights_sum += 0.25
            
            stacked = self.detect_stacked_imbalances()
            if stacked:
                bid_stacked = sum(1 for s in stacked if s.get('type') == 'bid_imbalance')
                ask_stacked = sum(1 for s in stacked if s.get('type') == 'ask_imbalance')
                total_stacked = bid_stacked + ask_stacked
                
                if total_stacked > 0:
                    stacked_signal = (ask_stacked - bid_stacked) / total_stacked
                    signal += stacked_signal * 0.2
                    weights_sum += 0.2
            
            if weights_sum > 0:
                signal = signal / weights_sum
            
            return max(-1.0, min(1.0, signal))
    
    def get_market_microstructure_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive market microstructure summary
        
        Returns:
            Dictionary with all tape analysis metrics
        """
        with self._lock:
            now = time.time()
            
            recent_trades = [t for t in self._trades if now - t.timestamp <= 60]
            buy_vol = sum(t.quantity for t in recent_trades if not t.is_buyer_maker)
            sell_vol = sum(t.quantity for t in recent_trades if t.is_buyer_maker)
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'tape_speed': round(self._tape_speed, 2),
                'tape_signal': round(self.get_tape_signal(), 4),
                'volume': {
                    'buy_volume_1m': round(buy_vol, 4),
                    'sell_volume_1m': round(sell_vol, 4),
                    'delta_1m': round(buy_vol - sell_vol, 4),
                    'total_1m': round(buy_vol + sell_vol, 4)
                },
                'running_delta': round(self._running_delta, 4),
                'large_prints_5m': len(self.get_large_prints(300)),
                'sweeps_5m': len(self.get_sweeps(300)),
                'delta_spikes_5m': len(self.get_delta_spikes(300)),
                'absorption_zones_5m': len(self.get_absorption_zones(300)),
                'imbalances': self.detect_imbalances(),
                'stacked_imbalances': self.detect_stacked_imbalances(),
                'current_footprint': self._current_footprint.to_dict() if self._current_footprint else None
            }
    
    def connect_to_stream(self, order_flow_stream: Any) -> None:
        """
        Connect to an OrderFlowStream for real-time data
        
        Args:
            order_flow_stream: OrderFlowStream instance
        """
        self.order_flow_stream = order_flow_stream
        
        original_callback = order_flow_stream.on_trade_callback
        
        def combined_callback(trade):
            self.process_trade(trade)
            if original_callback:
                original_callback(trade)
        
        order_flow_stream.on_trade_callback = combined_callback
        logger.info("TapeAnalyzer connected to OrderFlowStream")


__all__ = [
    'FootprintBar',
    'TapeAnalyzer',
    'PriceLevel',
    'ImbalanceLevel',
    'StackedImbalance',
    'AbsorptionZone',
    'SweepEvent',
    'LargePrint',
    'DeltaSpike',
    'ImbalanceType',
    'AuctionState'
]
