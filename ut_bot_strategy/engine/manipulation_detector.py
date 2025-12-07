"""
Manipulation Detector Module for Market Manipulation Pattern Detection

Identifies various manipulation patterns including:
- Stop Hunts: Price spikes through support/resistance with quick reversals
- Spoofing: Large orders appearing/disappearing in order book
- Liquidity Sweeps: Aggressive orders clearing multiple price levels
- Fake Breakouts: Low volume breakouts with immediate reversals
- Absorption: Large limit orders absorbing market orders

Integration with OrderFlowStream and SignalEngine for real-time detection.
"""

import logging
import time
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Any, Deque
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

try:
    from ..data.order_flow_stream import TradeData, DepthData, OrderFlowMetrics
except ImportError:
    TradeData = None
    DepthData = None
    OrderFlowMetrics = None

logger = logging.getLogger(__name__)


class ManipulationType(Enum):
    """Types of market manipulation"""
    STOP_HUNT = "stop_hunt"
    SPOOFING = "spoofing"
    LIQUIDITY_SWEEP = "liquidity_sweep"
    FAKE_BREAKOUT = "fake_breakout"
    ABSORPTION = "absorption"
    WASH_TRADING = "wash_trading"
    LAYERING = "layering"


class ManipulationSeverity(Enum):
    """Severity levels for manipulation events"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TradingRecommendation(Enum):
    """Trading recommendations based on manipulation analysis"""
    TRADE = "trade"
    WAIT = "wait"
    AVOID = "avoid"
    FADE = "fade"


@dataclass
class StopHuntEvent:
    """Detected stop hunt event"""
    timestamp: float
    direction: str
    trigger_price: float
    reversal_price: float
    swing_level: float
    penetration_depth: float
    reversal_speed: float
    volume_at_extreme: float
    absorption_after: float
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': datetime.fromtimestamp(self.timestamp).isoformat(),
            'direction': self.direction,
            'trigger_price': round(self.trigger_price, 4),
            'reversal_price': round(self.reversal_price, 4),
            'swing_level': round(self.swing_level, 4),
            'penetration_depth': round(self.penetration_depth, 6),
            'reversal_speed': round(self.reversal_speed, 4),
            'volume_at_extreme': round(self.volume_at_extreme, 4),
            'absorption_after': round(self.absorption_after, 4),
            'confidence': round(self.confidence, 4),
            'type': ManipulationType.STOP_HUNT.value
        }


@dataclass
class SpoofingScore:
    """Spoofing detection score and evidence"""
    timestamp: float
    score: float
    bid_churn_rate: float
    ask_churn_rate: float
    large_order_appearances: int
    large_order_cancellations: int
    phantom_liquidity_ratio: float
    far_orders_pulled: int
    evidence: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': datetime.fromtimestamp(self.timestamp).isoformat(),
            'score': round(self.score, 4),
            'bid_churn_rate': round(self.bid_churn_rate, 4),
            'ask_churn_rate': round(self.ask_churn_rate, 4),
            'large_order_appearances': self.large_order_appearances,
            'large_order_cancellations': self.large_order_cancellations,
            'phantom_liquidity_ratio': round(self.phantom_liquidity_ratio, 4),
            'far_orders_pulled': self.far_orders_pulled,
            'evidence': self.evidence,
            'type': ManipulationType.SPOOFING.value
        }


@dataclass
class SweepEvent:
    """Detected liquidity sweep event"""
    timestamp: float
    direction: str
    start_price: float
    end_price: float
    levels_cleared: int
    total_volume: float
    notional_value: float
    sweep_speed: float
    reversal_detected: bool
    reversal_price: Optional[float]
    confidence: float
    trade_ids: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': datetime.fromtimestamp(self.timestamp).isoformat(),
            'direction': self.direction,
            'start_price': round(self.start_price, 4),
            'end_price': round(self.end_price, 4),
            'levels_cleared': self.levels_cleared,
            'total_volume': round(self.total_volume, 4),
            'notional_value': round(self.notional_value, 2),
            'sweep_speed': round(self.sweep_speed, 4),
            'reversal_detected': self.reversal_detected,
            'reversal_price': round(self.reversal_price, 4) if self.reversal_price else None,
            'confidence': round(self.confidence, 4),
            'type': ManipulationType.LIQUIDITY_SWEEP.value
        }


@dataclass
class FakeBreakoutEvent:
    """Detected fake breakout event"""
    timestamp: float
    direction: str
    breakout_level: float
    breakout_price: float
    reversal_price: float
    breakout_volume: float
    reversal_volume: float
    volume_ratio: float
    delta_divergence: float
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': datetime.fromtimestamp(self.timestamp).isoformat(),
            'direction': self.direction,
            'breakout_level': round(self.breakout_level, 4),
            'breakout_price': round(self.breakout_price, 4),
            'reversal_price': round(self.reversal_price, 4),
            'breakout_volume': round(self.breakout_volume, 4),
            'reversal_volume': round(self.reversal_volume, 4),
            'volume_ratio': round(self.volume_ratio, 4),
            'delta_divergence': round(self.delta_divergence, 4),
            'confidence': round(self.confidence, 4),
            'type': ManipulationType.FAKE_BREAKOUT.value
        }


@dataclass
class AbsorptionEvent:
    """Detected absorption event"""
    timestamp: float
    price_level: float
    direction: str
    absorbed_volume: float
    total_trades: int
    price_stability: float
    wall_defended: bool
    wall_size: float
    defense_count: int
    duration_seconds: float
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': datetime.fromtimestamp(self.timestamp).isoformat(),
            'price_level': round(self.price_level, 4),
            'direction': self.direction,
            'absorbed_volume': round(self.absorbed_volume, 4),
            'total_trades': self.total_trades,
            'price_stability': round(self.price_stability, 4),
            'wall_defended': self.wall_defended,
            'wall_size': round(self.wall_size, 4),
            'defense_count': self.defense_count,
            'duration_seconds': round(self.duration_seconds, 2),
            'confidence': round(self.confidence, 4),
            'type': ManipulationType.ABSORPTION.value
        }


@dataclass
class ManipulationEvent:
    """Generic manipulation event wrapper"""
    timestamp: float
    event_type: ManipulationType
    severity: ManipulationSeverity
    confidence: float
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': datetime.fromtimestamp(self.timestamp).isoformat(),
            'event_type': self.event_type.value,
            'severity': self.severity.value,
            'confidence': round(self.confidence, 4),
            'description': self.description,
            'details': self.details
        }


@dataclass
class ManipulationAnalysis:
    """Complete manipulation analysis result"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    overall_score: float = 0.0
    stop_hunt_score: float = 0.0
    spoof_score: float = 0.0
    sweep_score: float = 0.0
    fake_breakout_score: float = 0.0
    absorption_score: float = 0.0
    events: List[ManipulationEvent] = field(default_factory=list)
    recommendation: TradingRecommendation = TradingRecommendation.TRADE
    risk_level: ManipulationSeverity = ManipulationSeverity.LOW
    analysis_notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'overall_score': round(self.overall_score, 4),
            'stop_hunt_score': round(self.stop_hunt_score, 4),
            'spoof_score': round(self.spoof_score, 4),
            'sweep_score': round(self.sweep_score, 4),
            'fake_breakout_score': round(self.fake_breakout_score, 4),
            'absorption_score': round(self.absorption_score, 4),
            'events': [e.to_dict() for e in self.events],
            'recommendation': self.recommendation.value,
            'risk_level': self.risk_level.value,
            'analysis_notes': self.analysis_notes,
            'event_count': len(self.events)
        }


class ManipulationDetector:
    """
    Market Manipulation Detection Engine
    
    Analyzes order flow and price data to detect manipulation patterns:
    - Stop hunts at key levels
    - Order book spoofing
    - Liquidity sweeps
    - Fake breakouts
    - Order absorption
    
    Thread-safe for integration with real-time data streams.
    """
    
    STOP_HUNT_REVERSAL_THRESHOLD = 0.003
    STOP_HUNT_VOLUME_MULTIPLIER = 2.0
    SPOOFING_ORDER_LIFETIME_THRESHOLD = 5.0
    SPOOFING_SIZE_THRESHOLD = 50000.0
    SWEEP_MIN_LEVELS = 3
    SWEEP_SPEED_THRESHOLD = 0.5
    BREAKOUT_VOLUME_RATIO_THRESHOLD = 0.5
    ABSORPTION_STABILITY_THRESHOLD = 0.001
    ABSORPTION_VOLUME_THRESHOLD = 100000.0
    
    MAX_EVENTS_HISTORY = 1000
    ANALYSIS_WINDOW_SECONDS = 300
    
    def __init__(
        self,
        stop_hunt_threshold: float = 0.003,
        spoofing_order_lifetime: float = 5.0,
        sweep_min_levels: int = 3,
        large_order_threshold: float = 50000.0,
        absorption_stability: float = 0.001,
        analysis_window: int = 300
    ):
        """
        Initialize Manipulation Detector
        
        Args:
            stop_hunt_threshold: Price reversal threshold for stop hunts (default 0.3%)
            spoofing_order_lifetime: Max order lifetime for spoof detection (seconds)
            sweep_min_levels: Minimum levels for sweep detection
            large_order_threshold: Threshold for large orders (notional value)
            absorption_stability: Price stability threshold for absorption
            analysis_window: Analysis window in seconds
        """
        self.stop_hunt_threshold = stop_hunt_threshold
        self.spoofing_order_lifetime = spoofing_order_lifetime
        self.sweep_min_levels = sweep_min_levels
        self.large_order_threshold = large_order_threshold
        self.absorption_stability = absorption_stability
        self.analysis_window = analysis_window
        
        self._lock = threading.RLock()
        
        self._stop_hunt_events: Deque[StopHuntEvent] = deque(maxlen=self.MAX_EVENTS_HISTORY)
        self._spoofing_scores: Deque[SpoofingScore] = deque(maxlen=self.MAX_EVENTS_HISTORY)
        self._sweep_events: Deque[SweepEvent] = deque(maxlen=self.MAX_EVENTS_HISTORY)
        self._fake_breakout_events: Deque[FakeBreakoutEvent] = deque(maxlen=self.MAX_EVENTS_HISTORY)
        self._absorption_events: Deque[AbsorptionEvent] = deque(maxlen=self.MAX_EVENTS_HISTORY)
        self._all_events: Deque[ManipulationEvent] = deque(maxlen=self.MAX_EVENTS_HISTORY)
        
        self._depth_history: Deque[DepthData] = deque(maxlen=500)
        self._order_book_states: Deque[Dict] = deque(maxlen=100)
        self._trade_history: Deque[TradeData] = deque(maxlen=10000)
        
        self._current_scores: Dict[str, float] = {
            'stop_hunt': 0.0,
            'spoofing': 0.0,
            'sweep': 0.0,
            'fake_breakout': 0.0,
            'absorption': 0.0,
            'overall': 0.0
        }
        
        self._swing_highs: List[float] = []
        self._swing_lows: List[float] = []
        self._key_levels: List[float] = []
        
        logger.info("ManipulationDetector initialized")
    
    def add_trade(self, trade: Any) -> None:
        """
        Add a trade to the history for analysis
        
        Args:
            trade: TradeData object from OrderFlowStream
        """
        with self._lock:
            self._trade_history.append(trade)
    
    def add_depth_snapshot(self, depth: Any) -> None:
        """
        Add order book depth snapshot for analysis
        
        Args:
            depth: DepthData object from OrderFlowStream
        """
        with self._lock:
            if self._depth_history:
                prev_depth = self._depth_history[-1]
                self._track_order_changes(prev_depth, depth)
            self._depth_history.append(depth)
    
    def update_swing_levels(self, swing_highs: List[float], swing_lows: List[float]) -> None:
        """
        Update swing high/low levels for stop hunt detection
        
        Args:
            swing_highs: List of swing high prices
            swing_lows: List of swing low prices
        """
        with self._lock:
            self._swing_highs = swing_highs.copy()
            self._swing_lows = swing_lows.copy()
            self._key_levels = sorted(set(swing_highs + swing_lows))
    
    def set_key_levels(self, levels: List[float]) -> None:
        """
        Set key support/resistance levels
        
        Args:
            levels: List of key price levels
        """
        with self._lock:
            self._key_levels = sorted(set(levels))
    
    def _track_order_changes(self, prev_depth: Any, curr_depth: Any) -> None:
        """Track order book changes between snapshots for spoofing detection"""
        if not prev_depth or not curr_depth:
            return
        
        prev_bids = {float(b[0]): float(b[1]) for b in prev_depth.bids}
        prev_asks = {float(a[0]): float(a[1]) for a in prev_depth.asks}
        curr_bids = {float(b[0]): float(b[1]) for b in curr_depth.bids}
        curr_asks = {float(a[0]): float(a[1]) for a in curr_depth.asks}
        
        bid_added = sum(1 for p in curr_bids if p not in prev_bids)
        bid_removed = sum(1 for p in prev_bids if p not in curr_bids)
        ask_added = sum(1 for p in curr_asks if p not in prev_asks)
        ask_removed = sum(1 for p in prev_asks if p not in curr_asks)
        
        large_bid_appeared = 0
        large_bid_cancelled = 0
        large_ask_appeared = 0
        large_ask_cancelled = 0
        
        for price, size in curr_bids.items():
            if price not in prev_bids and size * price > self.large_order_threshold:
                large_bid_appeared += 1
        
        for price, size in prev_bids.items():
            if price not in curr_bids and size * price > self.large_order_threshold:
                large_bid_cancelled += 1
        
        for price, size in curr_asks.items():
            if price not in prev_asks and size * price > self.large_order_threshold:
                large_ask_appeared += 1
        
        for price, size in prev_asks.items():
            if price not in curr_asks and size * price > self.large_order_threshold:
                large_ask_cancelled += 1
        
        state = {
            'timestamp': curr_depth.timestamp,
            'bid_added': bid_added,
            'bid_removed': bid_removed,
            'ask_added': ask_added,
            'ask_removed': ask_removed,
            'large_bid_appeared': large_bid_appeared,
            'large_bid_cancelled': large_bid_cancelled,
            'large_ask_appeared': large_ask_appeared,
            'large_ask_cancelled': large_ask_cancelled,
            'best_bid': curr_depth.bids[0][0] if curr_depth.bids else 0,
            'best_ask': curr_depth.asks[0][0] if curr_depth.asks else 0
        }
        
        self._order_book_states.append(state)
    
    def analyze_manipulation(
        self,
        candle_data: pd.DataFrame,
        order_flow_metrics: Optional[Any] = None
    ) -> ManipulationAnalysis:
        """
        Perform complete manipulation analysis
        
        Args:
            candle_data: DataFrame with OHLCV data
            order_flow_metrics: Optional OrderFlowMetrics object
            
        Returns:
            ManipulationAnalysis with scores and detected events
        """
        with self._lock:
            analysis = ManipulationAnalysis(timestamp=datetime.utcnow())
            
            try:
                swing_highs, swing_lows = self._calculate_swing_levels(candle_data)
                self._swing_highs = swing_highs
                self._swing_lows = swing_lows
                
                price_data = self._extract_price_data(candle_data)
                
                stop_hunt_result = self._detect_stop_hunts(price_data, swing_highs, swing_lows)
                analysis.stop_hunt_score = stop_hunt_result['score']
                for event in stop_hunt_result['events']:
                    self._add_manipulation_event(
                        event_type=ManipulationType.STOP_HUNT,
                        confidence=event.confidence,
                        description=f"Stop hunt detected: {event.direction} through {event.swing_level:.2f}",
                        details=event.to_dict()
                    )
                
                spoof_result = self._analyze_spoofing()
                analysis.spoof_score = spoof_result.score if spoof_result else 0.0
                if spoof_result and spoof_result.score > 0.3:
                    self._add_manipulation_event(
                        event_type=ManipulationType.SPOOFING,
                        confidence=spoof_result.score,
                        description=f"Spoofing detected: {len(spoof_result.evidence)} indicators",
                        details=spoof_result.to_dict()
                    )
                
                sweep_result = self._detect_sweeps(price_data)
                analysis.sweep_score = sweep_result['score']
                for event in sweep_result['events']:
                    self._add_manipulation_event(
                        event_type=ManipulationType.LIQUIDITY_SWEEP,
                        confidence=event.confidence,
                        description=f"Liquidity sweep: {event.levels_cleared} levels {event.direction}",
                        details=event.to_dict()
                    )
                
                fake_breakout_result = self._detect_fake_breakouts(
                    candle_data, swing_highs, swing_lows, order_flow_metrics
                )
                analysis.fake_breakout_score = fake_breakout_result['score']
                for event in fake_breakout_result['events']:
                    self._add_manipulation_event(
                        event_type=ManipulationType.FAKE_BREAKOUT,
                        confidence=event.confidence,
                        description=f"Fake breakout: {event.direction} at {event.breakout_level:.2f}",
                        details=event.to_dict()
                    )
                
                absorption_result = self._detect_absorption(order_flow_metrics)
                analysis.absorption_score = absorption_result['score']
                for event in absorption_result['events']:
                    self._add_manipulation_event(
                        event_type=ManipulationType.ABSORPTION,
                        confidence=event.confidence,
                        description=f"Absorption at {event.price_level:.2f}: {event.absorbed_volume:.0f} absorbed",
                        details=event.to_dict()
                    )
                
                analysis.overall_score = self._calculate_overall_score(analysis)
                
                cutoff = time.time() - 60
                recent_events = [e for e in self._all_events if e.timestamp >= cutoff]
                analysis.events = recent_events.copy()
                
                analysis.recommendation, analysis.risk_level = self._generate_recommendation(analysis)
                
                analysis.analysis_notes = self._generate_analysis_notes(analysis)
                
                self._update_current_scores(analysis)
                
            except Exception as e:
                logger.error(f"Error in manipulation analysis: {e}")
                analysis.analysis_notes.append(f"Analysis error: {str(e)}")
            
            return analysis
    
    def detect_stop_hunt(
        self,
        price_data: Dict[str, List[float]],
        swing_highs: List[float],
        swing_lows: List[float]
    ) -> Optional[StopHuntEvent]:
        """
        Detect stop hunt patterns
        
        Args:
            price_data: Dictionary with 'high', 'low', 'close', 'volume' lists
            swing_highs: List of swing high prices
            swing_lows: List of swing low prices
            
        Returns:
            StopHuntEvent if detected, None otherwise
        """
        with self._lock:
            result = self._detect_stop_hunts(price_data, swing_highs, swing_lows)
            if result['events']:
                return result['events'][-1]
            return None
    
    def _detect_stop_hunts(
        self,
        price_data: Dict[str, List[float]],
        swing_highs: List[float],
        swing_lows: List[float]
    ) -> Dict[str, Any]:
        """Internal stop hunt detection"""
        events = []
        score = 0.0
        
        if not price_data or len(price_data.get('close', [])) < 5:
            return {'score': 0.0, 'events': []}
        
        highs = price_data['high']
        lows = price_data['low']
        closes = price_data['close']
        volumes = price_data.get('volume', [0] * len(closes))
        
        avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        
        for i in range(-min(5, len(closes)), 0):
            current_high = highs[i]
            current_low = lows[i]
            current_close = closes[i]
            current_volume = volumes[i] if i < len(volumes) else 0
            
            for swing_high in swing_highs[-10:]:
                if current_high > swing_high and current_close < swing_high:
                    penetration = (current_high - swing_high) / swing_high
                    reversal = (current_high - current_close) / current_high
                    
                    if reversal > self.stop_hunt_threshold:
                        volume_spike = current_volume / avg_volume if avg_volume > 0 else 1
                        
                        confidence = min(1.0, (
                            0.3 * min(1.0, penetration / 0.005) +
                            0.3 * min(1.0, reversal / 0.01) +
                            0.2 * min(1.0, volume_spike / 3.0) +
                            0.2
                        ))
                        
                        event = StopHuntEvent(
                            timestamp=time.time(),
                            direction='bearish',
                            trigger_price=current_high,
                            reversal_price=current_close,
                            swing_level=swing_high,
                            penetration_depth=penetration,
                            reversal_speed=reversal,
                            volume_at_extreme=current_volume,
                            absorption_after=0.0,
                            confidence=confidence
                        )
                        events.append(event)
                        self._stop_hunt_events.append(event)
                        score = max(score, confidence)
            
            for swing_low in swing_lows[-10:]:
                if current_low < swing_low and current_close > swing_low:
                    penetration = (swing_low - current_low) / swing_low
                    reversal = (current_close - current_low) / current_close
                    
                    if reversal > self.stop_hunt_threshold:
                        volume_spike = current_volume / avg_volume if avg_volume > 0 else 1
                        
                        confidence = min(1.0, (
                            0.3 * min(1.0, penetration / 0.005) +
                            0.3 * min(1.0, reversal / 0.01) +
                            0.2 * min(1.0, volume_spike / 3.0) +
                            0.2
                        ))
                        
                        event = StopHuntEvent(
                            timestamp=time.time(),
                            direction='bullish',
                            trigger_price=current_low,
                            reversal_price=current_close,
                            swing_level=swing_low,
                            penetration_depth=penetration,
                            reversal_speed=reversal,
                            volume_at_extreme=current_volume,
                            absorption_after=0.0,
                            confidence=confidence
                        )
                        events.append(event)
                        self._stop_hunt_events.append(event)
                        score = max(score, confidence)
        
        return {'score': score, 'events': events}
    
    def detect_spoofing(self, depth_history: Optional[List] = None) -> SpoofingScore:
        """
        Detect spoofing patterns in order book
        
        Args:
            depth_history: Optional list of DepthData snapshots
            
        Returns:
            SpoofingScore with detection results
        """
        with self._lock:
            return self._analyze_spoofing(depth_history)
    
    def _analyze_spoofing(self, depth_history: Optional[List] = None) -> SpoofingScore:
        """Internal spoofing analysis"""
        now = time.time()
        evidence = []
        
        states = list(self._order_book_states)
        if not states:
            return SpoofingScore(
                timestamp=now,
                score=0.0,
                bid_churn_rate=0.0,
                ask_churn_rate=0.0,
                large_order_appearances=0,
                large_order_cancellations=0,
                phantom_liquidity_ratio=0.0,
                far_orders_pulled=0
            )
        
        recent_states = [s for s in states if s['timestamp'] >= now - 60]
        if len(recent_states) < 5:
            return SpoofingScore(
                timestamp=now,
                score=0.0,
                bid_churn_rate=0.0,
                ask_churn_rate=0.0,
                large_order_appearances=0,
                large_order_cancellations=0,
                phantom_liquidity_ratio=0.0,
                far_orders_pulled=0
            )
        
        total_bid_added = sum(s['bid_added'] for s in recent_states)
        total_bid_removed = sum(s['bid_removed'] for s in recent_states)
        total_ask_added = sum(s['ask_added'] for s in recent_states)
        total_ask_removed = sum(s['ask_removed'] for s in recent_states)
        
        time_span = recent_states[-1]['timestamp'] - recent_states[0]['timestamp']
        time_span = max(time_span, 1.0)
        
        bid_churn_rate = (total_bid_added + total_bid_removed) / time_span
        ask_churn_rate = (total_ask_added + total_ask_removed) / time_span
        
        large_appearances = sum(
            s['large_bid_appeared'] + s['large_ask_appeared']
            for s in recent_states
        )
        large_cancellations = sum(
            s['large_bid_cancelled'] + s['large_ask_cancelled']
            for s in recent_states
        )
        
        phantom_ratio = 0.0
        if large_appearances > 0:
            phantom_ratio = large_cancellations / large_appearances
            if phantom_ratio > 0.8:
                evidence.append(f"High phantom liquidity ratio: {phantom_ratio:.2%}")
        
        churn_threshold = 10.0
        if bid_churn_rate > churn_threshold:
            evidence.append(f"High bid churn rate: {bid_churn_rate:.1f}/s")
        if ask_churn_rate > churn_threshold:
            evidence.append(f"High ask churn rate: {ask_churn_rate:.1f}/s")
        
        if large_cancellations > 5:
            evidence.append(f"Large orders cancelled: {large_cancellations}")
        
        far_orders_pulled = 0
        if self._depth_history:
            far_orders_pulled = self._count_far_orders_pulled()
            if far_orders_pulled > 3:
                evidence.append(f"Far orders pulled: {far_orders_pulled}")
        
        score = 0.0
        score += min(0.25, (bid_churn_rate + ask_churn_rate) / (churn_threshold * 4))
        score += min(0.25, phantom_ratio * 0.25)
        score += min(0.25, large_cancellations / 20.0)
        score += min(0.25, far_orders_pulled / 10.0)
        
        spoof_score = SpoofingScore(
            timestamp=now,
            score=min(1.0, score),
            bid_churn_rate=bid_churn_rate,
            ask_churn_rate=ask_churn_rate,
            large_order_appearances=large_appearances,
            large_order_cancellations=large_cancellations,
            phantom_liquidity_ratio=phantom_ratio,
            far_orders_pulled=far_orders_pulled,
            evidence=evidence
        )
        
        self._spoofing_scores.append(spoof_score)
        return spoof_score
    
    def _count_far_orders_pulled(self) -> int:
        """Count orders placed far from market and pulled"""
        count = 0
        if len(self._depth_history) < 2:
            return 0
        
        for i in range(1, min(len(self._depth_history), 20)):
            prev = self._depth_history[-i-1]
            curr = self._depth_history[-i]
            
            if not prev.bids or not prev.asks or not curr.bids or not curr.asks:
                continue
            
            prev_mid = (prev.bids[0][0] + prev.asks[0][0]) / 2
            
            for bid in prev.bids[5:]:
                price = float(bid[0])
                size = float(bid[1])
                if size * price > self.large_order_threshold:
                    distance = (prev_mid - price) / prev_mid
                    if distance > 0.005:
                        if not any(float(b[0]) == price for b in curr.bids):
                            count += 1
            
            for ask in prev.asks[5:]:
                price = float(ask[0])
                size = float(ask[1])
                if size * price > self.large_order_threshold:
                    distance = (price - prev_mid) / prev_mid
                    if distance > 0.005:
                        if not any(float(a[0]) == price for a in curr.asks):
                            count += 1
        
        return count
    
    def detect_liquidity_sweep(
        self,
        trades: Optional[List] = None,
        price_levels: Optional[List[float]] = None
    ) -> Optional[SweepEvent]:
        """
        Detect liquidity sweep patterns
        
        Args:
            trades: Optional list of TradeData
            price_levels: Optional list of key price levels
            
        Returns:
            SweepEvent if detected, None otherwise
        """
        with self._lock:
            price_data = self._get_recent_price_data()
            result = self._detect_sweeps(price_data, trades)
            if result['events']:
                return result['events'][-1]
            return None
    
    def _detect_sweeps(
        self,
        price_data: Dict[str, List[float]],
        trades: Optional[List] = None
    ) -> Dict[str, Any]:
        """Internal sweep detection"""
        events = []
        score = 0.0
        
        trades = trades or list(self._trade_history)
        if len(trades) < 10:
            return {'score': 0.0, 'events': []}
        
        now = time.time()
        recent_trades = [t for t in trades if t.timestamp >= now - 30]
        
        if len(recent_trades) < 5:
            return {'score': 0.0, 'events': []}
        
        buy_trades = [t for t in recent_trades if not t.is_buyer_maker]
        sell_trades = [t for t in recent_trades if t.is_buyer_maker]
        
        for direction, direction_trades in [('up', buy_trades), ('down', sell_trades)]:
            if len(direction_trades) < 3:
                continue
            
            sorted_trades = sorted(direction_trades, key=lambda t: t.timestamp)
            
            sweep_start = None
            sweep_trades = []
            
            for i, trade in enumerate(sorted_trades):
                if not sweep_start:
                    sweep_start = trade
                    sweep_trades = [trade]
                    continue
                
                time_gap = trade.timestamp - sweep_trades[-1].timestamp
                
                if time_gap < 2.0:
                    sweep_trades.append(trade)
                else:
                    if len(sweep_trades) >= 3:
                        event = self._create_sweep_event(sweep_trades, direction)
                        if event and event.confidence > 0.3:
                            events.append(event)
                            self._sweep_events.append(event)
                            score = max(score, event.confidence)
                    
                    sweep_start = trade
                    sweep_trades = [trade]
            
            if len(sweep_trades) >= 3:
                event = self._create_sweep_event(sweep_trades, direction)
                if event and event.confidence > 0.3:
                    events.append(event)
                    self._sweep_events.append(event)
                    score = max(score, event.confidence)
        
        return {'score': score, 'events': events}
    
    def _create_sweep_event(self, trades: List, direction: str) -> Optional[SweepEvent]:
        """Create a sweep event from a group of trades"""
        if len(trades) < 3:
            return None
        
        prices = [t.price for t in trades]
        start_price = trades[0].price
        end_price = trades[-1].price
        
        if direction == 'up':
            levels_cleared = sum(1 for i in range(1, len(prices)) if prices[i] > prices[i-1])
        else:
            levels_cleared = sum(1 for i in range(1, len(prices)) if prices[i] < prices[i-1])
        
        if levels_cleared < self.sweep_min_levels:
            return None
        
        total_volume = sum(t.quantity for t in trades)
        notional_value = sum(t.notional_value for t in trades)
        
        time_span = trades[-1].timestamp - trades[0].timestamp
        sweep_speed = levels_cleared / time_span if time_span > 0 else 0
        
        reversal_detected = False
        reversal_price = None
        
        later_trades = [
            t for t in self._trade_history
            if t.timestamp > trades[-1].timestamp
            and t.timestamp < trades[-1].timestamp + 30
        ]
        
        if later_trades:
            if direction == 'up':
                min_after = min(t.price for t in later_trades)
                if min_after < end_price * 0.997:
                    reversal_detected = True
                    reversal_price = min_after
            else:
                max_after = max(t.price for t in later_trades)
                if max_after > end_price * 1.003:
                    reversal_detected = True
                    reversal_price = max_after
        
        confidence = min(1.0, (
            0.25 * min(1.0, levels_cleared / 5) +
            0.25 * min(1.0, notional_value / (self.large_order_threshold * 2)) +
            0.25 * min(1.0, sweep_speed / 2.0) +
            0.25 * (1.0 if reversal_detected else 0.0)
        ))
        
        return SweepEvent(
            timestamp=trades[0].timestamp,
            direction=direction,
            start_price=start_price,
            end_price=end_price,
            levels_cleared=levels_cleared,
            total_volume=total_volume,
            notional_value=notional_value,
            sweep_speed=sweep_speed,
            reversal_detected=reversal_detected,
            reversal_price=reversal_price,
            confidence=confidence,
            trade_ids=[t.trade_id for t in trades]
        )
    
    def _detect_fake_breakouts(
        self,
        candle_data: pd.DataFrame,
        swing_highs: List[float],
        swing_lows: List[float],
        order_flow_metrics: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Detect fake breakout patterns"""
        events = []
        score = 0.0
        
        if len(candle_data) < 10:
            return {'score': 0.0, 'events': []}
        
        recent = candle_data.tail(5)
        avg_volume = candle_data['volume'].tail(20).mean() if 'volume' in candle_data else 0
        
        for i in range(len(recent) - 1):
            row = recent.iloc[i]
            next_row = recent.iloc[i + 1]
            
            current_high = row['high']
            current_low = row['low']
            current_close = row['close']
            current_volume = row.get('volume', avg_volume)
            
            next_high = next_row['high']
            next_low = next_row['low']
            next_close = next_row['close']
            next_volume = next_row.get('volume', avg_volume)
            
            for swing_high in swing_highs[-10:]:
                if current_high > swing_high and current_close > swing_high:
                    if next_close < swing_high:
                        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                        reversal_volume_ratio = next_volume / avg_volume if avg_volume > 0 else 1
                        
                        if volume_ratio < 1.0 and reversal_volume_ratio > volume_ratio:
                            delta_divergence = 0.0
                            if order_flow_metrics:
                                delta_divergence = order_flow_metrics.delta_change
                            
                            confidence = min(1.0, (
                                0.3 * max(0, 1.0 - volume_ratio) +
                                0.3 * min(1.0, reversal_volume_ratio / 2.0) +
                                0.2 * (1.0 if delta_divergence < 0 else 0.5) +
                                0.2
                            ))
                            
                            event = FakeBreakoutEvent(
                                timestamp=time.time(),
                                direction='bearish',
                                breakout_level=swing_high,
                                breakout_price=current_high,
                                reversal_price=next_close,
                                breakout_volume=current_volume,
                                reversal_volume=next_volume,
                                volume_ratio=volume_ratio,
                                delta_divergence=delta_divergence,
                                confidence=confidence
                            )
                            events.append(event)
                            self._fake_breakout_events.append(event)
                            score = max(score, confidence)
            
            for swing_low in swing_lows[-10:]:
                if current_low < swing_low and current_close < swing_low:
                    if next_close > swing_low:
                        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                        reversal_volume_ratio = next_volume / avg_volume if avg_volume > 0 else 1
                        
                        if volume_ratio < 1.0 and reversal_volume_ratio > volume_ratio:
                            delta_divergence = 0.0
                            if order_flow_metrics:
                                delta_divergence = order_flow_metrics.delta_change
                            
                            confidence = min(1.0, (
                                0.3 * max(0, 1.0 - volume_ratio) +
                                0.3 * min(1.0, reversal_volume_ratio / 2.0) +
                                0.2 * (1.0 if delta_divergence > 0 else 0.5) +
                                0.2
                            ))
                            
                            event = FakeBreakoutEvent(
                                timestamp=time.time(),
                                direction='bullish',
                                breakout_level=swing_low,
                                breakout_price=current_low,
                                reversal_price=next_close,
                                breakout_volume=current_volume,
                                reversal_volume=next_volume,
                                volume_ratio=volume_ratio,
                                delta_divergence=delta_divergence,
                                confidence=confidence
                            )
                            events.append(event)
                            self._fake_breakout_events.append(event)
                            score = max(score, confidence)
        
        return {'score': score, 'events': events}
    
    def _detect_absorption(
        self,
        order_flow_metrics: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Detect absorption patterns"""
        events = []
        score = 0.0
        
        if len(self._trade_history) < 20:
            return {'score': 0.0, 'events': []}
        
        now = time.time()
        recent_trades = [t for t in self._trade_history if t.timestamp >= now - 60]
        
        if len(recent_trades) < 10:
            return {'score': 0.0, 'events': []}
        
        price_levels: Dict[float, List] = {}
        tick_size = 0.01
        
        for trade in recent_trades:
            rounded_price = round(trade.price / tick_size) * tick_size
            if rounded_price not in price_levels:
                price_levels[rounded_price] = []
            price_levels[rounded_price].append(trade)
        
        for price, trades in price_levels.items():
            if len(trades) < 5:
                continue
            
            total_volume = sum(t.quantity for t in trades)
            total_notional = sum(t.notional_value for t in trades)
            
            if total_notional < self.ABSORPTION_VOLUME_THRESHOLD * 0.5:
                continue
            
            prices = [t.price for t in trades]
            price_range = max(prices) - min(prices) if prices else 0
            price_stability = 1.0 - (price_range / price if price > 0 else 0)
            
            if price_stability < 0.99:
                continue
            
            buy_volume = sum(t.quantity for t in trades if not t.is_buyer_maker)
            sell_volume = sum(t.quantity for t in trades if t.is_buyer_maker)
            
            if buy_volume > sell_volume * 1.5:
                direction = 'buy_absorption'
            elif sell_volume > buy_volume * 1.5:
                direction = 'sell_absorption'
            else:
                direction = 'neutral_absorption'
            
            wall_defended = False
            wall_size = 0.0
            defense_count = 0
            
            if self._depth_history:
                latest_depth = self._depth_history[-1]
                for bid in latest_depth.bids[:5]:
                    if abs(float(bid[0]) - price) < tick_size * 2:
                        wall_size = float(bid[1]) * float(bid[0])
                        if wall_size > self.large_order_threshold:
                            wall_defended = True
                            defense_count = len(trades)
                            break
                
                if not wall_defended:
                    for ask in latest_depth.asks[:5]:
                        if abs(float(ask[0]) - price) < tick_size * 2:
                            wall_size = float(ask[1]) * float(ask[0])
                            if wall_size > self.large_order_threshold:
                                wall_defended = True
                                defense_count = len(trades)
                                break
            
            time_span = trades[-1].timestamp - trades[0].timestamp if len(trades) > 1 else 1.0
            
            confidence = min(1.0, (
                0.25 * min(1.0, total_notional / self.ABSORPTION_VOLUME_THRESHOLD) +
                0.25 * price_stability +
                0.25 * (1.0 if wall_defended else 0.0) +
                0.25 * min(1.0, len(trades) / 20)
            ))
            
            if confidence > 0.4:
                event = AbsorptionEvent(
                    timestamp=now,
                    price_level=price,
                    direction=direction,
                    absorbed_volume=total_notional,
                    total_trades=len(trades),
                    price_stability=price_stability,
                    wall_defended=wall_defended,
                    wall_size=wall_size,
                    defense_count=defense_count,
                    duration_seconds=time_span,
                    confidence=confidence
                )
                events.append(event)
                self._absorption_events.append(event)
                score = max(score, confidence)
        
        if order_flow_metrics and hasattr(order_flow_metrics, 'absorption_score'):
            score = max(score, order_flow_metrics.absorption_score)
        
        return {'score': score, 'events': events}
    
    def get_manipulation_score(self) -> float:
        """
        Get current overall manipulation score (0-1)
        
        Returns:
            Float between 0 and 1, higher = more manipulation detected
        """
        with self._lock:
            return self._current_scores.get('overall', 0.0)
    
    def _calculate_swing_levels(
        self,
        candle_data: pd.DataFrame,
        lookback: int = 5
    ) -> Tuple[List[float], List[float]]:
        """Calculate swing high and low levels from candle data"""
        swing_highs = []
        swing_lows = []
        
        if len(candle_data) < lookback * 2 + 1:
            return swing_highs, swing_lows
        
        highs = candle_data['high'].values
        lows = candle_data['low'].values
        
        for i in range(lookback, len(candle_data) - lookback):
            is_swing_high = True
            is_swing_low = True
            
            for j in range(1, lookback + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_swing_high = False
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_swing_low = False
            
            if is_swing_high:
                swing_highs.append(float(highs[i]))
            if is_swing_low:
                swing_lows.append(float(lows[i]))
        
        return swing_highs[-20:], swing_lows[-20:]
    
    def _extract_price_data(self, candle_data: pd.DataFrame) -> Dict[str, List[float]]:
        """Extract price data from DataFrame"""
        return {
            'open': candle_data['open'].tolist() if 'open' in candle_data else [],
            'high': candle_data['high'].tolist() if 'high' in candle_data else [],
            'low': candle_data['low'].tolist() if 'low' in candle_data else [],
            'close': candle_data['close'].tolist() if 'close' in candle_data else [],
            'volume': candle_data['volume'].tolist() if 'volume' in candle_data else []
        }
    
    def _get_recent_price_data(self) -> Dict[str, List[float]]:
        """Get recent price data from trade history"""
        trades = list(self._trade_history)
        if not trades:
            return {'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}
        
        prices = [t.price for t in trades[-100:]]
        volumes = [t.quantity for t in trades[-100:]]
        
        return {
            'open': [prices[0]] if prices else [],
            'high': [max(prices)] if prices else [],
            'low': [min(prices)] if prices else [],
            'close': [prices[-1]] if prices else [],
            'volume': [sum(volumes)] if volumes else []
        }
    
    def _add_manipulation_event(
        self,
        event_type: ManipulationType,
        confidence: float,
        description: str,
        details: Dict[str, Any]
    ) -> None:
        """Add a manipulation event to history"""
        severity = ManipulationSeverity.LOW
        if confidence >= 0.8:
            severity = ManipulationSeverity.CRITICAL
        elif confidence >= 0.6:
            severity = ManipulationSeverity.HIGH
        elif confidence >= 0.4:
            severity = ManipulationSeverity.MEDIUM
        
        event = ManipulationEvent(
            timestamp=time.time(),
            event_type=event_type,
            severity=severity,
            confidence=confidence,
            description=description,
            details=details
        )
        self._all_events.append(event)
    
    def _calculate_overall_score(self, analysis: ManipulationAnalysis) -> float:
        """Calculate weighted overall manipulation score"""
        weights = {
            'stop_hunt': 0.25,
            'spoofing': 0.20,
            'sweep': 0.20,
            'fake_breakout': 0.20,
            'absorption': 0.15
        }
        
        weighted_sum = (
            analysis.stop_hunt_score * weights['stop_hunt'] +
            analysis.spoof_score * weights['spoofing'] +
            analysis.sweep_score * weights['sweep'] +
            analysis.fake_breakout_score * weights['fake_breakout'] +
            analysis.absorption_score * weights['absorption']
        )
        
        max_score = max(
            analysis.stop_hunt_score,
            analysis.spoof_score,
            analysis.sweep_score,
            analysis.fake_breakout_score,
            analysis.absorption_score
        )
        
        overall = 0.7 * weighted_sum + 0.3 * max_score
        
        return min(1.0, overall)
    
    def _generate_recommendation(
        self,
        analysis: ManipulationAnalysis
    ) -> Tuple[TradingRecommendation, ManipulationSeverity]:
        """Generate trading recommendation based on analysis"""
        score = analysis.overall_score
        
        if score >= 0.7:
            return TradingRecommendation.AVOID, ManipulationSeverity.CRITICAL
        elif score >= 0.5:
            return TradingRecommendation.WAIT, ManipulationSeverity.HIGH
        elif score >= 0.3:
            if analysis.stop_hunt_score > 0.5 or analysis.fake_breakout_score > 0.5:
                return TradingRecommendation.FADE, ManipulationSeverity.MEDIUM
            return TradingRecommendation.WAIT, ManipulationSeverity.MEDIUM
        else:
            return TradingRecommendation.TRADE, ManipulationSeverity.LOW
    
    def _generate_analysis_notes(self, analysis: ManipulationAnalysis) -> List[str]:
        """Generate human-readable analysis notes"""
        notes = []
        
        if analysis.stop_hunt_score > 0.5:
            notes.append(f"Stop hunt activity detected (score: {analysis.stop_hunt_score:.2f})")
        
        if analysis.spoof_score > 0.5:
            notes.append(f"Possible spoofing in order book (score: {analysis.spoof_score:.2f})")
        
        if analysis.sweep_score > 0.5:
            notes.append(f"Liquidity sweep detected (score: {analysis.sweep_score:.2f})")
        
        if analysis.fake_breakout_score > 0.5:
            notes.append(f"Fake breakout pattern (score: {analysis.fake_breakout_score:.2f})")
        
        if analysis.absorption_score > 0.5:
            notes.append(f"Absorption detected at key levels (score: {analysis.absorption_score:.2f})")
        
        if not notes:
            notes.append("No significant manipulation patterns detected")
        
        notes.append(f"Overall risk: {analysis.risk_level.value.upper()}")
        notes.append(f"Recommendation: {analysis.recommendation.value.upper()}")
        
        return notes
    
    def _update_current_scores(self, analysis: ManipulationAnalysis) -> None:
        """Update stored current scores"""
        self._current_scores = {
            'stop_hunt': analysis.stop_hunt_score,
            'spoofing': analysis.spoof_score,
            'sweep': analysis.sweep_score,
            'fake_breakout': analysis.fake_breakout_score,
            'absorption': analysis.absorption_score,
            'overall': analysis.overall_score
        }
    
    def get_recent_events(self, seconds: int = 300) -> List[ManipulationEvent]:
        """
        Get manipulation events from the last N seconds
        
        Args:
            seconds: Number of seconds to look back
            
        Returns:
            List of ManipulationEvent objects
        """
        cutoff = time.time() - seconds
        with self._lock:
            return [e for e in self._all_events if e.timestamp >= cutoff]
    
    def get_stop_hunt_history(self, count: int = 10) -> List[StopHuntEvent]:
        """Get recent stop hunt events"""
        with self._lock:
            return list(self._stop_hunt_events)[-count:]
    
    def get_sweep_history(self, count: int = 10) -> List[SweepEvent]:
        """Get recent sweep events"""
        with self._lock:
            return list(self._sweep_events)[-count:]
    
    def get_spoofing_history(self, count: int = 10) -> List[SpoofingScore]:
        """Get recent spoofing scores"""
        with self._lock:
            return list(self._spoofing_scores)[-count:]
    
    def get_absorption_history(self, count: int = 10) -> List[AbsorptionEvent]:
        """Get recent absorption events"""
        with self._lock:
            return list(self._absorption_events)[-count:]
    
    def get_all_scores(self) -> Dict[str, float]:
        """Get all current manipulation scores"""
        with self._lock:
            return self._current_scores.copy()
    
    def clear_history(self) -> None:
        """Clear all event history"""
        with self._lock:
            self._stop_hunt_events.clear()
            self._spoofing_scores.clear()
            self._sweep_events.clear()
            self._fake_breakout_events.clear()
            self._absorption_events.clear()
            self._all_events.clear()
            self._trade_history.clear()
            self._depth_history.clear()
            self._order_book_states.clear()
            self._current_scores = {k: 0.0 for k in self._current_scores}
        
        logger.info("Manipulation detector history cleared")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get complete summary of manipulation detection state"""
        with self._lock:
            now = time.time()
            recent_cutoff = now - 300
            
            return {
                'current_scores': self._current_scores.copy(),
                'recent_stop_hunts': len([e for e in self._stop_hunt_events if e.timestamp >= recent_cutoff]),
                'recent_sweeps': len([e for e in self._sweep_events if e.timestamp >= recent_cutoff]),
                'recent_absorptions': len([e for e in self._absorption_events if e.timestamp >= recent_cutoff]),
                'total_events': len(self._all_events),
                'trades_analyzed': len(self._trade_history),
                'depth_snapshots': len(self._depth_history),
                'swing_highs': len(self._swing_highs),
                'swing_lows': len(self._swing_lows),
                'key_levels': len(self._key_levels)
            }
