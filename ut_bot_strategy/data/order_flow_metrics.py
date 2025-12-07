"""
Order Flow Metrics Service for Aggregating Order Flow Analysis Components

Provides a unified API for order flow metrics by aggregating:
- OrderFlowStream: Real-time WebSocket data (CVD, imbalance, large orders)
- TapeAnalyzer: Footprint analysis, tape reading, sweeps, absorption zones
- ManipulationDetector: Stop hunts, spoofing, fake breakouts

Advanced computed metrics for trading decisions:
- order_flow_bias: Combined score from CVD trend, imbalance, large orders
- manipulation_adjusted_confidence: Confidence adjusted for manipulation risk
- institutional_activity_score: Large orders, absorption, sweeps indicator
- smart_money_indicator: Delta divergence + absorption at key levels
"""

import logging
import threading
import time
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

try:
    from .order_flow_stream import OrderFlowStream, OrderFlowMetrics, TradeData, DepthData
except ImportError:
    OrderFlowStream = None
    OrderFlowMetrics = None
    TradeData = None
    DepthData = None

try:
    from ..engine.tape_analyzer import TapeAnalyzer
except ImportError:
    TapeAnalyzer = None

try:
    from ..engine.manipulation_detector import ManipulationDetector, TradingRecommendation
except ImportError:
    ManipulationDetector = None
    TradingRecommendation = None

logger = logging.getLogger(__name__)


class TradingBias(Enum):
    """Trading bias classification"""
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"


@dataclass
class CVDAnalysis:
    """CVD analysis with trend detection"""
    current_cvd: float = 0.0
    cvd_change: float = 0.0
    cvd_trend: str = "neutral"
    trend_strength: float = 0.0
    is_diverging: bool = False
    divergence_direction: str = "none"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'current_cvd': round(self.current_cvd, 4),
            'cvd_change': round(self.cvd_change, 4),
            'cvd_trend': self.cvd_trend,
            'trend_strength': round(self.trend_strength, 4),
            'is_diverging': self.is_diverging,
            'divergence_direction': self.divergence_direction
        }


@dataclass
class DeltaExtremes:
    """Delta extremes analysis (99th percentile spikes)"""
    has_extreme: bool = False
    extreme_value: float = 0.0
    extreme_percentile: float = 0.0
    direction: str = "none"
    recent_spikes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'has_extreme': self.has_extreme,
            'extreme_value': round(self.extreme_value, 4),
            'extreme_percentile': round(self.extreme_percentile, 2),
            'direction': self.direction,
            'recent_spikes': self.recent_spikes
        }


@dataclass
class ImbalanceAnalysis:
    """Bid/Ask imbalance analysis"""
    imbalance_ratio: float = 0.0
    is_significant: bool = False
    direction: str = "neutral"
    book_imbalance: float = 0.0
    volume_imbalance: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'imbalance_ratio': round(self.imbalance_ratio, 4),
            'is_significant': self.is_significant,
            'direction': self.direction,
            'book_imbalance': round(self.book_imbalance, 4),
            'volume_imbalance': round(self.volume_imbalance, 4)
        }


@dataclass
class AbsorptionAnalysis:
    """Absorption zones analysis"""
    active_zones: int = 0
    total_absorbed_volume: float = 0.0
    strongest_zone_price: float = 0.0
    strongest_zone_strength: float = 0.0
    zones: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'active_zones': self.active_zones,
            'total_absorbed_volume': round(self.total_absorbed_volume, 2),
            'strongest_zone_price': round(self.strongest_zone_price, 4),
            'strongest_zone_strength': round(self.strongest_zone_strength, 4),
            'zones': self.zones
        }


@dataclass
class ManipulationAnalysis:
    """Manipulation probability analysis"""
    overall_score: float = 0.0
    stop_hunt_score: float = 0.0
    spoofing_score: float = 0.0
    fake_breakout_score: float = 0.0
    sweep_score: float = 0.0
    is_high_risk: bool = False
    recommendation: str = "trade"
    notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'overall_score': round(self.overall_score, 4),
            'stop_hunt_score': round(self.stop_hunt_score, 4),
            'spoofing_score': round(self.spoofing_score, 4),
            'fake_breakout_score': round(self.fake_breakout_score, 4),
            'sweep_score': round(self.sweep_score, 4),
            'is_high_risk': self.is_high_risk,
            'recommendation': self.recommendation,
            'notes': self.notes
        }


@dataclass
class InstitutionalActivity:
    """Institutional activity indicators"""
    activity_score: float = 0.0
    large_order_count: int = 0
    large_buy_count: int = 0
    large_sell_count: int = 0
    total_large_notional: float = 0.0
    sweep_count: int = 0
    absorption_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'activity_score': round(self.activity_score, 4),
            'large_order_count': self.large_order_count,
            'large_buy_count': self.large_buy_count,
            'large_sell_count': self.large_sell_count,
            'total_large_notional': round(self.total_large_notional, 2),
            'sweep_count': self.sweep_count,
            'absorption_count': self.absorption_count
        }


@dataclass
class SmartMoneyIndicator:
    """Smart money indicator combining multiple signals"""
    score: float = 0.0
    delta_divergence: float = 0.0
    absorption_at_levels: float = 0.0
    large_order_direction: float = 0.0
    sweep_direction: float = 0.0
    direction: str = "neutral"
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'score': round(self.score, 4),
            'delta_divergence': round(self.delta_divergence, 4),
            'absorption_at_levels': round(self.absorption_at_levels, 4),
            'large_order_direction': round(self.large_order_direction, 4),
            'sweep_direction': round(self.sweep_direction, 4),
            'direction': self.direction,
            'confidence': round(self.confidence, 4)
        }


@dataclass
class OrderFlowMetricsConfig:
    """Configuration for Order Flow Metrics Service"""
    cvd_trend_threshold: float = 1000.0
    imbalance_threshold: float = 0.3
    manipulation_reject_threshold: float = 0.7
    large_order_threshold: float = 50000.0
    delta_percentile_threshold: float = 99.0
    lookback_seconds: int = 300


@dataclass
class CompleteOrderFlowMetrics:
    """Complete order flow metrics snapshot"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    cvd: CVDAnalysis = field(default_factory=CVDAnalysis)
    delta_extremes: DeltaExtremes = field(default_factory=DeltaExtremes)
    imbalance: ImbalanceAnalysis = field(default_factory=ImbalanceAnalysis)
    absorption: AbsorptionAnalysis = field(default_factory=AbsorptionAnalysis)
    manipulation: ManipulationAnalysis = field(default_factory=ManipulationAnalysis)
    tape_signal: float = 0.0
    order_flow_bias: float = 0.0
    institutional_activity: InstitutionalActivity = field(default_factory=InstitutionalActivity)
    smart_money: SmartMoneyIndicator = field(default_factory=SmartMoneyIndicator)
    microstructure_summary: Dict[str, Any] = field(default_factory=dict)
    raw_stream_metrics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'cvd': self.cvd.to_dict(),
            'delta_extremes': self.delta_extremes.to_dict(),
            'imbalance': self.imbalance.to_dict(),
            'absorption': self.absorption.to_dict(),
            'manipulation': self.manipulation.to_dict(),
            'tape_signal': round(self.tape_signal, 4),
            'order_flow_bias': round(self.order_flow_bias, 4),
            'institutional_activity': self.institutional_activity.to_dict(),
            'smart_money': self.smart_money.to_dict(),
            'microstructure_summary': self.microstructure_summary,
            'raw_stream_metrics': self.raw_stream_metrics
        }


class OrderFlowMetricsService:
    """
    Order Flow Metrics Service
    
    Aggregates all order flow analysis components into a unified API:
    - OrderFlowStream for real-time WebSocket data
    - TapeAnalyzer for footprint and tape reading
    - ManipulationDetector for manipulation pattern detection
    
    Provides advanced computed metrics for trading signal enhancement.
    """
    
    def __init__(self, config: Optional[OrderFlowMetricsConfig] = None):
        """
        Initialize Order Flow Metrics Service
        
        Args:
            config: Configuration for thresholds and parameters
        """
        self.config = config or OrderFlowMetricsConfig()
        
        self._stream: Optional[OrderFlowStream] = None
        self._tape_analyzer: Optional[TapeAnalyzer] = None
        self._manipulation_detector: Optional[ManipulationDetector] = None
        
        self._lock = threading.RLock()
        self._initialized = False
        
        self._cvd_history: List[Tuple[float, float]] = []
        self._delta_history: List[float] = []
        self._price_history: List[Tuple[float, float]] = []
        
        self._last_metrics: Optional[CompleteOrderFlowMetrics] = None
        self._last_update: float = 0.0
        
        logger.info("OrderFlowMetricsService initialized")
    
    def initialize(
        self,
        stream: Optional[Any] = None,
        tape_analyzer: Optional[Any] = None,
        manipulation_detector: Optional[Any] = None
    ) -> None:
        """
        Connect order flow analysis components
        
        Args:
            stream: OrderFlowStream instance
            tape_analyzer: TapeAnalyzer instance
            manipulation_detector: ManipulationDetector instance
        """
        with self._lock:
            if stream is not None:
                self._stream = stream
                if hasattr(stream, 'on_metrics_callback'):
                    original_callback = stream.on_metrics_callback
                    
                    def combined_callback(metrics):
                        self._on_stream_metrics_update(metrics)
                        if original_callback:
                            original_callback(metrics)
                    
                    stream.on_metrics_callback = combined_callback
                    
                logger.info("Connected to OrderFlowStream")
            
            if tape_analyzer is not None:
                self._tape_analyzer = tape_analyzer
                logger.info("Connected to TapeAnalyzer")
            
            if manipulation_detector is not None:
                self._manipulation_detector = manipulation_detector
                logger.info("Connected to ManipulationDetector")
            
            self._initialized = True
            logger.info("OrderFlowMetricsService fully initialized")
    
    def _on_stream_metrics_update(self, metrics: Any) -> None:
        """Handle real-time metrics updates from stream"""
        with self._lock:
            now = time.time()
            
            if hasattr(metrics, 'cumulative_delta'):
                self._cvd_history.append((now, metrics.cumulative_delta))
                if len(self._cvd_history) > 1000:
                    self._cvd_history = self._cvd_history[-500:]
            
            if hasattr(metrics, 'delta_change'):
                self._delta_history.append(metrics.delta_change)
                if len(self._delta_history) > 1000:
                    self._delta_history = self._delta_history[-500:]
            
            if hasattr(metrics, 'last_price') and metrics.last_price > 0:
                self._price_history.append((now, metrics.last_price))
                if len(self._price_history) > 1000:
                    self._price_history = self._price_history[-500:]
    
    def get_complete_metrics(self) -> CompleteOrderFlowMetrics:
        """
        Get complete order flow metrics snapshot
        
        Returns:
            CompleteOrderFlowMetrics with all aggregated analysis
        """
        with self._lock:
            metrics = CompleteOrderFlowMetrics(timestamp=datetime.utcnow())
            
            stream_metrics = None
            if self._stream:
                try:
                    stream_metrics = self._stream.get_current_metrics()
                    metrics.raw_stream_metrics = stream_metrics.to_dict() if stream_metrics else None
                except Exception as e:
                    logger.warning(f"Error getting stream metrics: {e}")
            
            metrics.cvd = self._analyze_cvd(stream_metrics)
            
            metrics.delta_extremes = self._analyze_delta_extremes()
            
            metrics.imbalance = self._analyze_imbalance(stream_metrics)
            
            metrics.absorption = self._analyze_absorption()
            
            metrics.manipulation = self._analyze_manipulation()
            
            if self._tape_analyzer:
                try:
                    metrics.tape_signal = self._tape_analyzer.get_tape_signal()
                except Exception as e:
                    logger.warning(f"Error getting tape signal: {e}")
            
            metrics.institutional_activity = self._calculate_institutional_activity(stream_metrics)
            
            metrics.smart_money = self._calculate_smart_money_indicator(
                metrics.cvd, metrics.absorption, stream_metrics
            )
            
            metrics.order_flow_bias = self._calculate_order_flow_bias(
                metrics.cvd, metrics.imbalance, stream_metrics, metrics.tape_signal
            )
            
            if self._tape_analyzer:
                try:
                    metrics.microstructure_summary = self._tape_analyzer.get_market_microstructure_summary()
                except Exception as e:
                    logger.warning(f"Error getting microstructure summary: {e}")
            
            self._last_metrics = metrics
            self._last_update = time.time()
            
            return metrics
    
    def _analyze_cvd(self, stream_metrics: Any) -> CVDAnalysis:
        """Analyze CVD with trend detection"""
        analysis = CVDAnalysis()
        
        if stream_metrics:
            analysis.current_cvd = getattr(stream_metrics, 'cumulative_delta', 0.0)
            analysis.cvd_change = getattr(stream_metrics, 'delta_change', 0.0)
        
        if len(self._cvd_history) >= 10:
            recent_cvd = [h[1] for h in self._cvd_history[-10:]]
            cvd_change = recent_cvd[-1] - recent_cvd[0]
            
            if abs(cvd_change) >= self.config.cvd_trend_threshold:
                analysis.cvd_trend = "bullish" if cvd_change > 0 else "bearish"
                analysis.trend_strength = min(1.0, abs(cvd_change) / (self.config.cvd_trend_threshold * 3))
            else:
                analysis.cvd_trend = "neutral"
                analysis.trend_strength = 0.0
            
            if len(self._price_history) >= 10:
                recent_prices = [h[1] for h in self._price_history[-10:]]
                price_change = recent_prices[-1] - recent_prices[0]
                
                if (cvd_change > 0 and price_change < 0) or (cvd_change < 0 and price_change > 0):
                    analysis.is_diverging = True
                    analysis.divergence_direction = "bullish" if cvd_change > 0 else "bearish"
        
        return analysis
    
    def _analyze_delta_extremes(self) -> DeltaExtremes:
        """Analyze delta for 99th percentile spikes"""
        analysis = DeltaExtremes()
        
        if len(self._delta_history) < 50:
            return analysis
        
        abs_deltas = [abs(d) for d in self._delta_history]
        threshold = np.percentile(abs_deltas, self.config.delta_percentile_threshold)
        
        if len(self._delta_history) > 0:
            current_delta = self._delta_history[-1]
            
            if abs(current_delta) >= threshold:
                analysis.has_extreme = True
                analysis.extreme_value = current_delta
                analysis.extreme_percentile = (
                    sum(1 for d in abs_deltas if abs(current_delta) > d) / len(abs_deltas) * 100
                )
                analysis.direction = "buy" if current_delta > 0 else "sell"
        
        recent_deltas = self._delta_history[-100:] if len(self._delta_history) >= 100 else self._delta_history
        analysis.recent_spikes = sum(1 for d in recent_deltas if abs(d) >= threshold)
        
        return analysis
    
    def _analyze_imbalance(self, stream_metrics: Any) -> ImbalanceAnalysis:
        """Analyze bid/ask imbalance"""
        analysis = ImbalanceAnalysis()
        
        if stream_metrics:
            analysis.book_imbalance = getattr(stream_metrics, 'bid_ask_imbalance', 0.0)
            
            buy_vol = getattr(stream_metrics, 'buy_volume', 0.0)
            sell_vol = getattr(stream_metrics, 'sell_volume', 0.0)
            total_vol = buy_vol + sell_vol
            
            if total_vol > 0:
                analysis.volume_imbalance = (buy_vol - sell_vol) / total_vol
            
            analysis.imbalance_ratio = (analysis.book_imbalance + analysis.volume_imbalance) / 2
            
            if abs(analysis.imbalance_ratio) >= self.config.imbalance_threshold:
                analysis.is_significant = True
                analysis.direction = "bullish" if analysis.imbalance_ratio > 0 else "bearish"
            else:
                analysis.direction = "neutral"
        
        return analysis
    
    def _analyze_absorption(self) -> AbsorptionAnalysis:
        """Analyze absorption zones"""
        analysis = AbsorptionAnalysis()
        
        if not self._tape_analyzer:
            return analysis
        
        try:
            zones = self._tape_analyzer.get_absorption_zones(self.config.lookback_seconds)
            analysis.active_zones = len(zones)
            
            if zones:
                analysis.zones = zones
                analysis.total_absorbed_volume = sum(z.get('absorbed_volume', 0) for z in zones)
                
                strongest = max(zones, key=lambda z: z.get('absorbed_volume', 0))
                analysis.strongest_zone_price = strongest.get('price', 0.0)
                
                max_vol = max(z.get('absorbed_volume', 0) for z in zones) if zones else 0
                analysis.strongest_zone_strength = min(1.0, max_vol / self.config.large_order_threshold) if self.config.large_order_threshold > 0 else 0.0
        except Exception as e:
            logger.warning(f"Error analyzing absorption: {e}")
        
        return analysis
    
    def _analyze_manipulation(self) -> ManipulationAnalysis:
        """Analyze manipulation probability"""
        analysis = ManipulationAnalysis()
        
        if not self._manipulation_detector:
            return analysis
        
        try:
            scores = self._manipulation_detector.get_all_scores()
            
            analysis.overall_score = scores.get('overall', 0.0)
            analysis.stop_hunt_score = scores.get('stop_hunt', 0.0)
            analysis.spoofing_score = scores.get('spoofing', 0.0)
            analysis.fake_breakout_score = scores.get('fake_breakout', 0.0)
            analysis.sweep_score = scores.get('sweep', 0.0)
            
            analysis.is_high_risk = analysis.overall_score >= self.config.manipulation_reject_threshold
            
            if analysis.overall_score >= 0.7:
                analysis.recommendation = "avoid"
            elif analysis.overall_score >= 0.5:
                analysis.recommendation = "wait"
            elif analysis.overall_score >= 0.3:
                analysis.recommendation = "caution"
            else:
                analysis.recommendation = "trade"
            
            if analysis.stop_hunt_score > 0.5:
                analysis.notes.append(f"Stop hunt activity: {analysis.stop_hunt_score:.1%}")
            if analysis.spoofing_score > 0.5:
                analysis.notes.append(f"Spoofing detected: {analysis.spoofing_score:.1%}")
            if analysis.fake_breakout_score > 0.5:
                analysis.notes.append(f"Fake breakout: {analysis.fake_breakout_score:.1%}")
            
        except Exception as e:
            logger.warning(f"Error analyzing manipulation: {e}")
        
        return analysis
    
    def _calculate_institutional_activity(self, stream_metrics: Any) -> InstitutionalActivity:
        """Calculate institutional activity score"""
        activity = InstitutionalActivity()
        
        if stream_metrics:
            activity.large_order_count = getattr(stream_metrics, 'large_order_count', 0)
            activity.large_buy_count = getattr(stream_metrics, 'large_buy_count', 0)
            activity.large_sell_count = getattr(stream_metrics, 'large_sell_count', 0)
        
        if self._stream:
            try:
                large_orders = self._stream.get_large_orders(self.config.lookback_seconds)
                activity.total_large_notional = sum(o.notional_value for o in large_orders)
            except Exception:
                pass
        
        if self._tape_analyzer:
            try:
                activity.sweep_count = len(self._tape_analyzer.get_sweeps(self.config.lookback_seconds))
                activity.absorption_count = len(self._tape_analyzer.get_absorption_zones(self.config.lookback_seconds))
            except Exception:
                pass
        
        score = 0.0
        score += min(0.3, activity.large_order_count * 0.05)
        score += min(0.25, activity.sweep_count * 0.1)
        score += min(0.25, activity.absorption_count * 0.1)
        score += min(0.2, activity.total_large_notional / (self.config.large_order_threshold * 10))
        
        activity.activity_score = min(1.0, score)
        
        return activity
    
    def _calculate_smart_money_indicator(
        self,
        cvd: CVDAnalysis,
        absorption: AbsorptionAnalysis,
        stream_metrics: Any
    ) -> SmartMoneyIndicator:
        """Calculate smart money indicator"""
        indicator = SmartMoneyIndicator()
        
        if cvd.is_diverging:
            indicator.delta_divergence = 0.3 if cvd.divergence_direction == "bullish" else -0.3
        
        if absorption.active_zones > 0 and absorption.strongest_zone_strength > 0.5:
            indicator.absorption_at_levels = absorption.strongest_zone_strength * 0.25
        
        if stream_metrics:
            large_buy = getattr(stream_metrics, 'large_buy_count', 0)
            large_sell = getattr(stream_metrics, 'large_sell_count', 0)
            total_large = large_buy + large_sell
            
            if total_large > 0:
                indicator.large_order_direction = (large_buy - large_sell) / total_large * 0.25
        
        if self._tape_analyzer:
            try:
                sweeps = self._tape_analyzer.get_sweeps(60)
                up_sweeps = sum(1 for s in sweeps if s.get('direction') == 'UP')
                down_sweeps = sum(1 for s in sweeps if s.get('direction') == 'DOWN')
                total_sweeps = up_sweeps + down_sweeps
                
                if total_sweeps > 0:
                    indicator.sweep_direction = (up_sweeps - down_sweeps) / total_sweeps * 0.2
            except Exception:
                pass
        
        indicator.score = (
            indicator.delta_divergence +
            indicator.absorption_at_levels +
            indicator.large_order_direction +
            indicator.sweep_direction
        )
        
        indicator.score = max(-1.0, min(1.0, indicator.score))
        
        if indicator.score > 0.3:
            indicator.direction = "bullish"
        elif indicator.score < -0.3:
            indicator.direction = "bearish"
        else:
            indicator.direction = "neutral"
        
        component_count = sum([
            1 if indicator.delta_divergence != 0 else 0,
            1 if indicator.absorption_at_levels != 0 else 0,
            1 if indicator.large_order_direction != 0 else 0,
            1 if indicator.sweep_direction != 0 else 0
        ])
        
        indicator.confidence = min(1.0, component_count * 0.25 * abs(indicator.score) * 2)
        
        return indicator
    
    def _calculate_order_flow_bias(
        self,
        cvd: CVDAnalysis,
        imbalance: ImbalanceAnalysis,
        stream_metrics: Any,
        tape_signal: float
    ) -> float:
        """
        Calculate combined order flow bias
        
        Returns:
            Float from -1 (bearish) to +1 (bullish)
        """
        bias = 0.0
        weights_sum = 0.0
        
        if cvd.cvd_trend != "neutral":
            cvd_score = cvd.trend_strength if cvd.cvd_trend == "bullish" else -cvd.trend_strength
            bias += cvd_score * 0.25
            weights_sum += 0.25
        
        if imbalance.is_significant:
            imb_score = imbalance.imbalance_ratio
            bias += imb_score * 0.25
            weights_sum += 0.25
        
        if stream_metrics:
            large_buy = getattr(stream_metrics, 'large_buy_count', 0)
            large_sell = getattr(stream_metrics, 'large_sell_count', 0)
            total_large = large_buy + large_sell
            
            if total_large > 0:
                large_score = (large_buy - large_sell) / total_large
                bias += large_score * 0.25
                weights_sum += 0.25
        
        if abs(tape_signal) > 0.01:
            bias += tape_signal * 0.25
            weights_sum += 0.25
        
        if weights_sum > 0:
            bias = bias / weights_sum
        
        return max(-1.0, min(1.0, bias))
    
    def get_trading_bias(self) -> Tuple[TradingBias, float, str]:
        """
        Get simplified trading bias for signal enhancement
        
        Returns:
            Tuple of (TradingBias enum, confidence float, reason string)
        """
        metrics = self.get_complete_metrics()
        
        bias_value = metrics.order_flow_bias
        
        reasons = []
        
        if metrics.cvd.cvd_trend != "neutral":
            reasons.append(f"CVD {metrics.cvd.cvd_trend}")
        
        if metrics.imbalance.is_significant:
            reasons.append(f"Imbalance {metrics.imbalance.direction}")
        
        if metrics.smart_money.direction != "neutral":
            reasons.append(f"Smart money {metrics.smart_money.direction}")
        
        if abs(metrics.tape_signal) > 0.3:
            tape_dir = "bullish" if metrics.tape_signal > 0 else "bearish"
            reasons.append(f"Tape {tape_dir}")
        
        if bias_value >= 0.6:
            bias = TradingBias.STRONG_BULLISH
        elif bias_value >= 0.3:
            bias = TradingBias.BULLISH
        elif bias_value <= -0.6:
            bias = TradingBias.STRONG_BEARISH
        elif bias_value <= -0.3:
            bias = TradingBias.BEARISH
        else:
            bias = TradingBias.NEUTRAL
        
        confidence = abs(bias_value)
        reason = ", ".join(reasons) if reasons else "No strong signals"
        
        return bias, confidence, reason
    
    def should_trade(self, base_signal: str) -> Tuple[bool, str]:
        """
        Determine if a trade should be taken based on order flow
        
        Args:
            base_signal: Base signal direction ('LONG', 'SHORT', or 'NEUTRAL')
            
        Returns:
            Tuple of (should_trade bool, reason string)
        """
        metrics = self.get_complete_metrics()
        
        if metrics.manipulation.is_high_risk:
            return False, f"High manipulation risk: {metrics.manipulation.overall_score:.1%}"
        
        if metrics.manipulation.overall_score >= 0.5:
            return False, f"Elevated manipulation: {metrics.manipulation.overall_score:.1%}"
        
        bias, confidence, reason = self.get_trading_bias()
        
        if base_signal.upper() == 'LONG':
            if bias in [TradingBias.STRONG_BEARISH, TradingBias.BEARISH]:
                return False, f"Order flow bearish contradicts LONG: {reason}"
            if bias == TradingBias.NEUTRAL and confidence < 0.1:
                return True, "Neutral order flow, proceed with caution"
            return True, f"Order flow supports LONG: {reason}"
        
        elif base_signal.upper() == 'SHORT':
            if bias in [TradingBias.STRONG_BULLISH, TradingBias.BULLISH]:
                return False, f"Order flow bullish contradicts SHORT: {reason}"
            if bias == TradingBias.NEUTRAL and confidence < 0.1:
                return True, "Neutral order flow, proceed with caution"
            return True, f"Order flow supports SHORT: {reason}"
        
        return True, "Neutral signal, order flow check passed"
    
    def adjust_signal_confidence(self, base_confidence: float) -> float:
        """
        Adjust signal confidence based on order flow analysis
        
        Args:
            base_confidence: Original signal confidence (0 to 1)
            
        Returns:
            Adjusted confidence (0 to 1)
        """
        metrics = self.get_complete_metrics()
        
        manipulation_factor = 1.0 - metrics.manipulation.overall_score
        adjusted = base_confidence * manipulation_factor
        
        if metrics.institutional_activity.activity_score > 0.5:
            adjusted *= 1.0 + (metrics.institutional_activity.activity_score - 0.5) * 0.2
        
        if metrics.smart_money.confidence > 0.5:
            adjusted *= 1.0 + (metrics.smart_money.confidence - 0.5) * 0.15
        
        if metrics.delta_extremes.has_extreme:
            adjusted *= 1.0 + (metrics.delta_extremes.extreme_percentile / 100 - 0.99) * 2
        
        return max(0.0, min(1.0, adjusted))
    
    def get_manipulation_adjusted_confidence(self, base_confidence: float) -> float:
        """
        Get confidence adjusted for manipulation risk
        
        Args:
            base_confidence: Original confidence (0 to 1)
            
        Returns:
            Adjusted confidence: base_confidence * (1 - manipulation_score)
        """
        with self._lock:
            if self._manipulation_detector:
                try:
                    score = self._manipulation_detector.get_manipulation_score()
                    return base_confidence * (1.0 - score)
                except Exception as e:
                    logger.warning(f"Error getting manipulation score: {e}")
            
            return base_confidence
    
    def get_institutional_activity_score(self) -> float:
        """
        Get institutional activity score (0 to 1)
        
        Returns:
            Score based on large orders, absorption, sweeps
        """
        metrics = self.get_complete_metrics()
        return metrics.institutional_activity.activity_score
    
    def get_smart_money_indicator(self) -> Tuple[float, str, float]:
        """
        Get smart money indicator
        
        Returns:
            Tuple of (score -1 to 1, direction string, confidence)
        """
        metrics = self.get_complete_metrics()
        sm = metrics.smart_money
        return sm.score, sm.direction, sm.confidence
    
    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized"""
        return self._initialized
    
    @property
    def has_stream(self) -> bool:
        """Check if stream is connected"""
        return self._stream is not None
    
    @property
    def has_tape_analyzer(self) -> bool:
        """Check if tape analyzer is connected"""
        return self._tape_analyzer is not None
    
    @property
    def has_manipulation_detector(self) -> bool:
        """Check if manipulation detector is connected"""
        return self._manipulation_detector is not None
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            'initialized': self._initialized,
            'has_stream': self.has_stream,
            'has_tape_analyzer': self.has_tape_analyzer,
            'has_manipulation_detector': self.has_manipulation_detector,
            'cvd_history_size': len(self._cvd_history),
            'delta_history_size': len(self._delta_history),
            'price_history_size': len(self._price_history),
            'last_update': datetime.fromtimestamp(self._last_update).isoformat() if self._last_update > 0 else None,
            'config': {
                'cvd_trend_threshold': self.config.cvd_trend_threshold,
                'imbalance_threshold': self.config.imbalance_threshold,
                'manipulation_reject_threshold': self.config.manipulation_reject_threshold,
                'large_order_threshold': self.config.large_order_threshold
            }
        }


__all__ = [
    'OrderFlowMetricsService',
    'OrderFlowMetricsConfig',
    'CompleteOrderFlowMetrics',
    'CVDAnalysis',
    'DeltaExtremes',
    'ImbalanceAnalysis',
    'AbsorptionAnalysis',
    'ManipulationAnalysis',
    'InstitutionalActivity',
    'SmartMoneyIndicator',
    'TradingBias'
]
