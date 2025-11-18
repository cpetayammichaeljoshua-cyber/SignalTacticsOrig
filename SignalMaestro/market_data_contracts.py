#!/usr/bin/env python3
"""
Shared Data Contracts for Market Intelligence System
Defines common interfaces and data structures
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import pandas as pd

class AnalyzerType(Enum):
    """Types of market analyzers"""
    LIQUIDITY = "liquidity"
    ORDER_FLOW = "order_flow"
    VOLUME_PROFILE = "volume_profile"
    FOOTPRINT = "footprint"
    FRACTALS = "fractals"
    INTERMARKET = "intermarket"

class MarketBias(Enum):
    """Market bias directions"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

class SignalStrength(Enum):
    """Signal strength levels"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

@dataclass
class MarketSnapshot:
    """
    Normalized market data snapshot shared across analyzers
    """
    symbol: str
    timestamp: datetime
    
    # OHLCV data
    ohlcv_df: pd.DataFrame
    current_price: float
    
    # Order book depth (optional)
    bids: Optional[List[tuple]] = None  # [(price, volume), ...]
    asks: Optional[List[tuple]] = None  # [(price, volume), ...]
    
    # Recent trades (optional)
    recent_trades: Optional[List[Dict]] = None  # [{'price': float, 'volume': float, 'is_buyer_maker': bool}, ...]
    
    # Funding rate (for futures)
    funding_rate: Optional[float] = None
    
    # Open interest
    open_interest: Optional[float] = None
    
    # Market metrics
    volume_24h: Optional[float] = None
    price_change_24h: Optional[float] = None
    
    # Intermarket data (correlation with other symbols)
    correlated_symbols: Optional[Dict[str, pd.DataFrame]] = None
    
    def __post_init__(self):
        """Validate required data"""
        if self.ohlcv_df is None or len(self.ohlcv_df) == 0:
            raise ValueError("OHLCV DataFrame is required")
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in self.ohlcv_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required OHLCV columns: {missing_cols}")

@dataclass
class AnalysisResult:
    """
    Standardized output from any market analyzer
    """
    analyzer_type: AnalyzerType
    timestamp: datetime
    
    # Core metrics
    score: float  # 0-100, overall confidence in analysis
    bias: MarketBias  # bullish, bearish, or neutral
    confidence: float  # 0-100, confidence in the bias
    
    # Detailed findings
    signals: List[Dict] = field(default_factory=list)  # Specific signals detected
    key_levels: List[Dict] = field(default_factory=list)  # Important price levels
    metrics: Dict[str, Any] = field(default_factory=dict)  # Analyzer-specific metrics
    
    # Risk/Trade suggestions
    suggested_entry: Optional[float] = None
    suggested_stop: Optional[float] = None
    suggested_targets: Optional[List[float]] = None
    
    # Veto flags (reasons to NOT trade)
    veto_flags: List[str] = field(default_factory=list)
    
    # Metadata
    processing_time_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)
    
    def has_veto(self) -> bool:
        """Check if this analysis vetoes trading"""
        return len(self.veto_flags) > 0

@dataclass
class MarketIntelSnapshot:
    """
    Unified market intelligence combining all analyzer results
    """
    symbol: str
    timestamp: datetime
    
    # Individual analyzer results
    analyzer_results: Dict[AnalyzerType, AnalysisResult] = field(default_factory=dict)
    
    # Consensus metrics
    consensus_bias: MarketBias = MarketBias.NEUTRAL
    consensus_confidence: float = 0.0  # 0-100
    overall_score: float = 50.0  # 0-100
    
    # Aggregate findings
    dominant_signals: List[Dict] = field(default_factory=list)
    critical_levels: List[Dict] = field(default_factory=list)
    
    # Risk assessment
    total_veto_count: int = 0
    veto_reasons: List[str] = field(default_factory=list)
    risk_level: str = "moderate"  # low, moderate, high, extreme
    
    # Trading suggestions (aggregated)
    recommended_entry: Optional[float] = None
    recommended_stop: Optional[float] = None
    recommended_targets: Optional[List[float]] = None
    recommended_leverage: Optional[float] = None
    
    # Performance metrics
    total_processing_time_ms: float = 0.0
    analyzers_active: int = 0
    analyzers_failed: int = 0
    
    def should_trade(self) -> bool:
        """Determine if conditions are favorable for trading"""
        return (
            self.total_veto_count == 0 and
            self.consensus_confidence >= 65 and
            self.overall_score >= 60
        )
    
    def get_signal_strength(self) -> SignalStrength:
        """Categorize overall signal strength"""
        if self.overall_score >= 85:
            return SignalStrength.VERY_STRONG
        elif self.overall_score >= 75:
            return SignalStrength.STRONG
        elif self.overall_score >= 60:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def get_analyzer_count(self, status: str = "active") -> int:
        """Get count of analyzers by status"""
        if status == "active":
            return self.analyzers_active
        elif status == "failed":
            return self.analyzers_failed
        else:
            return len(self.analyzer_results)

@dataclass
class FusedSignal:
    """
    Final trading signal after fusion of all intelligence
    """
    symbol: str
    timestamp: datetime
    
    # Trade direction
    direction: str  # "LONG" or "SHORT"
    
    # Confidence and strength
    confidence: float  # 0-100
    strength: SignalStrength
    
    # Entry and risk management
    entry_price: float
    stop_loss: float
    take_profit_levels: List[float]
    
    # Position sizing
    recommended_leverage: float
    risk_reward_ratio: float
    
    # Supporting analysis
    primary_reason: str
    supporting_factors: List[str]
    
    # Source intelligence
    intel_snapshot: MarketIntelSnapshot
    
    # Metadata
    signal_id: str = ""
    expiry_timestamp: Optional[datetime] = None
    
    def is_valid(self) -> bool:
        """Check if signal is still valid"""
        if self.expiry_timestamp and datetime.now() > self.expiry_timestamp:
            return False
        return self.confidence >= 65 and self.stop_loss != 0

@dataclass
class OrderBookSnapshot:
    """Order book depth snapshot"""
    symbol: str
    timestamp: datetime
    bids: List[tuple]  # [(price, volume), ...]
    asks: List[tuple]  # [(price, volume), ...]
    
    def get_bid_ask_imbalance(self, depth: int = 10) -> float:
        """
        Calculate bid/ask imbalance
        Positive = more buy pressure, Negative = more sell pressure
        """
        bid_volume = sum(vol for _, vol in self.bids[:depth])
        ask_volume = sum(vol for _, vol in self.asks[:depth])
        
        total = bid_volume + ask_volume
        if total == 0:
            return 0.0
        
        return (bid_volume - ask_volume) / total
    
    def get_spread(self) -> float:
        """Get current spread"""
        if not self.bids or not self.asks:
            return 0.0
        
        best_bid = self.bids[0][0]
        best_ask = self.asks[0][0]
        return best_ask - best_bid
    
    def get_spread_percent(self) -> float:
        """Get spread as percentage of mid price"""
        spread = self.get_spread()
        if not self.bids or not self.asks:
            return 0.0
        
        mid_price = (self.bids[0][0] + self.asks[0][0]) / 2
        if mid_price == 0:
            return 0.0
        
        return (spread / mid_price) * 100

@dataclass
class VolumeProfileLevel:
    """Single level in volume profile"""
    price: float
    volume: float
    percentage: float  # Percentage of total volume
    is_poc: bool = False  # Point of Control (highest volume)
    is_vah: bool = False  # Value Area High
    is_val: bool = False  # Value Area Low

@dataclass
class VolumeProfile:
    """Volume profile analysis"""
    symbol: str
    timestamp: datetime
    timeframe: str
    
    levels: List[VolumeProfileLevel]
    poc_price: float  # Point of Control
    value_area_high: float  # 70% volume area high
    value_area_low: float  # 70% volume area low
    
    total_volume: float
    
    def get_current_position(self, current_price: float) -> str:
        """Determine if price is in value area, above, or below"""
        if current_price >= self.value_area_high:
            return "above_value"
        elif current_price <= self.value_area_low:
            return "below_value"
        else:
            return "in_value"
