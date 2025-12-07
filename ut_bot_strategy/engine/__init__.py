"""
Signal Engine Module

Contains the signal generation engine and order flow analysis tools:
- SignalEngine: Combines UT Bot and STC for trading signals
- FootprintBar: Aggregates per-price-level bid/ask volume for candles
- TapeAnalyzer: Analyzes Time & Sales data for trading signals
- ManipulationDetector: Identifies market manipulation patterns
"""

from .signal_engine import SignalEngine
from .tape_analyzer import (
    FootprintBar,
    TapeAnalyzer,
    PriceLevel,
    ImbalanceLevel,
    StackedImbalance,
    AbsorptionZone,
    SweepEvent,
    LargePrint,
    DeltaSpike,
    ImbalanceType,
    AuctionState
)
from .manipulation_detector import (
    ManipulationDetector,
    ManipulationAnalysis,
    ManipulationEvent,
    ManipulationType,
    ManipulationSeverity,
    TradingRecommendation,
    StopHuntEvent,
    SpoofingScore,
    SweepEvent as ManipulationSweepEvent,
    FakeBreakoutEvent,
    AbsorptionEvent
)

__all__ = [
    'SignalEngine',
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
    'AuctionState',
    'ManipulationDetector',
    'ManipulationAnalysis',
    'ManipulationEvent',
    'ManipulationType',
    'ManipulationSeverity',
    'TradingRecommendation',
    'StopHuntEvent',
    'SpoofingScore',
    'ManipulationSweepEvent',
    'FakeBreakoutEvent',
    'AbsorptionEvent'
]
