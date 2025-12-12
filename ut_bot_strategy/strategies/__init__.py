"""
Trading Strategies Module

Contains specialized trading strategies for different market conditions:
- ScalpingStrategy: Ultra-fast 1m timeframe scalping with momentum detection
- (Future) SwingStrategy: Multi-day trend following
- (Future) RangeStrategy: Mean-reversion in sideways markets
"""

from .scalping_strategy import (
    ScalpingStrategy,
    ScalpingSignal,
    ScalpingMode,
    MomentumType,
    ScalpingConfig
)

__all__ = [
    'ScalpingStrategy',
    'ScalpingSignal', 
    'ScalpingMode',
    'MomentumType',
    'ScalpingConfig'
]
