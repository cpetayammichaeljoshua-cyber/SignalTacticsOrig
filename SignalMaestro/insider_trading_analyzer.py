#!/usr/bin/env python3
"""
Insider Trading Analyzer
Detects legitimate institutional/insider trading activity through market microstructure
NOT illegal - based on order flow, volume profile, accumulation/distribution patterns
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd

@dataclass
class InsiderSignal:
    """Insider trading signal"""
    detected: bool
    confidence: float  # 0-100
    activity_type: str  # accumulation, distribution, volume_surge, whale_activity
    description: str
    recommendation: str
    strength: float  # 0-100

class InsiderTradingAnalyzer:
    """Detects institutional trading patterns through microstructure analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def detect_insider_activity(self, market_data) -> InsiderSignal:
        """
        Detect insider/institutional trading activity
        Legitimate detection based on:
        - Large volume spikes
        - Accumulation/distribution patterns
        - Order flow imbalance
        - Market microstructure anomalies
        """
        try:
            if isinstance(market_data, list):
                if not market_data or len(market_data) < 50:
                    return InsiderSignal(False, 0, "none", "Insufficient data", "Wait", 0)
                market_data = pd.DataFrame(market_data, columns=['open', 'high', 'low', 'close', 'volume'])
            elif len(market_data) < 50:
                return InsiderSignal(False, 0, "none", "Insufficient data", "Wait", 0)
            
            recent = market_data.tail(50).copy()
            
            # Calculate volume metrics
            avg_volume = recent['volume'].mean()
            current_volume = recent['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Check for unusual volume (whale activity)
            if volume_ratio > 3.0:
                return InsiderSignal(
                    detected=True,
                    confidence=85.0,
                    activity_type="whale_activity",
                    description="🐋 Large volume spike detected - Institutional buying/selling",
                    recommendation="🎯 High probability move incoming",
                    strength=volume_ratio * 20
                )
            
            # Detect accumulation (large buyers)
            recent['close_position'] = (recent['close'] - recent['low']) / (recent['high'] - recent['low'])
            accumulation_score = (recent['close_position'] > 0.7).sum() / len(recent) * 100
            
            if accumulation_score > 70 and volume_ratio > 1.5:
                return InsiderSignal(
                    detected=True,
                    confidence=78.0,
                    activity_type="accumulation",
                    description="📈 Accumulation pattern detected - Institutional buyers entering",
                    recommendation="🎯 Bullish signal - Consider LONG",
                    strength=accumulation_score
                )
            
            # Detect distribution (large sellers)
            distribution_score = (recent['close_position'] < 0.3).sum() / len(recent) * 100
            
            if distribution_score > 70 and volume_ratio > 1.5:
                return InsiderSignal(
                    detected=True,
                    confidence=78.0,
                    activity_type="distribution",
                    description="📉 Distribution pattern detected - Institutional sellers exiting",
                    recommendation="🎯 Bearish signal - Consider SHORT",
                    strength=distribution_score
                )
            
            # Detect volume surge (without extreme ratio)
            if volume_ratio > 1.8:
                return InsiderSignal(
                    detected=True,
                    confidence=72.0,
                    activity_type="volume_surge",
                    description="📊 Volume surge detected - Institutional activity present",
                    recommendation="⚠️ Increased volatility expected",
                    strength=volume_ratio * 15
                )
            
            return InsiderSignal(
                detected=False,
                confidence=0,
                activity_type="none",
                description="No significant insider activity detected",
                recommendation="Monitor price action",
                strength=0
            )
            
        except Exception as e:
            self.logger.error(f"Insider activity detection error: {e}")
            return InsiderSignal(
                detected=False,
                confidence=0,
                activity_type="error",
                description=f"Detection error: {str(e)}",
                recommendation="Check system",
                strength=0
            )

# Global instance
insider_analyzer = InsiderTradingAnalyzer()
