#!/usr/bin/env python3
"""
Market Intelligence Analyzer
Advanced order flow and market microstructure analysis
Detects institutional activity, volume imbalances, and price action patterns
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

class MarketIntelligence(Enum):
    """Market intelligence signals"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"

class InstitutionalActivity(Enum):
    """Institutional activity patterns"""
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    RANGING = "ranging"
    BREAKOUT = "breakout"
    NONE = "none"

@dataclass
class VolumeAnalysis:
    """Volume analysis result"""
    total_volume: float
    buy_volume: float
    sell_volume: float
    volume_ratio: float  # buy/sell ratio
    volume_imbalance: float  # -1 to 1
    avg_volume_period: float
    volume_trend: str  # increasing, decreasing, stable
    unusual_volume: bool

@dataclass
class OrderFlowIntelligence:
    """Order flow intelligence data"""
    market_structure: str  # "bullish", "bearish", "ranging"
    support_levels: List[float]
    resistance_levels: List[float]
    volume_poc: float  # Point of control
    institutional_signal: InstitutionalActivity
    momentum_score: float  # -100 to 100
    trend_strength: float  # 0 to 100
    volatility_regime: str  # "low", "normal", "high", "extreme"

class MarketIntelligenceAnalyzer:
    """Advanced market intelligence and order flow analyzer"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.min_bars_for_analysis = 100
        self.volume_threshold_multiplier = 1.5  # 150% of average = unusual
        
    async def analyze_volume_profile(self, ohlcv_data: pd.DataFrame) -> VolumeAnalysis:
        """Analyze volume profile and detect volume anomalies"""
        try:
            if len(ohlcv_data) < self.min_bars_for_analysis:
                raise ValueError(f"Insufficient data: {len(ohlcv_data)} < {self.min_bars_for_analysis}")
            
            # Calculate volume metrics
            total_volume = ohlcv_data['volume'].sum()
            
            # Estimate buy/sell volume based on price direction
            ohlcv_data['price_change'] = ohlcv_data['close'] - ohlcv_data['open']
            ohlcv_data['is_bullish'] = ohlcv_data['price_change'] >= 0
            
            buy_volume = ohlcv_data[ohlcv_data['is_bullish']]['volume'].sum()
            sell_volume = ohlcv_data[~ohlcv_data['is_bullish']]['volume'].sum()
            
            volume_ratio = buy_volume / sell_volume if sell_volume > 0 else 0
            volume_imbalance = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0
            
            # Detect unusual volume
            avg_volume = ohlcv_data['volume'].rolling(20).mean().iloc[-1]
            current_volume = ohlcv_data['volume'].iloc[-1]
            unusual_volume = current_volume > (avg_volume * self.volume_threshold_multiplier)
            
            # Volume trend
            recent_volume = ohlcv_data['volume'].iloc[-20:].mean()
            older_volume = ohlcv_data['volume'].iloc[-50:-20].mean()
            if recent_volume > older_volume * 1.1:
                volume_trend = "increasing"
            elif recent_volume < older_volume * 0.9:
                volume_trend = "decreasing"
            else:
                volume_trend = "stable"
            
            return VolumeAnalysis(
                total_volume=float(total_volume),
                buy_volume=float(buy_volume),
                sell_volume=float(sell_volume),
                volume_ratio=float(volume_ratio),
                volume_imbalance=float(volume_imbalance),
                avg_volume_period=float(avg_volume),
                volume_trend=volume_trend,
                unusual_volume=bool(unusual_volume)
            )
        except Exception as e:
            self.logger.error(f"Volume analysis error: {e}")
            return VolumeAnalysis(0, 0, 0, 0, 0, 0, "stable", False)
    
    async def detect_institutional_activity(self, ohlcv_data: pd.DataFrame) -> InstitutionalActivity:
        """Detect patterns suggesting institutional trading activity"""
        try:
            if len(ohlcv_data) < 50:
                return InstitutionalActivity.NONE
            
            # Get last 50 candles for recent activity
            recent = ohlcv_data.tail(50).copy()
            
            # Calculate metrics
            price_range = recent['high'] - recent['low']
            avg_range = price_range.mean()
            
            # Count lower closes (accumulation signal)
            recent['close_vs_mid'] = recent['close'] - ((recent['high'] + recent['low']) / 2)
            accumulation_score = len(recent[recent['close_vs_mid'] > 0])
            distribution_score = len(recent[recent['close_vs_mid'] < 0])
            
            # Calculate price direction
            price_movement = recent['close'].iloc[-1] - recent['close'].iloc[0]
            
            # Detect consolidation
            high_std = recent['high'].std()
            low_std = recent['low'].std()
            consolidation_level = (high_std + low_std) / 2
            
            # Determine activity type
            if abs(price_movement) > avg_range * 5:
                # Strong breakout
                return InstitutionalActivity.BREAKOUT
            elif accumulation_score > distribution_score * 1.3 and price_movement > 0:
                return InstitutionalActivity.ACCUMULATION
            elif distribution_score > accumulation_score * 1.3 and price_movement < 0:
                return InstitutionalActivity.DISTRIBUTION
            elif consolidation_level < avg_range * 0.5:
                return InstitutionalActivity.RANGING
            else:
                return InstitutionalActivity.NONE
        except Exception as e:
            self.logger.error(f"Institutional activity detection error: {e}")
            return InstitutionalActivity.NONE
    
    async def analyze_order_flow_intelligence(self, ohlcv_data: pd.DataFrame) -> OrderFlowIntelligence:
        """Comprehensive order flow intelligence analysis"""
        try:
            if len(ohlcv_data) < self.min_bars_for_analysis:
                raise ValueError("Insufficient data for analysis")
            
            # Volume analysis
            volume_analysis = await self.analyze_volume_profile(ohlcv_data)
            
            # Institutional activity
            institutional = await self.detect_institutional_activity(ohlcv_data)
            
            # Market structure
            recent = ohlcv_data.tail(20).copy()
            higher_highs = len(recent[recent['high'] > recent['high'].shift(1).fillna(0)])
            higher_lows = len(recent[recent['low'] > recent['low'].shift(1).fillna(0)])
            lower_highs = len(recent[recent['high'] < recent['high'].shift(1).fillna(0)])
            lower_lows = len(recent[recent['low'] < recent['low'].shift(1).fillna(0)])
            
            if higher_highs > lower_highs and higher_lows > lower_lows:
                market_structure = "bullish"
                momentum_score = min(100, 50 + (volume_analysis.volume_imbalance * 50))
            elif lower_highs > higher_highs and lower_lows > higher_lows:
                market_structure = "bearish"
                momentum_score = max(-100, -50 + (volume_analysis.volume_imbalance * 50))
            else:
                market_structure = "ranging"
                momentum_score = 0
            
            # Support and Resistance
            support_levels = self._find_support_levels(ohlcv_data, 3)
            resistance_levels = self._find_resistance_levels(ohlcv_data, 3)
            
            # Point of Control (volume-weighted price)
            volume_poc = (ohlcv_data['close'] * ohlcv_data['volume']).sum() / ohlcv_data['volume'].sum()
            
            # Trend strength
            closes = ohlcv_data['close'].tail(50).values
            trend_up = len(np.where(np.diff(closes) > 0)[0])
            trend_strength = (trend_up / len(closes)) * 100 if len(closes) > 0 else 50
            
            # Volatility regime
            volatility = ohlcv_data['close'].pct_change().std() * 100
            if volatility > 5:
                volatility_regime = "extreme"
            elif volatility > 2:
                volatility_regime = "high"
            elif volatility > 0.5:
                volatility_regime = "normal"
            else:
                volatility_regime = "low"
            
            return OrderFlowIntelligence(
                market_structure=market_structure,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                volume_poc=float(volume_poc),
                institutional_signal=institutional,
                momentum_score=float(momentum_score),
                trend_strength=float(trend_strength),
                volatility_regime=volatility_regime
            )
        except Exception as e:
            self.logger.error(f"Order flow intelligence error: {e}")
            return OrderFlowIntelligence("ranging", [], [], 0, InstitutionalActivity.NONE, 0, 50, "normal")
    
    def _find_support_levels(self, ohlcv_data: pd.DataFrame, count: int = 3) -> List[float]:
        """Find support levels using swing lows"""
        try:
            lows = ohlcv_data['low'].tail(100).values
            if len(lows) < 20:
                return []
            
            # Find local minima
            supports = []
            for i in range(10, len(lows) - 10):
                if lows[i] < lows[i-10:i].mean() and lows[i] < lows[i+1:i+11].mean():
                    supports.append(float(lows[i]))
            
            # Return top count unique levels
            supports = sorted(list(set([round(s, 5) for s in supports])))
            return supports[-count:] if len(supports) >= count else supports
        except:
            return []
    
    def _find_resistance_levels(self, ohlcv_data: pd.DataFrame, count: int = 3) -> List[float]:
        """Find resistance levels using swing highs"""
        try:
            highs = ohlcv_data['high'].tail(100).values
            if len(highs) < 20:
                return []
            
            # Find local maxima
            resistances = []
            for i in range(10, len(highs) - 10):
                if highs[i] > highs[i-10:i].mean() and highs[i] > highs[i+1:i+11].mean():
                    resistances.append(float(highs[i]))
            
            # Return top count unique levels
            resistances = sorted(list(set([round(r, 5) for r in resistances])))
            return resistances[-count:] if len(resistances) >= count else resistances
        except:
            return []
    
    async def get_market_intelligence_summary(self, ohlcv_data: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive market intelligence summary"""
        try:
            volume_analysis = await self.analyze_volume_profile(ohlcv_data)
            order_flow = await self.analyze_order_flow_intelligence(ohlcv_data)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'volume': {
                    'total': volume_analysis.total_volume,
                    'buy_sell_ratio': volume_analysis.volume_ratio,
                    'imbalance': volume_analysis.volume_imbalance,
                    'trend': volume_analysis.volume_trend,
                    'unusual': volume_analysis.unusual_volume
                },
                'market_structure': {
                    'structure': order_flow.market_structure,
                    'support': order_flow.support_levels,
                    'resistance': order_flow.resistance_levels,
                    'poc': order_flow.volume_poc
                },
                'institutional': {
                    'activity': order_flow.institutional_signal.value,
                    'momentum_score': order_flow.momentum_score,
                    'trend_strength': order_flow.trend_strength
                },
                'volatility': order_flow.volatility_regime,
                'signal': self._get_intelligence_signal(order_flow, volume_analysis)
            }
        except Exception as e:
            self.logger.error(f"Market intelligence summary error: {e}")
            return {'error': str(e)}
    
    def _get_intelligence_signal(self, order_flow: OrderFlowIntelligence, 
                                 volume: VolumeAnalysis) -> str:
        """Generate trading signal from market intelligence"""
        score = 0
        
        # Market structure score
        if order_flow.market_structure == "bullish":
            score += 25
        elif order_flow.market_structure == "bearish":
            score -= 25
        
        # Institutional activity
        if order_flow.institutional_signal == InstitutionalActivity.ACCUMULATION:
            score += 20
        elif order_flow.institutional_signal == InstitutionalActivity.DISTRIBUTION:
            score -= 20
        
        # Volume imbalance
        score += volume.volume_imbalance * 30
        
        # Momentum
        score += order_flow.momentum_score * 0.25
        
        # Decision
        if score > 40:
            return MarketIntelligence.STRONG_BUY.value
        elif score > 15:
            return MarketIntelligence.BUY.value
        elif score < -40:
            return MarketIntelligence.STRONG_SELL.value
        elif score < -15:
            return MarketIntelligence.SELL.value
        else:
            return MarketIntelligence.NEUTRAL.value

# Create global instance
market_analyzer = MarketIntelligenceAnalyzer()
