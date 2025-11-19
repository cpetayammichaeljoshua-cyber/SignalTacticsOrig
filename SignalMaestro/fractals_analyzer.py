#!/usr/bin/env python3
"""
Fractals Detection and Analysis Module
Identifies market structure, swing points, and fractal patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

from SignalMaestro.market_data_contracts import (
    AnalysisResult, AnalyzerType, MarketBias, MarketSnapshot
)

class FractalsAnalyzer:
    """
    Analyzes market fractals:
    - Williams Fractals (5-bar patterns)
    - Market structure (higher highs, lower lows)
    - Swing points
    - Trend structure
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.fractal_period = 5  # Williams Fractal uses 5 bars
        
    def analyze(self, market_snapshot: MarketSnapshot) -> AnalysisResult:
        """
        Main analysis entry point
        
        Args:
            market_snapshot: Standardized market data
            
        Returns:
            AnalysisResult with fractals analysis
        """
        start_time = datetime.now()
        
        df = market_snapshot.ohlcv_df.copy()
        
        # Detect fractals
        fractals_data = self._detect_fractals(df)
        
        # Analyze market structure
        structure_data = self._analyze_market_structure(df, fractals_data)
        
        # Identify trend
        trend_data = self._identify_trend(df, structure_data)
        
        # Find key swing levels
        swing_levels = self._identify_swing_levels(fractals_data)
        
        # Determine bias
        bias, confidence = self._determine_bias(structure_data, trend_data)
        
        # Calculate score
        score = self._calculate_score(structure_data, trend_data, fractals_data)
        
        # Check veto conditions
        veto_flags = self._check_veto_conditions(structure_data, trend_data)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return AnalysisResult(
            analyzer_type=AnalyzerType.FRACTALS,
            timestamp=market_snapshot.timestamp,
            score=score,
            bias=bias,
            confidence=confidence,
            signals=[
                {
                    'type': 'market_structure',
                    'structure': structure_data['current_structure'],
                    'trend': trend_data['trend']
                }
            ],
            key_levels=swing_levels,
            metrics={
                'fractals': fractals_data,
                'structure': structure_data,
                'trend': trend_data
            },
            veto_flags=veto_flags,
            processing_time_ms=processing_time
        )
    
    def _detect_fractals(self, df: pd.DataFrame) -> Dict:
        """
        Detect Williams Fractals
        Up Fractal: Middle candle high is highest of 5 consecutive candles
        Down Fractal: Middle candle low is lowest of 5 consecutive candles
        """
        if len(df) < self.fractal_period:
            return {
                'up_fractals': [],
                'down_fractals': [],
                'total_fractals': 0
            }
        
        up_fractals = []
        down_fractals = []
        
        for i in range(2, len(df) - 2):
            # Up fractal (resistance)
            if (df.iloc[i]['high'] > df.iloc[i-1]['high'] and
                df.iloc[i]['high'] > df.iloc[i-2]['high'] and
                df.iloc[i]['high'] > df.iloc[i+1]['high'] and
                df.iloc[i]['high'] > df.iloc[i+2]['high']):
                
                up_fractals.append({
                    'index': i,
                    'price': float(df.iloc[i]['high']),
                    'timestamp': df.iloc[i].get('timestamp', i)
                })
            
            # Down fractal (support)
            if (df.iloc[i]['low'] < df.iloc[i-1]['low'] and
                df.iloc[i]['low'] < df.iloc[i-2]['low'] and
                df.iloc[i]['low'] < df.iloc[i+1]['low'] and
                df.iloc[i]['low'] < df.iloc[i+2]['low']):
                
                down_fractals.append({
                    'index': i,
                    'price': float(df.iloc[i]['low']),
                    'timestamp': df.iloc[i].get('timestamp', i)
                })
        
        return {
            'up_fractals': up_fractals,
            'down_fractals': down_fractals,
            'total_fractals': len(up_fractals) + len(down_fractals),
            'recent_up': up_fractals[-5:] if up_fractals else [],
            'recent_down': down_fractals[-5:] if down_fractals else []
        }
    
    def _analyze_market_structure(self, df: pd.DataFrame, fractals_data: Dict) -> Dict:
        """
        Analyze market structure using fractals
        Higher Highs (HH), Higher Lows (HL) = Uptrend
        Lower Highs (LH), Lower Lows (LL) = Downtrend
        """
        up_fractals = fractals_data.get('up_fractals', [])
        down_fractals = fractals_data.get('down_fractals', [])
        
        if len(up_fractals) < 2 or len(down_fractals) < 2:
            return {
                'current_structure': 'undefined',
                'structure_strength': 0,
                'breaks': [],
                'confirmation': False
            }
        
        # Analyze recent highs
        recent_highs = [f['price'] for f in up_fractals[-3:]]
        recent_lows = [f['price'] for f in down_fractals[-3:]]
        
        # Check for higher highs
        higher_highs = all(recent_highs[i] < recent_highs[i+1] for i in range(len(recent_highs)-1))
        
        # Check for higher lows
        higher_lows = all(recent_lows[i] < recent_lows[i+1] for i in range(len(recent_lows)-1))
        
        # Check for lower highs
        lower_highs = all(recent_highs[i] > recent_highs[i+1] for i in range(len(recent_highs)-1))
        
        # Check for lower lows
        lower_lows = all(recent_lows[i] > recent_lows[i+1] for i in range(len(recent_lows)-1))
        
        # Determine structure
        if higher_highs and higher_lows:
            structure = 'bullish'  # HH + HL
            strength = 90
            confirmation = True
        elif lower_highs and lower_lows:
            structure = 'bearish'  # LH + LL
            strength = 90
            confirmation = True
        elif higher_highs and not lower_lows:
            structure = 'bullish'  # Partial bullish
            strength = 70
            confirmation = False
        elif lower_highs and not higher_lows:
            structure = 'bearish'  # Partial bearish
            strength = 70
            confirmation = False
        else:
            structure = 'ranging'  # No clear structure
            strength = 50
            confirmation = False
        
        # Detect structure breaks
        breaks = self._detect_structure_breaks(df, up_fractals, down_fractals)
        
        return {
            'current_structure': structure,
            'structure_strength': strength,
            'higher_highs': higher_highs,
            'higher_lows': higher_lows,
            'lower_highs': lower_highs,
            'lower_lows': lower_lows,
            'breaks': breaks,
            'confirmation': confirmation
        }
    
    def _detect_structure_breaks(self, df: pd.DataFrame, 
                                 up_fractals: List[Dict], 
                                 down_fractals: List[Dict]) -> List[Dict]:
        """Detect when market structure is broken"""
        breaks = []
        
        if not up_fractals or not down_fractals or len(df) < 10:
            return breaks
        
        recent_df = df.tail(20)
        
        # Check if recent price broke above recent swing high
        if up_fractals:
            last_swing_high = up_fractals[-1]['price']
            recent_high = recent_df['high'].max()
            
            if recent_high > last_swing_high * 1.001:  # 0.1% breakout
                breaks.append({
                    'type': 'breakout_high',
                    'level': last_swing_high,
                    'direction': 'bullish'
                })
        
        # Check if recent price broke below recent swing low
        if down_fractals:
            last_swing_low = down_fractals[-1]['price']
            recent_low = recent_df['low'].min()
            
            if recent_low < last_swing_low * 0.999:  # 0.1% breakdown
                breaks.append({
                    'type': 'breakdown_low',
                    'level': last_swing_low,
                    'direction': 'bearish'
                })
        
        return breaks
    
    def _identify_trend(self, df: pd.DataFrame, structure_data: Dict) -> Dict:
        """Identify overall trend using multiple methods"""
        if len(df) < 50:
            return {
                'trend': 'neutral',
                'strength': 0,
                'confirmation': False
            }
        
        recent = df.tail(50)
        
        # Method 1: Price slope
        price_slope = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / len(recent)
        
        # Method 2: Moving averages
        sma20 = recent['close'].rolling(window=20).mean().iloc[-1]
        sma50 = recent['close'].rolling(window=50).mean().iloc[-1] if len(recent) >= 50 else sma20
        current_price = recent['close'].iloc[-1]
        
        # Method 3: Market structure
        structure = structure_data['current_structure']
        
        # Combine methods
        bullish_signals = 0
        bearish_signals = 0
        
        if price_slope > 0:
            bullish_signals += 1
        elif price_slope < 0:
            bearish_signals += 1
        
        if current_price > sma20 > sma50:
            bullish_signals += 1
        elif current_price < sma20 < sma50:
            bearish_signals += 1
        
        if structure == 'bullish':
            bullish_signals += 1
        elif structure == 'bearish':
            bearish_signals += 1
        
        # Determine trend
        if bullish_signals >= 2:
            trend = 'bullish'
            strength = (bullish_signals / 3) * 100
            confirmation = bullish_signals == 3
        elif bearish_signals >= 2:
            trend = 'bearish'
            strength = (bearish_signals / 3) * 100
            confirmation = bearish_signals == 3
        else:
            trend = 'neutral'
            strength = 50
            confirmation = False
        
        return {
            'trend': trend,
            'strength': strength,
            'confirmation': confirmation,
            'price_above_sma20': current_price > sma20,
            'sma20_above_sma50': sma20 > sma50
        }
    
    def _identify_swing_levels(self, fractals_data: Dict) -> List[Dict]:
        """Identify key swing levels from fractals"""
        levels = []
        
        # Recent up fractals (resistance)
        for fractal in fractals_data.get('recent_up', []):
            levels.append({
                'price': fractal['price'],
                'type': 'fractal_resistance',
                'strength': 70
            })
        
        # Recent down fractals (support)
        for fractal in fractals_data.get('recent_down', []):
            levels.append({
                'price': fractal['price'],
                'type': 'fractal_support',
                'strength': 70
            })
        
        return levels
    
    def _determine_bias(self, structure_data: Dict, trend_data: Dict) -> Tuple[MarketBias, float]:
        """Determine overall bias from structure and trend"""
        structure = structure_data['current_structure']
        trend = trend_data['trend']
        
        # Both agree
        if structure == 'bullish' and trend == 'bullish':
            return MarketBias.BULLISH, 90
        elif structure == 'bearish' and trend == 'bearish':
            return MarketBias.BEARISH, 90
        
        # Structure takes priority
        elif structure == 'bullish':
            return MarketBias.BULLISH, 70
        elif structure == 'bearish':
            return MarketBias.BEARISH, 70
        
        # Trend alone
        elif trend == 'bullish':
            return MarketBias.BULLISH, 60
        elif trend == 'bearish':
            return MarketBias.BEARISH, 60
        
        # Neutral
        else:
            return MarketBias.NEUTRAL, 50
    
    def _calculate_score(self, structure_data: Dict, 
                        trend_data: Dict, fractals_data: Dict) -> float:
        """Calculate overall fractals score"""
        score = 50.0
        
        # Strong structure
        if structure_data['confirmation']:
            score += 20
        
        # Strong trend
        if trend_data['confirmation']:
            score += 15
        
        # Structure breaks (momentum)
        if len(structure_data['breaks']) > 0:
            score += 10
        
        # Good fractal count
        if fractals_data['total_fractals'] > 10:
            score += 5
        
        return min(max(score, 0), 100)
    
    def _check_veto_conditions(self, structure_data: Dict, trend_data: Dict) -> List[str]:
        """Check for veto conditions"""
        veto_flags = []
        
        # Conflicting signals
        if structure_data['current_structure'] == 'bullish' and trend_data['trend'] == 'bearish':
            veto_flags.append("Structure and trend conflict (structure bullish, trend bearish)")
        elif structure_data['current_structure'] == 'bearish' and trend_data['trend'] == 'bullish':
            veto_flags.append("Structure and trend conflict (structure bearish, trend bullish)")
        
        # Ranging market
        if structure_data['current_structure'] == 'ranging':
            veto_flags.append("Ranging market structure")
        
        return veto_flags
