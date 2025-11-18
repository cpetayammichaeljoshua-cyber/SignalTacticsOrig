#!/usr/bin/env python3
"""
Advanced Order Flow Analysis Module
CVD (Cumulative Volume Delta), bid/ask imbalance, and smart money detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import logging

from SignalMaestro.market_data_contracts import (
    AnalysisResult, AnalyzerType, MarketBias, MarketSnapshot
)

class AdvancedOrderFlowAnalyzer:
    """
    Analyzes order flow to detect:
    - CVD (Cumulative Volume Delta)
    - Bid/Ask imbalance
    - Buying/Selling pressure
    - Smart money accumulation/distribution
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cvd_history = []
        
    def analyze(self, market_snapshot: MarketSnapshot) -> AnalysisResult:
        """
        Main analysis entry point
        
        Args:
            market_snapshot: Standardized market data
            
        Returns:
            AnalysisResult with order flow analysis
        """
        start_time = datetime.now()
        
        df = market_snapshot.ohlcv_df
        
        # Calculate CVD (Cumulative Volume Delta)
        cvd_data = self._calculate_cvd(df)
        
        # Analyze bid/ask imbalance
        imbalance_data = self._analyze_order_book_imbalance(market_snapshot)
        
        # Detect buying/selling pressure
        pressure_data = self._analyze_pressure(df, cvd_data)
        
        # Detect accumulation/distribution
        smart_money = self._detect_smart_money(df, cvd_data)
        
        # Determine overall bias
        bias, confidence = self._determine_bias(cvd_data, pressure_data, smart_money)
        
        # Calculate overall score
        score = self._calculate_score(cvd_data, pressure_data, smart_money, confidence)
        
        # Identify veto flags
        veto_flags = self._check_veto_conditions(cvd_data, pressure_data)
        
        # Key levels from order flow
        key_levels = self._identify_key_levels(df, cvd_data)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return AnalysisResult(
            analyzer_type=AnalyzerType.ORDER_FLOW,
            timestamp=market_snapshot.timestamp,
            score=score,
            bias=bias,
            confidence=confidence,
            signals=[
                {
                    'type': 'cvd',
                    'value': cvd_data['current_cvd'],
                    'trend': cvd_data['cvd_trend']
                },
                {
                    'type': 'pressure',
                    'buying_pressure': pressure_data['buying_pressure'],
                    'selling_pressure': pressure_data['selling_pressure']
                }
            ],
            key_levels=key_levels,
            metrics={
                'cvd': cvd_data,
                'pressure': pressure_data,
                'smart_money': smart_money,
                'imbalance': imbalance_data
            },
            veto_flags=veto_flags,
            processing_time_ms=processing_time
        )
    
    def _calculate_cvd(self, df: pd.DataFrame) -> Dict:
        """
        Calculate Cumulative Volume Delta
        CVD = sum of (volume when price increases - volume when price decreases)
        """
        if len(df) < 2:
            return {
                'current_cvd': 0,
                'cvd_trend': 'neutral',
                'cvd_divergence': False,
                'cvd_strength': 0
            }
        
        # Calculate volume delta for each candle
        df = df.copy()
        df['price_change'] = df['close'] - df['open']
        
        # Positive delta when price increases, negative when decreases
        df['volume_delta'] = np.where(
            df['price_change'] > 0,
            df['volume'],
            -df['volume']
        )
        
        # Cumulative sum
        df['cvd'] = df['volume_delta'].cumsum()
        
        current_cvd = df['cvd'].iloc[-1]
        
        # Determine CVD trend
        recent_cvd = df['cvd'].tail(20)
        cvd_slope = (recent_cvd.iloc[-1] - recent_cvd.iloc[0]) / len(recent_cvd)
        
        if cvd_slope > 0:
            cvd_trend = 'bullish'
        elif cvd_slope < 0:
            cvd_trend = 'bearish'
        else:
            cvd_trend = 'neutral'
        
        # Check for divergence (price going one way, CVD going another)
        price_slope = (df['close'].iloc[-1] - df['close'].iloc[-20]) / 20
        cvd_divergence = (price_slope > 0 and cvd_slope < 0) or (price_slope < 0 and cvd_slope > 0)
        
        # CVD strength
        cvd_std = recent_cvd.std()
        cvd_strength = min(abs(cvd_slope) / cvd_std * 100, 100) if cvd_std != 0 else 0
        
        self.cvd_history.append({
            'timestamp': datetime.now(),
            'cvd': current_cvd,
            'trend': cvd_trend
        })
        
        # Keep only recent history
        self.cvd_history = self.cvd_history[-1000:]
        
        return {
            'current_cvd': float(current_cvd),
            'cvd_trend': cvd_trend,
            'cvd_divergence': cvd_divergence,
            'cvd_strength': float(cvd_strength),
            'cvd_slope': float(cvd_slope)
        }
    
    def _analyze_order_book_imbalance(self, snapshot: MarketSnapshot) -> Dict:
        """Analyze order book bid/ask imbalance"""
        if not snapshot.bids or not snapshot.asks:
            return {
                'imbalance': 0.0,
                'bias': 'neutral',
                'strength': 0
            }
        
        # Calculate imbalance in top 10 levels
        bid_volume = sum(vol for _, vol in snapshot.bids[:10])
        ask_volume = sum(vol for _, vol in snapshot.asks[:10])
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return {
                'imbalance': 0.0,
                'bias': 'neutral',
                'strength': 0
            }
        
        imbalance = (bid_volume - ask_volume) / total_volume
        
        if imbalance > 0.2:
            bias = 'bullish'
        elif imbalance < -0.2:
            bias = 'bearish'
        else:
            bias = 'neutral'
        
        strength = abs(imbalance) * 100
        
        return {
            'imbalance': float(imbalance),
            'bias': bias,
            'strength': float(strength),
            'bid_volume': float(bid_volume),
            'ask_volume': float(ask_volume)
        }
    
    def _analyze_pressure(self, df: pd.DataFrame, cvd_data: Dict) -> Dict:
        """Analyze buying and selling pressure"""
        recent = df.tail(20)
        
        # Calculate buying pressure (green candles with high volume)
        buying_candles = recent[recent['close'] > recent['open']]
        buying_pressure = buying_candles['volume'].sum() if len(buying_candles) > 0 else 0
        
        # Calculate selling pressure (red candles with high volume)
        selling_candles = recent[recent['close'] < recent['open']]
        selling_pressure = selling_candles['volume'].sum() if len(selling_candles) > 0 else 0
        
        total_pressure = buying_pressure + selling_pressure
        
        if total_pressure == 0:
            pressure_ratio = 0
        else:
            pressure_ratio = (buying_pressure - selling_pressure) / total_pressure
        
        # Determine dominant pressure
        if pressure_ratio > 0.3:
            dominant = 'buyers'
        elif pressure_ratio < -0.3:
            dominant = 'sellers'
        else:
            dominant = 'balanced'
        
        return {
            'buying_pressure': float(buying_pressure),
            'selling_pressure': float(selling_pressure),
            'pressure_ratio': float(pressure_ratio),
            'dominant': dominant,
            'strength': abs(pressure_ratio) * 100
        }
    
    def _detect_smart_money(self, df: pd.DataFrame, cvd_data: Dict) -> Dict:
        """
        Detect smart money accumulation/distribution
        Smart money often moves against retail (counter-trend)
        """
        if len(df) < 50:
            return {
                'activity': 'none',
                'phase': 'neutral',
                'confidence': 0
            }
        
        recent = df.tail(50)
        
        # Look for patterns:
        # Accumulation: Price falling but volume increasing (buying the dip)
        # Distribution: Price rising but volume decreasing (selling the rip)
        
        price_trend = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
        volume_trend = (recent['volume'].tail(10).mean() - recent['volume'].head(10).mean()) / recent['volume'].head(10).mean()
        
        cvd_trend = cvd_data['cvd_trend']
        
        # Accumulation signals
        if price_trend < -0.02 and volume_trend > 0.1 and cvd_trend == 'bullish':
            activity = 'accumulation'
            phase = 'bullish'
            confidence = min((abs(volume_trend) + abs(price_trend)) * 50, 100)
        # Distribution signals
        elif price_trend > 0.02 and volume_trend < -0.1 and cvd_trend == 'bearish':
            activity = 'distribution'
            phase = 'bearish'
            confidence = min((abs(volume_trend) + abs(price_trend)) * 50, 100)
        else:
            activity = 'none'
            phase = 'neutral'
            confidence = 0
        
        return {
            'activity': activity,
            'phase': phase,
            'confidence': float(confidence)
        }
    
    def _determine_bias(self, cvd_data: Dict, pressure_data: Dict, smart_money: Dict) -> tuple:
        """Determine overall order flow bias"""
        bullish_signals = 0
        bearish_signals = 0
        
        # CVD trend
        if cvd_data['cvd_trend'] == 'bullish':
            bullish_signals += 1
        elif cvd_data['cvd_trend'] == 'bearish':
            bearish_signals += 1
        
        # Pressure
        if pressure_data['dominant'] == 'buyers':
            bullish_signals += 1
        elif pressure_data['dominant'] == 'sellers':
            bearish_signals += 1
        
        # Smart money
        if smart_money['phase'] == 'bullish':
            bullish_signals += 1
        elif smart_money['phase'] == 'bearish':
            bearish_signals += 1
        
        # Determine bias
        if bullish_signals > bearish_signals:
            bias = MarketBias.BULLISH
            confidence = (bullish_signals / 3) * 100
        elif bearish_signals > bullish_signals:
            bias = MarketBias.BEARISH
            confidence = (bearish_signals / 3) * 100
        else:
            bias = MarketBias.NEUTRAL
            confidence = 50
        
        # Boost confidence if multiple strong signals
        if cvd_data['cvd_strength'] > 70:
            confidence = min(confidence + 10, 100)
        if pressure_data['strength'] > 70:
            confidence = min(confidence + 10, 100)
        
        return bias, confidence
    
    def _calculate_score(self, cvd_data: Dict, pressure_data: Dict, 
                        smart_money: Dict, confidence: float) -> float:
        """Calculate overall order flow score (0-100)"""
        score = 50.0  # Base score
        
        # Add CVD strength
        score += cvd_data['cvd_strength'] * 0.2
        
        # Add pressure strength
        score += pressure_data['strength'] * 0.2
        
        # Add smart money confidence
        score += smart_money['confidence'] * 0.1
        
        # Add general confidence
        score += confidence * 0.1
        
        return min(max(score, 0), 100)
    
    def _check_veto_conditions(self, cvd_data: Dict, pressure_data: Dict) -> List[str]:
        """Check for conditions that should veto trading"""
        veto_flags = []
        
        # Divergence is a warning sign
        if cvd_data['cvd_divergence']:
            veto_flags.append("CVD divergence detected")
        
        # Extremely low pressure/volume
        total_pressure = pressure_data['buying_pressure'] + pressure_data['selling_pressure']
        if total_pressure < 100:  # Arbitrary threshold
            veto_flags.append("Extremely low volume/pressure")
        
        return veto_flags
    
    def _identify_key_levels(self, df: pd.DataFrame, cvd_data: Dict) -> List[Dict]:
        """Identify key price levels based on order flow"""
        levels = []
        
        recent = df.tail(50)
        
        # Find high volume nodes
        volume_threshold = recent['volume'].quantile(0.8)
        high_volume_bars = recent[recent['volume'] > volume_threshold]
        
        for idx, row in high_volume_bars.iterrows():
            levels.append({
                'price': float(row['close']),
                'type': 'high_volume_node',
                'volume': float(row['volume']),
                'importance': 'high'
            })
        
        return levels[-5:]  # Return top 5
