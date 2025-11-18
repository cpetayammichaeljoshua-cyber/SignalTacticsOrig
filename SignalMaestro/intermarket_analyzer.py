#!/usr/bin/env python3
"""
Intermarket Data Correlation Analysis
Analyzes correlations with related markets and indices
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import logging

from SignalMaestro.market_data_contracts import (
    AnalysisResult, AnalyzerType, MarketBias, MarketSnapshot
)

class IntermarketAnalyzer:
    """
    Analyzes intermarket correlations:
    - Correlation with BTC, ETH, major indices
    - Lead/lag relationships
    - Risk-on/risk-off sentiment
    - Sector rotation
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.correlation_period = 50
        
    def analyze(self, market_snapshot: MarketSnapshot) -> AnalysisResult:
        """
        Main analysis entry point
        
        Args:
            market_snapshot: Standardized market data
            
        Returns:
            AnalysisResult with intermarket analysis
        """
        start_time = datetime.now()
        
        df = market_snapshot.ohlcv_df
        correlated_data = market_snapshot.correlated_symbols or {}
        
        # Calculate correlations
        correlations = self._calculate_correlations(df, correlated_data)
        
        # Detect divergences
        divergences = self._detect_divergences(df, correlated_data)
        
        # Analyze market sentiment
        sentiment = self._analyze_market_sentiment(correlations, divergences)
        
        # Identify leading indicators
        leaders = self._identify_leading_indicators(df, correlated_data)
        
        # Determine bias
        bias, confidence = self._determine_bias(correlations, sentiment, divergences)
        
        # Calculate score
        score = self._calculate_score(correlations, sentiment, divergences)
        
        # Check veto conditions
        veto_flags = self._check_veto_conditions(divergences, sentiment)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return AnalysisResult(
            analyzer_type=AnalyzerType.INTERMARKET,
            timestamp=market_snapshot.timestamp,
            score=score,
            bias=bias,
            confidence=confidence,
            signals=[
                {
                    'type': 'intermarket',
                    'sentiment': sentiment['overall'],
                    'correlations': correlations
                }
            ],
            key_levels=[],
            metrics={
                'correlations': correlations,
                'divergences': divergences,
                'sentiment': sentiment,
                'leaders': leaders
            },
            veto_flags=veto_flags,
            processing_time_ms=processing_time
        )
    
    def _calculate_correlations(self, df: pd.DataFrame, 
                                correlated_data: Dict[str, pd.DataFrame]) -> Dict:
        """Calculate price correlations with related symbols"""
        correlations = {}
        
        if not correlated_data or len(df) < self.correlation_period:
            return correlations
        
        # Get returns for target symbol
        df_returns = df['close'].pct_change().dropna()
        
        for symbol, corr_df in correlated_data.items():
            if len(corr_df) < self.correlation_period:
                continue
            
            try:
                # Align data
                aligned_df = corr_df.tail(len(df))
                corr_returns = aligned_df['close'].pct_change().dropna()
                
                # Calculate correlation
                min_len = min(len(df_returns), len(corr_returns))
                if min_len < 10:
                    continue
                
                corr = df_returns.tail(min_len).corr(corr_returns.tail(min_len))
                
                # Calculate recent trend
                recent_trend = 'bullish' if corr_returns.tail(10).mean() > 0 else 'bearish'
                
                correlations[symbol] = {
                    'correlation': float(corr),
                    'strength': abs(corr),
                    'type': 'positive' if corr > 0.3 else ('negative' if corr < -0.3 else 'weak'),
                    'recent_trend': recent_trend
                }
                
            except Exception as e:
                self.logger.warning(f"Error calculating correlation for {symbol}: {e}")
        
        return correlations
    
    def _detect_divergences(self, df: pd.DataFrame, 
                           correlated_data: Dict[str, pd.DataFrame]) -> Dict:
        """Detect divergences between correlated markets"""
        divergences = []
        
        if not correlated_data or len(df) < 20:
            return {'detected': [], 'count': 0}
        
        # Calculate trend for main symbol
        main_trend = 'bullish' if df['close'].iloc[-1] > df['close'].iloc[-20] else 'bearish'
        
        for symbol, corr_df in correlated_data.items():
            if len(corr_df) < 20:
                continue
            
            try:
                # Calculate trend for correlated symbol
                corr_trend = 'bullish' if corr_df['close'].iloc[-1] > corr_df['close'].iloc[-20] else 'bearish'
                
                # Check for divergence (opposite trends)
                if main_trend != corr_trend:
                    divergences.append({
                        'symbol': symbol,
                        'main_trend': main_trend,
                        'corr_trend': corr_trend,
                        'severity': 'moderate'
                    })
                    
            except Exception as e:
                self.logger.warning(f"Error detecting divergence for {symbol}: {e}")
        
        return {
            'detected': divergences,
            'count': len(divergences)
        }
    
    def _analyze_market_sentiment(self, correlations: Dict, divergences: Dict) -> Dict:
        """Analyze overall market sentiment from intermarket data"""
        if not correlations:
            return {
                'overall': 'neutral',
                'risk_appetite': 'moderate',
                'confidence': 0
            }
        
        # Count bullish vs bearish trends
        bullish_count = sum(1 for c in correlations.values() if c['recent_trend'] == 'bullish')
        bearish_count = sum(1 for c in correlations.values() if c['recent_trend'] == 'bearish')
        total = bullish_count + bearish_count
        
        if total == 0:
            return {
                'overall': 'neutral',
                'risk_appetite': 'moderate',
                'confidence': 0
            }
        
        bullish_ratio = bullish_count / total
        
        if bullish_ratio > 0.65:
            overall = 'risk_on'
            risk_appetite = 'high'
        elif bullish_ratio < 0.35:
            overall = 'risk_off'
            risk_appetite = 'low'
        else:
            overall = 'neutral'
            risk_appetite = 'moderate'
        
        # Factor in divergences (reduce confidence)
        confidence = bullish_ratio * 100 if overall == 'risk_on' else (1 - bullish_ratio) * 100
        
        if divergences['count'] > 0:
            confidence *= 0.8  # Reduce confidence by 20% for each divergence
        
        return {
            'overall': overall,
            'risk_appetite': risk_appetite,
            'confidence': float(confidence),
            'bullish_count': bullish_count,
            'bearish_count': bearish_count
        }
    
    def _identify_leading_indicators(self, df: pd.DataFrame, 
                                    correlated_data: Dict[str, pd.DataFrame]) -> Dict:
        """Identify which correlated markets lead (change before target symbol)"""
        leaders = {}
        
        if not correlated_data or len(df) < 20:
            return leaders
        
        # Simple lead detection: which symbols changed direction first
        df_direction = 'bullish' if df['close'].iloc[-1] > df['close'].iloc[-5] else 'bearish'
        
        for symbol, corr_df in correlated_data.items():
            if len(corr_df) < 20:
                continue
            
            try:
                # Check if correlated symbol changed direction earlier
                corr_direction = 'bullish' if corr_df['close'].iloc[-6] > corr_df['close'].iloc[-10] else 'bearish'
                
                if corr_direction == df_direction:
                    leaders[symbol] = {
                        'is_leader': True,
                        'direction': corr_direction,
                        'reliability': 70  # Placeholder
                    }
                    
            except Exception as e:
                self.logger.warning(f"Error identifying leader for {symbol}: {e}")
        
        return leaders
    
    def _determine_bias(self, correlations: Dict, sentiment: Dict, 
                       divergences: Dict) -> tuple:
        """Determine bias from intermarket analysis"""
        overall_sentiment = sentiment['overall']
        
        if overall_sentiment == 'risk_on':
            bias = MarketBias.BULLISH
            confidence = sentiment['confidence']
        elif overall_sentiment == 'risk_off':
            bias = MarketBias.BEARISH
            confidence = sentiment['confidence']
        else:
            bias = MarketBias.NEUTRAL
            confidence = 50
        
        # Reduce confidence if there are divergences
        if divergences['count'] > 2:
            confidence *= 0.7
        
        return bias, min(confidence, 100)
    
    def _calculate_score(self, correlations: Dict, 
                        sentiment: Dict, divergences: Dict) -> float:
        """Calculate overall intermarket score"""
        score = 50.0
        
        # Strong correlations
        if correlations:
            avg_correlation = np.mean([abs(c['correlation']) for c in correlations.values()])
            score += avg_correlation * 20
        
        # Clear sentiment
        if sentiment['overall'] != 'neutral':
            score += 15
        
        # Few divergences
        if divergences['count'] == 0:
            score += 15
        elif divergences['count'] > 2:
            score -= 10
        
        return min(max(score, 0), 100)
    
    def _check_veto_conditions(self, divergences: Dict, sentiment: Dict) -> List[str]:
        """Check for veto conditions"""
        veto_flags = []
        
        # Multiple divergences
        if divergences['count'] >= 3:
            veto_flags.append(f"Multiple intermarket divergences detected ({divergences['count']})")
        
        # Unclear sentiment with low confidence
        if sentiment['overall'] == 'neutral' and sentiment['confidence'] < 40:
            veto_flags.append("Unclear market sentiment")
        
        return veto_flags
