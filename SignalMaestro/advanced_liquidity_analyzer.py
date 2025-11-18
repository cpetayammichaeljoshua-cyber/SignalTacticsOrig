#!/usr/bin/env python3
"""
Advanced Liquidity Analysis Module
Detects liquidity grabs, sweeps, and manipulation patterns
POV: Liquidity grab/swept detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

from SignalMaestro.market_data_contracts import (
    AnalysisResult, AnalyzerType, MarketBias, MarketSnapshot
)

@dataclass
class LiquidityZone:
    """Represents a liquidity zone"""
    price: float
    volume: float
    zone_type: str  # 'high', 'low', 'support', 'resistance'
    timestamp: datetime
    strength: float  # 0-100
    swept: bool = False
    grabbed: bool = False

@dataclass
class LiquidityEvent:
    """Represents a liquidity grab/sweep event"""
    event_type: str  # 'grab', 'sweep', 'accumulation', 'distribution'
    price: float
    volume: float
    timestamp: datetime
    confidence: float  # 0-100
    direction: str  # 'bullish', 'bearish'
    target_zone: LiquidityZone

class AdvancedLiquidityAnalyzer:
    """
    Advanced liquidity analysis for detecting:
    - Liquidity grabs (stop hunts)
    - Liquidity sweeps
    - Accumulation/distribution zones
    - Smart money flow
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.liquidity_zones: List[LiquidityZone] = []
        self.liquidity_events: List[LiquidityEvent] = []
        
        # Configuration
        self.lookback_periods = 100
        self.volume_threshold = 1.5  # 1.5x average volume
        self.wick_ratio = 0.6  # Wick must be 60% of candle
        self.sweep_confirmation_bars = 3
        
    def analyze(self, market_snapshot: MarketSnapshot) -> AnalysisResult:
        """
        Main analysis entry point (standardized interface)
        
        Args:
            market_snapshot: Standardized market data
            
        Returns:
            AnalysisResult with liquidity analysis
        """
        start_time = datetime.now()
        
        df = market_snapshot.ohlcv_df
        liquidity_data = self.analyze_liquidity(df)
        
        # Determine bias
        bias_str = liquidity_data.get('bias', 'neutral')
        if bias_str == 'bullish':
            bias = MarketBias.BULLISH
        elif bias_str == 'bearish':
            bias = MarketBias.BEARISH
        else:
            bias = MarketBias.NEUTRAL
        
        # Calculate confidence
        confidence = liquidity_data.get('liquidity_score', 50)
        
        # Generate signals
        signals = []
        if liquidity_data.get('recent_events'):
            signals = [
                {
                    'type': 'liquidity_event',
                    'events': liquidity_data['recent_events']
                }
            ]
        
        # Key levels
        key_levels = []
        for level in liquidity_data.get('high_liquidity_levels', []):
            key_levels.append({
                'price': level['price'],
                'type': level['type'],
                'strength': level['strength']
            })
        
        # Check veto conditions
        veto_flags = []
        if liquidity_data.get('liquidity_score', 0) < 40:
            veto_flags.append("Low liquidity score")
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return AnalysisResult(
            analyzer_type=AnalyzerType.LIQUIDITY,
            timestamp=market_snapshot.timestamp,
            score=liquidity_data.get('liquidity_score', 50),
            bias=bias,
            confidence=confidence,
            signals=signals,
            key_levels=key_levels,
            metrics=liquidity_data,
            veto_flags=veto_flags,
            processing_time_ms=processing_time
        )
        
    def analyze_liquidity(self, df: pd.DataFrame) -> Dict:
        """
        Complete liquidity analysis
        
        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume
            
        Returns:
            Dictionary with liquidity analysis results
        """
        if len(df) < self.lookback_periods:
            return {
                'liquidity_zones': [],
                'recent_events': [],
                'liquidity_score': 50,
                'bias': 'neutral'
            }
        
        # Identify liquidity zones
        self.liquidity_zones = self._identify_liquidity_zones(df)
        
        # Detect liquidity grabs
        grab_events = self._detect_liquidity_grabs(df)
        
        # Detect liquidity sweeps
        sweep_events = self._detect_liquidity_sweeps(df)
        
        # Analyze smart money flow
        smart_money_flow = self._analyze_smart_money_flow(df)
        
        # Combine all events
        self.liquidity_events = grab_events + sweep_events
        
        # Calculate overall liquidity score
        liquidity_score = self._calculate_liquidity_score(df)
        
        # Determine market bias
        bias = self._determine_liquidity_bias()
        
        return {
            'liquidity_zones': [self._zone_to_dict(z) for z in self.liquidity_zones[-10:]],
            'recent_events': [self._event_to_dict(e) for e in self.liquidity_events[-5:]],
            'liquidity_score': liquidity_score,
            'bias': bias,
            'smart_money_flow': smart_money_flow,
            'high_liquidity_levels': self._get_key_liquidity_levels(),
            'grab_signals': len(grab_events),
            'sweep_signals': len(sweep_events)
        }
    
    def _identify_liquidity_zones(self, df: pd.DataFrame) -> List[LiquidityZone]:
        """Identify key liquidity zones where stops likely cluster"""
        zones = []
        
        # Find swing highs and lows (liquidity pools)
        df['swing_high'] = df['high'].rolling(window=20, center=True).apply(
            lambda x: 1 if x[10] == max(x) else 0, raw=True
        )
        df['swing_low'] = df['low'].rolling(window=20, center=True).apply(
            lambda x: 1 if x[10] == min(x) else 0, raw=True
        )
        
        # Identify high volume nodes
        volume_mean = df['volume'].rolling(window=50).mean()
        high_volume_bars = df[df['volume'] > volume_mean * self.volume_threshold]
        
        # Create liquidity zones from swing points
        for idx, row in df.iterrows():
            if pd.notna(row.get('swing_high', 0)) and row.get('swing_high', 0) == 1:
                volume_at_level = df.loc[idx, 'volume']
                zones.append(LiquidityZone(
                    price=row['high'],
                    volume=volume_at_level,
                    zone_type='resistance',
                    timestamp=row.get('timestamp', idx),
                    strength=self._calculate_zone_strength(df, idx, 'high')
                ))
            
            if pd.notna(row.get('swing_low', 0)) and row.get('swing_low', 0) == 1:
                volume_at_level = df.loc[idx, 'volume']
                zones.append(LiquidityZone(
                    price=row['low'],
                    volume=volume_at_level,
                    zone_type='support',
                    timestamp=row.get('timestamp', idx),
                    strength=self._calculate_zone_strength(df, idx, 'low')
                ))
        
        return sorted(zones, key=lambda x: x.timestamp, reverse=True)
    
    def _calculate_zone_strength(self, df: pd.DataFrame, idx, level_type: str) -> float:
        """Calculate strength of a liquidity zone (0-100)"""
        strength = 50.0  # Base strength
        
        try:
            # Factor 1: Volume at the level
            if idx in df.index:
                volume = df.loc[idx, 'volume']
                avg_volume = df['volume'].rolling(window=50).mean().loc[idx]
                if avg_volume > 0:
                    volume_ratio = volume / avg_volume
                    strength += min(volume_ratio * 10, 30)
            
            # Factor 2: Number of touches (multiple tests = stronger level)
            price = df.loc[idx, level_type]
            price_tolerance = price * 0.001  # 0.1% tolerance
            
            if level_type == 'high':
                touches = len(df[(df['high'] >= price - price_tolerance) & 
                                (df['high'] <= price + price_tolerance)])
            else:
                touches = len(df[(df['low'] >= price - price_tolerance) & 
                                (df['low'] <= price + price_tolerance)])
            
            strength += min(touches * 5, 20)
            
        except Exception as e:
            self.logger.warning(f"Error calculating zone strength: {e}")
        
        return min(strength, 100.0)
    
    def _detect_liquidity_grabs(self, df: pd.DataFrame) -> List[LiquidityEvent]:
        """
        Detect liquidity grabs (stop hunts)
        Characteristics:
        - Price spikes above/below key level
        - Long wick with small body
        - Quick reversal
        - Often accompanied by volume spike
        """
        events = []
        
        for i in range(20, len(df)):
            current = df.iloc[i]
            previous_high = df.iloc[i-20:i]['high'].max()
            previous_low = df.iloc[i-20:i]['low'].min()
            
            body_size = abs(current['close'] - current['open'])
            full_range = current['high'] - current['low']
            
            if full_range == 0:
                continue
            
            upper_wick = current['high'] - max(current['open'], current['close'])
            lower_wick = min(current['open'], current['close']) - current['low']
            
            # Bullish liquidity grab (sweep below then reverse up)
            if (current['low'] < previous_low and
                current['close'] > current['open'] and
                lower_wick / full_range > self.wick_ratio):
                
                confidence = self._calculate_grab_confidence(df, i, 'bullish')
                
                # Find the zone that was grabbed
                target_zone = self._find_nearest_zone(current['low'], 'support')
                
                events.append(LiquidityEvent(
                    event_type='grab',
                    price=current['low'],
                    volume=current['volume'],
                    timestamp=current.get('timestamp', i),
                    confidence=confidence,
                    direction='bullish',
                    target_zone=target_zone
                ))
            
            # Bearish liquidity grab (sweep above then reverse down)
            if (current['high'] > previous_high and
                current['close'] < current['open'] and
                upper_wick / full_range > self.wick_ratio):
                
                confidence = self._calculate_grab_confidence(df, i, 'bearish')
                
                target_zone = self._find_nearest_zone(current['high'], 'resistance')
                
                events.append(LiquidityEvent(
                    event_type='grab',
                    price=current['high'],
                    volume=current['volume'],
                    timestamp=current.get('timestamp', i),
                    confidence=confidence,
                    direction='bearish',
                    target_zone=target_zone
                ))
        
        return events
    
    def _detect_liquidity_sweeps(self, df: pd.DataFrame) -> List[LiquidityEvent]:
        """
        Detect liquidity sweeps
        Characteristics:
        - Price moves through multiple liquidity zones
        - Strong directional move
        - High volume
        - Sustained momentum
        """
        events = []
        
        for i in range(30, len(df)):
            window = df.iloc[i-5:i+1]
            
            # Bullish sweep (taking out multiple lows)
            lows_broken = 0
            for j in range(len(window)-1):
                if window.iloc[j+1]['low'] < window.iloc[:j+1]['low'].min():
                    lows_broken += 1
            
            if lows_broken >= 2 and window.iloc[-1]['close'] > window.iloc[0]['close']:
                confidence = min(lows_broken * 25 + 25, 100)
                
                events.append(LiquidityEvent(
                    event_type='sweep',
                    price=window.iloc[-1]['low'],
                    volume=window.iloc[-1]['volume'],
                    timestamp=window.iloc[-1].get('timestamp', i),
                    confidence=confidence,
                    direction='bullish',
                    target_zone=self._find_nearest_zone(window.iloc[-1]['low'], 'support')
                ))
            
            # Bearish sweep (taking out multiple highs)
            highs_broken = 0
            for j in range(len(window)-1):
                if window.iloc[j+1]['high'] > window.iloc[:j+1]['high'].max():
                    highs_broken += 1
            
            if highs_broken >= 2 and window.iloc[-1]['close'] < window.iloc[0]['close']:
                confidence = min(highs_broken * 25 + 25, 100)
                
                events.append(LiquidityEvent(
                    event_type='sweep',
                    price=window.iloc[-1]['high'],
                    volume=window.iloc[-1]['volume'],
                    timestamp=window.iloc[-1].get('timestamp', i),
                    confidence=confidence,
                    direction='bearish',
                    target_zone=self._find_nearest_zone(window.iloc[-1]['high'], 'resistance')
                ))
        
        return events
    
    def _analyze_smart_money_flow(self, df: pd.DataFrame) -> Dict:
        """Analyze smart money flow patterns"""
        if len(df) < 20:
            return {'flow': 'neutral', 'strength': 0}
        
        recent = df.tail(20)
        
        # Calculate money flow
        typical_price = (recent['high'] + recent['low'] + recent['close']) / 3
        money_flow = typical_price * recent['volume']
        
        positive_flow = money_flow[recent['close'] > recent['open']].sum()
        negative_flow = money_flow[recent['close'] < recent['open']].sum()
        
        total_flow = positive_flow + negative_flow
        if total_flow == 0:
            return {'flow': 'neutral', 'strength': 0}
        
        flow_ratio = (positive_flow - negative_flow) / total_flow
        
        if flow_ratio > 0.3:
            flow = 'bullish'
        elif flow_ratio < -0.3:
            flow = 'bearish'
        else:
            flow = 'neutral'
        
        strength = abs(flow_ratio) * 100
        
        return {
            'flow': flow,
            'strength': strength,
            'positive_flow': float(positive_flow),
            'negative_flow': float(negative_flow)
        }
    
    def _calculate_grab_confidence(self, df: pd.DataFrame, idx: int, direction: str) -> float:
        """Calculate confidence level for liquidity grab"""
        confidence = 50.0
        
        try:
            current = df.iloc[idx]
            
            # Volume confirmation
            avg_volume = df.iloc[max(0, idx-20):idx]['volume'].mean()
            if current['volume'] > avg_volume * 1.5:
                confidence += 20
            
            # Reversal strength
            if direction == 'bullish':
                reversal = (current['close'] - current['low']) / (current['high'] - current['low'])
                confidence += reversal * 30
            else:
                reversal = (current['high'] - current['close']) / (current['high'] - current['low'])
                confidence += reversal * 30
            
        except Exception as e:
            self.logger.warning(f"Error calculating grab confidence: {e}")
        
        return min(confidence, 100.0)
    
    def _calculate_liquidity_score(self, df: pd.DataFrame) -> float:
        """Calculate overall liquidity health score (0-100)"""
        score = 50.0
        
        try:
            recent_df = df.tail(50)
            
            # Factor 1: Volume consistency
            volume_std = recent_df['volume'].std()
            volume_mean = recent_df['volume'].mean()
            if volume_mean > 0:
                cv = volume_std / volume_mean
                score += (1 - min(cv, 1)) * 20
            
            # Factor 2: Spread tightness
            spreads = (recent_df['high'] - recent_df['low']) / recent_df['close']
            avg_spread = spreads.mean()
            score += (1 - min(avg_spread * 10, 1)) * 15
            
            # Factor 3: Recent liquidity events
            # Just use the most recent events (last 10)
            recent_events = self.liquidity_events[-10:] if self.liquidity_events else []
            score += min(len(recent_events) * 5, 15)
            
        except Exception as e:
            self.logger.warning(f"Error calculating liquidity score: {e}")
        
        return min(max(score, 0), 100)
    
    def _determine_liquidity_bias(self) -> str:
        """Determine overall liquidity bias"""
        if not self.liquidity_events:
            return 'neutral'
        
        recent_events = self.liquidity_events[-10:]
        
        bullish = sum(1 for e in recent_events if e.direction == 'bullish')
        bearish = sum(1 for e in recent_events if e.direction == 'bearish')
        
        if bullish > bearish * 1.5:
            return 'bullish'
        elif bearish > bullish * 1.5:
            return 'bearish'
        else:
            return 'neutral'
    
    def _find_nearest_zone(self, price: float, zone_type: str) -> Optional[LiquidityZone]:
        """Find nearest liquidity zone to given price"""
        matching_zones = [z for z in self.liquidity_zones if z.zone_type == zone_type]
        
        if not matching_zones:
            return None
        
        return min(matching_zones, key=lambda z: abs(z.price - price))
    
    def _get_key_liquidity_levels(self) -> List[Dict]:
        """Get key liquidity levels for trading decisions"""
        if not self.liquidity_zones:
            return []
        
        # Sort by strength and get top 5
        sorted_zones = sorted(self.liquidity_zones, key=lambda z: z.strength, reverse=True)
        
        return [
            {
                'price': z.price,
                'type': z.zone_type,
                'strength': z.strength
            }
            for z in sorted_zones[:5]
        ]
    
    def _zone_to_dict(self, zone: LiquidityZone) -> Dict:
        """Convert LiquidityZone to dictionary"""
        return {
            'price': float(zone.price),
            'volume': float(zone.volume),
            'type': zone.zone_type,
            'strength': float(zone.strength),
            'swept': zone.swept,
            'grabbed': zone.grabbed
        }
    
    def _event_to_dict(self, event: LiquidityEvent) -> Dict:
        """Convert LiquidityEvent to dictionary"""
        return {
            'type': event.event_type,
            'price': float(event.price),
            'volume': float(event.volume),
            'confidence': float(event.confidence),
            'direction': event.direction
        }
    
    def get_trading_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Generate trading signal based on liquidity analysis
        
        Returns:
            Signal dict or None
        """
        analysis = self.analyze_liquidity(df)
        
        if analysis['liquidity_score'] < 60:
            return None
        
        recent_events = self.liquidity_events[-3:] if self.liquidity_events else []
        
        # Strong bullish setup: liquidity grab followed by sweep
        bullish_grabs = [e for e in recent_events if e.event_type == 'grab' and e.direction == 'bullish']
        if bullish_grabs and analysis['bias'] == 'bullish':
            latest_grab = bullish_grabs[-1]
            return {
                'direction': 'LONG',
                'reason': f'Bullish liquidity grab at {latest_grab.price:.4f}',
                'confidence': latest_grab.confidence,
                'entry_zone': latest_grab.price * 1.002,  # Slightly above grab
                'stop_loss': latest_grab.price * 0.998,
                'targets': [
                    latest_grab.price * 1.01,
                    latest_grab.price * 1.02,
                    latest_grab.price * 1.03
                ]
            }
        
        # Strong bearish setup
        bearish_grabs = [e for e in recent_events if e.event_type == 'grab' and e.direction == 'bearish']
        if bearish_grabs and analysis['bias'] == 'bearish':
            latest_grab = bearish_grabs[-1]
            return {
                'direction': 'SHORT',
                'reason': f'Bearish liquidity grab at {latest_grab.price:.4f}',
                'confidence': latest_grab.confidence,
                'entry_zone': latest_grab.price * 0.998,  # Slightly below grab
                'stop_loss': latest_grab.price * 1.002,
                'targets': [
                    latest_grab.price * 0.99,
                    latest_grab.price * 0.98,
                    latest_grab.price * 0.97
                ]
            }
        
        return None
