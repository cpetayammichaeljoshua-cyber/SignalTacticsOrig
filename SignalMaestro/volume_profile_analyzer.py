#!/usr/bin/env python3
"""
Volume Profile and Footprint Chart Analyzer
Analyzes volume distribution at price levels
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import logging

from SignalMaestro.market_data_contracts import (
    AnalysisResult, AnalyzerType, MarketBias, MarketSnapshot,
    VolumeProfile, VolumeProfileLevel
)

class VolumeProfileAnalyzer:
    """
    Analyzes volume profile and footprint charts:
    - Point of Control (POC)
    - Value Area (70% of volume)
    - High/Low Volume Nodes
    - Bid/Ask footprint
    """
    
    def __init__(self, num_bins: int = 50):
        self.logger = logging.getLogger(__name__)
        self.num_bins = num_bins
        
    def analyze(self, market_snapshot: MarketSnapshot) -> AnalysisResult:
        """
        Main analysis entry point
        
        Args:
            market_snapshot: Standardized market data
            
        Returns:
            AnalysisResult with volume profile analysis
        """
        start_time = datetime.now()
        
        df = market_snapshot.ohlcv_df
        current_price = market_snapshot.current_price
        
        # Build volume profile
        volume_profile = self._build_volume_profile(df, market_snapshot.symbol)
        
        # Analyze footprint (if tick data available)
        footprint_data = self._analyze_footprint(market_snapshot)
        
        # Determine price position relative to value area
        price_position = self._analyze_price_position(current_price, volume_profile)
        
        # Identify support/resistance from volume
        sr_levels = self._identify_volume_sr_levels(volume_profile)
        
        # Determine bias
        bias, confidence = self._determine_bias(volume_profile, current_price, price_position)
        
        # Calculate score
        score = self._calculate_score(volume_profile, price_position, footprint_data)
        
        # Check veto conditions
        veto_flags = self._check_veto_conditions(price_position, footprint_data)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return AnalysisResult(
            analyzer_type=AnalyzerType.VOLUME_PROFILE,
            timestamp=market_snapshot.timestamp,
            score=score,
            bias=bias,
            confidence=confidence,
            signals=[
                {
                    'type': 'volume_profile',
                    'poc': volume_profile.poc_price,
                    'vah': volume_profile.value_area_high,
                    'val': volume_profile.value_area_low,
                    'position': price_position
                }
            ],
            key_levels=sr_levels,
            metrics={
                'volume_profile': self._volume_profile_to_dict(volume_profile),
                'footprint': footprint_data,
                'price_position': price_position
            },
            veto_flags=veto_flags,
            processing_time_ms=processing_time
        )
    
    def _build_volume_profile(self, df: pd.DataFrame, symbol: str) -> VolumeProfile:
        """Build volume profile from OHLCV data"""
        if len(df) == 0:
            raise ValueError("Empty DataFrame")
        
        # Determine price range
        price_min = df['low'].min()
        price_max = df['high'].max()
        
        # Create price bins
        bins = np.linspace(price_min, price_max, self.num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Distribute volume across price bins
        volume_at_price = np.zeros(self.num_bins)
        
        for idx, row in df.iterrows():
            # Approximate volume distribution within each candle
            candle_range = row['high'] - row['low']
            if candle_range == 0:
                # All volume at close price
                bin_idx = np.digitize([row['close']], bins)[0] - 1
                if 0 <= bin_idx < self.num_bins:
                    volume_at_price[bin_idx] += row['volume']
            else:
                # Distribute volume evenly across candle range
                candle_bins = (bins >= row['low']) & (bins <= row['high'])
                relevant_bins = np.where(candle_bins[:-1] | candle_bins[1:])[0]
                
                if len(relevant_bins) > 0:
                    volume_per_bin = row['volume'] / len(relevant_bins)
                    for bin_idx in relevant_bins:
                        if 0 <= bin_idx < self.num_bins:
                            volume_at_price[bin_idx] += volume_per_bin
        
        total_volume = volume_at_price.sum()
        
        # Find POC (Point of Control - highest volume)
        poc_idx = np.argmax(volume_at_price)
        poc_price = bin_centers[poc_idx]
        
        # Find Value Area (70% of volume around POC)
        value_area_volume = total_volume * 0.70
        
        # Expand from POC to find value area
        cumulative_volume = volume_at_price[poc_idx]
        lower_idx = poc_idx
        upper_idx = poc_idx
        
        while cumulative_volume < value_area_volume:
            # Expand to side with more volume
            if lower_idx > 0 and upper_idx < self.num_bins - 1:
                if volume_at_price[lower_idx - 1] > volume_at_price[upper_idx + 1]:
                    lower_idx -= 1
                    cumulative_volume += volume_at_price[lower_idx]
                else:
                    upper_idx += 1
                    cumulative_volume += volume_at_price[upper_idx]
            elif lower_idx > 0:
                lower_idx -= 1
                cumulative_volume += volume_at_price[lower_idx]
            elif upper_idx < self.num_bins - 1:
                upper_idx += 1
                cumulative_volume += volume_at_price[upper_idx]
            else:
                break
        
        value_area_low = bin_centers[lower_idx]
        value_area_high = bin_centers[upper_idx]
        
        # Create VolumeProfileLevels
        levels = []
        for i, (price, volume) in enumerate(zip(bin_centers, volume_at_price)):
            percentage = (volume / total_volume * 100) if total_volume > 0 else 0
            levels.append(VolumeProfileLevel(
                price=float(price),
                volume=float(volume),
                percentage=float(percentage),
                is_poc=(i == poc_idx),
                is_vah=(i == upper_idx),
                is_val=(i == lower_idx)
            ))
        
        return VolumeProfile(
            symbol=symbol,
            timestamp=datetime.now(),
            timeframe='session',
            levels=levels,
            poc_price=float(poc_price),
            value_area_high=float(value_area_high),
            value_area_low=float(value_area_low),
            total_volume=float(total_volume)
        )
    
    def _analyze_footprint(self, snapshot: MarketSnapshot) -> Dict:
        """
        Analyze footprint chart (bid/ask volume at each price level)
        Requires recent_trades data
        """
        if not snapshot.recent_trades:
            return {
                'available': False,
                'bid_volume': 0,
                'ask_volume': 0,
                'imbalance': 0
            }
        
        bid_volume = 0
        ask_volume = 0
        
        for trade in snapshot.recent_trades:
            if trade.get('is_buyer_maker', False):
                # Seller initiated (hit the bid)
                bid_volume += trade['volume']
            else:
                # Buyer initiated (hit the ask)
                ask_volume += trade['volume']
        
        total = bid_volume + ask_volume
        imbalance = (ask_volume - bid_volume) / total if total > 0 else 0
        
        return {
            'available': True,
            'bid_volume': float(bid_volume),
            'ask_volume': float(ask_volume),
            'imbalance': float(imbalance),
            'interpretation': 'bullish' if imbalance > 0.2 else ('bearish' if imbalance < -0.2 else 'neutral')
        }
    
    def _analyze_price_position(self, current_price: float, volume_profile: VolumeProfile) -> Dict:
        """Analyze current price position relative to volume profile"""
        position_type = volume_profile.get_current_position(current_price)
        
        # Calculate distance from POC
        poc_distance_pct = ((current_price - volume_profile.poc_price) / volume_profile.poc_price) * 100
        
        # Calculate distance from value area
        if current_price > volume_profile.value_area_high:
            va_distance_pct = ((current_price - volume_profile.value_area_high) / current_price) * 100
            va_position = 'above'
        elif current_price < volume_profile.value_area_low:
            va_distance_pct = ((volume_profile.value_area_low - current_price) / current_price) * 100
            va_position = 'below'
        else:
            va_distance_pct = 0
            va_position = 'inside'
        
        return {
            'type': position_type,
            'poc_distance_pct': float(poc_distance_pct),
            'va_position': va_position,
            'va_distance_pct': float(va_distance_pct)
        }
    
    def _identify_volume_sr_levels(self, volume_profile: VolumeProfile) -> List[Dict]:
        """Identify support/resistance levels from volume profile"""
        levels = []
        
        # POC is always a key level
        levels.append({
            'price': volume_profile.poc_price,
            'type': 'poc',
            'strength': 100,
            'volume': max(level.volume for level in volume_profile.levels)
        })
        
        # Value area boundaries
        levels.append({
            'price': volume_profile.value_area_high,
            'type': 'value_area_high',
            'strength': 80,
            'volume': 0
        })
        
        levels.append({
            'price': volume_profile.value_area_low,
            'type': 'value_area_low',
            'strength': 80,
            'volume': 0
        })
        
        # High volume nodes (HVN)
        sorted_levels = sorted(volume_profile.levels, key=lambda x: x.volume, reverse=True)
        for level in sorted_levels[1:4]:  # Top 3 excluding POC
            if level.percentage > 2:  # At least 2% of total volume
                levels.append({
                    'price': level.price,
                    'type': 'high_volume_node',
                    'strength': min(level.percentage * 10, 100),
                    'volume': level.volume
                })
        
        return levels
    
    def _determine_bias(self, volume_profile: VolumeProfile, 
                       current_price: float, price_position: Dict) -> tuple:
        """Determine bias based on volume profile"""
        # Price above value area = bullish
        if price_position['va_position'] == 'above':
            bias = MarketBias.BULLISH
            confidence = min(price_position['va_distance_pct'] * 10 + 60, 90)
        
        # Price below value area = bearish
        elif price_position['va_position'] == 'below':
            bias = MarketBias.BEARISH
            confidence = min(price_position['va_distance_pct'] * 10 + 60, 90)
        
        # Price near POC = neutral
        elif abs(price_position['poc_distance_pct']) < 0.5:
            bias = MarketBias.NEUTRAL
            confidence = 50
        
        # Price above POC but in value area = slightly bullish
        elif price_position['poc_distance_pct'] > 0:
            bias = MarketBias.BULLISH
            confidence = 60
        
        # Price below POC but in value area = slightly bearish
        else:
            bias = MarketBias.BEARISH
            confidence = 60
        
        return bias, confidence
    
    def _calculate_score(self, volume_profile: VolumeProfile, 
                        price_position: Dict, footprint_data: Dict) -> float:
        """Calculate overall volume profile score"""
        score = 50.0
        
        # Well-defined value area = higher score
        va_range = volume_profile.value_area_high - volume_profile.value_area_low
        if va_range > 0:
            va_ratio = va_range / volume_profile.poc_price
            score += min(va_ratio * 1000, 20)  # Up to +20
        
        # Footprint confirmation
        if footprint_data.get('available', False):
            if footprint_data['interpretation'] != 'neutral':
                score += 15
        
        # Clear price position
        if price_position['va_position'] != 'inside':
            score += 15
        
        return min(max(score, 0), 100)
    
    def _check_veto_conditions(self, price_position: Dict, footprint_data: Dict) -> List[str]:
        """Check for veto conditions"""
        veto_flags = []
        
        # No strong veto conditions from volume profile
        # It's more of a confirmation tool
        
        return veto_flags
    
    def _volume_profile_to_dict(self, vp: VolumeProfile) -> Dict:
        """Convert VolumeProfile to dict"""
        return {
            'poc': vp.poc_price,
            'vah': vp.value_area_high,
            'val': vp.value_area_low,
            'total_volume': vp.total_volume,
            'num_levels': len(vp.levels)
        }
