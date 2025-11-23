#!/usr/bin/env python3
"""
Order Flow Trading Analysis Module
Analyzes order flow, volume profile, cumulative delta, and smart money activity
Integrates with market intelligence for enhanced trading signals
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class OrderFlowMetrics:
    """Order flow analysis metrics"""
    cumulative_delta: float
    delta_strength: float  # 0-1 normalized
    buy_volume: float
    sell_volume: float
    buy_sell_ratio: float
    volume_profile_score: float  # 0-1
    smart_money_activity: float  # 0-1
    order_imbalance: float  # -1 to 1
    liquidity_levels: Dict[str, float]
    support_demand: float  # 0-1
    resistance_supply: float  # 0-1
    timestamp: datetime

class OrderFlowAnalyzer:
    """Advanced order flow analysis for high-frequency trading"""
    
    def __init__(self):
        self.logger = logger
        self.volume_profiles = {}
        
    async def analyze_order_flow(self, market_data: pd.DataFrame, 
                                  symbol: str = "FXSUSDT") -> OrderFlowMetrics:
        """
        Comprehensive order flow analysis
        
        Args:
            market_data: OHLCV data with volume information
            symbol: Trading symbol
            
        Returns:
            OrderFlowMetrics with complete order flow analysis
        """
        try:
            if len(market_data) < 20:
                return self._default_metrics()
            
            # Calculate basic order flow metrics
            cumulative_delta = self._calculate_cumulative_delta(market_data)
            delta_strength = self._normalize_metric(abs(cumulative_delta), 0, 100000)
            
            # Volume analysis
            buy_vol, sell_vol = self._calculate_buy_sell_volume(market_data)
            buy_sell_ratio = buy_vol / sell_vol if sell_vol > 0 else 1.0
            
            # Volume profile
            vol_profile_score = self._analyze_volume_profile(market_data)
            
            # Smart money detection
            smart_money_score = self._detect_smart_money_activity(market_data, buy_vol, sell_vol)
            
            # Order imbalance
            order_imbalance = (buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-6)
            
            # Support/Resistance demand
            support_demand = self._analyze_support_demand(market_data, buy_vol, sell_vol)
            resistance_supply = self._analyze_resistance_supply(market_data, buy_vol, sell_vol)
            
            # Liquidity levels
            liquidity = self._identify_liquidity_levels(market_data)
            
            return OrderFlowMetrics(
                cumulative_delta=float(cumulative_delta),
                delta_strength=float(delta_strength),
                buy_volume=float(buy_vol),
                sell_volume=float(sell_vol),
                buy_sell_ratio=float(buy_sell_ratio),
                volume_profile_score=float(vol_profile_score),
                smart_money_activity=float(smart_money_score),
                order_imbalance=float(order_imbalance),
                liquidity_levels=liquidity,
                support_demand=float(support_demand),
                resistance_supply=float(resistance_supply),
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Order flow analysis error: {e}")
            return self._default_metrics()
    
    def _calculate_cumulative_delta(self, market_data: pd.DataFrame) -> float:
        """Calculate cumulative delta (buying pressure - selling pressure)"""
        try:
            close = market_data['close'].values
            volume = market_data['volume'].values
            
            # Simplified delta: if close > open, volume is buying, else selling
            open_prices = market_data['open'].values if 'open' in market_data else close[:-1]
            
            delta = 0.0
            for i in range(len(close)):
                close_price = close[i]
                open_price = open_prices[i] if i < len(open_prices) else close[i-1]
                vol = volume[i]
                
                if close_price > open_price:
                    delta += vol  # Buying volume
                elif close_price < open_price:
                    delta -= vol  # Selling volume
            
            return delta
        except Exception as e:
            self.logger.debug(f"Delta calculation error: {e}")
            return 0.0
    
    def _calculate_buy_sell_volume(self, market_data: pd.DataFrame) -> Tuple[float, float]:
        """Estimate buy and sell volume from price action"""
        try:
            close = market_data['close'].values
            volume = market_data['volume'].values
            
            buy_volume = 0.0
            sell_volume = 0.0
            
            # If available, use open for comparison; otherwise use lag
            if 'open' in market_data.columns:
                open_prices = market_data['open'].values
            else:
                open_prices = np.concatenate(([close[0:1]], close[:-1]))
            
            for i in range(len(close)):
                if close[i] > open_prices[i]:
                    buy_volume += volume[i]
                else:
                    sell_volume += volume[i]
            
            return buy_volume, sell_volume
        except Exception:
            return 0.0, 0.0
    
    def _analyze_volume_profile(self, market_data: pd.DataFrame) -> float:
        """Analyze volume profile at different price levels (0-1 score)"""
        try:
            if len(market_data) < 10:
                return 0.5
            
            close = market_data['close'].values
            volume = market_data['volume'].values
            
            # Create price bins
            price_min = float(np.amin(close))
            price_max = float(np.amax(close))
            if price_max == price_min:
                return 0.5
            
            bins = 10
            bin_edges = np.linspace(price_min, price_max, bins + 1)
            vol_profile = np.zeros(bins)
            
            for price, vol in zip(close, volume):
                bin_idx = min(int((price - price_min) / (price_max - price_min) * bins), bins - 1)
                vol_profile[bin_idx] += vol
            
            # POC (Point of Control) - highest volume
            poc_idx = np.argmax(vol_profile)
            poc_price = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2
            current_price = close[-1]
            
            # Score based on proximity to POC
            distance_to_poc = abs(current_price - poc_price) / (price_max - price_min)
            profile_score = 1.0 - min(distance_to_poc, 1.0)
            
            return float(profile_score)
        except Exception:
            return 0.5
    
    def _detect_smart_money_activity(self, market_data: pd.DataFrame, 
                                      buy_vol: float, sell_vol: float) -> float:
        """Detect smart money activity patterns"""
        try:
            close = market_data['close'].values
            volume = market_data['volume'].values
            
            # Check for volume accumulation patterns
            recent_vol = float(np.mean(volume[-5:]))
            historical_vol = float(np.mean(volume[:-5]))
            
            vol_ratio = recent_vol / (historical_vol + 1e-6)
            
            # Large volume with small price change = accumulation
            price_change = abs(close[-1] - close[-10]) / (close[-10] + 1e-6)
            
            if vol_ratio > 1.5 and price_change < 0.02:
                # Accumulation detected
                smart_score = 0.8
            elif vol_ratio > 2.0:
                # Unusual activity
                smart_score = 0.7
            else:
                smart_score = 0.5
            
            # Adjust by buy/sell ratio
            buy_sell_imbalance = abs(buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-6)
            smart_score *= (0.5 + buy_sell_imbalance)
            
            return min(float(smart_score), 1.0)
        except Exception:
            return 0.5
    
    def _analyze_support_demand(self, market_data: pd.DataFrame, 
                                buy_vol: float, sell_vol: float) -> float:
        """Analyze support level demand"""
        try:
            close = market_data['close'].values
            volume = market_data['volume'].values
            
            # Find support (local minimum)
            support = float(np.amin(close[-20:]))
            current = close[-1]
            
            distance_to_support = current - support
            total_range = float(np.amax(close[-20:])) - support
            
            if total_range == 0:
                return 0.5
            
            proximity = 1.0 - (distance_to_support / total_range)
            
            # Factor in buy volume
            vol_factor = min(buy_vol / (sell_vol + buy_vol + 1e-6), 1.0)
            
            demand_score = (proximity * 0.6) + (vol_factor * 0.4)
            return float(demand_score)
        except Exception:
            return 0.5
    
    def _analyze_resistance_supply(self, market_data: pd.DataFrame,
                                   buy_vol: float, sell_vol: float) -> float:
        """Analyze resistance level supply"""
        try:
            close = market_data['close'].values
            volume = market_data['volume'].values
            
            # Find resistance (local maximum)
            resistance = float(np.amax(close[-20:]))
            current = close[-1]
            
            distance_to_resistance = resistance - current
            total_range = resistance - float(np.amin(close[-20:]))
            
            if total_range == 0:
                return 0.5
            
            proximity = 1.0 - (distance_to_resistance / total_range)
            
            # Factor in sell volume
            vol_factor = min(sell_vol / (sell_vol + buy_vol + 1e-6), 1.0)
            
            supply_score = (proximity * 0.6) + (vol_factor * 0.4)
            return float(supply_score)
        except Exception:
            return 0.5
    
    def _identify_liquidity_levels(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Identify key liquidity levels (support/resistance)"""
        try:
            close = market_data['close'].values
            volume = market_data['volume'].values
            
            recent_data = close[-50:]
            
            support = float(np.amin(recent_data))
            resistance = float(np.amax(recent_data))
            
            mid_point = (support + resistance) / 2
            
            return {
                'support': support,
                'resistance': resistance,
                'midpoint': mid_point,
                'current': float(close[-1])
            }
        except Exception:
            return {}
    
    def _normalize_metric(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize metric to 0-1 range"""
        if max_val == min_val:
            return 0.5
        normalized = (value - min_val) / (max_val - min_val)
        return min(max(float(normalized), 0.0), 1.0)
    
    def _default_metrics(self) -> OrderFlowMetrics:
        """Return default metrics"""
        return OrderFlowMetrics(
            cumulative_delta=0.0,
            delta_strength=0.5,
            buy_volume=0.0,
            sell_volume=0.0,
            buy_sell_ratio=1.0,
            volume_profile_score=0.5,
            smart_money_activity=0.5,
            order_imbalance=0.0,
            liquidity_levels={},
            support_demand=0.5,
            resistance_supply=0.5,
            timestamp=datetime.utcnow()
        )
    
    def get_order_flow_signal(self, metrics: OrderFlowMetrics) -> str:
        """Generate trading signal from order flow metrics"""
        try:
            if metrics.order_imbalance > 0.3 and metrics.smart_money_activity > 0.65:
                return "STRONG_BUY"
            elif metrics.order_imbalance > 0.1 and metrics.smart_money_activity > 0.55:
                return "BUY"
            elif metrics.order_imbalance < -0.3 and metrics.smart_money_activity > 0.65:
                return "STRONG_SELL"
            elif metrics.order_imbalance < -0.1 and metrics.smart_money_activity > 0.55:
                return "SELL"
            else:
                return "NEUTRAL"
        except Exception:
            return "NEUTRAL"
