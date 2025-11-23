#!/usr/bin/env python3
"""
Insider Trading Analysis Module
Detects large institutional trades, order flow imbalances, and smart money activity
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class InsiderMetrics:
    """Insider trading metrics"""
    large_buy_volume: float
    large_sell_volume: float
    institutional_activity_score: float  # 0-1
    whale_accumulation: float  # 0-1
    unusual_volume: float  # 0-1
    order_concentration: float  # 0-1
    sentiment_score: float  # -1 to 1 (negative = selling, positive = buying)
    manipulation_risk: float  # 0-1 (pump and dump risk)
    timestamp: datetime

class InsiderTradingAnalyzer:
    """Analyzes insider and institutional trading patterns"""
    
    def __init__(self, whale_threshold_percentile: float = 90.0):
        self.logger = logger
        self.whale_threshold_percentile = whale_threshold_percentile
        
    async def analyze_insider_trading(self, market_data: pd.DataFrame,
                                      symbol: str = "FXSUSDT") -> InsiderMetrics:
        """
        Analyze insider and institutional trading patterns
        
        Args:
            market_data: OHLCV data
            symbol: Trading symbol
            
        Returns:
            InsiderMetrics with institutional activity analysis
        """
        try:
            if len(market_data) < 20:
                return self._default_metrics()
            
            volume = np.asarray(market_data['volume'].values, dtype=float)
            close = np.asarray(market_data['close'].values, dtype=float)
            
            # Identify large volume trades
            large_buy_vol, large_sell_vol = self._detect_large_trades(market_data)
            
            # Institutional activity score
            inst_activity = self._calculate_institutional_activity(volume)
            
            # Whale accumulation detection
            whale_accum = self._detect_whale_accumulation(market_data)
            
            # Unusual volume detection
            unusual_vol = self._detect_unusual_volume(volume)
            
            # Order concentration (how concentrated are large orders)
            order_conc = self._calculate_order_concentration(volume)
            
            # Market sentiment (-1 to 1)
            sentiment = self._calculate_sentiment(close, volume)
            
            # Pump and dump risk detection
            manip_risk = self._detect_manipulation_risk(market_data)
            
            return InsiderMetrics(
                large_buy_volume=float(large_buy_vol),
                large_sell_volume=float(large_sell_vol),
                institutional_activity_score=float(inst_activity),
                whale_accumulation=float(whale_accum),
                unusual_volume=float(unusual_vol),
                order_concentration=float(order_conc),
                sentiment_score=float(sentiment),
                manipulation_risk=float(manip_risk),
                timestamp=datetime.utcnow()
            )
        except Exception as e:
            self.logger.error(f"Insider trading analysis error: {e}")
            return self._default_metrics()
    
    def _detect_large_trades(self, market_data: pd.DataFrame) -> Tuple[float, float]:
        """Detect large buy and sell trades"""
        try:
            close = np.asarray(market_data['close'].values, dtype=float)
            volume = np.asarray(market_data['volume'].values, dtype=float)
            
            # Threshold for large trade
            avg_volume = float(np.mean(volume))
            large_threshold = avg_volume * 2.0
            
            buy_volume = 0.0
            sell_volume = 0.0
            
            for i in range(len(close)):
                if volume[i] > large_threshold:
                    if i > 0 and close[i] > close[i-1]:
                        buy_volume += volume[i]
                    elif i > 0 and close[i] < close[i-1]:
                        sell_volume += volume[i]
            
            return buy_volume, sell_volume
        except Exception:
            return 0.0, 0.0
    
    def _calculate_institutional_activity(self, volume: np.ndarray) -> float:
        """Calculate institutional activity score"""
        try:
            recent_vol = float(np.mean(volume[-5:]))
            historical_vol = float(np.mean(volume[:-5]))
            
            if historical_vol == 0:
                return 0.5
            
            vol_ratio = recent_vol / historical_vol
            
            # High institutional activity = high volume ratio
            if vol_ratio > 3.0:
                return 0.9
            elif vol_ratio > 2.0:
                return 0.7
            elif vol_ratio > 1.5:
                return 0.6
            else:
                return 0.4
        except Exception:
            return 0.5
    
    def _detect_whale_accumulation(self, market_data: pd.DataFrame) -> float:
        """Detect whale (large holder) accumulation patterns"""
        try:
            volume = np.asarray(market_data['volume'].values, dtype=float)
            
            # Calculate volume percentile
            vol_percentile = float(np.percentile(volume, self.whale_threshold_percentile))
            recent_large_trades = int(np.sum(volume[-10:] > vol_percentile))
            
            # More large trades = more whale activity
            whale_score = min(recent_large_trades / 10.0, 1.0)
            return float(whale_score)
        except Exception:
            return 0.5
    
    def _detect_unusual_volume(self, volume: np.ndarray) -> float:
        """Detect unusual volume spikes"""
        try:
            mean_vol = float(np.mean(volume))
            std_vol = float(np.std(volume))
            
            if std_vol == 0:
                return 0.5
            
            recent_vol = volume[-1]
            z_score = abs((recent_vol - mean_vol) / std_vol)
            
            # Z-score > 2 = unusual
            if z_score > 3.0:
                return 0.9
            elif z_score > 2.0:
                return 0.7
            elif z_score > 1.5:
                return 0.5
            else:
                return 0.3
        except Exception:
            return 0.5
    
    def _calculate_order_concentration(self, volume: np.ndarray) -> float:
        """Calculate how concentrated large orders are"""
        try:
            volume = np.asarray(volume, dtype=float)
            avg_vol = float(np.mean(volume))
            
            # Count candles with >2x average volume
            concentrated = int(np.sum(volume > (avg_vol * 2.0)))
            
            concentration_score = min(float(concentrated) / len(volume), 1.0)
            return float(concentration_score)
        except Exception:
            return 0.5
    
    def _calculate_sentiment(self, close: np.ndarray, volume: np.ndarray) -> float:
        """Calculate market sentiment (-1 to 1)"""
        try:
            close = np.asarray(close, dtype=float)
            volume = np.asarray(volume, dtype=float)
            buy_volume = 0.0
            sell_volume = 0.0
            
            for i in range(1, len(close)):
                if close[i] > close[i-1]:
                    buy_volume += float(volume[i])
                else:
                    sell_volume += float(volume[i])
            
            total = buy_volume + sell_volume
            if total == 0:
                return 0.0
            
            sentiment = (buy_volume - sell_volume) / total
            return float(sentiment)
        except Exception:
            return 0.0
    
    def _detect_manipulation_risk(self, market_data: pd.DataFrame) -> float:
        """Detect pump and dump manipulation risk"""
        try:
            close = np.asarray(market_data['close'].values, dtype=float)
            volume = np.asarray(market_data['volume'].values, dtype=float)
            
            # Risk factors:
            # 1. Large volume with small price movement = distribution
            # 2. Price spike with volume spike = potential pump
            
            recent_price_change = abs(float(close[-1]) - float(close[-5])) / (float(close[-5]) + 1e-9)
            recent_vol_avg = float(np.mean(volume[-5:]))
            historical_vol_avg = float(np.mean(volume[:-5]))
            
            vol_ratio = recent_vol_avg / (historical_vol_avg + 1e-6)
            
            # Large volume with small price change = distribution risk
            if vol_ratio > 2.0 and recent_price_change < 0.05:
                return 0.8
            # Price spike with volume = pump risk
            elif recent_price_change > 0.10 and vol_ratio > 1.5:
                return 0.7
            # Moderate risk
            elif vol_ratio > 1.5:
                return 0.4
            else:
                return 0.2
        except Exception:
            return 0.3
    
    def _default_metrics(self) -> InsiderMetrics:
        """Return default metrics"""
        return InsiderMetrics(
            large_buy_volume=0.0,
            large_sell_volume=0.0,
            institutional_activity_score=0.5,
            whale_accumulation=0.5,
            unusual_volume=0.5,
            order_concentration=0.5,
            sentiment_score=0.0,
            manipulation_risk=0.3,
            timestamp=datetime.utcnow()
        )
    
    def get_insider_signal(self, metrics: InsiderMetrics) -> str:
        """Generate signal from insider metrics"""
        try:
            # Combine institutional activity and whale accumulation
            institutional_pressure = (
                metrics.institutional_activity_score * 0.4 +
                metrics.whale_accumulation * 0.3 +
                (1.0 - metrics.manipulation_risk) * 0.3
            )
            
            # Sentiment-based adjustment
            if metrics.sentiment_score > 0.4:
                if institutional_pressure > 0.65:
                    return "INSIDER_BUY"
                elif institutional_pressure > 0.50:
                    return "BUY"
            elif metrics.sentiment_score < -0.4:
                if institutional_pressure > 0.65:
                    return "INSIDER_SELL"
                elif institutional_pressure > 0.50:
                    return "SELL"
            
            return "HOLD"
        except Exception:
            return "HOLD"
