"""
MARKET INTELLIGENCE ENGINE
Integrates sentiment analysis, market prediction, and technical analysis
for enhanced high-frequency trading signals.
"""
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class MarketIntelligenceEngine:
    """
    Enhanced market intelligence for high-frequency trading signals.
    Combines sentiment, prediction, and technical analysis.
    """
    
    def __init__(self):
        self.signal_cache = {}
        self.market_regime = "neutral"
        self.confidence_threshold = 0.65
        logger.info("✅ Market Intelligence Engine initialized")
    
    async def analyze_market(self, symbol: str, market_data: pd.DataFrame, 
                            ai_orchestrator=None) -> Dict[str, Any]:
        """
        Comprehensive market analysis combining multiple signals.
        """
        analysis = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'scores': {},
            'signals': {},
            'recommendation': 'HOLD',
            'confidence': 0.0,
            'reasoning': [],
        }
        
        try:
            # Technical Analysis
            technical_score = self._analyze_technical(market_data)
            analysis['scores']['technical'] = technical_score
            
            # Volume Analysis
            volume_score = self._analyze_volume(market_data)
            analysis['scores']['volume'] = volume_score
            
            # Trend Analysis
            trend_score = self._analyze_trend(market_data)
            analysis['scores']['trend'] = trend_score
            
            # Momentum Analysis
            momentum_score = self._analyze_momentum(market_data)
            analysis['scores']['momentum'] = momentum_score
            
            # AI-Enhanced Analysis (if available)
            if ai_orchestrator:
                try:
                    ai_score = await ai_orchestrator.generate_ai_signal(
                        symbol, market_data
                    )
                    analysis['scores']['ai'] = ai_score
                except Exception as e:
                    logger.debug(f"AI analysis unavailable: {e}")
            
            # Calculate weighted confidence
            scores = [s for s in analysis['scores'].values() if isinstance(s, (int, float))]
            if scores:
                analysis['confidence'] = float(np.mean(scores))
            
            # Generate recommendation
            analysis['recommendation'], analysis['reasoning'] = self._generate_recommendation(
                analysis['scores']
            )
            
        except Exception as e:
            logger.error(f"Market intelligence error: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _analyze_technical(self, market_data: pd.DataFrame) -> float:
        """Technical analysis score (0-1)"""
        try:
            if len(market_data) < 20:
                return 0.5
            
            close_series = market_data['close']
            close_val = close_series.iloc[-1] if hasattr(close_series, 'iloc') else close_series[-1]
            close = float(close_val)
            ma_20_val = close_series.rolling(20).mean().iloc[-1]
            ma_20 = float(ma_20_val)
            ma_50_series = close_series.rolling(50).mean()
            ma_50_val = ma_50_series.iloc[-1] if len(market_data) >= 50 else ma_20
            ma_50 = float(ma_50_val)
            
            # Price position relative to moving averages
            if close > ma_20 > ma_50:
                return 0.8  # Strong uptrend
            elif close < ma_20 < ma_50:
                return 0.2  # Strong downtrend
            else:
                return 0.5  # Consolidation
        except Exception:
            return 0.5
    
    def _analyze_volume(self, market_data: pd.DataFrame) -> float:
        """Volume analysis score (0-1)"""
        try:
            if len(market_data) < 10:
                return 0.5
            
            vol_series = market_data['volume']
            current_vol_val = vol_series.iloc[-1] if hasattr(vol_series, 'iloc') else vol_series[-1]
            current_volume = float(current_vol_val)
            avg_vol_val = vol_series.rolling(10).mean().iloc[-1]
            avg_volume = float(avg_vol_val)
            
            # Volume surge indicator
            if current_volume > avg_volume * 1.5:
                return 0.8  # Strong volume
            elif current_volume > avg_volume:
                return 0.6  # Moderate volume
            else:
                return 0.4  # Weak volume
        except Exception:
            return 0.5
    
    def _analyze_trend(self, market_data: pd.DataFrame) -> float:
        """Trend strength analysis (0-1)"""
        try:
            if len(market_data) < 20:
                return 0.5
            
            close_prices = np.array(market_data['close'].tail(20).tolist(), dtype=float)
            
            # Calculate trend strength using linear regression
            x = np.arange(len(close_prices))
            coefficients = np.polyfit(x, close_prices, 1)
            slope = float(coefficients[0])
            
            # Normalize slope to 0-1
            price_range = float(np.max(close_prices) - np.min(close_prices))
            if price_range == 0:
                return 0.5
            
            normalized_slope = min(1.0, max(0.0, 0.5 + (slope / price_range)))
            return float(normalized_slope)
        except Exception:
            return 0.5
    
    def _analyze_momentum(self, market_data: pd.DataFrame) -> float:
        """Momentum analysis (0-1)"""
        try:
            if len(market_data) < 14:
                return 0.5
            
            # RSI-like momentum
            close = market_data['close']
            deltas = close.diff()
            gain = (deltas.where(deltas > 0, 0)).rolling(window=14).mean()
            loss = (-deltas.where(deltas < 0, 0)).rolling(window=14).mean()
            
            loss_last = loss.iloc[-1] if hasattr(loss, 'iloc') else loss[-1]
            loss_val = float(loss_last)
            gain_last = gain.iloc[-1] if hasattr(gain, 'iloc') else gain[-1]
            gain_val = float(gain_last)
            rs = gain_val / loss_val if loss_val != 0 else 1.0
            rsi = 100 - (100 / (1 + rs)) if rs > 0 else 50
            
            # Convert RSI to 0-1 scale
            return float(rsi / 100.0)
        except Exception:
            return 0.5
    
    def _generate_recommendation(self, scores: Dict[str, float]) -> tuple:
        """Generate trading recommendation based on scores"""
        valid_scores = [s for s in scores.values() if isinstance(s, (int, float))]
        
        if not valid_scores:
            return "HOLD", ["Insufficient data for recommendation"]
        
        avg_score = np.mean(valid_scores)
        reasoning = []
        
        # Score interpretation
        for metric, score in scores.items():
            if isinstance(score, (int, float)):
                if score > 0.7:
                    reasoning.append(f"🟢 {metric.upper()}: BULLISH ({score:.2f})")
                elif score < 0.3:
                    reasoning.append(f"🔴 {metric.upper()}: BEARISH ({score:.2f})")
                else:
                    reasoning.append(f"🟡 {metric.upper()}: NEUTRAL ({score:.2f})")
        
        # Recommendation
        if avg_score > 0.70:
            recommendation = "BUY"
            reasoning.append("📈 Overall: STRONG BUY signal")
        elif avg_score > 0.55:
            recommendation = "BUY_WEAK"
            reasoning.append("📈 Overall: Weak BUY signal")
        elif avg_score < 0.30:
            recommendation = "SELL"
            reasoning.append("📉 Overall: STRONG SELL signal")
        elif avg_score < 0.45:
            recommendation = "SELL_WEAK"
            reasoning.append("📉 Overall: Weak SELL signal")
        else:
            recommendation = "HOLD"
            reasoning.append("🟡 Overall: NEUTRAL - HOLD position")
        
        return recommendation, reasoning
    
    async def calculate_dynamic_sltp(self, entry_price: float, direction: str, 
                                     market_data: pd.DataFrame,
                                     risk_percent: float = 1.0) -> Dict[str, float]:
        """Calculate dynamic SL/TP based on market volatility"""
        try:
            if len(market_data) < 20:
                # Default: 2% SL, 4% TP
                if direction.upper() == "BUY":
                    return {
                        'stop_loss': entry_price * 0.98,
                        'take_profit': entry_price * 1.04,
                        'atr_based': False
                    }
                else:
                    return {
                        'stop_loss': entry_price * 1.02,
                        'take_profit': entry_price * 0.96,
                        'atr_based': False
                    }
            
            # Calculate ATR (Average True Range)
            high = market_data['high']
            low = market_data['low']
            close = market_data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr_val = tr.rolling(14).mean().iloc[-1]
            atr = float(atr_val)
            
            # Dynamic risk calculation
            risk_factor = (atr / entry_price) * risk_percent
            
            if direction.upper() == "BUY":
                stop_loss = entry_price - (atr * 2)
                take_profit = entry_price + (atr * 3)
            else:
                stop_loss = entry_price + (atr * 2)
                take_profit = entry_price - (atr * 3)
            
            return {
                'stop_loss': float(stop_loss),
                'take_profit': float(take_profit),
                'atr': atr,
                'atr_based': True
            }
        except Exception as e:
            logger.error(f"SL/TP calculation error: {e}")
            return {'stop_loss': 0.0, 'take_profit': 0.0, 'atr_based': False}
    
    def get_market_regime(self, market_data: pd.DataFrame) -> str:
        """Detect current market regime"""
        try:
            if len(market_data) < 20:
                return "unknown"
            
            close = market_data['close']
            returns = close.pct_change()
            volatility = returns.std()
            
            # Volatility-based regime
            if volatility > 0.02:
                return "high_volatility"
            elif volatility > 0.01:
                return "normal"
            else:
                return "low_volatility"
        except Exception:
            return "unknown"
    
    async def get_market_health_check(self) -> Dict[str, Any]:
        """Quick health check of market conditions"""
        return {
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'market_regime': self.market_regime,
            'confidence_threshold': self.confidence_threshold,
            'cache_size': len(self.signal_cache)
        }
