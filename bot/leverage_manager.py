"""
Adaptive leverage management system
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class LeverageConfig:
    """Configuration for leverage management"""
    base_leverage: int = 10
    max_leverage: int = 50
    min_leverage: int = 1
    confidence_threshold: float = 0.7
    volatility_adjustment: bool = True
    performance_adjustment: bool = True
    max_position_size: float = 0.1  # 10% of portfolio

class AdaptiveLeverageManager:
    """Manages leverage based on multiple factors"""
    
    def __init__(self, config: LeverageConfig):
        self.config = config
        self.performance_history = {}
        self.volatility_history = {}
        self.leverage_history = {}
        
    def calculate_leverage(self, 
                          symbol: str,
                          signal_confidence: float,
                          current_volatility: float,
                          account_balance: float,
                          recent_performance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Calculate optimal leverage for a trade"""
        
        try:
            # Start with base leverage
            leverage = float(self.config.base_leverage)
            
            # Factor 1: Signal confidence adjustment
            confidence_multiplier = self._calculate_confidence_multiplier(signal_confidence)
            leverage *= confidence_multiplier
            
            # Factor 2: Volatility adjustment
            volatility_multiplier = 1.0
            if self.config.volatility_adjustment:
                volatility_multiplier = self._calculate_volatility_multiplier(symbol, current_volatility)
                leverage *= volatility_multiplier
            
            # Factor 3: Performance adjustment
            performance_multiplier = 1.0
            if self.config.performance_adjustment and recent_performance:
                performance_multiplier = self._calculate_performance_multiplier(symbol, recent_performance)
                leverage *= performance_multiplier
            
            # Factor 4: Risk management constraints
            leverage = self._apply_risk_constraints(leverage, account_balance)
            
            # Ensure leverage is within bounds
            leverage = max(self.config.min_leverage, min(self.config.max_leverage, leverage))
            leverage = int(round(leverage))
            
            # Calculate position size
            position_size = self._calculate_position_size(leverage, account_balance, current_volatility)
            
            # Store in history
            self._update_leverage_history(symbol, leverage, signal_confidence, current_volatility)
            
            result = {
                'leverage': leverage,
                'position_size': position_size,
                'confidence_multiplier': confidence_multiplier,
                'volatility_multiplier': volatility_multiplier if self.config.volatility_adjustment else 1.0,
                'performance_multiplier': performance_multiplier if self.config.performance_adjustment and recent_performance else 1.0,
                'risk_score': self._calculate_risk_score(leverage, current_volatility, signal_confidence)
            }
            
            logger.debug(f"🎯 Calculated leverage for {symbol}: {leverage}x (confidence: {signal_confidence:.2f}, volatility: {current_volatility:.4f})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error calculating leverage for {symbol}: {e}")
            return {
                'leverage': self.config.base_leverage,
                'position_size': 0.01,
                'confidence_multiplier': 1.0,
                'volatility_multiplier': 1.0,
                'performance_multiplier': 1.0,
                'risk_score': 0.5
            }
    
    def _calculate_confidence_multiplier(self, confidence: float) -> float:
        """Calculate leverage multiplier based on signal confidence"""
        try:
            if confidence >= self.config.confidence_threshold:
                # High confidence: increase leverage
                multiplier = 1.0 + (confidence - self.config.confidence_threshold) * 2.0
            else:
                # Low confidence: decrease leverage
                multiplier = 0.5 + (confidence / self.config.confidence_threshold) * 0.5
            
            # Cap the multiplier
            return max(0.2, min(3.0, multiplier))
            
        except Exception as e:
            logger.error(f"Error calculating confidence multiplier: {e}")
            return 1.0
    
    def _calculate_volatility_multiplier(self, symbol: str, current_volatility: float) -> float:
        """Calculate leverage multiplier based on volatility"""
        try:
            # Store volatility history
            if symbol not in self.volatility_history:
                self.volatility_history[symbol] = []
            
            self.volatility_history[symbol].append(current_volatility)
            self.volatility_history[symbol] = self.volatility_history[symbol][-100:]  # Keep last 100
            
            # Calculate percentile of current volatility
            if len(self.volatility_history[symbol]) < 10:
                return 1.0  # Not enough data
            
            volatility_percentile = np.percentile(self.volatility_history[symbol], 
                                                [current_volatility * 100])[0] / 100.0
            
            # High volatility: reduce leverage, Low volatility: increase leverage
            if volatility_percentile > 0.8:
                multiplier = 0.5  # High volatility
            elif volatility_percentile > 0.6:
                multiplier = 0.7
            elif volatility_percentile < 0.2:
                multiplier = 1.3  # Low volatility
            elif volatility_percentile < 0.4:
                multiplier = 1.1
            else:
                multiplier = 1.0  # Normal volatility
            
            return multiplier
            
        except Exception as e:
            logger.error(f"Error calculating volatility multiplier: {e}")
            return 1.0
    
    def _calculate_performance_multiplier(self, symbol: str, recent_performance: Dict[str, Any]) -> float:
        """Calculate leverage multiplier based on recent performance"""
        try:
            win_rate = recent_performance.get('win_rate', 0.5)
            avg_return = recent_performance.get('avg_return', 0.0)
            max_drawdown = recent_performance.get('max_drawdown', 0.0)
            trade_count = recent_performance.get('trade_count', 0)
            
            # Need minimum trades for reliable performance metrics
            if trade_count < 5:
                return 1.0
            
            # Performance score based on multiple factors
            performance_score = 0
            
            # Win rate component (0-1)
            if win_rate > 0.6:
                performance_score += 0.4
            elif win_rate > 0.5:
                performance_score += 0.2
            elif win_rate < 0.4:
                performance_score -= 0.3
            
            # Average return component
            if avg_return > 0.02:  # 2% average return
                performance_score += 0.3
            elif avg_return > 0.01:
                performance_score += 0.1
            elif avg_return < -0.01:
                performance_score -= 0.3
            
            # Drawdown component
            if max_drawdown < 0.05:  # Less than 5% drawdown
                performance_score += 0.2
            elif max_drawdown > 0.15:  # More than 15% drawdown
                performance_score -= 0.4
            
            # Convert score to multiplier
            if performance_score > 0.5:
                multiplier = 1.5  # Good performance
            elif performance_score > 0.2:
                multiplier = 1.2
            elif performance_score < -0.3:
                multiplier = 0.5  # Poor performance
            elif performance_score < 0:
                multiplier = 0.7
            else:
                multiplier = 1.0
            
            return multiplier
            
        except Exception as e:
            logger.error(f"Error calculating performance multiplier: {e}")
            return 1.0
    
    def _apply_risk_constraints(self, leverage: float, account_balance: float) -> float:
        """Apply risk management constraints"""
        try:
            # Maximum position size constraint
            max_leverage_by_position = self.config.max_position_size * 100  # Convert to leverage
            leverage = min(leverage, max_leverage_by_position)
            
            # Account balance constraint (smaller accounts should use lower leverage)
            if account_balance < 1000:  # Small account
                leverage *= 0.5
            elif account_balance < 5000:
                leverage *= 0.7
            elif account_balance > 50000:  # Large account
                leverage *= 1.2
            
            return leverage
            
        except Exception as e:
            logger.error(f"Error applying risk constraints: {e}")
            return leverage
    
    def _calculate_position_size(self, leverage: int, account_balance: float, volatility: float) -> float:
        """Calculate position size based on leverage and risk"""
        try:
            # Base position size as percentage of account
            base_position_pct = 0.02  # 2% base risk
            
            # Adjust for volatility (higher volatility = smaller position)
            volatility_adjustment = max(0.5, min(2.0, 1.0 / (volatility * 100 + 1)))
            
            # Calculate position size
            position_size = account_balance * base_position_pct * volatility_adjustment / leverage
            
            # Ensure minimum and maximum position sizes
            min_position = account_balance * 0.001  # 0.1% minimum
            max_position = account_balance * self.config.max_position_size
            
            position_size = max(min_position, min(max_position, position_size))
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return account_balance * 0.01  # 1% fallback
    
    def _calculate_risk_score(self, leverage: int, volatility: float, confidence: float) -> float:
        """Calculate overall risk score for the position"""
        try:
            # Leverage risk (0-1, higher leverage = higher risk)
            leverage_risk = min(1.0, leverage / self.config.max_leverage)
            
            # Volatility risk (0-1, higher volatility = higher risk)
            volatility_risk = min(1.0, volatility * 1000)  # Scale volatility
            
            # Confidence risk (0-1, lower confidence = higher risk)
            confidence_risk = 1.0 - confidence
            
            # Combined risk score
            risk_score = (leverage_risk * 0.4 + volatility_risk * 0.3 + confidence_risk * 0.3)
            
            return max(0.0, min(1.0, risk_score))
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 0.5  # Medium risk fallback
    
    def _update_leverage_history(self, symbol: str, leverage: int, confidence: float, volatility: float):
        """Update leverage history for analysis"""
        try:
            if symbol not in self.leverage_history:
                self.leverage_history[symbol] = []
            
            self.leverage_history[symbol].append({
                'timestamp': datetime.now(),
                'leverage': leverage,
                'confidence': confidence,
                'volatility': volatility
            })
            
            # Keep last 500 entries
            self.leverage_history[symbol] = self.leverage_history[symbol][-500:]
            
        except Exception as e:
            logger.error(f"Error updating leverage history: {e}")
    
    def get_leverage_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get leverage usage statistics"""
        try:
            if symbol not in self.leverage_history or not self.leverage_history[symbol]:
                return {}
            
            history = self.leverage_history[symbol]
            leverages = [entry['leverage'] for entry in history[-50:]]  # Last 50
            
            return {
                'avg_leverage': np.mean(leverages),
                'max_leverage': np.max(leverages),
                'min_leverage': np.min(leverages),
                'leverage_std': np.std(leverages),
                'total_positions': len(history),
                'recent_trend': self._calculate_leverage_trend(leverages)
            }
            
        except Exception as e:
            logger.error(f"Error getting leverage statistics: {e}")
            return {}
    
    def _calculate_leverage_trend(self, leverages: List[int]) -> str:
        """Calculate recent leverage trend"""
        try:
            if len(leverages) < 10:
                return "insufficient_data"
            
            recent_avg = np.mean(leverages[-5:])
            older_avg = np.mean(leverages[-10:-5])
            
            if recent_avg > older_avg * 1.1:
                return "increasing"
            elif recent_avg < older_avg * 0.9:
                return "decreasing"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Error calculating leverage trend: {e}")
            return "unknown"
    
    def adjust_leverage_for_market_conditions(self, base_leverage: int, market_conditions: Dict[str, Any]) -> int:
        """Adjust leverage based on overall market conditions"""
        try:
            multiplier = 1.0
            
            # Market sentiment adjustment
            sentiment = market_conditions.get('sentiment', 'neutral')
            if sentiment == 'extreme_fear':
                multiplier *= 0.5
            elif sentiment == 'fear':
                multiplier *= 0.7
            elif sentiment == 'greed':
                multiplier *= 1.2
            elif sentiment == 'extreme_greed':
                multiplier *= 0.8  # Reduce leverage in extreme greed
            
            # Overall market volatility
            market_volatility = market_conditions.get('market_volatility', 'normal')
            if market_volatility == 'high':
                multiplier *= 0.6
            elif market_volatility == 'low':
                multiplier *= 1.3
            
            # Correlation adjustment (if positions are highly correlated)
            correlation = market_conditions.get('position_correlation', 0.5)
            if correlation > 0.8:
                multiplier *= 0.7  # Reduce leverage for highly correlated positions
            
            adjusted_leverage = int(base_leverage * multiplier)
            return max(self.config.min_leverage, min(self.config.max_leverage, adjusted_leverage))
            
        except Exception as e:
            logger.error(f"Error adjusting leverage for market conditions: {e}")
            return base_leverage
