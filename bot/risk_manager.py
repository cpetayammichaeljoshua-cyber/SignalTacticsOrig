"""
Risk management system for the trading bot
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RiskMetrics:
    """Risk metrics for portfolio monitoring"""
    portfolio_value: float
    daily_pnl: float
    unrealized_pnl: float
    max_drawdown: float
    var_95: float  # Value at Risk 95%
    exposure: float
    leverage_ratio: float
    position_count: int

class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self, config):
        self.config = config
        self.position_history = {}
        self.pnl_history = []
        self.drawdown_history = []
        self.risk_events = []
        
        # Risk limits
        self.max_daily_loss = config.max_drawdown * 0.3  # 30% of max drawdown as daily limit
        self.max_position_size = 0.2  # 20% of portfolio per position
        self.max_total_exposure = 3.0  # 300% total exposure
        self.correlation_threshold = 0.7
        
    def evaluate_trade_risk(self, 
                           symbol: str,
                           signal: Dict[str, Any],
                           leverage: int,
                           position_size: float,
                           current_positions: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate risk for a potential trade"""
        
        try:
            risk_assessment = {
                'approved': True,
                'risk_level': RiskLevel.LOW,
                'risk_score': 0.0,
                'warnings': [],
                'recommendations': []
            }
            
            # Check individual position risk
            position_risk = self._assess_position_risk(symbol, signal, leverage, position_size)
            risk_assessment['position_risk'] = position_risk
            
            # Check portfolio risk
            portfolio_risk = self._assess_portfolio_risk(current_positions, symbol, position_size)
            risk_assessment['portfolio_risk'] = portfolio_risk
            
            # Check correlation risk
            correlation_risk = self._assess_correlation_risk(symbol, current_positions)
            risk_assessment['correlation_risk'] = correlation_risk
            
            # Check market condition risk
            market_risk = self._assess_market_risk(signal)
            risk_assessment['market_risk'] = market_risk
            
            # Calculate overall risk score
            risk_assessment['risk_score'] = self._calculate_overall_risk(
                position_risk, portfolio_risk, correlation_risk, market_risk
            )
            
            # Determine risk level and approval
            risk_assessment['risk_level'] = self._determine_risk_level(risk_assessment['risk_score'])
            risk_assessment['approved'] = self._should_approve_trade(risk_assessment)
            
            # Generate warnings and recommendations
            self._generate_risk_warnings(risk_assessment)
            
            logger.debug(f"🛡️ Risk assessment for {symbol}: {risk_assessment['risk_level'].value} "
                        f"(score: {risk_assessment['risk_score']:.2f}, approved: {risk_assessment['approved']})")
            
            return risk_assessment
            
        except Exception as e:
            logger.error(f"❌ Error evaluating trade risk for {symbol}: {e}")
            return {
                'approved': False,
                'risk_level': RiskLevel.CRITICAL,
                'risk_score': 1.0,
                'warnings': [f'Risk evaluation error: {e}'],
                'recommendations': ['Avoid trading until risk system is fixed']
            }
    
    def _assess_position_risk(self, symbol: str, signal: Dict[str, Any], leverage: int, position_size: float) -> Dict[str, Any]:
        """Assess risk for individual position"""
        try:
            risk_factors = {}
            
            # Leverage risk
            leverage_risk = min(1.0, leverage / self.config.max_leverage)
            risk_factors['leverage'] = leverage_risk
            
            # Position size risk
            size_risk = position_size / (self.max_position_size * 1000)  # Assuming 1000 as base portfolio
            risk_factors['size'] = min(1.0, size_risk)
            
            # Signal confidence risk (inverse)
            confidence_risk = 1.0 - signal.get('confidence', 0.5)
            risk_factors['confidence'] = confidence_risk
            
            # Volatility risk
            volatility = signal.get('volatility', 0.02)
            volatility_risk = min(1.0, volatility * 50)  # Scale volatility
            risk_factors['volatility'] = volatility_risk
            
            # Combined position risk
            position_risk_score = float(np.mean(list(risk_factors.values())))
            
            return {
                'risk_score': position_risk_score,
                'factors': risk_factors,
                'level': self._determine_risk_level(position_risk_score)
            }
            
        except Exception as e:
            logger.error(f"Error assessing position risk: {e}")
            return {'risk_score': 1.0, 'factors': {}, 'level': RiskLevel.CRITICAL}
    
    def _assess_portfolio_risk(self, current_positions: Dict[str, Any], new_symbol: str, new_position_size: float) -> Dict[str, Any]:
        """Assess portfolio-level risk"""
        try:
            risk_factors = {}
            
            # Calculate current exposure
            total_exposure = sum(pos.get('exposure', 0) for pos in current_positions.values())
            new_exposure = total_exposure + new_position_size
            
            # Exposure risk
            exposure_risk = min(1.0, new_exposure / self.max_total_exposure)
            risk_factors['exposure'] = exposure_risk
            
            # Concentration risk
            position_count = len(current_positions) + 1
            concentration_risk = 1.0 / max(1, position_count * 0.2)  # Favor diversification
            risk_factors['concentration'] = min(1.0, concentration_risk)
            
            # Sector/correlation risk (simplified)
            same_type_positions = sum(1 for symbol in current_positions.keys() 
                                    if self._get_asset_type(symbol) == self._get_asset_type(new_symbol))
            sector_risk = same_type_positions / max(1, len(current_positions) + 1)
            risk_factors['sector'] = sector_risk
            
            # Portfolio risk score
            portfolio_risk_score = float(np.mean(list(risk_factors.values())))
            
            return {
                'risk_score': portfolio_risk_score,
                'factors': risk_factors,
                'total_exposure': new_exposure,
                'position_count': position_count,
                'level': self._determine_risk_level(portfolio_risk_score)
            }
            
        except Exception as e:
            logger.error(f"Error assessing portfolio risk: {e}")
            return {'risk_score': 1.0, 'factors': {}, 'level': RiskLevel.CRITICAL}
    
    def _assess_correlation_risk(self, symbol: str, current_positions: Dict[str, Any]) -> Dict[str, Any]:
        """Assess correlation risk with existing positions"""
        try:
            if not current_positions:
                return {'risk_score': 0.0, 'correlations': {}, 'level': RiskLevel.LOW}
            
            correlations = {}
            high_correlation_count = 0
            
            for existing_symbol in current_positions.keys():
                # Simplified correlation calculation (in real implementation, use historical price data)
                correlation = self._estimate_correlation(symbol, existing_symbol)
                correlations[existing_symbol] = correlation
                
                if correlation > self.correlation_threshold:
                    high_correlation_count += 1
            
            # Correlation risk based on number of highly correlated positions
            correlation_risk = high_correlation_count / len(current_positions) if current_positions else 0
            
            return {
                'risk_score': correlation_risk,
                'correlations': correlations,
                'high_correlation_count': high_correlation_count,
                'level': self._determine_risk_level(correlation_risk)
            }
            
        except Exception as e:
            logger.error(f"Error assessing correlation risk: {e}")
            return {'risk_score': 1.0, 'correlations': {}, 'level': RiskLevel.CRITICAL}
    
    def _assess_market_risk(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Assess market condition risk"""
        try:
            risk_factors = {}
            
            # Volatility regime risk
            volatility_regime = signal.get('volatility_regime', 'normal')
            if volatility_regime == 'high':
                risk_factors['volatility_regime'] = 0.8
            elif volatility_regime == 'low':
                risk_factors['volatility_regime'] = 0.3
            else:
                risk_factors['volatility_regime'] = 0.5
            
            # Trend strength risk (contrarian positions in strong trends are risky)
            trend = signal.get('trend', 'sideways')
            signal_direction = signal.get('signal', 0)
            
            if trend in ['strong_uptrend', 'strong_downtrend']:
                # Check if signal is against the trend
                trend_risk = 0.3 if self._signal_with_trend(signal_direction, trend) else 0.8
            else:
                trend_risk = 0.4
            
            risk_factors['trend'] = trend_risk
            
            # Market risk score
            market_risk_score = float(np.mean(list(risk_factors.values())))
            
            return {
                'risk_score': market_risk_score,
                'factors': risk_factors,
                'level': self._determine_risk_level(market_risk_score)
            }
            
        except Exception as e:
            logger.error(f"Error assessing market risk: {e}")
            return {'risk_score': 0.5, 'factors': {}, 'level': RiskLevel.MEDIUM}
    
    def _calculate_overall_risk(self, position_risk: Dict[str, Any], portfolio_risk: Dict[str, Any], 
                               correlation_risk: Dict[str, Any], market_risk: Dict[str, Any]) -> float:
        """Calculate overall risk score"""
        try:
            weights = {
                'position': 0.3,
                'portfolio': 0.3,
                'correlation': 0.2,
                'market': 0.2
            }
            
            overall_risk = (
                position_risk['risk_score'] * weights['position'] +
                portfolio_risk['risk_score'] * weights['portfolio'] +
                correlation_risk['risk_score'] * weights['correlation'] +
                market_risk['risk_score'] * weights['market']
            )
            
            return min(1.0, max(0.0, overall_risk))
            
        except Exception as e:
            logger.error(f"Error calculating overall risk: {e}")
            return 1.0
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from risk score"""
        if risk_score < 0.3:
            return RiskLevel.LOW
        elif risk_score < 0.6:
            return RiskLevel.MEDIUM
        elif risk_score < 0.8:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _should_approve_trade(self, risk_assessment: Dict[str, Any]) -> bool:
        """Determine if trade should be approved based on risk"""
        try:
            risk_level = risk_assessment['risk_level']
            risk_score = risk_assessment['risk_score']
            
            # Never approve critical risk trades
            if risk_level == RiskLevel.CRITICAL:
                return False
            
            # High risk trades need very good signals
            if risk_level == RiskLevel.HIGH and risk_score > 0.75:
                return False
            
            # Check specific risk factors
            portfolio_risk = risk_assessment.get('portfolio_risk', {})
            if portfolio_risk.get('total_exposure', 0) > self.max_total_exposure:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error determining trade approval: {e}")
            return False
    
    def _generate_risk_warnings(self, risk_assessment: Dict[str, Any]):
        """Generate risk warnings and recommendations"""
        try:
            warnings = []
            recommendations = []
            
            risk_level = risk_assessment['risk_level']
            risk_score = risk_assessment['risk_score']
            
            # General risk warnings
            if risk_level == RiskLevel.CRITICAL:
                warnings.append("CRITICAL RISK: Trade blocked due to excessive risk")
                recommendations.append("Review risk parameters and market conditions")
            elif risk_level == RiskLevel.HIGH:
                warnings.append("HIGH RISK: Consider reducing position size")
                recommendations.append("Monitor position closely and set tight stops")
            
            # Specific factor warnings
            portfolio_risk = risk_assessment.get('portfolio_risk', {})
            if portfolio_risk.get('total_exposure', 0) > self.max_total_exposure * 0.8:
                warnings.append("High portfolio exposure detected")
                recommendations.append("Consider closing some positions to reduce exposure")
            
            correlation_risk = risk_assessment.get('correlation_risk', {})
            if correlation_risk.get('high_correlation_count', 0) > 2:
                warnings.append("High correlation risk with existing positions")
                recommendations.append("Diversify positions to reduce correlation risk")
            
            position_risk = risk_assessment.get('position_risk', {})
            if position_risk.get('factors', {}).get('leverage', 0) > 0.8:
                warnings.append("High leverage detected")
                recommendations.append("Consider reducing leverage for this position")
            
            risk_assessment['warnings'] = warnings
            risk_assessment['recommendations'] = recommendations
            
        except Exception as e:
            logger.error(f"Error generating risk warnings: {e}")
    
    def monitor_portfolio_risk(self, current_positions: Dict[str, Any], account_balance: float) -> Dict[str, Any]:
        """Monitor overall portfolio risk"""
        try:
            # Calculate current metrics
            total_exposure = sum(pos.get('exposure', 0) for pos in current_positions.values())
            unrealized_pnl = sum(pos.get('unrealized_pnl', 0) for pos in current_positions.values())
            
            # Calculate daily P&L
            daily_pnl = self._calculate_daily_pnl(current_positions)
            
            # Calculate max drawdown
            current_drawdown = self._calculate_current_drawdown(account_balance, unrealized_pnl)
            
            # Update histories
            self.pnl_history.append({'timestamp': datetime.now(), 'pnl': daily_pnl})
            self.drawdown_history.append({'timestamp': datetime.now(), 'drawdown': current_drawdown})
            
            # Keep last 1000 entries
            self.pnl_history = self.pnl_history[-1000:]
            self.drawdown_history = self.drawdown_history[-1000:]
            
            # Risk metrics
            risk_metrics = RiskMetrics(
                portfolio_value=account_balance + unrealized_pnl,
                daily_pnl=daily_pnl,
                unrealized_pnl=unrealized_pnl,
                max_drawdown=current_drawdown,
                var_95=self._calculate_var_95(),
                exposure=total_exposure,
                leverage_ratio=total_exposure / account_balance if account_balance > 0 else 0,
                position_count=len(current_positions)
            )
            
            # Risk alerts
            alerts = self._generate_portfolio_alerts(risk_metrics)
            
            return {
                'metrics': risk_metrics,
                'alerts': alerts,
                'risk_level': self._assess_portfolio_risk_level(risk_metrics),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error monitoring portfolio risk: {e}")
            return {'metrics': None, 'alerts': ['Error monitoring portfolio risk'], 'risk_level': RiskLevel.CRITICAL}
    
    def _get_asset_type(self, symbol: str) -> str:
        """Get asset type for correlation estimation"""
        if 'BTC' in symbol:
            return 'bitcoin'
        elif 'ETH' in symbol:
            return 'ethereum'
        elif symbol.endswith('USDT'):
            return 'crypto'
        else:
            return 'other'
    
    def _estimate_correlation(self, symbol1: str, symbol2: str) -> float:
        """Estimate correlation between two symbols (simplified)"""
        # In a real implementation, this would use historical price data
        # For now, using asset type-based estimation
        type1 = self._get_asset_type(symbol1)
        type2 = self._get_asset_type(symbol2)
        
        if type1 == type2:
            return 0.8  # High correlation for same asset type
        elif 'crypto' in [type1, type2]:
            return 0.6  # Medium correlation for crypto assets
        else:
            return 0.3  # Low correlation for different types
    
    def _signal_with_trend(self, signal: int, trend: str) -> bool:
        """Check if signal aligns with trend"""
        if trend in ['strong_uptrend', 'uptrend'] and signal == 1:
            return True
        elif trend in ['strong_downtrend', 'downtrend'] and signal == -1:
            return True
        else:
            return False
    
    def _calculate_daily_pnl(self, current_positions: Dict[str, Any]) -> float:
        """Calculate daily P&L"""
        # Simplified calculation - in reality would use position opening times
        return sum(pos.get('daily_pnl', 0) for pos in current_positions.values())
    
    def _calculate_current_drawdown(self, account_balance: float, unrealized_pnl: float) -> float:
        """Calculate current drawdown from peak"""
        current_value = account_balance + unrealized_pnl
        
        if not hasattr(self, 'peak_value'):
            self.peak_value = current_value
        
        if current_value > self.peak_value:
            self.peak_value = current_value
            return 0.0
        
        drawdown = (self.peak_value - current_value) / self.peak_value
        return drawdown
    
    def _calculate_var_95(self) -> float:
        """Calculate 95% Value at Risk"""
        if len(self.pnl_history) < 20:
            return 0.0
        
        pnl_values = [entry['pnl'] for entry in self.pnl_history[-100:]]
        return float(np.percentile(pnl_values, 5))  # 5th percentile for 95% VaR
    
    def _generate_portfolio_alerts(self, metrics: RiskMetrics) -> List[str]:
        """Generate portfolio risk alerts"""
        alerts = []
        
        if metrics.max_drawdown > self.config.max_drawdown * 0.8:
            alerts.append("WARNING: Approaching maximum drawdown limit")
        
        if metrics.daily_pnl < -self.max_daily_loss:
            alerts.append("ALERT: Daily loss limit exceeded")
        
        if metrics.leverage_ratio > 3.0:
            alerts.append("WARNING: High leverage ratio detected")
        
        if metrics.exposure > self.max_total_exposure:
            alerts.append("CRITICAL: Total exposure limit exceeded")
        
        return alerts
    
    def _assess_portfolio_risk_level(self, metrics: RiskMetrics) -> RiskLevel:
        """Assess overall portfolio risk level"""
        risk_score = 0
        
        # Drawdown component
        if metrics.max_drawdown > self.config.max_drawdown * 0.8:
            risk_score += 0.3
        
        # Leverage component
        if metrics.leverage_ratio > 2.0:
            risk_score += 0.2
        
        # Exposure component
        if metrics.exposure > self.max_total_exposure * 0.8:
            risk_score += 0.3
        
        # VaR component
        if abs(metrics.var_95) > metrics.portfolio_value * 0.05:
            risk_score += 0.2
        
        return self._determine_risk_level(risk_score)
