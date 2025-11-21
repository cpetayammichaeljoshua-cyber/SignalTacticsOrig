#!/usr/bin/env python3
"""
Advanced Multi-Layer Signal Validation System
Validates signals through comprehensive checks before execution:
- Technical validation (price, SL/TP validity)
- Market condition validation (spread, liquidity, volatility)
- Risk management validation (exposure limits, drawdown)
- Order flow validation (ATAS integration)
- ML-based signal quality scoring
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import ccxt


@dataclass
class ValidationResult:
    """Result of signal validation"""
    is_valid: bool
    confidence_score: float  # 0-100
    validation_details: Dict[str, Any]
    warnings: List[str]
    errors: List[str]
    passed_checks: List[str]
    failed_checks: List[str]
    order_flow_score: float = 0.0
    final_decision: str = "REJECT"  # APPROVE, REJECT, REVIEW


class AdvancedSignalValidator:
    """
    Multi-layer validation system for trading signals
    Only approved signals proceed to execution and Telegram notification
    """
    
    def __init__(self, exchange: Optional[ccxt.Exchange] = None, atas_fetcher=None):
        self.logger = logging.getLogger(__name__)
        self.exchange = exchange
        self.atas_fetcher = atas_fetcher
        
        # Validation thresholds
        self.min_confidence_score = 75.0  # Minimum overall confidence
        self.max_spread_pct = 0.1  # Maximum 0.1% spread
        self.min_volume_24h = 1000000  # $1M minimum 24h volume
        self.max_position_exposure = 5.0  # Max 5% per position
        self.max_total_exposure = 20.0  # Max 20% total
        
        # Order flow thresholds
        self.min_order_flow_score = 60.0  # Minimum order flow quality
        self.order_flow_weight = 0.25  # 25% weight in final decision
        
        # Risk management
        self.max_leverage = 30
        self.min_risk_reward_ratio = 1.5
        self.max_stop_loss_pct = 2.0
        
        # Performance tracking
        self.total_validated = 0
        self.total_approved = 0
        self.total_rejected = 0
        self.rejection_reasons = {}
        
        self.logger.info("✅ Advanced Signal Validator initialized")
    
    async def validate_signal(
        self,
        signal: Any,
        current_positions: Dict[str, Any] = None,
        account_balance: float = 1000.0
    ) -> ValidationResult:
        """
        Comprehensive signal validation
        
        Args:
            signal: Trading signal to validate
            current_positions: Currently open positions
            account_balance: Account balance for exposure calculations
            
        Returns:
            ValidationResult with detailed validation outcome
        """
        self.total_validated += 1
        
        validation_details = {}
        warnings = []
        errors = []
        passed_checks = []
        failed_checks = []
        
        try:
            # Layer 1: Technical Validation
            tech_valid, tech_score, tech_details = await self._validate_technical(signal)
            validation_details['technical'] = tech_details
            
            if tech_valid:
                passed_checks.append("Technical Validation")
            else:
                failed_checks.append("Technical Validation")
                errors.extend(tech_details.get('errors', []))
            
            # Layer 2: Market Condition Validation
            market_valid, market_score, market_details = await self._validate_market_conditions(signal)
            validation_details['market'] = market_details
            
            if market_valid:
                passed_checks.append("Market Conditions")
            else:
                failed_checks.append("Market Conditions")
                warnings.extend(market_details.get('warnings', []))
            
            # Layer 3: Risk Management Validation
            risk_valid, risk_score, risk_details = await self._validate_risk_management(
                signal, current_positions, account_balance
            )
            validation_details['risk'] = risk_details
            
            if risk_valid:
                passed_checks.append("Risk Management")
            else:
                failed_checks.append("Risk Management")
                errors.extend(risk_details.get('errors', []))
            
            # Layer 4: Order Flow Validation (ATAS Integration)
            flow_valid, flow_score, flow_details = await self._validate_order_flow(signal)
            validation_details['order_flow'] = flow_details
            
            if flow_valid:
                passed_checks.append("Order Flow Analysis")
            else:
                failed_checks.append("Order Flow Analysis")
                warnings.extend(flow_details.get('warnings', []))
            
            # Layer 5: Signal Quality Scoring
            quality_valid, quality_score, quality_details = await self._validate_signal_quality(signal)
            validation_details['quality'] = quality_details
            
            if quality_valid:
                passed_checks.append("Signal Quality")
            else:
                failed_checks.append("Signal Quality")
                warnings.extend(quality_details.get('warnings', []))
            
            # Calculate final confidence score (weighted average)
            confidence_score = (
                tech_score * 0.25 +
                market_score * 0.20 +
                risk_score * 0.20 +
                flow_score * self.order_flow_weight +
                quality_score * 0.10
            )
            
            # Make final decision
            is_valid = all([tech_valid, market_valid, risk_valid])
            
            if is_valid and confidence_score >= self.min_confidence_score:
                final_decision = "APPROVE"
                self.total_approved += 1
            elif is_valid and confidence_score >= self.min_confidence_score * 0.85:
                final_decision = "REVIEW"  # Borderline case
                warnings.append("Signal quality borderline - requires review")
            else:
                final_decision = "REJECT"
                self.total_rejected += 1
                
                # Track rejection reasons
                for check in failed_checks:
                    self.rejection_reasons[check] = self.rejection_reasons.get(check, 0) + 1
            
            result = ValidationResult(
                is_valid=is_valid and final_decision == "APPROVE",
                confidence_score=confidence_score,
                validation_details=validation_details,
                warnings=warnings,
                errors=errors,
                passed_checks=passed_checks,
                failed_checks=failed_checks,
                order_flow_score=flow_score,
                final_decision=final_decision
            )
            
            # Log validation result
            status_emoji = "✅" if result.is_valid else "❌"
            self.logger.info(f"\n{status_emoji} SIGNAL VALIDATION: {final_decision}")
            self.logger.info(f"   Symbol: {self._get_attr(signal, 'symbol', 'UNKNOWN')}")
            self.logger.info(f"   Confidence Score: {confidence_score:.1f}%")
            self.logger.info(f"   Passed: {len(passed_checks)}/{len(passed_checks) + len(failed_checks)} checks")
            
            if failed_checks:
                self.logger.info(f"   Failed Checks: {', '.join(failed_checks)}")
            
            if errors:
                self.logger.info(f"   Errors: {len(errors)}")
                for error in errors[:3]:  # Show first 3 errors
                    self.logger.info(f"      - {error}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Validation error: {e}")
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                validation_details={'error': str(e)},
                warnings=[],
                errors=[f"Validation exception: {e}"],
                passed_checks=[],
                failed_checks=["System Error"],
                final_decision="REJECT"
            )
    
    async def _validate_technical(self, signal: Any) -> Tuple[bool, float, Dict]:
        """Validate technical aspects of signal"""
        details = {'checks': [], 'errors': []}
        score = 100.0
        
        try:
            entry_price = self._get_attr(signal, 'entry_price', 0)
            stop_loss = self._get_attr(signal, 'stop_loss', 0)
            tp1 = self._get_attr(signal, 'take_profit_1', 0)
            direction = self._get_attr(signal, 'direction', 'UNKNOWN')
            
            # Check 1: Valid prices
            if entry_price <= 0:
                details['errors'].append("Invalid entry price")
                return False, 0.0, details
            
            if stop_loss <= 0:
                details['errors'].append("Invalid stop loss")
                return False, 0.0, details
            
            if tp1 <= 0:
                details['errors'].append("Invalid take profit")
                return False, 0.0, details
            
            # Check 2: SL/TP placement logic
            if direction == 'LONG':
                if stop_loss >= entry_price:
                    details['errors'].append("Stop loss must be below entry for LONG")
                    return False, 0.0, details
                
                if tp1 <= entry_price:
                    details['errors'].append("Take profit must be above entry for LONG")
                    return False, 0.0, details
            
            elif direction == 'SHORT':
                if stop_loss <= entry_price:
                    details['errors'].append("Stop loss must be above entry for SHORT")
                    return False, 0.0, details
                
                if tp1 >= entry_price:
                    details['errors'].append("Take profit must be below entry for SHORT")
                    return False, 0.0, details
            
            # Check 3: Stop loss not too wide
            sl_distance_pct = abs((entry_price - stop_loss) / entry_price) * 100
            if sl_distance_pct > self.max_stop_loss_pct:
                details['errors'].append(f"Stop loss too wide: {sl_distance_pct:.2f}%")
                score -= 30
            
            # Check 4: Risk/Reward ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(tp1 - entry_price)
            rr_ratio = reward / risk if risk > 0 else 0
            
            if rr_ratio < self.min_risk_reward_ratio:
                details['errors'].append(f"Poor R/R ratio: 1:{rr_ratio:.2f}")
                score -= 20
            
            details['checks'] = ['Prices valid', 'SL/TP placement correct']
            details['sl_distance_pct'] = sl_distance_pct
            details['risk_reward_ratio'] = rr_ratio
            
            return score >= 50, score, details
            
        except Exception as e:
            details['errors'].append(f"Technical validation error: {e}")
            return False, 0.0, details
    
    async def _validate_market_conditions(self, signal: Any) -> Tuple[bool, float, Dict]:
        """Validate current market conditions"""
        details = {'checks': [], 'warnings': []}
        score = 100.0
        
        try:
            symbol = self._get_attr(signal, 'symbol', '')
            
            if not self.exchange:
                details['warnings'].append("No exchange connection - using default scores")
                return True, 80.0, details
            
            # Fetch market data
            ticker = await self.exchange.fetch_ticker(symbol)
            
            # Check 1: Spread
            if ticker.get('bid') and ticker.get('ask'):
                spread_pct = ((ticker['ask'] - ticker['bid']) / ticker['bid']) * 100
                
                if spread_pct > self.max_spread_pct:
                    details['warnings'].append(f"Wide spread: {spread_pct:.3f}%")
                    score -= 20
                
                details['spread_pct'] = spread_pct
            
            # Check 2: Volume
            volume_24h = ticker.get('quoteVolume', 0)
            
            if volume_24h < self.min_volume_24h:
                details['warnings'].append(f"Low 24h volume: ${volume_24h:,.0f}")
                score -= 30
            
            details['volume_24h'] = volume_24h
            
            # Check 3: Price stability (not halted or circuit breaker)
            if ticker.get('last') and ticker.get('close'):
                price_change_pct = abs((ticker['last'] - ticker['close']) / ticker['close']) * 100
                
                if price_change_pct > 10:
                    details['warnings'].append(f"Extreme volatility: {price_change_pct:.1f}%")
                    score -= 15
                
                details['price_change_pct'] = price_change_pct
            
            details['checks'] = ['Spread checked', 'Volume analyzed', 'Stability verified']
            
            return score >= 50, score, details
            
        except Exception as e:
            details['warnings'].append(f"Market validation error: {e}")
            return True, 70.0, details
    
    async def _validate_risk_management(
        self,
        signal: Any,
        current_positions: Dict[str, Any],
        account_balance: float
    ) -> Tuple[bool, float, Dict]:
        """Validate risk management parameters"""
        details = {'checks': [], 'errors': []}
        score = 100.0
        
        try:
            leverage = self._get_attr(signal, 'leverage', 10)
            position_size = self._get_attr(signal, 'position_size_usdt', 0)
            entry_price = self._get_attr(signal, 'entry_price', 0)
            stop_loss = self._get_attr(signal, 'stop_loss', 0)
            
            # Check 1: Leverage limit
            if leverage > self.max_leverage:
                details['errors'].append(f"Leverage too high: {leverage}x (max: {self.max_leverage}x)")
                return False, 0.0, details
            
            # Check 2: Position exposure
            position_exposure_pct = (position_size / account_balance) * 100
            
            if position_exposure_pct > self.max_position_exposure:
                details['errors'].append(
                    f"Position exposure too high: {position_exposure_pct:.1f}% (max: {self.max_position_exposure}%)"
                )
                return False, 0.0, details
            
            # Check 3: Total exposure (including current positions)
            current_exposure = 0.0
            if current_positions:
                for pos in current_positions.values():
                    current_exposure += pos.get('position_size', 0)
            
            total_exposure_pct = ((current_exposure + position_size) / account_balance) * 100
            
            if total_exposure_pct > self.max_total_exposure:
                details['errors'].append(
                    f"Total exposure too high: {total_exposure_pct:.1f}% (max: {self.max_total_exposure}%)"
                )
                return False, 0.0, details
            
            # Check 4: Risk per trade
            risk_amount = abs(entry_price - stop_loss) * (position_size / entry_price) * leverage
            risk_pct = (risk_amount / account_balance) * 100
            
            if risk_pct > 5.0:  # Max 5% risk per trade
                details['errors'].append(f"Risk per trade too high: {risk_pct:.1f}%")
                score -= 40
            
            details['checks'] = [
                'Leverage validated',
                'Position exposure checked',
                'Total exposure calculated',
                'Risk per trade verified'
            ]
            details['leverage'] = leverage
            details['position_exposure_pct'] = position_exposure_pct
            details['total_exposure_pct'] = total_exposure_pct
            details['risk_pct'] = risk_pct
            
            return score >= 50, score, details
            
        except Exception as e:
            details['errors'].append(f"Risk validation error: {e}")
            return False, 0.0, details
    
    async def _validate_order_flow(self, signal: Any) -> Tuple[bool, float, Dict]:
        """Validate using ATAS order flow data"""
        details = {'checks': [], 'warnings': [], 'flow_data': {}}
        score = 100.0
        
        try:
            if not self.atas_fetcher:
                details['warnings'].append("ATAS order flow not available - using default score")
                return True, 75.0, details
            
            symbol = self._get_attr(signal, 'symbol', '')
            direction = self._get_attr(signal, 'direction', '')
            
            # Fetch order flow data from ATAS
            flow_data = await self.atas_fetcher.fetch_order_flow(symbol)
            
            if not flow_data:
                details['warnings'].append("No order flow data available")
                return True, 70.0, details
            
            # Analyze order flow alignment with signal
            buy_volume = flow_data.get('buy_volume', 0)
            sell_volume = flow_data.get('sell_volume', 0)
            total_volume = buy_volume + sell_volume
            
            if total_volume > 0:
                buy_pct = (buy_volume / total_volume) * 100
                sell_pct = (sell_volume / total_volume) * 100
                
                details['flow_data'] = {
                    'buy_volume': buy_volume,
                    'sell_volume': sell_volume,
                    'buy_pct': buy_pct,
                    'sell_pct': sell_pct
                }
                
                # Check alignment
                if direction == 'LONG' and buy_pct < 55:
                    details['warnings'].append(f"Weak buy pressure for LONG: {buy_pct:.1f}%")
                    score -= 30
                elif direction == 'SHORT' and sell_pct < 55:
                    details['warnings'].append(f"Weak sell pressure for SHORT: {sell_pct:.1f}%")
                    score -= 30
                
                # Check for strong confirmation
                if direction == 'LONG' and buy_pct >= 65:
                    details['checks'].append("Strong buy flow confirmation")
                    score += 10  # Bonus
                elif direction == 'SHORT' and sell_pct >= 65:
                    details['checks'].append("Strong sell flow confirmation")
                    score += 10  # Bonus
            
            # Delta analysis
            delta = flow_data.get('delta', 0)
            if direction == 'LONG' and delta < 0:
                details['warnings'].append(f"Negative delta for LONG: {delta}")
                score -= 20
            elif direction == 'SHORT' and delta > 0:
                details['warnings'].append(f"Positive delta for SHORT: {delta}")
                score -= 20
            
            details['checks'].append('Order flow analyzed')
            
            return score >= self.min_order_flow_score, min(score, 100), details
            
        except Exception as e:
            details['warnings'].append(f"Order flow validation error: {e}")
            return True, 70.0, details
    
    async def _validate_signal_quality(self, signal: Any) -> Tuple[bool, float, Dict]:
        """Validate overall signal quality metrics"""
        details = {'checks': [], 'warnings': []}
        score = 100.0
        
        try:
            signal_strength = self._get_attr(signal, 'signal_strength', 0)
            consensus_confidence = self._get_attr(signal, 'consensus_confidence', 0)
            strategies_agree = self._get_attr(signal, 'strategies_agree', 0)
            total_strategies = self._get_attr(signal, 'total_strategies', 1)
            
            # Check signal strength
            if signal_strength < 70:
                details['warnings'].append(f"Low signal strength: {signal_strength:.1f}%")
                score -= 30
            
            # Check consensus
            if consensus_confidence < 60:
                details['warnings'].append(f"Low consensus: {consensus_confidence:.1f}%")
                score -= 25
            
            # Check strategy agreement
            agreement_pct = (strategies_agree / total_strategies) * 100 if total_strategies > 0 else 0
            
            if agreement_pct < 50:
                details['warnings'].append(f"Weak strategy agreement: {agreement_pct:.1f}%")
                score -= 20
            
            details['checks'] = ['Signal strength checked', 'Consensus analyzed', 'Strategy agreement verified']
            details['signal_strength'] = signal_strength
            details['consensus_confidence'] = consensus_confidence
            details['agreement_pct'] = agreement_pct
            
            return score >= 50, score, details
            
        except Exception as e:
            details['warnings'].append(f"Quality validation error: {e}")
            return True, 70.0, details
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        approval_rate = (self.total_approved / self.total_validated * 100) if self.total_validated > 0 else 0
        
        return {
            'total_validated': self.total_validated,
            'total_approved': self.total_approved,
            'total_rejected': self.total_rejected,
            'approval_rate': approval_rate,
            'rejection_reasons': self.rejection_reasons
        }
    
    def _get_attr(self, obj: Any, attr: str, default: Any = None) -> Any:
        """Get attribute from object or dict"""
        if hasattr(obj, attr):
            return getattr(obj, attr)
        elif isinstance(obj, dict):
            return obj.get(attr, default)
        return default


async def demo_validator():
    """Demo the advanced signal validator"""
    print("\n" + "="*80)
    print("🔍 ADVANCED SIGNAL VALIDATOR DEMO")
    print("="*80)
    
    validator = AdvancedSignalValidator()
    
    # Test signal
    test_signal = {
        'symbol': 'ETH/USDT:USDT',
        'direction': 'LONG',
        'entry_price': 3500.00,
        'stop_loss': 3482.50,
        'take_profit_1': 3542.00,
        'take_profit_2': 3570.00,
        'take_profit_3': 3600.00,
        'leverage': 20,
        'signal_strength': 85.5,
        'consensus_confidence': 75.0,
        'strategies_agree': 4,
        'total_strategies': 5,
        'risk_reward_ratio': 2.4,
        'position_size_usdt': 50.0
    }
    
    result = await validator.validate_signal(test_signal, account_balance=1000.0)
    
    print(f"\n✅ Validation Complete:")
    print(f"   Decision: {result.final_decision}")
    print(f"   Confidence: {result.confidence_score:.1f}%")
    print(f"   Passed: {len(result.passed_checks)} checks")
    print(f"   Failed: {len(result.failed_checks)} checks")
    
    if result.warnings:
        print(f"\n⚠️  Warnings:")
        for warning in result.warnings:
            print(f"   - {warning}")
    
    if result.errors:
        print(f"\n❌ Errors:")
        for error in result.errors:
            print(f"   - {error}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_validator())
