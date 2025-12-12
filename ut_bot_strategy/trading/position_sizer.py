"""
Advanced Volatility-Based Position Sizer for Optimal Risk Management

Features:
- ATR volatility-based position sizing
- Kelly Criterion for optimal bet sizing
- Maximum portfolio exposure limits
- Signal confidence adjustments
- Correlation-aware position sizing
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PositionSizeResult:
    """Result of position size calculation"""
    recommended_size: float
    max_size: float
    risk_per_trade: float
    stop_distance_percent: float
    atr_multiple: float
    kelly_fraction: float
    position_value_usd: float
    portfolio_exposure_percent: float
    confidence_adjusted_size: float


class VolatilityPositionSizer:
    """
    Advanced volatility-based position sizer for optimal risk management
    
    Features:
    - Calculates position sizes based on ATR volatility
    - Implements Kelly Criterion for optimal sizing
    - Respects maximum portfolio exposure limits
    - Adjusts size based on signal confidence
    - Considers correlation with other positions
    """
    
    def __init__(
        self,
        max_risk_per_trade: float = 2.0,
        max_portfolio_exposure: float = 50.0,
        volatility_scalar: float = 1.0,
        use_kelly: bool = True,
        kelly_fraction_limit: float = 0.25,
        min_position_size: float = 0.0001,
        correlation_penalty: float = 0.1
    ):
        """
        Initialize the VolatilityPositionSizer
        
        Args:
            max_risk_per_trade: Maximum risk per trade as percentage (default 2%)
            max_portfolio_exposure: Maximum portfolio exposure as percentage (default 50%)
            volatility_scalar: Scalar to adjust volatility sensitivity (default 1.0)
            use_kelly: Whether to use Kelly Criterion for sizing (default True)
            kelly_fraction_limit: Maximum Kelly fraction allowed (default 0.25)
            min_position_size: Minimum position size in base units
            correlation_penalty: Penalty per correlated position (default 0.1 = 10%)
        """
        self.max_risk_per_trade = max_risk_per_trade / 100.0
        self.max_portfolio_exposure = max_portfolio_exposure / 100.0
        self.volatility_scalar = volatility_scalar
        self.use_kelly = use_kelly
        self.kelly_fraction_limit = kelly_fraction_limit
        self.min_position_size = min_position_size
        self.correlation_penalty = correlation_penalty
        
        logger.info(
            f"VolatilityPositionSizer initialized: "
            f"max_risk={max_risk_per_trade}%, max_exposure={max_portfolio_exposure}%, "
            f"volatility_scalar={volatility_scalar}, use_kelly={use_kelly}, "
            f"kelly_limit={kelly_fraction_limit}"
        )
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        account_balance: float,
        confidence: float,
        atr: float,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        current_positions: Optional[List[Dict]] = None,
        avg_volatility: Optional[float] = None
    ) -> PositionSizeResult:
        """
        Main position size calculation method
        
        Args:
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            account_balance: Available account balance in USD
            confidence: Signal confidence score (0-1)
            atr: Current Average True Range value
            win_rate: Historical win rate (0-1) for Kelly calculation
            avg_win: Average winning trade percentage
            avg_loss: Average losing trade percentage
            current_positions: List of current open positions for correlation check
            avg_volatility: Average historical volatility for comparison
            
        Returns:
            PositionSizeResult with all sizing metrics
        """
        try:
            stop_distance = abs(entry_price - stop_loss)
            stop_distance_percent = stop_distance / entry_price
            
            if stop_distance_percent <= 0:
                logger.warning("Invalid stop distance, using fallback of 2%")
                stop_distance_percent = 0.02
                stop_distance = entry_price * 0.02
            
            atr_percent = (atr / entry_price) if entry_price > 0 else 0.01
            atr_multiple = stop_distance / atr if atr > 0 else 1.5
            
            risk_amount = account_balance * self.max_risk_per_trade
            
            base_position_value = risk_amount / stop_distance_percent
            base_size = base_position_value / entry_price
            
            kelly_fraction = 0.0
            if self.use_kelly and win_rate is not None and avg_win is not None and avg_loss is not None:
                kelly_fraction = self.calculate_kelly_fraction(win_rate, avg_win, avg_loss)
                kelly_adjusted_risk = min(self.max_risk_per_trade, kelly_fraction)
                kelly_risk_amount = account_balance * kelly_adjusted_risk
                kelly_position_value = kelly_risk_amount / stop_distance_percent
                kelly_size = kelly_position_value / entry_price
                
                base_size = min(base_size, kelly_size)
            
            if avg_volatility is not None and avg_volatility > 0:
                current_volatility = atr_percent
                volatility_adjusted_size = self.adjust_for_volatility(
                    base_size, current_volatility, avg_volatility
                )
            else:
                volatility_adjusted_size = self._apply_volatility_scalar(base_size, atr_percent)
            
            confidence_adjusted_size = volatility_adjusted_size * self._confidence_multiplier(confidence)
            
            correlation_adjustment = 1.0
            if current_positions:
                correlation_adjustment = self._calculate_correlation_adjustment(current_positions)
            
            correlation_adjusted_size = confidence_adjusted_size * correlation_adjustment
            
            max_exposure_value = self.calculate_max_exposure(account_balance, self.max_portfolio_exposure * 100)
            max_size = max_exposure_value / entry_price
            
            recommended_size = min(correlation_adjusted_size, max_size)
            recommended_size = max(recommended_size, self.min_position_size)
            
            position_value_usd = recommended_size * entry_price
            portfolio_exposure_percent = (position_value_usd / account_balance) * 100 if account_balance > 0 else 0
            
            result = PositionSizeResult(
                recommended_size=recommended_size,
                max_size=max_size,
                risk_per_trade=risk_amount,
                stop_distance_percent=stop_distance_percent * 100,
                atr_multiple=atr_multiple,
                kelly_fraction=kelly_fraction,
                position_value_usd=position_value_usd,
                portfolio_exposure_percent=portfolio_exposure_percent,
                confidence_adjusted_size=confidence_adjusted_size
            )
            
            logger.info(
                f"Position size calculated: {recommended_size:.6f} units, "
                f"value=${position_value_usd:.2f}, exposure={portfolio_exposure_percent:.1f}%, "
                f"kelly={kelly_fraction:.3f}, atr_mult={atr_multiple:.2f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            fallback_size = (account_balance * 0.01) / entry_price if entry_price > 0 else self.min_position_size
            return PositionSizeResult(
                recommended_size=fallback_size,
                max_size=fallback_size,
                risk_per_trade=account_balance * 0.01,
                stop_distance_percent=2.0,
                atr_multiple=1.5,
                kelly_fraction=0.0,
                position_value_usd=fallback_size * entry_price,
                portfolio_exposure_percent=1.0,
                confidence_adjusted_size=fallback_size
            )
    
    def calculate_kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly Criterion fraction for optimal position sizing
        
        Kelly Formula: f* = (p * b - q) / b
        Where:
        - p = probability of winning (win_rate)
        - q = probability of losing (1 - win_rate)
        - b = ratio of average win to average loss
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade as percentage (e.g., 5.0 for 5%)
            avg_loss: Average losing trade as percentage (e.g., 2.0 for 2%)
            
        Returns:
            Kelly fraction (0 to kelly_fraction_limit)
        """
        try:
            win_rate = max(0.0, min(1.0, win_rate))
            
            if avg_loss <= 0:
                logger.warning("Invalid avg_loss for Kelly calculation, returning 0")
                return 0.0
            
            b = avg_win / avg_loss
            
            p = win_rate
            q = 1.0 - win_rate
            
            kelly = (p * b - q) / b if b > 0 else 0.0
            
            kelly = max(0.0, kelly)
            
            limited_kelly = min(kelly, self.kelly_fraction_limit)
            
            logger.debug(
                f"Kelly calculation: win_rate={win_rate:.2f}, avg_win={avg_win:.2f}%, "
                f"avg_loss={avg_loss:.2f}%, b={b:.2f}, raw_kelly={kelly:.3f}, "
                f"limited_kelly={limited_kelly:.3f}"
            )
            
            return limited_kelly
            
        except Exception as e:
            logger.error(f"Error calculating Kelly fraction: {e}")
            return 0.0
    
    def adjust_for_volatility(
        self,
        base_size: float,
        current_volatility: float,
        avg_volatility: float
    ) -> float:
        """
        Adjust position size based on volatility comparison
        
        When volatility is higher than average, reduce position size.
        When volatility is lower than average, allow larger position.
        
        Args:
            base_size: Base position size before volatility adjustment
            current_volatility: Current volatility (e.g., ATR as percentage)
            avg_volatility: Average historical volatility
            
        Returns:
            Volatility-adjusted position size
        """
        try:
            if avg_volatility <= 0:
                logger.warning("Invalid avg_volatility, returning base size")
                return base_size
            
            volatility_ratio = current_volatility / avg_volatility
            
            adjustment_factor = 1.0 / volatility_ratio
            
            adjustment_factor = max(0.5, min(1.5, adjustment_factor))
            
            adjustment_factor = 1.0 + (adjustment_factor - 1.0) * self.volatility_scalar
            
            adjusted_size = base_size * adjustment_factor
            
            logger.debug(
                f"Volatility adjustment: current={current_volatility:.4f}, "
                f"avg={avg_volatility:.4f}, ratio={volatility_ratio:.2f}, "
                f"factor={adjustment_factor:.2f}, adjusted_size={adjusted_size:.6f}"
            )
            
            return adjusted_size
            
        except Exception as e:
            logger.error(f"Error adjusting for volatility: {e}")
            return base_size
    
    def calculate_max_exposure(
        self,
        account_balance: float,
        max_exposure_percent: Optional[float] = None
    ) -> float:
        """
        Calculate maximum position value based on exposure limits
        
        Args:
            account_balance: Available account balance in USD
            max_exposure_percent: Maximum exposure as percentage (overrides default)
            
        Returns:
            Maximum position value in USD
        """
        try:
            if max_exposure_percent is not None:
                exposure_limit = max_exposure_percent / 100.0
            else:
                exposure_limit = self.max_portfolio_exposure
            
            max_exposure = account_balance * exposure_limit
            
            logger.debug(
                f"Max exposure calculated: balance=${account_balance:.2f}, "
                f"limit={exposure_limit*100:.1f}%, max_value=${max_exposure:.2f}"
            )
            
            return max_exposure
            
        except Exception as e:
            logger.error(f"Error calculating max exposure: {e}")
            return account_balance * 0.1
    
    def calculate_risk_reward(
        self,
        entry: float,
        stop: float,
        target: float
    ) -> float:
        """
        Calculate risk-to-reward ratio for a trade
        
        Args:
            entry: Entry price
            stop: Stop loss price
            target: Take profit target price
            
        Returns:
            Risk-reward ratio (target distance / stop distance)
        """
        try:
            risk_distance = abs(entry - stop)
            reward_distance = abs(target - entry)
            
            if risk_distance <= 0:
                logger.warning("Invalid risk distance, returning 0")
                return 0.0
            
            rr_ratio = reward_distance / risk_distance
            
            logger.debug(
                f"Risk-reward calculated: entry={entry:.4f}, stop={stop:.4f}, "
                f"target={target:.4f}, R:R={rr_ratio:.2f}"
            )
            
            return rr_ratio
            
        except Exception as e:
            logger.error(f"Error calculating risk-reward: {e}")
            return 0.0
    
    def _confidence_multiplier(self, confidence: float) -> float:
        """
        Calculate position size multiplier based on signal confidence
        
        Args:
            confidence: Signal confidence score (0-1)
            
        Returns:
            Multiplier between 0.5 and 1.0
        """
        confidence = max(0.0, min(1.0, confidence))
        
        return 0.5 + (confidence * 0.5)
    
    def _apply_volatility_scalar(self, base_size: float, atr_percent: float) -> float:
        """
        Apply volatility scalar when no average volatility is provided
        
        Args:
            base_size: Base position size
            atr_percent: ATR as percentage of price
            
        Returns:
            Adjusted position size
        """
        if atr_percent > 0.03:
            reduction = min(0.5, (atr_percent - 0.03) * 10)
            return base_size * (1.0 - reduction) * self.volatility_scalar
        elif atr_percent < 0.01:
            boost = min(0.3, (0.01 - atr_percent) * 20)
            return base_size * (1.0 + boost) * self.volatility_scalar
        
        return base_size * self.volatility_scalar
    
    def _calculate_correlation_adjustment(self, current_positions: List[Dict]) -> float:
        """
        Calculate position size adjustment based on correlation with existing positions
        
        Reduces position size when there are correlated open positions to prevent
        concentration risk.
        
        Args:
            current_positions: List of current open positions with 'symbol' and 'direction'
            
        Returns:
            Adjustment multiplier (0.5 to 1.0)
        """
        try:
            num_positions = len(current_positions)
            
            if num_positions == 0:
                return 1.0
            
            same_direction_count = sum(
                1 for pos in current_positions 
                if pos.get('direction', '').upper() in ['LONG', 'BUY']
            )
            opposite_direction_count = num_positions - same_direction_count
            
            correlation_factor = same_direction_count * self.correlation_penalty
            
            hedge_bonus = opposite_direction_count * 0.05
            
            adjustment = 1.0 - correlation_factor + hedge_bonus
            
            adjustment = max(0.5, min(1.0, adjustment))
            
            logger.debug(
                f"Correlation adjustment: {num_positions} positions, "
                f"same_dir={same_direction_count}, opposite={opposite_direction_count}, "
                f"adjustment={adjustment:.2f}"
            )
            
            return adjustment
            
        except Exception as e:
            logger.error(f"Error calculating correlation adjustment: {e}")
            return 0.8
    
    def get_sizing_summary(self, result: PositionSizeResult) -> str:
        """
        Format position sizing result as human-readable summary
        
        Args:
            result: PositionSizeResult from calculation
            
        Returns:
            Formatted string summary
        """
        return (
            f"📊 Position Sizing Summary\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📈 Recommended Size: {result.recommended_size:.6f}\n"
            f"📉 Max Allowed Size: {result.max_size:.6f}\n"
            f"💵 Position Value: ${result.position_value_usd:.2f}\n"
            f"⚠️ Risk Amount: ${result.risk_per_trade:.2f}\n"
            f"📏 Stop Distance: {result.stop_distance_percent:.2f}%\n"
            f"📊 ATR Multiple: {result.atr_multiple:.2f}x\n"
            f"🎲 Kelly Fraction: {result.kelly_fraction:.3f}\n"
            f"💼 Portfolio Exposure: {result.portfolio_exposure_percent:.1f}%\n"
            f"🎯 Confidence Adj Size: {result.confidence_adjusted_size:.6f}"
        )
    
    def validate_position(
        self,
        result: PositionSizeResult,
        account_balance: float,
        min_order_value: float = 10.0
    ) -> Tuple[bool, str]:
        """
        Validate if the calculated position meets trading requirements
        
        Args:
            result: PositionSizeResult from calculation
            account_balance: Available account balance
            min_order_value: Minimum order value allowed by exchange
            
        Returns:
            Tuple of (is_valid, reason)
        """
        if result.position_value_usd < min_order_value:
            return False, f"Position value ${result.position_value_usd:.2f} below minimum ${min_order_value}"
        
        if result.position_value_usd > account_balance:
            return False, f"Position value ${result.position_value_usd:.2f} exceeds balance ${account_balance:.2f}"
        
        if result.portfolio_exposure_percent > self.max_portfolio_exposure * 100:
            return False, f"Exposure {result.portfolio_exposure_percent:.1f}% exceeds max {self.max_portfolio_exposure*100:.1f}%"
        
        if result.recommended_size < self.min_position_size:
            return False, f"Position size {result.recommended_size:.6f} below minimum {self.min_position_size}"
        
        return True, "Position validated successfully"
