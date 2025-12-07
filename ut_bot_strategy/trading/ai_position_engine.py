"""
AI-Driven Position Engine for Dynamic TP/SL/Position Calculation

Features:
- Dynamic stop-loss calculation based on ATR, volatility, and market structure
- Multi-target take profits (TP1, TP2, TP3) with configurable allocation
- Intelligent position sizing based on risk percentage and account balance
- Optimal leverage calculation based on volatility and signal confidence
- Margin requirements management
- Trailing stop logic with progressive SL movement
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    """Trade direction enum"""
    LONG = "LONG"
    SHORT = "SHORT"


class TrailingStopState(Enum):
    """Trailing stop state machine"""
    INITIAL = "initial"
    AT_ENTRY = "at_entry"
    AT_TP1 = "at_tp1"
    AT_TP2 = "at_tp2"


@dataclass
class TPLevel:
    """Take profit level with allocation"""
    price: float
    allocation_percent: float
    risk_reward: float
    hit: bool = False


@dataclass
class TradeSetup:
    """Complete trade setup with all parameters"""
    entry_price: float
    stop_loss: float
    tp1: TPLevel
    tp2: TPLevel
    tp3: TPLevel
    position_size: float
    leverage: int
    margin_required: float
    direction: str
    risk_amount: float
    total_risk_reward: float
    trailing_stop_state: TrailingStopState = TrailingStopState.INITIAL
    current_trailing_sl: Optional[float] = None
    confidence_score: float = 0.0
    reasoning: str = ""


@dataclass
class TPAllocation:
    """Take profit allocation configuration"""
    tp1_percent: float = 40.0
    tp2_percent: float = 35.0
    tp3_percent: float = 25.0
    
    def validate(self) -> bool:
        """Validate allocations sum to 100%"""
        total = self.tp1_percent + self.tp2_percent + self.tp3_percent
        return abs(total - 100.0) < 0.01


class AIPositionEngine:
    """
    AI-driven position calculation engine with dynamic TP/SL management
    
    Features:
    - ATR-based dynamic stop loss with volatility and AI adjustments
    - Multi-target take profits with configurable risk-reward ratios
    - Intelligent position sizing based on risk management
    - Volatility-adaptive leverage calculation
    - Trailing stop management with progressive SL movement
    """
    
    def __init__(
        self,
        min_leverage: int = 2,
        max_leverage: int = 20,
        base_leverage: int = 5,
        default_risk_percent: float = 2.0,
        max_position_percent: float = 50.0,
        atr_sl_multiplier: float = 1.5,
        volatility_low_threshold: float = 0.5,
        volatility_high_threshold: float = 2.0,
        tp_allocation: Optional[TPAllocation] = None
    ):
        """
        Initialize the AI Position Engine
        
        Args:
            min_leverage: Minimum allowed leverage
            max_leverage: Maximum allowed leverage
            base_leverage: Default leverage for neutral conditions
            default_risk_percent: Default risk per trade as percentage
            max_position_percent: Maximum position size as percentage of balance
            atr_sl_multiplier: Base ATR multiplier for stop loss
            volatility_low_threshold: ATR% below this = low volatility
            volatility_high_threshold: ATR% above this = high volatility
            tp_allocation: Take profit allocation configuration
        """
        self.min_leverage = min_leverage
        self.max_leverage = max_leverage
        self.base_leverage = base_leverage
        self.default_risk_percent = default_risk_percent
        self.max_position_percent = max_position_percent / 100.0
        self.atr_sl_multiplier = atr_sl_multiplier
        self.vol_low = volatility_low_threshold
        self.vol_high = volatility_high_threshold
        
        self.tp_allocation = tp_allocation or TPAllocation()
        if not self.tp_allocation.validate():
            logger.warning("TP allocations don't sum to 100%, using defaults")
            self.tp_allocation = TPAllocation()
        
        self.volatility_leverage_scale = {
            0.3: 1.3,
            0.5: 1.1,
            1.0: 1.0,
            1.5: 0.7,
            2.0: 0.5,
            3.0: 0.3
        }
        
        self.market_structure_adjustments = {
            'trending': {'sl_mult': 1.2, 'tp_mult': 1.3},
            'ranging': {'sl_mult': 0.9, 'tp_mult': 0.85},
            'volatile': {'sl_mult': 1.5, 'tp_mult': 1.1},
            'breakout': {'sl_mult': 1.8, 'tp_mult': 1.5},
            'consolidation': {'sl_mult': 0.8, 'tp_mult': 0.8}
        }
        
        logger.info(
            f"AIPositionEngine initialized: leverage={min_leverage}x-{max_leverage}x, "
            f"TP allocation: {self.tp_allocation.tp1_percent}%/{self.tp_allocation.tp2_percent}%/{self.tp_allocation.tp3_percent}%"
        )
    
    def calculate_dynamic_sl(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        volatility_score: float,
        ai_adjustment: float = 0.0,
        market_structure: str = "ranging"
    ) -> Tuple[float, str]:
        """
        Calculate dynamic stop loss based on ATR, volatility, and market structure
        
        Args:
            entry_price: Trade entry price
            direction: "LONG" or "SHORT"
            atr: Average True Range value
            volatility_score: Volatility score (0-1, where 1 is low volatility)
            ai_adjustment: AI-based adjustment factor (-0.5 to 0.5)
            market_structure: Market structure type
            
        Returns:
            Tuple of (stop_loss_price, reasoning)
        """
        try:
            ai_adjustment = max(-0.5, min(0.5, ai_adjustment))
            
            structure_adj = self.market_structure_adjustments.get(
                market_structure, 
                self.market_structure_adjustments['ranging']
            )
            
            volatility_multiplier = 1.0
            if volatility_score < 0.3:
                volatility_multiplier = 1.4
            elif volatility_score < 0.5:
                volatility_multiplier = 1.2
            elif volatility_score > 0.8:
                volatility_multiplier = 0.8
            
            adjusted_multiplier = (
                self.atr_sl_multiplier * 
                structure_adj['sl_mult'] * 
                volatility_multiplier * 
                (1.0 + ai_adjustment)
            )
            
            sl_distance = atr * adjusted_multiplier
            
            direction_upper = direction.upper()
            if direction_upper == "LONG":
                stop_loss = entry_price - sl_distance
            elif direction_upper == "SHORT":
                stop_loss = entry_price + sl_distance
            else:
                raise ValueError(f"Invalid direction: {direction}")
            
            min_sl_distance = entry_price * 0.005
            actual_distance = abs(entry_price - stop_loss)
            min_enforced = False
            
            if actual_distance < min_sl_distance:
                logger.info(f"SL distance {actual_distance:.4f} ({actual_distance/entry_price*100:.3f}%) below minimum 0.5%, enforcing minimum distance")
                min_enforced = True
                if direction_upper == "LONG":
                    stop_loss = entry_price - min_sl_distance
                else:
                    stop_loss = entry_price + min_sl_distance
            
            sl_percent = abs(entry_price - stop_loss) / entry_price * 100
            
            reasoning = (
                f"SL calculated: ATR={atr:.6f}, multiplier={adjusted_multiplier:.2f}, "
                f"volatility={volatility_score:.2f}, structure={market_structure}, "
                f"AI adj={ai_adjustment:.2f}, distance={sl_percent:.2f}%"
                f"{' (min enforced)' if min_enforced else ''}"
            )
            
            logger.debug(reasoning)
            
            return stop_loss, reasoning
            
        except Exception as e:
            logger.error(f"Error calculating dynamic SL: {e}")
            fallback_distance = entry_price * 0.02
            if direction.upper() == "LONG":
                return entry_price - fallback_distance, f"Fallback SL: {e}"
            return entry_price + fallback_distance, f"Fallback SL: {e}"
    
    def calculate_multi_tp(
        self,
        entry_price: float,
        stop_loss: float,
        direction: str,
        risk_reward_ratios: Optional[List[float]] = None,
        market_structure: str = "ranging"
    ) -> List[TPLevel]:
        """
        Calculate multi-target take profits with allocation percentages
        
        Args:
            entry_price: Trade entry price
            stop_loss: Stop loss price
            direction: "LONG" or "SHORT"
            risk_reward_ratios: Risk-reward ratios for TP1, TP2, TP3
            market_structure: Market structure for TP adjustment
            
        Returns:
            List of TPLevel objects
        """
        try:
            if risk_reward_ratios is None:
                risk_reward_ratios = [1.0, 2.0, 3.0]
            
            if len(risk_reward_ratios) < 3:
                risk_reward_ratios = risk_reward_ratios + [3.0] * (3 - len(risk_reward_ratios))
            
            sl_distance = abs(entry_price - stop_loss)
            
            structure_adj = self.market_structure_adjustments.get(
                market_structure,
                self.market_structure_adjustments['ranging']
            )
            tp_multiplier = structure_adj['tp_mult']
            
            direction_upper = direction.upper()
            tp_levels = []
            
            allocations = [
                self.tp_allocation.tp1_percent,
                self.tp_allocation.tp2_percent,
                self.tp_allocation.tp3_percent
            ]
            
            min_sl_distance = entry_price * 0.005
            if sl_distance < min_sl_distance:
                logger.info(f"SL distance {sl_distance:.4f} too small for TP calculation, using minimum {min_sl_distance:.4f}")
                sl_distance = min_sl_distance
            
            min_rr_floors = [1.0, 2.0, 3.0]
            
            for i, (rr, alloc) in enumerate(zip(risk_reward_ratios[:3], allocations)):
                adjusted_rr = rr * tp_multiplier
                
                min_rr = min_rr_floors[i]
                if adjusted_rr < min_rr:
                    logger.debug(f"TP{i+1} R:R {adjusted_rr:.2f} below minimum {min_rr:.1f}R, enforcing floor")
                    adjusted_rr = min_rr
                
                tp_distance = sl_distance * adjusted_rr
                
                if direction_upper == "LONG":
                    tp_price = entry_price + tp_distance
                else:
                    tp_price = entry_price - tp_distance
                
                tp_levels.append(TPLevel(
                    price=tp_price,
                    allocation_percent=alloc,
                    risk_reward=adjusted_rr,
                    hit=False
                ))
            
            logger.debug(
                f"Multi-TP calculated: TP1={tp_levels[0].price:.4f} ({tp_levels[0].risk_reward:.1f}R), "
                f"TP2={tp_levels[1].price:.4f} ({tp_levels[1].risk_reward:.1f}R), "
                f"TP3={tp_levels[2].price:.4f} ({tp_levels[2].risk_reward:.1f}R)"
            )
            
            return tp_levels
            
        except Exception as e:
            logger.error(f"Error calculating multi-TP: {e}")
            fallback_tp = entry_price * (1.02 if direction.upper() == "LONG" else 0.98)
            return [
                TPLevel(price=fallback_tp, allocation_percent=40.0, risk_reward=1.0),
                TPLevel(price=fallback_tp * (1.02 if direction.upper() == "LONG" else 0.98), allocation_percent=35.0, risk_reward=2.0),
                TPLevel(price=fallback_tp * (1.04 if direction.upper() == "LONG" else 0.96), allocation_percent=25.0, risk_reward=3.0)
            ]
    
    def calculate_position_size(
        self,
        account_balance: float,
        risk_percent: float,
        entry_price: float,
        stop_loss: float,
        leverage: int = 1
    ) -> Tuple[float, float, float]:
        """
        Calculate position size based on risk percentage and account balance
        
        Args:
            account_balance: Available account balance in quote currency
            risk_percent: Risk per trade as percentage (e.g., 2.0 for 2%)
            entry_price: Trade entry price
            stop_loss: Stop loss price
            leverage: Applied leverage
            
        Returns:
            Tuple of (position_size_in_base, position_value, risk_amount)
        """
        try:
            risk_decimal = risk_percent / 100.0
            risk_amount = account_balance * risk_decimal
            
            sl_distance = abs(entry_price - stop_loss)
            sl_percent = sl_distance / entry_price
            
            if sl_percent <= 0:
                logger.warning("Invalid SL distance, using fallback")
                sl_percent = 0.02
            
            position_value_from_risk = risk_amount / sl_percent
            
            max_position_value = account_balance * self.max_position_percent * leverage
            
            position_value = min(position_value_from_risk, max_position_value)
            
            position_size = position_value / entry_price
            
            logger.debug(
                f"Position size calculated: {position_size:.6f} units, "
                f"value=${position_value:.2f}, risk=${risk_amount:.2f}"
            )
            
            return position_size, position_value, risk_amount
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            fallback_size = (account_balance * 0.01) / entry_price
            return fallback_size, fallback_size * entry_price, account_balance * 0.01
    
    def calculate_optimal_leverage(
        self,
        volatility_score: float,
        signal_confidence: float,
        min_lev: Optional[int] = None,
        max_lev: Optional[int] = None
    ) -> Tuple[int, str]:
        """
        Calculate optimal leverage based on volatility and signal confidence
        
        Args:
            volatility_score: Volatility score (0-1, where 1 is low volatility/safer)
            signal_confidence: Signal confidence score (0-1)
            min_lev: Minimum leverage override
            max_lev: Maximum leverage override
            
        Returns:
            Tuple of (optimal_leverage, reasoning)
        """
        try:
            min_lev = min_lev or self.min_leverage
            max_lev = max_lev or self.max_leverage
            
            vol_multiplier = 1.0
            for threshold, mult in sorted(self.volatility_leverage_scale.items()):
                if (1.0 - volatility_score) * 3 <= threshold:
                    vol_multiplier = mult
                    break
            else:
                vol_multiplier = 0.3
            
            confidence_multiplier = 0.5 + (signal_confidence * 0.5)
            
            raw_leverage = self.base_leverage * vol_multiplier * confidence_multiplier
            
            optimal_leverage = int(round(raw_leverage))
            optimal_leverage = max(min_lev, min(max_lev, optimal_leverage))
            
            reasoning = (
                f"Leverage={optimal_leverage}x: vol_score={volatility_score:.2f} "
                f"(mult={vol_multiplier:.2f}), confidence={signal_confidence:.2f} "
                f"(mult={confidence_multiplier:.2f})"
            )
            
            logger.debug(reasoning)
            
            return optimal_leverage, reasoning
            
        except Exception as e:
            logger.error(f"Error calculating optimal leverage: {e}")
            return self.min_leverage, f"Fallback leverage: {e}"
    
    def calculate_margin_required(
        self,
        position_value: float,
        leverage: int
    ) -> float:
        """
        Calculate margin required for a position
        
        Args:
            position_value: Total position value
            leverage: Applied leverage
            
        Returns:
            Required margin amount
        """
        try:
            if leverage <= 0:
                leverage = 1
            return position_value / leverage
        except Exception as e:
            logger.error(f"Error calculating margin: {e}")
            return position_value
    
    def get_complete_trade_setup(
        self,
        signal: Dict[str, Any],
        account_balance: float,
        market_data: Dict[str, Any]
    ) -> TradeSetup:
        """
        Generate complete trade setup from signal and market data
        
        Args:
            signal: Signal dictionary with direction, entry price, indicators
            market_data: Market data with ATR, volatility metrics
            
        Returns:
            Complete TradeSetup with all parameters
        """
        try:
            entry_price = signal.get('entry_price', signal.get('price', 0))
            direction = signal.get('direction', 'LONG').upper()
            
            if entry_price <= 0:
                raise ValueError("Invalid entry price")
            
            atr = market_data.get('atr', entry_price * 0.01)
            atr_percent = market_data.get('atr_percent', 1.0)
            
            if atr < entry_price * 0.001:
                logger.info(f"ATR {atr:.6f} seems too small (< 0.1% of entry), recalculating from atr_percent={atr_percent:.2f}%")
                atr = entry_price * (atr_percent / 100) if atr_percent > 0 else entry_price * 0.01
                logger.info(f"ATR adjusted to {atr:.6f}")
            
            volatility_score = market_data.get('volatility_score', 0.5)
            ai_adjustment = signal.get('ai_adjustment', 0.0)
            market_structure = market_data.get('market_structure', 'ranging')
            signal_confidence = signal.get('confidence', signal.get('signal_strength', 0.7))
            risk_percent = signal.get('risk_percent', self.default_risk_percent)
            rr_ratios = signal.get('risk_reward_ratios', [1.0, 2.0, 3.0])
            
            stop_loss, sl_reasoning = self.calculate_dynamic_sl(
                entry_price=entry_price,
                direction=direction,
                atr=atr,
                volatility_score=volatility_score,
                ai_adjustment=ai_adjustment,
                market_structure=market_structure
            )
            
            tp_levels = self.calculate_multi_tp(
                entry_price=entry_price,
                stop_loss=stop_loss,
                direction=direction,
                risk_reward_ratios=rr_ratios,
                market_structure=market_structure
            )
            
            optimal_leverage, lev_reasoning = self.calculate_optimal_leverage(
                volatility_score=volatility_score,
                signal_confidence=signal_confidence
            )
            
            position_size, position_value, risk_amount = self.calculate_position_size(
                account_balance=account_balance,
                risk_percent=risk_percent,
                entry_price=entry_price,
                stop_loss=stop_loss,
                leverage=optimal_leverage
            )
            
            margin_required = self.calculate_margin_required(
                position_value=position_value,
                leverage=optimal_leverage
            )
            
            weighted_rr = sum(
                tp.risk_reward * (tp.allocation_percent / 100)
                for tp in tp_levels
            )
            
            overall_confidence = (volatility_score + signal_confidence) / 2
            
            reasoning = f"{sl_reasoning} | {lev_reasoning}"
            
            trade_setup = TradeSetup(
                entry_price=entry_price,
                stop_loss=stop_loss,
                tp1=tp_levels[0],
                tp2=tp_levels[1],
                tp3=tp_levels[2],
                position_size=position_size,
                leverage=optimal_leverage,
                margin_required=margin_required,
                direction=direction,
                risk_amount=risk_amount,
                total_risk_reward=weighted_rr,
                trailing_stop_state=TrailingStopState.INITIAL,
                current_trailing_sl=stop_loss,
                confidence_score=overall_confidence,
                reasoning=reasoning
            )
            
            logger.info(
                f"Trade setup generated: {direction} @ {entry_price:.4f}, "
                f"SL={stop_loss:.4f}, TP1={tp_levels[0].price:.4f}, "
                f"TP2={tp_levels[1].price:.4f}, TP3={tp_levels[2].price:.4f}, "
                f"Size={position_size:.6f}, Leverage={optimal_leverage}x, "
                f"Margin=${margin_required:.2f}"
            )
            
            return trade_setup
            
        except Exception as e:
            logger.error(f"Error generating complete trade setup: {e}")
            raise
    
    def update_trailing_stop(
        self,
        trade_setup: TradeSetup,
        current_price: float,
        tp_hit: int = 0
    ) -> TradeSetup:
        """
        Update trailing stop based on TP hits
        
        Trailing stop logic:
        - Initial: SL at original stop loss
        - After TP1 hit: Move SL to entry (breakeven)
        - After TP2 hit: Move SL to TP1 price
        
        Args:
            trade_setup: Current trade setup
            current_price: Current market price
            tp_hit: Which TP was just hit (1, 2, or 3)
            
        Returns:
            Updated TradeSetup with new trailing stop
        """
        try:
            direction = trade_setup.direction.upper()
            
            if tp_hit == 1 and trade_setup.trailing_stop_state == TrailingStopState.INITIAL:
                trade_setup.tp1.hit = True
                trade_setup.current_trailing_sl = trade_setup.entry_price
                trade_setup.trailing_stop_state = TrailingStopState.AT_ENTRY
                
                logger.info(
                    f"TP1 hit - Moving SL to entry (breakeven): {trade_setup.entry_price:.4f}"
                )
            
            elif tp_hit == 2 and trade_setup.trailing_stop_state in [
                TrailingStopState.INITIAL, 
                TrailingStopState.AT_ENTRY
            ]:
                trade_setup.tp1.hit = True
                trade_setup.tp2.hit = True
                trade_setup.current_trailing_sl = trade_setup.tp1.price
                trade_setup.trailing_stop_state = TrailingStopState.AT_TP1
                
                logger.info(
                    f"TP2 hit - Moving SL to TP1: {trade_setup.tp1.price:.4f}"
                )
            
            elif tp_hit == 3:
                trade_setup.tp1.hit = True
                trade_setup.tp2.hit = True
                trade_setup.tp3.hit = True
                trade_setup.current_trailing_sl = trade_setup.tp2.price
                trade_setup.trailing_stop_state = TrailingStopState.AT_TP2
                
                logger.info(
                    f"TP3 hit - Trade fully closed at {trade_setup.tp3.price:.4f}"
                )
            
            return trade_setup
            
        except Exception as e:
            logger.error(f"Error updating trailing stop: {e}")
            return trade_setup
    
    def check_tp_hit(
        self,
        trade_setup: TradeSetup,
        current_price: float
    ) -> int:
        """
        Check which TP level was hit by current price
        
        Args:
            trade_setup: Current trade setup
            current_price: Current market price
            
        Returns:
            TP level hit (0 if none, 1/2/3 for respective TP)
        """
        try:
            direction = trade_setup.direction.upper()
            
            if direction == "LONG":
                if not trade_setup.tp3.hit and current_price >= trade_setup.tp3.price:
                    return 3
                elif not trade_setup.tp2.hit and current_price >= trade_setup.tp2.price:
                    return 2
                elif not trade_setup.tp1.hit and current_price >= trade_setup.tp1.price:
                    return 1
            else:
                if not trade_setup.tp3.hit and current_price <= trade_setup.tp3.price:
                    return 3
                elif not trade_setup.tp2.hit and current_price <= trade_setup.tp2.price:
                    return 2
                elif not trade_setup.tp1.hit and current_price <= trade_setup.tp1.price:
                    return 1
            
            return 0
            
        except Exception as e:
            logger.error(f"Error checking TP hit: {e}")
            return 0
    
    def check_sl_hit(
        self,
        trade_setup: TradeSetup,
        current_price: float
    ) -> bool:
        """
        Check if stop loss was hit
        
        Args:
            trade_setup: Current trade setup
            current_price: Current market price
            
        Returns:
            True if SL hit, False otherwise
        """
        try:
            direction = trade_setup.direction.upper()
            trailing_sl = trade_setup.current_trailing_sl or trade_setup.stop_loss
            
            if direction == "LONG":
                return current_price <= trailing_sl
            else:
                return current_price >= trailing_sl
            
        except Exception as e:
            logger.error(f"Error checking SL hit: {e}")
            return False
    
    def get_remaining_position(
        self,
        trade_setup: TradeSetup
    ) -> Tuple[float, float]:
        """
        Calculate remaining position size after TP hits
        
        Args:
            trade_setup: Current trade setup
            
        Returns:
            Tuple of (remaining_percent, remaining_size)
        """
        try:
            closed_percent = 0.0
            
            if trade_setup.tp1.hit:
                closed_percent += trade_setup.tp1.allocation_percent
            if trade_setup.tp2.hit:
                closed_percent += trade_setup.tp2.allocation_percent
            if trade_setup.tp3.hit:
                closed_percent += trade_setup.tp3.allocation_percent
            
            remaining_percent = 100.0 - closed_percent
            remaining_size = trade_setup.position_size * (remaining_percent / 100.0)
            
            return remaining_percent, remaining_size
            
        except Exception as e:
            logger.error(f"Error calculating remaining position: {e}")
            return 100.0, trade_setup.position_size
    
    def format_trade_summary(self, trade_setup: TradeSetup) -> str:
        """
        Format trade setup as human-readable summary
        
        Args:
            trade_setup: Trade setup to format
            
        Returns:
            Formatted summary string
        """
        try:
            sl_distance = abs(trade_setup.entry_price - trade_setup.stop_loss)
            sl_percent = (sl_distance / trade_setup.entry_price) * 100
            
            summary = f"""
╔═══════════════════════════════════════════╗
║         AI TRADE SETUP SUMMARY            ║
╠═══════════════════════════════════════════╣
║ Direction: {trade_setup.direction:>8}                     ║
║ Entry:     {trade_setup.entry_price:>12.4f}                ║
║ Stop Loss: {trade_setup.stop_loss:>12.4f} ({sl_percent:.2f}%)        ║
╠═══════════════════════════════════════════╣
║ TP1: {trade_setup.tp1.price:>10.4f} ({trade_setup.tp1.risk_reward:.1f}R) → {trade_setup.tp1.allocation_percent:.0f}% ║
║ TP2: {trade_setup.tp2.price:>10.4f} ({trade_setup.tp2.risk_reward:.1f}R) → {trade_setup.tp2.allocation_percent:.0f}% ║
║ TP3: {trade_setup.tp3.price:>10.4f} ({trade_setup.tp3.risk_reward:.1f}R) → {trade_setup.tp3.allocation_percent:.0f}% ║
╠═══════════════════════════════════════════╣
║ Position:  {trade_setup.position_size:>12.6f}                ║
║ Leverage:  {trade_setup.leverage:>12}x               ║
║ Margin:    ${trade_setup.margin_required:>11.2f}                ║
║ Risk:      ${trade_setup.risk_amount:>11.2f}                ║
║ RR Ratio:  {trade_setup.total_risk_reward:>12.2f}                ║
║ Confidence:{trade_setup.confidence_score:>12.2f}                ║
╚═══════════════════════════════════════════╝
"""
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Error formatting trade summary: {e}")
            return f"Trade Setup: {trade_setup.direction} @ {trade_setup.entry_price}"
