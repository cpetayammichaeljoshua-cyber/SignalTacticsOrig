#!/usr/bin/env python3
"""
Signal Fusion Engine
Combines Ichimoku signals with market intelligence to produce final trade decisions
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from SignalMaestro.market_data_contracts import (
    MarketIntelSnapshot, FusedSignal, SignalStrength, MarketBias
)
from SignalMaestro.ichimoku_sniper_strategy import IchimokuSignal

class SignalFusionEngine:
    """
    Fuses Ichimoku strategy signals with comprehensive market intelligence
    to produce final, high-confidence trading signals
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Fusion configuration
        self.min_intel_score = 60  # Minimum intelligence score
        self.min_consensus_confidence = 65  # Minimum consensus confidence
        self.max_risk_level = "moderate"  # Maximum acceptable risk
        
        # Signal tracking
        self.generated_signals: List[FusedSignal] = []
        self.signal_counter = 0
        
    def fuse_signal(self, ichimoku_signal: Optional[IchimokuSignal],
                    intel_snapshot: MarketIntelSnapshot,
                    current_price: float) -> Optional[FusedSignal]:
        """
        Fuse Ichimoku signal with market intelligence
        
        Args:
            ichimoku_signal: Signal from Ichimoku strategy (can be None)
            intel_snapshot: Market intelligence snapshot
            current_price: Current market price
            
        Returns:
            FusedSignal if all conditions met, None otherwise
        """
        self.logger.info("=" * 70)
        self.logger.info("🔬 SIGNAL FUSION PROCESS")
        self.logger.info("=" * 70)
        
        # Step 1: Check if intelligence allows trading
        if not intel_snapshot.should_trade():
            self.logger.info("❌ Intelligence veto: Not favorable for trading")
            self.logger.info(f"   Score: {intel_snapshot.overall_score:.1f}/100")
            self.logger.info(f"   Confidence: {intel_snapshot.consensus_confidence:.1f}%")
            self.logger.info(f"   Veto count: {intel_snapshot.total_veto_count}")
            if intel_snapshot.veto_reasons:
                self.logger.info(f"   Veto reasons: {', '.join(intel_snapshot.veto_reasons[:3])}")
            return None
        
        self.logger.info("✅ Intelligence check passed")
        self.logger.info(f"   Score: {intel_snapshot.overall_score:.1f}/100")
        self.logger.info(f"   Consensus: {intel_snapshot.consensus_bias.value} ({intel_snapshot.consensus_confidence:.1f}%)")
        
        # Step 2: Determine trade direction
        if ichimoku_signal:
            # Use Ichimoku direction
            direction = ichimoku_signal.direction
            self.logger.info(f"✅ Ichimoku signal: {direction}")
        else:
            # Use intelligence consensus
            if intel_snapshot.consensus_bias == MarketBias.BULLISH:
                direction = "LONG"
            elif intel_snapshot.consensus_bias == MarketBias.BEARISH:
                direction = "SHORT"
            else:
                self.logger.info("❌ No clear direction from intelligence")
                return None
            
            self.logger.info(f"✅ Intelligence direction: {direction}")
        
        # Step 3: Verify intelligence agrees with direction
        if not self._intelligence_agrees(direction, intel_snapshot):
            self.logger.info("❌ Intelligence disagrees with signal direction")
            return None
        
        self.logger.info("✅ Intelligence confirms direction")
        
        # Step 4: Calculate confidence
        confidence = self._calculate_fused_confidence(ichimoku_signal, intel_snapshot)
        
        if confidence < 65:
            self.logger.info(f"❌ Fused confidence too low: {confidence:.1f}%")
            return None
        
        self.logger.info(f"✅ Fused confidence: {confidence:.1f}%")
        
        # Step 5: Determine entry and risk management
        entry_price = self._calculate_entry(ichimoku_signal, intel_snapshot, current_price, direction)
        stop_loss = self._calculate_stop_loss(ichimoku_signal, intel_snapshot, entry_price, direction)
        take_profits = self._calculate_take_profits(ichimoku_signal, intel_snapshot, entry_price, direction)
        
        # Step 6: Calculate leverage and risk/reward
        leverage = self._calculate_leverage(intel_snapshot, confidence)
        risk_reward = self._calculate_risk_reward(entry_price, stop_loss, take_profits)
        
        # Step 7: Build signal
        signal_strength = intel_snapshot.get_signal_strength()
        
        # Generate reasons
        primary_reason, supporting_factors = self._generate_reasons(
            ichimoku_signal, intel_snapshot, direction
        )
        
        # Generate signal ID
        self.signal_counter += 1
        signal_id = f"FUSED-{intel_snapshot.symbol}-{self.signal_counter:04d}"
        
        # Expiry (30 minutes for 30m timeframe)
        expiry = datetime.now() + timedelta(minutes=30)
        
        fused_signal = FusedSignal(
            symbol=intel_snapshot.symbol,
            timestamp=datetime.now(),
            direction=direction,
            confidence=confidence,
            strength=signal_strength,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_levels=take_profits,
            recommended_leverage=leverage,
            risk_reward_ratio=risk_reward,
            primary_reason=primary_reason,
            supporting_factors=supporting_factors,
            intel_snapshot=intel_snapshot,
            signal_id=signal_id,
            expiry_timestamp=expiry
        )
        
        self.generated_signals.append(fused_signal)
        
        self.logger.info("=" * 70)
        self.logger.info("✅ FUSED SIGNAL GENERATED")
        self.logger.info("=" * 70)
        self.logger.info(f"   ID: {signal_id}")
        self.logger.info(f"   Direction: {direction}")
        self.logger.info(f"   Entry: ${entry_price:.4f}")
        self.logger.info(f"   Stop Loss: ${stop_loss:.4f}")
        self.logger.info(f"   Take Profits: {[f'${tp:.4f}' for tp in take_profits]}")
        self.logger.info(f"   Leverage: {leverage}x")
        self.logger.info(f"   Risk/Reward: 1:{risk_reward:.2f}")
        self.logger.info(f"   Confidence: {confidence:.1f}%")
        self.logger.info(f"   Strength: {signal_strength.value}")
        self.logger.info("=" * 70)
        
        return fused_signal
    
    def _intelligence_agrees(self, direction: str, intel_snapshot: MarketIntelSnapshot) -> bool:
        """Check if intelligence consensus agrees with direction"""
        if direction == "LONG" and intel_snapshot.consensus_bias == MarketBias.BEARISH:
            return False
        if direction == "SHORT" and intel_snapshot.consensus_bias == MarketBias.BULLISH:
            return False
        return True
    
    def _calculate_fused_confidence(self, ichimoku_signal: Optional[IchimokuSignal],
                                    intel_snapshot: MarketIntelSnapshot) -> float:
        """Calculate combined confidence from both sources"""
        ichimoku_confidence = ichimoku_signal.confidence if ichimoku_signal else 70
        intel_confidence = intel_snapshot.consensus_confidence
        
        # Weighted average (70% intelligence, 30% Ichimoku)
        fused = (intel_confidence * 0.7) + (ichimoku_confidence * 0.3)
        
        # Boost if both strongly agree
        if ichimoku_signal and ichimoku_confidence > 80 and intel_confidence > 80:
            fused = min(fused + 10, 100)
        
        return fused
    
    def _calculate_entry(self, ichimoku_signal: Optional[IchimokuSignal],
                        intel_snapshot: MarketIntelSnapshot,
                        current_price: float,
                        direction: str) -> float:
        """Calculate optimal entry price"""
        if ichimoku_signal and ichimoku_signal.entry_price:
            # Use Ichimoku entry
            return ichimoku_signal.entry_price
        
        # Use intelligence recommendation or current price
        if intel_snapshot.recommended_entry:
            return intel_snapshot.recommended_entry
        
        # Default to current price with small adjustment
        if direction == "LONG":
            return current_price * 1.0005  # 0.05% above
        else:
            return current_price * 0.9995  # 0.05% below
    
    def _calculate_stop_loss(self, ichimoku_signal: Optional[IchimokuSignal],
                            intel_snapshot: MarketIntelSnapshot,
                            entry_price: float,
                            direction: str) -> float:
        """Calculate stop loss"""
        if ichimoku_signal and ichimoku_signal.stop_loss:
            return ichimoku_signal.stop_loss
        
        if intel_snapshot.recommended_stop:
            return intel_snapshot.recommended_stop
        
        # Default stop loss (1% risk)
        if direction == "LONG":
            return entry_price * 0.99
        else:
            return entry_price * 1.01
    
    def _calculate_take_profits(self, ichimoku_signal: Optional[IchimokuSignal],
                                intel_snapshot: MarketIntelSnapshot,
                                entry_price: float,
                                direction: str) -> List[float]:
        """Calculate take profit levels"""
        if ichimoku_signal and ichimoku_signal.take_profit_levels:
            return ichimoku_signal.take_profit_levels
        
        if intel_snapshot.recommended_targets:
            return intel_snapshot.recommended_targets[:3]
        
        # Default: 1%, 2%, 3%
        if direction == "LONG":
            return [
                entry_price * 1.01,
                entry_price * 1.02,
                entry_price * 1.03
            ]
        else:
            return [
                entry_price * 0.99,
                entry_price * 0.98,
                entry_price * 0.97
            ]
    
    def _calculate_leverage(self, intel_snapshot: MarketIntelSnapshot, confidence: float) -> float:
        """Calculate recommended leverage"""
        if intel_snapshot.recommended_leverage:
            return intel_snapshot.recommended_leverage
        
        # Base on confidence and risk
        if confidence >= 85 and intel_snapshot.risk_level == "low":
            return 10
        elif confidence >= 75 and intel_snapshot.risk_level in ["low", "moderate"]:
            return 7
        elif confidence >= 65:
            return 5
        else:
            return 3
    
    def _calculate_risk_reward(self, entry: float, stop: float, 
                              take_profits: List[float]) -> float:
        """Calculate risk/reward ratio"""
        risk = abs(entry - stop)
        if risk == 0:
            return 0
        
        # Use first TP for R:R calculation
        reward = abs(take_profits[0] - entry)
        
        return reward / risk
    
    def _generate_reasons(self, ichimoku_signal: Optional[IchimokuSignal],
                         intel_snapshot: MarketIntelSnapshot,
                         direction: str) -> tuple:
        """Generate trading reasons"""
        primary_reason = f"{direction} signal with {intel_snapshot.consensus_confidence:.0f}% consensus"
        
        supporting_factors = []
        
        # Add Ichimoku reason
        if ichimoku_signal:
            supporting_factors.append(f"Ichimoku Sniper confirmation")
        
        # Add top intelligence signals
        for signal in intel_snapshot.dominant_signals[:3]:
            analyzer = signal.get('analyzer', 'unknown')
            sig_type = signal.get('type', 'signal')
            supporting_factors.append(f"{analyzer}: {sig_type}")
        
        # Add overall score
        supporting_factors.append(f"Overall intelligence score: {intel_snapshot.overall_score:.0f}/100")
        
        return primary_reason, supporting_factors[:5]  # Max 5 factors
    
    def get_recent_signals(self, limit: int = 10) -> List[FusedSignal]:
        """Get recent fused signals"""
        return self.generated_signals[-limit:]
    
    def clear_expired_signals(self):
        """Remove expired signals"""
        now = datetime.now()
        self.generated_signals = [
            sig for sig in self.generated_signals
            if sig.expiry_timestamp and sig.expiry_timestamp > now
        ]
