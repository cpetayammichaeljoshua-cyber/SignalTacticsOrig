#!/usr/bin/env python3
"""
Comprehensive Trading Dashboard
Real-time visualization of all market intelligence
"""

from typing import Dict, Optional
from datetime import datetime

from SignalMaestro.market_data_contracts import MarketIntelSnapshot, FusedSignal, AnalyzerType

class ComprehensiveDashboard:
    """
    Formats market intelligence for display
    """
    
    @staticmethod
    def format_intel_snapshot(intel: MarketIntelSnapshot) -> str:
        """Format intelligence snapshot for display"""
        lines = []
        
        lines.append("=" * 80)
        lines.append(f"📊 COMPREHENSIVE MARKET INTELLIGENCE - {intel.symbol}")
        lines.append("=" * 80)
        lines.append(f"🕐 Timestamp: {intel.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Overall Assessment
        lines.append("🎯 OVERALL ASSESSMENT")
        lines.append("-" * 80)
        lines.append(f"   Consensus Bias: {intel.consensus_bias.value.upper()} ({intel.consensus_confidence:.1f}% confidence)")
        lines.append(f"   Overall Score: {intel.overall_score:.1f}/100")
        lines.append(f"   Signal Strength: {intel.get_signal_strength().value.upper()}")
        lines.append(f"   Risk Level: {intel.risk_level.upper()}")
        lines.append(f"   Trade Status: {'✅ TRADEABLE' if intel.should_trade() else '❌ DO NOT TRADE'}")
        lines.append("")
        
        # Analyzer Results
        lines.append("🔬 ANALYZER RESULTS")
        lines.append("-" * 80)
        
        for analyzer_type, result in intel.analyzer_results.items():
            bias_emoji = "🟢" if result.bias.value == "bullish" else ("🔴" if result.bias.value == "bearish" else "⚪")
            lines.append(f"   {bias_emoji} {analyzer_type.value.upper()}")
            lines.append(f"      Score: {result.score:.1f}/100 | Bias: {result.bias.value} | Confidence: {result.confidence:.1f}%")
            
            if result.veto_flags:
                lines.append(f"      ⚠️  Veto: {', '.join(result.veto_flags[:2])}")
        
        lines.append("")
        
        # Key Signals
        if intel.dominant_signals:
            lines.append("🎪 DOMINANT SIGNALS")
            lines.append("-" * 80)
            for i, signal in enumerate(intel.dominant_signals[:5], 1):
                lines.append(f"   {i}. [{signal.get('analyzer', 'unknown').upper()}] {signal.get('type', 'signal')}")
                if 'value' in signal:
                    lines.append(f"      Value: {signal['value']}")
            lines.append("")
        
        # Critical Levels
        if intel.critical_levels:
            lines.append("📍 CRITICAL PRICE LEVELS")
            lines.append("-" * 80)
            for i, level in enumerate(intel.critical_levels[:5], 1):
                lines.append(f"   {i}. ${level.get('price', 0):.4f} - {level.get('type', 'unknown')} (Strength: {level.get('strength', 0):.0f})")
            lines.append("")
        
        # Veto Flags
        if intel.veto_reasons:
            lines.append("⚠️  VETO FLAGS")
            lines.append("-" * 80)
            for i, reason in enumerate(intel.veto_reasons[:5], 1):
                lines.append(f"   {i}. {reason}")
            lines.append("")
        
        # Recommendations
        lines.append("💡 TRADING RECOMMENDATIONS")
        lines.append("-" * 80)
        if intel.recommended_entry:
            lines.append(f"   Entry: ${intel.recommended_entry:.4f}")
        if intel.recommended_stop:
            lines.append(f"   Stop Loss: ${intel.recommended_stop:.4f}")
        if intel.recommended_targets:
            lines.append(f"   Take Profits: {', '.join([f'${tp:.4f}' for tp in intel.recommended_targets])}")
        if intel.recommended_leverage:
            lines.append(f"   Leverage: {intel.recommended_leverage}x")
        lines.append("")
        
        # Performance
        lines.append("⚡ PERFORMANCE METRICS")
        lines.append("-" * 80)
        lines.append(f"   Analyzers Active: {intel.analyzers_active}")
        lines.append(f"   Analyzers Failed: {intel.analyzers_failed}")
        lines.append(f"   Processing Time: {intel.total_processing_time_ms:.0f}ms")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    @staticmethod
    def format_fused_signal(signal: FusedSignal) -> str:
        """Format fused signal for display"""
        lines = []
        
        direction_emoji = "🚀" if signal.direction == "LONG" else "📉"
        
        lines.append("=" * 80)
        lines.append(f"{direction_emoji} FUSED TRADING SIGNAL - {signal.symbol}")
        lines.append("=" * 80)
        lines.append(f"🆔 Signal ID: {signal.signal_id}")
        lines.append(f"🕐 Generated: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"⏰ Expires: {signal.expiry_timestamp.strftime('%Y-%m-%d %H:%M:%S') if signal.expiry_timestamp else 'N/A'}")
        lines.append("")
        
        # Trade Details
        lines.append(f"📈 TRADE DETAILS")
        lines.append("-" * 80)
        lines.append(f"   Direction: {signal.direction}")
        lines.append(f"   Entry Price: ${signal.entry_price:.4f}")
        lines.append(f"   Stop Loss: ${signal.stop_loss:.4f}")
        lines.append("")
        
        lines.append("   Take Profit Levels:")
        for i, tp in enumerate(signal.take_profit_levels, 1):
            lines.append(f"      TP{i}: ${tp:.4f}")
        lines.append("")
        
        # Risk Management
        lines.append("⚡ RISK MANAGEMENT")
        lines.append("-" * 80)
        lines.append(f"   Recommended Leverage: {signal.recommended_leverage}x")
        lines.append(f"   Risk/Reward Ratio: 1:{signal.risk_reward_ratio:.2f}")
        lines.append(f"   Confidence: {signal.confidence:.1f}%")
        lines.append(f"   Signal Strength: {signal.strength.value.upper()}")
        lines.append("")
        
        # Reasoning
        lines.append("💭 REASONING")
        lines.append("-" * 80)
        lines.append(f"   Primary: {signal.primary_reason}")
        lines.append("   Supporting Factors:")
        for i, factor in enumerate(signal.supporting_factors, 1):
            lines.append(f"      {i}. {factor}")
        lines.append("")
        
        # Intelligence Summary
        intel = signal.intel_snapshot
        lines.append("🔬 INTELLIGENCE SUMMARY")
        lines.append("-" * 80)
        lines.append(f"   Overall Score: {intel.overall_score:.1f}/100")
        lines.append(f"   Consensus: {intel.consensus_bias.value} ({intel.consensus_confidence:.1f}%)")
        lines.append(f"   Risk Level: {intel.risk_level}")
        lines.append(f"   Active Analyzers: {intel.analyzers_active}")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    @staticmethod
    def format_telegram_signal(signal: FusedSignal) -> str:
        """Format signal for Telegram (Cornix compatible)"""
        direction_emoji = "🚀" if signal.direction == "LONG" else "📉"
        
        lines = []
        lines.append(f"{direction_emoji} **{signal.symbol} {signal.direction}** {direction_emoji}")
        lines.append("")
        lines.append(f"**Confidence:** {signal.confidence:.0f}% | **Strength:** {signal.strength.value.upper()}")
        lines.append(f"**Leverage:** {signal.recommended_leverage}x | **R:R:** 1:{signal.risk_reward_ratio:.2f}")
        lines.append("")
        lines.append(f"**Entry:** `{signal.entry_price:.4f}`")
        lines.append(f"**Stop Loss:** `{signal.stop_loss:.4f}`")
        lines.append("")
        lines.append("**Take Profit Targets:**")
        for i, tp in enumerate(signal.take_profit_levels, 1):
            lines.append(f"  TP{i}: `{tp:.4f}`")
        lines.append("")
        lines.append(f"💡 {signal.primary_reason}")
        lines.append("")
        lines.append(f"🔬 Intelligence: {signal.intel_snapshot.overall_score:.0f}/100")
        lines.append(f"⚡ Generated: {signal.timestamp.strftime('%H:%M:%S')}")
        lines.append("")
        lines.append("⚠️ **Risk Management:** Use proper position sizing")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_compact_status(intel: MarketIntelSnapshot) -> str:
        """Compact one-line status"""
        return (f"{intel.symbol} | "
                f"Score: {intel.overall_score:.0f}/100 | "
                f"{intel.consensus_bias.value.upper()} {intel.consensus_confidence:.0f}% | "
                f"Risk: {intel.risk_level} | "
                f"{'✅ TRADE' if intel.should_trade() else '❌ WAIT'}")
