#!/usr/bin/env python3
"""
Market Intelligence Command Addon
New Telegram commands for market intelligence, insider trading, and order flow analysis
"""

# These methods should be added to FXSUSDTTelegramBot class

async def cmd_market_intelligence(self, update, context):
    """Display comprehensive market intelligence analysis"""
    chat_id = str(update.effective_chat.id)
    try:
        market_data = await self.trader.get_market_data('FXSUSDT', '1m', 200)
        if market_data is None or len(market_data) < 50:
            await self.send_message(chat_id, "❌ Insufficient market data for analysis")
            return
        
        mi_summary = await self.market_intelligence.get_market_intelligence_summary(market_data)
        
        msg = """📊 **MARKET INTELLIGENCE REPORT**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**VOLUME ANALYSIS:**
"""
        if mi_summary.get('volume'):
            vol = mi_summary['volume']
            msg += f"• Buy/Sell Ratio: {vol.get('buy_sell_ratio', 0):.2f}x\n"
            msg += f"• Volume Imbalance: {vol.get('imbalance', 0)*100:.1f}%\n"
            msg += f"• Trend: {vol.get('trend', 'stable').upper()}\n"
            msg += f"• Unusual: {'🔴 YES' if vol.get('unusual') else '🟢 NO'}\n"
        
        msg += "\n**MARKET STRUCTURE:**\n"
        if mi_summary.get('market_structure'):
            ms = mi_summary['market_structure']
            msg += f"• Direction: {ms.get('structure', 'unknown').upper()}\n"
            msg += f"• Support: {len(ms.get('support', []))} zones detected\n"
            msg += f"• Resistance: {len(ms.get('resistance', []))} zones detected\n"
        
        msg += "\n**INSTITUTIONAL SIGNALS:**\n"
        if mi_summary.get('institutional'):
            inst = mi_summary['institutional']
            msg += f"• Activity: {inst.get('activity', 'none').upper()}\n"
            msg += f"• Momentum: {inst.get('momentum_score', 0):.1f}\n"
            msg += f"• Trend Strength: {inst.get('trend_strength', 0):.1f}%\n"
        
        msg += "\n**VOLATILITY:** " + mi_summary.get('volatility', 'normal').upper() + "\n"
        msg += f"**SIGNAL:** {mi_summary.get('signal', 'neutral').upper()}\n"
        
        await self.send_message(chat_id, msg)
    except Exception as e:
        await self.send_message(chat_id, f"❌ Market intelligence error: {str(e)}")

async def cmd_insider_detection(self, update, context):
    """Detect insider/institutional trading activity"""
    chat_id = str(update.effective_chat.id)
    try:
        market_data = await self.trader.get_market_data('FXSUSDT', '1m', 200)
        if market_data is None or len(market_data) < 50:
            await self.send_message(chat_id, "❌ Insufficient data")
            return
        
        insider_signal = await self.insider_analyzer.detect_insider_activity(market_data)
        
        if insider_signal.detected:
            msg = f"""🐋 **INSIDER ACTIVITY DETECTED**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Type:** {insider_signal.activity_type.upper()}
**Confidence:** {insider_signal.confidence:.1f}%
**Strength:** {insider_signal.strength:.1f}

📝 {insider_signal.description}

🎯 {insider_signal.recommendation}"""
        else:
            msg = """🟢 **NO SIGNIFICANT INSIDER ACTIVITY**

Market is trading normally without institutional patterns.
Monitor for changes in volume and price action."""
        
        await self.send_message(chat_id, msg)
    except Exception as e:
        await self.send_message(chat_id, f"❌ Error: {str(e)}")

async def cmd_order_flow(self, update, context):
    """Display order flow analysis"""
    chat_id = str(update.effective_chat.id)
    try:
        market_data = await self.trader.get_market_data('FXSUSDT', '1m', 200)
        if market_data is None or len(market_data) < 50:
            await self.send_message(chat_id, "❌ Insufficient data")
            return
        
        current_price = await self.trader.get_current_price()
        order_flow = await self.smart_sltp.analyze_order_flow(market_data, current_price)
        
        msg = f"""📈 **ORDER FLOW ANALYSIS**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**FLOW DIRECTION:** {order_flow.direction.value.upper()}
**STRENGTH:** {order_flow.strength:.1f}%

**VOLUME METRICS:**
• Buy Volume: {order_flow.aggressive_buy_ratio*100:.1f}%
• Sell Volume: {order_flow.aggressive_sell_ratio*100:.1f}%
• Imbalance: {order_flow.volume_imbalance*100:+.1f}%

**DELTA:**
• Net: {order_flow.net_delta:+.0f}
• Cumulative: {order_flow.cumulative_delta:+.0f}

**KEY ZONES:**
• Absorption Zones: {len(order_flow.absorption_zones)}
• Rejection Zones: {len(order_flow.rejection_zones)}"""
        
        await self.send_message(chat_id, msg)
    except Exception as e:
        await self.send_message(chat_id, f"❌ Error: {str(e)}")
