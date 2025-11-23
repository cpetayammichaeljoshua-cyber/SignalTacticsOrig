"""
Enhanced signal formatting with market intelligence insights
"""
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class EnhancedSignalFormatter:
    """Format signals with market intelligence context"""
    
    @staticmethod
    def format_signal_with_intelligence(signal: Dict[str, Any], 
                                       market_analysis: Dict[str, Any]) -> str:
        """Format trading signal with market intelligence"""
        try:
            msg = "🎯 FXSUSDT TRADING SIGNAL\n"
            msg += "=" * 40 + "\n"
            
            # Signal direction
            direction = signal.get('direction', 'HOLD').upper()
            if direction == 'BUY':
                msg += "📈 Direction: BUY\n"
            elif direction == 'SELL':
                msg += "📉 Direction: SELL\n"
            else:
                msg += f"🟡 Direction: {direction}\n"
            
            # Entry price
            if 'entry' in signal:
                msg += f"💰 Entry: ${signal['entry']:.2f}\n"
            
            # Stop Loss & Take Profit
            if 'stop_loss' in signal:
                msg += f"🛑 Stop Loss: ${signal['stop_loss']:.2f}\n"
            if 'take_profit' in signal:
                msg += f"🎯 Take Profit: ${signal['take_profit']:.2f}\n"
            
            # Market Intelligence Insights
            if market_analysis:
                msg += "\n📊 MARKET INTELLIGENCE:\n"
                
                recommendation = market_analysis.get('recommendation', 'HOLD')
                msg += f"• Signal: {recommendation}\n"
                
                if 'confidence' in market_analysis:
                    confidence = market_analysis['confidence']
                    bars = '█' * int(confidence * 10)
                    msg += f"• Confidence: {bars} {confidence*100:.0f}%\n"
                
                if 'scores' in market_analysis:
                    scores = market_analysis['scores']
                    for metric, score in scores.items():
                        if isinstance(score, (int, float)):
                            msg += f"  • {metric.upper()}: {score:.2f}\n"
                
                if 'reasoning' in market_analysis:
                    msg += "\n💡 Analysis:\n"
                    for reason in market_analysis['reasoning'][:3]:
                        msg += f"  {reason}\n"
            
            msg += "\n" + "=" * 40 + "\n"
            msg += f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            msg += "Cornix Compatible ✅"
            
            return msg
        except Exception as e:
            logger.error(f"Signal formatting error: {e}")
            return f"⚠️ Signal Generation Error: {e}"

