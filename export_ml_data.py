
#!/usr/bin/env python3
"""
Manual ML Data Export Utility for Ultimate Trading Bot
Exports all tracked trades for machine learning training
"""

import asyncio
import json
import os
import sys
import pandas as pd
from datetime import datetime

# Add SignalMaestro to path
sys.path.append('SignalMaestro')

async def export_ml_data():
    """Export ML training data manually"""
    try:
        # Import trade tracker
        from telegram_trade_tracker import TelegramTradeTracker
        
        # Get bot token from environment
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not bot_token:
            print("❌ Error: TELEGRAM_BOT_TOKEN environment variable not set")
            return
        
        # Initialize tracker
        tracker = TelegramTradeTracker(bot_token)
        
        print("📊 Exporting ML training data...")
        
        # Export data
        await tracker.export_ml_training_data()
        
        # Get summary
        summary = await tracker.get_training_data_summary()
        
        print(f"""✅ ML Data Export Complete!

📁 Files Generated:
• tracked_trades.json ({summary.get('total_trades_tracked', 0)} trades)
• ml_training_data.csv (ML-ready format)

📊 Data Summary:
• Total Trades: {summary.get('total_trades_tracked', 0)}
• Completed Trades: {summary.get('completed_trades', 0)}
• Win Rate: {summary.get('win_rate', 0):.1f}%
• Avg P&L: {summary.get('avg_profit_loss', 0):.2f}%
• Unique Symbols: {summary.get('unique_symbols', 0)}

🧠 Ready for ML training!""")
        
    except ImportError:
        print("❌ Error: Trade tracker module not available")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(export_ml_data())
