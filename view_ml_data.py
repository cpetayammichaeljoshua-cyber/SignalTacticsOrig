
#!/usr/bin/env python3
"""
ML Data Viewer Utility
View and analyze tracked trading data for ML training
"""

import json
import pandas as pd
import os
from datetime import datetime

def view_ml_data():
    """View ML training data"""
    try:
        # Check for data files
        json_file = "tracked_trades.json"
        csv_file = "ml_training_data.csv"
        
        if os.path.exists(json_file):
            print(f"📊 Loading data from {json_file}...")
            
            with open(json_file, 'r') as f:
                trades = json.load(f)
            
            if not trades:
                print("📭 No trades found in data file")
                return
            
            df = pd.DataFrame(trades)
            
            print(f"""
📈 ML TRAINING DATA ANALYSIS
═══════════════════════════════

📊 BASIC STATISTICS:
• Total Trades: {len(df)}
• Completed Trades: {len(df[df['trade_status'] == 'closed'])}
• Symbols: {df['symbol'].nunique() if 'symbol' in df.columns else 0}
• Date Range: {df['timestamp'].min() if 'timestamp' in df.columns else 'N/A'} to {df['timestamp'].max() if 'timestamp' in df.columns else 'N/A'}

💰 PROFIT/LOSS ANALYSIS:""")
            
            if 'profit_loss' in df.columns:
                completed = df[df['profit_loss'].notna()]
                if len(completed) > 0:
                    profitable = completed[completed['profit_loss'] > 0]
                    losing = completed[completed['profit_loss'] < 0]
                    
                    print(f"""• Profitable Trades: {len(profitable)} ({len(profitable)/len(completed)*100:.1f}%)
• Losing Trades: {len(losing)} ({len(losing)/len(completed)*100:.1f}%)
• Average P&L: {completed['profit_loss'].mean():.2f}%
• Best Trade: {completed['profit_loss'].max():.2f}%
• Worst Trade: {completed['profit_loss'].min():.2f}%""")
            
            print("\n🎯 TARGET PERFORMANCE:")
            if all(col in df.columns for col in ['tp1_hit', 'tp2_hit', 'tp3_hit', 'sl_hit']):
                tp1_rate = df['tp1_hit'].sum() / len(df) * 100
                tp2_rate = df['tp2_hit'].sum() / len(df) * 100
                tp3_rate = df['tp3_hit'].sum() / len(df) * 100
                sl_rate = df['sl_hit'].sum() / len(df) * 100
                
                print(f"""• TP1 Hit Rate: {tp1_rate:.1f}%
• TP2 Hit Rate: {tp2_rate:.1f}%
• TP3 Hit Rate: {tp3_rate:.1f}%
• Stop Loss Rate: {sl_rate:.1f}%""")
            
            print("\n📊 SYMBOL BREAKDOWN:")
            if 'symbol' in df.columns:
                symbol_stats = df.groupby('symbol').agg({
                    'profit_loss': ['count', 'mean'],
                    'tp1_hit': 'sum',
                    'sl_hit': 'sum'
                }).round(2)
                
                print(symbol_stats.head(10))
            
            print(f"\n📁 Data files available:")
            print(f"• JSON: {json_file} ({os.path.getsize(json_file)} bytes)")
            if os.path.exists(csv_file):
                print(f"• CSV: {csv_file} ({os.path.getsize(csv_file)} bytes)")
            
            print("\n🧠 Ready for ML model training!")
            
        else:
            print(f"❌ No data file found: {json_file}")
            print("Run the Ultimate Trading Bot to start collecting data or use /export command")
    
    except Exception as e:
        print(f"❌ Error viewing ML data: {e}")

if __name__ == "__main__":
    view_ml_data()
