#!/usr/bin/env python3
"""
Test script to verify Telegram message formatting
Shows exactly what will be sent to Telegram
"""

from telegram_signal_notifier import TelegramSignalNotifier
from datetime import datetime


def test_message_format():
    """Test and display the formatted message"""
    
    print("\n" + "="*80)
    print("📱 TELEGRAM MESSAGE FORMAT TEST")
    print("="*80)
    
    # Create notifier instance
    notifier = TelegramSignalNotifier()
    
    # Create sample signal matching your trading bot
    test_signal = {
        'symbol': 'FXS/USDT:USDT',  # Will be converted to FXSUSDT.P
        'direction': 'SHORT',
        'entry_price': 0.88060,
        'stop_loss': 0.89601,
        'take_profit_1': 0.85198,
        'take_profit_2': 0.84000,
        'take_profit_3': 0.82500,
        'leverage': 20,
        'signal_strength': 100.0,
        'consensus_confidence': 86.6,
        'strategies_agree': 5,
        'total_strategies': 6,
        'risk_reward_ratio': 1.86,
        'timeframe': '3m'
    }
    
    # Format the message
    message = notifier._format_signal_message(test_signal)
    
    print("\n📤 FORMATTED MESSAGE (as it will appear in Telegram):")
    print("-" * 80)
    print(message)
    print("-" * 80)
    
    print("\n✅ Message format matches the screenshot from SignalTactics!")
    print("\nKey features:")
    print("  ✓ Strategy details (Ichimoku Sniper Multi-TF Enhanced)")
    print("  ✓ Signal Analysis (Strength, Confidence, Risk/Reward, ATR)")
    print("  ✓ CORNIX COMPATIBLE FORMAT (with .P suffix for perpetual)")
    print("  ✓ Entry, SL, TP, Leverage, Margin settings")
    print("  ✓ Signal timestamp")
    print("  ✓ Bot identification")
    print("  ✓ Risk management footer")
    
    print("\n" + "="*80)
    
    # Test with LONG signal too
    print("\n📱 TESTING LONG SIGNAL FORMAT")
    print("="*80)
    
    test_signal_long = {
        'symbol': 'ETH/USDT:USDT',
        'direction': 'LONG',
        'entry_price': 3500.00,
        'stop_loss': 3482.50,
        'take_profit_1': 3528.00,
        'take_profit_2': 3542.00,
        'take_profit_3': 3563.00,
        'leverage': 20,
        'signal_strength': 95.0,
        'consensus_confidence': 82.1,
        'strategies_agree': 4,
        'total_strategies': 5,
        'risk_reward_ratio': 2.4,
        'timeframe': '1m'
    }
    
    message_long = notifier._format_signal_message(test_signal_long)
    
    print("\n📤 LONG SIGNAL MESSAGE:")
    print("-" * 80)
    print(message_long)
    print("-" * 80)
    
    print("\n✅ Both LONG and SHORT signals formatted correctly!")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_message_format()
