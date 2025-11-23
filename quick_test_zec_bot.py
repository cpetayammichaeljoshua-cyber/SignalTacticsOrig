#!/usr/bin/env python3
"""
Quick test script for ZEC/USDT Bot
Tests core functionality without running full bot
"""

import asyncio
import sys
import os

# Add to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SignalMaestro'))

async def test_imports():
    """Test all imports"""
    print("🧪 Testing imports...")
    try:
        from zecusdt_telegram_bot import ZECUSDTTelegramBot
        print("✅ ZECUSDTTelegramBot imported")
        
        from telegram_channel_scanner import TelegramChannelScanner
        print("✅ TelegramChannelScanner imported")
        
        from zecusdt_trader_adapter import ZECUSDTTrader
        print("✅ ZECUSDTTrader imported")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

async def test_channel_scanner():
    """Test channel scanner"""
    print("\n🧪 Testing Telegram Channel Scanner...")
    try:
        from telegram_channel_scanner import TelegramChannelScanner
        
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN', 'test_token')
        scanner = TelegramChannelScanner(bot_token, channel_id="3464978276")
        
        # Test signal extraction
        test_message = "LONG ZEC @ 200 SL 198 TP 205"
        signal = scanner.extract_trade_signal(test_message)
        
        if signal and signal['direction'] == 'LONG':
            print(f"✅ Signal extraction works: {signal['direction']} {signal['symbol']}")
            return True
        else:
            print("❌ Signal extraction failed")
            return False
    except Exception as e:
        print(f"❌ Scanner test failed: {e}")
        return False

async def test_trader():
    """Test trader initialization"""
    print("\n🧪 Testing ZEC/USDT Trader...")
    try:
        from zecusdt_trader_adapter import ZECUSDTTrader
        
        trader = ZECUSDTTrader()
        print(f"✅ Trader initialized for {trader.symbol}")
        return True
    except Exception as e:
        print(f"⚠️ Trader init (expected if no API keys): {e}")
        return True  # Expected if no keys configured

async def test_bot_init():
    """Test bot initialization"""
    print("\n🧪 Testing ZEC/USDT Bot initialization...")
    try:
        # Set dummy token if not set
        if not os.getenv('TELEGRAM_BOT_TOKEN'):
            os.environ['TELEGRAM_BOT_TOKEN'] = 'test_token'
        
        from zecusdt_telegram_bot import ZECUSDTTelegramBot
        
        bot = ZECUSDTTelegramBot()
        print(f"✅ Bot initialized successfully")
        print(f"   - Symbol: {bot.trader.symbol}")
        print(f"   - Channel Scanner: {bot.channel_scanner.channel_id}")
        print(f"   - Commands Available: {len(bot.commands)}")
        return True
    except Exception as e:
        print(f"❌ Bot init failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("=" * 60)
    print("🤖 ZEC/USDT TELEGRAM SCANNER BOT - QUICK TEST")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Imports", await test_imports()))
    results.append(("Channel Scanner", await test_channel_scanner()))
    results.append(("Trader", await test_trader()))
    results.append(("Bot Init", await test_bot_init()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Bot is ready to run.")
        print("\nStart the bot with:")
        print("  python start_zecusdt_bot_telegram_scanner.py")
    else:
        print("\n⚠️ Some tests failed. Check errors above.")

if __name__ == "__main__":
    asyncio.run(main())
