
#!/usr/bin/env python3
"""
Quick Start: High-Frequency Scalping Bot with Telegram Integration
Launch the complete system with one command
"""

import asyncio
import os
import sys

# Ensure environment is set
if not os.getenv('TELEGRAM_BOT_TOKEN'):
    print("❌ ERROR: TELEGRAM_BOT_TOKEN not set!")
    print("Please set it in Replit Secrets")
    sys.exit(1)

if not os.getenv('BINANCE_API_KEY'):
    print("⚠️ WARNING: BINANCE_API_KEY not set")
    print("Bot will run in demo mode")

print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   HIGH-FREQUENCY SCALPING BOT + TELEGRAM INTEGRATION         ║
║                                                               ║
║   ⚡ 5-Second Scans                                           ║
║   🎯 6+ Strategy Consensus                                    ║
║   📡 Auto Telegram Push                                       ║
║   🚀 Ultra-Fast Signal Generation                             ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
""")

# Import and run
from start_high_frequency_scalping_bot import main

if __name__ == "__main__":
    asyncio.run(main())
