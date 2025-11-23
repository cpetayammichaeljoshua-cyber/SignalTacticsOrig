# ZEC/USDT Telegram Scanner Bot - Setup Instructions

## Quick Start (3 steps)

### Step 1: Set Telegram Bot Secrets
In your Replit **Secrets** tab, add:
```
TELEGRAM_BOT_TOKEN=your_bot_token_from_botfather
ADMIN_CHAT_ID=your_telegram_user_id
```

### Step 2: Set Binance API Secrets
In your Replit **Secrets** tab, add:
```
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
```

### Step 3: Run the Bot
```bash
python start_zecusdt_bot_telegram_scanner.py
```

## Features Included

### 📡 Telegram Channel Scanning
- Monitors channel **3464978276** continuously
- Extracts trade signals automatically
- Recognizes LONG/SHORT directions
- Parses entry prices, stop loss, take profit
- Detects leverage levels
- Validates signals with 75% confidence threshold

### 🤖 ZEC/USDT Trading
- Binance Futures integration
- Ichimoku Cloud strategy
- Multi-timeframe analysis
- Smart SL/TP calculation
- Rate limiting (1 trade per 30 minutes)

### 💬 Telegram Commands
Use any of these in a chat with the bot:

**Market Info:**
- `/price` - Current ZEC/USDT price
- `/dashboard` - Market analysis dashboard
- `/market` - General market overview
- `/channel_status` - Channel scanner status

**Account:**
- `/balance` - Your account balance
- `/position` - Open positions
- `/risk` - Risk management info

**Trading:**
- `/dynamic_sltp LONG` - Smart SL/TP for long
- `/dynamic_sltp SHORT` - Smart SL/TP for short
- `/scan` - Manual market scan

**Analysis:**
- `/stats` - Performance statistics
- `/backtest [days]` - Backtest strategy
- `/optimize` - Optimize parameters
- `/leverage AUTO` - Optimal leverage

**Admin:**
- `/admin status` - Detailed bot status
- `/admin restart` - Restart scanner
- `/admin config` - View configuration

## How Channel Signals Work

The bot recognizes signals like:
```
LONG ZEC @ 200 SL 198 TP 205
SHORT ZECUSDT Entry 199 Stop 197 Target 194
Buy 201, Leverage 5x, SL 200, TP 203
```

When detected, signals are:
1. ✅ Validated against confidence threshold
2. 📊 Formatted with full trade details
3. 📡 Relayed to the main signal channel
4. 📈 Stored for analysis

## File Structure

```
.
├── start_zecusdt_bot_telegram_scanner.py    # Main entry point
├── quick_test_zec_bot.py                     # Test script
├── ZECUSDT_BOT_README.md                     # Full documentation
├── SETUP_INSTRUCTIONS.md                     # This file
└── SignalMaestro/
    ├── zecusdt_telegram_bot.py               # Bot implementation
    ├── telegram_channel_scanner.py           # Channel scanning
    ├── zecusdt_trader_adapter.py             # ZEC/USDT trader
    └── ... (other strategy files)
```

## Testing

Run the test script to verify everything works:
```bash
python quick_test_zec_bot.py
```

Expected output:
```
✅ Imports
✅ Channel Scanner
✅ Trader
✅ Bot Init
```

## Troubleshooting

**"Missing TELEGRAM_BOT_TOKEN"**
→ Add TELEGRAM_BOT_TOKEN to Replit Secrets

**"Missing BINANCE_API_KEY"**
→ Add BINANCE_API_KEY to Replit Secrets

**No signals detected**
→ Check channel ID 3464978276 is correct
→ Verify signal format matches patterns
→ Check confidence threshold is ≥75%

**Bot connection errors**
→ Check internet connection
→ Verify API keys are valid
→ Check Binance account supports ZEC/USDT futures

## Performance

- Channel scan interval: 45-60 seconds
- Signal detection latency: <30 seconds
- API response time: <2 seconds
- Rate limit: 1 trade per 30 minutes

## Safety

- ✅ 75% confidence threshold on all signals
- ✅ API keys stored securely in Replit Secrets
- ✅ Automatic rate limiting
- ✅ Comprehensive error handling
- ✅ Real-time logging

## Support

Check logs folder for detailed information:
```bash
tail -f logs/comprehensive_fxsusdt_*.log
```

For issues, see the main README at: `ZECUSDT_BOT_README.md`
