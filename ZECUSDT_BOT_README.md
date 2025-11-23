# ZECUSDT Telegram Scanner Trading Bot

A sophisticated trading bot that scans Telegram channel **3464978276** for trading signals and automatically executes trades on ZEC/USDT futures.

## Features

✅ **Telegram Channel Monitoring**
- Real-time scanning of channel 3464978276
- Automatic signal extraction from messages
- Intelligent parsing of trade direction (LONG/SHORT), prices, and leverage

✅ **Advanced Signal Processing**
- Direction Detection: Extracts LONG/SHORT signals
- Price Parsing: Entry price, Stop Loss, Take Profit
- Leverage Detection: Identifies position sizing
- Confidence Scoring: 75% minimum confidence threshold
- Symbol Recognition: Automatically identifies ZEC/USDT

✅ **ZEC/USDT Futures Trading**
- Direct Binance Futures integration
- Multi-timeframe Ichimoku Cloud strategy
- Rate limiting (1 trade per 30 minutes)
- Dynamic position management

✅ **Comprehensive Commands**
- `/price` - Current ZEC/USDT price & 24h stats
- `/balance` - Account balance information
- `/position` - Open positions
- `/dashboard` - Market analysis dashboard
- `/channel_status` - Scanner status
- `/dynamic_sltp` - Smart SL/TP calculation
- And 20+ more analysis commands

## Setup

### 1. Install Dependencies
```bash
pip install python-telegram-bot aiohttp python-binance pandas scikit-learn numpy
```

### 2. Configure Secrets (Replit Secrets)
Add these secrets in your Replit environment:
```
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
ADMIN_CHAT_ID=your_admin_chat_id  # Optional
```

### 3. Run the Bot
```bash
python start_zecusdt_bot_telegram_scanner.py
```

## Channel Signal Format

The bot recognizes signals in the following formats from channel 3464978276:

```
LONG ZEC @ 200 SL 198 TP 205
SHORT ZECUSDT 199 Stop 197 Target 194
Entry 201, Leverage 5x, SL 200, TP 203
```

## How It Works

1. **Continuous Scanning**: Monitors channel every 45 seconds
2. **Signal Extraction**: Parses messages for trading information
3. **Validation**: Ensures signals meet 75% confidence threshold
4. **Rate Limiting**: One trade per 30 minutes minimum
5. **Execution**: Sends validated signals to @SignalTactics channel
6. **Relaying**: Forwards channel signals with full formatting

## Signal Confidence

Signals require:
- Clear direction (LONG or SHORT)
- Valid entry price
- Minimum 75% confidence score
- All price levels > 0

## Files

- `start_zecusdt_bot_telegram_scanner.py` - Main entry point
- `SignalMaestro/zecusdt_telegram_bot.py` - Bot implementation
- `SignalMaestro/telegram_channel_scanner.py` - Channel scanning module
- `SignalMaestro/zecusdt_trader_adapter.py` - ZEC/USDT trading interface

## Status Commands

- `/status` - Bot uptime and signal count
- `/channel_status` - Channel scanner details
- `/dashboard` - Market analysis
- `/admin status` - Detailed admin info

## Safety Features

- ✅ 75% minimum confidence threshold
- ✅ Rate limiting (max 1 signal per 30 minutes)
- ✅ API key encryption via Replit Secrets
- ✅ Comprehensive error handling
- ✅ Real-time logging

## Troubleshooting

**Bot not starting?**
- Check TELEGRAM_BOT_TOKEN is set
- Check BINANCE_API_KEY and BINANCE_API_SECRET are set
- View logs for error messages

**No signals detected?**
- Verify channel ID 3464978276 is correct
- Check message format matches expected patterns
- Ensure 75% confidence threshold is met

**Trading errors?**
- Verify Binance account has ZEC/USDT futures enabled
- Check account balance
- Ensure API keys have trading permissions

## Performance

Typical performance:
- Signal detection latency: <30 seconds
- API response time: <2 seconds
- Channel scan interval: 45 seconds base / 60 seconds active

## Support

For issues or questions, check the logs directory or enable debug logging.
