# UT Bot + STC Trading Signal Bot - Production Deployment ✅

## Project Overview
Advanced cryptocurrency trading signal bot for ETH/USDT on 5-minute timeframe combining:
- **UT Bot Alerts**: ATR-based trailing stop indicator for entry signals
- **STC (Schaff Trend Cycle)**: Momentum oscillator for trend confirmation
- Telegram signal notifications with rich formatting
- Automatic stop loss and take profit calculation
- Risk management with 1:1.5 reward ratio

## Recent Changes (Latest) - December 6, 2025

### ✅ Dependency Fixes for Deployment
- Fixed numpy/pandas-ta/numba version conflicts
- Replaced pandas-ta with ta library (more stable, Python 3.11 compatible)
- Updated pyproject.toml with compatible version constraints:
  - numpy>=1.24.0,<2.0 (binary compatible with pandas)
  - pandas>=2.0.0,<2.1 (stable version)
  - matplotlib>=3.8.0,<3.9
  - scikit-learn>=1.3.0,<1.5
- Migrated deprecated tool.uv.dev-dependencies to [dependency-groups] format
- Constrained Python version to >=3.11,<3.12 for package compatibility

## Changes - November 28, 2025

### ✅ New UT Bot + STC Strategy Implementation
- **UT Bot Alerts Indicator**: Converted from TradingView Pine Script to Python
- **STC Indicator**: Implemented with modified settings (Length=80, Fast=27, Slow=50)
- **Signal Engine**: Combined indicator logic with complete strategy rules
- **Telegram Integration**: Rich formatted signals with entry, SL, TP
- **Binance Data Fetcher**: Real-time ETH/USDT 5m data from Binance
- **Continuous Monitoring**: Async orchestrator for 24/7 operation

## Architecture

### Core Components (UT Bot + STC Strategy)
1. **UTBotAlerts** - ATR-based trailing stop indicator (converted from Pine Script)
2. **STCIndicator** - Schaff Trend Cycle oscillator with modified settings
3. **SignalEngine** - Combines indicators for signal generation
4. **BinanceDataFetcher** - Real-time OHLCV data from Binance
5. **TelegramSignalBot** - Rich formatted signal notifications
6. **TradingOrchestrator** - Main bot controller and monitoring loop

### Strategy Rules
**LONG Entry Conditions:**
- ✅ UT Bot issues BUY signal (price crosses above trailing stop)
- ✅ STC line is GREEN color
- ✅ STC line is pointing UPWARD
- ✅ STC value is BELOW 75

**SHORT Entry Conditions:**
- ✅ UT Bot issues SELL signal (price crosses below trailing stop)
- ✅ STC line is RED color
- ✅ STC line is pointing DOWNWARD
- ✅ STC value is ABOVE 25

**Risk Management:**
- ✅ Stop Loss: Recent swing low (LONG) or swing high (SHORT)
- ✅ Take Profit: 1.5x the risk amount (R:R = 1:1.5)
- ✅ Swing lookback: 5 bars for SL placement

### Indicator Settings

**UT Bot Alerts (Pine Script Converted):**
- Key Value (Sensitivity): 2.0
- ATR Period: 6
- Use Heikin Ashi: Enabled (ON)

**STC Indicator (Modified from original):**
- Length: 80 (changed from 12)
- Fast Length: 27
- Slow Length: 50
- Smoothing Factor (AAA): 0.5

### Auto-Leverage Trading System

**Leverage Configuration:**
- Min Leverage: 1x
- Max Leverage: 20x
- Base Leverage: 5x
- Risk Per Trade: 2%
- Max Position: 50% of balance

**Dynamic Leverage Calculation:**
- Volatility-adjusted: Lower leverage in high volatility
- Signal strength multiplier: Higher confidence = higher leverage
- Automatic position sizing based on stop loss distance
- Isolated margin for risk protection

## Project Structure

```
ut_bot_strategy/
├── __init__.py           # Package initialization
├── config.py             # Configuration settings
├── orchestrator.py       # Main bot controller
├── indicators/
│   ├── __init__.py
│   ├── ut_bot_alerts.py  # UT Bot indicator
│   └── stc_indicator.py  # STC indicator
├── engine/
│   ├── __init__.py
│   └── signal_engine.py  # Signal generation logic
├── data/
│   ├── __init__.py
│   └── binance_fetcher.py # Binance data fetching
├── trading/
│   ├── __init__.py
│   ├── leverage_calculator.py  # Auto-leverage calculation
│   └── futures_executor.py     # Binance Futures trading
└── telegram/
    ├── __init__.py
    └── telegram_bot.py   # Telegram notifications
main.py                   # Entry point
```

## Setup & Deployment

### 1. Set Replit Secrets
Required environment variables:
- `TELEGRAM_BOT_TOKEN` - Your Telegram bot token
- `TELEGRAM_CHAT_ID` - Target chat ID for signals
- `BINANCE_API_KEY` - Binance API key
- `BINANCE_API_SECRET` - Binance API secret

### 2. Run the Bot
```bash
python main.py
```

## Signal Format Example

```
🟢 UT BOT + STC SIGNAL 🟢

📈 Direction: LONG
💱 Pair: ETH/USDT
⏰ Timeframe: 5m

━━━━━━━━━━━━━━━━━━━━━

💰 Entry Price: $3,450.25
🛑 Stop Loss: $3,420.50
🎯 Take Profit: $3,494.88

━━━━━━━━━━━━━━━━━━━━━

📊 Risk: 0.86%
🎲 Risk:Reward: 1:1.5

CONFIRMATION:
✅ UT Bot LONG Signal
✅ STC Green ↑
✅ All conditions met
```

## Performance Notes
- Based on "Quantum Trading Strategy" with 55% win rate in backtests
- Modified STC settings (80/27/50) for better confirmation
- Swing-based stop loss placement for optimal risk management
- 1:1.5 Risk:Reward ratio for positive expectancy

## Final Production Status

✅ **UT Bot Alerts Indicator**: Fully converted from Pine Script
✅ **STC Indicator**: Implemented with modified settings
✅ **Signal Engine**: Complete strategy logic implemented
✅ **Telegram Integration**: Rich formatted notifications
✅ **Binance Data**: Real-time 5m ETH/USDT data
✅ **Continuous Monitoring**: 24/7 async operation

**🚀 UT Bot + STC Signal Bot is fully deployed and production-ready!**
