# AI-Powered Trading Signal System

## Overview

This is an AI-powered cryptocurrency trading signal system that combines technical indicators (UT Bot Alerts and STC) with AI-driven analysis for ETH/USDT trading on Binance. The system generates trading signals, manages positions with dynamic TP/SL calculations, and sends Cornix-compatible signals via Telegram. It includes backtesting capabilities, trade learning from historical outcomes, and optional automated futures trading execution.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Signal Generation Strategy

The system uses a dual-indicator approach for signal confirmation:

**Problem**: Need reliable entry signals that minimize false positives while capturing strong trends.

**Solution**: Combine UT Bot Alerts (ATR-based trailing stop) with STC (Schaff Trend Cycle) indicator for multi-layered confirmation.

- **LONG signals** require: UT Bot BUY + STC green + STC rising + STC < 75
- **SHORT signals** require: UT Bot SELL + STC red + STC falling + STC > 25
- Stop loss placed at recent swing high/low (configurable lookback period)
- Take profit calculated at 1.5x risk by default (configurable risk:reward ratio)

**Rationale**: The UT Bot provides directional bias while STC confirms momentum, reducing whipsaw trades. The modified STC parameters (length: 80, fast: 27) are optimized based on backtesting research.

### AI Analysis Engine

**Problem**: Static indicator strategies don't adapt to changing market conditions or learn from past trades.

**Solution**: Integrate OpenAI GPT-based AI Trading Brain with persistent SQLite learning database.

**Architecture**:
- `AITradingBrain`: Provides signal confidence scoring, market regime analysis, and trade outcome learning
- `TradeLearningDB`: SQLite database tracking trades, outcomes, and AI insights for continuous improvement
- Caching mechanism to avoid redundant API calls
- Graceful fallback when OpenAI API is unavailable

**Trade-offs**: 
- Pros: Adaptive learning, contextual market analysis, improved signal quality over time
- Cons: Dependency on external API, potential latency, requires API credits

### Position Sizing and Risk Management

**Problem**: Fixed position sizing doesn't account for volatility, signal strength, or account risk tolerance.

**Solution**: AI Position Engine with dynamic calculations.

**Features**:
- `AIPositionEngine`: Calculates dynamic TP/SL based on ATR, volatility, and market structure
- Multi-target take profits (TP1, TP2, TP3) with configurable allocation percentages
- Trailing stop logic that progressively moves SL as targets are hit
- `LeverageCalculator`: Volatility-adjusted leverage with signal strength multipliers
- Position sizing based on configurable risk percentage (default 2% per trade)

**Design pattern**: The engine uses a state machine approach for trailing stops (INITIAL → AT_ENTRY → AT_TP1 → AT_TP2) to systematically protect profits.

### Data Pipeline

**Problem**: Need reliable real-time and historical market data with fallback mechanisms.

**Solution**: Multi-source data fetcher with library fallbacks.

**Architecture**:
- Primary: Binance official Python SDK (`python-binance`)
- Fallback: CCXT library for broader exchange support
- Supports both REST API (historical) and WebSocket (real-time)
- Optional Heikin Ashi transformation for smoother price action
- Built-in rate limiting and error handling

### Automated Trading Execution

**Problem**: Manual trade execution is slow and error-prone; need automated futures trading.

**Solution**: `FuturesExecutor` with comprehensive order management.

**Features**:
- Market, limit, and stop order execution via CCXT
- Leverage and margin type configuration (CROSS/ISOLATED)
- Position tracking and management
- Order result validation and retry logic
- Decimal precision handling for different trading pairs

**Design decision**: Uses CCXT for broader exchange compatibility rather than exchange-specific SDKs, enabling future multi-exchange support.

### Notification System

**Problem**: Traders need timely, actionable signals in a standardized format.

**Solution**: Multi-bot Telegram architecture with different purposes.

**Components**:
- `ProductionSignalBot`: Sends Cornix-compatible trading signals with rate limiting (max 6/hour)
- `InteractiveCommandBot`: Admin-only command interface for monitoring and control
- `TelegramSignalBot`: Basic signal notifications

**Cornix Format**: Signals include structured format for automated trading bot integration:
```
#ETHUSDT LONG
Entry: 1800.50
TP1: 1825.00
TP2: 1850.00
SL: 1775.00
Leverage: 10x
```

**Security**: Admin whitelist via chat IDs, command rate limiting to prevent spam.

### Backtesting Framework

**Problem**: Need to validate strategy performance before live deployment.

**Solution**: Historical simulation engine using the same indicator logic as live trading.

**Features**:
- `BacktestRunner`: Simulates trades on historical data
- `BacktestMetrics`: Comprehensive statistics (win rate, profit factor, drawdown, Sharpe ratio)
- Configurable lookback periods (default 30 days, max 5000 candles)
- Trade-by-trade record keeping with exit reasons

**Design principle**: Uses identical signal generation code as live system to ensure backtest accuracy.

### Orchestration Layer

**Problem**: Need to coordinate multiple async components in a continuous monitoring loop.

**Solution**: Central orchestrator pattern.

**Architecture**:
- `TradingOrchestrator`: Main coordinator for all components
- Async event loop for concurrent operations
- Signal handlers for graceful shutdown
- Continuous monitoring with configurable timeframes (1m, 5m, 15m, 1h)
- Health check mechanisms and error recovery

**Entry point**: `main.py` initializes orchestrator with all components and starts the monitoring loop.

## External Dependencies

### APIs and Services

- **OpenAI GPT API**: AI-powered trade analysis and learning
  - Required env var: `OPENAI_API_KEY`
  - Used for: Signal confidence scoring, market regime detection, trade outcome analysis
  - Fallback: System continues without AI features if unavailable

- **Binance API**: Market data and futures trading
  - Required env vars: `BINANCE_API_KEY`, `BINANCE_API_SECRET`
  - Used for: OHLCV data fetching, order execution, position management
  - Alternatives: CCXT provides fallback to other exchanges

- **Telegram Bot API**: Signal delivery and bot commands
  - Required env vars: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
  - Used for: Signal notifications, interactive commands, performance reports
  - Admin whitelist: `ADMIN_CHAT_IDS` (comma-separated)

### Python Libraries

**Core Dependencies**:
- `pandas`, `numpy`: Data manipulation and calculations
- `aiohttp`: Async HTTP requests for APIs
- `aiosqlite`: Async SQLite database operations
- `python-telegram-bot`: Telegram bot framework

**Exchange Libraries**:
- `python-binance`: Binance official SDK (primary)
- `ccxt`: Multi-exchange trading library (fallback)

**AI/ML**:
- `openai`: GPT API client (optional, with graceful degradation)

### Data Storage

- **SQLite Database**: `ut_bot_strategy/data/trade_learning.db`
  - Tables: `trades` (position records), `ai_learnings` (AI insights), `performance_metrics` (summaries)
  - Schema: Auto-created on first run via `TradeLearningDB.initialize()`
  - Purpose: Persistent trade history and AI learning data

### Configuration

All configurable parameters are centralized in `ut_bot_strategy/config.py`:
- Indicator settings (UT Bot, STC parameters)
- Trading rules (risk:reward ratio, swing lookback)
- Leverage configuration (min/max, volatility thresholds)
- Timeframes and monitoring intervals

Environment variables override defaults for sensitive credentials.