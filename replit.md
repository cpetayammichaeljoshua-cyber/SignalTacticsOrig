# Trading Signal Bot

## Overview

This repository contains a comprehensive cryptocurrency trading automation system that processes and forwards trading signals via Telegram. The bot combines multiple trading strategies, machine learning-enhanced analysis, and automated signal forwarding capabilities. It's designed for continuous operation with advanced restart management, health monitoring, and external uptime services integration.

**NEW: Comprehensive FXSUSDT Intelligence System** - Advanced multi-analyzer trading system integrating liquidity analysis, order flow (CVD), volume profiles, fractals, and intermarket correlations.

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Changes

### November 18, 2025
- **NEW: Comprehensive FXSUSDT Trading Intelligence System**: Built advanced multi-analyzer system with 5 specialized analysis modules:
  - **Liquidity Analysis**: POV liquidity grab/swept detection, stop hunt identification, smart money flow tracking
  - **Order Flow Analysis**: CVD (Cumulative Volume Delta) tracking, bid/ask imbalance, buying/selling pressure
  - **Volume Profile & Footprint Charts**: Point of Control, Value Area, HVN/LVN, footprint analysis
  - **Fractals Analysis**: Williams Fractals, market structure (HH/HL/LH/LL), swing points, trend confirmation
  - **Intermarket Correlations**: BTC/ETH correlation, risk-on/off sentiment, divergence detection
- **Market Intelligence Engine**: Central orchestrator that runs all analyzers in parallel, produces unified intelligence snapshots with consensus bias and scoring
- **Signal Fusion Engine**: Combines Ichimoku Sniper signals with market intelligence to produce high-confidence fused signals
- **Async Data Fetcher**: Efficient market data fetching with 10-second cache TTL to prevent redundant API calls
- **Comprehensive Dashboard**: Real-time visualization and formatting for all indicators
- **Standardized Data Contracts**: `MarketSnapshot`, `AnalysisResult`, `MarketIntelSnapshot`, `FusedSignal`
- **Weighted Consensus System**: Each analyzer contributes to final decision with configurable weights
- **Veto System**: Any analyzer can veto a trade based on unfavorable conditions
- **New Workflow**: "Comprehensive FXSUSDT Bot" - runs complete intelligence system
- **Documentation**: Added comprehensive README_COMPREHENSIVE_SYSTEM.md

### October 1, 2025
- **Fixed Critical Syntax Error**: Resolved syntax error in `advanced_time_fibonacci_strategy.py` that prevented AdvancedTimeFibonacciStrategy from registering. Changed invalid dictionary unpacking from `**ml_prediction if ml_prediction else {}` to `**(ml_prediction or {})`. Strategy now registers successfully (9 strategies total).
- **Fixed Import Errors**: Resolved missing imports in `fxsusdt_telegram_bot.py` and `ai_sentiment_analyzer.py` (time, feedparser, asyncio_throttle, BeautifulSoup).
- **Fixed Type Errors**: Corrected 15+ type errors across multiple files including None assignments, Telegram token validation, OpenAI client initialization, and chat member access.
- **Verified Command Handlers**: Confirmed all Telegram command handlers are functional (/balance, /leverage, /risk, /market, /position, /stats, /help, /start).
- **Verified Position Sizing**: Confirmed position sizing correctly calculates based on account balance × risk percentage, with stop loss distance and leverage multipliers properly applied.
- **OpenAI Integration**: Using Replit's python_openai integration with GPT-5 model (latest release, August 7, 2025) for AI sentiment analysis with secure API key management.
- **Dependencies Added**: Installed feedparser, asyncio-throttle, and beautifulsoup4 packages for enhanced functionality.

## System Architecture

### Core Bot Architecture

The system follows a modular architecture with multiple specialized bot implementations:

- **Signal Processing Layer**: Multiple signal parsers handle various trading signal formats from text messages using regex patterns
- **Strategy Layer**: Advanced trading strategies including Time-Fibonacci theory, ML-enhanced analysis, and multi-timeframe confluence
- **Execution Layer**: Integration with Binance and Kraken exchanges for live trading and market data
- **Communication Layer**: Telegram bot integration for receiving signals and sending formatted responses
- **Persistence Layer**: SQLite database for trade history, user settings, and ML training data

### Trading Strategy Components

The bot implements several sophisticated trading strategies:

1. **Perfect Scalping Strategy**: Uses technical indicators across 3m-4h timeframes with 1:3 risk-reward ratios
2. **Time-Fibonacci Strategy**: Combines market session analysis with Fibonacci retracements for optimal entry timing
3. **ML-Enhanced Strategy**: Machine learning models that learn from past trades to improve future signal quality
4. **Ultimate Scalping Strategy**: Multi-indicator confluence system with dynamic stop-loss management

### Process Management

Robust process management ensures continuous operation:

- **Daemon System**: Auto-restart functionality with configurable retry limits and cooldown periods
- **Health Monitoring**: Memory usage, CPU monitoring, and heartbeat checks
- **Status Tracking**: JSON-based status files for monitoring bot health and performance
- **Keep-Alive Service**: HTTP server for external ping services to maintain Replit uptime

### ML Learning System

Advanced machine learning capabilities for trade optimization:

- **Trade Analysis**: Learns from winning and losing trades to improve signal selection
- **Performance Tracking**: Monitors win rates, profit/loss patterns, and strategy effectiveness
- **Market Insights**: Analyzes optimal trading sessions, symbol performance, and market conditions
- **Adaptive Parameters**: Dynamically adjusts trading parameters based on historical performance

### Risk Management

Comprehensive risk management system:

- **Position Sizing**: Automated calculation based on account balance and risk tolerance
- **Stop-Loss Management**: Dynamic stop-loss adjustment as trades progress through take-profit levels
- **Rate Limiting**: Controlled signal generation to prevent overtrading (2-3 trades per hour)
- **Validation System**: Multi-layer signal validation before execution

## External Dependencies

### Trading Platforms

- **Binance API**: Primary exchange for live trading and market data retrieval
- **Kraken API**: Backup exchange for market data when Binance is unavailable
- **Cornix Integration**: Automated signal forwarding to Cornix trading bots

### Communication Services

- **Telegram Bot API**: Core communication platform for receiving and sending trading signals
- **Telegram Channels**: Signal forwarding to designated channels (@SignalTactics)

### Technical Analysis Libraries

- **pandas-ta**: Technical indicator calculations (RSI, MACD, EMA, Bollinger Bands)
- **talib**: Advanced technical analysis functions when available
- **matplotlib**: Chart generation for signal visualization

### Machine Learning Stack

- **scikit-learn**: Random Forest and Gradient Boosting models for trade prediction
- **pandas/numpy**: Data manipulation and numerical analysis
- **SQLite**: Local database for storing trade history and ML training data

### Infrastructure Services

- **aiohttp**: Asynchronous HTTP client for API communications
- **Flask**: Webhook server for receiving external signals
- **psutil**: Process monitoring and system resource tracking
- **External Ping Services**: Kaffeine, UptimeRobot for maintaining Replit uptime

### Development Libraries

- **asyncio**: Asynchronous programming for concurrent operations
- **logging**: Comprehensive logging system with file rotation
- **signal/threading**: Process management and graceful shutdown handling