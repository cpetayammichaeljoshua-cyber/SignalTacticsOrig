# Trading Bot System

## Overview

This is a comprehensive cryptocurrency trading bot system designed for automated signal generation, processing, and execution. The system combines machine learning, technical analysis, and risk management to provide profitable trading signals across multiple timeframes. It integrates with Telegram for signal distribution, Binance for market data and execution, and Cornix for trade automation. The bot features advanced scalping strategies, ML-enhanced trade validation, and sophisticated position management with dynamic stop-loss and take-profit levels.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Core Trading Engine
The system uses a modular architecture with separate components for signal generation, risk management, and execution. The main trading bot (`bot/trading_bot.py`) orchestrates all operations, while specialized modules handle ML-based signal generation (`bot/ml_signal_generator.py`), adaptive leverage management (`bot/leverage_manager.py`), and comprehensive risk assessment (`bot/risk_manager.py`). The data processor (`bot/data_processor.py`) handles feature engineering and technical indicator calculations.

### Signal Processing Pipeline
Trading signals flow through a multi-stage pipeline starting with the signal parser (`SignalMaestro/signal_parser.py`) that handles various signal formats. The system supports multiple strategy implementations including the Advanced Time-Fibonacci Strategy (`SignalMaestro/advanced_time_fibonacci_strategy.py`) and Ultimate Scalping Strategy (`SignalMaestro/ultimate_scalping_strategy.py`). Each signal undergoes ML validation, risk assessment, and technical analysis before execution.

### Machine Learning Framework
The ML system learns from trade outcomes to improve future performance. Models are stored in `ml_models/` with performance metrics tracking accuracy and win rates. The ML analyzer (`SignalMaestro/ml_trade_analyzer.py`) continuously updates models based on closed trades, while the Telegram scanner (`SignalMaestro/telegram_trade_scanner.py`) collects historical trade data for training.

### Telegram Integration
The bot provides multiple Telegram interfaces including a simple signal forwarder (`SignalMaestro/simple_signal_bot.py`), an enhanced signal processor (`SignalMaestro/enhanced_signal_bot.py`), and automated signal distribution systems. The system supports professional signal formatting, chart generation, and rate-limited responses to prevent spam.

### Position Management
Advanced position management includes dynamic stop-loss and take-profit adjustments. When TP1 is hit, stop-loss moves to entry; when TP2 is hit, stop-loss moves to TP1; TP3 triggers full position closure. The system supports multiple leverage levels (10x-50x) with cross-margin configuration and sophisticated risk controls.

### Data Storage
Uses SQLite databases for persistent storage of trade history, user preferences, and ML training data. Key data files include `persistent_trade_logs.json` for trade tracking and various status JSON files for monitoring bot health and performance.

### Process Management
Robust process management with daemon systems (`SignalMaestro/bot_daemon.py`, `SignalMaestro/replit_daemon.py`) that provide auto-restart capabilities, health monitoring, and uptime management. The system includes keep-alive services and external ping integration for continuous operation.

## External Dependencies

### Trading Infrastructure
- **Binance API**: Primary exchange for market data and trade execution via ccxt library
- **Kraken API**: Alternative data source when Binance is unavailable
- **Cornix Platform**: Trade automation and signal distribution via webhook integration

### Telegram Services
- **Telegram Bot API**: Core messaging and user interaction platform
- **Signal Channels**: Integration with trading signal channels for data collection and distribution

### Machine Learning Stack
- **scikit-learn**: Primary ML framework for signal prediction and risk assessment
- **pandas/numpy**: Data processing and numerical computations
- **talib**: Technical analysis indicators (optional, with fallback implementations)

### Web Services
- **Flask/aiohttp**: Webhook servers for external signal reception
- **matplotlib**: Chart generation for technical analysis visualization
- **Uptime Monitoring**: External ping services for continuous operation assurance

### Development Tools
- **SQLite**: Local database for trade history and configuration storage
- **Logging**: Comprehensive logging system with file rotation and structured output