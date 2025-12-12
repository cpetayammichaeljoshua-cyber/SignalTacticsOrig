# UT Bot + STC Trading Signal Bot

## Overview
This project is an advanced cryptocurrency trading signal bot designed for **ALL liquid Binance USDT-M Futures pairs**. It integrates a wide array of sophisticated analytics, including the UT Bot and STC indicators, real-time order flow analysis, manipulation detection, and multi-source market intelligence. The bot aims to provide highly accurate trading signals by confirming trends across multiple timeframes, tracking whale activity, and understanding the broader economic and derivatives market context. Its primary purpose is to identify high-probability trading opportunities with robust risk management, delivering actionable intelligence via Telegram notifications. The project's ambition is to capitalize on a comprehensive, data-driven approach to cryptocurrency futures trading.

## User Preferences
I prefer clear, concise explanations and direct answers. For coding, I favor a modular and object-oriented approach. I value iterative development and continuous integration. Before implementing significant changes or new features, please outline the proposed approach and discuss potential impacts. I expect the bot to provide detailed reasoning behind its signals and decisions, even if AI-driven. Do not make changes to files related to deployment configurations without explicit approval.

## System Architecture

### UI/UX Decisions
The primary user interface is Telegram, providing rich-formatted signal notifications. These notifications are designed for clarity, presenting key trade parameters, market intelligence summaries, and confirmation factors at a glance.

### Technical Implementations
The core of the system is built around a **Signal Engine** that synthesizes data from numerous sources to generate trading signals with a confidence score.
- **Indicators**: UTBotAlerts (ATR-based trailing stop) and STCIndicator (Schaff Trend Cycle oscillator) form the base signal.
- **Multi-Timeframe Confirmation**: Analyzes 1m, 5m, 15m, 1h, and 4h timeframes, assigning weights to higher timeframes for bias detection.
- **Order Flow Analysis**: Real-time processing of Binance WebSocket trade streams and order book depth to detect CVD, imbalance, and manipulation (stop hunts, spoofing, liquidity sweeps).
- **Dynamic Multi-Asset Scanning**: Auto-discovers and scans all liquid Binance USDT-M perpetual futures pairs, filtering by volume and sorting for priority.
- **AI Trading Brain**: Utilizes AI for position sizing and analysis, incorporating market intelligence context (Fear & Greed, news sentiment, multi-timeframe confirmation) and includes a fallback rule-based analysis.
- **Position Sizer**: Implements ATR volatility-based sizing combined with Kelly Criterion for optimal bet sizing, considering signal confidence and portfolio exposure limits.
- **External Intelligence Manager**: Centralizes coordination of all market intelligence sources, tracks per-source reliability, and applies confidence boosts or penalties based on source alignment or conflict.

### Feature Specifications
- **Comprehensive Market Intelligence**: Integration of various external data sources for a holistic market view.
- **Derivatives Data Analysis**: Real-time analysis of funding rates, open interest, Long/Short ratios, and liquidations for enhanced signal quality.
- **Whale Tracking**: Monitors large ($100K+) trades in real-time to identify smart money flow and potential manipulation.
- **Economic Calendar Awareness**: Tracks high-impact macro events to adjust trading confidence and warn of potential volatility.
- **Risk Management**: Enforces a minimum stop-loss distance and employs a multi-take-profit system, aiming for a 1:1.5 reward ratio.
- **Telegram Integration**: Delivers comprehensive, formatted trading signals including direction, pair, timeframe, entry, SL/TP, risk analysis, and detailed market intelligence context.

### System Design Choices
The system adopts a modular architecture, with distinct components for data fetching, indicator calculation, signal generation, external data integration, and notification. Asynchronous programming (`asyncio`) is used for parallel data fetching and real-time stream processing to ensure efficiency and responsiveness. Data caching mechanisms (e.g., 60-second for Binance derivatives, 1-hour for dynamic pairs) are implemented to comply with API rate limits and optimize performance.

## External Dependencies

- **Binance API**: For real-time OHLCV data, futures public API (funding rates, open interest, L/S ratios, taker volume), and WebSocket streams for trades, order book depth, and liquidations. (No API key required for public data/websockets).
- **Alternative.me Fear & Greed Index API**: For current and historical fear/greed values. (Free, no API key required).
- **CoinGecko API**: For trending coins, market statistics, and global market cap data. (Free tier available, optional API key for higher limits).
- **CryptoPanic API**: For crypto news and sentiment analysis. (Free tier available, optional API key).
- **Telegram Bot API**: For sending signal notifications.
- **OpenAI API**: For the AI-powered position sizing and analysis (AI Trading Brain).