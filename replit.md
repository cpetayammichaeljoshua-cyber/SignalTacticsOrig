# UT Bot + STC Trading Signal Bot - Production Deployment ✅

## Project Overview
Advanced cryptocurrency trading signal bot for ETH/USDT on 5-minute timeframe combining:
- **UT Bot Alerts**: ATR-based trailing stop indicator for entry signals
- **STC (Schaff Trend Cycle)**: Momentum oscillator for trend confirmation
- **Order Flow Analysis**: Real-time CVD, delta, imbalance detection
- **Manipulation Detection**: Stop hunts, spoofing, liquidity sweeps
- **Multi-Source Market Intelligence**: Fear & Greed Index, CoinGecko, News Sentiment
- **Multi-Timeframe Confirmation**: 1m, 5m, 15m, 1h, 4h alignment analysis
- **Multi-Asset Scanning**: Top 20 USDT-M futures pairs
- Telegram signal notifications with rich formatting
- AI-powered position sizing and analysis
- Risk management with 1:1.5 reward ratio

## Recent Changes (Latest) - December 12, 2025

### ✅ Multi-Source Market Intelligence Integration
Added comprehensive external data sources for enhanced signal quality:

**New Package Created: `ut_bot_strategy/external_data/`**
- `fear_greed_client.py` - Alternative.me Fear & Greed Index API (FREE, no key required)
  - Current fear/greed value (0-100)
  - Historical data support
  - 5-minute caching
  - Graceful degradation on API failure
  
- `market_data_aggregator.py` - CoinGecko API integration (FREE tier)
  - Trending coins detection
  - Market stats for top cryptos
  - Global market cap data ($3.2T tracked)
  - Rate limiting (30 calls/min)
  - Optional COINGECKO_API_KEY env var

- `news_sentiment_client.py` - CryptoPanic API integration (FREE tier)
  - Crypto news with sentiment labels
  - Filter by currency (ETH, BTC)
  - Bullish/bearish/hot/important filters
  - Aggregated sentiment summary
  - Optional CRYPTOPANIC_API_KEY env var

### ✅ Multi-Asset Scanner
**New Package: `ut_bot_strategy/scanning/`**
- `multi_asset_scanner.py` - Scans 20 top USDT-M futures pairs
  - BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT, DOGEUSDT, ADAUSDT, AVAXUSDT, LINKUSDT, DOTUSDT
  - MATICUSDT, LTCUSDT, SHIBUSDT, UNIUSDT, ATOMUSDT, XLMUSDT, ETCUSDT, FILUSDT, APTUSDT, NEARUSDT
  - Parallel data fetching with asyncio
  - Signal engine integration per symbol
  - Composite scoring and opportunity ranking
  - Fear/greed and news sentiment alignment

### ✅ Multi-Timeframe Confirmation System
**New Package: `ut_bot_strategy/confirmation/`**
- `multi_timeframe.py` - Analyzes 5 timeframes: 1m, 5m, 15m, 1h, 4h
  - Weight distribution: 4h (30%), 1h (25%), 15m (20%), 5m (15%), 1m (10%)
  - UT Bot + STC indicators on each timeframe
  - Weighted alignment score calculation
  - Higher timeframe bias detection
  - Recommendations: STRONG_CONFIRM/CONFIRM/NEUTRAL/CONFLICT

### ✅ Enhanced Signal Engine
Updated `signal_engine.py` with multi-source confidence calculation:
- Base indicator confidence: 40%
- Order flow alignment: 20%
- Multi-timeframe confirmation: 15%
- Fear/Greed alignment: 10%
- News sentiment: 10%
- Market breadth: 5%

New `market_intelligence` field in signals with:
- fear_greed_value, fear_greed_classification
- news_sentiment_score, news_bias
- market_breadth_score
- mtf_alignment_score
- overall_intelligence_score
- component_scores breakdown

### ✅ Advanced Position Sizer
**New: `ut_bot_strategy/trading/position_sizer.py`**
- ATR volatility-based position sizing
- Kelly Criterion for optimal bet sizing
- Maximum portfolio exposure limits (50%)
- Signal confidence adjustments
- Correlation-aware sizing

### ✅ Enhanced AI Trading Brain
Updated `ai_trading_brain.py` with market intelligence analysis:
- Fear & Greed context in prompts (contrarian approach)
- News sentiment alignment analysis
- Multi-timeframe confirmation reasoning
- Market breadth assessment
- Enhanced position sizing recommendations
- Fallback rule-based analysis when OpenAI unavailable

### ✅ Enhanced Telegram Signals
Updated signal format with comprehensive market context:
```
🟢 UT BOT + STC SIGNAL 🟢

📈 Direction: LONG
💱 Pair: ETH/USDT
⏰ Timeframe: 5m

━━━━━ MARKET INTELLIGENCE ━━━━━

🎭 Fear & Greed: 45 (Fear)
📰 News Sentiment: +0.35 (Bullish)
📊 Market Breadth: 65% Bullish
🔄 MTF Alignment: 85% (4h, 1h, 15m confirm)
🧠 AI Confidence: 78%

━━━━━━━ TRADE SETUP ━━━━━━━

💰 Entry: $3,450.25
🛑 Stop Loss: $3,420.50
🎯 Take Profit: $3,494.88

━━━━━━ RISK ANALYSIS ━━━━━━

📊 Risk: 0.86%
🎲 Risk:Reward: 1:1.5
📈 Order Flow: Bullish (+0.45)
⚠️ Manipulation Score: 0.12

━━━━━━ CONFIRMATION ━━━━━━

✅ UT Bot LONG Signal
✅ STC Green ↑
✅ Fear supports LONG
✅ News sentiment aligned
✅ Higher TF confirms
```

## Previous Changes - December 7, 2025

### ✅ Order Flow Analysis Enhancement
- Real-time trade stream via Binance Futures WebSocket
- Order book depth updates (20 levels @ 100ms)
- CVD tracking, large order detection, buy/sell imbalance
- Manipulation detection (stop hunts, spoofing, liquidity sweeps)

### ✅ Dynamic TP/SL Enhancement
- Minimum SL distance enforcement (0.5% of entry price)
- True ATR calculation using Wilder's smoothing
- Multi-TP system (1:1, 1:2, 1:3 R:R ratios)

## Architecture

### Core Components
1. **UTBotAlerts** - ATR-based trailing stop indicator
2. **STCIndicator** - Schaff Trend Cycle oscillator
3. **SignalEngine** - Multi-source signal generation with confidence scoring
4. **BinanceDataFetcher** - Real-time OHLCV data
5. **TelegramSignalBot** - Enhanced signal notifications
6. **TradingOrchestrator** - Main bot controller

### External Data Clients
1. **FearGreedClient** - Alternative.me API (FREE)
2. **MarketDataAggregator** - CoinGecko API (FREE tier)
3. **NewsSentimentClient** - CryptoPanic API (FREE tier)

### Analysis Services
1. **MultiTimeframeConfirmation** - 5-timeframe analysis
2. **MultiAssetScanner** - 20-pair opportunity scanning
3. **VolatilityPositionSizer** - ATR/Kelly position sizing

### Order Flow Analysis
1. **OrderFlowStream** - WebSocket trade/depth streams
2. **TapeAnalyzer** - Footprint and absorption detection
3. **ManipulationDetector** - Stop hunt/spoofing detection
4. **OrderFlowMetricsService** - Aggregated metrics

## Project Structure

```
ut_bot_strategy/
├── __init__.py
├── config.py                 # Configuration with new external data settings
├── orchestrator.py           # Main bot controller with intelligence integration
├── ai/
│   ├── __init__.py
│   └── ai_trading_brain.py   # Enhanced AI analysis with market context
├── confirmation/
│   ├── __init__.py
│   └── multi_timeframe.py    # Multi-timeframe confirmation system
├── data/
│   ├── __init__.py
│   ├── binance_fetcher.py
│   ├── order_flow_stream.py
│   ├── order_flow_metrics.py
│   └── trade_learning_db.py
├── engine/
│   ├── __init__.py
│   ├── signal_engine.py      # Enhanced with multi-source confidence
│   ├── tape_analyzer.py
│   └── manipulation_detector.py
├── external_data/
│   ├── __init__.py
│   ├── fear_greed_client.py   # Alternative.me API
│   ├── market_data_aggregator.py  # CoinGecko API
│   └── news_sentiment_client.py   # CryptoPanic API
├── indicators/
│   ├── __init__.py
│   ├── ut_bot_alerts.py
│   └── stc_indicator.py
├── scanning/
│   ├── __init__.py
│   └── multi_asset_scanner.py  # Multi-asset opportunity scanner
├── telegram/
│   ├── __init__.py
│   ├── telegram_bot.py        # Enhanced with market intelligence display
│   └── production_signal_bot.py
└── trading/
    ├── __init__.py
    ├── ai_position_engine.py
    ├── futures_executor.py
    ├── leverage_calculator.py
    └── position_sizer.py      # Advanced volatility-based sizing
main.py                        # Entry point with all integrations
```

## Environment Variables

### Required:
- `TELEGRAM_BOT_TOKEN` - Telegram bot token
- `TELEGRAM_CHAT_ID` - Target chat ID
- `BINANCE_API_KEY` - Binance API key
- `BINANCE_API_SECRET` - Binance API secret

### Optional (FREE APIs work without these):
- `COINGECKO_API_KEY` - CoinGecko API key (free tier: 10K calls/month)
- `CRYPTOPANIC_API_KEY` - CryptoPanic API key (free tier available)

**Note:** Fear & Greed Index requires NO API key (completely free).

## API Rate Limits

| API | Free Tier | Rate Limit | Key Required |
|-----|-----------|------------|--------------|
| Fear & Greed | Unlimited | None | No |
| CoinGecko | 10K calls/month | 30/min | Optional |
| CryptoPanic | Limited | Varies | Optional |

## Confidence Calculation Weights

| Factor | Weight | Description |
|--------|--------|-------------|
| Base Indicators | 40% | UT Bot + STC signal strength |
| Order Flow | 20% | CVD trend, delta, manipulation score |
| MTF Confirmation | 15% | Timeframe alignment score |
| Fear/Greed | 10% | Contrarian sentiment alignment |
| News Sentiment | 10% | News bias alignment |
| Market Breadth | 5% | Overall market direction |

## Production Status

✅ **UT Bot Alerts Indicator**: Fully converted from Pine Script
✅ **STC Indicator**: Implemented with modified settings (80/27/50)
✅ **Signal Engine**: Multi-source confidence calculation
✅ **Order Flow Analysis**: CVD, delta, manipulation detection
✅ **External Data**: Fear/Greed, CoinGecko, News Sentiment
✅ **Multi-Timeframe**: 5-timeframe confirmation system
✅ **Multi-Asset Scanner**: 20 USDT-M pairs scanning
✅ **Position Sizer**: ATR/Kelly criterion sizing
✅ **AI Brain**: Enhanced with market intelligence
✅ **Telegram**: Rich formatted signals with full context
✅ **Railway Deployment**: Configuration ready

**🚀 UT Bot + STC Signal Bot with Multi-Source Intelligence is fully deployed and production-ready!**
