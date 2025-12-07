# Railway.com Deployment Guide

## Quick Start

1. **Connect your GitHub repository** to Railway
2. **Set environment variables** (see below)
3. **Deploy** - Railway will automatically detect Python and build

## Required Environment Variables

Set these in your Railway project settings:

| Variable | Description | Required |
|----------|-------------|----------|
| `TELEGRAM_BOT_TOKEN` | Your Telegram bot token from @BotFather | Yes |
| `TELEGRAM_CHAT_ID` | Target chat ID for signal notifications | Yes |
| `BINANCE_API_KEY` | Binance API key for market data | Yes |
| `BINANCE_API_SECRET` | Binance API secret | Yes |
| `OPENAI_API_KEY` | OpenAI API key for AI analysis (optional) | No |

## Files for Deployment

The following files configure Railway deployment:

- `Procfile` - Defines the web process
- `railway.json` - Railway-specific configuration
- `nixpacks.toml` - Nixpacks build configuration
- `requirements.txt` - Python dependencies
- `runtime.txt` - Python version specification

## Deployment Steps

### 1. Create Railway Project
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Initialize project
railway init
```

### 2. Link GitHub Repository
- Go to Railway dashboard
- Click "New Project" > "Deploy from GitHub repo"
- Select your repository

### 3. Configure Environment Variables
```bash
# Or set via CLI
railway variables set TELEGRAM_BOT_TOKEN=your_token
railway variables set TELEGRAM_CHAT_ID=your_chat_id
railway variables set BINANCE_API_KEY=your_api_key
railway variables set BINANCE_API_SECRET=your_api_secret
```

### 4. Deploy
```bash
railway up
```

## Features Deployed

- **Order Flow Analysis**: Real-time bid/ask imbalance, CVD, delta tracking
- **Tape Analysis**: Footprint charts, large print detection, absorption zones
- **Manipulation Detection**: Stop hunts, spoofing, liquidity sweeps
- **AI Signal Enhancement**: Dynamic TP/SL, confidence adjustment
- **Telegram Notifications**: Cornix-compatible signal format

## Monitoring

View logs in Railway dashboard or via CLI:
```bash
railway logs
```

## Troubleshooting

### Common Issues

1. **WebSocket Connection Fails**
   - Check if Binance is accessible from Railway's servers
   - Ensure API keys have correct permissions

2. **Telegram Not Sending**
   - Verify bot token is correct
   - Ensure chat ID is correct (use @userinfobot to get your chat ID)

3. **Database Errors**
   - SQLite databases are ephemeral on Railway
   - Consider using Railway PostgreSQL for persistence

### Health Check

The bot logs status every 30 seconds. Check logs for:
```
INFO - Fetched 200 candles via CCXT
INFO - All components initialized successfully
```

## Scaling

Railway automatically handles scaling. For high-frequency trading:
- Consider upgrading to a paid plan for more resources
- Use Railway volumes for persistent storage
