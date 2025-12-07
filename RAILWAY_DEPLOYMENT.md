# Railway.com Deployment Guide

## Quick Deploy

1. **Push code to GitHub** or connect Railway directly to your Replit

2. **Create new Railway project**:
   - Go to https://railway.app
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository

3. **Configure Environment Variables** (in Railway dashboard):
   - `TELEGRAM_BOT_TOKEN` - Your Telegram bot token
   - `TELEGRAM_CHAT_ID` - Target chat ID for signals
   - `BINANCE_API_KEY` - Binance API key
   - `BINANCE_API_SECRET` - Binance API secret

4. **Deploy**:
   - Railway will auto-detect the configuration
   - Build uses `requirements.txt` for dependencies
   - Start command: `python main.py`

## Configuration Files

- `nixpacks.toml` - Build configuration for Nixpacks
- `railway.json` - Railway deployment settings  
- `Procfile` - Process type (worker for background bot)
- `requirements.txt` - Python dependencies
- `runtime.txt` - Python version (3.11.10)

## Important Notes

- This is a **worker** process (not a web server)
- Secrets are set in Railway dashboard, NOT in code
- Bot runs 24/7 with auto-restart on failure
- Order Flow WebSocket connections auto-reconnect

## Monitoring

Check Railway logs for:
- "AI Trading Signal Bot Started" - Bot initialized
- "Order Flow Active" - WebSocket connected
- Signal generation and Telegram notifications
