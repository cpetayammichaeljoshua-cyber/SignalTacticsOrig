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
   - Railway will use the custom Dockerfile
   - Build uses `requirements.txt` for dependencies
   - Start command: `python main.py`

## Configuration Files

- `Dockerfile` - Custom Docker build (used by Railway)
- `railway.json` - Railway deployment settings (uses DOCKERFILE builder)
- `.dockerignore` - Excludes unnecessary files from Docker build
- `Procfile` - Process type (worker for background bot)
- `requirements.txt` - Python dependencies
- `runtime.txt` - Python version (3.11.10)

## Important Notes

- This is a **worker** process (not a web server)
- Secrets are set in Railway dashboard, NOT in code
- Bot runs 24/7 with auto-restart on failure
- Order Flow WebSocket connections auto-reconnect
- Using custom Dockerfile bypasses Nixpacks auto-detection issues

## Why Custom Dockerfile?

Railway's Nixpacks builder auto-detects `pyproject.toml` and uses `pip install -e .` 
which can cause issues. The custom Dockerfile gives full control over:
- Python version (3.11-slim)
- Dependency installation (requirements.txt only)
- Build process (no secrets in build args)

## Monitoring

Check Railway logs for:
- "AI Trading Signal Bot Started" - Bot initialized
- "Order Flow Active" - WebSocket connected
- Signal generation and Telegram notifications

## Troubleshooting

If deployment fails:
1. Check Railway logs for errors
2. Verify all environment variables are set in Railway dashboard
3. Make sure GitHub repo is synced with latest code
4. Check that Dockerfile is in root directory
