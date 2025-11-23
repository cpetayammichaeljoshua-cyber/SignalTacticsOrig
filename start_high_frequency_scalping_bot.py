#!/usr/bin/env python3
"""
HIGH-FREQUENCY SCALPING BOT - FXSUSDT
Production entry point with integrated Market Intelligence Engine
"""
import asyncio
import os
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    logger.info("🚀 STARTING HIGH-FREQUENCY SCALPING BOT")
    logger.info("=" * 80)
    
    try:
        # Import core components
        from SignalMaestro.fxsusdt_telegram_bot import FXSUSDTTelegramBot
        from SignalMaestro.market_intelligence_engine import MarketIntelligenceEngine
        
        logger.info("✅ Importing Market Intelligence Engine")
        
        # Initialize bot
        bot = FXSUSDTTelegramBot()
        
        # Initialize Market Intelligence Engine
        market_intelligence = MarketIntelligenceEngine()
        logger.info("✅ Market Intelligence Engine initialized")
        
        # Initialize AI Orchestrator for enhanced signals (OPTIONAL)
        ai_orchestrator = None
        try:
            from SignalMaestro.ai_orchestrator import AIOrchestrator
            ai_orchestrator = AIOrchestrator()
            logger.info("✅ AI Orchestrator initialized")
        except ImportError:
            logger.warning("⚠️ AI components not available (PyTorch/ML libraries)")
        except Exception as e:
            logger.warning(f"AI Orchestrator unavailable: {e}")
        
        # Configure for HIGH-FREQUENCY trading
        bot.min_signal_interval_minutes = 2
        market_intelligence.confidence_threshold = 0.60  # Lower threshold for HFT
        
        # Log configuration
        logger.info("=" * 80)
        logger.info("📊 HIGH-FREQUENCY TRADING CONFIGURATION")
        logger.info("=" * 80)
        logger.info(f"✅ Rate Limit: {bot.min_signal_interval_minutes} minutes (HIGH-FREQUENCY)")
        logger.info(f"✅ Signals: Frequent trades to @SignalTactics")
        logger.info(f"✅ Commands: {len(bot.commands)} available")
        logger.info(f"✅ Market Intelligence: ACTIVE")
        logger.info(f"✅ Confidence Threshold: {market_intelligence.confidence_threshold}")
        logger.info(f"✅ AI Enhancement: {'ENABLED' if ai_orchestrator else 'STANDARD'}")
        logger.info("=" * 80)
        
        # Start telegram polling
        logger.info("🔗 Starting Telegram bot polling...")
        await bot.start_telegram_polling()
        
    except KeyboardInterrupt:
        logger.info("⏹️ Bot stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    if not os.getenv('TELEGRAM_BOT_TOKEN'):
        logger.error("❌ Missing TELEGRAM_BOT_TOKEN in secrets")
        logger.error("Please set TELEGRAM_BOT_TOKEN in Replit Secrets")
        sys.exit(1)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot shutdown")
        sys.exit(0)
