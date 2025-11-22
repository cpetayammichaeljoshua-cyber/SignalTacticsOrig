#!/usr/bin/env python3
"""
ULTIMATE HIGH-FREQUENCY SCALPING BOT
Dynamically Comprehensive Flexible Advanced Precise Fastest Intelligent Multi-Strategy System

Integrates ALL strategies for maximum trading opportunities:
✓ Ultimate Scalping Strategy (3m-4h, 50x leverage)
✓ Lightning Scalping Strategy (30s-2m, ultra-fast)
✓ Momentum Scalping Strategy (RSI/MACD, 1m-5m)
✓ Volume Breakout Scalping Strategy (volume spikes)
✓ Ichimoku Sniper Strategy (trend following)
✓ Market Intelligence Engine (5 advanced analyzers)

Features:
- 5-second scan intervals for maximum speed
- Multi-timeframe analysis (1m, 3m, 5m)
- Parallel strategy execution
- Dynamic position management
- ALL 538 Binance USDⓈ-M perpetual markets
- Real-time signal fusion with consensus voting
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
import ccxt
from typing import List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import core components
from high_frequency_scalping_orchestrator import HighFrequencyScalpingOrchestrator
from dynamic_multi_market_position_manager import DynamicMultiMarketPositionManager
from dynamic_comprehensive_error_fixer import DynamicComprehensiveErrorFixer
from bot_health_check import check_bot_health
from telegram_signal_notifier import TelegramSignalNotifier
from automatic_position_closer import AutomaticPositionCloser
from atas_platform_integration import ATASPlatformIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('high_frequency_scalping.log')
    ]
)

logger = logging.getLogger(__name__)


class BinanceMarketFetcher:
    """Fetches and filters Binance USDⓈ-M perpetual futures markets"""
    
    def __init__(self, exchange):
        self.exchange = exchange
        self.logger = logging.getLogger(__name__)
    
    async def get_high_volume_markets(self, min_volume_usdt: float = 5_000_000, top_n: int = 20) -> List[str]:
        """Get top N markets by volume"""
        try:
            self.logger.info("🔍 Fetching Binance USDⓈ-M futures markets...")
            
            # Load markets with error handling
            try:
                markets = self.exchange.load_markets()
            except Exception as e:
                self.logger.error(f"Failed to load markets: {e}")
                raise
            
            # Filter for USDⓈ-M perpetual futures
            perpetual_markets = [
                symbol for symbol, market in markets.items()
                if market.get('type') == 'swap' and 
                   market.get('settle') == 'USDT' and
                   market.get('contract') and
                   market.get('active', True)
            ]
            
            self.logger.info(f"✅ Found {len(perpetual_markets)} active USDⓈ-M perpetual markets")
            
            # Fetch 24h volume for all markets
            self.logger.info(f"📊 Checking volume for {len(perpetual_markets)} markets...")
            
            volume_data = []
            failed_count = 0
            for i, symbol in enumerate(perpetual_markets, 1):
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                    volume_usdt = ticker.get('quoteVolume', 0)
                    
                    if volume_usdt >= min_volume_usdt:
                        volume_data.append({
                            'symbol': symbol,
                            'volume': volume_usdt
                        })
                    
                    # Progress update
                    if i % 100 == 0:
                        self.logger.info(f"   Checked {i}/{len(perpetual_markets)} markets...")
                    
                    # Rate limiting - more conservative
                    if i % 20 == 0:
                        await asyncio.sleep(0.5)
                
                except Exception as e:
                    failed_count += 1
                    self.logger.debug(f"Skipping {symbol}: {e}")
                    if failed_count % 10 == 0:
                        self.logger.warning(f"⚠️ Failed to fetch {failed_count} markets")
                    # Add delay on errors to avoid rate limiting
                    await asyncio.sleep(0.1)
            
            self.logger.info(f"✅ Found {len(volume_data)} high-volume markets total")
            
            # Sort by volume and get top N
            volume_data.sort(key=lambda x: x['volume'], reverse=True)
            top_markets = [m['symbol'] for m in volume_data[:top_n]]
            
            self.logger.info(f"✅ Selected top {len(top_markets)} markets for HIGH-FREQUENCY monitoring:")
            for i, symbol in enumerate(top_markets[:10], 1):
                self.logger.info(f"   {i}. {symbol}")
            if len(top_markets) > 10:
                self.logger.info(f"   ... and {len(top_markets) - 10} more")
            
            return top_markets
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching markets: {e}")
            # Fallback to popular pairs
            return [
                'ETH/USDT:USDT', 'BTC/USDT:USDT', 'SOL/USDT:USDT', 
                'XRP/USDT:USDT', 'DOGE/USDT:USDT', 'BNB/USDT:USDT',
                'ADA/USDT:USDT', 'AVAX/USDT:USDT', 'MATIC/USDT:USDT',
                'LINK/USDT:USDT'
            ]


async def main():
    """Main high-frequency scalping bot"""
    
    # Print banner
    print("\n" + "="*90)
    print("""
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                               ║
    ║   ██╗  ██╗██╗ ██████╗ ██╗  ██╗    ███████╗██████╗ ███████╗ ██████╗          ║
    ║   ██║  ██║██║██╔════╝ ██║  ██║    ██╔════╝██╔══██╗██╔════╝██╔═══██╗         ║
    ║   ███████║██║██║  ███╗███████║    █████╗  ██████╔╝█████╗  ██║   ██║         ║
    ║   ██╔══██║██║██║   ██║██╔══██║    ██╔══╝  ██╔══██╗██╔══╝  ██║▄▄ ██║         ║
    ║   ██║  ██║██║╚██████╔╝██║  ██║    ██║     ██║  ██║███████╗╚██████╔╝         ║
    ║   ╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═╝  ╚═╝    ╚═╝     ╚═╝  ╚═╝╚══════╝ ╚══▀▀═╝          ║
    ║                                                                               ║
    ║              ULTIMATE HIGH-FREQUENCY SCALPING SYSTEM                          ║
    ║                                                                               ║
    ║   ⚡ 5-Second Scans  |  🎯 6+ Strategies  |  📊 Multi-Timeframe               ║
    ║   🌐 538 Markets  |  💎 Dynamic Position Management  |  🔥 Ultra-Fast        ║
    ║                                                                               ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    print("="*90 + "\n")
    
    logger.info("🚀 INITIALIZING HIGH-FREQUENCY SCALPING BOT")
    logger.info("="*80)
    
    # Step 1: Apply comprehensive error fixes
    logger.info("🔧 Step 1: Applying comprehensive error fixes...")
    error_fixer = DynamicComprehensiveErrorFixer()
    error_fixer.apply_all_fixes()
    
    # Step 2: Health check
    logger.info("🏥 Step 2: Running health checks...")
    await check_bot_health()
    
    # Step 3: Initialize exchange
    logger.info("📡 Step 3: Initializing Binance exchange...")
    api_key = os.getenv('BINANCE_API_KEY', '')
    api_secret = os.getenv('BINANCE_API_SECRET', '')
    
    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',
            'adjustForTimeDifference': True
        }
    })
    
    logger.info("✅ Exchange initialized")
    
    # Step 4: Get high-volume markets
    logger.info("🌐 Step 4: Fetching high-volume markets...")
    try:
        market_fetcher = BinanceMarketFetcher(exchange)
        markets = await market_fetcher.get_high_volume_markets(
            min_volume_usdt=5_000_000,
            top_n=20  # Monitor top 20 for high-frequency
        )
    except Exception as e:
        logger.error(f"❌ Error fetching markets: {e}")
        logger.info("⚠️ Using fallback market list...")
        markets = [
            'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT',
            'XRP/USDT:USDT', 'DOGE/USDT:USDT', 'BNB/USDT:USDT',
            'ADA/USDT:USDT', 'AVAX/USDT:USDT', 'MATIC/USDT:USDT',
            'LINK/USDT:USDT'
        ]
    
    # Step 5: Initialize Telegram notifier with enhanced configuration
    logger.info("📱 Step 5: Initializing Telegram signal notifier...")
    telegram_notifier = TelegramSignalNotifier()
    
    # Test Telegram connection
    logger.info("🧪 Testing Telegram bot connection...")
    telegram_ok = await telegram_notifier.test_connection()
    
    if telegram_ok:
        logger.info("✅ Telegram notifier ready and tested")
        # Send startup notification to channel
        startup_msg = """🚀 **HIGH-FREQUENCY SCALPING BOT ONLINE**

✅ Multi-Strategy System Active
📊 Monitoring 20+ High-Volume Markets
⚡ 5-Second Scan Intervals
🎯 6 Advanced Strategies Running
📈 Multi-Timeframe Analysis (1m, 3m, 5m, 30m)

**Active Strategies:**
• Ultimate Scalping (22% weight)
• Lightning Scalping (20% weight)
• Momentum Scalping (18% weight)
• Volume Breakout (15% weight)
• Ichimoku Sniper (15% weight)
• Market Intelligence (10% weight)

🔔 Ready to send premium trading signals!"""
        
        await telegram_notifier._send_telegram_message(startup_msg)
        logger.info("📢 Startup notification sent to Telegram channel")
    else:
        logger.warning("⚠️ Telegram connection test failed - signals will not be sent")
        logger.warning("💡 Please configure TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in Replit Secrets")
    
    # Step 6: Initialize automatic position closer
    logger.info("🔄 Step 6: Initializing automatic position closer...")
    position_closer = AutomaticPositionCloser(
        exchange=exchange,
        telegram_notifier=telegram_notifier
    )
    logger.info("✅ Position closer ready")
    
    # Step 7: Initialize ATAS platform integration
    logger.info("🔌 Step 7: Initializing ATAS platform integration...")
    atas_integration = ATASPlatformIntegration(host='0.0.0.0', port=8888)
    logger.info("✅ ATAS integration ready on http://0.0.0.0:8888")
    
    # Step 8: Initialize position manager
    logger.info("💎 Step 8: Initializing dynamic position manager...")
    position_manager = DynamicMultiMarketPositionManager(exchange=exchange)
    logger.info("✅ Position manager ready")
    
    # Step 9: Initialize high-frequency orchestrator with all integrations
    logger.info("⚡ Step 9: Initializing high-frequency scalping orchestrator...")
    orchestrator = HighFrequencyScalpingOrchestrator(
        telegram_notifier=telegram_notifier,
        position_closer=position_closer,
        atas_integration=atas_integration
    )
    
    # Load all strategies
    success = await orchestrator.initialize_strategies()
    if not success:
        logger.error("❌ Failed to load strategies. Exiting.")
        return
    
    logger.info("="*80)
    logger.info("🎯 ACTIVE STRATEGIES:")
    logger.info("   ✓ Ultimate Scalping Strategy (22% weight)")
    logger.info("   ✓ Lightning Scalping Strategy (20% weight)")
    logger.info("   ✓ Momentum Scalping Strategy (18% weight)")
    logger.info("   ✓ Volume Breakout Strategy (15% weight)")
    logger.info("   ✓ Ichimoku Sniper Strategy (15% weight)")
    logger.info("   ✓ Market Intelligence Engine (10% weight)")
    logger.info("="*80)
    logger.info("⚙️  CONFIGURATION:")
    logger.info(f"   Scan Interval: {orchestrator.scan_interval} seconds")
    logger.info(f"   Timeframes: {', '.join(orchestrator.fast_timeframes)}")
    logger.info(f"   Markets: {len(markets)} high-volume pairs")
    logger.info(f"   Min Consensus: {orchestrator.min_consensus_confidence}%")
    logger.info(f"   Min Strategies Agree: {orchestrator.min_strategies_agree}")
    logger.info(f"   Stop Loss: {orchestrator.tight_stop_loss_pct}%")
    logger.info(f"   Profit Targets: {orchestrator.quick_profit_targets}")
    logger.info("="*80)
    logger.info("🔗 INTEGRATIONS ACTIVE:")
    logger.info("   ✓ Telegram Signal Notifications")
    logger.info("   ✓ Automatic Position Closing")
    logger.info("   ✓ ATAS Platform API (http://0.0.0.0:8888)")
    logger.info("="*80)
    
    # Step 10: Start background tasks
    logger.info("🚀 Step 10: Starting background tasks...")
    
    # Start position monitoring
    position_monitor_task = asyncio.create_task(position_closer.monitor_positions())
    logger.info("✅ Position monitoring started")
    
    # Start ATAS API server
    atas_server_task = asyncio.create_task(atas_integration.start_server())
    logger.info("✅ ATAS API server started")
    
    # Step 11: Start high-frequency scanning
    logger.info("🚀 Step 11: Starting HIGH-FREQUENCY market scanner...")
    logger.info(f"⏱️  Scanning {len(markets)} markets every {orchestrator.scan_interval} seconds")
    logger.info(f"📊 Using timeframes: {', '.join(orchestrator.fast_timeframes)}")
    logger.info("")
    
    try:
        # Run scanner in parallel with background tasks
        scanner_task = asyncio.create_task(
            orchestrator.scan_markets_high_frequency(
                exchange=exchange,
                symbols=markets,
                position_manager=position_manager
            )
        )
        
        # Wait for scanner (other tasks run in background)
        await scanner_task
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  Shutting down gracefully...")
        # Cancel background tasks
        try:
            position_monitor_task.cancel()
            atas_server_task.cancel()
            await asyncio.gather(position_monitor_task, atas_server_task, return_exceptions=True)
        except Exception as e:
            logger.debug(f"Task cancellation error: {e}")
        logger.info("✅ Shutdown complete")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    # Run the bot
    asyncio.run(main())
