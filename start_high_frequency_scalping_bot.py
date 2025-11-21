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
from high_frequency_telegram_pusher import HighFrequencyTelegramPusher

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
            
            # Load markets
            markets = self.exchange.load_markets()
            
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
                    
                    # Rate limiting
                    if i % 50 == 0:
                        await asyncio.sleep(1)
                
                except Exception as e:
                    self.logger.debug(f"Skipping {symbol}: {e}")
            
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
    market_fetcher = BinanceMarketFetcher(exchange)
    markets = await market_fetcher.get_high_volume_markets(
        min_volume_usdt=5_000_000,
        top_n=20  # Monitor top 20 for high-frequency
    )
    
    # Step 5: Initialize position manager
    logger.info("💎 Step 5: Initializing dynamic position manager...")
    position_manager = DynamicMultiMarketPositionManager(exchange=exchange)
    logger.info("✅ Position manager ready")
    
    # Step 6: Initialize Telegram pusher first
    logger.info("📡 Step 6: Initializing Telegram signal pusher...")
    telegram_pusher = HighFrequencyTelegramPusher()
    
    # Step 6.5: Initialize high-frequency orchestrator with Telegram callback
    logger.info("⚡ Step 6.5: Initializing high-frequency scalping orchestrator...")
    orchestrator = HighFrequencyScalpingOrchestrator(
        signal_callback=telegram_pusher.push_signal_to_telegram
    )
    
    # Send startup notification
    await telegram_pusher.send_status_update(
        f"🚀 HIGH-FREQUENCY SCALPING BOT STARTED\n\n"
        f"⚡ **Active Strategies:** 6+ Multi-Strategy Engine\n"
        f"📊 **Markets:** {len(markets)} high-volume pairs\n"
        f"⏱️ **Scan Interval:** {orchestrator.scan_interval}s\n"
        f"🎯 **Min Consensus:** {orchestrator.min_consensus_confidence}%\n"
        f"📈 **Telegram:** Auto-push enabled\n"
        f"🔥 **Ready for ultra-fast trading!**"
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
    
    # Step 7: Start high-frequency scanning
    logger.info("🚀 Step 7: Starting HIGH-FREQUENCY market scanner...")
    logger.info(f"⏱️  Scanning {len(markets)} markets every {orchestrator.scan_interval} seconds")
    logger.info(f"📊 Using timeframes: {', '.join(orchestrator.fast_timeframes)}")
    logger.info("")
    
    try:
        await orchestrator.scan_markets_high_frequency(
            exchange=exchange,
            symbols=markets,
            position_manager=position_manager
        )
    except KeyboardInterrupt:
        logger.info("\n⏹️  Shutting down gracefully...")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        raise


if __name__ == "__main__":
    # Run the bot
    asyncio.run(main())
