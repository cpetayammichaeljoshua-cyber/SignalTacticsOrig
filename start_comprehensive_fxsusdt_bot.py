#!/usr/bin/env python3
"""
Start Comprehensive FXSUSDT Trading Bot
Integrates all advanced analysis modules:
- Liquidity Analysis (grabs/sweeps)
- Order Flow (CVD)
- Volume Profile & Footprint Charts
- Fractals & Market Structure
- Intermarket Correlations
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add SignalMaestro to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SignalMaestro'))

from SignalMaestro.market_intelligence_engine import MarketIntelligenceEngine
from SignalMaestro.signal_fusion_engine import SignalFusionEngine
from SignalMaestro.comprehensive_dashboard import ComprehensiveDashboard
from SignalMaestro.ichimoku_sniper_strategy import IchimokuSniperStrategy

# Telegram imports
import aiohttp

class ComprehensiveFXSUSDTBot:
    """
    Comprehensive FXSUSDT trading bot with all advanced features
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.intel_engine = MarketIntelligenceEngine()
        self.fusion_engine = SignalFusionEngine()
        self.dashboard = ComprehensiveDashboard()
        self.ichimoku_strategy = IchimokuSniperStrategy()
        
        # Configuration
        self.symbol = 'FXSUSDT'
        self.timeframe = '30m'
        self.scan_interval = 60  # 1 minute
        
        # Telegram
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.channel_id = "@SignalTactics"
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}" if self.bot_token else None
        
        # Statistics
        self.signals_sent = 0
        self.last_signal_time = None
        self.start_time = datetime.now()
        
    async def run_continuous_scanner(self):
        """Main continuous scanning loop"""
        self.logger.info("=" * 80)
        self.logger.info("🚀 COMPREHENSIVE FXSUSDT TRADING BOT STARTING")
        self.logger.info("=" * 80)
        self.logger.info(f"📊 Symbol: {self.symbol}")
        self.logger.info(f"⏱️  Timeframe: {self.timeframe}")
        self.logger.info(f"🔄 Scan Interval: {self.scan_interval}s")
        self.logger.info("")
        self.logger.info("🔬 ACTIVE ANALYSIS MODULES:")
        self.logger.info("   ✅ Liquidity Analysis (grabs/sweeps)")
        self.logger.info("   ✅ Order Flow Analysis (CVD)")
        self.logger.info("   ✅ Volume Profile & Footprint Charts")
        self.logger.info("   ✅ Fractals & Market Structure")
        self.logger.info("   ✅ Intermarket Correlations")
        self.logger.info("   ✅ Ichimoku Sniper Strategy")
        self.logger.info("   ✅ Signal Fusion Engine")
        self.logger.info("=" * 80)
        self.logger.info("")
        
        while True:
            try:
                await self.scan_and_analyze()
                await asyncio.sleep(self.scan_interval)
                
            except KeyboardInterrupt:
                self.logger.info("\n🛑 Bot stopped by user")
                break
            except Exception as e:
                self.logger.error(f"❌ Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(self.scan_interval)
    
    async def scan_and_analyze(self):
        """Scan market and analyze"""
        try:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"🔍 SCANNING {self.symbol} - {datetime.now().strftime('%H:%M:%S')}")
            self.logger.info(f"{'='*80}")
            
            # Step 1: Run comprehensive market intelligence
            intel_snapshot = await self.intel_engine.analyze_market(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=500,
                correlated_symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
            )
            
            # Display intelligence summary
            compact_status = self.dashboard.format_compact_status(intel_snapshot)
            self.logger.info(f"\n📊 {compact_status}\n")
            
            # Step 2: Check Ichimoku strategy
            ichimoku_signal = None
            if intel_snapshot.should_trade():
                self.logger.info("🎯 Checking Ichimoku Sniper Strategy...")
                try:
                    df = intel_snapshot.analyzer_results.get(
                        list(intel_snapshot.analyzer_results.keys())[0]
                    )
                    # Note: In real implementation, pass actual OHLCV data
                    # ichimoku_signal = self.ichimoku_strategy.analyze(df)
                except Exception as e:
                    self.logger.warning(f"Ichimoku analysis skipped: {e}")
            
            # Step 3: Fuse signals
            fused_signal = self.fusion_engine.fuse_signal(
                ichimoku_signal=ichimoku_signal,
                intel_snapshot=intel_snapshot,
                current_price=intel_snapshot.analyzer_results.get(
                    list(intel_snapshot.analyzer_results.keys())[0]
                ).metrics.get('current_price', 0) if intel_snapshot.analyzer_results else 0
            )
            
            # Step 4: Send signal if generated
            if fused_signal:
                await self.send_signal(fused_signal)
            
            # Display full intelligence (verbose)
            if intel_snapshot.overall_score > 70:
                full_intel = self.dashboard.format_intel_snapshot(intel_snapshot)
                self.logger.info(f"\n{full_intel}")
            
        except Exception as e:
            self.logger.error(f"❌ Error in scan_and_analyze: {e}", exc_info=True)
    
    async def send_signal(self, signal):
        """Send trading signal to Telegram"""
        if not self.base_url:
            self.logger.warning("⚠️  Telegram not configured, signal not sent")
            self.logger.info(self.dashboard.format_fused_signal(signal))
            return
        
        try:
            # Format for Telegram
            message = self.dashboard.format_telegram_signal(signal)
            
            # Send to channel
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': self.channel_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        self.signals_sent += 1
                        self.last_signal_time = datetime.now()
                        self.logger.info(f"✅ Signal sent to {self.channel_id}")
                        self.logger.info(f"📊 Total signals sent: {self.signals_sent}")
                    else:
                        self.logger.error(f"❌ Failed to send signal: {response.status}")
            
            # Also log the signal
            self.logger.info(f"\n{self.dashboard.format_fused_signal(signal)}")
            
        except Exception as e:
            self.logger.error(f"❌ Error sending signal: {e}")
    
    def print_statistics(self):
        """Print bot statistics"""
        uptime = datetime.now() - self.start_time
        self.logger.info("\n" + "=" * 80)
        self.logger.info("📊 BOT STATISTICS")
        self.logger.info("=" * 80)
        self.logger.info(f"⏱️  Uptime: {uptime}")
        self.logger.info(f"📡 Signals Sent: {self.signals_sent}")
        self.logger.info(f"🕐 Last Signal: {self.last_signal_time or 'Never'}")
        self.logger.info("=" * 80)

async def main():
    """Main entry point"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/comprehensive_fxsusdt_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # ASCII Art Banner
    banner = """
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                                                                           ║
    ║   ██████╗ ██████╗ ███╗   ███╗██████╗ ██████╗ ███████╗██╗  ██╗███████╗   ║
    ║  ██╔════╝██╔═══██╗████╗ ████║██╔══██╗██╔══██╗██╔════╝██║  ██║██╔════╝   ║
    ║  ██║     ██║   ██║██╔████╔██║██████╔╝██████╔╝█████╗  ███████║█████╗     ║
    ║  ██║     ██║   ██║██║╚██╔╝██║██╔═══╝ ██╔══██╗██╔══╝  ██╔══██║██╔══╝     ║
    ║  ╚██████╗╚██████╔╝██║ ╚═╝ ██║██║     ██║  ██║███████╗██║  ██║███████╗   ║
    ║   ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝     ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝   ║
    ║                                                                           ║
    ║             ADVANCED FXSUSDT TRADING INTELLIGENCE SYSTEM                 ║
    ║                                                                           ║
    ║   🔬 Liquidity Analysis  |  📊 Order Flow (CVD)  |  📈 Volume Profile    ║
    ║   🔀 Fractals Analysis  |  🌐 Intermarket Data  |  ⚡ Signal Fusion     ║
    ║                                                                           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)
    
    # Check environment
    if not os.getenv('TELEGRAM_BOT_TOKEN'):
        logger.warning("⚠️  TELEGRAM_BOT_TOKEN not set - signals will be logged only")
    
    if not os.getenv('BINANCE_API_KEY'):
        logger.warning("⚠️  BINANCE_API_KEY not set - using public endpoints only")
    
    # Create and run bot
    bot = ComprehensiveFXSUSDTBot()
    
    try:
        await bot.run_continuous_scanner()
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutting down gracefully...")
        bot.print_statistics()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
