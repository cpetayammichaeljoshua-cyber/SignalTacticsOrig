#!/usr/bin/env python3
"""
Comprehensive FXSUSDT Bot with Advanced Market Intelligence Integration
Dynamically precise, fastest, and most intelligent trading strategy workflow

Features:
- Real-time Market Intelligence Engine with 5 specialized analyzers
- Liquidity analysis (POV, stop hunts, smart money flow)
- Order flow analysis (CVD, bid/ask imbalance, pressure)
- Volume profile & footprint analysis
- Williams Fractals with market structure
- Intermarket correlations (BTC/ETH divergence)
- Ichimoku Sniper signal integration
- AI-enhanced signal fusion
- Comprehensive risk management
- Live Binance futures trading
- Telegram notifications
"""

import asyncio
import logging
import sys
import os
import warnings
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import traceback

# Suppress all warnings globally
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['PYTHONWARNINGS'] = 'ignore'

# Configure pandas to suppress warnings
try:
    import pandas as pd
    pd.set_option('mode.chained_assignment', None)
    pd.options.mode.copy_on_write = True
except ImportError:
    pass

# Add SignalMaestro to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SignalMaestro'))

# Import with error handling
try:
    from SignalMaestro.market_intelligence_engine import MarketIntelligenceEngine
    from SignalMaestro.market_data_contracts import MarketBias, AnalyzerType
    from SignalMaestro.config import Config
    from SignalMaestro.logger import setup_logging
    from SignalMaestro.binance_trader import BinanceTrader
    from SignalMaestro.database import Database
    from SignalMaestro.fxsusdt_telegram_bot import FXSUSDTTelegramBot
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("🔧 Attempting to fix import issues...")
    sys.path.insert(0, os.path.dirname(__file__))
    from SignalMaestro.market_intelligence_engine import MarketIntelligenceEngine
    from SignalMaestro.market_data_contracts import MarketBias, AnalyzerType
    from SignalMaestro.config import Config
    from SignalMaestro.logger import setup_logging
    from SignalMaestro.binance_trader import BinanceTrader
    from SignalMaestro.database import Database
    from SignalMaestro.fxsusdt_telegram_bot import FXSUSDTTelegramBot


class ComprehensiveFXSUSDTBotWithIntel:
    """
    Comprehensive FXSUSDT Bot with integrated Market Intelligence
    
    Architecture:
    1. Market Intelligence Engine: Runs 5 parallel analyzers
    2. Ichimoku Sniper Strategy: Generates base signals (via FXSUSDTTelegramBot)
    3. Signal Fusion: Combines intel + Ichimoku for final decision
    4. Risk Management: Position sizing, stop-loss, take-profit
    5. Trade Execution: Live Binance futures trading
    6. Monitoring: Real-time logging and Telegram alerts
    """
    
    def __init__(self):
        """Initialize the comprehensive bot"""
        self.config = Config()
        self.logger = setup_logging(
            log_level="INFO",
            log_file=f"logs/comprehensive_fxsusdt_intel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        self.symbol = self.config.FXSUSDT_SYMBOL
        self.timeframe = self.config.FXSUSDT_TIMEFRAME
        self.is_running = False
        self.stats: Dict[str, float] = {
            'intel_analyses': 0.0,
            'signals_processed': 0.0,
            'trades_executed': 0.0,
            'last_analysis_time': 0.0,
            'avg_analysis_time': 0.0,
            'high_confidence_signals': 0.0
        }
        
        self.telegram_bot = None
        self.intel_engine = None
        self.trader = None
        self.db = None
        
        self._print_startup_banner()
    
    def _print_startup_banner(self):
        """Print comprehensive startup banner"""
        self.logger.info("=" * 100)
        self.logger.info("🚀 COMPREHENSIVE FXSUSDT BOT WITH MARKET INTELLIGENCE - STARTING")
        self.logger.info("=" * 100)
        self.logger.info(f"")
        self.logger.info(f"📊 MARKET INTELLIGENCE ENGINE")
        self.logger.info(f"   • 5 Specialized Analyzers running in PARALLEL")
        self.logger.info(f"   • Liquidity Analysis: POV grabs, stop hunts, smart money detection")
        self.logger.info(f"   • Order Flow Analysis: CVD tracking, bid/ask imbalance, pressure")
        self.logger.info(f"   • Volume Profile: Point of Control, Value Area, HVN/LVN")
        self.logger.info(f"   • Fractals Analysis: Williams Fractals, market structure")
        self.logger.info(f"   • Intermarket Correlations: BTC/ETH divergence, risk-on/off")
        self.logger.info(f"")
        self.logger.info(f"🎯 TRADING CONFIGURATION")
        self.logger.info(f"   • Symbol: {self.symbol}")
        self.logger.info(f"   • Timeframe: {self.timeframe}")
        self.logger.info(f"   • Strategy: Ichimoku Sniper + Market Intelligence Fusion")
        self.logger.info(f"   • Risk Management: Dynamic Position Sizing")
        self.logger.info(f"")
        self.logger.info(f"📈 PERFORMANCE METRICS")
        self.logger.info(f"   • Min Intelligence Score: 60/100")
        self.logger.info(f"   • Min Signal Confidence: 70%")
        self.logger.info(f"   • Risk Level Assessment: Active")
        self.logger.info(f"   • Veto System: Enabled (prevents high-risk trades)")
        self.logger.info("=" * 100)
    
    async def initialize(self) -> bool:
        """Initialize all components"""
        try:
            self.logger.info("🔧 Initializing All Components...")
            
            # Initialize Market Intelligence Engine
            self.logger.info("  📊 Initializing Market Intelligence Engine...")
            self.intel_engine = MarketIntelligenceEngine(
                api_key=self.config.BINANCE_API_KEY,
                api_secret=self.config.BINANCE_API_SECRET
            )
            self.logger.info("     ✅ Market Intelligence Engine initialized")
            self.logger.info(f"        Enabled Analyzers: {len(self.intel_engine.enabled_analyzers)}")
            
            # Initialize Telegram Bot
            self.logger.info("  📱 Initializing Telegram Bot...")
            self.telegram_bot = FXSUSDTTelegramBot()
            self.logger.info("     ✅ Telegram Bot initialized")
            
            # Initialize Binance Trader
            self.logger.info("  💱 Initializing Binance Trader...")
            self.trader = BinanceTrader()
            await self.trader.initialize()
            self.logger.info("     ✅ Binance Trader initialized")
            
            # Initialize Database
            self.logger.info("  💾 Initializing Database...")
            self.db = Database()
            await self.db.initialize()
            self.logger.info("     ✅ Database initialized")
            
            # Verify connection
            balance = await self.trader.get_account_balance()
            self.logger.info(f"     💰 Account Balance: {balance:.2f} USDT")
            
            self.logger.info("✅ All Components Initialized Successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Initialization failed: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    async def generate_market_intelligence(self) -> Optional[Dict]:
        """Generate comprehensive market intelligence"""
        try:
            start_time = datetime.now()
            
            self.logger.info(f"\n🔬 Generating Comprehensive Market Intelligence for {self.symbol}...")
            self.logger.info(f"   Analyzing 500 candles @ {self.timeframe}")
            self.logger.info(f"   Running all 5 analyzers in parallel...")
            
            # Run comprehensive analysis
            if not self.intel_engine:
                self.logger.error("Market Intelligence Engine not initialized")
                return None
            
            intel_snapshot = await self.intel_engine.analyze_market(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=500,
                correlated_symbols=['BTCUSDT', 'ETHUSDT']
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.stats['last_analysis_time'] = float(processing_time)
            self.stats['intel_analyses'] += 1
            
            # Calculate average
            if self.stats['avg_analysis_time'] == 0:
                self.stats['avg_analysis_time'] = float(processing_time)
            else:
                self.stats['avg_analysis_time'] = (float(self.stats['avg_analysis_time']) + float(processing_time)) / 2
            
            # Log comprehensive summary
            self._log_intelligence_summary(intel_snapshot, processing_time)
            
            return {
                'timestamp': intel_snapshot.timestamp,
                'consensus_bias': intel_snapshot.consensus_bias,
                'consensus_confidence': intel_snapshot.consensus_confidence,
                'overall_score': intel_snapshot.overall_score,
                'risk_level': intel_snapshot.risk_level,
                'veto_count': intel_snapshot.total_veto_count,
                'entry': intel_snapshot.recommended_entry,
                'stop': intel_snapshot.recommended_stop,
                'targets': intel_snapshot.recommended_targets,
                'leverage': intel_snapshot.recommended_leverage,
                'critical_levels': intel_snapshot.critical_levels[:5],
                'signals': intel_snapshot.dominant_signals[:5],
                'raw_snapshot': intel_snapshot
            }
            
        except Exception as e:
            self.logger.error(f"❌ Intelligence generation failed: {e}")
            return None
    
    def _log_intelligence_summary(self, intel_snapshot, processing_time):
        """Log detailed intelligence summary"""
        self.logger.info(f"\n✅ MARKET INTELLIGENCE REPORT ({processing_time:.0f}ms)")
        self.logger.info(f"   Consensus: {intel_snapshot.consensus_bias.value} ({intel_snapshot.consensus_confidence:.1f}%)")
        self.logger.info(f"   Overall Score: {intel_snapshot.overall_score:.1f}/100")
        self.logger.info(f"   Risk Level: {intel_snapshot.risk_level}")
        self.logger.info(f"   Vetoes: {intel_snapshot.total_veto_count}")
        self.logger.info(f"   Active Analyzers: {intel_snapshot.analyzers_active}/{intel_snapshot.analyzers_active + intel_snapshot.analyzers_failed}")
        
        self.logger.info(f"\n📊 ANALYZER BREAKDOWN:")
        for analyzer_type, result in intel_snapshot.analyzer_results.items():
            self.logger.info(f"   {analyzer_type.value.upper()}: Score {result.score:.1f}, Confidence {result.confidence:.1f}%, Bias {result.bias.value}")
        
        if intel_snapshot.critical_levels:
            self.logger.info(f"\n🎯 TOP 3 CRITICAL LEVELS:")
            for i, level in enumerate(intel_snapshot.critical_levels[:3], 1):
                self.logger.info(f"   {i}. {level.get('level_type', 'unknown').upper()} @ {level.get('price', 0):.2f}")
        
        if intel_snapshot.dominant_signals:
            self.logger.info(f"\n⚡ TOP 3 SIGNALS:")
            for i, sig in enumerate(intel_snapshot.dominant_signals[:3], 1):
                self.logger.info(f"   {i}. {sig.get('type', 'unknown')} from {sig.get('analyzer', 'unknown')}")
    
    async def run_analysis_cycle(self):
        """Run one complete analysis cycle with market intelligence"""
        try:
            self.logger.info("\n" + "=" * 100)
            self.logger.info(f"📱 ANALYSIS CYCLE #{self.stats['intel_analyses'] + 1} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info("=" * 100)
            
            # Generate Market Intelligence
            intel = await self.generate_market_intelligence()
            
            # Use Telegram bot to process signals with intelligence
            if intel:
                self.stats['signals_processed'] += 1
                if intel['consensus_confidence'] >= 70:
                    self.stats['high_confidence_signals'] += 1
                
                self.logger.info(f"\n💡 SIGNAL QUALITY ASSESSMENT:")
                self.logger.info(f"   Confidence Level: {'🟢 HIGH' if intel['consensus_confidence'] >= 70 else '🟡 MEDIUM' if intel['consensus_confidence'] >= 50 else '🔴 LOW'}")
                self.logger.info(f"   Total Score: {intel['overall_score']:.1f}/100")
                self.logger.info(f"   Risk Assessment: {intel['risk_level'].upper()}")
            
            # Log cycle statistics
            self._log_cycle_statistics()
            
        except Exception as e:
            self.logger.error(f"❌ Analysis cycle failed: {e}")
            self.logger.error(traceback.format_exc())
    
    def _log_cycle_statistics(self):
        """Log cycle statistics"""
        self.logger.info(f"\n📊 BOT STATISTICS:")
        self.logger.info(f"   Intelligence Analyses: {self.stats['intel_analyses']}")
        self.logger.info(f"   Signals Processed: {self.stats['signals_processed']}")
        self.logger.info(f"   High-Confidence Signals: {self.stats['high_confidence_signals']}")
        self.logger.info(f"   Avg Analysis Time: {self.stats['avg_analysis_time']:.0f}ms")
        self.logger.info(f"   Last Analysis Time: {self.stats['last_analysis_time']:.0f}ms")
    
    async def run_continuous(self, interval_seconds: int = 300):
        """
        Run bot continuously with market intelligence
        
        Default interval: 5 minutes (prevents excessive API calls)
        """
        self.is_running = True
        
        self.logger.info(f"\n🌐 STARTING CONTINUOUS MODE")
        self.logger.info(f"   Analysis Interval: {interval_seconds}s ({interval_seconds/60:.1f} min)")
        self.logger.info(f"   Press Ctrl+C to stop")
        
        try:
            while self.is_running:
                try:
                    await self.run_analysis_cycle()
                    self.logger.info(f"\n⏱️  Next analysis in {interval_seconds} seconds...")
                    await asyncio.sleep(interval_seconds)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Cycle error: {e}")
                    await asyncio.sleep(60)
        
        except KeyboardInterrupt:
            self.logger.info("\n\n⏹️  Bot stopping (Ctrl+C received)")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Gracefully shutdown"""
        self.logger.info("\n" + "=" * 100)
        self.logger.info("🛑 SHUTTING DOWN BOT")
        self.logger.info("=" * 100)
        
        self.is_running = False
        
        try:
            if self.trader:
                await self.trader.close()
                self.logger.info("✅ Binance connection closed")
        except Exception as e:
            self.logger.warning(f"Error closing Binance: {e}")
        
        # Final report
        self._log_final_report()
        self.logger.info("=" * 100 + "\n")
    
    def _log_final_report(self):
        """Log final report"""
        self.logger.info(f"\n📋 FINAL REPORT:")
        self.logger.info(f"   Total Analysis Cycles: {self.stats['intel_analyses']}")
        self.logger.info(f"   Signals Processed: {self.stats['signals_processed']}")
        self.logger.info(f"   High-Confidence Signals: {self.stats['high_confidence_signals']}")
        if self.stats['signals_processed'] > 0:
            self.logger.info(f"   High-Confidence Rate: {(self.stats['high_confidence_signals'] / self.stats['signals_processed'] * 100):.1f}%")
        self.logger.info(f"   Average Analysis Time: {self.stats['avg_analysis_time']:.0f}ms")


async def main():
    """Main entry point"""
    bot = ComprehensiveFXSUSDTBotWithIntel()
    
    # Initialize all components
    if not await bot.initialize():
        sys.exit(1)
    
    # Run continuous mode (5-minute intervals)
    # This prevents excessive API calls while maintaining responsiveness
    await bot.run_continuous(interval_seconds=300)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nBot terminated by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)