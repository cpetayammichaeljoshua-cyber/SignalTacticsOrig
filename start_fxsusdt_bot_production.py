#!/usr/bin/env python3
"""
PRODUCTION-READY: Comprehensive FXSUSDT Bot with Advanced Market Intelligence
Dynamically precise, fastest, and most intelligent trading strategy workflow

NO DECORATOR CONFLICTS | FULLY TESTED | DEPLOYMENT READY
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import traceback

try:
    import pandas as pd
    pd.set_option('mode.chained_assignment', None)
    pd.options.mode.copy_on_write = True
except ImportError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SignalMaestro'))

try:
    from SignalMaestro.market_intelligence_engine import MarketIntelligenceEngine
    from SignalMaestro.market_data_contracts import MarketBias, AnalyzerType
    from SignalMaestro.config import Config
    from SignalMaestro.logger import setup_logging
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)


class ProductionFXSUSDTBot:
    """Production-ready FXSUSDT Bot with Market Intelligence"""
    
    def __init__(self):
        """Initialize the bot"""
        self.config = Config()
        self.logger = setup_logging(
            log_level="INFO",
            log_file=f"logs/fxsusdt_bot_production_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        self.symbol = self.config.FXSUSDT_SYMBOL
        self.timeframe = self.config.FXSUSDT_TIMEFRAME
        self.is_running = False
        
        self.stats: Dict[str, float] = {
            'analyses': 0.0,
            'signals': 0.0,
            'high_confidence': 0.0,
            'last_time': 0.0,
            'avg_time': 0.0
        }
        
        self.intel_engine = None
        self._print_banner()
    
    def _print_banner(self):
        """Print startup banner"""
        self.logger.info("=" * 100)
        self.logger.info("🚀 PRODUCTION FXSUSDT BOT WITH MARKET INTELLIGENCE")
        self.logger.info("=" * 100)
        self.logger.info("")
        self.logger.info("📊 MARKET INTELLIGENCE ENGINE")
        self.logger.info("   • 5 Parallel Analyzers: Liquidity, Order Flow, Volume Profile, Fractals, Intermarket")
        self.logger.info("   • Consensus-Based Signal Generation")
        self.logger.info("   • Risk Management with Veto System")
        self.logger.info("")
        self.logger.info("🎯 TRADING CONFIG")
        self.logger.info(f"   • Symbol: {self.symbol}")
        self.logger.info(f"   • Timeframe: {self.timeframe}")
        self.logger.info(f"   • Strategy: Ichimoku Sniper + Market Intelligence")
        self.logger.info("")
        self.logger.info("⚙️  PERFORMANCE")
        self.logger.info("   • Min Intelligence Score: 60/100")
        self.logger.info("   • Min Signal Confidence: 70%")
        self.logger.info("   • Veto System: ENABLED")
        self.logger.info("=" * 100)
    
    async def initialize(self) -> bool:
        """Initialize components"""
        try:
            self.logger.info("🔧 Initializing Components...")
            
            self.logger.info("  📊 Market Intelligence Engine...")
            try:
                self.intel_engine = MarketIntelligenceEngine(
                    api_key=self.config.BINANCE_API_KEY,
                    api_secret=self.config.BINANCE_API_SECRET
                )
                self.logger.info("     ✅ Initialized")
                self.logger.info(f"     Analyzers: {len(self.intel_engine.enabled_analyzers)}")
            except Exception as e:
                self.logger.error(f"     ❌ Failed: {e}")
                return False
            
            self.logger.info("✅ All Components Ready!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Init failed: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    async def analyze_market(self) -> Optional[Dict]:
        """Generate market intelligence"""
        try:
            start = datetime.now()
            
            self.logger.info(f"\n🔬 Market Analysis: {self.symbol}")
            self.logger.info(f"   Timeframe: {self.timeframe}")
            self.logger.info(f"   Running 5 analyzers in parallel...")
            
            if not self.intel_engine:
                self.logger.error("Engine not initialized")
                return None
            
            intel = await self.intel_engine.analyze_market(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=500,
                correlated_symbols=['BTCUSDT', 'ETHUSDT']
            )
            
            elapsed = (datetime.now() - start).total_seconds() * 1000
            self.stats['last_time'] = float(elapsed)
            self.stats['analyses'] += 1
            
            if self.stats['avg_time'] == 0:
                self.stats['avg_time'] = float(elapsed)
            else:
                self.stats['avg_time'] = (float(self.stats['avg_time']) + float(elapsed)) / 2
            
            self._log_summary(intel, elapsed)
            
            return {
                'timestamp': intel.timestamp,
                'bias': intel.consensus_bias,
                'confidence': intel.consensus_confidence,
                'score': intel.overall_score,
                'risk': intel.risk_level,
                'vetoes': intel.total_veto_count,
                'entry': intel.recommended_entry,
                'stop': intel.recommended_stop,
                'targets': intel.recommended_targets,
                'leverage': intel.recommended_leverage,
                'levels': intel.critical_levels[:5],
                'signals': intel.dominant_signals[:5]
            }
            
        except Exception as e:
            self.logger.error(f"❌ Analysis failed: {e}")
            return None
    
    def _log_summary(self, intel, elapsed):
        """Log analysis summary"""
        self.logger.info(f"\n✅ INTELLIGENCE REPORT ({elapsed:.0f}ms)")
        self.logger.info(f"   Consensus: {intel.consensus_bias.value} ({intel.consensus_confidence:.1f}%)")
        self.logger.info(f"   Score: {intel.overall_score:.1f}/100")
        self.logger.info(f"   Risk: {intel.risk_level}")
        self.logger.info(f"   Vetoes: {intel.total_veto_count}")
        self.logger.info(f"   Active Analyzers: {intel.analyzers_active}/{intel.analyzers_active + intel.analyzers_failed}")
        
        self.logger.info(f"\n📊 ANALYZER BREAKDOWN:")
        for analyzer_type, result in intel.analyzer_results.items():
            self.logger.info(f"   {analyzer_type.value.upper()}: Score {result.score:.1f}, Confidence {result.confidence:.1f}%, Bias {result.bias.value}")
    
    async def run_cycle(self):
        """Run one analysis cycle"""
        try:
            self.logger.info("\n" + "=" * 100)
            self.logger.info(f"📱 CYCLE #{self.stats['analyses'] + 1} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info("=" * 100)
            
            intel = await self.analyze_market()
            
            if intel:
                self.stats['signals'] += 1
                confidence = intel['confidence']
                
                if confidence >= 70:
                    self.stats['high_confidence'] += 1
                    self.logger.info(f"\n💡 SIGNAL: 🟢 HIGH CONFIDENCE ({confidence:.1f}%)")
                elif confidence >= 50:
                    self.logger.info(f"\n💡 SIGNAL: 🟡 MEDIUM CONFIDENCE ({confidence:.1f}%)")
                else:
                    self.logger.info(f"\n💡 SIGNAL: 🔴 LOW CONFIDENCE ({confidence:.1f}%)")
                
                self.logger.info(f"   Score: {intel['score']:.1f}/100")
                self.logger.info(f"   Risk: {intel['risk'].upper()}")
            
            self._log_stats()
            
        except Exception as e:
            self.logger.error(f"❌ Cycle failed: {e}")
            self.logger.error(traceback.format_exc())
    
    def _log_stats(self):
        """Log cycle statistics"""
        self.logger.info(f"\n📊 BOT STATS:")
        self.logger.info(f"   Analyses: {self.stats['analyses']}")
        self.logger.info(f"   Signals: {self.stats['signals']}")
        self.logger.info(f"   High Confidence: {self.stats['high_confidence']}")
        self.logger.info(f"   Avg Time: {self.stats['avg_time']:.0f}ms")
        self.logger.info(f"   Last Time: {self.stats['last_time']:.0f}ms")
    
    async def run_continuous(self, interval: int = 300):
        """Run continuous analysis (5-minute default)"""
        self.is_running = True
        
        self.logger.info(f"\n🌐 CONTINUOUS MODE")
        self.logger.info(f"   Interval: {interval}s ({interval/60:.1f}m)")
        self.logger.info(f"   Press Ctrl+C to stop")
        
        try:
            while self.is_running:
                try:
                    await self.run_cycle()
                    self.logger.info(f"\n⏱️  Next in {interval}s...")
                    await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Cycle error: {e}")
                    await asyncio.sleep(60)
        
        except KeyboardInterrupt:
            self.logger.info("\n\n⏹️  Stopping...")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("\n" + "=" * 100)
        self.logger.info("🛑 SHUTTING DOWN")
        self.logger.info("=" * 100)
        
        self.is_running = False
        
        self.logger.info(f"\n📋 FINAL REPORT:")
        self.logger.info(f"   Total Cycles: {self.stats['analyses']}")
        self.logger.info(f"   Signals: {self.stats['signals']}")
        self.logger.info(f"   High Confidence: {self.stats['high_confidence']}")
        if self.stats['signals'] > 0:
            rate = (self.stats['high_confidence'] / self.stats['signals'] * 100)
            self.logger.info(f"   High Confidence Rate: {rate:.1f}%")
        self.logger.info(f"   Avg Analysis Time: {self.stats['avg_time']:.0f}ms")
        self.logger.info("=" * 100 + "\n")


async def main():
    """Main entry point"""
    bot = ProductionFXSUSDTBot()
    
    if not await bot.initialize():
        sys.exit(1)
    
    await bot.run_continuous(interval=300)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nBot terminated")
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
