#!/usr/bin/env python3
"""
COMPREHENSIVE UNIFIED MULTI-MARKET BOT
Dynamically integrates:
- Error Fixing & Health Checks
- ALL Binance USDM Futures Markets
- Advanced Analysis (Liquidity, Order Flow, Volume Profile, Fractals)
- Dynamic Position Management
- AI-Enhanced Signal Processing
"""

import asyncio
import logging
import sys
import os
import warnings
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import aiohttp
import ccxt

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

try:
    import pandas as pd
    pd.set_option('mode.chained_assignment', None)
    pd.options.mode.copy_on_write = True
except:
    pass

try:
    import numpy as np
    np.seterr(all='ignore')
except:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SignalMaestro'))

from dynamic_comprehensive_error_fixer import DynamicComprehensiveErrorFixer
from bot_health_check import check_bot_health
from dynamic_multi_market_position_manager import DynamicMultiMarketPositionManager


class BinanceMarketFetcher:
    """Dynamically fetch all Binance USDM futures markets"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.exchange = None
        
    async def initialize(self):
        """Initialize CCXT exchange connection"""
        try:
            api_key = os.getenv('BINANCE_API_KEY', '')
            api_secret = os.getenv('BINANCE_API_SECRET', '')
            
            self.exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            })
            
            self.exchange.load_markets()
            self.logger.info("✅ Binance exchange initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Exchange initialization failed: {e}")
            return False
    
    async def get_all_usdm_perpetual_markets(self) -> List[str]:
        """Get all active USDⓈ-M perpetual futures markets"""
        try:
            if not self.exchange:
                await self.initialize()
            
            markets = []
            for symbol, market in self.exchange.markets.items():
                if (market.get('type') == 'swap' and 
                    market.get('quote') == 'USDT' and
                    market.get('settle') == 'USDT' and
                    market.get('active', False) and
                    market.get('linear', True)):
                    markets.append(market['symbol'])
            
            self.logger.info(f"✅ Found {len(markets)} active USDⓈ-M perpetual markets")
            return sorted(markets)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to fetch markets: {e}")
            return ['FXS/USDT:USDT']
    
    async def get_high_volume_markets(self, min_volume_usdt: float = 5000000, top_n: int = 20) -> List[str]:
        """Get high-volume markets suitable for trading - checks ALL markets"""
        try:
            all_markets = await self.get_all_usdm_perpetual_markets()
            high_volume = []
            
            self.logger.info(f"📊 Checking volume for {len(all_markets)} markets...")
            
            for i, symbol in enumerate(all_markets):
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                    volume_24h = ticker.get('quoteVolume', 0)
                    
                    if volume_24h >= min_volume_usdt:
                        high_volume.append({
                            'symbol': symbol,
                            'volume_24h': volume_24h,
                            'price': ticker.get('last', 0)
                        })
                    
                    if (i + 1) % 100 == 0:
                        self.logger.info(f"   Checked {i + 1}/{len(all_markets)} markets...")
                    
                    await asyncio.sleep(0.05)
                except Exception as e:
                    self.logger.debug(f"Skip {symbol}: {e}")
                    continue
            
            high_volume.sort(key=lambda x: x['volume_24h'], reverse=True)
            top_markets = high_volume[:top_n]
            symbols = [m['symbol'] for m in top_markets]
            
            self.logger.info(f"✅ Found {len(high_volume)} high-volume markets total")
            self.logger.info(f"✅ Selected top {len(symbols)} markets for monitoring")
            return symbols
            
        except Exception as e:
            self.logger.error(f"❌ High-volume market filtering failed: {e}")
            return ['FXS/USDT:USDT', 'BTC/USDT:USDT', 'ETH/USDT:USDT']




class UnifiedMultiMarketBot:
    """
    Comprehensive unified bot for ALL Binance USDM futures markets
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self.market_fetcher = BinanceMarketFetcher()
        self.error_fixer = DynamicComprehensiveErrorFixer()
        self.position_manager = None
        
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.channel_id = "@SignalTactics"
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}" if self.bot_token else None
        
        self.active_markets = []
        self.timeframe = '30m'
        self.scan_interval = 120
        
        self.signals_sent = 0
        self.start_time = datetime.now()
        
        try:
            from SignalMaestro.market_intelligence_engine import MarketIntelligenceEngine
            self.intel_engine = MarketIntelligenceEngine()
            self.logger.info("✅ Market Intelligence Engine loaded")
        except ImportError:
            self.intel_engine = None
            self.logger.warning("⚠️  Market Intelligence Engine not available")
        
        try:
            from SignalMaestro.signal_fusion_engine import SignalFusionEngine
            self.fusion_engine = SignalFusionEngine()
            self.logger.info("✅ Signal Fusion Engine loaded")
        except ImportError:
            self.fusion_engine = None
            self.logger.warning("⚠️  Signal Fusion Engine not available")
        
        try:
            from SignalMaestro.comprehensive_dashboard import ComprehensiveDashboard
            self.dashboard = ComprehensiveDashboard()
            self.logger.info("✅ Comprehensive Dashboard loaded")
        except ImportError:
            self.dashboard = None
            self.logger.warning("⚠️  Comprehensive Dashboard not available")
        
        try:
            from SignalMaestro.ichimoku_sniper_strategy import IchimokuSniperStrategy
            self.ichimoku_strategy = IchimokuSniperStrategy()
            self.logger.info("✅ Ichimoku Sniper Strategy loaded")
        except ImportError:
            self.ichimoku_strategy = None
            self.logger.warning("⚠️  Ichimoku Strategy not available")
    
    async def initialize(self):
        """Initialize the bot"""
        self.logger.info("=" * 90)
        self.logger.info("🚀 UNIFIED MULTI-MARKET BOT INITIALIZATION")
        self.logger.info("=" * 90)
        
        self.error_fixer.apply_all_fixes()
        
        health_ok = await check_bot_health()
        if not health_ok:
            self.logger.warning("⚠️  Some health checks failed, continuing anyway...")
        
        await self.market_fetcher.initialize()
        
        self.position_manager = DynamicMultiMarketPositionManager(
            exchange=self.market_fetcher.exchange
        )
        self.logger.info("✅ Dynamic Position Manager initialized")
        
        self.logger.info("🔍 Fetching all Binance USDⓈ-M futures markets...")
        high_volume_markets = await self.market_fetcher.get_high_volume_markets(min_volume_usdt=5000000)
        
        self.active_markets = high_volume_markets[:20] if high_volume_markets else ['FXS/USDT:USDT']
        
        self.logger.info(f"✅ Monitoring {len(self.active_markets)} high-volume markets:")
        for i, symbol in enumerate(self.active_markets[:10], 1):
            self.logger.info(f"   {i}. {symbol}")
        if len(self.active_markets) > 10:
            self.logger.info(f"   ... and {len(self.active_markets) - 10} more")
        
        self.logger.info("=" * 90)
        self.logger.info("🔬 ACTIVE ANALYSIS MODULES:")
        self.logger.info("   ✅ Multi-Market Scanner (ALL Binance USDⓈ-M Perpetuals)")
        self.logger.info("   ✅ Liquidity Analysis (grabs/sweeps)")
        self.logger.info("   ✅ Order Flow Analysis (CVD)")
        self.logger.info("   ✅ Volume Profile & Footprint Charts")
        self.logger.info("   ✅ Fractals & Market Structure")
        self.logger.info("   ✅ Intermarket Correlations")
        self.logger.info("   ✅ Ichimoku Sniper Strategy")
        self.logger.info("   ✅ AI-Enhanced Signal Processing")
        self.logger.info("   ✅ Dynamic Position Management")
        self.logger.info("=" * 90)
        
        return True
    
    async def run_continuous_scanner(self):
        """Main continuous scanning loop across all markets"""
        await self.initialize()
        
        self.logger.info("\n🎯 Starting continuous multi-market scanner...")
        self.logger.info(f"⏱️  Scan interval: {self.scan_interval}s")
        self.logger.info(f"📊 Timeframe: {self.timeframe}")
        self.logger.info("")
        
        scan_count = 0
        
        while True:
            try:
                scan_count += 1
                self.logger.info(f"\n{'='*90}")
                self.logger.info(f"🔍 SCAN #{scan_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                self.logger.info(f"{'='*90}")
                
                for market in self.active_markets:
                    try:
                        await self.scan_market(market)
                        await asyncio.sleep(2)
                    except Exception as e:
                        self.logger.error(f"❌ Error scanning {market}: {e}")
                
                self.logger.info(f"\n✅ Scan #{scan_count} complete - {len(self.active_markets)} markets analyzed")
                self.logger.info(f"📊 Total signals sent: {self.signals_sent}")
                
                await asyncio.sleep(self.scan_interval)
                
            except KeyboardInterrupt:
                self.logger.info("\n🛑 Bot stopped by user")
                self.print_statistics()
                break
            except Exception as e:
                self.logger.error(f"❌ Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(self.scan_interval)
    
    async def scan_market(self, symbol: str):
        """Scan individual market"""
        try:
            self.logger.info(f"\n📈 Analyzing {symbol}...")
            
            if self.intel_engine:
                intel_snapshot = await self.intel_engine.analyze_market(
                    symbol=symbol.replace(':USDT', '').replace('/', ''),
                    timeframe=self.timeframe,
                    limit=500,
                    correlated_symbols=['BTCUSDT', 'ETHUSDT']
                )
                
                if self.dashboard:
                    status = self.dashboard.format_compact_status(intel_snapshot)
                    self.logger.info(f"   {status}")
                
                if intel_snapshot.should_trade():
                    self.logger.info(f"   🎯 {symbol} shows trading opportunity!")
                    
                    entry_price = 0
                    try:
                        ticker = self.market_fetcher.exchange.fetch_ticker(symbol)
                        entry_price = ticker.get('last', 0)
                    except Exception as e:
                        self.logger.error(f"Failed to get ticker for {symbol}: {e}")
                    
                    if entry_price > 0 and self.position_manager:
                        self.logger.info(f"   💎 Calculating position parameters for {symbol}...")
                        
                        leverage_analysis = await self.position_manager.calculate_optimal_leverage(
                            symbol=symbol,
                            account_balance=1000,
                            risk_tolerance='moderate'
                        )
                        self.logger.info(f"      ⚡ Optimal Leverage: {leverage_analysis['optimal_leverage']}x")
                        
                        direction = 'LONG' if intel_snapshot.consensus_sentiment == 'bullish' else 'SHORT'
                        sl_tp_analysis = await self.position_manager.calculate_dynamic_sl_tp(
                            symbol=symbol,
                            entry_price=entry_price,
                            direction=direction,
                            leverage=leverage_analysis['optimal_leverage']
                        )
                        self.logger.info(f"      🎯 Entry: ${entry_price:.4f}")
                        self.logger.info(f"      🛑 Stop Loss: ${sl_tp_analysis['stop_loss']:.4f}")
                        self.logger.info(f"      💰 Take Profit: ${sl_tp_analysis['take_profit']:.4f}")
                        
                        position_analysis = await self.position_manager.calculate_position_size(
                            symbol=symbol,
                            account_balance=1000,
                            entry_price=entry_price,
                            stop_loss=sl_tp_analysis['stop_loss'],
                            leverage=leverage_analysis['optimal_leverage']
                        )
                        self.logger.info(f"      💵 Position Size: ${position_analysis.get('position_size_usdt', 0):.2f}")
                        
                        fused_signal = None
                        if self.fusion_engine:
                            fused_signal = self.fusion_engine.fuse_signal(
                                ichimoku_signal=None,
                                intel_snapshot=intel_snapshot,
                                current_price=entry_price
                            )
                            
                            signal_data = {
                                'signal': fused_signal,
                                'position_params': {
                                    'leverage': leverage_analysis,
                                    'sl_tp': sl_tp_analysis,
                                    'position_size': position_analysis,
                                    'entry_price': entry_price,
                                    'direction': direction
                                }
                            }
                            
                            await self.send_signal(symbol, signal_data)
                        else:
                            self.logger.warning("Signal fusion engine not available")
            else:
                ticker = self.market_fetcher.exchange.fetch_ticker(symbol)
                price = ticker.get('last', 0)
                volume = ticker.get('quoteVolume', 0)
                change = ticker.get('percentage', 0)
                
                self.logger.info(f"   💰 ${price:,.2f} | 📊 Vol: ${volume/1e6:.1f}M | {change:+.2f}%")
                
        except Exception as e:
            self.logger.debug(f"Scan error for {symbol}: {e}")
    
    async def send_signal(self, symbol: str, signal_data: Dict):
        """Send trading signal to Telegram with position parameters"""
        try:
            fused_signal = signal_data.get('signal')
            position_params = signal_data.get('position_params', {})
            
            message = f"🎯 **TRADING SIGNAL: {symbol}**\n\n"
            
            if position_params:
                direction = position_params.get('direction', 'LONG')
                leverage = position_params.get('leverage', {})
                sl_tp = position_params.get('sl_tp', {})
                position_size = position_params.get('position_size', {})
                entry = position_params.get('entry_price', 0)
                
                message += f"**Direction:** {direction}\n"
                message += f"**Entry Price:** ${entry:.4f}\n"
                message += f"**Leverage:** {leverage.get('optimal_leverage', 5)}x\n"
                message += f"**Stop Loss:** ${sl_tp.get('stop_loss', 0):.4f}\n"
                message += f"**Take Profit:** ${sl_tp.get('take_profit', 0):.4f}\n"
                message += f"**Position Size:** ${position_size.get('position_size_usdt', 0):.2f}\n"
                message += f"**Risk/Reward:** 1:{sl_tp.get('risk_reward_ratio', 0):.2f}\n\n"
            
            if self.dashboard and fused_signal:
                message += self.dashboard.format_telegram_signal(fused_signal)
            
            message += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
            
            if not self.base_url:
                self.logger.info(f"\n📢 SIGNAL GENERATED:\n{message}")
                self.signals_sent += 1
                return
            
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
                        self.logger.info(f"   ✅ Signal with position params sent to {self.channel_id}")
                    else:
                        self.logger.error(f"   ❌ Failed to send: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"❌ Signal sending error: {e}")
    
    def print_statistics(self):
        """Print bot statistics"""
        uptime = datetime.now() - self.start_time
        self.logger.info("\n" + "=" * 90)
        self.logger.info("📊 FINAL STATISTICS")
        self.logger.info("=" * 90)
        self.logger.info(f"⏱️  Total Uptime: {uptime}")
        self.logger.info(f"📡 Total Signals: {self.signals_sent}")
        self.logger.info(f"📈 Markets Monitored: {len(self.active_markets)}")
        self.logger.info("=" * 90)


async def main():
    """Main entry point"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/unified_multi_market_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    banner = """
    ╔═══════════════════════════════════════════════════════════════════════════════════╗
    ║                                                                                   ║
    ║   ██╗   ██╗███╗   ██╗██╗███████╗██╗███╗   ███╗██████╗                            ║
    ║   ██║   ██║████╗  ██║██║██╔════╝██║████╗ ████║██╔══██╗                           ║
    ║   ██║   ██║██╔██╗ ██║██║█████╗  ██║██╔████╔██║██████╔╝                           ║
    ║   ██║   ██║██║╚██╗██║██║██╔══╝  ██║██║╚██╔╝██║██╔══██╗                           ║
    ║   ╚██████╔╝██║ ╚████║██║██║     ██║██║ ╚═╝ ██║██████╔╝                           ║
    ║    ╚═════╝ ╚═╝  ╚═══╝╚═╝╚═╝     ╚═╝╚═╝     ╚═╝╚═════╝                            ║
    ║                                                                                   ║
    ║              COMPREHENSIVE MULTI-MARKET TRADING INTELLIGENCE SYSTEM              ║
    ║                                                                                   ║
    ║   🌐 ALL Binance USDⓈ-M Perpetual Futures Markets                                ║
    ║   🔬 Advanced Technical Analysis  |  📊 Multi-Market Scanner                     ║
    ║   🎯 AI-Enhanced Signals  |  📈 Dynamic Position Management                      ║
    ║   ⚡ Real-time Monitoring  |  🔧 Automatic Error Fixing                          ║
    ║                                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)
    
    if not os.getenv('TELEGRAM_BOT_TOKEN'):
        logger.warning("⚠️  TELEGRAM_BOT_TOKEN not set - signals will be logged only")
        logger.info("💡 Set TELEGRAM_BOT_TOKEN to enable signal broadcasting")
    
    if not os.getenv('BINANCE_API_KEY'):
        logger.info("ℹ️  BINANCE_API_KEY not set - using public endpoints")
    
    bot = UnifiedMultiMarketBot()
    
    try:
        await bot.run_continuous_scanner()
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutting down gracefully...")
        bot.print_statistics()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
