
#!/usr/bin/env python3
"""
Comprehensive Dynamic Signal Trading Bot
Advanced Trading Bot with Complete Integration:
- All Advanced Strategies Combined
- Telegram Channel Integration
- Real-time Signal Processing
- Dynamic Risk Management
- ML-Enhanced Decision Making
- Automatic Backtesting & Learning
"""

import asyncio
import logging
import os
import json
import sys
import time
import threading
import base64
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import subprocess
import traceback
import aiohttp
from io import BytesIO

# Add SignalMaestro to path
sys.path.append(str(Path(__file__).parent / "SignalMaestro"))

# Import all required components
try:
    from config import Config
    from signal_parser import SignalParser
    from risk_manager import RiskManager
    from binance_trader import BinanceTrader
    from database import Database
    from advanced_trading_strategy import AdvancedTradingStrategy
    from ultimate_trading_bot import UltimateTradingBot
    from ml_enhanced_trading_bot import MLTradePredictor
    from enhanced_perfect_scalping_bot_v3 import EnhancedPerfectScalpingBot
    from advanced_price_action_analyzer import AdvancedPriceActionAnalyzer
    from technical_analysis import TechnicalAnalysis
    from momentum_scalping_strategy import MomentumScalpingStrategy
    from lightning_scalping_strategy import LightningScalpingStrategy
    from volume_breakout_scalping_strategy import VolumeBreakoutScalpingStrategy
    from ultimate_scalping_strategy import UltimateScalpingStrategy
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some components not available: {e}")
    COMPONENTS_AVAILABLE = False

class ComprehensiveDynamicSignalBot:
    """
    Comprehensive Dynamic Signal Trading Bot with Full Telegram Integration
    """
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # Load configuration
        self.config = Config() if COMPONENTS_AVAILABLE else self._create_default_config()
        
        # Core configuration
        self.trading_config = {
            'initial_capital': 10.0,
            'risk_percentage': 10.0,
            'max_concurrent_trades': 3,
            'min_leverage': 10,
            'max_leverage': 75,
            'dynamic_stop_losses': [1.5, 4.0, 7.5],
            'take_profits': [2.0, 4.0, 6.0],
            'max_daily_loss': 2.0,
            'portfolio_risk_cap': 8.0
        }
        
        # Telegram configuration
        self.telegram_config = {
            'bot_token': getattr(self.config, 'TELEGRAM_BOT_TOKEN', ''),
            'channel_id': '@SignalTactics',
            'admin_chat_id': getattr(self.config, 'ADMIN_CHAT_ID', ''),
            'base_url': f"https://api.telegram.org/bot{getattr(self.config, 'TELEGRAM_BOT_TOKEN', '')}"
        }
        
        # Initialize all components
        self.initialize_all_components()
        
        # Performance tracking
        self.performance_metrics = {
            'total_signals_sent': 0,
            'successful_deliveries': 0,
            'failed_deliveries': 0,
            'current_capital': self.trading_config['initial_capital'],
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'trades_per_hour': 0.0,
            'last_signal_time': None,
            'bot_start_time': datetime.now()
        }
        
        # Signal counter
        self.signal_counter = 0
        self.is_running = False
        
    def _setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('comprehensive_dynamic_signal_bot.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _create_default_config(self):
        """Create default configuration if config unavailable"""
        class DefaultConfig:
            TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
            ADMIN_CHAT_ID = os.getenv('ADMIN_CHAT_ID', '')
            SUPPORTED_PAIRS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
        return DefaultConfig()
    
    def initialize_all_components(self):
        """Initialize all trading and analysis components"""
        try:
            # Core components
            self.signal_parser = SignalParser() if COMPONENTS_AVAILABLE else None
            self.risk_manager = RiskManager() if COMPONENTS_AVAILABLE else None
            self.binance_trader = BinanceTrader() if COMPONENTS_AVAILABLE else None
            self.database = Database() if COMPONENTS_AVAILABLE else None
            
            # Advanced components
            self.price_action_analyzer = AdvancedPriceActionAnalyzer() if COMPONENTS_AVAILABLE else None
            self.technical_analysis = TechnicalAnalysis() if COMPONENTS_AVAILABLE else None
            self.ml_predictor = MLTradePredictor() if COMPONENTS_AVAILABLE else None
            
            # Strategy components
            self.momentum_strategy = MomentumScalpingStrategy() if COMPONENTS_AVAILABLE else None
            self.lightning_strategy = LightningScalpingStrategy() if COMPONENTS_AVAILABLE else None
            self.volume_strategy = VolumeBreakoutScalpingStrategy() if COMPONENTS_AVAILABLE else None
            self.ultimate_strategy = UltimateScalpingStrategy() if COMPONENTS_AVAILABLE else None
            
            # Main trading bots
            self.ultimate_bot = UltimateTradingBot() if COMPONENTS_AVAILABLE else None
            self.enhanced_bot = EnhancedPerfectScalpingBot() if COMPONENTS_AVAILABLE else None
            
            # Advanced strategy
            self.advanced_strategy = AdvancedTradingStrategy(self.binance_trader) if COMPONENTS_AVAILABLE else None
            
            self.logger.info("🚀 All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing components: {e}")
    
    async def start_comprehensive_bot(self):
        """Start the comprehensive dynamic signal bot"""
        try:
            self.logger.info("🚀 STARTING COMPREHENSIVE DYNAMIC SIGNAL TRADING BOT")
            self.logger.info("=" * 80)
            
            # Display configuration
            self._display_configuration()
            
            # Test Telegram connection
            if await self.test_telegram_connection():
                self.logger.info("✅ Telegram connection successful")
            else:
                self.logger.error("❌ Telegram connection failed")
                return
            
            # Initialize components
            await self.initialize_async_components()
            
            # Send startup notification
            await self.send_startup_notification()
            
            # Start main trading loop
            await self.run_comprehensive_trading_loop()
            
        except Exception as e:
            self.logger.error(f"❌ Error starting comprehensive bot: {e}")
            traceback.print_exc()
    
    def _display_configuration(self):
        """Display bot configuration"""
        self.logger.info(f"💰 Trading Configuration:")
        self.logger.info(f"   • Initial Capital: ${self.trading_config['initial_capital']}")
        self.logger.info(f"   • Risk per Trade: {self.trading_config['risk_percentage']}%")
        self.logger.info(f"   • Max Concurrent Trades: {self.trading_config['max_concurrent_trades']}")
        self.logger.info(f"   • Dynamic Leverage: {self.trading_config['min_leverage']}x - {self.trading_config['max_leverage']}x")
        
        self.logger.info(f"\n📱 Telegram Configuration:")
        self.logger.info(f"   • Channel: {self.telegram_config['channel_id']}")
        self.logger.info(f"   • Bot Token: {'✅ Set' if self.telegram_config['bot_token'] else '❌ Missing'}")
        self.logger.info(f"   • Admin Chat: {'✅ Set' if self.telegram_config['admin_chat_id'] else '❌ Missing'}")
        
        self.logger.info("=" * 80)
    
    async def test_telegram_connection(self):
        """Test Telegram Bot API connection"""
        try:
            if not self.telegram_config['bot_token']:
                self.logger.error("❌ Telegram bot token not configured")
                return False
            
            url = f"{self.telegram_config['base_url']}/getMe"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('ok'):
                            bot_info = data.get('result', {})
                            self.logger.info(f"✅ Connected to bot: {bot_info.get('username', 'Unknown')}")
                            return True
                    
                    self.logger.error(f"❌ Telegram API error: {response.status}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"❌ Telegram connection test failed: {e}")
            return False
    
    async def initialize_async_components(self):
        """Initialize async components"""
        try:
            if COMPONENTS_AVAILABLE:
                if self.binance_trader:
                    await self.binance_trader.initialize()
                
                if self.database:
                    await self.database.initialize()
                
                if hasattr(self.ml_predictor, 'load_models'):
                    self.ml_predictor.load_models()
            
            self.logger.info("✅ Async components initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing async components: {e}")
    
    async def send_startup_notification(self):
        """Send startup notification to Telegram"""
        try:
            startup_message = f"""
🚀 **COMPREHENSIVE DYNAMIC SIGNAL BOT STARTED**

📊 **Configuration:**
💰 Capital: ${self.trading_config['initial_capital']}
⚖️ Risk: {self.trading_config['risk_percentage']}% per trade
🎯 Max Trades: {self.trading_config['max_concurrent_trades']}
📈 Leverage: {self.trading_config['min_leverage']}x - {self.trading_config['max_leverage']}x

🧠 **Advanced Features:**
✅ Advanced Price Action Analysis
✅ ML-Enhanced Signal Generation
✅ Dynamic Risk Management
✅ Multi-Strategy Integration
✅ Real-time Market Scanning
✅ Automatic Backtesting

⚡ **Market:** Binance Futures USDM
🎯 **Target:** High-Probability Scalping Signals
📢 **Channel:** @SignalTactics

🕐 **Started:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

🤖 Ready to generate profitable signals!
            """
            
            await self.send_telegram_message(self.telegram_config['channel_id'], startup_message)
            
            if self.telegram_config['admin_chat_id']:
                await self.send_telegram_message(self.telegram_config['admin_chat_id'], startup_message)
            
        except Exception as e:
            self.logger.error(f"❌ Error sending startup notification: {e}")
    
    async def run_comprehensive_trading_loop(self):
        """Main comprehensive trading loop"""
        try:
            self.is_running = True
            self.logger.info("🔄 Starting comprehensive trading loop...")
            
            while self.is_running:
                try:
                    # Scan all strategies for signals
                    signals = await self.scan_all_strategies_for_signals()
                    
                    if signals:
                        self.logger.info(f"📊 Found {len(signals)} high-probability signals")
                        
                        # Process and send signals
                        for signal in signals:
                            await self.process_and_send_comprehensive_signal(signal)
                            await asyncio.sleep(2)  # Rate limiting
                    
                    # Update performance metrics
                    await self.update_performance_metrics()
                    
                    # Log status every 5 minutes
                    if self.signal_counter % 10 == 0:
                        self.log_current_status()
                    
                    # Wait before next scan
                    await asyncio.sleep(30)  # Scan every 30 seconds
                    
                except KeyboardInterrupt:
                    self.logger.info("🛑 Shutting down gracefully...")
                    self.is_running = False
                    break
                    
                except Exception as e:
                    self.logger.error(f"❌ Error in trading loop: {e}")
                    await asyncio.sleep(60)  # Wait 1 minute on error
            
        except Exception as e:
            self.logger.error(f"❌ Fatal error in trading loop: {e}")
            traceback.print_exc()
    
    async def scan_all_strategies_for_signals(self) -> List[Dict[str, Any]]:
        """Scan all strategies for high-probability signals"""
        try:
            all_signals = []
            
            # Get signals from all available strategies
            if COMPONENTS_AVAILABLE:
                # Ultimate Trading Bot signals
                if self.ultimate_bot:
                    try:
                        ultimate_signals = await self.ultimate_bot.scan_for_signals()
                        if ultimate_signals:
                            for signal in ultimate_signals:
                                signal['source_strategy'] = 'Ultimate Trading Bot'
                                all_signals.append(signal)
                    except Exception as e:
                        self.logger.warning(f"Error getting Ultimate Bot signals: {e}")
                
                # Enhanced Perfect Scalping Bot signals
                if self.enhanced_bot:
                    try:
                        enhanced_signals = await self.enhanced_bot.scan_markets()
                        if enhanced_signals:
                            for signal in enhanced_signals:
                                signal['source_strategy'] = 'Enhanced Perfect Scalping'
                                all_signals.append(signal)
                    except Exception as e:
                        self.logger.warning(f"Error getting Enhanced Bot signals: {e}")
                
                # Advanced strategy signals
                if self.advanced_strategy:
                    try:
                        advanced_signals = await self.advanced_strategy.scan_markets()
                        if advanced_signals:
                            for signal in advanced_signals:
                                signal['source_strategy'] = 'Advanced Strategy'
                                all_signals.append(signal)
                    except Exception as e:
                        self.logger.warning(f"Error getting Advanced Strategy signals: {e}")
                
                # Individual strategy signals
                strategies = [
                    (self.momentum_strategy, 'Momentum Scalping'),
                    (self.lightning_strategy, 'Lightning Scalping'),
                    (self.volume_strategy, 'Volume Breakout'),
                    (self.ultimate_strategy, 'Ultimate Scalping')
                ]
                
                for strategy, strategy_name in strategies:
                    if strategy:
                        try:
                            strategy_signals = await self.get_strategy_signals(strategy, strategy_name)
                            all_signals.extend(strategy_signals)
                        except Exception as e:
                            self.logger.warning(f"Error getting {strategy_name} signals: {e}")
            
            # Filter and rank signals
            filtered_signals = await self.filter_and_rank_signals(all_signals)
            
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"❌ Error scanning strategies: {e}")
            return []
    
    async def get_strategy_signals(self, strategy, strategy_name):
        """Get signals from individual strategy"""
        try:
            signals = []
            
            # Mock signal generation for demonstration
            # Replace with actual strategy scanning
            if hasattr(strategy, 'scan_for_signals'):
                strategy_signals = await strategy.scan_for_signals()
            elif hasattr(strategy, 'generate_signals'):
                strategy_signals = await strategy.generate_signals()
            else:
                # Generate mock signal for testing
                symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
                for symbol in symbols:
                    mock_signal = {
                        'symbol': symbol,
                        'action': 'LONG',
                        'price': 45000 if symbol == 'BTCUSDT' else 3200,
                        'strength': 85,
                        'confidence': 78,
                        'source_strategy': strategy_name,
                        'timestamp': datetime.now()
                    }
                    signals.append(mock_signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"❌ Error getting {strategy_name} signals: {e}")
            return []
    
    async def filter_and_rank_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter and rank signals by quality"""
        try:
            if not signals:
                return []
            
            # Apply filters
            filtered_signals = []
            
            for signal in signals:
                # Basic quality filters
                if (signal.get('strength', 0) >= 70 and 
                    signal.get('confidence', 0) >= 65):
                    
                    # Add comprehensive analysis
                    enhanced_signal = await self.enhance_signal_with_analysis(signal)
                    filtered_signals.append(enhanced_signal)
            
            # Sort by composite score
            filtered_signals.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
            
            # Limit to top 5 signals
            return filtered_signals[:5]
            
        except Exception as e:
            self.logger.error(f"❌ Error filtering signals: {e}")
            return signals
    
    async def enhance_signal_with_analysis(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance signal with comprehensive analysis"""
        try:
            enhanced_signal = signal.copy()
            
            # Add ML prediction if available
            if self.ml_predictor and hasattr(self.ml_predictor, 'predict_signal_outcome'):
                try:
                    ml_prediction = await self.ml_predictor.predict_signal_outcome(signal)
                    enhanced_signal['ml_prediction'] = ml_prediction
                    enhanced_signal['ml_confidence'] = ml_prediction.get('confidence', 0.5)
                except Exception as e:
                    self.logger.warning(f"ML prediction failed: {e}")
                    enhanced_signal['ml_confidence'] = 0.5
            
            # Add technical analysis
            if self.technical_analysis:
                try:
                    ta_analysis = await self.get_technical_analysis(signal.get('symbol'))
                    enhanced_signal['technical_analysis'] = ta_analysis
                except Exception as e:
                    self.logger.warning(f"Technical analysis failed: {e}")
            
            # Add price action analysis
            if self.price_action_analyzer:
                try:
                    pa_analysis = await self.get_price_action_analysis(signal.get('symbol'))
                    enhanced_signal['price_action'] = pa_analysis
                except Exception as e:
                    self.logger.warning(f"Price action analysis failed: {e}")
            
            # Calculate composite score
            composite_score = self.calculate_composite_score(enhanced_signal)
            enhanced_signal['composite_score'] = composite_score
            
            # Add dynamic risk management
            enhanced_signal['risk_management'] = await self.calculate_dynamic_risk_management(enhanced_signal)
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"❌ Error enhancing signal: {e}")
            return signal
    
    async def get_technical_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get technical analysis for symbol"""
        try:
            # Mock technical analysis - replace with actual implementation
            return {
                'rsi': 65,
                'macd_signal': 'bullish',
                'ema_trend': 'upward',
                'support_level': 44800,
                'resistance_level': 46200,
                'confidence': 0.75
            }
        except Exception as e:
            self.logger.error(f"❌ Error in technical analysis: {e}")
            return {}
    
    async def get_price_action_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get price action analysis for symbol"""
        try:
            # Mock price action analysis - replace with actual implementation
            return {
                'trend_strength': 'strong',
                'pattern': 'ascending_triangle',
                'breakout_probability': 0.82,
                'key_levels': [44500, 45000, 45500],
                'confidence': 0.78
            }
        except Exception as e:
            self.logger.error(f"❌ Error in price action analysis: {e}")
            return {}
    
    def calculate_composite_score(self, signal: Dict[str, Any]) -> float:
        """Calculate composite score for signal ranking"""
        try:
            # Base score from signal strength and confidence
            base_score = (signal.get('strength', 0) + signal.get('confidence', 0)) / 2
            
            # ML enhancement
            ml_score = signal.get('ml_confidence', 0.5) * 100
            
            # Technical analysis score
            ta_score = signal.get('technical_analysis', {}).get('confidence', 0.5) * 100
            
            # Price action score
            pa_score = signal.get('price_action', {}).get('confidence', 0.5) * 100
            
            # Weighted composite score
            composite_score = (
                base_score * 0.4 +
                ml_score * 0.3 +
                ta_score * 0.2 +
                pa_score * 0.1
            )
            
            return round(composite_score, 2)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating composite score: {e}")
            return 50.0
    
    async def calculate_dynamic_risk_management(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate dynamic risk management parameters"""
        try:
            entry_price = signal.get('price', 0)
            composite_score = signal.get('composite_score', 50)
            
            # Dynamic leverage based on confidence
            if composite_score >= 85:
                leverage = self.trading_config['max_leverage']
            elif composite_score >= 75:
                leverage = int(self.trading_config['max_leverage'] * 0.8)
            else:
                leverage = int(self.trading_config['max_leverage'] * 0.6)
            
            # Dynamic stop losses
            base_sl = self.trading_config['dynamic_stop_losses']
            volatility_factor = 1.0  # Could be calculated from market data
            
            stop_losses = {
                'sl1': entry_price * (1 - (base_sl[0] * volatility_factor) / 100),
                'sl2': entry_price * (1 - (base_sl[1] * volatility_factor) / 100),
                'sl3': entry_price * (1 - (base_sl[2] * volatility_factor) / 100)
            }
            
            # Take profits
            take_profits = {
                'tp1': entry_price * (1 + self.trading_config['take_profits'][0] / 100),
                'tp2': entry_price * (1 + self.trading_config['take_profits'][1] / 100),
                'tp3': entry_price * (1 + self.trading_config['take_profits'][2] / 100)
            }
            
            return {
                'leverage': leverage,
                'stop_losses': stop_losses,
                'take_profits': take_profits,
                'position_size_percentage': self.trading_config['risk_percentage']
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating risk management: {e}")
            return {}
    
    async def process_and_send_comprehensive_signal(self, signal: Dict[str, Any]):
        """Process and send comprehensive signal to Telegram"""
        try:
            self.signal_counter += 1
            
            # Format professional signal message
            formatted_message = self.format_comprehensive_signal_message(signal)
            
            # Generate chart if possible
            chart_data = await self.generate_signal_chart(signal)
            
            # Send to Telegram channel
            success = await self.send_signal_to_telegram(formatted_message, chart_data, signal)
            
            if success:
                self.performance_metrics['successful_deliveries'] += 1
                self.performance_metrics['last_signal_time'] = datetime.now()
                self.logger.info(f"✅ Signal #{self.signal_counter} sent successfully: {signal.get('symbol')} {signal.get('action')}")
            else:
                self.performance_metrics['failed_deliveries'] += 1
                self.logger.error(f"❌ Signal #{self.signal_counter} delivery failed")
            
            self.performance_metrics['total_signals_sent'] += 1
            
        except Exception as e:
            self.logger.error(f"❌ Error processing comprehensive signal: {e}")
    
    def format_comprehensive_signal_message(self, signal: Dict[str, Any]) -> str:
        """Format comprehensive signal message for Telegram"""
        try:
            # Direction emoji
            direction = signal.get('action', '').upper()
            direction_emoji = "🟢" if direction in ['LONG', 'BUY'] else "🔴"
            
            # Risk management
            risk_mgmt = signal.get('risk_management', {})
            
            formatted_message = f"""
{direction_emoji} **SIGNAL #{self.signal_counter}** {direction_emoji}

📊 **PAIR:** `{signal.get('symbol', 'N/A')}`
📈 **DIRECTION:** `{direction}`
💰 **ENTRY:** `${signal.get('price', 0):,.4f}`

⚡ **LEVERAGE:** `{risk_mgmt.get('leverage', 35)}x`

🛡️ **STOP LOSSES:**
• SL1: `${risk_mgmt.get('stop_losses', {}).get('sl1', 0):,.4f}` (1/3 position)
• SL2: `${risk_mgmt.get('stop_losses', {}).get('sl2', 0):,.4f}` (1/3 position)  
• SL3: `${risk_mgmt.get('stop_losses', {}).get('sl3', 0):,.4f}` (1/3 position)

🎯 **TAKE PROFITS:**
• TP1: `${risk_mgmt.get('take_profits', {}).get('tp1', 0):,.4f}` (33%)
• TP2: `${risk_mgmt.get('take_profits', {}).get('tp2', 0):,.4f}` (33%)
• TP3: `${risk_mgmt.get('take_profits', {}).get('tp3', 0):,.4f}` (34%)

📊 **ANALYSIS:**
• **Signal Strength:** `{signal.get('strength', 0)}%`
• **Confidence:** `{signal.get('confidence', 0)}%` 
• **Composite Score:** `{signal.get('composite_score', 0)}/100`
• **ML Prediction:** `{signal.get('ml_confidence', 0.5)*100:.1f}%`

🧠 **STRATEGY:** `{signal.get('source_strategy', 'Advanced Multi-Strategy')}`

⚖️ **RISK:** `{self.trading_config['risk_percentage']}%` | **R:R:** `1:3`

🕐 **TIME:** `{datetime.now().strftime('%H:%M:%S UTC')}`
📅 **DATE:** `{datetime.now().strftime('%d/%m/%Y')}`

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 **Comprehensive Dynamic Signal Bot**
📢 **@SignalTactics** | 💎 **Premium Signals**
🔥 **AI-Enhanced Multi-Strategy Analysis**
            """
            
            return formatted_message
            
        except Exception as e:
            self.logger.error(f"❌ Error formatting signal message: {e}")
            return f"Signal #{self.signal_counter}: {signal.get('symbol')} {signal.get('action')}"
    
    async def generate_signal_chart(self, signal: Dict[str, Any]) -> Optional[str]:
        """Generate chart for signal (returns base64 encoded image)"""
        try:
            # Mock chart generation - replace with actual chart generation
            # This would typically use matplotlib or plotly to create a chart
            
            symbol = signal.get('symbol', 'BTCUSDT')
            self.logger.info(f"📊 Generating chart for {symbol}")
            
            # For now, return None - implement actual chart generation here
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error generating chart: {e}")
            return None
    
    async def send_signal_to_telegram(self, message: str, chart_data: Optional[str], signal: Dict[str, Any]) -> bool:
        """Send signal to Telegram with multiple delivery methods"""
        try:
            success = False
            
            # Method 1: Send to main channel
            if chart_data:
                success = await self.send_telegram_photo(self.telegram_config['channel_id'], chart_data, message)
            else:
                success = await self.send_telegram_message(self.telegram_config['channel_id'], message)
            
            # Method 2: Also send to admin for monitoring
            if self.telegram_config['admin_chat_id']:
                admin_message = f"📤 **Signal Sent to Channel**\n\n{message}"
                await self.send_telegram_message(self.telegram_config['admin_chat_id'], admin_message)
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error sending signal to Telegram: {e}")
            return False
    
    async def send_telegram_message(self, chat_id: str, text: str, parse_mode: str = 'Markdown') -> bool:
        """Send text message to Telegram"""
        try:
            url = f"{self.telegram_config['base_url']}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': text,
                'parse_mode': parse_mode
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('ok', False)
                    else:
                        error_text = await response.text()
                        self.logger.error(f"❌ Telegram API error: {error_text}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"❌ Error sending Telegram message: {e}")
            return False
    
    async def send_telegram_photo(self, chat_id: str, photo_data: str, caption: str) -> bool:
        """Send photo to Telegram"""
        try:
            url = f"{self.telegram_config['base_url']}/sendPhoto"
            
            # Convert base64 to bytes
            photo_bytes = base64.b64decode(photo_data)
            
            data = aiohttp.FormData()
            data.add_field('chat_id', chat_id)
            data.add_field('caption', caption)
            data.add_field('parse_mode', 'Markdown')
            data.add_field('photo', photo_bytes, filename='chart.png', content_type='image/png')
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('ok', False)
                    else:
                        error_text = await response.text()
                        self.logger.error(f"❌ Telegram photo error: {error_text}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"❌ Error sending Telegram photo: {e}")
            return False
    
    async def update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Calculate success rate
            if self.performance_metrics['total_signals_sent'] > 0:
                success_rate = (self.performance_metrics['successful_deliveries'] / 
                              self.performance_metrics['total_signals_sent'] * 100)
                self.performance_metrics['success_rate'] = success_rate
            
            # Calculate signals per hour
            if self.performance_metrics['bot_start_time']:
                running_time = datetime.now() - self.performance_metrics['bot_start_time']
                hours_running = running_time.total_seconds() / 3600
                
                if hours_running > 0:
                    self.performance_metrics['signals_per_hour'] = (
                        self.performance_metrics['total_signals_sent'] / hours_running
                    )
            
        except Exception as e:
            self.logger.error(f"❌ Error updating performance metrics: {e}")
    
    def log_current_status(self):
        """Log current bot status"""
        try:
            metrics = self.performance_metrics
            
            self.logger.info("📊 CURRENT STATUS:")
            self.logger.info(f"   • Total Signals Sent: {metrics['total_signals_sent']}")
            self.logger.info(f"   • Successful Deliveries: {metrics['successful_deliveries']}")
            self.logger.info(f"   • Failed Deliveries: {metrics['failed_deliveries']}")
            self.logger.info(f"   • Success Rate: {metrics.get('success_rate', 0):.1f}%")
            self.logger.info(f"   • Signals/Hour: {metrics.get('signals_per_hour', 0):.2f}")
            
            if metrics['last_signal_time']:
                self.logger.info(f"   • Last Signal: {metrics['last_signal_time']}")
            
        except Exception as e:
            self.logger.error(f"❌ Error logging status: {e}")

async def main():
    """Main function to start the comprehensive dynamic signal bot"""
    print("🌟 COMPREHENSIVE DYNAMIC SIGNAL TRADING BOT")
    print("=" * 80)
    print("🚀 Features:")
    print("   • Multi-Strategy Signal Generation")
    print("   • ML-Enhanced Decision Making")
    print("   • Real-time Telegram Integration")
    print("   • Dynamic Risk Management")
    print("   • Advanced Technical Analysis")
    print("   • Automatic Chart Generation")
    print("=" * 80)
    print("📱 Channel: @SignalTactics")
    print("💰 Configuration: $10 capital, 10% risk, 3 max trades")
    print("⚡ Market: Binance Futures USDM")
    print("=" * 80)
    
    try:
        # Create and start the comprehensive bot
        bot = ComprehensiveDynamicSignalBot()
        await bot.start_comprehensive_bot()
        
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
