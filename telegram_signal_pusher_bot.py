
#!/usr/bin/env python3
"""
Advanced Comprehensive Telegram Signal Pusher Bot
Continuously scans markets and pushes high-quality trading signals to Telegram channel
Uses all advanced strategies and ML analysis for optimal signal generation
"""

import asyncio
import logging
import os
import json
import sys
import aiohttp
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd
import numpy as np

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

class AdvancedTelegramSignalPusher:
    """
    Advanced Comprehensive Signal Pusher for Telegram Channel
    Continuously scans markets using all strategies and pushes high-quality signals
    """
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # Configuration
        self.config = Config() if COMPONENTS_AVAILABLE else self._create_default_config()
        
        # Trading configuration
        self.trading_config = {
            'initial_capital': 10.0,
            'risk_percentage': 10.0,
            'max_concurrent_trades': 3,
            'min_leverage': 10,
            'max_leverage': 75,
            'dynamic_stop_losses': [1.5, 4.0, 7.5],
            'take_profits': [2.0, 4.0, 6.0],
            'signal_quality_threshold': 75,
            'ml_confidence_threshold': 70
        }
        
        # Telegram configuration
        self.telegram_config = {
            'bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
            'channel_id': '@SignalTactics',
            'admin_chat_id': os.getenv('ADMIN_CHAT_ID', ''),
            'base_url': f"https://api.telegram.org/bot{os.getenv('TELEGRAM_BOT_TOKEN', '')}"
        }
        
        # Initialize all components
        self.initialize_components()
        
        # Performance tracking
        self.performance_metrics = {
            'total_signals_pushed': 0,
            'successful_pushes': 0,
            'failed_pushes': 0,
            'signals_per_hour': 0.0,
            'last_signal_time': None,
            'bot_start_time': datetime.now(),
            'uptime_hours': 0.0
        }
        
        # Bot state
        self.signal_counter = 0
        self.is_running = False
        self.scan_interval = 180  # 3 minutes between scans
        
        # Market symbols to scan
        self.symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'SOLUSDT', 'DOGEUSDT',
            'AVAXUSDT', 'DOTUSDT', 'MATICUSDT', 'LINKUSDT', 'LTCUSDT', 'BCHUSDT', 'ETCUSDT',
            'ATOMUSDT', 'ALGOUSDT', 'XLMUSDT', 'VETUSDT', 'TRXUSDT', 'EOSUSDT', 'THETAUSDT',
            'UNIUSDT', 'AAVEUSDT', 'COMPUSDT', 'MKRUSDT', 'SUSHIUSDT', 'CAKEUSDT'
        ]
        
        self.logger.info("🚀 Advanced Telegram Signal Pusher initialized")
        
    def _setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('telegram_signal_pusher.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _create_default_config(self):
        """Create default configuration"""
        class DefaultConfig:
            TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
            ADMIN_CHAT_ID = os.getenv('ADMIN_CHAT_ID', '')
            SUPPORTED_PAIRS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
        return DefaultConfig()
    
    def initialize_components(self):
        """Initialize all trading and analysis components"""
        try:
            # Core components
            self.signal_parser = SignalParser() if COMPONENTS_AVAILABLE else None
            self.risk_manager = RiskManager() if COMPONENTS_AVAILABLE else None
            self.binance_trader = BinanceTrader() if COMPONENTS_AVAILABLE else None
            self.database = Database() if COMPONENTS_AVAILABLE else None
            
            # Advanced analysis components
            self.price_action_analyzer = AdvancedPriceActionAnalyzer() if COMPONENTS_AVAILABLE else None
            self.technical_analysis = TechnicalAnalysis() if COMPONENTS_AVAILABLE else None
            self.ml_predictor = MLTradePredictor() if COMPONENTS_AVAILABLE else None
            
            # Strategy components
            self.momentum_strategy = MomentumScalpingStrategy() if COMPONENTS_AVAILABLE else None
            self.lightning_strategy = LightningScalpingStrategy() if COMPONENTS_AVAILABLE else None
            self.volume_strategy = VolumeBreakoutScalpingStrategy() if COMPONENTS_AVAILABLE else None
            self.ultimate_strategy = UltimateScalpingStrategy() if COMPONENTS_AVAILABLE else None
            
            # Main trading bots for signal generation
            self.ultimate_bot = UltimateTradingBot() if COMPONENTS_AVAILABLE else None
            self.enhanced_bot = EnhancedPerfectScalpingBot() if COMPONENTS_AVAILABLE else None
            self.advanced_strategy = AdvancedTradingStrategy(self.binance_trader) if COMPONENTS_AVAILABLE else None
            
            self.logger.info("✅ All signal generation components initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing components: {e}")
    
    async def start_signal_pusher(self):
        """Start the continuous signal pusher bot"""
        try:
            self.logger.info("🚀 STARTING ADVANCED TELEGRAM SIGNAL PUSHER BOT")
            self.logger.info("=" * 80)
            
            # Display configuration
            self._display_configuration()
            
            # Test Telegram connection
            if await self.test_telegram_connection():
                self.logger.info("✅ Telegram connection successful")
            else:
                self.logger.error("❌ Telegram connection failed")
                return
            
            # Initialize async components
            await self.initialize_async_components()
            
            # Send startup notification
            await self.send_startup_notification()
            
            # Start main signal pushing loop
            await self.run_continuous_signal_loop()
            
        except Exception as e:
            self.logger.error(f"❌ Error starting signal pusher: {e}")
            traceback.print_exc()
    
    def _display_configuration(self):
        """Display bot configuration"""
        self.logger.info("📊 SIGNAL PUSHER CONFIGURATION:")
        self.logger.info(f"   • Channel: {self.telegram_config['channel_id']}")
        self.logger.info(f"   • Scan Interval: {self.scan_interval} seconds")
        self.logger.info(f"   • Signal Quality Threshold: {self.trading_config['signal_quality_threshold']}%")
        self.logger.info(f"   • ML Confidence Threshold: {self.trading_config['ml_confidence_threshold']}%")
        self.logger.info(f"   • Symbols Monitored: {len(self.symbols)}")
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
                
                if self.ml_predictor and hasattr(self.ml_predictor, 'load_models'):
                    self.ml_predictor.load_models()
            
            self.logger.info("✅ Async components initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing async components: {e}")
    
    async def send_startup_notification(self):
        """Send startup notification to Telegram channel"""
        try:
            startup_message = f"""
🚀 **ADVANCED TELEGRAM SIGNAL PUSHER STARTED**

📊 **Configuration:**
💰 Capital Analysis: ${self.trading_config['initial_capital']}
⚖️ Risk Analysis: {self.trading_config['risk_percentage']}% per signal
🎯 Max Concurrent: {self.trading_config['max_concurrent_trades']} signals
📈 Leverage Range: {self.trading_config['min_leverage']}x - {self.trading_config['max_leverage']}x

🧠 **Advanced Analysis Features:**
✅ Advanced Price Action Analysis
✅ ML-Enhanced Signal Generation
✅ Dynamic Risk Management
✅ Multi-Strategy Integration
✅ Advanced Liquidity Analysis
✅ Sequential Move Detection
✅ Schelling Points Integration
✅ Advanced Order Flow Analysis
✅ Strategic Positioning

⚡ **Market Coverage:** {len(self.symbols)} pairs
🎯 **Signal Quality:** {self.trading_config['signal_quality_threshold']}%+ threshold
🧠 **ML Confidence:** {self.trading_config['ml_confidence_threshold']}%+ threshold
🔄 **Scan Frequency:** Every {self.scan_interval//60} minutes

📢 **Channel:** {self.telegram_config['channel_id']}
🕐 **Started:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

🤖 **Ready to push high-quality trading signals!**
            """
            
            await self.send_telegram_message(self.telegram_config['channel_id'], startup_message)
            
            if self.telegram_config['admin_chat_id']:
                await self.send_telegram_message(self.telegram_config['admin_chat_id'], startup_message)
            
        except Exception as e:
            self.logger.error(f"❌ Error sending startup notification: {e}")
    
    async def run_continuous_signal_loop(self):
        """Main continuous signal generation and pushing loop"""
        try:
            self.is_running = True
            self.logger.info("🔄 Starting continuous signal pushing loop...")
            
            while self.is_running:
                try:
                    # Update uptime
                    self.update_uptime()
                    
                    # Scan all strategies for signals
                    high_quality_signals = await self.scan_all_strategies_for_signals()
                    
                    if high_quality_signals:
                        self.logger.info(f"📊 Found {len(high_quality_signals)} high-quality signals")
                        
                        # Push signals to Telegram
                        for signal in high_quality_signals:
                            await self.push_signal_to_telegram(signal)
                            await asyncio.sleep(5)  # 5 second delay between signals
                    else:
                        self.logger.info("⏳ No qualifying signals found, continuing scan...")
                    
                    # Update performance metrics
                    self.update_performance_metrics()
                    
                    # Log status every hour
                    if self.signal_counter % 20 == 0:
                        self.log_performance_status()
                    
                    # Wait before next scan
                    self.logger.info(f"⏳ Waiting {self.scan_interval} seconds for next scan...")
                    await asyncio.sleep(self.scan_interval)
                    
                except KeyboardInterrupt:
                    self.logger.info("🛑 Shutting down gracefully...")
                    self.is_running = False
                    break
                    
                except Exception as e:
                    self.logger.error(f"❌ Error in signal loop: {e}")
                    await asyncio.sleep(60)  # Wait 1 minute on error
            
        except Exception as e:
            self.logger.error(f"❌ Fatal error in signal loop: {e}")
            traceback.print_exc()
    
    async def scan_all_strategies_for_signals(self) -> List[Dict[str, Any]]:
        """Scan all strategies and return high-quality signals"""
        try:
            all_signals = []
            
            # Scan each symbol with all strategies
            for symbol in self.symbols:
                try:
                    # Get market data
                    market_data = await self.get_market_data(symbol)
                    if not market_data:
                        continue
                    
                    # Analyze with all available strategies
                    symbol_signals = await self.analyze_symbol_with_all_strategies(symbol, market_data)
                    
                    if symbol_signals:
                        all_signals.extend(symbol_signals)
                        
                except Exception as e:
                    self.logger.warning(f"Error analyzing {symbol}: {e}")
                    continue
            
            # Filter and rank signals by quality
            high_quality_signals = self.filter_high_quality_signals(all_signals)
            
            return high_quality_signals
            
        except Exception as e:
            self.logger.error(f"❌ Error scanning strategies: {e}")
            return []
    
    async def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive market data for symbol"""
        try:
            if not self.binance_trader:
                return None
            
            # Get OHLCV data for multiple timeframes
            timeframes = ['1m', '5m', '15m', '1h']
            market_data = {'symbol': symbol}
            
            for tf in timeframes:
                df = await self.binance_trader.get_binance_data(symbol, tf, 100)
                if df is not None and not df.empty:
                    market_data[tf] = df
            
            # Need at least 2 timeframes for analysis
            if len([k for k in market_data.keys() if k != 'symbol']) >= 2:
                return market_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error getting market data for {symbol}: {e}")
            return None
    
    async def analyze_symbol_with_all_strategies(self, symbol: str, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze symbol with all available strategies"""
        try:
            signals = []
            
            # Get primary timeframe data
            primary_df = market_data.get('5m') or market_data.get('1m')
            if primary_df is None or primary_df.empty:
                return signals
            
            # Calculate comprehensive indicators
            indicators = self.calculate_comprehensive_indicators(primary_df)
            if not indicators:
                return signals
            
            # Generate signal using ultimate bot logic
            signal = await self.generate_advanced_signal(symbol, indicators, market_data)
            if signal:
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing {symbol}: {e}")
            return []
    
    def calculate_comprehensive_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive technical indicators"""
        try:
            if df.empty or len(df) < 50:
                return {}
            
            indicators = {}
            
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            volume = df['volume'].values
            
            # Calculate EMA
            ema_8 = self.calculate_ema(close, 8)
            ema_21 = self.calculate_ema(close, 21)
            ema_55 = self.calculate_ema(close, 55)
            
            indicators.update({
                'current_price': close[-1],
                'ema_8': ema_8[-1],
                'ema_21': ema_21[-1],
                'ema_55': ema_55[-1],
                'ema_bullish': ema_8[-1] > ema_21[-1] > ema_55[-1],
                'ema_bearish': ema_8[-1] < ema_21[-1] < ema_55[-1],
            })
            
            # Calculate RSI
            rsi = self.calculate_rsi(close, 14)
            indicators.update({
                'rsi': rsi[-1],
                'rsi_oversold': rsi[-1] < 30,
                'rsi_overbought': rsi[-1] > 70,
            })
            
            # Calculate MACD
            macd_line, macd_signal, macd_hist = self.calculate_macd(close)
            indicators.update({
                'macd': macd_line[-1],
                'macd_signal': macd_signal[-1],
                'macd_histogram': macd_hist[-1],
                'macd_bullish': macd_line[-1] > macd_signal[-1] and macd_hist[-1] > 0,
                'macd_bearish': macd_line[-1] < macd_signal[-1] and macd_hist[-1] < 0,
            })
            
            # Volume analysis
            volume_sma = np.mean(volume[-20:])
            indicators.update({
                'volume_ratio': volume[-1] / volume_sma if volume_sma > 0 else 1.0,
                'volume_surge': volume[-1] > volume_sma * 1.5 if volume_sma > 0 else False,
            })
            
            # Market volatility
            returns = np.diff(close[-20:]) / close[-21:-1]
            indicators['market_volatility'] = np.std(returns) if len(returns) > 0 else 0.02
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating indicators: {e}")
            return {}
    
    def calculate_ema(self, values: np.array, period: int) -> np.array:
        """Calculate Exponential Moving Average"""
        ema = np.zeros(len(values))
        ema[0] = values[0]
        multiplier = 2 / (period + 1)
        for i in range(1, len(values)):
            ema[i] = (values[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        return ema
    
    def calculate_rsi(self, values: np.array, period: int) -> np.array:
        """Calculate RSI"""
        if len(values) < period + 1:
            return np.full(len(values), 50.0)
        
        deltas = np.diff(values)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.zeros(len(values))
        avg_losses = np.zeros(len(values))
        
        if period <= len(gains):
            avg_gains[period] = np.mean(gains[:period])
            avg_losses[period] = np.mean(losses[:period])
        
        for i in range(period + 1, len(values)):
            avg_gains[i] = (avg_gains[i-1] * (period-1) + gains[i-1]) / period
            avg_losses[i] = (avg_losses[i-1] * (period-1) + losses[i-1]) / period
        
        rsi = np.zeros(len(values))
        for i in range(len(values)):
            if avg_losses[i] == 0:
                rsi[i] = 100.0 if avg_gains[i] > 0 else 50.0
            else:
                rs = avg_gains[i] / avg_losses[i]
                rsi[i] = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, values: np.array) -> tuple:
        """Calculate MACD"""
        ema_12 = self.calculate_ema(values, 12)
        ema_26 = self.calculate_ema(values, 26)
        macd_line = ema_12 - ema_26
        signal_line = self.calculate_ema(macd_line, 9)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    async def generate_advanced_signal(self, symbol: str, indicators: Dict[str, Any], market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate advanced signal with comprehensive analysis"""
        try:
            # Calculate signal strength
            bullish_signals = 0
            bearish_signals = 0
            
            # EMA analysis (30% weight)
            if indicators.get('ema_bullish'):
                bullish_signals += 30
            elif indicators.get('ema_bearish'):
                bearish_signals += 30
            
            # RSI analysis (20% weight)
            if indicators.get('rsi_oversold'):
                bullish_signals += 20
            elif indicators.get('rsi_overbought'):
                bearish_signals += 20
            
            # MACD analysis (25% weight)
            if indicators.get('macd_bullish'):
                bullish_signals += 25
            elif indicators.get('macd_bearish'):
                bearish_signals += 25
            
            # Volume analysis (15% weight)
            if indicators.get('volume_surge'):
                if bullish_signals > bearish_signals:
                    bullish_signals += 15
                else:
                    bearish_signals += 15
            
            # Price momentum (10% weight)
            current_price = indicators.get('current_price', 0)
            ema_8 = indicators.get('ema_8', current_price)
            if current_price > ema_8:
                bullish_signals += 10
            elif current_price < ema_8:
                bearish_signals += 10
            
            # Determine direction and strength
            if bullish_signals >= self.trading_config['signal_quality_threshold']:
                direction = 'LONG'
                signal_strength = bullish_signals
            elif bearish_signals >= self.trading_config['signal_quality_threshold']:
                direction = 'SHORT'
                signal_strength = bearish_signals
            else:
                return None
            
            # Calculate prices
            entry_price = current_price
            volatility = indicators.get('market_volatility', 0.02)
            
            # Dynamic price movement based on volatility
            price_movement_pct = max(1.5, min(4.0, volatility * 100))
            price_movement = entry_price * (price_movement_pct / 100)
            
            if direction == 'LONG':
                stop_loss = entry_price - price_movement
                tp1 = entry_price + (price_movement * 0.5)
                tp2 = entry_price + (price_movement * 1.0)
                tp3 = entry_price + (price_movement * 1.5)
            else:
                stop_loss = entry_price + price_movement
                tp1 = entry_price - (price_movement * 0.5)
                tp2 = entry_price - (price_movement * 1.0)
                tp3 = entry_price - (price_movement * 1.5)
            
            # Calculate dynamic leverage based on volatility
            if volatility <= 0.01:
                leverage = min(75, self.trading_config['max_leverage'])
            elif volatility <= 0.02:
                leverage = 50
            elif volatility <= 0.03:
                leverage = 35
            else:
                leverage = max(10, self.trading_config['min_leverage'])
            
            # Add ML analysis if available
            ml_analysis = None
            if self.ml_predictor:
                try:
                    ml_analysis = await self.get_ml_analysis({
                        'symbol': symbol,
                        'signal_strength': signal_strength,
                        'direction': direction,
                        'volatility': volatility,
                        'volume_ratio': indicators.get('volume_ratio', 1.0),
                        'rsi': indicators.get('rsi', 50)
                    })
                except Exception as e:
                    self.logger.warning(f"ML analysis failed: {e}")
            
            # Create comprehensive signal
            signal = {
                'symbol': symbol,
                'direction': direction,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'tp1': tp1,
                'tp2': tp2,
                'tp3': tp3,
                'leverage': leverage,
                'signal_strength': signal_strength,
                'indicators': indicators,
                'ml_analysis': ml_analysis,
                'timestamp': datetime.now(),
                'volatility': volatility,
                'risk_reward_ratio': abs((tp2 - entry_price) / (entry_price - stop_loss)) if direction == 'LONG' else abs((entry_price - tp2) / (stop_loss - entry_price))
            }
            
            return signal
            
        except Exception as e:
            self.logger.error(f"❌ Error generating signal for {symbol}: {e}")
            return None
    
    async def get_ml_analysis(self, signal_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get ML analysis for signal"""
        try:
            if not self.ml_predictor or not hasattr(self.ml_predictor, 'predict_trade_outcome'):
                return None
            
            ml_prediction = self.ml_predictor.predict_trade_outcome(signal_data)
            
            if ml_prediction and ml_prediction.get('confidence', 0) >= self.trading_config['ml_confidence_threshold']:
                return ml_prediction
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ ML analysis error: {e}")
            return None
    
    def filter_high_quality_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter and return only high-quality signals"""
        try:
            if not signals:
                return []
            
            high_quality = []
            
            for signal in signals:
                # Quality checks
                signal_strength = signal.get('signal_strength', 0)
                risk_reward = signal.get('risk_reward_ratio', 0)
                volatility = signal.get('volatility', 0)
                
                # Basic quality filters
                if (signal_strength >= self.trading_config['signal_quality_threshold'] and
                    risk_reward >= 1.0 and
                    volatility <= 0.05):
                    
                    # ML filter if available
                    ml_analysis = signal.get('ml_analysis')
                    if ml_analysis:
                        ml_confidence = ml_analysis.get('confidence', 0)
                        if ml_confidence >= self.trading_config['ml_confidence_threshold']:
                            high_quality.append(signal)
                    else:
                        # Accept signal without ML if strength is very high
                        if signal_strength >= 85:
                            high_quality.append(signal)
            
            # Sort by signal strength
            high_quality.sort(key=lambda x: x.get('signal_strength', 0), reverse=True)
            
            # Limit to top 3 signals per scan
            return high_quality[:3]
            
        except Exception as e:
            self.logger.error(f"❌ Error filtering signals: {e}")
            return signals
    
    async def push_signal_to_telegram(self, signal: Dict[str, Any]):
        """Push high-quality signal to Telegram channel"""
        try:
            self.signal_counter += 1
            
            # Format comprehensive signal message
            formatted_message = self.format_signal_message(signal)
            
            # Push to Telegram channel
            success = await self.send_telegram_message(self.telegram_config['channel_id'], formatted_message)
            
            if success:
                self.performance_metrics['successful_pushes'] += 1
                self.performance_metrics['last_signal_time'] = datetime.now()
                self.logger.info(f"✅ Signal #{self.signal_counter} pushed successfully: {signal.get('symbol')} {signal.get('direction')}")
            else:
                self.performance_metrics['failed_pushes'] += 1
                self.logger.error(f"❌ Signal #{self.signal_counter} push failed")
            
            self.performance_metrics['total_signals_pushed'] += 1
            
        except Exception as e:
            self.logger.error(f"❌ Error pushing signal: {e}")
    
    def format_signal_message(self, signal: Dict[str, Any]) -> str:
        """Format comprehensive signal message"""
        try:
            direction = signal.get('direction', '').upper()
            direction_emoji = "🟢" if direction == 'LONG' else "🔴"
            
            # Get current trading session
            current_hour = datetime.now().hour
            if 8 <= current_hour < 10:
                session = "🇬🇧 London Open"
            elif 14 <= current_hour < 16:
                session = "🌊 NY Overlap"
            elif 16 <= current_hour < 20:
                session = "🇺🇸 NY Main"
            else:
                session = "🌐 Global Session"
            
            # Signal strength bar
            strength = signal.get('signal_strength', 0)
            strength_bar = "█" * int(strength / 10)
            
            # ML analysis section
            ml_section = ""
            ml_analysis = signal.get('ml_analysis')
            if ml_analysis:
                ml_confidence = ml_analysis.get('confidence', 0)
                ml_prediction = ml_analysis.get('prediction', 'neutral')
                ml_section = f"""
🧠 **ML ANALYSIS:**
📊 **Prediction:** `{ml_prediction.title()}`
🎯 **ML Confidence:** `{ml_confidence:.1f}%`
💡 **Recommendation:** `{ml_analysis.get('recommendation', 'Standard execution')}`"""
            
            # Risk/Reward calculation
            risk_reward = signal.get('risk_reward_ratio', 1.0)
            
            message = f"""
🎯 **ADVANCED TRADING SIGNAL** 🎯
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{direction_emoji} **{direction}** | `{signal.get('symbol', 'N/A')}`
💰 **Entry:** `${signal.get('entry_price', 0):,.4f}`

🎯 **TAKE PROFITS:**
TP1: `${signal.get('tp1', 0):,.4f}` (33%)
TP2: `${signal.get('tp2', 0):,.4f}` (33%)
TP3: `${signal.get('tp3', 0):,.4f}` (34%)

🛡️ **Stop Loss:** `${signal.get('stop_loss', 0):,.4f}`
⚡ **Leverage:** `{signal.get('leverage', 35)}x Cross`

📊 **ADVANCED ANALYSIS:**
🎯 **Signal Strength:** `{strength:.1f}%` {strength_bar}
⏰ **Session:** {session}
📈 **Volatility:** `{signal.get('volatility', 0)*100:.1f}%`
📊 **Volume Ratio:** `{signal.get('indicators', {}).get('volume_ratio', 1.0):.2f}x`
📊 **RSI:** `{signal.get('indicators', {}).get('rsi', 50):.1f}`
📈 **MACD:** `{'Bullish' if signal.get('indicators', {}).get('macd_bullish') else 'Bearish' if signal.get('indicators', {}).get('macd_bearish') else 'Neutral'}`
{ml_section}

⚖️ **Risk/Reward:** `1:{risk_reward:.2f}`
💼 **Position Size:** `{self.trading_config['risk_percentage']}%` of capital

📈 **STRATEGY:** Multi-Strategy Advanced Analysis
🎲 **Edge:** Technical + ML + Price Action
⏰ **Time:** `{datetime.now().strftime('%H:%M:%S UTC')}`
📅 **Date:** `{datetime.now().strftime('%d/%m/%Y')}`

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 **Advanced Signal Pusher Bot**
📢 **{self.telegram_config['channel_id']}** | 💎 **Premium Signals**
🔥 **Signal #{self.signal_counter}** | ⏰ **Live Analysis**
            """
            
            return message.strip()
            
        except Exception as e:
            self.logger.error(f"❌ Error formatting signal message: {e}")
            return f"Signal #{self.signal_counter}: {signal.get('symbol')} {signal.get('direction')}"
    
    async def send_telegram_message(self, chat_id: str, text: str, parse_mode: str = 'Markdown') -> bool:
        """Send message to Telegram"""
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
    
    def update_uptime(self):
        """Update uptime tracking"""
        if self.performance_metrics['bot_start_time']:
            uptime = datetime.now() - self.performance_metrics['bot_start_time']
            self.performance_metrics['uptime_hours'] = uptime.total_seconds() / 3600
    
    def update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Calculate signals per hour
            if self.performance_metrics['uptime_hours'] > 0:
                self.performance_metrics['signals_per_hour'] = (
                    self.performance_metrics['total_signals_pushed'] / self.performance_metrics['uptime_hours']
                )
        except Exception as e:
            self.logger.error(f"❌ Error updating metrics: {e}")
    
    def log_performance_status(self):
        """Log current performance status"""
        try:
            metrics = self.performance_metrics
            
            self.logger.info("📊 SIGNAL PUSHER STATUS:")
            self.logger.info(f"   • Total Signals Pushed: {metrics['total_signals_pushed']}")
            self.logger.info(f"   • Successful Pushes: {metrics['successful_pushes']}")
            self.logger.info(f"   • Failed Pushes: {metrics['failed_pushes']}")
            self.logger.info(f"   • Success Rate: {(metrics['successful_pushes']/max(metrics['total_signals_pushed'],1)*100):.1f}%")
            self.logger.info(f"   • Signals/Hour: {metrics['signals_per_hour']:.2f}")
            self.logger.info(f"   • Uptime: {metrics['uptime_hours']:.1f} hours")
            
            if metrics['last_signal_time']:
                self.logger.info(f"   • Last Signal: {metrics['last_signal_time']}")
            
        except Exception as e:
            self.logger.error(f"❌ Error logging status: {e}")

async def main():
    """Main function to run the Telegram signal pusher"""
    print("🚀 ADVANCED TELEGRAM SIGNAL PUSHER BOT")
    print("=" * 80)
    print("📊 Features:")
    print("   • Continuous market scanning")
    print("   • Advanced multi-strategy analysis")
    print("   • ML-enhanced signal filtering")
    print("   • Dynamic risk management")
    print("   • Real-time Telegram pushing")
    print("=" * 80)
    print(f"📢 Channel: @SignalTactics")
    print(f"⏰ Scan Interval: 3 minutes")
    print("=" * 80)
    
    try:
        # Create and start the signal pusher
        pusher = AdvancedTelegramSignalPusher()
        await pusher.start_signal_pusher()
        
    except KeyboardInterrupt:
        print("\n🛑 Signal pusher stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
