"""
Main trading bot implementation
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import os
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException

from bot.ml_signal_generator import SignalGenerator
from bot.leverage_manager import AdaptiveLeverageManager, LeverageConfig
from bot.risk_manager import RiskManager, RiskLevel
from bot.data_processor import DataProcessor

logger = logging.getLogger(__name__)

class TradingBot:
    """Main trading bot orchestrator"""
    
    def __init__(self, config):
        self.config = config
        self.is_running = False
        self.binance_client: Optional[BinanceClient] = None
        
        # Initialize components
        self.signal_generator = SignalGenerator(config)
        self.leverage_manager = AdaptiveLeverageManager(LeverageConfig(
            base_leverage=config.min_leverage * 2,
            max_leverage=config.max_leverage,
            min_leverage=config.min_leverage
        ))
        self.risk_manager = RiskManager(config)
        
        # State tracking
        self.market_data = {}
        self.current_positions = {}
        self.account_info = {}
        self.performance_metrics = {}
        self.daily_trade_count = 0
        self.last_trade_date = None
        
        # Performance tracking
        self.trade_history = []
        self.pnl_history = []
        self.balance_history = []
        
    async def start(self):
        """Start the trading bot"""
        try:
            logger.info("🚀 Starting Ultimate Crypto Trading Bot...")
            
            # Initialize Binance client
            await self._initialize_binance_client()
            
            # Initialize signal generator
            await self.signal_generator.initialize()
            
            # Start main trading loop
            self.is_running = True
            
            # Run initial setup
            await self._initial_setup()
            
            # Start trading loop
            await self._trading_loop()
            
        except Exception as e:
            logger.error(f"❌ Error starting trading bot: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the trading bot"""
        try:
            logger.info("🛑 Stopping trading bot...")
            self.is_running = False
            
            # Close any open positions if needed (implement based on strategy)
            await self._cleanup_positions()
            
            logger.info("✅ Trading bot stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Error stopping trading bot: {e}")
    
    async def _initialize_binance_client(self):
        """Initialize Binance API client"""
        try:
            api_key = os.getenv("BINANCE_API_KEY")
            api_secret = os.getenv("BINANCE_API_SECRET")
            
            if not api_key or not api_secret:
                raise ValueError("Binance API credentials not found in environment variables")
            
            # Initialize client (testnet for safety - change to False for production)
            self.binance_client = BinanceClient(api_key, api_secret, testnet=True)
            
            # Test connection
            await self._test_binance_connection()
            
            logger.info("✅ Binance client initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Binance client: {e}")
            raise
    
    async def _test_binance_connection(self):
        """Test Binance API connection"""
        try:
            # Get account info
            account_info = self.binance_client.get_account()
            logger.info(f"✅ Connected to Binance. Account type: {account_info.get('accountType', 'Unknown')}")
            
            # Get server time
            server_time = self.binance_client.get_server_time()
            logger.debug(f"Binance server time: {server_time}")
            
        except BinanceAPIException as e:
            logger.error(f"❌ Binance API error: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Error testing Binance connection: {e}")
            raise
    
    async def _initial_setup(self):
        """Perform initial setup"""
        try:
            logger.info("🔧 Performing initial setup...")
            
            # Get initial account info
            await self._update_account_info()
            
            # Get initial market data
            await self._update_market_data()
            
            # Initialize performance tracking
            self._initialize_performance_tracking()
            
            logger.info("✅ Initial setup completed")
            
        except Exception as e:
            logger.error(f"❌ Error in initial setup: {e}")
            raise
    
    async def _trading_loop(self):
        """Main trading loop"""
        logger.info("🔄 Starting main trading loop...")
        
        try:
            while self.is_running:
                loop_start = datetime.now()
                
                try:
                    # Update market data
                    await self._update_market_data()
                    
                    # Update account info
                    await self._update_account_info()
                    
                    # Generate signals
                    signals = await self.signal_generator.generate_signals(self.market_data)
                    
                    # Process signals and execute trades
                    await self._process_signals(signals)
                    
                    # Update performance metrics
                    await self._update_performance_metrics()
                    
                    # Risk monitoring
                    await self._monitor_risk()
                    
                    # Log status
                    self._log_status(signals)
                    
                except Exception as e:
                    logger.error(f"❌ Error in trading loop iteration: {e}")
                    await asyncio.sleep(30)  # Wait before retrying
                
                # Wait for next iteration
                loop_duration = (datetime.now() - loop_start).total_seconds()
                sleep_time = max(0, self.config.update_interval - loop_duration)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
        
        except Exception as e:
            logger.error(f"❌ Fatal error in trading loop: {e}")
            await self.stop()
    
    async def _update_market_data(self):
        """Update market data for all symbols"""
        try:
            for symbol in self.config.symbols:
                try:
                    # Get klines (OHLCV data)
                    klines = self.binance_client.get_historical_klines(
                        symbol, 
                        BinanceClient.KLINE_INTERVAL_1MINUTE,
                        f"{self.config.lookback_period + 50} minutes ago UTC"
                    )
                    
                    if klines:
                        # Convert to DataFrame
                        df = pd.DataFrame(klines, columns=[
                            'timestamp', 'open', 'high', 'low', 'close', 'volume',
                            'close_time', 'quote_asset_volume', 'number_of_trades',
                            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                            'ignore'
                        ])
                        
                        # Convert timestamp and numerical columns
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        numerical_columns = ['open', 'high', 'low', 'close', 'volume']
                        for col in numerical_columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        # Store market data
                        self.market_data[symbol] = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
                        
                        logger.debug(f"📊 Updated market data for {symbol}: {len(df)} candles")
                    
                except Exception as e:
                    logger.error(f"❌ Error updating market data for {symbol}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"❌ Error in market data update: {e}")
    
    async def _update_account_info(self):
        """Update account information"""
        try:
            account_info = self.binance_client.get_account()
            
            self.account_info = {
                'total_balance': float(account_info.get('totalWalletBalance', 0)),
                'available_balance': float(account_info.get('availableBalance', 0)),
                'total_unrealized_pnl': float(account_info.get('totalUnrealizedProfit', 0)),
                'total_margin_balance': float(account_info.get('totalMarginBalance', 0)),
                'total_position_initial_margin': float(account_info.get('totalPositionInitialMargin', 0)),
                'max_withdraw_amount': float(account_info.get('maxWithdrawAmount', 0))
            }
            
            # Get position information
            positions = self.binance_client.futures_position_information()
            
            self.current_positions = {}
            for position in positions:
                if float(position['positionAmt']) != 0:
                    symbol = position['symbol']
                    self.current_positions[symbol] = {
                        'symbol': symbol,
                        'position_amount': float(position['positionAmt']),
                        'entry_price': float(position['entryPrice']),
                        'mark_price': float(position['markPrice']),
                        'unrealized_pnl': float(position['unRealizedProfit']),
                        'percentage': float(position['percentage']),
                        'leverage': int(float(position['leverage'])),
                        'exposure': abs(float(position['positionAmt'])) * float(position['markPrice'])
                    }
            
            logger.debug(f"💰 Account balance: ${self.account_info['total_balance']:.2f}, "
                        f"Open positions: {len(self.current_positions)}")
            
        except Exception as e:
            logger.error(f"❌ Error updating account info: {e}")
    
    async def _process_signals(self, signals: Dict[str, Dict[str, Any]]):
        """Process trading signals and execute trades"""
        try:
            for symbol, signal in signals.items():
                try:
                    # Skip if no signal
                    if signal['signal'] == 0:
                        continue
                    
                    # Check daily trade limit
                    if not self._check_daily_trade_limit():
                        logger.warning(f"⚠️ Daily trade limit reached")
                        break
                    
                    # Calculate leverage
                    leverage_info = self.leverage_manager.calculate_leverage(
                        symbol=symbol,
                        signal_confidence=signal['confidence'],
                        current_volatility=signal.get('volatility', 0.02),
                        account_balance=self.account_info.get('available_balance', 1000)
                    )
                    
                    logger.info(f"🎯 Adaptive leverage calculated: {leverage_info['leverage']}x "
                               f"(Performance factor: {leverage_info['performance_multiplier']:.2f})")
                    
                    # Evaluate trade risk
                    risk_assessment = self.risk_manager.evaluate_trade_risk(
                        symbol=symbol,
                        signal=signal,
                        leverage=leverage_info['leverage'],
                        position_size=leverage_info['position_size'],
                        current_positions=self.current_positions
                    )
                    
                    # Check if trade is approved
                    if not risk_assessment['approved']:
                        logger.warning(f"⚠️ Trade rejected for {symbol}: {risk_assessment['warnings']}")
                        continue
                    
                    # Execute trade
                    await self._execute_trade(symbol, signal, leverage_info, risk_assessment)
                    
                except Exception as e:
                    logger.error(f"❌ Error processing signal for {symbol}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"❌ Error processing signals: {e}")
    
    async def _execute_trade(self, symbol: str, signal: Dict[str, Any], 
                           leverage_info: Dict[str, Any], risk_assessment: Dict[str, Any]):
        """Execute a trade"""
        try:
            # This is a simulation of trade execution
            # In a real implementation, you would place actual orders here
            
            logger.info(f"🎯 SIMULATED TRADE EXECUTION for {symbol}:")
            logger.info(f"   Signal: {'BUY' if signal['signal'] == 1 else 'SELL'}")
            logger.info(f"   Confidence: {signal['confidence']:.3f}")
            logger.info(f"   Leverage: {leverage_info['leverage']}x")
            logger.info(f"   Position Size: ${leverage_info['position_size']:.2f}")
            logger.info(f"   Risk Level: {risk_assessment['risk_level'].value}")
            logger.info(f"   Reason: {signal.get('reason', 'N/A')}")
            
            # Update trade count
            self._update_trade_count()
            
            # Record trade
            trade_record = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'signal': signal['signal'],
                'confidence': signal['confidence'],
                'leverage': leverage_info['leverage'],
                'position_size': leverage_info['position_size'],
                'risk_level': risk_assessment['risk_level'].value,
                'reason': signal.get('reason', 'N/A')
            }
            
            self.trade_history.append(trade_record)
            
            # In a real implementation, you would:
            # 1. Set leverage for the symbol
            # 2. Calculate quantity based on position size
            # 3. Place market or limit order
            # 4. Set stop-loss and take-profit orders
            # 5. Handle order execution errors
            
            logger.info(f"✅ Trade recorded for {symbol}")
            
        except Exception as e:
            logger.error(f"❌ Error executing trade for {symbol}: {e}")
    
    async def _monitor_risk(self):
        """Monitor portfolio risk"""
        try:
            portfolio_risk = self.risk_manager.monitor_portfolio_risk(
                self.current_positions,
                self.account_info.get('total_balance', 1000)
            )
            
            # Log alerts if any
            if portfolio_risk.get('alerts'):
                for alert in portfolio_risk['alerts']:
                    logger.warning(f"🚨 RISK ALERT: {alert}")
            
            # Log risk level if high
            risk_level = portfolio_risk.get('risk_level')
            if risk_level and risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                logger.warning(f"⚠️ Portfolio risk level: {risk_level.value}")
            
        except Exception as e:
            logger.error(f"❌ Error monitoring risk: {e}")
    
    async def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            current_balance = self.account_info.get('total_balance', 0)
            current_pnl = self.account_info.get('total_unrealized_pnl', 0)
            
            # Store balance history
            self.balance_history.append({
                'timestamp': datetime.now(),
                'balance': current_balance,
                'unrealized_pnl': current_pnl
            })
            
            # Keep last 1000 entries
            self.balance_history = self.balance_history[-1000:]
            
            # Calculate performance metrics
            if len(self.balance_history) >= 2:
                initial_balance = self.balance_history[0]['balance']
                total_return = (current_balance - initial_balance) / initial_balance if initial_balance > 0 else 0
                
                self.performance_metrics = {
                    'total_return': total_return,
                    'current_balance': current_balance,
                    'unrealized_pnl': current_pnl,
                    'total_trades': len(self.trade_history),
                    'active_positions': len(self.current_positions),
                    'daily_trades': self.daily_trade_count
                }
            
        except Exception as e:
            logger.error(f"❌ Error updating performance metrics: {e}")
    
    def _check_daily_trade_limit(self) -> bool:
        """Check if daily trade limit is reached"""
        current_date = datetime.now().date()
        
        if self.last_trade_date != current_date:
            self.daily_trade_count = 0
            self.last_trade_date = current_date
        
        return self.daily_trade_count < self.config.max_daily_trades
    
    def _update_trade_count(self):
        """Update daily trade count"""
        self.daily_trade_count += 1
    
    def _log_status(self, signals: Dict[str, Dict[str, Any]]):
        """Log bot status"""
        try:
            active_signals = {k: v for k, v in signals.items() if v['signal'] != 0}
            
            if active_signals:
                logger.info(f"🎯 Active signals: {len(active_signals)}")
                for symbol, signal in active_signals.items():
                    direction = "🟢 BUY" if signal['signal'] == 1 else "🔴 SELL"
                    logger.info(f"   {symbol}: {direction}, Confidence: {signal['confidence']:.1%}, "
                               f"Strength: {signal.get('tech_strength', 0):.1%}")
            
        except Exception as e:
            logger.error(f"❌ Error logging status: {e}")
    
    def _initialize_performance_tracking(self):
        """Initialize performance tracking"""
        try:
            initial_balance = self.account_info.get('total_balance', 1000)
            
            self.balance_history = [{
                'timestamp': datetime.now(),
                'balance': initial_balance,
                'unrealized_pnl': 0
            }]
            
            logger.info(f"📊 Performance tracking initialized with balance: ${initial_balance:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Error initializing performance tracking: {e}")
    
    async def _cleanup_positions(self):
        """Cleanup positions on shutdown (if needed)"""
        try:
            # In a real implementation, you might want to close all positions
            # For now, just log the current positions
            if self.current_positions:
                logger.info(f"ℹ️ Bot stopping with {len(self.current_positions)} open positions")
                for symbol, position in self.current_positions.items():
                    logger.info(f"   {symbol}: {position['position_amount']:.4f} "
                               f"(PnL: ${position['unrealized_pnl']:.2f})")
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up positions: {e}")
    
    # Public methods for web dashboard
    def get_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        return {
            'is_running': self.is_running,
            'account_info': self.account_info,
            'current_positions': self.current_positions,
            'performance_metrics': self.performance_metrics,
            'daily_trade_count': self.daily_trade_count,
            'total_symbols': len(self.config.symbols)
        }
    
    def get_trade_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trade history"""
        return self.trade_history[-limit:] if self.trade_history else []
    
    def get_performance_history(self) -> List[Dict[str, Any]]:
        """Get performance history"""
        return self.balance_history
    
    def get_signal_statistics(self) -> Dict[str, Any]:
        """Get signal statistics for all symbols"""
        stats = {}
        for symbol in self.config.symbols:
            stats[symbol] = self.signal_generator.get_signal_statistics(symbol)
        return stats
