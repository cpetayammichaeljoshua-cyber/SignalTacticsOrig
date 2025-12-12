"""
Main Orchestrator for UT Bot + STC Trading Signal Bot

Coordinates all components:
- Data fetching from Binance
- Indicator calculations
- Signal generation
- Auto-leverage trading execution
- Telegram notifications
- External data (Fear/Greed, CoinGecko, News Sentiment)
- Multi-timeframe confirmation
- Continuous monitoring loop
"""

import os
import sys
import asyncio
import logging
import signal
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from .config import Config, load_config
from .data.binance_fetcher import BinanceDataFetcher
from .engine.signal_engine import SignalEngine
from .telegram.telegram_bot import TelegramSignalBot
from .trading.leverage_calculator import LeverageCalculator
from .trading.futures_executor import FuturesExecutor
from .external_data.fear_greed_client import FearGreedClient
from .external_data.market_data_aggregator import MarketDataAggregator
from .external_data.news_sentiment_client import NewsSentimentClient
from .external_data.derivatives_client import BinanceDerivativesClient
from .external_data.liquidation_monitor import LiquidationMonitor
from .confirmation.multi_timeframe import MultiTimeframeConfirmation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ut_bot_signals.log')
    ]
)
logger = logging.getLogger(__name__)


class TradingOrchestrator:
    """
    Main orchestrator for the trading signal bot
    
    Manages the complete flow:
    1. Fetch market data from Binance
    2. Calculate UT Bot and STC indicators
    3. Check for valid entry signals
    4. Send signals to Telegram
    5. Monitor continuously
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the Trading Orchestrator
        
        Args:
            config: Configuration object (loads from environment if not provided)
        """
        self.config = config or load_config()
        self.running = False
        self._shutdown_event = asyncio.Event()
        self._last_signal_time: Optional[datetime] = None
        self._last_market_update: Optional[datetime] = None
        self._signal_count = 0
        self._error_count = 0
        
        self.data_fetcher = BinanceDataFetcher(
            api_key=self.config.binance_api_key,
            api_secret=self.config.binance_api_secret,
            symbol=self.config.trading.symbol,
            interval=self.config.trading.timeframe
        )
        
        self.signal_engine = SignalEngine(
            ut_key_value=self.config.ut_bot.key_value,
            ut_atr_period=self.config.ut_bot.atr_period,
            ut_use_heikin_ashi=self.config.ut_bot.use_heikin_ashi,
            stc_length=self.config.stc.length,
            stc_fast_length=self.config.stc.fast_length,
            stc_slow_length=self.config.stc.slow_length,
            swing_lookback=self.config.trading.swing_lookback,
            risk_reward_ratio=self.config.trading.risk_reward_ratio,
            min_risk_percent=self.config.trading.min_risk_percent,
            max_risk_percent=self.config.trading.max_risk_percent
        )
        
        self.telegram_bot = TelegramSignalBot(
            bot_token=self.config.telegram_bot_token,
            chat_id=self.config.telegram_chat_id
        )
        
        leverage_config = self.config.trading.leverage
        self.leverage_calculator = LeverageCalculator(
            min_leverage=leverage_config.min_leverage,
            max_leverage=leverage_config.max_leverage,
            base_leverage=leverage_config.base_leverage,
            risk_per_trade_percent=leverage_config.risk_per_trade_percent,
            max_position_percent=leverage_config.max_position_percent,
            volatility_low_threshold=leverage_config.volatility_low_threshold,
            volatility_high_threshold=leverage_config.volatility_high_threshold,
            signal_strength_multiplier=leverage_config.signal_strength_multiplier
        )
        
        self.futures_executor = FuturesExecutor(
            api_key=self.config.binance_api_key,
            api_secret=self.config.binance_api_secret,
            symbol=self.config.trading.symbol
        )
        
        self.auto_trading_enabled = leverage_config.enabled
        self.use_isolated_margin = leverage_config.use_isolated_margin
        self._current_position = None
        self._trade_count = 0
        
        self.fear_greed_client: Optional[FearGreedClient] = None
        self.market_data_client: Optional[MarketDataAggregator] = None
        self.news_client: Optional[NewsSentimentClient] = None
        self.mtf_confirmation: Optional[MultiTimeframeConfirmation] = None
        
        if self.config.external_data.fear_greed_enabled:
            self.fear_greed_client = FearGreedClient()
        
        if self.config.external_data.coingecko_enabled:
            self.market_data_client = MarketDataAggregator(
                api_key=self.config.coingecko_api_key
            )
        
        if self.config.external_data.cryptopanic_enabled:
            self.news_client = NewsSentimentClient(
                api_key=self.config.cryptopanic_api_key
            )
        
        self.derivatives_client = BinanceDerivativesClient()
        self.liquidation_monitor = LiquidationMonitor()
        
        self.signal_engine.set_derivatives_client(self.derivatives_client)
        
        if self.config.multi_timeframe.enabled:
            self.mtf_confirmation = MultiTimeframeConfirmation(
                api_key=self.config.binance_api_key,
                api_secret=self.config.binance_api_secret,
                primary_timeframe=self.config.trading.timeframe,
                ut_key_value=self.config.ut_bot.key_value,
                ut_atr_period=self.config.ut_bot.atr_period,
                ut_use_heikin_ashi=self.config.ut_bot.use_heikin_ashi,
                stc_length=self.config.stc.length,
                stc_fast_length=self.config.stc.fast_length,
                stc_slow_length=self.config.stc.slow_length
            )
        
        self._last_external_data_refresh: Optional[datetime] = None
        self._market_intelligence: Dict[str, Any] = {}
        
        logger.info("Trading Orchestrator initialized")
        logger.info(f"Symbol: {self.config.trading.symbol}")
        logger.info(f"Timeframe: {self.config.trading.timeframe}")
        logger.info(f"UT Bot settings: Key={self.config.ut_bot.key_value}, ATR={self.config.ut_bot.atr_period}")
        logger.info(f"STC settings: Length={self.config.stc.length}, Fast={self.config.stc.fast_length}, Slow={self.config.stc.slow_length}")
        logger.info(f"Auto-trading: {'ENABLED' if self.auto_trading_enabled else 'DISABLED'}")
        if self.auto_trading_enabled:
            logger.info(f"Leverage range: {leverage_config.min_leverage}x-{leverage_config.max_leverage}x")
        logger.info(f"Fear/Greed Index: {'ENABLED' if self.fear_greed_client else 'DISABLED'}")
        logger.info(f"CoinGecko Market Data: {'ENABLED' if self.market_data_client else 'DISABLED'}")
        logger.info(f"News Sentiment: {'ENABLED' if self.news_client else 'DISABLED'}")
        logger.info(f"Multi-Timeframe Confirmation: {'ENABLED' if self.mtf_confirmation else 'DISABLED'}")
        logger.info("Derivatives Client: ENABLED")
        logger.info("Liquidation Monitor: ENABLED")
    
    def _can_send_signal(self) -> bool:
        """Check if we can send a new signal (respecting cooldown)"""
        if self._last_signal_time is None:
            return True
        
        cooldown = timedelta(minutes=self.config.bot.signal_cooldown_minutes)
        return datetime.now() - self._last_signal_time > cooldown
    
    def _can_send_market_update(self) -> bool:
        """Check if we can send a market update"""
        if not self.config.bot.send_market_updates:
            return False
        
        if self._last_market_update is None:
            return True
        
        interval = timedelta(minutes=self.config.bot.market_update_interval_minutes)
        return datetime.now() - self._last_market_update > interval
    
    def _should_refresh_external_data(self) -> bool:
        """Check if external data should be refreshed"""
        if self._last_external_data_refresh is None:
            return True
        
        interval = timedelta(minutes=self.config.external_data.refresh_interval_minutes)
        return datetime.now() - self._last_external_data_refresh > interval
    
    async def refresh_external_data(self) -> Dict[str, Any]:
        """
        Refresh external market intelligence data
        
        Returns:
            Dictionary with market intelligence data
        """
        if not self._should_refresh_external_data():
            return self._market_intelligence
        
        try:
            fear_greed_data = None
            if self.fear_greed_client:
                try:
                    fear_greed_data = await self.fear_greed_client.get_current()
                    if fear_greed_data:
                        self._market_intelligence['fear_greed'] = {
                            'value': fear_greed_data.value,
                            'classification': fear_greed_data.value_classification,
                            'timestamp': fear_greed_data.timestamp.isoformat() if fear_greed_data.timestamp else None
                        }
                        logger.info(f"Fear/Greed Index: {fear_greed_data.value} ({fear_greed_data.value_classification})")
                except Exception as e:
                    logger.warning(f"Error fetching Fear/Greed data: {e}")
            
            if self.market_data_client:
                try:
                    global_data = await self.market_data_client.get_global_market_data()
                    if global_data:
                        self._market_intelligence['global_market'] = {
                            'total_market_cap_usd': global_data.total_market_cap_usd,
                            'btc_dominance': global_data.btc_dominance,
                            'eth_dominance': global_data.eth_dominance,
                            'market_cap_change_24h': global_data.market_cap_change_percentage_24h
                        }
                        logger.info(f"Global Market Cap: ${global_data.total_market_cap_usd:,.0f}")
                except Exception as e:
                    logger.warning(f"Error fetching CoinGecko data: {e}")
            
            if self.news_client:
                try:
                    news_summary = await self.news_client.get_sentiment_summary()
                    if news_summary:
                        self._market_intelligence['news_sentiment'] = {
                            'average_sentiment': news_summary.average_sentiment,
                            'sentiment_label': news_summary.sentiment_label,
                            'total_articles': news_summary.total_news
                        }
                        logger.info(f"News Sentiment: {news_summary.sentiment_label} ({news_summary.average_sentiment:.2f})")
                except Exception as e:
                    logger.warning(f"Error fetching news sentiment: {e}")
            
            self._last_external_data_refresh = datetime.now()
            logger.debug("External data refreshed successfully")
            
        except Exception as e:
            logger.error(f"Error refreshing external data: {e}")
        
        return self._market_intelligence
    
    def get_mtf_confirmation(self, signal_type: str) -> tuple:
        """
        Get multi-timeframe confirmation for a signal
        
        Args:
            signal_type: LONG or SHORT
            
        Returns:
            Tuple of (should_trade, alignment_score, recommendation)
        """
        if not self.mtf_confirmation:
            return True, 1.0, 'NO_MTF'
        
        try:
            mtf_result = self.mtf_confirmation.analyze(self.config.trading.symbol)
            
            alignment_score = mtf_result.alignment_score
            recommendation = mtf_result.recommendation
            
            should_trade = (
                alignment_score >= self.config.multi_timeframe.min_alignment_score and
                recommendation in ['STRONG_CONFIRM', 'CONFIRM']
            )
            
            self._market_intelligence['mtf_confirmation'] = {
                'alignment_score': alignment_score,
                'recommendation': recommendation,
                'higher_tf_bias': mtf_result.higher_timeframe_bias,
                'confirming_timeframes': mtf_result.confirming_timeframes,
                'conflicting_timeframes': mtf_result.conflicting_timeframes
            }
            
            logger.info(f"MTF Confirmation: score={alignment_score:.2f}, recommendation={recommendation}")
            
            return should_trade, alignment_score, recommendation
            
        except Exception as e:
            logger.warning(f"Error getting MTF confirmation: {e}")
            return True, 1.0, 'ERROR'
    
    async def fetch_and_process(self) -> tuple:
        """
        Fetch data and process for signals
        
        Returns:
            Tuple of (signal, dataframe) if valid signal found, (None, None) otherwise
        """
        try:
            await self.refresh_external_data()
            
            df = self.data_fetcher.fetch_historical_data(
                limit=max(200, self.config.trading.min_candles_required + 50)
            )
            
            if df is None or len(df) < self.config.trading.min_candles_required:
                logger.warning(f"Insufficient data: {len(df) if df is not None else 0} candles")
                return None, None
            
            logger.debug(f"Fetched {len(df)} candles, latest: {df.index[-1]}")
            
            derivatives_data = None
            try:
                symbol = self.config.trading.symbol.replace('/', '').upper()
                derivatives_data = await self.derivatives_client.get_derivatives_intelligence(symbol)
                if derivatives_data:
                    logger.debug(f"Derivatives data fetched: score={derivatives_data.derivatives_score:.2f}")
            except Exception as e:
                logger.warning(f"Error fetching derivatives data: {e}")
            
            signal = self.signal_engine.generate_signal(df, derivatives_data=derivatives_data)
            
            if signal and self._can_send_signal():
                signal_type = signal.get('type', 'NEUTRAL')
                
                if self.config.multi_timeframe.enabled:
                    should_trade, alignment_score, recommendation = self.get_mtf_confirmation(signal_type)
                    
                    signal['mtf_alignment_score'] = alignment_score
                    signal['mtf_recommendation'] = recommendation
                    
                    if not should_trade:
                        logger.info(f"Signal {signal_type} filtered by MTF: alignment={alignment_score:.2f}, recommendation={recommendation}")
                        return None, df
                
                signal['market_intelligence'] = self._market_intelligence.copy()
                
                try:
                    liquidation_metrics = self.liquidation_monitor.get_metrics()
                    if liquidation_metrics:
                        signal['market_intelligence']['liquidation'] = {
                            'long_liquidations_usd': liquidation_metrics.long_liquidations_usd,
                            'short_liquidations_usd': liquidation_metrics.short_liquidations_usd,
                            'total_liquidations_usd': liquidation_metrics.total_liquidations_usd,
                            'liquidation_imbalance': liquidation_metrics.liquidation_imbalance,
                            'large_liquidation_count': liquidation_metrics.large_liquidation_count,
                            'liquidation_intensity': liquidation_metrics.liquidation_intensity,
                            'signal_bias': liquidation_metrics.signal_bias
                        }
                        logger.debug(f"Liquidation metrics added: imbalance={liquidation_metrics.liquidation_imbalance:.2f}")
                except Exception as e:
                    logger.warning(f"Error getting liquidation metrics: {e}")
                
                logger.info(f"Valid {signal_type} signal generated with market intelligence!")
                return signal, df
            elif signal:
                logger.info(f"Signal generated but in cooldown period")
            
            if self._can_send_market_update():
                state = self.signal_engine.get_market_state(df)
                await self.telegram_bot.send_market_update(state)
                self._last_market_update = datetime.now()
            
            return None, df
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Error in fetch_and_process: {e}")
            
            if self._error_count <= 3:
                await self.telegram_bot.send_error(str(e), "Data fetch/process error")
            
            return None, None
    
    async def execute_trade(self, signal: Dict, df) -> Optional[Dict]:
        """
        Execute auto-leverage trade based on signal
        
        Args:
            signal: Trading signal
            df: DataFrame with price data for ATR calculation
            
        Returns:
            Trade execution result or None
        """
        try:
            if not await self.futures_executor.initialize():
                logger.error("Failed to initialize futures executor")
                return None
            
            balance = await self.futures_executor.get_account_balance()
            available_balance = balance.get('free', 0)
            
            if available_balance < 10:
                logger.warning(f"Insufficient balance: ${available_balance:.2f}")
                return None
            
            current_price = signal.get('entry_price', 0)
            atr = df['high'].iloc[-14:].max() - df['low'].iloc[-14:].min()
            atr = atr / 14
            
            leverage_result = self.leverage_calculator.calculate_optimal_leverage(
                signal=signal,
                account_balance=available_balance,
                current_price=current_price,
                atr=atr
            )
            
            is_valid, validation_msg = self.leverage_calculator.validate_trade(
                leverage_result=leverage_result,
                account_balance=available_balance,
                min_order_size=0.001
            )
            
            if not is_valid:
                logger.warning(f"Trade validation failed: {validation_msg}")
                return None
            
            logger.info(f"Executing trade: {leverage_result.leverage}x leverage, {leverage_result.position_size:.4f} ETH")
            
            trade_result = await self.futures_executor.execute_trade(
                signal=signal,
                leverage=leverage_result.leverage,
                quantity=leverage_result.position_size,
                use_isolated=self.use_isolated_margin
            )
            
            if trade_result.get('success'):
                self._trade_count += 1
                self._current_position = {
                    'direction': signal.get('direction'),
                    'entry_price': trade_result['entry'].price,
                    'quantity': leverage_result.position_size,
                    'leverage': leverage_result.leverage,
                    'stop_loss': signal.get('stop_loss'),
                    'take_profit': signal.get('take_profit'),
                    'timestamp': datetime.now()
                }
                logger.info(f"Trade executed successfully! Total trades: {self._trade_count}")
            
            return {
                'trade_result': trade_result,
                'leverage_result': leverage_result,
                'balance': available_balance
            }
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return None
    
    async def process_signal(self, signal: Dict, df=None) -> bool:
        """
        Process and send a trading signal, optionally execute trade
        
        Args:
            signal: Signal dictionary
            df: DataFrame for trade execution (optional)
            
        Returns:
            True if signal was sent successfully
        """
        try:
            trade_info = None
            if self.auto_trading_enabled and df is not None:
                try:
                    existing_position = await self.futures_executor.get_position()
                    if existing_position:
                        logger.info(f"Existing {existing_position.side} position, skipping new trade")
                    else:
                        trade_info = await self.execute_trade(signal, df)
                except Exception as trade_error:
                    logger.warning(f"Could not execute trade: {trade_error}")
                    trade_info = None
            
            success = await self.telegram_bot.send_signal(signal, trade_info)
            
            if success:
                self._last_signal_time = datetime.now()
                self._signal_count += 1
                self._error_count = 0
                logger.info(f"Signal sent successfully. Total signals: {self._signal_count}")
            else:
                logger.error("Failed to send signal to Telegram")
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            return False
    
    async def run_cycle(self):
        """Run a single monitoring cycle"""
        signal, df = await self.fetch_and_process()
        
        if signal:
            await self.process_signal(signal, df)
    
    async def run(self):
        """
        Main monitoring loop
        
        Continuously monitors the market and sends signals
        """
        if not self.config.validate():
            logger.error("Invalid configuration. Cannot start bot.")
            return
        
        self.running = True
        logger.info("Starting UT Bot + STC Signal Bot...")
        
        try:
            await self.liquidation_monitor.start()
            logger.info("Liquidation monitor started")
        except Exception as e:
            logger.warning(f"Failed to start liquidation monitor: {e}")
        
        if self.config.bot.enable_startup_notification:
            await self.telegram_bot.send_startup_notification()
        
        try:
            while self.running and not self._shutdown_event.is_set():
                try:
                    await self.run_cycle()
                except Exception as e:
                    logger.error(f"Error in monitoring cycle: {e}")
                    self._error_count += 1
                
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.config.bot.check_interval_seconds
                    )
                except asyncio.TimeoutError:
                    pass
                
        except asyncio.CancelledError:
            logger.info("Bot cancelled")
        except Exception as e:
            logger.error(f"Fatal error in run loop: {e}")
        finally:
            await self.shutdown()
    
    async def shutdown(self, reason: str = "Manual shutdown") -> None:
        """
        Gracefully shutdown the bot
        
        Args:
            reason: Shutdown reason for notification
        """
        logger.info(f"Shutting down: {reason}")
        self.running = False
        self._shutdown_event.set()
        
        try:
            if self.config.bot.enable_shutdown_notification:
                await self.telegram_bot.send_shutdown_notification(reason)
        except Exception as e:
            logger.warning(f"Error sending shutdown notification: {e}")
        
        try:
            await self.telegram_bot.close()
        except Exception as e:
            logger.warning(f"Error closing telegram bot: {e}")
        
        try:
            await self.futures_executor.close()
        except Exception as e:
            logger.warning(f"Error closing futures executor: {e}")
        
        try:
            await self.data_fetcher.close()
        except Exception as e:
            logger.warning(f"Error closing data fetcher: {e}")
        
        if self.fear_greed_client:
            try:
                await self.fear_greed_client.close()
                logger.info("Fear/Greed client closed")
            except Exception as e:
                logger.warning(f"Error closing fear/greed client: {e}")
        
        if self.market_data_client:
            try:
                await self.market_data_client.close()
                logger.info("CoinGecko client closed")
            except Exception as e:
                logger.warning(f"Error closing market data client: {e}")
        
        if self.news_client:
            try:
                await self.news_client.close()
                logger.info("News sentiment client closed")
            except Exception as e:
                logger.warning(f"Error closing news client: {e}")
        
        try:
            await self.derivatives_client.close()
            logger.info("Derivatives client closed")
        except Exception as e:
            logger.warning(f"Error closing derivatives client: {e}")
        
        try:
            await self.liquidation_monitor.stop()
            logger.info("Liquidation monitor stopped")
        except Exception as e:
            logger.warning(f"Error stopping liquidation monitor: {e}")
        
        await asyncio.sleep(0.5)
        logger.info("Bot shutdown complete")
    
    def stop(self):
        """Signal the bot to stop"""
        self._shutdown_event.set()
        self.running = False
    
    def get_status(self) -> Dict:
        """Get current bot status"""
        return {
            'running': self.running,
            'signal_count': self._signal_count,
            'error_count': self._error_count,
            'last_signal_time': self._last_signal_time,
            'config': self.config.to_dict()
        }


def setup_signal_handlers(orchestrator: TradingOrchestrator):
    """Setup system signal handlers for graceful shutdown"""
    def handle_shutdown(signum, frame):
        logger.info(f"Received signal {signum}")
        orchestrator.stop()
    
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)


async def main():
    """Main entry point"""
    config = load_config()
    orchestrator = TradingOrchestrator(config)
    setup_signal_handlers(orchestrator)
    
    try:
        await orchestrator.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        await orchestrator.shutdown("Keyboard interrupt")


if __name__ == "__main__":
    asyncio.run(main())
