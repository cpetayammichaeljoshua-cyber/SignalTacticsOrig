"""
Main Orchestrator for UT Bot + STC Trading Signal Bot

Coordinates all components:
- Data fetching from Binance
- Indicator calculations
- Signal generation
- Telegram notifications
- Continuous monitoring loop
"""

import os
import sys
import asyncio
import logging
import signal
from datetime import datetime, timedelta
from typing import Optional, Dict

from .config import Config, load_config
from .data.binance_fetcher import BinanceDataFetcher
from .engine.signal_engine import SignalEngine
from .telegram.telegram_bot import TelegramSignalBot

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
        
        logger.info("Trading Orchestrator initialized")
        logger.info(f"Symbol: {self.config.trading.symbol}")
        logger.info(f"Timeframe: {self.config.trading.timeframe}")
        logger.info(f"UT Bot settings: Key={self.config.ut_bot.key_value}, ATR={self.config.ut_bot.atr_period}")
        logger.info(f"STC settings: Length={self.config.stc.length}, Fast={self.config.stc.fast_length}, Slow={self.config.stc.slow_length}")
    
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
    
    async def fetch_and_process(self) -> Optional[Dict]:
        """
        Fetch data and process for signals
        
        Returns:
            Signal dictionary if valid signal found, None otherwise
        """
        try:
            df = self.data_fetcher.fetch_historical_data(
                limit=max(200, self.config.trading.min_candles_required + 50)
            )
            
            if df is None or len(df) < self.config.trading.min_candles_required:
                logger.warning(f"Insufficient data: {len(df) if df is not None else 0} candles")
                return None
            
            logger.debug(f"Fetched {len(df)} candles, latest: {df.index[-1]}")
            
            signal = self.signal_engine.generate_signal(df)
            
            if signal and self._can_send_signal():
                logger.info(f"Valid {signal['type']} signal generated!")
                return signal
            elif signal:
                logger.info(f"Signal generated but in cooldown period")
            
            if self._can_send_market_update():
                state = self.signal_engine.get_market_state(df)
                await self.telegram_bot.send_market_update(state)
                self._last_market_update = datetime.now()
            
            return None
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Error in fetch_and_process: {e}")
            
            if self._error_count <= 3:
                await self.telegram_bot.send_error(str(e), "Data fetch/process error")
            
            return None
    
    async def process_signal(self, signal: Dict) -> bool:
        """
        Process and send a trading signal
        
        Args:
            signal: Signal dictionary
            
        Returns:
            True if signal was sent successfully
        """
        try:
            success = await self.telegram_bot.send_signal(signal)
            
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
        signal = await self.fetch_and_process()
        
        if signal:
            await self.process_signal(signal)
    
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
    
    async def shutdown(self, reason: str = "Manual shutdown"):
        """
        Gracefully shutdown the bot
        
        Args:
            reason: Shutdown reason for notification
        """
        logger.info(f"Shutting down: {reason}")
        self.running = False
        self._shutdown_event.set()
        
        if self.config.bot.enable_shutdown_notification:
            await self.telegram_bot.send_shutdown_notification(reason)
        
        await self.telegram_bot.close()
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
