"""
Backtest Runner for UT Bot + STC Trading Strategy

Simulates historical trades using the same strategy logic as the live bot.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

from ..engine.signal_engine import SignalEngine
from ..data.binance_fetcher import BinanceDataFetcher

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtest parameters"""
    symbol: str = "ETHUSDT"
    timeframe: str = "5m"
    lookback_days: int = 30
    max_candles: int = 5000
    ut_key_value: float = 2.0
    ut_atr_period: int = 6
    ut_use_heikin_ashi: bool = True
    stc_length: int = 80
    stc_fast_length: int = 27
    stc_slow_length: int = 50
    swing_lookback: int = 5
    risk_reward_ratio: float = 1.5
    min_risk_percent: float = 0.3
    max_risk_percent: float = 3.0


@dataclass
class TradeRecord:
    """Record of a single simulated trade"""
    trade_id: int
    direction: str
    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_percent: float = 0.0
    pnl_absolute: float = 0.0
    risk_reward_achieved: float = 0.0
    holding_bars: int = 0
    is_winner: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'trade_id': self.trade_id,
            'direction': self.direction,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_price': self.exit_price,
            'exit_reason': self.exit_reason,
            'pnl_percent': self.pnl_percent,
            'pnl_absolute': self.pnl_absolute,
            'risk_reward_achieved': self.risk_reward_achieved,
            'holding_bars': self.holding_bars,
            'is_winner': self.is_winner
        }


class BacktestRunner:
    """
    Runs backtests on historical data using UT Bot + STC strategy
    
    Features:
    - Bar-by-bar simulation with realistic entry/exit logic
    - Stop loss and take profit handling
    - Comprehensive trade recording
    - Performance metrics calculation
    """
    
    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        data_fetcher: Optional[BinanceDataFetcher] = None
    ):
        """
        Initialize the backtest runner
        
        Args:
            config: Backtest configuration
            data_fetcher: Optional data fetcher instance (creates new if not provided)
        """
        self.config = config or BacktestConfig()
        self.data_fetcher = data_fetcher
        
        self.signal_engine = SignalEngine(
            ut_key_value=self.config.ut_key_value,
            ut_atr_period=self.config.ut_atr_period,
            ut_use_heikin_ashi=self.config.ut_use_heikin_ashi,
            stc_length=self.config.stc_length,
            stc_fast_length=self.config.stc_fast_length,
            stc_slow_length=self.config.stc_slow_length,
            swing_lookback=self.config.swing_lookback,
            risk_reward_ratio=self.config.risk_reward_ratio,
            min_risk_percent=self.config.min_risk_percent,
            max_risk_percent=self.config.max_risk_percent
        )
        
        self.trades: List[TradeRecord] = []
        self._trade_counter = 0
        self._current_position: Optional[TradeRecord] = None
        
    def _create_data_fetcher(self) -> BinanceDataFetcher:
        """Create a data fetcher if not provided"""
        if self.data_fetcher:
            return self.data_fetcher
        return BinanceDataFetcher(
            symbol=self.config.symbol,
            interval=self.config.timeframe
        )
    
    async def run_backtest(
        self,
        lookback_days: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Run a complete backtest simulation
        
        Args:
            lookback_days: Override config lookback days
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with backtest results and metrics
        """
        days = lookback_days or self.config.lookback_days
        
        bars_per_day = {
            '1m': 1440,
            '5m': 288,
            '15m': 96,
            '30m': 48,
            '1h': 24,
            '4h': 6,
            '1d': 1
        }
        
        bars_needed = days * bars_per_day.get(self.config.timeframe, 288)
        bars_needed = min(bars_needed, self.config.max_candles)
        
        logger.info(f"Starting backtest: {days} days, ~{bars_needed} bars")
        
        fetcher = self._create_data_fetcher()
        df = fetcher.fetch_historical_data(limit=bars_needed + 200)
        
        if df is None or len(df) < 100:
            logger.error(f"Insufficient data for backtest: {len(df) if df is not None else 0} bars")
            return {
                'success': False,
                'error': 'Insufficient historical data',
                'trades': [],
                'metrics': {}
            }
        
        logger.info(f"Fetched {len(df)} candles for backtest")
        
        self.trades = []
        self._trade_counter = 0
        self._current_position = None
        
        df = self.signal_engine.calculate_indicators(df)
        
        warmup_period = max(self.config.stc_length, self.config.ut_atr_period, 100) + 50
        
        total_bars = len(df) - warmup_period
        
        for i in range(warmup_period, len(df)):
            current_bar = df.iloc[i]
            current_time = df.index[i]
            current_high = current_bar['high']
            current_low = current_bar['low']
            current_close = current_bar['close']
            
            if self._current_position:
                self._check_exit(current_time, current_high, current_low, current_close, i - warmup_period)
            
            if not self._current_position:
                history_df = df.iloc[:i+1].copy()
                signal = self.signal_engine.generate_signal(history_df)
                
                if signal:
                    self._open_position(signal, current_time, current_close)
            
            if progress_callback and (i - warmup_period) % 100 == 0:
                progress = (i - warmup_period) / total_bars * 100
                await progress_callback(progress)
        
        if self._current_position:
            last_bar = df.iloc[-1]
            last_time = df.index[-1]
            self._close_position(last_time, last_bar['close'], "End of backtest")
        
        from .backtest_metrics import BacktestMetrics
        metrics = BacktestMetrics(self.trades)
        
        return {
            'success': True,
            'symbol': self.config.symbol,
            'timeframe': self.config.timeframe,
            'period_days': days,
            'total_bars': len(df),
            'start_date': df.index[warmup_period].strftime('%Y-%m-%d %H:%M'),
            'end_date': df.index[-1].strftime('%Y-%m-%d %H:%M'),
            'trades': [t.to_dict() for t in self.trades],
            'metrics': metrics.calculate_all_metrics()
        }
    
    def _open_position(self, signal: Dict[str, Any], entry_time: datetime, entry_price: float):
        """Open a new position based on signal"""
        self._trade_counter += 1
        
        self._current_position = TradeRecord(
            trade_id=self._trade_counter,
            direction=signal.get('type', 'LONG'),
            entry_time=entry_time,
            entry_price=entry_price,
            stop_loss=signal.get('stop_loss', 0),
            take_profit=signal.get('take_profit', 0)
        )
        
        logger.debug(f"Opened {self._current_position.direction} at {entry_price:.2f}")
    
    def _check_exit(self, current_time: datetime, high: float, low: float, close: float, bars_held: int):
        """Check if current position should be exited"""
        if not self._current_position:
            return
        
        pos = self._current_position
        pos.holding_bars = bars_held
        
        if pos.direction == 'LONG':
            if low <= pos.stop_loss:
                self._close_position(current_time, pos.stop_loss, "Stop Loss")
            elif high >= pos.take_profit:
                self._close_position(current_time, pos.take_profit, "Take Profit")
        else:
            if high >= pos.stop_loss:
                self._close_position(current_time, pos.stop_loss, "Stop Loss")
            elif low <= pos.take_profit:
                self._close_position(current_time, pos.take_profit, "Take Profit")
    
    def _close_position(self, exit_time: datetime, exit_price: float, exit_reason: str):
        """Close the current position and record the trade"""
        if not self._current_position:
            return
        
        pos = self._current_position
        pos.exit_time = exit_time
        pos.exit_price = exit_price
        pos.exit_reason = exit_reason
        
        if pos.direction == 'LONG':
            pos.pnl_percent = (exit_price - pos.entry_price) / pos.entry_price * 100
            pos.pnl_absolute = exit_price - pos.entry_price
        else:
            pos.pnl_percent = (pos.entry_price - exit_price) / pos.entry_price * 100
            pos.pnl_absolute = pos.entry_price - exit_price
        
        pos.is_winner = pos.pnl_percent > 0
        
        risk = abs(pos.entry_price - pos.stop_loss)
        if risk > 0:
            if pos.direction == 'LONG':
                pos.risk_reward_achieved = (exit_price - pos.entry_price) / risk
            else:
                pos.risk_reward_achieved = (pos.entry_price - exit_price) / risk
        
        self.trades.append(pos)
        self._current_position = None
        
        logger.debug(f"Closed {pos.direction} trade: {pos.pnl_percent:.2f}% ({pos.exit_reason})")
