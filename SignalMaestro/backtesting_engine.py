#!/usr/bin/env python3
"""
Comprehensive Backtesting Engine for Ultimate Trading Bot
Simulates exact bot behavior with dynamic leverage, ML filtering, and 3-level stop loss system
Designed for $10 capital with 10% risk per trade and 3 maximum concurrent positions
"""

import asyncio
import logging
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from decimal import Decimal, ROUND_DOWN
import sqlite3
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Import existing bot components
try:
    from config import Config
    from technical_analysis import TechnicalAnalysis
    from ultimate_scalping_strategy import UltimateScalpingStrategy, UltimateSignal
    from dynamic_stop_loss_system import (
        StopLossConfig, DynamicStopLoss, StopLossLevel, 
        VolatilityLevel, MarketSession, MarketAnalyzer
    )
    from ml_trade_analyzer import MLTradeAnalyzer
    EXISTING_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some existing components not available: {e}")
    EXISTING_COMPONENTS_AVAILABLE = False

# Machine Learning imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

@dataclass
class BacktestConfig:
    """Backtesting configuration parameters"""
    initial_capital: float = 10.0  # $10 USD
    risk_percentage: float = 10.0  # 10% risk per trade
    max_concurrent_trades: int = 3  # Maximum 3 concurrent positions
    commission_rate: float = 0.0004  # 0.04% futures commission
    
    # Dynamic leverage settings
    min_leverage: int = 10
    max_leverage: int = 75
    
    # Stop loss levels (percentage)
    sl1_percent: float = 1.5
    sl2_percent: float = 4.0
    sl3_percent: float = 7.5
    
    # Position closure percentages at each SL level
    sl1_close_percent: float = 33.0
    sl2_close_percent: float = 33.0
    sl3_close_percent: float = 34.0
    
    # Take profit levels
    tp1_percent: float = 2.0
    tp2_percent: float = 4.0
    tp3_percent: float = 6.0
    
    # Time settings
    start_date: datetime = None
    end_date: datetime = None
    timeframes: List[str] = None
    
    def __post_init__(self):
        if self.start_date is None:
            self.start_date = datetime.now() - timedelta(days=60)
        if self.end_date is None:
            self.end_date = datetime.now() - timedelta(days=1)
        if self.timeframes is None:
            self.timeframes = ['3m', '5m', '15m', '1h', '4h']

@dataclass
class Position:
    """Trading position tracking"""
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    entry_time: datetime
    size: float
    leverage: int
    margin_used: float
    
    # Stop loss levels
    sl1_price: float
    sl2_price: float
    sl3_price: float
    sl1_triggered: bool = False
    sl2_triggered: bool = False
    sl3_triggered: bool = False
    
    # Take profit levels
    tp1_price: float
    tp2_price: float
    tp3_price: float
    tp1_triggered: bool = False
    tp2_triggered: bool = False
    tp3_triggered: bool = False
    
    # Position status
    remaining_size: float = None
    current_pnl: float = 0.0
    max_pnl: float = 0.0
    min_pnl: float = 0.0
    is_closed: bool = False
    close_time: Optional[datetime] = None
    close_reason: str = ""
    final_pnl: float = 0.0
    
    def __post_init__(self):
        if self.remaining_size is None:
            self.remaining_size = self.size

@dataclass
class TradeResult:
    """Individual trade result for analysis"""
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    size: float
    leverage: int
    pnl: float
    pnl_percentage: float
    duration_minutes: float
    close_reason: str
    commission_paid: float
    sl_levels_hit: List[str]
    tp_levels_hit: List[str]

@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics"""
    # Basic metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # PnL metrics
    total_pnl: float = 0.0
    total_pnl_percentage: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    
    # Streak metrics
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    current_win_streak: int = 0
    current_loss_streak: int = 0
    
    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_percentage: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    # Trade metrics
    avg_trade_duration: float = 0.0  # minutes
    avg_win_amount: float = 0.0
    avg_loss_amount: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Frequency metrics
    trades_per_hour: float = 0.0
    trades_per_day: float = 0.0
    
    # Commission metrics
    total_commission: float = 0.0
    
    # Capital metrics
    final_capital: float = 0.0
    peak_capital: float = 0.0
    
    # Leverage analysis
    avg_leverage_used: float = 0.0
    leverage_efficiency: float = 0.0

class DynamicLeverageCalculator:
    """Calculates dynamic leverage based on market conditions"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_leverage(self, volatility: float, volume_ratio: float = 1.0, 
                          trend_strength: float = 0.5) -> int:
        """
        Calculate dynamic leverage based on market conditions
        
        Args:
            volatility: Market volatility (ATR percentage)
            volume_ratio: Volume compared to average
            trend_strength: Trend strength indicator (0-1)
        """
        try:
            # Base leverage calculation (inverse relationship with volatility)
            if volatility <= 0.001:  # Very low volatility
                base_leverage = self.config.max_leverage
            elif volatility <= 0.005:  # Ultra low volatility
                base_leverage = self.config.max_leverage
            elif volatility <= 0.01:  # Low volatility
                base_leverage = 70
            elif volatility <= 0.02:  # Medium volatility
                base_leverage = 55
            elif volatility <= 0.03:  # High volatility
                base_leverage = 35
            elif volatility <= 0.05:  # Very high volatility
                base_leverage = 20
            else:  # Extreme volatility
                base_leverage = self.config.min_leverage
            
            # Volume adjustment
            if volume_ratio > 1.5:  # High volume
                base_leverage = min(base_leverage * 1.1, self.config.max_leverage)
            elif volume_ratio < 0.8:  # Low volume
                base_leverage = max(base_leverage * 0.9, self.config.min_leverage)
            
            # Trend strength adjustment
            if trend_strength > 0.8:  # Strong trend
                base_leverage = min(base_leverage * 1.05, self.config.max_leverage)
            elif trend_strength < 0.3:  # Weak trend
                base_leverage = max(base_leverage * 0.95, self.config.min_leverage)
            
            # Ensure within bounds
            leverage = max(self.config.min_leverage, 
                          min(self.config.max_leverage, int(base_leverage)))
            
            return leverage
            
        except Exception as e:
            self.logger.error(f"Error calculating leverage: {e}")
            return 35  # Safe default

    def get_volatility_category(self, volatility: float) -> str:
        """Get volatility category string"""
        if volatility <= 0.005:
            return "ULTRA LOW"
        elif volatility <= 0.01:
            return "LOW"
        elif volatility <= 0.02:
            return "MEDIUM"
        elif volatility <= 0.03:
            return "HIGH"
        else:
            return "EXTREME"

    def calculate_efficiency(self, leverage: int, volatility: float) -> float:
        """Calculate leverage efficiency percentage"""
        max_possible = self.config.max_leverage
        efficiency = (leverage / max_possible) * 100
        return min(100.0, efficiency)

class MLSignalFilter:
    """Machine Learning signal filtering system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Try to load existing models
        self._load_models()
    
    def _load_models(self):
        """Load existing ML models if available"""
        try:
            model_path = Path("ml_models/signal_classifier.pkl")
            scaler_path = Path("ml_models/scaler.pkl")
            
            if model_path.exists() and scaler_path.exists():
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.is_trained = True
                self.logger.info("✅ ML models loaded successfully")
            else:
                self.logger.info("ℹ️ No existing ML models found - using signal strength filtering")
                
        except Exception as e:
            self.logger.warning(f"Could not load ML models: {e}")
    
    def should_take_signal(self, signal_data: Dict[str, Any]) -> Tuple[bool, float, str]:
        """
        Determine if signal should be taken based on ML prediction
        
        Returns:
            (should_take, confidence, reason)
        """
        try:
            # Extract key features
            signal_strength = signal_data.get('signal_strength', 50)
            volatility = signal_data.get('volatility', 0.02)
            volume_ratio = signal_data.get('volume_ratio', 1.0)
            rsi = signal_data.get('rsi', 50)
            trend_strength = signal_data.get('trend_strength', 0.5)
            
            # If ML model is available, use it
            if self.is_trained and self.model is not None:
                features = np.array([[
                    signal_strength, volatility, volume_ratio, rsi, trend_strength
                ]])
                
                # Scale features
                features_scaled = self.scaler.transform(features)
                
                # Get prediction
                prediction = self.model.predict(features_scaled)[0]
                confidence = self.model.predict_proba(features_scaled)[0].max() * 100
                
                should_take = prediction == 1 and confidence >= 65
                reason = f"ML: {confidence:.1f}% confidence"
                
                return should_take, confidence, reason
            
            # Fallback to rule-based filtering
            confidence = self._calculate_fallback_confidence(signal_data)
            should_take = confidence >= 70  # High threshold for quality
            reason = f"Rule-based: {confidence:.1f}% score"
            
            return should_take, confidence, reason
            
        except Exception as e:
            self.logger.error(f"Error in ML signal filtering: {e}")
            # Conservative fallback
            return signal_data.get('signal_strength', 0) >= 80, 50, "Fallback filter"
    
    def _calculate_fallback_confidence(self, signal_data: Dict[str, Any]) -> float:
        """Calculate confidence using rule-based approach"""
        confidence = 0
        
        # Signal strength weight (40%)
        signal_strength = signal_data.get('signal_strength', 50)
        confidence += (signal_strength / 100) * 40
        
        # Volume confirmation weight (20%)
        volume_ratio = signal_data.get('volume_ratio', 1.0)
        if volume_ratio > 1.2:
            confidence += 20
        elif volume_ratio > 1.0:
            confidence += 15
        elif volume_ratio > 0.8:
            confidence += 10
        
        # Trend strength weight (20%)
        trend_strength = signal_data.get('trend_strength', 0.5)
        confidence += trend_strength * 20
        
        # Volatility adjustment weight (10%)
        volatility = signal_data.get('volatility', 0.02)
        if 0.005 <= volatility <= 0.025:  # Optimal range
            confidence += 10
        elif volatility > 0.05:  # Too high
            confidence -= 5
        
        # RSI confirmation weight (10%)
        rsi = signal_data.get('rsi', 50)
        direction = signal_data.get('direction', 'BUY')
        
        if direction == 'BUY' and rsi < 40:  # Oversold for buy
            confidence += 10
        elif direction == 'SELL' and rsi > 60:  # Overbought for sell
            confidence += 10
        elif 45 <= rsi <= 55:  # Neutral zone
            confidence += 5
        
        return min(100, max(0, confidence))

class BacktestingEngine:
    """Main backtesting engine"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize components
        self.leverage_calculator = DynamicLeverageCalculator(config)
        self.ml_filter = MLSignalFilter()
        self.technical_analysis = TechnicalAnalysis() if EXISTING_COMPONENTS_AVAILABLE else None
        
        # Trading state
        self.capital = config.initial_capital
        self.peak_capital = config.initial_capital
        self.active_positions: Dict[str, Position] = {}
        self.trade_history: List[TradeResult] = []
        
        # Performance tracking
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.daily_returns: List[float] = []
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        
        # Exchange connection
        self.exchange = None
        
        self.logger.info(f"🚀 Backtesting Engine initialized with ${config.initial_capital} capital")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for backtesting"""
        logger = logging.getLogger('BacktestingEngine')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def initialize_exchange(self):
        """Initialize exchange connection for data fetching"""
        try:
            self.exchange = ccxt.binance({
                'sandbox': False,  # Use mainnet for historical data
                'enableRateLimit': True,
                'timeout': 30000,
            })
            
            await self.exchange.load_markets()
            self.logger.info("✅ Exchange connection initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize exchange: {e}")
            raise
    
    async def fetch_historical_data(self, symbol: str, timeframe: str, 
                                   limit: int = 1000) -> List[List]:
        """Fetch historical OHLCV data"""
        try:
            if not self.exchange:
                await self.initialize_exchange()
            
            # Fetch data
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol, timeframe, limit=limit
            )
            
            self.logger.info(f"📊 Fetched {len(ohlcv)} {timeframe} candles for {symbol}")
            return ohlcv
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol} {timeframe}: {e}")
            return []
    
    def calculate_position_size(self, entry_price: float, leverage: int, 
                              stop_loss_price: float) -> Tuple[float, float]:
        """
        Calculate position size based on risk management
        
        Returns:
            (position_size, margin_required)
        """
        try:
            # Calculate risk amount (10% of capital)
            risk_amount = self.capital * (self.config.risk_percentage / 100)
            
            # Calculate price difference to stop loss
            price_diff = abs(entry_price - stop_loss_price)
            price_diff_percent = price_diff / entry_price
            
            # Calculate position value needed
            position_value = risk_amount / price_diff_percent
            
            # Calculate position size
            position_size = position_value / entry_price
            
            # Calculate margin required
            margin_required = position_value / leverage
            
            # Ensure we don't exceed available capital
            available_capital = self.capital - sum(
                pos.margin_used for pos in self.active_positions.values()
            )
            
            if margin_required > available_capital:
                # Scale down position to fit available capital
                scale_factor = available_capital / margin_required
                position_size *= scale_factor
                margin_required = available_capital
            
            return position_size, margin_required
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0, 0.0
    
    def calculate_stop_loss_levels(self, entry_price: float, direction: str) -> Tuple[float, float, float]:
        """Calculate 3-level stop loss prices"""
        if direction.lower() == 'long':
            sl1 = entry_price * (1 - self.config.sl1_percent / 100)
            sl2 = entry_price * (1 - self.config.sl2_percent / 100)
            sl3 = entry_price * (1 - self.config.sl3_percent / 100)
        else:  # short
            sl1 = entry_price * (1 + self.config.sl1_percent / 100)
            sl2 = entry_price * (1 + self.config.sl2_percent / 100)
            sl3 = entry_price * (1 + self.config.sl3_percent / 100)
        
        return sl1, sl2, sl3
    
    def calculate_take_profit_levels(self, entry_price: float, direction: str) -> Tuple[float, float, float]:
        """Calculate 3-level take profit prices"""
        if direction.lower() == 'long':
            tp1 = entry_price * (1 + self.config.tp1_percent / 100)
            tp2 = entry_price * (1 + self.config.tp2_percent / 100)
            tp3 = entry_price * (1 + self.config.tp3_percent / 100)
        else:  # short
            tp1 = entry_price * (1 - self.config.tp1_percent / 100)
            tp2 = entry_price * (1 - self.config.tp2_percent / 100)
            tp3 = entry_price * (1 - self.config.tp3_percent / 100)
        
        return tp1, tp2, tp3
    
    async def process_signal(self, signal: Dict[str, Any], current_time: datetime) -> bool:
        """Process trading signal and potentially open position"""
        try:
            # Check if we can take more positions
            if len(self.active_positions) >= self.config.max_concurrent_trades:
                return False
            
            # Extract signal data
            symbol = signal.get('symbol', '')
            direction = signal.get('direction', '')
            entry_price = signal.get('entry_price', 0.0)
            
            if not all([symbol, direction, entry_price]):
                return False
            
            # ML filtering
            should_take, confidence, reason = self.ml_filter.should_take_signal(signal)
            
            if not should_take:
                self.logger.debug(f"❌ Signal filtered out: {symbol} - {reason}")
                return False
            
            # Calculate volatility and leverage
            volatility = signal.get('volatility', 0.02)
            volume_ratio = signal.get('volume_ratio', 1.0)
            trend_strength = signal.get('trend_strength', 0.5)
            
            leverage = self.leverage_calculator.calculate_leverage(
                volatility, volume_ratio, trend_strength
            )
            
            # Calculate stop loss and take profit levels
            sl1, sl2, sl3 = self.calculate_stop_loss_levels(entry_price, direction)
            tp1, tp2, tp3 = self.calculate_take_profit_levels(entry_price, direction)
            
            # Calculate position size
            stop_loss_price = sl3  # Use final stop loss for risk calculation
            position_size, margin_required = self.calculate_position_size(
                entry_price, leverage, stop_loss_price
            )
            
            if position_size <= 0 or margin_required <= 0:
                return False
            
            # Create position
            position = Position(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                entry_time=current_time,
                size=position_size,
                leverage=leverage,
                margin_used=margin_required,
                sl1_price=sl1,
                sl2_price=sl2,
                sl3_price=sl3,
                tp1_price=tp1,
                tp2_price=tp2,
                tp3_price=tp3
            )
            
            # Add to active positions
            self.active_positions[symbol] = position
            
            self.logger.info(
                f"📈 OPENED {direction.upper()} {symbol} @ ${entry_price:.4f} | "
                f"Size: {position_size:.6f} | Leverage: {leverage}x | "
                f"Margin: ${margin_required:.2f} | Confidence: {confidence:.1f}%"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")
            return False
    
    def update_position(self, symbol: str, current_price: float, current_time: datetime):
        """Update position with current market price"""
        if symbol not in self.active_positions:
            return
        
        position = self.active_positions[symbol]
        
        if position.is_closed:
            return
        
        # Calculate current PnL
        if position.direction.lower() == 'long':
            pnl = (current_price - position.entry_price) * position.remaining_size
        else:
            pnl = (position.entry_price - current_price) * position.remaining_size
        
        position.current_pnl = pnl
        position.max_pnl = max(position.max_pnl, pnl)
        position.min_pnl = min(position.min_pnl, pnl)
        
        # Check stop loss and take profit levels
        self._check_exit_conditions(symbol, current_price, current_time)
    
    def _check_exit_conditions(self, symbol: str, current_price: float, current_time: datetime):
        """Check if position should be closed due to SL or TP"""
        position = self.active_positions[symbol]
        
        if position.is_closed:
            return
        
        # Check stop loss levels
        if position.direction.lower() == 'long':
            # Long position stop losses
            if not position.sl3_triggered and current_price <= position.sl3_price:
                self._trigger_stop_loss(position, 3, current_price, current_time)
            elif not position.sl2_triggered and current_price <= position.sl2_price:
                self._trigger_stop_loss(position, 2, current_price, current_time)
            elif not position.sl1_triggered and current_price <= position.sl1_price:
                self._trigger_stop_loss(position, 1, current_price, current_time)
            
            # Long position take profits
            elif not position.tp1_triggered and current_price >= position.tp1_price:
                self._trigger_take_profit(position, 1, current_price, current_time)
            elif not position.tp2_triggered and current_price >= position.tp2_price:
                self._trigger_take_profit(position, 2, current_price, current_time)
            elif not position.tp3_triggered and current_price >= position.tp3_price:
                self._trigger_take_profit(position, 3, current_price, current_time)
        
        else:  # short position
            # Short position stop losses
            if not position.sl3_triggered and current_price >= position.sl3_price:
                self._trigger_stop_loss(position, 3, current_price, current_time)
            elif not position.sl2_triggered and current_price >= position.sl2_price:
                self._trigger_stop_loss(position, 2, current_price, current_time)
            elif not position.sl1_triggered and current_price >= position.sl1_price:
                self._trigger_stop_loss(position, 1, current_price, current_time)
            
            # Short position take profits
            elif not position.tp1_triggered and current_price <= position.tp1_price:
                self._trigger_take_profit(position, 1, current_price, current_time)
            elif not position.tp2_triggered and current_price <= position.tp2_price:
                self._trigger_take_profit(position, 2, current_price, current_time)
            elif not position.tp3_triggered and current_price <= position.tp3_price:
                self._trigger_take_profit(position, 3, current_price, current_time)
    
    def _trigger_stop_loss(self, position: Position, level: int, exit_price: float, exit_time: datetime):
        """Trigger stop loss level"""
        if level == 1 and not position.sl1_triggered:
            close_percent = self.config.sl1_close_percent
            position.sl1_triggered = True
            reason = "SL1"
        elif level == 2 and not position.sl2_triggered:
            close_percent = self.config.sl2_close_percent
            position.sl2_triggered = True
            reason = "SL2"
        elif level == 3 and not position.sl3_triggered:
            close_percent = self.config.sl3_close_percent
            position.sl3_triggered = True
            reason = "SL3"
        else:
            return
        
        # Calculate amount to close
        close_size = position.remaining_size * (close_percent / 100)
        
        # Execute partial closure
        self._close_position_partial(position, close_size, exit_price, exit_time, reason)
        
        self.logger.info(
            f"🛑 {reason} triggered for {position.symbol} @ ${exit_price:.4f} | "
            f"Closed {close_percent}% ({close_size:.6f})"
        )
    
    def _trigger_take_profit(self, position: Position, level: int, exit_price: float, exit_time: datetime):
        """Trigger take profit level"""
        if level == 1 and not position.tp1_triggered:
            close_percent = 33.0  # Close 33% at TP1
            position.tp1_triggered = True
            reason = "TP1"
        elif level == 2 and not position.tp2_triggered:
            close_percent = 50.0  # Close 50% of remaining at TP2
            position.tp2_triggered = True
            reason = "TP2"
        elif level == 3 and not position.tp3_triggered:
            close_percent = 100.0  # Close all remaining at TP3
            position.tp3_triggered = True
            reason = "TP3"
        else:
            return
        
        # Calculate amount to close
        close_size = position.remaining_size * (close_percent / 100)
        
        # Execute partial closure
        self._close_position_partial(position, close_size, exit_price, exit_time, reason)
        
        self.logger.info(
            f"🎯 {reason} triggered for {position.symbol} @ ${exit_price:.4f} | "
            f"Closed {close_percent}% ({close_size:.6f})"
        )
    
    def _close_position_partial(self, position: Position, close_size: float, 
                               exit_price: float, exit_time: datetime, reason: str):
        """Close part of a position"""
        # Calculate PnL for this partial close
        if position.direction.lower() == 'long':
            pnl = (exit_price - position.entry_price) * close_size
        else:
            pnl = (position.entry_price - exit_price) * close_size
        
        # Calculate commission
        trade_value = close_size * exit_price
        commission = trade_value * self.config.commission_rate
        pnl -= commission
        
        # Update capital
        self.capital += pnl
        
        # Update position
        position.remaining_size -= close_size
        position.final_pnl += pnl
        
        # Create trade result
        duration_minutes = (exit_time - position.entry_time).total_seconds() / 60
        
        trade_result = TradeResult(
            symbol=position.symbol,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_time=position.entry_time,
            exit_time=exit_time,
            size=close_size,
            leverage=position.leverage,
            pnl=pnl,
            pnl_percentage=(pnl / (close_size * position.entry_price / position.leverage)) * 100,
            duration_minutes=duration_minutes,
            close_reason=reason,
            commission_paid=commission,
            sl_levels_hit=[],
            tp_levels_hit=[]
        )
        
        self.trade_history.append(trade_result)
        
        # Check if position is fully closed
        if position.remaining_size <= 1e-8:  # Account for floating point precision
            position.is_closed = True
            position.close_time = exit_time
            position.close_reason = reason
            
            # Release margin
            self.capital += position.margin_used
            
            # Remove from active positions
            del self.active_positions[position.symbol]
        
        # Update equity curve
        self.equity_curve.append((exit_time, self.capital))
        
        # Update peak capital and drawdown
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_capital - self.capital) / self.peak_capital * 100
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
    
    def calculate_metrics(self) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics"""
        if not self.trade_history:
            return BacktestMetrics()
        
        metrics = BacktestMetrics()
        
        # Basic counts
        metrics.total_trades = len(self.trade_history)
        metrics.winning_trades = sum(1 for trade in self.trade_history if trade.pnl > 0)
        metrics.losing_trades = metrics.total_trades - metrics.winning_trades
        metrics.win_rate = (metrics.winning_trades / metrics.total_trades) * 100 if metrics.total_trades > 0 else 0
        
        # PnL metrics
        metrics.total_pnl = sum(trade.pnl for trade in self.trade_history)
        metrics.total_pnl_percentage = (metrics.total_pnl / self.config.initial_capital) * 100
        metrics.gross_profit = sum(trade.pnl for trade in self.trade_history if trade.pnl > 0)
        metrics.gross_loss = abs(sum(trade.pnl for trade in self.trade_history if trade.pnl < 0))
        metrics.profit_factor = metrics.gross_profit / metrics.gross_loss if metrics.gross_loss > 0 else float('inf')
        
        # Streak calculations
        current_win_streak = 0
        current_loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        for trade in self.trade_history:
            if trade.pnl > 0:
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            else:
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)
        
        metrics.max_consecutive_wins = max_win_streak
        metrics.max_consecutive_losses = max_loss_streak
        metrics.current_win_streak = current_win_streak
        metrics.current_loss_streak = current_loss_streak
        
        # Risk metrics
        metrics.max_drawdown = self.max_drawdown
        metrics.max_drawdown_percentage = self.max_drawdown
        
        # Calculate Sharpe ratio
        returns = [trade.pnl / self.config.initial_capital for trade in self.trade_history]
        if len(returns) > 1:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            metrics.sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        
        # Trade metrics
        metrics.avg_trade_duration = np.mean([trade.duration_minutes for trade in self.trade_history])
        
        winning_trades_pnl = [trade.pnl for trade in self.trade_history if trade.pnl > 0]
        losing_trades_pnl = [trade.pnl for trade in self.trade_history if trade.pnl < 0]
        
        metrics.avg_win_amount = np.mean(winning_trades_pnl) if winning_trades_pnl else 0
        metrics.avg_loss_amount = np.mean(losing_trades_pnl) if losing_trades_pnl else 0
        metrics.largest_win = max(winning_trades_pnl) if winning_trades_pnl else 0
        metrics.largest_loss = min(losing_trades_pnl) if losing_trades_pnl else 0
        
        # Frequency metrics
        if self.trade_history:
            total_time = (self.trade_history[-1].exit_time - self.trade_history[0].entry_time)
            total_hours = total_time.total_seconds() / 3600
            total_days = total_hours / 24
            
            metrics.trades_per_hour = metrics.total_trades / total_hours if total_hours > 0 else 0
            metrics.trades_per_day = metrics.total_trades / total_days if total_days > 0 else 0
        
        # Commission metrics
        metrics.total_commission = sum(trade.commission_paid for trade in self.trade_history)
        
        # Capital metrics
        metrics.final_capital = self.capital
        metrics.peak_capital = self.peak_capital
        
        # Leverage metrics
        metrics.avg_leverage_used = np.mean([trade.leverage for trade in self.trade_history])
        
        return metrics
    
    async def run_backtest(self, symbols: List[str]) -> BacktestMetrics:
        """Run the complete backtest"""
        self.logger.info(f"🚀 Starting backtest from {self.config.start_date} to {self.config.end_date}")
        self.logger.info(f"💰 Initial capital: ${self.config.initial_capital}")
        self.logger.info(f"⚡ Risk per trade: {self.config.risk_percentage}%")
        self.logger.info(f"📊 Max concurrent trades: {self.config.max_concurrent_trades}")
        
        try:
            # Initialize exchange
            await self.initialize_exchange()
            
            # Fetch historical data for all symbols
            data_cache = {}
            for symbol in symbols:
                self.logger.info(f"📥 Fetching data for {symbol}...")
                data_cache[symbol] = {}
                
                for timeframe in self.config.timeframes:
                    data = await self.fetch_historical_data(symbol, timeframe, 1000)
                    if data:
                        data_cache[symbol][timeframe] = data
                
                # Small delay to respect rate limits
                await asyncio.sleep(0.1)
            
            # Simulate trading
            await self._simulate_trading(data_cache)
            
            # Calculate final metrics
            metrics = self.calculate_metrics()
            
            self.logger.info("✅ Backtest completed successfully")
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Backtest failed: {e}")
            raise
        finally:
            if self.exchange:
                await self.exchange.close()
    
    async def _simulate_trading(self, data_cache: Dict[str, Dict[str, List]]):
        """Simulate trading based on historical data"""
        # This is a simplified simulation - in a real implementation,
        # you would iterate through historical data chronologically
        # and generate signals using the existing bot's strategy
        
        self.logger.info("🔄 Simulating trading...")
        
        # For demonstration, we'll create some sample signals
        # In a real implementation, this would use the existing bot's signal generation logic
        
        sample_signals = [
            {
                'symbol': 'BTCUSDT',
                'direction': 'long',
                'entry_price': 58000.0,
                'signal_strength': 85,
                'volatility': 0.015,
                'volume_ratio': 1.3,
                'trend_strength': 0.8,
                'rsi': 35,
            },
            {
                'symbol': 'ETHUSDT',
                'direction': 'short',
                'entry_price': 2300.0,
                'signal_strength': 78,
                'volatility': 0.020,
                'volume_ratio': 1.1,
                'trend_strength': 0.7,
                'rsi': 65,
            },
            {
                'symbol': 'BNBUSDT',
                'direction': 'long',
                'entry_price': 220.0,
                'signal_strength': 82,
                'volatility': 0.012,
                'volume_ratio': 1.5,
                'trend_strength': 0.9,
                'rsi': 28,
            }
        ]
        
        current_time = self.config.start_date
        
        for i, signal in enumerate(sample_signals):
            await self.process_signal(signal, current_time)
            current_time += timedelta(hours=1)
            
            # Simulate price movements and position updates
            symbol = signal['symbol']
            if symbol in self.active_positions:
                # Simulate some price movements
                entry_price = signal['entry_price']
                
                # Simulate favorable movement first
                if signal['direction'] == 'long':
                    price_move = entry_price * 1.025  # +2.5%
                else:
                    price_move = entry_price * 0.975  # -2.5%
                
                self.update_position(symbol, price_move, current_time + timedelta(minutes=30))
                
                # Then simulate stop loss hit
                if signal['direction'] == 'long':
                    sl_price = entry_price * 0.985  # -1.5% (SL1)
                else:
                    sl_price = entry_price * 1.015  # +1.5% (SL1)
                
                self.update_position(symbol, sl_price, current_time + timedelta(hours=2))
        
        self.logger.info(f"📈 Simulation completed with {len(self.trade_history)} trades")

# Example usage and testing functions
async def main():
    """Example usage of the backtesting engine"""
    
    # Configure backtest
    config = BacktestConfig(
        initial_capital=10.0,
        risk_percentage=10.0,
        max_concurrent_trades=3,
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now() - timedelta(days=1)
    )
    
    # Initialize engine
    engine = BacktestingEngine(config)
    
    # Define symbols to test
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
    
    try:
        # Run backtest
        metrics = await engine.run_backtest(symbols)
        
        # Print results
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f"Initial Capital: ${config.initial_capital:.2f}")
        print(f"Final Capital: ${metrics.final_capital:.2f}")
        print(f"Total PnL: ${metrics.total_pnl:.2f} ({metrics.total_pnl_percentage:.2f}%)")
        print(f"Total Trades: {metrics.total_trades}")
        print(f"Win Rate: {metrics.win_rate:.2f}%")
        print(f"Profit Factor: {metrics.profit_factor:.2f}")
        print(f"Max Drawdown: {metrics.max_drawdown:.2f}%")
        print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"Avg Leverage: {metrics.avg_leverage_used:.1f}x")
        print("="*50)
        
    except Exception as e:
        print(f"Error running backtest: {e}")

if __name__ == "__main__":
    asyncio.run(main())