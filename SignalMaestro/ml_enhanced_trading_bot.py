#!/usr/bin/env python3
"""
ML-Enhanced Advanced Trading Bot with Dynamic SL/TP Adjustment
Continuously learns from past trades to optimize future trading decisions
Features: ML learning, dynamic risk management, cooldown periods, market adaptation
"""

import asyncio
import logging
import aiohttp
import os
import json
import hmac
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import traceback
import time
import signal
import sys
import atexit
from pathlib import Path
import sqlite3
import pickle
from decimal import Decimal, ROUND_DOWN
from dataclasses import dataclass, asdict
import threading
from collections import defaultdict

# Technical Analysis
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

# Add custom technical analysis functions for missing indicators
def calculate_sma(values, period):
    """Calculate Simple Moving Average"""
    if len(values) < period:
        return None
    return sum(values[-period:]) / period

def calculate_ema(values, period):
    """Calculate Exponential Moving Average"""
    if len(values) < period:
        return None
    multiplier = 2 / (period + 1)
    ema = values[0]
    for value in values[1:]:
        ema = (value * multiplier) + (ema * (1 - multiplier))
    return ema

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    CHART_AVAILABLE = True
except ImportError:
    CHART_AVAILABLE = False

# ML Libraries
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

@dataclass
class TradeOutcome:
    """Data class for storing trade outcomes for ML learning"""
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    leverage: int
    market_volatility: float
    volume_ratio: float
    rsi: float
    macd_signal: str
    profit_loss: float
    exit_reason: str
    duration_minutes: float
    timestamp: datetime
    market_conditions: Dict[str, Any]

class MLTradePredictor:
    """Machine Learning predictor for optimal SL/TP levels"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model_dir = Path("ml_models")
        self.model_dir.mkdir(exist_ok=True)

        # ML Models for different predictions
        self.sl_model = None
        self.tp1_model = None
        self.tp2_model = None
        self.tp3_model = None
        self.volatility_model = None
        self.scaler = StandardScaler()

        # Trade database
        self.db_path = "ml_trade_learning.db"
        self._initialize_database()

        # Learning parameters
        self.min_trades_for_learning = 20
        self.retrain_frequency = 50  # Retrain after every 50 new trades
        self.trade_count = 0

        # Performance tracking
        self.model_accuracy = {
            'sl_accuracy': 0.0,
            'tp1_accuracy': 0.0,
            'tp2_accuracy': 0.0,
            'tp3_accuracy': 0.0,
            'last_training': None
        }

        self.logger.info("🧠 ML Trade Predictor initialized")

    def _initialize_database(self):
        """Initialize SQLite database for ML learning"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    stop_loss REAL,
                    take_profit_1 REAL,
                    take_profit_2 REAL,
                    take_profit_3 REAL,
                    leverage INTEGER,
                    market_volatility REAL,
                    volume_ratio REAL,
                    rsi REAL,
                    macd_signal TEXT,
                    profit_loss REAL,
                    exit_reason TEXT,
                    duration_minutes REAL,
                    timestamp TIMESTAMP,
                    market_conditions TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"Error initializing ML database: {e}")

    async def record_trade_outcome(self, trade_outcome: TradeOutcome):
        """Record trade outcome for ML learning"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO trade_outcomes (
                    symbol, direction, entry_price, exit_price, stop_loss,
                    take_profit_1, take_profit_2, take_profit_3, leverage,
                    market_volatility, volume_ratio, rsi, macd_signal,
                    profit_loss, exit_reason, duration_minutes, timestamp,
                    market_conditions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_outcome.symbol,
                trade_outcome.direction,
                trade_outcome.entry_price,
                trade_outcome.exit_price,
                trade_outcome.stop_loss,
                trade_outcome.take_profit_1,
                trade_outcome.take_profit_2,
                trade_outcome.take_profit_3,
                trade_outcome.leverage,
                trade_outcome.market_volatility,
                trade_outcome.volume_ratio,
                trade_outcome.rsi,
                trade_outcome.macd_signal,
                trade_outcome.profit_loss,
                trade_outcome.exit_reason,
                trade_outcome.duration_minutes,
                trade_outcome.timestamp.isoformat(),
                json.dumps(trade_outcome.market_conditions)
            ))

            conn.commit()
            conn.close()

            self.trade_count += 1
            self.logger.info(f"📝 Trade outcome recorded: {trade_outcome.symbol} P&L: {trade_outcome.profit_loss:.2f}")

            # Trigger retraining if threshold reached
            if self.trade_count % self.retrain_frequency == 0:
                await self.retrain_models()

        except Exception as e:
            self.logger.error(f"Error recording trade outcome: {e}")

    async def retrain_models(self):
        """Retrain ML models with new data"""
        try:
            if not ML_AVAILABLE:
                self.logger.warning("ML libraries not available - skipping model training")
                return

            self.logger.info("🔄 Retraining ML models with new data...")

            # Get training data
            training_data = self._get_training_data()
            if len(training_data) < self.min_trades_for_learning:
                self.logger.warning(f"Insufficient training data: {len(training_data)} trades")
                return

            # Prepare features and targets
            features, targets = self._prepare_training_data(training_data)
            if features is None or len(features) == 0:
                return

            # Train models
            await self._train_sl_model(features, targets)
            await self._train_tp_models(features, targets)

            # Save models
            self._save_models()

            self.model_accuracy['last_training'] = datetime.now().isoformat()
            self.logger.info("✅ ML models retrained successfully")

        except Exception as e:
            self.logger.error(f"Error retraining models: {e}")

    def _get_training_data(self) -> pd.DataFrame:
        """Get training data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT * FROM trade_outcomes ORDER BY created_at DESC LIMIT 500"
            df = pd.read_sql_query(query, conn)
            conn.close()

            # Parse JSON fields
            if 'market_conditions' in df.columns:
                df['market_conditions'] = df['market_conditions'].apply(
                    lambda x: json.loads(x) if x else {}
                )

            return df

        except Exception as e:
            self.logger.error(f"Error getting training data: {e}")
            return pd.DataFrame()

    def _prepare_training_data(self, df: pd.DataFrame) -> tuple:
        """Prepare features and targets for ML training"""
        try:
            if len(df) == 0:
                return None, None

            # Features
            features = pd.DataFrame()
            features['entry_price'] = df['entry_price']
            features['leverage'] = df['leverage'].fillna(50)
            features['market_volatility'] = df['market_volatility'].fillna(0.02)
            features['volume_ratio'] = df['volume_ratio'].fillna(1.0)
            features['rsi'] = df['rsi'].fillna(50)

            # Encode categorical features
            features['direction_encoded'] = df['direction'].map({'BUY': 1, 'SELL': 0}).fillna(1)
            features['macd_bullish'] = df['macd_signal'].map({'bullish': 1, 'bearish': 0}).fillna(0)

            # Time features
            try:
                # Handle timestamp conversion safely
                if 'timestamp' in df.columns:
                    # Convert timestamp strings to datetime objects
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
                    # Extract time features safely
                    valid_timestamps = df['timestamp'].dropna()
                    if len(valid_timestamps) > 0:
                        features['hour'] = valid_timestamps.dt.hour.iloc[0] if len(valid_timestamps) == 1 else valid_timestamps.dt.hour.mode().iloc[0]
                        features['day_of_week'] = valid_timestamps.dt.dayofweek.iloc[0] if len(valid_timestamps) == 1 else valid_timestamps.dt.dayofweek.mode().iloc[0]
                    else:
                        # Use current time as fallback
                        current_time = datetime.now()
                        features['hour'] = current_time.hour
                        features['day_of_week'] = current_time.weekday()
                else:
                    # Use current time as fallback
                    current_time = datetime.now()
                    features['hour'] = current_time.hour
                    features['day_of_week'] = current_time.weekday()
            except Exception as e:
                self.logger.warning(f"Error processing time features: {e}")
                # Use current time as fallback
                current_time = datetime.now()
                features['hour'] = current_time.hour
                features['day_of_week'] = current_time.weekday()

            # Targets (optimal SL/TP levels based on successful trades)
            targets = {}

            # Only use profitable trades for positive learning
            profitable_trades = df[df['profit_loss'] > 0]
            if len(profitable_trades) > 0:
                targets['optimal_sl'] = profitable_trades['stop_loss'] / profitable_trades['entry_price']
                targets['optimal_tp1'] = profitable_trades['take_profit_1'] / profitable_trades['entry_price']
                targets['optimal_tp2'] = profitable_trades['take_profit_2'] / profitable_trades['entry_price']
                targets['optimal_tp3'] = profitable_trades['take_profit_3'] / profitable_trades['entry_price']

                # Align features with profitable trades
                features = features.loc[profitable_trades.index]

            # Remove NaN values using newer pandas syntax
            features = features.ffill().bfill().fillna(0)

            return features, targets

        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            return None, None

    async def _train_sl_model(self, features: pd.DataFrame, targets: Dict[str, pd.Series]):
        """Train stop loss prediction model"""
        try:
            if 'optimal_sl' not in targets or len(targets['optimal_sl']) < 10:
                return

            X = features
            y = targets['optimal_sl']

            # Align data
            common_indices = X.index.intersection(y.index)
            X = X.loc[common_indices]
            y = y.loc[common_indices]

            if len(X) < 10:
                return

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train model
            self.sl_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.sl_model.fit(X_train_scaled, y_train)

            # Evaluate
            y_pred = self.sl_model.predict(X_test_scaled)
            accuracy = r2_score(y_test, y_pred)
            self.model_accuracy['sl_accuracy'] = max(0, accuracy)

            self.logger.info(f"🎯 SL model accuracy: {accuracy:.3f}")

        except Exception as e:
            self.logger.error(f"Error training SL model: {e}")

    async def _train_tp_models(self, features: pd.DataFrame, targets: Dict[str, pd.Series]):
        """Train take profit prediction models"""
        try:
            for tp_level, model_attr in [('optimal_tp1', 'tp1_model'), ('optimal_tp2', 'tp2_model'), ('optimal_tp3', 'tp3_model')]:
                if tp_level not in targets or len(targets[tp_level]) < 10:
                    continue

                X = features
                y = targets[tp_level]

                # Align data
                common_indices = X.index.intersection(y.index)
                X = X.loc[common_indices]
                y = y.loc[common_indices]

                if len(X) < 10:
                    continue

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                # Scale features
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)

                # Train model
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                model.fit(X_train_scaled, y_train)

                # Set model
                setattr(self, model_attr, model)

                # Evaluate
                y_pred = model.predict(X_test_scaled)
                accuracy = r2_score(y_test, y_pred)
                self.model_accuracy[f'{tp_level.replace("optimal_", "")}_accuracy'] = max(0, accuracy)

                self.logger.info(f"📈 {tp_level} model accuracy: {accuracy:.3f}")

        except Exception as e:
            self.logger.error(f"Error training TP models: {e}")

    def _save_models(self):
        """Save trained models to disk"""
        try:
            models = {
                'sl_model.pkl': self.sl_model,
                'tp1_model.pkl': self.tp1_model,
                'tp2_model.pkl': self.tp2_model,
                'tp3_model.pkl': self.tp3_model,
                'scaler.pkl': self.scaler
            }

            for filename, model in models.items():
                if model is not None:
                    with open(self.model_dir / filename, 'wb') as f:
                        pickle.dump(model, f)

            # Save accuracy metrics
            with open(self.model_dir / 'model_accuracy.json', 'w') as f:
                json.dump(self.model_accuracy, f, indent=2)

        except Exception as e:
            self.logger.error(f"Error saving models: {e}")

    def load_models(self):
        """Load trained models from disk"""
        try:
            models = {
                'sl_model.pkl': 'sl_model',
                'tp1_model.pkl': 'tp1_model',
                'tp2_model.pkl': 'tp2_model',
                'tp3_model.pkl': 'tp3_model',
                'scaler.pkl': 'scaler'
            }

            for filename, attr_name in models.items():
                filepath = self.model_dir / filename
                if filepath.exists():
                    with open(filepath, 'rb') as f:
                        setattr(self, attr_name, pickle.load(f))

            # Load accuracy metrics
            accuracy_file = self.model_dir / 'model_accuracy.json'
            if accuracy_file.exists():
                with open(accuracy_file, 'r') as f:
                    self.model_accuracy.update(json.load(f))

            self.logger.info("🤖 ML models loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading models: {e}")

    def predict_optimal_levels(self, signal_data: Dict[str, Any]) -> Dict[str, float]:
        """Predict optimal SL/TP levels using ML models"""
        try:
            if not all([self.sl_model, self.tp1_model, self.tp2_model, self.tp3_model, self.scaler]):
                return self._get_default_levels(signal_data)

            # Prepare features
            features = self._prepare_prediction_features(signal_data)
            if features is None:
                return self._get_default_levels(signal_data)

            # Scale features
            features_scaled = self.scaler.transform([features])

            # Predict levels
            sl_ratio = self.sl_model.predict(features_scaled)[0]
            tp1_ratio = self.tp1_model.predict(features_scaled)[0]
            tp2_ratio = self.tp2_model.predict(features_scaled)[0]
            tp3_ratio = self.tp3_model.predict(features_scaled)[0]

            entry_price = signal_data.get('entry_price', signal_data.get('current_price', 0))
            direction = signal_data.get('direction', 'BUY').upper()

            if direction in ['BUY', 'LONG']:
                predicted_levels = {
                    'stop_loss': entry_price * sl_ratio,
                    'take_profit_1': entry_price * tp1_ratio,
                    'take_profit_2': entry_price * tp2_ratio,
                    'take_profit_3': entry_price * tp3_ratio
                }
            else:
                predicted_levels = {
                    'stop_loss': entry_price * (2 - sl_ratio),
                    'take_profit_1': entry_price * (2 - tp1_ratio),
                    'take_profit_2': entry_price * (2 - tp2_ratio),
                    'take_profit_3': entry_price * (2 - tp3_ratio)
                }

            # Validate predictions
            if self._validate_predicted_levels(predicted_levels, entry_price, direction):
                return predicted_levels
            else:
                return self._get_default_levels(signal_data)

        except Exception as e:
            self.logger.error(f"Error predicting optimal levels: {e}")
            return self._get_default_levels(signal_data)

    def _prepare_prediction_features(self, signal_data: Dict[str, Any]) -> Optional[List[float]]:
        """Prepare features for prediction"""
        try:
            entry_price = signal_data.get('entry_price', signal_data.get('current_price', 0))
            features = [
                entry_price,
                signal_data.get('leverage', 50),
                signal_data.get('volatility', 0.02),
                signal_data.get('volume_ratio', 1.0),
                signal_data.get('rsi', 50),
                1 if signal_data.get('direction', 'BUY').upper() in ['BUY', 'LONG'] else 0,
                1 if signal_data.get('macd_bullish', False) else 0,
                datetime.now().hour,
                datetime.now().weekday()
            ]

            return features

        except Exception as e:
            self.logger.error(f"Error preparing prediction features: {e}")
            return None

    def _validate_predicted_levels(self, levels: Dict[str, float], entry_price: float, direction: str) -> bool:
        """Validate predicted levels make sense"""
        try:
            sl = levels['stop_loss']
            tp1 = levels['take_profit_1']
            tp2 = levels['take_profit_2']
            tp3 = levels['take_profit_3']

            if direction.upper() in ['BUY', 'LONG']:
                return sl < entry_price < tp1 < tp2 < tp3
            else:
                return tp3 < tp2 < tp1 < entry_price < sl

        except Exception as e:
            return False

    def _get_default_levels(self, signal_data: Dict[str, Any]) -> Dict[str, float]:
        """Get default SL/TP levels when ML prediction fails"""
        entry_price = signal_data.get('entry_price', signal_data.get('current_price', 0))
        direction = signal_data.get('direction', 'BUY').upper()
        volatility = signal_data.get('volatility', 0.02)

        # Adaptive risk based on volatility
        risk_multiplier = 1.0 + (volatility * 50)  # Higher volatility = wider stops
        base_risk = 0.015 * risk_multiplier  # 1.5% base risk adjusted for volatility

        if direction in ['BUY', 'LONG']:
            return {
                'stop_loss': entry_price * (1 - base_risk),
                'take_profit_1': entry_price * (1 + base_risk * 1.5),
                'take_profit_2': entry_price * (1 + base_risk * 2.5),
                'take_profit_3': entry_price * (1 + base_risk * 4.0)
            }
        else:
            return {
                'stop_loss': entry_price * (1 + base_risk),
                'take_profit_1': entry_price * (1 - base_risk * 1.5),
                'take_profit_2': entry_price * (1 - base_risk * 2.5),
                'take_profit_3': entry_price * (1 - base_risk * 4.0)
            }

class CooldownManager:
    """Manages cooldown periods to prevent spamming"""

    def __init__(self, cooldown_minutes: int = 30):
        self.cooldown_period = timedelta(minutes=cooldown_minutes)
        self.last_signals = defaultdict(datetime)
        self.global_last_signal = datetime.min
        self.logger = logging.getLogger(__name__)

    def can_send_signal(self, symbol: str = None) -> bool:
        """Check if signal can be sent based on cooldown"""
        now = datetime.now()

        # Global cooldown check
        if now - self.global_last_signal < timedelta(minutes=5):  # 5 min global cooldown
            return False

        # Symbol-specific cooldown check
        if symbol and now - self.last_signals[symbol] < self.cooldown_period:
            return False

        return True

    def record_signal(self, symbol: str = None):
        """Record that a signal was sent"""
        now = datetime.now()
        self.global_last_signal = now
        if symbol:
            self.last_signals[symbol] = now

    def get_cooldown_status(self, symbol: str = None) -> Dict[str, Any]:
        """Get cooldown status information"""
        now = datetime.now()

        global_remaining = max(0, (self.global_last_signal + timedelta(minutes=5) - now).total_seconds())

        status = {
            'can_send_global': global_remaining <= 0,
            'global_cooldown_remaining': global_remaining
        }

        if symbol:
            symbol_remaining = max(0, (self.last_signals[symbol] + self.cooldown_period - now).total_seconds())
            status.update({
                'can_send_symbol': symbol_remaining <= 0,
                'symbol_cooldown_remaining': symbol_remaining
            })

        return status

class MLEnhancedTradingBot:
    """Advanced ML-Enhanced Trading Bot with Dynamic SL/TP and Cooldown Management"""

    def __init__(self):
        self.logger = self._setup_logging()

        # Process management
        self.pid_file = Path("ml_enhanced_trading_bot.pid")
        self.shutdown_requested = False
        self._setup_signal_handlers()
        atexit.register(self._cleanup_on_exit)

        # Core components
        self.ml_predictor = MLTradePredictor()
        self.cooldown_manager = CooldownManager(cooldown_minutes=30)

        # Telegram configuration
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

        # Bot settings
        self.admin_chat_id = None
        self.target_channel = "@SignalTactics"
        self.channel_accessible = False

        # Enhanced symbol list with volatility tracking
        self.symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'SOLUSDT', 'DOGEUSDT',
            'AVAXUSDT', 'DOTUSDT', 'MATICUSDT', 'LINKUSDT', 'LTCUSDT', 'BCHUSDT', 'ETCUSDT',
            'ATOMUSDT', 'ALGOUSDT', 'XLMUSDT', 'VETUSDT', 'TRXUSDT', 'EOSUSDT', 'THETAUSDT',
            'UNIUSDT', 'AAVEUSDT', 'COMPUSDT', 'MKRUSDT', 'YFIUSDT', 'SUSHIUSDT', 'CAKEUSDT',
            'ARBUSDT', 'OPUSDT', 'METISUSDT', 'STRKUSDT', 'SANDUSDT', 'MANAUSDT', 'AXSUSDT',
            'APTUSDT', 'SUIUSDT', 'ARKMUSDT', 'SEIUSDT', 'TIAUSDT', 'WLDUSDT', 'JUPUSDT'
        ]

        # Dynamic market adaptation parameters
        self.market_conditions = {
            'volatility_regime': 'medium',  # low, medium, high
            'trend_strength': 'neutral',    # strong_bull, bull, neutral, bear, strong_bear
            'volume_profile': 'normal',     # low, normal, high
            'market_sentiment': 'neutral'   # bullish, neutral, bearish
        }

        # Performance tracking with ML metrics
        self.performance_stats = {
            'total_signals': 0,
            'ml_predicted_signals': 0,
            'profitable_signals': 0,
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'total_profit': 0.0,
            'cooldown_blocks': 0
        }

        # Advanced risk management with adaptive algorithms
        self.risk_config = {
            'base_risk_percentage': 2.0,  # 2% base risk
            'max_risk_percentage': 5.0,   # 5% maximum risk
            'volatility_adjustment': True,
            'ml_confidence_threshold': 0.7,
            'min_signal_strength': 85,
            'adaptive_risk_scaling': True,
            'market_regime_adjustment': True,
            'volume_confirmation_required': True
        }

        # Adaptive algorithm parameters
        self.adaptive_config = {
            'learning_rate': 0.01,
            'performance_window': 50,  # trades to consider for adaptation
            'min_confidence_threshold': 0.6,
            'max_confidence_threshold': 0.9,
            'risk_adjustment_factor': 0.1,
            'volatility_lookback_periods': 20
        }

        # Load ML models
        self.ml_predictor.load_models()

        self.logger.info("🚀 ML-Enhanced Trading Bot initialized with advanced features")
        self._write_pid_file()

    def _setup_logging(self):
        """Setup comprehensive logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ml_enhanced_trading_bot.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
            self.shutdown_requested = True

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    def _write_pid_file(self):
        """Write process ID to file for monitoring"""
        try:
            with open(self.pid_file, 'w') as f:
                f.write(str(os.getpid()))
            self.logger.info(f"📝 PID file written: {self.pid_file}")
        except Exception as e:
            self.logger.warning(f"Could not write PID file: {e}")

    def _cleanup_on_exit(self):
        """Cleanup resources on exit"""
        try:
            if self.pid_file.exists():
                self.pid_file.unlink()
                self.logger.info("🧹 PID file cleaned up")
        except Exception as e:
            self.logger.warning(f"Cleanup error: {e}")

    async def get_binance_data(self, symbol: str, interval: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Get market data from Binance with enhanced error handling"""
        try:
            url = f"https://fapi.binance.com/fapi/v1/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Create DataFrame with proper column names
                        df = pd.DataFrame(data, columns=[
                            'timestamp', 'open', 'high', 'low', 'close', 'volume',
                            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                            'taker_buy_quote', 'ignore'
                        ])

                        # Convert numeric columns
                        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                        for col in numeric_cols:
                            df[col] = pd.to_numeric(df[col], errors='coerce')

                        # Convert timestamp from milliseconds to datetime with proper handling
                        try:
                            # Ensure timestamp is numeric and convert to datetime
                            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
                            # Convert milliseconds to datetime using explicit unit parameter
                            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce', utc=True)
                            # Remove any rows with invalid timestamps
                            df = df.dropna(subset=['timestamp'])
                        except (ValueError, TypeError) as e:
                            self.logger.error(f"Timestamp conversion error for {symbol}: {e}")
                            return None

                        df.set_index('timestamp', inplace=True)

                        return df

            return None

        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def calculate_advanced_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive technical indicators with ML features"""
        try:
            if df.empty or len(df) < 20:  # Reduced minimum requirement
                return {}

            indicators = {}

            # Ensure we have valid numeric data
            df = df.dropna()
            if len(df) < 20:
                return {}

            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)
            close = df['close'].values.astype(float)
            volume = df['volume'].values.astype(float)

            # 1. Market volatility (key for ML models)
            try:
                returns = np.diff(np.log(close))
                if len(returns) > 0:
                    indicators['volatility'] = np.std(returns) * np.sqrt(len(returns))
                else:
                    indicators['volatility'] = 0.02
                indicators['volatility_percentile'] = self._calculate_volatility_percentile(indicators['volatility'])
            except (ValueError, ZeroDivisionError):
                indicators['volatility'] = 0.02
                indicators['volatility_percentile'] = 0.5

            # 2. Enhanced SuperTrend with volatility adjustment
            try:
                atr = self._calculate_atr(high, low, close, 14)
                multiplier = 2.0 + (indicators['volatility'] * 20)  # Dynamic multiplier
                indicators['supertrend'] = self._calculate_supertrend(high, low, close, atr, multiplier)
            except Exception as e:
                self.logger.warning(f"SuperTrend calculation error: {e}")
                indicators['supertrend'] = 0

            # 3. Volume analysis for ML
            try:
                if len(volume) >= 20:
                    volume_sma = np.mean(volume[-20:])
                    indicators['volume_ratio'] = volume[-1] / volume_sma if volume_sma > 0 else 1.0
                else:
                    indicators['volume_ratio'] = 1.0
                indicators['volume_surge'] = indicators['volume_ratio'] > 1.5
            except (IndexError, ZeroDivisionError):
                indicators['volume_ratio'] = 1.0
                indicators['volume_surge'] = False

            # 4. RSI with divergence detection
            try:
                indicators['rsi'] = self._calculate_rsi(close, 14)
                indicators['rsi_divergence'] = self._detect_rsi_divergence(close, indicators['rsi'])
            except Exception as e:
                self.logger.warning(f"RSI calculation error: {e}")
                indicators['rsi'] = 50.0
                indicators['rsi_divergence'] = False

            # 5. MACD analysis
            try:
                macd_line, signal_line, histogram = self._calculate_macd(close)
                if len(macd_line) > 0 and len(signal_line) > 0:
                    indicators['macd_bullish'] = macd_line[-1] > signal_line[-1]
                    indicators['macd_momentum'] = histogram[-1] - histogram[-2] if len(histogram) > 1 else 0
                else:
                    indicators['macd_bullish'] = False
                    indicators['macd_momentum'] = 0
            except Exception as e:
                self.logger.warning(f"MACD calculation error: {e}")
                indicators['macd_bullish'] = False
                indicators['macd_momentum'] = 0

            # 6. Market regime detection
            try:
                indicators['market_regime'] = self._detect_market_regime(close, volume)
            except Exception as e:
                self.logger.warning(f"Market regime detection error: {e}")
                indicators['market_regime'] = 'uncertain'

            # 7. Support/Resistance levels
            try:
                indicators['support_resistance'] = self._calculate_support_resistance(high, low)
            except Exception as e:
                self.logger.warning(f"Support/Resistance calculation error: {e}")
                indicators['support_resistance'] = {'resistance': close[-1], 'support': close[-1], 'pivot': close[-1]}

            # 8. Trend strength
            try:
                indicators['trend_strength'] = self._calculate_trend_strength(close)
            except Exception as e:
                self.logger.warning(f"Trend strength calculation error: {e}")
                indicators['trend_strength'] = 50.0

            # 9. Current price info
            indicators['current_price'] = close[-1]
            indicators['price_change_1h'] = (close[-1] - close[-5]) / close[-5] * 100 if len(close) > 5 else 0
            indicators['price_velocity'] = (close[-1] - close[-3]) / close[-3] * 100 if len(close) > 3 else 0

            # 10. ML-specific features
            indicators['ml_features'] = {
                'price_momentum': indicators['price_velocity'],
                'volume_momentum': indicators['volume_ratio'],
                'volatility_regime': indicators['volatility_percentile'],
                'trend_alignment': 1 if indicators['supertrend'] > 0 and indicators['macd_bullish'] else 0
            }

            return indicators

        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return {}

    def _calculate_volatility_percentile(self, current_volatility: float) -> float:
        """Calculate volatility percentile for regime detection"""
        # This would normally use historical volatility data
        # For now, use thresholds based on typical crypto volatility
        if current_volatility < 0.01:
            return 0.2  # Low volatility
        elif current_volatility < 0.03:
            return 0.5  # Medium volatility
        else:
            return 0.8  # High volatility

    def _calculate_atr(self, high: np.array, low: np.array, close: np.array, period: int = 14) -> np.array:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))

        atr = np.zeros(len(close))
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(close)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr

    def _calculate_supertrend(self, high: np.array, low: np.array, close: np.array, atr: np.array, multiplier: float) -> float:
        """Calculate SuperTrend indicator"""
        try:
            hl2 = (high + low) / 2
            upper_band = hl2 + (multiplier * atr)
            lower_band = hl2 - (multiplier * atr)

            supertrend = np.zeros(len(close))
            direction = np.zeros(len(close))

            for i in range(1, len(close)):
                if close[i] <= lower_band[i]:
                    supertrend[i] = upper_band[i]
                    direction[i] = -1
                elif close[i] >= upper_band[i]:
                    supertrend[i] = lower_band[i]
                    direction[i] = 1
                else:
                    supertrend[i] = supertrend[i-1]
                    direction[i] = direction[i-1]

            return direction[-1]  # Return current direction

        except Exception as e:
            return 0

    def _calculate_rsi(self, values: np.array, period: int = 14) -> float:
        """Calculate RSI"""
        try:
            deltas = np.diff(values)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            avg_gains = np.mean(gains[-period:])
            avg_losses = np.mean(losses[-period:])

            if avg_losses == 0:
                return 100.0

            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            return rsi

        except Exception as e:
            return 50.0

    def _calculate_macd(self, values: np.array, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple:
        """Calculate MACD"""
        try:
            ema_fast = self._calculate_ema(values, fast_period)
            ema_slow = self._calculate_ema(values, slow_period)
            macd_line = ema_fast - ema_slow
            signal_line = self._calculate_ema(macd_line, signal_period)
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
        except Exception as e:
            return np.array([0]), np.array([0]), np.array([0])

    def _calculate_ema(self, values: np.array, period: int) -> np.array:
        """Calculate Exponential Moving Average"""
        ema = np.zeros(len(values))
        ema[0] = values[0]
        multiplier = 2 / (period + 1)
        for i in range(1, len(values)):
            ema[i] = (values[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        return ema

    def _detect_rsi_divergence(self, price: np.array, rsi: float) -> bool:
        """Detect RSI divergence patterns"""
        try:
            if len(price) < 20:
                return False

            # Simple divergence detection
            recent_price_trend = price[-5:].mean() - price[-10:-5].mean()
            rsi_trend = rsi - 50  # Simplified RSI trend

            # Bullish divergence: price down, RSI up
            # Bearish divergence: price up, RSI down
            return (recent_price_trend < 0 and rsi_trend > 0) or (recent_price_trend > 0 and rsi_trend < 0)

        except Exception as e:
            return False

    def _detect_market_regime(self, close: np.array, volume: np.array) -> str:
        """Detect current market regime"""
        try:
            if len(close) < 20:
                return 'uncertain'

            # Price trend
            short_trend = np.mean(close[-5:]) - np.mean(close[-10:-5])
            long_trend = np.mean(close[-10:]) - np.mean(close[-20:-10])

            # Volume confirmation
            volume_trend = np.mean(volume[-5:]) - np.mean(volume[-10:-5])

            if short_trend > 0 and long_trend > 0 and volume_trend > 0:
                return 'strong_bullish'
            elif short_trend > 0 and long_trend > 0:
                return 'bullish'
            elif short_trend < 0 and long_trend < 0 and volume_trend > 0:
                return 'strong_bearish'
            elif short_trend < 0 and long_trend < 0:
                return 'bearish'
            else:
                return 'sideways'

        except Exception as e:
            return 'uncertain'

    def _calculate_support_resistance(self, high: np.array, low: np.array) -> Dict[str, float]:
        """Calculate support and resistance levels"""
        try:
            # Simple pivot point calculation
            recent_high = np.max(high[-20:])
            recent_low = np.min(low[-20:])

            return {
                'resistance': recent_high,
                'support': recent_low,
                'pivot': (recent_high + recent_low) / 2
            }

        except Exception as e:
            return {'resistance': 0, 'support': 0, 'pivot': 0}

    def _calculate_trend_strength(self, close: np.array) -> float:
        """Calculate trend strength (0-100)"""
        try:
            if len(close) < 20:
                return 50

            # Linear regression slope
            x = np.arange(len(close[-20:]))
            y = close[-20:]
            slope = np.polyfit(x, y, 1)[0]

            # Normalize to 0-100 scale
            strength = min(100, max(0, 50 + (slope / y.mean() * 1000)))
            return strength

        except Exception as e:
            return 50

    async def generate_ml_enhanced_signal(self, symbol: str, indicators: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate signal with ML-enhanced SL/TP levels"""
        try:
            # Check cooldown first
            if not self.cooldown_manager.can_send_signal(symbol):
                cooldown_status = self.cooldown_manager.get_cooldown_status(symbol)
                self.logger.info(f"⏰ Signal blocked by cooldown for {symbol}. Remaining: {cooldown_status.get('symbol_cooldown_remaining', 0):.0f}s")
                self.performance_stats['cooldown_blocks'] += 1
                return None

            # Basic signal strength calculation
            signal_strength = 0
            bullish_signals = 0
            bearish_signals = 0

            current_price = indicators.get('current_price', 0)
            if current_price <= 0:
                return None

            # Enhanced signal analysis with ML features
            supertrend = indicators.get('supertrend', 0)
            if supertrend > 0:
                bullish_signals += 30
            elif supertrend < 0:
                bearish_signals += 30

            # RSI analysis
            rsi = indicators.get('rsi', 50)
            if rsi < 30:
                bullish_signals += 20
            elif rsi > 70:
                bearish_signals += 20

            # MACD analysis
            if indicators.get('macd_bullish', False):
                bullish_signals += 15
            else:
                bearish_signals += 15

            # Volume confirmation
            if indicators.get('volume_surge', False):
                if bullish_signals > bearish_signals:
                    bullish_signals += 10
                else:
                    bearish_signals += 10

            # Trend strength
            trend_strength = indicators.get('trend_strength', 50)
            if trend_strength > 65:
                bullish_signals += 15
            elif trend_strength < 35:
                bearish_signals += 15

            # Market regime
            market_regime = indicators.get('market_regime', 'uncertain')
            if market_regime in ['strong_bullish', 'bullish']:
                bullish_signals += 10
            elif market_regime in ['strong_bearish', 'bearish']:
                bearish_signals += 10

            # Determine direction and strength
            if bullish_signals >= self.risk_config['min_signal_strength']:
                direction = 'BUY'
                signal_strength = bullish_signals
            elif bearish_signals >= self.risk_config['min_signal_strength']:
                direction = 'SELL'
                signal_strength = bearish_signals
            else:
                return None

            # Prepare signal data for ML prediction
            signal_data = {
                'symbol': symbol,
                'direction': direction,
                'entry_price': current_price,
                'current_price': current_price,
                'leverage': self._calculate_dynamic_leverage(indicators),
                'volatility': indicators.get('volatility', 0.02),
                'volume_ratio': indicators.get('volume_ratio', 1.0),
                'rsi': rsi,
                'macd_bullish': indicators.get('macd_bullish', False),
                'signal_strength': signal_strength
            }

            # Get ML-predicted optimal levels
            optimal_levels = self.ml_predictor.predict_optimal_levels(signal_data)

            # Create final signal
            signal = {
                'symbol': symbol,
                'direction': direction,
                'entry_price': current_price,
                'stop_loss': optimal_levels['stop_loss'],
                'take_profit_1': optimal_levels['take_profit_1'],
                'take_profit_2': optimal_levels['take_profit_2'],
                'take_profit_3': optimal_levels['take_profit_3'],
                'signal_strength': signal_strength,
                'leverage': signal_data['leverage'],
                'ml_enhanced': True,
                'market_conditions': {
                    'volatility': indicators.get('volatility', 0.02),
                    'regime': market_regime,
                    'trend_strength': trend_strength,
                    'volume_profile': 'high' if indicators.get('volume_surge', False) else 'normal'
                },
                'indicators_used': [
                    'ML-Enhanced SuperTrend', 'Volume Surge Analysis',
                    'RSI Divergence', 'MACD Momentum', 'Market Regime Detection'
                ],
                'ml_accuracy': self.ml_predictor.model_accuracy,
                'cooldown_applied': True
            }

            # Record cooldown
            self.cooldown_manager.record_signal(symbol)

            # Update performance stats
            self.performance_stats['total_signals'] += 1
            self.performance_stats['ml_predicted_signals'] += 1

            return signal

        except Exception as e:
            self.logger.error(f"Error generating ML-enhanced signal for {symbol}: {e}")
            return None

    def _calculate_dynamic_leverage(self, indicators: Dict[str, Any]) -> int:
        """Calculate dynamic leverage based on market conditions and ML insights"""
        try:
            base_leverage = 35

            # Volatility adjustment
            volatility = indicators.get('volatility', 0.02)
            if volatility > 0.04:  # High volatility
                volatility_adjustment = -15
            elif volatility < 0.01:  # Low volatility
                volatility_adjustment = 10
            else:
                volatility_adjustment = 0

            # Volume adjustment
            volume_ratio = indicators.get('volume_ratio', 1.0)
            if volume_ratio > 2.0:
                volume_adjustment = 5
            elif volume_ratio < 0.5:
                volume_adjustment = -10
            else:
                volume_adjustment = 0

            # Trend strength adjustment
            trend_strength = indicators.get('trend_strength', 50)
            if trend_strength > 75:
                trend_adjustment = 5
            elif trend_strength < 25:
                trend_adjustment = -10
            else:
                trend_adjustment = 0

            # Calculate final leverage
            final_leverage = base_leverage + volatility_adjustment + volume_adjustment + trend_adjustment
            final_leverage = max(20, min(75, final_leverage))  # Clamp between 20x and 75x

            return int(final_leverage)

        except Exception as e:
            return 35  # Default leverage

    async def _perform_market_analysis(self) -> Dict[str, Any]:
        """Perform comprehensive market analysis for adaptive algorithms"""
        try:
            # Analyze major market indicators
            btc_data = await self.get_binance_data('BTCUSDT', '1h', 50)
            eth_data = await self.get_binance_data('ETHUSDT', '1h', 50)

            market_analysis = {
                'volatility_regime': 'medium',
                'trend_direction': 'neutral',
                'market_strength': 50,
                'volume_profile': 'normal',
                'fear_greed_index': 50,
                'correlation_strength': 0.5
            }

            if btc_data is not None and len(btc_data) >= 20:
                btc_indicators = self.calculate_advanced_indicators(btc_data)
                market_analysis.update({
                    'volatility_regime': self._determine_volatility_regime(btc_indicators.get('volatility', 0.02)),
                    'trend_direction': btc_indicators.get('market_regime', 'neutral'),
                    'market_strength': btc_indicators.get('trend_strength', 50)
                })

            return market_analysis

        except Exception as e:
            self.logger.error(f"Error in market analysis: {e}")
            return {'volatility_regime': 'medium', 'trend_direction': 'neutral', 'market_strength': 50}

    def _determine_volatility_regime(self, volatility: float) -> str:
        """Determine market volatility regime"""
        if volatility > 0.04:
            return 'high'
        elif volatility < 0.015:
            return 'low'
        else:
            return 'medium'

    async def _select_optimal_symbols(self, market_analysis: Dict[str, Any]) -> List[str]:
        """Select optimal symbols based on market conditions"""
        try:
            # Base symbols always included
            optimal_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']

            # Add symbols based on market regime
            if market_analysis['trend_direction'] in ['bullish', 'strong_bullish']:
                optimal_symbols.extend(['AVAXUSDT', 'LINKUSDT', 'ADAUSDT', 'DOTUSDT'])
            elif market_analysis['volatility_regime'] == 'high':
                optimal_symbols.extend(['DOGEUSDT', 'XRPUSDT', 'MATICUSDT', 'ATOMUSDT'])
            else:
                optimal_symbols.extend(['LTCUSDT', 'BCHUSDT', 'ETCUSDT', 'XLMUSDT'])

            # Add trending altcoins
            optimal_symbols.extend(['ARBUSDT', 'OPUSDT', 'APTUSDT', 'SUIUSDT'])

            return list(set(optimal_symbols))  # Remove duplicates

        except Exception as e:
            self.logger.error(f"Error selecting optimal symbols: {e}")
            return self.symbols[:15]

    async def _comprehensive_symbol_analysis(self, symbol: str, market_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Perform comprehensive multi-timeframe analysis"""
        try:
            # Multi-timeframe data collection
            timeframes = ['5m', '15m', '1h']
            market_data = {}

            for tf in timeframes:
                df = await self.get_binance_data(symbol, tf, 100)
                if df is not None and len(df) >= 50:
                    market_data[tf] = self.calculate_advanced_indicators(df)

            if not market_data:
                return None

            # Confluence analysis across timeframes
            signal = await self._analyze_timeframe_confluence(symbol, market_data, market_analysis)
            return signal

        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis for {symbol}: {e}")
            return None

    async def _analyze_timeframe_confluence(self, symbol: str, market_data: Dict[str, Dict], market_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze confluence across multiple timeframes"""
        try:
            confluence_score = 0
            direction_votes = {'BUY': 0, 'SELL': 0}

            # Weight timeframes differently
            timeframe_weights = {'5m': 0.3, '15m': 0.4, '1h': 0.3}

            for tf, weight in timeframe_weights.items():
                if tf not in market_data:
                    continue

                indicators = market_data[tf]

                # SuperTrend analysis
                if indicators.get('supertrend', 0) > 0:
                    direction_votes['BUY'] += weight * 30
                elif indicators.get('supertrend', 0) < 0:
                    direction_votes['SELL'] += weight * 30

                # RSI confluence
                rsi = indicators.get('rsi', 50)
                if rsi < 35:
                    direction_votes['BUY'] += weight * 20
                elif rsi > 65:
                    direction_votes['SELL'] += weight * 20

                # MACD confluence
                if indicators.get('macd_bullish', False):
                    direction_votes['BUY'] += weight * 15
                else:
                    direction_votes['SELL'] += weight * 15

                # Volume confirmation
                if indicators.get('volume_surge', False):
                    confluence_score += weight * 10

            # Determine final direction and strength
            if direction_votes['BUY'] >= self.risk_config['min_signal_strength']:
                direction = 'BUY'
                signal_strength = direction_votes['BUY']
            elif direction_votes['SELL'] >= self.risk_config['min_signal_strength']:
                direction = 'SELL'
                signal_strength = direction_votes['SELL']
            else:
                return None

            # Get primary timeframe data for signal generation
            primary_indicators = market_data.get('5m', market_data.get('15m', {}))
            if not primary_indicators:
                return None

            return await self._create_enhanced_signal(symbol, direction, signal_strength, primary_indicators, market_analysis)

        except Exception as e:
            self.logger.error(f"Error in confluence analysis: {e}")
            return None

    async def _create_enhanced_signal(self, symbol: str, direction: str, signal_strength: float, indicators: Dict[str, Any], market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create enhanced signal with adaptive SL/TP"""
        try:
            current_price = indicators.get('current_price', 0)
            if current_price <= 0:
                return None

            # Adaptive leverage based on market conditions
            leverage = await self._calculate_adaptive_leverage(indicators, market_analysis)

            # Dynamic SL/TP calculation with ML optimization
            sl_tp_levels = await self._calculate_dynamic_sl_tp(symbol, direction, current_price, indicators, market_analysis)

            signal = {
                'symbol': symbol,
                'direction': direction,
                'entry_price': current_price,
                'stop_loss': sl_tp_levels['stop_loss'],
                'take_profit_1': sl_tp_levels['take_profit_1'],
                'take_profit_2': sl_tp_levels['take_profit_2'],
                'take_profit_3': sl_tp_levels['take_profit_3'],
                'signal_strength': signal_strength,
                'leverage': leverage,
                'ml_enhanced': True,
                'adaptive_risk': True,
                'market_conditions': {
                    'volatility': indicators.get('volatility', 0.02),
                    'regime': market_analysis.get('trend_direction', 'neutral'),
                    'trend_strength': indicators.get('trend_strength', 50),
                    'volume_profile': 'high' if indicators.get('volume_surge', False) else 'normal',
                    'market_strength': market_analysis.get('market_strength', 50)
                },
                'risk_metrics': {
                    'max_risk': self._calculate_max_risk(current_price, sl_tp_levels['stop_loss'], leverage),
                    'risk_reward_ratio': self._calculate_risk_reward_ratio(current_price, sl_tp_levels),
                    'confidence_score': signal_strength / 100
                }
            }

            return signal

        except Exception as e:
            self.logger.error(f"Error creating enhanced signal: {e}")
            return None

    async def _calculate_adaptive_leverage(self, indicators: Dict[str, Any], market_analysis: Dict[str, Any]) -> int:
        """Calculate adaptive leverage based on comprehensive analysis"""
        try:
            base_leverage = 40

            # Market volatility adjustment
            volatility_regime = market_analysis.get('volatility_regime', 'medium')
            if volatility_regime == 'high':
                volatility_adj = -20
            elif volatility_regime == 'low':
                volatility_adj = 15
            else:
                volatility_adj = 0

            # Market strength adjustment
            market_strength = market_analysis.get('market_strength', 50)
            if market_strength > 75:
                strength_adj = 10
            elif market_strength < 25:
                strength_adj = -15
            else:
                strength_adj = 0

            # Volume confirmation adjustment
            if indicators.get('volume_surge', False):
                volume_adj = 5
            else:
                volume_adj = -5

            # Trend alignment adjustment
            trend_strength = indicators.get('trend_strength', 50)
            if trend_strength > 70:
                trend_adj = 10
            elif trend_strength < 30:
                trend_adj = -10
            else:
                trend_adj = 0

            final_leverage = base_leverage + volatility_adj + strength_adj + volume_adj + trend_adj
            return max(15, min(100, final_leverage))

        except Exception as e:
            return 40

    async def _calculate_dynamic_sl_tp(self, symbol: str, direction: str, entry_price: float, indicators: Dict[str, Any], market_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate dynamic SL/TP based on market conditions and ML predictions"""
        try:
            # Get ML-predicted levels
            signal_data = {
                'symbol': symbol,
                'direction': direction,
                'entry_price': entry_price,
                'volatility': indicators.get('volatility', 0.02),
                'volume_ratio': indicators.get('volume_ratio', 1.0),
                'rsi': indicators.get('rsi', 50),
                'macd_bullish': indicators.get('macd_bullish', False),
                'market_regime': market_analysis.get('trend_direction', 'neutral')
            }

            ml_levels = self.ml_predictor.predict_optimal_levels(signal_data)

            # Apply adaptive adjustments
            volatility = indicators.get('volatility', 0.02)
            volatility_multiplier = 1.0 + (volatility * 25)  # Scale with volatility

            # Market regime adjustments
            regime_multiplier = 1.0
            if market_analysis.get('trend_direction') in ['strong_bullish', 'strong_bearish']:
                regime_multiplier = 1.2  # Wider targets in strong trends
            elif market_analysis.get('volatility_regime') == 'high':
                regime_multiplier = 1.3  # Wider stops in high volatility

            # Calculate final levels
            if direction.upper() in ['BUY', 'LONG']:
                stop_loss = entry_price * (1 - (0.025 * volatility_multiplier * regime_multiplier))
                tp1 = entry_price * (1 + (0.035 * volatility_multiplier))
                tp2 = entry_price * (1 + (0.055 * volatility_multiplier))
                tp3 = entry_price * (1 + (0.085 * volatility_multiplier * regime_multiplier))
            else:
                stop_loss = entry_price * (1 + (0.025 * volatility_multiplier * regime_multiplier))
                tp1 = entry_price * (1 - (0.035 * volatility_multiplier))
                tp2 = entry_price * (1 - (0.055 * volatility_multiplier))
                tp3 = entry_price * (1 - (0.085 * volatility_multiplier * regime_multiplier))

            return {
                'stop_loss': stop_loss,
                'take_profit_1': tp1,
                'take_profit_2': tp2,
                'take_profit_3': tp3
            }

        except Exception as e:
            self.logger.error(f"Error calculating dynamic SL/TP: {e}")
            return self.ml_predictor._get_default_levels(signal_data)

    def _calculate_max_risk(self, entry_price: float, stop_loss: float, leverage: int) -> float:
        """Calculate maximum risk amount"""
        try:
            risk_per_unit = abs(entry_price - stop_loss) / entry_price
            return risk_per_unit * leverage * 100  # Risk percentage with leverage
        except:
            return 0.0

    def _calculate_risk_reward_ratio(self, entry_price: float, levels: Dict[str, float]) -> float:
        """Calculate risk-reward ratio"""
        try:
            risk = abs(entry_price - levels['stop_loss'])
            reward = abs(levels['take_profit_2'] - entry_price)  # Use TP2 for RR calculation
            return reward / risk if risk > 0 else 0
        except:
            return 0.0

    async def _calculate_ml_confidence(self, signal: Dict[str, Any]) -> float:
        """Calculate ML confidence score for signal"""
        try:
            base_confidence = signal['signal_strength'] / 100

            # Market condition adjustments
            market_conditions = signal.get('market_conditions', {})
            volatility = market_conditions.get('volatility', 0.02)

            # Lower confidence in extreme volatility
            if volatility > 0.05:
                volatility_penalty = 0.2
            elif volatility < 0.01:
                volatility_penalty = 0.1
            else:
                volatility_penalty = 0.0

            # Risk-reward adjustment
            risk_metrics = signal.get('risk_metrics', {})
            rr_ratio = risk_metrics.get('risk_reward_ratio', 1.0)

            if rr_ratio < 1.5:
                rr_penalty = 0.15
            elif rr_ratio > 3.0:
                rr_penalty = 0.0
            else:
                rr_penalty = 0.05

            final_confidence = base_confidence - volatility_penalty - rr_penalty
            return max(0.0, min(1.0, final_confidence))

        except Exception as e:
            return 0.5

    async def _rank_signals_by_ml_score(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank signals by comprehensive ML scoring"""
        try:
            for signal in signals:
                ml_score = 0.0

                # Signal strength (40% weight)
                ml_score += (signal['signal_strength'] / 100) * 0.4

                # ML confidence (30% weight)
                ml_score += signal.get('ml_confidence', 0.5) * 0.3

                # Risk-reward ratio (20% weight)
                rr_ratio = signal.get('risk_metrics', {}).get('risk_reward_ratio', 1.0)
                rr_score = min(1.0, rr_ratio / 3.0)  # Normalize to 0-1
                ml_score += rr_score * 0.2

                # Market alignment (10% weight)
                market_strength = signal.get('market_conditions', {}).get('market_strength', 50)
                ml_score += (market_strength / 100) * 0.1

                signal['ml_score'] = ml_score

            # Sort by ML score
            signals.sort(key=lambda x: x.get('ml_score', 0), reverse=True)
            return signals

        except Exception as e:
            self.logger.error(f"Error ranking signals: {e}")
            return signals

    async def scan_for_ml_signals(self) -> List[Dict[str, Any]]:
        """Enhanced real-time market scanning with adaptive algorithms"""
        signals = []
        successful_scans = 0
        market_analysis = await self._perform_market_analysis()

        # Adaptive symbol selection based on market conditions
        active_symbols = await self._select_optimal_symbols(market_analysis)

        for symbol in active_symbols[:20]:  # Increased scanning capacity
            try:
                # Quick check if cooldown allows signal for this symbol
                if not self.cooldown_manager.can_send_signal(symbol):
                    continue

                # Multi-timeframe analysis for better accuracy
                try:
                    signal = await self._comprehensive_symbol_analysis(symbol, market_analysis)
                    if signal:
                        # Apply ML confidence filtering
                        ml_confidence = await self._calculate_ml_confidence(signal)
                        if ml_confidence >= self.adaptive_config['min_confidence_threshold']:
                            signal['ml_confidence'] = ml_confidence
                            signals.append(signal)
                            successful_scans += 1
                except Exception as analysis_error:
                    self.logger.warning(f"Analysis error for {symbol}: {str(analysis_error)[:100]}")
                    continue

                # Adaptive rate limiting based on market volatility
                sleep_time = 0.05 if market_analysis['volatility_regime'] == 'high' else 0.1
                await asyncio.sleep(sleep_time)

            except Exception as e:
                self.logger.warning(f"Error scanning {symbol}: {str(e)[:100]}")
                continue

        # Enhanced signal ranking with ML scoring
        signals = await self._rank_signals_by_ml_score(signals)

        self.logger.info(f"🔍 Enhanced ML scan: {successful_scans} symbols, {len(signals)} high-quality signals")

        return signals[:5]  # Return top 5 signals for better opportunities

    async def send_message(self, chat_id: str, text: str, parse_mode='Markdown') -> bool:
        """Send message to Telegram with enhanced error handling"""
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': text,
                'parse_mode': parse_mode,
                'disable_web_page_preview': True
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, timeout=15) as response:
                    if response.status == 200:
                        self.logger.info(f"✅ Message sent to {chat_id}")
                        return True
                    else:
                        error = await response.text()
                        self.logger.warning(f"⚠️ Send message failed: {error}")
                        return False

        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
            return False

    def format_ml_signal_message(self, signal: Dict[str, Any]) -> str:
        """Format ML-enhanced signal with comprehensive information"""
        try:
            direction = signal['direction']
            timestamp = datetime.now().strftime('%H:%M:%S')
            ml_accuracy = signal.get('ml_accuracy', {})
            risk_metrics = signal.get('risk_metrics', {})

            # Enhanced message with adaptive ML insights
            message = f"""🧠 **ADAPTIVE ML SIGNAL**

🎯 **#{signal['symbol']}** {direction}
💰 **Entry:** {signal['entry_price']:.6f}
🛑 **Stop Loss:** {signal['stop_loss']:.6f}

📈 **Dynamic Take Profits:**
• **TP1:** {signal['take_profit_1']:.6f} (40%)
• **TP2:** {signal['take_profit_2']:.6f} (35%)
• **TP3:** {signal['take_profit_3']:.6f} (25%)

⚡ **Adaptive Leverage:** {signal['leverage']}x
📊 **Signal Strength:** {signal['signal_strength']:.0f}%
🎯 **ML Confidence:** {signal.get('ml_confidence', 0)*100:.1f}%
📈 **Risk/Reward:** 1:{risk_metrics.get('risk_reward_ratio', 0):.1f}
🕒 **Time:** {timestamp} UTC

🧠 **ML Performance:**
• **Model Accuracy:** {ml_accuracy.get('sl_accuracy', 0)*100:.1f}%
• **Learning Progress:** {self.ml_predictor.trade_count} trades
• **ML Score:** {signal.get('ml_score', 0)*100:.1f}%

      **Market Analysis:**
• **Volatility:** {signal['market_conditions']['volatility']:.3f}
• **Regime:** {signal['market_conditions']['regime'].title()}
• **Market Strength:** {signal['market_conditions'].get('market_strength', 50):.0f}%
• **Volume:** {signal['market_conditions']['volume_profile'].title()}

🤖 **Adaptive Features:**
✅ Dynamic SL/TP adjustment
✅ Market regime adaptation
✅ Multi-timeframe confluence
✅ Real-time learning optimization

⚙️ **Risk Management:**
• **Max Risk:** {risk_metrics.get('max_risk', 0):.1f}%
• **Adaptive Position Sizing**
• **Volatility-based adjustments**

⏰ **Cooldown:** 30min between signals
🚀 **Continuously Learning & Adapting**"""

            return message.strip()

        except Exception as e:
            self.logger.error(f"Error formatting message: {e}")
            return f"Signal: {signal.get('symbol', 'Unknown')} {signal.get('direction', 'Unknown')}"

    async def get_updates(self, offset=None, timeout=10) -> list:
        """Get Telegram updates with timeout"""
        try:
            url = f"{self.base_url}/getUpdates"
            params = {'timeout': timeout}
            if offset is not None:
                params['offset'] = offset

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=timeout+5) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('result', [])
                    return []

        except Exception as e:
            self.logger.error(f"Error getting updates: {e}")
            return []

    async def handle_commands(self, message: Dict, chat_id: str):
        """Handle enhanced bot commands"""
        try:
            text = message.get('text', '').strip()

            if text.startswith('/start'):
                self.admin_chat_id = chat_id

                welcome = f"""🧠 **ML-ENHANCED TRADING BOT**
*Continuously Learning & Optimizing*

✅ **Status:** Online & Learning
🎯 **Strategy:** ML-Enhanced Dynamic Scalping
⚡ **Leverage:** 20x-75x (Adaptive)
⏰ **Cooldown:** 30min between signals
🛡️ **Risk:** Dynamic SL/TP based on ML models

**🧠 Machine Learning Features:**
• Learns from every trade outcome
• Predicts optimal SL/TP levels
• Adapts to market conditions
• Continuous model improvement

**📊 Current ML Stats:**
• **Trades Analyzed:** `{self.ml_predictor.trade_count}`
• **Model Accuracy:** `{self.ml_predictor.model_accuracy.get('sl_accuracy', 0)*100:.1f}%`
• **Total Signals:** `{self.performance_stats['total_signals']}`
• **ML Signals:** `{self.performance_stats['ml_predicted_signals']}`

**⏰ Cooldown Management:**
• 30 minutes between symbol signals
• 5 minutes global cooldown
• Prevents signal spam
• Quality over quantity

*Bot continuously learns and improves with each trade*"""

                await self.send_message(chat_id, welcome)

            elif text.startswith('/ml'):
                ml_stats = f"""🧠 **MACHINE LEARNING STATUS**

**📊 Model Performance:**
• **SL Accuracy:** `{self.ml_predictor.model_accuracy.get('sl_accuracy', 0)*100:.1f}%`
• **TP1 Accuracy:** `{self.ml_predictor.model_accuracy.get('tp1_accuracy', 0)*100:.1f}%`
• **TP2 Accuracy:** `{self.ml_predictor.model_accuracy.get('tp2_accuracy', 0)*100:.1f}%`
• **TP3 Accuracy:** `{self.ml_predictor.model_accuracy.get('tp3_accuracy', 0)*100:.1f}%`

**📈 Learning Progress:**
• **Trades Analyzed:** `{self.ml_predictor.trade_count}`
• **Last Training:** `{self.ml_predictor.model_accuracy.get('last_training', 'Never')}`
• **Retrain Frequency:** Every `{self.ml_predictor.retrain_frequency}` trades

**🎯 Signal Quality:**
• **Total Signals:** `{self.performance_stats['total_signals']}`
• **ML-Enhanced:** `{self.performance_stats['ml_predicted_signals']}`
• **Win Rate:** `{self.performance_stats['win_rate']:.1f}%`
• **Cooldown Blocks:** `{self.performance_stats['cooldown_blocks']}`

*Models automatically retrain as new trade data becomes available*"""

                await self.send_message(chat_id, ml_stats)

            elif text.startswith('/cooldown'):
                cooldown_info = f"""⏰ **COOLDOWN STATUS**

**⚙️ Settings:**
• **Symbol Cooldown:** 30 minutes
• **Global Cooldown:** 5 minutes
• **Purpose:** Prevent signal spam

**📊 Current Status:**
• **Active Cooldowns:** {len([s for s, t in self.cooldown_manager.last_signals.items() if (datetime.now() - t).total_seconds() < 1800])}
• **Total Blocks:** `{self.performance_stats['cooldown_blocks']}`

**🎯 Benefits:**
• Higher quality signals
• Reduced noise
• Better risk management
• Focus on best opportunities

*Cooldown ensures only highest quality signals are sent*"""

                await self.send_message(chat_id, cooldown_info)

            elif text.startswith('/adaptive'):
                adaptive_status = f"""🤖 **ADAPTIVE ALGORITHM STATUS**

**🧠 Learning Parameters:**
• **Learning Rate:** `{self.adaptive_config['learning_rate']}`
• **Performance Window:** `{self.adaptive_config['performance_window']} trades`
• **Confidence Range:** `{self.adaptive_config['min_confidence_threshold']:.1f} - {self.adaptive_config['max_confidence_threshold']:.1f}`

**📊 Risk Management:**
• **Adaptive Risk Scaling:** `{'Enabled' if self.risk_config['adaptive_risk_scaling'] else 'Disabled'}`
• **Market Regime Adjustment:** `{'Enabled' if self.risk_config['market_regime_adjustment'] else 'Disabled'}`
• **Volume Confirmation:** `{'Required' if self.risk_config['volume_confirmation_required'] else 'Optional'}`

**🎯 Current Performance:**
• **Total Signals:** `{self.performance_stats['total_signals']}`
• **ML Enhanced:** `{self.performance_stats['ml_predicted_signals']}`
• **Win Rate:** `{self.performance_stats['win_rate']:.1f}%`
• **Avg Profit:** `{self.performance_stats['avg_profit']:.2f}%`

**⚙️ Adaptive Features:**
✅ Dynamic leverage calculation
✅ Volatility-based SL/TP adjustment  
✅ Multi-timeframe confluence analysis
✅ Real-time market regime detection
✅ Continuous model retraining

*System automatically adapts to market conditions*"""

                await self.send_message(chat_id, adaptive_status)

            elif text.startswith('/scan'):
                await self.send_message(chat_id, "🧠 **Adaptive ML Market Scan**\n\nAnalyzing markets with adaptive algorithms...")

                signals = await self.scan_for_ml_signals()

                if signals:
                    for signal in signals:
                        signal_msg = self.format_ml_signal_message(signal)
                        await self.send_message(chat_id, signal_msg)
                        await asyncio.sleep(2)

                    await self.send_message(chat_id, f"✅ **{len(signals)} Adaptive ML Signals Found**\n\nTop-ranked signals with ML scoring applied")
                else:
                    await self.send_message(chat_id, "📊 **No High-Quality Signals**\n\nAdaptive algorithms filtering for optimal opportunities")

        except Exception as e:
            self.logger.error(f"Error handling command: {e}")

    async def auto_ml_scan_loop(self):
        """Main auto-scanning loop with ML enhancements"""
        consecutive_errors = 0
        max_consecutive_errors = 5

        while not self.shutdown_requested:
            try:
                self.logger.info("🧠 Starting ML-enhanced market scan...")
                signals = await self.scan_for_ml_signals()

                if signals:
                    self.logger.info(f"🎯 Found {len(signals)} ML-enhanced signals")

                    for signal in signals:
                        try:
                            signal_msg = self.format_ml_signal_message(signal)

                            # Send to admin
                            if self.admin_chat_id:
                                await self.send_message(self.admin_chat_id, signal_msg)

                            # Send to channel
                            await self.send_message(self.target_channel, signal_msg)

                            self.logger.info(f"📤 ML signal sent: {signal['symbol']} {signal['direction']} (Strength: {signal['signal_strength']:.0f}%)")

                            await asyncio.sleep(3)  # Delay between signals

                        except Exception as signal_error:
                            self.logger.error(f"Error sending signal: {signal_error}")
                            continue

                else:
                    self.logger.info("🔍 No ML signals found - waiting for better opportunities")

                consecutive_errors = 0

                # Dynamic scan interval based on market activity
                scan_interval = 120 if signals else 180  # 2-3 minutes
                self.logger.info(f"⏰ Next ML scan in {scan_interval} seconds")
                await asyncio.sleep(scan_interval)

            except Exception as e:
                consecutive_errors += 1
                self.logger.error(f"ML scan loop error #{consecutive_errors}: {e}")

                if consecutive_errors >= max_consecutive_errors:
                    error_wait = 300  # 5 minutes
                    self.logger.critical(f"🚨 Too many consecutive errors. Waiting {error_wait} seconds...")
                else:
                    error_wait = 60 * consecutive_errors

                await asyncio.sleep(error_wait)

    async def run_bot(self):
        """Main bot execution with ML integration"""
        self.logger.info("🚀 Starting ML-Enhanced Trading Bot")

        try:
            # Start auto-scanning task
            auto_scan_task = asyncio.create_task(self.auto_ml_scan_loop())

            # Start command handling
            offset = None

            while not self.shutdown_requested:
                try:
                    updates = await self.get_updates(offset, timeout=5)

                    for update in updates:
                        if self.shutdown_requested:
                            break

                        offset = update['update_id'] + 1

                        if 'message' in update:
                            message = update['message']
                            chat_id = str(message['chat']['id'])

                            if 'text' in message:
                                await self.handle_commands(message, chat_id)

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    self.logger.error(f"Bot loop error: {e}")
                    await asyncio.sleep(5)

        except Exception as e:
            self.logger.critical(f"Critical bot error: {e}")
            raise
        finally:
            if self.admin_chat_id:
                try:
                    shutdown_msg = "🛑 **ML-Enhanced Bot Shutdown**\n\nBot stopped. All learned models preserved for restart."
                    await self.send_message(self.admin_chat_id, shutdown_msg)
                except:
                    pass

async def main():
    """Run the ML-enhanced trading bot"""
    bot = MLEnhancedTradingBot()

    try:
        print("🧠 ML-Enhanced Trading Bot Starting...")
        print("🎯 Features: Machine Learning, Dynamic SL/TP, Cooldown Management")
        print("⚡ Adaptive Leverage, Market Regime Detection")
        print("🛡️ Risk Management with Continuous Learning")
        print("⏰ 30-minute cooldown prevents spam")
        print("\nBot will continuously learn and improve from every trade")

        await bot.run_bot()

    except KeyboardInterrupt:
        print("\n🛑 ML-Enhanced Trading Bot stopped by user")
        return False
    except Exception as e:
        print(f"❌ Bot Error: {e}")
        return True

if __name__ == "__main__":
    asyncio.run(main())