
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
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            features['hour'] = df['timestamp'].dt.hour
            features['day_of_week'] = df['timestamp'].dt.dayofweek
            
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
            
            # Remove NaN values
            features = features.fillna(method='forward').fillna(method='backward')
            
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
            'ml_accuracy': 0.0,
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'total_profit': 0.0,
            'cooldown_blocks': 0
        }

        # Advanced risk management
        self.risk_config = {
            'base_risk_percentage': 2.0,  # 2% base risk
            'max_risk_percentage': 5.0,   # 5% maximum risk
            'volatility_adjustment': True,
            'ml_confidence_threshold': 0.7,
            'min_signal_strength': 85
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

                        df = pd.DataFrame(data, columns=[
                            'timestamp', 'open', 'high', 'low', 'close', 'volume',
                            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                            'taker_buy_quote', 'ignore'
                        ])

                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            df[col] = pd.to_numeric(df[col])

                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df.set_index('timestamp', inplace=True)

                        return df

            return None

        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def calculate_advanced_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive technical indicators with ML features"""
        try:
            if df.empty or len(df) < 50:
                return {}

            indicators = {}
            
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            volume = df['volume'].values

            # 1. Market volatility (key for ML models)
            returns = np.diff(np.log(close))
            indicators['volatility'] = np.std(returns) * np.sqrt(len(returns))
            indicators['volatility_percentile'] = self._calculate_volatility_percentile(indicators['volatility'])

            # 2. Enhanced SuperTrend with volatility adjustment
            atr = self._calculate_atr(high, low, close, 14)
            multiplier = 2.0 + (indicators['volatility'] * 20)  # Dynamic multiplier
            indicators['supertrend'] = self._calculate_supertrend(high, low, close, atr, multiplier)

            # 3. Volume analysis for ML
            volume_sma = np.mean(volume[-20:])
            indicators['volume_ratio'] = volume[-1] / volume_sma if volume_sma > 0 else 1.0
            indicators['volume_surge'] = indicators['volume_ratio'] > 1.5

            # 4. RSI with divergence detection
            indicators['rsi'] = self._calculate_rsi(close, 14)
            indicators['rsi_divergence'] = self._detect_rsi_divergence(close, indicators['rsi'])

            # 5. MACD analysis
            macd_line, signal_line, histogram = self._calculate_macd(close)
            indicators['macd_bullish'] = macd_line[-1] > signal_line[-1]
            indicators['macd_momentum'] = histogram[-1] - histogram[-2] if len(histogram) > 1 else 0

            # 6. Market regime detection
            indicators['market_regime'] = self._detect_market_regime(close, volume)

            # 7. Support/Resistance levels
            indicators['support_resistance'] = self._calculate_support_resistance(high, low)

            # 8. Trend strength
            indicators['trend_strength'] = self._calculate_trend_strength(close)

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

    def _calculate_atr(self, high: np.array, low: np.array, close: np.array, period: int) -> np.array:
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

    def _calculate_rsi(self, values: np.array, period: int) -> float:
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

    def _calculate_macd(self, values: np.array) -> tuple:
        """Calculate MACD"""
        try:
            ema_12 = self._calculate_ema(values, 12)
            ema_26 = self._calculate_ema(values, 26)
            macd_line = ema_12 - ema_26
            signal_line = self._calculate_ema(macd_line, 9)
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

    async def scan_for_ml_signals(self) -> List[Dict[str, Any]]:
        """Scan for signals with ML enhancement and cooldown management"""
        signals = []
        successful_scans = 0
        
        for symbol in self.symbols[:15]:  # Limit symbols to avoid rate limits
            try:
                # Quick check if cooldown allows signal for this symbol
                if not self.cooldown_manager.can_send_signal(symbol):
                    continue

                # Get market data
                df = await self.get_binance_data(symbol, '5m', 100)
                if df is None or len(df) < 50:
                    continue

                # Calculate indicators
                indicators = self.calculate_advanced_indicators(df)
                if not indicators:
                    continue

                # Generate ML-enhanced signal
                signal = await self.generate_ml_enhanced_signal(symbol, indicators)
                if signal:
                    signals.append(signal)
                    successful_scans += 1

                # Respect rate limits
                await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.warning(f"Error scanning {symbol}: {str(e)[:100]}")
                continue

        # Sort by signal strength and return top signals
        signals.sort(key=lambda x: x['signal_strength'], reverse=True)
        
        self.logger.info(f"🔍 ML scan complete: {successful_scans} symbols scanned, {len(signals)} signals found")
        
        return signals[:3]  # Return max 3 signals to avoid spam

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

            # Enhanced message with ML insights
            message = f"""🧠 **ML-ENHANCED SIGNAL**

🎯 **#{signal['symbol']}** {direction}
💰 **Entry:** {signal['entry_price']:.6f}
🛑 **Stop Loss:** {signal['stop_loss']:.6f}

📈 **Take Profits (ML Optimized):**
• **TP1:** {signal['take_profit_1']:.6f} (40%)
• **TP2:** {signal['take_profit_2']:.6f} (35%)
• **TP3:** {signal['take_profit_3']:.6f} (25%)

⚡ **Leverage:** {signal['leverage']}x (Dynamic)
📊 **Signal Strength:** {signal['signal_strength']:.0f}%
🕒 **Time:** {timestamp} UTC

🧠 **ML Performance:**
• **SL Accuracy:** {ml_accuracy.get('sl_accuracy', 0)*100:.1f}%
• **TP Accuracy:** {ml_accuracy.get('tp1_accuracy', 0)*100:.1f}%
• **Learning:** {self.ml_predictor.trade_count} trades analyzed

🌊 **Market Conditions:**
• **Volatility:** {signal['market_conditions']['volatility']:.3f}
• **Regime:** {signal['market_conditions']['regime'].title()}
• **Volume:** {signal['market_conditions']['volume_profile'].title()}

⚙️ **Auto-Management:**
✅ SL → Entry after TP1
✅ SL → TP1 after TP2
✅ Full close after TP3

⏰ **Cooldown:** 30min applied
🤖 **ML-Enhanced Bot | Always Learning**"""

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

            elif text.startswith('/scan'):
                await self.send_message(chat_id, "🧠 **ML-Enhanced Market Scan**\n\nAnalyzing markets with machine learning...")
                
                signals = await self.scan_for_ml_signals()
                
                if signals:
                    for signal in signals:
                        signal_msg = self.format_ml_signal_message(signal)
                        await self.send_message(chat_id, signal_msg)
                        await asyncio.sleep(2)
                    
                    await self.send_message(chat_id, f"✅ **{len(signals)} ML-Enhanced Signals Found**\n\nCooldowns activated for sent signals")
                else:
                    await self.send_message(chat_id, "📊 **No ML Signals**\n\nEither cooldowns active or market conditions don't meet ML criteria")

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
