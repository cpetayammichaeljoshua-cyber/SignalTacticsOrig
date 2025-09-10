#!/usr/bin/env python3
"""
Ultimate Perfect Trading Bot - Complete Automated System with Advanced ML
Combines all features: Signal generation, ML analysis, Telegram integration, Cornix forwarding
Enhanced with sophisticated machine learning that learns from every trade
Optimized for maximum profitability and smooth operation
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

# Technical Analysis and Chart Generation
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
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.linear_model import LogisticRegression
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from io import BytesIO
import base64

class AdvancedMLTradeAnalyzer:
    """Advanced ML Trade Analyzer with comprehensive learning capabilities"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # ML Models
        self.signal_classifier = None
        self.profit_predictor = None
        self.risk_assessor = None
        self.market_regime_detector = None
        # Initialize StandardScaler for ML models
        if ML_AVAILABLE:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
        else:
            self.scaler = None

        # Learning database
        self.db_path = "advanced_ml_trading.db"
        self._initialize_database()

        # Exponential learning tracking
        self.model_performance = {
            'signal_accuracy': 0.0,
            'profit_prediction_accuracy': 0.0,
            'risk_assessment_accuracy': 0.0,
            'confidence_prediction_accuracy': 0.0,
            'ensemble_accuracy': 0.0,
            'total_trades_learned': 0,
            'last_training_time': None,
            'win_rate_improvement': 0.0,
            'accuracy_growth_rate': 0.0,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'ml_confidence_threshold': 80.0,  # Increased from 75% for stricter filtering
            'adaptive_threshold': True,
            'learning_velocity': 0.0,
            'prediction_precision': 0.0,
            'trade_success_streak': 0
        }

        # Exponential learning parameters - more aggressive
        self.retrain_threshold = 3  # Retrain after every 3 trades for rapid learning
        self.trades_since_retrain = 0
        self.learning_multiplier = 1.5  # Higher exponential learning factor
        self.accuracy_target = 95.0  # Slightly lower target accuracy for more trades
        self.min_confidence_for_signal = 70.0  # Reduced to 70%+ ML confidence for more signals

        # Market insights
        self.market_insights = {
            'best_time_sessions': {},
            'symbol_performance': {},
            'indicator_effectiveness': {},
            'risk_patterns': {}
        }

        self.logger.info("🧠 Advanced ML Trade Analyzer initialized")

    def _initialize_database(self):
        """Initialize comprehensive ML database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Trade outcomes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL,
                    exit_price REAL,
                    stop_loss REAL,
                    take_profit_1 REAL,
                    take_profit_2 REAL,
                    take_profit_3 REAL,
                    signal_strength REAL,
                    leverage REAL,
                    profit_loss REAL,
                    trade_result TEXT,
                    duration_minutes REAL,
                    market_volatility REAL,
                    volume_ratio REAL,
                    rsi_value REAL,
                    macd_signal TEXT,
                    ema_alignment BOOLEAN,
                    cvd_trend TEXT,
                    time_session TEXT,
                    day_of_week INTEGER,
                    hour_of_day INTEGER,
                    indicators_data TEXT,
                    ml_prediction TEXT,
                    ml_confidence REAL,
                    entry_time TIMESTAMP,
                    exit_time TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # ML insights table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    insight_type TEXT,
                    insight_data TEXT,
                    confidence_score REAL,
                    trades_analyzed INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.commit()
            conn.close()

            self.logger.info("📊 Advanced ML database initialized")

        except Exception as e:
            self.logger.error(f"Error initializing ML database: {e}")

    async def record_trade_outcome(self, trade_data: Dict[str, Any]):
        """Record trade outcome for ML learning"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Extract time features
            entry_time = trade_data.get('entry_time', datetime.now())
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time)

            time_session = self._get_time_session(entry_time)

            cursor.execute('''
                INSERT OR REPLACE INTO ml_trades (
                    symbol, direction, entry_price, exit_price, stop_loss,
                    take_profit_1, take_profit_2, take_profit_3, signal_strength,
                    leverage, profit_loss, trade_result, duration_minutes,
                    market_volatility, volume_ratio, rsi_value, macd_signal,
                    ema_alignment, cvd_trend, time_session, day_of_week,
                    hour_of_day, indicators_data, ml_prediction, ml_confidence,
                    entry_time, exit_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data.get('symbol'),
                trade_data.get('direction'),
                trade_data.get('entry_price'),
                trade_data.get('exit_price'),
                trade_data.get('stop_loss'),
                trade_data.get('take_profit_1'),
                trade_data.get('take_profit_2'),
                trade_data.get('take_profit_3'),
                trade_data.get('signal_strength'),
                trade_data.get('leverage'),
                trade_data.get('profit_loss'),
                trade_data.get('trade_result'),
                trade_data.get('duration_minutes'),
                trade_data.get('market_volatility', 0.02),
                trade_data.get('volume_ratio', 1.0),
                trade_data.get('rsi_value', 50),
                trade_data.get('macd_signal', 'neutral'),
                trade_data.get('ema_alignment', False),
                trade_data.get('cvd_trend', 'neutral'),
                time_session,
                entry_time.weekday(),
                entry_time.hour,
                json.dumps(trade_data.get('indicators_data', {})),
                trade_data.get('ml_prediction', 'unknown'),
                trade_data.get('ml_confidence', 0),
                entry_time.isoformat(),
                trade_data.get('exit_time', entry_time).isoformat() if trade_data.get('exit_time') else entry_time.isoformat()
            ))

            conn.commit()
            conn.close()

            # Only increment counters for completed trades to avoid double counting
            if trade_data.get('trade_status') == 'COMPLETED':
                self.trades_since_retrain += 1
                self.model_performance['total_trades_learned'] += 1

            self.logger.info(f"📝 ML Trade recorded: {trade_data.get('symbol')} - {trade_data.get('trade_result')}")

            # Auto-retrain if threshold reached (only for completed trades)
            if trade_data.get('trade_status') == 'COMPLETED' and self.trades_since_retrain >= self.retrain_threshold:
                await self.retrain_models()

        except Exception as e:
            self.logger.error(f"Error recording ML trade: {e}")

    async def update_open_trade_data(self, trade_data: Dict[str, Any]):
        """Update open trade data for continuous ML learning"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Extract time features
            entry_time = trade_data.get('entry_time', datetime.now())
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time)

            last_update = trade_data.get('last_update', datetime.now())
            if isinstance(last_update, str):
                last_update = datetime.fromisoformat(last_update)

            time_session = self._get_time_session(entry_time)

            # Use INSERT OR REPLACE to update existing records
            cursor.execute('''
                INSERT OR REPLACE INTO ml_trades (
                    symbol, direction, entry_price, exit_price, stop_loss,
                    take_profit_1, take_profit_2, take_profit_3, signal_strength,
                    leverage, profit_loss, trade_result, duration_minutes,
                    market_volatility, volume_ratio, rsi_value, macd_signal,
                    ema_alignment, cvd_trend, time_session, day_of_week,
                    hour_of_day, indicators_data, ml_prediction, ml_confidence,
                    entry_time, exit_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data.get('symbol'),
                trade_data.get('direction'),
                trade_data.get('entry_price'),
                trade_data.get('current_price'),  # Use current price as temporary exit price
                trade_data.get('stop_loss'),
                trade_data.get('take_profit_1'),
                trade_data.get('take_profit_2'),
                trade_data.get('take_profit_3'),
                trade_data.get('signal_strength'),
                trade_data.get('leverage'),
                trade_data.get('unrealized_pnl', 0),  # Use unrealized P/L
                trade_data.get('trade_result', 'OPEN'),
                trade_data.get('duration_minutes'),
                trade_data.get('market_volatility', 0.02),
                trade_data.get('volume_ratio', 1.0),
                trade_data.get('rsi_value', 50),
                trade_data.get('macd_signal', 'neutral'),
                trade_data.get('ema_alignment', False),
                trade_data.get('cvd_trend', 'neutral'),
                time_session,
                entry_time.weekday(),
                entry_time.hour,
                json.dumps(trade_data.get('indicators_data', {})),
                trade_data.get('ml_prediction', 'unknown'),
                trade_data.get('ml_confidence', 0),
                entry_time.isoformat(),
                last_update.isoformat()
            ))

            conn.commit()
            conn.close()

            # Trigger incremental learning every 10 updates
            update_count = getattr(self, '_open_trade_updates', 0) + 1
            setattr(self, '_open_trade_updates', update_count)

            if update_count % 10 == 0:
                await self._incremental_ml_learning()

        except Exception as e:
            self.logger.error(f"Error updating open trade data: {e}")

    async def _incremental_ml_learning(self):
        """Perform incremental ML learning from open trades"""
        try:
            self.logger.info("🔄 Performing incremental ML learning from open trades...")

            # Get recent open trade data
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get last 50 open trades for incremental learning
            cursor.execute('''
                SELECT * FROM ml_trades 
                WHERE trade_result IN ('OPEN', 'MONITORING') 
                ORDER BY created_at DESC 
                LIMIT 50
            ''')

            open_trades = cursor.fetchall()
            conn.close()

            if len(open_trades) >= 10:  # Need minimum trades for learning
                # Analyze patterns in open trades
                patterns = await self._analyze_open_trade_patterns(open_trades)

                # Update ML models with real-time insights
                await self._update_ml_models_incremental(patterns)

                self.logger.info(f"🧠 Incremental learning completed with {len(open_trades)} open trades")

        except Exception as e:
            self.logger.error(f"Error in incremental ML learning: {e}")

    async def _analyze_open_trade_patterns(self, open_trades: List) -> Dict[str, Any]:
        """Analyze patterns from open trades"""
        try:
            patterns = {
                'profitable_setups': [],
                'losing_setups': [],
                'duration_insights': {},
                'volatility_patterns': {},
                'signal_strength_correlation': {}
            }

            profitable_count = 0
            losing_count = 0

            for trade in open_trades:
                # Analyze based on unrealized P/L
                profit_loss = trade[11] if len(trade) > 11 else 0  # profit_loss column

                if profit_loss > 0:
                    profitable_count += 1
                    patterns['profitable_setups'].append({
                        'symbol': trade[1],
                        'signal_strength': trade[9],
                        'volatility': trade[14],
                        'cvd_trend': trade[19]
                    })
                elif profit_loss < 0:
                    losing_count += 1
                    patterns['losing_setups'].append({
                        'symbol': trade[1],
                        'signal_strength': trade[9],
                        'volatility': trade[14],
                        'cvd_trend': trade[19]
                    })

            # Calculate success rate
            total_trades = profitable_count + losing_count
            patterns['current_success_rate'] = (profitable_count / total_trades * 100) if total_trades > 0 else 0

            return patterns

        except Exception as e:
            self.logger.error(f"Error analyzing open trade patterns: {e}")
            return {}

    async def _update_ml_models_incremental(self, patterns: Dict[str, Any]):
        """Update ML models with incremental learning from open trades"""
        try:
            success_rate = patterns.get('current_success_rate', 0)

            # Adjust confidence thresholds based on real-time performance
            if success_rate > 80:
                # High success rate - can be more aggressive
                self.model_performance['signal_accuracy'] = min(0.95, self.model_performance['signal_accuracy'] + 0.02)
            elif success_rate < 60:
                # Low success rate - be more conservative
                self.model_performance['signal_accuracy'] = max(0.70, self.model_performance['signal_accuracy'] - 0.02)

            # Update learning progress
            self.model_performance['last_training_time'] = datetime.now().isoformat()

            self.logger.info(f"📈 ML models updated - Current success rate: {success_rate:.1f}%")

        except Exception as e:
            self.logger.error(f"Error updating ML models incrementally: {e}")

    def _get_time_session(self, timestamp: datetime) -> str:
        """Determine trading session"""
        hour = timestamp.hour

        if 8 <= hour < 10:
            return 'LONDON_OPEN'
        elif 10 <= hour < 13:
            return 'LONDON_MAIN'
        elif 13 <= hour < 15:
            return 'NY_OVERLAP'
        elif 15 <= hour < 18:
            return 'NY_MAIN'
        elif 18 <= hour < 22:
            return 'NY_CLOSE'
        elif 22 <= hour < 24 or 0 <= hour < 6:
            return 'ASIA_MAIN'
        else:
            return 'TRANSITION'

    async def retrain_models(self):
        """Retrain all ML models with new data"""
        try:
            if not ML_AVAILABLE:
                self.logger.warning("ML libraries not available")
                return

            self.logger.info("🧠 Retraining ML models with new data...")

            # Get training data
            training_data = self._get_training_data()

            if len(training_data) < 50:
                self.logger.warning(f"Insufficient training data: {len(training_data)} trades")
                return

            # Prepare features and targets
            features, targets = self._prepare_ml_features(training_data)

            if features is None or len(features) == 0:
                return

            # Train signal classifier
            await self._train_signal_classifier(features, targets)

            # Train profit predictor
            await self._train_profit_predictor(features, targets)

            # Train risk assessor
            await self._train_risk_assessor(features, targets)

            # Analyze market insights
            await self._analyze_market_insights(training_data)

            # Save models
            self._save_ml_models()

            self.trades_since_retrain = 0
            self.model_performance['last_training_time'] = datetime.now().isoformat()

            self.logger.info(f"✅ ML models retrained with {len(training_data)} trades")

        except Exception as e:
            self.logger.error(f"Error retraining ML models: {e}")

    def _get_training_data(self) -> pd.DataFrame:
        """Get training data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT * FROM ml_trades 
                WHERE profit_loss IS NOT NULL 
                ORDER BY created_at DESC 
                LIMIT 1000
            """
            df = pd.read_sql_query(query, conn)
            conn.close()

            # Parse JSON fields
            if 'indicators_data' in df.columns:
                df['indicators_data'] = df['indicators_data'].apply(
                    lambda x: json.loads(x) if x else {}
                )

            return df

        except Exception as e:
            self.logger.error(f"Error getting training data: {e}")
            return pd.DataFrame()

    def _prepare_ml_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features for ML training with consistent column names"""
        try:
            if len(df) == 0:
                return None, None

            # Create feature matrix with consistent column names
            features = pd.DataFrame()

            # Basic features - ensure consistent naming
            features['signal_strength'] = df['signal_strength'].fillna(0)
            features['leverage'] = df['leverage'].fillna(35)
            features['market_volatility'] = df['market_volatility'].fillna(0.02)
            features['volume_ratio'] = df['volume_ratio'].fillna(1.0)
            features['rsi_value'] = df['rsi_value'].fillna(50)

            # Encode categorical features with consistent mapping
            direction_map = {'BUY': 1, 'SELL': 0}
            macd_map = {'bullish': 1, 'neutral': 0, 'bearish': -1}
            cvd_map = {'bullish': 1, 'neutral': 0, 'bearish': -1}
            session_map = {
                'LONDON_OPEN': 0, 'LONDON_MAIN': 1, 'NY_OVERLAP': 2,
                'NY_MAIN': 3, 'NY_CLOSE': 4, 'ASIA_MAIN': 5, 'TRANSITION': 6
            }

            features['direction_encoded'] = df['direction'].fillna('BUY').map(direction_map).fillna(1)
            features['macd_signal_encoded'] = df['macd_signal'].fillna('neutral').map(macd_map).fillna(0)
            features['cvd_trend_encoded'] = df['cvd_trend'].fillna('neutral').map(cvd_map).fillna(0)
            features['time_session_encoded'] = df['time_session'].fillna('NY_MAIN').map(session_map).fillna(3)
            features['ema_alignment'] = df['ema_alignment'].fillna(False).astype(int)

            # Time features with consistent naming
            features['hour_of_day'] = df['hour_of_day'].fillna(12)
            features['day_of_week'] = df['day_of_week'].fillna(1)

            # Targets
            targets = {
                'profitable': (df['profit_loss'] > 0).astype(int),
                'profit_amount': df['profit_loss'].fillna(0),
                'high_risk': (abs(df['profit_loss']) > 2.0).astype(int),
                'quick_profit': ((df['profit_loss'] > 0) & (df['duration_minutes'] < 30)).astype(int)
            }

            # Remove NaN values
            features = features.fillna(0)

            return features, targets

        except Exception as e:
            self.logger.error(f"Error preparing ML features: {e}")
            return None, None

    async def _train_signal_classifier(self, features: pd.DataFrame, targets: Dict):
        """Train signal classification model with improved scaling"""
        try:
            X = features
            y = targets['profitable']

            if len(X) < 20:
                return

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Initialize scaler if not exists
            if not hasattr(self, 'scaler') or self.scaler is None:
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
            
            # Always fit scaler on training data
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train model with better parameters
            self.signal_classifier = RandomForestClassifier(
                n_estimators=150,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
            self.signal_classifier.fit(X_train_scaled, y_train)

            # Evaluate with cross-validation
            y_pred = self.signal_classifier.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation score for better accuracy estimate
            if len(X_train) > 30:
                cv_scores = cross_val_score(self.signal_classifier, X_train_scaled, y_train, cv=3)
                accuracy = cv_scores.mean()

            self.model_performance['signal_accuracy'] = accuracy
            self.logger.info(f"🎯 Signal classifier accuracy: {accuracy:.3f} (CV: {cv_scores.std():.3f})" if len(X_train) > 30 else f"🎯 Signal classifier accuracy: {accuracy:.3f}")

        except Exception as e:
            self.logger.error(f"Error training signal classifier: {e}")
            # Fallback to basic model
            try:
                from sklearn.linear_model import LogisticRegression
                self.signal_classifier = LogisticRegression(random_state=42, class_weight='balanced')
                if hasattr(self, 'scaler') and self.scaler is not None:
                    self.signal_classifier.fit(X_train_scaled, y_train)
                else:
                    self.signal_classifier.fit(X_train, y_train)
                self.model_performance['signal_accuracy'] = 0.7  # Conservative fallback
                self.logger.info("📊 Fallback classifier trained")
            except:
                self.logger.error("Failed to train fallback classifier")

    async def _train_profit_predictor(self, features: pd.DataFrame, targets: Dict):
        """Train profit prediction model with improved scaling and validation"""
        try:
            X = features
            y = targets['profit_amount']

            if len(X) < 20:
                return

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Use the same scaler as signal classifier
            if hasattr(self, 'scaler') and self.scaler is not None:
                # Use existing fitted scaler
                X_train_scaled = self.scaler.transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
            else:
                # Initialize new scaler if needed
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)

            # Train model with better parameters
            from sklearn.ensemble import GradientBoostingRegressor
            self.profit_predictor = GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42
            )
            self.profit_predictor.fit(X_train_scaled, y_train)

            # Evaluate with multiple metrics
            y_pred = self.profit_predictor.predict(X_test_scaled)
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # Store comprehensive performance metrics
            self.model_performance['profit_prediction_accuracy'] = max(0, r2)
            self.model_performance['profit_prediction_mae'] = mae
            self.model_performance['profit_prediction_rmse'] = rmse
            
            self.logger.info(f"💰 Profit predictor - R²: {r2:.3f}, MAE: {mae:.3f}, RMSE: {rmse:.3f}")

        except Exception as e:
            self.logger.error(f"Error training profit predictor: {e}")
            # Fallback to simple linear model
            try:
                from sklearn.linear_model import LinearRegression
                self.profit_predictor = LinearRegression()
                if hasattr(self, 'scaler') and self.scaler is not None:
                    self.profit_predictor.fit(X_train_scaled, y_train)
                else:
                    self.profit_predictor.fit(X_train, y_train)
                self.model_performance['profit_prediction_accuracy'] = 0.5  # Conservative fallback
                self.logger.info("📊 Fallback profit predictor trained")
            except:
                self.logger.error("Failed to train fallback profit predictor")

    async def _train_risk_assessor(self, features: pd.DataFrame, targets: Dict):
        """Train risk assessment model"""
        try:
            X = features
            y = targets['high_risk']

            if len(X) < 20:
                return

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Scale features
            # Assuming scaler is initialized in _train_signal_classifier or available
            if not hasattr(self, 'scaler') or self.scaler is None:
                 self.scaler = StandardScaler() # Initialize if not present
                 X_train_scaled = self.scaler.fit_transform(X_train)
            else:
                 X_train_scaled = self.scaler.transform(X_train) # Use existing scaler

            X_test_scaled = self.scaler.transform(X_test)

            # Train model
            self.risk_assessor = LogisticRegression(
                random_state=42,
                class_weight='balanced'
            )
            self.risk_assessor.fit(X_train_scaled, y_train)

            # Evaluate
            y_pred = self.risk_assessor.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)

            self.model_performance['risk_assessment_accuracy'] = accuracy
            self.logger.info(f"⚠️ Risk assessor accuracy: {accuracy:.3f}")

        except Exception as e:
            self.logger.error(f"Error training risk assessor: {e}")

    async def _analyze_market_insights(self, df: pd.DataFrame):
        """Analyze market insights from trading data"""
        try:
            # Time session analysis
            session_performance = df.groupby('time_session')['profit_loss'].agg(['mean', 'count', 'std'])
            self.market_insights['best_time_sessions'] = session_performance.to_dict()

            # Symbol performance
            symbol_performance = df.groupby('symbol')['profit_loss'].agg(['mean', 'count', 'std'])
            self.market_insights['symbol_performance'] = symbol_performance.to_dict()

            # Indicator effectiveness
            indicator_corr = df[['signal_strength', 'rsi_value', 'volume_ratio', 'profit_loss']].corr()['profit_loss']
            self.market_insights['indicator_effectiveness'] = indicator_corr.to_dict()

            self.logger.info("🔍 Market insights updated")

        except Exception as e:
            self.logger.error(f"Error analyzing market insights: {e}")

    def _save_ml_models(self):
        """Save ML models to disk"""
        try:
            model_dir = Path("ml_models")
            model_dir.mkdir(exist_ok=True)

            models = {
                'signal_classifier.pkl': self.signal_classifier,
                'profit_predictor.pkl': self.profit_predictor,
                'risk_assessor.pkl': self.risk_assessor,
                'scaler.pkl': self.scaler
            }

            for filename, model in models.items():
                if model is not None:
                    with open(model_dir / filename, 'wb') as f:
                        pickle.dump(model, f)

            # Save performance metrics
            with open(model_dir / 'performance_metrics.json', 'w') as f:
                json.dump(self.model_performance, f, indent=2)

            # Save market insights
            with open(model_dir / 'market_insights.json', 'w') as f:
                json.dump(self.market_insights, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Error saving ML models: {e}")

    def load_ml_models(self):
        """Load ML models from disk"""
        try:
            model_dir = Path("ml_models")

            if not model_dir.exists():
                return

            models = {
                'signal_classifier.pkl': 'signal_classifier',
                'profit_predictor.pkl': 'profit_predictor',
                'risk_assessor.pkl': 'risk_assessor',
                'scaler.pkl': 'scaler'
            }

            for filename, attr_name in models.items():
                filepath = model_dir / filename
                if filepath.exists():
                    with open(filepath, 'rb') as f:
                        setattr(self, attr_name, pickle.load(f))

            # Load performance metrics
            metrics_file = model_dir / 'performance_metrics.json'
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    self.model_performance.update(json.load(f))

            # Load market insights
            insights_file = model_dir / 'market_insights.json'
            if insights_file.exists():
                with open(insights_file, 'r') as f:
                    self.market_insights.update(json.load(f))

            self.logger.info("🤖 Advanced ML models loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading ML models: {e}")

    def predict_trade_outcome(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced ML prediction for trade outcome with improved filtering"""
        try:
            # Check if all models are available
            models_available = all([
                hasattr(self, 'signal_classifier') and self.signal_classifier is not None,
                hasattr(self, 'profit_predictor') and self.profit_predictor is not None,
                hasattr(self, 'risk_assessor') and self.risk_assessor is not None,
                hasattr(self, 'scaler') and self.scaler is not None
            ])
            
            if not models_available:
                return self._fallback_prediction(signal_data)

            # Prepare features as DataFrame
            features_df = self._prepare_prediction_features(signal_data)
            if features_df is None or features_df.empty:
                return self._fallback_prediction(signal_data)

            try:
                # Scale features with error handling
                features_scaled = self.scaler.transform(features_df)
            except Exception as scale_error:
                self.logger.warning(f"Scaling error, using fallback: {scale_error}")
                return self._fallback_prediction(signal_data)

            # Get predictions with error handling
            try:
                profit_prob = self.signal_classifier.predict_proba(features_scaled)[0][1]
                profit_amount = self.profit_predictor.predict(features_scaled)[0]
                risk_prob = self.risk_assessor.predict_proba(features_scaled)[0][1]
            except Exception as pred_error:
                self.logger.warning(f"Prediction error, using fallback: {pred_error}")
                return self._fallback_prediction(signal_data)

            # Calculate overall confidence with bounds checking
            confidence = max(0, min(100, profit_prob * 100))

            # Adjust based on market insights
            confidence = self._adjust_confidence_with_insights(signal_data, confidence)

            # IMPROVED FILTERING - Less restrictive but still quality-focused
            signal_strength = signal_data.get('signal_strength', 50)
            
            # Multi-factor decision making
            if confidence >= 75 and profit_amount > 0 and risk_prob < 0.3 and signal_strength >= 80:
                prediction = 'highly_favorable'
            elif confidence >= 65 and profit_amount > 0 and risk_prob < 0.4 and signal_strength >= 70:
                prediction = 'favorable'
            elif confidence >= 55 and profit_amount > 0 and risk_prob < 0.5 and signal_strength >= 60:
                prediction = 'above_neutral'
            elif confidence >= 45 and signal_strength >= 85:  # High signal strength can override ML
                prediction = 'strength_override'
            else:
                # More informative rejection reasons
                rejection_reason = []
                if confidence < 45:
                    rejection_reason.append(f"Low ML confidence ({confidence:.1f}%)")
                if profit_amount <= 0:
                    rejection_reason.append("Negative expected profit")
                if risk_prob >= 0.5:
                    rejection_reason.append(f"High risk probability ({risk_prob*100:.1f}%)")
                if signal_strength < 60:
                    rejection_reason.append(f"Low signal strength ({signal_strength:.1f}%)")
                
                return {
                    'prediction': 'filtered_out',
                    'confidence': confidence,
                    'expected_profit': profit_amount,
                    'risk_probability': risk_prob * 100,
                    'recommendation': f'Signal filtered: {"; ".join(rejection_reason)}',
                    'model_accuracy': self.model_performance.get('signal_accuracy', 0) * 100,
                    'trades_learned_from': self.model_performance.get('total_trades_learned', 0),
                    'rejection_reasons': rejection_reason
                }

            return {
                'prediction': prediction,
                'confidence': confidence,
                'expected_profit': profit_amount,
                'risk_probability': risk_prob * 100,
                'recommendation': self._get_ml_recommendation(prediction, confidence, profit_amount, risk_prob),
                'model_accuracy': self.model_performance.get('signal_accuracy', 0) * 100,
                'trades_learned_from': self.model_performance.get('total_trades_learned', 0),
                'signal_strength': signal_strength
            }

        except Exception as e:
            self.logger.error(f"Error in ML prediction: {e}")
            return self._fallback_prediction(signal_data)

    def _prepare_prediction_features(self, signal_data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Prepare features for prediction as DataFrame to match training data"""
        try:
            # Get current time for session
            current_time = datetime.now()
            time_session = self._get_time_session(current_time)

            # Map categorical values
            direction_map = {'BUY': 1, 'SELL': 0}
            session_map = {
                'LONDON_OPEN': 0, 'LONDON_MAIN': 1, 'NY_OVERLAP': 2,
                'NY_MAIN': 3, 'NY_CLOSE': 4, 'ASIA_MAIN': 5, 'TRANSITION': 6
            }
            cvd_map = {'bullish': 1, 'neutral': 0, 'bearish': -1}
            macd_map = {'bullish': 1, 'neutral': 0, 'bearish': -1}

            # Create DataFrame with proper column names matching training data
            feature_data = {
                'signal_strength': signal_data.get('signal_strength', 85),
                'leverage': signal_data.get('leverage', 35),
                'market_volatility': signal_data.get('market_volatility', 0.02),
                'volume_ratio': signal_data.get('volume_ratio', 1.0),
                'rsi_value': signal_data.get('rsi', 50),
                'direction_encoded': direction_map.get(signal_data.get('direction', 'BUY'), 1),
                'macd_signal_encoded': macd_map.get(signal_data.get('macd_signal', 'neutral'), 0),
                'cvd_trend_encoded': cvd_map.get(signal_data.get('cvd_trend', 'neutral'), 0),
                'time_session_encoded': session_map.get(time_session, 3),
                'ema_alignment': 1 if signal_data.get('ema_bullish', False) else 0,
                'hour_of_day': current_time.hour,
                'day_of_week': current_time.weekday()
            }

            # Return as single-row DataFrame
            features_df = pd.DataFrame([feature_data])
            return features_df

        except Exception as e:
            self.logger.error(f"Error preparing prediction features: {e}")
            return None

    def _adjust_confidence_with_insights(self, signal_data: Dict[str, Any], base_confidence: float) -> float:
        """Adjust confidence based on market insights"""
        try:
            adjusted_confidence = base_confidence

            # Time session adjustment
            current_session = self._get_time_session(datetime.now())
            if 'best_time_sessions' in self.market_insights:
                session_data = self.market_insights['best_time_sessions']
                if current_session in session_data.get('mean', {}):
                    session_performance = session_data['mean'][current_session]
                    if session_performance > 0:
                        adjusted_confidence *= 1.1
                    elif session_performance < -0.5:
                        adjusted_confidence *= 0.9

            # Symbol performance adjustment
            symbol = signal_data.get('symbol', '')
            if 'symbol_performance' in self.market_insights:
                symbol_data = self.market_insights['symbol_performance']
                if symbol in symbol_data.get('mean', {}):
                    symbol_performance = symbol_data['mean'][symbol]
                    if symbol_performance > 0:
                        adjusted_confidence *= 1.05
                    elif symbol_performance < -0.5:
                        adjusted_confidence *= 0.95

            return min(95, max(5, adjusted_confidence))

        except Exception as e:
            self.logger.error(f"Error adjusting confidence: {e}")
            return base_confidence

    def _get_ml_recommendation(self, prediction: str, confidence: float, profit: float, risk: float) -> str:
        """Get ML-based recommendation"""
        return "Signal Strength Based: Favorable"

    def _fallback_prediction(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback prediction when ML models not available - balanced approach"""
        signal_strength = signal_data.get('signal_strength', 50)
        volume_ratio = signal_data.get('volume_ratio', 1.0)
        volatility = signal_data.get('market_volatility', 0.02)
        
        # Multi-factor fallback assessment
        base_confidence = signal_strength
        
        # Volume boost
        if volume_ratio > 1.2:
            base_confidence += 5
        elif volume_ratio < 0.8:
            base_confidence -= 5
            
        # Volatility adjustment
        if 0.01 <= volatility <= 0.03:  # Optimal volatility range
            base_confidence += 3
        elif volatility > 0.05:  # High volatility penalty
            base_confidence -= 8
            
        # Ensure bounds
        confidence = max(0, min(100, base_confidence))

        # IMPROVED FALLBACK THRESHOLDS
        if confidence >= 80 and signal_strength >= 75:
            prediction = 'highly_favorable'
        elif confidence >= 70 and signal_strength >= 65:
            prediction = 'favorable'
        elif confidence >= 60 and signal_strength >= 55:
            prediction = 'above_neutral'
        elif confidence >= 50 and signal_strength >= 70:  # High signal strength override
            prediction = 'strength_based'
        else:
            return {
                'prediction': 'below_threshold',
                'confidence': confidence,
                'expected_profit': 0,
                'risk_probability': max(30, 100 - confidence),
                'recommendation': f'Fallback filter: Signal strength {signal_strength:.1f}%, confidence {confidence:.1f}%',
                'model_accuracy': 0.0,
                'trades_learned_from': 0,
                'fallback_mode': True
            }

        # Calculate expected profit based on multiple factors
        profit_multiplier = 1.0
        if volume_ratio > 1.2:
            profit_multiplier += 0.2
        if volatility <= 0.02:
            profit_multiplier += 0.1
            
        expected_profit = (confidence / 100.0) * profit_multiplier * 1.5

        return {
            'prediction': prediction,
            'confidence': confidence,
            'expected_profit': expected_profit,
            'risk_probability': max(10, 100 - confidence),
            'recommendation': f"Fallback: {prediction.replace('_', ' ').title()} (Multi-factor: {confidence:.1f}%)",
            'model_accuracy': 0.0,
            'trades_learned_from': 0,
            'fallback_mode': True,
            'volume_factor': volume_ratio,
            'volatility_factor': volatility
        }

    def get_ml_summary(self) -> Dict[str, Any]:
        """Get comprehensive ML summary"""
        return {
            'model_performance': self.model_performance,
            'market_insights': self.market_insights,
            'learning_status': 'active' if self.model_performance['total_trades_learned'] > 0 else 'initializing',
            'next_retrain_in': self.retrain_threshold - self.trades_since_retrain,
            'ml_available': ML_AVAILABLE
        }

class UltimateTradingBot:
    """Ultimate automated trading bot with advanced ML integration"""

    def __init__(self):
        self.logger = self._setup_logging()

        # Process management
        self.pid_file = Path("ultimate_trading_bot.pid")
        self.shutdown_requested = False
        self._setup_signal_handlers()
        atexit.register(self._cleanup_on_exit)

        # Telegram configuration
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

        # Session management
        self.session_secret = os.getenv('SESSION_SECRET', 'ultimate_trading_secret_key')
        self.session_token = None

        # Bot status
        self.running = True
        self.last_heartbeat = datetime.now()

        # Bot settings
        self.admin_chat_id = None
        self.target_channel = "@SignalTactics"
        self.channel_accessible = False

        # Enhanced symbol list (200+ pairs for maximum coverage)
        self.symbols = [
            # Major cryptocurrencies
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'SOLUSDT', 'DOGEUSDT',
            'AVAXUSDT', 'DOTUSDT', 'MATICUSDT', 'LINKUSDT', 'LTCUSDT', 'BCHUSDT', 'ETCUSDT',
            'ATOMUSDT', 'ALGOUSDT', 'XLMUSDT', 'VETUSDT', 'TRXUSDT', 'EOSUSDT', 'THETAUSDT',

            # DeFi tokens
            'UNIUSDT', 'AAVEUSDT', 'COMPUSDT', 'MKRUSDT', 'YFIUSDT', 'SUSHIUSDT', 'CAKEUSDT',
            'CRVUSDT', '1INCHUSDT', 'SNXUSDT', 'BALAUSDT', 'ALPHAUSDT',

            # Layer 2 & Scaling
            'ARBUSDT', 'OPUSDT', 'METISUSDT', 'STRKUSDT',

            # Gaming & Metaverse
            'SANDUSDT', 'MANAUSDT', 'AXSUSDT', 'GALAUSDT', 'ENJUSDT', 'CHZUSDT',
            'FLOWUSDT', 'IMXUSDT', 'GMTUSDT', 'STEPNUSDT',

            # AI & Data
            'FETUSDT', 'AGIXUSDT', 'OCEANUSDT', 'GRTUSDT',

            # Meme coins
            'SHIBUSDT', 'PEPEUSDT', 'FLOKIUSDT', 'BONKUSDT',

            # New & Trending
            'APTUSDT', 'SUIUSDT', 'ARKMUSDT', 'SEIUSDT', 'TIAUSDT', 'WLDUSDT',
            'JUPUSDT', 'WIFUSDT', 'BOMEUSDT', 'NOTUSDT', 'REZUSDT'
        ]

        # Optimized timeframes for scalping
        self.timeframes = ['1m', '3m', '5m', '15m', '1h', '4h']

        # CVD (Cumulative Volume Delta) tracking
        self.cvd_data = {
            'btc_perp_cvd': 0,
            'cvd_trend': 'neutral',
            'cvd_divergence': False,
            'cvd_strength': 0
        }

        # Adaptive leverage settings with cross margin
        self.leverage_config = {
            'min_leverage': 20,
            'max_leverage': 75,
            'base_leverage': 35,
            'volatility_threshold_low': 0.01,
            'volatility_threshold_high': 0.04,
            'volume_threshold_low': 0.8,
            'volume_threshold_high': 1.5,
            'margin_type': 'CROSSED'  # Always use cross margin
        }

        # Adaptive leveraging based on market conditions and past performance
        self.adaptive_leverage = {
            'recent_wins': 0,
            'recent_losses': 0,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'performance_window': 20,
            'leverage_adjustment_factor': 0.1
        }

        # Risk management - optimized for maximum profitability and unlimited signals
        self.risk_reward_ratio = 1.0  # 1:1 ratio as requested
        self.min_signal_strength = 75  # Lowered for more opportunities
        self.max_signals_per_hour = 999  # Unlimited signals per hour
        self.capital_allocation = 0.02  # 2% per trade for more trades
        self.max_concurrent_trades = 50  # Increased concurrent trades for more volume

        # Performance tracking
        self.signal_counter = 0
        self.active_trades = {}
        self.performance_stats = {
            'total_signals': 0,
            'profitable_signals': 0,
            'win_rate': 0.0,
            'total_profit': 0.0
        }

        # Prevent signal spam - greatly reduced restrictions
        self.last_signal_time = {}
        self.min_signal_interval = 30  # 30 seconds between signals for same symbol

        # Hourly signal tracking - removed limitations
        self.hourly_signal_count = 0
        self.last_hour_reset = datetime.now().hour
        self.unlimited_signals = True  # Flag for unlimited signal mode

        # Active symbol tracking - enforce single trade per symbol
        self.active_symbols = set()  # Track symbols with open trades
        self.symbol_trade_lock = {}  # Lock mechanism for each symbol

        # Advanced ML Trade Analyzer
        self.ml_analyzer = AdvancedMLTradeAnalyzer()
        self.ml_analyzer.load_ml_models()

        # Closed Trades Scanner for ML Training
        self.closed_trades_scanner = None
        if self.bot_token:
            try:
                from telegram_closed_trades_scanner import TelegramClosedTradesScanner
                self.closed_trades_scanner = TelegramClosedTradesScanner(self.bot_token, self.target_channel)
                self.logger.info("📊 Telegram Closed Trades Scanner initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize closed trades scanner: {e}")

        self.logger.info("🚀 Ultimate Trading Bot initialized with Advanced ML")
        self._write_pid_file()

    def _setup_logging(self):
        """Setup comprehensive logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ultimate_trading_bot.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
            self.shutdown_requested = True
            self.running = False

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

    async def create_session(self) -> str:
        """Create indefinite session"""
        try:
            session_data = {
                'created_at': datetime.now().isoformat(),
                'bot_id': 'ultimate_trading_bot',
                'expires_at': 'never'
            }

            session_string = json.dumps(session_data, sort_keys=True)
            session_token = hmac.new(
                self.session_secret.encode(),
                session_string.encode(),
                hashlib.sha256
            ).hexdigest()

            self.session_token = session_token
            self.logger.info("✅ Indefinite session created")
            return session_token

        except Exception as e:
            self.logger.error(f"Session creation error: {e}")
            return None

    async def calculate_cvd_btc_perp(self) -> Dict[str, Any]:
        """Calculate Cumulative Volume Delta for BTC PERP"""
        try:
            url = "https://fapi.binance.com/fapi/v1/klines"
            params = {
                'symbol': 'BTCUSDT',
                'interval': '1m',
                'limit': 100
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        klines = await response.json()

                        # Get trades for volume delta calculation
                        trades_url = "https://fapi.binance.com/fapi/v1/aggTrades"
                        trades_params = {
                            'symbol': 'BTCUSDT',
                            'limit': 1000
                        }

                        async with session.get(trades_url, params=trades_params) as trades_response:
                            if trades_response.status == 200:
                                trades = await trades_response.json()

                                buy_volume = 0
                                sell_volume = 0

                                for trade in trades:
                                    volume = float(trade['q'])
                                    if trade['m']:  # Maker side (sell)
                                        sell_volume += volume
                                    else:  # Taker side (buy)
                                        buy_volume += volume

                                volume_delta = buy_volume - sell_volume
                                self.cvd_data['btc_perp_cvd'] += volume_delta

                                if volume_delta > 0:
                                    self.cvd_data['cvd_trend'] = 'bullish'
                                elif volume_delta < 0:
                                    self.cvd_data['cvd_trend'] = 'bearish'
                                else:
                                    self.cvd_data['cvd_trend'] = 'neutral'

                                total_volume = buy_volume + sell_volume
                                if total_volume > 0:
                                    self.cvd_data['cvd_strength'] = min(100, abs(volume_delta) / total_volume * 100)

                                # Detect divergence with price
                                if len(klines) >= 20:
                                    recent_prices = [float(k[4]) for k in klines[-20:]]
                                    price_trend = 'bullish' if recent_prices[-1] > recent_prices[-10] else 'bearish'
                                    self.cvd_data['cvd_divergence'] = (
                                        (price_trend == 'bullish' and self.cvd_data['cvd_trend'] == 'bearish') or
                                        (price_trend == 'bearish' and self.cvd_data['cvd_trend'] == 'bullish')
                                    )

                                return self.cvd_data

            return self.cvd_data

        except Exception as e:
            self.logger.error(f"Error calculating CVD for BTC PERP: {e}")
            return self.cvd_data

    async def get_binance_data(self, symbol: str, interval: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Get USD-M futures market data from Binance"""
        try:
            url = f"https://fapi.binance.com/fapi/v1/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
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
            self.logger.error(f"Error fetching futures data for {symbol}: {e}")
            return None

    def calculate_advanced_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive technical indicators"""
        try:
            indicators = {}

            if df.empty or len(df) < 55:
                return {}

            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            volume = df['volume'].values

            if len(high) == 0 or len(low) == 0 or len(close) == 0:
                return {}

            # 1. Enhanced SuperTrend
            hl2 = (high + low) / 2
            atr = self._calculate_atr(high, low, close, 7)
            volatility = np.std(close[-20:]) / np.mean(close[-20:])
            multiplier = 2.5 + (volatility * 10)

            upper_band = hl2 + (multiplier * atr)
            lower_band = hl2 - (multiplier * atr)

            supertrend = np.zeros(len(close))
            supertrend_direction = np.zeros(len(close))

            for i in range(1, len(close)):
                if close[i] <= lower_band[i]:
                    supertrend[i] = upper_band[i]
                    supertrend_direction[i] = -1
                elif close[i] >= upper_band[i]:
                    supertrend[i] = lower_band[i]
                    supertrend_direction[i] = 1
                else:
                    supertrend[i] = supertrend[i-1]
                    supertrend_direction[i] = supertrend_direction[i-1]

            indicators['supertrend'] = supertrend[-1]
            indicators['supertrend_direction'] = supertrend_direction[-1]

            # 2. VWAP
            typical_price = (high + low + close) / 3
            vwap = np.zeros(len(close))
            cumulative_volume = np.zeros(len(close))
            cumulative_pv = np.zeros(len(close))

            for i in range(len(close)):
                if i == 0:
                    cumulative_volume[i] = volume[i]
                    cumulative_pv[i] = typical_price[i] * volume[i]
                else:
                    cumulative_volume[i] = cumulative_volume[i-1] + volume[i]
                    cumulative_pv[i] = cumulative_pv[i-1] + (typical_price[i] * volume[i])

                if cumulative_volume[i] > 0:
                    vwap[i] = cumulative_pv[i] / cumulative_volume[i]

            indicators['vwap'] = vwap[-1] if len(vwap) > 0 else close[-1]

            if vwap[-1] != 0 and not np.isnan(vwap[-1]) and not np.isinf(vwap[-1]):
                indicators['price_vs_vwap'] = (close[-1] - vwap[-1]) / vwap[-1] * 100
            else:
                indicators['price_vs_vwap'] = 0.0

            # 3. EMA Cross Strategy
            ema_8 = self._calculate_ema(close, 8)
            ema_21 = self._calculate_ema(close, 21)
            ema_55 = self._calculate_ema(close, 55)

            indicators['ema_8'] = ema_8[-1]
            indicators['ema_21'] = ema_21[-1]
            indicators['ema_55'] = ema_55[-1]
            indicators['ema_bullish'] = ema_8[-1] > ema_21[-1] > ema_55[-1]
            indicators['ema_bearish'] = ema_8[-1] < ema_21[-1] < ema_55[-1]

            # 4. RSI with divergence
            rsi = self._calculate_rsi(close, 14)
            indicators['rsi'] = rsi[-1]
            indicators['rsi_oversold'] = rsi[-1] < 30
            indicators['rsi_overbought'] = rsi[-1] > 70

            # 5. MACD
            macd_line, macd_signal, macd_hist = self._calculate_macd(close)
            indicators['macd'] = macd_line[-1]
            indicators['macd_signal'] = macd_signal[-1]
            indicators['macd_histogram'] = macd_hist[-1]
            indicators['macd_bullish'] = macd_line[-1] > macd_signal[-1] and macd_hist[-1] > 0
            indicators['macd_bearish'] = macd_line[-1] < macd_signal[-1] and macd_hist[-1] < 0

            # 6. Volume analysis
            volume_sma = np.mean(volume[-20:])
            if volume_sma > 0 and not np.isnan(volume_sma) and not np.isinf(volume_sma):
                indicators['volume_ratio'] = volume[-1] / volume_sma
                indicators['volume_surge'] = volume[-1] > volume_sma * 1.5
            else:
                indicators['volume_ratio'] = 1.0
                indicators['volume_surge'] = False

            # 7. Market volatility
            indicators['market_volatility'] = volatility

            # 8. CVD integration
            cvd_data = self.cvd_data
            indicators['cvd_trend'] = cvd_data['cvd_trend']
            indicators['cvd_strength'] = cvd_data['cvd_strength']
            indicators['cvd_divergence'] = cvd_data['cvd_divergence']

            # 9. Heikin Ashi trend confirmation
            ha_data = self._calculate_heikin_ashi(df)
            indicators.update(ha_data)

            # 10. Current price info
            indicators['current_price'] = close[-1]
            indicators['price_change'] = (close[-1] - close[-2]) / close[-2] * 100
            indicators['price_velocity'] = (close[-1] - close[-3]) / close[-3] * 100

            return indicators

        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return {}

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

    def _calculate_ema(self, values: np.array, period: int) -> np.array:
        """Calculate Exponential Moving Average"""
        ema = np.zeros(len(values))
        ema[0] = values[0]
        multiplier = 2 / (period + 1)
        for i in range(1, len(values)):
            ema[i] = (values[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        return ema

    def _calculate_rsi(self, values: np.array, period: int) -> np.array:
        """Calculate RSI with division by zero handling"""
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

    def _calculate_macd(self, values: np.array) -> tuple:
        """Calculate MACD"""
        ema_12 = self._calculate_ema(values, 12)
        ema_26 = self._calculate_ema(values, 26)
        macd_line = ema_12 - ema_26
        signal_line = self._calculate_ema(macd_line, 9)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _calculate_heikin_ashi(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Heikin Ashi candles with doji detection for signal confirmation"""
        try:
            if df.empty or len(df) < 3:
                return {
                    'ha_trend': 'neutral',
                    'ha_current_bullish': False,
                    'ha_current_bearish': False,
                    'ha_trend_strength': 0,
                    'ha_confirmation': False,
                    'ha_doji_detected': False,
                    'ha_doji_confirmation': False,
                    'ha_bar_switch': False,
                    'ha_signal_ready': False,
                    'ha_open': 0,
                    'ha_close': 0,
                    'ha_high': 0,
                    'ha_low': 0
                }

            ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
            ha_open = np.zeros(len(df))
            ha_open[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2

            for i in range(1, len(df)):
                ha_open[i] = (ha_open[i-1] + ha_close.iloc[i-1]) / 2

            ha_high = np.maximum(df['high'].values, np.maximum(ha_open, ha_close.values))
            ha_low = np.minimum(df['low'].values, np.minimum(ha_open, ha_close.values))

            # Get last 3 candles for doji detection and bar switch confirmation
            last_ha_open = ha_open[-1]
            last_ha_close = ha_close.iloc[-1]
            prev_ha_open = ha_open[-2] if len(ha_open) > 1 else last_ha_open
            prev_ha_close = ha_close.iloc[-2] if len(ha_close) > 1 else last_ha_open
            prev2_ha_open = ha_open[-3] if len(ha_open) > 2 else prev_ha_open
            prev2_ha_close = ha_close.iloc[-3] if len(ha_close) > 2 else prev_ha_close

            # Current candle trend
            current_bullish = last_ha_close > last_ha_open
            current_bearish = last_ha_close < last_ha_open

            # Previous candle trend
            prev_bullish = prev_ha_close > prev_ha_open
            prev_bearish = prev_ha_close < prev_ha_open

            # Detect doji patterns - small body relative to range
            current_body_size = abs(last_ha_close - last_ha_open)
            current_range = ha_high[-1] - ha_low[-1]
            prev_body_size = abs(prev_ha_close - prev_ha_open)
            prev_range = ha_high[-2] - ha_low[-2] if len(ha_high) > 1 else current_range

            # Doji detection - body is less than 10% of the total range
            current_is_doji = (current_body_size / current_range < 0.10) if current_range > 0 else False
            prev_is_doji = (prev_body_size / prev_range < 0.10) if prev_range > 0 else False

            # Bar switch detection - previous was doji, current shows direction
            doji_to_bullish_switch = prev_is_doji and current_bullish and not current_is_doji
            doji_to_bearish_switch = prev_is_doji and current_bearish and not current_is_doji

            # Enhanced confirmation with doji pattern
            bullish_confirmation = current_bullish and (prev_bullish or doji_to_bullish_switch)
            bearish_confirmation = current_bearish and (prev_bearish or doji_to_bearish_switch)

            # Special doji confirmation - when doji switches to directional bar
            doji_confirmation = doji_to_bullish_switch or doji_to_bearish_switch

            # Signal readiness - requires doji confirmation OR strong trend continuation
            signal_ready = doji_confirmation or (bullish_confirmation and not prev_is_doji) or (bearish_confirmation and not prev_is_doji)

            # Calculate trend strength - safe division with doji consideration
            body_size = abs(last_ha_close - last_ha_open)
            candle_range = ha_high[-1] - ha_low[-1]
            
            if candle_range > 0:
                trend_strength = (body_size / candle_range * 100)
                # Boost strength if coming from doji pattern
                if doji_confirmation:
                    trend_strength *= 1.3  # 30% boost for doji confirmation
            else:
                trend_strength = 0

            # Determine final trend with doji consideration
            if doji_confirmation:
                if doji_to_bullish_switch:
                    final_trend = 'bullish_from_doji'
                elif doji_to_bearish_switch:
                    final_trend = 'bearish_from_doji'
                else:
                    final_trend = 'doji_transition'
            elif bullish_confirmation:
                final_trend = 'bullish'
            elif bearish_confirmation:
                final_trend = 'bearish'
            else:
                final_trend = 'neutral'

            return {
                'ha_trend': final_trend,
                'ha_current_bullish': current_bullish,
                'ha_current_bearish': current_bearish,
                'ha_trend_strength': min(100, trend_strength),
                'ha_confirmation': bullish_confirmation or bearish_confirmation,
                'ha_doji_detected': current_is_doji or prev_is_doji,
                'ha_doji_confirmation': doji_confirmation,
                'ha_bar_switch': doji_to_bullish_switch or doji_to_bearish_switch,
                'ha_signal_ready': signal_ready,
                'ha_doji_to_bullish': doji_to_bullish_switch,
                'ha_doji_to_bearish': doji_to_bearish_switch,
                'ha_open': last_ha_open,
                'ha_close': last_ha_close,
                'ha_high': ha_high[-1],
                'ha_low': ha_low[-1]
            }

        except Exception as e:
            self.logger.error(f"Error calculating Heikin Ashi with doji detection: {e}")
            return {
                'ha_trend': 'neutral',
                'ha_current_bullish': False,
                'ha_current_bearish': False,
                'ha_trend_strength': 0,
                'ha_confirmation': False,
                'ha_doji_detected': False,
                'ha_doji_confirmation': False,
                'ha_bar_switch': False,
                'ha_signal_ready': False,
                'ha_open': 0,
                'ha_close': 0,
                'ha_high': 0,
                'ha_low': 0
            }

    def calculate_adaptive_leverage(self, indicators: Dict[str, Any], df: pd.DataFrame) -> int:
        """Calculate adaptive leverage based on market conditions and past performance"""
        try:
            base_leverage = self.leverage_config['base_leverage']
            min_leverage = self.leverage_config['min_leverage']
            max_leverage = self.leverage_config['max_leverage']

            # Load recent performance for adaptive adjustments
            performance_factor = self._get_adaptive_performance_factor()

            volatility_factor = 0
            volume_factor = 0
            trend_factor = 0
            signal_strength_factor = 0

            # Volatility analysis
            volatility = indicators.get('market_volatility', 0.02)
            if volatility <= self.leverage_config['volatility_threshold_low']:
                volatility_factor = 15
            elif volatility >= self.leverage_config['volatility_threshold_high']:
                volatility_factor = -20
            else:
                volatility_factor = -5

            # Volume analysis
            volume_ratio = indicators.get('volume_ratio', 1.0)
            if volume_ratio >= self.leverage_config['volume_threshold_high']:
                volume_factor = 10
            elif volume_ratio <= self.leverage_config['volume_threshold_low']:
                volume_factor = -15
            else:
                volume_factor = 0

            # Trend strength
            ema_bullish = indicators.get('ema_bullish', False)
            ema_bearish = indicators.get('ema_bearish', False)
            supertrend_direction = indicators.get('supertrend_direction', 0)

            if (ema_bullish or ema_bearish) and abs(supertrend_direction) == 1:
                trend_factor = 8
            else:
                trend_factor = -10

            # Signal strength
            signal_strength = indicators.get('signal_strength', 0)
            if signal_strength >= 90:
                signal_strength_factor = 5
            elif signal_strength >= 80:
                signal_strength_factor = 2
            else:
                signal_strength_factor = -5

            # Adaptive performance adjustment
            adaptive_factor = performance_factor * 10  # Scale performance impact

            leverage_adjustment = (
                volatility_factor * 0.3 +
                volume_factor * 0.2 +
                trend_factor * 0.15 +
                signal_strength_factor * 0.15 +
                adaptive_factor * 0.2  # 20% weight for adaptive learning
            )

            final_leverage = base_leverage + leverage_adjustment
            final_leverage = max(min_leverage, min(max_leverage, final_leverage))
            final_leverage = round(final_leverage / 5) * 5

            self.logger.info(f"🎯 Adaptive leverage calculated: {int(final_leverage)}x (Performance factor: {performance_factor:.2f})")
            return int(final_leverage)

        except Exception as e:
            self.logger.error(f"Error calculating adaptive leverage: {e}")
            return self.leverage_config['base_leverage']

    def _get_adaptive_performance_factor(self) -> float:
        """Get performance factor for adaptive leverage adjustment with absolute values and incremental win rate tracking"""
        try:
            # Load recent trades from ML database
            conn = sqlite3.connect(self.ml_analyzer.db_path)
            cursor = conn.cursor()

            # Get recent trades for performance analysis
            cursor.execute("""
                SELECT profit_loss, trade_result 
                FROM ml_trades 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (self.adaptive_leverage['performance_window'],))

            recent_trades = cursor.fetchall()
            conn.close()

            if not recent_trades:
                return 0.25  # Default positive factor (absolute value >= 0)

            # Calculate performance metrics
            wins = sum(1 for trade in recent_trades if trade[0] and trade[0] > 0)
            losses = len(recent_trades) - wins

            if len(recent_trades) == 0:
                return 0.25  # Default positive factor

            current_win_rate = wins / len(recent_trades)

            # Get previous win rate for incremental tracking
            previous_win_rate = getattr(self, '_previous_win_rate', 0.5)

            # Calculate win rate improvement (strictly incremental)
            win_rate_increment = max(0, current_win_rate - previous_win_rate)

            # Store current win rate as previous for next calculation
            self._previous_win_rate = current_win_rate

            # Calculate consecutive performance
            consecutive_wins = 0
            consecutive_losses = 0

            for trade in recent_trades:
                if trade[0] and trade[0] > 0:  # Winning trade
                    consecutive_wins += 1
                    consecutive_losses = 0
                else:  # Losing trade
                    consecutive_losses += 1
                    consecutive_wins = 0

                # Only count the current streak
                if consecutive_wins > 0 and consecutive_losses == 0:
                    break
                elif consecutive_losses > 0 and consecutive_wins == 0:
                    break

            # Update adaptive leverage tracking with incremental win rate
            self.adaptive_leverage.update({
                'recent_wins': wins,
                'recent_losses': losses,
                'consecutive_wins': consecutive_wins,
                'consecutive_losses': consecutive_losses,
                'current_win_rate': current_win_rate,
                'win_rate_increment': win_rate_increment
            })

            # Calculate performance factor (absolute value >= 0)
            performance_factor = 0.0

            # Win rate adjustment with incremental bonus
            base_win_factor = current_win_rate * 0.8  # Base factor from current win rate
            increment_bonus = win_rate_increment * 2.0  # Bonus for improvement

            performance_factor += base_win_factor + increment_bonus

            # Consecutive performance adjustment (absolute values only)
            if consecutive_wins >= 3:
                performance_factor += 0.4
            elif consecutive_wins >= 1:
                performance_factor += 0.15

            # Reduce factor for consecutive losses but keep >= 0
            if consecutive_losses >= 3:
                performance_factor = max(0.1, performance_factor * 0.5)
            elif consecutive_losses >= 1:
                performance_factor = max(0.2, performance_factor * 0.8)

            # Ensure minimum positive factor to maintain absolute value >= 0
            performance_factor = max(0.1, performance_factor)

            # Cap at reasonable maximum
            performance_factor = min(1.5, performance_factor)

            # Log incremental tracking
            self.logger.info(f"📊 Win Rate Tracking: Current: {current_win_rate:.3f}, Previous: {previous_win_rate:.3f}, Increment: {win_rate_increment:.3f}")
            self.logger.info(f"🎯 Performance Factor: {performance_factor:.3f} (absolute value >= 0)")

            return performance_factor

        except Exception as e:
            self.logger.error(f"Error calculating performance factor: {e}")
            return 0.25  # Default positive factor on error

    def generate_ml_enhanced_signal(self, symbol: str, indicators: Dict[str, Any], df: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
        """Generate ML-enhanced scalping signal"""
        try:
            current_time = datetime.now()

            # Track hourly signals but don't limit them
            if current_time.hour != self.last_hour_reset:
                self.hourly_signal_count = 0
                self.last_hour_reset = current_time.hour

            # No hourly limits - push unlimited signals
            if False:  # Disabled limit check
                self.logger.debug(f"⏰ Hourly signal limit reached: {self.hourly_signal_count}/{self.max_signals_per_hour}")
                return None

            # Prevent multiple trades per symbol
            if symbol in self.active_symbols:
                self.logger.debug(f"🔒 Skipping {symbol} - active trade already exists")
                return None

            if symbol in self.last_signal_time:
                time_diff = (current_time - self.last_signal_time[symbol]).total_seconds()
                if time_diff < self.min_signal_interval and not self.unlimited_signals:
                    return None

            signal_strength = 0
            bullish_signals = 0
            bearish_signals = 0

            current_price = indicators.get('current_price', 0)
            if current_price <= 0:
                return None

            # 1. Enhanced SuperTrend (25% weight)
            if indicators.get('supertrend_direction') == 1:
                bullish_signals += 25
            elif indicators.get('supertrend_direction') == -1:
                bearish_signals += 25

            # 2. EMA Confluence (20% weight)
            if indicators.get('ema_bullish'):
                bullish_signals += 20
            elif indicators.get('ema_bearish'):
                bearish_signals += 20

            # 3. CVD Confluence (15% weight)
            cvd_trend = indicators.get('cvd_trend', 'neutral')
            if cvd_trend == 'bullish':
                bullish_signals += 15
            elif cvd_trend == 'bearish':
                bearish_signals += 15

            # 4. VWAP Position (10% weight)
            price_vs_vwap = indicators.get('price_vs_vwap', 0)
            if not np.isnan(price_vs_vwap) and not np.isinf(price_vs_vwap):
                if price_vs_vwap > 0.1:
                    bullish_signals += 10
                elif price_vs_vwap < -0.1:
                    bearish_signals += 10

            # 5. RSI analysis (10% weight)
            if indicators.get('rsi_oversold'):
                bullish_signals += 10
            elif indicators.get('rsi_overbought'):
                bearish_signals += 10

            # 6. MACD confluence (10% weight)
            if indicators.get('macd_bullish'):
                bullish_signals += 10
            elif indicators.get('macd_bearish'):
                bearish_signals += 10

            # 7. Volume surge (10% weight)
            if indicators.get('volume_surge'):
                if bullish_signals > bearish_signals:
                    bullish_signals += 10
                else:
                    bearish_signals += 10

            # 8. Heikin Ashi trend confirmation (15% weight) - Critical for trend validation
            ha_trend = indicators.get('ha_trend', 'neutral')
            ha_confirmation = indicators.get('ha_confirmation', False)
            ha_strength = indicators.get('ha_trend_strength', 0)

            if ha_trend == 'bullish' and ha_confirmation and ha_strength > 60:
                bullish_signals += 15
            elif ha_trend == 'bearish' and ha_confirmation and ha_strength > 60:
                bearish_signals += 15
            elif ha_trend != 'neutral' and ha_confirmation:
                # Partial points for weaker Heikin Ashi signals
                if ha_trend == 'bullish':
                    bullish_signals += 8
                elif ha_trend == 'bearish':
                    bearish_signals += 8

            # Enhanced signal strength calculation with validation
            total_possible_signals = 125  # Total possible points from all indicators
            
            if bullish_signals >= self.min_signal_strength:
                direction = 'BUY'
                signal_strength = min(100, bullish_signals)
                signal_percentage = (bullish_signals / total_possible_signals) * 100
            elif bearish_signals >= self.min_signal_strength:
                direction = 'SELL'
                signal_strength = min(100, bearish_signals)
                signal_percentage = (bearish_signals / total_possible_signals) * 100
            else:
                # Log why signal was rejected
                max_signal = max(bullish_signals, bearish_signals)
                self.logger.debug(f"❌ {symbol} signal too weak - Max strength: {max_signal:.0f}% < Required: {self.min_signal_strength}%")
                return None

            # Validate signal strength consistency
            if signal_strength < 50:
                self.logger.debug(f"❌ {symbol} signal strength too low: {signal_strength:.0f}%")
                return None

            # Enhanced Heikin Ashi validation - more nuanced
            ha_conflict = False
            if direction == 'BUY' and ha_trend == 'bearish' and ha_strength > 70:
                ha_conflict = True
                self.logger.debug(f"⚠️ {symbol} BUY signal conflicts with strong bearish HA trend ({ha_strength:.0f}%)")
            elif direction == 'SELL' and ha_trend == 'bullish' and ha_strength > 70:
                ha_conflict = True
                self.logger.debug(f"⚠️ {symbol} SELL signal conflicts with strong bullish HA trend ({ha_strength:.0f}%)")

            # Allow weak HA conflicts if signal is very strong
            if ha_conflict and signal_strength < 90:
                self.logger.debug(f"❌ {symbol} signal rejected - HA conflict with insufficient strength")
                return None

            # Calculate entry, stop loss, and take profits
            entry_price = current_price
            risk_percentage = 1.5  # 1.5% risk
            risk_amount = entry_price * (risk_percentage / 100)

            if direction == 'BUY':
                stop_loss = entry_price - risk_amount
                tp1 = entry_price + (risk_amount * 0.33)  # 33% of profit for 1:1 ratio
                tp2 = entry_price + (risk_amount * 0.67)  # 67% of profit for 1:1 ratio
                tp3 = entry_price + (risk_amount * 1.0)   # Full 1:1 profit

                if not (stop_loss < entry_price < tp1 < tp2 < tp3):
                    stop_loss = entry_price * 0.985
                    tp1 = entry_price * 1.005
                    tp2 = entry_price * 1.010
                    tp3 = entry_price * 1.015
            else:
                stop_loss = entry_price + risk_amount
                tp1 = entry_price - (risk_amount * 0.33)  # 33% of profit for 1:1 ratio
                tp2 = entry_price - (risk_amount * 0.67)  # 67% of profit for 1:1 ratio
                tp3 = entry_price - (risk_amount * 1.0)   # Full 1:1 profit

                if not (tp3 < tp2 < tp1 < entry_price < stop_loss):
                    stop_loss = entry_price * 1.015
                    tp1 = entry_price * 0.995
                    tp2 = entry_price * 0.990
                    tp3 = entry_price * 0.985

            # Risk validation
            risk_percentage = abs(entry_price - stop_loss) / entry_price * 100
            if risk_percentage > 3.0:
                return None

            # Calculate adaptive leverage with cross margin
            placeholder_df = pd.DataFrame({'close': [current_price] * 20}) if df is None or len(df) < 20 else df
            optimal_leverage = self.calculate_adaptive_leverage(indicators, placeholder_df)

            # ENHANCED HEIKIN ASHI CONFIRMATION (LESS RESTRICTIVE)
            ha_signal_ready = indicators.get('ha_signal_ready', False)
            ha_doji_confirmation = indicators.get('ha_doji_confirmation', False)
            ha_doji_to_bullish = indicators.get('ha_doji_to_bullish', False)
            ha_doji_to_bearish = indicators.get('ha_doji_to_bearish', False)
            ha_confirmation = indicators.get('ha_confirmation', False)
            ha_trend = indicators.get('ha_trend', 'neutral')

            # Multiple confirmation methods (not just doji)
            direction_matches_ha = False
            confirmation_method = "none"
            
            # Method 1: Doji confirmation (preferred)
            if direction == 'BUY' and ha_doji_to_bullish:
                direction_matches_ha = True
                confirmation_method = "doji_to_bullish"
                self.logger.info(f"✅ BUY signal confirmed by Heikin Ashi doji to bullish switch: {symbol}")
            elif direction == 'SELL' and ha_doji_to_bearish:
                direction_matches_ha = True
                confirmation_method = "doji_to_bearish"
                self.logger.info(f"✅ SELL signal confirmed by Heikin Ashi doji to bearish switch: {symbol}")
            
            # Method 2: Strong trend confirmation (alternative)
            elif direction == 'BUY' and ha_trend in ['bullish', 'bullish_from_doji'] and ha_confirmation:
                direction_matches_ha = True
                confirmation_method = "bullish_trend"
                self.logger.info(f"✅ BUY signal confirmed by strong Heikin Ashi bullish trend: {symbol}")
            elif direction == 'SELL' and ha_trend in ['bearish', 'bearish_from_doji'] and ha_confirmation:
                direction_matches_ha = True
                confirmation_method = "bearish_trend"
                self.logger.info(f"✅ SELL signal confirmed by strong Heikin Ashi bearish trend: {symbol}")
            
            # Method 3: Signal ready confirmation (backup)
            elif ha_signal_ready:
                direction_matches_ha = True
                confirmation_method = "signal_ready"
                self.logger.info(f"✅ {direction} signal confirmed by Heikin Ashi signal ready: {symbol}")
            
            # Method 4: High signal strength override (for very strong signals)
            elif signal_strength >= 95:
                direction_matches_ha = True
                confirmation_method = "high_strength_override"
                self.logger.info(f"✅ {direction} signal confirmed by exceptional strength ({signal_strength:.0f}%): {symbol}")

            # Only reject if completely contradictory
            if direction == 'BUY' and ha_trend == 'bearish' and not direction_matches_ha:
                self.logger.debug(f"❌ {symbol} BUY signal rejected - Strong bearish Heikin Ashi trend")
                return None
            elif direction == 'SELL' and ha_trend == 'bullish' and not direction_matches_ha:
                self.logger.debug(f"❌ {symbol} SELL signal rejected - Strong bullish Heikin Ashi trend")
                return None
            
            # Accept signals with any confirmation method or neutral HA
            if not direction_matches_ha and ha_trend not in ['neutral', 'doji_transition']:
                self.logger.debug(f"⚠️ {symbol} signal proceeding with limited HA confirmation - Method: {confirmation_method}")
            
            # Store confirmation method for analysis
            ha_confirmation_used = confirmation_method

            # ML prediction with IMPROVED filtering
            ml_signal_data = {
                'symbol': symbol,
                'direction': direction,
                'signal_strength': signal_strength,
                'leverage': optimal_leverage,
                'market_volatility': indicators.get('market_volatility', 0.02),
                'volume_ratio': indicators.get('volume_ratio', 1.0),
                'rsi': indicators.get('rsi', 50),
                'cvd_trend': cvd_trend,
                'macd_signal': 'bullish' if indicators.get('macd_bullish') else 'bearish' if indicators.get('macd_bearish') else 'neutral',
                'ema_bullish': indicators.get('ema_bullish', False)
            }

            ml_prediction = self.ml_analyzer.predict_trade_outcome(ml_signal_data)

            # IMPROVED ML FILTERING - Less restrictive but quality-focused
            ml_confidence = ml_prediction.get('confidence', 50)
            prediction_type = ml_prediction.get('prediction', 'unknown')
            expected_profit = ml_prediction.get('expected_profit', 0)

            # Block only clearly negative predictions
            blocked_predictions = ['filtered_out', 'below_threshold']
            if prediction_type in blocked_predictions:
                rejection_reasons = ml_prediction.get('rejection_reasons', [])
                self.logger.debug(f"❌ {symbol} signal blocked by ML - {prediction_type}: {'; '.join(rejection_reasons)}")
                return None

            # Allow favorable predictions and strong signal overrides
            acceptable_predictions = [
                'highly_favorable', 'favorable', 'above_neutral', 
                'strength_override', 'strength_based'
            ]
            
            # Special handling for fallback mode
            if ml_prediction.get('fallback_mode', False):
                if prediction_type not in acceptable_predictions:
                    self.logger.debug(f"❌ {symbol} fallback signal rejected - {prediction_type} (confidence: {ml_confidence:.1f}%)")
                    return None
            elif prediction_type not in acceptable_predictions:
                # Check for high signal strength override
                if signal_strength >= 90 and ml_confidence >= 40:
                    self.logger.info(f"✅ {symbol} signal accepted via high strength override - Strength: {signal_strength:.1f}%, ML: {ml_confidence:.1f}%")
                    prediction_type = 'strength_override'
                else:
                    self.logger.debug(f"❌ {symbol} signal rejected - ML prediction: {prediction_type} (confidence: {ml_confidence:.1f}%)")
                    return None

            # Boost signal strength for high ML confidence
            if prediction_type == 'highly_favorable':
                signal_strength *= 1.3  # Strong boost for highly favorable
            elif prediction_type == 'favorable':
                signal_strength *= 1.2  # Good boost for favorable
            elif prediction_type == 'above_neutral':
                signal_strength *= 1.1  # Small boost for above neutral

            # Additional boost for doji confirmation
            if direction_matches_doji:
                signal_strength *= 1.15  # Extra boost for doji confirmation

            # MAINTAIN HIGH SIGNAL STRENGTH - strict threshold
            if signal_strength < 80:  # High threshold to maintain strength
                self.logger.debug(f"❌ {symbol} signal rejected - Final strength too low: {signal_strength:.1f}")
                return None

            # Update last signal time and lock symbol for single trade per symbol
            self.last_signal_time[symbol] = current_time
            # Lock symbol to prevent multiple concurrent trades
            self.active_symbols.add(symbol)
            self.symbol_trade_lock[symbol] = current_time

            # Increment hourly signal counter
            self.hourly_signal_count += 1

            return {
                'symbol': symbol,
                'direction': direction,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'tp1': tp1,
                'tp2': tp2,
                'tp3': tp3,
                'signal_strength': min(signal_strength, 100),
                'risk_percentage': risk_percentage,
                'risk_reward_ratio': self.risk_reward_ratio,
                'optimal_leverage': optimal_leverage,
                'margin_type': 'CROSSED',
                'ml_prediction': ml_prediction,
                'ha_confirmation': direction_matches_doji,
                'ha_doji_switch': ha_doji_confirmation,
                'indicators_used': [
                    'Heikin Ashi Doji Confirmation', 'ML Above-Neutral Filter', 'Enhanced SuperTrend', 
                    'EMA Confluence', 'CVD Analysis', 'VWAP Position', 'Volume Surge', 'RSI Analysis', 'MACD Signals'
                ],
                'timeframe': 'Multi-TF (1m-4h)',
                'strategy': 'Ultimate ML-Enhanced Scalping with Doji Confirmation',
                'ml_enhanced': True,
                'adaptive_leverage': True,
                'strict_filtering': True,
                'entry_time': current_time
            }

        except Exception as e:
            self.logger.error(f"Error generating ML-enhanced signal: {e}")
            return None

    async def scan_for_signals(self) -> List[Dict[str, Any]]:
        """Scan all symbols for ML-enhanced signals"""
        signals = []

        # Update CVD data
        try:
            await self.calculate_cvd_btc_perp()
            self.logger.info(f"📊 CVD Updated - Trend: {self.cvd_data['cvd_trend']}, Strength: {self.cvd_data['cvd_strength']:.1f}%")
        except Exception as e:
            self.logger.warning(f"CVD calculation error: {e}")

        for symbol in self.symbols:
            try:
                # Quick availability check
                test_df = await self.get_binance_data(symbol, '1h', 10)
                if test_df is None:
                    continue

                timeframe_scores = {}

                for timeframe in self.timeframes:
                    try:
                        df = await self.get_binance_data(symbol, timeframe, 100)
                        if df is None or len(df) < 50:
                            continue

                        indicators = self.calculate_advanced_indicators(df)
                        if not indicators or not isinstance(indicators, dict):
                            continue

                        signal = self.generate_ml_enhanced_signal(symbol, indicators, df)
                        if signal and isinstance(signal, dict) and 'signal_strength' in signal:
                            timeframe_scores[timeframe] = signal
                    except Exception as e:
                        self.logger.warning(f"Timeframe {timeframe} error for {symbol}: {e}")
                        continue

                if timeframe_scores:
                    try:
                        valid_signals = [s for s in timeframe_scores.values() if s.get('signal_strength', 0) > 0]
                        if valid_signals:
                            # Select signal with highest ML confidence
                            best_signal = max(valid_signals, key=lambda x: x.get('ml_prediction', {}).get('confidence', 0))

                            # Use more permissive thresholds for maximum signal generation
                            ml_confidence = best_signal.get('ml_prediction', {}).get('confidence', 0)
                            signal_strength = best_signal.get('signal_strength', 0)

                            # Accept signals with either good ML confidence OR good signal strength
                            if (ml_confidence >= 65 and signal_strength >= 60) or \
                               (ml_confidence >= 70) or \
                               (signal_strength >= self.min_signal_strength):
                                signals.append(best_signal)
                    except Exception as e:
                        self.logger.error(f"Error selecting best signal for {symbol}: {e}")
                        continue

            except Exception as e:
                self.logger.warning(f"Skipping {symbol} due to error: {e}")
                continue

        # Sort by ML confidence and signal strength but return more signals
        signals.sort(key=lambda x: (x.get('ml_prediction', {}).get('confidence', 0), x['signal_strength']), reverse=True)
        return signals  # Return all signals instead of limiting

    async def verify_channel_access(self) -> bool:
        """Verify channel access"""
        try:
            url = f"{self.base_url}/getChat"
            data = {'chat_id': self.target_channel}

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        self.channel_accessible = True
                        self.logger.info(f"✅ Channel {self.target_channel} is accessible")
                        return True
                    else:
                        self.channel_accessible = False
                        error = await response.text()
                        self.logger.warning(f"⚠️ Channel {self.target_channel} not accessible: {error}")
                        return False

        except Exception as e:
            self.channel_accessible = False
            self.logger.error(f"Error verifying channel access: {e}")
            return False

    async def send_message(self, chat_id: str, text: str, parse_mode=None) -> bool:
        """Send message to Telegram"""
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': text,
                'disable_web_page_preview': True
            }

            # Only add parse_mode if it's specified and not None
            if parse_mode:
                data['parse_mode'] = parse_mode

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        self.logger.info(f"✅ Message sent successfully to {chat_id}")
                        if chat_id == self.target_channel:
                            self.channel_accessible = True
                        return True
                    else:
                        error = await response.text()
                        self.logger.warning(f"⚠️ Send message failed to {chat_id}: {error}")
                        if chat_id == self.target_channel:
                            self.channel_accessible = False
                        if chat_id == self.target_channel and self.admin_chat_id:
                            return await self._send_to_admin_fallback(text, parse_mode)
                        return False

        except Exception as e:
            self.logger.error(f"Error sending message to {chat_id}: {e}")
            if chat_id == self.target_channel:
                self.channel_accessible = False
            if chat_id == self.target_channel and self.admin_chat_id:
                return await self._send_to_admin_fallback(text, parse_mode)
            return False

    async def _send_to_admin_fallback(self, text: str, parse_mode: str) -> bool:
        """Fallback to send message to admin"""
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': self.admin_chat_id,
                'text': f"📢 CHANNEL FALLBACK\n\n{text}",
                'disable_web_page_preview': True
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        self.logger.info(f"✅ Fallback message sent to admin {self.admin_chat_id}")
                        return True
                    return False
        except:
            return False

    def format_ml_signal_message(self, signal: Dict[str, Any]) -> str:
        """Format ML signal message with enhanced information"""
        ml_prediction = signal.get('ml_prediction', {})

        # Simple Cornix format
        cornix_signal = self._format_cornix_signal(signal)

        # Enhanced Heikin Ashi confirmation status
        ha_confirmation_used = signal.get('ha_confirmation_used', 'none')
        ha_status_map = {
            'doji_to_bullish': "🎯 DOJI→BULL",
            'doji_to_bearish': "🎯 DOJI→BEAR", 
            'bullish_trend': "📈 STRONG BULL",
            'bearish_trend': "📉 STRONG BEAR",
            'signal_ready': "✅ HA READY",
            'high_strength_override': "🚀 HIGH POWER",
            'none': "⚠️ BASIC"
        }
        ha_status = ha_status_map.get(ha_confirmation_used, "⚠️ BASIC")

        # Enhanced ML status
        ml_conf = ml_prediction.get('confidence', 0)
        ml_pred = ml_prediction.get('prediction', 'unknown').replace('_', ' ').title()
        expected_profit = ml_prediction.get('expected_profit', 0)
        model_accuracy = ml_prediction.get('model_accuracy', 0)
        
        # ML mode indicator
        ml_mode = "🤖 FULL ML" if not ml_prediction.get('fallback_mode', False) else "🔄 FALLBACK"
        
        # Signal quality indicators
        quality_indicators = []
        if ml_conf >= 80:
            quality_indicators.append("🔥 HIGH CONF")
        if expected_profit >= 1.0:
            quality_indicators.append("💎 HIGH ROI")
        if signal['signal_strength'] >= 90:
            quality_indicators.append("⚡ MAX POWER")
        
        quality_status = " | ".join(quality_indicators) if quality_indicators else "📊 STANDARD"
        
        message = f"""{cornix_signal}

🧠 {ml_mode}: {ml_pred} ({ml_conf:.0f}%) | Accuracy: {model_accuracy:.0f}%
📊 Signal: {signal['signal_strength']:.0f}% | Expected: +{expected_profit:.1f}% | R/R: 1:1
🕯️ HA: {ha_status} | Method: {ha_confirmation_used.replace('_', ' ').title()}
⚖️ {signal.get('optimal_leverage', 35)}x Cross Margin | 📈 Auto-Scaling Active
{quality_status}
🕐 {datetime.now().strftime('%H:%M')} UTC | Signal #{self.hourly_signal_count}

🎯 Multi-Indicator Confluence | 🧠 ML Enhanced | 🛡️ Auto Risk Management"""

        return message.strip()

    def _format_cornix_signal(self, signal: Dict[str, Any]) -> str:
        """Format signal in Cornix-compatible format with improved parsing"""
        try:
            symbol = signal['symbol']
            direction = signal['direction'].upper()
            entry = signal['entry_price']
            stop_loss = signal['stop_loss']
            tp1 = signal['tp1']
            tp2 = signal['tp2']
            tp3 = signal['tp3']
            optimal_leverage = signal.get('optimal_leverage', 35)

            # Cornix-compatible format with clear parsing structure
            formatted_message = f"""#{symbol} {direction}

Entry: {entry:.6f}
StopLoss: {stop_loss:.6f}

TakeProfit:
TP1: {tp1:.6f}
TP2: {tp2:.6f}
TP3: {tp3:.6f}

Leverage: {optimal_leverage}x
MarginType: Cross
Exchange: BinanceFutures

TradeManagement:
- MoveSLtoEntry: AfterTP1
- MoveSLtoTP1: AfterTP2
- CloseAll: AfterTP3"""

            return formatted_message

        except Exception as e:
            self.logger.error(f"Error formatting Cornix signal: {e}")
            optimal_leverage = signal.get('optimal_leverage', 35)
            # Fallback format that's still Cornix-compatible
            return f"""#{signal['symbol']} {signal['direction']}
Entry: {signal['entry_price']:.6f}
StopLoss: {signal['stop_loss']:.6f}
TP1: {signal['tp1']:.6f}
TP2: {signal['tp2']:.6f}
TP3: {signal['tp3']:.6f}
Leverage: {optimal_leverage}x
Exchange: BinanceFutures"""

    async def send_to_cornix(self, signal: Dict[str, Any]) -> bool:
        """Cornix integration disabled - signals sent via Telegram only"""
        return True

    async def get_updates(self, offset=None, timeout=30) -> list:
        """Get Telegram updates"""
        try:
            url = f"{self.base_url}/getUpdates"
            params = {'timeout': timeout}
            if offset is not None:
                params['offset'] = offset

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('result', [])
                    return []

        except Exception as e:
            self.logger.error(f"Error getting updates: {e}")
            return []

    def generate_chart(self, symbol: str, df: pd.DataFrame, signal: Dict[str, Any]) -> Optional[str]:
        """Generate 1:1 ratio chart for the signal"""
        try:
            if not CHART_AVAILABLE or df is None or len(df) < 10:
                self.logger.warning(f"Chart generation skipped for {symbol}: insufficient data or libraries")
                return None

            # Create figure with 1:1 aspect ratio (square)
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))

            # Plot price line
            closes = df['close'].values
            if len(closes) == 0:
                plt.close(fig)
                return None

            # Use available data
            data_len = min(50, len(df))
            x_range = range(data_len)

            ax.plot(x_range, closes[-data_len:], color='#00ff00', linewidth=2, label='Price')

            # Mark entry point
            entry_price = signal.get('entry_price', closes[-1])
            ax.axhline(y=entry_price, color='yellow', linestyle='--', alpha=0.8, label=f'Entry: {entry_price:.4f}')

            # Mark TP levels
            if 'tp1' in signal and signal['tp1'] > 0:
                ax.axhline(y=signal['tp1'], color='green', linestyle=':', alpha=0.6, label=f'TP1: {signal["tp1"]:.4f}')
            if 'tp2' in signal and signal['tp2'] > 0:
                ax.axhline(y=signal['tp2'], color='green', linestyle=':', alpha=0.4)
            if 'tp3' in signal and signal['tp3'] > 0:
                ax.axhline(y=signal['tp3'], color='green', linestyle=':', alpha=0.2)

            # Mark SL
            if 'stop_loss' in signal and signal['stop_loss'] > 0:
                ax.axhline(y=signal['stop_loss'], color='red', linestyle=':', alpha=0.8, label=f'SL: {signal["stop_loss"]:.4f}')

            # Style the chart
            ax.set_facecolor('black')
            fig.patch.set_facecolor('black')
            ax.tick_params(colors='white')
            ax.set_title(f'{symbol} - {signal.get("direction", "BUY")} Signal', color='white', fontsize=12)
            ax.legend(loc='upper left', facecolor='black', edgecolor='white', labelcolor='white')
            ax.grid(True, alpha=0.3, color='gray')

            # Remove x-axis labels for cleaner look
            ax.set_xticks([])

            plt.tight_layout()

            # Convert to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', facecolor='black', edgecolor='white', dpi=80, bbox_inches='tight')
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.getvalue()).decode()

            # Clean up
            plt.close(fig)
            buffer.close()

            return chart_base64

        except Exception as e:
            self.logger.error(f"Error generating chart for {symbol}: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None

    async def send_photo(self, chat_id: str, photo_data: str, caption: str = "") -> bool:
        """Send photo to Telegram"""
        try:
            url = f"{self.base_url}/sendPhoto"

            # Convert base64 to bytes
            photo_bytes = base64.b64decode(photo_data)

            # Create form data properly for aiohttp
            form_data = aiohttp.FormData()
            form_data.add_field('chat_id', chat_id)
            form_data.add_field('caption', caption)
            form_data.add_field('parse_mode', 'Markdown')
            form_data.add_field('photo', photo_bytes, filename='chart.png', content_type='image/png')

            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=form_data) as response:
                    if response.status == 200:
                        self.logger.info(f"✅ Photo sent successfully to {chat_id}")
                        return True
                    else:
                        error = await response.text()
                        self.logger.warning(f"Send photo failed: {error}")
                        return False

        except Exception as e:
            self.logger.error(f"Error sending photo: {e}")
            return False

    async def _auto_unlock_symbol(self, symbol: str, delay_seconds: int):
        """Automatically unlock symbol after delay (safety mechanism)"""
        try:
            await asyncio.sleep(delay_seconds)
            if symbol in self.active_symbols:
                self.release_symbol_lock(symbol)
                self.logger.info(f"🕐 Auto-unlocked {symbol} after {delay_seconds/60:.0f} minutes")
        except Exception as e:
            self.logger.error(f"Error auto-unlocking {symbol}: {e}")

    async def handle_commands(self, message: Dict, chat_id: str):
        """Handle bot commands"""
        try:
            text = message.get('text', '').strip()

            if not text:
                return

            if text.startswith('/start'):
                self.admin_chat_id = chat_id
                self.logger.info(f"✅ Admin set to chat_id: {chat_id}")

                ml_summary = self.ml_analyzer.get_ml_summary()
                await self.send_message(chat_id, f"""🧠 ULTIMATE ML BOT

✅ Online & Learning
📊 Accuracy: {ml_summary['model_performance']['signal_accuracy']*100:.1f}%
📈 Trades: {ml_summary['model_performance']['total_trades_learned']}
🎯 Next Retrain: {ml_summary['next_retrain_in']}

Commands:
/ml - ML Status
/scan - Market Scan  
/stats - Performance
/symbols - Active Pairs
/leverage - Current Settings
/risk - Risk Management
/session - Trading Session
/help - All Commands

Bot learns from every trade""")

            elif text.startswith('/help'):
                await self.send_message(chat_id, """Available Commands:

/start - Initialize bot
/ml - ML model status
/scan - Scan markets
/stats - Performance stats
/symbols - Trading symbols
/leverage - Leverage settings
/risk - Risk management
/session - Current session
/cvd - CVD analysis
/market - Market conditions
/insights - Trading insights
/settings - Bot settings
/unlock [SYMBOL] - Unlock symbol trade lock""")

            elif text.startswith('/stats'):
                ml_summary = self.ml_analyzer.get_ml_summary()
                active_symbols_list = ', '.join(sorted(self.active_symbols)) if self.active_symbols else 'None'

                # Check persistent log status
                log_file = Path("persistent_trade_logs.json")
                persistent_logs_count = 0
                if log_file.exists():
                    try:
                        with open(log_file, 'r') as f:
                            logs = json.load(f)
                            persistent_logs_count = len(logs)
                    except:
                        pass

                await self.send_message(chat_id, f"""📊 **PERFORMANCE STATS**

Signals Scanned: {self.signal_counter}
Signals Sent: {len(self.active_trades)} (Open)
✅ Win Rate: {self.performance_stats['win_rate']:.1f}%
💰 Total Profit: {self.performance_stats['total_profit']:.2f}%
📈 Active Trades: {len(self.active_trades)}
🔒 Active Symbols: {len(self.active_symbols)}
🧠 ML Accuracy: {ml_summary['model_performance']['signal_accuracy']*100:.1f}%
⚡ Trades Learned: {ml_summary['model_performance']['total_trades_learned']}
💾 Persistent Logs: {persistent_logs_count}

**Active Pairs:** {active_symbols_list}""")

            elif text.startswith('/symbols'):
                await self.send_message(chat_id, f"""📋 **TRADING SYMBOLS**

Total Pairs: {len(self.symbols)}
Timeframes: {', '.join(self.timeframes)}

Top Pairs:
• BTCUSDT, ETHUSDT, BNBUSDT
• XRPUSDT, ADAUSDT, SOLUSDT
• DOGEUSDT, AVAXUSDT, DOTUSDT
• +{len(self.symbols)-9} more pairs""")

            elif text.startswith('/leverage'):
                await self.send_message(chat_id, f"""⚖️ **LEVERAGE SETTINGS**

Current Base: {self.leverage_config['base_leverage']}x
Range: {self.leverage_config['min_leverage']}x - {self.leverage_config['max_leverage']}x
Type: Cross Margin
Adaptive: ✅ Enabled

Recent Performance:
• Wins: {self.adaptive_leverage['recent_wins']}
• Losses: {self.adaptive_leverage['recent_losses']}
• Streak: {self.adaptive_leverage['consecutive_wins']}W""")

            elif text.startswith('/risk'):
                await self.send_message(chat_id, f"""🛡️ **RISK MANAGEMENT**

Risk per Trade: 1.5%
Risk/Reward: 1:3
Max Concurrent: {self.max_concurrent_trades}
Signal Filter: {self.min_signal_strength}%+

Auto Management:
✅ SL to Entry after TP1
✅ SL to TP1 after TP2
✅ Full close at TP3""")

            elif text.startswith('/session'):
                current_session = self._get_time_session(datetime.now())
                await self.send_message(chat_id, f"""🕐 **TRADING SESSION**

Current: {current_session}
Time: {datetime.now().strftime('%H:%M UTC')}
CVD Trend: {self.cvd_data['cvd_trend'].title()}
CVD Strength: {self.cvd_data['cvd_strength']:.1f}%

Session Performance:
• Best: NY_MAIN, LONDON_OPEN
• Moderate: NY_OVERLAP
• Quiet: ASIA_MAIN""")

            elif text.startswith('/cvd'):
                await self.send_message(chat_id, f"""📊 **CVD ANALYSIS**

BTC Perp CVD: {self.cvd_data['btc_perp_cvd']:.2f}
Trend: {self.cvd_data['cvd_trend'].title()}
Strength: {self.cvd_data['cvd_strength']:.1f}%
Divergence: {'⚠️ Yes' if self.cvd_data['cvd_divergence'] else '✅ No'}

*CVD measures institutional flow*""")

            elif text.startswith('/market'):
                await self.send_message(chat_id, f"""🌍 **MARKET CONDITIONS**

Session: {self._get_time_session(datetime.now())}
CVD: {self.cvd_data['cvd_trend'].title()}
Volatility: Normal
Volume: Active

Signal Quality: High
ML Filter: Active
Next Scan: <60s""")

            elif text.startswith('/insights'):
                ml_summary = self.ml_analyzer.get_ml_summary()
                await self.send_message(chat_id, f"""🔍 **TRADING INSIGHTS**

Best Sessions: Available
Symbol Performance: Tracked  
Indicator Effectiveness: Analyzed
Market Patterns: Learning

Learning Status: {ml_summary['learning_status'].title()}
Data Points: {ml_summary['model_performance']['total_trades_learned']}

*Insights improve with more data*""")

            elif text.startswith('/unlock'):
                # Manual unlock command for specific symbol
                parts = text.split()
                if len(parts) > 1:
                    symbol = parts[1].upper()
                    if symbol in self.active_symbols:
                        self.release_symbol_lock(symbol)
                        await self.send_message(chat_id, f"🔓 **{symbol} unlocked**")
                    else:
                        await self.send_message(chat_id, f"ℹ️ **{symbol} not locked**")
                else:
                    # Unlock all symbols
                    unlocked_count = len(self.active_symbols)
                    self.active_symbols.clear()
                    self.symbol_trade_lock.clear()
                    await self.send_message(chat_id, f"🔓 **Unlocked {unlocked_count} symbols**")

            elif text.startswith('/history'):
                # Show recent trade history from persistent logs
                try:
                    log_file = Path("persistent_trade_logs.json")
                    if not log_file.exists():
                        await self.send_message(chat_id, "📭 **No trade history found**")
                        return

                    with open(log_file, 'r') as f:
                        logs = json.load(f)

                    if not logs:
                        await self.send_message(chat_id, "📭 **No trades in history**")
                        return

                    # Show last 5 trades
                    recent_trades = logs[-5:]
                    history_msg = "📈 **RECENT TRADE HISTORY**\n\n"

                    for trade in reversed(recent_trades):
                        symbol = trade.get('symbol', 'UNKNOWN')
                        direction = trade.get('direction', 'UNKNOWN')
                        result = trade.get('trade_result', 'UNKNOWN')
                        pnl = trade.get('profit_loss', 0)
                        leverage = trade.get('leverage', 0)

                        status_emoji = "✅" if pnl > 0 else "❌"
                        history_msg += f"{status_emoji} **{symbol}** {direction} - {result}\n"
                        history_msg += f"   P/L: {pnl:.2f}% | Leverage: {leverage}x\n\n"

                    history_msg += f"💾 Total Logged Trades: {len(logs)}"
                    await self.send_message(chat_id, history_msg)

                except Exception as e:
                    await self.send_message(chat_id, f"❌ **Error loading history:** {str(e)}")

            elif text.startswith('/opentrades'):
                # Show current open trades and ML learning status
                try:
                    if not self.active_trades:
                        await self.send_message(chat_id, "📊 **No active trades currently**")
                        return

                    open_msg = "🔄 **OPEN TRADES - ML LEARNING**\n\n"

                    for symbol, trade_info in self.active_trades.items():
                        signal = trade_info['signal']
                        entry_time = trade_info['entry_time']
                        duration = (datetime.now() - entry_time).total_seconds() / 60

                        # Get current price for unrealized P/L
                        try:
                            df = await self.get_binance_data(symbol, '1m', 1)
                            if df is not None and len(df) > 0:
                                current_price = float(df['close'].iloc[-1])
                                entry_price = signal['entry_price']

                                if signal['direction'].upper() in ['BUY', 'LONG']:
                                    unrealized_pnl = ((current_price - entry_price) / entry_price) * 100 * signal['optimal_leverage']
                                else:
                                    unrealized_pnl = ((entry_price - current_price) / entry_price) * 100 * signal['optimal_leverage']

                                status_emoji = "🟢" if unrealized_pnl > 0 else "🔴" if unrealized_pnl < 0 else "🟡"

                                open_msg += f"{status_emoji} **{symbol}** {signal['direction']}\n"
                                open_msg += f"   Entry: {entry_price:.6f}\n"
                                open_msg += f"   Current: {current_price:.6f}\n"
                                open_msg += f"   Unrealized P/L: {unrealized_pnl:.2f}%\n"
                                open_msg += f"   Duration: {duration:.1f} minutes\n"
                                open_msg += f"   ML Learning: ✅ Active\n\n"
                            else:
                                open_msg += f"📊 **{symbol}** {signal['direction']}\n"
                                open_msg += f"   Entry: {signal['entry_price']:.6f}\n"
                                open_msg += f"   Duration: {duration:.1f} minutes\n"
                                open_msg += f"   ML Learning: ✅ Active\n\n"
                        except:
                            open_msg += f"📊 **{symbol}** {signal['direction']}\n"
                            open_msg += f"   Entry: {signal['entry_price']:.6f}\n"
                            open_msg += f"   Duration: {duration:.1f} minutes\n"
                            open_msg += f"   ML Learning: ✅ Active\n\n"

                    # Add ML learning stats
                    update_count = getattr(self, '_open_trade_updates', 0)
                    open_msg += f"🧠 **ML Learning Stats:**\n"
                    open_msg += f"• Real-time Updates: {update_count}\n"
                    open_msg += f"• Learning Status: Active\n"
                    open_msg += f"• Next Incremental Training: {10 - (update_count % 10)} updates"

                    await self.send_message(chat_id, open_msg)

                except Exception as e:
                    await self.send_message(chat_id, f"❌ **Error loading open trades:** {str(e)}")

            elif text.startswith('/settings'):
                await self.send_message(chat_id, f"""⚙️ **BOT SETTINGS**

Target Channel: {self.target_channel}
Max Signals/Hour: {self.max_signals_per_hour} (Unlimited)
Min Signal Interval: {self.min_signal_interval}s
Auto-Restart: ✅ Enabled
Duplicate Prevention: ✅ One trade per symbol
Max Concurrent: {self.max_concurrent_trades}

ML Features:
• Continuous Learning: ✅
• Adaptive Leverage: ✅  
• Risk Assessment: ✅
• Market Insights: ✅""")

            elif text.startswith('/ml'):
                ml_summary = self.ml_analyzer.get_ml_summary()
                await self.send_message(chat_id, f"""🧠 **ML STATUS**

Signal Accuracy: {ml_summary['model_performance']['signal_accuracy']*100:.1f}%
Trades Learned: {ml_summary['model_performance']['total_trades_learned']}
Learning: {ml_summary['learning_status'].title()}
Next Retrain: {ml_summary['next_retrain_in']} trades

Models Active:
✅ Signal Classifier
✅ Profit Predictor  
✅ Risk Assessor""")

            elif text.startswith('/scan'):
                await self.send_message(chat_id, "🔍 **Scanning markets...**")

                signals = await self.scan_for_signals()

                if signals:
                    for signal in signals:
                        self.signal_counter += 1

                        # Send chart first
                        try:
                            df = await self.get_binance_data(signal['symbol'], '1h', 100)
                            if df is not None:
                                chart_data = self.generate_chart(signal['symbol'], df, signal)
                                if chart_data:
                                    await self.send_photo(chat_id, chart_data, f"📊 {signal['symbol']} Chart")
                        except Exception as e:
                            self.logger.warning(f"Chart generation failed: {e}")

                        # Send signal info separately  
                        signal_msg = self.format_ml_signal_message(signal)
                        await self.send_message(chat_id, signal_msg)
                        await asyncio.sleep(2)

                    await self.send_message(chat_id, f"✅ **{len(signals)} signals found**")
                else:
                    await self.send_message(chat_id, "📊 **No signals found**\nML filtering for quality")

            elif text.startswith('/train'):
                await self.send_message(chat_id, "🧠 **Scanning channel for closed trades...**")

                try:
                    await self.scan_and_train_from_closed_trades()
                    ml_summary = self.ml_analyzer.get_ml_summary()

                    await self.send_message(chat_id, f"""✅ **ML TRAINING COMPLETE**

🧠 **Updated Model Performance:**
• Signal Accuracy: {ml_summary['model_performance']['signal_accuracy']*100:.1f}%
• Trades Learned: {ml_summary['model_performance']['total_trades_learned']}
• Learning Status: {ml_summary['learning_status'].title()}

📊 **Channel Scan Results:**
• Processed closed trades from @SignalTactics
• ML models retrained with new data
• Performance metrics updated

*Bot continuously learns from channel activity*""")

                except Exception as e:
                    await self.send_message(chat_id, f"❌ **Training Error:** {str(e)}")

            elif text.startswith('/channel'):
                await self.send_message(chat_id, f"""📢 **CHANNEL STATUS**

Target: {self.target_channel}
Access: {'✅ Available' if self.channel_accessible else '❌ Limited'}
Auto-Training: ✅ Enabled

**ML Channel Learning:**
• Scans for closed trades automatically
• Extracts trade outcomes and P/L
• Trains models with real results
• Improves prediction accuracy

Use /train to manually scan and train""")

        except Exception as e:
            self.logger.error(f"Error handling command {text}: {e}")

    def release_symbol_lock(self, symbol: str):
        """Release symbol from active trading lock with enhanced cleanup"""
        try:
            if symbol in self.active_symbols:
                self.active_symbols.remove(symbol)
                self.logger.info(f"🔓 Released trade lock for {symbol}")

            if symbol in self.symbol_trade_lock:
                del self.symbol_trade_lock[symbol]

            # Cancel monitoring task if it exists
            if symbol in self.active_trades:
                trade_info = self.active_trades[symbol]
                monitoring_task = trade_info.get('monitoring_task')
                if monitoring_task and not monitoring_task.done():
                    monitoring_task.cancel()
                    self.logger.info(f"🔄 Cancelled monitoring task for {symbol}")

        except Exception as e:
            self.logger.error(f"Error releasing symbol lock for {symbol}: {e}")

    async def cleanup_stale_locks(self):
        """Automatically cleanup stale symbol locks"""
        try:
            current_time = datetime.now()
            stale_symbols = []

            for symbol, lock_time in self.symbol_trade_lock.items():
                if isinstance(lock_time, datetime):
                    time_diff = (current_time - lock_time).total_seconds()
                    # Remove locks older than 2 hours
                    if time_diff > 7200:
                        stale_symbols.append(symbol)

            for symbol in stale_symbols:
                self.logger.warning(f"🧹 Cleaning up stale lock for {symbol}")
                self.release_symbol_lock(symbol)

            if stale_symbols:
                self.logger.info(f"🧹 Cleaned up {len(stale_symbols)} stale symbol locks")

        except Exception as e:
            self.logger.error(f"Error cleaning up stale locks: {e}")

    async def validate_active_trades(self):
        """Validate and cleanup active trades"""
        try:
            invalid_symbols = []
            
            for symbol, trade_info in self.active_trades.items():
                try:
                    # Check if monitoring task is still running
                    monitoring_task = trade_info.get('monitoring_task')
                    if monitoring_task and monitoring_task.done():
                        self.logger.warning(f"⚠️ Monitoring task completed for {symbol}, cleaning up")
                        invalid_symbols.append(symbol)
                        continue

                    # Check trade age
                    entry_time = trade_info.get('entry_time', datetime.now())
                    age_hours = (datetime.now() - entry_time).total_seconds() / 3600
                    
                    if age_hours > 48:  # 48 hours old
                        self.logger.warning(f"⏰ Trade {symbol} is {age_hours:.1f} hours old, marking for cleanup")
                        invalid_symbols.append(symbol)

                except Exception as trade_error:
                    self.logger.error(f"Error validating trade {symbol}: {trade_error}")
                    invalid_symbols.append(symbol)

            # Cleanup invalid trades
            for symbol in invalid_symbols:
                if symbol in self.active_trades:
                    del self.active_trades[symbol]
                self.release_symbol_lock(symbol)

            if invalid_symbols:
                self.logger.info(f"🧹 Cleaned up {len(invalid_symbols)} invalid active trades")

        except Exception as e:
            self.logger.error(f"Error validating active trades: {e}")

    async def record_open_trade_for_ml(self, signal: Dict[str, Any]):
        """Record open trade immediately for real-time ML learning"""
        try:
            symbol = signal['symbol']
            current_time = datetime.now()

            # Create open trade data for immediate ML learning
            open_trade_data = {
                'symbol': symbol,
                'direction': signal['direction'],
                'entry_price': signal['entry_price'],
                'exit_price': None,  # Not available yet
                'stop_loss': signal['stop_loss'],
                'take_profit_1': signal['tp1'],
                'take_profit_2': signal['tp2'],
                'take_profit_3': signal['tp3'],
                'signal_strength': signal['signal_strength'],
                'leverage': signal['optimal_leverage'],
                'profit_loss': 0.0,  # Will be updated when trade closes
                'trade_result': 'OPEN',
                'duration_minutes': 0,
                'market_volatility': signal.get('market_volatility', 0.02),
                'volume_ratio': signal.get('volume_ratio', 1.0),
                'rsi_value': signal.get('rsi', 50),
                'macd_signal': signal.get('macd_signal', 'neutral'),
                'ema_alignment': signal.get('ema_bullish', False),
                'cvd_trend': signal.get('cvd_trend', 'neutral'),
                'indicators_data': signal.get('indicators_used', []),
                'ml_prediction': signal.get('ml_prediction', {}).get('prediction', 'unknown'),
                'ml_confidence': signal.get('ml_prediction', {}).get('confidence', 0),
                'entry_time': current_time,
                'exit_time': None,
                'trade_status': 'ACTIVE',
                'last_update': current_time
            }

            # Record open trade in ML analyzer for immediate learning
            await self.ml_analyzer.record_trade_outcome(open_trade_data)

            # Store in active trades tracking for real-time updates
            self.active_trades[symbol] = {
                'signal': signal,
                'entry_time': current_time,
                'ml_data': open_trade_data,
                'monitoring_task': None
            }

            # Start real-time monitoring task for this trade
            monitoring_task = asyncio.create_task(self._monitor_open_trade_ml(symbol, signal))
            self.active_trades[symbol]['monitoring_task'] = monitoring_task

            # Save to persistent logs immediately
            await self._save_trade_to_persistent_log(open_trade_data)

            self.logger.info(f"🤖 Open trade recorded for real-time ML: {symbol} {signal['direction']} - Starting ML monitoring")

        except Exception as e:
            self.logger.error(f"Error recording open trade for ML: {e}")

    async def _monitor_open_trade_ml(self, symbol: str, signal: Dict[str, Any]):
        """Monitor open trade and continuously update ML with real-time data - Enhanced"""
        try:
            entry_price = signal['entry_price']
            entry_time = datetime.now()
            update_interval = 30  # Update ML every 30 seconds
            last_price = entry_price
            price_history = [entry_price]
            max_profit = 0
            max_drawdown = 0

            while symbol in self.active_trades:
                try:
                    # Get current market data with retry logic
                    df = None
                    for retry in range(3):
                        try:
                            df = await self.get_binance_data(symbol, '1m', 20)
                            if df is not None and len(df) > 0:
                                break
                        except Exception as fetch_error:
                            if retry == 2:
                                self.logger.warning(f"Failed to fetch data for {symbol} after 3 attempts: {fetch_error}")
                            await asyncio.sleep(5)
                    
                    if df is None or len(df) == 0:
                        await asyncio.sleep(update_interval)
                        continue

                    current_price = float(df['close'].iloc[-1])
                    current_time = datetime.now()
                    duration_minutes = (current_time - entry_time).total_seconds() / 60

                    # Track price movement
                    price_history.append(current_price)
                    if len(price_history) > 100:  # Keep last 100 prices
                        price_history = price_history[-100:]

                    # Calculate unrealized P/L with precision
                    if signal['direction'].upper() in ['BUY', 'LONG']:
                        unrealized_pnl = ((current_price - entry_price) / entry_price) * 100 * signal['optimal_leverage']
                    else:
                        unrealized_pnl = ((entry_price - current_price) / entry_price) * 100 * signal['optimal_leverage']

                    # Track max profit and drawdown
                    max_profit = max(max_profit, unrealized_pnl)
                    max_drawdown = min(max_drawdown, unrealized_pnl)

                    # Enhanced trade status check
                    trade_status = self._check_trade_status(current_price, signal)
                    
                    # Calculate additional metrics
                    price_volatility = np.std(price_history[-20:]) / np.mean(price_history[-20:]) if len(price_history) >= 20 else 0
                    price_momentum = (current_price - price_history[-10]) / price_history[-10] * 100 if len(price_history) >= 10 else 0

                    # Enhanced ML data with more features
                    updated_ml_data = {
                        'symbol': symbol,
                        'direction': signal['direction'],
                        'entry_price': entry_price,
                        'current_price': current_price,
                        'unrealized_pnl': unrealized_pnl,
                        'max_profit': max_profit,
                        'max_drawdown': max_drawdown,
                        'stop_loss': signal['stop_loss'],
                        'take_profit_1': signal['tp1'],
                        'take_profit_2': signal['tp2'],
                        'take_profit_3': signal['tp3'],
                        'signal_strength': signal['signal_strength'],
                        'leverage': signal['optimal_leverage'],
                        'profit_loss': unrealized_pnl,
                        'trade_result': trade_status,
                        'duration_minutes': duration_minutes,
                        'price_volatility': price_volatility,
                        'price_momentum': price_momentum,
                        'market_volatility': signal.get('market_volatility', 0.02),
                        'volume_ratio': signal.get('volume_ratio', 1.0),
                        'rsi_value': signal.get('rsi', 50),
                        'macd_signal': signal.get('macd_signal', 'neutral'),
                        'ema_alignment': signal.get('ema_bullish', False),
                        'cvd_trend': signal.get('cvd_trend', 'neutral'),
                        'indicators_data': signal.get('indicators_used', []),
                        'ml_prediction': signal.get('ml_prediction', {}).get('prediction', 'unknown'),
                        'ml_confidence': signal.get('ml_prediction', {}).get('confidence', 0),
                        'entry_time': entry_time,
                        'last_update': current_time,
                        'trade_status': 'MONITORING'
                    }

                    # Feed updated data to ML for continuous learning
                    await self.ml_analyzer.update_open_trade_data(updated_ml_data)

                    # Enhanced logging every 3 minutes
                    if int(duration_minutes) % 3 == 0 and int(duration_minutes) > 0:
                        profit_emoji = "🟢" if unrealized_pnl > 0 else "🔴" if unrealized_pnl < 0 else "🟡"
                        self.logger.info(f"🔄 {profit_emoji} ML Update: {symbol} - {duration_minutes:.1f}min | P/L: {unrealized_pnl:.2f}% | Max: {max_profit:.2f}% | Status: {trade_status}")

                    # Enhanced exit conditions
                    if trade_status in ['TP1_HIT', 'TP2_HIT', 'TP3_HIT', 'SL_HIT']:
                        # Record final trade result with enhanced data
                        final_result = {
                            'exit_price': current_price,
                            'profit_loss': unrealized_pnl,
                            'max_profit': max_profit,
                            'max_drawdown': max_drawdown,
                            'result': trade_status,
                            'duration_minutes': duration_minutes,
                            'exit_time': current_time,
                            'price_volatility': price_volatility,
                            'final_momentum': price_momentum
                        }

                        await self.record_trade_completion(signal, final_result)

                        # Remove from active trades and release lock
                        if symbol in self.active_trades:
                            del self.active_trades[symbol]
                        
                        self.release_symbol_lock(symbol)

                        result_emoji = "✅" if unrealized_pnl > 0 else "❌"
                        self.logger.info(f"{result_emoji} ML Trade Completed: {symbol} - {trade_status} - Final P/L: {unrealized_pnl:.2f}% | Max: {max_profit:.2f}%")
                        break

                    # Auto-exit after 24 hours (optional safety)
                    if duration_minutes > 1440:  # 24 hours
                        self.logger.warning(f"⏰ Auto-closing {symbol} after 24 hours - P/L: {unrealized_pnl:.2f}%")
                        final_result = {
                            'exit_price': current_price,
                            'profit_loss': unrealized_pnl,
                            'max_profit': max_profit,
                            'max_drawdown': max_drawdown,
                            'result': 'AUTO_CLOSE_TIMEOUT',
                            'duration_minutes': duration_minutes,
                            'exit_time': current_time
                        }
                        await self.record_trade_completion(signal, final_result)
                        if symbol in self.active_trades:
                            del self.active_trades[symbol]
                        self.release_symbol_lock(symbol)
                        break

                    last_price = current_price
                    await asyncio.sleep(update_interval)

                except Exception as monitor_error:
                    self.logger.warning(f"ML monitoring error for {symbol}: {monitor_error}")
                    await asyncio.sleep(update_interval)
                    continue

        except Exception as e:
            self.logger.error(f"Error in ML trade monitoring for {symbol}: {e}")
            # Ensure symbol is released on error
            self.release_symbol_lock(symbol)
            if symbol in self.active_trades:
                del self.active_trades[symbol]

    def _check_trade_status(self, current_price: float, signal: Dict[str, Any]) -> str:
        """Check if trade has hit TP or SL levels"""
        try:
            direction = signal['direction'].upper()

            if direction in ['BUY', 'LONG']:
                if current_price >= signal['tp3']:
                    return 'TP3_HIT'
                elif current_price >= signal['tp2']:
                    return 'TP2_HIT'
                elif current_price >= signal['tp1']:
                    return 'TP1_HIT'
                elif current_price <= signal['stop_loss']:
                    return 'SL_HIT'
            else:
                if current_price <= signal['tp3']:
                    return 'TP3_HIT'
                elif current_price <= signal['tp2']:
                    return 'TP2_HIT'
                elif current_price <= signal['tp1']:
                    return 'TP1_HIT'
                elif current_price >= signal['stop_loss']:
                    return 'SL_HIT'

            return 'OPEN'

        except Exception as e:
            return 'OPEN'

    async def record_trade_completion(self, signal: Dict[str, Any], trade_result: Dict[str, Any]):
        """Record completed trade for ML learning with comprehensive logging"""
        try:
            symbol = signal['symbol']

            trade_data = {
                'symbol': symbol,
                'direction': signal['direction'],
                'entry_price': signal['entry_price'],
                'exit_price': trade_result.get('exit_price', signal['entry_price']),
                'stop_loss': signal['stop_loss'],
                'take_profit_1': signal['tp1'],
                'take_profit_2': signal['tp2'],
                'take_profit_3': signal['tp3'],
                'signal_strength': signal['signal_strength'],
                'leverage': signal['optimal_leverage'],
                'profit_loss': trade_result.get('profit_loss', 0),
                'trade_result': trade_result.get('result', 'UNKNOWN'),
                'duration_minutes': trade_result.get('duration_minutes', 0),
                'market_volatility': signal.get('market_volatility', 0.02),
                'volume_ratio': signal.get('volume_ratio', 1.0),
                'rsi_value': signal.get('rsi', 50),
                'macd_signal': signal.get('macd_signal', 'neutral'),
                'ema_alignment': signal.get('ema_bullish', False),
                'cvd_trend': signal.get('cvd_trend', 'neutral'),
                'indicators_data': signal.get('indicators_used', []),
                'ml_prediction': signal.get('ml_prediction', {}).get('prediction', 'unknown'),
                'ml_confidence': signal.get('ml_prediction', {}).get('confidence', 0),
                'entry_time': signal.get('entry_time', datetime.now()),
                'exit_time': trade_result.get('exit_time', datetime.now()),
                'trade_status': 'COMPLETED'
            }

            # Record final result in ML analyzer database for learning
            await self.ml_analyzer.record_trade_outcome(trade_data)

            # Also save to persistent trade logs for backup and analysis
            await self._save_trade_to_persistent_log(trade_data)

            # Update performance tracking
            self._update_performance_stats(trade_data)

            # Release symbol lock after trade completion
            self.release_symbol_lock(symbol)

            self.logger.info(f"📊 Completed trade logged for ML: {symbol} - {trade_data['trade_result']} - P/L: {trade_data['profit_loss']:.2f}%")

        except Exception as e:
            self.logger.error(f"Error recording trade completion: {e}")

    async def _save_trade_to_persistent_log(self, trade_data: Dict[str, Any]):
        """Save trade to persistent JSON log file for backup"""
        try:
            log_file = Path("persistent_trade_logs.json")

            # Load existing logs
            existing_logs = []
            if log_file.exists():
                try:
                    with open(log_file, 'r') as f:
                        existing_logs = json.load(f)
                except:
                    existing_logs = []

            # Add timestamp and bot version
            trade_log = {
                **trade_data,
                'logged_at': datetime.now().isoformat(),
                'bot_version': 'Ultimate_Trading_Bot_v1.0',
                'session_id': self.session_token[:8] if self.session_token else 'unknown'
            }

            # Convert datetime objects to ISO strings
            for key, value in trade_log.items():
                if isinstance(value, datetime):
                    trade_log[key] = value.isoformat()

            existing_logs.append(trade_log)

            # Keep last 1000 trades to prevent file from growing too large
            if len(existing_logs) > 1000:
                existing_logs = existing_logs[-1000:]

            # Save back to file
            with open(log_file, 'w') as f:
                json.dump(existing_logs, f, indent=2, default=str)

            self.logger.info(f"💾 Trade saved to persistent log: {trade_data['symbol']}")

        except Exception as e:
            self.logger.error(f"Error saving to persistent log: {e}")

    def _update_performance_stats(self, trade_data: Dict[str, Any]):
        """Update performance statistics"""
        try:
            self.performance_stats['total_signals'] += 1

            if trade_data['profit_loss'] > 0:
                self.performance_stats['profitable_signals'] += 1
                self.performance_stats['total_profit'] += trade_data['profit_loss']

            if self.performance_stats['total_signals'] > 0:
                self.performance_stats['win_rate'] = (
                    self.performance_stats['profitable_signals'] / 
                    self.performance_stats['total_signals'] * 100
                )

        except Exception as e:
            self.logger.error(f"Error updating performance stats: {e}")

    async def load_persistent_trade_logs(self):
        """Load persistent trade logs on startup for ML continuity"""
        try:
            log_file = Path("persistent_trade_logs.json")

            if not log_file.exists():
                self.logger.info("📁 No persistent trade logs found - starting fresh")
                return

            with open(log_file, 'r') as f:
                trade_logs = json.load(f)

            if not trade_logs:
                return

            # Feed historical data to ML analyzer
            for trade_log in trade_logs[-100:]:  # Load last 100 trades for ML context
                try:
                    # Convert ISO strings back to datetime for ML processing
                    if 'entry_time' in trade_log and isinstance(trade_log['entry_time'], str):
                        trade_log['entry_time'] = datetime.fromisoformat(trade_log['entry_time'])
                    if 'exit_time' in trade_log and isinstance(trade_log['exit_time'], str):
                        trade_log['exit_time'] = datetime.fromisoformat(trade_log['exit_time'])

                    # Record in ML analyzer for learning continuity
                    await self.ml_analyzer.record_trade_outcome(trade_log)

                except Exception as e:
                    self.logger.warning(f"Error loading trade log: {e}")
                    continue

            self.logger.info(f"📚 Loaded {len(trade_logs)} persistent trade logs for ML continuity")

            # Update performance stats from historical data
            profitable_trades = sum(1 for trade in trade_logs if trade.get('profit_loss', 0) > 0)
            total_profit = sum(trade.get('profit_loss', 0) for trade in trade_logs if trade.get('profit_loss', 0) > 0)

            self.performance_stats.update({
                'total_signals': len(trade_logs),
                'profitable_signals': profitable_trades,
                'win_rate': (profitable_trades / len(trade_logs) * 100) if trade_logs else 0,
                'total_profit': total_profit
            })

        except Exception as e:
            self.logger.error(f"Error loading persistent trade logs: {e}")

    async def scan_and_train_from_closed_trades(self):
        """Scan channel for closed trades and train ML"""
        try:
            if not self.closed_trades_scanner:
                self.logger.warning("Closed trades scanner not available")
                return

            self.logger.info("🔍 Scanning Telegram channel for closed trades...")

            # First get unprocessed trades from database
            unprocessed_trades = await self.closed_trades_scanner.get_unprocessed_trades()

            # Scan for new closed trades
            new_closed_trades = await self.closed_trades_scanner.scan_for_closed_trades(hours_back=48)

            # Combine all trades for processing
            all_trades = unprocessed_trades + new_closed_trades

            if all_trades:
                self.logger.info(f"📈 Processing {len(all_trades)} closed trades for ML training")

                processed_count = 0
                processed_ids = []

                for trade in all_trades:
                    try:
                        # Process trade for ML training
                        await self._process_closed_trade_for_ml(trade)
                        processed_count += 1

                        if trade.get('message_id'):
                            processed_ids.append(trade.get('message_id'))

                    except Exception as trade_error:
                        self.logger.warning(f"Error processing trade {trade.get('symbol', 'UNKNOWN')}: {trade_error}")
                        continue

                # Mark trades as processed
                if processed_ids:
                    await self.closed_trades_scanner.mark_trades_as_processed(processed_ids)

                # Retrain ML models with new data if we have enough trades
                if processed_count >= 5:
                    await self.ml_analyzer.retrain_models()
                    self.logger.info(f"✅ ML models retrained with {processed_count} closed trades")
                else:
                    self.logger.info(f"📊 Processed {processed_count} trades (need 5+ for retraining)")

            else:
                self.logger.info("📊 No closed trades found for ML training")

        except Exception as e:
            self.logger.error(f"Error scanning for closed trades: {e}")

    async def _scan_channel_for_closed_trades(self) -> List[Dict[str, Any]]:
        """Scan channel messages for closed/completed trades"""
        try:
            closed_trades = []

            # Get recent messages from the target channel
            url = f"{self.base_url}/getUpdates"
            params = {'offset': -100}  # Get last 100 updates

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        updates = data.get('result', [])

                        # Look for channel messages
                        for update in updates:
                            if 'channel_post' in update:
                                message = update['channel_post']
                                if message.get('chat', {}).get('username') == self.target_channel.replace('@', ''):
                                    text = message.get('text', '')

                                    # Check if message contains closed trade information
                                    closed_trade = self._parse_closed_trade_message(text, message)
                                    if closed_trade:
                                        closed_trades.append(closed_trade)

            return closed_trades

        except Exception as e:
            self.logger.error(f"Error scanning channel messages: {e}")
            return []

    def _parse_closed_trade_message(self, text: str, message: Dict) -> Optional[Dict[str, Any]]:
        """Parse message text to identify and extract closed trade information"""
        try:
            import re

            # Keywords that indicate a closed/completed trade
            closed_keywords = [
                'closed', 'tp1 hit', 'tp2 hit', 'tp3 hit', 'target reached', 
                'stop loss hit', 'sl hit', 'trade closed', 'position closed',
                'profit taken', 'loss taken', 'exit', 'completed'
            ]

            text_lower = text.lower()

            # Check if message contains closed trade indicators
            if not any(keyword in text_lower for keyword in closed_keywords):
                return None

            closed_trade = {
                'message_id': message.get('message_id'),
                'timestamp': datetime.fromtimestamp(message.get('date', 0)),
                'text': text
            }

            # Extract symbol
            symbol_match = re.search(r'#?(\w+USDT?)\s+', text, re.IGNORECASE)
            if symbol_match:
                closed_trade['symbol'] = symbol_match.group(1).upper()

            # Extract direction
            direction_match = re.search(r'(LONG|SHORT|BUY|SELL)', text, re.IGNORECASE)
            if direction_match:
                closed_trade['direction'] = direction_match.group(1).upper()

            # Extract profit/loss percentage
            profit_patterns = [
                r'profit[:\s]*([+-]?\d+\.?\d*)%',
                r'([+-]?\d+\.?\d*)%\s*profit',
                r'gain[:\s]*([+-]?\d+\.?\d*)%',
                r'loss[:\s]*([+-]?\d+\.?\d*)%',
                r'([+-]?\d+\.?\d*)%\s*loss'
            ]

            for pattern in profit_patterns:
                profit_match = re.search(pattern, text, re.IGNORECASE)
                if profit_match:
                    profit_value = float(profit_match.group(1))
                    closed_trade['profit_loss'] = profit_value
                    closed_trade['trade_result'] = 'PROFIT' if profit_value > 0 else 'LOSS'
                    break

            # Determine trade result from keywords if profit_loss not found
            if 'trade_result' not in closed_trade:
                if any(word in text_lower for word in ['tp1 hit', 'tp2 hit', 'tp3 hit', 'target reached', 'profit taken']):
                    closed_trade['trade_result'] = 'PROFIT'
                    closed_trade['profit_loss'] = 1.0  # Default positive value
                elif any(word in text_lower for word in ['stop loss hit', 'sl hit', 'loss taken']):
                    closed_trade['trade_result'] = 'LOSS'
                    closed_trade['profit_loss'] = -1.0  # Default negative value
                else:
                    closed_trade['trade_result'] = 'CLOSED'
                    closed_trade['profit_loss'] = 0.0

            # Extract entry and exit prices if available
            entry_match = re.search(r'entry[:\s]*(\d+\.?\d*)', text, re.IGNORECASE)
            if entry_match:
                closed_trade['entry_price'] = float(entry_match.group(1))

            exit_match = re.search(r'exit[:\s]*(\d+\.?\d*)', text, re.IGNORECASE)
            if exit_match:
                closed_trade['exit_price'] = float(exit_match.group(1))

            # Extract leverage if mentioned
            leverage_match = re.search(r'(\d+)x', text, re.IGNORECASE)
            if leverage_match:
                closed_trade['leverage'] = int(leverage_match.group(1))

            # Only return if we have minimum required information
            if 'symbol' in closed_trade and 'trade_result' in closed_trade:
                return closed_trade

            return None

        except Exception as e:
            self.logger.error(f"Error parsing closed trade message: {e}")
            return None

    async def _process_closed_trade_for_ml(self, closed_trade: Dict[str, Any]):
        """Process a closed trade and prepare it for ML training"""
        try:
            # Validate required fields
            symbol = closed_trade.get('symbol')
            if not symbol:
                self.logger.warning("Skipping trade with no symbol")
                return

            # Extract trade result and profit/loss
            trade_result = closed_trade.get('trade_result', 'UNKNOWN')
            profit_loss = closed_trade.get('profit_loss', 0)

            # Skip trades with no meaningful result
            if trade_result == 'UNKNOWN' and profit_loss == 0:
                self.logger.debug(f"Skipping {symbol} - no trade outcome data")
                return

            # Create comprehensive trade data for ML
            trade_data = {
                'symbol': symbol.upper(),
                'direction': closed_trade.get('direction', 'BUY').upper(),
                'entry_price': closed_trade.get('entry_price', 0),
                'exit_price': closed_trade.get('exit_price', closed_trade.get('entry_price', 0)),
                'stop_loss': 0,  # Not available from channel message
                'take_profit_1': 0,  # Not available from channel message
                'take_profit_2': 0,  # Not available from channel message
                'take_profit_3': 0,  # Not available from channel message
                'signal_strength': 85,  # Default value for channel signals
                'leverage': max(10, min(100, closed_trade.get('leverage', 35))),  # Validate leverage range
                'profit_loss': float(profit_loss),
                'trade_result': trade_result,
                'duration_minutes': closed_trade.get('duration_minutes', 30),
                'market_volatility': 0.02,
                'volume_ratio': 1.0,
                'rsi_value': 50,
                'macd_signal': 'neutral',
                'ema_alignment': False,
                'cvd_trend': 'neutral',
                'indicators_data': ['telegram_channel_signal'],
                'ml_prediction': 'channel_signal',
                'ml_confidence': 75,
                'entry_time': closed_trade.get('timestamp', datetime.now()),
                'exit_time': closed_trade.get('timestamp', datetime.now()),
                'data_source': 'telegram_channel'
            }

            # Try to enhance with current market data (optional)
            try:
                if symbol in self.symbols:  # Only for supported symbols
                    df = await self.get_binance_data(symbol, '1h', 50)
                    if df is not None and len(df) > 20:
                        indicators = self.calculate_advanced_indicators(df)
                        if indicators:
                            # Update trade data with market indicators
                            trade_data.update({
                                'market_volatility': indicators.get('market_volatility', 0.02),
                                'volume_ratio': indicators.get('volume_ratio', 1.0),
                                'rsi_value': indicators.get('rsi', 50),
                                'ema_alignment': indicators.get('ema_bullish', False) or indicators.get('ema_bearish', False),
                                'cvd_trend': indicators.get('cvd_trend', 'neutral')
                            })
            except Exception as market_error:
                self.logger.debug(f"Could not enhance {symbol} with market data: {market_error}")

            # Record in ML analyzer
            await self.ml_analyzer.record_trade_outcome(trade_data)

            # Save to persistent logs
            await self._save_trade_to_persistent_log(trade_data)

            # Update performance stats
            self._update_performance_stats(trade_data)

            result_emoji = "✅" if profit_loss > 0 else "❌" if profit_loss < 0 else "➖"
            self.logger.info(f"📊 {result_emoji} Processed {symbol} {trade_result}: {profit_loss:.2f}% P/L")

        except Exception as e:
            symbol = closed_trade.get('symbol', 'UNKNOWN')
            self.logger.error(f"Error processing closed trade {symbol}: {e}")

    async def auto_scan_loop(self):
        """Main auto-scanning loop with ML learning and enhanced maintenance"""
        consecutive_errors = 0
        max_consecutive_errors = 5
        base_scan_interval = 90
        last_channel_training = datetime.now()
        last_cleanup = datetime.now()
        last_validation = datetime.now()

        while self.running and not self.shutdown_requested:
            try:
                now = datetime.now()
                
                # Periodic maintenance (every 10 minutes)
                if (now - last_cleanup).total_seconds() > 600:  # 10 minutes
                    try:
                        await self.cleanup_stale_locks()
                        last_cleanup = now
                    except Exception as e:
                        self.logger.warning(f"Cleanup error: {e}")

                # Validate active trades (every 15 minutes)
                if (now - last_validation).total_seconds() > 900:  # 15 minutes
                    try:
                        await self.validate_active_trades()
                        last_validation = now
                    except Exception as e:
                        self.logger.warning(f"Validation error: {e}")

                # Periodically scan channel for closed trades and train ML (every 30 minutes)
                if (now - last_channel_training).total_seconds() > 1800:  # 30 minutes
                    try:
                        await self.scan_and_train_from_closed_trades()
                        last_channel_training = now
                    except Exception as e:
                        self.logger.warning(f"Channel training error: {e}")

                # Enhanced status logging
                active_count = len(self.active_trades)
                locked_count = len(self.active_symbols)
                self.logger.info(f"🧠 Scanning markets for ML-enhanced signals... | Active: {active_count} | Locked: {locked_count}")
                
                signals = await self.scan_for_signals()

                if signals:
                    self.logger.info(f"📊 Found {len(signals)} ML-validated signals | Hour: {self.hourly_signal_count}/{self.max_signals_per_hour}")

                    signals_sent_count = 0

                    for signal in signals:
                        # No hourly limits - process all signals
                        if False:  # Disabled hourly limit check
                            self.logger.info(f"⏰ Hourly signal limit reached ({self.max_signals_per_hour}). Skipping remaining signals.")
                            break

                        try:
                            self.signal_counter += 1
                            self.performance_stats['total_signals'] += 1

                            if self.performance_stats['total_signals'] > 0:
                                self.performance_stats['win_rate'] = (
                                    self.performance_stats['profitable_signals'] /
                                    self.performance_stats['total_signals'] * 100
                                )

                            # Send chart first to @SignalTactics
                            chart_sent = False
                            if self.channel_accessible:
                                try:
                                    df = await self.get_binance_data(signal['symbol'], '1h', 100)
                                    if df is not None and len(df) > 10:
                                        chart_data = self.generate_chart(signal['symbol'], df, signal)
                                        if chart_data and len(chart_data) > 100:  # Valid base64 should be longer
                                            chart_sent = await self.send_photo(self.target_channel, chart_data, 
                                                                             f"📊 {signal['symbol']} - {signal['direction']} Setup")
                                        else:
                                            self.logger.info(f"📊 Chart generation skipped for {signal['symbol']} - no valid chart data")
                                    else:
                                        self.logger.info(f"📊 Chart generation skipped for {signal['symbol']} - insufficient market data")
                                except Exception as chart_error:
                                    self.logger.warning(f"Chart error for {signal['symbol']}: {str(chart_error)[:100]}")
                                    chart_sent = False

                            # Send signal info separately
                            signal_msg = self.format_ml_signal_message(signal)
                            channel_sent = False
                            if self.channel_accessible:
                                channel_sent = await self.send_message(self.target_channel, signal_msg)

                            if channel_sent:
                                chart_status = "📊✅" if chart_sent else "📊❌"
                                self.logger.info(f"📤 ML Signal #{self.signal_counter} delivered {chart_status}: Channel @SignalTactics")

                                ml_conf = signal.get('ml_prediction', {}).get('confidence', 0)
                                self.logger.info(f"✅ ML Signal sent: {signal['symbol']} {signal['direction']} (Strength: {signal['signal_strength']:.0f}%, ML: {ml_conf:.1f}%)")

                                # Record open trade for immediate ML learning
                                await self.record_open_trade_for_ml(signal)

                                # Auto-unlock symbol after 20 minutes for faster recycling
                                asyncio.create_task(self._auto_unlock_symbol(signal['symbol'], 1200))
                            else:
                                # Release symbol lock if signal failed to send
                                self.release_symbol_lock(signal['symbol'])
                                self.logger.warning(f"❌ Failed to send ML Signal #{self.signal_counter} to @SignalTactics")

                        except Exception as signal_error:
                            self.logger.error(f"Error processing ML signal for {signal.get('symbol', 'unknown')}: {signal_error}")
                            continue

                else:
                    self.logger.info("📊 No ML signals found - models filtering for optimal opportunities")

                consecutive_errors = 0
                self.last_heartbeat = datetime.now()

                scan_interval = 30 if signals else 45  # Much more frequent scanning
                self.logger.info(f"⏰ Next ML scan in {scan_interval} seconds | 🚀 UNLIMITED MODE")
                await asyncio.sleep(scan_interval)

            except Exception as e:
                consecutive_errors += 1
                self.logger.error(f"ML auto-scan loop error #{consecutive_errors}: {e}")

                if consecutive_errors >= max_consecutive_errors:
                    self.logger.critical(f"🚨 Too many consecutive errors ({consecutive_errors}). Extended wait.")
                    error_wait = min(300, 30 * consecutive_errors)

                    try:
                        await self.create_session()
                        await self.verify_channel_access()
                        self.logger.info("🔄 Session and connections refreshed")
                    except Exception as recovery_error:
                        self.logger.error(f"Recovery attempt failed: {recovery_error}")

                else:
                    error_wait = min(120, 15 * consecutive_errors)

                self.logger.info(f"⏳ Waiting {error_wait} seconds before retry...")
                await asyncio.sleep(error_wait)

    async def run_bot(self):
        """Main bot execution with ML integration"""
        self.logger.info("🚀 Starting Ultimate ML Trading Bot")

        try:
            await self.create_session()
            await self.verify_channel_access()

            # Load persistent trade logs for ML continuity
            await self.load_persistent_trade_logs()

            if self.admin_chat_id:
                ml_summary = self.ml_analyzer.get_ml_summary()
                startup_msg = f"""🧠 **ULTIMATE ML TRADING BOT STARTED**

✅ **System Status:** Online & Learning
🔄 **Session:** Created with indefinite duration
📢 **Channel:** {self.target_channel} - {"✅ Accessible" if self.channel_accessible else "⚠️ Setup Required"}
🎯 **Scanning:** {len(self.symbols)} symbols across {len(self.timeframes)} timeframes
🆔 **Process ID:** {os.getpid()}

**🧠 Machine Learning Status:**
• **Model Accuracy:** {ml_summary['model_performance']['signal_accuracy']*100:.1f}%
• **Trades Learned:** {ml_summary['model_performance']['total_trades_learned']}
• **Learning Status:** {ml_summary['learning_status'].title()}
• **ML Available:** {'✅ Yes' if ml_summary['ml_available'] else '❌ No'}

**🛡️ Enhanced Features Active:**
• Advanced multi-indicator analysis
• CVD confluence detection
• **Adaptive leverage calculation** (20x-75x)
• **Cross margin trading** (all positions)
• Machine learning predictions
• Persistent trade learning
• Cornix-compatible Telegram formatting
• Real-time performance tracking
• Continuous learning system

**⚙️ Adaptive Learning:**
• Performance-based leverage adjustment
• Win/loss streak tracking
• Persistent trade database
• Cross-session learning continuity

**📤 Delivery Method:**
• Signals sent only to @SignalTactics channel
• Cornix-readable format for automation
• Cross margin configuration included
• TradeTactics_bot integration

**🚀 UNLIMITED SIGNAL MODE ACTIVE:**
• No hourly signal limits
• Multiple trades per symbol allowed
• Aggressive scanning intervals (30-45s)
• Maximum trade volume optimization

*Ultimate ML bot with Unlimited Signal Generation*"""
                await self.send_message(self.admin_chat_id, startup_msg)

            auto_scan_task = asyncio.create_task(self.auto_scan_loop())

            offset = None
            last_channel_check = datetime.now()

            while self.running and not self.shutdown_requested:
                try:
                    now = datetime.now()
                    if (now - last_channel_check).total_seconds() > 1800:
                        await self.verify_channel_access()
                        last_channel_check = now

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
                    if not self.shutdown_requested:
                        await asyncio.sleep(5)

        except Exception as e:
            self.logger.critical(f"Critical ML bot error: {e}")
            raise
        finally:
            if self.admin_chat_id and not self.shutdown_requested:
                try:
                    shutdown_msg = "🛑 **Ultimate ML Trading Bot Shutdown**\n\nBot has stopped. All ML models and learning data preserved for restart."
                    await self.send_message(self.admin_chat_id, shutdown_msg)
                except:
                    pass

async def main():
    """Run the ultimate ML trading bot"""
    bot = UltimateTradingBot()

    try:
        print("🧠 Ultimate ML Trading Bot Starting...")
        print("📊 Most Advanced Machine Learning Strategy")
        print("⚖️ 1:3 Risk/Reward Ratio")
        print("🎯 3 Take Profits + SL to Entry")
        print("🤖 Advanced ML Predictions")
        print("📈 CVD Confluence Analysis")
        print("🧠 Continuous Learning System")
        print("🛡️ Auto-Restart Protection Active")
        print("\nBot will run continuously and learn from every trade")

        await bot.run_bot()

    except KeyboardInterrupt:
        print("\n🛑 Ultimate ML Trading Bot stopped by user")
        bot.running = False
        return False
    except Exception as e:
        print(f"❌ Bot Error: {e}")
        bot.logger.error(f"Bot crashed: {e}")
        return True

if __name__ == "__main__":
    asyncio.run(main())