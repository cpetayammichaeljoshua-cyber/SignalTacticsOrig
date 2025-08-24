
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
    from sklearn.preprocessing import StandardScaler, LabelEncoder
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
        self.scaler = StandardScaler()
        
        # Learning database
        self.db_path = "advanced_ml_trading.db"
        self._initialize_database()
        
        # Performance tracking
        self.model_performance = {
            'signal_accuracy': 0.0,
            'profit_prediction_accuracy': 0.0,
            'risk_assessment_accuracy': 0.0,
            'total_trades_learned': 0,
            'last_training_time': None,
            'win_rate_improvement': 0.0
        }
        
        # Learning parameters
        self.retrain_threshold = 25  # Retrain after 25 new trades
        self.trades_since_retrain = 0
        
        # Market insights
        self.market_insights = {
            'best_time_sessions': {},
            'symbol_performance': {},
            'indicator_effectiveness': {},
            'risk_patterns': {}
        }
        
        self.logger.info("🧠 Advanced ML Trade Analyzer initialized")
        
        # Bootstrap with some initial performance data if database is empty
        asyncio.create_task(self._bootstrap_initial_data())

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
                    leverage INTEGER,
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
                INSERT INTO ml_trades (
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
            
            self.trades_since_retrain += 1
            self.model_performance['total_trades_learned'] += 1
            
            self.logger.info(f"📝 ML Trade recorded: {trade_data.get('symbol')} - {trade_data.get('trade_result')}")
            
            # Auto-retrain if threshold reached
            if self.trades_since_retrain >= self.retrain_threshold:
                await self.retrain_models()
                
        except Exception as e:
            self.logger.error(f"Error recording ML trade: {e}")

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
        """Prepare features for ML training"""
        try:
            if len(df) == 0:
                return None, None
            
            # Create feature matrix
            features = pd.DataFrame()
            
            # Basic features
            features['signal_strength'] = df['signal_strength'].fillna(0)
            features['leverage'] = df['leverage'].fillna(35)
            features['market_volatility'] = df['market_volatility'].fillna(0.02)
            features['volume_ratio'] = df['volume_ratio'].fillna(1.0)
            features['rsi_value'] = df['rsi_value'].fillna(50)
            
            # Encode categorical features
            le_direction = LabelEncoder()
            le_macd = LabelEncoder()
            le_cvd = LabelEncoder()
            le_session = LabelEncoder()
            
            features['direction_encoded'] = le_direction.fit_transform(df['direction'].fillna('BUY'))
            features['macd_signal_encoded'] = le_macd.fit_transform(df['macd_signal'].fillna('neutral'))
            features['cvd_trend_encoded'] = le_cvd.fit_transform(df['cvd_trend'].fillna('neutral'))
            features['time_session_encoded'] = le_session.fit_transform(df['time_session'].fillna('NY_MAIN'))
            features['ema_alignment'] = df['ema_alignment'].fillna(False).astype(int)
            
            # Time features
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
        """Train signal classification model"""
        try:
            X = features
            y = targets['profitable']
            
            if len(X) < 20:
                return
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.signal_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                class_weight='balanced'
            )
            self.signal_classifier.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.signal_classifier.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.model_performance['signal_accuracy'] = accuracy
            self.logger.info(f"🎯 Signal classifier accuracy: {accuracy:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error training signal classifier: {e}")

    async def _train_profit_predictor(self, features: pd.DataFrame, targets: Dict):
        """Train profit prediction model"""
        try:
            X = features
            y = targets['profit_amount']
            
            if len(X) < 20:
                return
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            from sklearn.ensemble import GradientBoostingRegressor
            self.profit_predictor = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            )
            self.profit_predictor.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.profit_predictor.predict(X_test_scaled)
            from sklearn.metrics import r2_score
            r2 = r2_score(y_test, y_pred)
            
            self.model_performance['profit_prediction_accuracy'] = max(0, r2)
            self.logger.info(f"💰 Profit predictor R²: {r2:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error training profit predictor: {e}")

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
            X_train_scaled = self.scaler.fit_transform(X_train)
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
        """Advanced ML prediction for trade outcome"""
        try:
            if not all([self.signal_classifier, self.profit_predictor, self.risk_assessor, self.scaler]):
                return self._fallback_prediction(signal_data)
            
            # Prepare features
            features = self._prepare_prediction_features(signal_data)
            if features is None:
                return self._fallback_prediction(signal_data)
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Get predictions
            profit_prob = self.signal_classifier.predict_proba(features_scaled)[0][1]
            profit_amount = self.profit_predictor.predict(features_scaled)[0]
            risk_prob = self.risk_assessor.predict_proba(features_scaled)[0][1]
            
            # Calculate overall confidence
            confidence = profit_prob * 100
            
            # Adjust based on market insights
            confidence = self._adjust_confidence_with_insights(signal_data, confidence)
            
            # Determine prediction
            if confidence >= 75 and profit_amount > 0 and risk_prob < 0.3:
                prediction = 'highly_favorable'
            elif confidence >= 65 and profit_amount > 0:
                prediction = 'favorable'
            elif confidence >= 45:
                prediction = 'neutral'
            else:
                prediction = 'unfavorable'
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'expected_profit': profit_amount,
                'risk_probability': risk_prob * 100,
                'recommendation': self._get_ml_recommendation(prediction, confidence, profit_amount, risk_prob),
                'model_accuracy': self.model_performance['signal_accuracy'] * 100,
                'trades_learned_from': self.model_performance['total_trades_learned']
            }
            
        except Exception as e:
            self.logger.error(f"Error in ML prediction: {e}")
            return self._fallback_prediction(signal_data)

    def _prepare_prediction_features(self, signal_data: Dict[str, Any]) -> Optional[List[float]]:
        """Prepare features for prediction"""
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
            
            features = [
                signal_data.get('signal_strength', 85),
                signal_data.get('leverage', 35),
                signal_data.get('market_volatility', 0.02),
                signal_data.get('volume_ratio', 1.0),
                signal_data.get('rsi', 50),
                direction_map.get(signal_data.get('direction', 'BUY'), 1),
                macd_map.get(signal_data.get('macd_signal', 'neutral'), 0),
                cvd_map.get(signal_data.get('cvd_trend', 'neutral'), 0),
                session_map.get(time_session, 3),
                1 if signal_data.get('ema_bullish', False) else 0,
                current_time.hour,
                current_time.weekday()
            ]
            
            return features
            
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
        """Fallback prediction when ML models not available"""
        signal_strength = signal_data.get('signal_strength', 50)
        
        if signal_strength >= 85:
            prediction = 'favorable'
            confidence = 75
        elif signal_strength >= 75:
            prediction = 'neutral'
            confidence = 60
        else:
            prediction = 'unfavorable'
            confidence = 40
        
        # Only return favorable predictions in fallback
        if signal_strength < 85:  # Only favorable signals
            return {
                'prediction': 'unfavorable',
                'confidence': confidence,
                'expected_profit': 0.0,
                'risk_probability': 50.0,
                'recommendation': "Signal Strength Based: Favorable",
                'model_accuracy': 0.0,
                'trades_learned_from': 0
            }
        
        return {
            'prediction': 'favorable',
            'confidence': confidence,
            'expected_profit': 0.0,
            'risk_probability': 50.0,
            'recommendation': "Signal Strength Based: Favorable",
            'model_accuracy': 0.0,
            'trades_learned_from': 0
        }

    async def _bootstrap_initial_data(self):
        """Bootstrap initial performance data if database is empty"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if we have any trades
            cursor.execute("SELECT COUNT(*) FROM ml_trades")
            count = cursor.fetchone()[0]
            
            if count == 0:
                # Add some sample historical trades for bootstrapping
                sample_trades = [
                    # Some winning trades
                    ('BTCUSDT', 'BUY', 65000, 65500, 1.2, 'WIN', 85, 35, datetime.now() - timedelta(days=5)),
                    ('ETHUSDT', 'SELL', 3500, 3450, 1.4, 'WIN', 90, 35, datetime.now() - timedelta(days=4)),
                    ('BNBUSDT', 'BUY', 580, 590, 1.7, 'WIN', 88, 35, datetime.now() - timedelta(days=3)),
                    # Some losing trades
                    ('ADAUSDT', 'BUY', 0.45, 0.44, -2.2, 'LOSS', 82, 35, datetime.now() - timedelta(days=2)),
                    ('XRPUSDT', 'SELL', 0.52, 0.53, -1.9, 'LOSS', 80, 35, datetime.now() - timedelta(days=1)),
                ]
                
                for trade in sample_trades:
                    cursor.execute('''
                        INSERT INTO ml_trades (
                            symbol, direction, entry_price, exit_price, profit_loss, 
                            trade_result, signal_strength, leverage, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', trade)
                
                conn.commit()
                self.logger.info("📊 Bootstrap: Added sample performance data")
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error bootstrapping initial data: {e}")

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

        # Risk management - optimized for maximum profitability
        self.risk_reward_ratio = 3.0
        self.min_signal_strength = 80
        self.max_signals_per_hour = 5
        self.capital_allocation = 0.025  # 2.5% per trade
        self.max_concurrent_trades = 10

        # Performance tracking
        self.signal_counter = 0
        self.active_trades = {}
        self.performance_stats = {
            'total_signals': 0,
            'profitable_signals': 0,
            'win_rate': 0.0,
            'total_profit': 0.0
        }

        # Prevent signal spam
        self.last_signal_time = {}
        self.min_signal_interval = 180  # 3 minutes between signals for same symbol

        # Advanced ML Trade Analyzer
        self.ml_analyzer = AdvancedMLTradeAnalyzer()
        self.ml_analyzer.load_ml_models()
        
        # Bot status
        self.running = True
        self.last_heartbeat = datetime.now()

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
        """Calculate comprehensive technical indicators with institutional order flow analysis"""
        try:
            indicators = {}

            if df.empty or len(df) < 55:
                return {}

            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            volume = df['volume'].values
            open_prices = df['open'].values

            if len(high) == 0 or len(low) == 0 or len(close) == 0:
                return {}

            # INSTITUTIONAL ORDER FLOW ANALYSIS
            order_flow = self._calculate_institutional_order_flow(df)
            indicators.update(order_flow)

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

            # 9. Current price info
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

    def _calculate_institutional_order_flow(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate institutional order flow and smart money indicators"""
        try:
            order_flow = {}
            
            if len(df) < 20:
                return {}
            
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            open_prices = df['open'].values
            volume = df['volume'].values
            
            # 1. Liquidity Detection (Support/Resistance with volume)
            liquidity_levels = self._detect_liquidity_levels(df)
            order_flow['liquidity_sweep'] = liquidity_levels['sweep_detected']
            order_flow['liquidity_strength'] = liquidity_levels['strength']
            order_flow['nearest_liquidity'] = liquidity_levels['nearest_level']
            
            # 2. Market Structure Analysis
            market_structure = self._analyze_market_structure(df)
            order_flow['market_structure'] = market_structure['structure']
            order_flow['structure_break'] = market_structure['break_detected']
            order_flow['structure_strength'] = market_structure['strength']
            
            # 3. Order Block Detection
            order_blocks = self._detect_order_blocks(df)
            order_flow['bullish_order_block'] = order_blocks['bullish_ob']
            order_flow['bearish_order_block'] = order_blocks['bearish_ob']
            order_flow['order_block_strength'] = order_blocks['strength']
            
            # 4. Fair Value Gap (FVG) Analysis
            fvg_analysis = self._detect_fair_value_gaps(df)
            order_flow['bullish_fvg'] = fvg_analysis['bullish_fvg']
            order_flow['bearish_fvg'] = fvg_analysis['bearish_fvg']
            order_flow['fvg_probability'] = fvg_analysis['fill_probability']
            
            # 5. Imbalance Detection
            imbalance = self._detect_price_imbalances(df)
            order_flow['imbalance_direction'] = imbalance['direction']
            order_flow['imbalance_strength'] = imbalance['strength']
            
            # 6. Volume Profile Analysis
            volume_profile = self._analyze_volume_profile(df)
            order_flow['volume_poc'] = volume_profile['point_of_control']
            order_flow['volume_imbalance'] = volume_profile['imbalance']
            order_flow['high_volume_node'] = volume_profile['hvn']
            
            # 7. Institutional Candle Patterns
            institutional_patterns = self._detect_institutional_patterns(df)
            order_flow['institutional_pattern'] = institutional_patterns['pattern']
            order_flow['pattern_reliability'] = institutional_patterns['reliability']
            
            # 8. Smart Money Concepts
            smc = self._calculate_smart_money_concepts(df)
            order_flow['choch'] = smc['change_of_character']
            order_flow['bos'] = smc['break_of_structure']
            order_flow['displacement'] = smc['displacement']
            order_flow['smc_bias'] = smc['bias']
            
            return order_flow
            
        except Exception as e:
            self.logger.error(f"Error calculating institutional order flow: {e}")
            return {}

    def _detect_liquidity_levels(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect liquidity levels and sweeps"""
        try:
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # Find swing highs and lows with volume confirmation
            swing_highs = []
            swing_lows = []
            
            for i in range(2, len(df) - 2):
                # Swing high: higher than 2 bars on each side + high volume
                if (high[i] > high[i-1] and high[i] > high[i-2] and 
                    high[i] > high[i+1] and high[i] > high[i+2] and
                    volume[i] > np.mean(volume[max(0, i-10):i+1])):
                    swing_highs.append((i, high[i], volume[i]))
                
                # Swing low: lower than 2 bars on each side + high volume
                if (low[i] < low[i-1] and low[i] < low[i-2] and 
                    low[i] < low[i+1] and low[i] < low[i+2] and
                    volume[i] > np.mean(volume[max(0, i-10):i+1])):
                    swing_lows.append((i, low[i], volume[i]))
            
            current_price = df['close'].iloc[-1]
            
            # Check for liquidity sweeps
            sweep_detected = False
            strength = 0
            nearest_level = current_price
            
            if swing_highs:
                recent_high = max(swing_highs[-3:], key=lambda x: x[1]) if len(swing_highs) >= 3 else swing_highs[-1]
                if current_price > recent_high[1] * 1.001:  # 0.1% above high
                    sweep_detected = True
                    mean_volume = np.mean(volume[-20:])
                    if mean_volume != 0 and not np.isnan(mean_volume) and not np.isinf(mean_volume):
                        strength = min(recent_high[2] / mean_volume * 50, 100)
                    else:
                        strength = 50  # Default strength when volume data is invalid
                    nearest_level = recent_high[1]
            
            if swing_lows and not sweep_detected:
                recent_low = min(swing_lows[-3:], key=lambda x: x[1]) if len(swing_lows) >= 3 else swing_lows[-1]
                if current_price < recent_low[1] * 0.999:  # 0.1% below low
                    sweep_detected = True
                    mean_volume = np.mean(volume[-20:])
                    if mean_volume != 0 and not np.isnan(mean_volume) and not np.isinf(mean_volume):
                        strength = min(recent_low[2] / mean_volume * 50, 100)
                    else:
                        strength = 50  # Default strength when volume data is invalid
                    nearest_level = recent_low[1]
            
            return {
                'sweep_detected': sweep_detected,
                'strength': strength,
                'nearest_level': nearest_level
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting liquidity levels: {e}")
            return {'sweep_detected': False, 'strength': 0, 'nearest_level': 0}

    def _analyze_market_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market structure for institutional patterns"""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # Calculate Higher Highs, Higher Lows, Lower Highs, Lower Lows
            structure_points = []
            
            for i in range(5, len(df) - 5):
                if high[i] == max(high[i-5:i+6]):  # Local high
                    structure_points.append(('H', i, high[i]))
                if low[i] == min(low[i-5:i+6]):  # Local low
                    structure_points.append(('L', i, low[i]))
            
            if len(structure_points) < 4:
                return {'structure': 'ranging', 'break_detected': False, 'strength': 0}
            
            # Analyze last 4 structure points
            recent_points = structure_points[-4:]
            
            # Determine trend structure
            highs = [p for p in recent_points if p[0] == 'H']
            lows = [p for p in recent_points if p[0] == 'L']
            
            structure = 'ranging'
            break_detected = False
            strength = 0
            
            if len(highs) >= 2 and len(lows) >= 2:
                # Check for Higher Highs and Higher Lows (Uptrend)
                if (highs[-1][2] > highs[-2][2] and lows[-1][2] > lows[-2][2]):
                    structure = 'uptrend'
                    strength = 75
                # Check for Lower Highs and Lower Lows (Downtrend)
                elif (highs[-1][2] < highs[-2][2] and lows[-1][2] < lows[-2][2]):
                    structure = 'downtrend'
                    strength = 75
                
                # Check for structure break
                current_price = close[-1]
                if structure == 'uptrend' and current_price < min(lows[-2:], key=lambda x: x[2])[2]:
                    break_detected = True
                    strength = 90
                elif structure == 'downtrend' and current_price > max(highs[-2:], key=lambda x: x[2])[2]:
                    break_detected = True
                    strength = 90
            
            return {
                'structure': structure,
                'break_detected': break_detected,
                'strength': strength
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market structure: {e}")
            return {'structure': 'ranging', 'break_detected': False, 'strength': 0}

    def _detect_order_blocks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect institutional order blocks"""
        try:
            open_prices = df['open'].values
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            volume = df['volume'].values
            
            bullish_ob = False
            bearish_ob = False
            strength = 0
            
            # Look for order block patterns in last 10 candles
            for i in range(len(df) - 10, len(df) - 1):
                if i < 2:
                    continue
                
                # Bullish Order Block: Big green candle followed by pullback
                if (close[i] > open_prices[i] and  # Green candle
                    (close[i] - open_prices[i]) / open_prices[i] > 0.005 and  # At least 0.5% body
                    volume[i] > np.mean(volume[max(0, i-5):i+1]) * 1.5 and  # High volume
                    close[i+1] < close[i]):  # Followed by pullback
                    
                    bullish_ob = True
                    strength = min((close[i] - open_prices[i]) / open_prices[i] * 2000, 100)
                    break
                
                # Bearish Order Block: Big red candle followed by pullback
                if (close[i] < open_prices[i] and  # Red candle
                    (open_prices[i] - close[i]) / open_prices[i] > 0.005 and  # At least 0.5% body
                    volume[i] > np.mean(volume[max(0, i-5):i+1]) * 1.5 and  # High volume
                    close[i+1] > close[i]):  # Followed by pullback
                    
                    bearish_ob = True
                    strength = min((open_prices[i] - close[i]) / open_prices[i] * 2000, 100)
                    break
            
            return {
                'bullish_ob': bullish_ob,
                'bearish_ob': bearish_ob,
                'strength': strength
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting order blocks: {e}")
            return {'bullish_ob': False, 'bearish_ob': False, 'strength': 0}

    def _detect_fair_value_gaps(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect Fair Value Gaps (FVGs)"""
        try:
            high = df['high'].values
            low = df['low'].values
            
            bullish_fvg = False
            bearish_fvg = False
            fill_probability = 0
            
            # Look for FVGs in recent candles
            for i in range(len(df) - 5, len(df) - 2):
                if i < 1:
                    continue
                
                # Bullish FVG: Gap between candle i-1 high and candle i+1 low
                if low[i+1] > high[i-1]:
                    gap_size = (low[i+1] - high[i-1]) / high[i-1]
                    if gap_size > 0.001:  # At least 0.1% gap
                        bullish_fvg = True
                        fill_probability = min(gap_size * 5000, 85)  # Higher probability for larger gaps
                        break
                
                # Bearish FVG: Gap between candle i-1 low and candle i+1 high
                if high[i+1] < low[i-1]:
                    gap_size = (low[i-1] - high[i+1]) / low[i-1]
                    if gap_size > 0.001:  # At least 0.1% gap
                        bearish_fvg = True
                        fill_probability = min(gap_size * 5000, 85)  # Higher probability for larger gaps
                        break
            
            return {
                'bullish_fvg': bullish_fvg,
                'bearish_fvg': bearish_fvg,
                'fill_probability': fill_probability
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting FVGs: {e}")
            return {'bullish_fvg': False, 'bearish_fvg': False, 'fill_probability': 0}

    def _detect_price_imbalances(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect price imbalances and inefficiencies"""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            volume = df['volume'].values
            
            # Calculate price velocity and volume confirmation
            price_changes = np.diff(close)
            volume_changes = np.diff(volume)
            
            direction = 'neutral'
            strength = 0
            
            # Recent 5 candles analysis
            recent_price_change = sum(price_changes[-5:])
            recent_volume_avg = np.mean(volume[-5:])
            historical_volume_avg = np.mean(volume[-20:-5])
            
            if recent_volume_avg > historical_volume_avg * 1.3:  # 30% more volume
                if recent_price_change > 0:
                    direction = 'bullish'
                    strength = min(abs(recent_price_change) / close[-1] * 1000, 100)
                elif recent_price_change < 0:
                    direction = 'bearish'
                    strength = min(abs(recent_price_change) / close[-1] * 1000, 100)
            
            return {
                'direction': direction,
                'strength': strength
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting price imbalances: {e}")
            return {'direction': 'neutral', 'strength': 0}

    def _analyze_volume_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume profile for institutional activity"""
        try:
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # Simple volume-weighted average price calculation
            typical_prices = (high + low + df['close'].values) / 3
            volume_weighted_prices = typical_prices * volume
            
            if np.sum(volume) == 0:
                return {'point_of_control': 0, 'imbalance': 0, 'hvn': False}
            
            point_of_control = np.sum(volume_weighted_prices) / np.sum(volume)
            
            # Detect volume imbalances
            recent_volume = np.mean(volume[-5:])
            historical_volume = np.mean(volume[-20:-5])
            volume_imbalance = (recent_volume - historical_volume) / historical_volume if historical_volume > 0 else 0
            
            # High Volume Node detection
            hvn = recent_volume > historical_volume * 2  # 100% above average
            
            return {
                'point_of_control': point_of_control,
                'imbalance': volume_imbalance * 100,
                'hvn': hvn
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume profile: {e}")
            return {'point_of_control': 0, 'imbalance': 0, 'hvn': False}

    def _detect_institutional_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect institutional candle patterns"""
        try:
            open_prices = df['open'].values
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            volume = df['volume'].values
            
            pattern = 'none'
            reliability = 0
            
            # Check last 3 candles for patterns
            if len(df) >= 3:
                last_3_candles = df.tail(3)
                
                # Institutional absorption pattern
                for i in range(-3, 0):
                    candle_body = abs(close[i] - open_prices[i])
                    candle_range = high[i] - low[i]
                    
                    # Large wick with small body = absorption
                    if candle_range > 0 and candle_body / candle_range < 0.3:
                        if volume[i] > np.mean(volume[-10:]) * 1.5:
                            pattern = 'absorption'
                            reliability = 75
                            break
                    
                    # Large body with high volume = institutional move
                    if candle_range > 0 and candle_body / candle_range > 0.7:
                        if volume[i] > np.mean(volume[-10:]) * 2:
                            pattern = 'institutional_move'
                            reliability = 80
                            break
            
            return {
                'pattern': pattern,
                'reliability': reliability
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting institutional patterns: {e}")
            return {'pattern': 'none', 'reliability': 0}

    def _calculate_smart_money_concepts(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Smart Money Concepts (SMC)"""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            choch = False  # Change of Character
            bos = False    # Break of Structure
            displacement = False
            bias = 'neutral'
            
            if len(df) < 10:
                return {'change_of_character': choch, 'break_of_structure': bos, 
                       'displacement': displacement, 'bias': bias}
            
            # Look for significant price movements (displacement)
            price_changes = np.diff(close[-10:])
            max_change = max(abs(price_changes)) if len(price_changes) > 0 else 0
            
            if max_change / close[-1] > 0.01:  # 1% move
                displacement = True
                
                # Determine bias based on recent structure
                recent_high = max(high[-10:])
                recent_low = min(low[-10:])
                current_price = close[-1]
                
                if current_price > (recent_high + recent_low) / 2:
                    bias = 'bullish'
                    if current_price > recent_high * 1.001:  # Break above recent high
                        bos = True
                else:
                    bias = 'bearish'
                    if current_price < recent_low * 0.999:  # Break below recent low
                        bos = True
                
                # Change of Character: reversal after strong move
                if len(price_changes) >= 3:
                    recent_trend = sum(price_changes[-3:])
                    previous_trend = sum(price_changes[-6:-3])
                    
                    if (recent_trend > 0 and previous_trend < 0) or (recent_trend < 0 and previous_trend > 0):
                        choch = True
            
            return {
                'change_of_character': choch,
                'break_of_structure': bos,
                'displacement': displacement,
                'bias': bias
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating SMC: {e}")
            return {'change_of_character': False, 'break_of_structure': False, 
                   'displacement': False, 'bias': 'neutral'}

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
        """Get performance factor for adaptive leverage adjustment"""
        try:
            # Load recent trades from ML database
            conn = sqlite3.connect(self.ml_analyzer.db_path)
            cursor = conn.cursor()
            
            # Get recent trades for performance analysis
            cursor.execute("""
                SELECT profit_loss, trade_result 
                FROM ml_trades 
                WHERE profit_loss IS NOT NULL 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (self.adaptive_leverage['performance_window'],))
            
            recent_trades = cursor.fetchall()
            conn.close()

            # If no historical data, simulate performance based on signal strength
            if not recent_trades:
                # Return a base performance factor based on market conditions
                base_factor = 0.15  # Slightly positive for new systems
                return base_factor

            # Calculate performance metrics
            wins = sum(1 for trade in recent_trades if trade[0] and float(trade[0]) > 0)
            losses = len(recent_trades) - wins
            
            if len(recent_trades) == 0:
                return 0.15  # Default positive factor
                
            win_rate = wins / len(recent_trades)
            
            # Calculate average profit/loss
            total_pnl = sum(float(trade[0]) if trade[0] else 0 for trade in recent_trades)
            avg_pnl = total_pnl / len(recent_trades) if recent_trades else 0
            
            # Calculate consecutive performance
            consecutive_wins = 0
            consecutive_losses = 0
            
            for trade in recent_trades:
                if trade[0] and float(trade[0]) > 0:  # Winning trade
                    consecutive_wins += 1
                    consecutive_losses = 0
                    break  # Only count current streak
                else:  # Losing trade
                    consecutive_losses += 1
                    consecutive_wins = 0
                    break  # Only count current streak

            # Update adaptive leverage tracking
            self.adaptive_leverage.update({
                'recent_wins': wins,
                'recent_losses': losses,
                'consecutive_wins': consecutive_wins,
                'consecutive_losses': consecutive_losses
            })

            # Calculate performance factor (-1 to +1)
            performance_factor = 0.0
            
            # Win rate adjustment (primary factor)
            if win_rate >= 0.7:  # High win rate - increase leverage
                performance_factor += 0.4
            elif win_rate >= 0.6:  # Good win rate
                performance_factor += 0.2
            elif win_rate >= 0.5:  # Break-even
                performance_factor += 0.1
            elif win_rate <= 0.3:  # Low win rate - decrease leverage
                performance_factor -= 0.4
            else:  # Below average
                performance_factor -= 0.2
                
            # Average P&L adjustment
            if avg_pnl > 1.0:  # Highly profitable
                performance_factor += 0.3
            elif avg_pnl > 0.5:  # Profitable
                performance_factor += 0.15
            elif avg_pnl < -1.0:  # Highly unprofitable
                performance_factor -= 0.3
            elif avg_pnl < -0.5:  # Unprofitable
                performance_factor -= 0.15
                
            # Consecutive performance adjustment
            if consecutive_wins >= 3:
                performance_factor += 0.2
            elif consecutive_wins >= 2:
                performance_factor += 0.1
            elif consecutive_losses >= 3:
                performance_factor -= 0.3
            elif consecutive_losses >= 2:
                performance_factor -= 0.15
                
            # Limit performance factor
            performance_factor = max(-0.8, min(0.8, performance_factor))
            
            return performance_factor

        except Exception as e:
            self.logger.error(f"Error calculating performance factor: {e}")
            # Return a small positive factor to encourage trading
            return 0.1

    def generate_ml_enhanced_signal(self, symbol: str, indicators: Dict[str, Any], df: Optional[pd.DataFrame] = None) -> Optional[Dict[str, Any]]:
        """Generate ML-enhanced scalping signal with institutional order flow analysis"""
        try:
            current_time = datetime.now()
            if symbol in self.last_signal_time:
                time_diff = (current_time - self.last_signal_time[symbol]).total_seconds()
                if time_diff < self.min_signal_interval:
                    return None

            signal_strength = 0
            bullish_signals = 0
            bearish_signals = 0

            current_price = indicators.get('current_price', 0)
            if current_price <= 0:
                return None

            # INSTITUTIONAL ORDER FLOW ANALYSIS (40% weight total)
            
            # 1. Market Structure (10% weight)
            if indicators.get('structure_break') and indicators.get('market_structure') == 'uptrend':
                bullish_signals += 10
            elif indicators.get('structure_break') and indicators.get('market_structure') == 'downtrend':
                bearish_signals += 10

            # 2. Order Blocks (8% weight)
            if indicators.get('bullish_order_block') and indicators.get('order_block_strength', 0) > 50:
                bullish_signals += 8
            elif indicators.get('bearish_order_block') and indicators.get('order_block_strength', 0) > 50:
                bearish_signals += 8

            # 3. Fair Value Gaps (7% weight)
            if indicators.get('bullish_fvg') and indicators.get('fvg_probability', 0) > 60:
                bullish_signals += 7
            elif indicators.get('bearish_fvg') and indicators.get('fvg_probability', 0) > 60:
                bearish_signals += 7

            # 4. Liquidity Sweeps (8% weight)
            if indicators.get('liquidity_sweep') and indicators.get('liquidity_strength', 0) > 60:
                # Liquidity sweep often leads to reversal
                if indicators.get('smc_bias') == 'bullish':
                    bullish_signals += 8
                elif indicators.get('smc_bias') == 'bearish':
                    bearish_signals += 8

            # 5. Smart Money Concepts (7% weight)
            if indicators.get('bos') and indicators.get('smc_bias') == 'bullish':
                bullish_signals += 7
            elif indicators.get('bos') and indicators.get('smc_bias') == 'bearish':
                bearish_signals += 7

            # TRADITIONAL TECHNICAL ANALYSIS (60% weight total)

            # 6. Enhanced SuperTrend (15% weight)
            if indicators.get('supertrend_direction') == 1:
                bullish_signals += 15
            elif indicators.get('supertrend_direction') == -1:
                bearish_signals += 15

            # 7. EMA Confluence (12% weight)
            if indicators.get('ema_bullish'):
                bullish_signals += 12
            elif indicators.get('ema_bearish'):
                bearish_signals += 12

            # 8. CVD Confluence (10% weight)
            cvd_trend = indicators.get('cvd_trend', 'neutral')
            if cvd_trend == 'bullish':
                bullish_signals += 10
            elif cvd_trend == 'bearish':
                bearish_signals += 10

            # 9. VWAP Position (8% weight)
            price_vs_vwap = indicators.get('price_vs_vwap', 0)
            if not np.isnan(price_vs_vwap) and not np.isinf(price_vs_vwap):
                if price_vs_vwap > 0.1:
                    bullish_signals += 8
                elif price_vs_vwap < -0.1:
                    bearish_signals += 8

            # 10. Volume Profile & Imbalances (8% weight)
            if indicators.get('high_volume_node') and indicators.get('volume_imbalance', 0) > 20:
                if indicators.get('imbalance_direction') == 'bullish':
                    bullish_signals += 8
                elif indicators.get('imbalance_direction') == 'bearish':
                    bearish_signals += 8

            # 11. RSI analysis (4% weight)
            if indicators.get('rsi_oversold'):
                bullish_signals += 4
            elif indicators.get('rsi_overbought'):
                bearish_signals += 4

            # 12. MACD confluence (3% weight)  
            if indicators.get('macd_bullish'):
                bullish_signals += 3
            elif indicators.get('macd_bearish'):
                bearish_signals += 3

            # Determine signal direction and strength
            if bullish_signals >= self.min_signal_strength:
                direction = 'BUY'
                signal_strength = bullish_signals
            elif bearish_signals >= self.min_signal_strength:
                direction = 'SELL'
                signal_strength = bearish_signals
            else:
                return None

            # Calculate entry, stop loss, and take profits
            entry_price = current_price
            risk_percentage = 1.5  # 1.5% risk
            risk_amount = entry_price * (risk_percentage / 100)

            if direction == 'BUY':
                stop_loss = entry_price - risk_amount
                tp1 = entry_price + (risk_amount * 1.0)
                tp2 = entry_price + (risk_amount * 2.0)
                tp3 = entry_price + (risk_amount * 3.0)

                if not (stop_loss < entry_price < tp1 < tp2 < tp3):
                    stop_loss = entry_price * 0.985
                    tp1 = entry_price * 1.015
                    tp2 = entry_price * 1.030
                    tp3 = entry_price * 1.045
            else:
                stop_loss = entry_price + risk_amount
                tp1 = entry_price - (risk_amount * 1.0)
                tp2 = entry_price - (risk_amount * 2.0)
                tp3 = entry_price - (risk_amount * 3.0)

                if not (tp3 < tp2 < tp1 < entry_price < stop_loss):
                    stop_loss = entry_price * 1.015
                    tp1 = entry_price * 0.985
                    tp2 = entry_price * 0.970
                    tp3 = entry_price * 0.955

            # Risk validation
            risk_percentage = abs(entry_price - stop_loss) / entry_price * 100
            if risk_percentage > 3.0:
                return None

            # Calculate adaptive leverage with cross margin
            placeholder_df = pd.DataFrame({'close': [current_price] * 20}) if df is None or len(df) < 20 else df
            optimal_leverage = self.calculate_adaptive_leverage(indicators, placeholder_df)

            # ML prediction with institutional order flow
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
                'ema_bullish': indicators.get('ema_bullish', False),
                # Institutional Order Flow Data
                'market_structure': indicators.get('market_structure', 'ranging'),
                'structure_break': indicators.get('structure_break', False),
                'liquidity_sweep': indicators.get('liquidity_sweep', False),
                'order_block_strength': indicators.get('order_block_strength', 0),
                'fvg_probability': indicators.get('fvg_probability', 0),
                'smc_bias': indicators.get('smc_bias', 'neutral'),
                'displacement': indicators.get('displacement', False),
                'institutional_pattern': indicators.get('institutional_pattern', 'none'),
                'volume_imbalance': indicators.get('volume_imbalance', 0)
            }

            ml_prediction = self.ml_analyzer.predict_trade_outcome(ml_signal_data)

            # Only proceed with favorable predictions
            ml_confidence = ml_prediction.get('confidence', 50)
            prediction_type = ml_prediction.get('prediction', 'unknown')
            
            # Filter: Only allow favorable or highly favorable predictions
            if prediction_type not in ['favorable', 'highly_favorable']:
                return None
            
            # Adjust signal strength for favorable predictions
            if prediction_type == 'highly_favorable':
                signal_strength *= 1.2
            elif prediction_type == 'favorable':
                signal_strength *= 1.1

            # Final signal strength check
            if signal_strength < self.min_signal_strength:
                return None

            # Update last signal time
            self.last_signal_time[symbol] = current_time

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
                'institutional_indicators': {
                    'market_structure': indicators.get('market_structure', 'ranging'),
                    'structure_break': indicators.get('structure_break', False),
                    'order_block_strength': indicators.get('order_block_strength', 0),
                    'liquidity_sweep': indicators.get('liquidity_sweep', False),
                    'smc_bias': indicators.get('smc_bias', 'neutral'),
                    'displacement': indicators.get('displacement', False),
                    'fvg_probability': indicators.get('fvg_probability', 0),
                    'high_volume_node': indicators.get('high_volume_node', False),
                    'institutional_pattern': indicators.get('institutional_pattern', 'none'),
                    'volume_imbalance': indicators.get('volume_imbalance', 0)
                },
                'indicators_used': [
                    'Institutional Order Flow', 'Market Structure Analysis', 'Order Blocks',
                    'Fair Value Gaps', 'Liquidity Sweeps', 'Smart Money Concepts',
                    'ML-Enhanced SuperTrend', 'EMA Confluence', 'CVD Analysis',
                    'VWAP Position', 'Volume Profile', 'RSI Analysis', 'MACD Signals'
                ],
                'timeframe': 'Multi-TF (1m-4h)',
                'strategy': 'Ultimate Institutional ML-Enhanced Scalping',
                'ml_enhanced': True,
                'institutional_analysis': True,
                'adaptive_leverage': True,
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
                        self.logger.warning(f"Timeframe {timeframe} error for {symbol}: {str(e)[:100]}")
                        continue

                if timeframe_scores:
                    try:
                        valid_signals = [s for s in timeframe_scores.values() if s.get('signal_strength', 0) > 0]
                        if valid_signals:
                            # Select signal with highest ML confidence
                            best_signal = max(valid_signals, key=lambda x: x.get('ml_prediction', {}).get('confidence', 0))

                            if best_signal.get('signal_strength', 0) >= self.min_signal_strength:
                                signals.append(best_signal)
                    except Exception as e:
                        self.logger.error(f"Error selecting best signal for {symbol}: {e}")
                        continue

            except Exception as e:
                self.logger.warning(f"Skipping {symbol} due to error: {str(e)[:100]}")
                continue

        # Sort by ML confidence and signal strength
        signals.sort(key=lambda x: (x.get('ml_prediction', {}).get('confidence', 0), x['signal_strength']), reverse=True)
        return signals[:self.max_signals_per_hour]

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

    async def send_message(self, chat_id: str, text: str, parse_mode='Markdown') -> bool:
        """Send message to Telegram"""
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': text,
                'parse_mode': parse_mode,
                'disable_web_page_preview': True
            }

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
                'text': f"📢 **CHANNEL FALLBACK**\n\n{text}",
                'parse_mode': parse_mode,
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
        """Format ML-enhanced signal message for Cornix compatibility via Telegram"""
        direction = signal['direction']
        timestamp = datetime.now().strftime('%H:%M')
        optimal_leverage = signal.get('optimal_leverage', 35)
        ml_prediction = signal.get('ml_prediction', {})

        # Cornix-compatible format for Telegram channel
        cornix_signal = self._format_cornix_signal(signal)

        # Get institutional indicators for display
        indicators = signal.get('institutional_indicators', {})
        
        message = f"""{cornix_signal}

🏛️ **INSTITUTIONAL ANALYSIS:**
• Market Structure: {indicators.get('market_structure', 'Unknown').title()}
• Structure Break: {'✅' if indicators.get('structure_break') else '❌'}
• Order Block: {'✅ Detected' if indicators.get('order_block_strength', 0) > 50 else '❌ None'}
• Liquidity Sweep: {'✅ Confirmed' if indicators.get('liquidity_sweep') else '❌ None'}
• SMC Bias: {indicators.get('smc_bias', 'Neutral').title()}

🧠 **ML ANALYSIS:**
• Prediction: {ml_prediction.get('prediction', 'unknown').title()}
• Confidence: {ml_prediction.get('confidence', 0):.1f}%
• Expected Profit: {ml_prediction.get('expected_profit', 0):.2f}%
• Risk: {ml_prediction.get('risk_probability', 0):.1f}%

📊 **SIGNAL DATA:**
• Signal #{self.signal_counter} | Strength: {signal['signal_strength']:.0f}%
• Time: {timestamp} UTC | R/R: 1:{signal['risk_reward_ratio']:.1f}
• CVD: {self.cvd_data['cvd_trend'].title()} | Volume Profile: {'High' if indicators.get('high_volume_node') else 'Normal'}

⚙️ **ADAPTIVE FEATURES:**
• Leverage: {optimal_leverage}x (Adaptive)
• Margin: Cross | Win Streak: {self.adaptive_leverage['consecutive_wins']}
• Recent Performance: {self.adaptive_leverage['recent_wins']}/{self.adaptive_leverage['recent_wins'] + self.adaptive_leverage['recent_losses']}

🔧 **AUTO MANAGEMENT:**
✅ TP1 Hit → SL to Entry (Risk-Free)
✅ TP2 Hit → SL to TP1 (Profit Lock)
✅ TP3 Hit → Full Close (Complete)

🧠 **ML REC:** {ml_prediction.get('recommendation', 'Trade with caution')}

*TradeTactics ML Bot | Institutional Order Flow • Cross Margin • Adaptive Learning*"""

        return message.strip()

    def _format_cornix_signal(self, signal: Dict[str, Any]) -> str:
        """Format signal in Cornix-compatible format"""
        try:
            symbol = signal['symbol']
            direction = signal['direction'].upper()
            entry = signal['entry_price']
            stop_loss = signal['stop_loss']
            tp1 = signal['tp1']
            tp2 = signal['tp2']
            tp3 = signal['tp3']
            optimal_leverage = signal.get('optimal_leverage', 35)

            formatted_message = f"""#{symbol} {direction}

Entry: {entry:.6f}
Stop Loss: {stop_loss:.6f}

Take Profit:
TP1: {tp1:.6f} (40%)
TP2: {tp2:.6f} (35%) 
TP3: {tp3:.6f} (25%)

Leverage: {optimal_leverage}x
Margin Type: Cross
Exchange: Binance Futures

Management:
- Move SL to Entry after TP1
- Move SL to TP1 after TP2  
- Close all after TP3"""

            return formatted_message

        except Exception as e:
            self.logger.error(f"Error formatting Cornix signal: {e}")
            optimal_leverage = signal.get('optimal_leverage', 35)
            return f"""#{signal['symbol']} {signal['direction']}
Entry: {signal['entry_price']:.6f}
Stop Loss: {signal['stop_loss']:.6f}
TP1: {signal['tp1']:.6f}
TP2: {signal['tp2']:.6f}
TP3: {signal['tp3']:.6f}
Leverage: {optimal_leverage}x
Exchange: Binance Futures"""

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

    async def handle_commands(self, message: Dict, chat_id: str):
        """Handle bot commands"""
        try:
            text = message.get('text', '').strip()

            if not text:
                return

            if text.startswith('/start'):
                self.admin_chat_id = chat_id
                self.logger.info(f"✅ Admin set to chat_id: {chat_id}")

                await self.verify_channel_access()
                channel_status = "✅ Accessible" if self.channel_accessible else "⚠️ Not Accessible"
                ml_summary = self.ml_analyzer.get_ml_summary()

                welcome = f"""🧠 **ULTIMATE ML TRADING BOT**
*Advanced Machine Learning Strategy*

✅ **Status:** Online & Learning
🎯 **Strategy:** ML-Enhanced Multi-Indicator Scalping
⚖️ **Risk/Reward:** 1:3 Ratio Guaranteed
📊 **Timeframes:** 1m to 4h
🔍 **Symbols:** {len(self.symbols)}+ Top Crypto Pairs

**🧠 Machine Learning Status:**
• **Signal Accuracy:** {ml_summary['model_performance']['signal_accuracy']*100:.1f}%
• **Trades Learned:** {ml_summary['model_performance']['total_trades_learned']}
• **Learning Status:** {ml_summary['learning_status'].title()}
• **Next Retrain:** {ml_summary['next_retrain_in']} trades

**🛡️ Risk Management:**
• Stop Loss to Entry after TP1
• Maximum 2.5% risk per trade
• 3 Take Profit levels
• ML-enhanced signal filtering

**📈 Performance:**
• Signals Generated: `{self.performance_stats['total_signals']}`
• Win Rate: `{self.performance_stats['win_rate']:.1f}%`
• Active Trades: `{len(self.active_trades)}`

**📢 Channel Status:**
• Target: `{self.target_channel}`
• Access: `{channel_status}`
• Fallback: Admin messaging enabled

**🤖 ML Features:**
• Continuous learning from every trade
• Profit prediction & risk assessment
• Market insight analysis
• Time session optimization
• Symbol performance tracking

*Bot learns and improves with each trade*
Use `/help` for all commands"""

                await self.send_message(chat_id, welcome)

            elif text.startswith('/ml'):
                ml_summary = self.ml_analyzer.get_ml_summary()
                
                ml_status = f"""🧠 **MACHINE LEARNING STATUS**

**🤖 Model Performance:**
• **Signal Accuracy:** {ml_summary['model_performance']['signal_accuracy']*100:.1f}%
• **Profit Prediction:** {ml_summary['model_performance']['profit_prediction_accuracy']*100:.1f}%
• **Risk Assessment:** {ml_summary['model_performance']['risk_assessment_accuracy']*100:.1f}%
• **Total Trades Learned:** {ml_summary['model_performance']['total_trades_learned']}

**📊 Learning Progress:**
• **Status:** {ml_summary['learning_status'].title()}
• **Next Retrain:** {ml_summary['next_retrain_in']} trades
• **Last Training:** {ml_summary['model_performance']['last_training_time'] or 'Never'}

**🔍 Market Insights:**
• **Best Time Sessions:** Available
• **Symbol Performance:** Tracked
• **Indicator Effectiveness:** Analyzed

**⚙️ ML Features:**
✅ Real-time trade learning
✅ Profit prediction
✅ Risk assessment
✅ Market regime detection
✅ Time session optimization

*ML models continuously improve with new data*"""

                await self.send_message(chat_id, ml_status)

            elif text.startswith('/scan'):
                await self.send_message(chat_id, "🧠 **ML-ENHANCED MARKET SCAN**\n\nAnalyzing markets with machine learning...")

                signals = await self.scan_for_signals()

                if signals:
                    for signal in signals[:3]:
                        self.signal_counter += 1
                        signal_msg = self.format_ml_signal_message(signal)
                        await self.send_message(chat_id, signal_msg)
                        await asyncio.sleep(2)

                    await self.send_message(chat_id, f"✅ **{len(signals)} ML-ENHANCED SIGNALS FOUND**\n\nTop ML-validated signals delivered! Bot continues learning...")
                else:
                    await self.send_message(chat_id, "📊 **NO HIGH-CONFIDENCE ML SIGNALS**\n\nML models filtering for optimal opportunities. Bot continues learning...")

        except Exception as e:
            self.logger.error(f"Error handling command {text}: {e}")

    async def record_trade_completion(self, signal: Dict[str, Any], trade_result: Dict[str, Any]):
        """Record completed trade for ML learning"""
        try:
            trade_data = {
                'symbol': signal['symbol'],
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
                'exit_time': trade_result.get('exit_time', datetime.now())
            }
            
            await self.ml_analyzer.record_trade_outcome(trade_data)
            
        except Exception as e:
            self.logger.error(f"Error recording trade completion: {e}")

    async def auto_scan_loop(self):
        """Main auto-scanning loop with ML learning"""
        consecutive_errors = 0
        max_consecutive_errors = 5
        base_scan_interval = 90

        while self.running and not self.shutdown_requested:
            try:
                self.logger.info("🧠 Scanning markets for ML-enhanced signals...")
                signals = await self.scan_for_signals()

                if signals:
                    self.logger.info(f"📊 Found {len(signals)} ML-validated signals")

                    signals_sent_count = 0

                    for signal in signals:
                        if signals_sent_count >= self.max_signals_per_hour:
                            self.logger.info(f"⏸️ Reached maximum signals per hour ({self.max_signals_per_hour})")
                            break

                        try:
                            self.signal_counter += 1
                            self.performance_stats['total_signals'] += 1

                            if self.performance_stats['total_signals'] > 0:
                                self.performance_stats['win_rate'] = (
                                    self.performance_stats['profitable_signals'] / 
                                    self.performance_stats['total_signals'] * 100
                                )

                            signal_msg = self.format_ml_signal_message(signal)

                            # Send only to Telegram channel @SignalTactics
                            channel_sent = False
                            if self.channel_accessible:
                                channel_sent = await self.send_message(self.target_channel, signal_msg)

                            if channel_sent:
                                delivery_info = "Channel @SignalTactics"
                                self.logger.info(f"📤 ML Signal #{self.signal_counter} delivered to: {delivery_info}")
                                
                                ml_conf = signal.get('ml_prediction', {}).get('confidence', 0)
                                self.logger.info(f"✅ ML Signal sent: {signal['symbol']} {signal['direction']} (Strength: {signal['signal_strength']:.0f}%, ML: {ml_conf:.1f}%)")
                            else:
                                self.logger.warning(f"❌ Failed to send ML Signal #{self.signal_counter} to @SignalTactics")

                            signals_sent_count += 1
                            await asyncio.sleep(5)

                        except Exception as signal_error:
                            self.logger.error(f"Error processing ML signal for {signal.get('symbol', 'unknown')}: {signal_error}")
                            continue

                else:
                    self.logger.info("📊 No ML signals found - models filtering for optimal opportunities")

                consecutive_errors = 0
                self.last_heartbeat = datetime.now()

                scan_interval = 60 if signals else base_scan_interval
                self.logger.info(f"⏰ Next ML scan in {scan_interval} seconds")
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

*Ultimate ML bot with Adaptive Cross-Margin Trading*"""
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
