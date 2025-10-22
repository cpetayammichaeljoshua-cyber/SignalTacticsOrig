
#!/usr/bin/env python3
"""
Advanced Auto-Training ML System with OpenAI Integration
Dynamically trains machine learning models for optimal trading performance
Features:
- Real-time model training from live trades
- OpenAI-enhanced pattern recognition
- Adaptive learning rate adjustment
- Multi-model ensemble for robust predictions
- Continuous performance optimization
"""

import asyncio
import logging
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import sqlite3
from pathlib import Path
import pickle
import time
from collections import deque, defaultdict

# Machine Learning imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# OpenAI integration
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

@dataclass
class TrainingResult:
    """Training session result"""
    timestamp: datetime
    model_type: str
    accuracy: float
    precision: float
    recall: float
    training_samples: int
    learning_rate: float
    performance_improvement: float
    openai_insights: Optional[Dict[str, Any]] = None

@dataclass
class SignalPrediction:
    """ML-enhanced signal prediction"""
    symbol: str
    direction: str
    confidence: float
    expected_profit: float
    risk_score: float
    optimal_entry: float
    optimal_sl: float
    optimal_tp: List[float]
    model_consensus: float
    openai_sentiment: float
    reasoning: str

class AdvancedAutoTrainMLSystem:
    """
    Advanced auto-training ML system with OpenAI integration
    Continuously learns and improves from trading performance
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # OpenAI client
        self.openai_client = None
        if OPENAI_AVAILABLE:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key and api_key != "your_openai_api_key_here":
                try:
                    self.openai_client = OpenAI(api_key=api_key)
                    self.logger.info("🤖 OpenAI integration active for enhanced learning")
                except Exception as e:
                    self.logger.warning(f"⚠️ OpenAI initialization failed: {e}")
        
        # ML Models ensemble
        self.models = {
            'signal_classifier': None,
            'profit_predictor': None,
            'risk_assessor': None,
            'entry_optimizer': None,
            'exit_optimizer': None
        }
        
        self.scalers = {
            'features': StandardScaler(),
            'targets': StandardScaler()
        }
        
        # Training configuration
        self.training_config = {
            'min_samples': 50,
            'batch_size': 32,
            'learning_rate': 0.001,
            'retrain_interval': 10,  # Retrain every 10 new trades
            'performance_threshold': 0.65,  # 65% minimum accuracy
            'auto_tune': True,
            'openai_enhancement': True
        }
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.training_history = []
        self.trade_outcomes = deque(maxlen=5000)
        
        # Database
        self.db_path = "SignalMaestro/auto_train_ml.db"
        self._initialize_database()
        
        # Auto-training state
        self.trade_counter = 0
        self.last_training = None
        self.is_training = False
        self.training_lock = asyncio.Lock()
        
        # Load existing models
        self._load_models()
        
        self.logger.info("🧠 Advanced Auto-Train ML System initialized")

    def _initialize_database(self):
        """Initialize ML training database"""
        try:
            Path("SignalMaestro").mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Training results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    model_type TEXT NOT NULL,
                    accuracy REAL NOT NULL,
                    precision_score REAL NOT NULL,
                    recall_score REAL NOT NULL,
                    training_samples INTEGER NOT NULL,
                    learning_rate REAL NOT NULL,
                    performance_improvement REAL NOT NULL,
                    openai_insights TEXT,
                    model_state BLOB
                )
            ''')

            # Trade outcomes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    profit_loss REAL,
                    win INTEGER NOT NULL,
                    indicators TEXT NOT NULL,
                    market_conditions TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Model performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_type TEXT NOT NULL,
                    accuracy REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    avg_profit REAL NOT NULL,
                    total_trades INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.commit()
            conn.close()
            self.logger.info("📊 Auto-train ML database initialized")

        except Exception as e:
            self.logger.error(f"❌ Database initialization failed: {e}")

    async def record_trade_outcome(self, trade_data: Dict[str, Any]):
        """Record trade outcome for training"""
        try:
            self.trade_outcomes.append(trade_data)
            self.trade_counter += 1
            
            # Store in database
            await self._store_trade_outcome(trade_data)
            
            # Check if auto-training should trigger
            if self.trade_counter >= self.training_config['retrain_interval']:
                await self.auto_train()
                self.trade_counter = 0
                
        except Exception as e:
            self.logger.error(f"❌ Trade outcome recording failed: {e}")

    async def auto_train(self):
        """Automatically train ML models with latest data"""
        if self.is_training:
            self.logger.info("⏳ Training already in progress")
            return
        
        async with self.training_lock:
            try:
                self.is_training = True
                self.logger.info("🎓 Starting auto-training session...")
                
                # Prepare training data
                training_data = await self._prepare_training_data()
                
                if len(training_data) < self.training_config['min_samples']:
                    self.logger.warning(f"⚠️ Insufficient data: {len(training_data)} samples")
                    return
                
                # Get OpenAI insights if available
                openai_insights = None
                if self.openai_client and self.training_config['openai_enhancement']:
                    openai_insights = await self._get_openai_training_insights(training_data)
                
                # Train each model
                results = []
                for model_name in self.models.keys():
                    result = await self._train_model(model_name, training_data, openai_insights)
                    if result:
                        results.append(result)
                
                # Evaluate performance improvement
                improvement = await self._evaluate_training_improvement(results)
                
                # Save models
                self._save_models()
                
                # Store training results
                for result in results:
                    await self._store_training_result(result)
                
                self.last_training = datetime.now()
                self.logger.info(f"✅ Auto-training complete: {improvement:.1%} improvement")
                
            except Exception as e:
                self.logger.error(f"❌ Auto-training failed: {e}")
            finally:
                self.is_training = False

    async def _prepare_training_data(self) -> pd.DataFrame:
        """Prepare training data from trade outcomes"""
        try:
            if not self.trade_outcomes:
                return pd.DataFrame()
            
            data = []
            for trade in self.trade_outcomes:
                # Extract features
                features = {
                    'rsi': trade.get('rsi', 50),
                    'macd': trade.get('macd', 0),
                    'volume_ratio': trade.get('volume_ratio', 1.0),
                    'volatility': trade.get('volatility', 0.02),
                    'trend_strength': trade.get('trend_strength', 0),
                    'market_regime': self._encode_market_regime(trade.get('market_regime', 'neutral')),
                    'time_of_day': trade.get('hour', 12),
                    'day_of_week': trade.get('day_of_week', 1),
                }
                
                # Add targets
                features.update({
                    'win': trade.get('win', 0),
                    'profit_loss': trade.get('profit_loss', 0),
                    'direction': 1 if trade.get('direction') == 'BUY' else 0
                })
                
                data.append(features)
            
            df = pd.DataFrame(data)
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Training data preparation failed: {e}")
            return pd.DataFrame()

    async def _get_openai_training_insights(self, training_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Get OpenAI insights for training optimization"""
        try:
            if not self.openai_client:
                return None
            
            # Calculate statistics
            win_rate = training_data['win'].mean()
            avg_profit = training_data['profit_loss'].mean()
            
            # Create analysis prompt
            prompt = f"""
Analyze this trading bot performance data and provide insights for ML optimization:

Performance Metrics:
- Win Rate: {win_rate:.1%}
- Average Profit: {avg_profit:.2%}
- Total Trades: {len(training_data)}

Recent Market Conditions:
- Average RSI: {training_data['rsi'].mean():.1f}
- Average Volatility: {training_data['volatility'].mean():.3f}
- Trend Strength: {training_data['trend_strength'].mean():.2f}

Provide JSON response with:
{{
    "optimization_recommendations": ["list of specific improvements"],
    "risk_adjustment_factor": float (0.5-1.5),
    "confidence_threshold": float (0.0-1.0),
    "entry_timing_bias": "early" or "late" or "balanced",
    "market_regime_preference": "trending" or "ranging" or "volatile",
    "feature_importance": {{"rsi": float, "macd": float, "volume": float}}
}}
"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert quantitative trading analyst."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            insights = json.loads(response.choices[0].message.content)
            self.logger.info("🤖 OpenAI training insights received")
            return insights
            
        except Exception as e:
            self.logger.warning(f"⚠️ OpenAI insights failed: {e}")
            return None

    async def _train_model(self, model_name: str, training_data: pd.DataFrame, 
                          openai_insights: Optional[Dict[str, Any]]) -> Optional[TrainingResult]:
        """Train individual ML model"""
        try:
            if not SKLEARN_AVAILABLE:
                return None
            
            # Prepare features and targets based on model type
            if model_name == 'signal_classifier':
                X = training_data[['rsi', 'macd', 'volume_ratio', 'volatility', 
                                  'trend_strength', 'market_regime', 'time_of_day', 'day_of_week']]
                y = training_data['win']
                model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
                
            elif model_name == 'profit_predictor':
                X = training_data[['rsi', 'macd', 'volume_ratio', 'volatility', 
                                  'trend_strength', 'direction']]
                y = training_data['profit_loss']
                model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
                
            elif model_name == 'risk_assessor':
                X = training_data[['volatility', 'trend_strength', 'market_regime', 'volume_ratio']]
                y = training_data['win']
                model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
                
            else:
                return None
            
            # Apply OpenAI insights if available
            if openai_insights:
                # Adjust model parameters based on insights
                learning_rate = self.training_config['learning_rate']
                if 'risk_adjustment_factor' in openai_insights:
                    learning_rate *= openai_insights['risk_adjustment_factor']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scalers['features'].fit_transform(X_train)
            X_test_scaled = self.scalers['features'].transform(X_test)
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            
            if model_name in ['signal_classifier', 'risk_assessor']:
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
                recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
            else:
                # For regression models
                accuracy = 1 - np.mean(np.abs(y_pred - y_test))
                precision = accuracy
                recall = accuracy
            
            # Store model
            self.models[model_name] = model
            
            # Calculate improvement
            previous_accuracy = self._get_previous_accuracy(model_name)
            improvement = accuracy - previous_accuracy
            
            result = TrainingResult(
                timestamp=datetime.now(),
                model_type=model_name,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                training_samples=len(training_data),
                learning_rate=learning_rate,
                performance_improvement=improvement,
                openai_insights=openai_insights
            )
            
            self.logger.info(f"📈 {model_name}: {accuracy:.1%} accuracy (+{improvement:.1%})")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ {model_name} training failed: {e}")
            return None

    async def predict_signal_quality(self, signal_data: Dict[str, Any]) -> SignalPrediction:
        """Predict signal quality using ensemble of ML models"""
        try:
            # Extract features
            features = np.array([[
                signal_data.get('rsi', 50),
                signal_data.get('macd', 0),
                signal_data.get('volume_ratio', 1.0),
                signal_data.get('volatility', 0.02),
                signal_data.get('trend_strength', 0),
                self._encode_market_regime(signal_data.get('market_regime', 'neutral')),
                datetime.now().hour,
                datetime.now().weekday()
            ]])
            
            # Scale features
            features_scaled = self.scalers['features'].transform(features)
            
            # Get predictions from all models
            predictions = {}
            if self.models['signal_classifier'] is not None:
                predictions['win_probability'] = self.models['signal_classifier'].predict_proba(features_scaled)[0][1]
            else:
                predictions['win_probability'] = 0.5
            
            if self.models['profit_predictor'] is not None:
                predictions['expected_profit'] = self.models['profit_predictor'].predict(features_scaled[:, :6])[0]
            else:
                predictions['expected_profit'] = 0.01
            
            if self.models['risk_assessor'] is not None:
                predictions['risk_score'] = 1 - self.models['risk_assessor'].predict_proba(features_scaled[:, [3, 4, 5, 2]])[0][1]
            else:
                predictions['risk_score'] = 0.5
            
            # Get OpenAI sentiment if available
            openai_sentiment = 0.0
            if self.openai_client and self.training_config['openai_enhancement']:
                openai_sentiment = await self._get_openai_signal_sentiment(signal_data)
            
            # Calculate model consensus
            model_consensus = np.mean([
                predictions['win_probability'],
                (predictions['expected_profit'] + 0.05) / 0.1,  # Normalize to 0-1
                1 - predictions['risk_score']
            ])
            
            # Combine with OpenAI sentiment
            final_confidence = (model_consensus * 0.7 + openai_sentiment * 0.3) if openai_sentiment > 0 else model_consensus
            
            # Generate reasoning
            reasoning = self._generate_prediction_reasoning(predictions, openai_sentiment)
            
            # Calculate optimal levels
            entry_price = signal_data.get('entry_price', 0)
            volatility = signal_data.get('volatility', 0.02)
            
            prediction = SignalPrediction(
                symbol=signal_data.get('symbol', ''),
                direction=signal_data.get('direction', 'BUY'),
                confidence=final_confidence,
                expected_profit=predictions['expected_profit'],
                risk_score=predictions['risk_score'],
                optimal_entry=entry_price,
                optimal_sl=entry_price * (1 - volatility * 2) if signal_data.get('direction') == 'BUY' else entry_price * (1 + volatility * 2),
                optimal_tp=[
                    entry_price * (1 + volatility * 2) if signal_data.get('direction') == 'BUY' else entry_price * (1 - volatility * 2),
                    entry_price * (1 + volatility * 3) if signal_data.get('direction') == 'BUY' else entry_price * (1 - volatility * 3),
                    entry_price * (1 + volatility * 5) if signal_data.get('direction') == 'BUY' else entry_price * (1 - volatility * 5)
                ],
                model_consensus=model_consensus,
                openai_sentiment=openai_sentiment,
                reasoning=reasoning
            )
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"❌ Signal prediction failed: {e}")
            return self._create_fallback_prediction(signal_data)

    async def _get_openai_signal_sentiment(self, signal_data: Dict[str, Any]) -> float:
        """Get OpenAI sentiment analysis for signal"""
        try:
            if not self.openai_client:
                return 0.0
            
            prompt = f"""
Analyze this trading signal and rate its quality from 0.0 to 1.0:

Symbol: {signal_data.get('symbol')}
Direction: {signal_data.get('direction')}
RSI: {signal_data.get('rsi', 50)}
MACD: {signal_data.get('macd', 0)}
Volatility: {signal_data.get('volatility', 0.02)}
Trend Strength: {signal_data.get('trend_strength', 0)}
Market Regime: {signal_data.get('market_regime', 'neutral')}

Respond with JSON:
{{"sentiment_score": float between 0.0 and 1.0, "reasoning": "brief explanation"}}
"""

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a trading signal analyst."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get('sentiment_score', 0.0)
            
        except Exception as e:
            self.logger.debug(f"OpenAI sentiment failed: {e}")
            return 0.0

    def _generate_prediction_reasoning(self, predictions: Dict[str, Any], openai_sentiment: float) -> str:
        """Generate human-readable reasoning"""
        reasoning_parts = []
        
        win_prob = predictions.get('win_probability', 0)
        if win_prob > 0.7:
            reasoning_parts.append(f"High win probability ({win_prob:.1%})")
        elif win_prob < 0.4:
            reasoning_parts.append(f"Low win probability ({win_prob:.1%})")
        
        expected_profit = predictions.get('expected_profit', 0)
        if expected_profit > 0.03:
            reasoning_parts.append(f"Strong profit potential ({expected_profit:.1%})")
        
        risk_score = predictions.get('risk_score', 0.5)
        if risk_score < 0.3:
            reasoning_parts.append("Low risk")
        elif risk_score > 0.7:
            reasoning_parts.append("High risk")
        
        if openai_sentiment > 0.7:
            reasoning_parts.append("Positive AI sentiment")
        
        return "; ".join(reasoning_parts) if reasoning_parts else "Standard analysis"

    def _encode_market_regime(self, regime: str) -> int:
        """Encode market regime to numeric value"""
        regime_map = {
            'trending': 1,
            'ranging': 0,
            'volatile': 2,
            'neutral': 0
        }
        return regime_map.get(regime, 0)

    def _get_previous_accuracy(self, model_name: str) -> float:
        """Get previous accuracy for comparison"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT accuracy FROM training_results 
                WHERE model_type = ? 
                ORDER BY timestamp DESC LIMIT 1
            ''', (model_name,))
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else 0.0
            
        except Exception:
            return 0.0

    async def _evaluate_training_improvement(self, results: List[TrainingResult]) -> float:
        """Evaluate overall training improvement"""
        if not results:
            return 0.0
        
        improvements = [r.performance_improvement for r in results]
        return np.mean(improvements)

    async def _store_trade_outcome(self, trade_data: Dict[str, Any]):
        """Store trade outcome in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trade_outcomes (
                    symbol, direction, entry_price, exit_price, profit_loss,
                    win, indicators, market_conditions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data.get('symbol', ''),
                trade_data.get('direction', ''),
                trade_data.get('entry_price', 0),
                trade_data.get('exit_price', 0),
                trade_data.get('profit_loss', 0),
                trade_data.get('win', 0),
                json.dumps(trade_data.get('indicators', {})),
                json.dumps(trade_data.get('market_conditions', {}))
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"❌ Trade outcome storage failed: {e}")

    async def _store_training_result(self, result: TrainingResult):
        """Store training result in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            model_state = pickle.dumps(self.models.get(result.model_type))
            
            cursor.execute('''
                INSERT INTO training_results (
                    model_type, accuracy, precision_score, recall_score,
                    training_samples, learning_rate, performance_improvement,
                    openai_insights, model_state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.model_type,
                result.accuracy,
                result.precision,
                result.recall,
                result.training_samples,
                result.learning_rate,
                result.performance_improvement,
                json.dumps(result.openai_insights) if result.openai_insights else None,
                model_state
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"❌ Training result storage failed: {e}")

    def _save_models(self):
        """Save trained models to disk"""
        try:
            model_dir = Path("SignalMaestro/ml_models")
            model_dir.mkdir(parents=True, exist_ok=True)
            
            for model_name, model in self.models.items():
                if model is not None:
                    with open(model_dir / f"{model_name}.pkl", 'wb') as f:
                        pickle.dump(model, f)
            
            # Save scalers
            with open(model_dir / "feature_scaler.pkl", 'wb') as f:
                pickle.dump(self.scalers['features'], f)
            
            self.logger.info("💾 Models saved successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Model saving failed: {e}")

    def _load_models(self):
        """Load trained models from disk"""
        try:
            model_dir = Path("SignalMaestro/ml_models")
            if not model_dir.exists():
                return
            
            for model_name in self.models.keys():
                model_path = model_dir / f"{model_name}.pkl"
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
            
            # Load scalers
            scaler_path = model_dir / "feature_scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scalers['features'] = pickle.load(f)
            
            self.logger.info("📂 Models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Model loading failed: {e}")

    def _create_fallback_prediction(self, signal_data: Dict[str, Any]) -> SignalPrediction:
        """Create fallback prediction when ML fails"""
        entry_price = signal_data.get('entry_price', 0)
        volatility = signal_data.get('volatility', 0.02)
        
        return SignalPrediction(
            symbol=signal_data.get('symbol', ''),
            direction=signal_data.get('direction', 'BUY'),
            confidence=0.5,
            expected_profit=0.01,
            risk_score=0.5,
            optimal_entry=entry_price,
            optimal_sl=entry_price * 0.98,
            optimal_tp=[entry_price * 1.02, entry_price * 1.03, entry_price * 1.05],
            model_consensus=0.5,
            openai_sentiment=0.0,
            reasoning="Fallback prediction - models not ready"
        )

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'models_trained': sum(1 for m in self.models.values() if m is not None),
            'total_trades': len(self.trade_outcomes),
            'last_training': self.last_training.isoformat() if self.last_training else None,
            'is_training': self.is_training,
            'openai_enabled': self.openai_client is not None,
            'auto_train_enabled': True,
            'retrain_interval': self.training_config['retrain_interval'],
            'trades_until_retrain': self.training_config['retrain_interval'] - self.trade_counter,
            'performance_threshold': self.training_config['performance_threshold']
        }


# Global instance
_auto_train_system = None

def get_auto_train_system() -> AdvancedAutoTrainMLSystem:
    """Get global auto-train system instance"""
    global _auto_train_system
    if _auto_train_system is None:
        _auto_train_system = AdvancedAutoTrainMLSystem()
    return _auto_train_system
