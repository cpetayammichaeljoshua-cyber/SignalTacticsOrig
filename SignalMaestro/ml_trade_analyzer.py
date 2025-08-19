
#!/usr/bin/env python3
"""
Machine Learning Trade Analyzer
Learns from losses and analyzes past trades to improve scalping performance
"""

import numpy as np
import pandas as pd
import json
import pickle
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import sqlite3
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class MLTradeAnalyzer:
    """Machine Learning analyzer for trade performance improvement"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model_dir = Path("ml_models")
        self.model_dir.mkdir(exist_ok=True)
        
        # Models for different aspects
        self.loss_prediction_model = None
        self.signal_strength_model = None
        self.entry_timing_model = None
        self.scaler = StandardScaler()
        
        # Trade database
        self.db_path = "trade_learning.db"
        self._initialize_database()
        
        # Learning parameters
        self.min_trades_for_learning = 10
        self.feature_importance_threshold = 0.01
        
        # Model performance tracking
        self.model_performance = {
            'loss_prediction_accuracy': 0.0,
            'signal_strength_accuracy': 0.0,
            'entry_timing_accuracy': 0.0,
            'last_training_time': None,
            'trades_analyzed': 0
        }
        
        self.logger.info("🧠 ML Trade Analyzer initialized")
    
    def _initialize_database(self):
        """Initialize SQLite database for trade storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    stop_loss REAL,
                    take_profit_1 REAL,
                    take_profit_2 REAL,
                    take_profit_3 REAL,
                    signal_strength REAL,
                    leverage INTEGER,
                    position_size REAL,
                    trade_result TEXT,
                    profit_loss REAL,
                    duration_minutes REAL,
                    entry_time TIMESTAMP,
                    exit_time TIMESTAMP,
                    market_conditions TEXT,
                    indicators_data TEXT,
                    cvd_trend TEXT,
                    volatility REAL,
                    volume_ratio REAL,
                    ema_alignment BOOLEAN,
                    rsi_value REAL,
                    macd_signal TEXT,
                    lessons_learned TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create learning insights table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    insight_type TEXT NOT NULL,
                    pattern_description TEXT,
                    success_rate REAL,
                    recommendation TEXT,
                    confidence_score REAL,
                    trades_analyzed INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("📊 Trade learning database initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
    
    async def record_trade(self, trade_data: Dict[str, Any]):
        """Record a trade for machine learning analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trades (
                    symbol, direction, entry_price, exit_price, stop_loss,
                    take_profit_1, take_profit_2, take_profit_3, signal_strength,
                    leverage, position_size, trade_result, profit_loss,
                    duration_minutes, entry_time, exit_time, market_conditions,
                    indicators_data, cvd_trend, volatility, volume_ratio,
                    ema_alignment, rsi_value, macd_signal, lessons_learned
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                trade_data.get('position_size'),
                trade_data.get('trade_result'),
                trade_data.get('profit_loss'),
                trade_data.get('duration_minutes'),
                trade_data.get('entry_time'),
                trade_data.get('exit_time'),
                json.dumps(trade_data.get('market_conditions', {})),
                json.dumps(trade_data.get('indicators_data', {})),
                trade_data.get('cvd_trend'),
                trade_data.get('volatility'),
                trade_data.get('volume_ratio'),
                trade_data.get('ema_alignment'),
                trade_data.get('rsi_value'),
                trade_data.get('macd_signal'),
                trade_data.get('lessons_learned')
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"📝 Trade recorded for ML analysis: {trade_data.get('symbol')} {trade_data.get('trade_result')}")
            
            # Trigger learning if we have enough data
            if self._get_trade_count() >= self.min_trades_for_learning:
                await self.analyze_and_learn()
                
        except Exception as e:
            self.logger.error(f"Error recording trade: {e}")
    
    def _get_trade_count(self) -> int:
        """Get total number of recorded trades"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM trades")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            self.logger.error(f"Error getting trade count: {e}")
            return 0
    
    async def analyze_and_learn(self):
        """Main learning function that analyzes trades and updates models"""
        try:
            self.logger.info("🧠 Starting ML analysis and learning process...")
            
            # Get trade data
            trades_df = self._get_trades_dataframe()
            if len(trades_df) < self.min_trades_for_learning:
                self.logger.warning(f"Not enough trades for learning: {len(trades_df)}")
                return
            
            # 1. Learn from losses
            loss_insights = await self._analyze_losses(trades_df)
            
            # 2. Analyze successful patterns
            success_patterns = await self._analyze_successful_patterns(trades_df)
            
            # 3. Train prediction models
            await self._train_prediction_models(trades_df)
            
            # 4. Generate trading insights
            insights = await self._generate_trading_insights(trades_df)
            
            # 5. Update model performance metrics
            self._update_performance_metrics(trades_df)
            
            # 6. Store insights
            await self._store_insights(loss_insights + success_patterns + insights)
            
            self.logger.info(f"✅ ML analysis complete. Analyzed {len(trades_df)} trades")
            
        except Exception as e:
            self.logger.error(f"Error in ML analysis: {e}")
    
    def _get_trades_dataframe(self) -> pd.DataFrame:
        """Get trades data as pandas DataFrame"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT * FROM trades ORDER BY created_at DESC LIMIT 1000"
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Parse JSON fields
            if 'market_conditions' in df.columns:
                df['market_conditions'] = df['market_conditions'].apply(
                    lambda x: json.loads(x) if x else {}
                )
            if 'indicators_data' in df.columns:
                df['indicators_data'] = df['indicators_data'].apply(
                    lambda x: json.loads(x) if x else {}
                )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting trades DataFrame: {e}")
            return pd.DataFrame()
    
    async def _analyze_losses(self, trades_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze losing trades to identify patterns and lessons"""
        try:
            loss_insights = []
            
            # Filter losing trades
            losing_trades = trades_df[trades_df['trade_result'].isin(['LOSS', 'STOP_LOSS'])]
            
            if len(losing_trades) == 0:
                return []
            
            # Analyze loss patterns by symbol
            symbol_losses = losing_trades.groupby('symbol').agg({
                'profit_loss': ['count', 'mean'],
                'signal_strength': 'mean',
                'volatility': 'mean'
            }).round(2)
            
            for symbol in symbol_losses.index:
                loss_count = symbol_losses.loc[symbol, ('profit_loss', 'count')]
                avg_loss = symbol_losses.loc[symbol, ('profit_loss', 'mean')]
                avg_signal_strength = symbol_losses.loc[symbol, ('signal_strength', 'mean')]
                
                if loss_count >= 3:  # Pattern detection threshold
                    insight = {
                        'type': 'loss_pattern',
                        'symbol': symbol,
                        'pattern': f"High loss frequency on {symbol}",
                        'recommendation': f"Reduce position size or avoid {symbol} temporarily",
                        'confidence': min(loss_count / 10 * 100, 95),
                        'data': {
                            'loss_count': loss_count,
                            'avg_loss': avg_loss,
                            'avg_signal_strength': avg_signal_strength
                        }
                    }
                    loss_insights.append(insight)
            
            # Analyze loss patterns by market conditions
            condition_losses = {}
            for _, trade in losing_trades.iterrows():
                conditions = trade.get('market_conditions', {})
                for condition, value in conditions.items():
                    if condition not in condition_losses:
                        condition_losses[condition] = []
                    condition_losses[condition].append(value)
            
            # Analyze signal strength vs losses
            if len(losing_trades) >= 5:
                low_strength_losses = losing_trades[losing_trades['signal_strength'] < 85]
                if len(low_strength_losses) > len(losing_trades) * 0.6:
                    insight = {
                        'type': 'signal_strength_lesson',
                        'pattern': "Majority of losses from signals < 85% strength",
                        'recommendation': "Increase minimum signal strength to 90%",
                        'confidence': 85,
                        'data': {
                            'low_strength_loss_ratio': len(low_strength_losses) / len(losing_trades),
                            'avg_loss_signal_strength': losing_trades['signal_strength'].mean()
                        }
                    }
                    loss_insights.append(insight)
            
            # CVD divergence analysis
            cvd_losses = losing_trades[losing_trades['cvd_trend'].notna()]
            if len(cvd_losses) >= 3:
                bearish_cvd_losses = cvd_losses[cvd_losses['cvd_trend'] == 'bearish']
                if len(bearish_cvd_losses) > len(cvd_losses) * 0.7:
                    insight = {
                        'type': 'cvd_lesson',
                        'pattern': "High loss rate during bearish CVD trend",
                        'recommendation': "Avoid long positions during bearish CVD",
                        'confidence': 80,
                        'data': {
                            'bearish_cvd_loss_ratio': len(bearish_cvd_losses) / len(cvd_losses)
                        }
                    }
                    loss_insights.append(insight)
            
            return loss_insights
            
        except Exception as e:
            self.logger.error(f"Error analyzing losses: {e}")
            return []
    
    async def _analyze_successful_patterns(self, trades_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze successful trades to identify winning patterns"""
        try:
            success_insights = []
            
            # Filter successful trades
            winning_trades = trades_df[trades_df['trade_result'].isin(['PROFIT', 'TP1', 'TP2', 'TP3'])]
            
            if len(winning_trades) == 0:
                return []
            
            # Analyze winning patterns by signal strength
            high_strength_wins = winning_trades[winning_trades['signal_strength'] >= 90]
            if len(high_strength_wins) > 0:
                win_rate = len(high_strength_wins) / len(winning_trades)
                if win_rate > 0.7:
                    insight = {
                        'type': 'success_pattern',
                        'pattern': "High win rate with signal strength >= 90%",
                        'recommendation': "Prioritize signals with 90%+ strength",
                        'confidence': min(win_rate * 100, 95),
                        'data': {
                            'high_strength_win_rate': win_rate,
                            'avg_profit': high_strength_wins['profit_loss'].mean()
                        }
                    }
                    success_insights.append(insight)
            
            # Analyze by leverage patterns
            leverage_analysis = winning_trades.groupby('leverage').agg({
                'profit_loss': ['count', 'mean', 'std']
            }).round(2)
            
            best_leverage = None
            best_performance = 0
            
            for leverage in leverage_analysis.index:
                count = leverage_analysis.loc[leverage, ('profit_loss', 'count')]
                avg_profit = leverage_analysis.loc[leverage, ('profit_loss', 'mean')]
                
                if count >= 3 and avg_profit > best_performance:
                    best_performance = avg_profit
                    best_leverage = leverage
            
            if best_leverage:
                insight = {
                    'type': 'leverage_optimization',
                    'pattern': f"Best performance with {best_leverage}x leverage",
                    'recommendation': f"Consider using {best_leverage}x leverage more frequently",
                    'confidence': 75,
                    'data': {
                        'optimal_leverage': best_leverage,
                        'avg_profit': best_performance
                    }
                }
                success_insights.append(insight)
            
            # Analyze timeframe patterns
            duration_wins = winning_trades[winning_trades['duration_minutes'].notna()]
            if len(duration_wins) >= 5:
                quick_wins = duration_wins[duration_wins['duration_minutes'] <= 30]
                if len(quick_wins) > len(duration_wins) * 0.6:
                    insight = {
                        'type': 'timing_pattern',
                        'pattern': "Majority of wins occur within 30 minutes",
                        'recommendation': "Focus on quick scalping entries/exits",
                        'confidence': 80,
                        'data': {
                            'quick_win_ratio': len(quick_wins) / len(duration_wins),
                            'avg_quick_profit': quick_wins['profit_loss'].mean()
                        }
                    }
                    success_insights.append(insight)
            
            return success_insights
            
        except Exception as e:
            self.logger.error(f"Error analyzing successful patterns: {e}")
            return []
    
    async def _train_prediction_models(self, trades_df: pd.DataFrame):
        """Train ML models for trade prediction"""
        try:
            if len(trades_df) < 20:
                self.logger.warning("Not enough data for model training")
                return
            
            # Prepare features
            features = self._prepare_features(trades_df)
            
            if features is None or len(features) == 0:
                return
            
            # Train loss prediction model
            await self._train_loss_prediction_model(features, trades_df)
            
            # Train signal strength optimization model
            await self._train_signal_strength_model(features, trades_df)
            
            # Train entry timing model
            await self._train_entry_timing_model(features, trades_df)
            
            # Save models
            self._save_models()
            
            self.logger.info("🤖 ML models trained and saved")
            
        except Exception as e:
            self.logger.error(f"Error training prediction models: {e}")
    
    def _prepare_features(self, trades_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepare features for ML models"""
        try:
            features = pd.DataFrame()
            
            # Basic features
            features['signal_strength'] = trades_df['signal_strength']
            features['leverage'] = trades_df['leverage']
            features['volatility'] = trades_df['volatility'].fillna(0)
            features['volume_ratio'] = trades_df['volume_ratio'].fillna(1)
            features['rsi_value'] = trades_df['rsi_value'].fillna(50)
            
            # Encode categorical features
            le_direction = LabelEncoder()
            le_cvd = LabelEncoder()
            le_macd = LabelEncoder()
            
            features['direction_encoded'] = le_direction.fit_transform(trades_df['direction'].fillna('BUY'))
            features['cvd_trend_encoded'] = le_cvd.fit_transform(trades_df['cvd_trend'].fillna('neutral'))
            features['macd_signal_encoded'] = le_macd.fit_transform(trades_df['macd_signal'].fillna('neutral'))
            features['ema_alignment'] = trades_df['ema_alignment'].fillna(False).astype(int)
            
            # Time-based features
            if 'entry_time' in trades_df.columns:
                trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
                features['hour'] = trades_df['entry_time'].dt.hour
                features['day_of_week'] = trades_df['entry_time'].dt.dayofweek
            
            # Remove rows with too many NaN values
            features = features.dropna()
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return None
    
    async def _train_loss_prediction_model(self, features: pd.DataFrame, trades_df: pd.DataFrame):
        """Train model to predict likely losses"""
        try:
            # Create binary target (1 = loss, 0 = profit)
            target = (trades_df['trade_result'].isin(['LOSS', 'STOP_LOSS'])).astype(int)
            
            # Align features and target
            common_indices = features.index.intersection(target.index)
            X = features.loc[common_indices]
            y = target.loc[common_indices]
            
            if len(X) < 10:
                return
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.loss_prediction_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.loss_prediction_model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.loss_prediction_model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.model_performance['loss_prediction_accuracy'] = accuracy
            self.logger.info(f"🎯 Loss prediction model accuracy: {accuracy:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error training loss prediction model: {e}")
    
    async def _train_signal_strength_model(self, features: pd.DataFrame, trades_df: pd.DataFrame):
        """Train model to optimize signal strength thresholds"""
        try:
            # Create target based on profitability
            target = trades_df['profit_loss'].fillna(0)
            
            # Align features and target
            common_indices = features.index.intersection(target.index)
            X = features.loc[common_indices]
            y = target.loc[common_indices]
            
            if len(X) < 10:
                return
            
            # Convert to classification problem (profitable vs not)
            y_binary = (y > 0).astype(int)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.signal_strength_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            )
            self.signal_strength_model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.signal_strength_model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.model_performance['signal_strength_accuracy'] = accuracy
            self.logger.info(f"📊 Signal strength model accuracy: {accuracy:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error training signal strength model: {e}")
    
    async def _train_entry_timing_model(self, features: pd.DataFrame, trades_df: pd.DataFrame):
        """Train model for optimal entry timing"""
        try:
            # Create target based on quick profits
            duration = trades_df['duration_minutes'].fillna(60)
            profit = trades_df['profit_loss'].fillna(0)
            
            # Good timing = profitable within 30 minutes
            target = ((duration <= 30) & (profit > 0)).astype(int)
            
            # Align features and target
            common_indices = features.index.intersection(target.index)
            X = features.loc[common_indices]
            y = target.loc[common_indices]
            
            if len(X) < 10 or y.sum() < 3:  # Need at least 3 positive examples
                return
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.entry_timing_model = RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                random_state=42
            )
            self.entry_timing_model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.entry_timing_model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.model_performance['entry_timing_accuracy'] = accuracy
            self.logger.info(f"⏰ Entry timing model accuracy: {accuracy:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error training entry timing model: {e}")
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            if self.loss_prediction_model:
                with open(self.model_dir / 'loss_prediction_model.pkl', 'wb') as f:
                    pickle.dump(self.loss_prediction_model, f)
            
            if self.signal_strength_model:
                with open(self.model_dir / 'signal_strength_model.pkl', 'wb') as f:
                    pickle.dump(self.signal_strength_model, f)
            
            if self.entry_timing_model:
                with open(self.model_dir / 'entry_timing_model.pkl', 'wb') as f:
                    pickle.dump(self.entry_timing_model, f)
            
            # Save scaler
            with open(self.model_dir / 'scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save performance metrics
            with open(self.model_dir / 'performance_metrics.json', 'w') as f:
                json.dump(self.model_performance, f, indent=2, default=str)
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            model_files = {
                'loss_prediction_model.pkl': 'loss_prediction_model',
                'signal_strength_model.pkl': 'signal_strength_model', 
                'entry_timing_model.pkl': 'entry_timing_model',
                'scaler.pkl': 'scaler'
            }
            
            for filename, attr_name in model_files.items():
                filepath = self.model_dir / filename
                if filepath.exists():
                    with open(filepath, 'rb') as f:
                        setattr(self, attr_name, pickle.load(f))
            
            # Load performance metrics
            metrics_file = self.model_dir / 'performance_metrics.json'
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    self.model_performance.update(json.load(f))
            
            self.logger.info("🤖 ML models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
    
    async def _generate_trading_insights(self, trades_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate actionable trading insights"""
        try:
            insights = []
            
            # Overall performance analysis
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['profit_loss'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            if win_rate < 0.6:
                insights.append({
                    'type': 'performance_warning',
                    'pattern': f"Win rate below 60% ({win_rate:.1%})",
                    'recommendation': "Review signal criteria and risk management",
                    'confidence': 90,
                    'data': {'current_win_rate': win_rate}
                })
            
            # Best performing symbols
            if total_trades >= 10:
                symbol_performance = trades_df.groupby('symbol')['profit_loss'].agg(['count', 'mean', 'sum'])
                symbol_performance = symbol_performance[symbol_performance['count'] >= 3]
                
                if len(symbol_performance) > 0:
                    best_symbol = symbol_performance['mean'].idxmax()
                    best_performance = symbol_performance.loc[best_symbol, 'mean']
                    
                    insights.append({
                        'type': 'symbol_recommendation',
                        'pattern': f"Best performing symbol: {best_symbol}",
                        'recommendation': f"Consider increasing allocation to {best_symbol}",
                        'confidence': 75,
                        'data': {
                            'symbol': best_symbol,
                            'avg_profit': best_performance,
                            'trade_count': symbol_performance.loc[best_symbol, 'count']
                        }
                    })
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating trading insights: {e}")
            return []
    
    def _update_performance_metrics(self, trades_df: pd.DataFrame):
        """Update model performance tracking"""
        try:
            self.model_performance['last_training_time'] = datetime.now().isoformat()
            self.model_performance['trades_analyzed'] = len(trades_df)
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    async def _store_insights(self, insights: List[Dict[str, Any]]):
        """Store insights in database"""
        try:
            if not insights:
                return
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for insight in insights:
                cursor.execute('''
                    INSERT INTO learning_insights (
                        insight_type, pattern_description, success_rate,
                        recommendation, confidence_score, trades_analyzed
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    insight.get('type'),
                    insight.get('pattern'),
                    insight.get('data', {}).get('win_rate', 0),
                    insight.get('recommendation'),
                    insight.get('confidence', 0),
                    self.model_performance['trades_analyzed']
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"💡 Stored {len(insights)} new insights")
            
        except Exception as e:
            self.logger.error(f"Error storing insights: {e}")
    
    def predict_trade_outcome(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict trade outcome using trained models"""
        try:
            if not self.loss_prediction_model or not self.scaler:
                return {'prediction': 'unknown', 'confidence': 0}
            
            # Prepare features
            features = self._prepare_signal_features(signal_data)
            if features is None:
                return {'prediction': 'unknown', 'confidence': 0}
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Predict loss probability
            loss_prob = self.loss_prediction_model.predict_proba(features_scaled)[0][1]
            
            # Predict signal strength optimization
            strength_pred = 0
            if self.signal_strength_model:
                strength_pred = self.signal_strength_model.predict_proba(features_scaled)[0][1]
            
            # Predict entry timing
            timing_pred = 0
            if self.entry_timing_model:
                timing_pred = self.entry_timing_model.predict_proba(features_scaled)[0][1]
            
            # Combine predictions
            overall_score = (1 - loss_prob) * 0.5 + strength_pred * 0.3 + timing_pred * 0.2
            
            prediction = 'favorable' if overall_score > 0.6 else 'unfavorable' if overall_score < 0.4 else 'neutral'
            
            return {
                'prediction': prediction,
                'confidence': overall_score * 100,
                'loss_probability': loss_prob * 100,
                'strength_score': strength_pred * 100,
                'timing_score': timing_pred * 100,
                'recommendation': self._get_prediction_recommendation(overall_score, loss_prob)
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting trade outcome: {e}")
            return {'prediction': 'unknown', 'confidence': 0, 'error': str(e)}
    
    def _prepare_signal_features(self, signal_data: Dict[str, Any]) -> Optional[List[float]]:
        """Prepare features from signal data for prediction"""
        try:
            features = [
                signal_data.get('signal_strength', 85),
                signal_data.get('optimal_leverage', 50),
                signal_data.get('volatility', 0.02),
                signal_data.get('volume_ratio', 1.0),
                signal_data.get('rsi', 50),
                1 if signal_data.get('direction') == 'BUY' else 0,
                1 if signal_data.get('cvd_trend') == 'bullish' else 0,
                1 if signal_data.get('macd_bullish', False) else 0,
                1 if signal_data.get('ema_bullish', False) else 0,
                datetime.now().hour,
                datetime.now().weekday()
            ]
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error preparing signal features: {e}")
            return None
    
    def _get_prediction_recommendation(self, overall_score: float, loss_prob: float) -> str:
        """Get trading recommendation based on predictions"""
        if loss_prob > 0.7:
            return "HIGH RISK - Consider skipping this trade"
        elif overall_score > 0.75:
            return "EXCELLENT - High probability trade"
        elif overall_score > 0.6:
            return "GOOD - Favorable conditions"
        elif overall_score > 0.4:
            return "NEUTRAL - Exercise caution"
        else:
            return "POOR - Unfavorable conditions"
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning progress and insights"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get trade statistics
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM trades")
            total_trades = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM trades WHERE profit_loss > 0")
            winning_trades = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM learning_insights")
            total_insights = cursor.fetchone()[0]
            
            # Get recent insights
            cursor.execute("SELECT * FROM learning_insights ORDER BY created_at DESC LIMIT 5")
            recent_insights = cursor.fetchall()
            
            conn.close()
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            return {
                'total_trades_analyzed': total_trades,
                'win_rate': win_rate,
                'total_insights_generated': total_insights,
                'model_performance': self.model_performance,
                'recent_insights': [
                    {
                        'type': insight[1],
                        'pattern': insight[2],
                        'recommendation': insight[4],
                        'confidence': insight[5]
                    }
                    for insight in recent_insights
                ],
                'learning_status': 'active' if total_trades >= self.min_trades_for_learning else 'collecting_data'
            }
            
        except Exception as e:
            self.logger.error(f"Error getting learning summary: {e}")
            return {'error': str(e)}
    
    def get_trade_recommendations(self, symbol: str) -> Dict[str, Any]:
        """Get specific recommendations for a symbol based on historical performance"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT * FROM trades WHERE symbol = ? ORDER BY created_at DESC LIMIT 20"
            df = pd.read_sql_query(query, conn, params=(symbol,))
            conn.close()
            
            if len(df) == 0:
                return {'recommendation': 'No historical data available'}
            
            # Analyze symbol-specific patterns
            win_rate = len(df[df['profit_loss'] > 0]) / len(df)
            avg_profit = df['profit_loss'].mean()
            best_leverage = df.groupby('leverage')['profit_loss'].mean().idxmax()
            avg_duration = df['duration_minutes'].mean()
            
            recommendation = "NEUTRAL"
            if win_rate > 0.7 and avg_profit > 0:
                recommendation = "FAVORABLE"
            elif win_rate < 0.4 or avg_profit < 0:
                recommendation = "AVOID"
            
            return {
                'symbol': symbol,
                'recommendation': recommendation,
                'historical_win_rate': win_rate,
                'avg_profit_loss': avg_profit,
                'optimal_leverage': best_leverage,
                'avg_trade_duration': avg_duration,
                'trade_count': len(df)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting trade recommendations for {symbol}: {e}")
            return {'error': str(e)}
