"""
Machine learning models for signal generation
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

class ConsistentDataTransformer(BaseEstimator, TransformerMixin):
    """Ensures consistent data format between training and prediction"""
    
    def __init__(self):
        self.feature_names_ = None
        self.n_features_ = None
    
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            self.n_features_ = len(self.feature_names_)
        else:
            self.n_features_ = X.shape[1] if len(X.shape) > 1 else 1
            self.feature_names_ = [f'feature_{i}' for i in range(self.n_features_)]
        return self
    
    def transform(self, X):
        # Always return numpy array with consistent shape
        if isinstance(X, pd.DataFrame):
            # Ensure we have the same features as during training
            if self.feature_names_:
                missing_features = set(self.feature_names_) - set(X.columns)
                if missing_features:
                    logger.warning(f"Missing features during prediction: {missing_features}")
                    # Add missing features as zeros
                    for feature in missing_features:
                        X[feature] = 0.0
                
                # Reorder columns to match training order
                X = X[self.feature_names_]
            
            return X.values
        else:
            # Already numpy array, ensure correct shape
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            
            # Ensure correct number of features
            if X.shape[1] != self.n_features_:
                logger.warning(f"Feature count mismatch: expected {self.n_features_}, got {X.shape[1]}")
                if X.shape[1] < self.n_features_:
                    # Pad with zeros
                    padding = np.zeros((X.shape[0], self.n_features_ - X.shape[1]))
                    X = np.concatenate([X, padding], axis=1)
                else:
                    # Truncate
                    X = X[:, :self.n_features_]
            
            return X

class MLSignalGenerator:
    """Machine learning-based signal generator"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.feature_columns = {}
        self.model_performance = {}
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models"""
        try:
            # Random Forest with consistent data handling
            rf_pipeline = Pipeline([
                ('data_transformer', ConsistentDataTransformer()),
                ('scaler', RobustScaler()),
                ('classifier', RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                ))
            ])
            
            # Gradient Boosting with consistent data handling
            gb_pipeline = Pipeline([
                ('data_transformer', ConsistentDataTransformer()),
                ('scaler', RobustScaler()),
                ('classifier', GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                ))
            ])
            
            # Logistic Regression with consistent data handling
            lr_pipeline = Pipeline([
                ('data_transformer', ConsistentDataTransformer()),
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(
                    random_state=42,
                    max_iter=1000
                ))
            ])
            
            self.models = {
                'random_forest': rf_pipeline,
                'gradient_boosting': gb_pipeline,
                'logistic_regression': lr_pipeline
            }
            
            logger.info(f"✅ Initialized {len(self.models)} ML models with consistent data handling")
            
        except Exception as e:
            logger.error(f"❌ Error initializing models: {e}")
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, symbol: str) -> Dict[str, Any]:
        """Train all models for a specific symbol"""
        try:
            if X.empty or y.empty:
                logger.warning(f"Empty training data for {symbol}")
                return {}
            
            logger.info(f"🎓 Training models for {symbol} with {len(X)} samples")
            
            # Store feature columns for this symbol
            self.feature_columns[symbol] = X.columns.tolist()
            
            # Ensure X is a DataFrame with proper column names
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=self.feature_columns.get(symbol, [f'feature_{i}' for i in range(X.shape[1])]))
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            results = {}
            
            for model_name, model in self.models.items():
                try:
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    auc = roc_auc_score(y_test, y_pred_proba)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                    
                    # Store results
                    results[model_name] = {
                        'accuracy': accuracy,
                        'auc': auc,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'n_samples': len(X_train)
                    }
                    
                    # Store model
                    model_key = f"{symbol}_{model_name}"
                    self.models[model_key] = model
                    
                    # Save model
                    self._save_model(model, model_key)
                    
                    logger.info(f"✅ {model_name} for {symbol}: Accuracy={accuracy:.3f}, AUC={auc:.3f}")
                    
                except Exception as e:
                    logger.error(f"❌ Error training {model_name} for {symbol}: {e}")
                    continue
            
            # Store performance metrics
            self.model_performance[symbol] = results
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error training models for {symbol}: {e}")
            return {}
    
    def generate_signal(self, X: pd.DataFrame, symbol: str, method: str = 'ensemble') -> Dict[str, Any]:
        """Generate trading signal using ML models"""
        try:
            if X.empty:
                return {'signal': 0, 'confidence': 0.0, 'probabilities': {}}
            
            # Ensure we have the right features and consistent format
            if symbol in self.feature_columns:
                expected_features = self.feature_columns[symbol]
                
                # If X doesn't have the expected columns, try to create a consistent DataFrame
                if not all(col in X.columns for col in expected_features):
                    logger.debug(f"Adjusting features for {symbol} prediction")
                    
                    # Create a new DataFrame with the expected features
                    adjusted_X = pd.DataFrame(index=X.index)
                    
                    for feature in expected_features:
                        if feature in X.columns:
                            adjusted_X[feature] = X[feature]
                        else:
                            # Use a default value for missing features
                            adjusted_X[feature] = 0.0
                    
                    X = adjusted_X
                else:
                    # Reorder columns to match training order
                    X = X[expected_features]
            
            # Ensure X is a proper DataFrame
            if not isinstance(X, pd.DataFrame):
                feature_names = self.feature_columns.get(symbol, [f'feature_{i}' for i in range(X.shape[1])])
                X = pd.DataFrame(X, columns=feature_names)
            
            predictions = {}
            probabilities = {}
            
            # Get predictions from all models
            for model_name in ['random_forest', 'gradient_boosting', 'logistic_regression']:
                model_key = f"{symbol}_{model_name}"
                
                if model_key in self.models:
                    try:
                        model = self.models[model_key]
                        
                        # Get the last row for prediction
                        X_pred = X.tail(1)
                        
                        # Make prediction
                        pred = model.predict(X_pred)[0]
                        pred_proba = model.predict_proba(X_pred)[0, 1]
                        
                        predictions[model_name] = pred
                        probabilities[model_name] = pred_proba
                        
                    except Exception as e:
                        logger.error(f"Error getting prediction from {model_name} for {symbol}: {e}")
                        continue
            
            if not predictions:
                return {'signal': 0, 'confidence': 0.0, 'probabilities': {}}
            
            # Generate ensemble signal
            if method == 'ensemble':
                signal, confidence = self._ensemble_signal(predictions, probabilities)
            elif method == 'best':
                signal, confidence = self._best_model_signal(symbol, predictions, probabilities)
            else:
                signal, confidence = self._majority_vote_signal(predictions, probabilities)
            
            result = {
                'signal': signal,
                'confidence': confidence,
                'probabilities': probabilities,
                'individual_predictions': predictions
            }
            
            logger.debug(f"Generated signal for {symbol}: {signal} (confidence: {confidence:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error generating signal for {symbol}: {e}")
            return {'signal': 0, 'confidence': 0.0, 'probabilities': {}}
    
    def _ensemble_signal(self, predictions: Dict[str, int], probabilities: Dict[str, float]) -> Tuple[int, float]:
        """Generate ensemble signal using weighted average"""
        try:
            if not predictions:
                return 0, 0.0
            
            # Weight models by their performance (if available)
            weights = {'random_forest': 0.4, 'gradient_boosting': 0.4, 'logistic_regression': 0.2}
            
            weighted_prob = 0.0
            total_weight = 0.0
            
            for model_name, prob in probabilities.items():
                weight = weights.get(model_name, 1.0)
                weighted_prob += prob * weight
                total_weight += weight
            
            if total_weight > 0:
                final_prob = weighted_prob / total_weight
            else:
                final_prob = np.mean(list(probabilities.values()))
            
            # Generate signal based on probability threshold
            signal = 1 if final_prob > 0.55 else -1 if final_prob < 0.45 else 0
            confidence = float(abs(final_prob - 0.5) * 2)  # Scale to 0-1
            
            return signal, confidence
            
        except Exception as e:
            logger.error(f"Error in ensemble signal: {e}")
            return 0, 0.0
    
    def _best_model_signal(self, symbol: str, predictions: Dict[str, int], probabilities: Dict[str, float]) -> Tuple[int, float]:
        """Use signal from best performing model"""
        try:
            if symbol not in self.model_performance:
                return self._majority_vote_signal(predictions, probabilities)
            
            # Find best model by AUC score
            best_model = None
            best_auc = 0
            
            for model_name, metrics in self.model_performance[symbol].items():
                if metrics['auc'] > best_auc:
                    best_auc = metrics['auc']
                    best_model = model_name
            
            if best_model and best_model in predictions:
                signal = 1 if predictions[best_model] == 1 else -1
                confidence = probabilities.get(best_model, 0.5)
                confidence = abs(confidence - 0.5) * 2
                return signal, confidence
            else:
                return self._majority_vote_signal(predictions, probabilities)
            
        except Exception as e:
            logger.error(f"Error in best model signal: {e}")
            return 0, 0.0
    
    def _majority_vote_signal(self, predictions: Dict[str, int], probabilities: Dict[str, float]) -> Tuple[int, float]:
        """Generate signal using majority vote"""
        try:
            if not predictions:
                return 0, 0.0
            
            # Count votes
            buy_votes = sum(1 for pred in predictions.values() if pred == 1)
            sell_votes = sum(1 for pred in predictions.values() if pred == 0)
            
            total_votes = len(predictions)
            
            if buy_votes > sell_votes:
                signal = 1
                confidence = buy_votes / total_votes
            elif sell_votes > buy_votes:
                signal = -1
                confidence = sell_votes / total_votes
            else:
                signal = 0
                confidence = 0.5
            
            # Adjust confidence based on average probability
            avg_prob = np.mean(list(probabilities.values()))
            confidence = float((confidence + abs(avg_prob - 0.5) * 2) / 2)
            
            return signal, confidence
            
        except Exception as e:
            logger.error(f"Error in majority vote signal: {e}")
            return 0, 0.0
    
    def _save_model(self, model, model_key: str):
        """Save model to disk"""
        try:
            model_file = self.model_dir / f"{model_key}.joblib"
            joblib.dump(model, model_file)
            logger.debug(f"Saved model: {model_file}")
        except Exception as e:
            logger.error(f"Error saving model {model_key}: {e}")
    
    def load_model(self, model_key: str):
        """Load model from disk"""
        try:
            model_file = self.model_dir / f"{model_key}.joblib"
            if model_file.exists():
                model = joblib.load(model_file)
                self.models[model_key] = model
                logger.debug(f"Loaded model: {model_file}")
                return model
            else:
                logger.warning(f"Model file not found: {model_file}")
                return None
        except Exception as e:
            logger.error(f"Error loading model {model_key}: {e}")
            return None
    
    def get_feature_importance(self, symbol: str, model_name: str = 'random_forest') -> Dict[str, float]:
        """Get feature importance from trained model"""
        try:
            model_key = f"{symbol}_{model_name}"
            
            if model_key not in self.models:
                return {}
            
            model = self.models[model_key]
            
            # Get the classifier from the pipeline
            if hasattr(model, 'named_steps'):
                classifier = model.named_steps['classifier']
            else:
                classifier = model
            
            if hasattr(classifier, 'feature_importances_'):
                importance_values = classifier.feature_importances_
                feature_names = self.feature_columns.get(symbol, [])
                
                if len(importance_values) == len(feature_names):
                    importance_dict = dict(zip(feature_names, importance_values))
                    # Sort by importance
                    return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def get_model_performance(self, symbol: str) -> Dict[str, Any]:
        """Get performance metrics for models"""
        return self.model_performance.get(symbol, {})
