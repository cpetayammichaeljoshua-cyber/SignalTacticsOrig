"""
ML-based signal generation system
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import asyncio

from bot.data_processor import DataProcessor
from models.ml_models import MLSignalGenerator

logger = logging.getLogger(__name__)

class SignalGenerator:
    """Main signal generation orchestrator"""
    
    def __init__(self, config):
        self.config = config
        self.data_processor = DataProcessor(
            lookback_period=config.lookback_period,
            feature_window=config.feature_window
        )
        self.ml_generator = MLSignalGenerator()
        
        # Signal history
        self.signal_history = {}
        self.last_retrain = {}
        
        # Performance tracking
        self.signal_performance = {}
        
    async def initialize(self):
        """Initialize the signal generator"""
        try:
            logger.info("🔧 Initializing ML Signal Generator...")
            
            # Try to load existing models
            for symbol in self.config.symbols:
                for model_name in ['random_forest', 'gradient_boosting', 'logistic_regression']:
                    model_key = f"{symbol}_{model_name}"
                    self.ml_generator.load_model(model_key)
            
            logger.info("✅ ML Signal Generator initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing signal generator: {e}")
            raise
    
    async def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Generate signals for all symbols"""
        signals = {}
        
        for symbol, data in market_data.items():
            try:
                signal = await self.generate_signal(symbol, data)
                signals[symbol] = signal
            except Exception as e:
                logger.error(f"❌ Error generating signal for {symbol}: {e}")
                signals[symbol] = {'signal': 0, 'confidence': 0.0, 'reason': f'Error: {e}'}
        
        return signals
    
    async def generate_signal(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate signal for a single symbol"""
        try:
            if data.empty:
                return {'signal': 0, 'confidence': 0.0, 'reason': 'No data available'}
            
            # Check if we need to retrain models
            await self._check_and_retrain(symbol, data)
            
            # Process data
            processed_data = self.data_processor.process_ohlcv_data(data)
            
            if processed_data.empty:
                return {'signal': 0, 'confidence': 0.0, 'reason': 'Data processing failed'}
            
            # Generate ML signal
            ml_signal = self.ml_generator.generate_signal(processed_data, symbol)
            
            # Add technical analysis confirmation
            tech_signal = self._get_technical_confirmation(processed_data)
            
            # Combine signals
            final_signal = self._combine_signals(ml_signal, tech_signal)
            
            # Add market context
            market_context = self._get_market_context(processed_data)
            final_signal.update(market_context)
            
            # Store signal history
            self._store_signal_history(symbol, final_signal)
            
            # Add timing and strength analysis
            final_signal.update({
                'timestamp': datetime.now(),
                'data_points': len(processed_data),
                'symbol': symbol
            })
            
            logger.debug(f"🎯 Generated signal for {symbol}: {final_signal['signal']} (confidence: {final_signal['confidence']:.3f})")
            
            return final_signal
            
        except Exception as e:
            logger.error(f"❌ Error generating signal for {symbol}: {e}")
            return {'signal': 0, 'confidence': 0.0, 'reason': f'Error: {e}'}
    
    async def _check_and_retrain(self, symbol: str, data: pd.DataFrame):
        """Check if models need retraining and retrain if necessary"""
        try:
            current_time = datetime.now()
            
            # Check if we need to retrain
            if symbol not in self.last_retrain:
                should_retrain = True
            else:
                time_since_retrain = current_time - self.last_retrain[symbol]
                should_retrain = time_since_retrain.total_seconds() > (self.config.retrain_interval * 3600)
            
            if should_retrain and len(data) >= self.config.lookback_period:
                logger.info(f"🎓 Retraining models for {symbol}...")
                
                # Process data for training
                processed_data = self.data_processor.process_ohlcv_data(data)
                
                if not processed_data.empty:
                    # Prepare training data
                    X, y = self.data_processor.prepare_training_data(processed_data)
                    
                    if not X.empty and not y.empty and len(y.unique()) > 1:
                        # Train models
                        results = self.ml_generator.train_models(X, y, symbol)
                        
                        if results:
                            self.last_retrain[symbol] = current_time
                            logger.info(f"✅ Models retrained for {symbol}")
                        else:
                            logger.warning(f"⚠️ Model retraining failed for {symbol}")
                    else:
                        logger.warning(f"⚠️ Insufficient training data for {symbol}")
                else:
                    logger.warning(f"⚠️ Data processing failed for {symbol}")
            
        except Exception as e:
            logger.error(f"❌ Error in model retraining for {symbol}: {e}")
    
    def _get_technical_confirmation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get technical analysis confirmation"""
        try:
            if data.empty:
                return {'tech_signal': 0, 'tech_strength': 0.0}
            
            latest = data.iloc[-1]
            signals = []
            
            # RSI signal
            if 'rsi' in latest and pd.notna(latest['rsi']):
                if latest['rsi'] < 30:
                    signals.append(1)  # Oversold - buy
                elif latest['rsi'] > 70:
                    signals.append(-1)  # Overbought - sell
                else:
                    signals.append(0)
            
            # MACD signal
            if all(col in latest for col in ['macd', 'macd_signal']) and all(pd.notna(latest[col]) for col in ['macd', 'macd_signal']):
                if latest['macd'] > latest['macd_signal']:
                    signals.append(1)  # Bullish
                else:
                    signals.append(-1)  # Bearish
            
            # Moving average signal
            if all(col in latest for col in ['close', 'sma_20']) and all(pd.notna(latest[col]) for col in ['close', 'sma_20']):
                if latest['close'] > latest['sma_20']:
                    signals.append(1)  # Above MA - bullish
                else:
                    signals.append(-1)  # Below MA - bearish
            
            # Bollinger Bands signal
            if all(col in latest for col in ['close', 'bb_lower', 'bb_upper']) and all(pd.notna(latest[col]) for col in ['close', 'bb_lower', 'bb_upper']):
                if latest['close'] < latest['bb_lower']:
                    signals.append(1)  # Oversold
                elif latest['close'] > latest['bb_upper']:
                    signals.append(-1)  # Overbought
            
            if not signals:
                return {'tech_signal': 0, 'tech_strength': 0.0}
            
            # Calculate consensus
            avg_signal = np.mean(signals)
            tech_signal = 1 if avg_signal > 0.2 else -1 if avg_signal < -0.2 else 0
            tech_strength = abs(avg_signal)
            
            return {'tech_signal': tech_signal, 'tech_strength': tech_strength}
            
        except Exception as e:
            logger.error(f"Error in technical confirmation: {e}")
            return {'tech_signal': 0, 'tech_strength': 0.0}
    
    def _combine_signals(self, ml_signal: Dict[str, Any], tech_signal: Dict[str, Any]) -> Dict[str, Any]:
        """Combine ML and technical signals"""
        try:
            ml_weight = 0.7
            tech_weight = 0.3
            
            # Get individual signals
            ml_sig = ml_signal.get('signal', 0)
            ml_conf = ml_signal.get('confidence', 0.0)
            
            tech_sig = tech_signal.get('tech_signal', 0)
            tech_strength = tech_signal.get('tech_strength', 0.0)
            
            # Combine signals
            combined_score = (ml_sig * ml_conf * ml_weight) + (tech_sig * tech_strength * tech_weight)
            
            # Generate final signal
            if combined_score > 0.3:
                final_signal = 1  # Buy
            elif combined_score < -0.3:
                final_signal = -1  # Sell
            else:
                final_signal = 0  # Hold
            
            # Calculate confidence
            final_confidence = min(abs(combined_score), 1.0)
            
            # Determine reason
            reasons = []
            if ml_sig != 0:
                reasons.append(f"ML: {ml_sig} ({ml_conf:.2f})")
            if tech_sig != 0:
                reasons.append(f"Tech: {tech_sig} ({tech_strength:.2f})")
            
            reason = ", ".join(reasons) if reasons else "Neutral signals"
            
            return {
                'signal': final_signal,
                'confidence': final_confidence,
                'reason': reason,
                'ml_signal': ml_signal,
                'tech_signal': tech_signal,
                'combined_score': combined_score
            }
            
        except Exception as e:
            logger.error(f"Error combining signals: {e}")
            return {'signal': 0, 'confidence': 0.0, 'reason': f'Error: {e}'}
    
    def _get_market_context(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get market context information"""
        try:
            if data.empty:
                return {}
            
            latest = data.iloc[-1]
            context = {}
            
            # Volatility
            if 'volatility' in latest and pd.notna(latest['volatility']):
                context['volatility'] = float(latest['volatility'])
                if latest['volatility'] > data['volatility'].quantile(0.8):
                    context['volatility_regime'] = 'high'
                elif latest['volatility'] < data['volatility'].quantile(0.2):
                    context['volatility_regime'] = 'low'
                else:
                    context['volatility_regime'] = 'normal'
            
            # Trend strength
            if all(col in latest for col in ['close', 'sma_20', 'sma_50']) and all(pd.notna(latest[col]) for col in ['close', 'sma_20', 'sma_50']):
                if latest['close'] > latest['sma_20'] > latest['sma_50']:
                    context['trend'] = 'strong_uptrend'
                elif latest['close'] < latest['sma_20'] < latest['sma_50']:
                    context['trend'] = 'strong_downtrend'
                elif latest['close'] > latest['sma_20']:
                    context['trend'] = 'uptrend'
                elif latest['close'] < latest['sma_20']:
                    context['trend'] = 'downtrend'
                else:
                    context['trend'] = 'sideways'
            
            # Volume analysis
            if all(col in latest for col in ['volume', 'volume_sma_20']) and all(pd.notna(latest[col]) for col in ['volume', 'volume_sma_20']):
                volume_ratio = latest['volume'] / latest['volume_sma_20'] if latest['volume_sma_20'] > 0 else 1
                context['volume_ratio'] = float(volume_ratio)
                if volume_ratio > 2.0:
                    context['volume_regime'] = 'high'
                elif volume_ratio < 0.5:
                    context['volume_regime'] = 'low'
                else:
                    context['volume_regime'] = 'normal'
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting market context: {e}")
            return {}
    
    def _store_signal_history(self, symbol: str, signal: Dict[str, Any]):
        """Store signal in history for performance tracking"""
        try:
            if symbol not in self.signal_history:
                self.signal_history[symbol] = []
            
            self.signal_history[symbol].append({
                'timestamp': datetime.now(),
                'signal': signal['signal'],
                'confidence': signal['confidence'],
                'reason': signal.get('reason', '')
            })
            
            # Keep only last 1000 signals
            self.signal_history[symbol] = self.signal_history[symbol][-1000:]
            
        except Exception as e:
            logger.error(f"Error storing signal history: {e}")
    
    def get_signal_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get signal statistics for a symbol"""
        try:
            if symbol not in self.signal_history:
                return {}
            
            history = self.signal_history[symbol]
            if not history:
                return {}
            
            recent_signals = history[-100:]  # Last 100 signals
            
            signal_counts = {'buy': 0, 'sell': 0, 'hold': 0}
            total_confidence = 0
            
            for entry in recent_signals:
                if entry['signal'] == 1:
                    signal_counts['buy'] += 1
                elif entry['signal'] == -1:
                    signal_counts['sell'] += 1
                else:
                    signal_counts['hold'] += 1
                
                total_confidence += entry['confidence']
            
            return {
                'total_signals': len(recent_signals),
                'signal_distribution': signal_counts,
                'average_confidence': total_confidence / len(recent_signals) if recent_signals else 0,
                'last_signal_time': recent_signals[-1]['timestamp'] if recent_signals else None
            }
            
        except Exception as e:
            logger.error(f"Error getting signal statistics: {e}")
            return {}
    
    def get_feature_importance(self, symbol: str) -> Dict[str, float]:
        """Get feature importance for a symbol"""
        return self.ml_generator.get_feature_importance(symbol)
    
    def get_model_performance(self, symbol: str) -> Dict[str, Any]:
        """Get model performance metrics"""
        return self.ml_generator.get_model_performance(symbol)
