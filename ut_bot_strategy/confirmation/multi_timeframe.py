"""
Multi-Timeframe Confirmation System for Enhanced Signal Validation

Analyzes multiple timeframes to confirm trading signals.
Higher timeframes carry more weight in the confirmation process.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd
import numpy as np

from ..data.binance_fetcher import BinanceDataFetcher
from ..indicators.ut_bot_alerts import UTBotAlerts
from ..indicators.stc_indicator import STCIndicator

logger = logging.getLogger(__name__)


@dataclass
class TimeframeAnalysis:
    """Analysis results for a single timeframe"""
    timeframe: str
    ut_bot_signal: str  # LONG/SHORT/NEUTRAL
    stc_color: str  # green/red/neutral
    stc_slope: str  # up/down/flat
    stc_value: float
    trend_direction: str  # bullish/bearish/neutral
    volatility: float
    weight: float


@dataclass
class MTFConfirmation:
    """Multi-timeframe confirmation result"""
    primary_timeframe: str
    signal_type: str  # LONG/SHORT/NEUTRAL
    alignment_score: float  # 0-1
    confirming_timeframes: List[str]
    conflicting_timeframes: List[str]
    higher_timeframe_bias: str  # bullish/bearish/neutral
    analyses: Dict[str, TimeframeAnalysis]
    recommendation: str  # STRONG_CONFIRM/CONFIRM/NEUTRAL/CONFLICT
    timestamp: datetime = field(default_factory=datetime.now)


class MultiTimeframeConfirmation:
    """
    Multi-Timeframe Confirmation System
    
    Analyzes multiple timeframes (1m, 5m, 15m, 1h, 4h) for the same symbol
    to validate trading signals. Higher timeframes have more weight.
    
    Weight distribution:
    - 4h: 0.30 (30%)
    - 1h: 0.25 (25%)
    - 15m: 0.20 (20%)
    - 5m: 0.15 (15%)
    - 1m: 0.10 (10%)
    """
    
    TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h']
    
    TIMEFRAME_WEIGHTS = {
        '1m': 0.10,
        '5m': 0.15,
        '15m': 0.20,
        '1h': 0.25,
        '4h': 0.30
    }
    
    HIGHER_TIMEFRAMES = ['1h', '4h']
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 primary_timeframe: str = '5m',
                 ut_key_value: float = 2.0,
                 ut_atr_period: int = 6,
                 ut_use_heikin_ashi: bool = True,
                 stc_length: int = 80,
                 stc_fast_length: int = 27,
                 stc_slow_length: int = 50,
                 candle_limit: int = 200):
        """
        Initialize Multi-Timeframe Confirmation System
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            primary_timeframe: Primary timeframe for signal generation
            ut_key_value: UT Bot sensitivity
            ut_atr_period: UT Bot ATR period
            ut_use_heikin_ashi: Use Heikin Ashi for UT Bot
            stc_length: STC stochastic length
            stc_fast_length: STC fast EMA period
            stc_slow_length: STC slow EMA period
            candle_limit: Number of candles to fetch per timeframe
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.primary_timeframe = primary_timeframe
        self.candle_limit = candle_limit
        
        self.ut_bot = UTBotAlerts(
            key_value=ut_key_value,
            atr_period=ut_atr_period,
            use_heikin_ashi=ut_use_heikin_ashi
        )
        
        self.stc = STCIndicator(
            length=stc_length,
            fast_length=stc_fast_length,
            slow_length=stc_slow_length
        )
        
        self._fetchers: Dict[str, BinanceDataFetcher] = {}
        self._analyses: Dict[str, TimeframeAnalysis] = {}
        self._last_analysis_time: Optional[datetime] = None
        
        logger.info(f"MultiTimeframeConfirmation initialized with primary_tf={primary_timeframe}")
    
    def _get_fetcher(self, symbol: str, timeframe: str) -> BinanceDataFetcher:
        """Get or create a data fetcher for the given symbol and timeframe"""
        key = f"{symbol}_{timeframe}"
        if key not in self._fetchers:
            self._fetchers[key] = BinanceDataFetcher(
                api_key=self.api_key,
                api_secret=self.api_secret,
                symbol=symbol,
                interval=timeframe
            )
        return self._fetchers[key]
    
    def _analyze_timeframe(self, symbol: str, timeframe: str) -> Optional[TimeframeAnalysis]:
        """
        Analyze a single timeframe
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe to analyze
            
        Returns:
            TimeframeAnalysis or None if analysis fails
        """
        try:
            fetcher = self._get_fetcher(symbol, timeframe)
            df = fetcher.fetch_historical_data(limit=self.candle_limit)
            
            if df is None or len(df) < 100:
                logger.warning(f"Insufficient data for {symbol} {timeframe}")
                return None
            
            ut_df = self.ut_bot.calculate(df)
            stc_df = self.stc.calculate(df)
            
            ut_signal = self._get_ut_bot_signal(ut_df)
            stc_state = self.stc.get_latest_state(stc_df)
            
            trend_direction = self._determine_trend(ut_df, stc_df)
            volatility = self._calculate_volatility(df)
            
            stc_slope = stc_state.get('slope', 'neutral')
            if stc_slope == 'neutral':
                stc_slope = 'flat'
            
            analysis = TimeframeAnalysis(
                timeframe=timeframe,
                ut_bot_signal=ut_signal,
                stc_color=stc_state.get('color', 'neutral'),
                stc_slope=stc_slope,
                stc_value=stc_state.get('stc', 50.0) or 50.0,
                trend_direction=trend_direction,
                volatility=volatility,
                weight=self.TIMEFRAME_WEIGHTS.get(timeframe, 0.1)
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol} {timeframe}: {e}")
            return None
    
    def _get_ut_bot_signal(self, df: pd.DataFrame) -> str:
        """Get UT Bot signal from calculated dataframe"""
        if len(df) < 2:
            return 'NEUTRAL'
        
        latest = df.iloc[-1]
        
        if latest.get('buy_signal', False):
            return 'LONG'
        elif latest.get('sell_signal', False):
            return 'SHORT'
        
        position = latest.get('position', 0)
        if position == 1:
            return 'LONG'
        elif position == -1:
            return 'SHORT'
        
        return 'NEUTRAL'
    
    def _determine_trend(self, ut_df: pd.DataFrame, stc_df: pd.DataFrame) -> str:
        """Determine overall trend direction from indicators"""
        if len(ut_df) < 2 or len(stc_df) < 2:
            return 'neutral'
        
        ut_latest = ut_df.iloc[-1]
        stc_latest = stc_df.iloc[-1]
        
        bullish_score = 0
        bearish_score = 0
        
        if ut_latest.get('position', 0) == 1:
            bullish_score += 1
        elif ut_latest.get('position', 0) == -1:
            bearish_score += 1
        
        if ut_latest.get('bar_color', '') == 'green':
            bullish_score += 1
        elif ut_latest.get('bar_color', '') == 'red':
            bearish_score += 1
        
        stc_color = stc_latest.get('stc_color', 'neutral')
        stc_slope = stc_latest.get('stc_slope', 'neutral')
        
        if stc_color == 'green':
            bullish_score += 1
        elif stc_color == 'red':
            bearish_score += 1
        
        if stc_slope == 'up':
            bullish_score += 1
        elif stc_slope == 'down':
            bearish_score += 1
        
        if bullish_score > bearish_score + 1:
            return 'bullish'
        elif bearish_score > bullish_score + 1:
            return 'bearish'
        else:
            return 'neutral'
    
    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculate recent volatility as percentage"""
        if len(df) < 20:
            return 0.0
        
        returns = df['close'].pct_change().dropna()
        if len(returns) < 10:
            return 0.0
        
        volatility = returns.tail(20).std() * 100
        return round(volatility, 4)
    
    def analyze(self, symbol: str) -> MTFConfirmation:
        """
        Perform full multi-timeframe analysis
        
        Args:
            symbol: Trading pair symbol (e.g., 'ETHUSDT')
            
        Returns:
            MTFConfirmation with complete analysis results
        """
        logger.info(f"Starting MTF analysis for {symbol}")
        
        self._analyses = {}
        
        for tf in self.TIMEFRAMES:
            analysis = self._analyze_timeframe(symbol, tf)
            if analysis:
                self._analyses[tf] = analysis
        
        if not self._analyses:
            logger.warning(f"No analyses completed for {symbol}")
            return MTFConfirmation(
                primary_timeframe=self.primary_timeframe,
                signal_type='NEUTRAL',
                alignment_score=0.0,
                confirming_timeframes=[],
                conflicting_timeframes=[],
                higher_timeframe_bias='neutral',
                analyses={},
                recommendation='NEUTRAL'
            )
        
        primary_analysis = self._analyses.get(self.primary_timeframe)
        signal_type = primary_analysis.ut_bot_signal if primary_analysis else 'NEUTRAL'
        
        alignment_score = self.get_alignment_score(signal_type)
        higher_tf_bias = self.get_higher_tf_bias()
        
        confirming = []
        conflicting = []
        
        for tf, analysis in self._analyses.items():
            if signal_type == 'LONG':
                if analysis.ut_bot_signal == 'LONG' or analysis.trend_direction == 'bullish':
                    confirming.append(tf)
                elif analysis.ut_bot_signal == 'SHORT' or analysis.trend_direction == 'bearish':
                    conflicting.append(tf)
            elif signal_type == 'SHORT':
                if analysis.ut_bot_signal == 'SHORT' or analysis.trend_direction == 'bearish':
                    confirming.append(tf)
                elif analysis.ut_bot_signal == 'LONG' or analysis.trend_direction == 'bullish':
                    conflicting.append(tf)
        
        recommendation = self._determine_recommendation(
            signal_type, alignment_score, higher_tf_bias, confirming, conflicting
        )
        
        self._last_analysis_time = datetime.now()
        
        mtf_confirmation = MTFConfirmation(
            primary_timeframe=self.primary_timeframe,
            signal_type=signal_type,
            alignment_score=alignment_score,
            confirming_timeframes=confirming,
            conflicting_timeframes=conflicting,
            higher_timeframe_bias=higher_tf_bias,
            analyses=self._analyses.copy(),
            recommendation=recommendation,
            timestamp=self._last_analysis_time
        )
        
        logger.info(f"MTF analysis complete: signal={signal_type}, alignment={alignment_score:.2f}, "
                   f"recommendation={recommendation}")
        
        return mtf_confirmation
    
    def get_alignment_score(self, signal_type: str) -> float:
        """
        Calculate how well timeframes align with the signal
        
        Args:
            signal_type: LONG, SHORT, or NEUTRAL
            
        Returns:
            Alignment score between 0 and 1
        """
        if not self._analyses or signal_type == 'NEUTRAL':
            return 0.0
        
        weighted_alignment = 0.0
        total_weight = 0.0
        
        for tf, analysis in self._analyses.items():
            weight = analysis.weight
            total_weight += weight
            
            tf_score = 0.0
            
            if signal_type == 'LONG':
                if analysis.ut_bot_signal == 'LONG':
                    tf_score += 0.4
                elif analysis.ut_bot_signal == 'NEUTRAL':
                    tf_score += 0.1
                
                if analysis.trend_direction == 'bullish':
                    tf_score += 0.3
                elif analysis.trend_direction == 'neutral':
                    tf_score += 0.1
                
                if analysis.stc_color == 'green':
                    tf_score += 0.2
                elif analysis.stc_color == 'neutral':
                    tf_score += 0.05
                
                if analysis.stc_slope == 'up':
                    tf_score += 0.1
                
            elif signal_type == 'SHORT':
                if analysis.ut_bot_signal == 'SHORT':
                    tf_score += 0.4
                elif analysis.ut_bot_signal == 'NEUTRAL':
                    tf_score += 0.1
                
                if analysis.trend_direction == 'bearish':
                    tf_score += 0.3
                elif analysis.trend_direction == 'neutral':
                    tf_score += 0.1
                
                if analysis.stc_color == 'red':
                    tf_score += 0.2
                elif analysis.stc_color == 'neutral':
                    tf_score += 0.05
                
                if analysis.stc_slope == 'down':
                    tf_score += 0.1
            
            weighted_alignment += tf_score * weight
        
        if total_weight > 0:
            alignment_score = weighted_alignment / total_weight
        else:
            alignment_score = 0.0
        
        return min(1.0, max(0.0, alignment_score))
    
    def get_higher_tf_bias(self) -> str:
        """
        Get direction bias from higher timeframes (1h and 4h)
        
        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        bullish_weight = 0.0
        bearish_weight = 0.0
        total_weight = 0.0
        
        for tf in self.HIGHER_TIMEFRAMES:
            analysis = self._analyses.get(tf)
            if not analysis:
                continue
            
            weight = analysis.weight
            total_weight += weight
            
            if analysis.ut_bot_signal == 'LONG':
                bullish_weight += weight * 0.5
            elif analysis.ut_bot_signal == 'SHORT':
                bearish_weight += weight * 0.5
            
            if analysis.trend_direction == 'bullish':
                bullish_weight += weight * 0.3
            elif analysis.trend_direction == 'bearish':
                bearish_weight += weight * 0.3
            
            if analysis.stc_color == 'green' and analysis.stc_slope == 'up':
                bullish_weight += weight * 0.2
            elif analysis.stc_color == 'red' and analysis.stc_slope == 'down':
                bearish_weight += weight * 0.2
        
        if total_weight == 0:
            return 'neutral'
        
        bullish_ratio = bullish_weight / total_weight
        bearish_ratio = bearish_weight / total_weight
        
        threshold = 0.3
        
        if bullish_ratio > bearish_ratio + threshold:
            return 'bullish'
        elif bearish_ratio > bullish_ratio + threshold:
            return 'bearish'
        else:
            return 'neutral'
    
    def should_trade(self, min_alignment: float = 0.5) -> bool:
        """
        Determine if a trade should be taken based on MTF analysis
        
        Args:
            min_alignment: Minimum alignment score required (default 0.5)
            
        Returns:
            True if trade is recommended
        """
        if not self._analyses:
            return False
        
        primary = self._analyses.get(self.primary_timeframe)
        if not primary or primary.ut_bot_signal == 'NEUTRAL':
            return False
        
        alignment = self.get_alignment_score(primary.ut_bot_signal)
        if alignment < min_alignment:
            return False
        
        higher_bias = self.get_higher_tf_bias()
        
        if primary.ut_bot_signal == 'LONG' and higher_bias == 'bearish':
            return False
        if primary.ut_bot_signal == 'SHORT' and higher_bias == 'bullish':
            return False
        
        return True
    
    def _determine_recommendation(self, signal_type: str, alignment_score: float,
                                   higher_tf_bias: str, confirming: List[str],
                                   conflicting: List[str]) -> str:
        """
        Determine trading recommendation based on analysis
        
        Returns:
            STRONG_CONFIRM, CONFIRM, NEUTRAL, or CONFLICT
        """
        if signal_type == 'NEUTRAL':
            return 'NEUTRAL'
        
        if len(conflicting) >= 3:
            return 'CONFLICT'
        
        bias_aligned = (
            (signal_type == 'LONG' and higher_tf_bias == 'bullish') or
            (signal_type == 'SHORT' and higher_tf_bias == 'bearish')
        )
        
        bias_conflicting = (
            (signal_type == 'LONG' and higher_tf_bias == 'bearish') or
            (signal_type == 'SHORT' and higher_tf_bias == 'bullish')
        )
        
        if bias_conflicting:
            return 'CONFLICT'
        
        if alignment_score >= 0.7 and bias_aligned and len(confirming) >= 3:
            return 'STRONG_CONFIRM'
        
        if alignment_score >= 0.5 and len(confirming) >= 2:
            return 'CONFIRM'
        
        if alignment_score >= 0.3:
            return 'NEUTRAL'
        
        return 'CONFLICT'
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get a summary of the latest analysis"""
        if not self._analyses:
            return {'status': 'no_analysis'}
        
        summary = {
            'timestamp': self._last_analysis_time.isoformat() if self._last_analysis_time else None,
            'timeframes_analyzed': list(self._analyses.keys()),
            'analyses': {}
        }
        
        for tf, analysis in self._analyses.items():
            summary['analyses'][tf] = {
                'ut_bot_signal': analysis.ut_bot_signal,
                'stc_color': analysis.stc_color,
                'stc_slope': analysis.stc_slope,
                'stc_value': round(analysis.stc_value, 2),
                'trend': analysis.trend_direction,
                'volatility': analysis.volatility,
                'weight': analysis.weight
            }
        
        return summary
