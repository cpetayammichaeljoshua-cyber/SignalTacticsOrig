"""
Signal Engine for UT Bot + STC Strategy

Combines UT Bot Alerts and STC indicators to generate trading signals.
Implements the complete strategy rules:
- LONG: UT Bot buy signal + STC green + STC pointing up + STC < 75
- SHORT: UT Bot sell signal + STC red + STC pointing down + STC > 25
- Stop loss at recent swing high/low
- Take profit at 1.5x risk

Enhanced with Order Flow Analysis:
- CVD trend integration for confirmation
- Manipulation detection filtering
- Institutional activity scoring
- Enhanced confidence calculation

Multi-Source Market Intelligence:
- Fear & Greed Index integration (extreme fear favors LONG, extreme greed favors SHORT)
- News sentiment analysis
- Market breadth (how many top coins moving in same direction)
- Multi-timeframe confirmation
- Whale activity tracking (large $100K+ trades)
- Economic calendar awareness (FOMC, CPI, NFP impact)

Confidence Weight Distribution:
- Base indicator confidence: 30%
- Order flow alignment: 15%
- Multi-timeframe confirmation: 12%
- Derivatives alignment: 10%
- Whale activity: 8%
- Fear/Greed alignment: 8%
- News sentiment: 8%
- Economic calendar: 5%
- Market breadth: 4%
"""

import logging
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
import pandas as pd
import numpy as np

from ..indicators.ut_bot_alerts import UTBotAlerts
from ..indicators.stc_indicator import STCIndicator
from ..external_data.derivatives_client import BinanceDerivativesClient, DerivativesData

logger = logging.getLogger(__name__)


class SignalEngine:
    """
    Combined Signal Engine for UT Bot + STC Strategy
    
    Generates confirmed trading signals by combining:
    1. UT Bot Alerts for entry signals
    2. STC indicator for confirmation
    3. Swing detection for stop loss
    4. Risk:Reward calculation for take profit
    5. Order flow analysis for enhanced confirmation (optional)
    """
    
    MANIPULATION_THRESHOLD = 0.7
    ORDER_FLOW_DIVERGENCE_THRESHOLD = 0.3
    
    def __init__(self, 
                 ut_key_value: float = 1.0,
                 ut_atr_period: int = 10,
                 ut_use_heikin_ashi: bool = False,
                 stc_length: int = 80,
                 stc_fast_length: int = 27,
                 stc_slow_length: int = 50,
                 swing_lookback: int = 5,
                 risk_reward_ratio: float = 1.5,
                 min_risk_percent: float = 0.5,
                 max_risk_percent: float = 3.0,
                 order_flow_enabled: bool = True,
                 manipulation_filter_enabled: bool = True,
                 order_flow_weight: float = 0.3):
        """
        Initialize Signal Engine
        
        Args:
            ut_key_value: UT Bot sensitivity (default 1.0)
            ut_atr_period: UT Bot ATR period (default 10)
            ut_use_heikin_ashi: Use Heikin Ashi for UT Bot (default False)
            stc_length: STC stochastic length (default 80)
            stc_fast_length: STC fast EMA period (default 27)
            stc_slow_length: STC slow EMA period (default 50)
            swing_lookback: Bars to look back for swing detection (default 5)
            risk_reward_ratio: Target R:R ratio (default 1.5)
            min_risk_percent: Minimum risk percentage (default 0.5%)
            max_risk_percent: Maximum risk percentage (default 3.0%)
            order_flow_enabled: Enable order flow analysis integration (default True)
            manipulation_filter_enabled: Filter signals on manipulation detection (default True)
            order_flow_weight: Weight for order flow in confidence calculation (default 0.3)
        """
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
        
        self.swing_lookback = swing_lookback
        self.risk_reward_ratio = risk_reward_ratio
        self.min_risk_percent = min_risk_percent
        self.max_risk_percent = max_risk_percent
        
        self.order_flow_enabled = order_flow_enabled
        self.manipulation_filter_enabled = manipulation_filter_enabled
        self.order_flow_weight = max(0.0, min(1.0, order_flow_weight))
        
        self._order_flow_metrics_service: Optional[Any] = None
        
        self._fear_greed_client: Optional[Any] = None
        self._news_client: Optional[Any] = None
        self._market_aggregator: Optional[Any] = None
        self._derivatives_client: Optional[BinanceDerivativesClient] = None
        self._whale_tracker: Optional[Any] = None
        self._economic_calendar: Optional[Any] = None
        
        self._last_signal_time: Optional[datetime] = None
        self._last_signal_type: Optional[str] = None
        self._signal_history: List[Dict] = []
        
        logger.info(f"SignalEngine initialized with order_flow_enabled={order_flow_enabled}, "
                   f"manipulation_filter_enabled={manipulation_filter_enabled}, "
                   f"order_flow_weight={order_flow_weight}")
    
    def set_order_flow_metrics(self, metrics_service: Any) -> None:
        """
        Connect to OrderFlowMetricsService for order flow analysis
        
        Args:
            metrics_service: OrderFlowMetricsService instance
        """
        self._order_flow_metrics_service = metrics_service
        logger.info("OrderFlowMetricsService connected to SignalEngine")
    
    def set_market_intelligence(self, 
                                fear_greed_client: Optional[Any] = None,
                                news_client: Optional[Any] = None,
                                market_aggregator: Optional[Any] = None) -> None:
        """
        Connect external data sources for multi-source market intelligence
        
        Args:
            fear_greed_client: FearGreedClient instance for Fear & Greed index
            news_client: NewsSentimentClient instance for news sentiment
            market_aggregator: MarketDataAggregator instance for market breadth
        """
        if fear_greed_client is not None:
            self._fear_greed_client = fear_greed_client
            logger.info("FearGreedClient connected to SignalEngine")
        
        if news_client is not None:
            self._news_client = news_client
            logger.info("NewsSentimentClient connected to SignalEngine")
        
        if market_aggregator is not None:
            self._market_aggregator = market_aggregator
            logger.info("MarketDataAggregator connected to SignalEngine")
    
    def set_derivatives_client(self, client: BinanceDerivativesClient) -> None:
        """
        Connect derivatives client for funding rate, OI, and L/S ratio data
        
        Args:
            client: BinanceDerivativesClient instance
        """
        self._derivatives_client = client
        logger.info("BinanceDerivativesClient connected to SignalEngine")
    
    def set_whale_tracker(self, whale_tracker: Any) -> None:
        """
        Connect whale tracker for large trade monitoring
        
        Args:
            whale_tracker: WhaleTracker instance for tracking $100K+ trades
        """
        self._whale_tracker = whale_tracker
        logger.info("WhaleTracker connected to SignalEngine")
    
    def set_economic_calendar(self, calendar_client: Any) -> None:
        """
        Connect economic calendar client for macro event awareness
        
        Args:
            calendar_client: EconomicCalendarClient instance for event tracking
        """
        self._economic_calendar = calendar_client
        logger.info("EconomicCalendarClient connected to SignalEngine")
    
    async def _get_whale_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get current whale activity metrics from connected tracker
        
        Returns:
            Dictionary with whale metrics or None if unavailable
        """
        if not self._whale_tracker:
            return None
        
        try:
            metrics = self._whale_tracker.get_current_metrics()
            if metrics is None:
                return None
            
            return {
                'whale_bias': getattr(metrics, 'whale_bias', 0.0),
                'smart_money_direction': getattr(metrics, 'smart_money_direction', 'NEUTRAL'),
                'whale_buy_volume': getattr(metrics, 'whale_buy_volume', 0.0),
                'whale_sell_volume': getattr(metrics, 'whale_sell_volume', 0.0),
                'net_whale_flow': getattr(metrics, 'net_whale_flow', 0.0),
                'accumulation_score': getattr(metrics, 'accumulation_score', 0.0),
                'distribution_score': getattr(metrics, 'distribution_score', 0.0),
                'whale_buy_count': getattr(metrics, 'whale_buy_count', 0),
                'whale_sell_count': getattr(metrics, 'whale_sell_count', 0)
            }
        except Exception as e:
            logger.warning(f"Error getting whale metrics: {e}")
            return None
    
    async def _get_economic_calendar_data(self) -> Optional[Dict[str, Any]]:
        """
        Get upcoming economic events from connected calendar client
        
        Returns:
            Dictionary with economic calendar data or None if unavailable
        """
        if not self._economic_calendar:
            return None
        
        try:
            summary = await self._economic_calendar.get_summary()
            if summary is None:
                return None
            
            next_event_info = None
            if summary.next_major_event:
                next_event_info = {
                    'name': summary.next_major_event.name,
                    'time': summary.next_major_event.time.isoformat() if hasattr(summary.next_major_event.time, 'isoformat') else str(summary.next_major_event.time),
                    'impact': summary.next_major_event.impact.value if hasattr(summary.next_major_event.impact, 'value') else str(summary.next_major_event.impact),
                    'minutes_until': summary.next_major_event.minutes_until
                }
            
            return {
                'has_imminent_event': summary.has_imminent_event,
                'should_avoid_trading': summary.should_avoid_trading,
                'total_events': summary.total_events,
                'high_impact_count': summary.high_impact_events,
                'next_major_event': next_event_info,
                'trading_recommendation': summary.trading_recommendation
            }
        except Exception as e:
            logger.warning(f"Error getting economic calendar data: {e}")
            return None
    
    def _calculate_whale_alignment(self, whale_metrics: Dict[str, Any], signal_type: str) -> float:
        """
        Calculate whale activity alignment score for signal
        
        Args:
            whale_metrics: Dict with whale activity data
            signal_type: 'LONG' or 'SHORT'
            
        Returns:
            Alignment score (0 to 1)
        """
        whale_bias = whale_metrics.get('whale_bias', 0.0)
        smart_money_direction = whale_metrics.get('smart_money_direction', 'NEUTRAL')
        
        if signal_type == 'LONG':
            base_alignment = (whale_bias + 1) / 2
            if smart_money_direction == 'BULLISH':
                base_alignment = min(1.0, base_alignment + 0.1)
            elif smart_money_direction == 'BEARISH':
                base_alignment = max(0.0, base_alignment - 0.1)
        elif signal_type == 'SHORT':
            base_alignment = (-whale_bias + 1) / 2
            if smart_money_direction == 'BEARISH':
                base_alignment = min(1.0, base_alignment + 0.1)
            elif smart_money_direction == 'BULLISH':
                base_alignment = max(0.0, base_alignment - 0.1)
        else:
            base_alignment = 0.5
        
        return max(0.0, min(1.0, base_alignment))
    
    def _calculate_economic_calendar_factor(self, calendar_data: Dict[str, Any]) -> float:
        """
        Calculate economic calendar factor for confidence adjustment
        
        High-impact events within 30 minutes reduce confidence.
        No imminent events return neutral score.
        
        Args:
            calendar_data: Dict with economic calendar data
            
        Returns:
            Factor score (0 to 1), where 1.0 = no impact, lower = reduce confidence
        """
        if calendar_data.get('should_avoid_trading', False):
            return 0.3
        
        if calendar_data.get('has_imminent_event', False):
            next_event = calendar_data.get('next_major_event')
            if next_event:
                minutes_until = next_event.get('minutes_until', 60)
                if minutes_until <= 15:
                    return 0.4
                elif minutes_until <= 30:
                    return 0.6
                elif minutes_until <= 60:
                    return 0.8
        
        return 1.0
    
    async def _get_derivatives_data(self, symbol: str = "ETHUSDT") -> Optional[DerivativesData]:
        """
        Fetch derivatives data from connected client
        
        Args:
            symbol: Trading pair symbol (default ETHUSDT)
            
        Returns:
            DerivativesData or None if unavailable
        """
        if not self._derivatives_client:
            return None
        
        try:
            derivatives_data = await self._derivatives_client.get_derivatives_intelligence(symbol)
            return derivatives_data
        except Exception as e:
            logger.warning(f"Error fetching derivatives data: {e}")
            return None
    
    def _calculate_derivatives_alignment(self, derivatives_data: DerivativesData, signal_type: str) -> float:
        """
        Calculate derivatives alignment score for signal
        
        Args:
            derivatives_data: DerivativesData with market metrics
            signal_type: 'LONG' or 'SHORT'
            
        Returns:
            Alignment score (0 to 1)
        """
        derivatives_score = derivatives_data.derivatives_score
        
        if signal_type == 'LONG':
            return (derivatives_score + 1) / 2
        elif signal_type == 'SHORT':
            return (-derivatives_score + 1) / 2
        
        return 0.5
    
    def _get_order_flow_data(self) -> Optional[Dict[str, Any]]:
        """
        Get current order flow metrics from connected service
        
        Returns:
            Dictionary with order flow data or None if not available
        """
        if not self._order_flow_metrics_service:
            return None
        
        try:
            metrics = self._order_flow_metrics_service.get_complete_metrics()
            
            imbalance_dir = "neutral"
            if hasattr(metrics, 'imbalance') and metrics.imbalance:
                if metrics.imbalance.direction == "bullish":
                    imbalance_dir = "bid"
                elif metrics.imbalance.direction == "bearish":
                    imbalance_dir = "ask"
            
            absorption_detected = False
            if hasattr(metrics, 'absorption') and metrics.absorption:
                absorption_detected = metrics.absorption.active_zones > 0
            
            return {
                'cvd_trend': metrics.cvd.cvd_trend if hasattr(metrics, 'cvd') else 'neutral',
                'cvd_value': metrics.cvd.current_cvd if hasattr(metrics, 'cvd') else 0.0,
                'cvd_trend_strength': metrics.cvd.trend_strength if hasattr(metrics, 'cvd') else 0.0,
                'delta_extreme': metrics.delta_extremes.has_extreme if hasattr(metrics, 'delta_extremes') else False,
                'delta_direction': metrics.delta_extremes.direction if hasattr(metrics, 'delta_extremes') else 'none',
                'imbalance_direction': imbalance_dir,
                'imbalance_ratio': metrics.imbalance.imbalance_ratio if hasattr(metrics, 'imbalance') else 0.0,
                'absorption_detected': absorption_detected,
                'absorption_zones': metrics.absorption.active_zones if hasattr(metrics, 'absorption') else 0,
                'manipulation_score': metrics.manipulation.overall_score if hasattr(metrics, 'manipulation') else 0.0,
                'manipulation_notes': metrics.manipulation.notes if hasattr(metrics, 'manipulation') else [],
                'manipulation_recommendation': metrics.manipulation.recommendation if hasattr(metrics, 'manipulation') else 'trade',
                'tape_signal': metrics.tape_signal if hasattr(metrics, 'tape_signal') else 0.0,
                'order_flow_bias': metrics.order_flow_bias if hasattr(metrics, 'order_flow_bias') else 0.0,
                'institutional_activity': metrics.institutional_activity.activity_score if hasattr(metrics, 'institutional_activity') else 0.0,
                'smart_money_direction': metrics.smart_money.direction if hasattr(metrics, 'smart_money') else 'neutral',
                'smart_money_score': metrics.smart_money.score if hasattr(metrics, 'smart_money') else 0.0
            }
        except Exception as e:
            logger.warning(f"Error getting order flow data: {e}")
            return None
    
    def _check_manipulation_filter(self, signal: Dict, order_flow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if signal should be filtered due to manipulation detection
        
        Args:
            signal: Signal dictionary
            order_flow_data: Order flow data dictionary
            
        Returns:
            Tuple of (should_filter: bool, reason: str)
        """
        if not self.manipulation_filter_enabled:
            return False, ""
        
        manipulation_score = order_flow_data.get('manipulation_score', 0.0)
        manipulation_recommendation = order_flow_data.get('manipulation_recommendation', 'trade')
        
        if manipulation_score >= self.MANIPULATION_THRESHOLD:
            notes = order_flow_data.get('manipulation_notes', [])
            reason = f"Manipulation detected (score: {manipulation_score:.2f})"
            if notes:
                reason += f" - {', '.join(notes[:2])}"
            return True, reason
        
        if manipulation_recommendation == 'avoid':
            return True, f"Order flow recommendation: avoid trading"
        
        return False, ""
    
    def _check_order_flow_divergence(self, signal: Dict, order_flow_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if order flow diverges from signal direction
        
        Args:
            signal: Signal dictionary
            order_flow_data: Order flow data dictionary
            
        Returns:
            Tuple of (should_filter: bool, reason: str)
        """
        signal_type = signal.get('type', '')
        order_flow_bias = order_flow_data.get('order_flow_bias', 0.0)
        cvd_trend = order_flow_data.get('cvd_trend', 'neutral')
        
        if signal_type == 'LONG':
            if order_flow_bias < -self.ORDER_FLOW_DIVERGENCE_THRESHOLD:
                return True, f"Order flow divergence: LONG signal but bearish order flow (bias: {order_flow_bias:.2f})"
            
            if cvd_trend == 'bearish' and abs(order_flow_bias) > 0.2:
                return True, f"Order flow divergence: LONG signal but bearish CVD trend"
        
        elif signal_type == 'SHORT':
            if order_flow_bias > self.ORDER_FLOW_DIVERGENCE_THRESHOLD:
                return True, f"Order flow divergence: SHORT signal but bullish order flow (bias: {order_flow_bias:.2f})"
            
            if cvd_trend == 'bullish' and abs(order_flow_bias) > 0.2:
                return True, f"Order flow divergence: SHORT signal but bullish CVD trend"
        
        return False, ""
    
    def _calculate_enhanced_confidence(self, base_confidence: float, order_flow_bias: float, 
                                        signal_type: str) -> float:
        """
        Calculate enhanced confidence by combining base confidence with order flow alignment
        
        Args:
            base_confidence: Base signal confidence (0 to 1)
            order_flow_bias: Order flow bias (-1 to 1)
            signal_type: 'LONG' or 'SHORT'
            
        Returns:
            Enhanced confidence (0 to 1)
        """
        base_weight = 1.0 - self.order_flow_weight
        of_weight = self.order_flow_weight
        
        if signal_type == 'LONG':
            of_alignment = (order_flow_bias + 1) / 2
        elif signal_type == 'SHORT':
            of_alignment = (-order_flow_bias + 1) / 2
        else:
            of_alignment = 0.5
        
        enhanced = (base_confidence * base_weight) + (of_alignment * of_weight)
        
        return max(0.0, min(1.0, enhanced))
    
    def _calculate_fear_greed_alignment(self, fear_greed_value: float, signal_type: str) -> float:
        """
        Calculate Fear & Greed alignment score
        
        Extreme Fear (0-30) favors LONG positions
        Extreme Greed (70-100) favors SHORT positions
        
        Args:
            fear_greed_value: Fear & Greed index value (0-100)
            signal_type: 'LONG' or 'SHORT'
            
        Returns:
            Alignment score (0 to 1)
        """
        if signal_type == 'LONG':
            if fear_greed_value <= 30:
                return 1.0 - (fear_greed_value / 30) * 0.3
            elif fear_greed_value <= 50:
                return 0.7 - ((fear_greed_value - 30) / 20) * 0.2
            elif fear_greed_value <= 70:
                return 0.5 - ((fear_greed_value - 50) / 20) * 0.2
            else:
                return 0.3 - ((fear_greed_value - 70) / 30) * 0.3
        elif signal_type == 'SHORT':
            if fear_greed_value >= 70:
                return 0.7 + ((fear_greed_value - 70) / 30) * 0.3
            elif fear_greed_value >= 50:
                return 0.5 + ((fear_greed_value - 50) / 20) * 0.2
            elif fear_greed_value >= 30:
                return 0.3 + ((fear_greed_value - 30) / 20) * 0.2
            else:
                return 0.3 - (fear_greed_value / 30) * 0.3
        
        return 0.5
    
    def _calculate_news_sentiment_alignment(self, news_sentiment: float, signal_type: str) -> float:
        """
        Calculate news sentiment alignment score
        
        Args:
            news_sentiment: News sentiment score (-1 to 1, where positive = bullish)
            signal_type: 'LONG' or 'SHORT'
            
        Returns:
            Alignment score (0 to 1)
        """
        if signal_type == 'LONG':
            return (news_sentiment + 1) / 2
        elif signal_type == 'SHORT':
            return (-news_sentiment + 1) / 2
        
        return 0.5
    
    def _calculate_market_breadth_alignment(self, market_breadth: float, signal_type: str) -> float:
        """
        Calculate market breadth alignment score
        
        Args:
            market_breadth: Percentage of top coins moving in bullish direction (0-100)
            signal_type: 'LONG' or 'SHORT'
            
        Returns:
            Alignment score (0 to 1)
        """
        normalized_breadth = market_breadth / 100.0
        
        if signal_type == 'LONG':
            return normalized_breadth
        elif signal_type == 'SHORT':
            return 1.0 - normalized_breadth
        
        return 0.5
    
    def _calculate_multi_source_confidence(self,
                                           base_confidence: float,
                                           signal_type: str,
                                           order_flow_bias: float = 0.0,
                                           market_context: Optional[Dict] = None,
                                           mtf_confirmation: Optional[Dict] = None,
                                           derivatives_data: Optional[DerivativesData] = None,
                                           whale_metrics: Optional[Dict] = None,
                                           calendar_data: Optional[Dict] = None) -> Tuple[float, Dict]:
        """
        Calculate multi-source confidence by weighing all market intelligence factors
        
        Weight distribution:
        - Base indicator confidence: 30%
        - Order flow alignment: 15%
        - Multi-timeframe confirmation: 12%
        - Derivatives alignment: 10%
        - Whale activity: 8%
        - Fear/Greed alignment: 8%
        - News sentiment: 8%
        - Economic calendar: 5%
        - Market breadth: 4%
        
        Args:
            base_confidence: Base signal confidence (0 to 1)
            signal_type: 'LONG' or 'SHORT'
            order_flow_bias: Order flow bias (-1 to 1)
            market_context: Dict containing fear_greed, news_sentiment, market_breadth
            mtf_confirmation: Dict containing alignment_score from multi-timeframe analysis
            derivatives_data: DerivativesData with funding, OI, L/S data
            whale_metrics: Dict containing whale activity data
            calendar_data: Dict containing economic calendar data
            
        Returns:
            Tuple of (multi_source_confidence, component_scores_dict)
        """
        WEIGHT_BASE = 0.30
        WEIGHT_ORDER_FLOW = 0.15
        WEIGHT_MTF = 0.12
        WEIGHT_DERIVATIVES = 0.10
        WEIGHT_WHALE = 0.08
        WEIGHT_FEAR_GREED = 0.08
        WEIGHT_NEWS = 0.08
        WEIGHT_CALENDAR = 0.05
        WEIGHT_BREADTH = 0.04
        
        component_scores = {
            'base_indicator': base_confidence,
            'order_flow_alignment': 0.5,
            'fear_greed_alignment': 0.5,
            'news_sentiment_alignment': 0.5,
            'mtf_alignment': 0.5,
            'market_breadth_alignment': 0.5,
            'derivatives_alignment': 0.5,
            'whale_alignment': 0.5,
            'economic_calendar_factor': 1.0
        }
        
        if signal_type == 'LONG':
            of_alignment = (order_flow_bias + 1) / 2
        elif signal_type == 'SHORT':
            of_alignment = (-order_flow_bias + 1) / 2
        else:
            of_alignment = 0.5
        component_scores['order_flow_alignment'] = of_alignment
        
        if market_context:
            fear_greed_value = market_context.get('fear_greed_value', 50.0)
            component_scores['fear_greed_alignment'] = self._calculate_fear_greed_alignment(
                fear_greed_value, signal_type
            )
            
            news_sentiment = market_context.get('news_sentiment_score', 0.0)
            component_scores['news_sentiment_alignment'] = self._calculate_news_sentiment_alignment(
                news_sentiment, signal_type
            )
            
            market_breadth = market_context.get('market_breadth_score', 50.0)
            component_scores['market_breadth_alignment'] = self._calculate_market_breadth_alignment(
                market_breadth, signal_type
            )
        
        if mtf_confirmation:
            mtf_score = mtf_confirmation.get('alignment_score', 0.5)
            component_scores['mtf_alignment'] = mtf_score
        
        if derivatives_data:
            component_scores['derivatives_alignment'] = self._calculate_derivatives_alignment(
                derivatives_data, signal_type
            )
        
        if whale_metrics:
            component_scores['whale_alignment'] = self._calculate_whale_alignment(
                whale_metrics, signal_type
            )
        
        if calendar_data:
            component_scores['economic_calendar_factor'] = self._calculate_economic_calendar_factor(
                calendar_data
            )
        
        multi_source_confidence = (
            component_scores['base_indicator'] * WEIGHT_BASE +
            component_scores['order_flow_alignment'] * WEIGHT_ORDER_FLOW +
            component_scores['mtf_alignment'] * WEIGHT_MTF +
            component_scores['derivatives_alignment'] * WEIGHT_DERIVATIVES +
            component_scores['whale_alignment'] * WEIGHT_WHALE +
            component_scores['fear_greed_alignment'] * WEIGHT_FEAR_GREED +
            component_scores['news_sentiment_alignment'] * WEIGHT_NEWS +
            component_scores['economic_calendar_factor'] * WEIGHT_CALENDAR +
            component_scores['market_breadth_alignment'] * WEIGHT_BREADTH
        )
        
        multi_source_confidence = max(0.0, min(1.0, multi_source_confidence))
        
        return multi_source_confidence, component_scores
    
    def _build_market_intelligence(self,
                                   market_context: Optional[Dict],
                                   mtf_confirmation: Optional[Dict],
                                   component_scores: Dict,
                                   overall_score: float,
                                   derivatives_data: Optional[DerivativesData] = None,
                                   whale_metrics: Optional[Dict] = None,
                                   calendar_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Build the market_intelligence field for signal output
        
        Args:
            market_context: Dict containing fear_greed, news_sentiment, market_breadth
            mtf_confirmation: Dict containing alignment_score from multi-timeframe analysis
            component_scores: Component scores from multi-source confidence calculation
            overall_score: Overall multi-source confidence score
            derivatives_data: DerivativesData with funding, OI, L/S data
            whale_metrics: Dict containing whale activity data
            calendar_data: Dict containing economic calendar data
            
        Returns:
            market_intelligence dictionary
        """
        intelligence = {
            'fear_greed_value': 50.0,
            'fear_greed_classification': 'Neutral',
            'news_sentiment_score': 0.0,
            'news_bias': 'neutral',
            'market_breadth_score': 50.0,
            'mtf_alignment_score': 0.5,
            'overall_intelligence_score': overall_score,
            'all_source_confidence': overall_score,
            'component_scores': component_scores,
            'derivatives_score': 0.0,
            'funding_rate': 0.0,
            'funding_trend': 'stable',
            'long_short_ratio': 1.0,
            'open_interest_change': 0.0,
            'market_sentiment': 'neutral',
            'whale_bias': 0.0,
            'whale_direction': 'NEUTRAL',
            'economic_event_warning': False,
            'next_major_event': None
        }
        
        if market_context:
            fear_greed_value = market_context.get('fear_greed_value', 50.0)
            intelligence['fear_greed_value'] = fear_greed_value
            
            if fear_greed_value <= 20:
                intelligence['fear_greed_classification'] = 'Extreme Fear'
            elif fear_greed_value <= 40:
                intelligence['fear_greed_classification'] = 'Fear'
            elif fear_greed_value <= 60:
                intelligence['fear_greed_classification'] = 'Neutral'
            elif fear_greed_value <= 80:
                intelligence['fear_greed_classification'] = 'Greed'
            else:
                intelligence['fear_greed_classification'] = 'Extreme Greed'
            
            news_sentiment = market_context.get('news_sentiment_score', 0.0)
            intelligence['news_sentiment_score'] = news_sentiment
            
            if news_sentiment >= 0.3:
                intelligence['news_bias'] = 'bullish'
            elif news_sentiment <= -0.3:
                intelligence['news_bias'] = 'bearish'
            else:
                intelligence['news_bias'] = 'neutral'
            
            intelligence['market_breadth_score'] = market_context.get('market_breadth_score', 50.0)
        
        if mtf_confirmation:
            intelligence['mtf_alignment_score'] = mtf_confirmation.get('alignment_score', 0.5)
            intelligence['mtf_recommendation'] = mtf_confirmation.get('recommendation', 'NEUTRAL')
            intelligence['higher_tf_bias'] = mtf_confirmation.get('higher_timeframe_bias', 'neutral')
            intelligence['confirming_timeframes'] = mtf_confirmation.get('confirming_timeframes', [])
            intelligence['conflicting_timeframes'] = mtf_confirmation.get('conflicting_timeframes', [])
        
        if derivatives_data:
            intelligence['derivatives_score'] = derivatives_data.derivatives_score
            intelligence['funding_rate'] = derivatives_data.funding_rate
            intelligence['funding_trend'] = derivatives_data.funding_rate_trend
            intelligence['long_short_ratio'] = derivatives_data.long_short_ratio
            intelligence['open_interest_change'] = derivatives_data.oi_change_24h
            intelligence['market_sentiment'] = derivatives_data.market_sentiment
        
        if whale_metrics:
            intelligence['whale_bias'] = whale_metrics.get('whale_bias', 0.0)
            intelligence['whale_direction'] = whale_metrics.get('smart_money_direction', 'NEUTRAL')
            intelligence['whale_buy_volume'] = whale_metrics.get('whale_buy_volume', 0.0)
            intelligence['whale_sell_volume'] = whale_metrics.get('whale_sell_volume', 0.0)
            intelligence['net_whale_flow'] = whale_metrics.get('net_whale_flow', 0.0)
            intelligence['accumulation_score'] = whale_metrics.get('accumulation_score', 0.0)
            intelligence['distribution_score'] = whale_metrics.get('distribution_score', 0.0)
        
        if calendar_data:
            has_imminent = calendar_data.get('has_imminent_event', False)
            should_avoid = calendar_data.get('should_avoid_trading', False)
            intelligence['economic_event_warning'] = has_imminent or should_avoid
            intelligence['next_major_event'] = calendar_data.get('next_major_event')
            intelligence['trading_recommendation'] = calendar_data.get('trading_recommendation', 'TRADE')
            intelligence['high_impact_event_count'] = calendar_data.get('high_impact_count', 0)
        
        return intelligence
    
    def _enhance_with_order_flow(self, signal: Dict) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Enhance signal with order flow data
        
        Args:
            signal: Base signal dictionary
            
        Returns:
            Tuple of (enhanced_signal or None if filtered, rejection_reason or None)
        """
        if not self.order_flow_enabled or not self._order_flow_metrics_service:
            return signal, None
        
        order_flow_data = self._get_order_flow_data()
        if not order_flow_data:
            signal['order_flow'] = {
                'cvd_trend': 'neutral',
                'delta_extreme': False,
                'imbalance_direction': 'neutral',
                'absorption_detected': False,
                'manipulation_score': 0.0,
                'tape_signal': 0.0,
                'enhanced_confidence': signal.get('confidence', 0.7)
            }
            signal['order_flow_bias'] = 0.0
            signal['manipulation_score'] = 0.0
            signal['institutional_activity'] = 0.0
            signal['tape_signal'] = 0.0
            signal['enhanced_confidence'] = signal.get('confidence', 0.7)
            return signal, None
        
        should_filter, filter_reason = self._check_manipulation_filter(signal, order_flow_data)
        if should_filter:
            logger.info(f"{signal['type']} signal rejected: {filter_reason}")
            return None, filter_reason
        
        should_filter, divergence_reason = self._check_order_flow_divergence(signal, order_flow_data)
        if should_filter:
            logger.info(f"{signal['type']} signal rejected: {divergence_reason}")
            return None, divergence_reason
        
        base_confidence = signal.get('confidence', 0.7)
        order_flow_bias = order_flow_data.get('order_flow_bias', 0.0)
        signal_type = signal.get('type', '')
        
        enhanced_confidence = self._calculate_enhanced_confidence(
            base_confidence, order_flow_bias, signal_type
        )
        
        signal['order_flow'] = {
            'cvd_trend': order_flow_data.get('cvd_trend', 'neutral'),
            'delta_extreme': order_flow_data.get('delta_extreme', False),
            'imbalance_direction': order_flow_data.get('imbalance_direction', 'neutral'),
            'absorption_detected': order_flow_data.get('absorption_detected', False),
            'manipulation_score': order_flow_data.get('manipulation_score', 0.0),
            'tape_signal': order_flow_data.get('tape_signal', 0.0),
            'enhanced_confidence': enhanced_confidence
        }
        
        signal['order_flow_bias'] = order_flow_data.get('order_flow_bias', 0.0)
        signal['manipulation_score'] = order_flow_data.get('manipulation_score', 0.0)
        signal['institutional_activity'] = order_flow_data.get('institutional_activity', 0.0)
        signal['tape_signal'] = order_flow_data.get('tape_signal', 0.0)
        signal['enhanced_confidence'] = enhanced_confidence
        
        signal['order_flow_details'] = {
            'cvd_value': order_flow_data.get('cvd_value', 0.0),
            'cvd_trend_strength': order_flow_data.get('cvd_trend_strength', 0.0),
            'delta_direction': order_flow_data.get('delta_direction', 'none'),
            'imbalance_ratio': order_flow_data.get('imbalance_ratio', 0.0),
            'absorption_zones': order_flow_data.get('absorption_zones', 0),
            'smart_money_direction': order_flow_data.get('smart_money_direction', 'neutral'),
            'smart_money_score': order_flow_data.get('smart_money_score', 0.0),
            'manipulation_recommendation': order_flow_data.get('manipulation_recommendation', 'trade')
        }
        
        logger.debug(f"Order flow enhanced signal: bias={order_flow_bias:.3f}, "
                    f"manipulation={signal['manipulation_score']:.3f}, "
                    f"enhanced_confidence={enhanced_confidence:.3f}")
        
        return signal, None
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all indicators on the data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicator values
        """
        df_ut = self.ut_bot.calculate(df)
        
        df_combined = self.stc.calculate(df_ut)
        
        return df_combined
    
    def find_swing_high(self, df: pd.DataFrame, lookback: Optional[int] = None) -> float:
        """
        Find recent swing high for stop loss (SHORT positions)
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Number of bars to look back
            
        Returns:
            Swing high price
        """
        lookback = lookback or self.swing_lookback
        recent_data = df.tail(lookback + 1)
        return float(recent_data['high'].max())
    
    def find_swing_low(self, df: pd.DataFrame, lookback: Optional[int] = None) -> float:
        """
        Find recent swing low for stop loss (LONG positions)
        
        Args:
            df: DataFrame with OHLCV data
            lookback: Number of bars to look back
            
        Returns:
            Swing low price
        """
        lookback = lookback or self.swing_lookback
        recent_data = df.tail(lookback + 1)
        return float(recent_data['low'].min())
    
    def calculate_risk_reward(self, entry_price: float, stop_loss: float, 
                             direction: str) -> Tuple[float, float, float]:
        """
        Calculate take profit based on risk:reward ratio
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            direction: 'LONG' or 'SHORT'
            
        Returns:
            Tuple of (take_profit, risk_amount, reward_amount)
        """
        risk = abs(entry_price - stop_loss)
        reward = risk * self.risk_reward_ratio
        
        if direction == 'LONG':
            take_profit = entry_price + reward
        else:
            take_profit = entry_price - reward
        
        return take_profit, risk, reward
    
    def calculate_risk_percent(self, entry_price: float, stop_loss: float) -> float:
        """
        Calculate risk as percentage of entry
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            
        Returns:
            Risk percentage
        """
        return abs(entry_price - stop_loss) / entry_price * 100
    
    def check_long_conditions(self, df: pd.DataFrame) -> Dict:
        """
        Check if LONG entry conditions are met
        
        Conditions:
        1. UT Bot gives BUY signal
        2. STC is GREEN color
        3. STC is pointing UP
        4. STC value is below 75
        
        Args:
            df: DataFrame with calculated indicators
            
        Returns:
            Dictionary with signal details
        """
        if len(df) < 3:
            return {'valid': False, 'reason': 'Insufficient data'}
        
        latest = df.iloc[-1]
        
        ut_buy_signal = latest.get('buy_signal', False)
        ut_above_stop = latest.get('above_stop', False)
        
        stc_value = latest.get('stc', 50)
        stc_color = latest.get('stc_color', 'neutral')
        stc_slope = latest.get('stc_slope', 'neutral')
        
        conditions = {
            'ut_buy_signal': bool(ut_buy_signal),
            'ut_above_stop': bool(ut_above_stop),
            'stc_green': stc_color == 'green',
            'stc_up': stc_slope == 'up',
            'stc_below_75': float(stc_value) < 75 if not pd.isna(stc_value) else False
        }
        
        all_conditions_met = all([
            conditions['ut_buy_signal'] or conditions['ut_above_stop'],
            conditions['stc_green'],
            conditions['stc_up'],
            conditions['stc_below_75']
        ])
        
        primary_signal = conditions['ut_buy_signal']
        
        return {
            'valid': all_conditions_met and primary_signal,
            'conditions': conditions,
            'stc_value': stc_value,
            'reason': self._get_condition_reason(conditions, 'LONG')
        }
    
    def check_short_conditions(self, df: pd.DataFrame) -> Dict:
        """
        Check if SHORT entry conditions are met
        
        Conditions:
        1. UT Bot gives SELL signal
        2. STC is RED color
        3. STC is pointing DOWN
        4. STC value is above 25
        
        Args:
            df: DataFrame with calculated indicators
            
        Returns:
            Dictionary with signal details
        """
        if len(df) < 3:
            return {'valid': False, 'reason': 'Insufficient data'}
        
        latest = df.iloc[-1]
        
        ut_sell_signal = latest.get('sell_signal', False)
        ut_below_stop = latest.get('below_stop', False)
        
        stc_value = latest.get('stc', 50)
        stc_color = latest.get('stc_color', 'neutral')
        stc_slope = latest.get('stc_slope', 'neutral')
        
        conditions = {
            'ut_sell_signal': bool(ut_sell_signal),
            'ut_below_stop': bool(ut_below_stop),
            'stc_red': stc_color == 'red',
            'stc_down': stc_slope == 'down',
            'stc_above_25': float(stc_value) > 25 if not pd.isna(stc_value) else False
        }
        
        all_conditions_met = all([
            conditions['ut_sell_signal'] or conditions['ut_below_stop'],
            conditions['stc_red'],
            conditions['stc_down'],
            conditions['stc_above_25']
        ])
        
        primary_signal = conditions['ut_sell_signal']
        
        return {
            'valid': all_conditions_met and primary_signal,
            'conditions': conditions,
            'stc_value': stc_value,
            'reason': self._get_condition_reason(conditions, 'SHORT')
        }
    
    def _get_condition_reason(self, conditions: Dict, direction: str) -> str:
        """Generate human-readable reason for signal validity"""
        failed = []
        
        if direction == 'LONG':
            if not conditions.get('ut_buy_signal'):
                failed.append('No UT Bot BUY signal')
            if not conditions.get('stc_green'):
                failed.append('STC not green')
            if not conditions.get('stc_up'):
                failed.append('STC not pointing up')
            if not conditions.get('stc_below_75'):
                failed.append('STC above 75')
        else:
            if not conditions.get('ut_sell_signal'):
                failed.append('No UT Bot SELL signal')
            if not conditions.get('stc_red'):
                failed.append('STC not red')
            if not conditions.get('stc_down'):
                failed.append('STC not pointing down')
            if not conditions.get('stc_above_25'):
                failed.append('STC below 25')
        
        if not failed:
            return 'All conditions met'
        return ', '.join(failed)
    
    def generate_signal(self, 
                        df: pd.DataFrame,
                        market_context: Optional[Dict] = None,
                        mtf_confirmation: Optional[Dict] = None,
                        derivatives_data: Optional[DerivativesData] = None,
                        whale_metrics: Optional[Dict] = None,
                        calendar_data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Generate trading signal based on strategy rules with order flow enhancement
        and multi-source market intelligence
        
        Args:
            df: DataFrame with OHLCV data
            market_context: Optional[Dict] containing:
                - fear_greed_value: Fear & Greed index (0-100)
                - news_sentiment_score: News sentiment (-1 to 1)
                - market_breadth_score: Market breadth (0-100)
            mtf_confirmation: Optional[Dict] containing:
                - alignment_score: Multi-timeframe alignment (0-1)
                - recommendation: MTF recommendation string
                - higher_timeframe_bias: 'bullish', 'bearish', or 'neutral'
                - confirming_timeframes: List of confirming timeframes
                - conflicting_timeframes: List of conflicting timeframes
            derivatives_data: Optional[DerivativesData] containing:
                - derivatives_score: Composite score (-1 to +1)
                - funding_rate: Current funding rate
                - funding_rate_trend: rising/falling/stable
                - long_short_ratio: Global L/S ratio
                - oi_change_24h: 24h OI change percentage
                - market_sentiment: Market sentiment from derivatives
            whale_metrics: Optional[Dict] containing:
                - whale_bias: Whale buy/sell bias (-1 to +1)
                - smart_money_direction: BULLISH/BEARISH/NEUTRAL
                - whale_buy_volume: Total whale buy volume
                - whale_sell_volume: Total whale sell volume
                - net_whale_flow: Net flow (buy - sell)
            calendar_data: Optional[Dict] containing:
                - has_imminent_event: Whether a high-impact event is imminent
                - should_avoid_trading: Whether to avoid new positions
                - next_major_event: Info about next major event
                - trading_recommendation: Calendar-based recommendation
            
        Returns:
            Signal dictionary or None if no valid signal
        """
        if len(df) < 100:
            logger.warning("Insufficient data for signal generation")
            return None
        
        df_calculated = self.calculate_indicators(df)
        
        latest = df_calculated.iloc[-1]
        index_value = df_calculated.index[-1]
        if isinstance(index_value, datetime):
            current_time = index_value
        elif isinstance(index_value, pd.Timestamp):
            current_time = index_value.to_pydatetime()
        else:
            timestamp_method = getattr(index_value, 'timestamp', None)
            if timestamp_method is not None and callable(timestamp_method):
                current_time = datetime.fromtimestamp(timestamp_method())
            else:
                current_time = datetime.now()
        entry_price = float(latest['close'])
        
        long_check = self.check_long_conditions(df_calculated)
        short_check = self.check_short_conditions(df_calculated)
        
        signal = None
        rejection_reason = None
        
        if long_check['valid']:
            stop_loss = self.find_swing_low(df_calculated)
            take_profit, risk, reward = self.calculate_risk_reward(entry_price, stop_loss, 'LONG')
            risk_percent = self.calculate_risk_percent(entry_price, stop_loss)
            
            if self.min_risk_percent <= risk_percent <= self.max_risk_percent:
                signal = {
                    'type': 'LONG',
                    'symbol': 'ETH/USDT',
                    'timeframe': '5m',
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'risk_amount': risk,
                    'reward_amount': reward,
                    'risk_percent': risk_percent,
                    'risk_reward_ratio': self.risk_reward_ratio,
                    'timestamp': current_time,
                    'conditions': long_check['conditions'],
                    'stc_value': long_check['stc_value'],
                    'ut_trailing_stop': float(latest.get('trailing_stop', 0)),
                    'atr': float(latest.get('atr', 0)),
                    'reason': long_check['reason'],
                    'confidence': 0.7,
                    'recommended_leverage': 16,
                    'leverage_config': {
                        'base_leverage': 12,
                        'margin_type': 'CROSS',
                        'auto_add_margin': True
                    }
                }
            else:
                logger.info(f"LONG signal rejected: Risk {risk_percent:.2f}% outside acceptable range")
        
        elif short_check['valid']:
            stop_loss = self.find_swing_high(df_calculated)
            take_profit, risk, reward = self.calculate_risk_reward(entry_price, stop_loss, 'SHORT')
            risk_percent = self.calculate_risk_percent(entry_price, stop_loss)
            
            if self.min_risk_percent <= risk_percent <= self.max_risk_percent:
                signal = {
                    'type': 'SHORT',
                    'symbol': 'ETH/USDT',
                    'timeframe': '5m',
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'risk_amount': risk,
                    'reward_amount': reward,
                    'risk_percent': risk_percent,
                    'risk_reward_ratio': self.risk_reward_ratio,
                    'timestamp': current_time,
                    'conditions': short_check['conditions'],
                    'stc_value': short_check['stc_value'],
                    'ut_trailing_stop': float(latest.get('trailing_stop', 0)),
                    'atr': float(latest.get('atr', 0)),
                    'reason': short_check['reason'],
                    'confidence': 0.7,
                    'recommended_leverage': 16,
                    'leverage_config': {
                        'base_leverage': 12,
                        'margin_type': 'CROSS',
                        'auto_add_margin': True
                    }
                }
            else:
                logger.info(f"SHORT signal rejected: Risk {risk_percent:.2f}% outside acceptable range")
        
        if signal:
            enhanced_signal, rejection_reason = self._enhance_with_order_flow(signal)
            
            if enhanced_signal is None:
                logger.info(f"Signal filtered by order flow: {rejection_reason}")
                return None
            
            signal = enhanced_signal
            
            base_confidence = signal.get('confidence', 0.7)
            order_flow_bias = signal.get('order_flow_bias', 0.0)
            signal_type = signal.get('type', '')
            
            multi_source_confidence, component_scores = self._calculate_multi_source_confidence(
                base_confidence=base_confidence,
                signal_type=signal_type,
                order_flow_bias=order_flow_bias,
                market_context=market_context,
                mtf_confirmation=mtf_confirmation,
                derivatives_data=derivatives_data,
                whale_metrics=whale_metrics,
                calendar_data=calendar_data
            )
            
            signal['multi_source_confidence'] = multi_source_confidence
            
            if calendar_data:
                has_imminent = calendar_data.get('has_imminent_event', False)
                should_avoid = calendar_data.get('should_avoid_trading', False)
                signal['event_warning'] = has_imminent or should_avoid
                
                next_event = calendar_data.get('next_major_event')
                if next_event and next_event.get('minutes_until', 60) <= 30:
                    calendar_factor = component_scores.get('economic_calendar_factor', 1.0)
                    signal['multi_source_confidence'] = multi_source_confidence * calendar_factor
                    logger.info(f"High-impact event within 30 minutes - confidence reduced by factor {calendar_factor:.2f}")
            else:
                signal['event_warning'] = False
            
            signal['market_intelligence'] = self._build_market_intelligence(
                market_context=market_context,
                mtf_confirmation=mtf_confirmation,
                component_scores=component_scores,
                overall_score=signal['multi_source_confidence'],
                derivatives_data=derivatives_data,
                whale_metrics=whale_metrics,
                calendar_data=calendar_data
            )
            
            if (self._last_signal_time == current_time and 
                self._last_signal_type == signal['type']):
                logger.debug("Duplicate signal detected, skipping")
                return None
            
            self._last_signal_time = current_time
            self._last_signal_type = signal['type']
            self._signal_history.append(signal)
            
            if len(self._signal_history) > 100:
                self._signal_history = self._signal_history[-100:]
            
            of_info = ""
            if 'order_flow_bias' in signal:
                of_info = f" (OF bias: {signal['order_flow_bias']:.2f}, confidence: {signal.get('enhanced_confidence', 0.7):.2f})"
            
            mi_info = f", multi-source: {multi_source_confidence:.2f}"
            
            logger.info(f"Generated {signal['type']} signal at {entry_price}{of_info}{mi_info}")
        
        return signal
    
    def get_market_state(self, df: pd.DataFrame) -> Dict:
        """
        Get current market state without generating a signal
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with current market state
        """
        if len(df) < 100:
            return {'error': 'Insufficient data'}
        
        df_calculated = self.calculate_indicators(df)
        latest = df_calculated.iloc[-1]
        
        state = {
            'timestamp': df_calculated.index[-1],
            'price': float(latest['close']),
            'ut_position': int(latest.get('position', 0)),
            'ut_bar_color': latest.get('bar_color', 'neutral'),
            'ut_trailing_stop': float(latest.get('trailing_stop', 0)),
            'stc_value': float(latest.get('stc', 0)),
            'stc_color': latest.get('stc_color', 'neutral'),
            'stc_slope': latest.get('stc_slope', 'neutral'),
            'atr': float(latest.get('atr', 0)),
            'long_conditions': self.check_long_conditions(df_calculated),
            'short_conditions': self.check_short_conditions(df_calculated)
        }
        
        if self.order_flow_enabled and self._order_flow_metrics_service:
            order_flow_data = self._get_order_flow_data()
            if order_flow_data:
                state['order_flow'] = {
                    'bias': order_flow_data.get('order_flow_bias', 0.0),
                    'cvd_trend': order_flow_data.get('cvd_trend', 'neutral'),
                    'manipulation_score': order_flow_data.get('manipulation_score', 0.0),
                    'institutional_activity': order_flow_data.get('institutional_activity', 0.0),
                    'tape_signal': order_flow_data.get('tape_signal', 0.0)
                }
        
        return state
    
    def get_signal_history(self) -> List[Dict]:
        """Get recent signal history"""
        return self._signal_history.copy()
    
    def get_order_flow_status(self) -> Dict[str, Any]:
        """Get order flow integration status"""
        return {
            'enabled': self.order_flow_enabled,
            'manipulation_filter_enabled': self.manipulation_filter_enabled,
            'order_flow_weight': self.order_flow_weight,
            'service_connected': self._order_flow_metrics_service is not None,
            'manipulation_threshold': self.MANIPULATION_THRESHOLD,
            'divergence_threshold': self.ORDER_FLOW_DIVERGENCE_THRESHOLD
        }
    
    def get_market_intelligence_status(self) -> Dict[str, Any]:
        """Get market intelligence integration status"""
        return {
            'fear_greed_connected': self._fear_greed_client is not None,
            'news_client_connected': self._news_client is not None,
            'market_aggregator_connected': self._market_aggregator is not None,
            'confidence_weights': {
                'base_indicator': 0.40,
                'order_flow_alignment': 0.20,
                'fear_greed_alignment': 0.10,
                'news_sentiment': 0.10,
                'mtf_confirmation': 0.15,
                'market_breadth': 0.05
            }
        }
