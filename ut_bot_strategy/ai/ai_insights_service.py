"""
AI Insights Service for UT Bot + STC Trading Strategy

Integrates OpenAI GPT-5 for intelligent signal analysis including:
- Signal confidence scoring
- Market sentiment analysis
- AI-adjusted leverage recommendations
"""

import os
import json
import asyncio
import hashlib
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class AIInsights:
    """
    Comprehensive AI insights for trading signals
    
    Attributes:
        confidence_score: AI confidence in the signal (0-100)
        signal_strength: Signal quality assessment (STRONG/MODERATE/WEAK)
        market_sentiment: Overall market sentiment (BULLISH/BEARISH/NEUTRAL)
        volatility_assessment: Market volatility level (LOW/MEDIUM/HIGH/EXTREME)
        ai_reasoning: Brief explanation of the AI's analysis
        recommended_leverage: AI-suggested leverage (1x-25x)
        timestamp: When the analysis was generated
    """
    confidence_score: float
    signal_strength: str
    market_sentiment: str
    volatility_assessment: str
    ai_reasoning: str
    recommended_leverage: int
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert AIInsights to dictionary"""
        return {
            'confidence_score': self.confidence_score,
            'signal_strength': self.signal_strength,
            'market_sentiment': self.market_sentiment,
            'volatility_assessment': self.volatility_assessment,
            'ai_reasoning': self.ai_reasoning,
            'recommended_leverage': self.recommended_leverage,
            'timestamp': self.timestamp.isoformat()
        }


class AIInsightsService:
    """
    AI-powered trading signal analysis service using OpenAI GPT-5
    
    Provides intelligent analysis of trading signals including confidence scoring,
    market sentiment analysis, and leverage recommendations.
    """
    
    # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
    # do not change this unless explicitly requested by the user
    MODEL = "gpt-5"
    REQUEST_TIMEOUT = 2.0
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize AI Insights Service
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client: Optional[OpenAI] = None
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = 300
        
        if self.api_key:
            self._client = OpenAI(api_key=self.api_key)
            logger.info("AIInsightsService initialized with OpenAI GPT-5")
        else:
            logger.warning("AIInsightsService initialized without API key - will use fallback values")
    
    def _get_cache_key(self, data: Dict[str, Any]) -> str:
        """Generate cache key from data"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if still valid"""
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if (datetime.now() - cached['timestamp']).total_seconds() < self._cache_ttl:
                logger.debug(f"Cache hit for key: {cache_key[:8]}...")
                return cached['data']
            else:
                del self._cache[cache_key]
        return None
    
    def _set_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Store result in cache"""
        self._cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }
        if len(self._cache) > 100:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]['timestamp'])
            del self._cache[oldest_key]
    
    async def analyze_signal(self, signal_data: Dict[str, Any], market_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a trading signal and return confidence assessment
        
        Args:
            signal_data: Signal information (type, entry, stop_loss, take_profit, etc.)
            market_features: Extracted market features from FeatureExtractor
            
        Returns:
            Dictionary with confidence_score (0-100), reasoning, and recommendation (STRONG/MODERATE/WEAK)
        """
        logger.info(f"Analyzing {signal_data.get('type', 'UNKNOWN')} signal with AI")
        
        cache_input = {'signal': signal_data, 'features': market_features, 'method': 'analyze_signal'}
        cache_key = self._get_cache_key(cache_input)
        
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        if not self._client:
            logger.warning("OpenAI client not available, using fallback signal analysis")
            return self._fallback_signal_analysis(signal_data, market_features)
        
        try:
            prompt = self._build_signal_analysis_prompt(signal_data, market_features)
            
            result = await asyncio.wait_for(
                self._call_openai(prompt, system_message=self._get_signal_analysis_system_prompt()),
                timeout=self.REQUEST_TIMEOUT
            )
            
            parsed_result = self._parse_signal_analysis_response(result)
            self._set_cache(cache_key, parsed_result)
            
            logger.info(f"AI signal analysis complete: confidence={parsed_result['confidence_score']}, recommendation={parsed_result['recommendation']}")
            return parsed_result
            
        except asyncio.TimeoutError:
            logger.warning(f"AI signal analysis timed out after {self.REQUEST_TIMEOUT}s, using fallback")
            return self._fallback_signal_analysis(signal_data, market_features)
        except Exception as e:
            logger.error(f"AI signal analysis failed: {e}, using fallback")
            return self._fallback_signal_analysis(signal_data, market_features)
    
    async def analyze_market_sentiment(self, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze overall market sentiment
        
        Args:
            price_data: Price and market data including features
            
        Returns:
            Dictionary with sentiment (BULLISH/BEARISH/NEUTRAL), volatility_level, and trend_analysis
        """
        logger.info("Analyzing market sentiment with AI")
        
        cache_input = {'price_data': price_data, 'method': 'analyze_sentiment'}
        cache_key = self._get_cache_key(cache_input)
        
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        if not self._client:
            logger.warning("OpenAI client not available, using fallback sentiment analysis")
            return self._fallback_sentiment_analysis(price_data)
        
        try:
            prompt = self._build_sentiment_analysis_prompt(price_data)
            
            result = await asyncio.wait_for(
                self._call_openai(prompt, system_message=self._get_sentiment_analysis_system_prompt()),
                timeout=self.REQUEST_TIMEOUT
            )
            
            parsed_result = self._parse_sentiment_analysis_response(result)
            self._set_cache(cache_key, parsed_result)
            
            logger.info(f"AI sentiment analysis complete: sentiment={parsed_result['sentiment']}, volatility={parsed_result['volatility_level']}")
            return parsed_result
            
        except asyncio.TimeoutError:
            logger.warning(f"AI sentiment analysis timed out after {self.REQUEST_TIMEOUT}s, using fallback")
            return self._fallback_sentiment_analysis(price_data)
        except Exception as e:
            logger.error(f"AI sentiment analysis failed: {e}, using fallback")
            return self._fallback_sentiment_analysis(price_data)
    
    async def get_leverage_recommendation(self, signal: Dict[str, Any], confidence: float, volatility: str) -> Dict[str, Any]:
        """
        Get AI-adjusted leverage recommendation
        
        Args:
            signal: Trading signal data
            confidence: Confidence score from analyze_signal (0-100)
            volatility: Volatility assessment (LOW/MEDIUM/HIGH/EXTREME)
            
        Returns:
            Dictionary with recommended_leverage (1-25), reasoning, and risk_level
        """
        logger.info(f"Getting AI leverage recommendation for {signal.get('type', 'UNKNOWN')} signal")
        
        cache_input = {'signal': signal, 'confidence': confidence, 'volatility': volatility, 'method': 'get_leverage'}
        cache_key = self._get_cache_key(cache_input)
        
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        if not self._client:
            logger.warning("OpenAI client not available, using fallback leverage calculation")
            return self._fallback_leverage_recommendation(signal, confidence, volatility)
        
        try:
            prompt = self._build_leverage_prompt(signal, confidence, volatility)
            
            result = await asyncio.wait_for(
                self._call_openai(prompt, system_message=self._get_leverage_system_prompt()),
                timeout=self.REQUEST_TIMEOUT
            )
            
            parsed_result = self._parse_leverage_response(result)
            self._set_cache(cache_key, parsed_result)
            
            logger.info(f"AI leverage recommendation: {parsed_result['recommended_leverage']}x, risk={parsed_result['risk_level']}")
            return parsed_result
            
        except asyncio.TimeoutError:
            logger.warning(f"AI leverage recommendation timed out after {self.REQUEST_TIMEOUT}s, using fallback")
            return self._fallback_leverage_recommendation(signal, confidence, volatility)
        except Exception as e:
            logger.error(f"AI leverage recommendation failed: {e}, using fallback")
            return self._fallback_leverage_recommendation(signal, confidence, volatility)
    
    async def get_comprehensive_insights(self, signal_data: Dict[str, Any], market_features: Dict[str, Any]) -> AIInsights:
        """
        Get comprehensive AI insights combining all analysis methods
        
        Args:
            signal_data: Trading signal information
            market_features: Extracted market features
            
        Returns:
            AIInsights dataclass with all analysis results
        """
        logger.info("Generating comprehensive AI insights")
        
        signal_analysis, sentiment_analysis = await asyncio.gather(
            self.analyze_signal(signal_data, market_features),
            self.analyze_market_sentiment(market_features),
            return_exceptions=True
        )
        
        if isinstance(signal_analysis, Exception):
            logger.error(f"Signal analysis failed: {signal_analysis}")
            signal_analysis = self._fallback_signal_analysis(signal_data, market_features)
        
        if isinstance(sentiment_analysis, Exception):
            logger.error(f"Sentiment analysis failed: {sentiment_analysis}")
            sentiment_analysis = self._fallback_sentiment_analysis(market_features)
        
        leverage_result = await self.get_leverage_recommendation(
            signal_data,
            signal_analysis['confidence_score'],
            sentiment_analysis['volatility_level']
        )
        
        insights = AIInsights(
            confidence_score=signal_analysis['confidence_score'],
            signal_strength=signal_analysis['recommendation'],
            market_sentiment=sentiment_analysis['sentiment'],
            volatility_assessment=sentiment_analysis['volatility_level'],
            ai_reasoning=signal_analysis['reasoning'],
            recommended_leverage=leverage_result['recommended_leverage'],
            timestamp=datetime.now()
        )
        
        logger.info(f"Comprehensive AI insights generated: confidence={insights.confidence_score}, strength={insights.signal_strength}, leverage={insights.recommended_leverage}x")
        return insights
    
    async def _call_openai(self, prompt: str, system_message: str) -> str:
        """
        Make async call to OpenAI API
        
        Args:
            prompt: User prompt
            system_message: System message for context
            
        Returns:
            Response content string
        """
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=1024
            )
        )
        return response.choices[0].message.content
    
    def _get_signal_analysis_system_prompt(self) -> str:
        """Get system prompt for signal analysis"""
        return """You are an expert cryptocurrency trading analyst specializing in technical analysis. 
Analyze trading signals and provide confidence assessments.
Always respond with valid JSON in this exact format:
{
    "confidence_score": <number 0-100>,
    "recommendation": "<STRONG|MODERATE|WEAK>",
    "reasoning": "<brief 1-2 sentence explanation>"
}
Consider signal alignment with indicators, market conditions, risk/reward ratio, and volume."""
    
    def _get_sentiment_analysis_system_prompt(self) -> str:
        """Get system prompt for sentiment analysis"""
        return """You are an expert market analyst specializing in cryptocurrency market sentiment.
Analyze market data and provide sentiment assessment.
Always respond with valid JSON in this exact format:
{
    "sentiment": "<BULLISH|BEARISH|NEUTRAL>",
    "volatility_level": "<LOW|MEDIUM|HIGH|EXTREME>",
    "trend_analysis": "<brief 1-2 sentence trend description>"
}
Consider price action, momentum, volatility metrics, and trend indicators."""
    
    def _get_leverage_system_prompt(self) -> str:
        """Get system prompt for leverage recommendation"""
        return """You are a risk management expert for cryptocurrency futures trading.
Recommend appropriate leverage based on signal quality and market conditions.
Always respond with valid JSON in this exact format:
{
    "recommended_leverage": <integer 1-25>,
    "risk_level": "<LOW|MEDIUM|HIGH|VERY_HIGH>",
    "reasoning": "<brief 1-2 sentence explanation>"
}
Be conservative with leverage in high volatility or low confidence situations."""
    
    def _build_signal_analysis_prompt(self, signal_data: Dict[str, Any], market_features: Dict[str, Any]) -> str:
        """Build prompt for signal analysis"""
        return f"""Analyze this trading signal:
Signal Type: {signal_data.get('type', 'UNKNOWN')}
Entry Price: {signal_data.get('entry_price', 'N/A')}
Stop Loss: {signal_data.get('stop_loss', 'N/A')}
Take Profit: {signal_data.get('take_profit', 'N/A')}
Risk/Reward: {signal_data.get('risk_reward', 'N/A')}

Market Features:
- Trend: {market_features.get('rolling_statistics', {}).get('trend_direction', 'UNKNOWN')}
- Momentum: {market_features.get('rolling_statistics', {}).get('momentum', 0)}%
- Volatility: {market_features.get('rolling_statistics', {}).get('volatility_level', 'UNKNOWN')}
- STC Value: {market_features.get('indicator_summary', {}).get('stc', {}).get('value', 50)}
- STC Direction: {market_features.get('indicator_summary', {}).get('stc', {}).get('direction', 'FLAT')}
- UT Bot Position: {market_features.get('indicator_summary', {}).get('ut_bot', {}).get('position', 0)}
- Volume Ratio: {market_features.get('volume_analysis', {}).get('volume_ratio', 1.0)}

Provide your analysis in JSON format."""
    
    def _build_sentiment_analysis_prompt(self, price_data: Dict[str, Any]) -> str:
        """Build prompt for sentiment analysis"""
        rolling_stats = price_data.get('rolling_statistics', {})
        market_context = price_data.get('market_context', {})
        
        return f"""Analyze market sentiment based on these metrics:
- Price vs EMA20: {rolling_stats.get('price_vs_ema20', 0)}%
- Price vs EMA50: {rolling_stats.get('price_vs_ema50', 0)}%
- Momentum ({market_context.get('data_points', 0)} periods): {rolling_stats.get('momentum', 0)}%
- Trend Direction: {rolling_stats.get('trend_direction', 'UNKNOWN')}
- Trend Strength: {rolling_stats.get('trend_strength', 0)}%
- Volatility: {rolling_stats.get('volatility', 0)}%
- 1H Return: {market_context.get('returns_1h', 0)}%
- 4H Return: {market_context.get('returns_4h', 0)}%
- Market Phase: {market_context.get('market_phase', 'UNKNOWN')}

Provide your analysis in JSON format."""
    
    def _build_leverage_prompt(self, signal: Dict[str, Any], confidence: float, volatility: str) -> str:
        """Build prompt for leverage recommendation"""
        return f"""Recommend leverage for this trade:
Signal Type: {signal.get('type', 'UNKNOWN')}
Confidence Score: {confidence}/100
Market Volatility: {volatility}
Risk/Reward Ratio: {signal.get('risk_reward', 1.5)}
Stop Loss Distance: {signal.get('risk_percent', 1.0)}%

Rules:
- Maximum leverage: 25x
- Minimum leverage: 1x
- Higher confidence = can use higher leverage
- Higher volatility = should use lower leverage
- Consider risk/reward ratio

Provide your recommendation in JSON format."""
    
    def _parse_signal_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse signal analysis response from AI"""
        try:
            data = json.loads(response)
            return {
                'confidence_score': max(0, min(100, float(data.get('confidence_score', 50)))),
                'recommendation': data.get('recommendation', 'MODERATE').upper(),
                'reasoning': str(data.get('reasoning', 'Analysis completed'))
            }
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.error(f"Failed to parse signal analysis response: {e}")
            return {
                'confidence_score': 50.0,
                'recommendation': 'MODERATE',
                'reasoning': 'Unable to parse AI response, using default assessment'
            }
    
    def _parse_sentiment_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse sentiment analysis response from AI"""
        try:
            data = json.loads(response)
            return {
                'sentiment': data.get('sentiment', 'NEUTRAL').upper(),
                'volatility_level': data.get('volatility_level', 'MEDIUM').upper(),
                'trend_analysis': str(data.get('trend_analysis', 'Market conditions unclear'))
            }
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.error(f"Failed to parse sentiment analysis response: {e}")
            return {
                'sentiment': 'NEUTRAL',
                'volatility_level': 'MEDIUM',
                'trend_analysis': 'Unable to parse AI response, using neutral assessment'
            }
    
    def _parse_leverage_response(self, response: str) -> Dict[str, Any]:
        """Parse leverage recommendation response from AI"""
        try:
            data = json.loads(response)
            leverage = int(data.get('recommended_leverage', 5))
            leverage = max(1, min(25, leverage))
            return {
                'recommended_leverage': leverage,
                'risk_level': data.get('risk_level', 'MEDIUM').upper(),
                'reasoning': str(data.get('reasoning', 'Leverage recommendation calculated'))
            }
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.error(f"Failed to parse leverage response: {e}")
            return {
                'recommended_leverage': 5,
                'risk_level': 'MEDIUM',
                'reasoning': 'Unable to parse AI response, using conservative leverage'
            }
    
    def _fallback_signal_analysis(self, signal_data: Dict[str, Any], market_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback signal analysis when AI is unavailable
        Uses rule-based logic based on indicator alignment
        """
        logger.debug("Using fallback signal analysis")
        
        score = 50.0
        reasons = []
        
        indicator_summary = market_features.get('indicator_summary', {})
        rolling_stats = market_features.get('rolling_statistics', {})
        volume_analysis = market_features.get('volume_analysis', {})
        
        signal_type = signal_data.get('type', 'UNKNOWN')
        
        ut_bot = indicator_summary.get('ut_bot', {})
        stc = indicator_summary.get('stc', {})
        
        if signal_type == 'LONG':
            if ut_bot.get('buy_signal'):
                score += 15
                reasons.append("UT Bot buy signal")
            if ut_bot.get('above_stop'):
                score += 5
                reasons.append("Price above trailing stop")
            if stc.get('color') == 'green':
                score += 10
                reasons.append("STC green")
            if stc.get('direction') == 'UP':
                score += 10
                reasons.append("STC rising")
            if stc.get('value', 50) < 75:
                score += 5
                reasons.append("STC below overbought")
        elif signal_type == 'SHORT':
            if ut_bot.get('sell_signal'):
                score += 15
                reasons.append("UT Bot sell signal")
            if ut_bot.get('below_stop'):
                score += 5
                reasons.append("Price below trailing stop")
            if stc.get('color') == 'red':
                score += 10
                reasons.append("STC red")
            if stc.get('direction') == 'DOWN':
                score += 10
                reasons.append("STC falling")
            if stc.get('value', 50) > 25:
                score += 5
                reasons.append("STC above oversold")
        
        if volume_analysis.get('is_high_volume'):
            score += 5
            reasons.append("High volume confirmation")
        
        trend = rolling_stats.get('trend_direction', 'NEUTRAL')
        if (signal_type == 'LONG' and trend == 'BULLISH') or (signal_type == 'SHORT' and trend == 'BEARISH'):
            score += 5
            reasons.append("Aligned with trend")
        elif (signal_type == 'LONG' and trend == 'BEARISH') or (signal_type == 'SHORT' and trend == 'BULLISH'):
            score -= 10
            reasons.append("Counter-trend trade")
        
        score = max(0, min(100, score))
        
        if score >= 75:
            recommendation = 'STRONG'
        elif score >= 50:
            recommendation = 'MODERATE'
        else:
            recommendation = 'WEAK'
        
        reasoning = '; '.join(reasons[:3]) if reasons else 'Basic indicator analysis'
        
        return {
            'confidence_score': score,
            'recommendation': recommendation,
            'reasoning': reasoning
        }
    
    def _fallback_sentiment_analysis(self, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback sentiment analysis when AI is unavailable
        Uses rule-based logic based on market metrics
        """
        logger.debug("Using fallback sentiment analysis")
        
        rolling_stats = price_data.get('rolling_statistics', {})
        market_context = price_data.get('market_context', {})
        
        trend = rolling_stats.get('trend_direction', 'NEUTRAL')
        momentum = rolling_stats.get('momentum', 0)
        volatility_level = rolling_stats.get('volatility_level', 'MEDIUM')
        market_phase = market_context.get('market_phase', 'RANGING')
        
        if trend == 'BULLISH' and momentum > 1:
            sentiment = 'BULLISH'
            trend_analysis = f"Market in {market_phase} with positive momentum of {momentum:.1f}%"
        elif trend == 'BEARISH' and momentum < -1:
            sentiment = 'BEARISH'
            trend_analysis = f"Market in {market_phase} with negative momentum of {momentum:.1f}%"
        else:
            sentiment = 'NEUTRAL'
            trend_analysis = f"Market in {market_phase} with mixed signals"
        
        return {
            'sentiment': sentiment,
            'volatility_level': volatility_level,
            'trend_analysis': trend_analysis
        }
    
    def _fallback_leverage_recommendation(self, signal: Dict[str, Any], confidence: float, volatility: str) -> Dict[str, Any]:
        """
        Fallback leverage recommendation when AI is unavailable
        Uses conservative rule-based calculation
        """
        logger.debug("Using fallback leverage calculation")
        
        base_leverage = 5
        
        if confidence >= 80:
            base_leverage = 12
        elif confidence >= 60:
            base_leverage = 8
        elif confidence >= 40:
            base_leverage = 5
        else:
            base_leverage = 3
        
        volatility_multiplier = {
            'LOW': 1.2,
            'MEDIUM': 1.0,
            'HIGH': 0.7,
            'EXTREME': 0.4
        }.get(volatility, 1.0)
        
        recommended_leverage = int(base_leverage * volatility_multiplier)
        recommended_leverage = max(1, min(25, recommended_leverage))
        
        if recommended_leverage >= 15:
            risk_level = 'VERY_HIGH'
        elif recommended_leverage >= 10:
            risk_level = 'HIGH'
        elif recommended_leverage >= 5:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        return {
            'recommended_leverage': recommended_leverage,
            'risk_level': risk_level,
            'reasoning': f"Based on {confidence:.0f}% confidence and {volatility} volatility"
        }
    
    def clear_cache(self) -> None:
        """Clear the results cache"""
        self._cache.clear()
        logger.info("AI insights cache cleared")
