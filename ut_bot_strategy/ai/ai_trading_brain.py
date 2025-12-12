"""
AI Trading Brain Module

Advanced AI-powered trading analysis using OpenAI GPT-5 for:
- Signal analysis with confidence scoring
- Learning from trade outcomes
- Dynamic market insights
- Parameter optimization suggestions
- Multi-source market intelligence integration

Features:
- SQLite database for persistent learning data
- Caching to avoid redundant API calls
- Fallback mode when OpenAI is unavailable
- Fear & Greed Index integration
- News sentiment analysis
- Market breadth data
- Multi-timeframe confirmation
"""

import os
import json
import logging
import hashlib
import asyncio
import aiosqlite
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None  # type: ignore
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


class AITradingBrain:
    """
    Advanced AI Trading Brain with GPT-5 integration.
    
    Provides intelligent trading analysis, learning capabilities,
    and parameter optimization using OpenAI's GPT-5 model.
    """
    
    MODEL = "gpt-5"
    
    def __init__(self, db_path: str = "ut_bot_strategy/data/ai_trading_brain.db",
                 cache_ttl_seconds: int = 300):
        """
        Initialize AI Trading Brain.
        
        Args:
            db_path: Path to SQLite database for learning data
            cache_ttl_seconds: Cache time-to-live in seconds (default 5 minutes)
        """
        self.db_path = db_path
        self.cache_ttl = cache_ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._initialized = False
        
        if OPENAI_AVAILABLE and OPENAI_API_KEY:
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            self.ai_available = True
            logger.info("AITradingBrain initialized with OpenAI GPT-5")
        else:
            self.openai_client = None
            self.ai_available = False
            if not OPENAI_AVAILABLE:
                logger.warning("OpenAI package not installed - running in fallback mode")
            else:
                logger.warning("OPENAI_API_KEY not set - running in fallback mode")
        
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self) -> None:
        """Initialize database tables."""
        if self._initialized:
            return
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS trade_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    outcome TEXT,
                    profit_loss REAL,
                    profit_loss_percent REAL,
                    signal_confidence REAL,
                    ai_analysis TEXT,
                    lessons_learned TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    closed_at TIMESTAMP
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS market_insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT,
                    insight_type TEXT NOT NULL,
                    insight_data TEXT NOT NULL,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS parameter_suggestions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parameter_name TEXT NOT NULL,
                    current_value TEXT,
                    suggested_value TEXT,
                    reason TEXT,
                    performance_impact REAL,
                    applied BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS learning_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    context TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_trade_outcomes_symbol 
                ON trade_outcomes(symbol)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_trade_outcomes_outcome 
                ON trade_outcomes(outcome)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_insights_symbol 
                ON market_insights(symbol)
            """)
            
            await db.commit()
        
        self._initialized = True
        logger.info("AITradingBrain database initialized")
    
    def _generate_cache_key(self, prefix: str, data: Any) -> str:
        """Generate a cache key from data."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        hash_value = hashlib.md5(data_str.encode()).hexdigest()[:16]
        return f"{prefix}_{hash_value}"
    
    def _get_cached(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if still valid."""
        if key in self._cache:
            cached = self._cache[key]
            if datetime.now() < cached['expires']:
                logger.debug(f"Cache hit for {key}")
                return cached['data']
            else:
                del self._cache[key]
        return None
    
    def _set_cached(self, key: str, data: Dict[str, Any]) -> None:
        """Cache result with TTL."""
        self._cache[key] = {
            'data': data,
            'expires': datetime.now() + timedelta(seconds=self.cache_ttl)
        }
    
    def _call_openai(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Make synchronous OpenAI API call with JSON response format.
        
        Args:
            messages: List of message dictionaries for the chat
            
        Returns:
            Parsed JSON response from GPT-5
        """
        if not self.ai_available or not self.openai_client:
            raise RuntimeError("OpenAI client not available")
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.MODEL,
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    def _build_analysis_prompt(self, signal_data: Dict[str, Any], 
                                market_intelligence: Optional[Dict[str, Any]] = None,
                                historical_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Build a comprehensive analysis prompt incorporating multi-source market intelligence.
        
        Args:
            signal_data: Dictionary containing signal information
            market_intelligence: Optional dict with fear_greed, news_sentiment, market_breadth, mtf_confirmation
            historical_context: Optional dict with historical performance data
            
        Returns:
            Formatted prompt string for OpenAI analysis
        """
        symbol = signal_data.get('symbol', 'UNKNOWN')
        direction = signal_data.get('direction', 'UNKNOWN')
        entry = signal_data.get('entry_price', 0)
        stop_loss = signal_data.get('stop_loss', 0)
        take_profit = signal_data.get('take_profit', 0)
        indicators = signal_data.get('indicators', {})
        
        risk_percent = abs(entry - stop_loss) / entry * 100 if entry > 0 else 0
        reward_percent = abs(take_profit - entry) / entry * 100 if entry > 0 else 0
        rr_ratio = reward_percent / risk_percent if risk_percent > 0 else 0
        
        prompt = f"""Analyze this trading signal with comprehensive market intelligence:

=== SIGNAL DATA ===
Symbol: {symbol}
Direction: {direction}
Entry Price: {entry}
Stop Loss: {stop_loss} (Risk: {risk_percent:.2f}%)
Take Profit: {take_profit} (Reward: {reward_percent:.2f}%)
Risk/Reward Ratio: {rr_ratio:.2f}

=== TECHNICAL INDICATORS ===
{json.dumps(indicators, indent=2)}
"""
        
        if market_intelligence:
            fear_greed = market_intelligence.get('fear_greed', {})
            news_sentiment = market_intelligence.get('news_sentiment', {})
            market_breadth = market_intelligence.get('market_breadth', {})
            mtf_confirmation = market_intelligence.get('mtf_confirmation', {})
            overall_score = market_intelligence.get('overall_intelligence_score', 0.5)
            
            fg_value = fear_greed.get('value', 50)
            fg_classification = fear_greed.get('classification', 'neutral')
            fg_trend = fear_greed.get('trend', 'stable')
            
            prompt += f"""
=== FEAR & GREED INDEX ===
Current Value: {fg_value}/100
Classification: {fg_classification}
Trend: {fg_trend}
Analysis Guidance:
- Extreme Fear (0-25): Contrarian BUY opportunity, favors LONG positions
- Fear (25-45): Cautious BUY bias
- Neutral (45-55): No clear sentiment edge
- Greed (55-75): Cautious SELL bias
- Extreme Greed (75-100): Contrarian SELL opportunity, favors SHORT positions
"""
            
            bullish_count = news_sentiment.get('bullish_count', 0)
            bearish_count = news_sentiment.get('bearish_count', 0)
            news_bias = news_sentiment.get('overall_bias', 'neutral')
            news_score = news_sentiment.get('sentiment_score', 0.0)
            
            prompt += f"""
=== NEWS SENTIMENT ===
Bullish News Count: {bullish_count}
Bearish News Count: {bearish_count}
Overall Bias: {news_bias}
Sentiment Score: {news_score:.2f} (-1 = very bearish, +1 = very bullish)
"""
            
            coins_up = market_breadth.get('coins_up', 0)
            coins_down = market_breadth.get('coins_down', 0)
            breadth_percentage = market_breadth.get('breadth_percentage', 50)
            market_trend = market_breadth.get('market_trend', 'mixed')
            
            prompt += f"""
=== MARKET BREADTH ===
Top Coins Moving Up: {coins_up}
Top Coins Moving Down: {coins_down}
Breadth Percentage: {breadth_percentage:.1f}% bullish
Market Trend: {market_trend}
"""
            
            confirming_tfs = mtf_confirmation.get('confirming_timeframes', [])
            conflicting_tfs = mtf_confirmation.get('conflicting_timeframes', [])
            mtf_alignment = mtf_confirmation.get('alignment_score', 0.5)
            higher_tf_bias = mtf_confirmation.get('higher_tf_bias', 'neutral')
            
            prompt += f"""
=== MULTI-TIMEFRAME CONFIRMATION ===
Confirming Timeframes: {', '.join(confirming_tfs) if confirming_tfs else 'None'}
Conflicting Timeframes: {', '.join(conflicting_tfs) if conflicting_tfs else 'None'}
MTF Alignment Score: {mtf_alignment:.2f}
Higher Timeframe Bias: {higher_tf_bias}
"""
            
            prompt += f"""
=== OVERALL MARKET INTELLIGENCE ===
Combined Intelligence Score: {overall_score:.2f} (0 = strongly against signal, 1 = strongly supports signal)
"""
        
        if historical_context:
            prompt += f"""
=== HISTORICAL PERFORMANCE ===
Win Rate on {symbol} {direction}: {historical_context.get('win_rate', 'N/A')}%
Average Profit: {historical_context.get('avg_profit', 'N/A')}%
Total Trades: {historical_context.get('total_trades', 0)}
"""
        
        prompt += """
=== ANALYSIS REQUIREMENTS ===
Consider these key factors:
1. Does the Fear & Greed level support or contradict this trade direction?
2. Does news sentiment align with the trade direction?
3. Do higher timeframes confirm the signal?
4. What is the overall risk assessment based on all factors?
5. Should position size be adjusted based on intelligence alignment?

Provide your comprehensive analysis with confidence score and recommendations."""
        
        return prompt
    
    async def analyze_with_market_context(self, signal_data: Dict[str, Any],
                                           market_intelligence: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze trading signal with comprehensive multi-source market intelligence.
        
        This enhanced analysis method incorporates:
        - Fear & Greed Index (contrarian indicator)
        - News sentiment analysis
        - Market breadth data
        - Multi-timeframe confirmation
        - Overall market intelligence scoring
        
        Args:
            signal_data: Dictionary containing signal information:
                - symbol: Trading pair (e.g., 'ETHUSDT')
                - direction: 'LONG' or 'SHORT'
                - entry_price: Proposed entry price
                - stop_loss: Stop loss level
                - take_profit: Take profit level(s)
                - indicators: Dict of indicator values
                - timeframe: Chart timeframe
                - base_confidence: Initial confidence before market intelligence
                
            market_intelligence: Dictionary containing:
                - fear_greed: Dict with value, classification, trend
                - news_sentiment: Dict with bullish_count, bearish_count, overall_bias, sentiment_score
                - market_breadth: Dict with coins_up, coins_down, breadth_percentage, market_trend
                - mtf_confirmation: Dict with confirming_timeframes, conflicting_timeframes, 
                                    alignment_score, higher_tf_bias
                - overall_intelligence_score: Combined score from all sources
                
        Returns:
            Dictionary with enhanced analysis:
                - confidence: Final adjusted confidence (0.0 to 1.0)
                - recommendation: 'EXECUTE', 'SKIP', or 'MODIFY'
                - analysis: Detailed analysis text
                - risk_assessment: Risk level and factors
                - suggested_adjustments: Position adjustments
                - market_assessment: 'bullish', 'bearish', or 'neutral'
                - intelligence_alignment: How well all sources align (0.0 to 1.0)
                - confidence_adjustment: Adjustment applied (-0.3 to +0.3)
                - reasoning: List of bullet points explaining the analysis
                - position_sizing: Dict with recommended position adjustments
        """
        await self.initialize()
        
        combined_data = {**signal_data, 'market_intelligence': market_intelligence}
        cache_key = self._generate_cache_key("market_context", combined_data)
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        if not self.ai_available:
            result = self._fallback_analyze_with_market_context(signal_data, market_intelligence)
            self._set_cached(cache_key, result)
            return result
        
        try:
            symbol = signal_data.get('symbol', 'UNKNOWN')
            direction = signal_data.get('direction', 'UNKNOWN')
            
            historical_context = await self._get_historical_performance(symbol, direction)
            
            analysis_prompt = self._build_analysis_prompt(
                signal_data, market_intelligence, historical_context
            )
            
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert trading analyst AI with deep understanding of market psychology, 
sentiment analysis, and multi-timeframe technical analysis. Analyze trading signals by considering 
all available market intelligence sources.

IMPORTANT: You must respond with valid JSON containing these exact fields:
- confidence: Final confidence score (0.0-1.0)
- recommendation: One of "EXECUTE", "SKIP", or "MODIFY"
- analysis: Detailed analysis text
- risk_assessment: Object with "level" (low/medium/high) and "factors" (array of risk factors)
- suggested_adjustments: Object with optional "entry", "stop_loss", "take_profit" adjustments
- market_assessment: One of "bullish", "bearish", or "neutral"
- intelligence_alignment: How well all intelligence sources align with the trade (0.0-1.0)
- confidence_adjustment: Adjustment to base confidence based on intelligence (-0.3 to +0.3)
- reasoning: Array of bullet point strings explaining key factors
- position_sizing: Object with:
  - size_multiplier: Position size adjustment (0.5 = half size, 1.0 = normal, 1.5 = 1.5x size)
  - max_leverage_recommended: Maximum recommended leverage
  - rationale: Brief explanation for sizing

Key Analysis Framework:
1. Fear & Greed: Extreme readings are contrarian - extreme fear favors LONG, extreme greed favors SHORT
2. News Sentiment: Aligning sentiment increases confidence, conflicting sentiment decreases it
3. Multi-Timeframe: Higher timeframe alignment is crucial - conflicting higher TFs are warning signs
4. Market Breadth: Strong breadth in trade direction adds confidence
5. Overall: Combine all factors for holistic risk assessment"""
                },
                {
                    "role": "user",
                    "content": analysis_prompt
                }
            ]
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._call_openai, messages)
            
            defaults = {
                'confidence': 0.5,
                'recommendation': 'SKIP',
                'analysis': 'Analysis incomplete',
                'risk_assessment': {'level': 'medium', 'factors': []},
                'suggested_adjustments': {},
                'market_assessment': 'neutral',
                'intelligence_alignment': 0.5,
                'confidence_adjustment': 0.0,
                'reasoning': [],
                'position_sizing': {
                    'size_multiplier': 1.0,
                    'max_leverage_recommended': 10,
                    'rationale': 'Default sizing'
                }
            }
            
            for key, default_value in defaults.items():
                if key not in result:
                    result[key] = default_value
            
            result['confidence_adjustment'] = max(-0.3, min(0.3, result.get('confidence_adjustment', 0.0)))
            result['intelligence_alignment'] = max(0.0, min(1.0, result.get('intelligence_alignment', 0.5)))
            
            result['ai_powered'] = True
            result['model'] = self.MODEL
            result['timestamp'] = datetime.now().isoformat()
            result['market_intelligence_used'] = True
            
            self._set_cached(cache_key, result)
            logger.info(f"Market context analysis for {symbol}: confidence={result['confidence']:.2f}, "
                       f"recommendation={result['recommendation']}, "
                       f"market_assessment={result['market_assessment']}, "
                       f"intelligence_alignment={result['intelligence_alignment']:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"AI market context analysis failed: {e}")
            result = self._fallback_analyze_with_market_context(signal_data, market_intelligence)
            self._set_cached(cache_key, result)
            return result
    
    def _fallback_analyze_with_market_context(self, signal_data: Dict[str, Any],
                                               market_intelligence: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback market context analysis when OpenAI is unavailable.
        Uses rule-based logic to analyze market intelligence factors.
        """
        entry = signal_data.get('entry_price', 0)
        stop_loss = signal_data.get('stop_loss', 0)
        take_profit = signal_data.get('take_profit', 0)
        direction = signal_data.get('direction', 'UNKNOWN')
        base_confidence = signal_data.get('base_confidence', 0.5)
        
        risk_percent = abs(entry - stop_loss) / entry * 100 if entry > 0 else 0
        reward_percent = abs(take_profit - entry) / entry * 100 if entry > 0 else 0
        rr_ratio = reward_percent / risk_percent if risk_percent > 0 else 0
        
        confidence_adjustment = 0.0
        reasoning = []
        alignment_scores = []
        
        fear_greed = market_intelligence.get('fear_greed', {})
        fg_value = fear_greed.get('value', 50)
        
        if direction == 'LONG':
            if fg_value <= 25:
                confidence_adjustment += 0.15
                reasoning.append(f"Extreme Fear ({fg_value}) supports LONG - contrarian buy opportunity")
                alignment_scores.append(1.0)
            elif fg_value <= 40:
                confidence_adjustment += 0.05
                reasoning.append(f"Fear zone ({fg_value}) mildly supports LONG")
                alignment_scores.append(0.7)
            elif fg_value >= 75:
                confidence_adjustment -= 0.15
                reasoning.append(f"Extreme Greed ({fg_value}) contradicts LONG - caution advised")
                alignment_scores.append(0.2)
            elif fg_value >= 60:
                confidence_adjustment -= 0.05
                reasoning.append(f"Greed zone ({fg_value}) mildly contradicts LONG")
                alignment_scores.append(0.4)
            else:
                alignment_scores.append(0.5)
        else:
            if fg_value >= 75:
                confidence_adjustment += 0.15
                reasoning.append(f"Extreme Greed ({fg_value}) supports SHORT - contrarian sell opportunity")
                alignment_scores.append(1.0)
            elif fg_value >= 60:
                confidence_adjustment += 0.05
                reasoning.append(f"Greed zone ({fg_value}) mildly supports SHORT")
                alignment_scores.append(0.7)
            elif fg_value <= 25:
                confidence_adjustment -= 0.15
                reasoning.append(f"Extreme Fear ({fg_value}) contradicts SHORT - caution advised")
                alignment_scores.append(0.2)
            elif fg_value <= 40:
                confidence_adjustment -= 0.05
                reasoning.append(f"Fear zone ({fg_value}) mildly contradicts SHORT")
                alignment_scores.append(0.4)
            else:
                alignment_scores.append(0.5)
        
        news = market_intelligence.get('news_sentiment', {})
        news_score = news.get('sentiment_score', 0.0)
        
        if direction == 'LONG':
            if news_score >= 0.5:
                confidence_adjustment += 0.1
                reasoning.append(f"Strong bullish news sentiment ({news_score:.2f}) aligns with LONG")
                alignment_scores.append(0.9)
            elif news_score >= 0.2:
                confidence_adjustment += 0.05
                reasoning.append(f"Mildly bullish news sentiment supports LONG")
                alignment_scores.append(0.7)
            elif news_score <= -0.5:
                confidence_adjustment -= 0.1
                reasoning.append(f"Strong bearish news sentiment ({news_score:.2f}) contradicts LONG")
                alignment_scores.append(0.2)
            elif news_score <= -0.2:
                confidence_adjustment -= 0.05
                reasoning.append(f"Mildly bearish news sentiment contradicts LONG")
                alignment_scores.append(0.4)
            else:
                alignment_scores.append(0.5)
        else:
            if news_score <= -0.5:
                confidence_adjustment += 0.1
                reasoning.append(f"Strong bearish news sentiment ({news_score:.2f}) aligns with SHORT")
                alignment_scores.append(0.9)
            elif news_score <= -0.2:
                confidence_adjustment += 0.05
                reasoning.append(f"Mildly bearish news sentiment supports SHORT")
                alignment_scores.append(0.7)
            elif news_score >= 0.5:
                confidence_adjustment -= 0.1
                reasoning.append(f"Strong bullish news sentiment ({news_score:.2f}) contradicts SHORT")
                alignment_scores.append(0.2)
            elif news_score >= 0.2:
                confidence_adjustment -= 0.05
                reasoning.append(f"Mildly bullish news sentiment contradicts SHORT")
                alignment_scores.append(0.4)
            else:
                alignment_scores.append(0.5)
        
        mtf = market_intelligence.get('mtf_confirmation', {})
        mtf_alignment = mtf.get('alignment_score', 0.5)
        higher_tf_bias = mtf.get('higher_tf_bias', 'neutral')
        
        if direction == 'LONG' and higher_tf_bias == 'bullish':
            confidence_adjustment += 0.1
            reasoning.append("Higher timeframes confirm bullish bias")
            alignment_scores.append(mtf_alignment)
        elif direction == 'SHORT' and higher_tf_bias == 'bearish':
            confidence_adjustment += 0.1
            reasoning.append("Higher timeframes confirm bearish bias")
            alignment_scores.append(mtf_alignment)
        elif higher_tf_bias != 'neutral' and higher_tf_bias != direction.lower():
            confidence_adjustment -= 0.1
            reasoning.append(f"Warning: Higher timeframes show {higher_tf_bias} bias, conflicts with {direction}")
            alignment_scores.append(1 - mtf_alignment)
        else:
            alignment_scores.append(0.5)
        
        breadth = market_intelligence.get('market_breadth', {})
        breadth_pct = breadth.get('breadth_percentage', 50)
        
        if direction == 'LONG' and breadth_pct >= 65:
            confidence_adjustment += 0.05
            reasoning.append(f"Strong market breadth ({breadth_pct:.0f}% bullish) supports LONG")
            alignment_scores.append(0.8)
        elif direction == 'SHORT' and breadth_pct <= 35:
            confidence_adjustment += 0.05
            reasoning.append(f"Weak market breadth ({breadth_pct:.0f}% bullish) supports SHORT")
            alignment_scores.append(0.8)
        elif direction == 'LONG' and breadth_pct <= 35:
            confidence_adjustment -= 0.05
            reasoning.append(f"Weak market breadth contradicts LONG")
            alignment_scores.append(0.3)
        elif direction == 'SHORT' and breadth_pct >= 65:
            confidence_adjustment -= 0.05
            reasoning.append(f"Strong market breadth contradicts SHORT")
            alignment_scores.append(0.3)
        else:
            alignment_scores.append(0.5)
        
        confidence_adjustment = max(-0.3, min(0.3, confidence_adjustment))
        final_confidence = max(0.0, min(1.0, base_confidence + confidence_adjustment))
        intelligence_alignment = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.5
        
        if final_confidence >= 0.7 and intelligence_alignment >= 0.6:
            recommendation = 'EXECUTE'
            size_multiplier = 1.0 + (intelligence_alignment - 0.5) * 0.5
        elif final_confidence >= 0.5:
            recommendation = 'MODIFY'
            size_multiplier = 0.75
        else:
            recommendation = 'SKIP'
            size_multiplier = 0.5
        
        if confidence_adjustment >= 0.1:
            market_assessment = 'bullish' if direction == 'LONG' else 'bearish'
        elif confidence_adjustment <= -0.1:
            market_assessment = 'bearish' if direction == 'LONG' else 'bullish'
        else:
            market_assessment = 'neutral'
        
        if risk_percent > 3:
            risk_level = 'high'
        elif risk_percent > 1.5:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'confidence': final_confidence,
            'recommendation': recommendation,
            'analysis': f"Fallback market context analysis: Base confidence {base_confidence:.2f} "
                       f"adjusted by {confidence_adjustment:+.2f} based on market intelligence. "
                       f"R/R ratio: {rr_ratio:.2f}",
            'risk_assessment': {
                'level': risk_level,
                'factors': ['risk_percent', 'intelligence_alignment', 'market_conditions']
            },
            'suggested_adjustments': {},
            'market_assessment': market_assessment,
            'intelligence_alignment': intelligence_alignment,
            'confidence_adjustment': confidence_adjustment,
            'reasoning': reasoning,
            'position_sizing': {
                'size_multiplier': round(size_multiplier, 2),
                'max_leverage_recommended': 10 if intelligence_alignment >= 0.6 else 5,
                'rationale': f"Based on intelligence alignment of {intelligence_alignment:.2f}"
            },
            'ai_powered': False,
            'fallback_mode': True,
            'market_intelligence_used': True,
            'timestamp': datetime.now().isoformat()
        }
    
    async def analyze_signal(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze trading signal with AI confidence scoring.
        
        Args:
            signal_data: Dictionary containing signal information:
                - symbol: Trading pair (e.g., 'ETHUSDT')
                - direction: 'LONG' or 'SHORT'
                - entry_price: Proposed entry price
                - stop_loss: Stop loss level
                - take_profit: Take profit level(s)
                - indicators: Dict of indicator values
                - timeframe: Chart timeframe
                
        Returns:
            Dictionary with:
                - confidence: 0.0 to 1.0 confidence score
                - recommendation: 'EXECUTE', 'SKIP', or 'MODIFY'
                - analysis: Detailed analysis text
                - risk_assessment: Risk level and factors
                - suggested_adjustments: Optional position adjustments
        """
        await self.initialize()
        
        cache_key = self._generate_cache_key("signal", signal_data)
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        if not self.ai_available:
            result = self._fallback_analyze_signal(signal_data)
            self._set_cached(cache_key, result)
            return result
        
        try:
            symbol = signal_data.get('symbol', 'UNKNOWN')
            direction = signal_data.get('direction', 'UNKNOWN')
            entry = signal_data.get('entry_price', 0)
            stop_loss = signal_data.get('stop_loss', 0)
            take_profit = signal_data.get('take_profit', 0)
            indicators = signal_data.get('indicators', {})
            
            risk_percent = abs(entry - stop_loss) / entry * 100 if entry > 0 else 0
            reward_percent = abs(take_profit - entry) / entry * 100 if entry > 0 else 0
            rr_ratio = reward_percent / risk_percent if risk_percent > 0 else 0
            
            historical_context = await self._get_historical_performance(symbol, direction)
            
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert trading analyst AI. Analyze trading signals and provide 
                    confidence scores based on technical indicators, risk-reward ratios, and market conditions.
                    Always respond with valid JSON containing: confidence (0.0-1.0), recommendation 
                    (EXECUTE/SKIP/MODIFY), analysis (detailed text), risk_assessment (object with level and factors),
                    and suggested_adjustments (object with optional entry, stop_loss, take_profit adjustments)."""
                },
                {
                    "role": "user",
                    "content": f"""Analyze this trading signal:

Symbol: {symbol}
Direction: {direction}
Entry Price: {entry}
Stop Loss: {stop_loss} (Risk: {risk_percent:.2f}%)
Take Profit: {take_profit} (Reward: {reward_percent:.2f}%)
Risk/Reward Ratio: {rr_ratio:.2f}

Indicators:
{json.dumps(indicators, indent=2)}

Historical Performance on {symbol} {direction} signals:
- Win Rate: {historical_context.get('win_rate', 'N/A')}%
- Average Profit: {historical_context.get('avg_profit', 'N/A')}%
- Total Trades: {historical_context.get('total_trades', 0)}

Provide your analysis with confidence score and recommendation."""
                }
            ]
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._call_openai, messages)
            
            if 'confidence' not in result:
                result['confidence'] = 0.5
            if 'recommendation' not in result:
                result['recommendation'] = 'SKIP'
            if 'analysis' not in result:
                result['analysis'] = 'Analysis incomplete'
            if 'risk_assessment' not in result:
                result['risk_assessment'] = {'level': 'medium', 'factors': []}
            if 'suggested_adjustments' not in result:
                result['suggested_adjustments'] = {}
            
            result['ai_powered'] = True
            result['model'] = self.MODEL
            result['timestamp'] = datetime.now().isoformat()
            
            self._set_cached(cache_key, result)
            logger.info(f"Signal analyzed for {symbol}: confidence={result['confidence']:.2f}, "
                       f"recommendation={result['recommendation']}")
            
            return result
            
        except Exception as e:
            logger.error(f"AI signal analysis failed: {e}")
            result = self._fallback_analyze_signal(signal_data)
            self._set_cached(cache_key, result)
            return result
    
    def _fallback_analyze_signal(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback signal analysis when OpenAI is unavailable."""
        entry = signal_data.get('entry_price', 0)
        stop_loss = signal_data.get('stop_loss', 0)
        take_profit = signal_data.get('take_profit', 0)
        
        risk_percent = abs(entry - stop_loss) / entry * 100 if entry > 0 else 0
        reward_percent = abs(take_profit - entry) / entry * 100 if entry > 0 else 0
        rr_ratio = reward_percent / risk_percent if risk_percent > 0 else 0
        
        confidence = 0.5
        if rr_ratio >= 1.5:
            confidence += 0.15
        if rr_ratio >= 2.0:
            confidence += 0.1
        if risk_percent <= 2.0:
            confidence += 0.1
        if risk_percent > 5.0:
            confidence -= 0.2
        
        indicators = signal_data.get('indicators', {})
        if indicators.get('stc_valid', False):
            confidence += 0.1
        if indicators.get('ut_bot_confirmed', False):
            confidence += 0.1
        
        confidence = max(0.0, min(1.0, confidence))
        
        if confidence >= 0.7:
            recommendation = 'EXECUTE'
        elif confidence >= 0.5:
            recommendation = 'MODIFY'
        else:
            recommendation = 'SKIP'
        
        return {
            'confidence': confidence,
            'recommendation': recommendation,
            'analysis': f"Fallback analysis: R/R ratio {rr_ratio:.2f}, Risk {risk_percent:.2f}%",
            'risk_assessment': {
                'level': 'low' if risk_percent < 2 else 'medium' if risk_percent < 4 else 'high',
                'factors': ['risk_percent', 'rr_ratio']
            },
            'suggested_adjustments': {},
            'ai_powered': False,
            'fallback_mode': True,
            'timestamp': datetime.now().isoformat()
        }
    
    async def learn_from_outcome(self, trade_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Learn from trade outcome to improve future analysis.
        
        Args:
            trade_result: Dictionary containing trade result:
                - trade_id: Unique trade identifier
                - symbol: Trading pair
                - direction: 'LONG' or 'SHORT'
                - entry_price: Actual entry price
                - exit_price: Exit price
                - stop_loss: Stop loss used
                - take_profit: Take profit used
                - outcome: 'WIN', 'LOSS', or 'BREAKEVEN'
                - profit_loss: Profit/loss amount
                - profit_loss_percent: Profit/loss percentage
                - signal_confidence: Original signal confidence
                
        Returns:
            Dictionary with:
                - lessons: List of learned lessons
                - pattern_identified: Any pattern found
                - improvement_suggestions: Suggestions for strategy
                - stored: Whether data was stored successfully
        """
        await self.initialize()
        
        try:
            trade_id = trade_result.get('trade_id', f"trade_{datetime.now().timestamp()}")
            symbol = trade_result.get('symbol', 'UNKNOWN')
            direction = trade_result.get('direction', 'UNKNOWN')
            outcome = trade_result.get('outcome', 'UNKNOWN')
            profit_loss_percent = trade_result.get('profit_loss_percent', 0)
            
            if self.ai_available:
                try:
                    messages = [
                        {
                            "role": "system",
                            "content": """You are an expert trading coach AI. Analyze trade outcomes and 
                            extract lessons for improvement. Always respond with valid JSON containing:
                            lessons (array of strings), pattern_identified (string or null),
                            improvement_suggestions (array of strings), and confidence_adjustment (number -0.2 to 0.2)."""
                        },
                        {
                            "role": "user",
                            "content": f"""Analyze this trade outcome and extract lessons:

Trade ID: {trade_id}
Symbol: {symbol}
Direction: {direction}
Entry: {trade_result.get('entry_price')}
Exit: {trade_result.get('exit_price')}
Stop Loss: {trade_result.get('stop_loss')}
Take Profit: {trade_result.get('take_profit')}
Outcome: {outcome}
P/L: {trade_result.get('profit_loss')} ({profit_loss_percent:.2f}%)
Original Confidence: {trade_result.get('signal_confidence', 'N/A')}

What lessons can be learned from this trade?"""
                        }
                    ]
                    
                    loop = asyncio.get_event_loop()
                    ai_lessons = await loop.run_in_executor(None, self._call_openai, messages)
                    
                except Exception as e:
                    logger.warning(f"AI learning analysis failed: {e}")
                    ai_lessons = self._fallback_learn_from_outcome(trade_result)
            else:
                ai_lessons = self._fallback_learn_from_outcome(trade_result)
            
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO trade_outcomes 
                    (trade_id, symbol, direction, entry_price, exit_price, stop_loss, 
                     take_profit, outcome, profit_loss, profit_loss_percent, 
                     signal_confidence, ai_analysis, lessons_learned, closed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_id, symbol, direction,
                    trade_result.get('entry_price'),
                    trade_result.get('exit_price'),
                    trade_result.get('stop_loss'),
                    trade_result.get('take_profit'),
                    outcome,
                    trade_result.get('profit_loss'),
                    profit_loss_percent,
                    trade_result.get('signal_confidence'),
                    json.dumps(ai_lessons),
                    json.dumps(ai_lessons.get('lessons', [])),
                    datetime.now().isoformat()
                ))
                await db.commit()
            
            logger.info(f"Learned from trade {trade_id}: {outcome} ({profit_loss_percent:.2f}%)")
            
            result = {
                'lessons': ai_lessons.get('lessons', []),
                'pattern_identified': ai_lessons.get('pattern_identified'),
                'improvement_suggestions': ai_lessons.get('improvement_suggestions', []),
                'confidence_adjustment': ai_lessons.get('confidence_adjustment', 0),
                'stored': True,
                'ai_powered': self.ai_available,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to learn from outcome: {e}")
            return {
                'lessons': [],
                'pattern_identified': None,
                'improvement_suggestions': [],
                'stored': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _fallback_learn_from_outcome(self, trade_result: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback learning when OpenAI is unavailable."""
        outcome = trade_result.get('outcome', 'UNKNOWN')
        profit_loss_percent = trade_result.get('profit_loss_percent', 0)
        direction = trade_result.get('direction', 'UNKNOWN')
        
        lessons = []
        suggestions = []
        
        if outcome == 'WIN':
            lessons.append(f"Successful {direction} trade - strategy confirmed")
            if profit_loss_percent > 3:
                lessons.append("Consider extending take profit for larger moves")
        elif outcome == 'LOSS':
            lessons.append(f"Failed {direction} trade - review entry criteria")
            if abs(profit_loss_percent) > 2:
                suggestions.append("Consider tighter stop loss placement")
            suggestions.append("Review indicator confluence before entry")
        
        return {
            'lessons': lessons,
            'pattern_identified': None,
            'improvement_suggestions': suggestions,
            'confidence_adjustment': 0.05 if outcome == 'WIN' else -0.05
        }
    
    async def get_market_insight(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get dynamic market analysis and insights.
        
        Args:
            market_data: Dictionary containing market information:
                - symbol: Trading pair
                - current_price: Current price
                - price_change_24h: 24h price change percentage
                - volume_24h: 24h trading volume
                - high_24h: 24h high
                - low_24h: 24h low
                - indicators: Current indicator values
                - timeframe: Analysis timeframe
                
        Returns:
            Dictionary with:
                - market_regime: Current market regime (trending/ranging/volatile)
                - bias: Market bias (bullish/bearish/neutral)
                - key_levels: Important support/resistance levels
                - opportunities: Potential trading opportunities
                - risks: Current market risks
                - confidence: Insight confidence score
        """
        await self.initialize()
        
        cache_key = self._generate_cache_key("insight", market_data)
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        if not self.ai_available:
            result = self._fallback_market_insight(market_data)
            self._set_cached(cache_key, result)
            return result
        
        try:
            symbol = market_data.get('symbol', 'UNKNOWN')
            current_price = market_data.get('current_price', 0)
            price_change = market_data.get('price_change_24h', 0)
            volume = market_data.get('volume_24h', 0)
            high_24h = market_data.get('high_24h', 0)
            low_24h = market_data.get('low_24h', 0)
            indicators = market_data.get('indicators', {})
            
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert market analyst AI. Analyze market conditions and 
                    provide actionable insights. Always respond with valid JSON containing:
                    market_regime (trending/ranging/volatile), bias (bullish/bearish/neutral),
                    key_levels (object with support and resistance arrays), opportunities (array of strings),
                    risks (array of strings), and confidence (0.0-1.0)."""
                },
                {
                    "role": "user",
                    "content": f"""Analyze the current market conditions for {symbol}:

Current Price: {current_price}
24h Change: {price_change:.2f}%
24h Volume: {volume}
24h High: {high_24h}
24h Low: {low_24h}
Price Range: {((high_24h - low_24h) / low_24h * 100) if low_24h > 0 else 0:.2f}%

Technical Indicators:
{json.dumps(indicators, indent=2)}

Provide comprehensive market analysis with regime identification, bias, key levels, 
opportunities, and risks."""
                }
            ]
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._call_openai, messages)
            
            if 'market_regime' not in result:
                result['market_regime'] = 'unknown'
            if 'bias' not in result:
                result['bias'] = 'neutral'
            if 'key_levels' not in result:
                result['key_levels'] = {'support': [], 'resistance': []}
            if 'opportunities' not in result:
                result['opportunities'] = []
            if 'risks' not in result:
                result['risks'] = []
            if 'confidence' not in result:
                result['confidence'] = 0.5
            
            result['ai_powered'] = True
            result['model'] = self.MODEL
            result['symbol'] = symbol
            result['timestamp'] = datetime.now().isoformat()
            
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO market_insights 
                    (symbol, timeframe, insight_type, insight_data, confidence)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    symbol,
                    market_data.get('timeframe', '1h'),
                    'full_analysis',
                    json.dumps(result),
                    result['confidence']
                ))
                await db.commit()
            
            self._set_cached(cache_key, result)
            logger.info(f"Market insight for {symbol}: {result['market_regime']}, "
                       f"bias={result['bias']}, confidence={result['confidence']:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"AI market insight failed: {e}")
            result = self._fallback_market_insight(market_data)
            self._set_cached(cache_key, result)
            return result
    
    def _fallback_market_insight(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback market insight when OpenAI is unavailable."""
        price_change = market_data.get('price_change_24h', 0)
        current_price = market_data.get('current_price', 0)
        high_24h = market_data.get('high_24h', 0)
        low_24h = market_data.get('low_24h', 0)
        
        price_range_percent = ((high_24h - low_24h) / low_24h * 100) if low_24h > 0 else 0
        
        if price_range_percent > 5:
            market_regime = 'volatile'
        elif abs(price_change) > 2:
            market_regime = 'trending'
        else:
            market_regime = 'ranging'
        
        if price_change > 1:
            bias = 'bullish'
        elif price_change < -1:
            bias = 'bearish'
        else:
            bias = 'neutral'
        
        support = low_24h if low_24h > 0 else current_price * 0.98
        resistance = high_24h if high_24h > 0 else current_price * 1.02
        
        return {
            'market_regime': market_regime,
            'bias': bias,
            'key_levels': {
                'support': [round(support, 6)],
                'resistance': [round(resistance, 6)]
            },
            'opportunities': [f"Watch for breakout above {resistance:.6f}"] if bias == 'bullish' else 
                            [f"Watch for breakdown below {support:.6f}"] if bias == 'bearish' else
                            ["Range trading opportunity"],
            'risks': ['High volatility'] if market_regime == 'volatile' else ['Low momentum'],
            'confidence': 0.6,
            'ai_powered': False,
            'fallback_mode': True,
            'symbol': market_data.get('symbol', 'UNKNOWN'),
            'timestamp': datetime.now().isoformat()
        }
    
    async def optimize_parameters(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest parameter optimizations based on performance.
        
        Args:
            performance_data: Dictionary containing performance metrics:
                - current_parameters: Dict of current parameter values
                - win_rate: Overall win rate percentage
                - profit_factor: Profit factor
                - avg_win: Average winning trade percentage
                - avg_loss: Average losing trade percentage
                - max_drawdown: Maximum drawdown percentage
                - total_trades: Total number of trades
                - recent_trades: List of recent trade results
                
        Returns:
            Dictionary with:
                - suggested_parameters: Dict of parameter suggestions
                - expected_improvement: Expected improvement percentage
                - rationale: Explanation for each suggestion
                - priority: Priority order for implementing changes
                - confidence: Suggestion confidence score
        """
        await self.initialize()
        
        cache_key = self._generate_cache_key("optimize", performance_data)
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        if not self.ai_available:
            result = self._fallback_optimize_parameters(performance_data)
            self._set_cached(cache_key, result)
            return result
        
        try:
            current_params = performance_data.get('current_parameters', {})
            win_rate = performance_data.get('win_rate', 50)
            profit_factor = performance_data.get('profit_factor', 1.0)
            avg_win = performance_data.get('avg_win', 0)
            avg_loss = performance_data.get('avg_loss', 0)
            max_drawdown = performance_data.get('max_drawdown', 0)
            total_trades = performance_data.get('total_trades', 0)
            
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert trading system optimizer AI. Analyze trading 
                    performance and suggest parameter optimizations. Always respond with valid JSON containing:
                    suggested_parameters (object with parameter names as keys and new values),
                    expected_improvement (number as percentage), rationale (object with parameter names 
                    and explanation strings), priority (array of parameter names in order),
                    and confidence (0.0-1.0)."""
                },
                {
                    "role": "user",
                    "content": f"""Analyze this trading performance and suggest optimizations:

Current Parameters:
{json.dumps(current_params, indent=2)}

Performance Metrics:
- Win Rate: {win_rate:.1f}%
- Profit Factor: {profit_factor:.2f}
- Average Win: {avg_win:.2f}%
- Average Loss: {avg_loss:.2f}%
- Max Drawdown: {max_drawdown:.2f}%
- Total Trades: {total_trades}

Suggest parameter adjustments to improve performance, considering:
1. Risk management (stop loss, position sizing)
2. Entry criteria (indicator thresholds)
3. Exit strategy (take profit levels, trailing stops)
4. Overall system robustness"""
                }
            ]
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._call_openai, messages)
            
            if 'suggested_parameters' not in result:
                result['suggested_parameters'] = {}
            if 'expected_improvement' not in result:
                result['expected_improvement'] = 0
            if 'rationale' not in result:
                result['rationale'] = {}
            if 'priority' not in result:
                result['priority'] = []
            if 'confidence' not in result:
                result['confidence'] = 0.5
            
            result['ai_powered'] = True
            result['model'] = self.MODEL
            result['timestamp'] = datetime.now().isoformat()
            
            async with aiosqlite.connect(self.db_path) as db:
                for param_name, suggested_value in result['suggested_parameters'].items():
                    current_value = current_params.get(param_name)
                    await db.execute("""
                        INSERT INTO parameter_suggestions 
                        (parameter_name, current_value, suggested_value, reason, performance_impact)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        param_name,
                        json.dumps(current_value),
                        json.dumps(suggested_value),
                        result['rationale'].get(param_name, ''),
                        result['expected_improvement']
                    ))
                await db.commit()
            
            self._set_cached(cache_key, result)
            logger.info(f"Parameter optimization suggestions: {len(result['suggested_parameters'])} changes, "
                       f"expected improvement: {result['expected_improvement']:.1f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"AI parameter optimization failed: {e}")
            result = self._fallback_optimize_parameters(performance_data)
            self._set_cached(cache_key, result)
            return result
    
    def _fallback_optimize_parameters(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback parameter optimization when OpenAI is unavailable."""
        win_rate = performance_data.get('win_rate', 50)
        profit_factor = performance_data.get('profit_factor', 1.0)
        max_drawdown = performance_data.get('max_drawdown', 0)
        current_params = performance_data.get('current_parameters', {})
        
        suggestions = {}
        rationale = {}
        priority = []
        
        if win_rate < 50:
            if 'ut_key_value' in current_params:
                suggestions['ut_key_value'] = current_params['ut_key_value'] * 1.1
                rationale['ut_key_value'] = "Increase sensitivity to reduce false signals"
                priority.append('ut_key_value')
        
        if max_drawdown > 20:
            if 'max_risk_percent' in current_params:
                suggestions['max_risk_percent'] = max(1.0, current_params['max_risk_percent'] * 0.8)
                rationale['max_risk_percent'] = "Reduce risk to limit drawdown"
                priority.append('max_risk_percent')
        
        if profit_factor < 1.5:
            if 'risk_reward_ratio' in current_params:
                suggestions['risk_reward_ratio'] = min(3.0, current_params['risk_reward_ratio'] * 1.2)
                rationale['risk_reward_ratio'] = "Increase R:R to improve profit factor"
                priority.append('risk_reward_ratio')
        
        expected_improvement = len(suggestions) * 5.0
        
        return {
            'suggested_parameters': suggestions,
            'expected_improvement': expected_improvement,
            'rationale': rationale,
            'priority': priority,
            'confidence': 0.5,
            'ai_powered': False,
            'fallback_mode': True,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _get_historical_performance(self, symbol: str, direction: str) -> Dict[str, Any]:
        """Get historical performance for a symbol and direction."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("""
                    SELECT 
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                        AVG(profit_loss_percent) as avg_profit
                    FROM trade_outcomes 
                    WHERE symbol = ? AND direction = ?
                """, (symbol, direction))
                
                row = await cursor.fetchone()
                
                if row and row[0] > 0:
                    total_trades = row[0]
                    wins = row[1] or 0
                    avg_profit = row[2] or 0
                    
                    return {
                        'total_trades': total_trades,
                        'win_rate': (wins / total_trades * 100) if total_trades > 0 else 0,
                        'avg_profit': avg_profit
                    }
                    
        except Exception as e:
            logger.debug(f"Could not fetch historical performance: {e}")
        
        return {
            'total_trades': 0,
            'win_rate': None,
            'avg_profit': None
        }
    
    async def get_learning_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all learned trading insights.
        
        Returns:
            Dictionary with learning statistics and insights
        """
        await self.initialize()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("""
                    SELECT 
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
                        AVG(profit_loss_percent) as avg_profit,
                        SUM(profit_loss) as total_profit
                    FROM trade_outcomes
                """)
                trade_stats = await cursor.fetchone()
                
                cursor = await db.execute("""
                    SELECT COUNT(*) as insight_count 
                    FROM market_insights
                """)
                insight_row = await cursor.fetchone()
                insight_count = insight_row[0] if insight_row else 0
                
                cursor = await db.execute("""
                    SELECT COUNT(*) as suggestion_count 
                    FROM parameter_suggestions WHERE applied = FALSE
                """)
                suggestion_row = await cursor.fetchone()
                pending_suggestions = suggestion_row[0] if suggestion_row else 0
                
                cursor = await db.execute("""
                    SELECT symbol, direction, COUNT(*) as count,
                           AVG(CASE WHEN outcome = 'WIN' THEN 1.0 ELSE 0.0 END) as win_rate
                    FROM trade_outcomes
                    GROUP BY symbol, direction
                    ORDER BY count DESC
                    LIMIT 10
                """)
                symbol_performance = await cursor.fetchall()
            
            total_trades = trade_stats[0] if trade_stats else 0
            wins = trade_stats[1] if trade_stats else 0
            losses = trade_stats[2] if trade_stats else 0
            avg_profit = trade_stats[3] if trade_stats else 0
            total_profit = trade_stats[4] if trade_stats else 0
            
            return {
                'total_trades_learned': total_trades or 0,
                'wins': wins or 0,
                'losses': losses or 0,
                'overall_win_rate': ((wins or 0) / total_trades * 100) if total_trades else 0,
                'average_profit_percent': avg_profit or 0,
                'total_profit': total_profit or 0,
                'market_insights_generated': insight_count,
                'pending_parameter_suggestions': pending_suggestions,
                'symbol_performance': [
                    {
                        'symbol': row[0],
                        'direction': row[1],
                        'trade_count': row[2],
                        'win_rate': row[3] * 100
                    }
                    for row in symbol_performance
                ],
                'ai_available': self.ai_available,
                'model': self.MODEL if self.ai_available else None,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get learning summary: {e}")
            return {
                'error': str(e),
                'ai_available': self.ai_available,
                'timestamp': datetime.now().isoformat()
            }
    
    def clear_cache(self) -> int:
        """Clear the analysis cache. Returns number of items cleared."""
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cleared {count} cached items")
        return count
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the AI Trading Brain."""
        return {
            'ai_available': self.ai_available,
            'model': self.MODEL,
            'openai_key_configured': bool(OPENAI_API_KEY),
            'openai_package_installed': OPENAI_AVAILABLE,
            'database_path': self.db_path,
            'cache_size': len(self._cache),
            'cache_ttl_seconds': self.cache_ttl,
            'initialized': self._initialized,
            'timestamp': datetime.now().isoformat()
        }
