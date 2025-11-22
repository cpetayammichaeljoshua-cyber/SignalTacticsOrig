#!/usr/bin/env python3
"""
Market Intelligence Engine
Orchestrates all analyzers and produces unified market intelligence
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from SignalMaestro.market_data_contracts import (
    MarketSnapshot, AnalysisResult, MarketIntelSnapshot,
    AnalyzerType, MarketBias, SignalStrength
)
from SignalMaestro.async_market_data_fetcher import AsyncMarketDataFetcher
from SignalMaestro.advanced_liquidity_analyzer import AdvancedLiquidityAnalyzer
from SignalMaestro.advanced_order_flow_analyzer import AdvancedOrderFlowAnalyzer
from SignalMaestro.volume_profile_analyzer import VolumeProfileAnalyzer
from SignalMaestro.fractals_analyzer import FractalsAnalyzer
from SignalMaestro.intermarket_analyzer import IntermarketAnalyzer

class MarketIntelligenceEngine:
    """
    Central orchestrator for all market analysis
    
    Responsibilities:
    - Fetch market data efficiently
    - Execute all analyzers in parallel
    - Produce unified intelligence snapshot
    - Maintain analyzer registry
    """
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Data fetcher
        self.data_fetcher = AsyncMarketDataFetcher(api_key, api_secret)
        
        # Register all analyzers
        self.analyzers = {
            AnalyzerType.LIQUIDITY: AdvancedLiquidityAnalyzer(),
            AnalyzerType.ORDER_FLOW: AdvancedOrderFlowAnalyzer(),
            AnalyzerType.VOLUME_PROFILE: VolumeProfileAnalyzer(),
            AnalyzerType.FRACTALS: FractalsAnalyzer(),
            AnalyzerType.INTERMARKET: IntermarketAnalyzer()
        }
        
        # Configuration
        self.enabled_analyzers = set(self.analyzers.keys())
        self.analyzer_weights = {
            AnalyzerType.LIQUIDITY: 1.2,  # Slightly higher weight
            AnalyzerType.ORDER_FLOW: 1.2,  # Slightly higher weight
            AnalyzerType.VOLUME_PROFILE: 1.0,
            AnalyzerType.FRACTALS: 1.0,
            AnalyzerType.INTERMARKET: 0.8  # Slightly lower weight
        }
        
        # Cache last snapshot
        self.last_snapshot: Optional[MarketIntelSnapshot] = None
        self.last_snapshot_time: Optional[datetime] = None
        
    async def analyze_market(self, symbol: str,
                            timeframe: str = '30m',
                            limit: int = 500,
                            correlated_symbols: Optional[List[str]] = None) -> MarketIntelSnapshot:
        """
        Complete market analysis
        
        Args:
            symbol: Trading symbol
            timeframe: Candlestick timeframe
            limit: Number of candles
            correlated_symbols: Symbols for correlation analysis
            
        Returns:
            MarketIntelSnapshot with complete intelligence
        """
        start_time = datetime.now()
        
        self.logger.info(f"🔬 Starting market intelligence analysis for {symbol}")
        
        try:
            # Step 1: Fetch market data
            market_snapshot = await self.data_fetcher.fetch_market_snapshot(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                include_orderbook=True,
                include_trades=True,
                correlated_symbols=correlated_symbols or ['BTCUSDT', 'ETHUSDT']
            )
            
            self.logger.info(f"✅ Market data fetched for {symbol}")
            
            # Step 2: Run all analyzers in parallel
            analyzer_results = await self._run_analyzers(market_snapshot)
            
            self.logger.info(f"✅ All analyzers completed ({len(analyzer_results)} active)")
            
            # Step 3: Produce unified intelligence
            intel_snapshot = self._produce_intelligence(
                symbol, market_snapshot, analyzer_results
            )
            
            # Cache result
            self.last_snapshot = intel_snapshot
            self.last_snapshot_time = datetime.now()
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            intel_snapshot.total_processing_time_ms = processing_time
            
            self.logger.info(f"✅ Intelligence snapshot produced in {processing_time:.0f}ms")
            self.logger.info(f"   Consensus: {intel_snapshot.consensus_bias.value} ({intel_snapshot.consensus_confidence:.1f}%)")
            self.logger.info(f"   Overall Score: {intel_snapshot.overall_score:.1f}/100")
            
            return intel_snapshot
            
        except Exception as e:
            self.logger.error(f"❌ Error in market analysis: {e}", exc_info=True)
            
            # Return empty snapshot on error
            return MarketIntelSnapshot(
                symbol=symbol,
                timestamp=datetime.now(),
                consensus_bias=MarketBias.NEUTRAL,
                consensus_confidence=0,
                overall_score=0,
                total_veto_count=1,
                veto_reasons=["Analysis failed"],
                analyzers_failed=len(self.enabled_analyzers)
            )
    
    async def _run_analyzers(self, market_snapshot: MarketSnapshot) -> Dict[AnalyzerType, AnalysisResult]:
        """Run all enabled analyzers in parallel"""
        tasks = []
        analyzer_types = []
        
        for analyzer_type in self.enabled_analyzers:
            analyzer = self.analyzers[analyzer_type]
            tasks.append(self._run_analyzer_safe(analyzer, market_snapshot, analyzer_type))
            analyzer_types.append(analyzer_type)
        
        # Execute all analyzers concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect successful results
        analyzer_results = {}
        for analyzer_type, result in zip(analyzer_types, results):
            if isinstance(result, AnalysisResult):
                analyzer_results[analyzer_type] = result
            else:
                self.logger.warning(f"Analyzer {analyzer_type.value} failed: {result}")
        
        return analyzer_results
    
    async def _run_analyzer_safe(self, analyzer: Any, 
                                 market_snapshot: MarketSnapshot,
                                 analyzer_type: AnalyzerType) -> AnalysisResult:
        """Run analyzer with error handling"""
        try:
            # Some analyzers might not be async
            result = analyzer.analyze(market_snapshot)
            
            # If it's a coroutine, await it
            if asyncio.iscoroutine(result):
                result = await result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error running {analyzer_type.value} analyzer: {e}")
            
            # Return empty result
            return AnalysisResult(
                analyzer_type=analyzer_type,
                timestamp=datetime.now(),
                score=0,
                bias=MarketBias.NEUTRAL,
                confidence=0,
                veto_flags=[f"Analyzer error: {str(e)[:100]}"]
            )
    
    def _produce_intelligence(self, symbol: str,
                             market_snapshot: MarketSnapshot,
                             analyzer_results: Dict[AnalyzerType, AnalysisResult]) -> MarketIntelSnapshot:
        """
        Produce unified market intelligence from all analyzer results
        """
        # Count active and failed analyzers
        analyzers_active = len(analyzer_results)
        analyzers_failed = len(self.enabled_analyzers) - analyzers_active
        
        # Collect all veto flags
        all_veto_flags = []
        for result in analyzer_results.values():
            all_veto_flags.extend(result.veto_flags)
        
        # Calculate consensus bias
        consensus_bias, consensus_confidence = self._calculate_consensus_bias(analyzer_results)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(analyzer_results)
        
        # Aggregate signals
        dominant_signals = self._aggregate_signals(analyzer_results)
        
        # Aggregate key levels
        critical_levels = self._aggregate_key_levels(analyzer_results)
        
        # Determine risk level
        risk_level = self._assess_risk_level(
            len(all_veto_flags), 
            overall_score, 
            consensus_confidence
        )
        
        # Generate trading recommendations
        recommendations = self._generate_recommendations(
            market_snapshot, analyzer_results, consensus_bias, overall_score
        )
        
        return MarketIntelSnapshot(
            symbol=symbol,
            timestamp=market_snapshot.timestamp,
            analyzer_results=analyzer_results,
            consensus_bias=consensus_bias,
            consensus_confidence=consensus_confidence,
            overall_score=overall_score,
            dominant_signals=dominant_signals,
            critical_levels=critical_levels,
            total_veto_count=len(all_veto_flags),
            veto_reasons=all_veto_flags,
            risk_level=risk_level,
            recommended_entry=recommendations.get('entry'),
            recommended_stop=recommendations.get('stop'),
            recommended_targets=recommendations.get('targets'),
            recommended_leverage=recommendations.get('leverage'),
            analyzers_active=analyzers_active,
            analyzers_failed=analyzers_failed
        )
    
    def _calculate_consensus_bias(self, analyzer_results: Dict[AnalyzerType, AnalysisResult]) -> tuple:
        """Calculate consensus bias from all analyzers"""
        if not analyzer_results:
            return MarketBias.NEUTRAL, 0
        
        bullish_weight = 0
        bearish_weight = 0
        total_weight = 0
        
        for analyzer_type, result in analyzer_results.items():
            weight = self.analyzer_weights.get(analyzer_type, 1.0)
            confidence_factor = result.confidence / 100
            
            if result.bias == MarketBias.BULLISH:
                bullish_weight += weight * confidence_factor
            elif result.bias == MarketBias.BEARISH:
                bearish_weight += weight * confidence_factor
            
            total_weight += weight
        
        if total_weight == 0:
            return MarketBias.NEUTRAL, 0
        
        bullish_score = (bullish_weight / total_weight) * 100
        bearish_score = (bearish_weight / total_weight) * 100
        
        if bullish_score > bearish_score * 1.5:
            consensus_bias = MarketBias.BULLISH
            consensus_confidence = bullish_score
        elif bearish_score > bullish_score * 1.5:
            consensus_bias = MarketBias.BEARISH
            consensus_confidence = bearish_score
        else:
            consensus_bias = MarketBias.NEUTRAL
            consensus_confidence = 50
        
        return consensus_bias, min(consensus_confidence, 100)
    
    def _calculate_overall_score(self, analyzer_results: Dict[AnalyzerType, AnalysisResult]) -> float:
        """Calculate weighted overall score"""
        if not analyzer_results:
            return 0
        
        weighted_score = 0
        total_weight = 0
        
        for analyzer_type, result in analyzer_results.items():
            weight = self.analyzer_weights.get(analyzer_type, 1.0)
            weighted_score += result.score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0
    
    def _aggregate_signals(self, analyzer_results: Dict[AnalyzerType, AnalysisResult]) -> List[Dict]:
        """Aggregate most important signals from all analyzers"""
        all_signals = []
        
        for analyzer_type, result in analyzer_results.items():
            for signal in result.signals[:2]:  # Top 2 signals from each
                all_signals.append({
                    'analyzer': analyzer_type.value,
                    'confidence': result.confidence,
                    **signal
                })
        
        # Sort by confidence
        all_signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return all_signals[:10]  # Top 10 overall
    
    def _aggregate_key_levels(self, analyzer_results: Dict[AnalyzerType, AnalysisResult]) -> List[Dict]:
        """Aggregate critical price levels from all analyzers"""
        all_levels = []
        
        for analyzer_type, result in analyzer_results.items():
            for level in result.key_levels:
                all_levels.append({
                    'source': analyzer_type.value,
                    **level
                })
        
        # Sort by strength/importance
        all_levels.sort(key=lambda x: x.get('strength', 0), reverse=True)
        
        return all_levels[:15]  # Top 15 levels
    
    def _assess_risk_level(self, veto_count: int, overall_score: float, 
                          consensus_confidence: float) -> str:
        """Assess overall risk level"""
        if veto_count >= 3 or overall_score < 40:
            return "extreme"
        elif veto_count >= 2 or overall_score < 55:
            return "high"
        elif veto_count >= 1 or consensus_confidence < 60:
            return "moderate"
        else:
            return "low"
    
    def _generate_recommendations(self, market_snapshot: MarketSnapshot,
                                 analyzer_results: Dict[AnalyzerType, AnalysisResult],
                                 consensus_bias: MarketBias,
                                 overall_score: float) -> Dict:
        """Generate trading recommendations"""
        current_price = market_snapshot.current_price
        
        # Collect suggestions from analyzers
        entries = []
        stops = []
        targets = []
        
        for result in analyzer_results.values():
            if result.suggested_entry:
                entries.append(result.suggested_entry)
            if result.suggested_stop:
                stops.append(result.suggested_stop)
            if result.suggested_targets:
                targets.extend(result.suggested_targets)
        
        # Calculate recommendations
        recommended_entry = sum(entries) / len(entries) if entries else current_price
        recommended_stop = sum(stops) / len(stops) if stops else None
        recommended_targets = sorted(set(targets))[:3] if targets else None
        
        # Calculate leverage based on score and confidence
        if overall_score >= 80:
            leverage = 10
        elif overall_score >= 70:
            leverage = 7
        elif overall_score >= 60:
            leverage = 5
        else:
            leverage = 3
        
        return {
            'entry': recommended_entry,
            'stop': recommended_stop,
            'targets': recommended_targets,
            'leverage': leverage
        }
    
    def enable_analyzer(self, analyzer_type: AnalyzerType):
        """Enable a specific analyzer"""
        self.enabled_analyzers.add(analyzer_type)
    
    def disable_analyzer(self, analyzer_type: AnalyzerType):
        """Disable a specific analyzer"""
        self.enabled_analyzers.discard(analyzer_type)
    
    def set_analyzer_weight(self, analyzer_type: AnalyzerType, weight: float):
        """Set weight for an analyzer"""
        self.analyzer_weights[analyzer_type] = weight
    
    def get_status(self) -> Dict:
        """Get engine status"""
        return {
            'enabled_analyzers': [a.value for a in self.enabled_analyzers],
            'analyzer_weights': {k.value: v for k, v in self.analyzer_weights.items()},
            'last_analysis': self.last_snapshot_time.isoformat() if self.last_snapshot_time else None,
            'cache_size': len(self.data_fetcher.cache)
        }
