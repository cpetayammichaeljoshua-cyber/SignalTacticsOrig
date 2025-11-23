#!/usr/bin/env python3
"""
High-Frequency Scalping Orchestrator - Ultimate Multi-Strategy Integration
Combines ALL scalping strategies for maximum trading opportunities
- 5-10 second scan intervals
- Multi-timeframe analysis (1m, 3m, 5m)
- Parallel strategy execution for speed
- Weighted signal fusion from 6+ strategies
- Dynamic position management with tight risk controls
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import aiohttp

@dataclass
class HighFrequencySignal:
    """Unified high-frequency scalping signal"""
    symbol: str
    direction: str  # LONG or SHORT
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float

    # Strategy consensus
    signal_strength: float  # 0-100
    strategies_agree: int  # Number of strategies agreeing
    total_strategies: int  # Total strategies analyzed
    consensus_confidence: float  # Percentage of agreement

    # Position parameters
    leverage: int
    position_size_usdt: float
    risk_reward_ratio: float

    # Timing
    timeframe: str
    expected_duration_seconds: int
    signal_latency_ms: float
    timestamp: datetime

    # Strategy breakdown
    strategy_votes: Dict[str, str]  # Strategy name -> direction
    strategy_scores: Dict[str, float]  # Strategy name -> strength

    # Market context
    market_volatility: str  # low, normal, high
    momentum_phase: str  # building, peak, declining
    volume_profile: str  # low, normal, high


class HighFrequencyScalpingOrchestrator:
    """
    Orchestrates multiple scalping strategies for high-frequency trading
    Combines signals from all strategies with intelligent weighting
    """

    def __init__(self, telegram_notifier=None, position_closer=None, atas_integration=None):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # High-frequency parameters
        self.scan_interval = 5  # 5 seconds between scans
        self.fast_timeframes = ['1m', '3m', '5m', '30m']  # Scalping timeframes + Ichimoku
        self.max_concurrent_positions = 5

        # Strategy weights (based on historical performance)
        self.strategy_weights = {
            'ultimate_scalping': 0.22,      # Most comprehensive
            'lightning_scalping': 0.20,     # Fastest execution
            'momentum_scalping': 0.18,      # RSI/MACD specialist
            'volume_breakout': 0.15,        # Volume specialist
            'ichimoku_sniper': 0.15,        # Trend specialist
            'market_intelligence': 0.10     # Market context
        }

        # Consensus requirements (PRODUCTION-GRADE FOR STABLE SIGNALS)
        self.min_strategies_agree = 3  # At least 3 strategies must agree for valid signal
        self.min_consensus_confidence = 50.0  # 50% agreement minimum (3+ out of 6 strategies)
        self.min_signal_strength = 65.0  # Minimum weighted strength for production reliability

        # Risk management for high-frequency
        self.max_risk_per_trade = 1.0  # 1% per trade
        self.max_total_exposure = 5.0  # 5% total exposure
        self.tight_stop_loss_pct = 0.5  # 0.5% stop loss for scalping
        self.quick_profit_targets = [0.8, 1.2, 1.8]  # % targets

        # Strategy instances (will be loaded dynamically)
        self.strategies = {}
        self.strategy_performance = {}  # Track win rates

        # Position tracking
        self.active_positions = {}
        self.last_signal_time = {}
        self.signals_generated = 0
        self.signals_executed = 0

        # Integrations
        self.telegram_notifier = telegram_notifier
        self.position_closer = position_closer
        self.atas_integration = atas_integration

    async def initialize_strategies(self):
        """Load and initialize all scalping strategies"""
        try:
            self.logger.info("🔄 Loading all scalping strategies...")

            # Import all strategies
            from SignalMaestro.ultimate_scalping_strategy import UltimateScalpingStrategy
            from SignalMaestro.lightning_scalping_strategy import LightningScalpingStrategy
            from SignalMaestro.momentum_scalping_strategy import MomentumScalpingStrategy
            from SignalMaestro.volume_breakout_scalping_strategy import VolumeBreakoutScalpingStrategy
            from SignalMaestro.ichimoku_sniper_strategy import IchimokuSniperStrategy
            from SignalMaestro.market_intelligence_engine import MarketIntelligenceEngine

            # Initialize strategies
            self.strategies = {
                'ultimate_scalping': UltimateScalpingStrategy(),
                'lightning_scalping': LightningScalpingStrategy(),
                'momentum_scalping': MomentumScalpingStrategy(),
                'volume_breakout': VolumeBreakoutScalpingStrategy(),
                'ichimoku_sniper': IchimokuSniperStrategy(),
                'market_intelligence': MarketIntelligenceEngine()
            }

            self.logger.info(f"✅ Loaded {len(self.strategies)} scalping strategies")
            for strategy_name in self.strategies.keys():
                self.logger.info(f"   ✓ {strategy_name}")

            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to load strategies: {e}")
            return False

    async def fetch_fast_ohlcv(self, exchange, symbol: str) -> Dict[str, List]:
        """Fetch OHLCV data for fast timeframes with error handling"""
        ohlcv_data = {}

        try:
            # Fetch data for each fast timeframe
            for timeframe in self.fast_timeframes:
                try:
                    # Fetch last 100 candles for analysis
                    candles = exchange.fetch_ohlcv(
                        symbol,
                        timeframe=timeframe,
                        limit=100
                    )
                    
                    # Validate candle data
                    if candles and len(candles) > 0:
                        ohlcv_data[timeframe] = candles
                    else:
                        self.logger.debug(f"No data received for {timeframe} {symbol}")
                        
                except Exception as e:
                    self.logger.debug(f"Failed to fetch {timeframe} data for {symbol}: {str(e)[:100]}")
                    # Continue with other timeframes if one fails
                    continue

            # Return data if we got at least one timeframe
            if ohlcv_data:
                return ohlcv_data
            else:
                self.logger.warning(f"⚠️ No OHLCV data available for {symbol}")
                return {}

        except Exception as e:
            self.logger.error(f"❌ Error fetching OHLCV for {symbol}: {str(e)[:100]}")
            return {}

    async def analyze_symbol_parallel(self, symbol: str, ohlcv_data: Dict[str, List]) -> Optional[HighFrequencySignal]:
        """Analyze symbol using all strategies in parallel"""
        start_time = time.time()

        try:
            # Run all strategy analyses in parallel for speed
            tasks = []
            strategy_names = []

            for strategy_name, strategy in self.strategies.items():
                if strategy_name == 'market_intelligence':
                    # Market intelligence uses different interface
                    task = self._analyze_with_intel(symbol, strategy)
                elif strategy_name == 'ichimoku_sniper':
                    # Ichimoku uses different interface
                    task = self._analyze_with_ichimoku(symbol, strategy, ohlcv_data)
                else:
                    # Standard scalping strategies with analyze_symbol method
                    task = strategy.analyze_symbol(symbol, ohlcv_data)

                tasks.append(task)
                strategy_names.append(strategy_name)

            # Execute all analyses in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect valid signals
            strategy_signals = {}
            for strategy_name, result in zip(strategy_names, results):
                if isinstance(result, Exception):
                    self.logger.debug(f"{strategy_name} error: {result}")
                    continue

                if result is not None:
                    strategy_signals[strategy_name] = result

            # Fuse signals from all strategies
            fused_signal = await self._fuse_signals(
                symbol=symbol,
                strategy_signals=strategy_signals,
                ohlcv_data=ohlcv_data
            )

            if fused_signal:
                # Add timing information
                signal_latency = (time.time() - start_time) * 1000
                fused_signal.signal_latency_ms = signal_latency

                self.logger.info(
                    f"   ⚡ Signal generated for {symbol} in {signal_latency:.0f}ms | "
                    f"{fused_signal.strategies_agree}/{fused_signal.total_strategies} strategies | "
                    f"Strength: {fused_signal.signal_strength:.1f}%"
                )

            return fused_signal

        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return None

    async def _analyze_with_intel(self, symbol: str, intel_engine) -> Any:
        """Analyze using market intelligence engine"""
        try:
            # Convert symbol format if needed (e.g., ETHUSDT instead of ETH/USDT:USDT)
            clean_symbol = symbol.replace('/USDT:USDT', 'USDT').replace('/', '')
            intel_snapshot = await intel_engine.analyze_market(clean_symbol)

            # Convert to signal format
            if intel_snapshot and intel_snapshot.should_trade():
                return {
                    'direction': 'LONG' if intel_snapshot.consensus_sentiment == 'bullish' else 'SHORT',
                    'signal_strength': intel_snapshot.overall_score,
                    'strategy_name': 'market_intelligence'
                }

            return None

        except Exception as e:
            self.logger.debug(f"Intel analysis error: {e}")
            return None

    async def _analyze_with_ichimoku(self, symbol: str, ichimoku_strategy, ohlcv_data: Dict[str, List]) -> Any:
        """Analyze using Ichimoku Sniper Strategy"""
        try:
            # Ichimoku uses 30m timeframe
            if '30m' in ohlcv_data and len(ohlcv_data['30m']) > 0:
                # Calculate Ichimoku components
                ichimoku_data = ichimoku_strategy.calculate_ichimoku_components(ohlcv_data['30m'])

                # Generate signal
                signal = ichimoku_strategy.generate_signal(ichimoku_data, '30m')

                if signal:
                    return signal

            # Fallback to other timeframes if 30m not available
            for tf in ['5m', '3m', '1m']:
                if tf in ohlcv_data and len(ohlcv_data[tf]) > 0:
                    ichimoku_data = ichimoku_strategy.calculate_ichimoku_components(ohlcv_data[tf])
                    signal = ichimoku_strategy.generate_signal(ichimoku_data, tf)
                    if signal:
                        return signal

            return None

        except Exception as e:
            self.logger.debug(f"Ichimoku analysis error: {e}")
            return None

    async def _fuse_signals(
        self,
        symbol: str,
        strategy_signals: Dict[str, Any],
        ohlcv_data: Dict[str, List]
    ) -> Optional[HighFrequencySignal]:
        """Fuse signals from multiple strategies into consensus signal"""

        if not strategy_signals:
            return None

        # Count votes by direction
        long_votes = 0
        short_votes = 0
        long_strength = 0.0
        short_strength = 0.0
        neutral_votes = 0

        strategy_votes = {}
        strategy_scores = {}

        for strategy_name, signal in strategy_signals.items():
            weight = self.strategy_weights.get(strategy_name, 0.1)

            # Extract direction and strength with proper validation
            direction = None
            strength = 0.0
            
            if hasattr(signal, 'direction'):
                direction = signal.direction
                strength = getattr(signal, 'signal_strength', 50)
            elif isinstance(signal, dict):
                direction = signal.get('direction', 'NEUTRAL')
                strength = signal.get('signal_strength', 50)
            else:
                continue

            # Validate strength is numeric and in valid range
            try:
                strength = float(strength)
                if strength < 0 or strength > 100:
                    strength = max(0, min(100, strength))
            except (ValueError, TypeError):
                strength = 50

            strategy_votes[strategy_name] = direction
            strategy_scores[strategy_name] = strength

            # Weight votes - FIXED: Only process clear LONG/SHORT signals
            if direction == 'LONG' or direction == 'BUY':
                long_votes += 1
                long_strength += strength * weight
            elif direction == 'SHORT' or direction == 'SELL':
                short_votes += 1
                short_strength += strength * weight
            else:
                # Track neutral/no signal
                neutral_votes += 1

        # Determine consensus
        total_strategies = len(strategy_signals)
        directional_votes = long_votes + short_votes

        # Need at least 50% of strategies to have directional signals
        if directional_votes < total_strategies * 0.5:
            self.logger.debug(f"Insufficient directional signals: {directional_votes}/{total_strategies}")
            return None

        if long_votes > short_votes:
            consensus_direction = 'LONG'
            strategies_agree = long_votes
            weighted_strength = long_strength / (long_votes if long_votes > 0 else 1)
        elif short_votes > long_votes:
            consensus_direction = 'SHORT'
            strategies_agree = short_votes
            weighted_strength = short_strength / (short_votes if short_votes > 0 else 1)
        else:
            # No consensus (equal votes)
            self.logger.debug(f"No clear consensus: {long_votes} LONG vs {short_votes} SHORT")
            return None

        # Check consensus requirements - PRODUCTION GRADE
        consensus_confidence = (strategies_agree / total_strategies) * 100

        if strategies_agree < self.min_strategies_agree:
            self.logger.debug(f"Insufficient agreements: {strategies_agree} < {self.min_strategies_agree}")
            return None

        if consensus_confidence < self.min_consensus_confidence:
            self.logger.debug(f"Low confidence: {consensus_confidence:.1f}% < {self.min_consensus_confidence}%")
            return None

        if weighted_strength < self.min_signal_strength:
            self.logger.debug(f"Low strength: {weighted_strength:.1f}% < {self.min_signal_strength}%")
            return None

        # Calculate entry price from latest candle
        primary_tf = '1m' if '1m' in ohlcv_data else '3m'
        latest_candle = ohlcv_data[primary_tf][-1]
        entry_price = latest_candle[4]  # Close price

        # Calculate tight SL/TP for scalping
        if consensus_direction == 'LONG':
            stop_loss = entry_price * (1 - self.tight_stop_loss_pct / 100)
            tp1 = entry_price * (1 + self.quick_profit_targets[0] / 100)
            tp2 = entry_price * (1 + self.quick_profit_targets[1] / 100)
            tp3 = entry_price * (1 + self.quick_profit_targets[2] / 100)
        else:
            stop_loss = entry_price * (1 + self.tight_stop_loss_pct / 100)
            tp1 = entry_price * (1 - self.quick_profit_targets[0] / 100)
            tp2 = entry_price * (1 - self.quick_profit_targets[1] / 100)
            tp3 = entry_price * (1 - self.quick_profit_targets[2] / 100)

        # Calculate risk/reward
        risk = abs(entry_price - stop_loss)
        reward = abs(tp2 - entry_price)
        risk_reward_ratio = reward / risk if risk > 0 else 1.5

        # Dynamic leverage based on signal strength
        leverage = self._calculate_dynamic_leverage(weighted_strength, consensus_confidence)

        # Create fused signal
        fused_signal = HighFrequencySignal(
            symbol=symbol,
            direction=consensus_direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            signal_strength=weighted_strength,
            strategies_agree=strategies_agree,
            total_strategies=total_strategies,
            consensus_confidence=consensus_confidence,
            leverage=leverage,
            position_size_usdt=0.0,  # Will be calculated by position manager
            risk_reward_ratio=risk_reward_ratio,
            timeframe=primary_tf,
            expected_duration_seconds=120,  # 2 minutes average for scalping
            signal_latency_ms=0.0,  # Will be set by caller
            timestamp=datetime.now(),
            strategy_votes=strategy_votes,
            strategy_scores=strategy_scores,
            market_volatility='normal',
            momentum_phase='building',
            volume_profile='normal'
        )

        self.signals_generated += 1

        return fused_signal

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol format for consistency across all strategies"""
        try:
            # Handle various formats: ETH/USDT:USDT, ETHUSDT, ETH/USDT -> ETH/USDT:USDT
            if ':' in symbol:
                return symbol  # Already normalized
            elif '/' in symbol:
                # ETH/USDT -> ETH/USDT:USDT
                if not symbol.endswith(':USDT'):
                    return symbol + ':USDT'
            else:
                # ETHUSDT -> ETH/USDT:USDT
                if symbol.endswith('USDT'):
                    base = symbol[:-4]
                    return f"{base}/USDT:USDT"
            return symbol
        except Exception as e:
            self.logger.error(f"Symbol normalization error: {e}")
            return symbol

    def _validate_signal(self, signal: HighFrequencySignal) -> bool:
        """Validate signal data before sending to Telegram/Cornix"""
        try:
            # Check required fields
            if not signal.symbol or not signal.direction:
                self.logger.error("❌ Signal missing symbol or direction")
                return False
            
            # Direction must be LONG or SHORT
            if signal.direction not in ['LONG', 'SHORT']:
                self.logger.error(f"❌ Invalid direction: {signal.direction}")
                return False
            
            # Check prices are positive
            if signal.entry_price <= 0:
                self.logger.error(f"❌ Invalid entry price: {signal.entry_price}")
                return False
                
            if signal.stop_loss <= 0:
                self.logger.error(f"❌ Invalid stop loss: {signal.stop_loss}")
                return False
            
            # Check at least one valid TP
            valid_tps = [tp for tp in [signal.take_profit_1, signal.take_profit_2, signal.take_profit_3] if tp > 0]
            if not valid_tps:
                self.logger.error("❌ Signal has no valid take profit levels")
                return False
            
            # Validate SL/TP logic with tolerance
            tolerance = 0.001  # 0.1% tolerance
            
            if signal.direction == 'LONG':
                if signal.stop_loss >= signal.entry_price * (1 + tolerance):
                    self.logger.error(f"❌ LONG: SL ({signal.stop_loss}) must be below entry ({signal.entry_price})")
                    return False
                if signal.take_profit_1 <= signal.entry_price * (1 - tolerance):
                    self.logger.error(f"❌ LONG: TP ({signal.take_profit_1}) must be above entry ({signal.entry_price})")
                    return False
                    
            else:  # SHORT
                if signal.stop_loss <= signal.entry_price * (1 - tolerance):
                    self.logger.error(f"❌ SHORT: SL ({signal.stop_loss}) must be above entry ({signal.entry_price})")
                    return False
                if signal.take_profit_1 >= signal.entry_price * (1 + tolerance):
                    self.logger.error(f"❌ SHORT: TP ({signal.take_profit_1}) must be below entry ({signal.entry_price})")
                    return False
            
            # Check leverage is within bounds
            if signal.leverage < 1 or signal.leverage > 125:
                self.logger.error(f"❌ Invalid leverage: {signal.leverage}")
                return False
            
            # Check signal strength and confidence
            if signal.signal_strength < self.min_signal_strength:
                self.logger.error(f"❌ Signal strength too low: {signal.signal_strength:.1f}%")
                return False
                
            if signal.consensus_confidence < self.min_consensus_confidence:
                self.logger.error(f"❌ Confidence too low: {signal.consensus_confidence:.1f}%")
                return False
            
            # Check risk/reward ratio minimum
            if signal.risk_reward_ratio < 1.0:
                self.logger.warning(f"⚠️ Low R/R ratio: {signal.risk_reward_ratio:.2f} (acceptable for high-frequency)")
            
            return True
        
        except Exception as e:
            self.logger.error(f"❌ Signal validation error: {e}")
            return False

    def _calculate_dynamic_leverage(self, strength: float, confidence: float) -> int:
        """Calculate leverage based on signal quality"""
        # Base leverage
        base_leverage = 10

        # Increase leverage for stronger signals
        if strength >= 85 and confidence >= 75:
            return min(30, base_leverage + 20)
        elif strength >= 80 and confidence >= 70:
            return min(25, base_leverage + 15)
        elif strength >= 75 and confidence >= 65:
            return min(20, base_leverage + 10)
        else:
            return base_leverage

    async def scan_markets_high_frequency(
        self,
        exchange,
        symbols: List[str],
        position_manager
    ):
        """Scan markets at high frequency for scalping opportunities"""

        scan_count = 0

        while True:
            try:
                scan_count += 1
                scan_start = time.time()

                self.logger.info(f"\n{'='*80}")
                self.logger.info(f"⚡ HIGH-FREQUENCY SCAN #{scan_count} - {datetime.now().strftime('%H:%M:%S')}")
                self.logger.info(f"{'='*80}")

                # Scan all symbols in parallel for maximum speed
                tasks = []
                for symbol in symbols:
                    task = self._scan_single_symbol(exchange, symbol, position_manager)
                    tasks.append(task)

                # Execute all scans concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Count signals
                signals_found = sum(1 for r in results if r and not isinstance(r, Exception))

                scan_duration = time.time() - scan_start

                self.logger.info(
                    f"✅ Scan completed in {scan_duration:.2f}s | "
                    f"Signals: {signals_found}/{len(symbols)} | "
                    f"Total generated: {self.signals_generated}"
                )

                # High-frequency interval
                await asyncio.sleep(self.scan_interval)

            except Exception as e:
                self.logger.error(f"Scan error: {e}")
                await asyncio.sleep(self.scan_interval)

    async def _scan_single_symbol(self, exchange, symbol: str, position_manager) -> Optional[HighFrequencySignal]:
        """Scan single symbol for trading opportunities"""
        try:
            # Normalize symbol format for consistency
            normalized_symbol = self._normalize_symbol(symbol)
            
            # Fetch fast timeframe data
            ohlcv_data = await self.fetch_fast_ohlcv(exchange, normalized_symbol)

            if not ohlcv_data:
                return None

            # Analyze with all strategies in parallel
            signal = await self.analyze_symbol_parallel(normalized_symbol, ohlcv_data)

            if signal:
                # Calculate position parameters
                if position_manager:
                    leverage_analysis = await position_manager.calculate_optimal_leverage(
                        symbol=symbol,
                        account_balance=1000,
                        risk_tolerance='aggressive'  # Scalping mode
                    )

                    sl_tp_analysis = await position_manager.calculate_dynamic_sl_tp(
                        symbol=symbol,
                        entry_price=signal.entry_price,
                        direction=signal.direction,
                        leverage=signal.leverage
                    )

                    position_analysis = await position_manager.calculate_position_size(
                        symbol=symbol,
                        account_balance=1000,
                        entry_price=signal.entry_price,
                        stop_loss=signal.stop_loss,
                        leverage=signal.leverage
                    )

                    signal.position_size_usdt = position_analysis.get('position_size_usdt', 0)

                    # Log signal details
                    self.logger.info(f"\n🎯 HIGH-FREQUENCY SIGNAL: {symbol}")
                    self.logger.info(f"   Direction: {signal.direction}")
                    self.logger.info(f"   Entry: ${signal.entry_price:.4f}")
                    self.logger.info(f"   Stop Loss: ${signal.stop_loss:.4f}")
                    self.logger.info(f"   TP1: ${signal.take_profit_1:.4f}")
                    self.logger.info(f"   TP2: ${signal.take_profit_2:.4f}")
                    self.logger.info(f"   TP3: ${signal.take_profit_3:.4f}")
                    self.logger.info(f"   Leverage: {signal.leverage}x")
                    self.logger.info(f"   Position Size: ${signal.position_size_usdt:.2f}")
                    self.logger.info(f"   R/R Ratio: 1:{signal.risk_reward_ratio:.2f}")
                    self.logger.info(f"   Consensus: {signal.consensus_confidence:.1f}% ({signal.strategies_agree}/{signal.total_strategies})")
                    self.logger.info(f"   Strength: {signal.signal_strength:.1f}%")

                    # Send signal to Telegram with TRADE ✅ confirmation
                    if self.telegram_notifier:
                        self.logger.info(f"📤 Attempting to send {symbol} signal to Telegram...")
                        try:
                            # Validate signal data before sending
                            if not self._validate_signal(signal):
                                self.logger.error(f"❌ Signal validation failed for {symbol}")
                                return None
                            
                            # Send signal
                            telegram_success = await self.telegram_notifier.send_signal(signal)
                            
                            if telegram_success:
                                self.logger.info(f"✅ TRADE ✅ - Signal sent successfully to Telegram for {symbol}")
                            else:
                                self.logger.warning(f"⚠️ Telegram notification failed for {symbol}")
                        except Exception as e:
                            self.logger.error(f"❌ Error sending signal: {e}")
                    else:
                        self.logger.warning("⚠️ Telegram notifier not initialized")


                    # Export to ATAS platform
                    if self.atas_integration:
                        asyncio.create_task(self.atas_integration.export_signal(signal))

                    # Add to position closer for monitoring (if in live mode)
                    if self.position_closer:
                        order_info = {
                            'entry_price': signal.entry_price,
                            'position_size': signal.position_size_usdt,
                            'quantity': signal.position_size_usdt / signal.entry_price
                        }
                        asyncio.create_task(self.position_closer.add_position(signal, order_info))

                return signal

            return None

        except Exception as e:
            self.logger.debug(f"Scan error for {symbol}: {e}")
            return None