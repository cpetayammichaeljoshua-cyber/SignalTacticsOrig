import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import random

class RealisticBacktester:
    """Realistic backtester with proper risk management and market simulation"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()

        # Realistic trading parameters
        self.initial_capital = config.get('initial_capital', 10.0)
        self.current_capital = self.initial_capital
        self.risk_per_trade = min(config.get('risk_percentage', 2.0), 3.0)  # Cap at 3%
        self.max_concurrent_trades = min(config.get('max_concurrent_trades', 3), 2)  # Cap at 2
        self.commission_rate = config.get('commission_rate', 0.0004)

        # Fixed risk sizing to prevent exponential growth
        self.use_fixed_risk = True
        self.fixed_risk_amount = self.initial_capital * (self.risk_per_trade / 100)

        # Trading tracking
        self.trades = []
        self.active_trades = []
        self.max_capital = self.initial_capital
        self.max_drawdown = 0.0

        # Market realism factors
        self.market_volatility = 1.0
        self.trend_strength = 0.5

    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    async def run_realistic_backtest(self, symbols: List[str], days: int = 7) -> Dict[str, Any]:
        """Run realistic backtest with proper controls"""

        self.logger.info(f"🚀 Starting Realistic Backtest - {days} days, {len(symbols)} symbols")

        try:
            # Generate realistic trade opportunities
            trade_opportunities = self._generate_realistic_signals(symbols, days)

            # Process trades chronologically
            for opportunity in sorted(trade_opportunities, key=lambda x: x['timestamp']):
                await self._process_trade_opportunity(opportunity)

            # Close remaining trades
            await self._close_remaining_trades()

            # Calculate final metrics
            results = self._calculate_realistic_metrics()

            return results

        except Exception as e:
            self.logger.error(f"Error in realistic backtest: {e}")
            return {}

    def _generate_realistic_signals(self, symbols: List[str], days: int) -> List[Dict[str, Any]]:
        """Generate realistic trading signals based on market behavior"""

        signals = []
        base_time = datetime.now() - timedelta(days=days)

        # Realistic signal frequency: 1-3 signals per day across all symbols
        total_signals = random.randint(days * 1, days * 3)

        for i in range(total_signals):
            signal_time = base_time + timedelta(
                hours=random.uniform(0, days * 24)
            )

            symbol = random.choice(symbols)
            direction = random.choice(['LONG', 'SHORT'])

            # Realistic price levels
            base_price = random.uniform(0.1, 100.0)

            # Calculate realistic SL/TP levels
            sl_distance = random.uniform(0.8, 2.0)  # 0.8-2% stop loss
            tp_distance = random.uniform(1.5, 4.0)   # 1.5-4% take profit

            if direction == 'LONG':
                entry_price = base_price
                stop_loss = entry_price * (1 - sl_distance / 100)
                take_profit = entry_price * (1 + tp_distance / 100)
            else:
                entry_price = base_price
                stop_loss = entry_price * (1 + sl_distance / 100)
                take_profit = entry_price * (1 - tp_distance / 100)

            # Signal strength affects win probability
            signal_strength = random.uniform(60, 85)

            signals.append({
                'timestamp': signal_time,
                'symbol': symbol,
                'direction': direction,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'signal_strength': signal_strength,
                'atr_value': base_price * 0.01  # 1% ATR
            })

        return signals

    async def _process_trade_opportunity(self, signal: Dict[str, Any]) -> bool:
        """Process a trading opportunity with realistic constraints"""

        try:
            # Check if we can take more trades
            if len(self.active_trades) >= self.max_concurrent_trades:
                return False

            # Check portfolio risk
            total_risk = len(self.active_trades) * self.fixed_risk_amount
            if total_risk >= self.initial_capital * 0.15:  # Max 15% total portfolio risk
                return False

            # Calculate position size with fixed risk
            position_size = self._calculate_realistic_position_size(signal)
            if not position_size:
                return False

            # Create trade
            trade = {
                'id': len(self.trades) + 1,
                'symbol': signal['symbol'],
                'direction': signal['direction'],
                'entry_price': signal['entry_price'],
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit'],
                'position_size': position_size,
                'entry_time': signal['timestamp'],
                'signal_strength': signal['signal_strength'],
                'status': 'ACTIVE'
            }

            # Add to active trades
            self.active_trades.append(trade)

            # Simulate trade outcome
            await self._simulate_realistic_trade_outcome(trade)

            return True

        except Exception as e:
            self.logger.error(f"Error processing trade opportunity: {e}")
            return False

    def _calculate_realistic_position_size(self, signal: Dict[str, Any]) -> Optional[float]:
        """Calculate position size with realistic risk management"""

        try:
            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']

            # Calculate stop loss distance
            sl_distance = abs(entry_price - stop_loss) / entry_price

            # Use fixed risk amount
            risk_amount = self.fixed_risk_amount

            # Calculate position size based on SL distance
            position_value = risk_amount / sl_distance
            position_size = position_value / entry_price

            # Ensure minimum position size
            min_position_value = 5.0  # $5 minimum
            if position_size * entry_price < min_position_value:
                position_size = min_position_value / entry_price

            # Cap maximum position size
            max_position_value = self.current_capital * 0.25  # Max 25% of capital
            if position_size * entry_price > max_position_value:
                position_size = max_position_value / entry_price

            return position_size

        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return None

    async def _simulate_realistic_trade_outcome(self, trade: Dict[str, Any]):
        """Simulate realistic trade outcome with market behavior"""

        try:
            # Realistic trade duration (15 minutes to 8 hours)
            duration_minutes = random.uniform(15, 480)
            exit_time = trade['entry_time'] + timedelta(minutes=duration_minutes)

            # Win probability based on signal strength and market conditions
            base_win_prob = (trade['signal_strength'] - 50) / 100  # Convert to 0.1-0.35 range
            base_win_prob = max(0.35, min(0.65, base_win_prob))  # Cap between 35-65%

            # Market regime impact
            market_factor = random.uniform(0.8, 1.2)
            win_probability = base_win_prob * market_factor
            win_probability = max(0.25, min(0.75, win_probability))  # Final cap 25-75%

            is_winner = random.random() < win_probability

            # Calculate exit price with realistic slippage
            if is_winner:
                if random.random() < 0.70:  # 70% hit full TP
                    exit_price = trade['take_profit']
                    # Add realistic slippage
                    slippage = random.uniform(-0.0005, -0.0002)  # 0.02-0.05% slippage
                    if trade['direction'] == 'LONG':
                        exit_price *= (1 + slippage)
                    else:
                        exit_price *= (1 - slippage)
                    exit_reason = "Take Profit"
                else:  # 30% partial profit
                    partial_profit = random.uniform(0.5, 0.8)  # 50-80% of TP
                    if trade['direction'] == 'LONG':
                        profit_distance = trade['take_profit'] - trade['entry_price']
                        exit_price = trade['entry_price'] + (profit_distance * partial_profit)
                    else:
                        profit_distance = trade['entry_price'] - trade['take_profit']
                        exit_price = trade['entry_price'] - (profit_distance * partial_profit)
                    exit_reason = "Partial Profit"
            else:
                if random.random() < 0.80:  # 80% hit stop loss
                    exit_price = trade['stop_loss']
                    # Worse slippage on stop losses
                    slippage = random.uniform(0.0003, 0.0008)  # 0.03-0.08% worse slippage
                    if trade['direction'] == 'LONG':
                        exit_price *= (1 - slippage)
                    else:
                        exit_price *= (1 + slippage)
                    exit_reason = "Stop Loss"
                else:  # 20% early exit with small loss
                    loss_factor = random.uniform(0.3, 0.7)  # 30-70% of SL
                    if trade['direction'] == 'LONG':
                        loss_distance = trade['entry_price'] - trade['stop_loss']
                        exit_price = trade['entry_price'] - (loss_distance * loss_factor)
                    else:
                        loss_distance = trade['stop_loss'] - trade['entry_price']
                        exit_price = trade['entry_price'] + (loss_distance * loss_factor)
                    exit_reason = "Early Exit"

            # Calculate P&L
            if trade['direction'] == 'LONG':
                pnl_per_unit = exit_price - trade['entry_price']
            else:
                pnl_per_unit = trade['entry_price'] - exit_price

            gross_pnl = pnl_per_unit * trade['position_size']

            # Calculate commission (both entry and exit)
            trade_value = trade['position_size'] * trade['entry_price']
            commission = trade_value * self.commission_rate * 2  # Entry + Exit

            net_pnl = gross_pnl - commission
            pnl_percentage = (net_pnl / trade_value) * 100

            # Update trade
            trade.update({
                'exit_price': exit_price,
                'exit_time': exit_time,
                'exit_reason': exit_reason,
                'duration_minutes': duration_minutes,
                'gross_pnl': gross_pnl,
                'commission': commission,
                'net_pnl': net_pnl,
                'pnl_percentage': pnl_percentage,
                'is_winner': is_winner,
                'status': 'CLOSED'
            })

            # Update capital
            self.current_capital += net_pnl

            # Track drawdown
            if self.current_capital > self.max_capital:
                self.max_capital = self.current_capital

            current_drawdown = (self.max_capital - self.current_capital) / self.max_capital * 100
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown

            # Move to completed trades
            self.active_trades.remove(trade)
            self.trades.append(trade)

            self.logger.debug(f"Trade completed: {trade['symbol']} {trade['direction']} - "
                            f"P&L: ${net_pnl:.2f} ({pnl_percentage:.1f}%)")

        except Exception as e:
            self.logger.error(f"Error simulating trade outcome: {e}")

    async def _close_remaining_trades(self):
        """Close any remaining active trades"""

        for trade in self.active_trades.copy():
            # Close at entry price (neutral outcome)
            trade.update({
                'exit_price': trade['entry_price'],
                'exit_time': datetime.now(),
                'exit_reason': "End of Backtest",
                'duration_minutes': 120,
                'gross_pnl': 0.0,
                'commission': trade['position_size'] * trade['entry_price'] * self.commission_rate * 2,
                'net_pnl': -trade['position_size'] * trade['entry_price'] * self.commission_rate * 2,
                'pnl_percentage': -self.commission_rate * 200,  # Commission only
                'is_winner': False,
                'status': 'CLOSED'
            })

            self.current_capital += trade['net_pnl']
            self.trades.append(trade)

        self.active_trades.clear()

    def _calculate_realistic_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive realistic metrics"""

        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'return_percentage': 0,
                'final_capital': self.current_capital
            }

        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['is_winner']])
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # P&L metrics
        total_pnl = sum(t['net_pnl'] for t in self.trades)
        return_percentage = (total_pnl / self.initial_capital * 100) if self.initial_capital > 0 else 0

        gross_profit = sum(t['net_pnl'] for t in self.trades if t['net_pnl'] > 0)
        gross_loss = abs(sum(t['net_pnl'] for t in self.trades if t['net_pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

        # Advanced metrics
        durations = [t['duration_minutes'] for t in self.trades]
        avg_duration = np.mean(durations) if durations else 0

        returns = [t['pnl_percentage'] for t in self.trades]
        avg_return = np.mean(returns) if returns else 0
        return_std = np.std(returns) if len(returns) > 1 else 0
        sharpe_ratio = avg_return / return_std if return_std > 0 else 0

        # Commission analysis
        total_commission = sum(t['commission'] for t in self.trades)

        # Consecutive metrics
        consecutive_wins = self._calculate_consecutive_metrics(True)
        consecutive_losses = self._calculate_consecutive_metrics(False)

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'return_percentage': return_percentage,
            'final_capital': self.current_capital,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': profit_factor,
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses,
            'avg_trade_duration_minutes': avg_duration,
            'trades_per_hour': total_trades / (7 * 24) if total_trades > 0 else 0,
            'trades_per_day': total_trades / 7 if total_trades > 0 else 0,
            'max_drawdown_pct': self.max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'peak_capital': self.max_capital,
            'total_commission': total_commission,
            'avg_commission_per_trade': total_commission / total_trades if total_trades > 0 else 0,
            'commission_impact_pct': (total_commission / abs(total_pnl) * 100) if total_pnl != 0 else 0
        }

    def _calculate_consecutive_metrics(self, for_wins: bool) -> int:
        """Calculate max consecutive wins or losses"""

        if not self.trades:
            return 0

        max_consecutive = 0
        current_consecutive = 0

        for trade in self.trades:
            if trade['is_winner'] == for_wins:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

# Entry point for realistic backtesting
async def run_realistic_backtest(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Run realistic backtest with proper risk controls"""

    if not config:
        config = {
            'initial_capital': 10.0,
            'risk_percentage': 2.0,  # Conservative 2%
            'max_concurrent_trades': 2,  # Conservative concurrent trades
            'commission_rate': 0.0004
        }

    backtester = RealisticBacktester(config)

    # Test symbols
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']

    results = await backtester.run_realistic_backtest(symbols, days=7)

    return results

if __name__ == "__main__":
    asyncio.run(run_realistic_backtest())