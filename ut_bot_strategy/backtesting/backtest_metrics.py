"""
Backtest Metrics Calculator

Comprehensive analysis of backtest results with detailed statistics.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)


class BacktestMetrics:
    """
    Calculate comprehensive metrics from backtest trade records
    
    Metrics include:
    - Win rate / Loss rate
    - Total and average PnL
    - Direction-based analysis (Long vs Short)
    - Risk-reward statistics
    - Performance periods
    - Streaks and drawdowns
    """
    
    def __init__(self, trades: List):
        """
        Initialize with trade records
        
        Args:
            trades: List of TradeRecord objects from backtest
        """
        self.trades = trades
        self._metrics: Dict[str, Any] = {}
    
    def calculate_all_metrics(self) -> Dict[str, Any]:
        """Calculate all metrics and return as dictionary"""
        if not self.trades:
            return self._empty_metrics()
        
        self._metrics = {
            'overview': self._calculate_overview(),
            'win_loss': self._calculate_win_loss(),
            'pnl': self._calculate_pnl(),
            'direction': self._calculate_direction_stats(),
            'risk_reward': self._calculate_risk_reward(),
            'streaks': self._calculate_streaks(),
            'timing': self._calculate_timing()
        }
        
        return self._metrics
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure when no trades"""
        return {
            'overview': {
                'total_trades': 0,
                'period_days': 0,
                'trades_per_day': 0
            },
            'win_loss': {
                'winners': 0,
                'losers': 0,
                'win_rate': 0,
                'loss_rate': 0
            },
            'pnl': {
                'total_pnl_percent': 0,
                'average_pnl_percent': 0,
                'average_winner_pnl': 0,
                'average_loser_pnl': 0,
                'profit_factor': 0,
                'expectancy': 0
            },
            'direction': {
                'long_trades': 0,
                'short_trades': 0,
                'long_win_rate': 0,
                'short_win_rate': 0,
                'long_pnl': 0,
                'short_pnl': 0
            },
            'risk_reward': {
                'average_rr': 0,
                'best_rr': 0,
                'worst_rr': 0
            },
            'streaks': {
                'max_win_streak': 0,
                'max_loss_streak': 0,
                'current_streak': 0
            },
            'timing': {
                'avg_holding_bars': 0,
                'best_trade': None,
                'worst_trade': None
            }
        }
    
    def _calculate_overview(self) -> Dict[str, Any]:
        """Calculate overview statistics"""
        total = len(self.trades)
        
        if total > 0 and self.trades[0].entry_time and self.trades[-1].exit_time:
            start = self.trades[0].entry_time
            end = self.trades[-1].exit_time
            period = (end - start).days if isinstance(end, datetime) and isinstance(start, datetime) else 0
        else:
            period = 0
        
        return {
            'total_trades': total,
            'period_days': max(period, 1),
            'trades_per_day': round(total / max(period, 1), 2)
        }
    
    def _calculate_win_loss(self) -> Dict[str, Any]:
        """Calculate win/loss statistics"""
        winners = [t for t in self.trades if t.is_winner]
        losers = [t for t in self.trades if not t.is_winner]
        
        total = len(self.trades)
        win_count = len(winners)
        loss_count = len(losers)
        
        return {
            'winners': win_count,
            'losers': loss_count,
            'win_rate': round(win_count / total * 100, 2) if total > 0 else 0,
            'loss_rate': round(loss_count / total * 100, 2) if total > 0 else 0
        }
    
    def _calculate_pnl(self) -> Dict[str, Any]:
        """Calculate profit/loss statistics"""
        if not self.trades:
            return self._empty_metrics()['pnl']
        
        pnls = [t.pnl_percent for t in self.trades]
        winners = [t.pnl_percent for t in self.trades if t.is_winner]
        losers = [t.pnl_percent for t in self.trades if not t.is_winner]
        
        total_pnl = sum(pnls)
        avg_pnl = statistics.mean(pnls) if pnls else 0
        avg_winner = statistics.mean(winners) if winners else 0
        avg_loser = statistics.mean(losers) if losers else 0
        
        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
        win_rate = len(winners) / len(self.trades) if self.trades else 0
        expectancy = (win_rate * avg_winner) + ((1 - win_rate) * avg_loser)
        
        return {
            'total_pnl_percent': round(total_pnl, 2),
            'average_pnl_percent': round(avg_pnl, 2),
            'average_winner_pnl': round(avg_winner, 2),
            'average_loser_pnl': round(avg_loser, 2),
            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 999.99,
            'expectancy': round(expectancy, 2)
        }
    
    def _calculate_direction_stats(self) -> Dict[str, Any]:
        """Calculate statistics by trade direction"""
        long_trades = [t for t in self.trades if t.direction == 'LONG']
        short_trades = [t for t in self.trades if t.direction == 'SHORT']
        
        long_winners = [t for t in long_trades if t.is_winner]
        short_winners = [t for t in short_trades if t.is_winner]
        
        long_pnl = sum(t.pnl_percent for t in long_trades)
        short_pnl = sum(t.pnl_percent for t in short_trades)
        
        return {
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'long_win_rate': round(len(long_winners) / len(long_trades) * 100, 2) if long_trades else 0,
            'short_win_rate': round(len(short_winners) / len(short_trades) * 100, 2) if short_trades else 0,
            'long_pnl': round(long_pnl, 2),
            'short_pnl': round(short_pnl, 2)
        }
    
    def _calculate_risk_reward(self) -> Dict[str, Any]:
        """Calculate risk/reward statistics"""
        rrs = [t.risk_reward_achieved for t in self.trades if t.risk_reward_achieved != 0]
        
        if not rrs:
            return {
                'average_rr': 0,
                'best_rr': 0,
                'worst_rr': 0
            }
        
        return {
            'average_rr': round(statistics.mean(rrs), 2),
            'best_rr': round(max(rrs), 2),
            'worst_rr': round(min(rrs), 2)
        }
    
    def _calculate_streaks(self) -> Dict[str, Any]:
        """Calculate winning and losing streaks"""
        if not self.trades:
            return self._empty_metrics()['streaks']
        
        max_win_streak = 0
        max_loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0
        
        for trade in self.trades:
            if trade.is_winner:
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            else:
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)
        
        current = current_win_streak if current_win_streak > 0 else -current_loss_streak
        
        return {
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
            'current_streak': current
        }
    
    def _calculate_timing(self) -> Dict[str, Any]:
        """Calculate timing-related statistics"""
        if not self.trades:
            return self._empty_metrics()['timing']
        
        holding_bars = [t.holding_bars for t in self.trades if t.holding_bars > 0]
        avg_holding = statistics.mean(holding_bars) if holding_bars else 0
        
        sorted_by_pnl = sorted(self.trades, key=lambda t: t.pnl_percent, reverse=True)
        
        best = sorted_by_pnl[0] if sorted_by_pnl else None
        worst = sorted_by_pnl[-1] if sorted_by_pnl else None
        
        return {
            'avg_holding_bars': round(avg_holding, 1),
            'best_trade': {
                'pnl': best.pnl_percent if best else 0,
                'direction': best.direction if best else '',
                'date': best.entry_time.strftime('%Y-%m-%d') if best and best.entry_time else ''
            },
            'worst_trade': {
                'pnl': worst.pnl_percent if worst else 0,
                'direction': worst.direction if worst else '',
                'date': worst.entry_time.strftime('%Y-%m-%d') if worst and worst.entry_time else ''
            }
        }
    
    def format_telegram_message(self) -> str:
        """Format metrics as HTML for Telegram"""
        if not self._metrics:
            self.calculate_all_metrics()
        
        m = self._metrics
        overview = m.get('overview', {})
        win_loss = m.get('win_loss', {})
        pnl = m.get('pnl', {})
        direction = m.get('direction', {})
        rr = m.get('risk_reward', {})
        streaks = m.get('streaks', {})
        timing = m.get('timing', {})
        
        win_rate = win_loss.get('win_rate', 0)
        win_emoji = "🟢" if win_rate >= 50 else "🟡" if win_rate >= 40 else "🔴"
        pnl_emoji = "📈" if pnl.get('total_pnl_percent', 0) >= 0 else "📉"
        
        message = f"""
📊 <b>BACKTEST RESULTS</b> 📊

━━━━━━━━━━━━━━━━━━━━━

<b>📋 Overview</b>
• Total Trades: <code>{overview.get('total_trades', 0)}</code>
• Period: <code>{overview.get('period_days', 0)} days</code>
• Trades/Day: <code>{overview.get('trades_per_day', 0)}</code>

━━━━━━━━━━━━━━━━━━━━━

<b>{win_emoji} Win/Loss Analysis</b>
• Winners: <code>{win_loss.get('winners', 0)}</code>
• Losers: <code>{win_loss.get('losers', 0)}</code>
• Win Rate: <code>{win_rate}%</code>
• Loss Rate: <code>{win_loss.get('loss_rate', 0)}%</code>

━━━━━━━━━━━━━━━━━━━━━

<b>{pnl_emoji} PnL Statistics</b>
• Total PnL: <code>{pnl.get('total_pnl_percent', 0)}%</code>
• Avg Trade: <code>{pnl.get('average_pnl_percent', 0)}%</code>
• Avg Winner: <code>+{pnl.get('average_winner_pnl', 0)}%</code>
• Avg Loser: <code>{pnl.get('average_loser_pnl', 0)}%</code>
• Profit Factor: <code>{pnl.get('profit_factor', 0)}</code>
• Expectancy: <code>{pnl.get('expectancy', 0)}%</code>

━━━━━━━━━━━━━━━━━━━━━

<b>📈 Direction Analysis</b>
• Long Trades: <code>{direction.get('long_trades', 0)}</code> ({direction.get('long_win_rate', 0)}% WR)
• Short Trades: <code>{direction.get('short_trades', 0)}</code> ({direction.get('short_win_rate', 0)}% WR)
• Long PnL: <code>{direction.get('long_pnl', 0)}%</code>
• Short PnL: <code>{direction.get('short_pnl', 0)}%</code>

━━━━━━━━━━━━━━━━━━━━━

<b>🎯 Risk/Reward</b>
• Avg R:R: <code>{rr.get('average_rr', 0)}</code>
• Best R:R: <code>{rr.get('best_rr', 0)}</code>
• Worst R:R: <code>{rr.get('worst_rr', 0)}</code>

━━━━━━━━━━━━━━━━━━━━━

<b>🔥 Streaks</b>
• Max Win Streak: <code>{streaks.get('max_win_streak', 0)}</code>
• Max Loss Streak: <code>{streaks.get('max_loss_streak', 0)}</code>
• Current Streak: <code>{streaks.get('current_streak', 0)}</code>

━━━━━━━━━━━━━━━━━━━━━

<b>⏱️ Timing</b>
• Avg Hold: <code>{timing.get('avg_holding_bars', 0)} bars</code>
• Best Trade: <code>+{timing.get('best_trade', {}).get('pnl', 0)}%</code>
• Worst Trade: <code>{timing.get('worst_trade', {}).get('pnl', 0)}%</code>

━━━━━━━━━━━━━━━━━━━━━

<i>UT Bot + STC Strategy Backtest</i>
"""
        return message.strip()
