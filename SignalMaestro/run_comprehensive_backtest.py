#!/usr/bin/env python3
"""
Comprehensive Backtest for Ultimate Trading Bot
Runs a complete backtest with signal generation and performance analysis
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtesting_engine import BacktestingEngine, BacktestConfig
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ComprehensiveBacktest:
    """Complete backtesting system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def run_backtest(self, days_back=30):
        """Run comprehensive backtest"""
        
        # Create configuration
        config = BacktestConfig(
            initial_capital=10.0,  # $10 USD
            risk_percentage=10.0,  # 10% risk per trade
            max_concurrent_trades=3,
            start_date=datetime.now() - timedelta(days=days_back),
            end_date=datetime.now() - timedelta(days=1),
            timeframes=['5m', '15m', '1h']
        )
        
        print("=" * 80)
        print("🚀 ULTIMATE TRADING BOT - COMPREHENSIVE BACKTEST")
        print("=" * 80)
        print(f"💰 Initial Capital: ${config.initial_capital}")
        print(f"📊 Risk per Trade: {config.risk_percentage}%")
        print(f"📈 Max Concurrent Trades: {config.max_concurrent_trades}")
        print(f"📅 Backtest Period: {config.start_date.strftime('%Y-%m-%d')} to {config.end_date.strftime('%Y-%m-%d')}")
        print(f"⏰ Duration: {days_back} days")
        print("=" * 80)
        
        # Initialize engine
        engine = BacktestingEngine(config)
        
        try:
            # Initialize exchange
            await engine.initialize_exchange()
            
            # Popular crypto pairs for testing
            test_symbols = [
                'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT',
                'SOL/USDT', 'DOGE/USDT', 'MATIC/USDT', 'DOT/USDT', 'LTC/USDT'
            ]
            
            signals_processed = 0
            signals_taken = 0
            
            print(f"📊 Testing {len(test_symbols)} trading pairs...")
            print("-" * 80)
            
            for i, symbol in enumerate(test_symbols):
                print(f"[{i+1}/{len(test_symbols)}] Processing {symbol}...")
                
                try:
                    # Fetch historical data
                    data = await engine.fetch_historical_data(symbol, '5m', limit=200)
                    
                    if not data or len(data) < 50:
                        print(f"❌ Insufficient data for {symbol}")
                        continue
                    
                    # Convert to DataFrame for analysis
                    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # Generate synthetic signals based on price action
                    signals = self.generate_test_signals(df, symbol)
                    
                    print(f"  📈 Generated {len(signals)} potential signals for {symbol}")
                    
                    # Process each signal
                    for signal in signals:
                        signals_processed += 1
                        
                        # Add realistic signal characteristics
                        signal.update({
                            'volatility': self.calculate_volatility(df),
                            'volume_ratio': np.random.uniform(0.8, 1.5),
                            'rsi': np.random.uniform(25, 75),
                            'trend_strength': np.random.uniform(0.3, 0.9)
                        })
                        
                        # Process signal
                        result = await engine.process_signal(signal, signal['timestamp'])
                        
                        if result:
                            signals_taken += 1
                            print(f"  ✅ Signal taken: {signal['direction']} {symbol} @ ${signal['entry_price']:.4f}")
                            
                            # Simulate price movement and position updates
                            await self.simulate_position_updates(engine, symbol, df, signal)
                
                except Exception as e:
                    print(f"❌ Error processing {symbol}: {e}")
                    continue
            
            print("-" * 80)
            print(f"📊 SIGNAL PROCESSING SUMMARY:")
            print(f"   Total Signals Generated: {signals_processed}")
            print(f"   Signals Taken: {signals_taken}")
            print(f"   Signal Acceptance Rate: {(signals_taken/signals_processed*100) if signals_processed > 0 else 0:.1f}%")
            print("-" * 80)
            
            # Calculate and display results
            metrics = await engine.calculate_metrics()
            self.display_results(engine, metrics)
            
            # Save results
            self.save_results(engine, metrics)
            
        except Exception as e:
            print(f"❌ Backtest failed: {e}")
            raise
        
        finally:
            if engine.exchange:
                await engine.exchange.close()
    
    def generate_test_signals(self, df, symbol):
        """Generate realistic trading signals from price data"""
        signals = []
        
        # Calculate indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['price_change'] = df['close'].pct_change()
        df['volume_sma'] = df['volume'].rolling(20).mean()
        
        # Generate signals based on simple conditions
        for i in range(50, len(df) - 10):  # Leave some buffer
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Simple momentum signals
            if (current['close'] > current['sma_20'] and 
                prev['close'] <= prev['sma_20'] and
                current['volume'] > current['volume_sma'] * 1.2):
                
                signals.append({
                    'symbol': symbol,
                    'direction': 'BUY',
                    'entry_price': current['close'],
                    'timestamp': current['timestamp'],
                    'signal_strength': min(95, 70 + abs(current['price_change']) * 1000)
                })
            
            elif (current['close'] < current['sma_20'] and 
                  prev['close'] >= prev['sma_20'] and
                  current['volume'] > current['volume_sma'] * 1.2):
                
                signals.append({
                    'symbol': symbol,
                    'direction': 'SELL',
                    'entry_price': current['close'],
                    'timestamp': current['timestamp'],
                    'signal_strength': min(95, 70 + abs(current['price_change']) * 1000)
                })
        
        return signals[:5]  # Limit to 5 signals per symbol for testing
    
    def calculate_volatility(self, df):
        """Calculate volatility from price data"""
        if len(df) < 20:
            return 0.02
        
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(24)  # Assuming 5m data
        return max(0.005, min(0.1, volatility))
    
    async def simulate_position_updates(self, engine, symbol, df, signal):
        """Simulate position updates with realistic price movements"""
        if symbol not in engine.active_positions:
            return
        
        position = engine.active_positions[symbol]
        entry_price = signal['entry_price']
        
        # Simulate 10 price updates over time
        price_changes = np.random.normal(0, 0.002, 10)  # Small random changes
        
        for i, price_change in enumerate(price_changes):
            current_price = entry_price * (1 + price_change)
            current_time = signal['timestamp'] + timedelta(minutes=i*5)
            
            # Update position
            engine.update_position(symbol, current_price, current_time)
            
            # Check if position is closed
            if position.is_closed:
                break
    
    def display_results(self, engine, metrics):
        """Display comprehensive backtest results"""
        print("\n" + "=" * 80)
        print("📊 BACKTEST RESULTS")
        print("=" * 80)
        
        # Basic metrics
        print(f"💰 FINANCIAL PERFORMANCE:")
        print(f"   Final Capital: ${engine.capital:.2f}")
        print(f"   Total PnL: ${metrics.total_pnl:.2f}")
        print(f"   Total Return: {metrics.total_pnl_percentage:.2f}%")
        print(f"   Peak Capital: ${metrics.peak_capital:.2f}")
        print(f"   Max Drawdown: {metrics.max_drawdown_percentage:.2f}%")
        
        print(f"\n📈 TRADING STATISTICS:")
        print(f"   Total Trades: {metrics.total_trades}")
        print(f"   Winning Trades: {metrics.winning_trades}")
        print(f"   Losing Trades: {metrics.losing_trades}")
        print(f"   Win Rate: {metrics.win_rate:.1f}%")
        print(f"   Profit Factor: {metrics.profit_factor:.2f}")
        
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"   Average Trade Duration: {metrics.avg_trade_duration:.1f} minutes")
        print(f"   Average Win: ${metrics.avg_win_amount:.2f}")
        print(f"   Average Loss: ${metrics.avg_loss_amount:.2f}")
        print(f"   Largest Win: ${metrics.largest_win:.2f}")
        print(f"   Largest Loss: ${metrics.largest_loss:.2f}")
        
        print(f"\n🎯 RISK METRICS:")
        print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"   Sortino Ratio: {metrics.sortino_ratio:.2f}")
        print(f"   Average Leverage: {metrics.avg_leverage_used:.1f}x")
        print(f"   Leverage Efficiency: {metrics.leverage_efficiency:.1f}%")
        
        print(f"\n📊 FREQUENCY ANALYSIS:")
        print(f"   Trades per Day: {metrics.trades_per_day:.1f}")
        print(f"   Trades per Hour: {metrics.trades_per_hour:.2f}")
        print(f"   Total Commission: ${metrics.total_commission:.2f}")
        
        # Active positions
        if engine.active_positions:
            print(f"\n🔄 ACTIVE POSITIONS:")
            for symbol, pos in engine.active_positions.items():
                print(f"   {symbol}: {pos.direction} ${pos.entry_price:.4f} | PnL: ${pos.current_pnl:.2f}")
        
        print("=" * 80)
    
    def save_results(self, engine, metrics):
        """Save backtest results to file"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'initial_capital': engine.config.initial_capital,
                'risk_percentage': engine.config.risk_percentage,
                'max_concurrent_trades': engine.config.max_concurrent_trades
            },
            'metrics': {
                'final_capital': engine.capital,
                'total_pnl': metrics.total_pnl,
                'total_return_pct': metrics.total_pnl_percentage,
                'total_trades': metrics.total_trades,
                'win_rate': metrics.win_rate,
                'profit_factor': metrics.profit_factor,
                'max_drawdown_pct': metrics.max_drawdown_percentage,
                'sharpe_ratio': metrics.sharpe_ratio,
                'avg_leverage': metrics.avg_leverage_used
            },
            'trade_history': [
                {
                    'symbol': trade.symbol,
                    'direction': trade.direction,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'pnl': trade.pnl,
                    'duration_minutes': trade.duration_minutes,
                    'leverage': trade.leverage
                }
                for trade in engine.trade_history
            ]
        }
        
        filename = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {filename}")

async def main():
    """Main function"""
    backtest = ComprehensiveBacktest()
    await backtest.run_backtest(days_back=7)  # Test with 7 days for speed

if __name__ == "__main__":
    asyncio.run(main())