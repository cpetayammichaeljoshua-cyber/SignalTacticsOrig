#!/usr/bin/env python3
"""
Dynamic Multi-Market Position Manager
Calculates optimal leverage, position sizes, and SL/TP for ALL markets
"""

import logging
from typing import Dict, Optional, Tuple
import ccxt
import asyncio


class DynamicMultiMarketPositionManager:
    """
    Advanced position management for all Binance USDM futures markets
    """
    
    def __init__(self, exchange: Optional[ccxt.Exchange] = None):
        self.logger = logging.getLogger(__name__)
        self.exchange = exchange
        
        self.max_leverage = 20
        self.min_leverage = 1
        self.risk_per_trade = 0.02
        self.max_position_size_pct = 0.10
        
        self.volatility_cache = {}
        self.atr_cache = {}
    
    async def calculate_optimal_leverage(
        self, 
        symbol: str, 
        account_balance: float = 1000,
        risk_tolerance: str = 'moderate'
    ) -> Dict:
        """
        Calculate optimal leverage based on market conditions
        
        Args:
            symbol: Trading pair (e.g., 'FXS/USDT:USDT')
            account_balance: Account balance in USDT
            risk_tolerance: 'conservative', 'moderate', or 'aggressive'
        
        Returns:
            Dict with leverage recommendation and analysis
        """
        try:
            volatility = await self.get_market_volatility(symbol)
            
            risk_multipliers = {
                'conservative': 0.5,
                'moderate': 1.0,
                'aggressive': 1.5
            }
            multiplier = risk_multipliers.get(risk_tolerance, 1.0)
            
            if volatility > 0.05:
                base_leverage = 3
            elif volatility > 0.03:
                base_leverage = 5
            elif volatility > 0.02:
                base_leverage = 8
            elif volatility > 0.01:
                base_leverage = 10
            else:
                base_leverage = 15
            
            optimal_leverage = int(base_leverage * multiplier)
            optimal_leverage = max(self.min_leverage, min(optimal_leverage, self.max_leverage))
            
            position_value = account_balance * self.max_position_size_pct * optimal_leverage
            
            return {
                'symbol': symbol,
                'optimal_leverage': optimal_leverage,
                'volatility': volatility,
                'volatility_category': self._categorize_volatility(volatility),
                'risk_tolerance': risk_tolerance,
                'max_position_value': position_value,
                'account_balance': account_balance,
                'recommendation': self._get_leverage_recommendation(optimal_leverage, volatility)
            }
            
        except Exception as e:
            self.logger.error(f"Leverage calculation error for {symbol}: {e}")
            return {
                'symbol': symbol,
                'optimal_leverage': 5,
                'error': str(e),
                'recommendation': 'Using default leverage of 5x due to calculation error'
            }
    
    async def calculate_dynamic_sl_tp(
        self,
        symbol: str,
        entry_price: float,
        direction: str,
        leverage: int = 5,
        risk_reward_ratio: float = 2.0
    ) -> Dict:
        """
        Calculate dynamic stop-loss and take-profit levels
        
        Args:
            symbol: Trading pair
            entry_price: Entry price
            direction: 'LONG' or 'SHORT'
            leverage: Leverage being used
            risk_reward_ratio: Risk/reward ratio (default 2.0)
        
        Returns:
            Dict with SL/TP levels and distances
        """
        try:
            atr = await self.get_atr(symbol)
            
            atr_multiplier_sl = 1.5 / (leverage / 5)
            atr_multiplier_tp = atr_multiplier_sl * risk_reward_ratio
            
            if direction.upper() == 'LONG':
                sl_price = entry_price - (atr * atr_multiplier_sl)
                tp_price = entry_price + (atr * atr_multiplier_tp)
            else:
                sl_price = entry_price + (atr * atr_multiplier_sl)
                tp_price = entry_price - (atr * atr_multiplier_tp)
            
            sl_distance_pct = abs(sl_price - entry_price) / entry_price * 100
            tp_distance_pct = abs(tp_price - entry_price) / entry_price * 100
            
            potential_loss_pct = sl_distance_pct * leverage
            potential_profit_pct = tp_distance_pct * leverage
            
            return {
                'symbol': symbol,
                'direction': direction,
                'entry_price': entry_price,
                'stop_loss': sl_price,
                'take_profit': tp_price,
                'atr': atr,
                'sl_distance_pct': sl_distance_pct,
                'tp_distance_pct': tp_distance_pct,
                'potential_loss_pct': potential_loss_pct,
                'potential_profit_pct': potential_profit_pct,
                'risk_reward_ratio': potential_profit_pct / potential_loss_pct if potential_loss_pct > 0 else 0,
                'leverage': leverage
            }
            
        except Exception as e:
            self.logger.error(f"SL/TP calculation error for {symbol}: {e}")
            
            fallback_sl_pct = 2.0 / (leverage / 5)
            fallback_tp_pct = fallback_sl_pct * risk_reward_ratio
            
            if direction.upper() == 'LONG':
                sl_price = entry_price * (1 - fallback_sl_pct / 100)
                tp_price = entry_price * (1 + fallback_tp_pct / 100)
            else:
                sl_price = entry_price * (1 + fallback_sl_pct / 100)
                tp_price = entry_price * (1 - fallback_tp_pct / 100)
            
            return {
                'symbol': symbol,
                'direction': direction,
                'entry_price': entry_price,
                'stop_loss': sl_price,
                'take_profit': tp_price,
                'error': str(e),
                'note': 'Using fallback calculation'
            }
    
    async def calculate_position_size(
        self,
        symbol: str,
        account_balance: float,
        entry_price: float,
        stop_loss: float,
        leverage: int = 5
    ) -> Dict:
        """
        Calculate optimal position size based on risk management
        
        Args:
            symbol: Trading pair
            account_balance: Account balance in USDT
            entry_price: Planned entry price
            stop_loss: Stop-loss price
            leverage: Leverage to use
        
        Returns:
            Dict with position size calculations
        """
        try:
            risk_amount = account_balance * self.risk_per_trade
            
            sl_distance = abs(entry_price - stop_loss)
            sl_distance_pct = sl_distance / entry_price
            
            position_value = risk_amount / sl_distance_pct
            
            max_position_value = account_balance * self.max_position_size_pct * leverage
            position_value = min(position_value, max_position_value)
            
            quantity = position_value / entry_price
            
            margin_required = position_value / leverage
            
            return {
                'symbol': symbol,
                'position_size_usdt': position_value,
                'quantity': quantity,
                'margin_required': margin_required,
                'leverage': leverage,
                'risk_amount': risk_amount,
                'risk_pct': self.risk_per_trade * 100,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'account_balance': account_balance,
                'position_size_pct': (position_value / account_balance) * 100
            }
            
        except Exception as e:
            self.logger.error(f"Position size calculation error: {e}")
            return {
                'error': str(e),
                'recommendation': 'Use minimum position size'
            }
    
    async def get_market_volatility(self, symbol: str, period: int = 24) -> float:
        """Calculate market volatility (standard deviation of returns)"""
        try:
            if symbol in self.volatility_cache:
                return self.volatility_cache[symbol]
            
            if self.exchange:
                ohlcv = await self.exchange.fetch_ohlcv(symbol, '1h', limit=period)
                
                if not ohlcv or len(ohlcv) < 2:
                    return 0.025
                
                closes = [candle[4] for candle in ohlcv]
                returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
                
                mean_return = sum(returns) / len(returns)
                variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
                volatility = variance ** 0.5
                
                self.volatility_cache[symbol] = volatility
                return volatility
            else:
                return 0.025
                
        except Exception as e:
            self.logger.debug(f"Volatility calculation error: {e}")
            return 0.025
    
    async def get_atr(self, symbol: str, period: int = 14) -> float:
        """Calculate Average True Range (ATR)"""
        try:
            if symbol in self.atr_cache:
                return self.atr_cache[symbol]
            
            if self.exchange:
                ohlcv = await self.exchange.fetch_ohlcv(symbol, '1h', limit=period + 1)
                
                if not ohlcv or len(ohlcv) < 2:
                    return 50.0
                
                true_ranges = []
                for i in range(1, len(ohlcv)):
                    high = ohlcv[i][2]
                    low = ohlcv[i][3]
                    prev_close = ohlcv[i-1][4]
                    
                    tr = max(
                        high - low,
                        abs(high - prev_close),
                        abs(low - prev_close)
                    )
                    true_ranges.append(tr)
                
                atr = sum(true_ranges) / len(true_ranges) if true_ranges else 50.0
                
                self.atr_cache[symbol] = atr
                return atr
            else:
                return 50.0
                
        except Exception as e:
            self.logger.debug(f"ATR calculation error: {e}")
            return 50.0
    
    def _categorize_volatility(self, volatility: float) -> str:
        """Categorize volatility level"""
        if volatility > 0.05:
            return 'Very High'
        elif volatility > 0.03:
            return 'High'
        elif volatility > 0.02:
            return 'Moderate'
        elif volatility > 0.01:
            return 'Low'
        else:
            return 'Very Low'
    
    def _get_leverage_recommendation(self, leverage: int, volatility: float) -> str:
        """Get leverage recommendation text"""
        vol_category = self._categorize_volatility(volatility)
        
        if leverage <= 3:
            return f"Conservative leverage ({leverage}x) suitable for {vol_category} volatility"
        elif leverage <= 7:
            return f"Moderate leverage ({leverage}x) balanced for {vol_category} volatility"
        elif leverage <= 12:
            return f"Aggressive leverage ({leverage}x) - monitor {vol_category} volatility closely"
        else:
            return f"Very aggressive leverage ({leverage}x) - extreme risk with {vol_category} volatility"
    
    def format_position_analysis(self, analysis: Dict) -> str:
        """Format position analysis for display"""
        lines = []
        lines.append(f"📊 POSITION ANALYSIS: {analysis.get('symbol', 'N/A')}")
        lines.append("=" * 60)
        
        if 'optimal_leverage' in analysis:
            lines.append(f"⚡ Optimal Leverage: {analysis['optimal_leverage']}x")
            lines.append(f"📈 Volatility: {analysis.get('volatility', 0)*100:.2f}% ({analysis.get('volatility_category', 'N/A')})")
            lines.append(f"💡 {analysis.get('recommendation', 'N/A')}")
        
        if 'stop_loss' in analysis:
            lines.append(f"\n🎯 Entry: ${analysis['entry_price']:,.4f}")
            lines.append(f"🛑 Stop Loss: ${analysis['stop_loss']:,.4f} ({analysis.get('sl_distance_pct', 0):.2f}%)")
            lines.append(f"💰 Take Profit: ${analysis['take_profit']:,.4f} ({analysis.get('tp_distance_pct', 0):.2f}%)")
            lines.append(f"📊 Risk/Reward: 1:{analysis.get('risk_reward_ratio', 0):.2f}")
        
        if 'position_size_usdt' in analysis:
            lines.append(f"\n💵 Position Size: ${analysis['position_size_usdt']:,.2f}")
            lines.append(f"📦 Quantity: {analysis['quantity']:,.4f}")
            lines.append(f"💳 Margin Required: ${analysis['margin_required']:,.2f}")
            lines.append(f"⚖️  Risk Amount: ${analysis['risk_amount']:,.2f} ({analysis.get('risk_pct', 0):.1f}%)")
        
        lines.append("=" * 60)
        return "\n".join(lines)


async def demo_position_manager():
    """Demo showing position manager capabilities"""
    print("\n" + "="*80)
    print("🚀 DYNAMIC MULTI-MARKET POSITION MANAGER DEMO")
    print("="*80)
    
    manager = DynamicMultiMarketPositionManager()
    
    test_markets = [
        ('FXS/USDT:USDT', 1000, 'moderate'),
        ('BTC/USDT:USDT', 5000, 'conservative'),
        ('ETH/USDT:USDT', 2000, 'aggressive')
    ]
    
    for symbol, balance, risk_tolerance in test_markets:
        print(f"\n{'='*80}")
        print(f"📈 Analyzing {symbol}")
        print(f"{'='*80}")
        
        leverage_analysis = await manager.calculate_optimal_leverage(
            symbol=symbol,
            account_balance=balance,
            risk_tolerance=risk_tolerance
        )
        
        print(f"\n⚡ LEVERAGE ANALYSIS:")
        print(f"   Optimal Leverage: {leverage_analysis['optimal_leverage']}x")
        print(f"   Volatility: {leverage_analysis.get('volatility', 0)*100:.2f}%")
        print(f"   Category: {leverage_analysis.get('volatility_category', 'N/A')}")
        print(f"   Recommendation: {leverage_analysis['recommendation']}")
        
        entry_price = 3.5 if 'FXS' in symbol else (50000 if 'BTC' in symbol else 3000)
        
        sl_tp_analysis = await manager.calculate_dynamic_sl_tp(
            symbol=symbol,
            entry_price=entry_price,
            direction='LONG',
            leverage=leverage_analysis['optimal_leverage']
        )
        
        print(f"\n🎯 SL/TP LEVELS:")
        print(f"   Entry: ${sl_tp_analysis['entry_price']:,.4f}")
        print(f"   Stop Loss: ${sl_tp_analysis['stop_loss']:,.4f}")
        print(f"   Take Profit: ${sl_tp_analysis['take_profit']:,.4f}")
        print(f"   Risk/Reward: 1:{sl_tp_analysis.get('risk_reward_ratio', 0):.2f}")
        
        position_analysis = await manager.calculate_position_size(
            symbol=symbol,
            account_balance=balance,
            entry_price=entry_price,
            stop_loss=sl_tp_analysis['stop_loss'],
            leverage=leverage_analysis['optimal_leverage']
        )
        
        print(f"\n💰 POSITION SIZING:")
        print(f"   Position Value: ${position_analysis.get('position_size_usdt', 0):,.2f}")
        print(f"   Margin Required: ${position_analysis.get('margin_required', 0):,.2f}")
        print(f"   Risk Amount: ${position_analysis.get('risk_amount', 0):,.2f}")
    
    print(f"\n{'='*80}")
    print("✅ Demo complete!")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(demo_position_manager())
