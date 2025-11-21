#!/usr/bin/env python3
"""
Automatic Position Closer with Dynamic Exit Strategies
Monitors open positions and automatically closes them based on:
- Take Profit targets hit
- Stop Loss triggered
- Time-based exits (scalping duration)
- Trailing stop activation
- Market condition changes
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import ccxt


@dataclass
class OpenPosition:
    """Track open position data"""
    symbol: str
    direction: str  # LONG or SHORT
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    leverage: int
    position_size: float
    quantity: float
    
    # Tracking
    position_id: str
    highest_price: float = 0.0
    lowest_price: float = 0.0
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    trailing_stop_active: bool = False
    trailing_stop_price: float = 0.0
    
    # Performance
    current_pnl: float = 0.0
    current_pnl_pct: float = 0.0
    max_profit: float = 0.0
    max_drawdown: float = 0.0


class AutomaticPositionCloser:
    """
    Monitors and automatically closes positions based on multiple exit strategies
    """
    
    def __init__(self, exchange: Optional[ccxt.Exchange] = None, telegram_notifier=None):
        self.logger = logging.getLogger(__name__)
        self.exchange = exchange
        self.telegram_notifier = telegram_notifier
        
        # Configuration
        self.check_interval = 2  # Check every 2 seconds for scalping
        self.max_position_duration = 300  # 5 minutes max for scalping
        self.trailing_stop_activation_pct = 1.0  # Activate trailing after 1% profit
        self.trailing_stop_distance_pct = 0.3  # Trail by 0.3%
        
        # Position tracking
        self.open_positions: Dict[str, OpenPosition] = {}
        self.closed_positions: List[Dict[str, Any]] = []
        
        # Statistics
        self.total_closed = 0
        self.total_wins = 0
        self.total_losses = 0
        self.total_pnl = 0.0
        
        self.logger.info("✅ Automatic Position Closer initialized")
    
    async def add_position(self, signal: Any, order_info: Dict[str, Any]) -> str:
        """
        Add a new position to monitor
        
        Args:
            signal: Trading signal that generated the position
            order_info: Order execution details
            
        Returns:
            position_id: Unique position identifier
        """
        try:
            # Generate position ID
            position_id = f"{self._get_attr(signal, 'symbol', 'UNKNOWN')}_{datetime.now().timestamp()}"
            
            # Extract signal data
            symbol = self._get_attr(signal, 'symbol', 'UNKNOWN')
            direction = self._get_attr(signal, 'direction', 'LONG')
            entry_price = order_info.get('entry_price', self._get_attr(signal, 'entry_price', 0))
            
            # Create position
            position = OpenPosition(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                entry_time=datetime.now(),
                stop_loss=self._get_attr(signal, 'stop_loss', 0),
                take_profit_1=self._get_attr(signal, 'take_profit_1', 0),
                take_profit_2=self._get_attr(signal, 'take_profit_2', 0),
                take_profit_3=self._get_attr(signal, 'take_profit_3', 0),
                leverage=self._get_attr(signal, 'leverage', 10),
                position_size=order_info.get('position_size', 0),
                quantity=order_info.get('quantity', 0),
                position_id=position_id,
                highest_price=entry_price,
                lowest_price=entry_price
            )
            
            # Add to tracking
            self.open_positions[position_id] = position
            
            self.logger.info(f"📊 Added position to monitor: {position_id}")
            self.logger.info(f"   Symbol: {symbol} | Direction: {direction} | Entry: ${entry_price:.4f}")
            
            # Notify via Telegram
            if self.telegram_notifier:
                await self.telegram_notifier.send_position_update(
                    symbol=symbol,
                    update_type='OPENED',
                    details={
                        'direction': direction,
                        'price': entry_price,
                        'leverage': self._get_attr(signal, 'leverage', 10),
                        'position_size': order_info.get('position_size', 0)
                    }
                )
            
            return position_id
            
        except Exception as e:
            self.logger.error(f"Error adding position: {e}")
            return ""
    
    async def monitor_positions(self):
        """Continuously monitor all open positions"""
        
        self.logger.info("🔄 Starting position monitoring...")
        
        while True:
            try:
                if not self.open_positions:
                    await asyncio.sleep(self.check_interval)
                    continue
                
                # Check each position
                positions_to_close = []
                
                for position_id, position in self.open_positions.items():
                    should_close, close_reason, close_price = await self._check_exit_conditions(position)
                    
                    if should_close:
                        positions_to_close.append((position_id, position, close_reason, close_price))
                
                # Close positions
                for position_id, position, reason, price in positions_to_close:
                    await self._close_position(position_id, position, reason, price)
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in position monitoring: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _check_exit_conditions(self, position: OpenPosition) -> tuple:
        """
        Check if position should be closed
        
        Returns:
            (should_close, reason, price)
        """
        try:
            # Get current price
            if self.exchange:
                ticker = await self.exchange.fetch_ticker(position.symbol)
                current_price = ticker['last']
            else:
                # Demo mode - simulate price movement
                current_price = position.entry_price * 1.005  # +0.5% for testing
            
            # Update price tracking
            position.highest_price = max(position.highest_price, current_price)
            position.lowest_price = min(position.lowest_price, current_price)
            
            # Calculate current PnL
            if position.direction == 'LONG':
                pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
            else:
                pnl_pct = ((position.entry_price - current_price) / position.entry_price) * 100
            
            position.current_pnl_pct = pnl_pct
            position.current_pnl = position.position_size * (pnl_pct / 100) * position.leverage
            position.max_profit = max(position.max_profit, position.current_pnl)
            
            # Check exit conditions
            
            # 1. Stop Loss hit
            if position.direction == 'LONG':
                if current_price <= position.stop_loss:
                    return True, "Stop Loss Hit", current_price
            else:
                if current_price >= position.stop_loss:
                    return True, "Stop Loss Hit", current_price
            
            # 2. Take Profit levels
            if position.direction == 'LONG':
                if not position.tp3_hit and current_price >= position.take_profit_3:
                    position.tp3_hit = True
                    return True, "Take Profit 3 Hit", current_price
                elif not position.tp2_hit and current_price >= position.take_profit_2:
                    position.tp2_hit = True
                    return True, "Take Profit 2 Hit", current_price
                elif not position.tp1_hit and current_price >= position.take_profit_1:
                    position.tp1_hit = True
                    return True, "Take Profit 1 Hit", current_price
            else:
                if not position.tp3_hit and current_price <= position.take_profit_3:
                    position.tp3_hit = True
                    return True, "Take Profit 3 Hit", current_price
                elif not position.tp2_hit and current_price <= position.take_profit_2:
                    position.tp2_hit = True
                    return True, "Take Profit 2 Hit", current_price
                elif not position.tp1_hit and current_price <= position.take_profit_1:
                    position.tp1_hit = True
                    return True, "Take Profit 1 Hit", current_price
            
            # 3. Trailing stop
            if position.current_pnl_pct >= self.trailing_stop_activation_pct:
                if not position.trailing_stop_active:
                    position.trailing_stop_active = True
                    self.logger.info(f"🎯 Trailing stop activated for {position.symbol}")
                
                # Update trailing stop
                if position.direction == 'LONG':
                    new_trailing_stop = position.highest_price * (1 - self.trailing_stop_distance_pct / 100)
                    position.trailing_stop_price = max(position.trailing_stop_price, new_trailing_stop)
                    
                    if current_price <= position.trailing_stop_price:
                        return True, "Trailing Stop Hit", current_price
                else:
                    new_trailing_stop = position.lowest_price * (1 + self.trailing_stop_distance_pct / 100)
                    if position.trailing_stop_price == 0:
                        position.trailing_stop_price = new_trailing_stop
                    else:
                        position.trailing_stop_price = min(position.trailing_stop_price, new_trailing_stop)
                    
                    if current_price >= position.trailing_stop_price:
                        return True, "Trailing Stop Hit", current_price
            
            # 4. Time-based exit for scalping
            position_age = (datetime.now() - position.entry_time).total_seconds()
            if position_age > self.max_position_duration:
                return True, "Time-Based Exit (Scalping)", current_price
            
            return False, "", 0.0
            
        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}")
            return False, "", 0.0
    
    async def _close_position(self, position_id: str, position: OpenPosition, reason: str, close_price: float):
        """Close position and record results"""
        try:
            # Calculate final PnL
            if position.direction == 'LONG':
                pnl_pct = ((close_price - position.entry_price) / position.entry_price) * 100
            else:
                pnl_pct = ((position.entry_price - close_price) / position.entry_price) * 100
            
            final_pnl = position.position_size * (pnl_pct / 100) * position.leverage
            
            # Update statistics
            self.total_closed += 1
            if final_pnl > 0:
                self.total_wins += 1
            else:
                self.total_losses += 1
            self.total_pnl += final_pnl
            
            # Record closed position
            closed_record = {
                'position_id': position_id,
                'symbol': position.symbol,
                'direction': position.direction,
                'entry_price': position.entry_price,
                'close_price': close_price,
                'entry_time': position.entry_time,
                'close_time': datetime.now(),
                'duration_seconds': (datetime.now() - position.entry_time).total_seconds(),
                'pnl': final_pnl,
                'pnl_pct': pnl_pct,
                'close_reason': reason,
                'leverage': position.leverage,
                'position_size': position.position_size
            }
            
            self.closed_positions.append(closed_record)
            
            # Log closure
            pnl_emoji = "💰" if final_pnl > 0 else "⚠️"
            self.logger.info(f"\n{pnl_emoji} POSITION CLOSED: {position.symbol}")
            self.logger.info(f"   Reason: {reason}")
            self.logger.info(f"   Entry: ${position.entry_price:.4f}")
            self.logger.info(f"   Exit: ${close_price:.4f}")
            self.logger.info(f"   PnL: ${final_pnl:.2f} ({pnl_pct:+.2f}%)")
            self.logger.info(f"   Duration: {closed_record['duration_seconds']:.0f}s")
            
            # Notify via Telegram
            if self.telegram_notifier:
                update_type = 'PROFIT' if final_pnl > 0 else 'LOSS'
                await self.telegram_notifier.send_position_update(
                    symbol=position.symbol,
                    update_type=update_type,
                    details={
                        'direction': position.direction,
                        'price': close_price,
                        'pnl': final_pnl,
                        'pnl_pct': pnl_pct,
                        'reason': reason
                    }
                )
            
            # Remove from open positions
            del self.open_positions[position_id]
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get trading statistics"""
        win_rate = (self.total_wins / self.total_closed * 100) if self.total_closed > 0 else 0
        
        return {
            'total_closed': self.total_closed,
            'total_wins': self.total_wins,
            'total_losses': self.total_losses,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'open_positions': len(self.open_positions)
        }
    
    def _get_attr(self, obj: Any, attr: str, default: Any = None) -> Any:
        """Get attribute from object or dict"""
        if hasattr(obj, attr):
            return getattr(obj, attr)
        elif isinstance(obj, dict):
            return obj.get(attr, default)
        return default


async def demo_position_closer():
    """Demo the automatic position closer"""
    print("\n" + "="*80)
    print("🔄 AUTOMATIC POSITION CLOSER DEMO")
    print("="*80)
    
    closer = AutomaticPositionCloser()
    
    # Simulate adding a position
    test_signal = {
        'symbol': 'ETH/USDT:USDT',
        'direction': 'LONG',
        'entry_price': 3500.00,
        'stop_loss': 3482.50,
        'take_profit_1': 3528.00,
        'take_profit_2': 3542.00,
        'take_profit_3': 3563.00,
        'leverage': 20
    }
    
    order_info = {
        'entry_price': 3500.00,
        'position_size': 100.0,
        'quantity': 0.0286
    }
    
    position_id = await closer.add_position(test_signal, order_info)
    print(f"\n✅ Position added: {position_id}")
    
    # Monitor for a short time
    print("\n🔄 Monitoring position for 10 seconds...")
    await asyncio.sleep(10)
    
    stats = closer.get_statistics()
    print(f"\n📊 Statistics:")
    print(f"   Open Positions: {stats['open_positions']}")
    print(f"   Closed Positions: {stats['total_closed']}")
    print(f"   Win Rate: {stats['win_rate']:.1f}%")
    print(f"   Total PnL: ${stats['total_pnl']:.2f}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_position_closer())
