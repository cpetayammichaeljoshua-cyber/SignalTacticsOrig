
#!/usr/bin/env python3
"""
Dynamic Position Manager for FXSUSDT.P
Handles auto leverage and dynamic stop loss/take profit using ATR
Implements 1:2 risk/reward ratio with position sizing
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import hmac
import hashlib

from fxsusdt_trader import FXSUSDTTrader

@dataclass
class PositionConfig:
    """Position configuration with dynamic parameters"""
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    atr_value: float
    leverage: int
    position_size: float
    risk_amount: float
    expected_return: float

class DynamicPositionManager:
    """Manages dynamic position sizing, leverage, and SL/TP for FXSUSDT.P"""
    
    def __init__(self, trader: FXSUSDTTrader):
        self.logger = logging.getLogger(__name__)
        self.trader = trader
        
        # Risk management parameters
        self.max_risk_per_trade = 0.02  # 2% of account
        self.min_leverage = 2
        self.max_leverage = 20
        self.risk_reward_ratio = 2.0  # 1:2 RR
        
        # ATR-based parameters
        self.atr_multiplier_sl = 1.5
        self.atr_multiplier_tp = 3.0  # For 1:2 RR
        
        # Volatility-based leverage scaling
        self.volatility_leverage_map = {
            0.00001: 20,  # Very low volatility
            0.00005: 15,  # Low volatility
            0.0001: 10,   # Medium volatility
            0.0002: 5,    # High volatility
            0.0005: 2     # Very high volatility
        }
        
        self.logger.info("⚙️ Dynamic Position Manager initialized")
    
    async def calculate_optimal_leverage(self, atr_value: float, account_balance: float) -> int:
        """Calculate optimal leverage based on ATR volatility"""
        try:
            if atr_value <= 0:
                return self.min_leverage
            
            # Find appropriate leverage based on ATR
            optimal_leverage = self.max_leverage
            
            for atr_threshold, leverage in sorted(self.volatility_leverage_map.items()):
                if atr_value >= atr_threshold:
                    optimal_leverage = leverage
                    break
            
            # Ensure within bounds
            optimal_leverage = max(self.min_leverage, min(optimal_leverage, self.max_leverage))
            
            self.logger.info(f"📊 ATR: {atr_value:.6f} → Optimal Leverage: {optimal_leverage}x")
            return optimal_leverage
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal leverage: {e}")
            return self.min_leverage
    
    async def calculate_position_size(self, entry_price: float, stop_loss: float, 
                                    leverage: int, account_balance: float) -> Tuple[float, float]:
        """Calculate position size based on risk management"""
        try:
            # Calculate risk amount (2% of account)
            risk_amount = account_balance * self.max_risk_per_trade
            
            # Calculate stop loss distance
            sl_distance = abs(entry_price - stop_loss)
            
            if sl_distance <= 0:
                return 0.0, 0.0
            
            # Calculate position size considering leverage
            # Risk = Position Size * Price * SL Distance / Leverage
            # Position Size = Risk * Leverage / (Price * SL Distance)
            position_size = (risk_amount * leverage) / (entry_price * sl_distance)
            
            # Calculate notional value
            notional_value = position_size * entry_price
            
            self.logger.info(f"💰 Risk: ${risk_amount:.2f} | Position: {position_size:.6f} | Notional: ${notional_value:.2f}")
            
            return position_size, notional_value
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0, 0.0
    
    async def set_leverage(self, leverage: int) -> bool:
        """Set leverage for FXSUSDT trading"""
        try:
            endpoint = "/fapi/v1/leverage"
            timestamp = int(time.time() * 1000)
            
            params = {
                'symbol': self.trader.symbol,
                'leverage': leverage,
                'timestamp': timestamp
            }
            
            # Create query string
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            signature = self.trader._generate_signature(query_string)
            
            url = f"{self.trader.base_url}{endpoint}"
            
            headers = {
                'X-MBX-APIKEY': self.trader.api_key,
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            data = f"{query_string}&signature={signature}"
            
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.logger.info(f"⚡ Leverage set to {leverage}x for {self.trader.symbol}")
                        return True
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Failed to set leverage: {response.status} - {error_text}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"Error setting leverage: {e}")
            return False
    
    async def place_market_order(self, direction: str, quantity: float) -> Optional[Dict[str, Any]]:
        """Place market order for FXSUSDT"""
        try:
            endpoint = "/fapi/v1/order"
            timestamp = int(time.time() * 1000)
            
            side = "BUY" if direction.upper() == "BUY" else "SELL"
            
            params = {
                'symbol': self.trader.symbol,
                'side': side,
                'type': 'MARKET',
                'quantity': f"{quantity:.6f}",
                'timestamp': timestamp
            }
            
            # Create query string
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            signature = self.trader._generate_signature(query_string)
            
            url = f"{self.trader.base_url}{endpoint}"
            
            headers = {
                'X-MBX-APIKEY': self.trader.api_key,
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            data = f"{query_string}&signature={signature}"
            
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.logger.info(f"✅ Market order placed: {side} {quantity:.6f} {self.trader.symbol}")
                        return result
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Failed to place order: {response.status} - {error_text}")
                        return None
                        
        except Exception as e:
            self.logger.error(f"Error placing market order: {e}")
            return None
    
    async def place_stop_order(self, direction: str, quantity: float, stop_price: float) -> Optional[Dict[str, Any]]:
        """Place stop loss order"""
        try:
            endpoint = "/fapi/v1/order"
            timestamp = int(time.time() * 1000)
            
            # For stop loss, we need opposite direction
            side = "SELL" if direction.upper() == "BUY" else "BUY"
            
            params = {
                'symbol': self.trader.symbol,
                'side': side,
                'type': 'STOP_MARKET',
                'quantity': f"{quantity:.6f}",
                'stopPrice': f"{stop_price:.5f}",
                'timeInForce': 'GTC',
                'timestamp': timestamp
            }
            
            # Create query string
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            signature = self.trader._generate_signature(query_string)
            
            url = f"{self.trader.base_url}{endpoint}"
            
            headers = {
                'X-MBX-APIKEY': self.trader.api_key,
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            data = f"{query_string}&signature={signature}"
            
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.logger.info(f"🛡️ Stop loss placed: {stop_price:.5f}")
                        return result
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Failed to place stop order: {response.status} - {error_text}")
                        return None
                        
        except Exception as e:
            self.logger.error(f"Error placing stop order: {e}")
            return None
    
    async def place_limit_order(self, direction: str, quantity: float, limit_price: float) -> Optional[Dict[str, Any]]:
        """Place take profit limit order"""
        try:
            endpoint = "/fapi/v1/order"
            timestamp = int(time.time() * 1000)
            
            # For take profit, we need opposite direction
            side = "SELL" if direction.upper() == "BUY" else "BUY"
            
            params = {
                'symbol': self.trader.symbol,
                'side': side,
                'type': 'LIMIT',
                'quantity': f"{quantity:.6f}",
                'price': f"{limit_price:.5f}",
                'timeInForce': 'GTC',
                'timestamp': timestamp
            }
            
            # Create query string
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            signature = self.trader._generate_signature(query_string)
            
            url = f"{self.trader.base_url}{endpoint}"
            
            headers = {
                'X-MBX-APIKEY': self.trader.api_key,
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            data = f"{query_string}&signature={signature}"
            
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.logger.info(f"🎯 Take profit placed: {limit_price:.5f}")
                        return result
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Failed to place limit order: {response.status} - {error_text}")
                        return None
                        
        except Exception as e:
            self.logger.error(f"Error placing limit order: {e}")
            return None
    
    async def execute_position(self, signal_data: Dict[str, Any]) -> Optional[PositionConfig]:
        """Execute complete position with dynamic parameters"""
        try:
            # Extract signal data
            direction = signal_data['action']
            entry_price = signal_data['entry_price']
            stop_loss = signal_data['stop_loss']
            take_profit = signal_data['take_profit']
            atr_value = signal_data['atr_value']
            
            # Get account balance
            account_info = await self.trader.get_account_balance()
            if not account_info:
                self.logger.error("❌ Failed to get account balance")
                return None
            
            balance = account_info['available_balance']
            if balance <= 0:
                self.logger.error("❌ Insufficient account balance")
                return None
            
            # Calculate optimal leverage
            optimal_leverage = await self.calculate_optimal_leverage(atr_value, balance)
            
            # Set leverage
            if not await self.set_leverage(optimal_leverage):
                self.logger.error("❌ Failed to set leverage")
                return None
            
            # Calculate position size
            position_size, notional_value = await self.calculate_position_size(
                entry_price, stop_loss, optimal_leverage, balance
            )
            
            if position_size <= 0:
                self.logger.error("❌ Invalid position size calculated")
                return None
            
            # Place market order
            market_order = await self.place_market_order(direction, position_size)
            if not market_order:
                self.logger.error("❌ Failed to place market order")
                return None
            
            # Place stop loss
            stop_order = await self.place_stop_order(direction, position_size, stop_loss)
            if not stop_order:
                self.logger.warning("⚠️ Failed to place stop loss order")
            
            # Place take profit
            tp_order = await self.place_limit_order(direction, position_size, take_profit)
            if not tp_order:
                self.logger.warning("⚠️ Failed to place take profit order")
            
            # Create position config
            risk_amount = balance * self.max_risk_per_trade
            sl_distance = abs(entry_price - stop_loss)
            tp_distance = abs(take_profit - entry_price)
            expected_return = (tp_distance / sl_distance) * risk_amount
            
            position_config = PositionConfig(
                symbol=self.trader.symbol,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                atr_value=atr_value,
                leverage=optimal_leverage,
                position_size=position_size,
                risk_amount=risk_amount,
                expected_return=expected_return
            )
            
            self.logger.info(f"✅ Position executed: {direction} {position_size:.6f} @ {entry_price:.5f}")
            self.logger.info(f"   Leverage: {optimal_leverage}x | Risk: ${risk_amount:.2f} | Expected: ${expected_return:.2f}")
            
            return position_config
            
        except Exception as e:
            self.logger.error(f"Error executing position: {e}")
            return None
