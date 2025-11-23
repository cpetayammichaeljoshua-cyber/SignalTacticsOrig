#!/usr/bin/env python3
"""
ZEC/USDT Trader Adapter - Wraps existing trader for ZEC/USDT symbol
"""

try:
    from SignalMaestro.fxsusdt_trader import FXSUSDTTrader as BaseTrader
except ImportError:
    # Fallback if fxsusdt_trader not available
    import logging
    class BaseTrader:
        def __init__(self):
            self.logger = logging.getLogger(self.__class__.__name__)
            self.symbol = 'ZECUSDT'

class ZECUSDTTrader(BaseTrader):
    """ZEC/USDT specific trader extending base trader functionality"""
    
    def __init__(self):
        try:
            super().__init__()
        except Exception as e:
            import logging
            self.logger = logging.getLogger(self.__class__.__name__)
            pass
        
        self.symbol = 'ZECUSDT'
        self.logger.info(f"🔄 ZECUSDTTrader initialized for {self.symbol}")
    
    async def get_current_price(self) -> float:
        """Get current ZEC/USDT price"""
        try:
            # Use the base implementation but with ZEC symbol
            from binance.um_futures import UMFutures
            
            client = UMFutures()
            ticker = client.ticker_price(symbol=self.symbol)
            return float(ticker['price'])
        except Exception as e:
            self.logger.error(f"Error getting price for {self.symbol}: {e}")
            return None
    
    async def get_24hr_ticker_stats(self, symbol: str = None) -> dict:
        """Get 24hr stats for ZEC/USDT"""
        symbol = symbol or self.symbol
        try:
            from binance.um_futures import UMFutures
            
            client = UMFutures()
            stats = client.ticker_24hr(symbol=symbol)
            return stats
        except Exception as e:
            self.logger.error(f"Error getting ticker stats for {symbol}: {e}")
            return None
    
    async def test_connection(self) -> bool:
        """Test connection to Binance"""
        try:
            price = await self.get_current_price()
            return price is not None
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
