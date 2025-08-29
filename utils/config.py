"""
Configuration management for the trading bot
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class Config:
    """Configuration class for the trading bot"""
    
    # API Credentials
    binance_api_key: str = ""
    binance_api_secret: str = ""
    telegram_bot_token: str = ""
    
    # Trading Parameters
    symbols: Optional[List[str]] = None
    base_position_size: float = 0.001  # BTC
    max_leverage: int = 50
    min_leverage: int = 1
    risk_per_trade: float = 0.02  # 2% of portfolio
    
    # ML Parameters
    lookback_period: int = 100
    retrain_interval: int = 24  # hours
    feature_window: int = 20
    
    # Risk Management
    max_daily_trades: int = 10
    max_drawdown: float = 0.15  # 15%
    stop_loss: float = 0.05  # 5%
    take_profit: float = 0.10  # 10%
    
    # Technical Settings
    update_interval: int = 60  # seconds
    data_retention: int = 1000  # candles
    
    def __post_init__(self):
        """Initialize configuration from environment variables"""
        self.binance_api_key = os.getenv("BINANCE_API_KEY", "")
        self.binance_api_secret = os.getenv("BINANCE_API_SECRET", "")
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        
        if self.symbols is None:
            self.symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT"]
        
        if not self.binance_api_key or not self.binance_api_secret:
            raise ValueError("Binance API credentials not found in environment variables")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "symbols": self.symbols,
            "base_position_size": self.base_position_size,
            "max_leverage": self.max_leverage,
            "min_leverage": self.min_leverage,
            "risk_per_trade": self.risk_per_trade,
            "lookback_period": self.lookback_period,
            "retrain_interval": self.retrain_interval,
            "feature_window": self.feature_window,
            "max_daily_trades": self.max_daily_trades,
            "max_drawdown": self.max_drawdown,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "update_interval": self.update_interval,
            "data_retention": self.data_retention
        }
