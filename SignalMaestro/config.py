"""
Configuration management for the trading bot
Handles environment variables and default settings
"""

import os
from typing import Dict, Any

class Config:
    """Configuration class for managing bot settings"""

    def __init__(self):
        # Telegram Configuration
        self.TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

        # Binance Configuration
        self.BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
        self.BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

        # Use mainnet by default for live trading
        self.BINANCE_TESTNET = os.getenv("BINANCE_TESTNET", "false").lower() == "true"

        # Session Secret
        self.SESSION_SECRET = os.getenv(
            "SESSION_SECRET",
            "snfpdA4kx+l9/+w0qMi3DzmND+CqQx/XUG0f4irkipuqmxNH++NpQ738XBdURGSuM574qS0iQsabmf6vmiwa2g=="
        )

        # Trading Configuration
        self.DEFAULT_RISK_PERCENTAGE = float(os.getenv("DEFAULT_RISK_PERCENTAGE", "2.0"))
        self.MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", "1000.0"))
        self.MIN_POSITION_SIZE = float(os.getenv("MIN_POSITION_SIZE", "10.0"))
        self.STOP_LOSS_PERCENTAGE = float(os.getenv("STOP_LOSS_PERCENTAGE", "3.0"))
        self.TAKE_PROFIT_PERCENTAGE = float(os.getenv("TAKE_PROFIT_PERCENTAGE", "6.0"))

        # Cornix Configuration
        self.CORNIX_WEBHOOK_URL = os.getenv("CORNIX_WEBHOOK_URL", "https://dashboard.cornix.io/tradingview/")
        self.CORNIX_BOT_UUID = os.getenv("CORNIX_BOT_UUID", "")

        # Database Configuration
        self.DATABASE_PATH = os.getenv("DATABASE_PATH", "trading_bot.db")

        # Server Configuration
        self.WEBHOOK_HOST = os.getenv("WEBHOOK_HOST", "0.0.0.0")
        self.WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", "5000"))

        # Logging Configuration
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FILE = os.getenv("LOG_FILE", "trading_bot.log")

        # Trading Pairs Configuration
        self.SUPPORTED_PAIRS = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOTUSDT",
            "LINKUSDT", "LTCUSDT", "BCHUSDT", "XLMUSDT", "EOSUSDT",
            "TRXUSDT", "XRPUSDT", "SOLUSDT", "AVAXUSDT", "MATICUSDT"
        ]

        # API Rate Limits
        self.BINANCE_REQUEST_TIMEOUT = int(os.getenv("BINANCE_REQUEST_TIMEOUT", "30"))
        self.MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
        self.RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1.0"))

        # Signal Processing
        self.SIGNAL_VALIDATION_ENABLED = os.getenv("SIGNAL_VALIDATION_ENABLED", "true").lower() == "true"
        self.AUTO_TRADE_ENABLED = os.getenv("AUTO_TRADE_ENABLED", "false").lower() == "true"

        # Security Settings
        self.AUTHORIZED_USERS = self._parse_authorized_users()
        self.ADMIN_USER_ID = os.getenv("ADMIN_USER_ID", "")
        self.ADMIN_USER_NAME = os.getenv("ADMIN_USER_NAME", "Trading Bot Admin")

        # Webhook Security
        self.WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")

    def _parse_authorized_users(self) -> list:
        """Parse authorized user IDs from environment variable"""
        users_str = os.getenv("AUTHORIZED_USERS", "")
        if not users_str:
            return []

        try:
            return [int(user_id.strip()) for user_id in users_str.split(",") if user_id.strip()]
        except ValueError:
            return []

    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading configuration as dictionary"""
        return {
            "default_risk_percentage": self.DEFAULT_RISK_PERCENTAGE,
            "max_position_size": self.MAX_POSITION_SIZE,
            "min_position_size": self.MIN_POSITION_SIZE,
            "stop_loss_percentage": self.STOP_LOSS_PERCENTAGE,
            "take_profit_percentage": self.TAKE_PROFIT_PERCENTAGE,
            "supported_pairs": self.SUPPORTED_PAIRS,
            "signal_validation_enabled": self.SIGNAL_VALIDATION_ENABLED,
            "auto_trade_enabled": self.AUTO_TRADE_ENABLED
        }

    def is_authorized_user(self, user_id: int) -> bool:
        """Check if user is authorized to use the bot"""
        if not self.AUTHORIZED_USERS:
            return True  # Allow all users if no restrictions set
        return user_id in self.AUTHORIZED_USERS

    def is_admin_user(self, user_id: int) -> bool:
        """Check if user is an admin"""
        if not self.ADMIN_USER_ID:
            return False
        return str(user_id) == self.ADMIN_USER_ID

    def validate_config(self) -> bool:
        """Validate that all required configuration is present"""
        required_fields = [
            "TELEGRAM_BOT_TOKEN",
            "BINANCE_API_KEY", 
            "BINANCE_API_SECRET"
        ]

        for field in required_fields:
            value = getattr(self, field, None)
            if not value or value == "":
                raise ValueError(f"Required configuration field '{field}' is missing or empty")

        return True
"""
Configuration management for Enhanced Perfect Scalping Bot V2
Handles environment variables and default settings
"""

import os
from typing import Optional

class Config:
    """Configuration class with environment variable support"""
    
    # Telegram Bot Configuration
    TELEGRAM_BOT_TOKEN: str = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID: str = os.getenv('TELEGRAM_CHAT_ID', '')
    
    # Trading Configuration
    BINANCE_API_KEY: str = os.getenv('BINANCE_API_KEY', '')
    BINANCE_SECRET_KEY: str = os.getenv('BINANCE_SECRET_KEY', '')
    BINANCE_TESTNET: bool = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
    
    # Cornix Configuration
    CORNIX_API_KEY: str = os.getenv('CORNIX_API_KEY', '')
    CORNIX_BOT_ID: str = os.getenv('CORNIX_BOT_ID', '')
    
    # Server Configuration
    WEBHOOK_HOST: str = os.getenv('WEBHOOK_HOST', '0.0.0.0')
    WEBHOOK_PORT: int = int(os.getenv('WEBHOOK_PORT', '5000'))
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE: str = os.getenv('LOG_FILE', 'logs/trading_bot.log')
    
    # Trading Parameters
    DEFAULT_RISK_PERCENT: float = float(os.getenv('DEFAULT_RISK_PERCENT', '2.0'))
    MAX_CONCURRENT_TRADES: int = int(os.getenv('MAX_CONCURRENT_TRADES', '5'))
    
    # Rate Limiting
    MESSAGE_RATE_LIMIT: int = int(os.getenv('MESSAGE_RATE_LIMIT', '3'))
    RATE_LIMIT_WINDOW: int = int(os.getenv('RATE_LIMIT_WINDOW', '3600'))
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate essential configuration"""
        required_fields = []
        
        # Only check for Telegram token as minimum requirement
        if not cls.TELEGRAM_BOT_TOKEN:
            print("Warning: TELEGRAM_BOT_TOKEN not set")
            
        return True  # Allow bot to run with minimal config
    
    @classmethod
    def get_config_summary(cls) -> dict:
        """Get configuration summary (without sensitive data)"""
        return {
            'telegram_configured': bool(cls.TELEGRAM_BOT_TOKEN),
            'binance_configured': bool(cls.BINANCE_API_KEY),
            'cornix_configured': bool(cls.CORNIX_API_KEY),
            'webhook_host': cls.WEBHOOK_HOST,
            'webhook_port': cls.WEBHOOK_PORT,
            'log_level': cls.LOG_LEVEL,
            'testnet_mode': cls.BINANCE_TESTNET
        }
#!/usr/bin/env python3
"""
Configuration Module
"""

import os
from typing import Optional

class Config:
    """Configuration class for trading bot"""
    
    def __init__(self):
        # Telegram Configuration
        self.TELEGRAM_BOT_TOKEN: Optional[str] = os.getenv('TELEGRAM_BOT_TOKEN')
        self.TELEGRAM_CHAT_ID: Optional[str] = os.getenv('TELEGRAM_CHAT_ID') or '@TradeTactics_bot'
        
        # Trading Configuration
        self.BINANCE_API_KEY: Optional[str] = os.getenv('BINANCE_API_KEY')
        self.BINANCE_API_SECRET: Optional[str] = os.getenv('BINANCE_API_SECRET')
        
        # Cornix Configuration
        self.CORNIX_API_KEY: Optional[str] = os.getenv('CORNIX_API_KEY')
        self.CORNIX_WEBHOOK_URL: Optional[str] = os.getenv('CORNIX_WEBHOOK_URL')
        
        # Bot Settings
        self.MAX_CONCURRENT_TRADES: int = 3
        self.DEFAULT_RISK_PERCENTAGE: float = 2.0
        self.DEFAULT_LEVERAGE: int = 50
        self.MARGIN_TYPE: str = "cross"
        
        # Rate Limiting
        self.MAX_MESSAGES_PER_HOUR: int = 3
        self.MIN_TRADE_INTERVAL_SECONDS: int = 900  # 15 minutes
        
    def validate(self) -> bool:
        """Validate required configuration"""
        if not self.TELEGRAM_BOT_TOKEN:
            print("Missing required environment variables:")
            print("- TELEGRAM_BOT_TOKEN (Required)")
            print("Please set TELEGRAM_BOT_TOKEN in the Secrets tab")
            return False
            
        # TELEGRAM_CHAT_ID is optional, will default to @TradeTactics_bot
        if not self.TELEGRAM_CHAT_ID:
            self.TELEGRAM_CHAT_ID = '@TradeTactics_bot'
            
        return True
