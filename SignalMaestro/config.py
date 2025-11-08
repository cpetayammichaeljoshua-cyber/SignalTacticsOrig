"""
Configuration management for the trading bot
Handles environment variables and default settings with proper validation
"""

import os
from typing import Dict, Any, List, Optional

class Config:
    """Configuration class for managing bot settings"""

    def __init__(self):
        # Telegram Configuration
        self.TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
        self.TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "@SignalTactics")
        self.ADMIN_USER_ID = os.getenv("ADMIN_USER_ID", "")
        self.ADMIN_USER_NAME = os.getenv("ADMIN_USER_NAME", "Trading Bot Admin")

        # Binance Configuration
        self.BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
        self.BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
        self.BINANCE_TESTNET = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

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
        self.DEFAULT_LEVERAGE = int(os.getenv("DEFAULT_LEVERAGE", "50"))
        self.MARGIN_TYPE = os.getenv("MARGIN_TYPE", "cross")

        # Cornix Configuration
        self.CORNIX_WEBHOOK_URL = os.getenv("CORNIX_WEBHOOK_URL", "https://dashboard.cornix.io/tradingview/")
        self.CORNIX_BOT_UUID = os.getenv("CORNIX_BOT_UUID", "")
        self.CORNIX_API_KEY = os.getenv("CORNIX_API_KEY", "")
        self.CORNIX_BOT_ID = os.getenv("CORNIX_BOT_ID", "")

        # Database Configuration
        self.DATABASE_PATH = os.getenv("DATABASE_PATH", "trading_bot.db")

        # Server Configuration
        self.WEBHOOK_HOST = os.getenv("WEBHOOK_HOST", "0.0.0.0")
        self.WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", "5000"))

        # Logging Configuration
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FILE = os.getenv("LOG_FILE", "logs/trading_bot.log")

        # Trading Pairs Configuration
        self.SUPPORTED_PAIRS = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOTUSDT",
            "LINKUSDT", "LTCUSDT", "BCHUSDT", "XLMUSDT", "EOSUSDT",
            "TRXUSDT", "XRPUSDT", "SOLUSDT", "AVAXUSDT", "MATICUSDT",
            "UNIUSDT", "AAVEUSDT", "SUSHIUSDT", "COMPUSDT", "MKRUSDT"
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
        self.WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")

        # Rate Limiting
        self.MAX_MESSAGES_PER_HOUR = int(os.getenv("MAX_MESSAGES_PER_HOUR", "3"))
        self.MIN_TRADE_INTERVAL_SECONDS = int(os.getenv("MIN_TRADE_INTERVAL_SECONDS", "900"))
        self.MAX_CONCURRENT_TRADES = int(os.getenv("MAX_CONCURRENT_TRADES", "3"))

        # Ensure logs directory exists
        os.makedirs(os.path.dirname(self.LOG_FILE), exist_ok=True)

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
            "auto_trade_enabled": self.AUTO_TRADE_ENABLED,
            "default_leverage": self.DEFAULT_LEVERAGE,
            "margin_type": self.MARGIN_TYPE
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

    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return status"""
        issues = []
        warnings = []

        # Check critical fields
        if not self.TELEGRAM_BOT_TOKEN:
            issues.append("TELEGRAM_BOT_TOKEN is required")

        # Check optional but recommended fields
        if not self.BINANCE_API_KEY and self.AUTO_TRADE_ENABLED:
            warnings.append("BINANCE_API_KEY not set - auto trading disabled")

        if not self.CORNIX_API_KEY:
            warnings.append("CORNIX_API_KEY not set - Cornix integration disabled")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "telegram_configured": bool(self.TELEGRAM_BOT_TOKEN),
            "binance_configured": bool(self.BINANCE_API_KEY and self.BINANCE_API_SECRET),
            "cornix_configured": bool(self.CORNIX_API_KEY)
        }