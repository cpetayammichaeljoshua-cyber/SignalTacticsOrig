
#!/usr/bin/env python3
"""
Comprehensive Bot Monitoring and Auto-Fix Script
Monitors the Ultimate Trading Bot and automatically resolves common issues
"""

import asyncio
import logging
import os
import json
import time
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class BotMonitorAndFix:
    """Comprehensive bot monitoring and auto-fix system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID', '@SignalTactics')
        
        # Monitoring thresholds
        self.max_consecutive_failures = 5
        self.max_rate_limit_warnings = 10
        self.min_signals_per_hour = 1
        
        # Auto-fix actions
        self.auto_fixes_applied = []
        self.monitoring_start_time = datetime.now()
        
    async def monitor_bot_health(self):
        """Monitor bot health and apply fixes automatically"""
        self.logger.info("🔍 Starting comprehensive bot monitoring...")
        
        issues_detected = []
        fixes_applied = []
        
        # Check 1: Telegram connectivity
        telegram_ok = await self._check_telegram_connection()
        if not telegram_ok:
            issues_detected.append("Telegram connectivity issue")
            fix_result = await self._fix_telegram_connection()
            if fix_result:
                fixes_applied.append("Telegram connection restored")
        
        # Check 2: Environment variables
        env_issues = await self._check_environment_variables()
        if env_issues:
            issues_detected.extend(env_issues)
            env_fixes = await self._fix_environment_issues(env_issues)
            fixes_applied.extend(env_fixes)
        
        # Check 3: Rate limiting issues
        rate_limit_issues = await self._check_rate_limiting()
        if rate_limit_issues:
            issues_detected.extend(rate_limit_issues)
            rate_fixes = await self._fix_rate_limiting()
            fixes_applied.extend(rate_fixes)
        
        # Check 4: Signal generation performance
        signal_issues = await self._check_signal_performance()
        if signal_issues:
            issues_detected.extend(signal_issues)
            signal_fixes = await self._optimize_signal_generation()
            fixes_applied.extend(signal_fixes)
        
        # Generate comprehensive report
        await self._generate_health_report(issues_detected, fixes_applied)
        
        return len(issues_detected) == 0, fixes_applied
    
    async def _check_telegram_connection(self) -> bool:
        """Check Telegram API connectivity"""
        try:
            if not self.bot_token:
                self.logger.error("❌ TELEGRAM_BOT_TOKEN not found")
                return False
            
            url = f"https://api.telegram.org/bot{self.bot_token}/getMe"
            timeout = aiohttp.ClientTimeout(total=10)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        bot_info = await response.json()
                        self.logger.info(f"✅ Telegram connection OK: @{bot_info['result']['username']}")
                        return True
                    else:
                        self.logger.error(f"❌ Telegram API error: {response.status}")
                        return False
        
        except Exception as e:
            self.logger.error(f"❌ Telegram connection check failed: {e}")
            return False
    
    async def _fix_telegram_connection(self) -> bool:
        """Attempt to fix Telegram connection issues"""
        self.logger.info("🔧 Attempting to fix Telegram connection...")
        
        try:
            # Send a test message to validate full connectivity
            test_message = f"🔧 <b>Bot Health Check</b>\n\n✅ Connection restored at {datetime.now().strftime('%H:%M UTC')}\n🔄 Monitoring active"
            
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': test_message,
                'parse_mode': 'HTML'
            }
            
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        self.logger.info("✅ Telegram connection fixed and validated")
                        return True
                    else:
                        self.logger.error(f"❌ Test message failed: {response.status}")
                        return False
        
        except Exception as e:
            self.logger.error(f"❌ Failed to fix Telegram connection: {e}")
            return False
    
    async def _check_environment_variables(self) -> List[str]:
        """Check critical environment variables"""
        issues = []
        
        critical_vars = {
            'TELEGRAM_BOT_TOKEN': 'Missing Telegram bot token',
            'TARGET_CHANNEL': 'Missing target channel',
            'TELEGRAM_CHAT_ID': 'Missing chat ID'
        }
        
        for var, description in critical_vars.items():
            if not os.getenv(var):
                issues.append(f"{description} ({var})")
                self.logger.warning(f"⚠️ {description}")
        
        # Check configurable parameters
        config_vars = {
            'MAX_MESSAGES_PER_HOUR': '8',
            'MIN_TRADE_INTERVAL_SECONDS': '120',
            'ORDER_FLOW_MIN_SIGNAL_STRENGTH': '78'
        }
        
        for var, default in config_vars.items():
            current_value = os.getenv(var, default)
            try:
                float(current_value)
            except ValueError:
                issues.append(f"Invalid {var}: {current_value}")
                self.logger.warning(f"⚠️ Invalid {var}: {current_value}")
        
        return issues
    
    async def _fix_environment_issues(self, issues: List[str]) -> List[str]:
        """Fix environment variable issues"""
        fixes = []
        
        # Apply default values for missing variables
        defaults = {
            'TARGET_CHANNEL': '@SignalTactics',
            'TELEGRAM_CHAT_ID': '@SignalTactics',
            'MAX_MESSAGES_PER_HOUR': '8',
            'MIN_TRADE_INTERVAL_SECONDS': '120',
            'ORDER_FLOW_MIN_SIGNAL_STRENGTH': '78',
            'DEFAULT_LEVERAGE': '35'
        }
        
        for var, default_value in defaults.items():
            if not os.getenv(var) or f"Missing" in str(issues) and var in str(issues):
                os.environ[var] = default_value
                fixes.append(f"Set {var} to {default_value}")
                self.logger.info(f"🔧 Set {var} = {default_value}")
        
        return fixes
    
    async def _check_rate_limiting(self) -> List[str]:
        """Check for rate limiting issues"""
        issues = []
        
        try:
            # Check current rate limit settings
            max_messages = int(os.getenv('MAX_MESSAGES_PER_HOUR', '8'))
            min_interval = int(os.getenv('MIN_TRADE_INTERVAL_SECONDS', '120'))
            
            if max_messages < 4:
                issues.append(f"Rate limit too restrictive: {max_messages}/hour")
            
            if min_interval > 300:  # 5 minutes
                issues.append(f"Trade interval too long: {min_interval}s")
        
        except Exception as e:
            issues.append(f"Rate limit configuration error: {e}")
        
        return issues
    
    async def _fix_rate_limiting(self) -> List[str]:
        """Fix rate limiting issues"""
        fixes = []
        
        # Optimize rate limiting for production
        optimal_settings = {
            'MAX_MESSAGES_PER_HOUR': '8',
            'MIN_TRADE_INTERVAL_SECONDS': '120'
        }
        
        for var, value in optimal_settings.items():
            current = os.getenv(var)
            if current != value:
                os.environ[var] = value
                fixes.append(f"Optimized {var}: {current} -> {value}")
                self.logger.info(f"🔧 Optimized {var} = {value}")
        
        return fixes
    
    async def _check_signal_performance(self) -> List[str]:
        """Check signal generation performance"""
        issues = []
        
        # This would normally check actual performance metrics
        # For now, we'll check configuration that affects performance
        
        try:
            min_strength = float(os.getenv('ORDER_FLOW_MIN_SIGNAL_STRENGTH', '78'))
            
            if min_strength > 85:
                issues.append(f"Signal strength threshold too high: {min_strength}%")
            elif min_strength < 70:
                issues.append(f"Signal strength threshold too low: {min_strength}%")
        
        except Exception as e:
            issues.append(f"Signal configuration error: {e}")
        
        return issues
    
    async def _optimize_signal_generation(self) -> List[str]:
        """Optimize signal generation settings"""
        fixes = []
        
        # Optimal signal generation settings
        optimal_signal_settings = {
            'ORDER_FLOW_MIN_SIGNAL_STRENGTH': '78'  # Balanced threshold
        }
        
        for var, value in optimal_signal_settings.items():
            current = os.getenv(var)
            if current != value:
                os.environ[var] = value
                fixes.append(f"Optimized signal {var}: {current} -> {value}")
                self.logger.info(f"🔧 Optimized signal {var} = {value}")
        
        return fixes
    
    async def _generate_health_report(self, issues: List[str], fixes: List[str]):
        """Generate and send comprehensive health report"""
        try:
            current_time = datetime.now()
            uptime = current_time - self.monitoring_start_time
            
            # Create health report
            if issues:
                status_emoji = "🚨" if len(issues) > 3 else "⚠️"
                status = "ISSUES DETECTED"
            else:
                status_emoji = "✅"
                status = "HEALTHY"
            
            report = f"""{status_emoji} <b>BOT HEALTH REPORT</b>

🕐 <b>Report Time:</b> {current_time.strftime('%Y-%m-%d %H:%M UTC')}
⏱️ <b>Monitoring Uptime:</b> {str(uptime).split('.')[0]}
🔍 <b>Status:</b> {status}

"""
            
            if issues:
                report += f"🚨 <b>ISSUES DETECTED ({len(issues)}):</b>\n"
                for i, issue in enumerate(issues[:5], 1):  # Limit to 5 issues
                    report += f"   {i}. {issue}\n"
                if len(issues) > 5:
                    report += f"   ... and {len(issues) - 5} more\n"
                report += "\n"
            
            if fixes:
                report += f"🔧 <b>AUTO-FIXES APPLIED ({len(fixes)}):</b>\n"
                for i, fix in enumerate(fixes[:5], 1):  # Limit to 5 fixes
                    report += f"   {i}. {fix}\n"
                if len(fixes) > 5:
                    report += f"   ... and {len(fixes) - 5} more\n"
                report += "\n"
            
            # Add configuration summary
            report += f"""📊 <b>CURRENT CONFIGURATION:</b>
• Max Signals/Hour: {os.getenv('MAX_MESSAGES_PER_HOUR', '8')}
• Min Trade Interval: {os.getenv('MIN_TRADE_INTERVAL_SECONDS', '120')}s
• Signal Strength Min: {os.getenv('ORDER_FLOW_MIN_SIGNAL_STRENGTH', '78')}%
• Default Leverage: {os.getenv('DEFAULT_LEVERAGE', '35')}x

🔄 <b>Next Check:</b> 15 minutes

<i>Automated monitoring and fixes by Production Bot Health System</i>"""
            
            # Send report
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': report,
                'parse_mode': 'HTML'
            }
            
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        self.logger.info("📊 Health report sent successfully")
                    else:
                        self.logger.warning(f"⚠️ Failed to send health report: {response.status}")
        
        except Exception as e:
            self.logger.error(f"❌ Error generating health report: {e}")

async def main():
    """Main monitoring function"""
    monitor = BotMonitorAndFix()
    
    try:
        print("🔍 Starting Bot Health Monitor and Auto-Fix System")
        print("=" * 60)
        
        healthy, fixes = await monitor.monitor_bot_health()
        
        if healthy:
            print("✅ Bot is healthy - no issues detected")
        else:
            print(f"🔧 Issues detected and {len(fixes)} auto-fixes applied")
        
        print("\n📊 Monitor completed successfully")
        
    except Exception as e:
        print(f"❌ Monitor failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(main())
