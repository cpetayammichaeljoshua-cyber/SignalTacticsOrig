
#!/usr/bin/env python3
"""
Bot Diagnostic and Fix Script
Identifies and fixes common issues in the Ultimate Trading Bot
"""

import os
import sys
from pathlib import Path
import importlib.util
import ast
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BotDiagnostic:
    """Comprehensive bot diagnostic and fix utility"""
    
    def __init__(self):
        self.issues_found = []
        self.fixes_applied = []
    
    def check_file_syntax(self, file_path: Path) -> bool:
        """Check Python file for syntax errors"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            ast.parse(content)
            logger.info(f"✅ Syntax check passed: {file_path}")
            return True
            
        except SyntaxError as e:
            logger.error(f"❌ Syntax error in {file_path}: Line {e.lineno}: {e.msg}")
            self.issues_found.append(f"Syntax error in {file_path}: Line {e.lineno}: {e.msg}")
            return False
        except Exception as e:
            logger.error(f"❌ Error checking {file_path}: {e}")
            return False
    
    def check_imports(self, file_path: Path) -> bool:
        """Check if all imports are available"""
        try:
            spec = importlib.util.spec_from_file_location("test_module", file_path)
            if spec is None:
                return False
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            logger.info(f"✅ Import check passed: {file_path}")
            return True
            
        except ImportError as e:
            logger.warning(f"⚠️ Import issue in {file_path}: {e}")
            self.issues_found.append(f"Import issue in {file_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Error checking imports in {file_path}: {e}")
            return False
    
    def check_environment_variables(self):
        """Check required environment variables"""
        required_vars = [
            'TELEGRAM_BOT_TOKEN',
            'TARGET_CHANNEL'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            logger.warning(f"⚠️ Missing environment variables: {missing_vars}")
            self.issues_found.append(f"Missing environment variables: {missing_vars}")
            return False
        
        logger.info("✅ Environment variables check passed")
        return True
    
    def run_full_diagnostic(self):
        """Run comprehensive diagnostic"""
        logger.info("🔍 Starting comprehensive bot diagnostic...")
        
        # Check critical files
        critical_files = [
            Path("SignalMaestro/ultimate_trading_bot.py"),
            Path("SignalMaestro/advanced_order_flow_scalping_strategy.py"),
            Path("SignalMaestro/enhanced_order_flow_integration.py"),
            Path("start_production_ultimate_bot.py")
        ]
        
        syntax_ok = True
        for file_path in critical_files:
            if file_path.exists():
                if not self.check_file_syntax(file_path):
                    syntax_ok = False
            else:
                logger.error(f"❌ Critical file missing: {file_path}")
                self.issues_found.append(f"Critical file missing: {file_path}")
        
        # Check imports (only if syntax is OK)
        if syntax_ok:
            for file_path in critical_files:
                if file_path.exists():
                    self.check_imports(file_path)
        
        # Check environment
        self.check_environment_variables()
        
        # Check directories
        required_dirs = [
            Path("SignalMaestro/ml_models"),
            Path("logs")
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                logger.info(f"📁 Creating directory: {dir_path}")
                dir_path.mkdir(parents=True, exist_ok=True)
                self.fixes_applied.append(f"Created directory: {dir_path}")
        
        # Summary
        logger.info("=" * 60)
        logger.info(f"📊 Diagnostic Summary:")
        logger.info(f"Issues found: {len(self.issues_found)}")
        logger.info(f"Fixes applied: {len(self.fixes_applied)}")
        
        if self.issues_found:
            logger.warning("⚠️ Issues found:")
            for issue in self.issues_found:
                logger.warning(f"  - {issue}")
        else:
            logger.info("✅ No critical issues found")
        
        if self.fixes_applied:
            logger.info("🔧 Fixes applied:")
            for fix in self.fixes_applied:
                logger.info(f"  - {fix}")
        
        return len(self.issues_found) == 0

if __name__ == "__main__":
    print("🔍 Ultimate Trading Bot Diagnostic Tool")
    print("=" * 50)
    
    diagnostic = BotDiagnostic()
    success = diagnostic.run_full_diagnostic()
    
    if success:
        print("\n✅ Diagnostic completed successfully - Bot ready for production!")
        sys.exit(0)
    else:
        print(f"\n❌ Diagnostic found {len(diagnostic.issues_found)} issues - Please review and fix")
        sys.exit(1)
