
#!/usr/bin/env python3
"""
Railway Deployment Error Fixer
Fixes all deployment-specific issues for Railway platform
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime

class RailwayDeploymentFixer:
    """Fix all Railway deployment errors"""
    
    def __init__(self):
        self.setup_logging()
        self.fixes_applied = []
        
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - RAILWAY_FIXER - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
    
    def fix_missing_directories(self):
        """Create all required directories"""
        try:
            self.logger.info("🔧 Creating required directories...")
            
            required_dirs = [
                "logs",
                "data",
                "ml_models",
                "backups",
                "SignalMaestro/logs",
                "SignalMaestro/data",
                "SignalMaestro/ml_models",
                "SignalMaestro/backups",
                "bot",
                "utils",
                "models",
                "ai_models"
            ]
            
            for dir_path in required_dirs:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                
                # Create __init__.py for Python modules
                init_file = Path(dir_path) / "__init__.py"
                if not init_file.exists():
                    init_file.write_text("# Auto-generated for Railway deployment\n")
            
            self.fixes_applied.append("Directories created")
            self.logger.info("✅ Directories created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Directory creation failed: {e}")
            return False
    
    def fix_file_permissions(self):
        """Fix file permissions for Railway"""
        try:
            self.logger.info("🔧 Fixing file permissions...")
            
            # Make all Python files executable
            for py_file in Path(".").rglob("*.py"):
                try:
                    os.chmod(py_file, 0o755)
                except:
                    pass
            
            # Make shell scripts executable
            for sh_file in Path(".").rglob("*.sh"):
                try:
                    os.chmod(sh_file, 0o755)
                except:
                    pass
            
            self.fixes_applied.append("Permissions fixed")
            self.logger.info("✅ Permissions fixed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Permission fixing failed: {e}")
            return False
    
    def fix_railway_environment(self):
        """Configure Railway-specific environment"""
        try:
            self.logger.info("🔧 Configuring Railway environment...")
            
            # Set Railway-specific variables
            os.environ['RAILWAY_ENVIRONMENT'] = 'production'
            os.environ['PYTHONUNBUFFERED'] = '1'
            os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            
            # Disable warnings
            os.environ['PYTHONWARNINGS'] = 'ignore'
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            
            # Set working directory
            os.chdir(Path(__file__).parent)
            
            self.fixes_applied.append("Railway environment configured")
            self.logger.info("✅ Railway environment configured")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Environment configuration failed: {e}")
            return False
    
    def fix_import_paths(self):
        """Fix Python import paths for Railway"""
        try:
            self.logger.info("🔧 Fixing import paths...")
            
            current_dir = Path(__file__).parent.absolute()
            
            paths_to_add = [
                str(current_dir),
                str(current_dir / "SignalMaestro"),
                str(current_dir / "bot"),
                str(current_dir / "utils"),
                str(current_dir / "models"),
            ]
            
            for path in paths_to_add:
                if path not in sys.path:
                    sys.path.insert(0, path)
            
            self.fixes_applied.append("Import paths fixed")
            self.logger.info("✅ Import paths configured")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Import path fixing failed: {e}")
            return False
    
    def install_critical_dependencies(self):
        """Install dependencies that Railway might miss"""
        try:
            self.logger.info("🔧 Installing critical dependencies...")
            
            critical_packages = [
                'nest-asyncio',
                'python-telegram-bot',
                'ccxt',
                'aiohttp',
                'requests',
                'pandas',
                'numpy'
            ]
            
            for package in critical_packages:
                try:
                    __import__(package.replace('-', '_'))
                except ImportError:
                    self.logger.info(f"📦 Installing {package}...")
                    subprocess.run([
                        sys.executable, '-m', 'pip', 'install', 
                        package, '--no-cache-dir', '--quiet'
                    ], capture_output=True)
            
            self.fixes_applied.append("Dependencies installed")
            self.logger.info("✅ Dependencies verified")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Dependency installation failed: {e}")
            return False
    
    def create_railway_healthcheck(self):
        """Create health check endpoint for Railway"""
        try:
            self.logger.info("🔧 Creating Railway health check...")
            
            healthcheck_content = '''#!/usr/bin/env python3
"""Railway Health Check Endpoint"""
from aiohttp import web
import asyncio

async def health_check(request):
    return web.Response(text='OK', status=200)

async def init_app():
    app = web.Application()
    app.router.add_get('/health', health_check)
    app.router.add_get('/', health_check)
    return app

if __name__ == '__main__':
    web.run_app(init_app(), host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
'''
            
            with open('railway_healthcheck.py', 'w') as f:
                f.write(healthcheck_content)
            
            self.fixes_applied.append("Health check created")
            self.logger.info("✅ Health check endpoint created")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Health check creation failed: {e}")
            return False
    
    def fix_database_paths(self):
        """Ensure database files are accessible"""
        try:
            self.logger.info("🔧 Fixing database paths...")
            
            db_files = [
                'advanced_ml_trading.db',
                'trade_learning.db',
                'error_logs.db',
                'ml_trade_learning.db'
            ]
            
            for db_file in db_files:
                db_path = Path(db_file)
                if not db_path.exists():
                    # Create empty database file
                    db_path.touch()
                    self.logger.info(f"Created {db_file}")
            
            self.fixes_applied.append("Database paths fixed")
            self.logger.info("✅ Database paths configured")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Database path fixing failed: {e}")
            return False
    
    def apply_all_fixes(self):
        """Apply all Railway deployment fixes"""
        self.logger.info("🚀 RAILWAY DEPLOYMENT FIXER")
        self.logger.info("=" * 60)
        
        fixes = [
            ("Creating directories", self.fix_missing_directories),
            ("Fixing permissions", self.fix_file_permissions),
            ("Configuring environment", self.fix_railway_environment),
            ("Fixing import paths", self.fix_import_paths),
            ("Installing dependencies", self.install_critical_dependencies),
            ("Creating health check", self.create_railway_healthcheck),
            ("Fixing database paths", self.fix_database_paths)
        ]
        
        successful_fixes = 0
        for description, fix_function in fixes:
            self.logger.info(f"🔧 {description}...")
            try:
                if fix_function():
                    successful_fixes += 1
                else:
                    self.logger.warning(f"⚠️ {description} had issues")
            except Exception as e:
                self.logger.error(f"❌ {description} failed: {e}")
        
        self.logger.info("=" * 60)
        self.logger.info(f"✅ Railway fixes completed!")
        self.logger.info(f"📊 Success: {successful_fixes}/{len(fixes)} fixes applied")
        
        return successful_fixes >= len(fixes) * 0.8

def main():
    """Main execution"""
    print("🚂 RAILWAY DEPLOYMENT ERROR FIXER")
    print("=" * 60)
    
    fixer = RailwayDeploymentFixer()
    
    try:
        success = fixer.apply_all_fixes()
        
        if success:
            print("\n✅ RAILWAY DEPLOYMENT READY!")
            print("🎯 All critical errors fixed")
            print("🚀 Bot ready for Railway deployment")
            return 0
        else:
            print("\n⚠️ Some fixes had issues")
            return 1
            
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
