
#!/bin/bash

echo "🚂 Starting Railway Deployment..."

# Run the Railway deployment fixer
python railway_deployment_fixer.py

# Run the comprehensive error fixer
python dynamic_comprehensive_error_fixer.py

# Run health check
python bot_health_check.py

# Start the main bot
python start_ultimate_fxsusdt_unified.py
