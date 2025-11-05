#!/bin/bash
# Local uptime monitoring script
# Generated on 2025-11-05T05:25:47.739012

ENDPOINTS=(
    "https://workspace.cagampangcristi.repl.co/health"
    "https://workspace.cagampangcristi.repl.co/ping"
    "https://workspace.cagampangcristi.repl.co/keepalive"
)

echo "🔍 Testing Trading Bot Endpoints..."
echo "Time: $(date)"
echo "----------------------------------------"

for endpoint in "${ENDPOINTS[@]}"; do
    echo -n "Testing $endpoint: "
    if curl -f -s "$endpoint" > /dev/null; then
        echo "✅ OK"
    else
        echo "❌ FAILED"
    fi
done

echo "----------------------------------------"
