
#!/usr/bin/env python3
"""
Setup script for uptime monitoring
Configures external ping services and provides setup instructions
"""

import os
import requests
import json
from datetime import datetime

class UptimeSetup:
    def __init__(self):
        self.repl_name = os.getenv('REPL_SLUG', 'trading-bot')
        self.repl_owner = os.getenv('REPL_OWNER', 'user')
        self.replit_url = f"https://{self.repl_name}.{self.repl_owner}.repl.co"
        
    def print_setup_instructions(self):
        """Print setup instructions for various uptime services"""
        print("🚀 Uptime Monitoring Setup Instructions")
        print("=" * 50)
        
        print(f"📡 Your Repl URL: {self.replit_url}")
        print(f"🏓 Ping Endpoint: {self.replit_url}/ping")
        print(f"❤️ Health Check: {self.replit_url}/health")
        print(f"⏰ Keep-Alive: {self.replit_url}/keepalive")
        
        print("\n🔧 External Service Setup:")
        print("-" * 30)
        
        print("1. KAFFEINE (Free Heroku app keep-alive)")
        print(f"   • Visit: https://kaffeine.herokuapp.com/")
        print(f"   • Add URL: {self.replit_url}/keepalive")
        print(f"   • Interval: 5 minutes")
        
        print("\n2. UPTIME ROBOT (Free tier available)")
        print(f"   • Visit: https://uptimerobot.com/")
        print(f"   • Add HTTP(S) monitor: {self.replit_url}/health")
        print(f"   • Check interval: 5 minutes")
        
        print("\n3. PINGDOM (Free tier available)")
        print(f"   • Visit: https://www.pingdom.com/")
        print(f"   • Add uptime check: {self.replit_url}/ping")
        print(f"   • Check interval: 5 minutes")
        
        print("\n4. FRESHPING (Free)")
        print(f"   • Visit: https://www.freshworks.com/website-monitoring/")
        print(f"   • Add check: {self.replit_url}/health")
        
        print("\n💡 Pro Tips:")
        print("   • Use multiple services for redundancy")
        print("   • Set up email/SMS alerts for downtime")
        print("   • Monitor response time trends")
        
    def test_endpoints(self):
        """Test if endpoints are accessible"""
        print(f"\n🧪 Testing endpoints...")
        
        endpoints = [
            ('/health', 'Health Check'),
            ('/ping', 'Ping'),
            ('/keepalive', 'Keep-Alive')
        ]
        
        for endpoint, name in endpoints:
            try:
                response = requests.get(f"{self.replit_url}{endpoint}", timeout=10)
                if response.status_code == 200:
                    print(f"   ✅ {name}: Working")
                else:
                    print(f"   ❌ {name}: Error {response.status_code}")
            except Exception as e:
                print(f"   ❌ {name}: Failed ({str(e)})")
    
    def generate_curl_commands(self):
        """Generate curl commands for testing"""
        print(f"\n🔧 Test Commands:")
        print(f"curl {self.replit_url}/health")
        print(f"curl {self.replit_url}/ping") 
        print(f"curl {self.replit_url}/keepalive")
        
    def run_setup(self):
        """Run the complete setup"""
        self.print_setup_instructions()
        self.test_endpoints()
        self.generate_curl_commands()
        
        print(f"\n✅ Setup complete!")
        print(f"🎯 Your bot should now stay alive with external pings")

if __name__ == "__main__":
    setup = UptimeSetup()
    setup.run_setup()
