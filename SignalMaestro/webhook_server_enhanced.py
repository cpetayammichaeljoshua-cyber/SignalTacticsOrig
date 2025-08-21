
#!/usr/bin/env python3
"""
Enhanced Webhook Server for Trading Signal Processing
Handles incoming signals and forwards to the enhanced bot
"""

from flask import Flask, request, jsonify
import asyncio
import logging
import json
from datetime import datetime
from threading import Thread
import os

from enhanced_perfect_scalping_bot_v2 import EnhancedPerfectScalpingBotV2

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global bot instance
trading_bot = None
bot_loop = None

def run_bot_in_background():
    """Run the trading bot in background asyncio loop"""
    global trading_bot, bot_loop
    
    try:
        bot_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(bot_loop)
        
        trading_bot = EnhancedPerfectScalpingBotV2()
        bot_loop.run_until_complete(trading_bot.start())
        
    except Exception as e:
        logger.error(f"Bot background error: {e}")

@app.route('/webhook', methods=['POST'])
async def handle_webhook():
    """Handle incoming webhook signals"""
    try:
        data = request.get_json()
        logger.info(f"Received webhook: {data}")
        
        if not trading_bot:
            return jsonify({'error': 'Bot not initialized'}), 500
        
        # Process signal asynchronously
        if bot_loop:
            asyncio.run_coroutine_threadsafe(
                trading_bot.process_signal(data),
                bot_loop
            )
        
        return jsonify({
            'status': 'success',
            'message': 'Signal received and processing',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/signal', methods=['POST'])
async def handle_manual_signal():
    """Handle manual signal input"""
    try:
        data = request.get_json()
        signal_text = data.get('signal', '')
        
        if not signal_text:
            return jsonify({'error': 'No signal provided'}), 400
        
        logger.info(f"Manual signal: {signal_text}")
        
        if trading_bot and bot_loop:
            asyncio.run_coroutine_threadsafe(
                trading_bot.process_signal(signal_text),
                bot_loop
            )
        
        return jsonify({
            'status': 'success',
            'message': 'Manual signal processed',
            'signal': signal_text
        })
        
    except Exception as e:
        logger.error(f"Manual signal error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
async def get_status():
    """Get bot status"""
    try:
        if not trading_bot:
            return jsonify({'status': 'bot_not_initialized'})
        
        # Get status from bot
        if bot_loop:
            future = asyncio.run_coroutine_threadsafe(
                trading_bot.get_status_report(),
                bot_loop
            )
            status = future.result(timeout=5.0)
        else:
            status = {'status': 'loop_not_running'}
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Status error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'bot_running': trading_bot is not None and trading_bot.running
    })

@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'name': 'Enhanced Perfect Scalping Bot V2',
        'version': '2.0',
        'status': 'running',
        'endpoints': {
            '/webhook': 'POST - Receive trading signals',
            '/signal': 'POST - Manual signal input',
            '/status': 'GET - Bot status',
            '/health': 'GET - Health check'
        }
    })

def initialize_app():
    """Initialize the Flask app and bot"""
    # Start bot in background thread
    bot_thread = Thread(target=run_bot_in_background, daemon=True)
    bot_thread.start()
    
    logger.info("🚀 Enhanced webhook server initialized")

if __name__ == '__main__':
    initialize_app()
    
    # Run Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
