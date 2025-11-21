# ATAS Platform Integration Guide

## Overview

This guide explains how to integrate your Python high-frequency scalping bot with the ATAS platform using the HTTP REST API bridge.

## Architecture

```
Python Trading Bot  →  REST API Server  →  ATAS C# Strategy
     (Signals)           (Port 8888)        (Executes Trades)
```

## Features

✅ **Real-time Signal Export** - Signals automatically pushed to API  
✅ **RESTful API** - Easy integration with ATAS C# strategies  
✅ **Signal History** - Last 50 active signals + 500 historical  
✅ **Symbol Filtering** - Query signals for specific markets  
✅ **Health Monitoring** - Built-in health check endpoints  

---

## Quick Start

### 1. Start the Python Bot

```bash
python start_high_frequency_scalping_bot.py
```

The ATAS API server will automatically start on `http://0.0.0.0:8888`

### 2. Verify API is Running

Open your browser or use curl:

```bash
# API Documentation
curl http://localhost:8888/

# Get all active signals
curl http://localhost:8888/api/signals

# Get latest signal
curl http://localhost:8888/api/signals/latest

# Health check
curl http://localhost:8888/api/health
```

---

## API Endpoints

### GET `/api/signals`
Retrieve all active trading signals

**Response:**
```json
{
  "status": "success",
  "count": 2,
  "signals": [
    {
      "timestamp": "2024-11-21T10:30:45.123456",
      "symbol": "ETHUSDT",
      "direction": "BUY",
      "entry_price": 3500.0,
      "stop_loss": 3482.5,
      "take_profit": 3542.0,
      "leverage": 20,
      "signal_strength": 85.5,
      "timeframe": "1m",
      "strategy_name": "HighFrequencyScalping",
      "metadata": {
        "consensus_confidence": 75.0,
        "strategies_agree": 4,
        "risk_reward_ratio": 2.4
      }
    }
  ],
  "timestamp": "2024-11-21T10:31:00"
}
```

### GET `/api/signals/latest`
Get the most recent signal

### GET `/api/signals/{symbol}`
Get signals for specific symbol (e.g., `/api/signals/ETHUSDT`)

### POST `/api/signals/acknowledge`
Mark signal as processed by ATAS

**Request:**
```json
{
  "timestamp": "2024-11-21T10:30:45.123456"
}
```

### GET `/api/health`
Health check endpoint

### GET `/api/stats`
API usage statistics

---

## ATAS C# Integration

### Example ATAS Strategy

Create a new strategy in Visual Studio:

```csharp
using System;
using System.Net.Http;
using System.Threading.Tasks;
using ATAS.Indicators;
using Newtonsoft.Json;

namespace ATAS.Indicators.Technical
{
    public class PythonSignalImporter : ChartStrategy
    {
        private readonly HttpClient _httpClient = new HttpClient();
        private const string API_URL = "http://localhost:8888/api/signals/latest";
        private string _lastSignalTimestamp = "";
        
        // Settings
        private int _checkInterval = 100;  // Check every 100 bars
        private bool _autoTrade = true;
        
        protected override void OnInitialize()
        {
            // Initialize strategy
            AddAlert("SignalAlert", "New signal received");
        }
        
        protected override void OnCalculate(int bar, decimal value)
        {
            // Only check on specific intervals
            if (bar % _checkInterval != 0)
                return;
            
            // Fetch latest signal
            var signal = FetchLatestSignal().Result;
            
            if (signal == null || signal.Timestamp == _lastSignalTimestamp)
                return;  // No new signal
            
            // New signal received
            _lastSignalTimestamp = signal.Timestamp;
            
            // Alert user
            AddAlert("SignalAlert", 
                $"New {signal.Direction} signal for {signal.Symbol} at {signal.EntryPrice}");
            
            // Auto-execute if enabled
            if (_autoTrade)
            {
                ExecuteSignal(signal);
            }
        }
        
        private void ExecuteSignal(SignalData signal)
        {
            // Calculate quantity based on risk management
            var quantity = CalculateQuantity(signal);
            
            // Place order
            if (signal.Direction == "BUY")
            {
                // Place buy market order
                TradingManager.PlaceMarketOrder(OrderSide.Buy, quantity);
                
                // Set stop loss and take profit
                TradingManager.PlaceStopOrder(OrderSide.Sell, quantity, signal.StopLoss);
                TradingManager.PlaceLimitOrder(OrderSide.Sell, quantity, signal.TakeProfit);
            }
            else if (signal.Direction == "SELL")
            {
                // Place sell market order
                TradingManager.PlaceMarketOrder(OrderSide.Sell, quantity);
                
                // Set stop loss and take profit
                TradingManager.PlaceStopOrder(OrderSide.Buy, quantity, signal.StopLoss);
                TradingManager.PlaceLimitOrder(OrderSide.Buy, quantity, signal.TakeProfit);
            }
        }
        
        private decimal CalculateQuantity(SignalData signal)
        {
            // Implement your position sizing logic
            decimal accountBalance = 10000;  // Get from account
            decimal riskPercent = 0.02m;  // 2% risk per trade
            
            decimal riskAmount = accountBalance * riskPercent;
            decimal slDistance = Math.Abs(signal.EntryPrice - signal.StopLoss);
            decimal quantity = riskAmount / slDistance;
            
            return quantity;
        }
        
        private async Task<SignalData> FetchLatestSignal()
        {
            try
            {
                var response = await _httpClient.GetAsync(API_URL);
                
                if (!response.IsSuccessStatusCode)
                    return null;
                
                var json = await response.Content.ReadAsStringAsync();
                var result = JsonConvert.DeserializeObject<ApiResponse>(json);
                
                if (result?.Status == "success")
                    return result.Signal;
                
                return null;
            }
            catch (Exception ex)
            {
                // Log error
                return null;
            }
        }
    }
    
    // Data models
    public class ApiResponse
    {
        public string Status { get; set; }
        public SignalData Signal { get; set; }
        public string Timestamp { get; set; }
    }
    
    public class SignalData
    {
        public string Timestamp { get; set; }
        public string Symbol { get; set; }
        public string Direction { get; set; }
        
        [JsonProperty("entry_price")]
        public decimal EntryPrice { get; set; }
        
        [JsonProperty("stop_loss")]
        public decimal StopLoss { get; set; }
        
        [JsonProperty("take_profit")]
        public decimal TakeProfit { get; set; }
        
        public int Leverage { get; set; }
        
        [JsonProperty("signal_strength")]
        public double SignalStrength { get; set; }
        
        public string Timeframe { get; set; }
        
        [JsonProperty("strategy_name")]
        public string StrategyName { get; set; }
    }
}
```

### Building the Strategy

1. **Create new C# Class Library project** in Visual Studio
2. **Add ATAS references** from ATAS installation directory
3. **Install Newtonsoft.Json** via NuGet: `Install-Package Newtonsoft.Json`
4. **Copy the code above** into your strategy file
5. **Build** to create DLL file
6. **Copy DLL** to `Documents\ATAS\Strategies\`
7. **Reload** strategies in ATAS platform

---

## Testing

### Test Signal Generation

Run the test script:

```bash
python telegram_signal_notifier.py
```

This will send a test signal to Telegram and verify the integration.

### Test ATAS API

```bash
python atas_platform_integration.py
```

This starts the API server and creates test signals. Then check:

```bash
curl http://localhost:8888/api/signals
```

---

## Configuration

### Change API Port

Edit `start_high_frequency_scalping_bot.py`:

```python
atas_integration = ATASPlatformIntegration(host='0.0.0.0', port=9999)  # Change port
```

Then update the ATAS C# strategy URL:

```csharp
private const string API_URL = "http://localhost:9999/api/signals/latest";
```

### Enable/Disable Auto-Trading

In ATAS strategy, change:

```csharp
private bool _autoTrade = false;  // Disable auto-trading (alerts only)
```

---

## Troubleshooting

### API Server Not Starting

**Check logs:**
```bash
tail -f high_frequency_scalping.log
```

**Verify port not in use:**
```bash
lsof -i :8888
```

### ATAS Cannot Connect

**Check firewall:**
- Allow port 8888 in firewall
- If running on different machines, use IP address instead of localhost

**Test connectivity:**
```bash
curl http://YOUR_SERVER_IP:8888/api/health
```

### No Signals Appearing

**Check signal generation:**
- Verify bot is running and scanning markets
- Check minimum consensus requirements
- Review logs for signal filtering

**Check API:**
```bash
curl http://localhost:8888/api/stats
```

---

## Signal Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│  Python High-Frequency Scalping Bot                     │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Strategy 1  │  │  Strategy 2  │  │  Strategy 3  │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                 │           │
│         └─────────────────┴─────────────────┘           │
│                           │                             │
│                    ┌──────▼──────┐                      │
│                    │   Signal    │                      │
│                    │   Fusion    │                      │
│                    └──────┬──────┘                      │
│                           │                             │
│         ┌─────────────────┼─────────────────┐           │
│         │                 │                 │           │
│    ┌────▼────┐      ┌────▼────┐      ┌────▼────┐      │
│    │Telegram │      │Position │      │  ATAS   │      │
│    │Notifier │      │ Closer  │      │   API   │      │
│    └─────────┘      └─────────┘      └────┬────┘      │
└─────────────────────────────────────────────┼──────────┘
                                              │
                                    ┌─────────▼─────────┐
                                    │   REST API        │
                                    │   Port 8888       │
                                    └─────────┬─────────┘
                                              │
                                    ┌─────────▼─────────┐
                                    │  ATAS Platform    │
                                    │  C# Strategy      │
                                    └───────────────────┘
```

---

## Advanced Usage

### Filter by Symbol in ATAS

```csharp
private string _targetSymbol = "BTCUSDT";
private const string API_URL = "http://localhost:8888/api/signals/BTCUSDT";
```

### Multiple Timeframe Signals

Query different endpoints for different timeframes:

```csharp
// Get all signals and filter by timeframe
var allSignals = await FetchAllSignals();
var scalping = allSignals.Where(s => s.Timeframe == "1m").ToList();
var swing = allSignals.Where(s => s.Timeframe == "5m").ToList();
```

### Signal Acknowledgment

After processing a signal, acknowledge it:

```csharp
private async Task AcknowledgeSignal(string timestamp)
{
    var content = new StringContent(
        JsonConvert.SerializeObject(new { timestamp = timestamp }),
        Encoding.UTF8,
        "application/json"
    );
    
    await _httpClient.PostAsync(
        "http://localhost:8888/api/signals/acknowledge",
        content
    );
}
```

---

## Best Practices

1. **Risk Management** - Always implement proper position sizing in ATAS
2. **Signal Validation** - Verify signal data before executing trades
3. **Error Handling** - Implement retry logic for API calls
4. **Logging** - Log all trades for review and optimization
5. **Testing** - Test with small positions before full deployment
6. **Monitoring** - Monitor both Python bot and ATAS strategy performance

---

## Support

For issues or questions:
- Check logs in `high_frequency_scalping.log`
- Review ATAS strategy logs in ATAS platform
- Test API endpoints manually with curl/browser
- Verify network connectivity between systems

---

## License

This integration bridge is part of the high-frequency scalping system.
