# USDJPY Assistant

A semi-automated trading assistant for USD/JPY on MetaTrader 5 (demo accounts).

## Features

- **Trading Presets**: Pre-configured templates for different trading styles
  - Aggressive/Conservative Scalping (M1)
  - Aggressive/Conservative Swing (M15/H4)
  - Mean Reversion (RSI oversold/overbought)
  - Trend Continuation (EMA/SMA cross)

- **Execution Policies**: Multiple ways to trigger trades
  - Confirmed Cross (EMA/SMA crossover with confirmation)
  - Price Level Trend (trade at specific price levels)
  - Indicator-Based (RSI + regime + optional MACD)

- **Risk Management**: Built-in safety controls
  - Max lots, spread limits, trade limits
  - Kill switch for emergency stop
  - Demo-only execution safeguards

- **Modern Web Interface**: Clean dashboard with
  - Run/Status control
  - Preset selection and preview
  - Profile editor
  - Logs and statistics

## Quick Start (Windows)

### Prerequisites

1. **Python 3.11 or 3.12** installed (NOT 3.13+)
   - Download from https://www.python.org/downloads/
   - Check "Add Python to PATH" during install

2. **MetaTrader 5** installed and configured
   - Download from your broker or https://www.metatrader5.com/
   - Log in to a **demo account**
   - Enable "Algo Trading" (Tools > Options > Expert Advisors)
   - Add your symbol (e.g., USDJPY.PRO) to Market Watch

3. **Node.js** (optional, for frontend development)
   - Only needed if you want to modify the UI
   - Download from https://nodejs.org/

### Installation

1. **Extract the zip** to any folder (e.g., `C:\usdjpy_assistant`)

2. **Run SETUP_ENV.bat** (double-click)
   - Creates Python virtual environment
   - Installs all dependencies
   - For MT5 trading on Windows, also run: `pip install -r requirements-mt5.txt`

3. **Run START_APP.bat** (double-click)
   - Starts the web application
   - Opens your browser to http://127.0.0.1:8000

### Usage

1. **Select a profile** from the sidebar dropdown

2. **Apply a preset** (Presets tab)
   - Choose a trading style that matches your goals
   - Preview the changes
   - Click "Apply Preset"

3. **Configure runtime** (Run/Status tab)
   - Set Mode to `ARMED_AUTO_DEMO` for automatic trading
   - Leave Kill Switch OFF
   - Click "Start Loop"

4. **Monitor** (Logs & Stats tab)
   - Watch execution attempts and rejections
   - Track win rate and average pips

## Folder Structure

```
usdjpy_assistant/
├── api/                  # FastAPI backend
├── core/                 # Trading logic
│   ├── presets.py        # Preset definitions
│   ├── profile.py        # Profile schema
│   ├── execution_engine.py
│   └── ...
├── frontend/             # React web UI
├── profiles/             # Profile JSON files
├── logs/                 # Runtime logs and database
├── START_APP.bat         # Start web app
├── SETUP_ENV.bat         # One-time setup
└── run_api.py            # API server script
```

## Preset Descriptions

| Preset | Style | Description |
|--------|-------|-------------|
| Aggressive Scalping | M1 | Fast trades, loose filters, tight TP/SL |
| Conservative Scalping | M1 | Strict spread/alignment, fewer trades |
| Aggressive Swing | M15/H4 | Moderate filters, wider targets |
| Conservative Swing | H4 | ATR filter, requires MACD confirmation |
| Mean Reversion Dip Buy | M15 | Buy when RSI oversold in uptrend |
| Mean Reversion Dip Sell | M15 | Sell when RSI overbought in downtrend |
| Trend Continuation | M1 | Trade with trend using EMA/SMA cross |

## Modes

- **DISARMED**: No trades placed (safe for testing)
- **ARMED_MANUAL_CONFIRM**: Signals logged, you confirm each trade
- **ARMED_AUTO_DEMO**: Automatic execution on demo accounts only

## Safety Features

- **Demo-only guard**: Will not execute on live/real accounts
- **Kill switch**: Immediately stops all trading
- **Idempotency**: Won't double-fire the same signal
- **Risk checks**: Validates every trade against your limits

## Troubleshooting

### "MT5 not initialized" or connection errors
- Make sure MetaTrader 5 is running
- Log in to your demo account
- Enable Algo Trading in MT5 settings

### Zero trades placed
- Check the Rejection Breakdown in Logs & Stats
- Common causes:
  - `indicator_based`: Conditions not met (regime/RSI)
  - `risk_reject`: Spread too high or at max trades
  - `order_failed`: MT5 rejected (check filling mode)

### Module not found errors
- Re-run SETUP_ENV.bat
- Make sure you're using Python 3.11 or 3.12

## Development

### Running in development mode

Backend (from project root):
```bash
.venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

Frontend (from frontend/ folder):
```bash
npm install
npm run dev
```

### Building frontend for production
```bash
cd frontend
npm run build
```

## License

For personal use only. Not financial advice.
