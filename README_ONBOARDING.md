## USDJPY Assistant v1 (MT5 Demo) - Onboarding

### What this is
- A **USDJPY-only** trading assistant for **MetaTrader 5** demo accounts.
- Pulls **H4/M15/M1**, computes **EMA/SMA cross + confirmation candle**, applies filters, enforces risk rules, and logs everything.
- Includes a **Streamlit UI** for advanced settings and manual confirmation.

### Folder contents
- **`profiles/v1/*.json`**: per-user configs (you and your uncle each have one)
- **`logs/<profile>/assistant.db`**: SQLite database with snapshots, trades, executions, pending signals
- **`RUN_UI.bat`**: launches the UI
- **`RUN_LOOP.bat`**: runs the 1-minute polling loop (reads runtime state from the UI)
- **`SETUP_ENV.bat`**: creates a venv and installs dependencies

### Setup (Windows)
1. Install **Python 3.11+** (3.12 recommended).
2. Install and open **MetaTrader 5**, login to a **demo** account.
3. In this folder, double click **`SETUP_ENV.bat`**.
4. Double click **`RUN_UI.bat`**.
5. (Optional) In another terminal, run **`RUN_LOOP.bat`**.

### How to use (recommended workflow)
1. In the UI, select your profile and confirm your **symbol** (e.g. `USDJPY.PRO`).
2. Click **Run Snapshot** to verify logging works.
3. Start the loop via `RUN_LOOP.bat`.
4. In UI, set Mode:
   - `DISARMED`: log-only
   - `ARMED_MANUAL_CONFIRM`: produces **pending signals** you can approve in UI
   - `ARMED_AUTO_DEMO`: auto-executes on demo
5. Use **Kill switch** any time to stop execution immediately.

### Your uncle’s setup
1. Copy `profiles/v1/cody_demo.json` to `profiles/v1/uncle_demo.json`.
2. Edit:
   - `profile_name`
   - `symbol` (his broker may name it differently)
   - risk limits (max lots/spread, require stop, etc.)
3. Run Doctor from UI to confirm his MT5 connection and symbol.

### Notes
- v1 is **M1 cadence**, not true HFT.
- Execution is **demo-only** by design.

