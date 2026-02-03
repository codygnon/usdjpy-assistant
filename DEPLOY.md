# Deploy to Railway or Render (uncle opens a link)

After connecting this repo to Railway or Render, set these so the app builds and runs.

## Build command (run once per deploy)

The frontend is pre-built and committed under `frontend/dist/`, so the PaaS does not need Node.js:

```bash
pip install -r requirements.txt
```

If you change the frontend, run `cd frontend && npm ci && npm run build` locally and commit the updated `frontend/dist/` before deploying.

## Start command

- **Option A – use Procfile:** Leave "Start Command" blank; the PaaS will use the `Procfile` in the repo (`web: uvicorn api.main:app --host 0.0.0.0 --port $PORT`).
- **Option B – explicit:** Set Start Command to:
  ```bash
  uvicorn api.main:app --host 0.0.0.0 --port $PORT
  ```

## Root directory

Leave as repo root (where `requirements.txt` and `frontend/` live).

## Notes

- **PORT** is set by the PaaS; the app binds to `0.0.0.0` and that port.
- **Profiles and logs** are stored on the server filesystem; they may be reset on redeploy unless you add a persistent volume (Railway) or move to a database later.
- **Run loop / MT5:** The web app runs in the cloud; actual trading (run_loop) still needs to run on a machine with MT5 or a broker connection.

## V2: FXCM / OANDA

Planned for a future version: broker adapters for FXCM and OANDA so the app can run and trade without MetaTrader 5. Until then, MT5 remains optional (install via `requirements-mt5.txt` on Windows) for legacy trading; the UI and API work without it for deployment to Linux PaaS.
