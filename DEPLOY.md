# Deploy to Railway or Render (uncle opens a link)

After connecting this repo to Railway or Render, set these so the app builds and runs.

## Build command (run once per deploy)

```bash
pip install -r requirements.txt && cd frontend && npm ci && npm run build
```

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
