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

## Persistent profiles and logs (Railway volume)

To keep profile editor settings and logs across redeploys, attach a **persistent volume** to your Railway service and point the app at it.

1. **Create a volume**
   - In the Railway dashboard, open your project and select the service that runs the app.
   - Use the **Command Palette** (e.g. `Ctrl+K` / `Cmd+K`) or right‑click the service → **Add Volume**, or use the **Volumes** section in the service settings.
   - Create a new volume and attach it to this service.

2. **Set the mount path**
   - When attaching the volume, set the **mount path** to `/data` (or another path you prefer).
   - Railway will expose this path to the app and set the env var **`RAILWAY_VOLUME_MOUNT_PATH`** (e.g. `/data`) automatically. The app uses this to store `profiles/` and `logs/` on the volume.

3. **Redeploy**
   - Redeploy the service (or push a new commit). After the deploy, the app will write profiles and logs under the volume. They will persist across future redeploys.

**Optional:** If you use a different path or run outside Railway, set **`USDJPY_DATA_DIR`** to the directory that should hold `profiles` and `logs` (e.g. `USDJPY_DATA_DIR=/data`).

**Limitations (Railway):** Each service can have only one volume; replicas cannot use volumes. Storage limits depend on your plan (e.g. Hobby 5GB).

## V2: FXCM / OANDA

Planned for a future version: broker adapters for FXCM and OANDA so the app can run and trade without MetaTrader 5. Until then, MT5 remains optional (install via `requirements-mt5.txt` on Windows) for legacy trading; the UI and API work without it for deployment to Linux PaaS.
