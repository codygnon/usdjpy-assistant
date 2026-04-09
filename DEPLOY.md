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

## Environment variables

### AI Trading Assistant (OpenAI)

The **AI Trading Assistant** tab calls OpenAI from the **same service** that runs `uvicorn` (your `web` process). It does **not** read secrets from the browser and is **not** configured by **Start loop** in the UI.

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | Yes, for chat | Server-side key from [platform.openai.com](https://platform.openai.com/api-keys) |
| `OPENAI_CHAT_MODEL` | No | Defaults to `gpt-4o-mini` in code if unset |

**Set these on Railway** (the service that runs the API), then redeploy or wait for Railway to restart the service so the new variables are visible to `uvicorn`.

#### Option A — Railway dashboard (no terminal)

1. Open [Railway](https://railway.app) → your **project** → the **service** that runs this app (the one using the `Procfile` / uvicorn).
2. Open the **Variables** tab (or **Settings → Variables**).
3. Click **New Variable** (or **RAW Editor**).
4. Add:
   - **Name:** `OPENAI_API_KEY`  
   - **Value:** your secret key (starts with `sk-...`).
5. Optionally add `OPENAI_CHAT_MODEL` = `gpt-4o-mini`.
6. Save. Railway usually triggers a **new deploy**; if not, use **Deploy** / **Restart** on that service.

#### Option B — Railway CLI (exact commands)

Run these **on your Mac** in any folder (or in your repo after `railway link`):

```bash
# One-time: install CLI and log in
npm i -g @railway/cli
railway login
```

From your **project repo root** (or any directory after linking):

```bash
cd /path/to/usdjpy_assistant
railway link
```

Select the same Railway project and **service** that hosts the API. Then set the secret (paste your real key; avoid sharing the line or committing it):

```bash
railway variable set OPENAI_API_KEY="sk-your-key-here"
```

Optional:

```bash
railway variable set OPENAI_CHAT_MODEL="gpt-4o-mini"
```

If you have multiple services, target the web service:

```bash
railway variable set OPENAI_API_KEY="sk-your-key-here" --service "YourServiceName"
```

After variables change, ensure a deploy completes (Railway normally redeploys automatically).

#### Verify

1. Open your deployed site → unlock profile → **AI Trading Assistant** → send a message.  
2. If the key is missing, the API returns **503** and the UI explains that `OPENAI_API_KEY` must be set on the server.

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
