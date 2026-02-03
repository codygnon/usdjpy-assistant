"""Run the FastAPI backend server.

Usage: python run_api.py [--port PORT]

When PORT env var is set (e.g. on Railway/Render), binds to 0.0.0.0 and that port
for production; otherwise uses 127.0.0.1 and 8000 for local dev.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import webbrowser
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the USDJPY Assistant API server.")
    ap.add_argument("--port", type=int, default=None, help="Port to run on (default: 8000 or $PORT)")
    ap.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parent
    port_env = os.environ.get("PORT")
    if port_env is not None:
        port = int(port_env)
        host = "0.0.0.0"
        reload = False
        open_browser = False
    else:
        port = args.port if args.port is not None else 8000
        host = "127.0.0.1"
        reload = True
        open_browser = not args.no_browser

    print(f"Starting USDJPY Assistant API on http://{host}:{port}")
    print("Press Ctrl+C to stop.")

    if open_browser:
        webbrowser.open(f"http://{host}:{port}")

    uvicorn_args = [
        sys.executable, "-m", "uvicorn",
        "api.main:app",
        "--host", host,
        "--port", str(port),
    ]
    if reload:
        uvicorn_args.append("--reload")

    try:
        subprocess.run(uvicorn_args, cwd=str(base_dir))
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
