"""Run the FastAPI backend server.

Usage: python run_api.py [--port PORT]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the USDJPY Assistant API server.")
    ap.add_argument("--port", type=int, default=8000, help="Port to run on (default: 8000)")
    ap.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parent
    
    print(f"Starting USDJPY Assistant API on http://127.0.0.1:{args.port}")
    print("Press Ctrl+C to stop.")
    
    if not args.no_browser:
        webbrowser.open(f"http://127.0.0.1:{args.port}")
    
    try:
        subprocess.run(
            [
                sys.executable, "-m", "uvicorn",
                "api.main:app",
                "--host", "127.0.0.1",
                "--port", str(args.port),
                "--reload",
            ],
            cwd=str(base_dir),
        )
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
