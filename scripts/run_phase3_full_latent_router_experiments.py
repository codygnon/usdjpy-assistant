#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")
OUT_DIR = ROOT / "research_out"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full-latent router experiments across 250k/500k/1000k")
    p.add_argument("--datasets", nargs="*", default=[
        str(OUT_DIR / "USDJPY_M1_OANDA_250k.csv"),
        str(OUT_DIR / "USDJPY_M1_OANDA_500k.csv"),
        str(OUT_DIR / "USDJPY_M1_OANDA_1000k.csv"),
    ])
    p.add_argument("--output-tag", default="phase3_full_latent_router_v1")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    index_rows = []
    for dataset in args.datasets:
        ds = Path(dataset)
        tag = f"{args.output_tag}_{ds.stem.replace('USDJPY_M1_OANDA_', '')}"
        out_prefix = OUT_DIR / tag
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "backtest_phase3_full_latent_router.py"),
            "--dataset",
            str(ds),
            "--output-prefix",
            str(out_prefix),
        ]
        print(f"[run] {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
        index_rows.append(
            {
                "dataset": ds.name,
                "output_prefix": str(out_prefix),
                "index_json": str(out_prefix.with_name(out_prefix.name + "_index.json")),
            }
        )
    out = OUT_DIR / f"{args.output_tag}_index.json"
    out.write_text(json.dumps(index_rows, indent=2), encoding="utf-8")
    print(json.dumps({"index": str(out), "datasets": len(index_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
