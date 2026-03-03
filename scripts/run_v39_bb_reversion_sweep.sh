#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/Users/codygnon/Documents/usdjpy_assistant}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/research_out/session_momentum_v39_winner_config.json}"
INPUT_CSV="${INPUT_CSV:-$ROOT_DIR/research_out/USDJPY_M1_OANDA_500k.csv}"
OUT_DIR="${OUT_DIR:-/tmp}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing config: $CONFIG_PATH" >&2
  exit 1
fi

if [[ ! -f "$INPUT_CSV" ]]; then
  echo "Missing input CSV: $INPUT_CSV" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
cd "$ROOT_DIR"

for sl in 6 8 10 12; do
  for mw in 6 10 15; do
    out_json="$OUT_DIR/bb_mid_SL${sl}_MW${mw}.json"
    out_log="$OUT_DIR/bb_mid_SL${sl}_MW${mw}.log"
    echo "--- Running SL=${sl} MW=${mw} ---"
    "$PYTHON_BIN" scripts/backtest_session_momentum.py \
      --config "$CONFIG_PATH" \
      --in "$INPUT_CSV" \
      --out "$out_json" \
      --bb-rev-enabled true \
      --bb-rev-std 2.0 \
      --bb-rev-confirm true \
      --bb-rev-tp-midline true \
      --bb-rev-sl-pips "$sl" \
      --bb-rev-min-width-pips "$mw" \
      --bb-rev-require-trend false \
      >"$out_log" 2>&1
    echo "EXIT=$?"
  done
done

"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

out_dir = Path(os.environ.get("OUT_DIR", "/tmp"))

print("=" * 110)
print(f"{'SL':>4} {'MW':>4} | {'Trades':>6} {'WR%':>6} {'Net_USD':>10} {'Avg/Tr':>8} | {'AvgWin':>7} {'AvgLoss':>7} {'AvgTP_p':>7} | {'SL_ex':>5} {'TP_ex':>5} {'MidCls':>6} | V39_ok")
print("-" * 110)

for sl in [6, 8, 10, 12]:
    for mw in [6, 10, 15]:
        f = out_dir / f"bb_mid_SL{sl}_MW{mw}.json"
        if not f.exists():
            print(f"{sl:4d} {mw:4d} | FAILED")
            continue

        d = json.load(f.open())
        trades = d["results"]["closed_trades"]
        bb = [t for t in trades if t.get("entry_regime") == "BB_REVERSION"]
        v39 = [t for t in trades if t.get("entry_regime") != "BB_REVERSION"]

        n = len(bb)
        if n == 0:
            print(f"{sl:4d} {mw:4d} | {0:6d} trades - no entries")
            continue

        bw = sum(1 for t in bb if t.get("pips", 0) > 0)
        bl = sum(1 for t in bb if t.get("pips", 0) < 0)
        bp = sum(t.get("usd", 0) for t in bb)
        aw = sum(t["pips"] for t in bb if t.get("pips", 0) > 0) / max(1, bw)
        al = sum(t["pips"] for t in bb if t.get("pips", 0) < 0) / max(1, bl)
        avg_tp = sum(t.get("tp1_pips", 0) for t in bb) / n
        sl_exits = sum(1 for t in bb if t.get("exit_reason") == "sl")
        tp_exits = sum(1 for t in bb if "tp" in t.get("exit_reason", ""))
        blocked = d["results"].get("blocked_reasons", {})
        mid_close = blocked.get("bb_rev_midline_too_close", 0)
        vp = sum(t.get("usd", 0) for t in v39)
        v39_ok = "YES" if len(v39) == 239 and abs(vp - 40311) < 5 else f"NO({len(v39)},${vp:,.0f})"

        marker = " <<<" if bp > 0 else ""
        print(f"{sl:4d} {mw:4d} | {n:6d} {bw / n * 100:6.1f} {bp:10,.0f} {bp / n:8,.0f} | {aw:7.1f} {al:7.1f} {avg_tp:7.1f} | {sl_exits:5d} {tp_exits:5d} {mid_close:6d} | {v39_ok}{marker}")
    print()

print("=" * 110)
print()
print("KEY: SL=stop loss pips, MW=min BB width pips, AvgTP_p=avg midline TP distance in pips")
print("V39_ok confirms V39 is untouched (239 trades, ~$40,311)")
print("<<< marks profitable BB configs")
PY
