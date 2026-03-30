#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/codygnon/Documents/usdjpy_assistant")


def _load_m1(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    return df.sort_values("time").reset_index(drop=True)


def _resample_m5(m1: pd.DataFrame) -> pd.DataFrame:
    rs = (
        m1.set_index("time")[["open", "high", "low", "close"]]
        .resample("5min", label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
        .reset_index()
    )
    rs["ema9"] = rs["close"].ewm(span=9, adjust=False).mean()
    rs["ema21"] = rs["close"].ewm(span=21, adjust=False).mean()
    return rs


def _read_blocked(blocked_json: Path) -> tuple[dict, list[dict]]:
    data = json.loads(blocked_json.read_text(encoding="utf-8"))
    blocked = list(data.get("blocked_trades", []))
    for row in blocked:
        row["entry_time"] = pd.Timestamp(row["entry_time"])
        row["outcome"] = "winner" if float(row.get("usd", 0.0)) > 0 else "loser"
        row["abs_usd"] = abs(float(row.get("usd", 0.0)))
        row["abs_pips"] = abs(float(row.get("pips", 0.0)))
    return data, blocked


def _pick_samples(blocked: list[dict]) -> list[dict]:
    quotas = [
        ("breakout", "loser", 6),
        ("breakout", "winner", 3),
        ("post_breakout_trend", "loser", 4),
        ("post_breakout_trend", "winner", 2),
    ]
    chosen: list[dict] = []
    seen: set[str] = set()
    for regime, outcome, count in quotas:
        rows = [
            r for r in blocked
            if r.get("regime_label") == regime and r.get("outcome") == outcome
        ]
        rows = sorted(rows, key=lambda r: (r["abs_usd"], r["entry_time"]), reverse=True)
        for row in rows[:count]:
            key = f"{row['entry_time'].isoformat()}|{row.get('reason')}"
            if key not in seen:
                chosen.append(row)
                seen.add(key)

    if len(chosen) < min(16, len(blocked)):
        remaining = [
            r for r in sorted(blocked, key=lambda r: (r["entry_time"], r["abs_usd"]))
            if f"{r['entry_time'].isoformat()}|{r.get('reason')}" not in seen
        ]
        if remaining:
            step = max(1, math.floor(len(remaining) / max(1, 16 - len(chosen))))
            for idx in range(0, len(remaining), step):
                row = remaining[idx]
                key = f"{row['entry_time'].isoformat()}|{row.get('reason')}"
                if key not in seen:
                    chosen.append(row)
                    seen.add(key)
                if len(chosen) >= 16:
                    break

    return sorted(chosen, key=lambda r: r["entry_time"])


def _svg_escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _render_chart(
    m5: pd.DataFrame,
    trade: dict,
    out_path: Path,
    bars_before: int = 24,
    bars_after: int = 18,
) -> None:
    entry_ts = pd.Timestamp(trade["entry_time"])
    idxs = np.where(m5["time"] <= entry_ts)[0]
    if len(idxs) == 0:
        return
    end_idx = idxs[-1]
    start_idx = max(0, end_idx - bars_before)
    stop_idx = min(len(m5), end_idx + bars_after + 1)
    window = m5.iloc[start_idx:stop_idx].copy()
    if window.empty:
        return

    width = 1400
    height = 800
    pad_l = 80
    pad_r = 30
    pad_t = 70
    pad_b = 80
    chart_w = width - pad_l - pad_r
    chart_h = height - pad_t - pad_b
    xs = np.linspace(pad_l, width - pad_r, len(window))
    lo = float(window["low"].min())
    hi = float(window["high"].max())
    if hi <= lo:
        hi = lo + 0.01
    span = hi - lo
    lo -= span * 0.05
    hi += span * 0.05

    def ymap(v: float) -> float:
        return pad_t + (hi - v) / (hi - lo) * chart_h

    entry_px = float(window.loc[window["time"] <= entry_ts, "close"].iloc[-1])
    side = str(trade.get("side", "")).upper()
    regime = str(trade.get("regime_label", ""))
    reason = str(trade.get("reason", ""))
    raw_profile = str(trade.get("raw_profile", ""))
    exit_reason = str(trade.get("exit_reason", ""))
    usd = float(trade.get("usd", 0.0))
    pips = float(trade.get("pips", 0.0))
    title = (
        f"{entry_ts.strftime('%Y-%m-%d %H:%M UTC')}  {side}  {regime}  "
        f"{pips:+.1f}p / ${usd:+.0f}\n"
        f"{reason} | profile={raw_profile} | exit={exit_reason}"
    )

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#0f172a"/>',
        f'<rect x="{pad_l}" y="{pad_t}" width="{chart_w}" height="{chart_h}" fill="#111827" stroke="#334155" stroke-width="1"/>',
    ]

    for frac in np.linspace(0, 1, 7):
        y = pad_t + frac * chart_h
        price = hi - frac * (hi - lo)
        lines.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width-pad_r}" y2="{y:.1f}" stroke="#1f2937" stroke-width="1"/>')
        lines.append(
            f'<text x="{pad_l-8}" y="{y+4:.1f}" text-anchor="end" fill="#94a3b8" font-size="12">{price:.3f}</text>'
        )

    tick_idx = np.linspace(0, len(window) - 1, min(8, len(window)), dtype=int)
    for idx in tick_idx:
        x = xs[idx]
        label = pd.Timestamp(window.iloc[idx]["time"]).strftime("%m-%d %H:%M")
        lines.append(f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{pad_t+chart_h}" stroke="#17202a" stroke-width="1"/>')
        lines.append(
            f'<text x="{x:.1f}" y="{height-24}" text-anchor="middle" fill="#94a3b8" font-size="12">{_svg_escape(label)}</text>'
        )

    up = "#22c55e"
    down = "#ef4444"
    candle_w = max(3.0, min(14.0, chart_w / max(10, len(window)) * 0.7))
    ema9_pts = []
    ema21_pts = []
    entry_x = None
    for i, row in enumerate(window.itertuples(index=False)):
        x = xs[i]
        o = float(row.open)
        h = float(row.high)
        l = float(row.low)
        c = float(row.close)
        color = up if c >= o else down
        lines.append(f'<line x1="{x:.1f}" y1="{ymap(l):.1f}" x2="{x:.1f}" y2="{ymap(h):.1f}" stroke="{color}" stroke-width="1.2"/>')
        y1 = ymap(max(o, c))
        y2 = ymap(min(o, c))
        body_h = max(1.5, y2 - y1)
        lines.append(
            f'<rect x="{x-candle_w/2:.1f}" y="{y1:.1f}" width="{candle_w:.1f}" height="{body_h:.1f}" fill="{color}" stroke="{color}"/>'
        )
        ema9_pts.append(f"{x:.1f},{ymap(float(row.ema9)):.1f}")
        ema21_pts.append(f"{x:.1f},{ymap(float(row.ema21)):.1f}")
        if pd.Timestamp(row.time) <= entry_ts:
            entry_x = x

    lines.append(f'<polyline fill="none" stroke="#f59e0b" stroke-width="2" points="{" ".join(ema9_pts)}"/>')
    lines.append(f'<polyline fill="none" stroke="#22c55e" stroke-width="2" points="{" ".join(ema21_pts)}"/>')

    if entry_x is not None:
        lines.append(
            f'<line x1="{entry_x:.1f}" y1="{pad_t}" x2="{entry_x:.1f}" y2="{pad_t+chart_h}" stroke="#38bdf8" stroke-width="2" stroke-dasharray="7 5"/>'
        )
    lines.append(
        f'<line x1="{pad_l}" y1="{ymap(entry_px):.1f}" x2="{width-pad_r}" y2="{ymap(entry_px):.1f}" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="3 6"/>'
    )

    title_lines = title.split("\n")
    lines.append(f'<text x="{pad_l}" y="30" fill="#e5e7eb" font-size="20" font-weight="bold">{_svg_escape(title_lines[0])}</text>')
    if len(title_lines) > 1:
        lines.append(f'<text x="{pad_l}" y="55" fill="#cbd5e1" font-size="15">{_svg_escape(title_lines[1])}</text>')
    lines.append(f'<text x="{width-180}" y="30" fill="#f59e0b" font-size="14">EMA 9</text>')
    lines.append(f'<text x="{width-180}" y="52" fill="#22c55e" font-size="14">EMA 21</text>')
    lines.append("</svg>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_report(
    out_dir: Path,
    dataset_name: str,
    blocked_json: Path,
    meta: dict,
    samples: list[dict],
) -> None:
    fs = meta.get("v44_filter_stats", {})
    report = [
        f"# V44 Regime Block Audit - {dataset_name}",
        "",
        f"Source blocked file: `{blocked_json}`",
        "",
        "## Summary",
        "",
        f"- Blocked V44 trades: `{fs.get('v44_blocked', 0)}`",
        f"- Passed V44 trades: `{fs.get('v44_passed', 0)}`",
        f"- Blocked winners: `{fs.get('blocked_winners', 0)}`",
        f"- Blocked losers: `{fs.get('blocked_losers', 0)}`",
        f"- Blocked net pips: `{fs.get('blocked_net_pips', 0)}`",
        "",
        "## Sample Charts",
        "",
    ]
    for sample in samples:
        ts = pd.Timestamp(sample["entry_time"])
        stem = f"{ts.strftime('%Y%m%d_%H%M')}_{sample['regime_label']}_{sample['outcome']}"
        img = out_dir / f"{stem}.svg"
        report.extend(
            [
                f"### {ts.strftime('%Y-%m-%d %H:%M UTC')} | {sample['regime_label']} | {sample['outcome']}",
                "",
                f"- Side: `{sample.get('side')}`",
                f"- Pips / USD: `{float(sample.get('pips', 0.0)):+.1f}p / ${float(sample.get('usd', 0.0)):+.2f}`",
                f"- Reason: `{sample.get('reason')}`",
                f"- Profile: `{sample.get('raw_profile')}`",
                f"- Exit: `{sample.get('exit_reason')}`",
                "",
                f"![{stem}]({img})",
                "",
            ]
        )
    (out_dir / "index.md").write_text("\n".join(report), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render chart audit for blocked V44 regime trades")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--blocked-json", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    blocked_json = Path(args.blocked_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta, blocked = _read_blocked(blocked_json)
    if not blocked:
        raise SystemExit("No blocked_trades found in the provided JSON.")

    m1 = _load_m1(str(dataset))
    m5 = _resample_m5(m1)
    samples = _pick_samples(blocked)
    for sample in samples:
        ts = pd.Timestamp(sample["entry_time"])
        stem = f"{ts.strftime('%Y%m%d_%H%M')}_{sample['regime_label']}_{sample['outcome']}"
        _render_chart(m5, sample, out_dir / f"{stem}.svg")

    _write_report(out_dir, dataset.name, blocked_json, meta, samples)
    print(f"Wrote {len(samples)} charts to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
