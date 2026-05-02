#!/usr/bin/env python3
"""Render an aesthetic PDF brief for Phase 6 lifecycle findings."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
FORENSIC_DIR = ROOT / "research_out" / "autonomous_fillmore_forensic_20260501"
OUT_PDF = FORENSIC_DIR / "Auto_Fillmore_Forensic_Audit_Phase6_Lifecycle.pdf"

BG = "#F7F4EE"
INK = "#1D2733"
MUTED = "#69717D"
GRID = "#D9D2C3"
RED = "#B5483F"
RED_DARK = "#7F2F2A"
TEAL = "#176B6D"
GOLD = "#B9852F"
BLUE = "#355C8C"
PANEL = "#FFFFFF"


def one(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x):.1f}"


def two(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x):.2f}"


def pct(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x) * 100:.1f}%"


def short(text: Any, n: int = 42) -> str:
    s = "" if text is None or pd.isna(text) else " ".join(str(text).split())
    return s if len(s) <= n else s[: n - 3].rstrip() + "..."


def fig_base(title: str, subtitle: str = ""):
    fig = plt.figure(figsize=(11, 8.5), facecolor=BG)
    fig.text(0.055, 0.94, title, fontsize=22, fontweight="bold", color=INK, ha="left")
    if subtitle:
        fig.text(0.055, 0.905, subtitle, fontsize=10.5, color=MUTED, ha="left")
    fig.add_artist(plt.Line2D([0.055, 0.945], [0.885, 0.885], color=GRID, linewidth=1.2))
    return fig


def card(fig, x: float, y: float, w: float, h: float, label: str, value: str, color: str = INK, note: str = ""):
    ax = fig.add_axes([x, y, w, h])
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.06, 0.72, label.upper(), transform=ax.transAxes, fontsize=8.2, color=MUTED, fontweight="bold")
    ax.text(0.06, 0.34, value, transform=ax.transAxes, fontsize=19, color=color, fontweight="bold")
    if note:
        ax.text(0.06, 0.12, note, transform=ax.transAxes, fontsize=7.7, color=MUTED)
    return ax


def wrapped_note(fig, text: str, x: float, y: float, w: float, size: float = 10.5, color: str = INK):
    lines = textwrap.wrap(text, width=max(30, int(w * 115)))
    fig.text(x, y, "\n".join(lines), fontsize=size, color=color, ha="left", va="top", linespacing=1.35)


def table_ax(fig, df: pd.DataFrame, x: float, y: float, w: float, h: float, font_size: float = 7.8):
    ax = fig.add_axes([x, y, w, h])
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="left", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size)
    tbl.scale(1, 1.32)
    for (row, _col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#E6DFD2")
        if row == 0:
            cell.set_facecolor("#EFE8DC")
            cell.set_text_props(weight="bold", color=INK)
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 else "#FBF8F1")
            cell.set_text_props(color=INK)
    return ax


def barh_ax(fig, labels, values, x: float, y: float, w: float, h: float, title: str):
    ax = fig.add_axes([x, y, w, h], facecolor=BG)
    labels = [short(v, 34) for v in labels]
    values = np.asarray(values, dtype=float)
    colors = [RED if v < 0 else TEAL for v in values]
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=colors, alpha=0.92)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.tick_params(axis="x", labelsize=8, colors=MUTED)
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    ax.grid(axis="x", color=GRID, alpha=0.7, linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def cover_page(pdf, capture, attr, hold):
    fig = fig_base("Auto Fillmore Phase 6", "Trade lifecycle audit - where the pips are lost after entry")
    overall = capture[capture.segment_type.eq("overall")].iloc[0]
    attr_map = {row.bucket: row for row in attr.itertuples(index=False)}
    best = hold[hold.segment_type.eq("hold_time_envelope")].sort_values("avg_pips", ascending=False).iloc[0]
    fig.text(0.055, 0.80, "Verdict: entry-generated damage, exit leakage second.", fontsize=26, color=RED_DARK, fontweight="bold")
    wrapped_note(
        fig,
        "The lifecycle evidence says the largest damage bucket is not slow loser babysitting. It is bad entries that never had meaningful favorable excursion, especially the known CLR fast failures. Exits still leak: winners leave pips behind and 24 red trades had at least 4p of MFE before closing red.",
        0.057,
        0.728,
        0.83,
        11.5,
    )
    card(fig, 0.055, 0.52, 0.205, 0.13, "Observed net", f"{one(attr_map['observed_net'].pips)}p", RED)
    card(fig, 0.285, 0.52, 0.205, 0.13, "Entry-failure proxy", f"{one(attr_map['entry_generated_terminal_proxy_mfe_le_2_mae_ge_6'].pips)}p", RED, "25 losses")
    card(fig, 0.515, 0.52, 0.205, 0.13, "Exit-reversal proxy", f"{one(attr_map['exit_generated_reversal_proxy_loss_with_mfe_ge_4'].pips)}p", GOLD, "24 losses")
    card(fig, 0.745, 0.52, 0.205, 0.13, "CLR fast failures", f"{one(attr_map['phase3_17_clr_fast_failures'].pips)}p", RED, "17 trades")
    card(fig, 0.055, 0.34, 0.205, 0.12, "Winner capture", f"{two(overall.winner_capture_median)}x", BLUE, "median of MFE")
    card(fig, 0.285, 0.34, 0.205, 0.12, "Loser capture", f"{two(overall.loser_capture_median)}x", RED, "MAE caveat")
    card(fig, 0.515, 0.34, 0.205, 0.12, "MFE left", f"{two(overall.winner_mfe_left_median)}p", GOLD, "median winner")
    card(fig, 0.745, 0.34, 0.205, 0.12, "Least-bad hold", short(best.segment, 8), TEAL, f"{two(best.avg_pips)}p/trade")
    wrapped_note(
        fig,
        "Telemetry warning: 88 of 113 losers realized more loss than stored MAE, so MAE does not appear exit-inclusive. Treat terminal-path ratios as directional evidence, not precise replay.",
        0.07,
        0.20,
        0.82,
        10.5,
        RED_DARK,
    )
    fig.text(0.055, 0.06, "Source: PHASE6_LIFECYCLE_ANALYSIS.md, phase6_entry_exit_attribution.csv, phase6_capture_efficiency.csv", fontsize=8, color=MUTED)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def capture_page(pdf, capture, hold):
    fig = fig_base("Capture And Hold", "Winners are not the mirror image of losers")
    gates = capture[capture.segment_type.eq("gate")].sort_values("net_pips")
    barh_ax(fig, gates.segment, gates.net_pips, 0.06, 0.52, 0.43, 0.29, "Net pips by gate")
    side = capture[capture.segment_type.eq("side")].copy()
    side["winner_capture_median"] = side["winner_capture_median"].map(two)
    side["loser_capture_median"] = side["loser_capture_median"].map(two)
    side["winner_mfe_left_median"] = side["winner_mfe_left_median"].map(lambda x: f"{two(x)}p")
    side["net_pips"] = side["net_pips"].map(lambda x: f"{one(x)}p")
    table_ax(fig, side[["segment", "n", "winner_capture_median", "loser_capture_median", "winner_mfe_left_median", "net_pips"]], 0.55, 0.60, 0.39, 0.16, 8.2)
    fig.text(0.55, 0.80, "Side lifecycle split", fontsize=12, fontweight="bold", color=INK)
    bins = hold[hold.segment_type.eq("hold_time_envelope")].copy()
    order = ["<=3m", "3-7m", "7-15m", "15-30m", ">30m"]
    bins["ord"] = bins["segment"].map({v: i for i, v in enumerate(order)})
    bins = bins.sort_values("ord")
    ax = fig.add_axes([0.07, 0.16, 0.84, 0.24], facecolor=BG)
    colors = [RED if v < 0 else TEAL for v in bins["avg_pips"]]
    ax.bar(bins["segment"], bins["avg_pips"], color=colors)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_title("Every hold-time bucket was negative", loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=8, colors=MUTED)
    ax.grid(axis="y", color=GRID, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    wrapped_note(fig, "This matters: shorter holding was least bad, not good. A lifecycle-only fix cannot rescue bad admissions unless entry quality also changes.", 0.07, 0.095, 0.84, 10, RED_DARK)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def runner_page(pdf, runner, caveat):
    fig = fig_base("Runner Regime And Caveats", "The runner clause changed behavior, but caveat exits were not the dominant mechanism")
    r = runner.copy()
    r["win_rate"] = r["win_rate"].map(pct)
    r["net_pips"] = r["net_pips"].map(lambda x: f"{one(x)}p")
    r["hold_median_min"] = r["hold_median_min"].map(lambda x: f"{one(x)}m")
    r["winner_capture_median"] = r["winner_capture_median"].map(two)
    r["loser_capture_median"] = r["loser_capture_median"].map(two)
    r["prompt_regime"] = r["prompt_regime"].map(lambda x: short(x, 30))
    table_ax(fig, r[["prompt_regime", "n", "win_rate", "net_pips", "hold_median_min", "winner_capture_median", "loser_capture_median", "runner_verdict"]], 0.055, 0.56, 0.89, 0.24, 7.6)
    wrapped_note(
        fig,
        "Runner/custom-exit v3 gets a committed measurable-harm verdict: 37.5% WR, -92.2p, 28.1m median hold, and only 0.42x winner capture. It did not produce the intended runner benefit.",
        0.06,
        0.47,
        0.86,
        10.5,
        RED_DARK,
    )
    c = caveat[caveat.test.isin(["hedge_density", "self_correction"])].copy()
    c["hold_median_min"] = c["hold_median_min"].map(lambda x: f"{one(x)}m")
    c["loser_hold_median_min"] = c["loser_hold_median_min"].map(lambda x: f"{one(x)}m")
    c["net_pips"] = c["net_pips"].map(lambda x: f"{one(x)}p")
    c["bucket"] = c["bucket"].map(lambda x: short(x, 28))
    table_ax(fig, c[["test", "bucket", "n", "loser_n", "hold_median_min", "loser_hold_median_min", "net_pips"]], 0.055, 0.13, 0.89, 0.24, 7.6)
    wrapped_note(fig, "Caveat laundering is real at entry, but Phase 6 did not find that self-correction text specifically creates longer-held losers. The non-caveat sample is tiny, so this is a cautious no.", 0.06, 0.065, 0.86, 9.5, INK)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def path_page(pdf, path, attr):
    fig = fig_base("Entry Versus Exit Attribution", "Terminal path evidence separates bad entries from exit leakage")
    overall = path[path.segment_type.eq("overall")].iloc[0]
    card(fig, 0.055, 0.70, 0.205, 0.12, "Entry failures", str(int(overall.early_entry_failure_n)), RED, f"{one(overall.early_entry_failure_pips)}p")
    card(fig, 0.285, 0.70, 0.205, 0.12, "Exit reversals", str(int(overall.exit_reversal_n)), GOLD, f"{one(overall.exit_reversal_pips)}p")
    card(fig, 0.515, 0.70, 0.205, 0.12, "Full adverse-ish", pct(overall.realized_loss_equals_mae_rate), BLUE, "losses near MAE")
    card(fig, 0.745, 0.70, 0.205, 0.12, "Winner MFE left", f"{two(overall.winner_mfe_left_median)}p", GOLD, "median")
    gates = path[path.segment_type.eq("gate")].copy().sort_values("early_entry_failure_pips")
    ax = fig.add_axes([0.07, 0.34, 0.84, 0.26], facecolor=BG)
    labels = [short(x, 24) for x in gates.segment]
    y = np.arange(len(labels))
    ax.barh(y - 0.18, gates.early_entry_failure_pips, height=0.34, color=RED, label="entry-failure proxy")
    ax.barh(y + 0.18, gates.exit_reversal_pips, height=0.34, color=GOLD, label="exit-reversal proxy")
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, alpha=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Bad-entry bucket dominates CLR damage", loc="left", fontsize=12, fontweight="bold", color=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    a = attr.copy()
    a["pips"] = a["pips"].map(lambda x: f"{one(x)}p")
    a["bucket"] = a["bucket"].map(lambda x: short(x, 34))
    table_ax(fig, a[["bucket", "n", "pips"]], 0.06, 0.08, 0.88, 0.16, 8.0)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def counter_page(pdf, counter, event):
    fig = fig_base("Counterfactual Bounds And Event Windows", "Useful diagnostics, not production rules")
    c = counter.copy()
    c["policy"] = c["policy"].map(lambda x: short(x, 42))
    c["counterfactual_pips"] = c["counterfactual_pips"].map(lambda x: f"{one(x)}p")
    c["delta_vs_actual_pips"] = c["delta_vs_actual_pips"].map(lambda x: f"{one(x)}p")
    c["profit_factor_pips"] = c["profit_factor_pips"].map(two)
    table_ax(fig, c[["policy", "counterfactual_pips", "delta_vs_actual_pips", "profit_factor_pips"]], 0.055, 0.60, 0.89, 0.22, 7.8)
    wrapped_note(
        fig,
        "The huge p50-MAE trailing result is a warning flare, not a patch recommendation: terminal MAE is not exit-inclusive on many losers, and path ordering is unknown.",
        0.06,
        0.50,
        0.86,
        10.5,
        RED_DARK,
    )
    e = event.copy()
    e["win_rate"] = e["win_rate"].map(pct)
    e["net_pips"] = e["net_pips"].map(lambda x: f"{one(x)}p")
    e["avg_pips"] = e["avg_pips"].map(lambda x: f"{one(x)}p")
    e["event_bucket"] = e["event_bucket"].map(lambda x: short(x, 28))
    table_ax(fig, e[["event_bucket", "n", "win_rate", "net_pips", "avg_pips", "sample_size_warning"]], 0.055, 0.22, 0.89, 0.17, 8.2)
    wrapped_note(
        fig,
        "Event-window check is bounded. FOMC and PCE timestamps are official; Japan CPI/Tokyo CPI and BOJ are partially date-only, so the report marks those assumptions explicitly.",
        0.06,
        0.12,
        0.86,
        10,
        INK,
    )
    fig.text(0.055, 0.06, "Primary report: PHASE6_LIFECYCLE_ANALYSIS.md", fontsize=8, color=MUTED)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    capture = pd.read_csv(FORENSIC_DIR / "phase6_capture_efficiency.csv")
    hold = pd.read_csv(FORENSIC_DIR / "phase6_hold_time_outcome.csv")
    runner = pd.read_csv(FORENSIC_DIR / "phase6_runner_regime_test.csv")
    caveat = pd.read_csv(FORENSIC_DIR / "phase6_caveat_exit_interaction.csv")
    path = pd.read_csv(FORENSIC_DIR / "phase6_path_asymmetry.csv")
    counter = pd.read_csv(FORENSIC_DIR / "phase6_lifecycle_counterfactuals.csv")
    attr = pd.read_csv(FORENSIC_DIR / "phase6_entry_exit_attribution.csv")
    event = pd.read_csv(FORENSIC_DIR / "phase6_event_window.csv")

    with PdfPages(OUT_PDF) as pdf:
        cover_page(pdf, capture, attr, hold)
        capture_page(pdf, capture, hold)
        runner_page(pdf, runner, caveat)
        path_page(pdf, path, attr)
        counter_page(pdf, counter, event)

    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
