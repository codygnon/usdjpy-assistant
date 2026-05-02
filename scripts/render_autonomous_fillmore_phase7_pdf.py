#!/usr/bin/env python3
"""Render an aesthetic PDF brief for Phase 7 interaction effects."""

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
OUT_PDF = FORENSIC_DIR / "Auto_Fillmore_Forensic_Audit_Phase7_Interactions.pdf"

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
OBSERVED_NET_PIPS = -308.0
OBSERVED_NET_USD = -7253.2365


def one(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x):.1f}"


def two(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x):.2f}"


def money(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"${float(x):,.0f}"


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
    ax.text(0.06, 0.34, value, transform=ax.transAxes, fontsize=18, color=color, fontweight="bold")
    if note:
        ax.text(0.06, 0.12, note, transform=ax.transAxes, fontsize=7.5, color=MUTED)
    return ax


def wrapped_note(fig, text: str, x: float, y: float, w: float, size: float = 10.5, color: str = INK):
    lines = textwrap.wrap(text, width=max(30, int(w * 115)))
    fig.text(x, y, "\n".join(lines), fontsize=size, color=color, ha="left", va="top", linespacing=1.35)


def table_ax(fig, df: pd.DataFrame, x: float, y: float, w: float, h: float, font_size: float = 7.6):
    ax = fig.add_axes([x, y, w, h])
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="left", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size)
    tbl.scale(1, 1.28)
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
    labels = [short(v, 36) for v in labels]
    values = np.asarray(values, dtype=float)
    colors = [RED if v < 0 else TEAL for v in values]
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=colors, alpha=0.92)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7.7)
    ax.tick_params(axis="x", labelsize=8, colors=MUTED)
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    ax.grid(axis="x", color=GRID, alpha=0.7, linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def cover_page(pdf, damage, preserved, veto):
    fig = fig_base("Auto Fillmore Phase 7", "Interaction effects - where layers combine into damage or preserved edge")
    core = damage[(damage.grid.eq("core_4d")) & (damage.n >= 10)].copy()
    full = damage[damage.grid.eq("full_10d")]
    top3_pips = core.sort_values("net_pips").head(3)
    top3_usd = core.sort_values("net_usd").head(3)
    pips_cov = max(0, -top3_pips["net_pips"].sum()) / abs(OBSERVED_NET_PIPS)
    usd_cov = max(0, -top3_usd["net_usd"].sum()) / abs(OBSERVED_NET_USD)
    robust = preserved[preserved["survives_leave_one_date"].eq(True)]
    pips_floor = veto[(veto["mode"].eq("greedy_by_pips")) & (veto["cumulative_coverage_abs_net_pips"] >= 0.70)].sort_values("sequence").head(1)
    usd_floor = veto[(veto["mode"].eq("greedy_by_usd")) & (veto["cumulative_coverage_abs_net_usd"] >= 0.70)].sort_values("sequence").head(1)
    fig.text(0.055, 0.80, "Damage is concentrated, but exact 10D cells are too thin.", fontsize=25, color=RED_DARK, fontweight="bold")
    wrapped_note(
        fig,
        "The exact multi-dimensional grid is underpowered: no 10D cell reaches N>=10. The useful signal appears in lower-cardinality interactions, especially sell-side caveat clusters across prompt regimes.",
        0.057,
        0.73,
        0.84,
        11.5,
    )
    card(fig, 0.055, 0.53, 0.205, 0.13, "Exact 10D N>=10", str(int((full.n >= 10).sum())), RED, f"max cell N={int(full.n.max())}")
    card(fig, 0.285, 0.53, 0.205, 0.13, "Top-3 pip cover", pct(pips_cov), RED)
    card(fig, 0.515, 0.53, 0.205, 0.13, "Top-3 USD cover", pct(usd_cov), GOLD)
    card(fig, 0.745, 0.53, 0.205, 0.13, "Robust edge cells", str(len(robust)), TEAL)
    card(fig, 0.055, 0.35, 0.435, 0.12, "70% pip floor", short(pips_floor.rule_id.iloc[0], 34) if not pips_floor.empty else "none", BLUE, f"{one(pips_floor.net_delta_pips.iloc[0])}p" if not pips_floor.empty else "")
    card(fig, 0.515, 0.35, 0.435, 0.12, "70% USD floor", short(usd_floor.rule_id.iloc[0], 34) if not usd_floor.empty else "none", BLUE, money(usd_floor.net_delta_usd.iloc[0]) if not usd_floor.empty else "")
    wrapped_note(
        fig,
        "The veto floor is diagnostic only. It shows the historical damage concentration Phase 9 must beat, not a final trading policy.",
        0.07,
        0.20,
        0.84,
        10.5,
        RED_DARK,
    )
    fig.text(0.055, 0.06, "Source: PHASE7_INTERACTION_EFFECTS.md and phase7_*.csv artifacts", fontsize=8, color=MUTED)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def damage_page(pdf, damage):
    fig = fig_base("Damage Concentration", "Core interaction cells with N>=10")
    core = damage[(damage.grid.eq("core_4d")) & (damage.n >= 10)].copy()
    core["label"] = core.apply(lambda r: f"{r['trigger_family']} | {r['side']} | {short(r['prompt_regime'], 20)}", axis=1)
    barh_ax(fig, core.sort_values("net_pips").head(7)["label"], core.sort_values("net_pips").head(7)["net_pips"], 0.06, 0.52, 0.87, 0.30, "Net pips by core cell")
    t = core.sort_values("net_usd").head(7).copy()
    t["cell"] = t["label"].map(lambda x: short(x, 38))
    t["WR"] = t["win_rate"].map(pct)
    t["pips"] = t["net_pips"].map(lambda x: f"{one(x)}p")
    t["USD"] = t["net_usd"].map(money)
    t["avg"] = t["avg_pips"].map(lambda x: f"{one(x)}p")
    table_ax(fig, t[["cell", "n", "WR", "pips", "USD", "avg"]], 0.055, 0.12, 0.89, 0.26, 7.8)
    wrapped_note(fig, "The same CLR sell caveat cell repeats across prompt regimes. Phase changes moved the loss around; they did not remove the failure mode.", 0.06, 0.44, 0.86, 10.5, RED_DARK)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def preserved_sizing_page(pdf, preserved, sizing):
    fig = fig_base("Preserved Edge And Sizing Compounding", "What to protect, and where variable sizing actually hurt")
    robust = preserved[preserved["survives_leave_one_date"].eq(True)].copy()
    robust["cell"] = robust["cell_signature"].map(lambda x: short(x, 46))
    robust["WR"] = robust["win_rate"].map(pct)
    robust["pips"] = robust["net_pips"].map(lambda x: f"{one(x)}p")
    robust["USD"] = robust["net_usd"].map(money)
    robust["LOO pips"] = robust["loo_min_net_pips"].map(lambda x: f"{one(x)}p")
    table_ax(fig, robust[["cell", "n", "WR", "pips", "USD", "LOO pips"]], 0.055, 0.57, 0.89, 0.22, 7.4)
    wrapped_note(fig, "Preserved edge does exist, but it is narrow: buy-side CLR in the Phase 2 zone-memory regime, especially 2-3.99 lots, and the Tuesday buy-CLR slice.", 0.06, 0.48, 0.86, 10.5, TEAL)
    s = sizing[sizing["outcome_type"].isin(["entry_failure", "exit_reversal", "other_loss", "winner"])].copy()
    ax = fig.add_axes([0.08, 0.15, 0.82, 0.25], facecolor=BG)
    colors = [RED if v < 0 else TEAL for v in s["sizing_delta_usd"]]
    ax.bar(s["outcome_type"], s["sizing_delta_usd"], color=colors)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_title("Sizing delta vs uniform 1 lot", loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8, colors=MUTED)
    ax.grid(axis="y", color=GRID, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    wrapped_note(fig, "Entry failures were expensive, but not uniquely oversized: 40% used above-median lots. Sizing damage is broader than one lifecycle bucket.", 0.06, 0.07, 0.86, 9.5, INK)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def stop_day_page(pdf, stop, day):
    fig = fig_base("Stop Overshoot + Day Effects", "The quiet anomaly and the calendar interaction")
    hyp = stop[stop["row_type"].eq("hypothesis_verdict")].copy()
    hyp["hypothesis"] = hyp["hypothesis"].map(lambda x: short(x, 34))
    table_ax(fig, hyp[["hypothesis", "evidence_weight", "verdict"]], 0.055, 0.63, 0.89, 0.18, 8.0)
    wrapped_note(fig, "Phase 3 stop overshoot is best explained by volatility/slippage or exit-fill mechanics: M5 ATR was 15.6p versus 5.1p in Phase A. The tighter-stop hypothesis is contradicted.", 0.06, 0.53, 0.86, 10.5, RED_DARK)
    d = day[day["row_type"].eq("day_summary")].copy().sort_values("net_pips")
    ax = fig.add_axes([0.08, 0.18, 0.82, 0.25], facecolor=BG)
    colors = [RED if v < 0 else TEAL for v in d["net_pips"]]
    ax.bar(d["day_of_week"], d["net_pips"], color=colors)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_title("Net pips by day", loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=8, colors=MUTED)
    ax.grid(axis="y", color=GRID, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    wrapped_note(fig, "Wednesday damage is mostly the runner/custom-exit v3 day; Tuesday's positive result survives only in the narrower buy-CLR slice, not day-only leave-one-date robustness.", 0.06, 0.08, 0.86, 9.5, INK)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def veto_page(pdf, veto):
    fig = fig_base("Minimum Veto Rule Floor", "In-sample damage avoidance benchmark for Phase 9")
    ind = veto[veto["mode"].eq("individual")].sort_values("net_delta_pips", ascending=False).copy()
    ind["rule"] = ind["rule_id"].map(lambda x: short(x, 38))
    ind["pips"] = ind["net_delta_pips"].map(lambda x: f"{one(x)}p")
    ind["USD"] = ind["net_delta_usd"].map(money)
    ind["pip cover"] = ind["coverage_abs_net_pips"].map(pct)
    ind["USD cover"] = ind["coverage_abs_net_usd"].map(pct)
    table_ax(fig, ind[["rule", "blocked_trades", "blocked_winners", "blocked_losers", "pips", "USD", "pip cover", "USD cover"]].head(7), 0.055, 0.56, 0.89, 0.25, 7.5)
    greedy = veto[veto["mode"].eq("greedy_by_usd")].copy().sort_values("sequence")
    ax = fig.add_axes([0.08, 0.18, 0.82, 0.25], facecolor=BG)
    ax.plot(greedy["sequence"], greedy["cumulative_coverage_abs_net_usd"] * 100, marker="o", color=BLUE, linewidth=2.2, label="USD")
    ax.plot(greedy["sequence"], greedy["cumulative_coverage_abs_net_pips"] * 100, marker="o", color=TEAL, linewidth=2.2, label="pips")
    ax.axhline(70, color=RED, linewidth=1.2, linestyle="--")
    ax.set_title("Greedy cumulative recovery vs observed net loss", loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.set_xlabel("rules added", fontsize=8, color=MUTED)
    ax.set_ylabel("% recovered", fontsize=8, color=MUTED)
    ax.tick_params(axis="both", labelsize=8, colors=MUTED)
    ax.grid(color=GRID, alpha=0.8)
    ax.legend(frameon=False, fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    wrapped_note(fig, "V1 alone clears the pip threshold; V1+V2 clears the USD threshold. This is a diagnostic floor, not a prescription.", 0.06, 0.08, 0.86, 10, RED_DARK)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    damage = pd.read_csv(FORENSIC_DIR / "phase7_damage_concentration.csv")
    preserved = pd.read_csv(FORENSIC_DIR / "phase7_preserved_edge_cells.csv")
    sizing = pd.read_csv(FORENSIC_DIR / "phase7_sizing_compounding.csv")
    stop = pd.read_csv(FORENSIC_DIR / "phase7_stop_overshoot_anomaly.csv")
    day = pd.read_csv(FORENSIC_DIR / "phase7_day_of_week_interaction.csv")
    veto = pd.read_csv(FORENSIC_DIR / "phase7_minimum_veto_rules.csv")

    with PdfPages(OUT_PDF) as pdf:
        cover_page(pdf, damage, preserved, veto)
        damage_page(pdf, damage)
        preserved_sizing_page(pdf, preserved, sizing)
        stop_day_page(pdf, stop, day)
        veto_page(pdf, veto)

    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
