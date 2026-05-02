#!/usr/bin/env python3
"""Render an aesthetic PDF brief for Phase 4 and Phase 5 forensic findings."""

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
OUT_PDF = FORENSIC_DIR / "Auto_Fillmore_Forensic_Audit_Phases_4_5.pdf"

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


def money(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"${float(x):,.0f}"


def one(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x):.1f}"


def pct(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x) * 100:.1f}%"


def short(text: Any, n: int = 52) -> str:
    s = str(text) if text is not None and not pd.isna(text) else ""
    s = " ".join(s.split())
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
    ax.text(0.06, 0.72, label.upper(), transform=ax.transAxes, fontsize=8.5, color=MUTED, fontweight="bold")
    ax.text(0.06, 0.34, value, transform=ax.transAxes, fontsize=20, color=color, fontweight="bold")
    if note:
        ax.text(0.06, 0.12, note, transform=ax.transAxes, fontsize=8, color=MUTED)
    return ax


def wrapped_note(fig, text: str, x: float, y: float, w: float, size: float = 10, color: str = INK):
    lines = textwrap.wrap(text, width=max(30, int(w * 115)))
    fig.text(x, y, "\n".join(lines), fontsize=size, color=color, ha="left", va="top", linespacing=1.35)


def table_ax(fig, df: pd.DataFrame, x: float, y: float, w: float, h: float, font_size: float = 7.8):
    ax = fig.add_axes([x, y, w, h])
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="left", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size)
    tbl.scale(1, 1.35)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#E6DFD2")
        if row == 0:
            cell.set_facecolor("#EFE8DC")
            cell.set_text_props(weight="bold", color=INK)
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 else "#FBF8F1")
            cell.set_text_props(color=INK)
    return ax


def barh_ax(fig, labels, values, x: float, y: float, w: float, h: float, title: str, color_negative: str = RED, color_positive: str = TEAL):
    ax = fig.add_axes([x, y, w, h], facecolor=BG)
    labels = [short(v, 32) for v in labels]
    values = np.asarray(values, dtype=float)
    colors = [color_negative if v < 0 else color_positive for v in values]
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


def phase4_cover(pdf: PdfPages, p4_clusters, sell_buy, fast, cognitive, prompt_rule):
    fig = fig_base("Auto Fillmore Phase 4", "LLM reasoning forensics - why the model talked itself into bad trades")
    fig.text(0.055, 0.80, "The reasoning layer is indicted.", fontsize=30, color=RED_DARK, fontweight="bold")
    wrapped_note(
        fig,
        "The model repeatedly named the reason not to trade, then used that acknowledgement as permission to trade anyway. "
        "This showed up as critical-level mixed caveat trades, momentum caveat trades, and overconfident level language on fast failures.",
        0.057,
        0.725,
        0.82,
        12,
    )
    card(fig, 0.055, 0.49, 0.205, 0.13, "Closed trades", "241", BLUE)
    card(fig, 0.285, 0.49, 0.205, 0.13, "Net result", "-308p", RED)
    card(fig, 0.515, 0.49, 0.205, 0.13, "Net USD", "$-7,253", RED)
    card(fig, 0.745, 0.49, 0.205, 0.13, "Sell CLR", "$-2,082", RED, "43 trades")
    top = p4_clusters.sort_values("net_usd").head(5).copy()
    barh_ax(fig, top["rationale_cluster"], top["net_usd"], 0.07, 0.11, 0.50, 0.28, "Worst rationale archetypes by USD")
    s = sell_buy.copy()
    s["win_rate"] = s["win_rate"].map(pct)
    s["net_pips"] = s["net_pips"].map(one)
    s["net_usd"] = s["net_usd"].map(money)
    table_ax(fig, s[["side", "closed", "win_rate", "net_pips", "net_usd"]], 0.62, 0.14, 0.32, 0.22, 8.5)
    fig.text(0.62, 0.39, "Sell-CLR vs Buy-CLR", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.055, 0.06, "Source artifacts: PHASE4_REASONING_FORENSICS.md, phase4_rationale_clusters.csv, phase4_sell_clr_vs_buy_clr.csv", fontsize=8, color=MUTED)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def phase4_details(pdf: PdfPages, fast, cognitive, prompt_rule):
    fig = fig_base("Phase 4 Evidence", "Caveat laundering, overconfident fast failures, and prompt-rule leakage")
    fast_counts = fast["rationale_evidence_claim"].value_counts().reset_index()
    fast_counts.columns = ["fast_failure_claim", "n"]
    table_ax(fig, fast_counts, 0.06, 0.61, 0.38, 0.22, 8.5)
    fig.text(0.06, 0.84, "17 CLR fast-failure rows", fontsize=12, fontweight="bold", color=INK)
    wrapped_note(
        fig,
        "16 of 17 fast failures carried strong/clean/fresh/material style reasoning against weak, thin, mixed, or ambiguous snapshot evidence. "
        "This is not missing data; this is the model overstating what the level evidence meant.",
        0.48,
        0.82,
        0.44,
        11,
        RED_DARK,
    )
    cog = cognitive[cognitive["test"].isin(["confirmation_bias", "self_correction_followthrough"])].copy()
    cog["place_rate"] = cog["place_rate"].map(pct)
    cog["win_rate"] = cog["win_rate"].map(pct)
    cog["net_usd"] = cog["net_usd"].map(money)
    cog["bucket"] = cog["bucket"].map(lambda x: short(x, 34))
    table_ax(fig, cog[["test", "bucket", "calls", "placed", "win_rate", "net_usd"]], 0.055, 0.28, 0.89, 0.24, 7.7)
    pr = prompt_rule.copy()
    pr["compliance_rate"] = pr["compliance_rate"].map(pct)
    pr["violation_closed_net_usd"] = pr["violation_closed_net_usd"].map(money)
    pr["rule"] = pr["rule"].map(lambda x: short(x, 42))
    table_ax(fig, pr[["rule", "universe_n", "compliant_n", "violation_n", "compliance_rate", "violation_closed_net_usd"]], 0.055, 0.055, 0.89, 0.16, 7.7)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def phase5_cover(pdf: PdfPages, decomp, corr, dist):
    fig = fig_base("Auto Fillmore Phase 5", "Sizing logic audit - how variable size turned a bad system into a much worse one")
    actual = decomp.loc[decomp["component"] == "actual_total_pnl", "component_usd"].iloc[0]
    one_lot = decomp.loc[decomp["component"] == "admission_cost_at_uniform_1_lot", "component_usd"].iloc[0]
    sizing = decomp.loc[decomp["component"] == "sizing_amplification_delta_vs_1_lot", "component_usd"].iloc[0]
    corr_pips = corr[(corr.scope_type == "overall") & (corr.target == "realized_pips") & (corr.method == "spearman")]["correlation"].iloc[0]
    fig.text(0.055, 0.80, "Verdict: random / edge-blind sizing.", fontsize=28, color=RED_DARK, fontweight="bold")
    wrapped_note(
        fig,
        "Chosen lots did not show a defensible positive relationship to realized edge. The same 241 entries at uniform 1 lot would still lose, but variable sizing added most of the dollar damage.",
        0.057,
        0.735,
        0.82,
        12,
    )
    card(fig, 0.055, 0.54, 0.205, 0.13, "Observed P&L", money(actual), RED)
    card(fig, 0.285, 0.54, 0.205, 0.13, "Uniform 1 lot", money(one_lot), RED)
    card(fig, 0.515, 0.54, 0.205, 0.13, "Sizing delta", money(sizing), RED)
    card(fig, 0.745, 0.54, 0.205, 0.13, "Lots-vs-pips rho", f"{corr_pips:.3f}", GOLD)
    d = decomp[decomp["reconciles_to_actual"].astype(str).eq("True")].copy()
    labels = ["Uniform 1 lot\nadmission", "Sizing\namplification", "Observed\nP&L"]
    values = [one_lot, sizing, actual]
    ax = fig.add_axes([0.08, 0.16, 0.48, 0.28], facecolor=BG)
    ax.bar(labels, values, color=[GOLD, RED, RED_DARK], width=0.55)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_title("P&L decomposition", loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=8, colors=MUTED)
    ax.grid(axis="y", color=GRID, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    overall = dist[dist["segment_type"].eq("overall")].copy()
    overall["mean_lots"] = overall["mean_lots"].map(one)
    overall["median_lots"] = overall["median_lots"].map(one)
    overall["p75_lots"] = overall["p75_lots"].map(one)
    overall["share_ge_4"] = overall["share_ge_4"].map(pct)
    overall["share_ge_8"] = overall["share_ge_8"].map(pct)
    table_ax(fig, overall[["n", "mean_lots", "median_lots", "p75_lots", "share_ge_4", "share_ge_8"]], 0.62, 0.20, 0.32, 0.17, 8.5)
    fig.text(0.62, 0.41, "Overall lot distribution", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.055, 0.06, "Source artifacts: PHASE5_SIZING_AUDIT.md, phase5_pl_decomposition.csv, phase5_edge_size_correlation.csv", fontsize=8, color=MUTED)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def phase5_details(pdf: PdfPages, dist, corr, caveat, policies, anomaly, telemetry):
    fig = fig_base("Phase 5 Evidence", "Where the size damage sits")
    lot = dist[dist["segment_type"].eq("gate")].sort_values("net_usd").copy()
    barh_ax(fig, lot["segment"], lot["net_usd"], 0.06, 0.53, 0.42, 0.28, "Gate net USD")
    side = dist[dist["segment_type"].eq("side")].copy()
    side["net_usd"] = side["net_usd"].map(money)
    side["mean_lots"] = side["mean_lots"].map(one)
    side["share_ge_4"] = side["share_ge_4"].map(pct)
    table_ax(fig, side[["segment", "n", "mean_lots", "share_ge_4", "net_usd"]], 0.55, 0.60, 0.38, 0.17, 8.5)
    cv = caveat[caveat["test"].eq("hedge_density")].copy()
    cv["mean_lots"] = cv["mean_lots"].map(one)
    cv["net_usd"] = cv["net_usd"].map(money)
    cv["bucket"] = cv["bucket"].map(lambda x: short(x, 24))
    table_ax(fig, cv[["bucket", "n", "mean_lots", "net_usd"]], 0.55, 0.33, 0.38, 0.18, 8.5)
    pol = policies.sort_values("counterfactual_usd", ascending=False).copy()
    pol["policy"] = pol["policy"].map(lambda x: short(x, 38))
    pol["counterfactual_usd"] = pol["counterfactual_usd"].map(money)
    pol["delta_vs_actual_usd"] = pol["delta_vs_actual_usd"].map(money)
    table_ax(fig, pol[["policy", "counterfactual_usd", "delta_vs_actual_usd", "trades_skipped"]].head(7), 0.06, 0.07, 0.87, 0.19, 7.7)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    fig = fig_base("8+ Lot Anomaly + Telemetry", "The one apparent large-lot exception is fragile")
    an = anomaly[anomaly["segment_type"].isin(["target", "comparison", "target_by_prompt_regime", "target_leave_one_date_out"])].head(10).copy()
    an["segment"] = an["segment"].map(lambda x: short(x, 34))
    an["win_rate"] = an["win_rate"].map(pct)
    an["net_usd"] = an["net_usd"].map(money)
    an["avg_usd"] = an["avg_usd"].map(money)
    table_ax(fig, an[["segment_type", "segment", "n", "win_rate", "net_pips", "net_usd", "avg_usd"]], 0.055, 0.52, 0.89, 0.30, 7.6)
    wrapped_note(
        fig,
        "Committed anomaly verdict: compositional artifact. The 8+ CLR cell is 24 buys / 1 sell, mostly Phase A, and leave-one-date checks move it from positive to negative. It is not a robust right-to-size-up rule.",
        0.06,
        0.44,
        0.86,
        11,
        RED_DARK,
    )
    tel = telemetry.copy()
    tel["mechanism_enabled"] = tel["mechanism_enabled"].map(lambda x: short(x, 70))
    table_ax(fig, tel[["priority", "field", "mechanism_enabled"]], 0.055, 0.08, 0.89, 0.27, 7.7)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(FORENSIC_DIR / name, low_memory=False)


def main() -> None:
    p4_clusters = load_csv("phase4_rationale_clusters.csv")
    sell_buy = load_csv("phase4_sell_clr_vs_buy_clr.csv")
    fast = load_csv("phase4_clr_fast_failure_rationale_audit.csv")
    cognitive = load_csv("phase4_cognitive_failure_tests.csv")
    prompt_rule = load_csv("phase4_prompt_rule_compliance.csv")
    decomp = load_csv("phase5_pl_decomposition.csv")
    corr = load_csv("phase5_edge_size_correlation.csv")
    dist = load_csv("phase5_sizing_distribution.csv")
    caveat = load_csv("phase5_caveat_sizing_interaction.csv")
    policies = load_csv("phase5_counterfactual_policies.csv")
    anomaly = load_csv("phase5_8lot_clr_anomaly.csv")
    telemetry = load_csv("phase5_required_telemetry.csv")

    with PdfPages(OUT_PDF) as pdf:
        phase4_cover(pdf, p4_clusters, sell_buy, fast, cognitive, prompt_rule)
        phase4_details(pdf, fast, cognitive, prompt_rule)
        phase5_cover(pdf, decomp, corr, dist)
        phase5_details(pdf, dist, corr, caveat, policies, anomaly, telemetry)

    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
