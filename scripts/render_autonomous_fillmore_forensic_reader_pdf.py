#!/usr/bin/env python3
"""Render a scan-friendly reader edition of the Auto Fillmore audit.

Unlike the full source book, this PDF is designed for human reading:
larger type, phase-by-phase cards, concise evidence tables, and the full
artifact map so no phase is lost.
"""

from __future__ import annotations

import math
import re
import textwrap
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research_out" / "autonomous_fillmore_forensic_20260501"
OUT_PDF = OUT / "Auto_Fillmore_Forensic_Audit_Phases_1_to_9_Reader_Edition.pdf"

BG = "#F6F1E8"
PAPER = "#FFFDF8"
INK = "#172233"
MUTED = "#65717F"
GRID = "#D8CFC0"
RED = "#A84037"
RED_DARK = "#74302B"
TEAL = "#176B6D"
GOLD = "#B7852F"
BLUE = "#335C89"
PANEL = "#FFFFFF"
SOFT = "#F1E8DA"

PAGE_NO = 0


def safe(text: object) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    return str(text).replace("$", r"\$")


def money(x: float | int | str) -> str:
    try:
        v = float(x)
    except Exception:
        return safe(x)
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v):,.0f}"


def money2(x: float | int | str) -> str:
    try:
        v = float(x)
    except Exception:
        return safe(x)
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v):,.2f}"


def pips(x: float | int | str) -> str:
    try:
        v = float(x)
    except Exception:
        return safe(x)
    return f"{v:+.1f}p"


def pct(x: float | int | str) -> str:
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return safe(x)


def short(text: object, n: int = 44) -> str:
    s = "" if text is None or pd.isna(text) else " ".join(str(text).split())
    return s if len(s) <= n else s[: n - 3].rstrip() + "..."


def wrap(text: str, width: int = 90) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False, break_on_hyphens=False))


def fig_base(title: str, subtitle: str = "", phase: str = ""):
    global PAGE_NO
    PAGE_NO += 1
    fig = plt.figure(figsize=(13.2, 8.5), facecolor=BG)
    fig.add_artist(plt.Rectangle((0.035, 0.045), 0.93, 0.89, transform=fig.transFigure, facecolor=PAPER, edgecolor=GRID, linewidth=1.0))
    fig.text(0.065, 0.91, safe(title), fontsize=21, fontweight="bold", color=INK, ha="left")
    if subtitle:
        fig.text(0.065, 0.875, safe(subtitle), fontsize=9.5, color=MUTED, ha="left")
    if phase:
        fig.text(0.935, 0.91, safe(phase), fontsize=9.5, color=RED_DARK, fontweight="bold", ha="right")
    fig.add_artist(plt.Line2D([0.065, 0.935], [0.855, 0.855], color=GRID, linewidth=0.9))
    fig.text(0.935, 0.055, str(PAGE_NO), fontsize=8, color=MUTED, ha="right")
    return fig


def add_card(fig, x, y, w, h, label: str, value: str, color=INK, note: str = ""):
    fig.add_artist(plt.Rectangle((x, y), w, h, transform=fig.transFigure, facecolor=PANEL, edgecolor=GRID, linewidth=0.8))
    fig.text(x + 0.018, y + h - 0.032, safe(label.upper()), fontsize=7.5, color=MUTED, fontweight="bold", ha="left")
    fs = 16 if len(value) < 20 else 12.5
    fig.text(x + 0.018, y + h * 0.38, safe(value), fontsize=fs, color=color, fontweight="bold", ha="left")
    if note:
        fig.text(x + 0.018, y + 0.022, safe(note), fontsize=7.3, color=MUTED, ha="left")


def add_note(fig, x, y, w, text: str, color=INK, size=9.5, weight="normal", width=70):
    fig.text(x, y, safe(wrap(text, width)), fontsize=size, color=color, fontweight=weight, ha="left", va="top", linespacing=1.28)


def add_bullets(fig, x, y, items: Iterable[str], color=INK, size=8.8, width=58, gap=0.056):
    yy = y
    for item in items:
        fig.text(x, yy, safe("• " + wrap(str(item), width).replace("\n", "\n  ")), fontsize=size, color=color, ha="left", va="top", linespacing=1.22)
        yy -= gap + 0.012 * max(0, wrap(str(item), width).count("\n"))
    return yy


def add_table(fig, df: pd.DataFrame, x, y, w, h, font_size=7.5, col_widths=None):
    ax = fig.add_axes([x, y, w, h])
    ax.axis("off")
    clean = df.copy()
    for col in clean.columns:
        clean[col] = clean[col].map(safe)
    tbl = ax.table(
        cellText=clean.values,
        colLabels=[safe(c) for c in clean.columns],
        cellLoc="left",
        colWidths=col_widths,
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size)
    tbl.scale(1, 1.24)
    for (row, _col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#E5DCCE")
        if row == 0:
            cell.set_facecolor("#EFE4D4")
            cell.set_text_props(weight="bold", color=INK)
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 else "#FBF6EE")
            cell.set_text_props(color=INK)
    return ax


def add_barh(fig, labels, values, x, y, w, h, title: str, value_fmt=pips):
    ax = fig.add_axes([x, y, w, h], facecolor=PAPER)
    vals = [float(v) for v in values]
    labs = [short(v, 34) for v in labels]
    colors = [RED if v < 0 else TEAL for v in vals]
    ax.barh(range(len(vals)), vals, color=colors, alpha=0.94)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labs, fontsize=7.6, color=INK)
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, fontweight="bold")
    ax.grid(axis="x", color=GRID, alpha=0.7)
    ax.tick_params(axis="x", labelsize=7.5, colors=MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for idx, val in enumerate(vals):
        ax.text(val, idx, " " + value_fmt(val), va="center", fontsize=7, color=INK)


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(OUT / name)


def cover(pdf):
    fig = plt.figure(figsize=(13.2, 8.5), facecolor=BG)
    fig.add_artist(plt.Rectangle((0.045, 0.06), 0.91, 0.86, transform=fig.transFigure, facecolor=PAPER, edgecolor=GRID, linewidth=1.1))
    fig.text(0.08, 0.80, "AUTO FILLMORE", fontsize=38, fontweight="bold", color=INK)
    fig.text(0.08, 0.725, "Forensic Audit Reader Edition", fontsize=25, fontweight="bold", color=RED_DARK)
    fig.text(0.08, 0.675, "Phases 1-9 | concise, scan-first, evidence-complete", fontsize=12, color=MUTED)
    add_note(
        fig,
        0.08,
        0.60,
        0.70,
        "This version keeps the full investigation arc but stops punishing your eyes: every phase, verdict, evidence gap, preserved edge, and redesign constraint is represented in a readable visual format.",
        size=13,
        width=78,
    )
    cards = [
        (0.08, 0.43, "Population", "241 trades", RED),
        (0.38, 0.43, "Net result", "-308.0p / -$7,253", RED),
        (0.08, 0.29, "Core failure", "Sell caveat-template collapse", RED_DARK),
        (0.38, 0.29, "Phase 9 replay", "+324.3p / +$6,420.90", TEAL),
    ]
    for x, y, label, value, color in cards:
        add_card(fig, x, y, 0.26, 0.105, label, value, color)
    add_note(
        fig,
        0.08,
        0.27,
        0.82,
        "The giant source book still exists for line-by-line archival reading. This reader edition is the one to use when making decisions.",
        color=BLUE,
        size=11.5,
        weight="bold",
        width=90,
    )
    fig.text(0.08, 0.105, "Source: PHASE1_BASELINE.md through PHASE9_OVERHAUL_BLUEPRINT.md plus phase CSV artifacts", fontsize=8.5, color=MUTED)
    pdf.savefig(fig)
    plt.close(fig)


def investigation_map(pdf):
    fig = fig_base("Investigation Map", "What each phase answered, in one page", "Overview")
    phases = [
        ("1 Baseline", "Data trustworthy? What is the loss shape?", "53.1% WR still loses: losers are too large."),
        ("2 Gates", "Which gate layer breaks?", "CLR sell collapses; momentum has pip edge but USD bleed."),
        ("3 Snapshot", "Did the LLM see bad data?", "Raw data is symmetric; missing side-normalized level/exposure."),
        ("4 Reasoning", "How did the model think?", "Sell-side caveat template collapse is primary cognitive failure."),
        ("5 Sizing", "Did size track edge?", "No. Random/edge-blind sizing adds -$5,187."),
        ("6 Lifecycle", "Entry or exit?", "Entry-generated damage dominates; runner clause harms."),
        ("7 Interactions", "Which combinations matter?", "Damage concentrated; preserved buy-CLR edge exists."),
        ("8 Synthesis", "What is the causal hierarchy?", "Primary: sell caveats, sizing, admission, entry failure."),
        ("9 Blueprint", "What replaces Fillmore?", "Deterministic shell, constrained LLM, replay passes floor."),
    ]
    y = 0.78
    for i, (phase, q, verdict) in enumerate(phases):
        x = 0.075 if i < 5 else 0.525
        yy = y - (i % 5) * 0.135
        fig.add_artist(plt.Rectangle((x, yy - 0.075), 0.39, 0.105, transform=fig.transFigure, facecolor=PANEL, edgecolor=GRID, linewidth=0.8))
        fig.text(x + 0.015, yy, phase, fontsize=10.5, color=RED_DARK, fontweight="bold")
        fig.text(x + 0.015, yy - 0.030, safe(wrap(q, 48)), fontsize=7.9, color=MUTED, va="top")
        fig.text(x + 0.015, yy - 0.061, safe(wrap(verdict, 52)), fontsize=8.3, color=INK, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def whole_story(pdf):
    fig = fig_base("The Whole Story", "The causal chain, compressed but complete", "Executive")
    add_note(fig, 0.075, 0.80, 0.84, "Fillmore did not fail because of one missing max-loss switch. It failed because weak admissions were allowed through, the LLM converted caveats into permission, and then random sizing amplified the negative pip base into a large dollar drawdown.", color=RED_DARK, size=13.5, weight="bold", width=95)
    chain = [
        ("Input", "Raw support/resistance, no side-normalized level quality, no exposure/risk-after-fill."),
        ("Gate", "CLR dominates volume; sell-CLR loses badly while buy-CLR contains preserved edge."),
        ("Reasoning", "Sell-side caveat templates treat 'mixed but tradeable' as enough."),
        ("Admission", "The placed-trade set loses -308.0p even at uniform 1 lot."),
        ("Sizing", "Variable lots add -$5,187.24 because size is edge-blind."),
        ("Lifecycle", "Entry failure dominates; exits leak but do not explain the main wound."),
        ("Telemetry", "Missing skip outcomes/path replay made the system slow to learn."),
        ("Blueprint", "Deterministic shell + constrained LLM + sizing/exit rules + full telemetry."),
    ]
    x0 = 0.075
    for idx, (name, detail) in enumerate(chain):
        x = x0 + (idx % 4) * 0.22
        y = 0.56 - (idx // 4) * 0.245
        fig.add_artist(plt.Rectangle((x, y), 0.19, 0.15, transform=fig.transFigure, facecolor=PANEL, edgecolor=GRID, linewidth=0.8))
        fig.text(x + 0.012, y + 0.112, name, fontsize=10, color=BLUE, fontweight="bold")
        fig.text(x + 0.012, y + 0.082, safe(wrap(detail, 27)), fontsize=7.8, color=INK, va="top", linespacing=1.18)
    add_note(fig, 0.075, 0.11, 0.84, "Phase 9's job is not fewer trades for its own sake. It is better expectancy: block the historically toxic failure modes, keep the narrow preserved edge, and force sizing/exits/telemetry into deterministic, replayable layers.", color=TEAL, size=11.0, weight="bold", width=100)
    pdf.savefig(fig)
    plt.close(fig)


def phase1(pdf):
    prompt = load("phase1_by_prompt_version.csv")
    gate = load("phase1_by_trigger_family.csv")
    side = load("phase1_by_side.csv")
    lots = load("phase1_by_lot_bucket.csv")
    fig = fig_base("Phase 1 - Baseline", "Data integrity, coverage, and the actual loss shape", "Phase 1")
    add_card(fig, 0.075, 0.69, 0.18, 0.11, "Closed trades", "241", RED)
    add_card(fig, 0.275, 0.69, 0.18, 0.11, "Win rate", "53.1%", GOLD)
    add_card(fig, 0.475, 0.69, 0.18, 0.11, "Net pips", "-308.0p", RED)
    add_card(fig, 0.675, 0.69, 0.23, 0.11, "Net USD", "-$7,253.24", RED)
    add_note(fig, 0.075, 0.60, 0.85, "Baseline verdict: this is structurally negative, not marginally unlucky. A 53.1% win rate still loses because the average loser is 1.64x the average winner in pips and about 1.86x in USD.", color=RED_DARK, size=11.0, weight="bold", width=105)
    table = pd.DataFrame(
        {
            "segment": ["avg winner", "avg loser", "loss/win"],
            "pips": ["+5.39p", "-8.83p", "1.64x"],
            "USD": ["+$88.06", "-$163.94", "1.86x"],
            "meaning": ["wins too small", "losses too large", "negative expectancy"],
        }
    )
    add_table(fig, table, 0.075, 0.36, 0.42, 0.16, 8.5)
    lot_t = lots[["_lot_bucket", "n", "win_rate", "net_pips", "net_usd"]].copy()
    lot_t["win_rate"] = lot_t["win_rate"].map(pct)
    lot_t["net_pips"] = lot_t["net_pips"].map(pips)
    lot_t["net_usd"] = lot_t["net_usd"].map(money)
    lot_t = lot_t.rename(columns={"_lot_bucket": "lot band", "win_rate": "WR", "net_pips": "pips", "net_usd": "USD"})
    add_table(fig, lot_t, 0.53, 0.28, 0.39, 0.29, 7.4)
    add_barh(fig, side["side"], side["net_pips"], 0.075, 0.11, 0.38, 0.18, "Side contribution")
    add_barh(fig, gate.head(5)["trigger_family"], gate.head(5)["net_pips"], 0.53, 0.08, 0.39, 0.18, "Gate contribution")
    pdf.savefig(fig)
    plt.close(fig)

    fig = fig_base("Phase 1 - What The Baseline Proved", "Prompt, side, gate, sizing, and evidence limits", "Phase 1")
    p = prompt[["prompt_version", "n", "win_rate", "net_pips", "net_usd"]].copy()
    p["prompt_version"] = p["prompt_version"].map(lambda x: short(x, 35))
    p["win_rate"] = p["win_rate"].map(pct)
    p["net_pips"] = p["net_pips"].map(pips)
    p["net_usd"] = p["net_usd"].map(money)
    p = p.rename(columns={"prompt_version": "prompt version", "win_rate": "WR", "net_pips": "pips", "net_usd": "USD"})
    add_table(fig, p, 0.075, 0.58, 0.55, 0.18, 8.0)
    add_bullets(
        fig,
        0.67,
        0.75,
        [
            "April 16-May 1 coverage is enough for placed-trade forensics.",
            "Skip-side outcomes are not audit-grade because forward outcomes were missing.",
            "Pips are the clean performance metric; USD exposes sizing damage.",
            "The investigation must target expectancy and asymmetry, not just max daily loss.",
        ],
        width=40,
    )
    add_note(fig, 0.075, 0.43, 0.82, "Phase 1 handoff: find out why sellers, CLR, and large-size bands are so bad, and whether the LLM is adding or destroying edge after the code gate fires.", color=BLUE, size=11, weight="bold", width=100)
    add_note(fig, 0.075, 0.28, 0.82, "Evidence gaps: no honest gate-only counterfactual, incomplete skip forward outcomes, and telemetry asymmetry between sidecar LLM calls and joined suggestion rows.", color=MUTED, size=9.5, width=100)
    pdf.savefig(fig)
    plt.close(fig)


def phase2(pdf):
    summary = load("phase2_gate_summary.csv")
    side = load("phase2_gate_side.csv")
    fig = fig_base("Phase 2 - Code Gate Audit", "Which gates are broken, conditional, or salvageable?", "Phase 2")
    t = summary[["gate", "n", "win_rate", "net_pips", "net_usd", "expectancy_pips", "mae_p75", "mfe_p75", "hold_median"]].copy()
    t["gate"] = t["gate"].map(lambda x: short(x, 27))
    for c in ["win_rate"]:
        t[c] = t[c].map(pct)
    for c in ["net_pips", "expectancy_pips", "mae_p75", "mfe_p75"]:
        t[c] = t[c].map(lambda v: f"{float(v):.1f}p")
    t["net_usd"] = t["net_usd"].map(money)
    t["hold_median"] = t["hold_median"].map(lambda v: f"{float(v):.1f}m")
    t = t.rename(columns={"win_rate": "WR", "net_pips": "pips", "net_usd": "USD", "expectancy_pips": "exp", "mae_p75": "MAE75", "mfe_p75": "MFE75", "hold_median": "hold"})
    add_table(fig, t, 0.055, 0.51, 0.89, 0.25, 7.2)
    add_note(fig, 0.075, 0.42, 0.84, "Verdict: Critical Level Reaction needs redesign, not deletion. Momentum is pip-positive but dollar-negative. Mean reversion is too small to convict. Non-primary gates are material damage and should not be hidden inside the primary gate architecture.", color=RED_DARK, size=11.0, weight="bold", width=110)
    add_barh(fig, summary["gate"], summary["net_pips"], 0.075, 0.12, 0.38, 0.22, "Net pips by gate")
    add_barh(fig, summary["gate"], summary["net_usd"], 0.54, 0.12, 0.38, 0.22, "Net USD by gate", value_fmt=money)
    pdf.savefig(fig)
    plt.close(fig)

    fig = fig_base("Phase 2 - CLR And Sell Pathology", "The largest gate damage center", "Phase 2")
    clr = side[side["gate_group"].eq("Critical Level Reaction")][["side", "n", "win_rate", "net_pips", "net_usd", "avg_win_pips", "avg_loss_pips"]].copy()
    clr["win_rate"] = clr["win_rate"].map(pct)
    for c in ["net_pips", "avg_win_pips", "avg_loss_pips"]:
        clr[c] = clr[c].map(pips)
    clr["net_usd"] = clr["net_usd"].map(money)
    clr = clr.rename(columns={"win_rate": "WR", "net_pips": "pips", "net_usd": "USD", "avg_win_pips": "avg win", "avg_loss_pips": "avg loss"})
    add_table(fig, clr, 0.075, 0.62, 0.45, 0.16, 8.8)
    add_card(fig, 0.58, 0.66, 0.15, 0.10, "Sell CLR", "-184.3p", RED)
    add_card(fig, 0.75, 0.66, 0.15, 0.10, "Buy CLR", "+0.6p", TEAL)
    add_note(fig, 0.075, 0.48, 0.84, "Sell-side aggregate damage is not explained by USDJPY macro drift: passive shorts had a +170.7p tailwind. That pushes the investigation toward gate conditioning and LLM reasoning, not external regime excuses.", color=RED_DARK, size=11, weight="bold", width=100)
    add_bullets(fig, 0.075, 0.32, ["CLR is 149 of 241 closed trades and -183.7p.", "Sell-CLR: 43 trades, 37.2% WR, -184.3p / -$2,081.56.", "Buy-CLR: 106 trades, 59.4% WR, roughly flat in pips.", "17 CLR fast failures later become a critical entry-failure proof set."], width=70, gap=0.043)
    add_note(fig, 0.58, 0.35, 0.30, "Gate verdicts: CLR REDESIGN. Momentum KEEP-CONDITIONAL with sizing redesign. Mean Reversion KEEP-CONDITIONAL. Non-primary gates KILL/EXPERIMENT ONLY.", color=BLUE, size=10.5, weight="bold", width=42)
    pdf.savefig(fig)
    plt.close(fig)


def phase3(pdf):
    info = load("phase3_field_information_content.csv")
    fast = load("phase3_clr_level_failure_17.csv")
    fig = fig_base("Phase 3 - Snapshot Quality Audit", "Was the LLM fed bad data, or did it misread usable data?", "Phase 3")
    add_note(fig, 0.075, 0.79, 0.82, "Executive finding: the snapshot was not the primary cause of sell-CLR collapse. Core raw market fields were symmetric across buy/sell. The real defect was that raw support/resistance was not converted into side-normalized decision packets with level quality and portfolio risk.", color=RED_DARK, size=12, weight="bold", width=100)
    t = info.head(8)[["field", "coverage_closed_snapshot_pct", "separation_score", "best_bucket_avg_pips", "worst_bucket_avg_pips"]].copy()
    t["field"] = t["field"].map(lambda x: short(x, 46))
    t["coverage_closed_snapshot_pct"] = t["coverage_closed_snapshot_pct"].map(pct)
    t["separation_score"] = t["separation_score"].map(lambda v: f"{float(v):.1f}")
    t["best_bucket_avg_pips"] = t["best_bucket_avg_pips"].map(pips)
    t["worst_bucket_avg_pips"] = t["worst_bucket_avg_pips"].map(pips)
    t = t.rename(columns={"coverage_closed_snapshot_pct": "coverage", "separation_score": "score", "best_bucket_avg_pips": "best", "worst_bucket_avg_pips": "worst"})
    add_table(fig, t, 0.055, 0.43, 0.89, 0.26, 6.8)
    add_bullets(fig, 0.075, 0.30, ["Keep: live price, spread, session, volatility, H1/M5/M1 technicals, support/resistance raw data, account primitives.", "Kill/reduce as cognition load: duplicate raw fields with no normalized meaning.", "Add: side-normalized level packet, open exposure, rolling P&L, risk-after-fill, path-time MAE/MFE, skip outcomes.", "Reformat: support/resistance into entry wall vs profit-path blocker."], width=85, gap=0.042)
    pdf.savefig(fig)
    plt.close(fig)

    fig = fig_base("Phase 3 - CLR Snapshot Forensic", "The 17 fast failures and side symmetry", "Phase 3")
    add_card(fig, 0.075, 0.69, 0.19, 0.11, "CLR fast failures", "17", RED, "top MAE + <=2p MFE")
    add_card(fig, 0.285, 0.69, 0.19, 0.11, "Weak/thin evidence", "8", GOLD)
    add_card(fig, 0.495, 0.69, 0.19, 0.11, "Mixed/ambiguous", "8", GOLD)
    add_card(fig, 0.705, 0.69, 0.19, 0.11, "Called strong", "1", RED)
    add_note(fig, 0.075, 0.55, 0.82, "Interpretation: when the LLM called many losing levels 'strong', that strength usually did not come from robust persisted level-quality data. It was a reasoning interpretation layered on top of weak/mixed raw evidence.", color=RED_DARK, size=11, weight="bold", width=100)
    add_bullets(fig, 0.075, 0.38, ["Buy/sell snapshot coverage was symmetric enough to refute 'sell snapshots were thinner'.", "Silent schema drift exists: later feature fields are output/schema artifacts, not stable market inputs.", "Momentum snapshots lacked open exposure and side-specific portfolio state, so sizing could not be fully rational.", "Skipped-call snapshots are bounded evidence only; skip forward quality is still unknown."], width=100, gap=0.048)
    pdf.savefig(fig)
    plt.close(fig)


def phase4(pdf):
    clusters = load("phase4_rationale_clusters.csv")
    cog = load("phase4_cognitive_failure_tests.csv")
    fig = fig_base("Phase 4 - LLM Reasoning Forensics", "The core of the investigation", "Phase 4")
    add_note(fig, 0.075, 0.79, 0.84, "Bottom line: given largely symmetric raw inputs, the LLM's reasoning collapses on shorts. It uses caveat/level language as permission to trade instead of as a brake. This is the primary cognitive root cause.", color=RED_DARK, size=12.5, weight="bold", width=100)
    t = clusters.head(8)[["rationale_cluster", "calls", "placed", "closed", "win_rate", "net_pips", "net_usd", "expectancy_usd"]].copy()
    t["rationale_cluster"] = t["rationale_cluster"].map(lambda x: short(x, 34))
    t["win_rate"] = t["win_rate"].map(pct)
    t["net_pips"] = t["net_pips"].map(pips)
    t["net_usd"] = t["net_usd"].map(money)
    t["expectancy_usd"] = t["expectancy_usd"].map(money2)
    t = t.rename(columns={"rationale_cluster": "cluster", "win_rate": "WR", "net_pips": "pips", "net_usd": "USD", "expectancy_usd": "exp USD"})
    add_table(fig, t, 0.055, 0.40, 0.89, 0.28, 7.0)
    add_note(fig, 0.075, 0.28, 0.84, "Toxic archetypes: momentum_with_caveat_trade and critical_level_mixed_caveat_trade dominate placed losses. Caveats are not just descriptive; in practice they became admission tickets.", color=RED_DARK, size=10.5, weight="bold", width=100)
    pdf.savefig(fig)
    plt.close(fig)

    fig = fig_base("Phase 4 - Cognitive Failure Tests", "What the model's language revealed", "Phase 4")
    rows = [
        ("Sell-CLR hypothesis", "H5 reasoning template collapse is primary.", "High"),
        ("Confirmation bias", "Ambiguous/contradictory reasoning still placed and lost.", "High"),
        ("Hedging as tell", "Hedge language exists, but not a clean standalone predictor.", "Medium"),
        ("Overconfidence", "Strong level language often outran snapshot evidence.", "High"),
        ("Hallucination audit", "40-row audit found unsupported claims worth instrumenting.", "Medium"),
        ("Prompt audit", "Runner and house-edge clauses changed behavior but did not solve selectivity.", "High"),
    ]
    add_table(fig, pd.DataFrame(rows, columns=["test", "finding", "confidence"]), 0.075, 0.48, 0.82, 0.27, 8.5)
    add_bullets(fig, 0.075, 0.34, ["Predictive/preserve: concrete side-specific evidence and buy-CLR zone-memory context.", "Toxic/eliminate: caveat trades on sells, structure-only catalysts, strong level claims without level packet proof.", "Prompt clauses to remove: runner preservation, red-pattern naming without terminal resolution, structure-only catalyst acceptance.", "Reasoning constraint: if it says 'mixed', 'but', 'however', or 'although', the caveat must resolve materially or the trade must skip."], width=103, gap=0.046)
    pdf.savefig(fig)
    plt.close(fig)


def phase5(pdf):
    decomp = load("phase5_pl_decomposition.csv")
    policies = load("phase5_counterfactual_policies.csv")
    corr = load("phase5_edge_size_correlation.csv")
    fig = fig_base("Phase 5 - Sizing Logic Audit", "Was sizing smart, random, or anti-Kelly?", "Phase 5")
    add_note(fig, 0.075, 0.79, 0.84, "Committed sizing verdict: random/edge-blind. The LLM's chosen lots do not show a defensible positive relationship to realized edge. This is not stable anti-Kelly; it is untrusted sizing authority.", color=RED_DARK, size=12, weight="bold", width=100)
    cards = [("Actual P&L", "-$7,253.24", RED), ("1-lot admission", "-$2,065.99", GOLD), ("Sizing delta", "-$5,187.24", RED), ("Overlap", "$0.00", TEAL)]
    x = 0.075
    for label, value, color in cards:
        add_card(fig, x, 0.62, 0.19, 0.11, label, value, color)
        x += 0.21
    t = policies[["policy", "counterfactual_usd", "delta_vs_actual_usd", "trades_altered", "mean_policy_lots"]].head(6).copy()
    t["policy"] = t["policy"].map(lambda x: short(x, 38))
    t["counterfactual_usd"] = t["counterfactual_usd"].map(money)
    t["delta_vs_actual_usd"] = t["delta_vs_actual_usd"].map(money)
    t["mean_policy_lots"] = t["mean_policy_lots"].map(lambda v: f"{float(v):.2f}")
    t = t.rename(columns={"counterfactual_usd": "P&L", "delta_vs_actual_usd": "delta", "mean_policy_lots": "mean lots"})
    add_table(fig, t, 0.055, 0.30, 0.89, 0.25, 7.2)
    add_note(fig, 0.075, 0.17, 0.84, "Phase 5 handoff: remove LLM sizing authority. Approximate volatility-scaled or fixed-fractional deterministic sizing, clipped and exposure-aware.", color=BLUE, size=11, weight="bold", width=100)
    pdf.savefig(fig)
    plt.close(fig)

    fig = fig_base("Phase 5 - Sizing Mechanics And Telemetry", "What correct sizing would need", "Phase 5")
    c = corr[
        (corr["scope_type"].eq("overall"))
        & (corr["target"].isin(["realized_pips", "win_flag", "realized_usd"]))
    ].head(6)[["target", "method", "n", "correlation", "ci_low", "ci_high", "verdict_hint"]].copy()
    for col in ["correlation", "ci_low", "ci_high"]:
        c[col] = c[col].map(lambda v: f"{float(v):.3f}")
    add_table(fig, c, 0.075, 0.58, 0.58, 0.20, 8.0)
    add_bullets(fig, 0.69, 0.76, ["Open USDJPY lots by side.", "Unrealized P&L by side.", "Rolling 20-trade P&L and lot-weighted P&L.", "Pip value per lot.", "Risk-after-fill in dollars.", "Session-volatility reference size."], width=38, gap=0.042)
    add_note(fig, 0.075, 0.41, 0.84, "8+ lot CLR anomaly verdict: compositional artifact, buy-CLR dominated, near-flat. It is not permission to size up generally.", color=RED_DARK, size=11, weight="bold", width=100)
    add_note(fig, 0.075, 0.27, 0.84, "Best bounded lesson: uniform 1 lot and volatility-scaled 1/ratio both recover roughly $5.2k in-sample. That does not prove forward edge, but it proves that variable LLM lots were a major amplifier.", color=INK, size=10, width=100)
    pdf.savefig(fig)
    plt.close(fig)


def phase6(pdf):
    attr = load("phase6_entry_exit_attribution.csv")
    runner = load("phase6_runner_regime_test.csv")
    capture = load("phase6_capture_efficiency.csv")
    fig = fig_base("Phase 6 - Trade Lifecycle", "Are entries wrong, exits wrong, or both?", "Phase 6")
    add_note(fig, 0.075, 0.79, 0.84, "Committed lifecycle verdict: entry-generated damage dominates. Exit leakage exists, but the largest negative leg is trades that never produced meaningful favorable excursion.", color=RED_DARK, size=12, weight="bold", width=100)
    t = attr[["bucket", "n", "pips"]].copy()
    t["bucket"] = t["bucket"].map(lambda x: short(x.replace("_", " "), 46))
    t["pips"] = t["pips"].map(pips)
    add_table(fig, t, 0.075, 0.53, 0.55, 0.22, 8.2)
    add_card(fig, 0.68, 0.66, 0.19, 0.10, "Entry proxy", "-305.1p", RED, "25 losses")
    add_card(fig, 0.68, 0.53, 0.19, 0.10, "Exit reversal", "-218.7p", GOLD, "24 losses")
    add_barh(fig, attr.head(4)["bucket"].map(lambda x: x.replace("_", " ")), attr.head(4)["pips"], 0.075, 0.16, 0.82, 0.25, "Pip attribution buckets")
    pdf.savefig(fig)
    plt.close(fig)

    fig = fig_base("Phase 6 - Runner And Exit Findings", "What changed behavior and what did not", "Phase 6")
    r = runner[["prompt_regime", "n", "win_rate", "net_pips", "hold_median_min", "winner_capture_median", "runner_verdict"]].copy()
    r["prompt_regime"] = r["prompt_regime"].map(lambda x: short(x, 30))
    r["win_rate"] = r["win_rate"].map(pct)
    r["net_pips"] = r["net_pips"].map(pips)
    r["hold_median_min"] = r["hold_median_min"].map(lambda v: f"{float(v):.1f}m")
    r["winner_capture_median"] = r["winner_capture_median"].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}x")
    r = r.rename(columns={"win_rate": "WR", "net_pips": "pips", "hold_median_min": "hold", "winner_capture_median": "win capture", "runner_verdict": "verdict"})
    add_table(fig, r, 0.055, 0.50, 0.89, 0.26, 7.2)
    add_bullets(fig, 0.075, 0.33, ["Runner/custom-exit v3 verdict: measurable harm.", "Caveat-laundered exits verdict: no; caveat laundering is entry/sizing, not exit hold-time.", "Winner capture: median 0.69x MFE; median 2.45p left behind.", "Path-time MAE/MFE is missing, so exit counterfactuals are upper bounds."], width=110, gap=0.045)
    pdf.savefig(fig)
    plt.close(fig)


def phase7(pdf):
    veto = load("phase7_minimum_veto_rules.csv")
    preserved = load("phase7_preserved_edge_cells.csv")
    fig = fig_base("Phase 7 - Interaction Effects", "Where layers combined into damage or preserved edge", "Phase 7")
    add_note(fig, 0.075, 0.79, 0.84, "Damage is concentrated enough to build a diagnostic floor, but exact 10-dimensional cells are too sparse. The useful signal is in lower-cardinality combinations: sell caveat clusters, mixed overlap entry signatures, and narrow buy-CLR preserved cells.", color=RED_DARK, size=12, weight="bold", width=100)
    ind = veto[veto["mode"].eq("individual")].head(7)[["rule_id", "blocked_trades", "blocked_winners", "net_delta_pips", "net_delta_usd", "coverage_abs_net_pips"]].copy()
    ind["rule_id"] = ind["rule_id"].map(lambda x: short(x, 35))
    ind["net_delta_pips"] = ind["net_delta_pips"].map(pips)
    ind["net_delta_usd"] = ind["net_delta_usd"].map(money)
    ind["coverage_abs_net_pips"] = ind["coverage_abs_net_pips"].map(pct)
    ind = ind.rename(columns={"rule_id": "rule", "blocked_trades": "blocked", "blocked_winners": "winners", "net_delta_pips": "pips", "net_delta_usd": "USD", "coverage_abs_net_pips": "pip cover"})
    add_table(fig, ind, 0.055, 0.40, 0.89, 0.30, 7.0)
    add_note(fig, 0.075, 0.27, 0.84, "Minimum floor: V1 alone recovers 90.4% of pip damage. V1+V2 recovers +300.5p and +$5,684.56 but blocks 52 winners. Phase 9 must beat this trade-off, not just copy it.", color=BLUE, size=11, weight="bold", width=100)
    pdf.savefig(fig)
    plt.close(fig)

    fig = fig_base("Phase 7 - Preserved Edge", "What the overhaul must not destroy", "Phase 7")
    p = preserved[preserved["survives_leave_one_date"].eq(True)].head(3)[["cell_signature", "n", "win_rate", "net_pips", "net_usd", "loo_min_net_usd"]].copy()
    p["cell_signature"] = p["cell_signature"].map(lambda x: short(x, 62))
    p["win_rate"] = p["win_rate"].map(pct)
    p["net_pips"] = p["net_pips"].map(pips)
    p["net_usd"] = p["net_usd"].map(money)
    p["loo_min_net_usd"] = p["loo_min_net_usd"].map(money)
    p = p.rename(columns={"cell_signature": "cell", "win_rate": "WR", "net_pips": "pips", "net_usd": "USD", "loo_min_net_usd": "LOO min"})
    add_table(fig, p, 0.055, 0.56, 0.89, 0.22, 7.1)
    add_bullets(fig, 0.075, 0.39, ["Preserved edge exists, but it is narrow.", "The robust pocket is buy-side CLR, especially Phase 2 zone-memory caveat-trade with 2-3.99 lots.", "Tuesday alone is not a generic edge; Tuesday buy-CLR is the preserved slice.", "8+ lot CLR is not preserved edge."], width=100, gap=0.046)
    add_note(fig, 0.075, 0.17, 0.84, "Phase 7 handoff: redesign aggressively against sell caveat templates and mixed-overlap entry failure, but explicitly protect buy-side CLR with valid level/zone-memory context.", color=TEAL, size=11, weight="bold", width=100)
    pdf.savefig(fig)
    plt.close(fig)


def phase8(pdf):
    fig = fig_base("Phase 8 - Root Cause Synthesis", "The final diagnostic hierarchy", "Phase 8")
    rows = [
        ("1", "Sell-side caveat-template collapse", "+278.4p / +$4,397.69 recoverable", "Primary cognitive failure"),
        ("2", "Random/edge-blind sizing amplification", "-$5,187.24", "Primary dollar failure"),
        ("3", "Admission layer negative pip expectancy", "-308.0p / -$2,065.99 at 1 lot", "Primary pip failure"),
        ("4", "Entry-generated level failure", "-305.1p across 25 trades", "Primary lifecycle source"),
        ("5", "Exit-reversal leakage", "-218.7p across 24 trades", "Secondary"),
        ("6", "Runner/custom-exit v3 harm", "-92.2p / -$1,777.09", "Secondary"),
    ]
    add_table(fig, pd.DataFrame(rows, columns=["rank", "root cause", "impact", "tier"]), 0.055, 0.42, 0.89, 0.34, 8.2)
    add_note(fig, 0.075, 0.29, 0.84, "Phase 8's synthesis: the system admits negative-expectancy trades, the LLM's short-side reasoning is the primary cognitive failure, and random sizing turns a pip problem into a dollar problem.", color=RED_DARK, size=12, weight="bold", width=100)
    pdf.savefig(fig)
    plt.close(fig)

    fig = fig_base("Phase 8 - Constraints For The Redesign", "What Phase 9 must obey", "Phase 8")
    add_card(fig, 0.075, 0.68, 0.24, 0.11, "Pip floor", "+278.4p", TEAL, "V1 alone")
    add_card(fig, 0.36, 0.68, 0.24, 0.11, "USD floor", "+$5,684.56", TEAL, "V1+V2")
    add_card(fig, 0.645, 0.68, 0.24, 0.11, "False positive cost", "52 winners", GOLD, "must beat/match")
    add_bullets(fig, 0.075, 0.53, ["Refuted: macro drift caused sells, thinner sell snapshots, anti-Kelly sizing, 8+ lot CLR edge, caveat-laundered exits, entry-failure oversizing.", "Ambiguous: path ordering, skip quality, exit-manager replay, open exposure effects, Phase 3 prompt vs volatility, full rendered context.", "Required telemetry: exposure by side, risk-after-fill, rolling P&L, unrealized P&L, pip value, side-normalized level packet, path-time MAE/MFE, skip outcomes, snapshot version, all gate candidates, exit replay."], width=106, gap=0.068)
    pdf.savefig(fig)
    plt.close(fig)


def phase9(pdf):
    replay = load("phase9_replay_summary.csv")
    fig = fig_base("Phase 9 - Overhaul Blueprint", "A new Auto Fillmore, not a patch pile", "Phase 9")
    add_note(fig, 0.075, 0.79, 0.84, "Design philosophy: deterministic where possible, model where necessary. The LLM becomes an evidence adjudicator inside a deterministic shell. Sizing and exits are no longer model-authorized.", color=BLUE, size=12, weight="bold", width=100)
    layers = [
        ("Telemetry", "versioned snapshot + full prompt"),
        ("Gate", "side-asymmetric eligibility"),
        ("Pre-veto", "block toxic rows before LLM"),
        ("LLM", "place/skip evidence JSON only"),
        ("Validator", "override bad reasoning"),
        ("Sizing", "fixed risk, clipped 1-4 lots"),
        ("Exit", "deterministic stop/profit/time rules"),
        ("Feedback", "replay every decision"),
    ]
    for idx, (name, detail) in enumerate(layers):
        x = 0.075 + (idx % 4) * 0.215
        y = 0.55 - (idx // 4) * 0.18
        fig.add_artist(plt.Rectangle((x, y), 0.18, 0.12, transform=fig.transFigure, facecolor=PANEL, edgecolor=GRID, linewidth=0.8))
        fig.text(x + 0.012, y + 0.078, name, fontsize=10, color=RED_DARK, fontweight="bold")
        fig.text(x + 0.012, y + 0.045, safe(wrap(detail, 24)), fontsize=7.8, color=INK, va="top")
    pdf.savefig(fig)
    plt.close(fig)

    fig = fig_base("Phase 9 - Replay Gate", "The blueprint passes the diagnostic floor", "Phase 9")
    add_card(fig, 0.075, 0.68, 0.19, 0.11, "Pip recovery", "+324.3p", TEAL, "floor +278.4p")
    add_card(fig, 0.285, 0.68, 0.19, 0.11, "Admission USD", "+$6,086.60", TEAL, "floor +$5,684.56")
    add_card(fig, 0.495, 0.68, 0.19, 0.11, "Cap-4 USD", "+$6,420.90", TEAL)
    add_card(fig, 0.705, 0.68, 0.19, 0.11, "Winners blocked", "45", GOLD, "floor 52")
    add_bullets(fig, 0.075, 0.51, ["Refined V1: sell caveat-template veto.", "Refined V2: mixed-overlap entry signature with protected-edge bypass.", "V5: hedge plus overconfidence validator.", "V6: sell-side level-language overreach validator.", "All three preserved buy-CLR cells survive in replay."], width=105, gap=0.046)
    add_note(fig, 0.075, 0.22, 0.84, "This does not prove forward edge. It proves the redesign is at least more defensible than the diagnostic floor and gives forward kill criteria so we can stop quickly if live data disagrees.", color=RED_DARK, size=11, weight="bold", width=100)
    pdf.savefig(fig)
    plt.close(fig)

    fig = fig_base("Phase 9 - What Ships In The New System", "Blueprint details that matter for implementation", "Phase 9")
    rows = [
        ("Gate", "CLR split buy/sell; momentum conditional; mean reversion small-N; non-primary killed."),
        ("LLM", "No sizing, no exits, JSON only, caveats terminal unless resolved."),
        ("Validator", "Caveat resolution, level overreach, loss asymmetry, sell burden, hedge+overconfidence."),
        ("Sizing", "Fixed-fractional 0.25%-0.5%, clipped 1-4, exposure/drawdown/vol modifiers."),
        ("Exit", "No stop widening, 1R profit lock, 30m backup time stop, no trailing until path telemetry."),
        ("Telemetry", "Exposure, risk-after-fill, pip value, level packet, path buckets, skip outcomes, prompt context."),
        ("Rollout", "paper -> 0.1x -> 0.5x -> full; audit after 200 trades or 4 weeks."),
        ("Killed", "LLM sizing, LLM exit extension, runner clauses, structure-only catalysts, day heuristics."),
    ]
    add_table(fig, pd.DataFrame(rows, columns=["layer", "spec"]), 0.055, 0.30, 0.89, 0.46, 8.0)
    add_note(fig, 0.075, 0.17, 0.84, "Forward kill examples: sell-side WR below 45% after 30 sells, any order above 4 lots, missing blocking telemetry, stop widening without replay, or cumulative paper/0.1x drawdown beyond thresholds.", color=RED_DARK, size=10.5, weight="bold", width=100)
    pdf.savefig(fig)
    plt.close(fig)


def appendix(pdf):
    fig = fig_base("Artifact Map", "Where the details live, without making the main read miserable", "Appendix")
    phases = [
        ("Phase 1", "PHASE1_BASELINE.md", "phase1_by_*.csv, phase1_closed_autonomous_trades.csv"),
        ("Phase 2", "PHASE2_GATE_AUDIT.md", "phase2_gate_*.csv, skip behavior, regime tables"),
        ("Phase 3", "PHASE3_SNAPSHOT_AUDIT.md", "phase3_schema_*.csv, field info, CLR snapshot rows"),
        ("Phase 4", "PHASE4_REASONING_FORENSICS.md", "phase4_reasoning_corpus.csv, clusters, prompt audit"),
        ("Phase 5", "PHASE5_SIZING_AUDIT.md", "phase5_pl_decomposition.csv, policies, correlations"),
        ("Phase 6", "PHASE6_LIFECYCLE_ANALYSIS.md", "phase6_capture, runner, entry_exit, event, path"),
        ("Phase 7", "PHASE7_INTERACTION_EFFECTS.md", "phase7_veto, preserved_edge, damage concentration"),
        ("Phase 8", "PHASE8_SYNTHESIS.md", "causal hierarchy, floor, ambiguity, handoff"),
        ("Phase 9", "PHASE9_OVERHAUL_BLUEPRINT.md", "architecture, gates, validators, replay, rollout"),
    ]
    add_table(fig, pd.DataFrame(phases, columns=["phase", "main report", "detail artifacts"]), 0.055, 0.28, 0.89, 0.50, 7.7)
    add_note(fig, 0.075, 0.16, 0.84, "The original full book is still useful as a source archive. This reader edition is meant for decisions: each phase is represented, every locked verdict is preserved, and the implementation constraints are explicit.", color=BLUE, size=11, weight="bold", width=100)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    global PAGE_NO
    PAGE_NO = 0
    with PdfPages(OUT_PDF) as pdf:
        cover(pdf)
        investigation_map(pdf)
        whole_story(pdf)
        phase1(pdf)
        phase2(pdf)
        phase3(pdf)
        phase4(pdf)
        phase5(pdf)
        phase6(pdf)
        phase7(pdf)
        phase8(pdf)
        phase9(pdf)
        appendix(pdf)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
