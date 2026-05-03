#!/usr/bin/env python3
"""Render a scan-first PDF brief for Auto Fillmore.

This is intentionally different from the archival phase book.  The goal is
readability: portrait pages, large type, strong section hierarchy, and enough
evidence to understand the Phase 1-9 audit and the v2 rebuild without reading
every source artifact.
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
FORENSIC = ROOT / "research_out" / "autonomous_fillmore_forensic_20260501"
DOCS = ROOT / "docs" / "fillmore_v2"
OUT = FORENSIC / "Auto_Fillmore_Visual_Brief_v2.pdf"

PAGE_W = 8.27
PAGE_H = 11.69

BG = "#F7F3EA"
PAPER = "#FFFCF5"
INK = "#172233"
MUTED = "#65717F"
LINE = "#D8CFC0"
RED = "#A84037"
RED_SOFT = "#F2DED8"
TEAL = "#176B6D"
TEAL_SOFT = "#DDEDEA"
BLUE = "#305F8D"
BLUE_SOFT = "#E1EAF3"
GOLD = "#B7852F"
GOLD_SOFT = "#F3E8CF"
PANEL = "#FFFFFF"
CHARCOAL = "#2D3748"

PAGE_NO = 0


def safe(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).replace("$", r"\$")


def wrap(text: object, width: int = 62) -> str:
    return "\n".join(
        textwrap.wrap(
            str(text),
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def money(value: object, digits: int = 0) -> str:
    try:
        v = float(value)
    except Exception:
        return safe(value)
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v):,.{digits}f}"


def pips(value: object) -> str:
    try:
        v = float(value)
    except Exception:
        return safe(value)
    return f"{v:+.1f}p"


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(FORENSIC / name)


def page(title: str, eyebrow: str = "", section: str = ""):
    global PAGE_NO
    PAGE_NO += 1
    fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor=BG)
    fig.add_artist(
        plt.Rectangle(
            (0.045, 0.035),
            0.91,
            0.93,
            transform=fig.transFigure,
            facecolor=PAPER,
            edgecolor=LINE,
            linewidth=0.9,
        )
    )
    if eyebrow:
        fig.text(0.09, 0.925, safe(eyebrow.upper()), fontsize=8.5, color=RED, fontweight="bold")
    fig.text(0.09, 0.885, safe(title), fontsize=22, color=INK, fontweight="bold")
    if section:
        fig.text(0.91, 0.925, safe(section), fontsize=8.5, color=MUTED, ha="right", fontweight="bold")
    fig.add_artist(plt.Line2D([0.09, 0.91], [0.852, 0.852], color=LINE, linewidth=0.9))
    fig.text(0.91, 0.052, f"{PAGE_NO}", fontsize=8, color=MUTED, ha="right")
    return fig


def text(fig, x: float, y: float, body: str, width: int = 62, size: float = 10.2, color=INK, weight="normal"):
    fig.text(
        x,
        y,
        safe(wrap(body, width)),
        fontsize=size,
        color=color,
        fontweight=weight,
        va="top",
        linespacing=1.28,
    )


def label(fig, x: float, y: float, body: str, color=RED):
    fig.text(x, y, safe(body.upper()), fontsize=8.2, color=color, fontweight="bold", va="top")


def card(
    fig,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    value: str,
    note: str = "",
    color=INK,
    fill=PANEL,
):
    fig.add_artist(
        plt.Rectangle((x, y), w, h, transform=fig.transFigure, facecolor=fill, edgecolor=LINE, linewidth=0.8)
    )
    fig.text(x + 0.018, y + h - 0.030, safe(title.upper()), fontsize=7.2, color=MUTED, fontweight="bold")
    fs = 18 if len(value) <= 18 else 13.4
    fig.text(x + 0.018, y + h * 0.45, safe(value), fontsize=fs, color=color, fontweight="bold", va="center")
    if note:
        fig.text(x + 0.018, y + 0.024, safe(wrap(note, 26)), fontsize=7.6, color=MUTED, va="bottom", linespacing=1.18)


def bullets(
    fig,
    x: float,
    y: float,
    items: Iterable[str],
    width: int = 48,
    size: float = 9.1,
    color=INK,
    gap: float = 0.054,
):
    yy = y
    for item in items:
        wrapped = wrap(item, width).replace("\n", "\n   ")
        fig.text(x, yy, safe("• " + wrapped), fontsize=size, color=color, va="top", linespacing=1.22)
        yy -= gap + 0.014 * wrapped.count("\n")
    return yy


def mini_table(fig, rows: list[list[str]], headers: list[str], x, y, w, h, col_widths=None, font_size=7.6):
    ax = fig.add_axes([x, y, w, h])
    ax.axis("off")
    tbl = ax.table(
        cellText=[[safe(c) for c in row] for row in rows],
        colLabels=[safe(hh) for hh in headers],
        cellLoc="left",
        loc="center",
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size)
    tbl.scale(1, 1.28)
    for (r, _c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#E6DCCE")
        if r == 0:
            cell.set_facecolor("#EFE5D7")
            cell.set_text_props(weight="bold", color=INK)
        else:
            cell.set_facecolor(PANEL if r % 2 else "#FBF6EE")
            cell.set_text_props(color=INK)
    return ax


def pill(fig, x: float, y: float, text_value: str, fill: str, color=INK):
    fig.add_artist(
        plt.Rectangle((x, y), 0.19, 0.034, transform=fig.transFigure, facecolor=fill, edgecolor=LINE, linewidth=0.55)
    )
    fig.text(x + 0.095, y + 0.017, safe(text_value), fontsize=7.8, color=color, ha="center", va="center", fontweight="bold")


def flow(fig, x: float, y: float, labels: list[str], fills: list[str]):
    w = 0.155
    h = 0.067
    gap = 0.015
    for i, item in enumerate(labels):
        xx = x + i * (w + gap)
        fig.add_artist(plt.Rectangle((xx, y), w, h, transform=fig.transFigure, facecolor=fills[i], edgecolor=LINE, linewidth=0.75))
        fig.text(xx + w / 2, y + h / 2, safe(wrap(item, 15)), fontsize=7.4, color=INK, ha="center", va="center", fontweight="bold")
        if i < len(labels) - 1:
            fig.text(xx + w + gap / 2, y + h / 2, ">", fontsize=12, color=MUTED, ha="center", va="center")


def bar_page(fig, labels, values, x, y, w, h, title: str, fmt=pips):
    ax = fig.add_axes([x, y, w, h], facecolor=PAPER)
    vals = [float(v) for v in values]
    colors = [RED if v < 0 else TEAL for v in vals]
    ax.barh(range(len(vals)), vals, color=colors, alpha=0.92)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels([str(s)[:28] for s in labels], fontsize=7.8, color=INK)
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, fontweight="bold")
    ax.grid(axis="x", color=LINE, alpha=0.65)
    ax.tick_params(axis="x", labelsize=7.6, colors=MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, val in enumerate(vals):
        ax.text(val, i, " " + fmt(val), va="center", fontsize=7.3, color=INK)


def cover(pdf):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor=BG)
    fig.add_artist(plt.Rectangle((0.045, 0.035), 0.91, 0.93, transform=fig.transFigure, facecolor=PAPER, edgecolor=LINE))
    fig.add_artist(plt.Rectangle((0.045, 0.035), 0.91, 0.27, transform=fig.transFigure, facecolor=INK, edgecolor=INK))
    fig.text(0.09, 0.84, "AUTO FILLMORE", fontsize=33, color=INK, fontweight="bold")
    fig.text(0.09, 0.785, "Visual Brief", fontsize=26, color=RED, fontweight="bold")
    text(
        fig,
        0.09,
        0.72,
        "A cleaner, scan-first explanation of the Phase 1-9 forensic audit, the v2 rebuild, and how Fillmore works for a new reader.",
        width=54,
        size=13,
        color=CHARCOAL,
        weight="bold",
    )
    card(fig, 0.09, 0.56, 0.245, 0.105, "Audit sample", "241 trades", "Apr 16-May 1")
    card(fig, 0.375, 0.56, 0.245, 0.105, "v1 result", "-$7,253", "-308.0p", RED)
    card(fig, 0.66, 0.56, 0.245, 0.105, "v2 result", "paper-ready", "not 0.1x yet", TEAL)
    fig.text(0.09, 0.245, "Designed to be read, not decoded", fontsize=22, color=PAPER, fontweight="bold")
    fig.text(
        0.09,
        0.195,
        safe(wrap("This version favors one idea per page, large labels, short evidence blocks, and clear distinctions between diagnosis, rebuild, and launch readiness.", 78)),
        fontsize=11,
        color="#E8EDF3",
        va="top",
        linespacing=1.25,
    )
    fig.text(0.09, 0.075, "Generated from local Phase 1-9 artifacts and v2 rebuild docs", fontsize=8.5, color="#C7D0DC")
    pdf.savefig(fig)
    plt.close(fig)


def how_to_read(pdf):
    fig = page("How To Read This", "orientation", "Reader guide")
    text(
        fig,
        0.09,
        0.80,
        "The earlier PDF tried to preserve too much source material. This one is a decision brief: it keeps the full story, but compresses it into the parts a human can scan.",
        width=60,
        size=12,
        weight="bold",
    )
    rows = [
        ["Pages 3-4", "What Fillmore is", "A plain-English mental model."],
        ["Pages 5-8", "What the investigation found", "Root causes, evidence, and what was ruled out."],
        ["Pages 9-13", "What v2 changed", "Architecture, rebuild steps, replay results, and readiness."],
        ["Pages 14-16", "Operator view", "What the UI should show, what is still blocked, and what to watch."],
    ]
    mini_table(fig, rows, ["Where", "Section", "Purpose"], 0.09, 0.47, 0.82, 0.24, [0.18, 0.28, 0.54], 8.6)
    label(fig, 0.09, 0.36, "Reading rule")
    bullets(
        fig,
        0.09,
        0.32,
        [
            "Red means damage or a blocked path.",
            "Teal means preserved edge, protection, or passed verification.",
            "Gold means operational caution: safe for Stage 1 paper, not cleared for scaled testing.",
        ],
        width=62,
        size=10,
    )
    pdf.savefig(fig)
    plt.close(fig)


def fillmore_plain_english(pdf):
    fig = page("What Fillmore Is", "plain-English overview", "System")
    text(
        fig,
        0.09,
        0.80,
        "Fillmore is an autonomous USDJPY trading agent. It watches live market data, detects a setup, sends a market snapshot to an LLM, and decides whether to place or skip.",
        width=62,
        size=12.2,
        weight="bold",
    )
    flow(
        fig,
        0.09,
        0.66,
        ["Market tick", "Code gate", "Snapshot", "LLM judgment", "Validation", "Audit row"],
        [GOLD_SOFT, GOLD_SOFT, PANEL, PANEL, TEAL_SOFT, TEAL_SOFT],
    )
    rows = [
        ["Gate", "A deterministic setup detector: critical level reaction, momentum continuation, or mean reversion."],
        ["Snapshot", "The packet of market, account, level, volatility, and regime fields shown to the model."],
        ["LLM", "The model that reads the evidence and argues place vs skip."],
        ["Validator", "A deterministic layer that can override the LLM if the argument is weak or contradictory."],
        ["Stage 1", "Live ticks and audit rows, but paper-guarded. v2 does not send broker orders."],
    ]
    mini_table(fig, rows, ["Term", "Meaning"], 0.09, 0.25, 0.82, 0.34, [0.20, 0.80], 8.4)
    text(
        fig,
        0.09,
        0.155,
        "The rebuild's core idea: keep the model useful, but remove its authority over sizing, exits, and unresolved caveats.",
        width=60,
        size=11.5,
        color=BLUE,
        weight="bold",
    )
    pdf.savefig(fig)
    plt.close(fig)


def v1_vs_v2(pdf):
    fig = page("v1 vs v2 In One Page", "the rebuild changed authority", "Comparison")
    rows = [
        ["Trade admission", "LLM could talk itself into marginal setups.", "Pre-vetoes block toxic setup cells before the LLM."],
        ["Reasoning", "Caveats often became decoration.", "Caveats must be resolved with evidence or the trade skips."],
        ["Sizing", "LLM-influenced and edge-blind.", "Deterministic risk function, hard capped."],
        ["Exits", "Prompt language could bias hold behavior.", "Stops, locks, and time rules are deterministic."],
        ["Telemetry", "Enough to audit placed trades, not enough to replay everything.", "Snapshot, prompt, gates, sizing, vetoes, validators, and state are persisted."],
        ["Launch state", "Original autonomous engine.", "Parallel v2 engine, default off, Stage 1 paper-guarded."],
    ]
    mini_table(fig, rows, ["Layer", "v1", "v2"], 0.07, 0.255, 0.86, 0.55, [0.19, 0.39, 0.42], 7.8)
    text(
        fig,
        0.09,
        0.15,
        "The rebuild is not just a better prompt. It is a new control system around the prompt.",
        width=62,
        size=12,
        color=TEAL,
        weight="bold",
    )
    pdf.savefig(fig)
    plt.close(fig)


def phase_map(pdf):
    fig = page("Phase 1-9 Audit Map", "what every phase contributed", "Forensic audit")
    phases = [
        ["1", "Baseline", "241 trades; 53.1% WR still lost -308.0p / -$7,253."],
        ["2", "Gate audit", "Sell-CLR and large-size damage isolated."],
        ["3", "Snapshot", "Core buy/sell data symmetric; key telemetry missing."],
        ["4", "Reasoning", "Sell-side caveat-template collapse confirmed."],
        ["5", "Sizing", "Sizing was random/edge-blind; -$5,187 amplification."],
        ["6", "Lifecycle", "Entry-generated damage dominated; runner language harmed."],
        ["7", "Interactions", "V1+V2 veto floor recovered 90.4% pips / 78.4% USD."],
        ["8", "Synthesis", "Locked root causes, preserved edge, refuted hypotheses."],
        ["9", "Blueprint", "Specified v2: deterministic shell, constrained LLM, telemetry-first."],
    ]
    x1, x2 = 0.09, 0.51
    y0 = 0.79
    for i, row in enumerate(phases):
        x = x1 if i < 5 else x2
        y = y0 - (i % 5) * 0.126
        fig.add_artist(plt.Rectangle((x, y - 0.075), 0.36, 0.092, transform=fig.transFigure, facecolor=PANEL, edgecolor=LINE, linewidth=0.75))
        fig.add_artist(plt.Rectangle((x, y - 0.075), 0.052, 0.092, transform=fig.transFigure, facecolor=RED if i < 7 else TEAL, edgecolor=LINE, linewidth=0.75))
        fig.text(x + 0.026, y - 0.030, row[0], fontsize=13, color=PAPER, ha="center", va="center", fontweight="bold")
        fig.text(x + 0.068, y - 0.006, row[1], fontsize=9.6, color=INK, fontweight="bold", va="top")
        fig.text(x + 0.068, y - 0.037, safe(wrap(row[2], 34)), fontsize=7.7, color=MUTED, va="top", linespacing=1.16)
    pdf.savefig(fig)
    plt.close(fig)


def loss_shape(pdf):
    side = read_csv("phase1_by_side.csv")
    gate = read_csv("phase1_by_trigger_family.csv").sort_values("net_pips").head(5)
    fig = page("The Loss Shape", "why a 53% win rate still failed", "Phases 1-2")
    card(fig, 0.09, 0.735, 0.18, 0.095, "Win rate", "53.1%", "not enough")
    card(fig, 0.30, 0.735, 0.18, 0.095, "Avg winner", "+5.39p", "+$88.06", TEAL)
    card(fig, 0.51, 0.735, 0.18, 0.095, "Avg loser", "-8.83p", "-$163.94", RED)
    card(fig, 0.72, 0.735, 0.18, 0.095, "Net", "-308.0p", "-$7,253", RED)
    bar_page(fig, side["side"], side["net_pips"], 0.10, 0.49, 0.34, 0.17, "By side")
    bar_page(fig, gate["trigger_family"], gate["net_pips"], 0.55, 0.42, 0.34, 0.24, "Worst gate families")
    text(
        fig,
        0.09,
        0.29,
        "The problem was not simply too many losses. It was loss asymmetry: winners were smaller, losers were larger, and variable sizing made the dollar losses much worse.",
        width=62,
        size=12,
        weight="bold",
    )
    bullets(
        fig,
        0.09,
        0.205,
        [
            "Sell-side aggregate damage was not explained by macro drift.",
            "Critical Level Reaction was the largest gate-level damage center.",
            "Momentum Continuation had pip edge but lost dollars because sizing was wrong.",
        ],
        width=62,
        size=9.6,
    )
    pdf.savefig(fig)
    plt.close(fig)


def root_causes(pdf):
    fig = page("Root Causes", "what actually caused the failure", "Phase 8")
    rows = [
        ["1", "Sell-side caveat-template collapse", "Primary cognitive failure. The LLM reused weak short templates and laundered caveats."],
        ["2", "Random / edge-blind sizing", "Added -$5,187.24 of dollar damage. Not anti-Kelly; just not tied to edge."],
        ["3", "Negative admission expectancy", "At uniform 1 lot, the same placed trades still lost -$2,065.99."],
        ["4", "Entry-generated level failure", "25 entry-failure proxy trades lost -305.1p; 17 CLR fast-failures lost -207.1p."],
        ["5", "Telemetry gaps", "No skip outcomes, path-time MAE/MFE, full rendered context, or exit replay."],
    ]
    mini_table(fig, rows, ["Rank", "Cause", "Evidence"], 0.07, 0.34, 0.86, 0.44, [0.08, 0.33, 0.59], 7.9)
    label(fig, 0.09, 0.245, "The shortest honest version")
    text(
        fig,
        0.09,
        0.215,
        "v1 let weak reasoning become action, then let edge-blind sizing amplify that action. The model was useful enough to create some good cells, but not disciplined enough to govern itself.",
        width=60,
        size=11.8,
        color=CHARCOAL,
        weight="bold",
    )
    pdf.savefig(fig)
    plt.close(fig)


def refuted_and_preserved(pdf):
    protected = read_csv("phase9_replay_protected_edge_check.csv")
    fig = page("What Not To Relitigate", "preserved edge and refuted theories", "Phase 8")
    label(fig, 0.09, 0.80, "Preserved edge")
    rows = []
    for _, r in protected.iterrows():
        rows.append([
            str(r["protected_cell"])[:35],
            str(int(r.get("found_rows", r.get("original_n", r.get("survivor_n", 0))))),
            pips(r["survivor_pips"]),
            money(r["survivor_usd_original_size"], 0),
            str(int(r["blocked_by_phase9"])),
        ])
    mini_table(fig, rows, ["Cell", "N", "Pips", "USD", "Blocked"], 0.09, 0.58, 0.82, 0.16, [0.48, 0.08, 0.15, 0.17, 0.12], 7.5)
    label(fig, 0.09, 0.49, "Refuted")
    bullets(
        fig,
        0.09,
        0.455,
        [
            "Sells did not fail because USDJPY macro drift punished shorts.",
            "Sell-CLR snapshots were not thinner than buy-CLR snapshots.",
            "Sizing was not anti-Kelly; it was random / edge-blind.",
            "The 8+ lot CLR cell was not real edge; it was compositional.",
            "Caveat-language losers were not mainly held too long at exit.",
        ],
        width=62,
        size=9.4,
    )
    text(
        fig,
        0.09,
        0.16,
        "The rebuild protects the good buy-CLR cells while attacking the sell/caveat and sizing failures. That is the line between redesign and just swinging a hammer.",
        width=62,
        size=11.2,
        color=TEAL,
        weight="bold",
    )
    pdf.savefig(fig)
    plt.close(fig)


def v2_architecture(pdf):
    fig = page("How v2 Works", "the new control architecture", "Rebuild")
    flow(
        fig,
        0.07,
        0.76,
        ["Telemetry", "Gate", "Pre-veto", "LLM", "Validator", "Sizing"],
        [BLUE_SOFT, GOLD_SOFT, RED_SOFT, PANEL, TEAL_SOFT, TEAL_SOFT],
    )
    rows = [
        ["Telemetry", "Captures exposure, pip value, risk-after-fill, rolling P&L, level packets, prompt/context, gate candidates."],
        ["Gate", "Keeps the three primary families; kills non-primary gate contamination by default."],
        ["Pre-veto", "Blocks toxic setup cells before spending an LLM call."],
        ["LLM", "Returns strict JSON. No sizing authority. No exit authority."],
        ["Validator", "Overrides place to skip when caveats, level claims, sell burden, or loss asymmetry fail."],
        ["Sizing/exit", "Deterministic lots, hard caps, deterministic stop/profit/time rules."],
    ]
    mini_table(fig, rows, ["Layer", "v2 behavior"], 0.09, 0.34, 0.82, 0.34, [0.20, 0.80], 8.1)
    text(
        fig,
        0.09,
        0.22,
        "The LLM is now one component inside a deterministic shell. It can argue from evidence; it cannot decide size, widen exits, or gloss over caveats.",
        width=62,
        size=11.7,
        color=BLUE,
        weight="bold",
    )
    pdf.savefig(fig)
    plt.close(fig)


def rebuild_steps(pdf):
    fig = page("What Was Built", "v2 rebuild steps", "Implementation")
    rows = [
        ["1", "Telemetry & snapshot", "v2 schema, state isolation, schema hash, blocking telemetry."],
        ["2", "Validators", "Strict JSON parse and five deterministic post-decision validators."],
        ["3", "Pre-vetoes", "V1 sell caveat-template and V2 mixed-overlap checks before LLM."],
        ["4", "Sizing", "Pure deterministic function in core/fillmore_v2_sizing.py."],
        ["5", "Gates", "Side-asymmetric CLR, conditional momentum, mean reversion retained."],
        ["6", "LLM/orchestrator", "Strict prompt/output contract and end-to-end run_decision path."],
        ["7", "Exit/tripwires", "Deterministic exit layer and rollout tripwire functions."],
        ["8", "Replay", "Pass A/B/C verification against forensic corpus."],
        ["9", "Live bridge", "v1/v2 engine flag, paper guard, v2 audit rows, readiness docs."],
    ]
    mini_table(fig, rows, ["Step", "Component", "What landed"], 0.07, 0.225, 0.86, 0.58, [0.08, 0.27, 0.65], 7.8)
    text(
        fig,
        0.09,
        0.135,
        "v2 was built in parallel. v1 remains default; v2 activates only through the engine flag and uses isolated v2 state.",
        width=62,
        size=11,
        color=TEAL,
        weight="bold",
    )
    pdf.savefig(fig)
    plt.close(fig)


def replay_results(pdf):
    fig = page("Replay Verification", "did v2 beat the diagnostic floor?", "Step 8")
    card(fig, 0.09, 0.73, 0.24, 0.10, "Pip recovery", "+282.5p", "floor +278.4p", TEAL)
    card(fig, 0.38, 0.73, 0.24, 0.10, "USD recovery", "+$6,370.77", "floor +$5,684.56", TEAL)
    card(fig, 0.67, 0.73, 0.24, 0.10, "Cap-to-4", "+$6,599.37", "beat target", TEAL)
    rows = [
        ["Recovery floors", "PASS", "Pips and USD met or exceeded the binding floor."],
        ["Protected cells", "PASS", "All three remained positive with exact parity to Phase 8."],
        ["False positives", "FAIL", "Blocked trades/winners exceeded ceiling due to legacy rationale parser."],
        ["Decision", "Stage 1", "User chose paper validation as the real forward gate."],
    ]
    mini_table(fig, rows, ["Check", "Status", "Interpretation"], 0.09, 0.43, 0.82, 0.23, [0.26, 0.16, 0.58], 8.2)
    text(
        fig,
        0.09,
        0.30,
        "Important nuance: the false-positive breach traces to a corpus-only adapter that reverse-engineers old free-text rationales. Production v2 emits structured JSON, so Stage 1 paper data becomes the binding validation sample.",
        width=62,
        size=10.8,
        color=CHARCOAL,
        weight="bold",
    )
    pdf.savefig(fig)
    plt.close(fig)


def readiness(pdf):
    fig = page("Live Testing Readiness", "what is safe today vs still blocked", "Stage 1")
    card(fig, 0.09, 0.735, 0.24, 0.095, "Engine", "v1 default", "v2 opt-in")
    card(fig, 0.38, 0.735, 0.24, 0.095, "v2 stage", "paper-live", "real ticks")
    card(fig, 0.67, 0.735, 0.24, 0.095, "Broker send", "blocked", "audit rows only", TEAL)
    label(fig, 0.09, 0.64, "Ready for Stage 1 paper-live")
    bullets(
        fig,
        0.09,
        0.605,
        [
            "v2 dispatch path refuses non-paper stage unless future code explicitly opts in.",
            "v2 does not call broker order-send functions in Stage 1.",
            "Rows are persisted with engine_version='v2' and replay telemetry.",
            "v2 state is isolated from v1 in runtime_state_fillmore_v2.json.",
        ],
        width=62,
        size=9.5,
    )
    label(fig, 0.09, 0.36, "Not cleared before 0.1x")
    bullets(
        fig,
        0.09,
        0.325,
        [
            "Real side-normalized level-quality builder.",
            "Better macro bias and catalyst classification.",
            "Skip-forward outcome capture with at least 98% coverage for T5.",
            "Stage progression reporting and 50-close Stage 1 paper report.",
            "Exit-layer live replay logs if v2 paper decisions become actual paper opens/closes.",
        ],
        width=62,
        size=9.5,
        color=RED,
    )
    pdf.savefig(fig)
    plt.close(fig)


def ui_page(pdf):
    fig = page("What The UI Should Communicate", "matching the v2 system", "Operator UI")
    rows = [
        ["Engine selector", "v1 legacy / v2 Stage 1. v1 remains default."],
        ["Paper guard", "v2 clearly says Stage 1 paper and no broker order-send."],
        ["First-tick audit", "Latest v2 row, missing telemetry, gate, decision, parse, pre-vetoes, validators, deterministic lots."],
        ["Deterministic sizing", "Old LLM lot controls hidden under v2; v2 shows authority and hard cap."],
        ["Before 0.1x", "The blocker checklist stays visible so paper-live does not get confused with scaled testing."],
        ["Rollback", "Set engine back to v1; v2 state remains isolated."],
    ]
    mini_table(fig, rows, ["UI area", "What it must say"], 0.08, 0.39, 0.84, 0.40, [0.28, 0.72], 8.0)
    text(
        fig,
        0.09,
        0.27,
        "The UI's job is not to make v2 look finished. Its job is to make the operating boundary impossible to miss: Stage 1 paper-live only, audit everything, no scaled sizing yet.",
        width=62,
        size=11.5,
        color=BLUE,
        weight="bold",
    )
    pdf.savefig(fig)
    plt.close(fig)


def first_hour(pdf):
    fig = page("First Hour Checklist", "what to watch when v2 is flipped", "Runbook")
    label(fig, 0.09, 0.80, "Must be true")
    bullets(
        fig,
        0.09,
        0.765,
        [
            "New rows appear in ai_suggestions with engine_version='v2'.",
            "snapshot_blocking_strikes stays at 0 when inputs are healthy.",
            "No lots above 4.",
            "Every sell-side placement has pre-veto and validator audit metadata.",
            "Every placed row has pip_value_per_lot, risk_after_fill_usd, rendered prompt/context, gate candidates, and deterministic sizing inputs.",
        ],
        width=62,
        size=9.6,
    )
    label(fig, 0.09, 0.44, "Rollback")
    text(
        fig,
        0.09,
        0.405,
        "Set the engine flag back to v1, then confirm no new engine_version='v2' rows appear after the rollback timestamp. v2 halt/strike state remains isolated.",
        width=62,
        size=10.4,
    )
    label(fig, 0.09, 0.29, "Known non-launch debt")
    text(
        fig,
        0.09,
        0.255,
        "Full suite has one unrelated pre-existing failure in tests/test_phase3_additive_runtime.py. It is documented as non-v2 launch debt and should not confuse Stage 1 launch/no-launch decisions.",
        width=62,
        size=9.8,
        color=MUTED,
    )
    pdf.savefig(fig)
    plt.close(fig)


def final_page(pdf):
    fig = page("The Bottom Line", "what changed and why it matters", "Summary")
    rows = [
        ["v1 problem", "Weak reasoning could become action, and action could become oversized exposure."],
        ["v2 answer", "Constrain the model with deterministic gates, vetoes, validators, sizing, exits, and telemetry."],
        ["Current status", "Ready for supervised Stage 1 paper-live testing, not scaled 0.1x testing."],
        ["What proves progress", "50 closed paper decisions with net pips >= 0, simulated PF >= 0.9, zero missing blocking telemetry, no protected-cell regression, no stop widening."],
    ]
    mini_table(fig, rows, ["Question", "Answer"], 0.08, 0.49, 0.84, 0.28, [0.26, 0.74], 8.6)
    text(
        fig,
        0.09,
        0.34,
        "The system is now designed to learn from itself. If Stage 1 fails, the next audit should be able to say exactly which deterministic layer, model output, telemetry field, or market regime failed.",
        width=62,
        size=12,
        color=TEAL,
        weight="bold",
    )
    fig.text(0.09, 0.13, "Primary sources", fontsize=9, color=RED, fontweight="bold")
    text(
        fig,
        0.09,
        0.105,
        "PHASE1_BASELINE.md through PHASE9_OVERHAUL_BLUEPRINT.md; docs/fillmore_v2/CHANGELOG.md; docs/fillmore_v2/step8_replay_results.md; docs/fillmore_v2/live_testing_readiness_20260502.md.",
        width=66,
        size=8.5,
        color=MUTED,
    )
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT) as pdf:
        cover(pdf)
        how_to_read(pdf)
        fillmore_plain_english(pdf)
        v1_vs_v2(pdf)
        phase_map(pdf)
        loss_shape(pdf)
        root_causes(pdf)
        refuted_and_preserved(pdf)
        v2_architecture(pdf)
        rebuild_steps(pdf)
        replay_results(pdf)
        readiness(pdf)
        ui_page(pdf)
        first_hour(pdf)
        final_page(pdf)
    print(OUT)


if __name__ == "__main__":
    main()
