#!/usr/bin/env python3
"""Render a concise, visual explainer PDF for Auto Fillmore.

Audience: a reader who does not already know Fillmore. The PDF explains:
  - What Fillmore is.
  - What the Phase 1-9 forensic investigation found.
  - What the v2 rebuild changed.
  - How v2 works now, and what is still required before scaled testing.
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
OUT = FORENSIC / "Auto_Fillmore_Explainer_Audit_and_v2_Rebuild.pdf"

BG = "#F6F1E8"
PAPER = "#FFFDF8"
INK = "#172233"
MUTED = "#65717F"
GRID = "#D8CFC0"
RED = "#A84037"
RED_DARK = "#74302B"
TEAL = "#176B6D"
TEAL_DARK = "#0E5355"
GOLD = "#B7852F"
BLUE = "#335C89"
PANEL = "#FFFFFF"
SOFT = "#F1E8DA"
GREEN_SOFT = "#E6F0EC"
RED_SOFT = "#F3E1DC"

PAGE_NO = 0


def safe(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).replace("$", r"\$")


def wrap(text: str, width: int = 72) -> str:
    return "\n".join(
        textwrap.wrap(str(text), width=width, break_long_words=False, break_on_hyphens=False)
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


def pct(value: object) -> str:
    try:
        v = float(value)
    except Exception:
        return safe(value)
    if abs(v) <= 1.0:
        v *= 100.0
    return f"{v:.1f}%"


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(FORENSIC / name)


def md_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def fig_base(title: str, subtitle: str = "", label: str = ""):
    global PAGE_NO
    PAGE_NO += 1
    fig = plt.figure(figsize=(13.2, 8.5), facecolor=BG)
    fig.add_artist(
        plt.Rectangle(
            (0.035, 0.045),
            0.93,
            0.89,
            transform=fig.transFigure,
            facecolor=PAPER,
            edgecolor=GRID,
            linewidth=1.0,
        )
    )
    fig.text(0.065, 0.91, safe(title), fontsize=22, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.065, 0.875, safe(subtitle), fontsize=9.8, color=MUTED)
    if label:
        fig.text(0.935, 0.91, safe(label), fontsize=9.5, color=RED_DARK, fontweight="bold", ha="right")
    fig.add_artist(plt.Line2D([0.065, 0.935], [0.855, 0.855], color=GRID, linewidth=0.9))
    fig.text(0.935, 0.055, str(PAGE_NO), fontsize=8, color=MUTED, ha="right")
    return fig


def add_card(fig, x, y, w, h, title: str, value: str, note: str = "", color=INK, fill=PANEL):
    fig.add_artist(
        plt.Rectangle((x, y), w, h, transform=fig.transFigure, facecolor=fill, edgecolor=GRID, linewidth=0.85)
    )
    fig.text(x + 0.015, y + h - 0.031, safe(title.upper()), fontsize=7.4, color=MUTED, fontweight="bold")
    fs = 15 if len(value) < 24 else 12
    fig.text(x + 0.015, y + h * 0.39, safe(value), fontsize=fs, color=color, fontweight="bold")
    if note:
        fig.text(x + 0.015, y + 0.021, safe(wrap(note, 36)), fontsize=7.2, color=MUTED, va="bottom")


def add_text(fig, x, y, text: str, width: int = 76, size: float = 9.5, color=INK, weight="normal"):
    fig.text(x, y, safe(wrap(text, width)), fontsize=size, color=color, fontweight=weight, va="top", linespacing=1.28)


def add_bullets(fig, x, y, items: Iterable[str], width: int = 64, size: float = 8.7, color=INK, gap=0.060):
    yy = y
    for item in items:
        wrapped = wrap(item, width).replace("\n", "\n  ")
        fig.text(x, yy, safe("- " + wrapped), fontsize=size, color=color, va="top", linespacing=1.24)
        yy -= gap + 0.013 * wrapped.count("\n")
    return yy


def add_table(fig, df: pd.DataFrame, x, y, w, h, font_size=7.4, col_widths=None):
    ax = fig.add_axes([x, y, w, h])
    ax.axis("off")
    clean = df.copy()
    for col in clean.columns:
        clean[col] = clean[col].map(safe)
    table = ax.table(
        cellText=clean.values,
        colLabels=[safe(c) for c in clean.columns],
        cellLoc="left",
        colWidths=col_widths,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.24)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#E5DCCE")
        if row == 0:
            cell.set_facecolor("#EFE4D4")
            cell.set_text_props(weight="bold", color=INK)
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 else "#FBF6EE")
            cell.set_text_props(color=INK)
    return ax


def add_flow(fig, y: float, labels: list[str], colors: list[str] | None = None):
    colors = colors or [PANEL] * len(labels)
    x0 = 0.075
    w = 0.145
    gap = 0.024
    for i, label in enumerate(labels):
        x = x0 + i * (w + gap)
        fig.add_artist(plt.Rectangle((x, y), w, 0.095, transform=fig.transFigure, facecolor=colors[i], edgecolor=GRID))
        fig.text(x + w / 2, y + 0.053, safe(wrap(label, 17)), fontsize=8.1, color=INK, ha="center", va="center", fontweight="bold")
        if i < len(labels) - 1:
            fig.text(x + w + gap / 2, y + 0.052, ">", fontsize=16, color=MUTED, ha="center", va="center")


def add_barh(fig, labels, values, x, y, w, h, title: str, value_formatter=pips):
    ax = fig.add_axes([x, y, w, h], facecolor=PAPER)
    vals = [float(v) for v in values]
    labs = [str(v)[:34] for v in labels]
    colors = [RED if v < 0 else TEAL for v in vals]
    ax.barh(range(len(vals)), vals, color=colors, alpha=0.94)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labs, fontsize=7.5, color=INK)
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, fontweight="bold")
    ax.grid(axis="x", color=GRID, alpha=0.65)
    ax.tick_params(axis="x", labelsize=7.3, colors=MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for idx, val in enumerate(vals):
        ax.text(val, idx, " " + value_formatter(val), va="center", fontsize=7.0, color=INK)


def cover(pdf):
    fig = plt.figure(figsize=(13.2, 8.5), facecolor=BG)
    fig.add_artist(plt.Rectangle((0.045, 0.06), 0.91, 0.86, transform=fig.transFigure, facecolor=PAPER, edgecolor=GRID, linewidth=1.1))
    fig.text(0.08, 0.80, "AUTO FILLMORE", fontsize=38, fontweight="bold", color=INK)
    fig.text(0.08, 0.725, "Investigation + v2 Rebuild Explainer", fontsize=24, fontweight="bold", color=RED_DARK)
    fig.text(0.08, 0.675, "What it is, why v1 failed, what v2 changed, and how live paper testing works", fontsize=12, color=MUTED)
    add_text(
        fig,
        0.08,
        0.595,
        "This PDF is written for two audiences: the operator deciding whether Fillmore is ready for Stage 1 paper-live testing, and a new reader trying to learn what Fillmore is. It condenses the Phase 1-9 forensic audit and the v2 rebuild without turning the evidence into a wall of text.",
        width=82,
        size=13,
    )
    add_card(fig, 0.08, 0.40, 0.24, 0.11, "Audit sample", "241 closed trades", "Apr 16-May 1 USDJPY", RED_DARK)
    add_card(fig, 0.35, 0.40, 0.24, 0.11, "v1 result", "-308.0p / -$7,253", "53.1% WR still lost", RED)
    add_card(fig, 0.62, 0.40, 0.24, 0.11, "v2 replay", "+324.3p / +$6,421", "recovery floor passed", TEAL)
    add_card(fig, 0.08, 0.25, 0.24, 0.11, "Current status", "Stage 1 paper-ready", "not 0.1x yet", BLUE)
    add_card(fig, 0.35, 0.25, 0.24, 0.11, "Core fix", "deterministic shell", "LLM constrained", TEAL_DARK)
    add_card(fig, 0.62, 0.25, 0.24, 0.11, "Safety guard", "paper-only dispatch", "no broker order-send", GOLD)
    fig.text(0.08, 0.105, "Sources: PHASE1-PHASE9 reports, v2 changelog, Step 8 replay results, live readiness docs", fontsize=8.5, color=MUTED)
    pdf.savefig(fig)
    plt.close(fig)


def page_what_is_fillmore(pdf):
    fig = fig_base("What Is Auto Fillmore?", "A plain-English model of the system", "Intro")
    add_text(
        fig,
        0.075,
        0.805,
        "Fillmore is an autonomous USDJPY trading agent. It watches the market every loop, waits for a code-defined setup, asks an LLM whether the setup deserves a trade, and logs the decision with enough evidence to audit later.",
        width=95,
        size=12,
        weight="bold",
    )
    add_flow(
        fig,
        0.645,
        ["Market tick", "Code gate", "Snapshot", "LLM decision", "Validator / sizing", "Paper audit row"],
        [SOFT, SOFT, PANEL, PANEL, GREEN_SOFT, GREEN_SOFT],
    )
    add_text(fig, 0.075, 0.545, "Key terms", width=30, size=12, color=RED_DARK, weight="bold")
    terms = pd.DataFrame(
        [
            ["Gate", "A deterministic detector for a setup: critical level reaction, momentum continuation, or mean reversion."],
            ["Snapshot", "The market/account/context packet sent to the LLM and persisted for audit."],
            ["LLM", "The model that judges evidence. In v2 it cannot size trades or manage exits."],
            ["Validator", "Deterministic checks that can override the LLM from place to skip."],
            ["Stage 1", "Paper-live validation: real live ticks, no real-capital order placement through v2."],
        ],
        columns=["Term", "Meaning"],
    )
    add_table(fig, terms, 0.075, 0.185, 0.85, 0.34, font_size=8.2, col_widths=[0.17, 0.83])
    add_text(
        fig,
        0.075,
        0.125,
        "The rebuild does not make the LLM smarter by trust. It makes the LLM smaller in authority: evidence only, no sizing, no exits, no loose caveats.",
        width=96,
        size=10,
        color=BLUE,
        weight="bold",
    )
    pdf.savefig(fig)
    plt.close(fig)


def page_v1_architecture(pdf):
    fig = fig_base("How v1 Worked", "The original loop mixed useful ideas with too much model discretion", "v1")
    add_flow(
        fig,
        0.705,
        ["1 sec polling", "Gate fires", "Large snapshot", "LLM says take/skip", "LLM size", "Order path"],
        [SOFT, SOFT, PANEL, PANEL, RED_SOFT, RED_SOFT],
    )
    left = [
        "The code gate triggered when it saw one of several setup families.",
        "The LLM received raw market context and wrote a rationale.",
        "The LLM could output trade direction, rationale, order type, and lots.",
        "Prompt patches tried to steer behavior over time, but the system still placed too many weak trades.",
    ]
    right = [
        "The LLM was allowed to turn caveats into permission instead of a stop signal.",
        "Sizing was tied loosely to conviction, not to a deterministic risk model.",
        "Snapshot fields were useful but not side-normalized enough for level quality.",
        "Telemetry gaps made skip quality, path timing, and exit replay hard to prove after the fact.",
    ]
    add_text(fig, 0.075, 0.58, "What was good", width=30, size=12, color=TEAL_DARK, weight="bold")
    add_bullets(fig, 0.075, 0.535, left, width=58)
    add_text(fig, 0.535, 0.58, "What broke", width=30, size=12, color=RED_DARK, weight="bold")
    add_bullets(fig, 0.535, 0.535, right, width=58)
    add_text(
        fig,
        0.075,
        0.185,
        "The key lesson: v1 was not only losing because it took bad trades. It lost because bad trade admission, loose reasoning, random sizing, and incomplete telemetry reinforced each other.",
        width=96,
        size=12,
        weight="bold",
        color=INK,
    )
    pdf.savefig(fig)
    plt.close(fig)


def page_phase_map(pdf):
    fig = fig_base("Phase 1-9 Investigation Map", "Every phase had a specific job", "Audit")
    phases = [
        ("1 Baseline", "Verified data and measured the loss shape.", "53.1% WR, -308.0p, -$7,253."),
        ("2 Gates", "Audited setup families and side behavior.", "Sell-CLR was catastrophic; momentum had pip edge but USD bleed."),
        ("3 Snapshot", "Checked what the model actually saw.", "Raw fields were symmetric; missing side-normalized level packet."),
        ("4 Reasoning", "Audited rationale text and prompts.", "Sell-side caveat-template collapse became the primary cognitive finding."),
        ("5 Sizing", "Separated pip edge from dollar damage.", "Sizing was random/edge-blind and added -$5,187."),
        ("6 Lifecycle", "Entry vs exit damage.", "Entry-generated damage dominated; runner language caused harm."),
        ("7 Interactions", "Found toxic and preserved cells.", "V1+V2 veto floor recovered 90.4% pips / 78.4% USD in-sample."),
        ("8 Synthesis", "Ranked root causes.", "Locked causal hierarchy, refuted hypotheses, preserved edge."),
        ("9 Blueprint", "Designed the replacement.", "Deterministic shell, constrained LLM, required telemetry, rollout gates."),
    ]
    for i, (phase, job, result) in enumerate(phases):
        x = 0.075 if i < 5 else 0.525
        y = 0.765 - (i % 5) * 0.135
        fig.add_artist(plt.Rectangle((x, y - 0.085), 0.39, 0.108, transform=fig.transFigure, facecolor=PANEL, edgecolor=GRID, linewidth=0.8))
        fig.text(x + 0.015, y, phase, fontsize=10.3, color=RED_DARK, fontweight="bold")
        fig.text(x + 0.015, y - 0.030, safe(wrap(job, 50)), fontsize=7.8, color=MUTED, va="top")
        fig.text(x + 0.015, y - 0.061, safe(wrap(result, 54)), fontsize=8.2, color=INK, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def page_baseline(pdf):
    side = read_csv("phase1_by_side.csv")
    gate = read_csv("phase1_by_trigger_family.csv").sort_values("net_pips").head(6)
    fig = fig_base("Baseline: The System Was Losing Despite Winning Often", "Phase 1 evidence", "Phases 1-2")
    add_card(fig, 0.075, 0.72, 0.18, 0.105, "Closed trades", "241", "Apr 16-May 1")
    add_card(fig, 0.275, 0.72, 0.18, 0.105, "Win rate", "53.1%", "not enough")
    add_card(fig, 0.475, 0.72, 0.18, 0.105, "Net pips", "-308.0p", "negative expectancy", RED)
    add_card(fig, 0.675, 0.72, 0.20, 0.105, "Net USD", "-$7,253", "sizing amplified", RED)
    add_barh(fig, side["side"], side["net_pips"], 0.075, 0.40, 0.37, 0.24, "Side damage by pips")
    add_barh(fig, gate["trigger_family"], gate["net_pips"], 0.535, 0.31, 0.37, 0.33, "Worst gate families by pips")
    add_text(
        fig,
        0.075,
        0.205,
        "The most important shape was asymmetry: average winner +5.39p / +$88.06, average loser -8.83p / -$163.94. A 53.1% win rate could not overcome losers that were larger and often oversized.",
        width=96,
        size=11.5,
        weight="bold",
    )
    pdf.savefig(fig)
    plt.close(fig)


def page_root_causes(pdf):
    fig = fig_base("Root Cause Hierarchy", "The audit separated causes from symptoms", "Phase 8")
    causes = pd.DataFrame(
        [
            ["1", "Sell-side caveat-template collapse", "Reasoning layer", "V1 alone recovered +278.4p / +$4,397.69"],
            ["2", "Random / edge-blind sizing", "Sizing layer", "-$5,187.24 of dollar loss came from variable sizing"],
            ["3", "Negative admission expectancy", "Gate + decision", "At uniform 1 lot, same trades still lost -$2,065.99"],
            ["4", "Entry-generated level failure", "Entry layer", "25 entry-failure proxy trades: -305.1p"],
            ["5", "Telemetry gaps", "Feedback layer", "Could not prove skip quality, path ordering, or full exit replay"],
        ],
        columns=["Rank", "Cause", "Layer", "Evidence"],
    )
    add_table(fig, causes, 0.065, 0.37, 0.87, 0.39, font_size=8.0, col_widths=[0.06, 0.29, 0.18, 0.47])
    add_text(
        fig,
        0.075,
        0.275,
        "Plain-language causal chain: the gate admitted too many marginal setups, the LLM rationalized short-side caveats, deterministic backstops were too weak, sizing did not track edge, and the logs did not capture enough to learn cleanly after the fact.",
        width=96,
        size=11.0,
        weight="bold",
        color=INK,
    )
    add_text(
        fig,
        0.075,
        0.155,
        "Refuted: shorts did not fail because the macro drift punished them; sell snapshots were not thinner than buy snapshots; sizing was not anti-Kelly, just edge-blind; the 8+ lot CLR cell was not real preserved edge.",
        width=96,
        size=9.5,
        color=MUTED,
    )
    pdf.savefig(fig)
    plt.close(fig)


def page_phases_1_3(pdf):
    fig = fig_base("Phases 1-3: Data, Gates, Snapshot", "What was proven before blaming the model", "Phases 1-3")
    rows = pd.DataFrame(
        [
            ["Phase 1", "Baseline", "Data was usable. 241 closed trades, -308.0p, -$7,253.24, PF 0.61.", "Trust the placed-trade analysis; skip-side was incomplete."],
            ["Phase 2", "Gate audit", "Critical Level Reaction had 149 trades and -183.7p. Sell-CLR: -184.3p / -$2,081.56.", "CLR needed side-asymmetric redesign."],
            ["Phase 2", "Momentum", "Momentum Continuation: +10.0p but -$1,241.24.", "Pip edge existed, sizing destroyed USD result."],
            ["Phase 3", "Snapshot", "Buy/sell CLR snapshots had symmetric core data.", "Sell collapse was not mainly a thinner-snapshot bug."],
            ["Phase 3", "Missing input", "No open exposure, rolling P&L, side-normalized level packet, snapshot_version, or path-time telemetry.", "Telemetry had to become a first-class layer."],
        ],
        columns=["Phase", "Topic", "Finding", "Meaning"],
    )
    add_table(fig, rows, 0.055, 0.245, 0.89, 0.50, font_size=7.7, col_widths=[0.09, 0.14, 0.43, 0.34])
    add_text(
        fig,
        0.075,
        0.155,
        "The snapshot was not exonerated completely: it lacked side-normalized level quality and portfolio risk fields. But the sell-CLR collapse could not be explained away as simple missing buy/sell coverage.",
        width=96,
        size=10.5,
        weight="bold",
    )
    pdf.savefig(fig)
    plt.close(fig)


def page_phases_4_6(pdf):
    fig = fig_base("Phases 4-6: Reasoning, Sizing, Lifecycle", "The core pathology came into focus", "Phases 4-6")
    add_card(fig, 0.075, 0.72, 0.25, 0.10, "Reasoning", "caveat laundering", "doubt became permission", RED)
    add_card(fig, 0.375, 0.72, 0.25, 0.10, "Sizing", "-$5,187", "variable sizing delta", RED)
    add_card(fig, 0.675, 0.72, 0.22, 0.10, "Lifecycle", "entry-dominant", "25 trades -305.1p", RED_DARK)
    rows = pd.DataFrame(
        [
            ["Phase 4", "Sell-CLR reasoning failed through template collapse: weak short rationales reused generic level/caveat language."],
            ["Phase 4", "The model often used caveats as narrative decoration instead of making them terminal."],
            ["Phase 5", "Sizing was random/edge-blind, not anti-Kelly. It did not reliably size bigger into better trades."],
            ["Phase 5", "Uniform 1-lot replay showed the admission layer still lost, but much less than actual USD loss."],
            ["Phase 6", "Entry-generated damage dominated: many losers had almost no favorable movement."],
            ["Phase 6", "Runner-preservation prompt language caused measurable harm and was removed in v2."],
        ],
        columns=["Phase", "Key read"],
    )
    add_table(fig, rows, 0.075, 0.275, 0.85, 0.36, font_size=8.3, col_widths=[0.12, 0.88])
    add_text(
        fig,
        0.075,
        0.175,
        "The insight that shaped v2: do not ask the model to be disciplined. Build discipline around it. Caveats, sizing, and exits became deterministic enforcement problems.",
        width=95,
        size=11.2,
        weight="bold",
        color=BLUE,
    )
    pdf.savefig(fig)
    plt.close(fig)


def page_phases_7_9(pdf):
    replay = read_csv("phase9_replay_summary.csv")
    fig = fig_base("Phases 7-9: Interactions to Blueprint", "The audit turned into a redesign spec", "Phases 7-9")
    add_text(
        fig,
        0.075,
        0.805,
        "Phase 7 found the minimum in-sample damage-control floor. Phase 8 locked the causal hierarchy and protected edge. Phase 9 designed a new architecture that had to beat the floor without destroying the protected cells.",
        width=96,
        size=11.2,
        weight="bold",
    )
    floor_rows = pd.DataFrame(
        [
            ["V1 floor", "+278.4p", "+$4,397.69", "Sell-side caveat cluster alone"],
            ["V1+V2 floor", "+300.5p", "+$5,684.56", "Binding diagnostic floor"],
            ["Phase 9 filter", "+324.3p", "+$6,086.60", "Admission filter replay"],
            ["Filter + cap 4", "-", "+$6,420.90", "Adds deterministic size cap"],
        ],
        columns=["Replay", "Pip recovery", "USD recovery", "Meaning"],
    )
    add_table(fig, floor_rows, 0.075, 0.47, 0.85, 0.25, font_size=8.4, col_widths=[0.18, 0.16, 0.18, 0.48])
    protected = pd.read_csv(FORENSIC / "phase9_replay_protected_edge_check.csv")
    protected_small = pd.DataFrame(
        {
            "Protected cell": protected["protected_cell"].map(lambda s: str(s)[:44]),
            "Blocked": protected["blocked_by_phase9"],
            "Survivor pips": protected["survivor_pips"].map(pips),
            "Survivor USD": protected["survivor_usd_original_size"].map(lambda v: money(v, 0)),
        }
    )
    add_table(fig, protected_small, 0.075, 0.165, 0.85, 0.20, font_size=7.7, col_widths=[0.55, 0.10, 0.16, 0.19])
    add_text(fig, 0.075, 0.115, "All three protected cells survived Phase 9 replay with zero rows blocked.", width=96, size=10.0, color=TEAL_DARK, weight="bold")
    pdf.savefig(fig)
    plt.close(fig)


def page_blueprint(pdf):
    fig = fig_base("The Phase 9 Blueprint", "The design philosophy of the replacement system", "Blueprint")
    add_flow(
        fig,
        0.72,
        ["Telemetry", "Gate", "Pre-veto", "LLM", "Validator", "Sizing", "Exit", "Feedback"],
        [SOFT, SOFT, RED_SOFT, PANEL, GREEN_SOFT, GREEN_SOFT, GREEN_SOFT, SOFT],
    )
    rows = pd.DataFrame(
        [
            ["Deterministic where possible", "Sizing, risk caps, stops, vetoes, and telemetry do not depend on model judgment."],
            ["Model where necessary", "The LLM only adjudicates evidence after deterministic filters clear the setup."],
            ["Side-asymmetric admission", "Buy-CLR and sell-CLR are different paths with different evidence burdens."],
            ["Caveats are terminal", "A caveat must be resolved with snapshot evidence or the trade is skipped."],
            ["Telemetry-first", "Every decision must be replayable from logs in the next audit."],
        ],
        columns=["Principle", "How it shows up"],
    )
    add_table(fig, rows, 0.075, 0.34, 0.85, 0.30, font_size=8.2, col_widths=[0.27, 0.73])
    add_text(
        fig,
        0.075,
        0.235,
        "The blueprint did not try to rescue v1 with another prompt patch. It changed authority: the LLM can argue for a trade, but deterministic layers decide whether that argument is valid, how large the trade can be, and how exits behave.",
        width=96,
        size=11.3,
        weight="bold",
    )
    pdf.savefig(fig)
    plt.close(fig)


def page_rebuild_steps(pdf):
    fig = fig_base("Auto Fillmore v2 Rebuild", "What was implemented after the audit", "Rebuild")
    steps = [
        ("1 Telemetry", "Snapshot schema, v2 state file, v2 DB columns, blocking-field halt policy."),
        ("2 Validators", "Strict JSON schema plus caveat, level-overreach, sell-burden, loss-asymmetry checks."),
        ("3 Pre-vetoes", "V1 sell caveat-template and V2 mixed-overlap rules before LLM token spend."),
        ("4 Sizing", "Deterministic risk function, 1-4 lot hard bound, drawdown/exposure/volatility throttles."),
        ("5 Gates", "Primary gates only: buy-CLR, sell-CLR, momentum, mean reversion. Non-primary killed."),
        ("6 LLM layer", "No sizing, no exits, structured JSON only, prompt forbids runner/loose caveats."),
        ("7 Exit layer", "Deterministic stop, profit lock, time stop, no LLM stop widening."),
        ("8 Replay", "Passes recovery and protected-cell floors; false-positive miss traced to legacy adapter."),
        ("9 Rollout", "Engine flag, tripwires, staged rollout, default v1."),
        ("Post-9", "Live-loop bridge, paper guard, first-tick checker, supervised session."),
    ]
    for i, (name, detail) in enumerate(steps):
        x = 0.075 if i < 5 else 0.525
        y = 0.78 - (i % 5) * 0.135
        fig.add_artist(plt.Rectangle((x, y - 0.083), 0.39, 0.105, transform=fig.transFigure, facecolor=PANEL, edgecolor=GRID, linewidth=0.8))
        fig.text(x + 0.015, y, name, fontsize=10.0, color=TEAL_DARK, fontweight="bold")
        fig.text(x + 0.015, y - 0.032, safe(wrap(detail, 55)), fontsize=8.0, color=INK, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def page_v1_v2_diff(pdf):
    fig = fig_base("v1 vs v2: What Changed?", "The core differences in one table", "Comparison")
    rows = pd.DataFrame(
        [
            ["Authority", "LLM had broad influence over trade + size.", "LLM only decides evidence; deterministic layers enforce."],
            ["Sizing", "Model/logic could amplify losers.", "Deterministic function, hard 1-4 lot bound, throttles."],
            ["Caveats", "Could be explained away in prose.", "Terminal unless resolved with snapshot evidence."],
            ["Sell side", "Same general framing as buys.", "Higher burden, pre-vetoes, sell-side validator."],
            ["Levels", "Raw support/resistance objects.", "Side-normalized level packet required for real CLR."],
            ["Exits", "Runner language could bias holding.", "No LLM stop widening; deterministic exit layer."],
            ["Telemetry", "Important gaps in skip/path/context logs.", "Versioned snapshot, full prompt/context, audit columns."],
            ["Rollout", "Prompt patches could blur epochs.", "Engine flag, separate v2 state, staged paper validation."],
        ],
        columns=["Area", "v1", "v2"],
    )
    add_table(fig, rows, 0.055, 0.145, 0.89, 0.63, font_size=7.6, col_widths=[0.16, 0.40, 0.44])
    pdf.savefig(fig)
    plt.close(fig)


def page_v2_how_it_works(pdf):
    fig = fig_base("How v2 Works Now", "The live paper-testing path", "v2")
    add_flow(
        fig,
        0.735,
        ["Engine flag v2", "Build snapshot", "Check blocking fields", "Run gates", "Run pre-vetoes", "Size", "LLM JSON", "Validate + persist"],
        [SOFT, SOFT, RED_SOFT, SOFT, RED_SOFT, GREEN_SOFT, PANEL, GREEN_SOFT],
    )
    rows = pd.DataFrame(
        [
            ["1", "Engine flag", "v1 stays default. v2 runs only after explicit operator opt-in."],
            ["2", "Snapshot", "Bridge fills account equity, exposure, P&L, pip value, sessions, volatility, side/SL/TP, and conservative level packet."],
            ["3", "Safety check", "Missing blocking fields create strikes and can halt v2 before LLM calls."],
            ["4", "Gate + veto", "Primary gate must be eligible, then pre-vetoes can skip before model cost."],
            ["5", "Sizing", "Lots come from deterministic sizing, not model confidence."],
            ["6", "LLM + validators", "LLM returns strict JSON; validators can override place to skip."],
            ["7", "Persistence", "Rows are written with engine_version='v2' and audit metadata."],
        ],
        columns=["Step", "Layer", "What happens"],
    )
    add_table(fig, rows, 0.065, 0.235, 0.87, 0.39, font_size=8.0, col_widths=[0.07, 0.18, 0.75])
    add_text(
        fig,
        0.075,
        0.145,
        "Important: current Stage 1 v2 is paper-guarded. It does not call broker order-send functions; it persists auditable decisions so we can validate behavior against live ticks first.",
        width=95,
        size=11.0,
        color=RED_DARK,
        weight="bold",
    )
    pdf.savefig(fig)
    plt.close(fig)


def page_readiness(pdf):
    fig = fig_base("Live Testing Readiness", "What is ready, what is not", "Stage 1")
    add_card(fig, 0.075, 0.72, 0.25, 0.105, "Ready", "Stage 1 paper-live", "operator opt-in", TEAL)
    add_card(fig, 0.375, 0.72, 0.25, 0.105, "Not ready", "0.1x / real capital", "requires more wiring", RED)
    add_card(fig, 0.675, 0.72, 0.22, 0.105, "Guard", "paper-only", "no broker order-send", GOLD)
    rows = pd.DataFrame(
        [
            ["Tests", "246 v2 tests pass; 26 targeted v1/v2 tests pass; full suite has 1 unrelated known failure."],
            ["Supervised session", "engine=v2, selected_gate=momentum_continuation, final_decision=place, risk_after_fill_usd populated."],
            ["First-tick checker", "Reports missing fields, halt reason, selected gate, final decision, parse status, pre-vetoes, validators."],
            ["Rollback", "Set engine flag back to v1; v2 state remains isolated in runtime_state_fillmore_v2.json."],
        ],
        columns=["Check", "Result"],
    )
    add_table(fig, rows, 0.075, 0.36, 0.85, 0.25, font_size=8.4, col_widths=[0.18, 0.82])
    add_text(fig, 0.075, 0.265, "First-hour watch list", width=40, size=12, color=RED_DARK, weight="bold")
    add_bullets(
        fig,
        0.075,
        0.225,
        [
            "v2 rows appear with engine_version='v2'.",
            "snapshot_blocking_strikes stays at 0 when inputs are healthy.",
            "No lots above 4.",
            "Rows include prompt/context, gate candidates, sizing inputs, risk_after_fill_usd, validators, and pre-veto logs.",
        ],
        width=92,
        size=8.8,
        gap=0.045,
    )
    pdf.savefig(fig)
    plt.close(fig)


def page_remaining(pdf):
    fig = fig_base("Before 0.1x Sizing", "The remaining gates are intentional, not optional", "Next")
    rows = pd.DataFrame(
        [
            ["Real level-quality builder", "Replace the Stage 1 conservative score=65 placeholder with true side-normalized level quality."],
            ["Macro/catalyst classification", "Upgrade neutral/structure-only defaults into actual evidence labels."],
            ["Skip-forward outcomes", "T5 requires at least 98% skip outcome coverage."],
            ["Stage reporting", "Daily script or dashboard for Stage 1 criteria and tripwires."],
            ["Exit replay logs", "Needed if v2 paper decisions become actual paper opens/closes."],
            ["50-close report", "Net pips >= 0, simulated PF >= 0.9, zero missing blocking telemetry, no protected-cell regression, no stop widening."],
        ],
        columns=["Requirement", "Why it matters"],
    )
    add_table(fig, rows, 0.065, 0.30, 0.87, 0.40, font_size=8.1, col_widths=[0.28, 0.72])
    add_text(
        fig,
        0.075,
        0.205,
        "Stage 1 answers: does v2 behave correctly under live ticks? Stage 2 answers: can it trade at tiny size? Those are different questions. The remaining work exists to keep those questions clean.",
        width=96,
        size=11.2,
        weight="bold",
        color=INK,
    )
    pdf.savefig(fig)
    plt.close(fig)


def page_glossary(pdf):
    fig = fig_base("Quick Glossary", "For readers learning the system", "Reference")
    rows = pd.DataFrame(
        [
            ["Pips", "Small price units for USDJPY. 10 pips = 0.10 JPY."],
            ["MAE / MFE", "Maximum adverse/favorable excursion: how far trade went against/for you while open."],
            ["Profit factor", "Gross wins divided by gross losses. Above 1 is profitable in the sample."],
            ["CLR", "Critical Level Reaction: a trade around a support/resistance level."],
            ["Caveat laundering", "The model names a weakness, then trades anyway without proving why the weakness is resolved."],
            ["Protected edge", "Small cells that actually worked and must not be destroyed by broad rules."],
            ["Pre-veto", "A deterministic skip before the LLM is called."],
            ["Validator", "A deterministic check after the LLM responds; can override place to skip."],
            ["Paper-live", "Live market ticks, but v2 does not place real-capital orders."],
        ],
        columns=["Term", "Meaning"],
    )
    add_table(fig, rows, 0.065, 0.16, 0.87, 0.62, font_size=8.2, col_widths=[0.20, 0.80])
    pdf.savefig(fig)
    plt.close(fig)


def render():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    global PAGE_NO
    PAGE_NO = 0
    with PdfPages(OUT) as pdf:
        cover(pdf)
        page_what_is_fillmore(pdf)
        page_v1_architecture(pdf)
        page_phase_map(pdf)
        page_baseline(pdf)
        page_root_causes(pdf)
        page_phases_1_3(pdf)
        page_phases_4_6(pdf)
        page_phases_7_9(pdf)
        page_blueprint(pdf)
        page_rebuild_steps(pdf)
        page_v1_v2_diff(pdf)
        page_v2_how_it_works(pdf)
        page_readiness(pdf)
        page_remaining(pdf)
        page_glossary(pdf)
    print(OUT)


if __name__ == "__main__":
    render()
