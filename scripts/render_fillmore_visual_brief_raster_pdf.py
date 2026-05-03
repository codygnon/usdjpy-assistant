#!/usr/bin/env python3
"""Render a robust raster PDF brief for Auto Fillmore.

The earlier matplotlib PDF looked acceptable on its cover but some interior
pages rendered badly in PDF viewers. This version draws every page with PIL and
saves a multi-page raster PDF. It trades selectable text for reliability and
scanability: if the PNG previews look right, the PDF pages look right.
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FORENSIC = ROOT / "research_out" / "autonomous_fillmore_forensic_20260501"
OUT = FORENSIC / "Auto_Fillmore_Visual_Brief_v3_readable.pdf"
PREVIEW_DIR = Path("/private/tmp/fillmore_visual_brief_v3_pages")

W, H = 1600, 2200
M = 150

BG = (247, 243, 234)
PAPER = (255, 252, 245)
INK = (23, 34, 51)
MUTED = (101, 113, 127)
LINE = (216, 207, 192)
RED = (168, 64, 55)
RED_SOFT = (244, 225, 219)
TEAL = (23, 107, 109)
TEAL_SOFT = (224, 239, 235)
BLUE = (51, 92, 137)
BLUE_SOFT = (227, 236, 245)
GOLD = (183, 133, 47)
GOLD_SOFT = (246, 236, 211)
WHITE = (255, 255, 255)
NAVY = (20, 31, 48)

FONT_DIR = Path("/System/Library/Fonts/Supplemental")
REGULAR = FONT_DIR / "Arial.ttf"
BOLD = FONT_DIR / "Arial Bold.ttf"
BLACK = FONT_DIR / "Arial Black.ttf"


def font(size: int, bold: bool = False, black: bool = False) -> ImageFont.FreeTypeFont:
    path = BLACK if black and BLACK.exists() else BOLD if bold and BOLD.exists() else REGULAR
    return ImageFont.truetype(str(path), size=size)


F = {
    "eyebrow": font(22, True),
    "title": font(50, True),
    "h2": font(34, True),
    "h3": font(25, True),
    "body": font(24),
    "body_b": font(24, True),
    "small": font(19),
    "small_b": font(19, True),
    "tiny": font(16),
    "metric": font(40, True),
    "hero": font(76, True, True),
}


def rgb(page: Image.Image) -> Image.Image:
    return page.convert("RGB")


def csv(name: str) -> pd.DataFrame:
    return pd.read_csv(FORENSIC / name)


def money(value: object, digits: int = 0) -> str:
    try:
        v = float(value)
    except Exception:
        return str(value)
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v):,.{digits}f}"


def pips(value: object) -> str:
    try:
        v = float(value)
    except Exception:
        return str(value)
    return f"{v:+.1f}p"


def text_width(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> int:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0]


def wrap_lines(draw: ImageDraw.ImageDraw, text: object, fnt: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    chunks: list[str] = []
    for para in str(text).split("\n"):
        words = para.split()
        if not words:
            chunks.append("")
            continue
        line = ""
        for word in words:
            candidate = word if not line else f"{line} {word}"
            if text_width(draw, candidate, fnt) <= max_w:
                line = candidate
            else:
                if line:
                    chunks.append(line)
                line = word
        if line:
            chunks.append(line)
    return chunks


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    body: object,
    fnt: ImageFont.FreeTypeFont,
    fill=INK,
    max_w: int = 900,
    line_gap: int = 8,
) -> int:
    x, y = xy
    for line in wrap_lines(draw, body, fnt, max_w):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + line_gap
    return y


def new_page(title: str, eyebrow: str = "", section: str = "", number: int = 0) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle((60, 55, W - 60, H - 70), fill=PAPER, outline=LINE, width=2)
    if eyebrow:
        d.text((M, 145), eyebrow.upper(), font=F["eyebrow"], fill=RED)
    d.text((M, 195), title, font=F["title"], fill=INK)
    if section:
        tw = text_width(d, section.upper(), F["small_b"])
        d.text((W - M - tw, 150), section.upper(), font=F["small_b"], fill=MUTED)
    d.line((M, 300, W - M, 300), fill=LINE, width=2)
    if number:
        d.text((W - M - 20, H - 125), str(number), font=F["tiny"], fill=MUTED)
    return img, d


def card(
    d: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    label: str,
    value: str,
    note: str = "",
    color=INK,
    fill=WHITE,
):
    d.rounded_rectangle((x, y, x + w, y + h), radius=8, fill=fill, outline=LINE, width=2)
    d.text((x + 26, y + 24), label.upper(), font=F["tiny"], fill=MUTED)
    d.text((x + 26, y + 70), value, font=F["metric"], fill=color)
    if note:
        draw_text(d, (x + 26, y + h - 58), note, F["tiny"], fill=MUTED, max_w=w - 52, line_gap=4)


def label(d: ImageDraw.ImageDraw, x: int, y: int, body: str, color=RED):
    d.text((x, y), body.upper(), font=F["small_b"], fill=color)


def bullets(d: ImageDraw.ImageDraw, x: int, y: int, items: Iterable[str], max_w: int = 1160, color=INK) -> int:
    yy = y
    for item in items:
        d.ellipse((x, yy + 11, x + 10, yy + 21), fill=TEAL)
        yy = draw_text(d, (x + 28, yy), item, F["body"], fill=color, max_w=max_w - 28, line_gap=8) + 12
    return yy


def table(
    d: ImageDraw.ImageDraw,
    x: int,
    y: int,
    widths: Sequence[int],
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    row_h: int = 74,
    header_h: int = 54,
    fnt: ImageFont.FreeTypeFont | None = None,
):
    fnt = fnt or F["small"]
    total_w = sum(widths)
    d.rounded_rectangle((x, y, x + total_w, y + header_h + row_h * len(rows)), radius=8, fill=WHITE, outline=LINE, width=2)
    d.rectangle((x, y, x + total_w, y + header_h), fill=(239, 229, 215), outline=LINE)
    xx = x
    for i, head in enumerate(headers):
        d.text((xx + 14, y + 17), str(head).upper(), font=F["tiny"], fill=INK)
        xx += widths[i]
        if i < len(widths) - 1:
            d.line((xx, y, xx, y + header_h + row_h * len(rows)), fill=(232, 222, 207), width=1)
    for r, row in enumerate(rows):
        yy = y + header_h + r * row_h
        if r % 2:
            d.rectangle((x, yy, x + total_w, yy + row_h), fill=(251, 246, 238))
        d.line((x, yy, x + total_w, yy), fill=(232, 222, 207), width=1)
        xx = x
        for c, cell in enumerate(row):
            max_lines = max(1, (row_h - 18) // (fnt.size + 5))
            lines = wrap_lines(d, cell, fnt, widths[c] - 24)[:max_lines]
            if len(wrap_lines(d, cell, fnt, widths[c] - 24)) > max_lines and lines:
                lines[-1] = lines[-1][: max(0, len(lines[-1]) - 2)] + "..."
            ty = yy + 12
            for line in lines:
                d.text((xx + 12, ty), line, font=fnt, fill=INK)
                ty += fnt.size + 5
            xx += widths[c]


def flow(d: ImageDraw.ImageDraw, x: int, y: int, labels: Sequence[str], fills: Sequence[tuple[int, int, int]]):
    w, h, gap = 188, 84, 20
    for i, item in enumerate(labels):
        xx = x + i * (w + gap)
        d.rounded_rectangle((xx, y, xx + w, y + h), radius=10, fill=fills[i], outline=LINE, width=2)
        lines = wrap_lines(d, item, F["small_b"], w - 22)
        ty = y + h // 2 - (len(lines) * 24) // 2
        for line in lines:
            tw = text_width(d, line, F["small_b"])
            d.text((xx + (w - tw) // 2, ty), line, font=F["small_b"], fill=INK)
            ty += 24
        if i < len(labels) - 1:
            d.text((xx + w + 5, y + 26), ">", font=F["h2"], fill=MUTED)


def bar_chart(d: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, labels: Sequence[str], values: Sequence[float], title: str):
    d.text((x, y), title, font=F["small_b"], fill=INK)
    y += 42
    vals = [float(v) for v in values]
    max_abs = max(abs(v) for v in vals) or 1
    zero = x + w // 2
    d.line((zero, y, zero, y + h), fill=MUTED, width=2)
    row_h = h // len(vals)
    for i, (lab, val) in enumerate(zip(labels, vals)):
        yy = y + i * row_h + 8
        lab_s = str(lab).replace("_", " ")[:24]
        d.text((x, yy), lab_s, font=F["tiny"], fill=INK)
        length = int((abs(val) / max_abs) * (w * 0.35))
        bx = zero if val >= 0 else zero - length
        color = TEAL if val >= 0 else RED
        d.rounded_rectangle((bx, yy + 22, bx + length, yy + 42), radius=4, fill=color)
        d.text((zero + (12 if val >= 0 else -118), yy + 20), pips(val), font=F["tiny"], fill=INK)


def cover(num: int) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle((60, 55, W - 60, H - 70), fill=PAPER, outline=LINE, width=2)
    d.text((M, 250), "AUTO FILLMORE", font=F["hero"], fill=INK)
    d.text((M, 360), "Visual Brief v3", font=F["title"], fill=RED)
    draw_text(
        d,
        (M, 485),
        "A readable, scan-first explanation of the Phase 1-9 forensic audit, the v2 rebuild, and how Fillmore works.",
        F["h3"],
        fill=INK,
        max_w=1050,
        line_gap=10,
    )
    card(d, M, 720, 360, 190, "Audit sample", "241 trades", "Apr 16-May 1")
    card(d, M + 420, 720, 360, 190, "v1 result", "-$7,253", "-308.0p", RED)
    card(d, M + 840, 720, 360, 190, "v2 status", "paper-ready", "not 0.1x yet", TEAL)
    d.rectangle((60, 1540, W - 60, H - 70), fill=NAVY)
    d.text((M, 1655), "Designed to be read, not decoded", font=F["h2"], fill=WHITE)
    draw_text(
        d,
        (M, 1760),
        "This version uses large type, short evidence blocks, and one idea per page. It is deliberately less dense than the archival investigation book.",
        F["body"],
        fill=(226, 233, 241),
        max_w=1100,
    )
    d.text((M, 2040), "Generated from local Phase 1-9 artifacts and v2 rebuild docs", font=F["tiny"], fill=(199, 208, 220))
    return img


def page_how_to_read(num: int) -> Image.Image:
    img, d = new_page("How To Read This", "orientation", "reader guide", num)
    draw_text(d, (M, 380), "The old PDF was too dense and some interior pages rendered badly. This rebuild is a decision brief: short sections, obvious labels, and enough evidence to understand the system without fighting the formatting.", F["body_b"], max_w=1160)
    table(
        d,
        M,
        650,
        [210, 330, 660],
        ["Where", "Section", "Purpose"],
        [
            ["Pages 3-4", "Fillmore basics", "Plain-English model of how the system works."],
            ["Pages 5-8", "Investigation", "Root causes, loss shape, preserved edge, refuted theories."],
            ["Pages 9-13", "v2 rebuild", "Architecture, implementation, replay, readiness."],
            ["Pages 14-15", "Operator view", "UI expectations, first-hour checklist, launch boundary."],
        ],
        row_h=95,
        fnt=F["small"],
    )
    label(d, M, 1160, "Legend")
    bullets(
        d,
        M,
        1210,
        [
            "Red means damage, vetoes, or blocked paths.",
            "Teal means protection, passed verification, or preserved edge.",
            "Gold means operational caution: okay for supervised Stage 1 paper, not scaled testing.",
        ],
    )
    return img


def page_what_is(num: int) -> Image.Image:
    img, d = new_page("What Fillmore Is", "plain-English overview", "system", num)
    draw_text(d, (M, 380), "Fillmore is an autonomous USDJPY trading agent. It watches live market data, detects a setup, sends a market snapshot to an LLM, and decides whether to place or skip.", F["body_b"], max_w=1160)
    flow(d, M, 610, ["Market tick", "Code gate", "Snapshot", "LLM judgment", "Validation", "Audit row"], [GOLD_SOFT, GOLD_SOFT, WHITE, WHITE, TEAL_SOFT, TEAL_SOFT])
    table(
        d,
        M,
        860,
        [250, 950],
        ["Term", "Meaning"],
        [
            ["Gate", "A deterministic setup detector: critical level reaction, momentum continuation, or mean reversion."],
            ["Snapshot", "The market/account/context packet sent to the LLM and stored for audit."],
            ["LLM", "The model that judges evidence. In v2 it cannot size trades or manage exits."],
            ["Validator", "Deterministic checks that can override a model place decision."],
            ["Stage 1", "Paper-live validation: real ticks, audit rows, no broker order-send through v2."],
        ],
        row_h=95,
        fnt=F["small"],
    )
    draw_text(d, (M, 1530), "The rebuild does not trust the model more. It gives the model less authority.", F["h3"], fill=BLUE, max_w=1160)
    return img


def page_v1_v2(num: int) -> Image.Image:
    img, d = new_page("v1 vs v2", "what changed", "comparison", num)
    table(
        d,
        M - 20,
        380,
        [230, 455, 555],
        ["Layer", "v1", "v2"],
        [
            ["Admission", "LLM could talk itself into marginal setups.", "Pre-vetoes block toxic setup cells before the LLM."],
            ["Reasoning", "Caveats often became decoration.", "Caveats must be resolved with evidence or the trade skips."],
            ["Sizing", "LLM-influenced and edge-blind.", "Deterministic risk function with hard cap."],
            ["Exits", "Prompt language could bias hold behavior.", "Stops, locks, and time rules are deterministic."],
            ["Telemetry", "Placed-trade audit was useful but incomplete.", "Prompt, snapshot, gates, sizing, vetoes, validators, and state are logged."],
            ["Launch", "Original autonomous engine.", "Parallel v2 engine, default off, Stage 1 paper-guarded."],
        ],
        row_h=120,
        fnt=F["small"],
    )
    draw_text(d, (M, 1425), "v2 is not a better prompt pasted on top of v1. It is a control system around the prompt.", F["h3"], fill=TEAL, max_w=1160)
    return img


def page_phase_map(num: int) -> Image.Image:
    img, d = new_page("Phase 1-9 Audit Map", "the full investigation arc", "forensic audit", num)
    phases = [
        ("1", "Baseline", "241 trades; 53.1% WR still lost -308.0p / -$7,253."),
        ("2", "Gates", "Sell-CLR and large-size damage isolated."),
        ("3", "Snapshot", "Core buy/sell data symmetric; key telemetry missing."),
        ("4", "Reasoning", "Sell-side caveat-template collapse confirmed."),
        ("5", "Sizing", "Sizing was random/edge-blind; -$5,187 amplification."),
        ("6", "Lifecycle", "Entry-generated damage dominated; runner language harmed."),
        ("7", "Interactions", "V1+V2 veto floor recovered 90.4% pips / 78.4% USD."),
        ("8", "Synthesis", "Locked root causes, preserved edge, refuted hypotheses."),
        ("9", "Blueprint", "Specified v2: deterministic shell, constrained LLM, telemetry-first."),
    ]
    x_positions = [M, M + 650]
    y0 = 390
    for i, (n, title, detail) in enumerate(phases):
        x = x_positions[0 if i < 5 else 1]
        y = y0 + (i % 5) * 260
        color = RED if i < 7 else TEAL
        d.rounded_rectangle((x, y, x + 560, y + 190), radius=10, fill=WHITE, outline=LINE, width=2)
        d.rounded_rectangle((x, y, x + 86, y + 190), radius=10, fill=color)
        d.text((x + 30, y + 65), n, font=F["h2"], fill=WHITE)
        d.text((x + 112, y + 28), title, font=F["h3"], fill=INK)
        draw_text(d, (x + 112, y + 78), detail, F["small"], fill=MUTED, max_w=410, line_gap=5)
    return img


def page_loss_shape(num: int) -> Image.Image:
    img, d = new_page("The Loss Shape", "why 53% winners still failed", "phases 1-2", num)
    card(d, M, 365, 270, 170, "Win rate", "53.1%", "not enough")
    card(d, M + 310, 365, 270, 170, "Avg winner", "+5.39p", "+$88.06", TEAL)
    card(d, M + 620, 365, 270, 170, "Avg loser", "-8.83p", "-$163.94", RED)
    card(d, M + 930, 365, 270, 170, "Net", "-308.0p", "-$7,253", RED)
    side = csv("phase1_by_side.csv")
    gate = csv("phase1_by_trigger_family.csv").sort_values("net_pips").head(5)
    bar_chart(d, M, 670, 540, 310, side["side"], side["net_pips"], "By side")
    bar_chart(d, M + 650, 670, 540, 420, gate["trigger_family"], gate["net_pips"], "Worst gate families")
    draw_text(d, (M, 1270), "The important shape was asymmetry: winners were smaller, losers were larger, and variable sizing made the dollar losses much worse.", F["body_b"], fill=INK, max_w=1160)
    bullets(d, M, 1410, [
        "Sell-side aggregate damage was not explained by USDJPY macro drift.",
        "Critical Level Reaction was the largest gate-level damage center.",
        "Momentum Continuation had pip edge but lost dollars because sizing was wrong.",
    ])
    return img


def page_root_causes(num: int) -> Image.Image:
    img, d = new_page("Root Causes", "causes, not symptoms", "phase 8", num)
    rows = [
        ["1", "Sell-side caveat-template collapse", "LLM reused weak short templates and laundered caveats."],
        ["2", "Random / edge-blind sizing", "-$5,187.24 of dollar damage came from variable sizing."],
        ["3", "Negative admission expectancy", "At uniform 1 lot, same placed trades still lost -$2,065.99."],
        ["4", "Entry-generated level failure", "25 entry-failure proxy trades lost -305.1p."],
        ["5", "Telemetry gaps", "No skip outcomes, path-time MAE/MFE, full context, or exit replay."],
    ]
    table(d, M - 20, 390, [90, 430, 720], ["Rank", "Cause", "Evidence"], rows, row_h=125, fnt=F["small"])
    draw_text(d, (M, 1130), "Shortest honest version: v1 let weak reasoning become action, then let edge-blind sizing amplify that action.", F["h3"], fill=RED, max_w=1160)
    draw_text(d, (M, 1280), "The model was useful enough to produce some good cells, but not disciplined enough to govern itself.", F["body_b"], fill=INK, max_w=1160)
    return img


def page_preserved_refuted(num: int) -> Image.Image:
    img, d = new_page("Preserved Edge + Refuted Theories", "what the rebuild must protect", "phase 8", num)
    protected = csv("phase9_replay_protected_edge_check.csv")
    rows = []
    for _, r in protected.iterrows():
        rows.append([
            str(r["protected_cell"]),
            str(int(r["survivor_n"])),
            pips(r["survivor_pips"]),
            money(r["survivor_usd_original_size"], 0),
            str(int(r["blocked_by_phase9"])),
        ])
    table(d, M - 20, 380, [540, 90, 160, 170, 130], ["Protected cell", "N", "Pips", "USD", "Blocked"], rows, row_h=95, fnt=F["tiny"])
    label(d, M, 780, "Refuted")
    bullets(d, M, 835, [
        "Sells did not fail because macro drift punished shorts.",
        "Sell-CLR snapshots were not thinner than buy-CLR snapshots.",
        "Sizing was not anti-Kelly; it was random / edge-blind.",
        "The 8+ lot CLR cell was not real edge; it was compositional.",
        "Caveat-language losers were not mainly held too long at exit.",
    ])
    draw_text(d, (M, 1430), "The rebuild protects the good buy-CLR cells while attacking the sell/caveat and sizing failures.", F["h3"], fill=TEAL, max_w=1160)
    return img


def page_v2_arch(num: int) -> Image.Image:
    img, d = new_page("How v2 Works", "deterministic shell, constrained model", "rebuild", num)
    flow(d, M - 30, 390, ["Telemetry", "Gate", "Pre-veto", "LLM", "Validator", "Sizing"], [BLUE_SOFT, GOLD_SOFT, RED_SOFT, WHITE, TEAL_SOFT, TEAL_SOFT])
    table(
        d,
        M - 20,
        620,
        [240, 1000],
        ["Layer", "v2 behavior"],
        [
            ["Telemetry", "Captures exposure, pip value, risk-after-fill, rolling P&L, level packets, prompt/context, and gate candidates."],
            ["Gate", "Keeps the three primary families; kills non-primary gate contamination by default."],
            ["Pre-veto", "Blocks toxic setup cells before spending an LLM call."],
            ["LLM", "Returns strict JSON. No sizing authority. No exit authority."],
            ["Validator", "Overrides place to skip when caveats, level claims, sell burden, or loss asymmetry fail."],
            ["Sizing / exit", "Deterministic lots, hard caps, deterministic stop/profit/time rules."],
        ],
        row_h=105,
        fnt=F["small"],
    )
    draw_text(d, (M, 1430), "The LLM can argue from evidence; it cannot decide size, widen exits, or gloss over caveats.", F["h3"], fill=BLUE, max_w=1160)
    return img


def page_rebuild_steps(num: int) -> Image.Image:
    img, d = new_page("What Was Built", "the nine implementation steps", "v2 rebuild", num)
    rows = [
        ["1", "Telemetry + snapshot", "v2 schema, state isolation, schema hash, blocking telemetry."],
        ["2", "Validators", "Strict JSON parse and five post-decision validators."],
        ["3", "Pre-vetoes", "V1 sell caveat-template and V2 mixed-overlap checks before LLM."],
        ["4", "Sizing", "Pure deterministic function in core/fillmore_v2_sizing.py."],
        ["5", "Gates", "Side-asymmetric CLR, conditional momentum, mean reversion retained."],
        ["6", "LLM/orchestrator", "Strict prompt/output contract and end-to-end run_decision path."],
        ["7", "Exit/tripwires", "Deterministic exit layer and rollout tripwire functions."],
        ["8", "Replay", "Pass A/B/C verification against forensic corpus."],
        ["9", "Live bridge", "v1/v2 engine flag, paper guard, v2 audit rows, readiness docs."],
    ]
    table(d, M - 20, 370, [80, 350, 810], ["Step", "Component", "What landed"], rows, row_h=102, fnt=F["tiny"])
    draw_text(d, (M, 1425), "v2 was built in parallel. v1 remains default; v2 activates only through the engine flag and uses isolated v2 state.", F["body_b"], fill=TEAL, max_w=1160)
    return img


def page_replay(num: int) -> Image.Image:
    img, d = new_page("Replay Verification", "did v2 beat the floor?", "step 8", num)
    card(d, M, 365, 360, 170, "Pip recovery", "+282.5p", "floor +278.4p", TEAL)
    card(d, M + 420, 365, 360, 170, "USD recovery", "+$6,370.77", "floor +$5,684.56", TEAL)
    card(d, M + 840, 365, 360, 170, "Cap-to-4", "+$6,599.37", "beat target", TEAL)
    table(
        d,
        M,
        680,
        [310, 190, 700],
        ["Check", "Status", "Interpretation"],
        [
            ["Recovery floors", "PASS", "Pips and USD met or exceeded the binding floor."],
            ["Protected cells", "PASS", "All three remained positive with exact parity to Phase 8."],
            ["False positives", "FAIL", "Blocked trades/winners exceeded ceiling due to legacy rationale parser."],
            ["Decision", "Stage 1", "User chose paper validation as the real forward gate."],
        ],
        row_h=115,
        fnt=F["small"],
    )
    draw_text(d, (M, 1250), "The false-positive breach traces to a corpus-only adapter that reverse-engineers old free-text rationales. Production v2 emits structured JSON, so Stage 1 paper data becomes the binding validation sample.", F["body_b"], fill=INK, max_w=1160)
    return img


def page_readiness(num: int) -> Image.Image:
    img, d = new_page("Live Testing Readiness", "safe today vs blocked later", "stage 1", num)
    card(d, M, 365, 360, 170, "Engine", "v1 default", "v2 opt-in")
    card(d, M + 420, 365, 360, 170, "v2 stage", "paper-live", "real ticks", GOLD)
    card(d, M + 840, 365, 360, 170, "Broker send", "blocked", "audit rows only", TEAL)
    label(d, M, 650, "Ready for Stage 1 paper-live")
    bullets(d, M, 700, [
        "v2 dispatch path refuses non-paper stage unless future code explicitly opts in.",
        "v2 does not call broker order-send functions in Stage 1.",
        "Rows persist with engine_version='v2' and replay telemetry.",
        "v2 state is isolated from v1 in runtime_state_fillmore_v2.json.",
    ])
    label(d, M, 1110, "Not cleared before 0.1x")
    bullets(d, M, 1160, [
        "Real side-normalized level-quality builder.",
        "Better macro bias and catalyst classification.",
        "Skip-forward outcome capture with at least 98% coverage for T5.",
        "Stage progression reporting and 50-close Stage 1 paper report.",
        "Exit-layer live replay logs if v2 paper decisions become actual paper opens/closes.",
    ], color=RED)
    return img


def page_ui(num: int) -> Image.Image:
    img, d = new_page("What The UI Should Show", "matching the v2 system", "operator UI", num)
    table(
        d,
        M - 20,
        380,
        [330, 910],
        ["UI area", "What it must communicate"],
        [
            ["Engine selector", "v1 legacy / v2 Stage 1. v1 remains default."],
            ["Paper guard", "v2 clearly says Stage 1 paper and no broker order-send."],
            ["First-tick audit", "Latest v2 row, missing telemetry, gate, decision, parse, pre-vetoes, validators, deterministic lots."],
            ["Deterministic sizing", "Old LLM lot controls hidden under v2; v2 shows authority and hard cap."],
            ["Before 0.1x", "The blocker checklist stays visible so paper-live does not get confused with scaled testing."],
            ["Rollback", "Set engine back to v1; v2 state remains isolated."],
        ],
        row_h=130,
        fnt=F["small"],
    )
    draw_text(d, (M, 1270), "The UI's job is not to make v2 look finished. Its job is to make the operating boundary impossible to miss.", F["h3"], fill=BLUE, max_w=1160)
    return img


def page_first_hour(num: int) -> Image.Image:
    img, d = new_page("First Hour Checklist", "when v2 is flipped", "runbook", num)
    label(d, M, 380, "Must be true")
    bullets(d, M, 430, [
        "New rows appear in ai_suggestions with engine_version='v2'.",
        "snapshot_blocking_strikes stays at 0 when inputs are healthy.",
        "No lots above 4.",
        "Any sell-side placement has validator/pre-veto audit metadata.",
        "Every placed row has pip_value_per_lot, risk_after_fill_usd, rendered prompt/context, gate candidates, and deterministic sizing inputs.",
    ])
    label(d, M, 980, "Rollback")
    draw_text(d, (M, 1030), "Set the engine flag back to v1, then confirm no new engine_version='v2' rows appear after the rollback timestamp. v2 halt/strike state remains isolated.", F["body"], fill=INK, max_w=1160)
    label(d, M, 1230, "Known non-launch debt")
    draw_text(d, (M, 1280), "Full suite has one unrelated pre-existing failure in tests/test_phase3_additive_runtime.py. It is documented as non-v2 launch debt and should not confuse Stage 1 launch/no-launch decisions.", F["small"], fill=MUTED, max_w=1160)
    return img


def page_bottom_line(num: int) -> Image.Image:
    img, d = new_page("The Bottom Line", "what changed and why it matters", "summary", num)
    table(
        d,
        M - 20,
        390,
        [300, 940],
        ["Question", "Answer"],
        [
            ["v1 problem", "Weak reasoning could become action, and action could become oversized exposure."],
            ["v2 answer", "Constrain the model with deterministic gates, vetoes, validators, sizing, exits, and telemetry."],
            ["Current status", "Ready for supervised Stage 1 paper-live testing, not scaled 0.1x testing."],
            ["What proves progress", "50 closed paper decisions with net pips >= 0, simulated PF >= 0.9, zero missing blocking telemetry, no protected-cell regression, no stop widening."],
        ],
        row_h=135,
        fnt=F["small"],
    )
    draw_text(d, (M, 1075), "The system is now designed to learn from itself. If Stage 1 fails, the next audit should be able to say exactly which deterministic layer, model output, telemetry field, or market regime failed.", F["h3"], fill=TEAL, max_w=1160)
    label(d, M, 1440, "Primary sources")
    draw_text(d, (M, 1490), "PHASE1_BASELINE.md through PHASE9_OVERHAUL_BLUEPRINT.md; docs/fillmore_v2/CHANGELOG.md; docs/fillmore_v2/step8_replay_results.md; docs/fillmore_v2/live_testing_readiness_20260502.md.", F["small"], fill=MUTED, max_w=1160)
    return img


PAGE_BUILDERS = [
    cover,
    page_how_to_read,
    page_what_is,
    page_v1_v2,
    page_phase_map,
    page_loss_shape,
    page_root_causes,
    page_preserved_refuted,
    page_v2_arch,
    page_rebuild_steps,
    page_replay,
    page_readiness,
    page_ui,
    page_first_hour,
    page_bottom_line,
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    pages: list[Image.Image] = []
    for idx, builder in enumerate(PAGE_BUILDERS, start=1):
        page = rgb(builder(idx))
        # The background is intentionally light. This catches truly blank pages
        # while allowing intentionally spacious cover pages.
        bbox = Image.eval(page, lambda px: 255 - px).getbbox()
        if bbox is None:
            raise RuntimeError(f"blank page generated: {idx}")
        pages.append(page)
        page.save(PREVIEW_DIR / f"page_{idx:02d}.png")
    pages[0].save(OUT, "PDF", resolution=150, save_all=True, append_images=pages[1:])
    print(OUT)
    print(PREVIEW_DIR)


if __name__ == "__main__":
    main()
