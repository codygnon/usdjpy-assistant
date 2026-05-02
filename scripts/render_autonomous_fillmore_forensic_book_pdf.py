#!/usr/bin/env python3
"""Render the full Auto Fillmore forensic audit, Phases 1-9, as a PDF book.

The source reports are markdown artifacts. This renderer keeps the full text
of the phase reports, paginates it into a print-friendly PDF, and adds a cover,
table of contents, running headers, and an evidence-source manifest.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
FORENSIC_DIR = ROOT / "research_out" / "autonomous_fillmore_forensic_20260501"
OUT_PDF = FORENSIC_DIR / "Auto_Fillmore_Forensic_Audit_Phases_1_to_9_Book.pdf"

PAGE_W = 8.5
PAGE_H = 11.0

BG = "#F7F4EE"
PAPER = "#FFFDF8"
INK = "#182230"
MUTED = "#69717D"
GRID = "#D9D2C3"
RED = "#A43D35"
RED_DARK = "#722B27"
TEAL = "#176B6D"
GOLD = "#B9852F"
BLUE = "#355C8C"
CODE_BG = "#F0E8DA"


PHASE_SOURCES = [
    ("Phase 1 - Baseline", "PHASE1_BASELINE.md"),
    ("Phase 2 - Code Gate Audit", "PHASE2_GATE_AUDIT.md"),
    ("Phase 3 - Snapshot Quality Audit", "PHASE3_SNAPSHOT_AUDIT.md"),
    ("Phase 4 - LLM Reasoning Forensics", "PHASE4_REASONING_FORENSICS.md"),
    ("Phase 5 - Sizing Logic Audit", "PHASE5_SIZING_AUDIT.md"),
    ("Phase 6 - Trade Lifecycle Analysis", "PHASE6_LIFECYCLE_ANALYSIS.md"),
    ("Phase 7 - Interaction Effects", "PHASE7_INTERACTION_EFFECTS.md"),
    ("Phase 8 - Root Cause Synthesis", "PHASE8_SYNTHESIS.md"),
    ("Phase 9 - Overhaul Blueprint", "PHASE9_OVERHAUL_BLUEPRINT.md"),
]


@dataclass
class PageState:
    pdf: PdfPages
    phase: str
    page_no: int = 0
    fig: plt.Figure | None = None
    y: float = 0.0


def strip_inline_md(text: str) -> str:
    text = text.replace("`", "")
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"_(.*?)_", r"\1", text)
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1 (\2)", text)
    return text


def mpl_safe(text: str) -> str:
    """Avoid accidental matplotlib mathtext parsing in financial text."""
    return text.replace("$", r"\$")


def add_footer(fig: plt.Figure, phase: str, page_no: int) -> None:
    fig.text(0.06, 0.965, "Auto Fillmore Forensic Audit", fontsize=8, color=MUTED, ha="left")
    fig.text(0.94, 0.965, mpl_safe(phase), fontsize=8, color=MUTED, ha="right")
    fig.add_artist(plt.Line2D([0.06, 0.94], [0.945, 0.945], color=GRID, linewidth=0.8))
    fig.add_artist(plt.Line2D([0.06, 0.94], [0.055, 0.055], color=GRID, linewidth=0.8))
    fig.text(0.94, 0.032, f"{page_no}", fontsize=8, color=MUTED, ha="right")
    fig.text(0.06, 0.032, "USDJPY autonomous system forensic record", fontsize=8, color=MUTED, ha="left")


def new_text_page(state: PageState, phase: str | None = None) -> None:
    if state.fig is not None:
        state.pdf.savefig(state.fig)
        plt.close(state.fig)
    if phase is not None:
        state.phase = phase
    state.page_no += 1
    state.fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor=PAPER)
    add_footer(state.fig, state.phase, state.page_no)
    state.y = 0.91


def ensure_space(state: PageState, needed: float) -> None:
    if state.y - needed < 0.085:
        new_text_page(state)


def draw_text(
    state: PageState,
    text: str,
    *,
    size: float = 9.5,
    color: str = INK,
    weight: str = "normal",
    family: str = "DejaVu Sans",
    indent: float = 0.0,
    line_height: float = 0.018,
) -> None:
    if not text:
        state.y -= line_height * 0.45
        return
    fig = state.fig
    assert fig is not None
    fig.text(0.075 + indent, state.y, mpl_safe(text), fontsize=size, color=color, fontweight=weight, family=family, ha="left", va="top")
    state.y -= line_height


def draw_wrapped(
    state: PageState,
    text: str,
    *,
    width: int = 108,
    size: float = 9.2,
    color: str = INK,
    weight: str = "normal",
    indent: float = 0.0,
    line_height: float = 0.0175,
) -> None:
    text = strip_inline_md(text).strip()
    if not text:
        state.y -= 0.008
        return
    lines = textwrap.wrap(text, width=max(20, width - int(indent * 120)), break_long_words=False, break_on_hyphens=False)
    ensure_space(state, line_height * max(1, len(lines)) + 0.008)
    for line in lines:
        draw_text(state, line, size=size, color=color, weight=weight, indent=indent, line_height=line_height)
    state.y -= 0.004


def draw_block(state: PageState, lines: list[str], *, title: str | None = None, max_width: int = 132) -> None:
    if title:
        draw_wrapped(state, title, size=8.4, color=MUTED, weight="bold")
    wrapped_lines: list[str] = []
    for raw in lines:
        raw = raw.rstrip("\n")
        chunks = textwrap.wrap(raw, width=max_width, replace_whitespace=False, drop_whitespace=False) or [""]
        wrapped_lines.extend(chunks)
    idx = 0
    while idx < len(wrapped_lines):
        available = max(5, int((state.y - 0.09) / 0.0135))
        chunk = wrapped_lines[idx : idx + available]
        ensure_space(state, 0.0135 * len(chunk) + 0.02)
        fig = state.fig
        assert fig is not None
        top = state.y + 0.006
        height = 0.0135 * len(chunk) + 0.018
        rect = plt.Rectangle((0.065, top - height), 0.87, height, transform=fig.transFigure, facecolor=CODE_BG, edgecolor=GRID, linewidth=0.6)
        fig.add_artist(rect)
        y = state.y - 0.004
        for line in chunk:
            fig.text(0.078, y, mpl_safe(line), fontsize=6.2, color=INK, family="DejaVu Sans Mono", ha="left", va="top")
            y -= 0.0135
        state.y = top - height - 0.012
        idx += len(chunk)
        if idx < len(wrapped_lines):
            new_text_page(state)


def render_cover(pdf: PdfPages, source_stats: list[tuple[str, int]]) -> None:
    fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor=BG)
    fig.add_artist(plt.Rectangle((0, 0), 1, 1, transform=fig.transFigure, color=BG))
    fig.add_artist(plt.Rectangle((0.06, 0.08), 0.88, 0.84, transform=fig.transFigure, facecolor=PAPER, edgecolor=GRID, linewidth=1.2))
    fig.text(0.10, 0.82, "AUTO FILLMORE", fontsize=34, fontweight="bold", color=INK, ha="left")
    fig.text(0.10, 0.765, "Forensic Audit Book", fontsize=27, fontweight="bold", color=RED_DARK, ha="left")
    fig.text(0.10, 0.718, "Phases 1-9 | Apr 16-May 1 live run | USDJPY autonomous system", fontsize=12, color=MUTED, ha="left")
    fig.add_artist(plt.Line2D([0.10, 0.90], [0.675, 0.675], color=GRID, linewidth=1.3))
    thesis = (
        "A complete evidence trail: baseline performance, gate behavior, snapshot quality, LLM reasoning, sizing, lifecycle, "
        "interaction effects, root-cause synthesis, and the final overhaul blueprint."
    )
    fig.text(0.10, 0.61, mpl_safe("\n".join(textwrap.wrap(thesis, 72))), fontsize=14, color=INK, ha="left", va="top", linespacing=1.35)

    cards = [
        (0.10, 0.435, "Closed trades", "241", RED),
        (0.52, 0.435, "Net result", "-308.0p / -$7,253", RED),
        (0.10, 0.315, "Primary failure", "Sell caveat-template collapse", RED_DARK),
        (0.52, 0.315, "Blueprint replay", "+324.3p / +$6,420.90", TEAL),
    ]
    for x, y, label, value, color in cards:
        fig.add_artist(plt.Rectangle((x, y), 0.36, 0.092, transform=fig.transFigure, facecolor="#FFFFFF", edgecolor=GRID, linewidth=0.8))
        fig.text(x + 0.014, y + 0.062, label.upper(), fontsize=7.3, color=MUTED, fontweight="bold")
        fig.text(x + 0.014, y + 0.023, mpl_safe(value), fontsize=11.2, color=color, fontweight="bold")

    fig.text(0.10, 0.252, "Included phase reports", fontsize=12, color=INK, fontweight="bold")
    for idx, (name, lines) in enumerate(source_stats):
        col_x = 0.12 if idx < 5 else 0.53
        row_y = 0.220 - (idx % 5) * 0.030
        fig.text(col_x, row_y, mpl_safe(f"{name}  |  {lines:,} lines"), fontsize=7.8, color=INK, ha="left")
    fig.text(0.10, 0.075, "Generated from local markdown artifacts. Tables and code blocks are preserved as monospaced source blocks.", fontsize=8.2, color=MUTED)
    pdf.savefig(fig)
    plt.close(fig)


def render_toc(pdf: PdfPages, source_stats: list[tuple[str, int]]) -> None:
    fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor=PAPER)
    fig.text(0.08, 0.91, "Table Of Contents", fontsize=24, fontweight="bold", color=INK)
    fig.text(0.08, 0.875, "The book is organized by the original investigation phases. Page numbers begin after the cover and contents.", fontsize=9, color=MUTED)
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.855, 0.855], color=GRID, linewidth=1.0))
    y = 0.81
    for idx, (name, lines) in enumerate(source_stats, start=1):
        color = RED_DARK if idx in (4, 8, 9) else INK
        fig.text(0.10, y, mpl_safe(f"{idx}. {name}"), fontsize=12.5, color=color, fontweight="bold")
        fig.text(0.82, y, f"{lines:,} lines", fontsize=9, color=MUTED, ha="right")
        y -= 0.043
    note = (
        "Reading guidance: Phases 1-7 build the factual record. Phase 8 is the causal synthesis. "
        "Phase 9 is the deployable redesign blueprint and retroactive replay gate."
    )
    fig.text(0.10, 0.30, mpl_safe("\n".join(textwrap.wrap(note, 82))), fontsize=10.5, color=INK, va="top", linespacing=1.35)
    fig.text(0.10, 0.14, "Design system", fontsize=11, fontweight="bold", color=INK)
    fig.text(0.10, 0.11, "Red = damage / failure. Teal = preserved edge or passing replay. Gold = caution / secondary effect. Blue = system design.", fontsize=8.8, color=MUTED)
    pdf.savefig(fig)
    plt.close(fig)


def render_phase_divider(state: PageState, phase_name: str, source_file: str, line_count: int) -> None:
    new_text_page(state, phase_name)
    fig = state.fig
    assert fig is not None
    fig.add_artist(plt.Rectangle((0.06, 0.18), 0.88, 0.62, transform=fig.transFigure, facecolor=BG, edgecolor=GRID, linewidth=1.0))
    fig.text(0.10, 0.68, mpl_safe(phase_name), fontsize=26, color=INK, fontweight="bold", ha="left")
    fig.text(0.10, 0.63, mpl_safe(source_file), fontsize=10, color=MUTED, ha="left")
    fig.add_artist(plt.Line2D([0.10, 0.88], [0.59, 0.59], color=GRID, linewidth=1.1))
    fig.text(0.10, 0.53, f"Full source text included: {line_count:,} lines", fontsize=14, color=RED_DARK, fontweight="bold", ha="left")
    fig.text(
        0.10,
        0.45,
        mpl_safe("\n".join(textwrap.wrap("The following pages preserve the markdown report content, including tables, rule lists, evidence gaps, replay metrics, and final verdicts.", 78))),
        fontsize=11,
        color=INK,
        ha="left",
        va="top",
        linespacing=1.35,
    )
    state.y = 0.14


def render_markdown(state: PageState, markdown: str) -> None:
    lines = markdown.splitlines()
    i = 0
    in_fence = False
    fence_lines: list[str] = []
    while i < len(lines):
        line = lines[i].rstrip()
        if line.startswith("```"):
            if in_fence:
                draw_block(state, fence_lines, title="code / diagram block")
                fence_lines = []
                in_fence = False
            else:
                in_fence = True
                fence_lines = []
            i += 1
            continue
        if in_fence:
            fence_lines.append(line)
            i += 1
            continue
        if not line.strip():
            state.y -= 0.006
            if state.y < 0.085:
                new_text_page(state)
            i += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i].rstrip())
                i += 1
            draw_block(state, table_lines, title="table")
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            level = len(heading.group(1))
            text = strip_inline_md(heading.group(2)).strip()
            if level == 1:
                ensure_space(state, 0.08)
                state.y -= 0.012
                draw_wrapped(state, text, width=60, size=17, color=RED_DARK, weight="bold", line_height=0.032)
                fig = state.fig
                assert fig is not None
                fig.add_artist(plt.Line2D([0.075, 0.925], [state.y + 0.006, state.y + 0.006], color=GRID, linewidth=0.8))
                state.y -= 0.014
            elif level == 2:
                ensure_space(state, 0.06)
                draw_wrapped(state, text, width=70, size=13, color=INK, weight="bold", line_height=0.025)
            elif level == 3:
                ensure_space(state, 0.045)
                draw_wrapped(state, text, width=82, size=11, color=BLUE, weight="bold", line_height=0.021)
            else:
                draw_wrapped(state, text, width=96, size=9.6, color=INK, weight="bold", line_height=0.018)
            i += 1
            continue
        if line.lstrip().startswith(("- ", "* ")):
            text = line.lstrip()[2:]
            draw_wrapped(state, "• " + text, width=100, size=8.8, indent=0.025, line_height=0.0165)
            i += 1
            continue
        if re.match(r"^\s*\d+\.\s+", line):
            draw_wrapped(state, line.strip(), width=102, size=8.8, indent=0.02, line_height=0.0165)
            i += 1
            continue
        if line.startswith("    ") or line.startswith("\t"):
            code_lines = []
            while i < len(lines) and (lines[i].startswith("    ") or lines[i].startswith("\t")):
                code_lines.append(lines[i])
                i += 1
            draw_block(state, code_lines, title="indented block")
            continue
        draw_wrapped(state, line, width=106, size=9.0, line_height=0.017)
        i += 1
    if in_fence and fence_lines:
        draw_block(state, fence_lines, title="code / diagram block")


def main() -> None:
    source_stats: list[tuple[str, int]] = []
    sources: list[tuple[str, str, str, int]] = []
    for phase_name, filename in PHASE_SOURCES:
        path = FORENSIC_DIR / filename
        text = path.read_text(encoding="utf-8")
        line_count = len(text.splitlines())
        source_stats.append((phase_name, line_count))
        sources.append((phase_name, filename, text, line_count))

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        render_cover(pdf, source_stats)
        render_toc(pdf, source_stats)
        state = PageState(pdf=pdf, phase="Front Matter", page_no=0)
        for phase_name, filename, text, line_count in sources:
            render_phase_divider(state, phase_name, filename, line_count)
            render_markdown(state, text)
        if state.fig is not None:
            state.pdf.savefig(state.fig)
            plt.close(state.fig)

    print(OUT_PDF)


if __name__ == "__main__":
    main()
