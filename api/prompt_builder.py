from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from api import autonomous_performance, suggestion_tracker
from api.ai_trading_chat import (
    autonomous_system_prompt_from_context,
    build_trade_suggestion_news_block,
    system_prompt_from_context,
)


PromptMode = Literal["chat", "suggest_manual", "suggest_autonomous", "thesis_monitor"]


def _hash_text(text: str | None) -> str | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _hash_json(payload: Any) -> str:
    try:
        raw = json.dumps(payload, sort_keys=True, default=str)
    except Exception:
        raw = repr(payload)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _word_count(text: str) -> int:
    return len(str(text or "").split())


def _fit_aux_memory_blocks(
    blocks: list[tuple[str, str, bool]],
    *,
    budget_words: int = 2400,
) -> str:
    budget_words = max(100, int(budget_words))
    selected: list[str] = []
    used = 0
    optional: list[tuple[str, str]] = []
    omitted: list[str] = []

    for name, text, required in blocks:
        block = str(text or "").strip()
        if not block:
            continue
        words = _word_count(block)
        if required:
            selected.append(block)
            used += words
        else:
            optional.append((name, block))

    remaining = max(0, budget_words - used)
    for name, block in optional:
        words = _word_count(block)
        if words <= remaining:
            selected.append(block)
            remaining -= words
        else:
            omitted.append(name)

    if omitted:
        selected.append(
            "=== PROMPT MEMORY BUDGET NOTE ===\n"
            "Some lower-priority history blocks were omitted to keep the prompt focused: "
            + ", ".join(omitted)
            + "."
        )
    return "\n\n".join(selected)


@dataclass
class FillmorePromptAssembly:
    system: str
    user: str
    model: str
    mode: PromptMode
    prompt_version: str
    context_hash: str
    prompt_hash: str
    news_block_hash: str | None = None
    learning_block_hash: str | None = None
    token_budget: int | None = None


class PromptBuilder:
    def __init__(
        self,
        *,
        profile: Any,
        profile_name: str,
        ctx: dict[str, Any],
        model: str,
        mode: PromptMode,
        base_system: str,
    ) -> None:
        self.profile = profile
        self.profile_name = profile_name
        self.ctx = ctx
        self.model = model
        self.mode = mode
        self.system_parts: list[str] = [str(base_system or "").strip()]
        self.news_block_hash: str | None = None
        self.learning_block_hash: str | None = None

    def append_system_block(
        self,
        block: str,
        *,
        kind: Literal["news", "learning"] | None = None,
    ) -> "PromptBuilder":
        text = str(block or "").strip()
        if not text:
            return self
        self.system_parts.append(text)
        hashed = _hash_text(text)
        if kind == "news":
            self.news_block_hash = hashed
        elif kind == "learning":
            self.learning_block_hash = hashed
        return self

    @classmethod
    def for_manual_suggest(
        cls,
        *,
        profile: Any,
        profile_name: str,
        ctx: dict[str, Any],
        model: str,
    ) -> "PromptBuilder":
        base_system = system_prompt_from_context(ctx, model)
        return cls(
            profile=profile,
            profile_name=profile_name,
            ctx=ctx,
            model=model,
            mode="suggest_manual",
            base_system=base_system,
        )

    @classmethod
    def for_autonomous_suggest(
        cls,
        *,
        profile: Any,
        profile_name: str,
        ctx: dict[str, Any],
        model: str,
        autonomous_config: dict[str, Any],
        risk_regime: dict[str, Any] | None = None,
    ) -> "PromptBuilder":
        base_system = autonomous_system_prompt_from_context(
            ctx,
            model,
            autonomous_config=autonomous_config,
            risk_regime=risk_regime,
        )
        return cls(
            profile=profile,
            profile_name=profile_name,
            ctx=ctx,
            model=model,
            mode="suggest_autonomous",
            base_system=base_system,
        )

    @classmethod
    def for_thesis_monitor(
        cls,
        *,
        profile: Any,
        profile_name: str,
        ctx: dict[str, Any],
        model: str,
    ) -> "PromptBuilder":
        base_system = autonomous_system_prompt_from_context(ctx, model)
        return cls(
            profile=profile,
            profile_name=profile_name,
            ctx=ctx,
            model=model,
            mode="thesis_monitor",
            base_system=base_system,
        )

    def append_news_block(
        self,
        *,
        symbol: str,
        rss_headline_count: int,
        web_result_count: int,
        parallel_timeout_sec: float,
        fallback_message: str | None = None,
    ) -> "PromptBuilder":
        try:
            block = build_trade_suggestion_news_block(
                symbol=symbol,
                rss_headline_count=rss_headline_count,
                web_result_count=web_result_count,
                parallel_timeout_sec=parallel_timeout_sec,
            )
        except Exception as exc:
            block = (
                fallback_message
                or (
                    "=== TRADE SUGGESTION — EXTERNAL MARKET NEWS (prefetched) ===\n"
                    f"News prefetch failed ({exc}); proceed using LIVE TRADING CONTEXT only.\n"
                )
            )
        self.system_parts.append(str(block).strip())
        self.news_block_hash = _hash_text(block)
        return self

    def append_learning_block(
        self,
        *,
        db_path: Path,
        days_back: int,
        max_recent_examples: int,
        current_ctx: dict[str, Any],
        fallback_message: str | None = None,
    ) -> "PromptBuilder":
        try:
            block = suggestion_tracker.build_learning_prompt_block(
                db_path,
                days_back=days_back,
                max_recent_examples=max_recent_examples,
                current_ctx=current_ctx,
            )
        except Exception as exc:
            block = (
                fallback_message
                or (
                    "=== FILLMORE LEARNING MEMORY (recent AI limit-order outcomes) ===\n"
                    f"Learning memory unavailable ({exc}); rely on LIVE TRADING CONTEXT and be selective.\n"
                )
            )
        self.system_parts.append(str(block).strip())
        self.learning_block_hash = _hash_text(block)
        return self

    def append_autonomous_memory(
        self,
        *,
        db_path: Path,
        risk_regime: dict[str, Any] | None,
        max_recent_examples: int = 6,
        reflection_limit: int = 8,
        today_limit: int = 10,
    ) -> "PromptBuilder":
        learning = suggestion_tracker.build_learning_prompt_block(
            db_path,
            days_back=180,
            max_recent_examples=max_recent_examples,
            current_ctx=self.ctx,
        )
        reflection_block = suggestion_tracker.build_autonomous_reflection_prompt_block(
            db_path,
            limit=reflection_limit,
            autonomous_only=True,
        )
        perf_block = autonomous_performance.build_performance_memory_block(
            db_path,
            risk_regime=risk_regime,
        )
        family_scorecard = autonomous_performance.build_family_scorecard_memory_block(db_path)
        today_block = suggestion_tracker.build_autonomous_today_block(db_path, max_items=today_limit)
        aux_memory = _fit_aux_memory_blocks([
            ("learning", learning, True),
            ("performance", perf_block, True),
            ("family_scorecard", family_scorecard, True),
            ("reflections", reflection_block, True),
            ("today", today_block, True),
        ])
        if aux_memory:
            self.system_parts.append(str(aux_memory).strip())
        self.learning_block_hash = _hash_text(learning)
        return self

    def build(
        self,
        *,
        user: str,
        prompt_version: str,
        token_budget: int | None = None,
    ) -> FillmorePromptAssembly:
        system = "\n\n".join(part for part in self.system_parts if str(part or "").strip())
        user_text = str(user or "").strip()
        prompt_hash = _hash_text(f"{system}\n\n---\n\n{user_text}") or _hash_json({"system": system, "user": user_text})
        return FillmorePromptAssembly(
            system=system,
            user=user_text,
            model=self.model,
            mode=self.mode,
            prompt_version=prompt_version,
            context_hash=_hash_json(self.ctx),
            prompt_hash=prompt_hash,
            news_block_hash=self.news_block_hash,
            learning_block_hash=self.learning_block_hash,
            token_budget=token_budget,
        )
