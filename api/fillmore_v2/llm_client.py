"""LLM client wrapper for Step 6 (PHASE9.4).

The orchestrator depends on a `LlmClient` Protocol, not the OpenAI SDK
directly. Tests inject `FakeLlmClient` to drive integration paths
deterministically without spending tokens. Real-LLM smoke testing happens
via `scripts/fillmore_v2_smoke_test.py`, not pytest.

The OpenAI client uses `response_format={"type": "json_object"}` so the
model can't return prose or markdown fences. Fenced output is treated as a
schema violation by `llm_output_schema.parse` regardless.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional, Protocol


@dataclass
class LlmCallResult:
    """Wraps everything the orchestrator needs to log a call.

    Includes the raw text so the audit trail captures exactly what the model
    returned (per PHASE9.8 item 10) — even if parsing fails.
    """
    raw_text: str
    model: str
    usage_tokens_in: Optional[int] = None
    usage_tokens_out: Optional[int] = None
    error: Optional[str] = None  # set on transport failure (timeout, 5xx, etc.)


class LlmClient(Protocol):
    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
    ) -> LlmCallResult:
        ...


# --- Production: OpenAI ------------------------------------------------------

class OpenAILlmClient:
    """Thin wrapper around openai.OpenAI. Imported lazily so tests don't
    require the SDK or an API key.
    """

    def __init__(self, *, openai_client: Any = None, max_tokens: int = 800):
        self._client = openai_client  # injectable for monkeypatching
        self._max_tokens = max_tokens

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        import openai  # noqa: WPS433 — lazy import
        self._client = openai.OpenAI()
        return self._client

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
    ) -> LlmCallResult:
        try:
            client = self._ensure_client()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=self._max_tokens,
                temperature=0.0,
            )
            choice = resp.choices[0]
            usage = getattr(resp, "usage", None)
            return LlmCallResult(
                raw_text=choice.message.content or "",
                model=model,
                usage_tokens_in=getattr(usage, "prompt_tokens", None) if usage else None,
                usage_tokens_out=getattr(usage, "completion_tokens", None) if usage else None,
            )
        except Exception as e:  # noqa: BLE001 — caller logs and forces skip
            return LlmCallResult(raw_text="", model=model, error=f"{type(e).__name__}: {e}")


# --- Testing: Fake LLM client -----------------------------------------------

@dataclass
class _FakeCall:
    system: str
    user: str
    model: str


class FakeLlmClient:
    """Returns scripted responses. Records every call for assertion in tests.

    Construct with either:
      - `FakeLlmClient(responses=[...])` — pop one per call (FIFO)
      - `FakeLlmClient(static_response="...")` — return the same text always
      - `FakeLlmClient(error="...")` — simulate transport failure
    """

    def __init__(
        self,
        *,
        responses: Optional[list[str]] = None,
        static_response: Optional[str] = None,
        error: Optional[str] = None,
    ):
        self._responses = list(responses) if responses else []
        self._static = static_response
        self._error = error
        self.calls: list[_FakeCall] = []

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
    ) -> LlmCallResult:
        self.calls.append(_FakeCall(system=system_prompt, user=user_prompt, model=model))
        if self._error is not None:
            return LlmCallResult(raw_text="", model=model, error=self._error)
        if self._responses:
            return LlmCallResult(raw_text=self._responses.pop(0), model=model)
        if self._static is not None:
            return LlmCallResult(raw_text=self._static, model=model)
        # Default: a benign skip so tests that don't care about the LLM still get a valid parse.
        return LlmCallResult(
            raw_text=json.dumps({"decision": "skip", "primary_thesis": "default fake"}),
            model=model,
        )
