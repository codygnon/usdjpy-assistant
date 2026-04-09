# AI Trading Assistant — Claude Code (backend only)

**Audience:** Claude Code agent working in parallel with Cursor.  
**Scope:** Python/FastAPI only. Do **not** edit `frontend/` or `frontend/src/*`.

**Partner plan:** [ai-trading-assistant-cursor.md](./ai-trading-assistant-cursor.md) implements the UI against the **frozen API contract** below. Both plans must match that contract exactly.

---

## Goal

Add a streaming chat endpoint that:

1. Loads the trader’s profile from disk (`profile_path`).
2. Fetches live broker context (OANDA or MT5) via existing adapters.
3. Builds a system prompt with account state, open positions, and recent closed trades (bounded size).
4. Streams the assistant reply from OpenAI Chat Completions (`gpt-4o-mini` by default).

Secrets: `OPENAI_API_KEY` and broker tokens stay **server-side only**.

---

## Frozen API contract (authoritative)

Implement this exactly so the Cursor-built frontend works without changes.

### `POST /api/data/{profile_name}/ai-chat`

**Path parameter:** `profile_name` — same stem/name the UI uses for other `/api/data/{profile_name}/...` routes.

**Query parameter:** `profile_path` — **required**. Same semantics as existing endpoints (e.g. `GET /api/data/{profile_name}/open-trades`). Use existing helper `_resolve_profile_path(profile_path)` from [api/main.py](../../api/main.py).

**Request headers:** `Content-Type: application/json`

**Request body (JSON):**

```json
{
  "message": "string",
  "history": [
    { "role": "user", "content": "string" },
    { "role": "assistant", "content": "string" }
  ]
}
```

- **`message`** is the **current** user turn only. **`history`** is prior turns only (alternating `user` / `assistant`), **excluding** the current `message`.
- Validate `message` is a non-empty string after strip (or allow empty with 400 — pick one and document).
- Cap `history` to the last **10** user/assistant pairs server-side (drop older entries).
- Reject unknown `role` values with **400**.

**Success: 200**

- `Content-Type: text/event-stream`
- `Cache-Control: no-cache`
- Body: **Server-Sent Events**. Each event is one or more lines of the form `data: <json>\n\n`.

**SSE JSON objects (only these `type` values):**

| `type`   | Other fields | Meaning |
|----------|----------------|--------|
| `delta`  | `text` (string) | Append `text` to the assistant reply (UTF-8). May be partial; may be empty. |
| `done`   | — | Stream finished successfully. Send once, then end the response. |

**Example stream:**

```
data: {"type":"delta","text":"You "}

data: {"type":"delta","text":"are long "}

data: {"type":"done"}

```

(Blank lines between events are fine per SSE spec.)

**Errors (not SSE — return JSON with FastAPI `HTTPException` or equivalent):**

| Status | When | Body shape |
|--------|------|------------|
| **503** | `OPENAI_API_KEY` env unset or empty | `{"detail": "OPENAI_API_KEY is not configured on the server"}` |
| **400** | Missing/invalid `profile_path`, bad JSON body | `{"detail": "<readable message>"}` |
| **404** | Profile file not found after resolve | `{"detail": "..."}` |
| **502** or **500** | Broker init failure, OpenAI failure mid-request | Prefer JSON `detail` before streaming starts; if failure mid-stream, emit `data: {"type":"error","message":"..."}\n\n` then end (optional — if simpler, only fail before stream) |

If you add `type: error` in the stream, document it in the module docstring so Cursor can handle it; otherwise keep failures **before** `StreamingResponse` starts.

---

## Files to create

### `api/ai_trading_chat.py`

Responsibilities:

1. **`build_trading_context(profile: ProfileV1, store: SqliteStore | None) -> dict`**  
   - Import `get_adapter` from `adapters.broker`, `load_profile_v1` from `core.profile` (callers pass loaded profile; store from `_store_for(profile_name)` if needed).  
   - `adapter.initialize()` / `adapter.shutdown()` in `try/finally`.  
   - **Account:** `adapter.get_account_info()` — normalize to plain dict (floats for balance/equity/margin as applicable). Both OANDA and MT5 expose this via [adapters/broker.py](../../adapters/broker.py).  
   - **Open positions:** `adapter.get_open_positions(profile.symbol)` — normalize list (instrument, side, volume/units, entry, unrealized if present).  
   - **Closed trades (recent):**  
     - If `broker_type == "oanda"`: use `get_closed_trade_summaries(days_back=30, symbol=profile.symbol, pip_size=profile.pip_size)` (or equivalent on `OandaAdapter`).  
     - Else MT5: use `get_mt5_full_report` or `get_mt5_report_stats` + recent deals pattern already used elsewhere in the API if needed — keep **last 20–30** closed items max.  
   - **OANDA-only extras:** If `getattr(profile, "broker_type", None) == "oanda"` and adapter has `get_order_book`, optionally fetch order book for `profile.symbol` and add top N bucket levels (cap JSON size). Skip entirely for MT5.  
   - **Never** put raw `oanda_token` or passwords in the context dict that becomes prompt text.  
   - Add `"as_of": "<ISO8601 UTC>"` timestamp.

2. **`system_prompt_from_context(ctx: dict) -> str`**  
   - Trading assistant persona: supports manual USDJPY trader; concise; lead with numbers; no imperative “take this trade”; can mention user-defined guardrails if present in context.  
   - Serialize `ctx` as readable bullet/JSON block for the model.

3. **`stream_openai_chat(*, system: str, user_message: str, history: list[dict], model: str)`**  
   - Use official `openai` Python SDK, async or sync stream — must yield UTF-8 text chunks for SSE `delta` events.  
   - Map OpenAI stream chunks to `{"type":"delta","text":...}`; after completion yield done signal (handled by route).

4. **Module docstring** must restate the SSE `delta` / `done` contract and error statuses.

**Dependencies:** Add `openai` to [requirements.txt](../../requirements.txt).

---

## Files to modify

### `api/main.py`

1. Import your new module (e.g. `from api.ai_trading_chat import ...` — adjust if package layout uses different import path; project runs with `sys.path` including repo root, so `api` package is valid).

2. Add Pydantic models for request body: `message: str`, `history: list` with `role` + `content`.

3. Register:

```python
@app.post("/api/data/{profile_name}/ai-chat")
```

4. Implementation sketch:

   - If not `os.environ.get("OPENAI_API_KEY", "").strip()`: raise HTTP 503 with frozen `detail` string.
   - Resolve `profile_path` query param; `load_profile_v1`; get `profile_name` from path or use path param consistently with other routes.
   - Run **blocking** broker + context build inside `ThreadPoolExecutor` with timeout (e.g. 15–25s), same pattern as [get_open_trades](../../api/main.py) / `API_OPEN_TRADES_SYNC_TIMEOUT_SEC`.
   - Build system prompt; call OpenAI stream; return `StreamingResponse(generator, media_type="text/event-stream", headers={"Cache-Control": "no-cache"})`.

5. Do not break existing routes.

---

## Configuration

| Env var | Required | Default |
|---------|----------|---------|
| `OPENAI_API_KEY` | Yes, for chat | — |
| `OPENAI_CHAT_MODEL` | No | `gpt-4o-mini` |

---

## Testing (Claude Code)

1. Export `OPENAI_API_KEY`, run `uvicorn api.main:app --reload --port 8000`.
2. `curl -N -X POST "http://127.0.0.1:8000/api/data/<profile>/ai-chat?profile_path=<urlencoded_path>" -H "Content-Type: application/json" -d '{"message":"Summarize my account","history":[]}'`  
   - Expect `text/event-stream` and `data: {"type":"delta"...}` lines.
3. Unset key: expect **503** JSON.
4. MT5 profile: must not crash if order book is skipped.

---

## Out of scope

- Frontend, Vite, React, `api.ts`.
- Anthropic / multi-provider switch (unless trivial env flag you add without changing the contract).
- Redis, background poller, web search.

---

## Checklist

- [ ] `api/ai_trading_chat.py` with context builder, prompt, OpenAI streaming helper
- [ ] `POST /api/data/{profile_name}/ai-chat` in `api/main.py` with timeouts
- [ ] `openai` in `requirements.txt`
- [ ] SSE output matches **delta** / **done** contract
- [ ] 503 when API key missing
- [ ] No secrets in prompts; no `frontend/` edits
