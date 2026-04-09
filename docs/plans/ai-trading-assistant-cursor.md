# AI Trading Assistant — Cursor (frontend only)

**Audience:** Cursor agent working in parallel with Claude Code.  
**Scope:** React/TypeScript only. Do **not** implement the FastAPI chat route or `api/ai_trading_chat.py` — that is [ai-trading-assistant-claude-code.md](./ai-trading-assistant-claude-code.md).

**Partner plan:** Backend must expose the **frozen API contract** below. Implement the UI to this spec so integration works when both branches merge.

---

## Goal

1. Add **AI Trading Assistant** as the **first** item in the main sidebar nav.
2. Nav order: **AI Trading Assistant** → Dashboard → Analysis → Presets → Profile Editor → Logs & Stats → Help / Guide (keep Help / Guide last).
3. New page: chat UI that POSTs to the backend and displays a **streaming** assistant reply.

---

## Frozen API contract (must match backend plan)

### `POST /api/data/{profile_name}/ai-chat?profile_path=<encoded>`

- **`profile_name`:** Use `selectedProfile.name` (same as other dashboard API calls).
- **`profile_path`:** Use `selectedProfile.path` — **URL-encode** when building the query string (e.g. `encodeURIComponent`).

**Request body:**

```json
{
  "message": "string",
  "history": [
    { "role": "user", "content": "string" },
    { "role": "assistant", "content": "string" }
  ]
}
```

Send **`message`** = latest user text only; **`history`** = prior messages only (no duplicate of `message`). Cap history at **10** pairs client-side to match server.

**Success:** `200`, `Content-Type: text/event-stream` (may include charset). Body is SSE:

- Lines starting with `data: ` followed by JSON.
- `{"type":"delta","text":"..."}` — append `text` to the current assistant message.
- `{"type":"done"}` — stream complete.

**Errors (JSON, not SSE):**

- **503** — e.g. `OPENAI_API_KEY` not configured: read `detail` from JSON body, show a clear message (“Ask admin to set OPENAI_API_KEY on the server”).
- Other **4xx/5xx** — show `detail` or status text; do not parse as SSE.

**Implementation note:** Use `fetch()`, not `EventSource` (EventSource cannot POST a body). Read `response.body.getReader()`, decode UTF-8, buffer incomplete lines, split on `\n\n`, parse `data: ` payloads as JSON.

---

## Files to modify

### [frontend/src/App.tsx](../../frontend/src/App.tsx)

1. Extend `Page` type: add `'ai-assistant'` (or `'aiAssistant'` — pick one id and use consistently).

2. In `<nav>`, add the **first** button:

   - Label: `AI Trading Assistant`
   - `onClick` → `setPage('ai-assistant')`
   - `active` when `page === 'ai-assistant'`

3. Keep existing nav order after it (Dashboard, Analysis, …, Help / Guide last).

4. In the main content switch (same place as `DashboardPage`, `LogsPage`, etc.):

   ```tsx
   {page === 'ai-assistant' && <AiTradingAssistantPage profile={selectedProfile} />}
   ```

5. Default page stays `dashboard` unless product asks otherwise.

---

## Files to create

### [frontend/src/AiTradingAssistantPage.tsx](../../frontend/src/AiTradingAssistantPage.tsx)

Props: `{ profile: { path: string; name: string } }` (match existing `Profile` type — import or inline).

**State:**

- `messages`: array of `{ role: 'user' | 'assistant'; content: string }` (or separate `user`/`assistant` rendering).
- `input`: controlled string for textarea.
- `sending`: boolean; disable Send while streaming.
- Optional: `error`: string | null.

**Behavior:**

1. On Send: append user message to `messages`, clear input, set `sending`.
2. Build `history` from prior turns (exclude the message you just added if the API expects history *before* current message — align with backend: typically send `history` as previous pairs + current `message` field; follow Claude plan: body has `message` + `history` of past turns only).
3. Call streaming helper from `api.ts` with `profile.name`, `profile.path`, `message`, `history`.
4. Append assistant `content` incrementally from `delta.text` chunks; on `done`, clear `sending`.
5. On non-OK response: parse JSON error if possible, set `error`, clear `sending`.
6. Auto-scroll message list to bottom on new content.
7. Styling: reuse existing `.card`, `.btn`, `var(--text-secondary)`, etc. for visual consistency.

**Edge cases:**

- Empty input: no-op or disable Send.
- Stop / AbortController (optional): cancel fetch on unmount or new Send.

---

### [frontend/src/api.ts](../../frontend/src/api.ts)

Add something like:

```ts
export async function streamAiChat(
  profileName: string,
  profilePath: string,
  body: { message: string; history: { role: 'user' | 'assistant'; content: string }[] },
  onDelta: (text: string) => void,
): Promise<void>
```

- URL: `` `${API_BASE}/data/${encodeURIComponent(profileName)}/ai-chat?profile_path=${encodeURIComponent(profilePath)}` ``
- `method: 'POST'`, `headers: { 'Content-Type': 'application/json' }`, `body: JSON.stringify(body)`
- If `!response.ok`, parse `text()` as JSON for `detail`, throw `Error(detail)`.
- If `ok`, check content-type includes `text/event-stream` (warn if not), then read stream and parse SSE as specified.

Alternatively return an `AsyncGenerator<string>` of deltas — either pattern is fine if `AiTradingAssistantPage` stays simple.

---

## Testing (Cursor)

1. Run Vite dev server (proxy `/api` to FastAPI per [frontend/vite.config.ts](../../frontend/vite.config.ts)).
2. With backend running and key set: send a message; see streaming text.
3. With backend returning 503: UI shows friendly message.
4. Unlock flow: page should only be reachable when profile is unlocked (same as other tabs — already gated in `App.tsx`).

---

## Out of scope

- Python, `api/main.py`, `requirements.txt`, OpenAI SDK.
- Changing the frozen contract without updating the Claude Code plan.

---

## Checklist

- [ ] `Page` + nav first item + render `AiTradingAssistantPage`
- [ ] `AiTradingAssistantPage.tsx` with chat UX and SSE parsing
- [ ] `streamAiChat` (or equivalent) in `api.ts`
- [ ] 503 / JSON error handling with user-visible copy
- [ ] No edits to `api/ai_trading_chat.py` or new Python routes in this task
