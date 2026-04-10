import { useCallback, useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import * as api from './api';

const MAX_HISTORY_MESSAGES = 20; // 10 user + 10 assistant pairs

export interface AiAssistantProfile {
  path: string;
  name: string;
}

type ChatLine = { role: 'user' | 'assistant'; content: string };

const CHAT_STORAGE_PREFIX = 'usdjpy_ai_chat:';
const MODEL_STORAGE_PREFIX = 'usdjpy_ai_chat_model:';

function chatStorageKey(profilePath: string): string {
  return `${CHAT_STORAGE_PREFIX}${profilePath}`;
}

function modelStorageKey(profilePath: string): string {
  return `${MODEL_STORAGE_PREFIX}${profilePath}`;
}

function loadModelFromStorage(profilePath: string): string | null {
  if (typeof sessionStorage === 'undefined') return null;
  try {
    const v = sessionStorage.getItem(modelStorageKey(profilePath));
    return v && v.trim() ? v.trim() : null;
  } catch {
    return null;
  }
}

function saveModelToStorage(profilePath: string, model: string): void {
  if (typeof sessionStorage === 'undefined') return;
  try {
    sessionStorage.setItem(modelStorageKey(profilePath), model);
  } catch {
    // ignore
  }
}

function loadChatFromStorage(profilePath: string): ChatLine[] {
  if (typeof sessionStorage === 'undefined') return [];
  try {
    const raw = sessionStorage.getItem(chatStorageKey(profilePath));
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (row): row is ChatLine =>
        row !== null &&
        typeof row === 'object' &&
        (row as ChatLine).role !== undefined &&
        ((row as ChatLine).role === 'user' || (row as ChatLine).role === 'assistant') &&
        typeof (row as ChatLine).content === 'string'
    );
  } catch {
    return [];
  }
}

function saveChatToStorage(profilePath: string, messages: ChatLine[]): void {
  if (typeof sessionStorage === 'undefined') return;
  try {
    sessionStorage.setItem(chatStorageKey(profilePath), JSON.stringify(messages));
  } catch {
    // quota or private mode
  }
}

function capHistory(messages: ChatLine[]): api.AiChatHistoryMessage[] {
  const slice = messages.slice(-MAX_HISTORY_MESSAGES);
  return slice.map(({ role, content }) => ({ role, content }));
}

/** Renders Markdown-style `**bold**` segments (no extra deps). */
function renderBoldSegments(text: string): ReactNode[] {
  const re = /\*\*([^*]+)\*\*/g;
  const out: ReactNode[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  let k = 0;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) {
      out.push(<span key={k++}>{text.slice(last, m.index)}</span>);
    }
    out.push(<strong key={k++}>{m[1]}</strong>);
    last = m.index + m[0].length;
  }
  if (last < text.length) {
    out.push(<span key={k++}>{text.slice(last)}</span>);
  }
  return out.length > 0 ? out : [text];
}

function AssistantFormattedBody({ content }: { content: string }): ReactNode {
  if (!content) return null;
  const lines = content.split('\n');
  return (
    <>
      {lines.map((line, li) => (
        <span key={li}>
          {li > 0 ? <br /> : null}
          {renderBoldSegments(line)}
        </span>
      ))}
    </>
  );
}

const FALLBACK_AI_MODELS: string[] = ['gpt-5-mini', 'gpt-4o-mini', 'gpt-4o'];

function signedPct(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return '-';
  return `${v > 0 ? '+' : ''}${v.toFixed(1)}%`;
}

function moveArrow(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return '-';
  if (v > 0) return 'UP';
  if (v < 0) return 'DOWN';
  return 'FLAT';
}

function countdownLabel(minutesToEvent: number): string {
  if (!Number.isFinite(minutesToEvent) || minutesToEvent < 0) return 'now';
  const h = Math.floor(minutesToEvent / 60);
  const m = Math.floor(minutesToEvent % 60);
  if (h <= 0) return `${m}m`;
  return `${h}h ${m}m`;
}

export default function AiTradingAssistantPage({ profile }: { profile: AiAssistantProfile }) {
  const [messages, setMessages] = useState<ChatLine[]>(() => loadChatFromStorage(profile.path));
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [modelList, setModelList] = useState<string[]>([]);
  const [chatModel, setChatModel] = useState<string>('');
  const [modelsReady, setModelsReady] = useState(false);
  const [toolStatus, setToolStatus] = useState<string | null>(null);
  const [rail, setRail] = useState<api.AiRailPayload | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    setMessages(loadChatFromStorage(profile.path));
  }, [profile.path]);

  useEffect(() => {
    let cancelled = false;
    api
      .getAiChatModels()
      .then((res) => {
        if (cancelled) return;
        const list = res.models.length > 0 ? res.models : [...FALLBACK_AI_MODELS];
        setModelList(list);
        const stored = loadModelFromStorage(profile.path);
        const def = list.includes(res.default_model) ? res.default_model : list[0];
        setChatModel(stored && list.includes(stored) ? stored : def);
        setModelsReady(true);
      })
      .catch(() => {
        if (cancelled) return;
        const list = [...FALLBACK_AI_MODELS];
        setModelList(list);
        const stored = loadModelFromStorage(profile.path);
        setChatModel(stored && list.includes(stored) ? stored : list[0]);
        setModelsReady(true);
      });
    return () => {
      cancelled = true;
    };
  }, [profile.path]);

  useEffect(() => {
    saveChatToStorage(profile.path, messages);
  }, [profile.path, messages]);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages, sending]);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    let mounted = true;
    const pull = () => {
      api.getAiRail(profile.name, profile.path, 7).then((r) => {
        if (mounted) setRail(r);
      }).catch(() => {});
    };
    pull();
    const id = setInterval(pull, 30000);
    return () => { mounted = false; clearInterval(id); };
  }, [profile.name, profile.path]);

  const send = useCallback(async (overrideText?: string) => {
    const text = (overrideText ?? input).trim();
    if (!text || sending) return;

    setError(null);
    abortRef.current?.abort();
    abortRef.current = new AbortController();
    const signal = abortRef.current.signal;

    const history = capHistory(messages);
    setInput('');
    setSending(true);
    setMessages((m) => [...m, { role: 'user', content: text }, { role: 'assistant', content: '' }]);

    try {
      await api.streamAiChat(
        profile.name,
        profile.path,
        { message: text, history, chat_model: chatModel || undefined },
        {
          signal,
          onDelta: (delta) => {
            setToolStatus(null);
            setMessages((m) => {
              if (m.length === 0) return m;
              const next = [...m];
              const last = next[next.length - 1];
              if (last?.role === 'assistant') {
                next[next.length - 1] = { role: 'assistant', content: last.content + delta };
              }
              return next;
            });
          },
          onToolStatus: (name) => {
            const labels: Record<string, string> = {
              get_candles: 'Fetching candles...',
              get_trade_history: 'Loading trades...',
              get_trade_details: 'Looking up trade...',
              analyze_trade_patterns: 'Analyzing patterns...',
              get_cross_asset_bias: 'Checking macro bias...',
              get_economic_calendar: 'Checking calendar...',
              get_news_headlines: 'Searching news...',
              web_search: 'Searching the web...',
            };
            setToolStatus(labels[name] || `Running ${name}...`);
          },
        }
      );
    } catch (e) {
      if ((e as Error).name === 'AbortError') {
        setMessages((m) => {
          if (m.length < 2) return m;
          const last = m[m.length - 1];
          const prev = m[m.length - 2];
          if (prev.role === 'user' && last.role === 'assistant' && !last.content) {
            return m.slice(0, -2);
          }
          return m;
        });
        return;
      }
      const msg = (e as Error).message || 'API fetch error';
      const friendly =
        /OPENAI_API_KEY|not configured|503/i.test(msg) || /503/.test(msg)
          ? 'The API server needs an OpenAI key. Set OPENAI_API_KEY in the environment where uvicorn runs (e.g. Railway variables), then restart.'
          : msg;
      setError(friendly);
      setMessages((m) => {
        if (m.length < 2) return m;
        const last = m[m.length - 1];
        const prev = m[m.length - 2];
        if (prev.role === 'user' && last.role === 'assistant' && !last.content) {
          return m.slice(0, -2);
        }
        return m;
      });
    } finally {
      setSending(false);
      setToolStatus(null);
      abortRef.current = null;
    }
  }, [chatModel, input, messages, profile.name, profile.path, sending]);

  const macroImplication = (() => {
    const b = (rail?.macro.combined_bias || '').toLowerCase();
    if (b === 'bullish') return 'Directional confirmation for longs; macro tailwind is aligned.';
    if (b === 'bearish') return 'Directional caution for longs; macro setup currently favors downside.';
    if (b === 'conflicting') return 'Mixed macro inputs; size down and rely more on level location.';
    return 'Macro signal is neutral; rely primarily on levels, session and volatility.';
  })();

  const sessionImplication = (() => {
    const label = String(rail?.session_vol.vol_label || '').toLowerCase();
    if (label.includes('elevated') || label.includes('above')) return 'Higher volatility regime; expect wider swings and faster invalidation.';
    if (label.includes('below') || label.includes('very low') || label.includes('compressed')) return 'Compressed regime; range behavior more likely than clean breakouts.';
    return 'Normal volatility regime; use standard target/stop expectations.';
  })();

  const ladderImplication = (() => {
    const nearestSup = rail?.levels.supports?.[0];
    const nearestRes = rail?.levels.resistances?.[0];
    const ds = nearestSup?.distance_pips != null ? Math.abs(nearestSup.distance_pips) : null;
    const dr = nearestRes?.distance_pips != null ? Math.abs(nearestRes.distance_pips) : null;
    if (ds != null && dr != null) {
      if (dr < 6) return 'Price is close to resistance; upside room is limited without a clean break.';
      if (ds < 6) return 'Price is close to support; downside room is limited unless support fails.';
      return 'Price sits between levels; entries are cleaner nearer the edges of the ladder.';
    }
    return 'Use nearest support/resistance distances to gauge immediate room before adding.';
  })();

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 360px', gap: 16, alignItems: 'start' }}>
      <div>
      <h2 className="page-title">AI Trading Assistant</h2>
      <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: 16, maxWidth: 720 }}>
        Context-aware help using your profile&apos;s broker connection on the server. Not financial advice.
      </p>

      <div className="form-group" style={{ marginBottom: 16, maxWidth: 360 }}>
        <label htmlFor="ai-chat-model">Model</label>
        <select
          id="ai-chat-model"
          value={chatModel}
          disabled={!modelsReady || sending || modelList.length === 0}
          onChange={(e) => {
            const v = e.target.value;
            setChatModel(v);
            saveModelToStorage(profile.path, v);
          }}
          style={{ width: '100%', padding: '8px 10px', borderRadius: 8 }}
        >
          {!modelsReady ? (
            <option value="">Loading…</option>
          ) : (
            modelList.map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))
          )}
        </select>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.75rem', margin: '6px 0 0' }}>
          Choices come from the server allowlist. Admins can set{' '}
          <code style={{ fontSize: '0.7rem' }}>AI_CHAT_ALLOWED_MODELS</code> (comma-separated).
        </p>
      </div>

      {error && (
        <div className="card mb-4" style={{ borderColor: 'var(--danger)' }}>
          <p style={{ color: 'var(--danger)', margin: 0 }}>{error}</p>
        </div>
      )}

      {messages.length === 0 && !sending && (
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 12, maxWidth: 720 }}>
          {[
            "How's my P&L today?",
            "What's the current setup?",
            "Position sizing help",
            "Analyze recent losses",
            "Any upcoming events?",
            "What's the latest USDJPY news?",
          ].map((chip) => (
            <button
              key={chip}
              type="button"
              onClick={() => void send(chip)}
              style={{
                padding: '6px 14px',
                borderRadius: 20,
                border: '1px solid var(--card-border, rgba(255,255,255,0.12))',
                background: 'var(--bg-secondary, rgba(0,0,0,0.2))',
                color: 'var(--text-secondary)',
                fontSize: '0.8rem',
                cursor: 'pointer',
              }}
            >
              {chip}
            </button>
          ))}
        </div>
      )}

      <div className="card" style={{ display: 'flex', flexDirection: 'column', maxWidth: 720, height: 'calc(100vh - 180px)', minHeight: 560 }}>
        <div
          ref={scrollRef}
          style={{
            flex: 1,
            overflowY: 'auto',
            padding: 12,
            marginBottom: 12,
            background: 'var(--bg-secondary, rgba(0,0,0,0.2))',
            borderRadius: 8,
            minHeight: 0,
          }}
        >
          {messages.length === 0 && (
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', margin: 0 }}>
              Ask about open positions, recent performance, or session context. Replies stream from the model.
            </p>
          )}
          {messages.map((line, i) => (
            <div
              key={i}
              style={{
                marginBottom: 12,
                textAlign: line.role === 'user' ? 'right' : 'left',
              }}
            >
              <div
                style={{
                  display: 'inline-block',
                  maxWidth: '92%',
                  padding: '10px 14px',
                  borderRadius: 10,
                  fontSize: '0.9rem',
                  lineHeight: 1.45,
                  whiteSpace: line.role === 'user' ? 'pre-wrap' : 'normal',
                  wordBreak: 'break-word',
                  background:
                    line.role === 'user' ? 'var(--accent, #3b82f6)' : 'var(--card-border, rgba(255,255,255,0.08))',
                  color: line.role === 'user' ? '#fff' : 'var(--text-primary)',
                  opacity: line.role === 'assistant' && sending && i === messages.length - 1 && !line.content ? 0.6 : 1,
                }}
              >
                {line.role === 'user' ? <strong>You</strong> : <strong>Assistant</strong>}
                <div style={{ marginTop: 4, fontWeight: 400 }}>
                  {line.role === 'assistant' ? (
                    <>
                      <AssistantFormattedBody content={line.content} />
                      {sending && i === messages.length - 1 && !line.content ? '…' : null}
                    </>
                  ) : (
                    line.content || (sending && i === messages.length - 1 ? '…' : '')
                  )}
                </div>
              </div>
            </div>
          ))}
          {toolStatus && (
            <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', fontStyle: 'italic', padding: '4px 0' }}>
              {toolStatus}
            </div>
          )}
        </div>

        <div className="form-group" style={{ marginBottom: 8 }}>
          <label>Message</label>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                void send();
              }
            }}
            placeholder="Type a message… (Enter to send, Shift+Enter for newline)"
            rows={3}
            disabled={sending}
            style={{ width: '100%', resize: 'vertical' }}
          />
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
          <button type="button" className="btn btn-primary" onClick={() => void send()} disabled={sending || !input.trim()}>
            {sending ? 'Sending…' : 'Send'}
          </button>
          {sending && (
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => {
                abortRef.current?.abort();
              }}
            >
              Stop
            </button>
          )}
          <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
            <button
              type="button"
              className="btn btn-secondary"
              disabled={sending || messages.length === 0}
              onClick={() => {
                const lines = messages.map((m) => {
                  const label = m.role === 'user' ? 'You' : 'Assistant';
                  return `### ${label}\n${m.content}\n`;
                });
                const header = `# USDJPY AI Chat Export\n# Profile: ${profile.name}\n# Date: ${new Date().toISOString()}\n\n`;
                const blob = new Blob([header + lines.join('\n---\n\n')], { type: 'text/markdown' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `usdjpy-chat-${new Date().toISOString().slice(0, 10)}.md`;
                a.click();
                URL.revokeObjectURL(url);
              }}
            >
              Export
            </button>
            <button
              type="button"
              className="btn btn-secondary"
              disabled={sending || messages.length === 0}
              onClick={() => {
                setMessages([]);
                saveChatToStorage(profile.path, []);
                setError(null);
              }}
            >
              Clear Chat
            </button>
          </div>
        </div>
      </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 12, position: 'sticky', top: 12 }}>
        <div className="card">
          <div style={{ fontWeight: 700, marginBottom: 6 }}>Macro Confirmation</div>
          <div style={{ color: 'var(--text-secondary)', fontSize: '0.82rem', marginBottom: 8 }}>
            Bias: <strong style={{ color: 'var(--text-primary)' }}>{rail?.macro.combined_bias || 'n/a'}</strong> ({rail?.macro.confidence || 'n/a'})
          </div>
          <div style={{ fontSize: '0.8rem', display: 'grid', gap: 4 }}>
            <div>DXY: {rail?.macro.dxy.value ?? '-'} | 1D {moveArrow(rail?.macro.dxy.one_day)} | 5D {moveArrow(rail?.macro.dxy.five_day)} ({signedPct(rail?.macro.dxy.five_day)})</div>
            <div>US10Y: {rail?.macro.us10y.value ?? '-'}% | 1D {moveArrow(rail?.macro.us10y.one_day)} | 5D {moveArrow(rail?.macro.us10y.five_day)}</div>
            <div>Oil: {rail?.macro.oil.value ?? '-'} | 1D {moveArrow(rail?.macro.oil.one_day)} | 5D {moveArrow(rail?.macro.oil.five_day)} ({signedPct(rail?.macro.oil.five_day)})</div>
            <div>Gold: {rail?.macro.gold.value ?? '-'} | 1D {moveArrow(rail?.macro.gold.one_day)} | 5D {moveArrow(rail?.macro.gold.five_day)}</div>
          </div>
          <div style={{ marginTop: 8, color: 'var(--text-secondary)', fontSize: '0.78rem' }}>{macroImplication}</div>
        </div>

        <div className="card">
          <div style={{ fontWeight: 700, marginBottom: 6 }}>Event Risk Countdown</div>
          {(rail?.events || []).length === 0 ? (
            <div style={{ color: 'var(--text-secondary)', fontSize: '0.82rem' }}>No upcoming high-impact USD/JPY events.</div>
          ) : (
            <div style={{ display: 'grid', gap: 6 }}>
              {(rail?.events || []).slice(0, 3).map((ev, idx) => {
                const mins = Number(ev.minutes_to_event || 0);
                const urgent = mins <= 60;
                const soon = mins > 60 && mins <= 180;
                const tone = urgent ? 'var(--danger)' : soon ? 'var(--warning, #d4a017)' : 'var(--text-secondary)';
                return (
                  <div key={`${ev.timestamp_utc}-${idx}`} style={{ fontSize: '0.8rem' }}>
                    <div style={{ color: 'var(--text-primary)' }}>{ev.currency} {ev.event}</div>
                    <div style={{ color: tone }}>
                      {countdownLabel(mins)} ({ev.time || ev.date})
                    </div>
                  </div>
                );
              })}
            </div>
          )}
          <div style={{ marginTop: 8, color: 'var(--text-secondary)', fontSize: '0.78rem' }}>
            {(rail?.events?.[0]?.minutes_to_event ?? 9999) <= 60
              ? 'Event risk elevated: avoid fresh adds into the release window.'
              : 'No immediate high-impact release window; event risk is manageable.'}
          </div>
        </div>

        <div className="card">
          <div style={{ fontWeight: 700, marginBottom: 6 }}>Key Levels Ladder</div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 6 }}>
            Mid: <span style={{ color: 'var(--text-primary)' }}>{rail?.levels.mid ?? '-'}</span>
          </div>
          <div style={{ fontSize: '0.8rem', display: 'grid', gap: 4 }}>
            <div style={{ color: 'var(--text-secondary)' }}>Resistance</div>
            {(rail?.levels.resistances || []).slice(0, 2).map((r, i) => (
              <div key={`r-${i}`}>{r.price} ({r.distance_pips ?? '-'}p)</div>
            ))}
            <div style={{ color: 'var(--text-secondary)', marginTop: 6 }}>Support</div>
            {(rail?.levels.supports || []).slice(0, 2).map((s, i) => (
              <div key={`s-${i}`}>{s.price} ({s.distance_pips ?? '-'}p)</div>
            ))}
          </div>
          <div style={{ marginTop: 8, color: 'var(--text-secondary)', fontSize: '0.78rem' }}>{ladderImplication}</div>
        </div>

        <div className="card">
          <div style={{ fontWeight: 700, marginBottom: 6 }}>Session &amp; Volatility</div>
          <div style={{ fontSize: '0.8rem', display: 'grid', gap: 4 }}>
            <div>Session: {rail?.session_vol.overlap || (rail?.session_vol.active_sessions || []).join(' + ') || 'none'}</div>
            <div>Next close: {rail?.session_vol.next_close || '-'}</div>
            <div>Spread: {rail?.session_vol.spread_pips ?? '-'}p</div>
            <div>ATR regime: {rail?.session_vol.vol_label || 'n/a'} ({rail?.session_vol.vol_ratio ?? '-'}x)</div>
          </div>
          <div style={{ marginTop: 8, color: 'var(--text-secondary)', fontSize: '0.78rem' }}>{sessionImplication}</div>
        </div>
      </div>
    </div>
  );
}
