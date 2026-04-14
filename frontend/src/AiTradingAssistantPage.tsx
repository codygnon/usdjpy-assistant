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
const SUGGEST_MODEL_STORAGE_PREFIX = 'usdjpy_ai_suggest_model:';

function chatStorageKey(profilePath: string): string {
  return `${CHAT_STORAGE_PREFIX}${profilePath}`;
}

function modelStorageKey(profilePath: string): string {
  return `${MODEL_STORAGE_PREFIX}${profilePath}`;
}

function suggestModelStorageKey(profilePath: string): string {
  return `${SUGGEST_MODEL_STORAGE_PREFIX}${profilePath}`;
}

function loadSuggestModelFromStorage(profilePath: string): string | null {
  if (typeof sessionStorage === 'undefined') return null;
  try {
    const v = sessionStorage.getItem(suggestModelStorageKey(profilePath));
    return v && v.trim() ? v.trim() : null;
  } catch {
    return null;
  }
}

function saveSuggestModelToStorage(profilePath: string, model: string): void {
  if (typeof sessionStorage === 'undefined') return;
  try {
    sessionStorage.setItem(suggestModelStorageKey(profilePath), model);
  } catch {
    // ignore
  }
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
  const [suggestModelList, setSuggestModelList] = useState<string[]>([]);
  const [suggestModel, setSuggestModel] = useState<string>('');
  const [toolStatus, setToolStatus] = useState<string | null>(null);
  const [rail, setRail] = useState<api.AiRailPayload | null>(null);
  // Trade suggestion state
  const [suggestion, setSuggestion] = useState<api.AiTradeSuggestion | null>(null);
  const [editDraft, setEditDraft] = useState<api.AiTradeSuggestion | null>(null);
  const [suggestLoading, setSuggestLoading] = useState(false);
  const [suggestError, setSuggestError] = useState<string | null>(null);
  const [placeLoading, setPlaceLoading] = useState(false);
  const [placeResult, setPlaceResult] = useState<{ status: string; order_id?: number | null; loop_auto_started?: boolean } | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const suggestPanelRef = useRef<HTMLDivElement>(null);
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

        // Suggest-model pool is separate (higher reasoning models).
        const sList = (res.suggest_models && res.suggest_models.length > 0)
          ? res.suggest_models
          : ['gpt-4o', ...list];
        setSuggestModelList(sList);
        const sStored = loadSuggestModelFromStorage(profile.path);
        const sDef = res.default_suggest_model && sList.includes(res.default_suggest_model)
          ? res.default_suggest_model
          : sList[0];
        setSuggestModel(sStored && sList.includes(sStored) ? sStored : sDef);

        setModelsReady(true);
      })
      .catch(() => {
        if (cancelled) return;
        const list = [...FALLBACK_AI_MODELS];
        setModelList(list);
        const stored = loadModelFromStorage(profile.path);
        setChatModel(stored && list.includes(stored) ? stored : list[0]);
        const sList = ['gpt-4o', ...list.filter((m) => m !== 'gpt-4o')];
        setSuggestModelList(sList);
        const sStored = loadSuggestModelFromStorage(profile.path);
        setSuggestModel(sStored && sList.includes(sStored) ? sStored : sList[0]);
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

  // Auto-scroll to suggestion panel when it appears
  useEffect(() => {
    if ((editDraft || suggestLoading) && suggestPanelRef.current) {
      suggestPanelRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [editDraft, suggestLoading]);

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

  const handleSuggestTrade = useCallback(async () => {
    setSuggestLoading(true);
    setSuggestError(null);
    setPlaceResult(null);
    setSuggestion(null);
    setEditDraft(null);
    setIsEditing(false);
    try {
      const s = await api.aiSuggestTrade(profile.name, profile.path, suggestModel || undefined);
      setSuggestion(s);
      setEditDraft(s);
    } catch (e) {
      setSuggestError((e as Error).message || 'Failed to get suggestion');
    } finally {
      setSuggestLoading(false);
    }
  }, [suggestModel, profile.name, profile.path]);

  // Build a diff of fields the user changed between the original suggestion and the edit draft.
  const buildEditedFields = useCallback((): Record<string, { before: unknown; after: unknown }> | null => {
    if (!suggestion || !editDraft) return null;
    const diff: Record<string, { before: unknown; after: unknown }> = {};
    const tracked: (keyof api.AiTradeSuggestion)[] = ['side', 'price', 'sl', 'tp', 'lots', 'time_in_force', 'exit_strategy'];
    for (const key of tracked) {
      const before = suggestion[key];
      const after = editDraft[key];
      if (before !== after) {
        diff[key] = { before, after };
      }
    }
    // Check exit_params numeric overrides
    const origParams = suggestion.exit_params || {};
    const draftParams = editDraft.exit_params || {};
    const allParamKeys = new Set([...Object.keys(origParams), ...Object.keys(draftParams)]);
    for (const pk of allParamKeys) {
      if ((origParams as Record<string, number>)[pk] !== (draftParams as Record<string, number>)[pk]) {
        diff[`exit_params.${pk}`] = {
          before: (origParams as Record<string, number>)[pk] ?? null,
          after: (draftParams as Record<string, number>)[pk] ?? null,
        };
      }
    }
    return Object.keys(diff).length > 0 ? diff : null;
  }, [suggestion, editDraft]);

  const handlePlaceOrder = useCallback(async () => {
    if (!editDraft) return;
    setPlaceLoading(true);
    setPlaceResult(null);
    setSuggestError(null);
    try {
      const editedFields = buildEditedFields();
      const res = await api.placeLimitOrder(profile.name, profile.path, {
        side: editDraft.side,
        price: editDraft.price,
        lots: editDraft.lots,
        sl: editDraft.sl,
        tp: editDraft.tp,
        time_in_force: editDraft.time_in_force || 'GTC',
        gtd_time_utc: editDraft.gtd_time_utc,
        comment: `ai_suggest:${editDraft.confidence}`,
        exit_strategy: editDraft.exit_strategy || 'none',
        exit_params: editDraft.exit_params || null,
        suggestion_id: suggestion?.suggestion_id || null,
        edited_fields: editedFields,
      });
      setPlaceResult({ status: res.status, order_id: res.order_id, loop_auto_started: res.loop_auto_started });
      setSuggestion(null);
      setEditDraft(null);
      setIsEditing(false);
    } catch (e) {
      setSuggestError((e as Error).message || 'Failed to place order');
    } finally {
      setPlaceLoading(false);
    }
  }, [buildEditedFields, editDraft, profile.name, profile.path, suggestion]);

  const handleReject = useCallback(() => {
    // Log the rejection to the suggestion tracker (fire-and-forget).
    if (suggestion?.suggestion_id) {
      const editedFields = buildEditedFields();
      api.logSuggestionAction(
        profile.name, suggestion.suggestion_id, 'rejected', editedFields,
      ).catch(() => {});  // best-effort
    }
    setSuggestion(null);
    setEditDraft(null);
    setIsEditing(false);
    setPlaceResult(null);
    setSuggestError(null);
  }, [buildEditedFields, profile.name, suggestion]);

  const updateDraft = useCallback((field: keyof api.AiTradeSuggestion, value: string) => {
    setEditDraft((prev) => {
      if (!prev) return prev;
      const numFields = ['price', 'sl', 'tp', 'lots'] as const;
      if ((numFields as readonly string[]).includes(field)) {
        const n = parseFloat(value);
        if (value === '' || value === '-' || value.endsWith('.')) return { ...prev, [field]: value as unknown as number };
        if (!isNaN(n)) return { ...prev, [field]: n };
        return prev;
      }
      return { ...prev, [field]: value };
    });
  }, []);

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

  const suggestionPanel = (editDraft || suggestLoading) ? (
    <div className="fillmore-suggest-panel fillmore-suggest-panel--rail" ref={suggestPanelRef}>
      <div className="fillmore-suggest-header">
        <div className="fillmore-suggest-title">Trade Suggestion</div>
        {editDraft && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span className={`fillmore-badge ${editDraft.confidence}`}>
              {editDraft.confidence.toUpperCase()}
            </span>
            {(suggestion?.model_used || editDraft.model_used) && (
              <span className="fillmore-badge model">
                {suggestion?.model_used || editDraft.model_used}
              </span>
            )}
          </div>
        )}
      </div>

      {suggestLoading && (
        <div className="fillmore-tool-status" style={{ justifyContent: 'center', padding: '24px 0' }}>
          Analyzing market context and generating trade suggestion...
        </div>
      )}

      {editDraft && !suggestLoading && (
        <div>
          <div className="fillmore-rationale">
            <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>Rationale:</span>{' '}
            {suggestion?.rationale || editDraft.rationale}
          </div>

          <div className="fillmore-order-grid">
            <div>
              <label>Side</label>
              {isEditing ? (
                <select value={editDraft.side} onChange={(e) => updateDraft('side', e.target.value)}>
                  <option value="buy">BUY</option>
                  <option value="sell">SELL</option>
                </select>
              ) : (
                <div className="field-value" style={{ color: editDraft.side === 'buy' ? '#4ade80' : '#f87171' }}>
                  {editDraft.side.toUpperCase()}
                </div>
              )}
            </div>
            <div>
              <label>Limit Price</label>
              {isEditing ? (
                <input type="text" value={String(editDraft.price)} onChange={(e) => updateDraft('price', e.target.value)} />
              ) : (
                <div className="field-value">{editDraft.price}</div>
              )}
            </div>
            <div>
              <label>Lots</label>
              {isEditing ? (
                <input type="text" value={String(editDraft.lots)} onChange={(e) => updateDraft('lots', e.target.value)} />
              ) : (
                <div className="field-value">{editDraft.lots}</div>
              )}
            </div>
            <div>
              <label>Stop Loss</label>
              {isEditing ? (
                <input type="text" value={String(editDraft.sl)} onChange={(e) => updateDraft('sl', e.target.value)} />
              ) : (
                <div className="field-value" style={{ color: '#f87171' }}>{editDraft.sl}</div>
              )}
            </div>
            <div>
              <label>Take Profit</label>
              {isEditing ? (
                <input type="text" value={String(editDraft.tp)} onChange={(e) => updateDraft('tp', e.target.value)} />
              ) : (
                <div className="field-value" style={{ color: '#4ade80' }}>{editDraft.tp}</div>
              )}
            </div>
            <div>
              <label>Expiration</label>
              {isEditing ? (
                <div style={{ display: 'flex', gap: 4 }}>
                  <select
                    value={editDraft.time_in_force || 'GTC'}
                    onChange={(e) => {
                      updateDraft('time_in_force', e.target.value);
                      if (e.target.value === 'GTC') updateDraft('gtd_time_utc', '');
                    }}
                  >
                    <option value="GTC">GTC</option>
                    <option value="GTD">GTD</option>
                  </select>
                  {(editDraft.time_in_force || 'GTC') === 'GTD' && (
                    <input
                      type="datetime-local"
                      value={(editDraft.gtd_time_utc || '').slice(0, 16)}
                      onChange={(e) => updateDraft('gtd_time_utc', e.target.value ? e.target.value + ':00Z' : '')}
                      style={{ flex: 1 }}
                    />
                  )}
                </div>
              ) : (
                <div className="field-value" style={{ color: 'var(--text-secondary)', fontWeight: 400 }}>
                  {editDraft.time_in_force || 'GTC'}
                  {editDraft.gtd_time_utc ? ` (${editDraft.gtd_time_utc.slice(0, 16)})` : ''}
                </div>
              )}
            </div>
          </div>

          {rail?.levels.mid && (
            <div className="fillmore-pip-row">
              Entry: {Math.abs(((editDraft.price - rail.levels.mid) / 0.01)).toFixed(1)}p from mid
              {' | '}SL: {Math.abs(((editDraft.price - editDraft.sl) / 0.01)).toFixed(1)}p
              {' | '}TP: {Math.abs(((editDraft.tp - editDraft.price) / 0.01)).toFixed(1)}p
              {' | '}R:R {(Math.abs((editDraft.tp - editDraft.price) / (editDraft.price - editDraft.sl)) || 0).toFixed(2)}
            </div>
          )}

          {(() => {
            const strategies = (suggestion?.available_exit_strategies || editDraft.available_exit_strategies || {}) as Record<string, api.AiExitStrategyInfo>;
            const currentId = editDraft.exit_strategy || 'none';
            const currentInfo = strategies[currentId];
            const strategyIds = Object.keys(strategies);
            const params = editDraft.exit_params || {};
            return (
              <div className="fillmore-exit-card">
                <div className="exit-header">
                  <div className="exit-label">Managed Exit Strategy</div>
                  {currentId !== 'none' && (
                    <div className="exit-sublabel">applied by run loop once filled</div>
                  )}
                </div>
                {isEditing && strategyIds.length > 0 ? (
                  <select
                    value={currentId}
                    onChange={(e) => {
                      const newId = e.target.value;
                      setEditDraft((prev) => {
                        if (!prev) return prev;
                        const info = strategies[newId];
                        const defaults = (info && info.defaults) || {};
                        return { ...prev, exit_strategy: newId, exit_params: { ...defaults } };
                      });
                    }}
                  >
                    {strategyIds.map((sid) => (
                      <option key={sid} value={sid}>{strategies[sid]?.label || sid}</option>
                    ))}
                  </select>
                ) : (
                  <div style={{ fontSize: '0.95rem', fontWeight: 600, color: currentId === 'none' ? '#facc15' : '#4ade80' }}>
                    {currentInfo?.label || (currentId === 'none' ? 'No managed exit (broker SL/TP only)' : currentId)}
                  </div>
                )}
                {currentInfo?.description && (
                  <div className="exit-desc">{currentInfo.description}</div>
                )}
                {currentId !== 'none' && Object.keys(params).length > 0 && (
                  <div className="exit-params-grid">
                    {Object.entries(params).map(([k, v]) => (
                      <div key={k}>
                        <div className="param-label">{k}</div>
                        {isEditing ? (
                          <input
                            type="text"
                            value={String(v)}
                            onChange={(e) => {
                              const raw = e.target.value;
                              setEditDraft((prev) => {
                                if (!prev) return prev;
                                const nextParams = { ...(prev.exit_params || {}) };
                                const n = parseFloat(raw);
                                if (raw === '' || raw === '-' || raw.endsWith('.')) {
                                  nextParams[k] = raw as unknown as number;
                                } else if (!isNaN(n)) {
                                  nextParams[k] = n;
                                }
                                return { ...prev, exit_params: nextParams };
                              });
                            }}
                          />
                        ) : (
                          <div className="param-value">{String(v)}</div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })()}

          <div className="fillmore-suggest-actions">
            {isEditing ? (
              <button type="button" className="btn btn-primary" onClick={() => setIsEditing(false)}>
                Done Editing
              </button>
            ) : (
              <>
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={() => void handlePlaceOrder()}
                  disabled={placeLoading}
                  style={{ background: '#16a34a', borderColor: '#16a34a' }}
                >
                  {placeLoading ? 'Placing...' : 'Place Order'}
                </button>
                <button type="button" className="btn btn-secondary" onClick={() => setIsEditing(true)}>
                  Edit
                </button>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={handleReject}
                  style={{ color: '#f87171', borderColor: '#f87171' }}
                >
                  Reject
                </button>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  ) : null;

  return (
    <div className="fillmore-page">
      {/* ===== LEFT: Chat + Suggest ===== */}
      <div className="fillmore-main">
        {/* Header */}
        <div className="fillmore-header">
          <div className="fillmore-brand">
            <div className="fillmore-avatar">F</div>
            <div>
              <div className="fillmore-title">Fillmore</div>
              <div className="fillmore-subtitle">USDJPY Trading Assistant</div>
            </div>
          </div>
          <div className="fillmore-header-controls">
            <select
              className="fillmore-model-select"
              value={chatModel}
              disabled={!modelsReady || sending || modelList.length === 0}
              onChange={(e) => {
                const v = e.target.value;
                setChatModel(v);
                saveModelToStorage(profile.path, v);
              }}
            >
              {!modelsReady ? (
                <option value="">Loading…</option>
              ) : (
                modelList.map((id) => (
                  <option key={id} value={id}>{id}</option>
                ))
              )}
            </select>
            {messages.length > 0 && (
              <button
                type="button"
                className="btn btn-secondary"
                style={{ fontSize: '0.82rem', padding: '6px 14px', borderRadius: 8 }}
                disabled={sending || messages.length === 0}
                onClick={() => {
                  setMessages([]);
                  saveChatToStorage(profile.path, []);
                  setError(null);
                }}
              >
                New Chat
              </button>
            )}
          </div>
        </div>

        {/* Error banner */}
        {error && <div className="fillmore-error">{error}</div>}

        {/* Chat scroll area */}
        <div className="fillmore-chat-scroll" ref={scrollRef}>
          {/* Welcome state */}
          {messages.length === 0 && !sending && (
            <div className="fillmore-welcome">
              <div className="fillmore-welcome-avatar">F</div>
              <h2>Fillmore. Let&apos;s go to work.</h2>
              <p>
                Elite USDJPY desk partner. I&apos;ve got your book, the tape, the macro, and the
                headlines — all live. Let&apos;s fill more banks.
              </p>
              <div className="fillmore-chips">
                {[
                  "How's my P&L today?",
                  "What's the current setup?",
                  "Position sizing help",
                  "Analyze recent losses",
                  "Any upcoming events?",
                  "What's the latest news?",
                ].map((chip) => (
                  <button
                    key={chip}
                    type="button"
                    className="fillmore-chip"
                    onClick={() => void send(chip)}
                  >
                    {chip}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Message list */}
          {messages.map((line, i) => (
            <div key={i} className={`fillmore-msg ${line.role}`}>
              <div className="fillmore-msg-avatar">
                {line.role === 'user' ? 'Y' : 'F'}
              </div>
              <div className="fillmore-msg-body">
                <div className="fillmore-msg-name">
                  {line.role === 'user' ? 'You' : 'Fillmore'}
                </div>
                <div className={`fillmore-msg-bubble${line.role === 'assistant' && sending && i === messages.length - 1 && !line.content ? ' streaming' : ''}`}>
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

          {/* Tool status */}
          {toolStatus && (
            <div className="fillmore-tool-status">{toolStatus}</div>
          )}
        </div>

        {/* Input area */}
        <div className="fillmore-input-area">
          <div className="fillmore-input-wrap">
            <textarea
              className="fillmore-textarea"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  void send();
                }
              }}
              placeholder="Ask Fillmore anything..."
              rows={1}
              disabled={sending}
            />
            {sending ? (
              <button
                type="button"
                className="fillmore-send-btn"
                onClick={() => abortRef.current?.abort()}
                title="Stop"
                style={{ background: 'var(--danger)' }}
              >
                &#9632;
              </button>
            ) : (
              <button
                type="button"
                className="fillmore-send-btn"
                onClick={() => void send()}
                disabled={!input.trim()}
                title="Send"
              >
                &#8593;
              </button>
            )}
          </div>
          <div className="fillmore-actions">
            <button
              type="button"
              className="btn btn-secondary"
              disabled={sending || messages.length === 0}
              onClick={() => {
                const lines = messages.map((m) => {
                  const label = m.role === 'user' ? 'You' : 'Fillmore';
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
          </div>
        </div>

      </div>

      {/* ===== RIGHT: Context sidebar ===== */}
      <div className="fillmore-sidebar">
        {/* Generate Suggestion — always visible at top of sidebar */}
        <div className="card">
          <div className="card-heading">Trade Suggestion</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {suggestModelList.length > 0 && (
              <select
                className="fillmore-model-select"
                style={{ width: '100%', fontSize: '0.85rem' }}
                value={suggestModel}
                onChange={(e) => {
                  const v = e.target.value;
                  setSuggestModel(v);
                  saveSuggestModelToStorage(profile.path, v);
                }}
                disabled={suggestLoading}
              >
                {suggestModelList.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            )}
            <button
              type="button"
              className="btn btn-primary"
              disabled={suggestLoading}
              onClick={() => void handleSuggestTrade()}
              style={{ width: '100%', borderRadius: 10, padding: '10px 18px', fontSize: '0.92rem' }}
            >
              {suggestLoading ? 'Analyzing...' : editDraft ? 'New Suggestion' : 'Generate Suggestion'}
            </button>
            {suggestError && <div style={{ color: '#f87171', fontSize: '0.84rem' }}>{suggestError}</div>}
            {placeResult && (
              <div style={{ color: '#4ade80', fontSize: '0.84rem' }}>
                Order {placeResult.status}{placeResult.order_id ? ` (ID: ${placeResult.order_id})` : ''}
                {placeResult.loop_auto_started && (
                  <div style={{ color: '#facc15', marginTop: 4 }}>Loop auto-started for exit management</div>
                )}
              </div>
            )}
          </div>
        </div>

        {suggestionPanel}

        <div className="card">
          <div className="card-heading">Macro Confirmation</div>
          <div className="card-row" style={{ marginBottom: 8 }}>
            Bias: <strong>{rail?.macro.combined_bias || 'n/a'}</strong> ({rail?.macro.confidence || 'n/a'})
          </div>
          <div style={{ display: 'grid', gap: 6 }}>
            <div className="card-row">DXY: {rail?.macro.dxy.value ?? '-'} | 1D {moveArrow(rail?.macro.dxy.one_day)} | 5D {moveArrow(rail?.macro.dxy.five_day)} ({signedPct(rail?.macro.dxy.five_day)})</div>
            <div className="card-row">US10Y: {rail?.macro.us10y.value ?? '-'}% | 1D {moveArrow(rail?.macro.us10y.one_day)} | 5D {moveArrow(rail?.macro.us10y.five_day)}</div>
            <div className="card-row">Oil: {rail?.macro.oil.value ?? '-'} | 1D {moveArrow(rail?.macro.oil.one_day)} | 5D {moveArrow(rail?.macro.oil.five_day)} ({signedPct(rail?.macro.oil.five_day)})</div>
            <div className="card-row">Gold: {rail?.macro.gold.value ?? '-'} | 1D {moveArrow(rail?.macro.gold.one_day)} | 5D {moveArrow(rail?.macro.gold.five_day)}</div>
          </div>
          <div className="card-hint">{macroImplication}</div>
        </div>

        <div className="card">
          <div className="card-heading">Event Risk Countdown</div>
          {(rail?.events || []).length === 0 ? (
            <div className="card-row" style={{ color: 'var(--text-secondary)' }}>No upcoming high-impact USD/JPY events.</div>
          ) : (
            <div style={{ display: 'grid', gap: 8 }}>
              {(rail?.events || []).slice(0, 3).map((ev, idx) => {
                const mins = Number(ev.minutes_to_event || 0);
                const urgent = mins <= 60;
                const soon = mins > 60 && mins <= 180;
                const tone = urgent ? 'var(--danger)' : soon ? 'var(--warning, #d4a017)' : 'var(--text-secondary)';
                return (
                  <div key={`${ev.timestamp_utc}-${idx}`}>
                    <div className="card-row">{ev.currency} {ev.event}</div>
                    <div style={{ color: tone, fontSize: '0.84rem' }}>
                      {countdownLabel(mins)} ({ev.time || ev.date})
                    </div>
                  </div>
                );
              })}
            </div>
          )}
          <div className="card-hint">
            {(rail?.events?.[0]?.minutes_to_event ?? 9999) <= 60
              ? 'Event risk elevated: avoid fresh adds into the release window.'
              : 'No immediate high-impact release window; event risk is manageable.'}
          </div>
        </div>

        <div className="card">
          <div className="card-heading">Key Levels Ladder</div>
          <div className="card-row" style={{ marginBottom: 8 }}>
            Mid: <strong>{rail?.levels.mid ?? '-'}</strong>
          </div>
          <div style={{ display: 'grid', gap: 6 }}>
            <div style={{ color: 'var(--text-secondary)', fontSize: '0.84rem', fontWeight: 600 }}>Resistance</div>
            {(rail?.levels.resistances || []).slice(0, 2).map((r, i) => (
              <div key={`r-${i}`} className="card-row">{r.price} ({r.distance_pips ?? '-'}p)</div>
            ))}
            <div style={{ color: 'var(--text-secondary)', fontSize: '0.84rem', fontWeight: 600, marginTop: 4 }}>Support</div>
            {(rail?.levels.supports || []).slice(0, 2).map((s, i) => (
              <div key={`s-${i}`} className="card-row">{s.price} ({s.distance_pips ?? '-'}p)</div>
            ))}
          </div>
          <div className="card-hint">{ladderImplication}</div>
        </div>

        <div className="card">
          <div className="card-heading">Session &amp; Volatility</div>
          <div style={{ display: 'grid', gap: 6 }}>
            <div className="card-row">Session: {rail?.session_vol.overlap || (rail?.session_vol.active_sessions || []).join(' + ') || 'none'}</div>
            <div className="card-row">Next close: {rail?.session_vol.next_close || '-'}</div>
            <div className="card-row">Spread: {rail?.session_vol.spread_pips ?? '-'}p</div>
            <div className="card-row">ATR regime: {rail?.session_vol.vol_label || 'n/a'} ({rail?.session_vol.vol_ratio ?? '-'}x)</div>
          </div>
          <div className="card-hint">{sessionImplication}</div>
        </div>
      </div>

      {/* Suggestion panel moved inside chat scroll area above */}
    </div>
  );
}
