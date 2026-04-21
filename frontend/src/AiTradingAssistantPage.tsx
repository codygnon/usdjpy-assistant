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

type ExpiryOptionValue =
  | 'no_expiry'
  | '1h'
  | '2h'
  | '3h'
  | '4h'
  | '5h'
  | '6h'
  | '12h'
  | '18h'
  | '1d'
  | '2d'
  | '1w'
  | '1m'
  | '2m'
  | '3m';

const EXPIRY_OPTIONS: { value: ExpiryOptionValue; label: string }[] = [
  { value: 'no_expiry', label: 'No Expiry' },
  { value: '1h', label: '1 Hour' },
  { value: '2h', label: '2 Hours' },
  { value: '3h', label: '3 Hours' },
  { value: '4h', label: '4 Hours' },
  { value: '5h', label: '5 Hours' },
  { value: '6h', label: '6 Hours' },
  { value: '12h', label: '12 Hours' },
  { value: '18h', label: '18 Hours' },
  { value: '1d', label: '1 Day' },
  { value: '2d', label: '2 Days' },
  { value: '1w', label: '1 Week' },
  { value: '1m', label: '1 Month' },
  { value: '2m', label: '2 Months' },
  { value: '3m', label: '3 Months' },
];

const DEFAULT_EXPIRY_OPTION: ExpiryOptionValue = '1d';

function addExpiryDuration(base: Date, option: ExpiryOptionValue): Date | null {
  if (option === 'no_expiry') return null;
  const out = new Date(base.getTime());
  switch (option) {
    case '1h':
      out.setUTCHours(out.getUTCHours() + 1);
      return out;
    case '2h':
      out.setUTCHours(out.getUTCHours() + 2);
      return out;
    case '3h':
      out.setUTCHours(out.getUTCHours() + 3);
      return out;
    case '4h':
      out.setUTCHours(out.getUTCHours() + 4);
      return out;
    case '5h':
      out.setUTCHours(out.getUTCHours() + 5);
      return out;
    case '6h':
      out.setUTCHours(out.getUTCHours() + 6);
      return out;
    case '12h':
      out.setUTCHours(out.getUTCHours() + 12);
      return out;
    case '18h':
      out.setUTCHours(out.getUTCHours() + 18);
      return out;
    case '1d':
      out.setUTCDate(out.getUTCDate() + 1);
      return out;
    case '2d':
      out.setUTCDate(out.getUTCDate() + 2);
      return out;
    case '1w':
      out.setUTCDate(out.getUTCDate() + 7);
      return out;
    case '1m':
      out.setUTCMonth(out.getUTCMonth() + 1);
      return out;
    case '2m':
      out.setUTCMonth(out.getUTCMonth() + 2);
      return out;
    case '3m':
      out.setUTCMonth(out.getUTCMonth() + 3);
      return out;
  }
}

function expiryOptionToOrderTiming(option: ExpiryOptionValue): { time_in_force: 'GTC' | 'GTD'; gtd_time_utc: string | null } {
  if (option === 'no_expiry') {
    return { time_in_force: 'GTC', gtd_time_utc: null };
  }
  const expiresAt = addExpiryDuration(new Date(), option);
  return {
    time_in_force: 'GTD',
    gtd_time_utc: expiresAt ? expiresAt.toISOString() : null,
  };
}

function expiryOptionLabel(option: string | null | undefined): string {
  return EXPIRY_OPTIONS.find((item) => item.value === option)?.label || '1 Day';
}

function inferExpiryOption(timeInForce: string | null | undefined, gtdTimeUtc: string | null | undefined): ExpiryOptionValue {
  const tif = String(timeInForce || '').trim().toUpperCase();
  if (tif === 'GTC') return 'no_expiry';
  if (tif !== 'GTD' || !gtdTimeUtc) return DEFAULT_EXPIRY_OPTION;

  const target = new Date(gtdTimeUtc);
  if (Number.isNaN(target.getTime())) return DEFAULT_EXPIRY_OPTION;
  const now = new Date();
  const targetMs = target.getTime();

  let best: ExpiryOptionValue = DEFAULT_EXPIRY_OPTION;
  let bestDiff = Number.POSITIVE_INFINITY;
  for (const option of EXPIRY_OPTIONS) {
    if (option.value === 'no_expiry') continue;
    const candidate = addExpiryDuration(now, option.value);
    if (!candidate) continue;
    const diff = Math.abs(candidate.getTime() - targetMs);
    if (diff < bestDiff) {
      bestDiff = diff;
      best = option.value;
    }
  }
  return best;
}

function normalizeSuggestionExpiry(suggestion: api.AiTradeSuggestion): api.AiTradeSuggestion {
  const option = inferExpiryOption(suggestion.time_in_force, suggestion.gtd_time_utc);
  const normalized = { ...suggestion, expiry_option: option };
  if (!suggestion.time_in_force || (String(suggestion.time_in_force).toUpperCase() === 'GTD' && !suggestion.gtd_time_utc)) {
    const timing = expiryOptionToOrderTiming(option);
    normalized.time_in_force = timing.time_in_force;
    normalized.gtd_time_utc = timing.gtd_time_utc;
  }
  return normalized;
}

function formatExpirySummary(option: string | null | undefined, gtdTimeUtc: string | null | undefined): string {
  const label = expiryOptionLabel(option);
  if (!gtdTimeUtc || option === 'no_expiry') return label;
  const dt = new Date(gtdTimeUtc);
  if (Number.isNaN(dt.getTime())) return label;
  return `${label} (${dt.toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })})`;
}

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

function formatAutonomousReason(reason: string | null | undefined): string {
  const raw = String(reason || '').trim();
  if (!raw) return 'n/a';
  return raw
    .replace(/^event_blackout:/, 'event blackout: ')
    .replace(/^repeat_setup_within_/, 'repeat setup within ')
    .replace(/^correlation_veto_within_/, 'correlation veto within ')
    .replace(/^loss_streak=/, 'loss streak=')
    .replace(/_/g, ' ');
}

function featurePill(label: string, active: boolean, detail?: string): ReactNode {
  return (
    <div
      key={`${label}-${detail || ''}`}
      style={{
        padding: '6px 8px',
        borderRadius: 8,
        border: active ? '1px solid rgba(74,222,128,0.35)' : '1px solid rgba(255,255,255,0.08)',
        background: active ? 'rgba(74,222,128,0.08)' : 'rgba(255,255,255,0.03)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
        <span style={{ fontSize: '0.75rem', color: '#e2e8f0', fontWeight: 600 }}>{label}</span>
        <span style={{ fontSize: '0.68rem', color: active ? '#4ade80' : 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: 0.5 }}>
          {active ? 'on' : 'off'}
        </span>
      </div>
      {detail && <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: 3, lineHeight: 1.35 }}>{detail}</div>}
    </div>
  );
}

function snapDeltaPips(requestedPrice: number | null | undefined, workingPrice: number | null | undefined): number | null {
  if (requestedPrice == null || workingPrice == null || !Number.isFinite(requestedPrice) || !Number.isFinite(workingPrice)) return null;
  return Math.abs(workingPrice - requestedPrice) / 0.01;
}

// ===========================================================================
// Autonomous Fillmore panel
// ===========================================================================

const AGG_LABELS: { value: api.AutonomousConfig['aggressiveness']; label: string }[] = [
  { value: 'conservative',    label: 'Conservative' },
  { value: 'balanced',        label: 'Balanced' },
  { value: 'aggressive',      label: 'Aggressive' },
];

const MODE_LABELS: { value: api.AutonomousConfig['mode']; label: string; color: string }[] = [
  { value: 'off',    label: 'OFF',    color: 'var(--text-secondary)' },
  { value: 'shadow', label: 'SHADOW', color: '#d4a017' },
  { value: 'paper',  label: 'PAPER',  color: '#4ade80' },
  { value: 'live',   label: 'LIVE',   color: '#f87171' },
];

function AutonomousFillmorePanel({
  profileName,
  modelOptions,
}: {
  profileName: string;
  modelOptions: string[];
}) {
  const [cfg, setCfg] = useState<api.AutonomousConfig | null>(null);
  const [gateModes, setGateModes] = useState<Record<string, api.AutonomousGateMode>>({});
  const [stats, setStats] = useState<api.AutonomousStats | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [showDecisions, setShowDecisions] = useState(false);
  const [showReasoning, setShowReasoning] = useState(false);
  const [reasoning, setReasoning] = useState<api.ReasoningFeed | null>(null);

  // Initial load.
  useEffect(() => {
    let cancelled = false;
    api.getAutonomousConfig(profileName).then((r) => {
      if (cancelled) return;
      setCfg(r.config);
      setGateModes(r.gate_modes);
    }).catch((e) => setErr(String(e?.message || e)));
    return () => { cancelled = true; };
  }, [profileName]);

  // Poll stats every 5s (cheap — reads runtime_state.json).
  useEffect(() => {
    let cancelled = false;
    const tick = () => {
      api.getAutonomousStats(profileName).then((s) => {
        if (!cancelled) setStats(s);
      }).catch(() => { /* silent */ });
    };
    tick();
    const id = window.setInterval(tick, 5000);
    return () => { cancelled = true; window.clearInterval(id); };
  }, [profileName]);

  useEffect(() => {
    if (!showReasoning) return undefined;
    let cancelled = false;
    const tick = () => {
      api.getAutonomousReasoning(profileName).then((r) => {
        if (!cancelled) setReasoning(r);
      }).catch(() => { /* silent */ });
    };
    tick();
    const id = window.setInterval(tick, 10000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [profileName, showReasoning]);

  const save = useCallback(async (patch: Partial<api.AutonomousConfig>) => {
    setBusy(true); setErr(null);
    try {
      const r = await api.updateAutonomousConfig(profileName, patch);
      setCfg(r.config);
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [profileName]);

  const resetThrottle = useCallback(async () => {
    setBusy(true); setErr(null);
    try {
      const nextStats = await api.resetAutonomousThrottle(profileName);
      setStats(nextStats);
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [profileName]);

  if (!cfg) {
    return (
      <div className="card autonomous-panel autonomous-panel--loading">
        <div className="card-heading autonomous-panel__title">Autonomous Fillmore</div>
        <div className="card-row autonomous-panel__muted">Loading…</div>
        {err && <div style={{ color: '#f87171', fontSize: '0.82rem', marginTop: 6 }}>{err}</div>}
      </div>
    );
  }

  const modeMeta = MODE_LABELS.find((m) => m.value === cfg.mode) || MODE_LABELS[0];
  const aggDesc = gateModes[cfg.aggressiveness]?.description;
  const expectedPct = gateModes[cfg.aggressiveness]?.expected_pass_rate_pct;

  const canEnable = cfg.mode !== 'off';
  const today = stats?.today;
  const thr = stats?.throttle;
  const window_ = stats?.window;
  const regime = stats?.risk_regime;
  const perf20 = stats?.performance?.rolling_20;
  const perf20View = perf20 ?? {
    trade_count: 0,
    closed_count: 0,
    win_rate: null,
    avg_win_pips: null,
    avg_loss_pips: null,
    profit_factor: null,
    avg_hold_minutes: null,
    fill_rate_limits: null,
    avg_time_to_fill_sec: null,
    avg_fill_vs_requested_pips: null,
    thesis_intervention_rate: null,
    avg_mae_pips: null,
    avg_mfe_pips: null,
    win_rate_by_confidence_json: {},
    win_rate_by_side_json: {},
    win_rate_by_session_json: {},
    prompt_version_breakdown_json: {},
    updated_utc: null,
  };
  const alerts = stats?.health_alerts || [];
  const orderMetrics = stats?.order_metrics;
  const autonomousModelOptions = modelOptions.includes(cfg.model)
    ? modelOptions
    : [cfg.model, ...modelOptions.filter((m) => m !== cfg.model)];

  // Budget bar color
  const budgetPct = today ? Math.min(100, today.budget_used_pct) : 0;
  const budgetColor = budgetPct >= 90 ? '#f87171' : budgetPct >= 60 ? '#facc15' : '#4ade80';
  const topReasons = Object.entries(window_?.top_block_reasons || {}).slice(0, 3);
  const topTriggerFamilies = Object.entries(window_?.trigger_families || {}).slice(0, 3);
  const familyOrderRows = Object.entries(orderMetrics?.by_trigger_family || {}).slice(0, 4);
  const reasoningCounts = reasoning
    ? {
        suggestions: reasoning.suggestions.length,
        thesisChecks: reasoning.thesis_checks.length,
        reflections: reasoning.reflections.length,
      }
    : null;

  return (
    <div className="card autonomous-panel">
      <div className="card-heading autonomous-panel__header">
        <div>
          <div className="autonomous-panel__eyebrow">Execution Engine</div>
          <span className="autonomous-panel__title">Autonomous Fillmore</span>
          <div className="autonomous-panel__subtitle">
            Continuous trade engine with event gating, defensive thesis review, and self-reflection memory.
          </div>
        </div>
        <span style={{
          fontSize: '0.72rem',
          padding: '2px 8px',
          borderRadius: 999,
          background: 'rgba(255,255,255,0.05)',
          color: modeMeta.color,
          fontWeight: 700,
          letterSpacing: 0.5,
        }}>
          {modeMeta.label}
        </span>
      </div>

      {err && <div style={{ color: '#f87171', fontSize: '0.82rem', marginBottom: 8 }}>{err}</div>}

      {cfg.mode === 'paper' && (
        <div className="autonomous-panel__notice">
          Paper mode sends real orders to your OANDA <strong>practice</strong> account (profile must have{' '}
          <code style={{ fontSize: '0.72rem' }}>oanda_environment: &quot;practice&quot;</code>
          {' '}and a practice API token). Fills and outcomes are normal broker trades and feed Fillmore learning.
        </div>
      )}

      {regime && (
        <div
          style={{
            marginBottom: 12,
            padding: '10px 12px',
            borderRadius: 14,
            border: regime.label === 'normal'
              ? '1px solid rgba(74,222,128,0.20)'
              : '1px solid rgba(250,204,21,0.28)',
            background: regime.label === 'normal'
              ? 'rgba(74,222,128,0.06)'
              : 'linear-gradient(180deg, rgba(250,204,21,0.12), rgba(15,23,42,0.55))',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
            <div>
              <div style={{ fontSize: '0.72rem', letterSpacing: 0.6, textTransform: 'uppercase', color: 'var(--text-secondary)' }}>
                Active Risk Regime
              </div>
              <div style={{ fontSize: '1rem', fontWeight: 700, color: regime.label === 'normal' ? '#4ade80' : '#facc15' }}>
                {regime.label}
                {regime.override_label ? ' · override' : ''}
              </div>
            </div>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', color: '#cbd5e1', fontSize: '0.78rem' }}>
              <div>lots {regime.risk_multiplier.toFixed(2)}x</div>
              <div>cooldown {regime.effective_min_llm_cooldown_sec}s</div>
              <div>max open {regime.effective_max_open_ai_trades}</div>
              <div>sizing {regime.risk_multiplier < 1 ? 'reduced' : 'normal'}</div>
            </div>
          </div>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginTop: 8, fontSize: '0.72rem', color: 'var(--text-secondary)' }}>
            <div>losses {thr?.consecutive_losses ?? 0}</div>
            <div>wins {thr?.consecutive_wins ?? 0}</div>
            <div>recovery wins {(regime.recovery_wins ?? 0)}</div>
            {regime.daily_drawdown_active && <div style={{ color: '#f87171' }}>daily drawdown trigger active</div>}
            {regime.override_until_utc && <div>override until {new Date(regime.override_until_utc).toLocaleTimeString()}</div>}
          </div>
        </div>
      )}

      {/* Master mode + enabled toggle */}
      <div className="autonomous-panel__grid autonomous-panel__grid--two">
        <div className="autonomous-panel__field">
          <label className="autonomous-panel__label">Mode</label>
          <select
            value={cfg.mode}
            disabled={busy}
            onChange={(e) => void save({ mode: e.target.value as api.AutonomousConfig['mode'], enabled: e.target.value !== 'off' })}
            className="autonomous-panel__control"
          >
            {MODE_LABELS.map((m) => (
              <option key={m.value} value={m.value}>{m.label}</option>
            ))}
          </select>
        </div>
        <div className="autonomous-panel__field">
          <label className="autonomous-panel__label">Enabled</label>
          <div className="autonomous-panel__toggle">
            <input
              type="checkbox"
              checked={cfg.enabled && canEnable}
              disabled={busy || !canEnable}
              onChange={(e) => void save({ enabled: e.target.checked })}
            />
            <span className="autonomous-panel__muted">
              {cfg.enabled && canEnable ? 'running' : 'paused'}
            </span>
          </div>
        </div>
      </div>

      {/* Aggressiveness */}
      <div className="autonomous-panel__field">
        <label className="autonomous-panel__label autonomous-panel__label--split">
          <span>Gate Aggressiveness</span>
          {expectedPct != null && (
            <span className="autonomous-panel__muted">
              ~{expectedPct}% pass rate
            </span>
          )}
        </label>
        <select
          value={cfg.aggressiveness}
          disabled={busy}
          onChange={(e) => void save({ aggressiveness: e.target.value as api.AutonomousConfig['aggressiveness'] })}
          className="autonomous-panel__control"
        >
          {AGG_LABELS.map((a) => (
            <option key={a.value} value={a.value}>{a.label}</option>
          ))}
        </select>
        {aggDesc && (
          <div className="autonomous-panel__hint">
            {aggDesc}
          </div>
        )}
      </div>

      <div className="autonomous-panel__feature-grid">
        {featurePill(
          'Event Blackout',
          cfg.event_blackout_enabled,
          cfg.event_blackout_enabled ? `Blocks high-impact USD/JPY events inside ${cfg.event_blackout_minutes}m.` : 'LLM may still fire near scheduled events.',
        )}
        {featurePill(
          'Multi-Trade',
          cfg.multi_trade_enabled,
          cfg.multi_trade_enabled ? `Can return up to ${cfg.max_suggestions_per_call} ideas per call.` : 'One idea max per LLM call.',
        )}
        {featurePill(
          'Correlation Veto',
          cfg.correlation_veto_enabled,
          cfg.correlation_veto_enabled ? `Blocks same-side stacking within ${cfg.correlation_distance_pips}p.` : 'No same-side proximity veto.',
        )}
        {featurePill(
          'Repeat Dedupe',
          cfg.repeat_setup_dedupe_enabled,
          cfg.repeat_setup_dedupe_enabled ? `Avoids repeat setups inside ${cfg.repeat_setup_bucket_pips}p over ${cfg.repeat_setup_window_min}m.` : 'Allows repeat-firing same bucket.',
        )}
        {featurePill(
          'Thesis Monitor',
          true,
          'Runs every ~3 minutes on open Fillmore trades and can only reduce risk.',
        )}
        {featurePill(
          'Reflection Memory',
          true,
          'Closed Fillmore trades generate short postmortems that feed autonomous prompts.',
        )}
      </div>

      {/* Budget + activity today */}
      <div className="autonomous-panel__section autonomous-panel__section--soft">
        <div className="autonomous-panel__section-head">
          <span className="autonomous-panel__section-title">Today</span>
          <span className="autonomous-panel__muted">live runtime</span>
        </div>
        <div className="autonomous-panel__metric-row">
          <span className="autonomous-panel__muted">Today's spend</span>
          <strong>${today?.spend_usd?.toFixed(3) ?? '0.000'} / ${cfg.daily_budget_usd.toFixed(2)}</strong>
        </div>
        <div className="autonomous-panel__bar">
          <div style={{ width: `${budgetPct}%`, height: '100%', background: budgetColor, transition: 'width 0.3s' }} />
        </div>
        <div className="autonomous-panel__stats-grid">
          <div className="autonomous-panel__stat">
            <div className="autonomous-panel__stat-label">LLM calls</div>
            <strong>{today?.llm_calls ?? 0}</strong>
          </div>
          <div className="autonomous-panel__stat">
            <div className="autonomous-panel__stat-label">Trades placed</div>
            <strong>{today?.trades_placed ?? 0}</strong>
          </div>
          <div className="autonomous-panel__stat">
            <div className="autonomous-panel__stat-label">P&L</div>
            <strong style={{ color: (today?.pnl_usd ?? 0) >= 0 ? '#4ade80' : '#f87171' }}>
              ${today?.pnl_usd?.toFixed(2) ?? '0.00'}
            </strong>
          </div>
        </div>
      </div>

      {stats && (
        <div className="autonomous-panel__section">
          <div className="autonomous-panel__section-head">
            <span className="autonomous-panel__section-title">Rolling Performance</span>
            <span className="autonomous-panel__muted">last 20 autonomous closes</span>
          </div>
          {!perf20 && (
            <div className="autonomous-panel__hint">
              Waiting for rolling performance stats to materialize.
            </div>
          )}
          <div className="autonomous-panel__stats-grid">
            <div className="autonomous-panel__stat">
              <div className="autonomous-panel__stat-label">Win rate</div>
              <strong>{perf20View.win_rate != null ? `${(perf20View.win_rate * 100).toFixed(0)}%` : '–'}</strong>
            </div>
            <div className="autonomous-panel__stat">
              <div className="autonomous-panel__stat-label">Profit factor</div>
              <strong>{perf20View.profit_factor != null ? perf20View.profit_factor.toFixed(2) : '–'}</strong>
            </div>
            <div className="autonomous-panel__stat">
              <div className="autonomous-panel__stat-label">Avg hold</div>
              <strong>{perf20View.avg_hold_minutes != null ? `${perf20View.avg_hold_minutes.toFixed(0)}m` : '–'}</strong>
            </div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 8, marginTop: 10 }}>
            <div className="autonomous-panel__chip">avg win {perf20View.avg_win_pips != null ? `${perf20View.avg_win_pips > 0 ? '+' : ''}${perf20View.avg_win_pips.toFixed(1)}p` : '–'}</div>
            <div className="autonomous-panel__chip">avg loss {perf20View.avg_loss_pips != null ? `${perf20View.avg_loss_pips.toFixed(1)}p` : '–'}</div>
            <div className="autonomous-panel__chip">limit fill {perf20View.fill_rate_limits != null ? `${(perf20View.fill_rate_limits * 100).toFixed(0)}%` : '–'}</div>
            <div className="autonomous-panel__chip">market fill vs req {perf20View.avg_fill_vs_requested_pips != null ? `${perf20View.avg_fill_vs_requested_pips > 0 ? '+' : ''}${perf20View.avg_fill_vs_requested_pips.toFixed(2)}p` : '–'}</div>
            <div className="autonomous-panel__chip">avg MAE {perf20View.avg_mae_pips != null ? `${perf20View.avg_mae_pips.toFixed(1)}p` : '–'}</div>
            <div className="autonomous-panel__chip">avg MFE {perf20View.avg_mfe_pips != null ? `${perf20View.avg_mfe_pips.toFixed(1)}p` : '–'}</div>
          </div>
        </div>
      )}

      {/* Gate window stats */}
      {window_ && window_.total > 0 && (
        <div className="autonomous-panel__section">
          <div className="autonomous-panel__section-head">
            <span className="autonomous-panel__section-title">Gate Window</span>
            <span className="autonomous-panel__muted">recent activity</span>
          </div>
          <div className="autonomous-panel__hint">
            Last {window_.total} ticks: <strong style={{ color: 'var(--text-primary)' }}>{window_.pass_rate_pct}%</strong> passed
          </div>
          {topReasons.length > 0 && (
            <div className="autonomous-panel__chip-row">
              {topReasons.map(([reason, count]) => (
                <div key={reason} className="autonomous-panel__chip">
                  {formatAutonomousReason(reason)} · {count}
                </div>
              ))}
            </div>
          )}
          {topTriggerFamilies.length > 0 && (
            <div className="autonomous-panel__chip-row" style={{ marginTop: 8 }}>
              {topTriggerFamilies.map(([family, count]) => (
                <div key={family} className="autonomous-panel__chip">
                  trigger {family.replace(/_/g, ' ')} · {count}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {orderMetrics && (
        <div className="autonomous-panel__section">
          <div className="autonomous-panel__section-head">
            <span className="autonomous-panel__section-title">Execution Mix</span>
            <span className="autonomous-panel__muted">trigger-aware order conversion</span>
          </div>
          <div className="autonomous-panel__chip-row">
            <div className="autonomous-panel__chip">
              suggested M/L {(orderMetrics.suggested.market ?? 0)}/{(orderMetrics.suggested.limit ?? 0)}
            </div>
            <div className="autonomous-panel__chip">
              placed M/L {(orderMetrics.placed.market ?? 0)}/{(orderMetrics.placed.limit ?? 0)}
            </div>
            <div className="autonomous-panel__chip">
              filled M/L {(orderMetrics.filled.market ?? 0)}/{(orderMetrics.filled.limit ?? 0)}
            </div>
            <div className="autonomous-panel__chip">
              dead limits {((orderMetrics.cancelled.limit ?? 0) + (orderMetrics.expired.limit ?? 0))}
            </div>
          </div>
          {familyOrderRows.length > 0 && (
            <div style={{ display: 'grid', gap: 8, marginTop: 10 }}>
              {familyOrderRows.map(([family, row]) => (
                <div
                  key={family}
                  style={{
                    borderRadius: 12,
                    padding: '10px 12px',
                    border: '1px solid rgba(255,255,255,0.08)',
                    background: 'rgba(255,255,255,0.03)',
                    fontSize: '0.78rem',
                    color: '#cbd5e1',
                  }}
                >
                  <div style={{ fontWeight: 700, color: 'var(--text-primary)', marginBottom: 6 }}>
                    {family.replace(/_/g, ' ')}
                  </div>
                  <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                    <div>placed M/L {(row.placed.market ?? 0)}/{(row.placed.limit ?? 0)}</div>
                    <div>filled M/L {(row.filled.market ?? 0)}/{(row.filled.limit ?? 0)}</div>
                    <div>limit fill {row.fill_rate.limit != null ? `${(row.fill_rate.limit * 100).toFixed(0)}%` : '–'}</div>
                    <div>limit ttf {row.avg_time_to_fill_sec.limit != null ? `${Math.round(row.avg_time_to_fill_sec.limit)}s` : '–'}</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Throttle indicator */}
      {thr?.active && (
        <div className="autonomous-panel__throttle">
          Throttled: {thr.reason || 'adaptive cooldown'}
          {thr.until_utc && <span> until {new Date(thr.until_utc).toLocaleTimeString()}</span>}
        </div>
      )}
      {thr?.active && (
        <button
          type="button"
          className="btn btn-secondary"
          disabled={busy}
          onClick={() => void resetThrottle()}
          style={{ width: '100%', fontSize: '0.78rem', padding: '6px 10px', marginBottom: 10 }}
        >
          Reset Throttle
        </button>
      )}

      {alerts.length > 0 && (
        <div className="autonomous-panel__section">
          <div className="autonomous-panel__section-head">
            <span className="autonomous-panel__section-title">Health</span>
            <span className="autonomous-panel__muted">runtime diagnostics</span>
          </div>
          <div style={{ display: 'grid', gap: 8 }}>
            {alerts.map((alert) => (
              <div
                key={alert.code}
                style={{
                  borderRadius: 10,
                  padding: '8px 10px',
                  border: alert.level === 'error' ? '1px solid rgba(248,113,113,0.35)' : '1px solid rgba(250,204,21,0.35)',
                  background: alert.level === 'error' ? 'rgba(248,113,113,0.08)' : 'rgba(250,204,21,0.08)',
                  color: alert.level === 'error' ? '#fecaca' : '#fde68a',
                  fontSize: '0.76rem',
                  lineHeight: 1.4,
                }}
              >
                {alert.msg}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Lot sizing — primary controls */}
      <div className="autonomous-panel__section autonomous-panel__section--soft">
        <div className="autonomous-panel__section-head">
          <span className="autonomous-panel__section-title">Lot Sizing</span>
          <span className="autonomous-panel__muted">
            range {Math.max(1, Math.round((cfg.base_lot_size ?? 5) - (cfg.lot_deviation ?? 4)))}-{Math.min(Math.round(cfg.max_lots_per_trade), Math.round((cfg.base_lot_size ?? 5) + (cfg.lot_deviation ?? 4)))} lots
          </span>
        </div>
        <div className="autonomous-panel__grid autonomous-panel__grid--three">
          <div className="autonomous-panel__field">
            <label className="autonomous-panel__label">Base size</label>
            <input
              type="number" step="1" min="1" max={cfg.max_lots_per_trade}
              defaultValue={cfg.base_lot_size ?? 5}
              onBlur={(e) => void save({ base_lot_size: Number(e.target.value) })}
              className="autonomous-panel__control"
            />
          </div>
          <div className="autonomous-panel__field">
            <label className="autonomous-panel__label">Deviation</label>
            <input
              type="number" step="1" min="0" max={cfg.max_lots_per_trade}
              defaultValue={cfg.lot_deviation ?? 4}
              onBlur={(e) => void save({ lot_deviation: Number(e.target.value) })}
              className="autonomous-panel__control"
            />
          </div>
          <div className="autonomous-panel__field">
            <label className="autonomous-panel__label">Hard ceiling</label>
            <input
              type="number" step="1" min="1"
              defaultValue={cfg.max_lots_per_trade}
              onBlur={(e) => void save({ max_lots_per_trade: Number(e.target.value) })}
              className="autonomous-panel__control"
            />
          </div>
        </div>
        <div className="autonomous-panel__hint">
          LLM sizes each trade to match conviction: {Math.max(1, Math.round((cfg.base_lot_size ?? 5) - (cfg.lot_deviation ?? 4)))} = thin edge, {Math.round(cfg.base_lot_size ?? 5)} = normal, {Math.min(Math.round(cfg.max_lots_per_trade), Math.round((cfg.base_lot_size ?? 5) + (cfg.lot_deviation ?? 4)))} = full send, 0 = skip.
        </div>
      </div>

      {/* Model */}
      <div className="autonomous-panel__field">
        <label className="autonomous-panel__label">Model</label>
        {autonomousModelOptions.length > 0 ? (
          <select
            value={cfg.model}
            disabled={busy}
            onChange={(e) => void save({ model: e.target.value })}
            className="autonomous-panel__control"
          >
            {autonomousModelOptions.map((modelId) => (
              <option key={modelId} value={modelId}>{modelId}</option>
            ))}
          </select>
        ) : (
          <input
            type="text"
            value={cfg.model}
            disabled={busy}
            onChange={(e) => setCfg({ ...cfg, model: e.target.value })}
            onBlur={(e) => void save({ model: e.target.value })}
            className="autonomous-panel__control"
          />
        )}
      </div>


      {/* Recent decisions (diagnostic) */}
      <button
        type="button"
        className="btn btn-secondary autonomous-panel__toggle-btn"
        onClick={() => setShowDecisions((v) => !v)}
      >
        {showDecisions ? 'Hide' : 'Show'} recent gate decisions
      </button>

      {showDecisions && stats?.recent_decisions && (
        <div className="autonomous-panel__log">
          {stats.recent_decisions.length === 0 ? (
            <div style={{ color: 'var(--text-secondary)' }}>no decisions yet</div>
          ) : (
            stats.recent_decisions.slice(0, 30).map((d, i) => {
              const time = new Date(d.t).toLocaleTimeString();
              const color = d.r === 'pass' ? '#4ade80' : d.l === 'hard' ? '#f87171' : d.l === 'throttle' ? '#facc15' : 'var(--text-secondary)';
              return (
                <div key={`${d.t}-${i}`} style={{ color, marginBottom: 2 }}>
                  {time} [{d.l}] {d.why}
                </div>
              );
            })
          )}
        </div>
      )}

      {/* Fillmore's Reasoning panel — displays stored analysis, thesis checks, reflections */}
      <button
        type="button"
        className="btn btn-secondary autonomous-panel__toggle-btn autonomous-panel__toggle-btn--spaced"
        onClick={() => {
          const next = !showReasoning;
          setShowReasoning(next);
          if (next && !reasoning) {
            api.getAutonomousReasoning(profileName).then(setReasoning).catch(() => {});
          }
        }}
      >
        {showReasoning ? 'Hide' : 'Show'} Fillmore&apos;s reasoning
        {reasoningCounts ? ` (${reasoningCounts.suggestions}/${reasoningCounts.thesisChecks}/${reasoningCounts.reflections})` : ''}
      </button>

      {showReasoning && reasoning && (
        <div className="autonomous-panel__reasoning">
          <div className="autonomous-panel__chip-row autonomous-panel__chip-row--legend">
            <div className="autonomous-panel__legend-chip autonomous-panel__legend-chip--suggestions">Suggestions</div>
            <div className="autonomous-panel__legend-chip autonomous-panel__legend-chip--thesis">Thesis monitor</div>
            <div className="autonomous-panel__legend-chip autonomous-panel__legend-chip--reflections">Reflections</div>
            <div className="autonomous-panel__legend-note">Auto-refreshes every 10s while open</div>
          </div>

          {/* Recent suggestions with analysis */}
          {reasoning.suggestions.length > 0 && (
            <>
              <div style={{ fontWeight: 600, marginBottom: 4, color: '#93c5fd' }}>Recent Suggestions</div>
              {reasoning.suggestions.map((s) => {
                const time = s.created_utc ? new Date(s.created_utc).toLocaleTimeString() : '?';
                const lotsVal = s.lots ?? 0;
                const lotsColor = lotsVal >= 7 ? '#4ade80' : lotsVal >= 3 ? '#facc15' : '#94a3b8';
                const outcomeLabel = s.win_loss || s.outcome_status || s.action || '';
                const outcomeColor = s.win_loss === 'win' ? '#4ade80' : s.win_loss === 'loss' ? '#f87171' : 'var(--text-secondary)';
                const analysisMatch = (s.rationale || '').match(/ANALYSIS:\n([\s\S]*)/);
                const analysisText = analysisMatch ? analysisMatch[1].trim() : null;
                const shortRationale = analysisMatch
                  ? (s.rationale || '').slice(0, (s.rationale || '').indexOf('\n\nANALYSIS:')).trim()
                  : (s.rationale || '').trim();
                const snapped = snapDeltaPips(s.requested_price, s.price);
                return (
                  <div key={s.suggestion_id} style={{ marginBottom: 10, borderBottom: '1px solid rgba(255,255,255,0.08)', paddingBottom: 8 }}>
                    <div>
                      <span style={{ color: '#e2e8f0' }}>{time}</span>{' '}
                      <span style={{ fontWeight: 600 }}>{(s.side || '').toUpperCase()}</span>{' '}
                      @{s.price?.toFixed(3)}{' '}
                      <span style={{ color: lotsColor }}>{lotsVal > 0 ? `${lotsVal} lots` : 'skip'}</span>{' '}
                      {outcomeLabel && <span style={{ color: outcomeColor }}>{outcomeLabel}{s.pips != null ? ` (${s.pips > 0 ? '+' : ''}${s.pips.toFixed(1)}p)` : ''}</span>}
                    </div>
                    {shortRationale && <div style={{ color: '#cbd5e1', marginTop: 2 }}>{shortRationale}</div>}
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 4 }}>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>
                        working @{s.price?.toFixed(3)}
                      </div>
                      {s.requested_price != null && (
                        <div style={{ fontSize: '0.7rem', color: snapped && snapped >= 0.1 ? '#fbbf24' : 'var(--text-secondary)' }}>
                          requested @{s.requested_price.toFixed(3)}
                          {snapped != null && snapped >= 0.1 ? ` · snapped ${snapped.toFixed(1)}p` : ''}
                        </div>
                      )}
                      {s.exit_strategy && (
                        <div style={{ fontSize: '0.7rem', color: '#a5b4fc' }}>
                          exit {s.exit_strategy}
                        </div>
                      )}
                    </div>
                    {s.exit_plan && s.exit_plan !== 'default' && (
                      <div style={{ color: '#a5b4fc', marginTop: 2, fontStyle: 'italic' }}>Exit plan: {s.exit_plan}</div>
                    )}
                    {analysisText && (
                      <details style={{ marginTop: 4 }}>
                        <summary style={{ color: '#93c5fd', cursor: 'pointer', fontSize: '0.72rem' }}>Full analysis</summary>
                        <pre style={{ whiteSpace: 'pre-wrap', color: '#94a3b8', fontSize: '0.72rem', margin: '4px 0 0 8px' }}>{analysisText}</pre>
                      </details>
                    )}
                  </div>
                );
              })}
            </>
          )}

          {/* Thesis monitor checks */}
          {reasoning.thesis_checks.length > 0 && (
            <>
              <div style={{ fontWeight: 600, marginBottom: 4, marginTop: 8, color: '#fbbf24' }}>Thesis Monitor</div>
              {reasoning.thesis_checks.slice(0, 10).map((tc) => {
                const time = tc.created_utc ? new Date(tc.created_utc).toLocaleTimeString() : '?';
                const actionColor = tc.action === 'hold' ? '#94a3b8' : tc.action === 'exit_now' ? '#f87171' : '#facc15';
                return (
                  <div key={`tc-${tc.id}`} style={{ marginBottom: 4 }}>
                    <span style={{ color: '#e2e8f0' }}>{time}</span>{' '}
                    <span style={{ color: actionColor, fontWeight: 600 }}>{tc.action}</span>{' '}
                    <span style={{ color: '#94a3b8' }}>{tc.reason}</span>
                    {tc.requested_new_sl != null && <span style={{ color: '#facc15' }}> SL→{tc.requested_new_sl.toFixed(3)}</span>}
                    {tc.requested_scale_out_pct != null && <span style={{ color: '#facc15' }}> scale {tc.requested_scale_out_pct}%</span>}
                    {tc.confidence && <span style={{ color: 'var(--text-secondary)' }}> · {tc.confidence}</span>}
                    {tc.execution_succeeded != null && (
                      <span style={{ color: tc.execution_succeeded ? '#4ade80' : '#f87171' }}>
                        {tc.execution_succeeded ? ' · executed' : ' · not executed'}
                      </span>
                    )}
                  </div>
                );
              })}
            </>
          )}

          {/* Self-reflections */}
          {reasoning.reflections.length > 0 && (
            <>
              <div style={{ fontWeight: 600, marginBottom: 4, marginTop: 8, color: '#a78bfa' }}>Self-Reflections</div>
              {reasoning.reflections.slice(0, 8).map((r) => {
                const time = r.created_utc ? new Date(r.created_utc).toLocaleTimeString() : '?';
                return (
                  <div key={`ref-${r.id}`} style={{ marginBottom: 6, borderLeft: '2px solid #a78bfa', paddingLeft: 6 }}>
                    <div style={{ color: '#e2e8f0', fontSize: '0.72rem' }}>{time}</div>
                    {r.summary_text && <div style={{ color: '#cbd5e1' }}>{r.summary_text}</div>}
                    {r.what_read_right && <div style={{ color: '#4ade80', fontSize: '0.72rem' }}>Right: {r.what_read_right}</div>}
                    {r.what_missed && <div style={{ color: '#f87171', fontSize: '0.72rem' }}>Missed: {r.what_missed}</div>}
                    <div style={{ color: 'var(--text-secondary)', fontSize: '0.68rem' }}>
                      {r.autonomous ? 'autonomous memory' : 'manual Fillmore memory'}
                    </div>
                  </div>
                );
              })}
            </>
          )}

          {reasoning.suggestions.length === 0 && reasoning.thesis_checks.length === 0 && reasoning.reflections.length === 0 && (
            <div style={{ color: 'var(--text-secondary)' }}>No reasoning data yet — Fillmore hasn&apos;t made any decisions.</div>
          )}

          <button
            type="button"
            className="btn btn-secondary"
            style={{ width: '100%', fontSize: '0.72rem', padding: '4px 8px', marginTop: 6 }}
            onClick={() => api.getAutonomousReasoning(profileName).then(setReasoning).catch(() => {})}
          >
            Refresh
          </button>
        </div>
      )}
    </div>
  );
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
              get_ai_suggestion_history: 'Loading Fillmore history...',
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
      const normalized = normalizeSuggestionExpiry(s);
      setSuggestion(normalized);
      setEditDraft(normalized);
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
    const tracked: (keyof api.AiTradeSuggestion)[] = ['side', 'price', 'sl', 'tp', 'lots', 'time_in_force', 'gtd_time_utc', 'exit_strategy', 'expiry_option'];
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
      const timing = expiryOptionToOrderTiming((editDraft.expiry_option as ExpiryOptionValue | undefined) || DEFAULT_EXPIRY_OPTION);
      const res = await api.placeLimitOrder(profile.name, profile.path, {
        side: editDraft.side,
        price: editDraft.price,
        lots: editDraft.lots,
        sl: editDraft.sl,
        tp: editDraft.tp,
        time_in_force: timing.time_in_force,
        gtd_time_utc: timing.gtd_time_utc,
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

  const updateExpiryOption = useCallback((option: ExpiryOptionValue) => {
    setEditDraft((prev) => {
      if (!prev) return prev;
      const timing = expiryOptionToOrderTiming(option);
      return {
        ...prev,
        expiry_option: option,
        time_in_force: timing.time_in_force,
        gtd_time_utc: timing.gtd_time_utc,
      };
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
                    value={(editDraft.expiry_option as ExpiryOptionValue | undefined) || DEFAULT_EXPIRY_OPTION}
                    onChange={(e) => updateExpiryOption(e.target.value as ExpiryOptionValue)}
                  >
                    {EXPIRY_OPTIONS.map((option) => (
                      <option key={option.value} value={option.value}>{option.label}</option>
                    ))}
                  </select>
                </div>
              ) : (
                <div className="field-value" style={{ color: 'var(--text-secondary)', fontWeight: 400 }}>
                  {formatExpirySummary(editDraft.expiry_option, editDraft.gtd_time_utc)}
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
        {/* Autonomous Fillmore — when enabled, trades happen without manual review */}
        <AutonomousFillmorePanel profileName={profile.name} modelOptions={suggestModelList} />

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
