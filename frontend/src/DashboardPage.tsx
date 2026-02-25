import { useEffect, useRef, useState, useCallback } from 'react';
import {
  getDashboard, getTradeEvents, startLoop, stopLoop, getRuntimeState, updateRuntimeState,
  type DashboardState, type TradeEvent, type FilterReport, type ContextItem,
  type DailySummary, type PositionInfo, type RuntimeState,
} from './api';

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------
const colors = {
  bg: '#0f172a',
  panel: '#1e293b',
  panelHover: '#334155',
  border: '#334155',
  textPrimary: '#e2e8f0',
  textSecondary: '#94a3b8',
  green: '#22c55e',
  red: '#ef4444',
  amber: '#f59e0b',
  blue: '#3b82f6',
};

const mono: React.CSSProperties = { fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace' };

function useDocumentVisible(): boolean {
  const [isVisible, setIsVisible] = useState(
    typeof document === 'undefined' ? true : document.visibilityState !== 'hidden'
  );

  useEffect(() => {
    const onVisibilityChange = () => setIsVisible(document.visibilityState !== 'hidden');
    document.addEventListener('visibilitychange', onVisibilityChange);
    return () => document.removeEventListener('visibilitychange', onVisibilityChange);
  }, []);

  return isVisible;
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function StatusDot({ running }: { running: boolean }) {
  return (
    <span style={{
      display: 'inline-block', width: 10, height: 10, borderRadius: '50%',
      backgroundColor: running ? colors.green : colors.red,
      marginRight: 6, boxShadow: running ? `0 0 6px ${colors.green}` : undefined,
    }} />
  );
}

function SideBadge({ side }: { side: string }) {
  const isBuy = side.toLowerCase() === 'buy';
  return (
    <span style={{
      display: 'inline-block', padding: '1px 6px', borderRadius: 3, fontSize: 11, fontWeight: 700,
      backgroundColor: isBuy ? colors.green + '22' : colors.red + '22',
      color: isBuy ? colors.green : colors.red,
      border: `1px solid ${isBuy ? colors.green + '44' : colors.red + '44'}`,
    }}>{side.toUpperCase()}</span>
  );
}

function EntryTypePill({ type }: { type: string | null }) {
  if (!type) return null;
  return (
    <span style={{
      display: 'inline-block', padding: '1px 5px', borderRadius: 3, fontSize: 10,
      backgroundColor: colors.blue + '22', color: colors.blue, border: `1px solid ${colors.blue}44`,
      marginLeft: 4,
    }}>{type}</span>
  );
}

function PipsValue({ pips }: { pips: number }) {
  const color = pips > 0 ? colors.green : pips < 0 ? colors.red : colors.textSecondary;
  return <span style={{ color, ...mono }}>{pips > 0 ? '+' : ''}{pips.toFixed(1)}p</span>;
}

// ---------------------------------------------------------------------------
// Header Bar
// ---------------------------------------------------------------------------

function HeaderBar({
  loopRunning, profileName, mode, tick, onToggleLoop, presetName, onModeChange, staleLabel,
}: {
  loopRunning: boolean;
  profileName: string;
  mode: string;
  tick: { bid: number; ask: number; spread: number } | null;
  onToggleLoop: () => void;
  presetName: string;
  onModeChange: (mode: string) => void;
  staleLabel?: string | null;
}) {
  const [utcTime, setUtcTime] = useState(new Date().toISOString().slice(11, 19));
  useEffect(() => {
    const id = setInterval(() => setUtcTime(new Date().toISOString().slice(11, 19)), 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 12, padding: '8px 16px',
      backgroundColor: colors.panel, borderBottom: `1px solid ${colors.border}`,
    }}>
      <button onClick={onToggleLoop} style={{
        padding: '4px 14px', borderRadius: 4, border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 12,
        backgroundColor: loopRunning ? colors.red : colors.green,
        color: '#fff',
      }}>{loopRunning ? 'Stop' : 'Start'}</button>
      <StatusDot running={loopRunning} />
      <span style={{ color: colors.textSecondary, fontSize: 12, ...mono }}>UTC {utcTime}</span>
      <span style={{ color: colors.textPrimary, fontSize: 13, fontWeight: 600 }}>{presetName || profileName}</span>
      <label style={{ color: colors.textSecondary, fontSize: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
        Mode
        <select
          value={mode || 'DISARMED'}
          onChange={(e) => onModeChange(e.target.value)}
          style={{
            background: colors.bg,
            color: colors.textPrimary,
            border: `1px solid ${colors.border}`,
            borderRadius: 4,
            fontSize: 12,
            padding: '2px 6px',
          }}
        >
          <option value="DISARMED">DISARMED</option>
          <option value="ARMED_MANUAL_CONFIRM">ARMED_MANUAL_CONFIRM</option>
          <option value="ARMED_AUTO_DEMO">ARMED_AUTO_DEMO</option>
        </select>
      </label>
      {staleLabel && <span style={{ color: colors.amber, fontSize: 12 }}>{staleLabel}</span>}
      <div style={{ flex: 1 }} />
      {tick && (
        <div style={{ display: 'flex', gap: 12, ...mono, fontSize: 12 }}>
          <span style={{ color: colors.textSecondary }}>Bid: <span style={{ color: colors.textPrimary }}>{tick.bid.toFixed(3)}</span></span>
          <span style={{ color: colors.textSecondary }}>Ask: <span style={{ color: colors.textPrimary }}>{tick.ask.toFixed(3)}</span></span>
          <span style={{ color: colors.textSecondary }}>Spread: <span style={{ color: colors.textPrimary }}>{tick.spread.toFixed(1)}p</span></span>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Context Panel
// ---------------------------------------------------------------------------

const CONTEXT_CATEGORY_LABELS: Record<string, string> = {
  m3_trend: 'M3 Trend',
  m1_emas: 'M1 EMAs',
  trend: 'Trend',
  zone_entry: 'Zone Entry',
  ema_zone: 'EMA Zone',
  filters: 'Filters',
  price: 'Price',
};

function ContextPanel({ items }: { items: ContextItem[] }) {
  const grouped: Record<string, ContextItem[]> = {};
  items.forEach(item => {
    if (!grouped[item.category]) grouped[item.category] = [];
    grouped[item.category].push(item);
  });

  return (
    <div style={{
      backgroundColor: colors.panel, borderRadius: 6, padding: 14,
      border: `1px solid ${colors.border}`, minHeight: 120,
    }}>
      <div style={{ fontSize: 15, fontWeight: 700, color: colors.textSecondary, marginBottom: 10, textTransform: 'uppercase', letterSpacing: 1 }}>Context</div>
      {Object.entries(grouped).map(([cat, catItems]) => (
        <div key={cat} style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 16, color: colors.blue, fontWeight: 600, marginBottom: 4, textTransform: 'uppercase' }}>{CONTEXT_CATEGORY_LABELS[cat] ?? cat}</div>
          {catItems.map(item => (
            <div key={item.key} style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}>
              <span style={{ color: colors.textSecondary, fontSize: 15 }}>{item.key}</span>
              <span style={{
                color: item.valueColor === 'green' ? colors.green : item.valueColor === 'red' ? colors.red : colors.textPrimary,
                fontSize: 15,
                ...mono,
              }}>{item.value}</span>
            </div>
          ))}
        </div>
      ))}
      {items.length === 0 && <div style={{ color: colors.textSecondary, fontSize: 15 }}>No context data (start loop for live context)</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Positions Panel — uses PositionInfo[] from dashboard state
// ---------------------------------------------------------------------------

function PositionsPanel({ trades }: { trades: PositionInfo[] }) {
  const longs = trades.filter(t => t.side.toLowerCase() === 'buy').length;
  const shorts = trades.filter(t => t.side.toLowerCase() === 'sell').length;
  const [expanded, setExpanded] = useState(true);
  useEffect(() => {
    setExpanded(trades.length <= 3);
  }, [trades.length]);

  return (
    <div style={{
      backgroundColor: colors.panel, borderRadius: 6, padding: 14,
      border: `1px solid ${colors.border}`, minHeight: 120,
    }}>
      <div style={{ fontSize: 15, fontWeight: 700, color: colors.textSecondary, marginBottom: 10, textTransform: 'uppercase', letterSpacing: 1 }}>
        Open Positions
      </div>
      {trades.length === 0 ? (
        <div style={{ color: colors.textSecondary, fontSize: 15 }}>No open positions</div>
      ) : (
        <>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
            <span style={{ color: colors.textPrimary, fontSize: 15 }}>{longs} Longs | {shorts} Shorts</span>
            <button
              type="button"
              onClick={() => setExpanded(e => !e)}
              style={{
                background: 'none', border: `1px solid ${colors.border}`, borderRadius: 4, padding: '4px 8px',
                color: colors.blue, fontSize: 13, cursor: 'pointer',
              }}
            >
              {expanded ? '▼ Hide table' : '▶ Show table'}
            </button>
          </div>
          {expanded && (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 15, marginTop: 10 }}>
              <thead>
                <tr style={{ color: colors.textSecondary, borderBottom: `1px solid ${colors.border}` }}>
                  <th style={{ textAlign: 'left', padding: '4px 6px', fontWeight: 500 }}>Side</th>
                  <th style={{ textAlign: 'right', padding: '4px 6px', fontWeight: 500 }}>Entry</th>
                  <th style={{ textAlign: 'right', padding: '4px 6px', fontWeight: 500 }}>Current</th>
                  <th style={{ textAlign: 'right', padding: '4px 6px', fontWeight: 500 }}>P&L</th>
                  <th style={{ textAlign: 'right', padding: '4px 6px', fontWeight: 500 }}>SL</th>
                  <th style={{ textAlign: 'right', padding: '4px 6px', fontWeight: 500 }}>TP</th>
                  <th style={{ textAlign: 'right', padding: '4px 6px', fontWeight: 500 }}>Size</th>
                </tr>
              </thead>
              <tbody>
                {trades.map(t => {
                  const currentPrice = Number.isFinite(t.current_price) ? t.current_price : t.entry_price;
                  const slPipsAway = t.stop_price != null
                    ? Math.abs(currentPrice - t.stop_price) / 0.01
                    : null;
                  const tpPipsAway = t.target_price != null
                    ? Math.abs(t.target_price - currentPrice) / 0.01
                    : null;
                  return (
                    <tr key={t.trade_id} style={{ borderBottom: `1px solid ${colors.border}22` }}>
                      <td style={{ padding: '4px 6px' }}><SideBadge side={t.side} /></td>
                      <td style={{ padding: '4px 6px', textAlign: 'right', ...mono, color: colors.textPrimary }}>{t.entry_price.toFixed(3)}</td>
                      <td style={{ padding: '4px 6px', textAlign: 'right', ...mono, color: colors.textPrimary }}>{currentPrice.toFixed(3)}</td>
                      <td style={{ padding: '4px 6px', textAlign: 'right' }}>
                        <PipsValue pips={Number(t.unrealized_pips ?? 0)} />
                      </td>
                      <td style={{ padding: '4px 6px', textAlign: 'right', color: colors.textSecondary, ...mono }}>
                        {t.stop_price != null
                          ? `${t.stop_price.toFixed(3)} (${Math.round(slPipsAway ?? 0)}p)`
                          : '—'}
                      </td>
                      <td style={{ padding: '4px 6px', textAlign: 'right', color: colors.textSecondary, ...mono }}>
                        {t.target_price != null
                          ? `${t.target_price.toFixed(3)} (${Math.round(tpPipsAway ?? 0)}p)`
                          : '—'}
                      </td>
                      <td style={{ padding: '4px 6px', textAlign: 'right', color: colors.textSecondary, ...mono }}>
                        {t.size_lots != null ? t.size_lots.toFixed(2) : '—'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Filter Status Table
// ---------------------------------------------------------------------------

function FilterRow({ filter, depth = 0 }: { filter: FilterReport; depth?: number }) {
  const [expanded, setExpanded] = useState(true);
  if (!filter.enabled) return null;
  const hasSubs = filter.sub_filters && filter.sub_filters.length > 0;

  return (
    <>
      <tr
        style={{ borderBottom: `1px solid ${colors.border}22`, cursor: hasSubs ? 'pointer' : 'default' }}
        onClick={hasSubs ? () => setExpanded(e => !e) : undefined}
      >
        <td style={{
          padding: '4px 8px', paddingLeft: 8 + depth * 16,
          borderLeft: `3px solid ${filter.is_clear ? colors.green : colors.red}`,
          color: colors.textPrimary, fontSize: 12,
        }}>
          {hasSubs && <span style={{ marginRight: 4, fontSize: 10 }}>{expanded ? '▼' : '▶'}</span>}
          {filter.display_name}
        </td>
        <td style={{ padding: '4px 8px', ...mono, fontSize: 12, color: colors.textPrimary }}>{filter.current_value}</td>
        <td style={{ padding: '4px 8px', fontSize: 11, color: colors.textSecondary }}>{filter.threshold}</td>
        <td style={{ padding: '4px 8px', textAlign: 'center' }}>
          {filter.is_clear ? (
            <span style={{ color: colors.green, fontSize: 14 }}>✓</span>
          ) : (
            <span title={filter.block_reason || ''} style={{ color: colors.red, fontSize: 14, cursor: 'help' }}>✗</span>
          )}
        </td>
        <td style={{ padding: '4px 8px', fontSize: 11, color: colors.red }}>
          {filter.block_reason && !filter.is_clear ? filter.block_reason : ''}
        </td>
      </tr>
      {hasSubs && expanded && filter.sub_filters.map(sub => (
        <FilterRow key={sub.filter_id} filter={sub} depth={depth + 1} />
      ))}
    </>
  );
}

function FilterTable({ filters }: { filters: FilterReport[] }) {
  const enabledFilters = filters.filter(f => f.enabled);
  if (enabledFilters.length === 0) {
    return (
      <div style={{
        backgroundColor: colors.panel, borderRadius: 6, padding: 12,
        border: `1px solid ${colors.border}`,
      }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: colors.textSecondary, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>Filter Status</div>
        <div style={{ color: colors.textSecondary, fontSize: 12 }}>No active filters (start loop for live filter status)</div>
      </div>
    );
  }
  const allClear = enabledFilters.every(f => f.is_clear);
  const blockers = enabledFilters.filter(f => !f.is_clear).map(f => f.display_name);

  return (
    <div style={{
      backgroundColor: colors.panel, borderRadius: 6, padding: 12,
      border: `1px solid ${colors.border}`,
    }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: colors.textSecondary, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>Filter Status</div>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ color: colors.textSecondary, borderBottom: `1px solid ${colors.border}`, fontSize: 11 }}>
            <th style={{ textAlign: 'left', padding: '4px 8px', fontWeight: 500 }}>Filter</th>
            <th style={{ textAlign: 'left', padding: '4px 8px', fontWeight: 500 }}>Current Value</th>
            <th style={{ textAlign: 'left', padding: '4px 8px', fontWeight: 500 }}>Threshold</th>
            <th style={{ textAlign: 'center', padding: '4px 8px', fontWeight: 500 }}>Status</th>
            <th style={{ textAlign: 'left', padding: '4px 8px', fontWeight: 500 }}>Reason</th>
          </tr>
        </thead>
        <tbody>
          {enabledFilters.map(f => (
            <FilterRow key={f.filter_id} filter={f} />
          ))}
        </tbody>
      </table>
      <div style={{
        marginTop: 8, padding: '4px 8px', fontSize: 12, borderRadius: 4,
        backgroundColor: allClear ? colors.green + '11' : colors.red + '11',
        color: allClear ? colors.green : colors.red,
        border: `1px solid ${allClear ? colors.green + '33' : colors.red + '33'}`,
      }}>
        Entry: {allClear ? 'CLEAR' : `BLOCKED by ${blockers.join(', ')}`}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Trade Execution Log (from trade_events.json)
// ---------------------------------------------------------------------------

function TradeCard({ event, isNew }: { event: TradeEvent; isNew: boolean }) {
  const isClose = event.event_type === 'close';
  const time = event.timestamp_utc?.slice(11, 19) || '';
  return (
    <div style={{
      padding: '6px 10px', marginBottom: 4, borderRadius: 4,
      backgroundColor: colors.panelHover,
      borderLeft: `3px solid ${isClose ? (event.pips && event.pips > 0 ? colors.green : colors.red) : colors.blue}`,
      animation: isNew ? 'dashHighlight 1.5s ease-out' : undefined,
      fontSize: 12,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
        <span style={{ color: colors.textSecondary, ...mono, fontSize: 11 }}>{time}</span>
        <span style={{
          fontSize: 10, fontWeight: 700, padding: '0 4px', borderRadius: 2,
          backgroundColor: isClose ? colors.amber + '22' : colors.blue + '22',
          color: isClose ? colors.amber : colors.blue,
        }}>{isClose ? 'CLOSE' : 'OPEN'}</span>
        <SideBadge side={event.side} />
        <EntryTypePill type={event.entry_type} />
        {event.trigger_type && <span style={{ color: colors.textSecondary, fontSize: 10 }}>{event.trigger_type}</span>}
        <span style={{ ...mono, color: colors.textPrimary }}>{event.price.toFixed(3)}</span>
      </div>
      {isClose && (
        <div style={{ display: 'flex', gap: 10, fontSize: 11, marginTop: 2 }}>
          {event.pips != null && <PipsValue pips={event.pips} />}
          {event.profit != null && (
            <span style={{ color: event.profit > 0 ? colors.green : colors.red, ...mono }}>
              {event.profit > 0 ? '+' : ''}{event.profit.toFixed(2)}
            </span>
          )}
          {event.exit_reason && <span style={{ color: colors.textSecondary }}>{event.exit_reason}</span>}
        </div>
      )}
    </div>
  );
}

function TradeLog({ title, events }: { title: string; events: TradeEvent[] }) {
  const prevCountRef = useRef(events.length);
  const newCount = events.length - prevCountRef.current;
  useEffect(() => { prevCountRef.current = events.length; }, [events.length]);

  return (
    <div style={{
      backgroundColor: colors.panel, borderRadius: 6, padding: 12,
      border: `1px solid ${colors.border}`,
    }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: colors.textSecondary, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>
        {title} ({events.length})
      </div>
      <div style={{ maxHeight: 300, overflowY: 'auto' }}>
        {events.length === 0 ? (
          <div style={{ color: colors.textSecondary, fontSize: 12, padding: 8 }}>No trade events yet</div>
        ) : (
          events.map((ev, i) => (
            <TradeCard key={ev.trade_id + ev.event_type + ev.timestamp_utc} event={ev} isNew={i < Math.max(0, newCount)} />
          ))
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Daily Summary Bar
// ---------------------------------------------------------------------------

function DailySummaryBar({ summary }: { summary: DailySummary | null }) {
  if (!summary || summary.trades_today === 0) {
    return (
      <div style={{
        padding: '6px 16px', backgroundColor: colors.panel,
        borderTop: `1px solid ${colors.border}`, display: 'flex', gap: 16,
        fontSize: 12, color: colors.textSecondary,
      }}>
        Today: No trades
      </div>
    );
  }
  const positive = summary.total_pips >= 0;
  return (
    <div style={{
      padding: '6px 16px', backgroundColor: colors.panel,
      borderTop: `1px solid ${colors.border}`, display: 'flex', gap: 16,
      fontSize: 12, color: positive ? colors.green : colors.red,
    }}>
      <span>Trades: <strong>{summary.trades_today}</strong></span>
      <span>W/L: <strong>{summary.wins}/{summary.losses}</strong></span>
      <span>Win Rate: <strong>{summary.win_rate.toFixed(1)}%</strong></span>
      <span style={mono}>Pips: <strong>{summary.total_pips > 0 ? '+' : ''}{summary.total_pips.toFixed(1)}</strong></span>
      <span style={mono}>Profit: <strong>{summary.total_profit > 0 ? '+' : ''}{summary.total_profit.toFixed(2)}</strong></span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main DashboardPage
// ---------------------------------------------------------------------------

interface DashboardPageProps {
  profileName: string;
  profilePath: string;
}

export default function DashboardPage({ profileName, profilePath }: DashboardPageProps) {
  const isPageVisible = useDocumentVisible();
  const [events, setEvents] = useState<TradeEvent[]>([]);
  const [dashState, setDashState] = useState<DashboardState | null>(null);
  const [runtime, setRuntime] = useState<RuntimeState | null>(null);
  const [trail, setTrail] = useState<Array<{ time: string; spread: number; blocked: number; trend: string }>>([]);
  const [trailExpanded, setTrailExpanded] = useState(false);

  const loopRunning = dashState?.loop_running ?? false;
  const tick = (dashState && Number.isFinite(dashState.bid) && Number.isFinite(dashState.ask))
    ? { bid: dashState.bid, ask: dashState.ask, spread: dashState.spread_pips }
    : null;

  // Poll trade events — 10s
  useEffect(() => {
    if (!isPageVisible) return;
    let mounted = true;
    const poll = () => {
      getTradeEvents(profileName, 30, profilePath)
        .then(e => { if (mounted) setEvents(e); })
        .catch(() => {});
    };
    poll();
    const id = setInterval(poll, 10000);
    return () => { mounted = false; clearInterval(id); };
  }, [profileName, profilePath, isPageVisible]);

  // Poll dashboard state — 10s
  useEffect(() => {
    if (!isPageVisible) return;
    let mounted = true;
    const poll = () => {
      getDashboard(profileName, profilePath)
        .then(s => {
          if (!mounted || !s || s.error) return;
          setDashState(s);
          const enabled = (s.filters || []).filter(f => f.enabled);
          const blocked = enabled.filter(f => !f.is_clear).length;
          const trend = (s.context || []).find(c => c.key.toLowerCase().includes('trend'))?.value ?? '—';
          const spread = Number(s.spread_pips ?? 0);
          const t = String(s.timestamp_utc || new Date().toISOString()).slice(11, 19);
          setTrail(prev => [...prev, { time: t, spread, blocked, trend }].slice(-60));
        })
        .catch(() => {});
    };
    poll();
    const id = setInterval(poll, 10000);
    return () => { mounted = false; clearInterval(id); };
  }, [profileName, profilePath, isPageVisible]);

  useEffect(() => {
    setTrail([]);
  }, [profileName]);

  const handleToggleLoop = useCallback(async () => {
    try {
      if (loopRunning) {
        await stopLoop(profileName);
      } else {
        await startLoop(profileName, profilePath);
      }
      const s = await getDashboard(profileName, profilePath);
      if (s && !s.error) setDashState(s);
    } catch (e) {
      console.error('Loop toggle error:', e);
    }
  }, [loopRunning, profileName, profilePath]);

  const handleModeChange = useCallback(async (mode: string) => {
    try {
      const rt = await getRuntimeState(profileName);
      setRuntime(rt);
      await updateRuntimeState(profileName, mode, rt.kill_switch);
      const s = await getDashboard(profileName, profilePath);
      if (s && !s.error) setDashState(s);
    } catch (e) {
      console.error('Mode change error:', e);
    }
  }, [profileName, profilePath]);

  const context: ContextItem[] = dashState?.context || [];
  const filters: FilterReport[] = dashState?.filters || [];
  const positions: PositionInfo[] = dashState?.positions || [];
  const dailySummary: DailySummary | null = dashState?.daily_summary || null;
  const presetName = dashState?.preset_name || '';
  const openEvents = events.filter((e) => e.event_type === 'open');
  const closedEvents = events.filter((e) => e.event_type === 'close');
  const recentOpenEvents = openEvents.slice(0, 30);
  const recentClosedEvents = closedEvents.slice(0, 30);
  const latestTrail = trail.length > 0 ? trail[trail.length - 1] : null;

  const staleLabel = (() => {
    if (!dashState?.stale) return null;
    const age = dashState.stale_age_seconds;
    if (age == null) return 'Stale data';
    return `Stale ${Math.round(age)}s`;
  })();

  return (
    <div style={{ backgroundColor: colors.bg, color: colors.textPrimary, minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <style>{`
        @keyframes dashHighlight {
          0% { background-color: ${colors.blue}33; }
          100% { background-color: ${colors.panelHover}; }
        }
      `}</style>

      {/* Header — tick comes from dashboard state first, then TA fallback */}
      <HeaderBar
        loopRunning={loopRunning}
        profileName={profileName}
        mode={dashState?.mode || runtime?.mode || 'DISARMED'}
        tick={tick}
        onToggleLoop={handleToggleLoop}
        presetName={presetName}
        onModeChange={handleModeChange}
        staleLabel={staleLabel}
      />

      {/* Main content */}
      <div style={{ flex: 1, padding: 12, display: 'flex', flexDirection: 'column', gap: 8, overflowY: 'auto' }}>
        {dashState?.data_source === 'none' && (
          <div style={{
            backgroundColor: colors.panel,
            border: `1px solid ${colors.border}`,
            borderRadius: 6,
            padding: 12,
            color: colors.textSecondary,
          }}>
            No run-loop dashboard data yet. Start the loop to populate dashboard context and filters.
          </div>
        )}
        {/* Two-column: Context + Trade Log */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
          <ContextPanel items={context} />
          <TradeLog title="Trade Log (Last 30)" events={recentOpenEvents} />
        </div>

        {/* Filter Status */}
        <FilterTable filters={filters} />

        {/* Open Positions */}
        <PositionsPanel trades={positions} />

        {/* Closed Trades */}
        <TradeLog title="Closed Trades (Last 30)" events={recentClosedEvents} />

        {/* Local dashboard history trail (last 60 polls) */}
        <div style={{
          backgroundColor: colors.panel, borderRadius: 6, padding: 12,
          border: `1px solid ${colors.border}`,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10, marginBottom: 8 }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: colors.textSecondary, textTransform: 'uppercase', letterSpacing: 1 }}>
              Recent Dashboard Trail ({trail.length})
            </div>
            <button
              type="button"
              onClick={() => setTrailExpanded((v) => !v)}
              style={{
                border: `1px solid ${colors.border}`,
                background: 'transparent',
                color: colors.blue,
                borderRadius: 4,
                fontSize: 11,
                padding: '2px 8px',
                cursor: 'pointer',
              }}
            >
              {trailExpanded ? 'Minimize' : 'Expand'}
            </button>
          </div>
          {trail.length === 0 ? (
            <div style={{ color: colors.textSecondary, fontSize: 12 }}>No samples yet</div>
          ) : !trailExpanded ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: 16, color: colors.textSecondary, fontSize: 12 }}>
              <span style={mono}>{latestTrail?.time}</span>
              <span>Spread <span style={mono}>{latestTrail?.spread.toFixed(1)}p</span></span>
              <span>Blocked <span style={{ ...mono, color: (latestTrail?.blocked || 0) > 0 ? colors.red : colors.green }}>{latestTrail?.blocked ?? 0}</span></span>
              <span>Trend <span style={{ color: colors.textPrimary }}>{latestTrail?.trend ?? '—'}</span></span>
            </div>
          ) : (
            <div style={{ maxHeight: 180, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ color: colors.textSecondary, borderBottom: `1px solid ${colors.border}` }}>
                    <th style={{ textAlign: 'left', padding: '3px 4px', fontWeight: 500 }}>Time</th>
                    <th style={{ textAlign: 'right', padding: '3px 4px', fontWeight: 500 }}>Spread</th>
                    <th style={{ textAlign: 'right', padding: '3px 4px', fontWeight: 500 }}>Blocked</th>
                    <th style={{ textAlign: 'left', padding: '3px 4px', fontWeight: 500 }}>Trend</th>
                  </tr>
                </thead>
                <tbody>
                  {[...trail].reverse().map((r, i) => (
                    <tr key={`${r.time}-${i}`} style={{ borderBottom: `1px solid ${colors.border}22` }}>
                      <td style={{ padding: '3px 4px', color: colors.textSecondary, ...mono }}>{r.time}</td>
                      <td style={{ padding: '3px 4px', textAlign: 'right', ...mono }}>{r.spread.toFixed(1)}p</td>
                      <td style={{ padding: '3px 4px', textAlign: 'right', ...mono, color: r.blocked > 0 ? colors.red : colors.green }}>{r.blocked}</td>
                      <td style={{ padding: '3px 4px' }}>{r.trend}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {/* Daily Summary Bar */}
      <DailySummaryBar summary={dailySummary} />
    </div>
  );
}
