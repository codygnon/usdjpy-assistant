import { useEffect, useRef, useState, useCallback } from 'react';
import {
  getDashboard, getTradeEvents, getLoopLog, startLoop, stopLoop, getRuntimeState,
  type DashboardState, type TradeEvent, type FilterReport, type ContextItem,
  type PositionInfo, type DailySummary,
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
  state, loopRunning, profileName, onToggleLoop,
}: {
  state: DashboardState | null;
  loopRunning: boolean;
  profileName: string;
  onToggleLoop: () => void;
}) {
  const [utcTime, setUtcTime] = useState(new Date().toISOString().slice(11, 19));
  useEffect(() => {
    const id = setInterval(() => setUtcTime(new Date().toISOString().slice(11, 19)), 1000);
    return () => clearInterval(id);
  }, []);

  const running = loopRunning;
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 12, padding: '8px 16px',
      backgroundColor: colors.panel, borderBottom: `1px solid ${colors.border}`,
    }}>
      <button onClick={onToggleLoop} style={{
        padding: '4px 14px', borderRadius: 4, border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 12,
        backgroundColor: running ? colors.red : colors.green,
        color: '#fff',
      }}>{running ? 'Stop' : 'Start'}</button>
      <StatusDot running={running} />
      <span style={{ color: colors.textSecondary, fontSize: 12, ...mono }}>UTC {utcTime}</span>
      <span style={{ color: colors.textPrimary, fontSize: 13, fontWeight: 600 }}>{state?.preset_name || profileName}</span>
      <span style={{ color: colors.textSecondary, fontSize: 12 }}>Mode: {state?.mode || '—'}</span>
      <div style={{ flex: 1 }} />
      {state && (
        <div style={{ display: 'flex', gap: 12, ...mono, fontSize: 12 }}>
          <span style={{ color: colors.textSecondary }}>Bid: <span style={{ color: colors.textPrimary }}>{state.bid.toFixed(3)}</span></span>
          <span style={{ color: colors.textSecondary }}>Ask: <span style={{ color: colors.textPrimary }}>{state.ask.toFixed(3)}</span></span>
          <span style={{ color: colors.textSecondary }}>Spread: <span style={{ color: colors.textPrimary }}>{state.spread_pips.toFixed(1)}p</span></span>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Context Panel
// ---------------------------------------------------------------------------

function ContextPanel({ items }: { items: ContextItem[] }) {
  const grouped: Record<string, ContextItem[]> = {};
  items.forEach(item => {
    if (!grouped[item.category]) grouped[item.category] = [];
    grouped[item.category].push(item);
  });

  return (
    <div style={{
      backgroundColor: colors.panel, borderRadius: 6, padding: 12,
      border: `1px solid ${colors.border}`, minHeight: 120,
    }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: colors.textSecondary, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>Context</div>
      {Object.entries(grouped).map(([cat, catItems]) => (
        <div key={cat} style={{ marginBottom: 8 }}>
          <div style={{ fontSize: 10, color: colors.blue, fontWeight: 600, marginBottom: 2, textTransform: 'uppercase' }}>{cat}</div>
          {catItems.map(item => (
            <div key={item.key} style={{ display: 'flex', justifyContent: 'space-between', padding: '1px 0' }}>
              <span style={{ color: colors.textSecondary, fontSize: 12 }}>{item.key}</span>
              <span style={{ color: colors.textPrimary, fontSize: 12, ...mono }}>{item.value}</span>
            </div>
          ))}
        </div>
      ))}
      {items.length === 0 && <div style={{ color: colors.textSecondary, fontSize: 12 }}>No context data</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Positions Panel
// ---------------------------------------------------------------------------

function PositionsPanel({ positions }: { positions: PositionInfo[] }) {
  return (
    <div style={{
      backgroundColor: colors.panel, borderRadius: 6, padding: 12,
      border: `1px solid ${colors.border}`, minHeight: 120,
    }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: colors.textSecondary, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>
        Open Positions ({positions.length})
      </div>
      {positions.length === 0 ? (
        <div style={{ color: colors.textSecondary, fontSize: 12 }}>No open positions</div>
      ) : (
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ color: colors.textSecondary, borderBottom: `1px solid ${colors.border}` }}>
              <th style={{ textAlign: 'left', padding: '2px 4px', fontWeight: 500 }}>Side</th>
              <th style={{ textAlign: 'left', padding: '2px 4px', fontWeight: 500 }}>Type</th>
              <th style={{ textAlign: 'right', padding: '2px 4px', fontWeight: 500 }}>Entry</th>
              <th style={{ textAlign: 'right', padding: '2px 4px', fontWeight: 500 }}>Current</th>
              <th style={{ textAlign: 'right', padding: '2px 4px', fontWeight: 500 }}>P&L</th>
              <th style={{ textAlign: 'right', padding: '2px 4px', fontWeight: 500 }}>Age</th>
              <th style={{ textAlign: 'right', padding: '2px 4px', fontWeight: 500 }}>SL</th>
              <th style={{ textAlign: 'right', padding: '2px 4px', fontWeight: 500 }}>TP</th>
              <th style={{ textAlign: 'center', padding: '2px 4px', fontWeight: 500 }}>BE</th>
            </tr>
          </thead>
          <tbody>
            {positions.map(pos => (
              <tr key={pos.trade_id} style={{ borderBottom: `1px solid ${colors.border}22` }}>
                <td style={{ padding: '3px 4px' }}><SideBadge side={pos.side} /></td>
                <td style={{ padding: '3px 4px' }}><EntryTypePill type={pos.entry_type} /></td>
                <td style={{ padding: '3px 4px', textAlign: 'right', ...mono, color: colors.textPrimary }}>{pos.entry_price.toFixed(3)}</td>
                <td style={{ padding: '3px 4px', textAlign: 'right', ...mono, color: colors.textPrimary }}>{pos.current_price.toFixed(3)}</td>
                <td style={{ padding: '3px 4px', textAlign: 'right' }}><PipsValue pips={pos.unrealized_pips} /></td>
                <td style={{ padding: '3px 4px', textAlign: 'right', color: colors.textSecondary, ...mono }}>{pos.age_minutes.toFixed(0)}m</td>
                <td style={{ padding: '3px 4px', textAlign: 'right', color: colors.textSecondary, ...mono }}>{pos.stop_price?.toFixed(3) ?? '—'}</td>
                <td style={{ padding: '3px 4px', textAlign: 'right', color: colors.textSecondary, ...mono }}>{pos.target_price?.toFixed(3) ?? '—'}</td>
                <td style={{ padding: '3px 4px', textAlign: 'center', fontSize: 10, color: pos.breakeven_applied ? colors.green : colors.textSecondary }}>
                  {pos.breakeven_applied ? 'YES' : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
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
// Trade Execution Log
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

function TradeLog({ events }: { events: TradeEvent[] }) {
  const prevCountRef = useRef(events.length);
  const newCount = events.length - prevCountRef.current;
  useEffect(() => { prevCountRef.current = events.length; }, [events.length]);

  return (
    <div style={{
      backgroundColor: colors.panel, borderRadius: 6, padding: 12,
      border: `1px solid ${colors.border}`,
    }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: colors.textSecondary, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>
        Trade Log ({events.length})
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
// Raw Logs (collapsible)
// ---------------------------------------------------------------------------

function RawLogs({ profileName }: { profileName: string }) {
  const [expanded, setExpanded] = useState(false);
  const [content, setContent] = useState('');

  useEffect(() => {
    if (!expanded) return;
    const fetch = () => getLoopLog(profileName, 100).then(r => setContent(r.content)).catch(() => {});
    fetch();
    const id = setInterval(fetch, 3000);
    return () => clearInterval(id);
  }, [expanded, profileName]);

  return (
    <div style={{ backgroundColor: colors.panel, borderRadius: 6, border: `1px solid ${colors.border}`, marginTop: 8 }}>
      <button
        onClick={() => setExpanded(e => !e)}
        style={{
          width: '100%', padding: '6px 12px', textAlign: 'left', border: 'none',
          backgroundColor: 'transparent', color: colors.textSecondary, cursor: 'pointer',
          fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 1,
        }}
      >
        {expanded ? '▼' : '▶'} Raw Logs
      </button>
      {expanded && (
        <pre style={{
          padding: '8px 12px', margin: 0, fontSize: 11, ...mono,
          color: colors.textSecondary, maxHeight: 300, overflowY: 'auto',
          backgroundColor: colors.bg, borderTop: `1px solid ${colors.border}`,
          whiteSpace: 'pre-wrap', wordBreak: 'break-all',
        }}>{content || 'No log output'}</pre>
      )}
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
  const [state, setState] = useState<DashboardState | null>(null);
  const [events, setEvents] = useState<TradeEvent[]>([]);
  const [loopRunning, setLoopRunning] = useState(false);

  // Poll dashboard state at 500ms
  useEffect(() => {
    let mounted = true;
    const poll = () => {
      getDashboard(profileName)
        .then(s => { if (mounted && s && !s.error) setState(s); })
        .catch(() => {});
    };
    poll();
    const id = setInterval(poll, 500);
    return () => { mounted = false; clearInterval(id); };
  }, [profileName]);

  // Poll trade events + loop status at 2000ms
  useEffect(() => {
    let mounted = true;
    const poll = () => {
      getTradeEvents(profileName, 100)
        .then(e => { if (mounted) setEvents(e); })
        .catch(() => {});
      getRuntimeState(profileName)
        .then(s => { if (mounted) setLoopRunning(s.loop_running); })
        .catch(() => {});
    };
    poll();
    const id = setInterval(poll, 2000);
    return () => { mounted = false; clearInterval(id); };
  }, [profileName]);

  const handleToggleLoop = useCallback(async () => {
    try {
      if (loopRunning) {
        await stopLoop(profileName);
        setLoopRunning(false);
      } else {
        await startLoop(profileName, profilePath);
        setLoopRunning(true);
      }
    } catch (e) {
      console.error('Loop toggle error:', e);
    }
  }, [loopRunning, profileName, profilePath]);

  const effectiveState = state ? { ...state, loop_running: loopRunning } : null;

  return (
    <div style={{ backgroundColor: colors.bg, color: colors.textPrimary, minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <style>{`
        @keyframes dashHighlight {
          0% { background-color: ${colors.blue}33; }
          100% { background-color: ${colors.panelHover}; }
        }
      `}</style>

      {/* Header */}
      <HeaderBar state={effectiveState} loopRunning={loopRunning} profileName={profileName} onToggleLoop={handleToggleLoop} />

      {/* Main content */}
      <div style={{ flex: 1, padding: 12, display: 'flex', flexDirection: 'column', gap: 8, overflowY: 'auto' }}>
        {/* Waiting banner: loop running but no data yet */}
        {loopRunning && !state && (
          <div style={{
            padding: '10px 16px', borderRadius: 6, fontSize: 13,
            backgroundColor: colors.amber + '11', color: colors.amber,
            border: `1px solid ${colors.amber}33`,
          }}>
            Loop started — waiting for first poll cycle...
          </div>
        )}
        {/* Not running banner */}
        {!loopRunning && !state && (
          <div style={{
            padding: '10px 16px', borderRadius: 6, fontSize: 13,
            backgroundColor: colors.textSecondary + '11', color: colors.textSecondary,
            border: `1px solid ${colors.textSecondary}33`,
          }}>
            Loop is not running. Press Start to begin.
          </div>
        )}
        {/* Two-column: Context + Positions */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
          <ContextPanel items={state?.context || []} />
          <PositionsPanel positions={state?.positions || []} />
        </div>

        {/* Filter Status */}
        <FilterTable filters={state?.filters || []} />

        {/* Trade Log */}
        <TradeLog events={events} />

        {/* Raw Logs */}
        <RawLogs profileName={profileName} />
      </div>

      {/* Daily Summary Bar */}
      <DailySummaryBar summary={state?.daily_summary || null} />
    </div>
  );
}
