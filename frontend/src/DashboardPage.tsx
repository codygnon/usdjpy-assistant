import { useEffect, useRef, useState, useCallback } from 'react';
import {
  getDashboard, getTradeEvents, getLoopLog, startLoop, stopLoop, getRuntimeState,
  getOpenTrades, getTechnicalAnalysis, getTrades, getFilterConfig,
  type DashboardState, type TradeEvent, type FilterReport, type ContextItem,
  type DailySummary, type OpenTrade, type RuntimeState, type TechnicalAnalysis,
  type FilterConfig,
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
  loopRunning, profileName, mode, tick, onToggleLoop, presetName,
}: {
  loopRunning: boolean;
  profileName: string;
  mode: string;
  tick: { bid: number; ask: number; spread: number } | null;
  onToggleLoop: () => void;
  presetName: string;
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
      <span style={{ color: colors.textSecondary, fontSize: 12 }}>Mode: {mode || '—'}</span>
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
          <div style={{ fontSize: 13, color: colors.blue, fontWeight: 600, marginBottom: 4, textTransform: 'uppercase' }}>{CONTEXT_CATEGORY_LABELS[cat] ?? cat}</div>
          {catItems.map(item => (
            <div key={item.key} style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}>
              <span style={{ color: colors.textSecondary, fontSize: 15 }}>{item.key}</span>
              <span style={{ color: colors.textPrimary, fontSize: 15, ...mono }}>{item.value}</span>
            </div>
          ))}
        </div>
      ))}
      {items.length === 0 && <div style={{ color: colors.textSecondary, fontSize: 15 }}>No context data (start loop for live context)</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Positions Panel — uses OpenTrade[] directly from getOpenTrades
// ---------------------------------------------------------------------------

function PositionsPanel({ trades, tick }: { trades: OpenTrade[]; tick: { bid: number; ask: number } | null }) {
  const mid = tick ? (tick.bid + tick.ask) / 2 : null;
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
                  const isBuy = t.side.toLowerCase() === 'buy';
                  const currentPrice = mid ?? t.entry_price;
                  const plPips = isBuy
                    ? (currentPrice - t.entry_price) / 0.01
                    : (t.entry_price - currentPrice) / 0.01;
                  return (
                    <tr key={t.trade_id} style={{ borderBottom: `1px solid ${colors.border}22` }}>
                      <td style={{ padding: '4px 6px' }}><SideBadge side={t.side} /></td>
                      <td style={{ padding: '4px 6px', textAlign: 'right', ...mono, color: colors.textPrimary }}>{t.entry_price.toFixed(3)}</td>
                      <td style={{ padding: '4px 6px', textAlign: 'right', ...mono, color: colors.textPrimary }}>{currentPrice.toFixed(3)}</td>
                      <td style={{ padding: '4px 6px', textAlign: 'right' }}>
                        {t.unrealized_pl != null
                          ? <span style={{ color: t.unrealized_pl >= 0 ? colors.green : colors.red, ...mono }}>{t.unrealized_pl >= 0 ? '+' : ''}{t.unrealized_pl.toFixed(2)}</span>
                          : <PipsValue pips={plPips} />
                        }
                      </td>
                      <td style={{ padding: '4px 6px', textAlign: 'right', color: colors.textSecondary, ...mono }}>{t.stop_price?.toFixed(3) ?? '—'}</td>
                      <td style={{ padding: '4px 6px', textAlign: 'right', color: colors.textSecondary, ...mono }}>{t.target_price?.toFixed(3) ?? '—'}</td>
                      <td style={{ padding: '4px 6px', textAlign: 'right', color: colors.textSecondary, ...mono }}>{t.size_lots}</td>
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
// Recent Trades (from DB)
// ---------------------------------------------------------------------------

function RecentTradesPanel({ profileName, profilePath }: { profileName: string; profilePath: string }) {
  const [trades, setTrades] = useState<Record<string, unknown>[]>([]);

  useEffect(() => {
    const fetch = () => {
      getTrades(profileName, 20, profilePath)
        .then(r => setTrades(r.trades))
        .catch(() => {});
    };
    fetch();
    const id = setInterval(fetch, 10000);
    return () => clearInterval(id);
  }, [profileName, profilePath]);

  return (
    <div style={{
      backgroundColor: colors.panel, borderRadius: 6, padding: 12,
      border: `1px solid ${colors.border}`,
    }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: colors.textSecondary, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>
        Recent Trades ({trades.length})
      </div>
      {trades.length === 0 ? (
        <div style={{ color: colors.textSecondary, fontSize: 12, padding: 8 }}>No trades recorded yet</div>
      ) : (
        <div style={{ maxHeight: 300, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ color: colors.textSecondary, borderBottom: `1px solid ${colors.border}`, fontSize: 11 }}>
                <th style={{ textAlign: 'left', padding: '2px 4px', fontWeight: 500 }}>Time</th>
                <th style={{ textAlign: 'left', padding: '2px 4px', fontWeight: 500 }}>Side</th>
                <th style={{ textAlign: 'right', padding: '2px 4px', fontWeight: 500 }}>Entry</th>
                <th style={{ textAlign: 'right', padding: '2px 4px', fontWeight: 500 }}>Exit</th>
                <th style={{ textAlign: 'right', padding: '2px 4px', fontWeight: 500 }}>Pips</th>
                <th style={{ textAlign: 'right', padding: '2px 4px', fontWeight: 500 }}>Profit</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((t, i) => {
                const pips = t.pips != null ? Number(t.pips) : null;
                const profit = t.profit != null ? Number(t.profit) : null;
                const ts = String(t.timestamp_utc || '').slice(5, 16);
                return (
                  <tr key={String(t.trade_id || i)} style={{ borderBottom: `1px solid ${colors.border}22` }}>
                    <td style={{ padding: '3px 4px', color: colors.textSecondary, ...mono, fontSize: 11 }}>{ts}</td>
                    <td style={{ padding: '3px 4px' }}><SideBadge side={String(t.side || '')} /></td>
                    <td style={{ padding: '3px 4px', textAlign: 'right', ...mono, color: colors.textPrimary }}>{Number(t.entry_price || 0).toFixed(3)}</td>
                    <td style={{ padding: '3px 4px', textAlign: 'right', ...mono, color: colors.textPrimary }}>
                      {t.exit_price != null ? Number(t.exit_price).toFixed(3) : '—'}
                    </td>
                    <td style={{ padding: '3px 4px', textAlign: 'right' }}>
                      {pips != null ? <PipsValue pips={pips} /> : <span style={{ color: colors.textSecondary }}>—</span>}
                    </td>
                    <td style={{ padding: '3px 4px', textAlign: 'right', ...mono }}>
                      {profit != null
                        ? <span style={{ color: profit >= 0 ? colors.green : colors.red }}>{profit >= 0 ? '+' : ''}{profit.toFixed(2)}</span>
                        : <span style={{ color: colors.textSecondary }}>—</span>
                      }
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
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
  // Independently fetched data — same endpoints Run/Status uses
  const [runtime, setRuntime] = useState<RuntimeState | null>(null);
  const [openTrades, setOpenTrades] = useState<OpenTrade[]>([]);
  const [taData, setTaData] = useState<TechnicalAnalysis | null>(null);
  const [events, setEvents] = useState<TradeEvent[]>([]);

  // Dashboard-specific data (only available when loop is running and writes dashboard_state.json)
  const [dashState, setDashState] = useState<DashboardState | null>(null);

  // Filter config from profile JSON (used as fallback when loop not running)
  const [filterConfig, setFilterConfig] = useState<FilterConfig | null>(null);

  const loopRunning = runtime?.loop_running ?? false;
  const tick = taData?.current_tick
    ? { bid: taData.current_tick.bid, ask: taData.current_tick.ask, spread: taData.current_tick.spread_pips }
    : null;

  // Poll runtime state — 3s
  useEffect(() => {
    let mounted = true;
    const poll = () => {
      getRuntimeState(profileName)
        .then(s => { if (mounted) setRuntime(s); })
        .catch(() => {});
    };
    poll();
    const id = setInterval(poll, 3000);
    return () => { mounted = false; clearInterval(id); };
  }, [profileName]);

  // Poll open positions — 5s
  useEffect(() => {
    let mounted = true;
    const poll = () => {
      getOpenTrades(profileName, profilePath)
        .then(t => { if (mounted) setOpenTrades(t); })
        .catch(() => {});
    };
    poll();
    const id = setInterval(poll, 5000);
    return () => { mounted = false; clearInterval(id); };
  }, [profileName, profilePath]);

  // Poll TA data (tick + indicators) — 3s
  useEffect(() => {
    let mounted = true;
    const poll = () => {
      getTechnicalAnalysis(profileName, profilePath)
        .then(ta => { if (mounted) setTaData(ta); })
        .catch(() => {});
    };
    poll();
    const id = setInterval(poll, 3000);
    return () => { mounted = false; clearInterval(id); };
  }, [profileName, profilePath]);

  // Poll trade events — 5s
  useEffect(() => {
    let mounted = true;
    const poll = () => {
      getTradeEvents(profileName, 100)
        .then(e => { if (mounted) setEvents(e); })
        .catch(() => {});
    };
    poll();
    const id = setInterval(poll, 5000);
    return () => { mounted = false; clearInterval(id); };
  }, [profileName]);

  // Poll filter config from profile JSON — 10s (slow, only changes on settings edit)
  useEffect(() => {
    let mounted = true;
    const poll = () => {
      getFilterConfig(profileName, profilePath)
        .then(fc => { if (mounted) setFilterConfig(fc); })
        .catch(() => {});
    };
    poll();
    const id = setInterval(poll, 10000);
    return () => { mounted = false; clearInterval(id); };
  }, [profileName, profilePath]);

  // Poll dashboard state for filters/context (when loop writes dashboard_state.json) — 3s
  useEffect(() => {
    let mounted = true;
    const poll = () => {
      getDashboard(profileName, profilePath)
        .then(s => { if (mounted && s && !s.error) setDashState(s); })
        .catch(() => {});
    };
    poll();
    const id = setInterval(poll, 3000);
    return () => { mounted = false; clearInterval(id); };
  }, [profileName, profilePath]);

  const handleToggleLoop = useCallback(async () => {
    try {
      if (loopRunning) {
        await stopLoop(profileName);
      } else {
        await startLoop(profileName, profilePath);
      }
      getRuntimeState(profileName).then(setRuntime).catch(() => {});
    } catch (e) {
      console.error('Loop toggle error:', e);
    }
  }, [loopRunning, profileName, profilePath]);

  // Build context: use loop's context when available, otherwise build preset-relevant from TA
  const context: ContextItem[] = (() => {
    if (dashState?.context && dashState.context.length > 0) return dashState.context;
    if (!taData) return [];
    const items: ContextItem[] = [];
    const preset = (dashState?.preset_name || '').toLowerCase();
    const isTrial4or5or6 = preset.includes('trial_4') || preset.includes('trial_5') || preset.includes('trial_6') || preset.includes('trial 4') || preset.includes('trial 5') || preset.includes('trial 6');
    const isTrial2or3 = preset.includes('trial_2') || preset.includes('trial_3') || preset.includes('trial 2') || preset.includes('trial 3') || preset.includes('hybrid') || preset.includes('counter_trend');

    // Price — always relevant
    if (tick) {
      items.push({ key: 'Bid', value: tick.bid.toFixed(3), category: 'price' });
      items.push({ key: 'Ask', value: tick.ask.toFixed(3), category: 'price' });
      items.push({ key: 'Spread', value: `${tick.spread.toFixed(1)}p`, category: 'price' });
    }

    const lastEma = (tf: string, name: string): number | null => {
      const series = taData.timeframes[tf]?.all_emas?.[name];
      return series && series.length > 0 ? series[series.length - 1].value : null;
    };
    const prevEma = (tf: string, name: string): number | null => {
      const series = taData.timeframes[tf]?.all_emas?.[name];
      return series && series.length > 1 ? series[series.length - 2].value : null;
    };
    const pip = 0.01;

    if (isTrial4or5or6) {
      // M3 Trend block: regime + EMAs that determine it + slope + 20 SMA
      const m3 = taData.timeframes['M3'];
      if (m3 && !m3.error) {
        const m3Regime = (m3.regime || 'unknown').toUpperCase();
        items.push({ key: 'M3 Trend', value: m3Regime, category: 'm3_trend' });
        const m3e5 = lastEma('M3', 'ema5'), m3e9 = lastEma('M3', 'ema9'), m3e21 = lastEma('M3', 'ema21');
        if (m3e5 != null) items.push({ key: 'M3 EMA 5', value: m3e5.toFixed(3), category: 'm3_trend' });
        if (m3e9 != null) items.push({ key: 'M3 EMA 9', value: m3e9.toFixed(3), category: 'm3_trend' });
        if (m3e21 != null) items.push({ key: 'M3 EMA 21', value: m3e21.toFixed(3), category: 'm3_trend' });
        const m3Ema9Series = m3.all_emas?.ema9;
        const m3Slope = m3Ema9Series && m3Ema9Series.length >= 2
          ? m3Ema9Series[m3Ema9Series.length - 1].value - m3Ema9Series[m3Ema9Series.length - 2].value
          : null;
        items.push({ key: 'M3 Slope', value: m3Slope != null ? (m3Slope > 0 ? '+' : m3Slope < 0 ? '-' : '0') : '—', category: 'm3_trend' });
        const m3Sma20 = m3.bollinger_series?.middle;
        const m3Sma20Val = m3Sma20 && m3Sma20.length > 0 ? m3Sma20[m3Sma20.length - 1].value : null;
        if (m3Sma20Val != null) items.push({ key: 'M3 20 SMA', value: m3Sma20Val.toFixed(3), category: 'm3_trend' });
      }
      // M1 EMAs 5/9/21/34
      for (const p of ['ema5', 'ema9', 'ema21', 'ema34']) {
        const v = lastEma('M1', p);
        if (v != null) items.push({ key: `M1 ${p.replace('ema', 'EMA ')}`, value: v.toFixed(3), category: 'm1_emas' });
      }
      // M1 EMA Zone Entry: EMA5 vs EMA9 (legacy)
      const e5 = lastEma('M1', 'ema5'), e9 = lastEma('M1', 'ema9');
      if (e5 != null && e9 != null) items.push({ key: 'EMA5-9 Spread', value: `${((e5 - e9) / pip).toFixed(1)}p`, category: 'zone_entry' });
      // EMA Zone Filter: EMA9 vs EMA17 spread, direction, change
      const e17 = lastEma('M1', 'ema17');
      if (e9 != null && e17 != null) {
        const spread917 = (e9 - e17) / pip;
        items.push({ key: 'EMA9-17 Spread', value: `${spread917.toFixed(1)}p`, category: 'ema_zone' });
        const prev9 = prevEma('M1', 'ema9');
        if (prev9 != null) {
          const dir = e9 > prev9 ? 'UP' : e9 < prev9 ? 'DOWN' : 'FLAT';
          items.push({ key: 'EMA9 Direction', value: dir, category: 'ema_zone' });
          const prevE17 = prevEma('M1', 'ema17');
          if (prevE17 != null) {
            const prevSpread = (prev9 - prevE17) / pip;
            const change = spread917 - prevSpread;
            items.push({ key: 'Spread Change', value: `${change > 0 ? '+' : ''}${change.toFixed(1)}p`, category: 'ema_zone' });
          }
        }
      }
      // ATR
      const m1 = taData.timeframes['M1'];
      if (m1?.atr?.value_pips != null) items.push({ key: 'M1 ATR', value: `${m1.atr.value_pips.toFixed(1)}p`, category: 'filters' });
      if (m3?.atr?.value_pips != null) items.push({ key: 'M3 ATR', value: `${m3.atr.value_pips.toFixed(1)}p`, category: 'filters' });
    } else if (isTrial2or3) {
      // M5 Trend
      const m5 = taData.timeframes['M5'];
      if (m5 && !m5.error) items.push({ key: 'M5 Regime', value: m5.regime, category: 'trend' });
      // M1 EMAs: 9, 13, 21
      for (const p of ['ema9', 'ema13', 'ema21']) {
        const v = lastEma('M1', p);
        if (v != null) items.push({ key: `M1 ${p.toUpperCase()}`, value: v.toFixed(3), category: 'zone_entry' });
      }
    } else {
      // Generic: show regime + RSI + ATR for each timeframe
      for (const [tf, d] of Object.entries(taData.timeframes)) {
        if (d.error) continue;
        items.push({ key: `${tf} Regime`, value: d.regime, category: tf });
        if (d.rsi?.value != null) items.push({ key: `${tf} RSI`, value: `${d.rsi.value.toFixed(1)} (${d.rsi.zone})`, category: tf });
        if (d.atr?.value_pips != null) items.push({ key: `${tf} ATR`, value: `${d.atr.value_pips.toFixed(1)}p`, category: tf });
      }
    }
    return items;
  })();

  // Filters: always use the dashboard API filters (same source as run loop). Fall back to filterConfig only when API returns none.
  const filters: FilterReport[] = (() => {
    const apiFilters = dashState?.filters;
    if (apiFilters && apiFilters.length > 0) return apiFilters as FilterReport[];

    const fc = filterConfig?.filters;
    if (!fc) return [];

    const result: FilterReport[] = [];
    const mkFilter = (id: string, name: string, enabled: boolean, isClear: boolean, currentValue: string, threshold: string, blockReason: string | null): FilterReport => ({
      filter_id: id, display_name: name, enabled, is_clear: isClear,
      current_value: currentValue, threshold, block_reason: blockReason,
      sub_filters: [], metadata: {},
    });

    // Spread
    if (fc.spread) {
      const maxPips = Number(fc.spread.max_pips ?? 5);
      const cur = tick?.spread ?? 0;
      const ok = cur <= maxPips;
      result.push(mkFilter('spread', 'Spread', true, tick ? ok : true,
        tick ? `${cur.toFixed(1)}p` : '—', `Max: ${maxPips.toFixed(1)}p`,
        tick && !ok ? `${cur.toFixed(1)}p > ${maxPips.toFixed(1)}p` : null));
    }

    // Session Filter
    if (fc.session_filter?.enabled) {
      const sessions = (fc.session_filter.sessions as string[]) || [];
      const now = new Date();
      const h = now.getUTCHours();
      const inTokyo = h >= 0 && h < 9;
      const inLondon = h >= 7 && h < 16;
      const inNY = h >= 13 && h < 22;
      const active: string[] = [];
      if (inTokyo) active.push('Tokyo');
      if (inLondon) active.push('London');
      if (inNY) active.push('NewYork');
      const ok = active.some(s => sessions.includes(s));
      result.push(mkFilter('session', 'Session Filter', true, ok,
        active.length > 0 ? active.join(', ') : 'Off-hours',
        sessions.join(', '),
        !ok ? 'Outside allowed sessions' : null));
    }

    // EMA Zone Filter
    if (fc.ema_zone_filter?.enabled) {
      const threshold = Number(fc.ema_zone_filter.threshold ?? 0.35);
      result.push(mkFilter('ema_zone', 'EMA Zone Filter', true, true,
        '—', `Score ≥ ${threshold.toFixed(2)}`, null));
    }

    // Rolling Danger Zone
    if (fc.rolling_danger_zone?.enabled) {
      const pct = Number(fc.rolling_danger_zone.pct ?? 0.15);
      const lookback = Number(fc.rolling_danger_zone.lookback ?? 100);
      result.push(mkFilter('rolling_danger', 'Rolling Danger Zone', true, true,
        '—', `${(pct * 100).toFixed(0)}% of ${lookback}-bar range`, null));
    }

    // RSI Divergence
    if (fc.rsi_divergence?.enabled) {
      result.push(mkFilter('rsi_div', 'RSI Divergence', true, true, '—', 'Active', null));
    }

    // Tiered ATR (Trial #4)
    if (fc.tiered_atr?.enabled) {
      const below = Number(fc.tiered_atr.block_below ?? 4);
      const allMax = Number(fc.tiered_atr.allow_all_max ?? 12);
      const pbMax = Number(fc.tiered_atr.pullback_max ?? 15);
      const m3atr = taData?.timeframes['M3']?.atr?.value_pips;
      let curVal = '—';
      let ok = true;
      let reason: string | null = null;
      if (m3atr != null) {
        curVal = `${m3atr.toFixed(1)}p`;
        if (m3atr < below || m3atr > pbMax) { ok = false; reason = `${curVal} outside ${below}-${pbMax}p`; }
        else if (m3atr > allMax) { reason = `${curVal} > ${allMax}p (pullback only)`; }
      }
      result.push(mkFilter('tiered_atr', 'Tiered ATR', true, ok, curVal,
        `<${below}p block, ${below}-${allMax}p all, ${allMax}-${pbMax}p PB only, >${pbMax}p block`, reason));
    }

    // M1 ATR (Trial #5)
    if (fc.m1_atr?.enabled) {
      const m1atr = taData?.timeframes['M1']?.atr?.value_pips;
      const tokyoMin = Number(fc.m1_atr.tokyo_min ?? 3);
      const londonMin = Number(fc.m1_atr.london_min ?? 3);
      const nyMin = Number(fc.m1_atr.ny_min ?? 3.5);
      const tokyoMax = Number(fc.m1_atr.tokyo_max ?? 12);
      const londonMax = Number(fc.m1_atr.london_max ?? 14);
      const nyMax = Number(fc.m1_atr.ny_max ?? 16);
      result.push(mkFilter('m1_atr', 'M1 ATR(7)', true, true,
        m1atr != null ? `${m1atr.toFixed(1)}p` : '—',
        `T: ${tokyoMin}-${tokyoMax}p, L: ${londonMin}-${londonMax}p, NY: ${nyMin}-${nyMax}p`, null));
    }

    // M3 ATR (Trial #5)
    if (fc.m3_atr?.enabled) {
      const m3atr = taData?.timeframes['M3']?.atr?.value_pips;
      const min = Number(fc.m3_atr.min ?? 5);
      const max = Number(fc.m3_atr.max ?? 16);
      let ok = true;
      let reason: string | null = null;
      if (m3atr != null) {
        ok = m3atr >= min && m3atr <= max;
        if (!ok) reason = `${m3atr.toFixed(1)}p outside ${min}-${max}p`;
      }
      result.push(mkFilter('m3_atr', 'M3 ATR(14)', true, ok,
        m3atr != null ? `${m3atr.toFixed(1)}p` : '—', `${min}-${max}p`, reason));
    }

    // Daily H/L
    if (fc.daily_hl?.enabled) {
      const buffer = Number(fc.daily_hl.buffer ?? 5);
      result.push(mkFilter('daily_hl', 'Daily H/L Filter', true, true,
        '—', `Buffer: ${buffer.toFixed(1)}p`, null));
    }

    // Trend Exhaustion
    if (fc.trend_exhaustion?.enabled) {
      const fresh = Number(fc.trend_exhaustion.fresh_max ?? 2);
      const mature = Number(fc.trend_exhaustion.mature_max ?? 3.5);
      result.push(mkFilter('trend_exhaustion', 'Trend Exhaustion', true, true,
        '—', `Fresh: ${fresh.toFixed(1)}p, Mature: ${mature.toFixed(1)}p`, null));
    }

    // Dead Zone
    if (fc.dead_zone?.enabled) {
      const h = new Date().getUTCHours();
      const inDeadZone = h >= 21 || h < 2;
      result.push(mkFilter('dead_zone', 'Dead Zone (21-02 UTC)', true, !inDeadZone,
        inDeadZone ? 'IN DEAD ZONE' : 'Clear',
        '21:00-02:00 UTC', inDeadZone ? 'Currently in dead zone' : null));
    }

    // Max Trades Per Side
    if (fc.max_trades?.enabled) {
      const perSide = Number(fc.max_trades.per_side ?? 5);
      result.push(mkFilter('max_trades', 'Max Trades/Side', true, true,
        '—', `${perSide} per side`, null));
    }

    return result;
  })();

  const dailySummary = dashState?.daily_summary || null;
  const presetName = dashState?.preset_name || filterConfig?.preset_name || '';

  return (
    <div style={{ backgroundColor: colors.bg, color: colors.textPrimary, minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <style>{`
        @keyframes dashHighlight {
          0% { background-color: ${colors.blue}33; }
          100% { background-color: ${colors.panelHover}; }
        }
      `}</style>

      {/* Header — always works since tick comes from getTechnicalAnalysis */}
      <HeaderBar
        loopRunning={loopRunning}
        profileName={profileName}
        mode={runtime?.mode || ''}
        tick={tick}
        onToggleLoop={handleToggleLoop}
        presetName={presetName}
      />

      {/* Main content */}
      <div style={{ flex: 1, padding: 12, display: 'flex', flexDirection: 'column', gap: 8, overflowY: 'auto' }}>
        {/* Two-column: Context + Positions */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
          <ContextPanel items={context} />
          <PositionsPanel trades={openTrades} tick={tick} />
        </div>

        {/* Filter Status */}
        <FilterTable filters={filters} />

        {/* Recent Trades from DB */}
        <RecentTradesPanel profileName={profileName} profilePath={profilePath} />

        {/* Trade Log (from trade_events.json, written by loop) */}
        {events.length > 0 && <TradeLog events={events} />}

        {/* Raw Logs */}
        <RawLogs profileName={profileName} />
      </div>

      {/* Daily Summary Bar */}
      <DailySummaryBar summary={dailySummary} />
    </div>
  );
}
