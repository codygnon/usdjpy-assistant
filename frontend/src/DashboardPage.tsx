import { useEffect, useRef, useState, useCallback, type ReactNode } from 'react';
import {
  getDashboard, getTradeEvents, getExecutions, getPhase3Decisions, getPhase3DefensiveMonitor, getPhase3PaperAcceptance, getPhase3Provenance, startLoop, stopLoop, getRuntimeState, updateRuntimeState,
  type DashboardState, type TradeEvent, type FilterReport, type ContextItem,
  type DailySummary, type Phase3AcceptanceSummary, type Phase3DecisionRow, type Phase3DefensiveMonitor, type Phase3Provenance, type RuntimeState,
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
const PHASE3_DEFENDED_PRESET_ID = 'phase3_integrated_v7_defended';

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
  exitSystemOnly, onExitSystemOnlyChange,
}: {
  loopRunning: boolean;
  profileName: string;
  mode: string;
  tick: { bid: number; ask: number; spread: number } | null;
  onToggleLoop: () => void;
  presetName: string;
  onModeChange: (mode: string) => void;
  staleLabel?: string | null;
  exitSystemOnly: boolean;
  onExitSystemOnlyChange: (enabled: boolean) => void;
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
      <button
        onClick={() => onExitSystemOnlyChange(!exitSystemOnly)}
        style={{
          padding: '2px 10px', borderRadius: 4, border: 'none', cursor: 'pointer',
          fontWeight: 600, fontSize: 11,
          backgroundColor: exitSystemOnly ? colors.amber : colors.panelHover,
          color: exitSystemOnly ? '#000' : colors.textSecondary,
        }}
      >{exitSystemOnly ? 'EXIT ONLY ON' : 'Exit Only'}</button>
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
  runtime: 'Runtime Health',
  decision: 'Decision State',
  session: 'Session',
  frozen: 'Frozen Package',
  v14: 'Tokyo / V14',
  london: 'London / V2',
  v44: 'New York / V44',
};

function isPhase3PresetName(name: string | null | undefined): boolean {
  const normalized = String(name || '').toLowerCase();
  return normalized.includes('phase3_integrated') || normalized.includes('phase 3');
}

function isDefendedPhase3Preset(name: string | null | undefined): boolean {
  return String(name || '').trim().toLowerCase() === PHASE3_DEFENDED_PRESET_ID;
}

function hasPhase3Payload(context: ContextItem[], filters: FilterReport[]): boolean {
  return context.some((item) => ['runtime', 'decision', 'frozen', 'v14', 'london', 'v44'].includes(item.category))
    || filters.some((filter) => filter.filter_id.startsWith('phase3_') || filter.filter_id.startsWith('ny_') || filter.filter_id.startsWith('tokyo') || filter.filter_id.startsWith('london'));
}

function getContextItem(items: ContextItem[], key: string): ContextItem | undefined {
  return items.find((item) => item.key === key);
}

function getContextValue(items: ContextItem[], key: string, fallback = 'waiting'): string {
  return getContextItem(items, key)?.value || fallback;
}

function getContextItemsByCategory(items: ContextItem[], category: string): ContextItem[] {
  return items.filter((item) => item.category === category);
}

function ContextValue({ item, fallback = 'waiting' }: { item?: ContextItem; fallback?: string }) {
  return (
    <span style={{
      color: item?.valueColor === 'green' ? colors.green : item?.valueColor === 'red' ? colors.red : colors.textPrimary,
      fontSize: 15,
      ...mono,
    }}>
      {item?.value || fallback}
    </span>
  );
}

function ContextSection({
  title,
  items,
  emptyLabel,
}: {
  title: string;
  items: ContextItem[];
  emptyLabel?: string;
}) {
  return (
    <div style={{
      backgroundColor: colors.panelHover,
      border: `1px solid ${colors.border}`,
      borderRadius: 8,
      padding: 12,
      minHeight: 120,
    }}>
      <div style={{ fontSize: 12, color: colors.blue, fontWeight: 700, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 0.8 }}>
        {title}
      </div>
      {items.length === 0 ? (
        <div style={{ color: colors.textSecondary, fontSize: 12 }}>{emptyLabel || 'No data yet'}</div>
      ) : (
        items.map((item) => (
          <div key={`${title}-${item.key}`} style={{ display: 'flex', justifyContent: 'space-between', gap: 10, padding: '3px 0' }}>
            <span style={{ color: colors.textSecondary, fontSize: 13 }}>{item.key}</span>
            <ContextValue item={item} />
          </div>
        ))
      )}
    </div>
  );
}

function Phase3StatusStrip({ items, presetName }: { items: ContextItem[]; presetName: string }) {
  const statusTiles = [
    {
      label: 'Package',
      value: getContextValue(items, 'Frozen Package', isDefendedPhase3Preset(presetName) ? 'Phase 3 Frozen V7 Defended' : 'Phase 3 active'),
      accent: colors.blue,
    },
    {
      label: 'Session',
      value: getContextValue(items, 'Active Session', 'not in session'),
      accent: colors.amber,
    },
    {
      label: 'Window',
      value: getContextValue(items, 'Window', 'waiting'),
      accent: colors.textSecondary,
    },
    {
      label: 'Ownership Cell',
      value: getContextValue(items, 'Ownership Cell', getContextValue(items, 'NY Ownership Cell', 'no eval yet')),
      accent: colors.green,
    },
    {
      label: 'Last Decision',
      value: getContextValue(items, 'Last decision', 'no eval yet'),
      accent: colors.textSecondary,
    },
    {
      label: 'Defensive State',
      value: getContextValue(
        items,
        'Defensive Flags',
        isDefendedPhase3Preset(presetName) ? getContextValue(items, 'Defensive Veto', 'veto armed') : 'clear'
      ),
      accent: colors.red,
    },
  ];

  return (
    <div style={{
      backgroundColor: colors.panel,
      borderRadius: 8,
      border: `1px solid ${colors.border}`,
      padding: 12,
    }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: colors.textSecondary, marginBottom: 10, textTransform: 'uppercase', letterSpacing: 1 }}>
        Phase 3 Status
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 10 }}>
        {statusTiles.map((tile) => (
          <div key={tile.label} style={{
            padding: 10,
            borderRadius: 8,
            backgroundColor: colors.panelHover,
            border: `1px solid ${tile.accent}44`,
          }}>
            <div style={{ fontSize: 11, color: colors.textSecondary, marginBottom: 6, textTransform: 'uppercase', letterSpacing: 0.8 }}>
              {tile.label}
            </div>
            <div style={{ fontSize: 13, color: tile.accent, fontWeight: 700, ...mono }}>
              {tile.value}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function Phase3ContextPanel({ items, presetName }: { items: ContextItem[]; presetName: string }) {
  const runtimeItems = getContextItemsByCategory(items, 'runtime');
  const frozenItems = getContextItemsByCategory(items, 'frozen');
  const decisionItems = getContextItemsByCategory(items, 'decision');
  const sessionItems = getContextItemsByCategory(items, 'session');
  const packageItems = [
    ...frozenItems.filter((item) => item.key === 'Frozen Package'),
    ...runtimeItems.filter((item) => item.key === 'Config Hash' || item.key === 'Config Date'),
  ];
  const runtimeHealthItems = [
    ...runtimeItems.filter((item) => item.key !== 'Config Hash' && item.key !== 'Config Date'),
    ...sessionItems.filter((item) => item.key === 'Phase 3 open'),
  ];
  const primarySessionItems = sessionItems.filter((item) =>
    ['UTC', 'Day', 'Active Session', 'Strategy', 'Window'].includes(item.key)
  );
  const defendedRuleItems = frozenItems.filter((item) => item.key !== 'Frozen Package');
  const sessionDetailItems = [
    ...getContextItemsByCategory(items, 'v14'),
    ...getContextItemsByCategory(items, 'london'),
    ...getContextItemsByCategory(items, 'v44'),
  ];

  return (
    <div style={{
      backgroundColor: colors.panel,
      borderRadius: 8,
      padding: 14,
      border: `1px solid ${colors.border}`,
      minHeight: 120,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 10, marginBottom: 10, flexWrap: 'wrap' }}>
        <div style={{ fontSize: 15, fontWeight: 700, color: colors.textSecondary, textTransform: 'uppercase', letterSpacing: 1 }}>
          Phase 3 Operator Context
        </div>
        <div style={{
          padding: '4px 10px',
          borderRadius: 999,
          fontSize: 11,
          fontWeight: 700,
          color: isDefendedPhase3Preset(presetName) ? colors.green : colors.blue,
          backgroundColor: isDefendedPhase3Preset(presetName) ? `${colors.green}18` : `${colors.blue}18`,
          border: `1px solid ${isDefendedPhase3Preset(presetName) ? colors.green : colors.blue}44`,
        }}>
          {isDefendedPhase3Preset(presetName) ? 'Frozen Defended Package Active' : 'Phase 3 Structured View'}
        </div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 10 }}>
        <ContextSection title="Package" items={packageItems} emptyLabel="No package metadata yet" />
        <ContextSection title="Runtime Health" items={runtimeHealthItems} emptyLabel="No runtime-health samples yet" />
        <ContextSection title="Session" items={primarySessionItems} emptyLabel="Not in an active session" />
        <ContextSection title="Decision State" items={decisionItems} emptyLabel="No Phase 3 evaluation yet" />
        <ContextSection title="Defended Rules" items={defendedRuleItems} emptyLabel="No frozen rule overlay active" />
        <ContextSection title="Session Detail" items={sessionDetailItems} emptyLabel="No session detail available yet" />
      </div>
    </div>
  );
}

function phase3FamilyLabel(family: string | null | undefined): string {
  switch (String(family || '')) {
    case 'v14':
      return 'V14';
    case 'london_v2_d':
      return 'L1';
    case 'london_v2_arb':
      return 'L2/ARB';
    case 'v44_ny':
      return 'V44';
    default:
      return String(family || '').trim() || 'Phase 3';
  }
}

function phase3SessionLabel(session: string | null | undefined): string {
  switch (String(session || '')) {
    case 'tokyo':
      return 'Tokyo';
    case 'london':
      return 'London';
    case 'ny':
      return 'NY';
    default:
      return String(session || '').trim() || '—';
  }
}

function StatusChip({ label, color }: { label: string; color: string }) {
  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 6,
      padding: '4px 8px',
      borderRadius: 999,
      fontSize: 11,
      fontWeight: 700,
      color,
      backgroundColor: `${color}18`,
      border: `1px solid ${color}44`,
      whiteSpace: 'nowrap',
    }}>
      {label}
    </span>
  );
}

function AttributionBadges({ event }: { event: Pick<TradeEvent, 'phase3_session' | 'phase3_strategy_family' | 'ownership_cell' | 'has_cell_attribution'> }) {
  const badges: Array<{ label: string; color: string }> = [];
  if (event.phase3_session) badges.push({ label: phase3SessionLabel(event.phase3_session), color: colors.amber });
  if (event.phase3_strategy_family) badges.push({ label: phase3FamilyLabel(event.phase3_strategy_family), color: colors.blue });
  if (event.ownership_cell) badges.push({ label: event.ownership_cell, color: colors.green });
  else if (event.has_cell_attribution) badges.push({ label: '@cell', color: colors.green });
  if (badges.length === 0) return null;
  return (
    <>
      {badges.map((badge) => (
        <StatusChip key={`${badge.label}-${badge.color}`} label={badge.label} color={badge.color} />
      ))}
    </>
  );
}

function Phase3SectionCard({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: ReactNode;
}) {
  return (
    <div style={{
      backgroundColor: colors.panel,
      borderRadius: 8,
      padding: 12,
      border: `1px solid ${colors.border}`,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'baseline', marginBottom: 8, flexWrap: 'wrap' }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: colors.textSecondary, textTransform: 'uppercase', letterSpacing: 1 }}>
          {title}
        </div>
        {subtitle && <div style={{ fontSize: 11, color: colors.textSecondary }}>{subtitle}</div>}
      </div>
      {children}
    </div>
  );
}

function Phase3MetricGrid({ rows }: { rows: Array<{ label: string; value: string | number | null | undefined; color?: string }> }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 10 }}>
      {rows.map((row) => (
        <div key={row.label} style={{ padding: 10, borderRadius: 8, backgroundColor: colors.panelHover, border: `1px solid ${colors.border}` }}>
          <div style={{ fontSize: 11, color: colors.textSecondary, marginBottom: 6, textTransform: 'uppercase', letterSpacing: 0.7 }}>{row.label}</div>
          <div style={{ fontSize: 13, fontWeight: 700, color: row.color || colors.textPrimary, ...mono }}>
            {row.value == null || row.value === '' ? '—' : String(row.value)}
          </div>
        </div>
      ))}
    </div>
  );
}

function Phase3DecisionSample({
  decisions,
  sessionFilter,
}: {
  decisions: Phase3DecisionRow[];
  sessionFilter?: 'tokyo' | 'london' | 'ny';
}) {
  const filtered = sessionFilter ? decisions.filter((row) => row.phase3_session === sessionFilter) : decisions;
  const sample = filtered.slice(-12).reverse();
  return (
    <Phase3SectionCard
      title="Recent Phase 3 Decisions"
      subtitle={sessionFilter ? `${phase3SessionLabel(sessionFilter)} sample` : 'Latest per-bar evaluations'}
    >
      {sample.length === 0 ? (
        <div style={{ color: colors.textSecondary, fontSize: 12 }}>No Phase 3 decision rows yet.</div>
      ) : (
        <div style={{ maxHeight: 260, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ color: colors.textSecondary, borderBottom: `1px solid ${colors.border}` }}>
                <th style={{ textAlign: 'left', padding: '4px 6px', fontWeight: 500 }}>Time</th>
                <th style={{ textAlign: 'left', padding: '4px 6px', fontWeight: 500 }}>Session</th>
                <th style={{ textAlign: 'center', padding: '4px 6px', fontWeight: 500 }}>Placed</th>
                <th style={{ textAlign: 'left', padding: '4px 6px', fontWeight: 500 }}>Reason</th>
              </tr>
            </thead>
            <tbody>
              {sample.map((row, idx) => (
                <tr key={`${row.timestamp_utc || 'row'}-${idx}`} style={{ borderBottom: `1px solid ${colors.border}22` }}>
                  <td style={{ padding: '4px 6px', ...mono, color: colors.textSecondary }}>{String(row.timestamp_utc || '').slice(11, 19)}</td>
                  <td style={{ padding: '4px 6px' }}>{phase3SessionLabel(row.phase3_session)}</td>
                  <td style={{ padding: '4px 6px', textAlign: 'center', color: row.placed ? colors.green : colors.red }}>{row.placed ? 'Yes' : 'No'}</td>
                  <td style={{ padding: '4px 6px', maxWidth: 460, overflow: 'hidden', textOverflow: 'ellipsis' }}>{row.reason_text || row.reason || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Phase3SectionCard>
  );
}

function Phase3ProvenanceCard({ provenance }: { provenance: Phase3Provenance | null }) {
  if (!provenance) {
    return (
      <Phase3SectionCard title="Decision Provenance">
        <div style={{ color: colors.textSecondary, fontSize: 12 }}>Decision provenance unavailable.</div>
      </Phase3SectionCard>
    );
  }
  const statusColor = provenance.outcome === 'placed' ? colors.green : provenance.outcome === 'blocked' ? colors.red : colors.amber;
  return (
    <Phase3SectionCard title="Decision Provenance" subtitle={provenance.generated_at_utc ? `Updated ${String(provenance.generated_at_utc).slice(11, 19)} UTC` : undefined}>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 10 }}>
        <StatusChip label={`Outcome: ${provenance.outcome.toUpperCase()}`} color={statusColor} />
        {provenance.session && <StatusChip label={phase3SessionLabel(provenance.session)} color={colors.amber} />}
        {provenance.strategy_family && <StatusChip label={phase3FamilyLabel(provenance.strategy_family)} color={colors.blue} />}
        {provenance.ownership_cell && <StatusChip label={provenance.ownership_cell} color={colors.green} />}
      </div>
      <Phase3MetricGrid rows={[
        { label: 'Package', value: provenance.package_id || provenance.preset_name || '—', color: colors.blue },
        { label: 'Window', value: provenance.window_label || 'waiting' },
        { label: 'Strategy Tag', value: provenance.strategy_tag || 'no evaluated Phase 3 decision yet' },
        { label: 'Regime', value: provenance.regime_label || '—' },
        { label: 'Blocking Rules', value: provenance.blocking_filter_ids.length > 0 ? provenance.blocking_filter_ids.join(', ') : 'clear' },
        { label: 'Last Block Reason', value: provenance.last_block_reason || provenance.reason || '—', color: provenance.outcome === 'blocked' ? colors.red : colors.textPrimary },
      ]} />
      <div style={{ marginTop: 10, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
        <div style={{ padding: 10, borderRadius: 8, backgroundColor: colors.panelHover, border: `1px solid ${colors.border}` }}>
          <div style={{ fontSize: 11, color: colors.textSecondary, marginBottom: 6, textTransform: 'uppercase', letterSpacing: 0.7 }}>Exit Policy</div>
          {provenance.exit_policy ? (
            <div style={{ color: colors.textPrimary, fontSize: 12, lineHeight: 1.6 }}>
              <div>{provenance.exit_policy.label}</div>
              <div style={mono}>TP1 {provenance.exit_policy.tp1_r.toFixed(2)}R / BE {provenance.exit_policy.be_offset_pips.toFixed(1)} / TP2 {provenance.exit_policy.tp2_r.toFixed(1)}R</div>
            </div>
          ) : (
            <div style={{ color: colors.textSecondary, fontSize: 12 }}>No exit policy assigned yet.</div>
          )}
        </div>
        <div style={{ padding: 10, borderRadius: 8, backgroundColor: colors.panelHover, border: `1px solid ${colors.border}` }}>
          <div style={{ fontSize: 11, color: colors.textSecondary, marginBottom: 6, textTransform: 'uppercase', letterSpacing: 0.7 }}>Frozen Modifiers</div>
          {provenance.frozen_modifiers.length === 0 ? (
            <div style={{ color: colors.textSecondary, fontSize: 12 }}>No frozen modifiers active.</div>
          ) : (
            provenance.frozen_modifiers.map((modifier) => (
              <div key={modifier.id} style={{ display: 'flex', justifyContent: 'space-between', gap: 8, fontSize: 12, padding: '2px 0' }}>
                <span style={{ color: colors.textSecondary }}>{modifier.label}</span>
                <span style={{ color: colors.textPrimary, ...mono }}>{modifier.value}</span>
              </div>
            ))
          )}
        </div>
      </div>
    </Phase3SectionCard>
  );
}

function Phase3AcceptanceWidget({ acceptance }: { acceptance: Phase3AcceptanceSummary | null }) {
  if (!acceptance || !acceptance.available) {
    return (
      <Phase3SectionCard title="Observed-Fire Acceptance">
        <div style={{ color: colors.textSecondary, fontSize: 12 }}>Paper acceptance artifact not available yet.</div>
      </Phase3SectionCard>
    );
  }
  const observed = acceptance.observed_summary?.OBSERVED_count ?? 0;
  const awaiting = acceptance.observed_summary?.IMPLEMENTED_AND_INSTRUMENTED_AWAITING_OBSERVED_FIRE_count ?? 0;
  const broken = acceptance.observed_summary?.BROKEN_count ?? 0;
  const verdictColor = acceptance.verdict?.includes('BLOCKED') ? colors.red : acceptance.verdict?.includes('ACCEPTED') ? colors.green : colors.amber;
  return (
    <Phase3SectionCard title="Observed-Fire Acceptance" subtitle={acceptance.package_under_test || undefined}>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 10 }}>
        <StatusChip label={acceptance.verdict || 'Awaiting validation'} color={verdictColor} />
        <StatusChip label={`Observed ${observed}`} color={colors.green} />
        <StatusChip label={`Awaiting ${awaiting}`} color={colors.amber} />
        <StatusChip label={`Broken ${broken}`} color={colors.red} />
      </div>
      <div style={{ color: colors.textSecondary, fontSize: 12, marginBottom: 10 }}>{acceptance.verdict_note || 'No acceptance note recorded.'}</div>
      <div style={{ display: 'grid', gap: 8 }}>
        {acceptance.rules.map((rule) => {
          const color = rule.status.includes('BROKEN') ? colors.red : rule.status.includes('OBSERVED') ? colors.green : colors.amber;
          return (
            <div key={rule.id} style={{ padding: 10, borderRadius: 8, backgroundColor: colors.panelHover, border: `1px solid ${colors.border}` }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, alignItems: 'center', marginBottom: 4, flexWrap: 'wrap' }}>
                <div style={{ color: colors.textPrimary, fontWeight: 700, fontSize: 13 }}>{rule.label}</div>
                <StatusChip label={rule.status} color={color} />
              </div>
              <div style={{ color: colors.textSecondary, fontSize: 12 }}>{rule.requirement}</div>
              {rule.evidence_pointer && <div style={{ color: colors.textSecondary, fontSize: 11, marginTop: 4 }}>{rule.evidence_pointer}</div>}
            </div>
          );
        })}
      </div>
      {acceptance.immediate_next_action && (
        <div style={{ marginTop: 10, padding: 10, borderRadius: 8, backgroundColor: `${colors.amber}10`, border: `1px solid ${colors.amber}33`, color: colors.amber, fontSize: 12 }}>
          Next action: {acceptance.immediate_next_action}
        </div>
      )}
    </Phase3SectionCard>
  );
}

function Phase3DefensiveMonitorPanel({ monitor }: { monitor: Phase3DefensiveMonitor | null }) {
  if (!monitor || !monitor.available) {
    return (
      <Phase3SectionCard title="Defensive Monitor">
        <div style={{ color: colors.textSecondary, fontSize: 12 }}>Defensive monitor artifact not available yet.</div>
      </Phase3SectionCard>
    );
  }
  const baseline500k = (monitor.research_baseline_blocked_trade_counts as Record<string, any> | null)?.['500k'];
  const baseline1000k = (monitor.research_baseline_blocked_trade_counts as Record<string, any> | null)?.['1000k'];
  return (
    <Phase3SectionCard title="Defensive Monitor" subtitle={monitor.frozen_package_id || undefined}>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 10 }}>
        <StatusChip label={monitor.strategy || 'v44_ny'} color={colors.blue} />
        <StatusChip label={monitor.ownership_cell || 'ambiguous/er_low/der_neg'} color={colors.green} />
        <StatusChip label={monitor.paper_monitor_executed ? 'Monitor Run' : 'Monitor Not Yet Run'} color={monitor.paper_monitor_executed ? colors.green : colors.amber} />
        {monitor.pause_recommended != null && <StatusChip label={monitor.pause_recommended ? 'Pause Recommended' : 'No Pause Flag'} color={monitor.pause_recommended ? colors.red : colors.green} />}
      </div>
      <Phase3MetricGrid rows={[
        { label: 'Log Path', value: monitor.log_path_used || 'not available' },
        { label: 'Guardrail Status', value: monitor.guardrail_status ? JSON.stringify(monitor.guardrail_status) : 'not available' },
        { label: '500k Baseline', value: baseline500k ? `${baseline500k.blocked_count} blocked / ${baseline500k.blocked_net_usd} USD` : '—' },
        { label: '1000k Baseline', value: baseline1000k ? `${baseline1000k.blocked_count} blocked / ${baseline1000k.blocked_net_usd} USD` : '—' },
      ]} />
      {monitor.paper_monitor_skip_reason && (
        <div style={{ marginTop: 10, color: colors.textSecondary, fontSize: 12 }}>{monitor.paper_monitor_skip_reason}</div>
      )}
      {monitor.searched_locations.length > 0 && (
        <div style={{ marginTop: 10 }}>
          <div style={{ fontSize: 11, color: colors.textSecondary, marginBottom: 6, textTransform: 'uppercase', letterSpacing: 0.7 }}>Searched Locations</div>
          <ul style={{ margin: 0, paddingLeft: 18, color: colors.textSecondary, fontSize: 12 }}>
            {monitor.searched_locations.slice(0, 6).map((location) => <li key={location}>{location}</li>)}
          </ul>
        </div>
      )}
      {monitor.next_command_when_log_exists && (
        <div style={{ marginTop: 10, padding: 10, borderRadius: 8, backgroundColor: `${colors.blue}10`, border: `1px solid ${colors.blue}33`, color: colors.textPrimary, fontSize: 12 }}>
          <div style={{ color: colors.textSecondary, marginBottom: 4 }}>Next command when log exists</div>
          <div style={mono}>{monitor.next_command_when_log_exists}</div>
        </div>
      )}
    </Phase3SectionCard>
  );
}

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
// Filter Status Table
// ---------------------------------------------------------------------------

function FilterRow({
  filter,
  depth = 0,
  emphasizeBlocked = false,
}: {
  filter: FilterReport;
  depth?: number;
  emphasizeBlocked?: boolean;
}) {
  const [expanded, setExpanded] = useState(true);
  if (!filter.enabled) return null;
  const hasSubs = filter.sub_filters && filter.sub_filters.length > 0;
  const rowTint = !filter.is_clear && emphasizeBlocked ? `${colors.red}12` : 'transparent';

  return (
    <>
      <tr
        style={{
          borderBottom: `1px solid ${colors.border}22`,
          cursor: hasSubs ? 'pointer' : 'default',
          backgroundColor: rowTint,
        }}
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
        <td style={{ padding: '4px 8px', fontSize: 11, color: colors.textSecondary }}>
          {filter.explanation ?? (filter.block_reason && !filter.is_clear ? filter.block_reason : '')}
        </td>
      </tr>
      {hasSubs && expanded && filter.sub_filters.map(sub => (
        <FilterRow key={sub.filter_id} filter={sub} depth={depth + 1} emphasizeBlocked={emphasizeBlocked} />
      ))}
    </>
  );
}

function classifyPhase3FilterGroup(filter: FilterReport): 'frozen' | 'entry' | 'session' | 'diagnostics' {
  if (filter.filter_id.startsWith('phase3_frozen_')) return 'frozen';
  if (['allowed_to_trade', 'last_decision', 'session', 'strategy'].includes(filter.filter_id)) return 'entry';
  if (
    filter.filter_id.includes('london') ||
    filter.filter_id.includes('tokyo') ||
    filter.filter_id.includes('ny_') ||
    filter.filter_id === 'regime' ||
    filter.filter_id === 'adx' ||
    filter.filter_id === 'atr' ||
    filter.filter_id.includes('range') ||
    filter.filter_id.includes('levels')
  ) {
    return 'session';
  }
  return 'diagnostics';
}

function FilterGroupTable({
  title,
  filters,
  emphasizeBlocked,
}: {
  title: string;
  filters: FilterReport[];
  emphasizeBlocked?: boolean;
}) {
  if (filters.length === 0) return null;
  return (
    <div style={{ marginTop: 10 }}>
      <div style={{ fontSize: 12, color: colors.blue, fontWeight: 700, marginBottom: 6, textTransform: 'uppercase', letterSpacing: 0.8 }}>
        {title}
      </div>
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
          {filters.map((filter) => (
            <FilterRow key={filter.filter_id} filter={filter} emphasizeBlocked={emphasizeBlocked} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function FilterTable({
  filters,
  candidateSide,
  candidateTrigger,
  lastBlockReason,
  isPhase3 = false,
}: {
  filters: FilterReport[];
  candidateSide: 'buy' | 'sell' | null;
  candidateTrigger: 'zone_entry' | 'tiered_pullback' | null;
  lastBlockReason?: string | null;
  isPhase3?: boolean;
}) {
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
  const hasCandidateContext = candidateSide != null && candidateTrigger != null;
  const isRelevantBlocker = (filter: FilterReport) => {
    switch (filter.filter_id) {
      case 'max_trades_buy':
        return candidateSide === 'buy';
      case 'max_trades_sell':
        return candidateSide === 'sell';
      case 'zone_entry_cap':
        return candidateTrigger === 'zone_entry';
      case 'tiered_pullback_cap':
        return candidateTrigger === 'tiered_pullback';
      case 'ema_zone_slope_filter':
        return candidateTrigger === 'zone_entry';
      default:
        return true;
    }
  };
  const relevantFilters = enabledFilters.filter(isRelevantBlocker);
  const relevantBlockers = relevantFilters.filter(f => !f.is_clear).map(f => f.display_name);
  const otherWarnings = enabledFilters
    .filter(f => !isRelevantBlocker(f) && !f.is_clear)
    .map(f => f.display_name);
  const allClear = relevantBlockers.length === 0;
  const groupedPhase3Filters = isPhase3
    ? {
        frozen: enabledFilters.filter((f) => classifyPhase3FilterGroup(f) === 'frozen'),
        entry: enabledFilters.filter((f) => classifyPhase3FilterGroup(f) === 'entry'),
        session: enabledFilters.filter((f) => classifyPhase3FilterGroup(f) === 'session'),
        diagnostics: enabledFilters.filter((f) => classifyPhase3FilterGroup(f) === 'diagnostics'),
      }
    : null;

  return (
    <div style={{
      backgroundColor: colors.panel, borderRadius: 6, padding: 12,
      border: `1px solid ${colors.border}`,
    }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: colors.textSecondary, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>Filter Status</div>
      <div style={{
        marginTop: 8, padding: '6px 10px', fontSize: 12, borderRadius: 6,
        backgroundColor: !hasCandidateContext
          ? colors.amber + '11'
          : allClear
            ? colors.green + '11'
            : colors.red + '11',
        color: !hasCandidateContext
          ? colors.amber
          : allClear
            ? colors.green
            : colors.red,
        border: `1px solid ${
          !hasCandidateContext
            ? colors.amber + '33'
            : allClear
              ? colors.green + '33'
              : colors.red + '33'
        }`,
      }}>
        {!hasCandidateContext
          ? 'Entry: WAITING (no active candidate)'
          : allClear
            ? 'Entry: CLEAR'
            : `Entry: BLOCKED by ${relevantBlockers.join(', ')}`}
      </div>
      {lastBlockReason && (
        <div style={{ marginTop: 6, padding: '6px 10px', fontSize: 11, color: colors.amber, backgroundColor: 'rgba(245, 158, 11, 0.1)', borderRadius: 6 }}>
          Top blocker: {lastBlockReason}
        </div>
      )}
      {otherWarnings.length > 0 && (
        <div style={{ marginTop: 6, fontSize: 11, color: colors.textSecondary }}>
          Other warnings: {otherWarnings.join(', ')}
        </div>
      )}
      {hasCandidateContext && (
        <div style={{ marginTop: 4, fontSize: 11, color: colors.textSecondary }}>
          Candidate: {candidateSide?.toUpperCase()} {candidateTrigger === 'zone_entry' ? 'zone entry' : 'tiered pullback'}
        </div>
      )}
      {isPhase3 && groupedPhase3Filters ? (
        <>
          <FilterGroupTable title="Frozen Rules" filters={groupedPhase3Filters.frozen} />
          <FilterGroupTable title="Entry Readiness" filters={groupedPhase3Filters.entry} emphasizeBlocked />
          <FilterGroupTable title="Session Gates" filters={groupedPhase3Filters.session} emphasizeBlocked />
          <FilterGroupTable title="Diagnostics" filters={groupedPhase3Filters.diagnostics} />
        </>
      ) : (
        <FilterGroupTable title="All Filters" filters={enabledFilters} />
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
        <AttributionBadges event={event} />
        {event.trigger_type && <span style={{ color: colors.textSecondary, fontSize: 10 }}>{event.trigger_type}</span>}
        <span style={{ ...mono, color: colors.textPrimary }}>{event.price.toFixed(3)}</span>
        {!isClose && event.spread_at_entry != null && (
          <span style={{ color: colors.textSecondary, fontSize: 10, ...mono, marginLeft: 4 }}>
            spread: {event.spread_at_entry.toFixed(1)}p
          </span>
        )}
      </div>
      {!isClose && event.context_snapshot?.trend_strength != null && (
        <div style={{ display: 'flex', gap: 8, fontSize: 10, marginTop: 2, color: colors.textSecondary }}>
          <span>Strength: <span style={{ color: colors.textPrimary }}>{String(event.context_snapshot.trend_strength)}</span></span>
          {event.context_snapshot.sl_distance_pips != null && (
            <span>SL: <span style={{ color: colors.red, ...mono }}>{Number(event.context_snapshot.sl_distance_pips).toFixed(1)}p</span></span>
          )}
          {event.context_snapshot.tp1_pips != null && (
            <span>TP1: <span style={{ color: colors.green, ...mono }}>{Number(event.context_snapshot.tp1_pips).toFixed(1)}p</span></span>
          )}
          {event.context_snapshot.lot_size != null && (
            <span>Lots: <span style={mono}>{Number(event.context_snapshot.lot_size).toFixed(2)}</span></span>
          )}
        </div>
      )}
      {isClose && (
        <div style={{ display: 'flex', gap: 10, fontSize: 11, marginTop: 2 }}>
          {event.pips != null && <PipsValue pips={event.pips} />}
          {event.profit != null && (
            <span style={{ color: event.profit > 0 ? colors.green : colors.red, ...mono }}>
              {event.profit > 0 ? '+' : ''}{event.profit.toFixed(2)}
            </span>
          )}
          {event.exit_reason && <span style={{ color: colors.textSecondary }}>{event.exit_reason}</span>}
          {event.context_snapshot?.spread_at_entry != null && (
            <span style={{ color: colors.textSecondary, ...mono, fontSize: 11 }}>
              entry spread: {Number(event.context_snapshot.spread_at_entry).toFixed(1)}p
            </span>
          )}
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
// Execution Log (reasons candidate trades were not placed + loop errors)
// ---------------------------------------------------------------------------

export type ExecutionRecord = {
  timestamp_utc?: string;
  mode?: string;
  attempted?: number;
  placed?: number;
  reason?: string;
  rule_id?: string;
  signal_id?: string;
};

function ExecutionLog({ executions }: { executions: ExecutionRecord[] }) {
  const recent = executions.slice(0, 80);
  return (
    <div style={{
      backgroundColor: colors.panel,
      borderRadius: 6,
      padding: 12,
      border: `1px solid ${colors.border}`,
    }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: colors.textSecondary, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>
        Execution Log ({recent.length})
      </div>
      <div style={{ fontSize: 10, color: colors.textSecondary, marginBottom: 6 }}>
        Reasons candidate trades were not placed; loop errors show here when recorded.
      </div>
      <div style={{ maxHeight: 280, overflowY: 'auto' }}>
        {recent.length === 0 ? (
          <div style={{ color: colors.textSecondary, fontSize: 12, padding: 8 }}>No execution attempts yet. Start the loop to see reasons.</div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11, ...mono }}>
            <thead>
              <tr style={{ color: colors.textSecondary, borderBottom: `1px solid ${colors.border}` }}>
                <th style={{ textAlign: 'left', padding: '4px 6px', fontWeight: 600 }}>Time</th>
                <th style={{ textAlign: 'left', padding: '4px 6px', fontWeight: 600 }}>Mode</th>
                <th style={{ textAlign: 'center', padding: '4px 6px', fontWeight: 600 }}>Placed</th>
                <th style={{ textAlign: 'left', padding: '4px 6px', fontWeight: 600 }}>Rule</th>
                <th style={{ textAlign: 'left', padding: '4px 6px', fontWeight: 600 }}>Reason</th>
              </tr>
            </thead>
            <tbody>
              {[...recent].reverse().map((e, i) => (
                <tr key={i} style={{ borderBottom: `1px solid ${colors.border}22` }}>
                  <td style={{ padding: '3px 6px', color: colors.textSecondary, whiteSpace: 'nowrap' }}>
                    {String(e.timestamp_utc ?? '').slice(11, 19)}
                  </td>
                  <td style={{ padding: '3px 6px', color: colors.textSecondary }}>{String(e.mode ?? '')}</td>
                  <td style={{ padding: '3px 6px', textAlign: 'center' }}>
                    <span style={{ color: e.placed ? colors.green : colors.amber }}>{e.placed ? 'Yes' : 'No'}</span>
                  </td>
                  <td style={{ padding: '3px 6px', color: colors.textSecondary, maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis' }} title={String(e.rule_id ?? '')}>
                    {String(e.rule_id ?? '').slice(-40)}
                  </td>
                  <td style={{ padding: '3px 6px', color: e.placed ? colors.green : colors.textPrimary, maxWidth: 320, overflow: 'hidden', textOverflow: 'ellipsis' }} title={String(e.reason ?? '')}>
                    {String(e.reason ?? '')}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
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
  const [executions, setExecutions] = useState<ExecutionRecord[]>([]);
  const [dashState, setDashState] = useState<DashboardState | null>(null);
  const [runtime, setRuntime] = useState<RuntimeState | null>(null);
  const [loopError, setLoopError] = useState<string | null>(null);
  const [dashboardOffline, setDashboardOffline] = useState(false);
  const [trail, setTrail] = useState<Array<{ time: string; spread: number; blocked: number; trend: string }>>([]);
  const [trailExpanded, setTrailExpanded] = useState(false);
  const [phase3Tab, setPhase3Tab] = useState<'overview' | 'tokyo' | 'london' | 'ny' | 'defensive' | 'paper'>('overview');
  const [phase3Provenance, setPhase3Provenance] = useState<Phase3Provenance | null>(null);
  const [phase3Acceptance, setPhase3Acceptance] = useState<Phase3AcceptanceSummary | null>(null);
  const [phase3Monitor, setPhase3Monitor] = useState<Phase3DefensiveMonitor | null>(null);
  const [phase3Decisions, setPhase3Decisions] = useState<Phase3DecisionRow[]>([]);

  const loopRunning = dashboardOffline ? false : (dashState?.loop_running ?? false);
  const tick = (dashState && Number.isFinite(dashState.bid) && Number.isFinite(dashState.ask))
    ? { bid: dashState.bid, ask: dashState.ask, spread: dashState.spread_pips }
    : null;

  // Load runtime state on mount (for exit_system_only toggle)
  useEffect(() => {
    getRuntimeState(profileName).then(setRuntime).catch(() => {});
  }, [profileName]);

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

  // Poll execution log (reasons trades not placed, loop errors) — 10s
  useEffect(() => {
    if (!isPageVisible) return;
    let mounted = true;
    const poll = () => {
      getExecutions(profileName, 80)
        .then(e => { if (mounted) setExecutions((e as ExecutionRecord[]) || []); })
        .catch(() => {});
    };
    poll();
    const id = setInterval(poll, 10000);
    return () => { mounted = false; clearInterval(id); };
  }, [profileName, isPageVisible]);

  // Poll dashboard state — 10s
  useEffect(() => {
    if (!isPageVisible) return;
    let mounted = true;
    const poll = () => {
      getDashboard(profileName, profilePath)
        .then(s => {
          if (!mounted || !s || s.error) {
            if (mounted) {
              setDashboardOffline(true);
              setDashState(prev => prev ? { ...prev, loop_running: false, stale: true } : prev);
            }
            return;
          }
          setDashboardOffline(false);
          setDashState(s);
          const enabled = (s.filters || []).filter(f => f.enabled);
          const blocked = enabled.filter(f => !f.is_clear).length;
          const trend = (s.context || []).find(c => c.key.toLowerCase().includes('trend'))?.value ?? '—';
          const spread = Number(s.spread_pips ?? 0);
          const t = String(s.timestamp_utc || new Date().toISOString()).slice(11, 19);
          setTrail(prev => [...prev, { time: t, spread, blocked, trend }].slice(-60));
        })
        .catch((e) => {
          if (!mounted) return;
          setDashboardOffline(true);
          setDashState(prev => prev ? { ...prev, loop_running: false, stale: true } : prev);
          setLoopError(prev => prev ?? (e instanceof Error ? e.message : String(e)));
        });
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
      setLoopError(null);
      if (loopRunning) {
        await stopLoop(profileName);
      } else {
        await startLoop(profileName, profilePath);
      }
      const s = await getDashboard(profileName, profilePath);
      if (s && !s.error) setDashState(s);
    } catch (e) {
      console.error('Loop toggle error:', e);
      setLoopError(e instanceof Error ? e.message : String(e));
    }
  }, [loopRunning, profileName, profilePath]);

  const handleModeChange = useCallback(async (mode: string) => {
    try {
      const rt = await getRuntimeState(profileName);
      setRuntime(rt);
      await updateRuntimeState(profileName, mode, rt.kill_switch, rt.exit_system_only);
      const s = await getDashboard(profileName, profilePath);
      if (s && !s.error) setDashState(s);
    } catch (e) {
      console.error('Mode change error:', e);
    }
  }, [profileName, profilePath]);

  const handleExitSystemOnlyChange = useCallback(async (enabled: boolean) => {
    try {
      const rt = await getRuntimeState(profileName);
      setRuntime(rt);
      await updateRuntimeState(profileName, rt.mode, rt.kill_switch, enabled);
      const updated = await getRuntimeState(profileName);
      setRuntime(updated);
      const s = await getDashboard(profileName, profilePath);
      if (s && !s.error) setDashState(s);
    } catch (e) {
      console.error('Exit system only toggle error:', e);
    }
  }, [profileName, profilePath]);

  const context: ContextItem[] = dashState?.context || [];
  const filters: FilterReport[] = dashState?.filters || [];
  const dailySummary: DailySummary | null = dashState?.daily_summary || null;
  const presetName = dashState?.preset_name || '';
  const isPhase3Preset = isPhase3PresetName(presetName) || hasPhase3Payload(context, filters);
  const loopLog: Array<{ts: string; level: string; msg: string}> = (dashState as unknown as Record<string, unknown>)?.loop_log as Array<{ts: string; level: string; msg: string}> || [];
  const openEvents = events.filter((e) => e.event_type === 'open');
  const closedEvents = events.filter((e) => e.event_type === 'close');
  const recentOpenEvents = openEvents.slice(0, 30);
  const recentClosedEvents = closedEvents.slice(0, 30);
  const latestTrail = trail.length > 0 ? trail[trail.length - 1] : null;

  const staleLabel = (() => {
    if (dashboardOffline) return 'Dashboard disconnected';
    if (!dashState?.stale) return null;
    const age = dashState.stale_age_seconds;
    if (age == null) return 'Stale data';
    return `Stale ${Math.round(age)}s`;
  })();

  useEffect(() => {
    setPhase3Tab('overview');
  }, [profileName]);

  useEffect(() => {
    if (!isPhase3Preset) {
      setPhase3Tab('overview');
      setPhase3Provenance(null);
      setPhase3Acceptance(null);
      setPhase3Monitor(null);
      setPhase3Decisions([]);
    }
  }, [isPhase3Preset]);

  useEffect(() => {
    if (!isPageVisible || !isPhase3Preset) return;
    let mounted = true;
    const poll = () => {
      getPhase3Provenance(profileName, profilePath)
        .then((data) => { if (mounted) setPhase3Provenance(data); })
        .catch(() => { if (mounted) setPhase3Provenance(null); });
      getPhase3PaperAcceptance()
        .then((data) => { if (mounted) setPhase3Acceptance(data); })
        .catch(() => { if (mounted) setPhase3Acceptance(null); });
      getPhase3DefensiveMonitor()
        .then((data) => { if (mounted) setPhase3Monitor(data); })
        .catch(() => { if (mounted) setPhase3Monitor(null); });
      getPhase3Decisions(profileName, 7, 2000)
        .then((rows) => { if (mounted) setPhase3Decisions(rows || []); })
        .catch(() => { if (mounted) setPhase3Decisions([]); });
    };
    poll();
    const id = setInterval(poll, 10000);
    return () => { mounted = false; clearInterval(id); };
  }, [isPageVisible, isPhase3Preset, profileName, profilePath]);

  const tabButtons: Array<{ id: 'overview' | 'tokyo' | 'london' | 'ny' | 'defensive' | 'paper'; label: string }> = [
    { id: 'overview', label: 'Overview' },
    { id: 'tokyo', label: 'Tokyo / V14' },
    { id: 'london', label: 'London / V2' },
    { id: 'ny', label: 'New York / V44' },
    { id: 'defensive', label: 'Defensive' },
    { id: 'paper', label: 'Paper Acceptance' },
  ];

  const filterContextForTab = (tab: typeof phase3Tab): ContextItem[] => {
    if (tab === 'overview') return context;
    if (tab === 'tokyo') return context.filter((item) => ['runtime', 'decision', 'session', 'frozen', 'v14'].includes(item.category));
    if (tab === 'london') return context.filter((item) => ['runtime', 'decision', 'session', 'frozen', 'london'].includes(item.category));
    if (tab === 'ny') return context.filter((item) => ['runtime', 'decision', 'session', 'frozen', 'v44'].includes(item.category));
    if (tab === 'defensive') return context.filter((item) => ['runtime', 'decision', 'frozen', 'v44'].includes(item.category));
    if (tab === 'paper') return context.filter((item) => ['runtime', 'decision', 'frozen'].includes(item.category));
    return context;
  };

  const renderPhase3Overview = () => (
    <>
      <Phase3ProvenanceCard provenance={phase3Provenance} />
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        <Phase3ContextPanel items={context} presetName={presetName} />
        <TradeLog title="Trade Log (Last 30)" events={recentOpenEvents} />
      </div>
      <Phase3DecisionSample decisions={phase3Decisions} />
      <FilterTable
        filters={filters}
        candidateSide={dashState?.entry_candidate_side ?? null}
        candidateTrigger={dashState?.entry_candidate_trigger ?? null}
        lastBlockReason={dashState?.last_block_reason ?? null}
        isPhase3
      />
      <TradeLog title="Closed Trades (Last 30)" events={recentClosedEvents} />
      <ExecutionLog executions={executions} />
    </>
  );

  const renderPhase3SessionTab = (session: 'tokyo' | 'london' | 'ny') => (
    <>
      <Phase3DecisionSample decisions={phase3Decisions} sessionFilter={session} />
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        <Phase3ContextPanel items={filterContextForTab(session)} presetName={presetName} />
        <TradeLog title={`${phase3SessionLabel(session)} Trade Log`} events={recentOpenEvents.filter((event) => event.phase3_session === session)} />
      </div>
      <FilterTable
        filters={filters}
        candidateSide={dashState?.entry_candidate_side ?? null}
        candidateTrigger={dashState?.entry_candidate_trigger ?? null}
        lastBlockReason={dashState?.last_block_reason ?? null}
        isPhase3
      />
      <ExecutionLog executions={executions} />
    </>
  );

  const renderPhase3DefensiveTab = () => (
    <>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        <Phase3DefensiveMonitorPanel monitor={phase3Monitor} />
        <Phase3ContextPanel items={filterContextForTab('defensive')} presetName={presetName} />
      </div>
      <FilterTable
        filters={filters}
        candidateSide={dashState?.entry_candidate_side ?? null}
        candidateTrigger={dashState?.entry_candidate_trigger ?? null}
        lastBlockReason={dashState?.last_block_reason ?? null}
        isPhase3
      />
      <ExecutionLog executions={executions} />
    </>
  );

  const renderPhase3PaperTab = () => (
    <>
      <Phase3AcceptanceWidget acceptance={phase3Acceptance} />
      <Phase3ProvenanceCard provenance={phase3Provenance} />
      <Phase3DecisionSample decisions={phase3Decisions} />
    </>
  );

  const renderPhase3TabContent = () => {
    switch (phase3Tab) {
      case 'overview':
        return renderPhase3Overview();
      case 'tokyo':
        return renderPhase3SessionTab('tokyo');
      case 'london':
        return renderPhase3SessionTab('london');
      case 'ny':
        return renderPhase3SessionTab('ny');
      case 'defensive':
        return renderPhase3DefensiveTab();
      case 'paper':
        return renderPhase3PaperTab();
      default:
        return renderPhase3Overview();
    }
  };

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
        exitSystemOnly={runtime?.exit_system_only ?? false}
        onExitSystemOnlyChange={handleExitSystemOnlyChange}
      />

      {/* Main content */}
      <div style={{ flex: 1, padding: 12, display: 'flex', flexDirection: 'column', gap: 8, overflowY: 'auto' }}>
        {loopError && (
          <div style={{
            backgroundColor: colors.panel,
            border: `1px solid ${colors.red}`,
            borderRadius: 6,
            padding: 12,
            color: colors.red,
            whiteSpace: 'pre-wrap',
          }}>
            {loopError}
          </div>
        )}
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
        {isPhase3Preset && (
          <Phase3StatusStrip items={context} presetName={presetName} />
        )}
        {isPhase3Preset ? (
          <>
            <div style={{
              backgroundColor: colors.panel,
              borderRadius: 8,
              border: `1px solid ${colors.border}`,
              padding: 8,
              display: 'flex',
              gap: 8,
              flexWrap: 'wrap',
            }}>
              {tabButtons.map((tab) => {
                const active = tab.id === phase3Tab;
                return (
                  <button
                    key={tab.id}
                    type="button"
                    onClick={() => setPhase3Tab(tab.id)}
                    style={{
                      border: `1px solid ${active ? colors.blue : colors.border}`,
                      backgroundColor: active ? `${colors.blue}18` : 'transparent',
                      color: active ? colors.blue : colors.textSecondary,
                      borderRadius: 999,
                      padding: '6px 12px',
                      fontSize: 12,
                      fontWeight: 700,
                      cursor: 'pointer',
                    }}
                  >
                    {tab.label}
                  </button>
                );
              })}
            </div>
            {renderPhase3TabContent()}
          </>
        ) : (
          <>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              <ContextPanel items={context} />
              <TradeLog title="Trade Log (Last 30)" events={recentOpenEvents} />
            </div>
            <FilterTable
              filters={filters}
              candidateSide={dashState?.entry_candidate_side ?? null}
              candidateTrigger={dashState?.entry_candidate_trigger ?? null}
              lastBlockReason={dashState?.last_block_reason ?? null}
              isPhase3={false}
            />
            <TradeLog title="Closed Trades (Last 30)" events={recentClosedEvents} />
            <ExecutionLog executions={executions} />
          </>
        )}

        {/* Loop Log — OG style */}
        {loopLog.length > 0 && (
          <div style={{
            backgroundColor: colors.panel, borderRadius: 6,
            border: `1px solid ${colors.border}`, padding: 12,
          }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: colors.textSecondary, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>
              Loop Log ({loopLog.length})
            </div>
            <div style={{ maxHeight: 300, overflowY: 'auto' }}>
              <pre style={{ margin: 0, fontFamily: 'monospace', fontSize: 11, lineHeight: 1.5, color: colors.textSecondary, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
{[...loopLog].reverse().map((e) => `${e.ts} [${e.level}] ${e.msg}`).join('\n')}
              </pre>
            </div>
          </div>
        )}

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
