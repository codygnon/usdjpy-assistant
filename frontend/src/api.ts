// API client for USDJPY Assistant

const API_BASE = '/api';

export interface ProfileInfo {
  path: string;
  name: string;
}

export interface Preset {
  id: string;
  name: string;
  description: string;
  pros: string[];
  cons: string[];
}

export interface RuntimeState {
  mode: string;
  kill_switch: boolean;
  exit_system_only: boolean;
  last_processed_bar_time_utc: string | null;
  loop_running: boolean;
}

export interface QuickStats {
  closed_trades: number;
  win_rate: number | null;
  avg_pips: number | null;
  total_profit?: number | null;
  total_commission?: number | null;
  total_swap?: number | null;
  display_currency?: string;
  source?: 'mt5' | 'oanda' | 'database';
  wins?: number;
  losses?: number;
  trades_with_profit?: number;
  trades_without_profit?: number;
  trades_with_position_id?: number;
}

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error: ${res.status} - ${text}`);
  }
  return res.json();
}

// Profiles
export async function listProfiles(): Promise<ProfileInfo[]> {
  return fetchJson<ProfileInfo[]>(`${API_BASE}/profiles`);
}

export async function getProfile(path: string): Promise<Record<string, unknown>> {
  return fetchJson<Record<string, unknown>>(`${API_BASE}/profiles/${encodeURIComponent(path)}`);
}

export async function saveProfile(path: string, data: Record<string, unknown>): Promise<void> {
  await fetchJson<unknown>(`${API_BASE}/profiles/${encodeURIComponent(path)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ profile_data: data }),
  });
}

export async function createProfile(name: string): Promise<ProfileInfo> {
  return fetchJson<ProfileInfo>(`${API_BASE}/profiles`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  });
}

export async function deleteProfile(path: string): Promise<{ status: string; path: string }> {
  return fetchJson<{ status: string; path: string }>(
    `${API_BASE}/profiles?path=${encodeURIComponent(path)}`,
    { method: 'DELETE' }
  );
}

// Authentication
export async function checkAuth(profilePath: string): Promise<{ has_password: boolean }> {
  return fetchJson<{ has_password: boolean }>(
    `${API_BASE}/auth/check?profile_path=${encodeURIComponent(profilePath)}`
  );
}

export async function authLogin(profilePath: string, password: string): Promise<{ success: boolean }> {
  return fetchJson<{ success: boolean }>(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ profile_path: profilePath, password }),
  });
}

export async function setPassword(
  profilePath: string,
  currentPassword: string | null,
  newPassword: string
): Promise<{ success: boolean }> {
  return fetchJson<{ success: boolean }>(`${API_BASE}/auth/set-password`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      profile_path: profilePath,
      current_password: currentPassword,
      new_password: newPassword,
    }),
  });
}

export async function removePassword(profilePath: string, password: string): Promise<{ success: boolean }> {
  return fetchJson<{ success: boolean }>(`${API_BASE}/auth/remove-password`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ profile_path: profilePath, password }),
  });
}

// Presets
export async function listPresets(): Promise<Preset[]> {
  return fetchJson<Preset[]>(`${API_BASE}/presets`);
}

export async function previewPreset(presetId: string, profilePath: string): Promise<Record<string, unknown>> {
  return fetchJson<Record<string, unknown>>(
    `${API_BASE}/presets/${presetId}/preview?profile_path=${encodeURIComponent(profilePath)}`
  );
}

export interface ApplyPresetOptions {
  vwap_session_filter_enabled?: boolean;
}

export async function applyPreset(
  presetId: string,
  profilePath: string,
  options?: ApplyPresetOptions
): Promise<Record<string, unknown>> {
  const body: { preset_id: string; options?: ApplyPresetOptions } = { preset_id: presetId };
  if (options && Object.keys(options).length > 0) {
    body.options = options;
  }
  return fetchJson<Record<string, unknown>>(
    `${API_BASE}/presets/${presetId}/apply?profile_path=${encodeURIComponent(profilePath)}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }
  );
}

// Runtime state
export async function getRuntimeState(profileName: string): Promise<RuntimeState> {
  return fetchJson<RuntimeState>(`${API_BASE}/runtime/${profileName}`);
}

export async function updateRuntimeState(
  profileName: string,
  mode: string,
  killSwitch: boolean,
  exitSystemOnly: boolean = false
): Promise<void> {
  await fetchJson<unknown>(`${API_BASE}/runtime/${profileName}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode, kill_switch: killSwitch, exit_system_only: exitSystemOnly }),
  });
}

// Temporary EMA Settings (Apply Temporary Settings menu)
export interface TempEmaSettings {
  m5_trend_ema_fast: number | null;
  m5_trend_ema_slow: number | null;
  m5_trend_source: 'closed_m5' | 'synthetic_live_m5' | null;
  m1_zone_entry_ema_slow: number | null;
  m1_pullback_cross_ema_slow: number | null;
  // Trial #4 fields (Zone Entry only - Tiered Pullback uses fixed tiers)
  m3_trend_ema_fast: number | null;
  m3_trend_ema_slow: number | null;
  m1_t4_zone_entry_ema_fast: number | null;
  m1_t4_zone_entry_ema_slow: number | null;
  // Uncle Parsh H1 Breakout EMA overrides
  up_m5_ema_fast: number | null;
  up_m5_ema_slow: number | null;
  // Uncle Parsh H1 Breakout: H1 Detection
  up_major_extremes_only: boolean | null;
  up_h1_lookback_hours: number | null;
  up_h1_swing_strength: number | null;
  up_h1_cluster_tolerance_pips: number | null;
  up_h1_min_touches_for_major: number | null;
  up_h1_min_distance_between_levels_pips: number | null;
  // Uncle Parsh H1 Breakout: M5 Catalyst
  up_power_close_body_pct: number | null;
  up_velocity_pips: number | null;
  // Uncle Parsh H1 Breakout: Exit Strategy
  up_initial_sl_spread_plus_pips: number | null;
  up_tp1_pips: number | null;
  up_tp1_close_pct: number | null;
  up_be_spread_plus_pips: number | null;
  up_trail_ema_period: number | null;
  // Uncle Parsh H1 Breakout: Discipline
  up_max_spread_pips: number | null;
  // Trial #8 exit strategy
  t8_exit_strategy: 'none' | 'tp1_be_trail' | 'ema_scale_runner' | null;
  t8_tp1_pips: number | null;
  t8_tp1_close_pct: number | null;
  t8_be_spread_plus_pips: number | null;
  t8_trail_ema_period: number | null;
  t8_m1_exit_ema_fast: number | null;
  t8_m1_exit_ema_slow: number | null;
  t8_scale_out_pct: number | null;
  t8_initial_sl_spread_plus_pips: number | null;
  // Trial #9 Exit Strategy + TP1/BE/Trail
  t9_exit_strategy: string | null;
  t9_hwm_trail_pips: number | null;
  t9_tp1_pips: number | null;
  t9_tp1_close_pct: number | null;
  t9_be_spread_plus_pips: number | null;
  t9_trail_ema_period: number | null;
  t9_trail_m5_ema_period: number | null;
  // Trial #10: Regime Gate / execution
  t10_regime_gate_enabled: boolean | null;
  t10_regime_london_sell_veto: boolean | null;
  t10_regime_london_start_hour_et: number | null;
  t10_regime_london_end_hour_et: number | null;
  t10_regime_boost_multiplier: number | null;
  t10_regime_buy_base_multiplier: number | null;
  t10_regime_sell_base_multiplier: number | null;
  t10_regime_chop_pause_enabled: boolean | null;
  t10_regime_chop_pause_minutes: number | null;
  t10_regime_chop_pause_lookback_trades: number | null;
  t10_regime_chop_pause_stop_rate: number | null;
  t10_tier17_nonboost_multiplier: number | null;
  t10_max_directional_lots_per_side: number | null;
  t10_bucketed_exit_enabled: boolean | null;
  t10_quick_tp1_pips: number | null;
  t10_quick_tp1_close_pct: number | null;
  t10_quick_be_spread_plus_pips: number | null;
  t10_runner_tp1_pips: number | null;
  t10_runner_tp1_close_pct: number | null;
  t10_runner_be_spread_plus_pips: number | null;
  t10_trail_escalation_enabled: boolean | null;
  t10_trail_escalation_tier1_pips: number | null;
  t10_trail_escalation_tier2_pips: number | null;
  t10_trail_escalation_m15_ema_period: number | null;
  t10_trail_escalation_m15_buffer_pips: number | null;
  t10_runner_score_sizing_enabled: boolean | null;
  t10_runner_base_lots: number | null;
  t10_runner_min_lots: number | null;
  t10_runner_max_lots: number | null;
  t10_atr_stop_enabled: boolean | null;
  t10_atr_stop_multiplier: number | null;
  t10_atr_stop_max_pips: number | null;
}

export async function getTempSettings(profileName: string): Promise<TempEmaSettings> {
  return fetchJson<TempEmaSettings>(`${API_BASE}/runtime/${profileName}/temp-settings`);
}

export async function updateTempSettings(
  profileName: string,
  settings: Partial<TempEmaSettings>
): Promise<void> {
  await fetchJson<unknown>(`${API_BASE}/runtime/${profileName}/temp-settings`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(settings),
  });
}

// Loop control
export async function startLoop(profileName: string, profilePath: string): Promise<{ status: string; pid?: number }> {
  return fetchJson<{ status: string; pid?: number }>(
    `${API_BASE}/loop/${profileName}/start?profile_path=${encodeURIComponent(profilePath)}`,
    { method: 'POST' }
  );
}

export async function stopLoop(profileName: string): Promise<{ status: string }> {
  return fetchJson<{ status: string }>(`${API_BASE}/loop/${profileName}/stop`, { method: 'POST' });
}

export async function getLoopLog(profileName: string, lines = 100): Promise<{ exists: boolean; content: string; total_lines: number }> {
  return fetchJson<{ exists: boolean; content: string; total_lines: number }>(
    `${API_BASE}/loop/${profileName}/log?lines=${lines}`
  );
}

// Data
export async function getSnapshots(profileName: string, limit = 20): Promise<Record<string, unknown>[]> {
  return fetchJson<Record<string, unknown>[]>(`${API_BASE}/data/${profileName}/snapshots?limit=${limit}`);
}

export interface TradesResponse {
  trades: Record<string, unknown>[];
  display_currency?: string;
}

export async function getTrades(
  profileName: string,
  limit = 50,
  profilePath?: string,
  sync = false
): Promise<TradesResponse> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (profilePath) params.set('profile_path', profilePath);
  if (sync) params.set('sync', 'true');
  const url = `${API_BASE}/data/${profileName}/trades?${params}`;
  const data = await fetchJson<Record<string, unknown>[] | { trades: Record<string, unknown>[]; display_currency: string }>(url);
  if (data != null && typeof data === 'object' && 'trades' in data && Array.isArray((data as { trades: unknown }).trades)) {
    return { trades: (data as { trades: Record<string, unknown>[] }).trades, display_currency: (data as { display_currency?: string }).display_currency };
  }
  return { trades: Array.isArray(data) ? data : [], display_currency: undefined };
}

export interface TradeHistoryDay {
  date: string;
  daily_profit: number;
  cum_profit: number;
  trade_count: number;
}

export interface TradeHistoryResponse {
  days: TradeHistoryDay[];
  display_currency?: string;
  source?: string;
}

export async function getTradeHistory(
  profileName: string,
  profilePath?: string,
  daysBack = 90
): Promise<TradeHistoryResponse> {
  const params = new URLSearchParams({ days_back: String(daysBack) });
  if (profilePath) params.set('profile_path', profilePath);
  return fetchJson<TradeHistoryResponse>(`${API_BASE}/data/${profileName}/trade-history?${params}`);
}

// Trade history detail (per-trade, not daily aggregate) for analytics
export interface TradeDetail {
  side: string;
  entry_price: number;
  exit_price: number;
  entry_time_utc: string;
  exit_time_utc: string;
  profit: number | null;
  pips: number | null;
  volume: number;
  spread_pips: number | null;
}

export interface TradeHistoryDetailResponse {
  trades: TradeDetail[];
  display_currency?: string;
  source?: string;
}

export async function getTradeHistoryDetail(
  profileName: string,
  profilePath?: string,
  daysBack = 90
): Promise<TradeHistoryDetailResponse> {
  const params = new URLSearchParams({ days_back: String(daysBack) });
  if (profilePath) params.set('profile_path', profilePath);
  return fetchJson<TradeHistoryDetailResponse>(`${API_BASE}/data/${profileName}/trade-history-detail?${params}`);
}

export async function getExecutions(profileName: string, limit = 50): Promise<Record<string, unknown>[]> {
  return fetchJson<Record<string, unknown>[]>(`${API_BASE}/data/${profileName}/executions?limit=${limit}`);
}

export async function getRejectionBreakdown(profileName: string): Promise<Record<string, number>> {
  return fetchJson<Record<string, number>>(`${API_BASE}/data/${profileName}/rejection-breakdown`);
}

export interface Phase3DecisionRow {
  timestamp_utc?: string | null;
  signal_id?: string | null;
  mode?: string | null;
  attempted?: number | boolean | null;
  placed?: number | boolean | null;
  reason?: string | null;
  reason_text?: string | null;
  blocking_filter_ids?: string[];
  phase3_session?: string | null;
}

export async function getPhase3Decisions(
  profileName: string,
  days = 3,
  limit = 5000,
  profilePath?: string
): Promise<Phase3DecisionRow[]> {
  const params = new URLSearchParams({ days: String(days), limit: String(limit) });
  if (profilePath) params.set('profile_path', profilePath);
  return fetchJson<Phase3DecisionRow[]>(`${API_BASE}/data/${profileName}/phase3-decisions?${params}`);
}

export async function getPhase3BlockersBreakdown(
  profileName: string,
  days = 7,
  limit = 20000,
  profilePath?: string
): Promise<Record<string, number>> {
  const params = new URLSearchParams({ days: String(days), limit: String(limit) });
  if (profilePath) params.set('profile_path', profilePath);
  return fetchJson<Record<string, number>>(`${API_BASE}/data/${profileName}/phase3-blockers-breakdown?${params}`);
}

export interface Phase3Provenance {
  generated_at_utc: string;
  package_id: string | null;
  preset_name: string;
  session: string | null;
  strategy_tag: string | null;
  strategy_family: string | null;
  window_label: string | null;
  ownership_cell: string | null;
  regime_label: string | null;
  defensive_flags: string[];
  attempted: boolean;
  placed: boolean;
  outcome: 'placed' | 'blocked' | 'waiting';
  reason: string | null;
  blocking_filter_ids: string[];
  last_block_reason: string | null;
  exit_policy: {
    label: string;
    tp1_r: number;
    be_offset_pips: number;
    tp2_r: number;
  } | null;
  frozen_modifiers: Array<{
    id: string;
    label: string;
    value: string;
  }>;
  data_freshness: {
    dashboard_timestamp_utc: string | null;
    decision_timestamp_utc: string | null;
  };
}

export interface Phase3AcceptanceSummary {
  available: boolean;
  generated_at_utc: string | null;
  package_under_test: string | null;
  verdict: string | null;
  verdict_note: string | null;
  observed_summary: {
    OBSERVED_count?: number;
    IMPLEMENTED_AND_INSTRUMENTED_AWAITING_OBSERVED_FIRE_count?: number;
    BROKEN_count?: number;
  } | null;
  rules: Array<{
    id: string;
    label: string;
    requirement: string;
    status: string;
    evidence_pointer?: string | null;
  }>;
  immediate_next_action: string | null;
}

export interface Phase3DefensiveMonitor {
  available: boolean;
  generated_at_utc: string | null;
  frozen_package_id: string | null;
  strategy: string | null;
  ownership_cell: string | null;
  paper_monitor_executed: boolean | null;
  paper_monitor_skip_reason: string | null;
  log_path_used: string | null;
  guardrail_status: Record<string, unknown> | null;
  pause_recommended: boolean | null;
  rollback_reference: Record<string, unknown> | null;
  next_command_when_log_exists: string | null;
  searched_locations: string[];
  research_baseline_blocked_trade_counts: Record<string, unknown> | null;
}

export async function getPhase3Provenance(profileName: string, profilePath?: string): Promise<Phase3Provenance> {
  const params = profilePath ? `?profile_path=${encodeURIComponent(profilePath)}` : '';
  return fetchJson<Phase3Provenance>(`${API_BASE}/data/${profileName}/phase3-provenance${params}`);
}

export async function getPhase3PaperAcceptance(): Promise<Phase3AcceptanceSummary> {
  return fetchJson<Phase3AcceptanceSummary>(`${API_BASE}/system/phase3-paper-acceptance`);
}

export async function getPhase3DefensiveMonitor(): Promise<Phase3DefensiveMonitor> {
  return fetchJson<Phase3DefensiveMonitor>(`${API_BASE}/system/phase3-defensive-monitor`);
}

export async function getQuickStats(profileName: string, profilePath?: string, sync = false): Promise<QuickStats> {
  const params = new URLSearchParams();
  if (profilePath) params.set('profile_path', profilePath);
  if (sync) params.set('sync', 'true');
  const qs = params.toString();
  const url = qs ? `${API_BASE}/data/${profileName}/stats?${qs}` : `${API_BASE}/data/${profileName}/stats`;
  return fetchJson<QuickStats>(url);
}

// Per-preset statistics
export interface PresetStats {
  total_trades: number;
  wins: number;
  losses: number;
  win_rate: number | null;
  total_pips: number;
  total_profit?: number | null;
  total_commission?: number;
  avg_pips: number | null;
  avg_rr: number | null;
  best_trade: number | null;
  worst_trade: number | null;
  win_streak: number;
  loss_streak: number;
  profit_factor: number | null;
  max_drawdown: number;
  /** Phase 3 Integrated only: stats per session (tokyo, london, ny) */
  by_session?: Record<string, PresetStats>;
}

export interface StatsByPreset {
  presets: Record<string, PresetStats>;
  source?: 'mt5' | 'oanda' | 'database';
  display_currency?: string;
}

export async function getStatsByPreset(profileName: string, profilePath?: string): Promise<StatsByPreset> {
  const url = profilePath
    ? `${API_BASE}/data/${profileName}/stats-by-preset?profile_path=${encodeURIComponent(profilePath)}`
    : `${API_BASE}/data/${profileName}/stats-by-preset`;
  return fetchJson<StatsByPreset>(url);
}

// MT5 Full Report (same as View -> Reports)
export interface Mt5Report {
  source: 'mt5' | 'oanda';
  display_currency?: string;
  summary: { balance: number; equity: number; margin: number; free_margin: number };
  closed_pl: {
    closed_trades: number;
    wins: number;
    losses: number;
    win_rate: number;
    total_profit: number;
    total_commission: number;
    total_swap: number;
    gross_profit: number;
    gross_loss: number;
    profit_factor: number;
    largest_profit_trade: number;
    largest_loss_trade: number;
    expected_payoff: number;
    avg_pips: number | null;
    total_pips: number;
  };
  long_short: {
    long_trades: number;
    long_wins: number;
    long_win_pct: number;
    short_trades: number;
    short_wins: number;
    short_win_pct: number;
  };
}

export async function getMt5Report(profileName: string, profilePath?: string): Promise<Mt5Report | null> {
  const url = profilePath
    ? `${API_BASE}/data/${profileName}/mt5-report?profile_path=${encodeURIComponent(profilePath)}`
    : `${API_BASE}/data/${profileName}/mt5-report`;
  const res = await fetch(url);
  if (!res.ok) return null;
  const data = await res.json();
  return data?.source === 'mt5' || data?.source === 'oanda' ? (data as Mt5Report) : null;
}

// Trade management
export async function closeTrade(
  profileName: string,
  tradeId: string,
  profilePath: string
): Promise<{ status: string; trade_id: string; exit_price: number; pips: number; r_multiple: number | null }> {
  return fetchJson(
    `${API_BASE}/trades/${profileName}/${tradeId}/close?profile_path=${encodeURIComponent(profilePath)}`,
    { method: 'POST' }
  );
}

export async function syncTrades(
  profileName: string,
  profilePath: string,
  forceProfitRefresh = true
): Promise<{ status: string; trades_updated: number; trades_imported: number; position_ids_backfilled: number; profit_backfilled?: number }> {
  const params = new URLSearchParams({ profile_path: profilePath });
  if (forceProfitRefresh) params.set('force_profit_refresh', 'true');
  return fetchJson(
    `${API_BASE}/data/${profileName}/sync-trades?${params}`,
    { method: 'POST' }
  );
}

// Technical Analysis
export interface TaRsi {
  value: number | null;
  zone: string;
  period: number;
}

export interface TaMacd {
  line: number | null;
  signal: number | null;
  histogram: number | null;
  direction: string;
}

export interface TaAtr {
  value: number | null;
  value_pips: number | null;
  state: string;
}

export interface TaPrice {
  current: number | null;
  recent_high: number | null;
  recent_low: number | null;
}

export interface OhlcBar {
  time: number;  // Unix timestamp
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface ScalpScoreLayer {
  name: string;
  score: number;
  max: number;
  components: Record<string, unknown>;
}

export interface ScalpScore {
  finalScore: number;
  direction: string;
  confidence: string;
  killSwitch: boolean;
  killReason: string | null;
  layers: ScalpScoreLayer[];
  timestamp: string;
}

export interface TaTimeframe {
  regime: string;
  rsi: TaRsi;
  macd: TaMacd;
  atr: TaAtr;
  price: TaPrice;
  summary: string;
  ohlc: OhlcBar[];
  all_emas?: Record<string, { time: number; value: number }[]>;
  bollinger_series?: { upper: { time: number; value: number }[]; middle: { time: number; value: number }[]; lower: { time: number; value: number }[] };
  scalp_score?: ScalpScore | null;
  error?: string;
}

export interface TechnicalAnalysis {
  timeframes: Record<string, TaTimeframe>;
  current_tick?: {
    bid: number;
    ask: number;
    spread_pips: number;
  };
}

export async function getTechnicalAnalysis(
  profileName: string,
  profilePath: string
): Promise<TechnicalAnalysis> {
  return fetchJson<TechnicalAnalysis>(
    `${API_BASE}/data/${profileName}/technical-analysis?profile_path=${encodeURIComponent(profilePath)}`
  );
}

export interface OpenTrade {
  trade_id: string;
  symbol: string;
  side: string;
  entry_price: number;
  stop_price: number | null;
  target_price: number | null;
  size_lots: number;
  timestamp_utc: string;
  mt5_order_id?: number;
  mt5_position_id?: number;
  unrealized_pl?: number;
  financing?: number;
  phase3_session?: string | null;
  phase3_strategy_family?: string | null;
  phase3_strategy_variant?: string | null;
  ownership_cell?: string | null;
  has_cell_attribution?: boolean;
}

export async function getOpenTrades(profileName: string, profilePath?: string, sync = true): Promise<OpenTrade[]> {
  const params = new URLSearchParams();
  if (profilePath) params.set('profile_path', profilePath);
  if (sync) params.set('sync', 'true');
  const qs = params.toString();
  return fetchJson<OpenTrade[]>(`${API_BASE}/data/${profileName}/open-trades${qs ? `?${qs}` : ''}`);
}

// Advanced Analytics
export interface AdvancedTrade {
  trade_id: string;
  side: string;
  entry_time_utc: string;
  exit_time_utc: string;
  entry_price: number;
  exit_price: number;
  pips: number | null;
  r_multiple: number | null;
  risk_pips: number | null;
  profit: number | null;
  duration_minutes: number | null;
  max_adverse_pips: number | null;
  max_favorable_pips: number | null;
  post_sl_recovery_pips: number | null;
  preset_name: string | null;
  exit_reason: string | null;
  entry_type: string | null;
  reversal_risk_tier: string | null;
  tier_number: number | null;
  phase3_session?: string | null;
  phase3_strategy_family?: string | null;
  phase3_strategy_variant?: string | null;
  ownership_cell?: string | null;
  has_cell_attribution?: boolean;
}

export interface AdvancedAnalyticsResponse {
  trades: AdvancedTrade[];
  display_currency: string;
  starting_balance?: number | null;
  total_profit_currency?: number | null;
}

export async function getAdvancedAnalytics(
  profileName: string, profilePath: string, daysBack = 365
): Promise<AdvancedAnalyticsResponse> {
  return fetchJson(`${API_BASE}/data/${profileName}/advanced-analytics?profile_path=${encodeURIComponent(profilePath)}&days_back=${daysBack}`);
}

// Filter Config (from profile JSON — used when loop is not running)
export interface FilterConfig {
  preset_name: string;
  filters: Record<string, Record<string, unknown>>;
}

export async function getFilterConfig(profileName: string, profilePath: string): Promise<FilterConfig> {
  return fetchJson<FilterConfig>(
    `${API_BASE}/data/${profileName}/filter-config?profile_path=${encodeURIComponent(profilePath)}`
  );
}

// Dashboard
export interface FilterReport {
  filter_id: string;
  display_name: string;
  enabled: boolean;
  is_clear: boolean;
  current_value: string;
  threshold: string;
  block_reason: string | null;
  /** Plain-English one-liner for user (pass or fail). */
  explanation?: string | null;
  sub_filters: FilterReport[];
  metadata: Record<string, unknown>;
}

export interface ContextItem {
  key: string;
  value: string;
  category: string;
  /** Slope-directed coloring: green = increasing, red = decreasing */
  valueColor?: 'green' | 'red';
}

export interface PositionInfo {
  trade_id: string;
  side: string;
  entry_price: number;
  size_lots: number | null;
  entry_type: string | null;
  current_price: number;
  unrealized_pips: number;
  age_minutes: number;
  stop_price: number | null;
  target_price: number | null;
  breakeven_applied: boolean;
  phase3_session?: string | null;
  phase3_strategy_family?: string | null;
  phase3_strategy_variant?: string | null;
  ownership_cell?: string | null;
  has_cell_attribution?: boolean;
}

export interface DailySummary {
  trades_today: number;
  wins: number;
  losses: number;
  total_pips: number;
  total_profit: number;
  win_rate: number;
}

export interface DashboardState {
  timestamp_utc: string | null;
  preset_name: string;
  mode: string;
  loop_running: boolean;
  entry_candidate_side?: 'buy' | 'sell' | null;
  entry_candidate_trigger?: 'zone_entry' | 'tiered_pullback' | null;
  last_block_reason?: string | null;
  filters: FilterReport[];
  context: ContextItem[];
  positions: PositionInfo[];
  daily_summary: DailySummary | null;
  bid: number;
  ask: number;
  spread_pips: number;
  stale?: boolean;
  stale_age_seconds?: number | null;
  data_source?: 'run_loop_file' | 'none';
  error?: string;
}

export interface TradeEvent {
  event_type: string;
  timestamp_utc: string;
  trade_id: string;
  side: string;
  entry_type: string | null;
  price: number;
  trigger_type: string | null;
  pips: number | null;
  profit: number | null;
  exit_reason: string | null;
  context_snapshot: Record<string, unknown>;
  spread_at_entry: number | null;
  phase3_session?: string | null;
  phase3_strategy_family?: string | null;
  phase3_strategy_variant?: string | null;
  ownership_cell?: string | null;
  has_cell_attribution?: boolean;
}

export async function getDashboard(profileName: string, profilePath?: string): Promise<DashboardState> {
  const params = profilePath ? `?profile_path=${encodeURIComponent(profilePath)}` : '';
  return fetchJson<DashboardState>(`${API_BASE}/data/${profileName}/dashboard${params}`);
}

export async function getTradeEvents(profileName: string, limit = 50, profilePath?: string): Promise<TradeEvent[]> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (profilePath) params.set('profile_path', profilePath);
  return fetchJson<TradeEvent[]>(`${API_BASE}/data/${profileName}/trade-events?${params.toString()}`);
}

export interface Trial7ReversalRiskStatus {
  enabled: boolean;
  available: boolean;
  score: number | null;
  tier: string | null;
  regime?: string | null;
  lot_multiplier?: number | null;
  min_tier_ema?: number | null;
  zone_block_entry?: boolean | null;
  use_managed_exit?: boolean | null;
  timestamp_utc?: string | null;
  source?: string;
}

export async function getTrial7ReversalRiskStatus(profileName: string, profilePath?: string): Promise<Trial7ReversalRiskStatus> {
  const params = profilePath ? `?profile_path=${encodeURIComponent(profilePath)}` : '';
  return fetchJson<Trial7ReversalRiskStatus>(`${API_BASE}/data/${profileName}/reversal-risk${params}`);
}

// --- AI Trading Assistant (streaming chat; POST + SSE) ---

export interface AiChatHistoryMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface AiChatRequestBody {
  message: string;
  history: AiChatHistoryMessage[];
  /** Must be in server allowlist from GET /api/ai-chat/models */
  chat_model?: string;
}

export interface AiChatModelsResponse {
  models: string[];
  default_model: string;
  suggest_models?: string[];
  default_suggest_model?: string;
}

export async function getAiChatModels(): Promise<AiChatModelsResponse> {
  return fetchJson<AiChatModelsResponse>(`${API_BASE}/ai-chat/models`);
}

export interface AiRailLevel {
  price: number;
  weight_pct: number;
  distance_pips: number | null;
  direction: 'support' | 'resistance';
}

export interface AiRailEvent {
  timestamp_utc: string;
  date: string;
  time: string;
  currency: string;
  event: string;
  impact: string;
  minutes_to_event: number;
}

export interface AiRailPayload {
  as_of: string | null;
  macro: {
    combined_bias: string;
    confidence: string;
    implication: string;
    dxy: { value: number | null; one_day: number | null; five_day: number | null };
    us10y: { value: number | null; one_day: number | null; five_day: number | null };
    oil: { value: number | null; one_day: number | null; five_day: number | null };
    gold: { value: number | null; one_day: number | null; five_day: number | null };
  };
  events: AiRailEvent[];
  levels: {
    mid: number | null;
    supports: AiRailLevel[];
    resistances: AiRailLevel[];
  };
  session_vol: {
    active_sessions: string[];
    overlap?: string | null;
    next_close?: string | null;
    warnings: string[];
    spread_pips?: number | null;
    vol_label?: string | null;
    vol_ratio?: number | null;
    recent_avg_pips?: number | null;
  };
}

export async function getAiRail(profileName: string, profilePath: string, daysAhead = 7): Promise<AiRailPayload> {
  const params = new URLSearchParams({
    profile_path: profilePath,
    days_ahead: String(daysAhead),
  });
  return fetchJson<AiRailPayload>(`${API_BASE}/data/${encodeURIComponent(profileName)}/ai-rail?${params.toString()}`);
}

// --- AI Trade Suggestion & Limit Order Placement ---

export interface AiExitStrategyInfo {
  id: string;
  label: string;
  description: string;
  defaults: Record<string, number>;
  trail_mode: string | null;
}

export interface AiTradeSuggestion {
  side: string;
  price: number;
  sl: number;
  tp: number;
  lots: number;
  time_in_force: string;
  gtd_time_utc: string | null;
  expiry_option?: string | null;
  rationale: string;
  confidence: string;
  exit_strategy?: string;
  exit_params?: Record<string, number> | null;
  available_exit_strategies?: Record<string, AiExitStrategyInfo>;
  model_used?: string;
  suggestion_id?: string | null;
}

export interface AiSuggestionHistoryItem {
  suggestion_id: string;
  created_utc: string;
  profile: string;
  model: string;
  side: string;
  limit_price: number;
  sl: number | null;
  tp: number | null;
  lots: number;
  time_in_force: string | null;
  gtd_time_utc: string | null;
  confidence: string | null;
  rationale: string | null;
  exit_strategy: string | null;
  exit_params: Record<string, unknown>;
  market_snapshot: Record<string, unknown>;
  action: string | null;
  action_utc: string | null;
  edited_fields: Record<string, unknown>;
  placed_order: Record<string, unknown>;
  oanda_order_id: string | null;
  trade_id: string | null;
  outcome_status: string | null;
  filled_at: string | null;
  fill_price: number | null;
  closed_at: string | null;
  exit_price: number | null;
  pnl: number | null;
  pips: number | null;
  win_loss: string | null;
}

export interface AiSuggestionHistoryResponse {
  total: number;
  limit: number;
  offset: number;
  items: AiSuggestionHistoryItem[];
}

export interface PlaceLimitOrderRequest {
  side: string;
  price: number;
  lots: number;
  sl?: number | null;
  tp?: number | null;
  time_in_force?: string;
  gtd_time_utc?: string | null;
  comment?: string;
  exit_strategy?: string | null;
  exit_params?: Record<string, number> | null;
  suggestion_id?: string | null;
  edited_fields?: Record<string, { before: unknown; after: unknown }> | null;
}

export interface PlaceLimitOrderResponse {
  status: string;
  order_id: number | null;
  side: string;
  price: number;
  lots: number;
  sl: number | null;
  tp: number | null;
  time_in_force: string;
  gtd_time_utc: string | null;
  exit_strategy?: string | null;
  exit_params?: Record<string, number> | null;
  suggestion_id?: string | null;
  loop_auto_started?: boolean;
}

export async function aiSuggestTrade(
  profileName: string,
  profilePath: string,
  suggestModel?: string,
): Promise<AiTradeSuggestion> {
  const params = new URLSearchParams({ profile_path: profilePath });
  return fetchJson<AiTradeSuggestion>(
    `${API_BASE}/data/${encodeURIComponent(profileName)}/ai-suggest-trade?${params.toString()}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ suggest_model: suggestModel || null }),
    },
  );
}

export async function placeLimitOrder(
  profileName: string,
  profilePath: string,
  order: PlaceLimitOrderRequest,
): Promise<PlaceLimitOrderResponse> {
  const params = new URLSearchParams({ profile_path: profilePath });
  return fetchJson<PlaceLimitOrderResponse>(
    `${API_BASE}/data/${encodeURIComponent(profileName)}/place-limit-order?${params.toString()}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(order),
    },
  );
}

export async function logSuggestionAction(
  profileName: string,
  suggestionId: string,
  action: 'placed' | 'rejected',
  editedFields?: Record<string, { before: unknown; after: unknown }> | null,
  oandaOrderId?: string | null,
): Promise<void> {
  await fetchJson<unknown>(
    `${API_BASE}/data/${encodeURIComponent(profileName)}/ai-suggestions/action`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        suggestion_id: suggestionId,
        action,
        edited_fields: editedFields || null,
        oanda_order_id: oandaOrderId || null,
      }),
    },
  );
}

export async function getAiSuggestionHistory(
  profileName: string,
  limit = 100,
  offset = 0,
): Promise<AiSuggestionHistoryResponse> {
  const params = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
  });
  return fetchJson<AiSuggestionHistoryResponse>(
    `${API_BASE}/data/${encodeURIComponent(profileName)}/ai-suggestions/history?${params.toString()}`
  );
}

// ---------- Autonomous Fillmore ----------

export interface AutonomousConfig {
  enabled: boolean;
  mode: 'off' | 'shadow' | 'paper' | 'live';
  aggressiveness: 'conservative' | 'balanced' | 'aggressive' | 'very_aggressive';
  limit_gtd_minutes: number;
  daily_budget_usd: number;
  min_llm_cooldown_sec: number;
  trading_hours: { tokyo: boolean; london: boolean; ny: boolean };
  max_lots_per_trade: number;
  base_lot_size: number;
  lot_deviation: number;
  max_open_ai_trades: number;
  max_daily_loss_usd: number;
  max_consecutive_errors: number;
  model: string;
  throttle_no_trade_streak: number;
  throttle_no_trade_cooldown_sec: number;
  throttle_loss_streak: number;
  throttle_loss_cooldown_sec: number;
  correlation_veto_enabled: boolean;
  correlation_distance_pips: number;
  repeat_setup_dedupe_enabled: boolean;
  repeat_setup_window_min: number;
  repeat_setup_bucket_pips: number;
  event_blackout_enabled: boolean;
  event_blackout_minutes: number;
  multi_trade_enabled: boolean;
  max_suggestions_per_call: number;
}

export interface AutonomousGateMode {
  description: string;
  expected_pass_rate_pct: number;
}

export interface AutonomousDecision {
  t: string;              // timestamp_utc
  r: 'pass' | 'block';
  l: 'hard' | 'signal' | 'throttle' | 'pass';
  why: string;
  mode: string;
  agg: string;
  x?: Record<string, unknown>;
}

export interface AutonomousStats {
  config: AutonomousConfig;
  gate_description?: string;
  expected_pass_rate_pct?: number;
  est_cost_per_llm_call_usd: number;
  window: {
    total: number;
    passes: number;
    blocks: number;
    pass_rate_pct: number;
    top_block_layers: Record<string, number>;
    top_block_reasons: Record<string, number>;
    trigger_families: Record<string, number>;
    trigger_reasons: Record<string, number>;
  };
  today: {
    llm_calls: number;
    spend_usd: number;
    budget_usd: number;
    budget_used_pct: number;
    trades_placed: number;
    pnl_usd: number;
  };
  throttle: {
    active: boolean;
    until_utc: string | null;
    reason: string | null;
    consecutive_no_trade_replies: number;
    consecutive_losses: number;
    consecutive_wins: number;
    consecutive_errors: number;
  };
  risk_regime: {
    label: 'normal' | 'defensive_soft' | 'defensive_hard' | string;
    streak_label: string;
    daily_drawdown_active: boolean;
    risk_multiplier: number;
    effective_min_llm_cooldown_sec: number;
    effective_max_open_ai_trades: number;
    override_label?: string | null;
    override_until_utc?: string | null;
    previous_regime_label?: string;
    recovery_wins?: number;
    regime_entered_trade_id?: string | null;
  };
  recent_gate_blocks: Record<string, number>;
  health_alerts: Array<{
    level: 'warning' | 'error' | string;
    code: string;
    msg: string;
  }>;
  order_metrics: {
    suggested: Record<string, number>;
    placed: Record<string, number>;
    filled: Record<string, number>;
    cancelled: Record<string, number>;
    expired: Record<string, number>;
    by_trigger_family: Record<string, {
      suggested: Record<string, number>;
      placed: Record<string, number>;
      filled: Record<string, number>;
      cancelled: Record<string, number>;
      expired: Record<string, number>;
      fill_rate: Record<string, number | null>;
      avg_time_to_fill_sec: Record<string, number | null>;
    }>;
  };
  performance: Record<string, {
    trade_count: number;
    closed_count: number;
    win_rate: number | null;
    avg_win_pips: number | null;
    avg_loss_pips: number | null;
    profit_factor: number | null;
    avg_hold_minutes: number | null;
    fill_rate_limits: number | null;
    avg_time_to_fill_sec: number | null;
    avg_fill_vs_requested_pips: number | null;
    thesis_intervention_rate: number | null;
    avg_mae_pips: number | null;
    avg_mfe_pips: number | null;
    win_rate_by_confidence_json: Record<string, number>;
    win_rate_by_side_json: Record<string, number>;
    win_rate_by_session_json: Record<string, number>;
    prompt_version_breakdown_json: Record<string, unknown>;
    updated_utc: string;
  }>;
  last_tick_utc: string | null;
  last_llm_call_utc: string | null;
  last_placed_order_id: string | null;
  last_suggestion_id: string | null;
  recent_decisions: AutonomousDecision[];
}

export async function getAutonomousConfig(profileName: string): Promise<{ config: AutonomousConfig; gate_modes: Record<string, AutonomousGateMode> }> {
  return fetchJson(`${API_BASE}/data/${encodeURIComponent(profileName)}/autonomous/config`);
}

export async function updateAutonomousConfig(
  profileName: string,
  patch: Partial<AutonomousConfig>,
): Promise<{ config: AutonomousConfig }> {
  return fetchJson(
    `${API_BASE}/data/${encodeURIComponent(profileName)}/autonomous/config`,
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    },
  );
}

export async function getAutonomousStats(profileName: string): Promise<AutonomousStats> {
  return fetchJson(`${API_BASE}/data/${encodeURIComponent(profileName)}/autonomous/stats`);
}

export async function resetAutonomousThrottle(profileName: string): Promise<AutonomousStats> {
  return fetchJson(`${API_BASE}/data/${encodeURIComponent(profileName)}/autonomous/reset-throttle`, {
    method: 'POST',
  });
}

export interface ReasoningSuggestion {
  suggestion_id: string;
  created_utc: string;
  side: string;
  requested_price: number | null;
  price: number;
  lots: number;
  quality: string | null;
  rationale: string | null;
  exit_plan: string | null;
  exit_strategy: string | null;
  action: string | null;
  outcome_status: string | null;
  win_loss: string | null;
  pips: number | null;
  pnl: number | null;
}

export interface ReasoningThesisCheck {
  id: number;
  profile: string;
  suggestion_id: string;
  trade_id: string;
  created_utc: string;
  action: string;
  reason: string;
  confidence: string;
  requested_new_sl: number | null;
  requested_scale_out_pct: number | null;
  execution_succeeded: boolean | null;
}

export interface ReasoningReflection {
  id: number;
  profile: string;
  suggestion_id: string;
  trade_id: string;
  created_utc: string;
  what_read_right: string;
  what_missed: string;
  summary_text: string;
  autonomous: boolean;
}

export interface ReasoningFeed {
  suggestions: ReasoningSuggestion[];
  thesis_checks: ReasoningThesisCheck[];
  reflections: ReasoningReflection[];
}

export async function getAutonomousReasoning(profileName: string): Promise<ReasoningFeed> {
  return fetchJson(`${API_BASE}/data/${encodeURIComponent(profileName)}/autonomous/reasoning`);
}

async function readApiErrorDetail(res: Response): Promise<string> {
  const text = await res.text();
  try {
    const j = JSON.parse(text) as { detail?: unknown };
    if (typeof j.detail === 'string') return j.detail;
    if (Array.isArray(j.detail)) {
      return j.detail.map((x) => (typeof x === 'object' && x && 'msg' in x ? String((x as { msg: unknown }).msg) : String(x))).join('; ');
    }
    return text || res.statusText;
  } catch {
    return text || res.statusText || `HTTP ${res.status}`;
  }
}

function parseSseBlock(buffer: string): { events: Record<string, unknown>[]; rest: string } {
  const events: Record<string, unknown>[] = [];
  const normalized = buffer.replace(/\r\n/g, '\n');
  const sep = '\n\n';
  let rest = normalized;
  let pos = rest.indexOf(sep);
  while (pos !== -1) {
    const block = rest.slice(0, pos);
    rest = rest.slice(pos + sep.length);
    pos = rest.indexOf(sep);
    for (const line of block.split('\n')) {
      const t = line.trimEnd();
      if (!t.startsWith('data:')) continue;
      const raw = t.slice(5).trimStart();
      if (raw === '[DONE]') {
        events.push({ type: 'done' });
        continue;
      }
      try {
        events.push(JSON.parse(raw) as Record<string, unknown>);
      } catch {
        // ignore malformed SSE payload
      }
    }
  }
  return { events, rest };
}

/**
 * POST /api/data/{profile_name}/ai-chat — Server-Sent Events with JSON payloads:
 * { "type": "delta", "text": "..." } and { "type": "done" }.
 */
export async function streamAiChat(
  profileName: string,
  profilePath: string,
  body: AiChatRequestBody,
  options: { onDelta: (text: string) => void; onToolStatus?: (name: string) => void; signal?: AbortSignal }
): Promise<void> {
  const url = `${API_BASE}/data/${encodeURIComponent(profileName)}/ai-chat?profile_path=${encodeURIComponent(profilePath)}`;
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify(body),
    signal: options.signal,
  });

  if (!res.ok) {
    const detail = await readApiErrorDetail(res);
    throw new Error(detail);
  }

  const reader = res.body?.getReader();
  if (!reader) {
    throw new Error('API fetch error: no response body');
  }

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const { events, rest } = parseSseBlock(buffer);
    buffer = rest;
    for (const ev of events) {
      const typ = ev.type;
      if (typ === 'delta' && typeof ev.text === 'string') {
        options.onDelta(ev.text);
      }
      if (typ === 'tool_status' && typeof ev.name === 'string' && options.onToolStatus) {
        options.onToolStatus(ev.name);
      }
      if (typ === 'done') {
        return;
      }
      if (typ === 'error' && typeof ev.message === 'string') {
        throw new Error(ev.message);
      }
    }
  }

  buffer += decoder.decode();
  if (buffer.trim()) {
    const { events } = parseSseBlock(buffer.endsWith('\n\n') ? buffer : `${buffer}\n\n`);
    for (const ev of events) {
      const typ = ev.type;
      if (typ === 'delta' && typeof ev.text === 'string') {
        options.onDelta(ev.text);
      }
      if (typ === 'tool_status' && typeof ev.name === 'string' && options.onToolStatus) {
        options.onToolStatus(ev.name);
      }
      if (typ === 'done') {
        return;
      }
      if (typ === 'error' && typeof ev.message === 'string') {
        throw new Error(ev.message);
      }
    }
  }
}
