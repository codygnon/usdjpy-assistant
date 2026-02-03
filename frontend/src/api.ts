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
  source?: 'mt5' | 'database';
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

export async function applyPreset(presetId: string, profilePath: string): Promise<Record<string, unknown>> {
  return fetchJson<Record<string, unknown>>(
    `${API_BASE}/presets/${presetId}/apply?profile_path=${encodeURIComponent(profilePath)}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ preset_id: presetId }),
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
  killSwitch: boolean
): Promise<void> {
  await fetchJson<unknown>(`${API_BASE}/runtime/${profileName}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode, kill_switch: killSwitch }),
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
  profilePath?: string
): Promise<TradesResponse> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (profilePath) params.set('profile_path', profilePath);
  const url = `${API_BASE}/data/${profileName}/trades?${params}`;
  const data = await fetchJson<Record<string, unknown>[] | { trades: Record<string, unknown>[]; display_currency: string }>(url);
  if (data != null && typeof data === 'object' && 'trades' in data && Array.isArray((data as { trades: unknown }).trades)) {
    return { trades: (data as { trades: Record<string, unknown>[] }).trades, display_currency: (data as { display_currency?: string }).display_currency };
  }
  return { trades: Array.isArray(data) ? data : [], display_currency: undefined };
}

export async function getExecutions(profileName: string, limit = 50): Promise<Record<string, unknown>[]> {
  return fetchJson<Record<string, unknown>[]>(`${API_BASE}/data/${profileName}/executions?limit=${limit}`);
}

export async function getRejectionBreakdown(profileName: string): Promise<Record<string, number>> {
  return fetchJson<Record<string, number>>(`${API_BASE}/data/${profileName}/rejection-breakdown`);
}

export async function getQuickStats(profileName: string, profilePath?: string): Promise<QuickStats> {
  const url = profilePath
    ? `${API_BASE}/data/${profileName}/stats?profile_path=${encodeURIComponent(profilePath)}`
    : `${API_BASE}/data/${profileName}/stats`;
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
}

export interface StatsByPreset {
  presets: Record<string, PresetStats>;
  source?: 'mt5' | 'database';
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
  source: 'mt5';
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
  return data?.source === 'mt5' ? (data as Mt5Report) : null;
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

export interface TaTimeframe {
  regime: string;
  rsi: TaRsi;
  macd: TaMacd;
  atr: TaAtr;
  price: TaPrice;
  summary: string;
  ohlc: OhlcBar[];
  ema_fast?: { time: number; value: number }[];
  ema_slow?: { time: number; value: number }[];
  ema_stack?: Record<string, { time: number; value: number }[]>;
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
}

export async function getOpenTrades(profileName: string): Promise<OpenTrade[]> {
  return fetchJson<OpenTrade[]>(`${API_BASE}/data/${profileName}/open-trades`);
}
