import { useEffect, useState, useRef } from 'react';
import { createChart, IChartApi, ISeriesApi, CandlestickData, Time, CandlestickSeries, LineSeries } from 'lightweight-charts';
import { ComposedChart, Area, Bar, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceLine, ScatterChart, Scatter, Cell, BarChart, LineChart, AreaChart } from 'recharts';
import * as api from './api';

type Page = 'run' | 'presets' | 'profile' | 'logs' | 'analysis' | 'guide';

interface Profile {
  path: string;
  name: string;
}

// Helper to get/set unlocked profiles from sessionStorage
const getUnlockedProfiles = (): Set<string> => {
  try {
    const stored = sessionStorage.getItem('unlockedProfiles');
    return stored ? new Set(JSON.parse(stored)) : new Set();
  } catch {
    return new Set();
  }
};

const setUnlockedProfilesStorage = (paths: Set<string>) => {
  sessionStorage.setItem('unlockedProfiles', JSON.stringify([...paths]));
};

export default function App() {
  const [page, setPage] = useState<Page>('run');
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [selectedProfile, setSelectedProfile] = useState<Profile | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showAddAccount, setShowAddAccount] = useState(false);
  const [addName, setAddName] = useState('');
  const [addError, setAddError] = useState<string | null>(null);
  const [addPending, setAddPending] = useState(false);
  const [addWithPassword, setAddWithPassword] = useState(false);
  const [addPassword, setAddPassword] = useState('');
  const [addPasswordConfirm, setAddPasswordConfirm] = useState('');
  const [confirmRemove, setConfirmRemove] = useState<Profile | null>(null);
  const [removePending, setRemovePending] = useState(false);

  // Auth state
  const [authStatus, setAuthStatus] = useState<Record<string, boolean>>({}); // path -> has_password
  const [unlockedProfiles, setUnlockedProfiles] = useState<Set<string>>(getUnlockedProfiles);
  const [loginPassword, setLoginPassword] = useState('');
  const [loginError, setLoginError] = useState<string | null>(null);
  const [loginPending, setLoginPending] = useState(false);

  const fetchProfiles = async () => {
    try {
      const list = await api.listProfiles();
      setProfiles(list);
      
      // Fetch auth status for all profiles
      const authMap: Record<string, boolean> = {};
      await Promise.all(
        list.map(async (p) => {
          try {
            const { has_password } = await api.checkAuth(p.path);
            authMap[p.path] = has_password;
          } catch {
            authMap[p.path] = false;
          }
        })
      );
      setAuthStatus(authMap);
      
      setSelectedProfile((prev) => {
        if (list.length === 0) return null;
        if (prev && list.some((p) => p.path === prev.path)) return prev;
        return list[0];
      });
    } catch (e) {
      setError((e as Error).message);
    }
  };

  useEffect(() => {
    fetchProfiles();
  }, []);

  // Check if selected profile needs login
  const selectedNeedsLogin = selectedProfile && 
    authStatus[selectedProfile.path] && 
    !unlockedProfiles.has(selectedProfile.path);

  const handleLogin = async () => {
    if (!selectedProfile) return;
    setLoginError(null);
    setLoginPending(true);
    try {
      const { success } = await api.authLogin(selectedProfile.path, loginPassword);
      if (success) {
        const newUnlocked = new Set(unlockedProfiles);
        newUnlocked.add(selectedProfile.path);
        setUnlockedProfiles(newUnlocked);
        setUnlockedProfilesStorage(newUnlocked);
        setLoginPassword('');
      } else {
        setLoginError('Incorrect password');
      }
    } catch (e) {
      setLoginError((e as Error).message);
    } finally {
      setLoginPending(false);
    }
  };

  const handleLogout = () => {
    if (!selectedProfile) return;
    const newUnlocked = new Set(unlockedProfiles);
    newUnlocked.delete(selectedProfile.path);
    setUnlockedProfiles(newUnlocked);
    setUnlockedProfilesStorage(newUnlocked);
    setLoginPassword('');
    setLoginError(null);
  };

  const handleCreateAccount = async () => {
    const name = addName.trim();
    if (!name) {
      setAddError('Enter a name');
      return;
    }
    if (addWithPassword) {
      if (!addPassword || addPassword.length < 4) {
        setAddError('Password must be at least 4 characters');
        return;
      }
      if (addPassword !== addPasswordConfirm) {
        setAddError('Passwords do not match');
        return;
      }
    }
    setAddError(null);
    setAddPending(true);
    try {
      const created = await api.createProfile(name);
      
      // Set password if requested
      if (addWithPassword && addPassword) {
        await api.setPassword(created.path, null, addPassword);
        // Add to unlocked so user can access immediately
        const newUnlocked = new Set(unlockedProfiles);
        newUnlocked.add(created.path);
        setUnlockedProfiles(newUnlocked);
        setUnlockedProfilesStorage(newUnlocked);
      }
      
      await fetchProfiles();
      setSelectedProfile(created);
      setAddName('');
      setAddPassword('');
      setAddPasswordConfirm('');
      setAddWithPassword(false);
      setShowAddAccount(false);
    } catch (e: unknown) {
      setAddError((e as Error).message);
    } finally {
      setAddPending(false);
    }
  };

  const handleRemoveAccount = async () => {
    if (!confirmRemove) return;
    setRemovePending(true);
    try {
      await api.deleteProfile(confirmRemove.path);
      fetchProfiles();
      setConfirmRemove(null);
    } catch (e: unknown) {
      setError((e as Error).message);
    } finally {
      setRemovePending(false);
    }
  };

  return (
    <div className="app">
      <aside className="sidebar">
        <h1>USDJPY Assistant</h1>
        <p className="subtitle">Trading Bot Dashboard</p>

        <div className="form-group">
          <label>Account</label>
          <select
            value={selectedProfile?.path || ''}
            onChange={(e) => {
              const p = profiles.find((x) => x.path === e.target.value);
              setSelectedProfile(p || null);
              setLoginPassword('');
              setLoginError(null);
            }}
          >
            {profiles.map((p) => (
              <option key={p.path} value={p.path}>
                {authStatus[p.path] ? '🔒 ' : ''}{p.name}
              </option>
            ))}
          </select>
          <div style={{ display: 'flex', gap: 8, marginTop: 8, flexWrap: 'wrap' }}>
            <button
              type="button"
              className="btn btn-secondary"
              style={{ fontSize: '0.8rem', padding: '6px 10px' }}
              onClick={() => { setShowAddAccount(!showAddAccount); setAddError(null); setAddName(''); setAddPassword(''); setAddPasswordConfirm(''); setAddWithPassword(false); }}
            >
              Add account
            </button>
            <button
              type="button"
              className="btn btn-danger"
              style={{ fontSize: '0.8rem', padding: '6px 10px' }}
              onClick={() => selectedProfile && setConfirmRemove(selectedProfile)}
              disabled={!selectedProfile || profiles.length === 0}
            >
              Remove account
            </button>
            {selectedProfile && authStatus[selectedProfile.path] && unlockedProfiles.has(selectedProfile.path) && (
              <button
                type="button"
                className="btn btn-secondary"
                style={{ fontSize: '0.8rem', padding: '6px 10px' }}
                onClick={handleLogout}
              >
                Lock
              </button>
            )}
          </div>
        </div>

        {showAddAccount && (
          <div className="card" style={{ marginTop: 12, padding: 12 }}>
            <div className="form-group" style={{ marginBottom: 8 }}>
              <label>New account name</label>
              <input
                type="text"
                value={addName}
                onChange={(e) => setAddName(e.target.value)}
                placeholder="e.g. uncle_demo"
              />
            </div>
            
            <div className="form-group" style={{ marginBottom: 8 }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={addWithPassword}
                  onChange={(e) => setAddWithPassword(e.target.checked)}
                  style={{ width: 'auto' }}
                />
                Set password (recommended)
              </label>
            </div>
            
            {addWithPassword && (
              <>
                <div className="form-group" style={{ marginBottom: 8 }}>
                  <label>Password</label>
                  <input
                    type="password"
                    value={addPassword}
                    onChange={(e) => setAddPassword(e.target.value)}
                    placeholder="Min 4 characters"
                  />
                </div>
                <div className="form-group" style={{ marginBottom: 8 }}>
                  <label>Confirm password</label>
                  <input
                    type="password"
                    value={addPasswordConfirm}
                    onChange={(e) => setAddPasswordConfirm(e.target.value)}
                    placeholder="Confirm password"
                  />
                </div>
                <p style={{ fontSize: '0.7rem', color: 'var(--warning)', marginBottom: 8, background: 'rgba(234, 179, 8, 0.1)', padding: 8, borderRadius: 4 }}>
                  Write your password down in a safe location. There is no forgot-password option.
                </p>
              </>
            )}
            
            {addError && <p style={{ color: 'var(--danger)', fontSize: '0.8rem', marginBottom: 8 }}>{addError}</p>}
            <div style={{ display: 'flex', gap: 8 }}>
              <button
                type="button"
                className="btn btn-primary"
                style={{ fontSize: '0.8rem' }}
                onClick={handleCreateAccount}
                disabled={addPending}
              >
                {addPending ? 'Creating…' : 'Create'}
              </button>
              <button
                type="button"
                className="btn btn-secondary"
                style={{ fontSize: '0.8rem' }}
                onClick={() => { setShowAddAccount(false); setAddError(null); setAddName(''); setAddPassword(''); setAddPasswordConfirm(''); setAddWithPassword(false); }}
                disabled={addPending}
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {confirmRemove && (
          <div
            style={{
              position: 'fixed',
              inset: 0,
              background: 'rgba(0,0,0,0.7)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 1000,
            }}
          >
            <div className="card" style={{ maxWidth: 400, margin: 16 }}>
              <h3 className="card-title">Remove account</h3>
              <p style={{ marginBottom: 16 }}>
                Remove <strong>{confirmRemove.name}</strong>? This will delete the profile file. Logs and trade
                history for this account are kept.
              </p>
              <div style={{ display: 'flex', gap: 8 }}>
                <button
                  type="button"
                  className="btn btn-danger"
                  onClick={handleRemoveAccount}
                  disabled={removePending}
                >
                  {removePending ? 'Removing…' : 'Remove'}
                </button>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => setConfirmRemove(null)}
                  disabled={removePending}
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}

        <nav style={{ marginTop: 24 }}>
          <button
            className={`nav-item ${page === 'run' ? 'active' : ''}`}
            onClick={() => setPage('run')}
          >
            Run / Status
          </button>
          <button
            className={`nav-item ${page === 'analysis' ? 'active' : ''}`}
            onClick={() => setPage('analysis')}
          >
            Analysis
          </button>
          <button
            className={`nav-item ${page === 'presets' ? 'active' : ''}`}
            onClick={() => setPage('presets')}
          >
            Presets
          </button>
          <button
            className={`nav-item ${page === 'profile' ? 'active' : ''}`}
            onClick={() => setPage('profile')}
          >
            Profile Editor
          </button>
          <button
            className={`nav-item ${page === 'logs' ? 'active' : ''}`}
            onClick={() => setPage('logs')}
          >
            Logs & Stats
          </button>
          <button
            className={`nav-item ${page === 'guide' ? 'active' : ''}`}
            onClick={() => setPage('guide')}
          >
            Help / Guide
          </button>
        </nav>
      </aside>

      <main className="main-content">
        {error && (
          <div className="card" style={{ borderColor: 'var(--danger)' }}>
            <p style={{ color: 'var(--danger)' }}>Error: {error}</p>
            <button className="btn btn-secondary" onClick={() => setError(null)}>
              Dismiss
            </button>
          </div>
        )}

        {selectedProfile ? (
          selectedNeedsLogin ? (
            <div className="card" style={{ maxWidth: 400, margin: '40px auto', textAlign: 'center' }}>
              <h2 style={{ marginBottom: 16 }}>Profile Locked</h2>
              <p style={{ color: 'var(--text-secondary)', marginBottom: 20 }}>
                <strong>{selectedProfile.name}</strong> is password-protected. Enter your password to continue.
              </p>
              <div className="form-group" style={{ marginBottom: 16, textAlign: 'left' }}>
                <label>Password</label>
                <input
                  type="password"
                  value={loginPassword}
                  onChange={(e) => setLoginPassword(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleLogin()}
                  placeholder="Enter password"
                  autoFocus
                />
              </div>
              {loginError && (
                <p style={{ color: 'var(--danger)', fontSize: '0.85rem', marginBottom: 12 }}>{loginError}</p>
              )}
              <button
                className="btn btn-primary"
                onClick={handleLogin}
                disabled={loginPending || !loginPassword}
                style={{ width: '100%', marginBottom: 16 }}
              >
                {loginPending ? 'Unlocking...' : 'Unlock'}
              </button>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                There is no forgot-password option. If you've lost your password, contact the administrator.
              </p>
            </div>
          ) : (
            <>
              {page === 'run' && <RunPage profile={selectedProfile} />}
              {page === 'analysis' && <AnalysisPage profile={selectedProfile} />}
              {page === 'presets' && <PresetsPage profile={selectedProfile} />}
              {page === 'profile' && <ProfilePage profile={selectedProfile} authStatus={authStatus} onAuthChange={fetchProfiles} />}
              {page === 'logs' && <LogsPage profile={selectedProfile} />}
              {page === 'guide' && <GuidePage />}
            </>
          )
        ) : (
          <div className="card">
            <p>No accounts yet. Use <strong>Add account</strong> in the sidebar to create one, or add a profile JSON in the profiles/ folder.</p>
          </div>
        )}
      </main>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Run / Status Page
// ---------------------------------------------------------------------------

function RunPage({ profile }: { profile: Profile }) {
  const [state, setState] = useState<api.RuntimeState | null>(null);
  const [log, setLog] = useState<string>('');
  const [loading, setLoading] = useState(false);

  const fetchState = () => {
    api.getRuntimeState(profile.name).then(setState).catch(console.error);
    api.getLoopLog(profile.name, 200).then((r) => setLog(r.content)).catch(console.error);
  };

  useEffect(() => {
    fetchState();
    const interval = setInterval(fetchState, 3000);
    return () => clearInterval(interval);
  }, [profile.name]);

  const handleModeChange = async (mode: string) => {
    if (!state) return;
    await api.updateRuntimeState(profile.name, mode, state.kill_switch);
    fetchState();
  };

  const handleKillSwitch = async (checked: boolean) => {
    if (!state) return;
    await api.updateRuntimeState(profile.name, state.mode, checked);
    fetchState();
  };

  const handleStart = async () => {
    setLoading(true);
    try {
      await api.startLoop(profile.name, profile.path);
      fetchState();
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    setLoading(true);
    try {
      await api.stopLoop(profile.name);
      fetchState();
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2 className="page-title">Run / Status</h2>

      <div className="grid-2">
        <div className="card">
          <h3 className="card-title">Loop Control</h3>
          
          <div className="flex-between mb-4">
            <span>Status:</span>
            {state?.loop_running ? (
              <span className="status status-running">
                <span className="status-dot" />
                Running
              </span>
            ) : (
              <span className="status status-stopped">
                <span className="status-dot" />
                Stopped
              </span>
            )}
          </div>

          <div className="flex gap-2">
            <button
              className="btn btn-success"
              onClick={handleStart}
              disabled={loading || state?.loop_running}
            >
              Start Loop
            </button>
            <button
              className="btn btn-danger"
              onClick={handleStop}
              disabled={loading || !state?.loop_running}
            >
              Stop Loop
            </button>
          </div>
        </div>

        <div className="card">
          <h3 className="card-title">Runtime Settings</h3>

          <div className="form-group">
            <label>Mode</label>
            <select
              value={state?.mode || 'DISARMED'}
              onChange={(e) => handleModeChange(e.target.value)}
            >
              <option value="DISARMED">DISARMED</option>
              <option value="ARMED_MANUAL_CONFIRM">ARMED_MANUAL_CONFIRM</option>
              <option value="ARMED_AUTO_DEMO">ARMED_AUTO_DEMO</option>
            </select>
          </div>

          <div className="form-group">
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={state?.kill_switch || false}
                onChange={(e) => handleKillSwitch(e.target.checked)}
                style={{ width: 'auto' }}
              />
              Kill Switch (disable all execution)
            </label>
          </div>
        </div>
      </div>

      <div className="card mt-4">
        <h3 className="card-title">Loop Log (last 200 lines)</h3>
        <div className="log-viewer">{log || '(no log yet)'}</div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Candlestick Chart Component
// ---------------------------------------------------------------------------

interface ChartTrade {
  trade_id: string;
  entry_price: number;
  stop_price: number | null;
  target_price: number | null;
  side: string;
  entry_time?: number;
  exit_time?: number;
  exit_price?: number;
}

interface CandlestickChartProps {
  ohlc: api.OhlcBar[];
  emaStack?: Record<string, { time: number; value: number }[]>;
  bollingerSeries?: { upper: { time: number; value: number }[]; middle: { time: number; value: number }[]; lower: { time: number; value: number }[] };
  height?: number;
}

const emaSeriesOptions = { lastValueVisible: false, priceLineVisible: false };
const emaColors: Record<string, string> = {
  ema5: '#ec4899', ema7: '#f97316', ema9: '#f59e0b', ema11: '#eab308',
  ema13: '#3b82f6', ema15: '#14b8a6', ema17: '#8b5cf6', ema21: '#6366f1',
  ema50: '#10b981', ema200: '#a855f7',
};

function CandlestickChart({ ohlc, emaStack, bollingerSeries, height = 300 }: CandlestickChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const emaSeriesRefs = useRef<Record<string, ISeriesApi<'Line'>>>({});
  const bbSeriesRefs = useRef<{ upper?: ISeriesApi<'Line'>; middle?: ISeriesApi<'Line'>; lower?: ISeriesApi<'Line'> }>({});
  const fitContentOnceRef = useRef(false);

  // Effect 1: Create chart once on mount; do not recreate on data/height change.
  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: height,
      layout: {
        background: { color: '#1a1a2e' },
        textColor: '#d1d5db',
      },
      grid: {
        vertLines: { color: '#2d2d44' },
        horzLines: { color: '#2d2d44' },
      },
      crosshair: {
        mode: 1,
      },
      rightPriceScale: {
        borderColor: '#2d2d44',
      },
      timeScale: {
        borderColor: '#2d2d44',
        timeVisible: true,
        secondsVisible: false,
      },
    });

    chartRef.current = chart;

    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#10b981',
      downColor: '#ef4444',
      borderUpColor: '#10b981',
      borderDownColor: '#ef4444',
      wickUpColor: '#10b981',
      wickDownColor: '#ef4444',
    });
    candlestickSeries.setData([]);
    seriesRef.current = candlestickSeries;

    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
          height: height,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
      emaSeriesRefs.current = {};
      bbSeriesRefs.current = {};
      fitContentOnceRef.current = false;
    };
  }, []);

  // Effect 2: Update data and layout when props change; do not destroy chart or call fitContent after first time.
  useEffect(() => {
    const chart = chartRef.current;
    const candlestickSeries = seriesRef.current;
    if (!chartContainerRef.current || !chart || !candlestickSeries) return;

    chart.applyOptions({ height });

    const chartData: CandlestickData<Time>[] = ohlc.map((bar) => ({
      time: bar.time as Time,
      open: bar.open,
      high: bar.high,
      low: bar.low,
      close: bar.close,
    }));
    candlestickSeries.setData(chartData);

    if (emaStack) {
      for (const [key, arr] of Object.entries(emaStack)) {
        if (!arr) continue;
        const color = emaColors[key] || '#94a3b8';
        const lineData = arr.map(d => ({ time: d.time as Time, value: d.value }));
        if (!emaSeriesRefs.current[key]) {
          const emaSeries = chart.addSeries(LineSeries, { color, lineWidth: 1, title: key.toUpperCase(), ...emaSeriesOptions });
          emaSeries.setData(lineData);
          emaSeriesRefs.current[key] = emaSeries;
        } else {
          emaSeriesRefs.current[key].setData(lineData);
        }
      }
    }

    if (bollingerSeries) {
      const bbColor = '#6b7280';
      if (!bbSeriesRefs.current.upper) {
        bbSeriesRefs.current.upper = chart.addSeries(LineSeries, { color: bbColor, lineWidth: 1, lineStyle: 2, title: 'BB Upper', ...emaSeriesOptions });
        bbSeriesRefs.current.middle = chart.addSeries(LineSeries, { color: bbColor, lineWidth: 1, title: 'BB Mid', ...emaSeriesOptions });
        bbSeriesRefs.current.lower = chart.addSeries(LineSeries, { color: bbColor, lineWidth: 1, lineStyle: 2, title: 'BB Lower', ...emaSeriesOptions });
      }
      const upper = bollingerSeries.upper?.length ? bollingerSeries.upper.map(d => ({ time: d.time as Time, value: d.value })) : [];
      const middle = bollingerSeries.middle?.length ? bollingerSeries.middle.map(d => ({ time: d.time as Time, value: d.value })) : [];
      const lower = bollingerSeries.lower?.length ? bollingerSeries.lower.map(d => ({ time: d.time as Time, value: d.value })) : [];
      bbSeriesRefs.current.upper?.setData(upper);
      bbSeriesRefs.current.middle?.setData(middle);
      bbSeriesRefs.current.lower?.setData(lower);
    }

    if (ohlc.length > 0 && !fitContentOnceRef.current) {
      chart.timeScale().fitContent();
      fitContentOnceRef.current = true;
    }
  }, [ohlc, emaStack, bollingerSeries, height]);

  return (
    <div style={{ position: 'relative', width: '100%', height: height, borderRadius: 6, overflow: 'hidden' }}>
      <div
        ref={chartContainerRef}
        style={{
          width: '100%',
          height: height,
          borderRadius: 6,
          overflow: 'hidden'
        }}
      />
      {ohlc.length === 0 && (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'var(--bg-tertiary)',
            borderRadius: 6,
            color: 'var(--text-secondary)'
          }}
        >
          No chart data available
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Analysis Page - Technical Analysis
// ---------------------------------------------------------------------------

function AnalysisPage({ profile }: { profile: Profile }) {
  const [ta, setTa] = useState<api.TechnicalAnalysis | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedTf, setExpandedTf] = useState<string | null>(null);
  const [enlargedTf, setEnlargedTf] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [chartTrades, setChartTrades] = useState<ChartTrade[]>([]);
  const [fullscreenTf, setFullscreenTf] = useState<string | null>(null);
  const [confirmCloseTrade, setConfirmCloseTrade] = useState<ChartTrade | null>(null);
  const [closingTrade, setClosingTrade] = useState(false);
  const [emaToggles, setEmaToggles] = useState<Record<string, boolean>>({});
  const [bbToggle, setBbToggle] = useState(false);
  const [tradesMinimized, setTradesMinimized] = useState(false);

  const EMA_PERIODS = [5, 7, 9, 11, 13, 15, 17, 21, 50, 200] as const;
  const EMA_COLORS: Record<string, string> = {
    ema5: '#ec4899', ema7: '#f97316', ema9: '#f59e0b', ema11: '#eab308',
    ema13: '#3b82f6', ema15: '#14b8a6', ema17: '#8b5cf6', ema21: '#6366f1',
    ema50: '#10b981', ema200: '#a855f7',
  };

  const getFilteredEmas = (allEmas?: Record<string, { time: number; value: number }[]>) => {
    if (!allEmas) return undefined;
    const filtered: Record<string, { time: number; value: number }[]> = {};
    for (const [key, arr] of Object.entries(allEmas)) {
      if (emaToggles[key] && arr && arr.length > 0) {
        filtered[key] = arr;
      }
    }
    return Object.keys(filtered).length > 0 ? filtered : undefined;
  };

  const fetchTa = async () => {
    try {
      const [taData, allTradesData] = await Promise.all([
        api.getTechnicalAnalysis(profile.name, profile.path),
        api.getTrades(profile.name, 250).then((r) => r.trades).catch(() => []),
      ]);
      setTa(taData);
      // Build chart trades: open + recently closed (with entry/exit times for markers)
      const chartTradesList: ChartTrade[] = allTradesData
        .filter((t: Record<string, unknown>) => t.trade_id && t.entry_price)
        .map((t: Record<string, unknown>) => {
          const tsUtc = t.timestamp_utc as string | undefined;
          const exitTsUtc = t.exit_timestamp_utc as string | undefined;
          const entryTime = tsUtc ? Math.floor(new Date(tsUtc).getTime() / 1000) : undefined;
          const exitTime = exitTsUtc ? Math.floor(new Date(exitTsUtc).getTime() / 1000) : undefined;
          return {
            trade_id: String(t.trade_id),
            entry_price: Number(t.entry_price),
            stop_price: t.stop_price != null ? Number(t.stop_price) : null,
            target_price: t.target_price != null ? Number(t.target_price) : null,
            side: String(t.side || 'buy'),
            entry_time: entryTime,
            exit_time: exitTime,
            exit_price: t.exit_price != null ? Number(t.exit_price) : undefined,
          };
        });
      setChartTrades(chartTradesList);
      setError(null);
      setLastUpdate(new Date());
    } catch (e: unknown) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    setLoading(true);
    fetchTa();
    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchTa, 30000);
    return () => clearInterval(interval);
  }, [profile.name, profile.path]);

  // ESC key to close fullscreen
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && fullscreenTf) {
        setFullscreenTf(null);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [fullscreenTf]);

  // Handle close trade
  const handleCloseTrade = async (trade: ChartTrade) => {
    setClosingTrade(true);
    try {
      await api.closeTrade(profile.name, trade.trade_id, profile.path);
      setConfirmCloseTrade(null);
      fetchTa(); // Refresh data
    } catch (e: unknown) {
      alert(`Failed to close trade: ${(e as Error).message}`);
    } finally {
      setClosingTrade(false);
    }
  };

  const toggleExpand = (tf: string) => {
    setExpandedTf(expandedTf === tf ? null : tf);
  };

  // Helper to get regime badge style
  const getRegimeBadge = (regime: string) => {
    const styles: Record<string, { bg: string; color: string }> = {
      bull: { bg: 'rgba(16, 185, 129, 0.2)', color: 'var(--success)' },
      bear: { bg: 'rgba(239, 68, 68, 0.2)', color: 'var(--danger)' },
      sideways: { bg: 'rgba(156, 163, 175, 0.2)', color: 'var(--text-secondary)' },
      unknown: { bg: 'rgba(156, 163, 175, 0.2)', color: 'var(--text-secondary)' },
    };
    const style = styles[regime] || styles.unknown;
    return (
      <span style={{
        padding: '4px 10px',
        borderRadius: 4,
        background: style.bg,
        color: style.color,
        fontWeight: 600,
        fontSize: '0.85rem',
        textTransform: 'uppercase',
      }}>
        {regime}
      </span>
    );
  };

  // Helper to get RSI zone style
  const getRsiDisplay = (rsi: api.TaRsi) => {
    const zoneColors: Record<string, string> = {
      oversold: 'var(--success)',
      overbought: 'var(--danger)',
      neutral: 'var(--text-secondary)',
    };
    return (
      <span>
        <span style={{ color: zoneColors[rsi.zone] || 'inherit' }}>
          {rsi.zone.charAt(0).toUpperCase() + rsi.zone.slice(1)}
        </span>
        {rsi.value !== null && (
          <span style={{ color: 'var(--text-secondary)', marginLeft: 4 }}>
            ({rsi.value.toFixed(1)})
          </span>
        )}
      </span>
    );
  };

  // Helper to get MACD display
  const getMacdDisplay = (macd: api.TaMacd) => {
    const dirColors: Record<string, string> = {
      positive: 'var(--success)',
      negative: 'var(--danger)',
      neutral: 'var(--text-secondary)',
    };
    return (
      <span style={{ color: dirColors[macd.direction] || 'inherit' }}>
        {macd.direction.charAt(0).toUpperCase() + macd.direction.slice(1)}
      </span>
    );
  };

  // Helper to get ATR state display
  const getAtrDisplay = (atr: api.TaAtr) => {
    const stateColors: Record<string, string> = {
      elevated: 'var(--warning)',
      low: 'var(--accent)',
      normal: 'var(--text-secondary)',
    };
    return (
      <span>
        <span style={{ color: stateColors[atr.state] || 'inherit' }}>
          {atr.state.charAt(0).toUpperCase() + atr.state.slice(1)}
        </span>
        {atr.value_pips !== null && (
          <span style={{ color: 'var(--text-secondary)', marginLeft: 4 }}>
            ({atr.value_pips} pips)
          </span>
        )}
      </span>
    );
  };

  const timeframeOrder = ['H4', 'H1', 'M30', 'M15', 'M5', 'M3', 'M1'];

  // Human-readable labels for the four key timeframes (1H, 30m, 5m, 3m) and others
  const timeframeLabel: Record<string, string> = {
    H4: '4H',
    H1: '1H',
    M30: '30m',
    M15: '15m',
    M5: '5m',
    M3: '3m',
    M1: '1m',
  };

  // Build EMA legend text from API data (e.g. "EMA 8/13/21" or "EMA 13, SMA 30")
  const getActiveEmaLegend = (): string => {
    const active = EMA_PERIODS.filter(p => emaToggles[`ema${p}`]);
    const parts: string[] = [];
    if (active.length > 0) parts.push(`EMA ${active.join('/')}`);
    if (bbToggle) parts.push('BB');
    return parts.length > 0 ? parts.join(' · ') : '';
  };

  return (
    <div>
      <div className="flex-between mb-4">
        <h2 className="page-title" style={{ marginBottom: 0, borderBottom: 'none', paddingBottom: 0 }}>
          Technical Analysis
        </h2>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          {lastUpdate && (
            <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
              Updated: {lastUpdate.toLocaleTimeString()}
            </span>
          )}
          <button
            className="btn btn-secondary"
            onClick={fetchTa}
            disabled={loading}
          >
            {loading ? 'Loading...' : 'Refresh'}
          </button>
        </div>
      </div>

      <p style={{ color: 'var(--text-secondary)', marginBottom: 16, fontSize: '0.9rem' }}>
        Real-time technical analysis across multiple timeframes. Auto-refreshes every 30 seconds.
        Click on a timeframe to expand detailed view.
      </p>

      {error && (
        <div className="card mb-4" style={{ borderColor: 'var(--danger)' }}>
          <p style={{ color: 'var(--danger)' }}>Error: {error}</p>
        </div>
      )}

      {/* Current Price Info */}
      {ta?.current_tick && (
        <div className="card mb-4">
          <div className="grid-3">
            <div className="stat-box">
              <div className="stat-value">{ta.current_tick.bid.toFixed(3)}</div>
              <div className="stat-label">Bid</div>
            </div>
            <div className="stat-box">
              <div className="stat-value">{ta.current_tick.ask.toFixed(3)}</div>
              <div className="stat-label">Ask</div>
            </div>
            <div className="stat-box">
              <div className="stat-value" style={{ color: ta.current_tick.spread_pips > 1 ? 'var(--warning)' : 'var(--success)' }}>
                {ta.current_tick.spread_pips} pips
              </div>
              <div className="stat-label">Spread</div>
            </div>
          </div>
        </div>
      )}

      {/* Timeframe Cards */}
      {ta && timeframeOrder.map((tf) => {
        const tfData = ta.timeframes[tf];
        if (!tfData) return null;
        
        const isExpanded = expandedTf === tf;
        
        return (
          <div key={tf} className="card mb-4">
            {/* Summary Row - only the arrow toggles expand/collapse */}
            <div className="flex-between" style={{ marginBottom: isExpanded ? 16 : 0 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                <div>
                  <h3 style={{ margin: 0, fontSize: '1.1rem', minWidth: 40 }}>{tf}</h3>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                    {timeframeLabel[tf] || tf}
                  </span>
                </div>
                {getRegimeBadge(tfData.regime)}
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
                <div>
                  <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginRight: 8 }}>RSI:</span>
                  {getRsiDisplay(tfData.rsi)}
                </div>
                <div>
                  <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginRight: 8 }}>MACD:</span>
                  {getMacdDisplay(tfData.macd)}
                </div>
                <div>
                  <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginRight: 8 }}>ATR:</span>
                  {getAtrDisplay(tfData.atr)}
                </div>
                <button
                  type="button"
                  onClick={() => toggleExpand(tf)}
                  style={{
                    background: 'none',
                    border: 'none',
                    padding: 4,
                    cursor: 'pointer',
                    color: 'var(--accent)',
                    fontSize: '0.9rem',
                  }}
                  aria-label={isExpanded ? 'Collapse chart' : 'Expand chart'}
                >
                  {isExpanded ? '▼' : '▶'}
                </button>
              </div>
            </div>

            {/* Summary Text */}
            {!isExpanded && (
              <p style={{ margin: '12px 0 0 0', color: 'var(--text-secondary)', fontSize: '0.85rem', fontStyle: 'italic' }}>
                {tfData.summary}
              </p>
            )}

            {/* Expanded Detail - click on chart does not collapse */}
            {isExpanded && (
              <div style={{ borderTop: '1px solid var(--border)', paddingTop: 16 }} onClick={(e) => e.stopPropagation()}>
                {/* Candlestick Chart */}
                <div style={{ marginBottom: 16 }}>
                  <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 8, marginBottom: 8 }}>
                    <h4 style={{ margin: 0, color: 'var(--accent)', fontSize: '0.9rem' }}>
                      {tf} Chart ({timeframeLabel[tf] || tf})
                    </h4>
                    {getActiveEmaLegend() && (
                      <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>
                        {getActiveEmaLegend()}
                      </span>
                    )}
                    <button
                      type="button"
                      className="btn btn-secondary"
                      style={{ fontSize: '0.75rem', marginLeft: 'auto' }}
                      onClick={() => setEnlargedTf(enlargedTf === tf ? null : tf)}
                    >
                      {enlargedTf === tf ? 'Shrink' : 'Enlarge'}
                    </button>
                    <button
                      type="button"
                      className="btn btn-secondary"
                      style={{ fontSize: '0.75rem' }}
                      onClick={() => setFullscreenTf(tf)}
                    >
                      Fullscreen
                    </button>
                  </div>
                  {/* EMA & BB Toggle Pills */}
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 8 }}>
                    {EMA_PERIODS.map(p => {
                      const key = `ema${p}`;
                      const active = !!emaToggles[key];
                      return (
                        <button
                          key={key}
                          type="button"
                          onClick={() => setEmaToggles(prev => ({ ...prev, [key]: !prev[key] }))}
                          style={{
                            background: active ? EMA_COLORS[key] : '#374151',
                            color: active ? '#fff' : '#9ca3af',
                            border: 'none',
                            borderRadius: 12,
                            padding: '2px 8px',
                            fontSize: '0.65rem',
                            fontWeight: 600,
                            cursor: 'pointer',
                          }}
                        >
                          {p}
                        </button>
                      );
                    })}
                    <button
                      type="button"
                      onClick={() => setBbToggle(prev => !prev)}
                      style={{
                        background: bbToggle ? '#6b7280' : '#374151',
                        color: bbToggle ? '#fff' : '#9ca3af',
                        border: 'none',
                        borderRadius: 12,
                        padding: '2px 8px',
                        fontSize: '0.65rem',
                        fontWeight: 600,
                        cursor: 'pointer',
                      }}
                    >
                      BB
                    </button>
                  </div>
                  <div style={{ position: 'relative' }}>
                    <CandlestickChart
                      ohlc={tfData.ohlc || []}
                      emaStack={getFilteredEmas(tfData.all_emas)}
                      bollingerSeries={bbToggle ? tfData.bollinger_series : undefined}
                      height={enlargedTf === tf ? 520 : 280}
                    />
                    {/* Active Trades Overlay */}
                    {(() => {
                      const activeTrades = chartTrades.filter(t => !t.exit_time && !t.exit_price);
                      if (activeTrades.length === 0) return null;
                      return (
                        <div style={{
                          position: 'absolute',
                          top: 8,
                          right: 8,
                          background: 'rgba(26, 26, 46, 0.9)',
                          border: '1px solid var(--border)',
                          borderRadius: 6,
                          padding: 8,
                          fontSize: '0.75rem',
                          maxWidth: 200,
                          zIndex: 10,
                        }}>
                          <div
                            style={{ fontWeight: 600, color: 'var(--text-secondary)', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}
                            onClick={() => setTradesMinimized(prev => !prev)}
                          >
                            <span>Active Trades ({activeTrades.length})</span>
                            <span style={{ fontSize: '0.6rem', marginLeft: 6 }}>{tradesMinimized ? '\u25B6' : '\u25BC'}</span>
                          </div>
                          {!tradesMinimized && activeTrades.map(t => (
                            <div key={t.trade_id} style={{
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'space-between',
                              gap: 8,
                              padding: '4px 0',
                              borderTop: '1px solid var(--border)'
                            }}>
                              <span style={{
                                color: t.side.toLowerCase() === 'buy' ? '#3b82f6' : '#ef4444',
                                fontWeight: 500
                              }}>
                                {t.side.toUpperCase()} @ {t.entry_price.toFixed(3)}
                              </span>
                              <button
                                onClick={() => setConfirmCloseTrade(t)}
                                style={{
                                  background: 'var(--danger)',
                                  border: 'none',
                                  borderRadius: 3,
                                  color: 'white',
                                  padding: '2px 6px',
                                  fontSize: '0.65rem',
                                  cursor: 'pointer',
                                }}
                              >
                                Close
                              </button>
                            </div>
                          ))}
                        </div>
                      );
                    })()}
                  </div>
                </div>

                {/* Plain English Summary */}
                <div style={{ 
                  padding: 12, 
                  background: 'var(--bg-tertiary)', 
                  borderRadius: 6, 
                  marginBottom: 16,
                  borderLeft: '4px solid var(--accent)'
                }}>
                  <p style={{ margin: 0, fontSize: '0.95rem' }}>{tfData.summary}</p>
                </div>

                <div className="grid-2">
                  {/* RSI Details */}
                  <div style={{ padding: 12, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                    <h4 style={{ margin: '0 0 12px 0', color: 'var(--accent)', fontSize: '0.9rem' }}>RSI (14)</h4>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                      <div>
                        <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Value:</span>
                        <span style={{ marginLeft: 8, fontWeight: 600 }}>
                          {tfData.rsi.value?.toFixed(2) ?? '-'}
                        </span>
                      </div>
                      <div>
                        <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Zone:</span>
                        <span style={{ marginLeft: 8 }}>{getRsiDisplay(tfData.rsi)}</span>
                      </div>
                    </div>
                    <p style={{ margin: '8px 0 0 0', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                      {tfData.rsi.zone === 'oversold' && 'Price may be undervalued. Potential buy opportunity if trend supports.'}
                      {tfData.rsi.zone === 'overbought' && 'Price may be overextended. Consider caution with longs.'}
                      {tfData.rsi.zone === 'neutral' && 'Momentum is balanced. Follow the trend direction.'}
                    </p>
                  </div>

                  {/* MACD Details */}
                  <div style={{ padding: 12, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                    <h4 style={{ margin: '0 0 12px 0', color: 'var(--accent)', fontSize: '0.9rem' }}>MACD (12,26,9)</h4>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                      <div>
                        <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Line:</span>
                        <span style={{ marginLeft: 8, fontWeight: 600 }}>
                          {tfData.macd.line?.toFixed(5) ?? '-'}
                        </span>
                      </div>
                      <div>
                        <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Signal:</span>
                        <span style={{ marginLeft: 8 }}>
                          {tfData.macd.signal?.toFixed(5) ?? '-'}
                        </span>
                      </div>
                      <div>
                        <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Histogram:</span>
                        <span style={{ marginLeft: 8, color: tfData.macd.histogram && tfData.macd.histogram > 0 ? 'var(--success)' : 'var(--danger)' }}>
                          {tfData.macd.histogram?.toFixed(5) ?? '-'}
                        </span>
                      </div>
                      <div>
                        <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Direction:</span>
                        <span style={{ marginLeft: 8 }}>{getMacdDisplay(tfData.macd)}</span>
                      </div>
                    </div>
                  </div>

                  {/* ATR Details */}
                  <div style={{ padding: 12, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                    <h4 style={{ margin: '0 0 12px 0', color: 'var(--accent)', fontSize: '0.9rem' }}>ATR (14)</h4>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                      <div>
                        <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Value:</span>
                        <span style={{ marginLeft: 8, fontWeight: 600 }}>
                          {tfData.atr.value?.toFixed(5) ?? '-'}
                        </span>
                      </div>
                      <div>
                        <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Pips:</span>
                        <span style={{ marginLeft: 8 }}>
                          {tfData.atr.value_pips ?? '-'}
                        </span>
                      </div>
                    </div>
                    <p style={{ margin: '8px 0 0 0', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                      {tfData.atr.state === 'elevated' && 'Volatility is high. Consider wider stops and smaller position sizes.'}
                      {tfData.atr.state === 'low' && 'Volatility is low. Smaller moves expected, watch for breakouts.'}
                      {tfData.atr.state === 'normal' && 'Volatility is within normal range.'}
                    </p>
                  </div>

                  {/* Bollinger Bands */}
                  {(() => {
                    const bb = (tfData as unknown as Record<string, unknown>).bollinger as Record<string, unknown> | undefined;
                    if (!bb || typeof bb !== 'object') return null;
                    return (
                      <div style={{ padding: 12, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                        <h4 style={{ margin: '0 0 12px 0', color: 'var(--accent)', fontSize: '0.9rem' }}>Bollinger Bands (20, 2)</h4>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                          <div>
                            <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Upper:</span>
                            <span style={{ marginLeft: 8, fontWeight: 600 }}>{bb.upper != null ? Number(bb.upper).toFixed(5) : '-'}</span>
                          </div>
                          <div>
                            <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Middle:</span>
                            <span style={{ marginLeft: 8 }}>{bb.middle != null ? Number(bb.middle).toFixed(5) : '-'}</span>
                          </div>
                          <div>
                            <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Lower:</span>
                            <span style={{ marginLeft: 8, fontWeight: 600 }}>{bb.lower != null ? Number(bb.lower).toFixed(5) : '-'}</span>
                          </div>
                        </div>
                        <p style={{ margin: '8px 0 0 0', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                          Price relative to bands indicates volatility and potential mean reversion. For display only; not used in automation.
                        </p>
                      </div>
                    );
                  })()}

                  {/* VWAP */}
                  {typeof (tfData as unknown as Record<string, unknown>).vwap === 'number' && (
                    <div style={{ padding: 12, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                      <h4 style={{ margin: '0 0 12px 0', color: 'var(--accent)', fontSize: '0.9rem' }}>VWAP</h4>
                      <div>
                        <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Value:</span>
                        <span style={{ marginLeft: 8, fontWeight: 600 }}>{Number((tfData as unknown as Record<string, unknown>).vwap).toFixed(5)}</span>
                      </div>
                      <p style={{ margin: '8px 0 0 0', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                        Volume-weighted average price (uses tick volume when available). For display only; not used in automation.
                      </p>
                    </div>
                  )}

                  {/* Price Levels */}
                  <div style={{ padding: 12, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                    <h4 style={{ margin: '0 0 12px 0', color: 'var(--accent)', fontSize: '0.9rem' }}>Price Levels (500-bar)</h4>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                      <div>
                        <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Current:</span>
                        <span style={{ marginLeft: 8, fontWeight: 600 }}>
                          {tfData.price.current?.toFixed(3) ?? '-'}
                        </span>
                      </div>
                      <div>
                        <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Recent High:</span>
                        <span style={{ marginLeft: 8, color: 'var(--success)' }}>
                          {tfData.price.recent_high?.toFixed(3) ?? '-'}
                        </span>
                      </div>
                      <div>
                        <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>Recent Low:</span>
                        <span style={{ marginLeft: 8, color: 'var(--danger)' }}>
                          {tfData.price.recent_low?.toFixed(3) ?? '-'}
                        </span>
                      </div>
                    </div>
                    <p style={{ margin: '8px 0 0 0', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                      {tfData.price.current && tfData.price.recent_high && tfData.price.recent_low && (
                        <>
                          Price is {((tfData.price.current - tfData.price.recent_low) / (tfData.price.recent_high - tfData.price.recent_low) * 100).toFixed(0)}% 
                          of the way from recent low to high.
                        </>
                      )}
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        );
      })}

      {loading && !ta && (
        <div className="card">
          <p style={{ color: 'var(--text-secondary)', textAlign: 'center' }}>Loading technical analysis...</p>
        </div>
      )}

      {/* Fullscreen Modal */}
      {fullscreenTf && ta && ta.timeframes[fullscreenTf] && (() => {
        const tfData = ta.timeframes[fullscreenTf];
        const chartHeight = typeof window !== 'undefined' ? window.innerHeight - 200 : 500;
        return (
          <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'var(--bg-primary)',
            zIndex: 1000,
            display: 'flex',
            flexDirection: 'column',
          }}>
            {/* Header */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '12px 20px',
              borderBottom: '1px solid var(--border)',
              background: 'var(--bg-secondary)',
            }}>
              <h2 style={{ margin: 0, fontSize: '1.2rem' }}>
                {fullscreenTf} Chart ({timeframeLabel[fullscreenTf] || fullscreenTf})
              </h2>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setFullscreenTf(null)}
              >
                Close (ESC)
              </button>
            </div>

            {/* EMA & BB Toggle Pills */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, padding: '0 16px' }}>
              {EMA_PERIODS.map(p => {
                const key = `ema${p}`;
                const active = !!emaToggles[key];
                return (
                  <button
                    key={key}
                    type="button"
                    onClick={() => setEmaToggles(prev => ({ ...prev, [key]: !prev[key] }))}
                    style={{
                      background: active ? EMA_COLORS[key] : '#374151',
                      color: active ? '#fff' : '#9ca3af',
                      border: 'none',
                      borderRadius: 12,
                      padding: '2px 8px',
                      fontSize: '0.65rem',
                      fontWeight: 600,
                      cursor: 'pointer',
                    }}
                  >
                    {p}
                  </button>
                );
              })}
              <button
                type="button"
                onClick={() => setBbToggle(prev => !prev)}
                style={{
                  background: bbToggle ? '#6b7280' : '#374151',
                  color: bbToggle ? '#fff' : '#9ca3af',
                  border: 'none',
                  borderRadius: 12,
                  padding: '2px 8px',
                  fontSize: '0.65rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                }}
              >
                BB
              </button>
            </div>

            {/* Chart */}
            <div style={{ flex: 1, padding: 16, overflow: 'hidden' }}>
              <CandlestickChart
                ohlc={tfData.ohlc || []}
                emaStack={getFilteredEmas(tfData.all_emas)}
                bollingerSeries={bbToggle ? tfData.bollinger_series : undefined}
                height={chartHeight}
              />
            </div>

            {/* Bottom Summary Panel */}
            <div style={{
              padding: '12px 20px',
              borderTop: '1px solid var(--border)',
              background: 'var(--bg-secondary)',
              display: 'flex',
              flexWrap: 'wrap',
              gap: 20,
              alignItems: 'center',
            }}>
              <div>
                <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginRight: 8 }}>Regime:</span>
                {getRegimeBadge(tfData.regime)}
              </div>
              <div>
                <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginRight: 8 }}>RSI:</span>
                {getRsiDisplay(tfData.rsi)}
              </div>
              <div>
                <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginRight: 8 }}>MACD:</span>
                {getMacdDisplay(tfData.macd)}
              </div>
              <div>
                <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', marginRight: 8 }}>ATR:</span>
                {getAtrDisplay(tfData.atr)}
              </div>
              <div style={{ flex: 1, minWidth: 200 }}>
                <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', fontStyle: 'italic' }}>
                  {tfData.summary}
                </span>
              </div>
            </div>
          </div>
        );
      })()}

      {/* Close Trade Confirmation Modal */}
      {confirmCloseTrade && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.7)',
          zIndex: 1100,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}>
          <div style={{
            background: 'var(--bg-secondary)',
            borderRadius: 8,
            padding: 24,
            maxWidth: 400,
            width: '90%',
            border: '1px solid var(--border)',
          }}>
            <h3 style={{ margin: '0 0 16px 0' }}>Close Trade?</h3>
            <p style={{ color: 'var(--text-secondary)', marginBottom: 8 }}>
              Are you sure you want to close this trade?
            </p>
            <div style={{
              padding: 12,
              background: 'var(--bg-tertiary)',
              borderRadius: 6,
              marginBottom: 20,
            }}>
              <div style={{
                color: confirmCloseTrade.side.toLowerCase() === 'buy' ? '#3b82f6' : '#ef4444',
                fontWeight: 600,
                fontSize: '1rem',
              }}>
                {confirmCloseTrade.side.toUpperCase()} @ {confirmCloseTrade.entry_price.toFixed(3)}
              </div>
              {confirmCloseTrade.stop_price && (
                <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: 4 }}>
                  SL: {confirmCloseTrade.stop_price.toFixed(3)}
                </div>
              )}
              {confirmCloseTrade.target_price && (
                <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  TP: {confirmCloseTrade.target_price.toFixed(3)}
                </div>
              )}
            </div>
            <div style={{ display: 'flex', gap: 12, justifyContent: 'flex-end' }}>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setConfirmCloseTrade(null)}
                disabled={closingTrade}
              >
                Cancel
              </button>
              <button
                type="button"
                className="btn btn-danger"
                onClick={() => handleCloseTrade(confirmCloseTrade)}
                disabled={closingTrade}
                style={{ background: 'var(--danger)' }}
              >
                {closingTrade ? 'Closing...' : 'Close Trade'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Custom Wizard Component
// ---------------------------------------------------------------------------

type ComboStrategy = 'ema_cross' | 'price_levels' | 'bollinger_bands' | 'vwap';

interface WizardConfig {
  tradingStyle: 'scalper' | 'day_trader' | 'swing_trader' | null;
  riskTolerance: 'conservative' | 'moderate' | 'aggressive' | null;
  entryStyle: 'ema_cross' | 'rsi_dips' | 'price_levels' | 'bollinger_bands' | 'vwap' | 'multiple' | null;
  emaStack: 'default' | 'alt1' | 'alt2' | null;
  multipleStrategyA: ComboStrategy | null;
  multipleStrategyB: ComboStrategy | null;
}

// EMA stack presets by trading style
const EMA_STACK_PRESETS = {
  scalper: {
    default: { label: 'Default (5, 9, 13)', values: [5, 9, 13] },
    alt1: { label: 'Ultra-fast (3, 5, 8)', values: [3, 5, 8] },
    alt2: { label: 'Standard (8, 13, 21)', values: [8, 13, 21] },
  },
  day_trader: {
    default: { label: 'Default (13, 21, 34)', values: [13, 21, 34] },
    alt1: { label: 'Faster (8, 13, 21)', values: [8, 13, 21] },
    alt2: { label: 'Slower (21, 34, 55)', values: [21, 34, 55] },
  },
  swing_trader: {
    default: { label: 'Default (21, 50, 100)', values: [21, 50, 100] },
    alt1: { label: 'Classic (20, 50, 200)', values: [20, 50, 200] },
    alt2: { label: 'Medium (34, 89, 144)', values: [34, 89, 144] },
  },
};

// Setting Control Component with +/- buttons and direct typing
function SettingControl({ 
  label, 
  value, 
  onAdjust,
  onChange,
  step = 1,
  min,
  max
}: { 
  label: string; 
  value: number | undefined; 
  onAdjust: (delta: number) => void;
  onChange?: (value: number) => void;
  step?: number;
  min?: number;
  max?: number;
}) {
  if (value === undefined || value === null) return null;
  
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = parseFloat(e.target.value);
    if (!isNaN(newValue) && onChange) {
      let clamped = newValue;
      if (min !== undefined) clamped = Math.max(min, clamped);
      if (max !== undefined) clamped = Math.min(max, clamped);
      onChange(clamped);
    }
  };
  
  return (
    <div style={{ 
      display: 'flex', 
      alignItems: 'center', 
      justifyContent: 'space-between',
      padding: '8px 12px',
      background: 'var(--bg-secondary)',
      borderRadius: 6,
      border: '1px solid var(--border)'
    }}>
      <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>{label}</span>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <button 
          onClick={() => onAdjust(-1)} 
          style={{ 
            width: 28, 
            height: 28, 
            borderRadius: 4, 
            border: '1px solid var(--border)', 
            background: 'var(--bg-tertiary)',
            color: 'var(--text-primary)',
            cursor: 'pointer',
            fontSize: '1rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
        >
          -
        </button>
        <input
          type="number"
          value={step < 1 ? value.toFixed(2) : value}
          onChange={handleInputChange}
          step={step}
          min={min}
          max={max}
          style={{ 
            width: 60, 
            textAlign: 'center', 
            fontWeight: 600,
            fontFamily: 'monospace',
            padding: '4px',
            borderRadius: 4,
            border: '1px solid var(--border)',
            background: 'var(--bg-tertiary)',
            color: 'var(--text-primary)',
            fontSize: '0.85rem'
          }}
        />
        <button 
          onClick={() => onAdjust(1)} 
          style={{ 
            width: 28, 
            height: 28, 
            borderRadius: 4, 
            border: '1px solid var(--border)', 
            background: 'var(--bg-tertiary)',
            color: 'var(--text-primary)',
            cursor: 'pointer',
            fontSize: '1rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
        >
          +
        </button>
      </div>
    </div>
  );
}

function CustomWizard({ profile, currentProfile, onComplete }: { profile: Profile; currentProfile: Record<string, unknown> | null; onComplete: () => void }) {
  const [step, setStep] = useState(1);
  const [config, setConfig] = useState<WizardConfig>({
    tradingStyle: null,
    riskTolerance: null,
    entryStyle: null,
    emaStack: 'default',
    multipleStrategyA: null,
    multipleStrategyB: null,
  });
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showEmaOptions, setShowEmaOptions] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [generatedProfile, setGeneratedProfile] = useState<Record<string, unknown> | null>(null);

  // Step definitions: 6 steps when "Multiple Strategies" selected, else 5
  const steps = config.entryStyle === 'multiple'
    ? [
        { num: 1, title: 'Trading Style', description: 'How do you want to trade?' },
        { num: 2, title: 'Risk Tolerance', description: 'How much risk per trade?' },
        { num: 3, title: 'Entry Style', description: 'How should the bot enter trades?' },
        { num: 4, title: 'Choose Strategies', description: 'Pick two strategies to combine' },
        { num: 5, title: 'Review', description: 'Review your configuration' },
        { num: 6, title: 'Fine-Tune', description: 'Adjust individual settings' },
      ]
    : [
        { num: 1, title: 'Trading Style', description: 'How do you want to trade?' },
        { num: 2, title: 'Risk Tolerance', description: 'How much risk per trade?' },
        { num: 3, title: 'Entry Style', description: 'How should the bot enter trades?' },
        { num: 4, title: 'Review', description: 'Review your configuration' },
        { num: 5, title: 'Fine-Tune', description: 'Adjust individual settings' },
      ];

  // State for fine-tuning the generated profile
  const [tweakedProfile, setTweakedProfile] = useState<Record<string, unknown> | null>(null);

  const maxStep = config.entryStyle === 'multiple' ? 6 : 5;

  // When switching from Multiple to a single entry style, clamp step so we don't sit on step 5/6 with only 5 steps
  useEffect(() => {
    if (config.entryStyle !== 'multiple' && step > 5) {
      setStep(5);
    }
  }, [config.entryStyle, step]);
  const canProceed = () => {
    switch (step) {
      case 1: return config.tradingStyle !== null;
      case 2: return config.riskTolerance !== null;
      case 3: return config.entryStyle !== null;
      case 4:
        if (config.entryStyle === 'multiple') {
          return config.multipleStrategyA !== null && config.multipleStrategyB !== null && config.multipleStrategyA !== config.multipleStrategyB;
        }
        return true;
      default: return true;
    }
  };

  // Build profile from wizard choices
  const buildProfileFromWizard = (): Record<string, unknown> => {
    const cur = currentProfile as Record<string, unknown> | null | undefined;
    // Base configuration; preserve broker and profile-editor-only fields from current profile
    const baseProfile: Record<string, unknown> = {
      schema_version: 1,
      profile_name: profile.name,
      symbol: cur?.symbol ?? 'USDJPY',
      pip_size: 0.01,
      broker_type: cur?.broker_type ?? 'mt5',
      oanda_token: cur?.oanda_token ?? null,
      oanda_account_id: cur?.oanda_account_id ?? null,
      oanda_environment: cur?.oanda_environment ?? 'practice',
      display_currency: cur?.display_currency ?? 'USD',
      deposit_amount: cur?.deposit_amount ?? null,
      leverage_ratio: cur?.leverage_ratio ?? null,
      created_utc: cur?.created_utc ?? new Date().toISOString().slice(0, 19).replace('T', ' ') + 'Z',
      risk: {},
      strategy: {
        timeframes: {},
        setups: {},
        filters: {
          alignment: { enabled: false, method: 'score', weights: { H4: 1, M15: 1, M1: 1 }, min_score_to_trade: -3 },
          ema_stack_filter: { enabled: false, timeframe: 'M1', periods: [8, 13, 21], min_separation_pips: 0 },
          atr_filter: { enabled: false, timeframe: 'M1', atr_period: 14, min_atr_pips: 0, max_atr_pips: null },
        },
      },
      trade_management: { target: {} },
      execution: { policies: [], loop_poll_seconds: 5, loop_poll_seconds_fast: 2 },
    };

    // Apply trading style
    const riskConfig = baseProfile.risk as Record<string, unknown>;
    const strategyConfig = baseProfile.strategy as Record<string, unknown>;
    const timeframes = strategyConfig.timeframes as Record<string, unknown>;
    const setups = strategyConfig.setups as Record<string, unknown>;
    const targetConfig = (baseProfile.trade_management as Record<string, unknown>).target as Record<string, unknown>;
    const executionConfig = baseProfile.execution as Record<string, unknown>;

    // Get selected EMA stack based on trading style and user selection
    const getEmaStack = () => {
      if (!config.tradingStyle) return [8, 13, 21]; // fallback
      const stylePresets = EMA_STACK_PRESETS[config.tradingStyle];
      const selection = config.emaStack || 'default';
      return stylePresets[selection]?.values || stylePresets.default.values;
    };
    const selectedEmaStack = getEmaStack();

    switch (config.tradingStyle) {
      case 'scalper':
        // Fast trades, M1 focused, small targets
        timeframes['M1'] = { ema_fast: 9, sma_slow: 21, ema_stack: selectedEmaStack };
        timeframes['M15'] = { ema_fast: 9, sma_slow: 21, ema_stack: selectedEmaStack };
        timeframes['H4'] = { ema_fast: 13, sma_slow: 30, ema_stack: selectedEmaStack };
        setups['m1_cross_entry'] = {
          enabled: true, timeframe: 'M1', ema: selectedEmaStack[0], sma: selectedEmaStack[2],
          confirmation: { confirm_bars: 1, require_close_on_correct_side: true, min_distance_pips: 0, max_wait_bars: 3 }
        };
        targetConfig['mode'] = 'fixed_pips';
        targetConfig['pips_default'] = 8;
        targetConfig['rr_default'] = 1.0;
        executionConfig['loop_poll_seconds'] = 3;
        executionConfig['loop_poll_seconds_fast'] = 1;
        break;

      case 'day_trader':
        // Medium trades, M15 focused, moderate targets
        timeframes['M1'] = { ema_fast: 13, sma_slow: 30, ema_stack: selectedEmaStack };
        timeframes['M15'] = { ema_fast: 13, sma_slow: 30, ema_stack: selectedEmaStack };
        timeframes['H4'] = { ema_fast: 13, sma_slow: 30, ema_stack: selectedEmaStack };
        setups['m15_cross_entry'] = {
          enabled: true, timeframe: 'M15', ema: selectedEmaStack[0], sma: selectedEmaStack[2],
          confirmation: { confirm_bars: 1, require_close_on_correct_side: true, min_distance_pips: 0, max_wait_bars: 5 }
        };
        targetConfig['mode'] = 'fixed_pips';
        targetConfig['pips_default'] = 15;
        targetConfig['rr_default'] = 1.5;
        executionConfig['loop_poll_seconds'] = 10;
        executionConfig['loop_poll_seconds_fast'] = 5;
        break;

      case 'swing_trader':
        // Longer trades, H4 focused, larger targets
        timeframes['M1'] = { ema_fast: 21, sma_slow: 50, ema_stack: selectedEmaStack };
        timeframes['M15'] = { ema_fast: 21, sma_slow: 50, ema_stack: selectedEmaStack };
        timeframes['H4'] = { ema_fast: 21, sma_slow: 50, ema_stack: selectedEmaStack };
        setups['h4_cross_entry'] = {
          enabled: true, timeframe: 'H4', ema: selectedEmaStack[0], sma: selectedEmaStack[2],
          confirmation: { confirm_bars: 1, require_close_on_correct_side: true, min_distance_pips: 0, max_wait_bars: 2 }
        };
        targetConfig['mode'] = 'fixed_pips';
        targetConfig['pips_default'] = 30;
        targetConfig['rr_default'] = 2.0;
        executionConfig['loop_poll_seconds'] = 60;
        executionConfig['loop_poll_seconds_fast'] = 30;
        break;
    }

    // Apply risk tolerance - scale from profile limits (Profile Editor is source of truth)
    const profileRisk = (currentProfile?.risk as Record<string, unknown>) || {};
    const profileMaxLots = (profileRisk.max_lots as number) ?? 0.1;
    const profileMaxSpread = (profileRisk.max_spread_pips as number) ?? 2;
    const profileMaxTrades = (profileRisk.max_trades_per_day as number) ?? 10;
    // Min stop pips: always use profile editor value (1–100); user can set >10 for scalping if desired
    const profileMinStop = (profileRisk.min_stop_pips as number);
    const minStopPips = typeof profileMinStop === 'number' && profileMinStop >= 0
      ? Math.max(1, Math.min(100, profileMinStop))
      : 10;

    switch (config.riskTolerance) {
      case 'conservative':
        riskConfig['max_lots'] = Math.round(profileMaxLots * 0.35 * 100) / 100 || 0.05;
        riskConfig['max_spread_pips'] = Math.round(profileMaxSpread * 0.25 * 10) / 10 || 0.5;
        riskConfig['min_stop_pips'] = minStopPips;
        riskConfig['max_trades_per_day'] = Math.max(1, Math.round(profileMaxTrades * 0.5));
        riskConfig['max_open_trades'] = 1;
        riskConfig['cooldown_minutes_after_loss'] = 15;
        riskConfig['require_stop'] = true;
        riskConfig['risk_per_trade_pct'] = 0.5;
        riskConfig['max_daily_loss_pct'] = 2;
        break;

      case 'moderate':
        riskConfig['max_lots'] = Math.round(profileMaxLots * 0.67 * 100) / 100 || 0.1;
        riskConfig['max_spread_pips'] = Math.round(profileMaxSpread * 0.5 * 10) / 10 || 0.8;
        riskConfig['min_stop_pips'] = minStopPips;
        riskConfig['max_trades_per_day'] = Math.max(1, Math.round(profileMaxTrades * 1.0));
        riskConfig['max_open_trades'] = 2;
        riskConfig['cooldown_minutes_after_loss'] = 5;
        riskConfig['require_stop'] = true;
        riskConfig['risk_per_trade_pct'] = 1;
        riskConfig['max_daily_loss_pct'] = 4;
        break;

      case 'aggressive':
        riskConfig['max_lots'] = Math.round(profileMaxLots * 1.0 * 100) / 100 || 0.15;
        riskConfig['max_spread_pips'] = Math.round(profileMaxSpread * 0.6 * 10) / 10 || 1.2;
        riskConfig['min_stop_pips'] = minStopPips;
        riskConfig['max_trades_per_day'] = Math.max(1, Math.round(profileMaxTrades * 1.0));
        riskConfig['max_open_trades'] = 3;
        riskConfig['cooldown_minutes_after_loss'] = 0;
        riskConfig['require_stop'] = true;
        riskConfig['risk_per_trade_pct'] = 2;
        riskConfig['max_daily_loss_pct'] = 6;
        break;
    }

    // Apply entry style - configure execution policies
    const policies: Record<string, unknown>[] = [];
    const setupId = Object.keys(setups)[0] || 'm1_cross_entry';
    const entryTimeframe = config.tradingStyle === 'swing_trader' ? 'H4' : config.tradingStyle === 'day_trader' ? 'M15' : 'M1';
    const tpPips = targetConfig['pips_default'] as number;
    const slPips = riskConfig['min_stop_pips'] as number;

    switch (config.entryStyle) {
      case 'ema_cross':
        policies.push({
          type: 'confirmed_cross',
          id: 'custom_confirmed_cross',
          enabled: true,
          setup_id: setupId,
        });
        break;

      case 'rsi_dips':
        policies.push({
          type: 'indicator_based',
          id: 'custom_rsi_buy',
          enabled: true,
          timeframe: entryTimeframe,
          regime: 'bull',
          side: 'buy',
          rsi_period: 14,
          rsi_oversold: 30,
          rsi_overbought: 70,
          rsi_zone: 'oversold',
          use_macd_cross: true,
          macd_fast: 12,
          macd_slow: 26,
          macd_signal: 9,
          tp_pips: tpPips,
          sl_pips: slPips,
        });
        policies.push({
          type: 'indicator_based',
          id: 'custom_rsi_sell',
          enabled: true,
          timeframe: entryTimeframe,
          regime: 'bear',
          side: 'sell',
          rsi_period: 14,
          rsi_oversold: 30,
          rsi_overbought: 70,
          rsi_zone: 'overbought',
          use_macd_cross: true,
          macd_fast: 12,
          macd_slow: 26,
          macd_signal: 9,
          tp_pips: tpPips,
          sl_pips: slPips,
        });
        break;

      case 'price_levels':
        (strategyConfig.filters as Record<string, unknown>).alignment = {
          enabled: true,
          method: 'score',
          weights: { H4: 2, M15: 1, M1: 1 },
          min_score_to_trade: 1,
        };
        policies.push({
          type: 'confirmed_cross',
          id: 'custom_confirmed_cross',
          enabled: true,
          setup_id: setupId,
        });
        break;

      case 'bollinger_bands':
        policies.push({
          type: 'bollinger_bands',
          id: 'custom_bb_buy',
          enabled: true,
          timeframe: entryTimeframe,
          period: 20,
          std_dev: 2,
          trigger: 'lower_band_buy',
          regime: 'bull',
          side: 'buy',
          tp_pips: tpPips,
          sl_pips: slPips,
        });
        policies.push({
          type: 'bollinger_bands',
          id: 'custom_bb_sell',
          enabled: true,
          timeframe: entryTimeframe,
          period: 20,
          std_dev: 2,
          trigger: 'upper_band_sell',
          regime: 'bear',
          side: 'sell',
          tp_pips: tpPips,
          sl_pips: slPips,
        });
        break;

      case 'vwap':
        policies.push({
          type: 'vwap',
          id: 'custom_vwap_buy',
          enabled: true,
          timeframe: entryTimeframe,
          trigger: 'cross_above',
          side: 'buy',
          tp_pips: tpPips,
          sl_pips: slPips,
        });
        policies.push({
          type: 'vwap',
          id: 'custom_vwap_sell',
          enabled: true,
          timeframe: entryTimeframe,
          trigger: 'cross_below',
          side: 'sell',
          tp_pips: tpPips,
          sl_pips: slPips,
        });
        break;

      case 'multiple': {
        // Build policies from user-chosen Strategy A and Strategy B (no RSI)
        const addPoliciesFor = (combo: ComboStrategy, prefix: string) => {
          if (combo === 'ema_cross') {
            policies.push({ type: 'confirmed_cross', id: `${prefix}_ema_cross`, enabled: true, setup_id: setupId });
            return;
          }
          if (combo === 'price_levels') {
            (strategyConfig.filters as Record<string, unknown>).alignment = {
              enabled: true,
              method: 'score',
              weights: { H4: 2, M15: 1, M1: 1 },
              min_score_to_trade: 1,
            };
            policies.push({ type: 'confirmed_cross', id: `${prefix}_confirmed_cross`, enabled: true, setup_id: setupId });
            return;
          }
          if (combo === 'bollinger_bands') {
            policies.push({
              type: 'bollinger_bands',
              id: `${prefix}_bb_buy`,
              enabled: true,
              timeframe: entryTimeframe,
              period: 20,
              std_dev: 2,
              trigger: 'lower_band_buy',
              regime: 'bull',
              side: 'buy',
              tp_pips: tpPips,
              sl_pips: slPips,
            });
            policies.push({
              type: 'bollinger_bands',
              id: `${prefix}_bb_sell`,
              enabled: true,
              timeframe: entryTimeframe,
              period: 20,
              std_dev: 2,
              trigger: 'upper_band_sell',
              regime: 'bear',
              side: 'sell',
              tp_pips: tpPips,
              sl_pips: slPips,
            });
            return;
          }
          if (combo === 'vwap') {
            policies.push({
              type: 'vwap',
              id: `${prefix}_vwap_buy`,
              enabled: true,
              timeframe: entryTimeframe,
              trigger: 'cross_above',
              side: 'buy',
              tp_pips: tpPips,
              sl_pips: slPips,
            });
            policies.push({
              type: 'vwap',
              id: `${prefix}_vwap_sell`,
              enabled: true,
              timeframe: entryTimeframe,
              trigger: 'cross_below',
              side: 'sell',
              tp_pips: tpPips,
              sl_pips: slPips,
            });
          }
        };
        if (config.multipleStrategyA) addPoliciesFor(config.multipleStrategyA, 'custom_ms_a');
        if (config.multipleStrategyB) addPoliciesFor(config.multipleStrategyB, 'custom_ms_b');
        break;
      }
    }

    executionConfig['policies'] = policies;

    // Generate custom preset name based on wizard choices
    // Format: "Custom: Style/Risk/Entry"
    const styleAbbrev = config.tradingStyle === 'scalper' ? 'Scalp' : 
                        config.tradingStyle === 'day_trader' ? 'DayT' : 'Swing';
    const riskAbbrev = config.riskTolerance === 'conservative' ? 'Cons' : 
                       config.riskTolerance === 'moderate' ? 'Mod' : 'Agg';
    const entryAbbrev = config.entryStyle === 'ema_cross' ? 'EMAC' :
                        config.entryStyle === 'rsi_dips' ? 'RSI' :
                        config.entryStyle === 'price_levels' ? 'PL' :
                        config.entryStyle === 'bollinger_bands' ? 'BB' :
                        config.entryStyle === 'vwap' ? 'VWAP' : 'MS';
    baseProfile['active_preset_name'] = `Custom: ${styleAbbrev}/${riskAbbrev}/${entryAbbrev}`;

    return baseProfile;
  };

  const handleNext = () => {
    if (step < maxStep) {
      setStep(step + 1);
    }
    // Generate profile when moving into Review step
    if ((step === 3 && config.entryStyle !== 'multiple') || (step === 4 && config.entryStyle === 'multiple')) {
      const built = buildProfileFromWizard();
      setGeneratedProfile(built);
      setTweakedProfile(JSON.parse(JSON.stringify(built)));
    }
  };

  const handleBack = () => {
    if (step > 1) {
      setStep(step - 1);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      // Use tweaked profile if on step 5, otherwise use generated profile
      const profileData = JSON.parse(JSON.stringify(tweakedProfile || generatedProfile || buildProfileFromWizard())) as Record<string, unknown>;
      const limits = (currentProfile?.risk as Record<string, unknown>) || {};
      const wizardRisk = (profileData.risk as Record<string, unknown>) || {};
      // Profile Editor risk (limits) are final: never overwrite with wizard values
      profileData.risk = limits;
      // Effective risk for this preset: wizard risk capped by profile limits
      const cap = (a: number, b: number) => (typeof a === 'number' && typeof b === 'number' ? Math.min(a, b) : a);
      profileData.effective_risk = {
        ...wizardRisk,
        max_lots: cap((wizardRisk.max_lots as number) ?? 0.1, (limits.max_lots as number) ?? 1),
        max_spread_pips: cap((wizardRisk.max_spread_pips as number) ?? 2, (limits.max_spread_pips as number) ?? 5),
        max_trades_per_day: cap((wizardRisk.max_trades_per_day as number) ?? 10, (limits.max_trades_per_day as number) ?? 100),
        max_open_trades: cap((wizardRisk.max_open_trades as number) ?? 2, (limits.max_open_trades as number) ?? 10),
      };
      // Broker and profile-editor-only fields: never overwrite with wizard; keep current profile values
      profileData.broker_type = currentProfile?.broker_type ?? 'mt5';
      profileData.oanda_token = currentProfile?.oanda_token ?? null;
      profileData.oanda_account_id = currentProfile?.oanda_account_id ?? null;
      profileData.oanda_environment = currentProfile?.oanda_environment ?? 'practice';
      profileData.display_currency = currentProfile?.display_currency ?? 'USD';
      profileData.deposit_amount = currentProfile?.deposit_amount ?? null;
      profileData.leverage_ratio = currentProfile?.leverage_ratio ?? null;
      if (currentProfile?.created_utc) profileData.created_utc = currentProfile.created_utc;
      await api.saveProfile(profile.path, profileData);
      onComplete();
    } catch (e: unknown) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  // Reset tweaked profile to wizard defaults
  const handleResetToDefaults = () => {
    if (generatedProfile) {
      setTweakedProfile(JSON.parse(JSON.stringify(generatedProfile)));
    }
  };

  // Update a nested value in tweaked profile
  const updateTweakedValue = (path: string[], value: unknown) => {
    if (!tweakedProfile) return;
    const updated = JSON.parse(JSON.stringify(tweakedProfile));
    let obj = updated;
    for (let i = 0; i < path.length - 1; i++) {
      if (obj[path[i]] === undefined) obj[path[i]] = {};
      obj = obj[path[i]];
    }
    obj[path[path.length - 1]] = value;
    setTweakedProfile(updated);
  };

  // Increment/decrement a numeric value
  const adjustValue = (path: string[], delta: number, min?: number, max?: number, step?: number) => {
    if (!tweakedProfile) return;
    let obj: Record<string, unknown> = tweakedProfile;
    for (let i = 0; i < path.length - 1; i++) {
      obj = obj[path[i]] as Record<string, unknown>;
      if (!obj) return;
    }
    const current = obj[path[path.length - 1]] as number;
    if (typeof current !== 'number') return;
    let newVal = current + delta * (step || 1);
    if (min !== undefined) newVal = Math.max(min, newVal);
    if (max !== undefined) newVal = Math.min(max, newVal);
    updateTweakedValue(path, Number(newVal.toFixed(4)));
  };

  // Check if fine-tuned values exceed profile editor limits
  const violatesProfileLimits = (() => {
    if (step !== 5 || !tweakedProfile || !currentProfile) return false;
    const profileRisk = (currentProfile.risk as Record<string, unknown>) || {};
    const profileMaxLots = profileRisk.max_lots as number | undefined;
    const profileMaxSpread = profileRisk.max_spread_pips as number | undefined;
    const profileMaxTrades = profileRisk.max_trades_per_day as number | undefined;
    const tweakedRisk = (tweakedProfile.risk as Record<string, unknown>) || {};
    const tweakedLots = tweakedRisk.max_lots as number | undefined;
    const tweakedSpread = tweakedRisk.max_spread_pips as number | undefined;
    const tweakedTrades = tweakedRisk.max_trades_per_day as number | undefined;
    const violatesLots = profileMaxLots != null && tweakedLots != null && tweakedLots > profileMaxLots;
    const violatesSpread = profileMaxSpread != null && tweakedSpread != null && tweakedSpread > profileMaxSpread;
    const violatesTrades = profileMaxTrades != null && tweakedTrades != null && tweakedTrades > profileMaxTrades;
    return violatesLots || violatesSpread || violatesTrades;
  })();

  // Style option card component
  const OptionCard = ({ 
    selected, 
    onClick, 
    title, 
    description, 
    icon,
    details 
  }: { 
    selected: boolean; 
    onClick: () => void; 
    title: string; 
    description: string;
    icon: string;
    details?: string[];
  }) => (
    <div
      onClick={onClick}
      style={{
        padding: 20,
        borderRadius: 8,
        border: `2px solid ${selected ? 'var(--accent)' : 'var(--border)'}`,
        background: selected ? 'rgba(59, 130, 246, 0.1)' : 'var(--bg-secondary)',
        cursor: 'pointer',
        transition: 'all 0.2s ease',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
        <span style={{ fontSize: '1.5rem' }}>{icon}</span>
        <h4 style={{ margin: 0, color: selected ? 'var(--accent)' : 'var(--text-primary)' }}>{title}</h4>
      </div>
      <p style={{ margin: '0 0 8px 0', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>{description}</p>
      {details && (
        <ul style={{ margin: 0, paddingLeft: 20, fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
          {details.map((d, i) => <li key={i}>{d}</li>)}
        </ul>
      )}
    </div>
  );

  return (
    <div>
      {/* Progress Indicator */}
      <div className="card mb-4">
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
          {steps.map((s) => (
            <div 
              key={s.num}
              style={{ 
                flex: 1, 
                textAlign: 'center',
                opacity: s.num <= step ? 1 : 0.5,
              }}
            >
              <div style={{
                width: 36,
                height: 36,
                borderRadius: '50%',
                background: s.num < step ? 'var(--success)' : s.num === step ? 'var(--accent)' : 'var(--bg-tertiary)',
                color: s.num <= step ? 'white' : 'var(--text-secondary)',
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontWeight: 600,
                marginBottom: 8,
              }}>
                {s.num < step ? '✓' : s.num}
              </div>
              <div style={{ fontSize: '0.85rem', fontWeight: s.num === step ? 600 : 400 }}>{s.title}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Step Content */}
      {step === 1 && (
        <div className="card">
          <h3 className="card-title">Step 1: Trading Style</h3>
          <p style={{ color: 'var(--text-secondary)', marginBottom: 20 }}>
            How do you want to trade? This determines the timeframe focus and trade duration.
          </p>
          <div style={{ display: 'grid', gap: 16 }}>
            <OptionCard
              selected={config.tradingStyle === 'scalper'}
              onClick={() => setConfig({ ...config, tradingStyle: 'scalper' })}
              title="Scalper"
              icon="⚡"
              description="Quick trades (1-5 minutes), small profits, many opportunities"
              details={['Focuses on M1 (1-minute) chart', '8 pip profit targets', 'Fast polling (1-3 seconds)', 'Best for active monitoring']}
            />
            <OptionCard
              selected={config.tradingStyle === 'day_trader'}
              onClick={() => setConfig({ ...config, tradingStyle: 'day_trader' })}
              title="Day Trader"
              icon="📈"
              description="Trades within the day (5-60 minutes), moderate targets"
              details={['Focuses on M15 (15-minute) chart', '15 pip profit targets', 'Moderate polling (5-10 seconds)', 'Good balance of activity and patience']}
            />
            <OptionCard
              selected={config.tradingStyle === 'swing_trader'}
              onClick={() => setConfig({ ...config, tradingStyle: 'swing_trader' })}
              title="Swing Trader"
              icon="🎯"
              description="Hold for hours or days, larger moves, fewer trades"
              details={['Focuses on H4 (4-hour) chart', '30+ pip profit targets', 'Slow polling (30-60 seconds)', 'Minimal monitoring needed']}
            />
          </div>

          {/* EMA Stack Selector - Optional Advanced Setting */}
          {config.tradingStyle && (
            <div style={{ marginTop: 20 }}>
              <button
                type="button"
                onClick={() => setShowEmaOptions(!showEmaOptions)}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: 'var(--accent)',
                  cursor: 'pointer',
                  fontSize: '0.9rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  padding: 0,
                }}
              >
                <span style={{ transform: showEmaOptions ? 'rotate(90deg)' : 'rotate(0)', transition: 'transform 0.2s' }}>▶</span>
                Advanced: EMA Stack (optional)
              </button>
              
              {showEmaOptions && (
                <div style={{ 
                  marginTop: 12, 
                  padding: 16, 
                  background: 'var(--bg-tertiary)', 
                  borderRadius: 8,
                  borderLeft: '3px solid var(--accent)'
                }}>
                  <p style={{ margin: '0 0 12px 0', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                    The EMA Stack defines the moving average periods used for trend detection. 
                    Leave as default unless you have a specific preference.
                  </p>
                  <div className="form-group" style={{ marginBottom: 0 }}>
                    <label>EMA Stack for {config.tradingStyle === 'scalper' ? 'Scalping' : config.tradingStyle === 'day_trader' ? 'Day Trading' : 'Swing Trading'}</label>
                    <select
                      value={config.emaStack || 'default'}
                      onChange={(e) => setConfig({ ...config, emaStack: e.target.value as 'default' | 'alt1' | 'alt2' })}
                    >
                      {config.tradingStyle && Object.entries(EMA_STACK_PRESETS[config.tradingStyle]).map(([key, preset]) => (
                        <option key={key} value={key}>{preset.label}</option>
                      ))}
                    </select>
                  </div>
                  <p style={{ margin: '8px 0 0 0', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                    Current selection: {config.tradingStyle && config.emaStack && 
                      EMA_STACK_PRESETS[config.tradingStyle][config.emaStack]?.values.join(', ')} EMA periods
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {step === 2 && (() => {
        const profileRisk = (currentProfile?.risk as Record<string, unknown>) || {};
        const pMaxLots = (profileRisk.max_lots as number) ?? 0.1;
        const pMaxSpread = (profileRisk.max_spread_pips as number) ?? 2;
        const pMaxTrades = (profileRisk.max_trades_per_day as number) ?? 10;
        const consLots = (Math.round(pMaxLots * 0.35 * 100) / 100) || 0.05;
        const consTrades = Math.max(1, Math.round(pMaxTrades * 0.5));
        const consSpread = (Math.round(pMaxSpread * 0.25 * 10) / 10) || 0.5;
        const modLots = (Math.round(pMaxLots * 0.67 * 100) / 100) || 0.1;
        const modTrades = Math.max(1, Math.round(pMaxTrades * 1.0));
        const modSpread = (Math.round(pMaxSpread * 0.5 * 10) / 10) || 0.8;
        const aggLots = (Math.round(pMaxLots * 1.0 * 100) / 100) || 0.15;
        const aggTrades = Math.max(1, Math.round(pMaxTrades * 1.0));
        const aggSpread = (Math.round(pMaxSpread * 0.6 * 10) / 10) || 1.2;
        return (
          <div className="card">
            <h3 className="card-title">Step 2: Risk Tolerance</h3>
            <p style={{ color: 'var(--text-secondary)', marginBottom: 20 }}>
              How much risk are you comfortable with per trade? Values scale from your Profile Editor limits.
            </p>
            <div style={{ display: 'grid', gap: 16 }}>
              <OptionCard
                selected={config.riskTolerance === 'conservative'}
                onClick={() => setConfig({ ...config, riskTolerance: 'conservative' })}
                title="Conservative"
                icon="🛡️"
                description="Lower risk, smaller positions, strict filters"
                details={[`${consLots} lots per trade`, `Max ${consTrades} trades per day`, `${consSpread} pip max spread`, '15 min cooldown after loss', '2% max daily loss']}
              />
              <OptionCard
                selected={config.riskTolerance === 'moderate'}
                onClick={() => setConfig({ ...config, riskTolerance: 'moderate' })}
                title="Moderate"
                icon="⚖️"
                description="Balanced risk and reward"
                details={[`${modLots} lots per trade`, `Max ${modTrades} trades per day`, `${modSpread} pip max spread`, '5 min cooldown after loss', '4% max daily loss']}
              />
              <OptionCard
                selected={config.riskTolerance === 'aggressive'}
                onClick={() => setConfig({ ...config, riskTolerance: 'aggressive' })}
                title="Aggressive"
                icon="🔥"
                description="Higher risk for potentially higher returns"
                details={[`${aggLots} lots per trade`, `Max ${aggTrades} trades per day`, `${aggSpread} pip max spread`, 'No cooldown after loss', '6% max daily loss']}
              />
            </div>
          </div>
        );
      })()}

      {step === 3 && (
        <div className="card">
          <h3 className="card-title">Step 3: Entry Style</h3>
          <p style={{ color: 'var(--text-secondary)', marginBottom: 20 }}>
            How should the bot decide when to enter a trade?
          </p>
          <div style={{ display: 'grid', gap: 16 }}>
            <OptionCard
              selected={config.entryStyle === 'ema_cross'}
              onClick={() => setConfig({ ...config, entryStyle: 'ema_cross', multipleStrategyA: null, multipleStrategyB: null })}
              title="EMA Cross"
              icon="✖️"
              description="Buy or sell based on EMA/SMA cross direction"
              details={['BUY: EMA crosses above SMA', 'SELL: EMA crosses below SMA', 'Waits for confirmation bar', 'Works in both trends']}
            />
            <OptionCard
              selected={config.entryStyle === 'rsi_dips'}
              onClick={() => setConfig({ ...config, entryStyle: 'rsi_dips', multipleStrategyA: null, multipleStrategyB: null })}
              title="RSI Dips"
              icon="📉"
              description="Buy oversold dips, sell overbought rallies"
              details={['BUY: RSI < 30 in uptrend', 'SELL: RSI > 70 in downtrend', 'Mean-reversion style', 'MACD confirmation included']}
            />
            <OptionCard
              selected={config.entryStyle === 'price_levels'}
              onClick={() => setConfig({ ...config, entryStyle: 'price_levels', multipleStrategyA: null, multipleStrategyB: null })}
              title="Price Levels"
              icon="📍"
              description="EMA cross with multi-timeframe alignment filter"
              details={['Bi-directional EMA cross', 'Requires H4/M15/M1 agreement', 'Higher quality signals', 'Fewer but better trades']}
            />
            <OptionCard
              selected={config.entryStyle === 'bollinger_bands'}
              onClick={() => setConfig({ ...config, entryStyle: 'bollinger_bands', multipleStrategyA: null, multipleStrategyB: null })}
              title="Bollinger Bands"
              icon="〰️"
              description="Mean reversion at lower/upper band"
              details={['BUY: price at/below lower band in uptrend', 'SELL: price at/above upper band in downtrend', 'M15, 20-period, 2 std dev', 'Clear levels for entry']}
            />
            <OptionCard
              selected={config.entryStyle === 'vwap'}
              onClick={() => setConfig({ ...config, entryStyle: 'vwap', multipleStrategyA: null, multipleStrategyB: null })}
              title="VWAP"
              icon="📊"
              description="Entries on price vs VWAP"
              details={['BUY: price crosses above VWAP', 'SELL: price crosses below VWAP', 'Volume-weighted average price', 'Good for intraday trend']}
            />
            <OptionCard
              selected={config.entryStyle === 'multiple'}
              onClick={() => setConfig({ ...config, entryStyle: 'multiple' })}
              title="Multiple Strategies"
              icon="🔀"
              description="Combine two strategies of your choice (EMA Cross, Price Levels, Bollinger Bands, or VWAP)"
              details={['Pick Strategy A and Strategy B in the next step', 'EMA Cross, Price Levels, Bollinger Bands, VWAP', 'Maximum flexibility', 'Good for active traders']}
            />
          </div>
        </div>
      )}

      {step === 4 && config.entryStyle === 'multiple' && (
        <div className="card">
          <h3 className="card-title">Step 4: Choose two strategies to combine</h3>
          <p style={{ color: 'var(--text-secondary)', marginBottom: 20 }}>
            Select Strategy A and Strategy B. They must be different. RSI is not available in combinations.
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, marginBottom: 16 }}>
            <div>
              <label style={{ display: 'block', marginBottom: 8, fontWeight: 600 }}>Strategy A</label>
              <select
                value={config.multipleStrategyA ?? ''}
                onChange={(e) => setConfig({ ...config, multipleStrategyA: (e.target.value || null) as ComboStrategy | null })}
                style={{
                  width: '100%',
                  padding: '10px 12px',
                  borderRadius: 6,
                  border: '1px solid var(--border)',
                  background: 'var(--bg-secondary)',
                  color: 'var(--text-primary)',
                  fontSize: '0.95rem',
                }}
              >
                <option value="">— Select —</option>
                <option value="ema_cross">EMA Cross</option>
                <option value="price_levels">Price Levels</option>
                <option value="bollinger_bands">Bollinger Bands</option>
                <option value="vwap">VWAP</option>
              </select>
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: 8, fontWeight: 600 }}>Strategy B</label>
              <select
                value={config.multipleStrategyB ?? ''}
                onChange={(e) => setConfig({ ...config, multipleStrategyB: (e.target.value || null) as ComboStrategy | null })}
                style={{
                  width: '100%',
                  padding: '10px 12px',
                  borderRadius: 6,
                  border: '1px solid var(--border)',
                  background: 'var(--bg-secondary)',
                  color: 'var(--text-primary)',
                  fontSize: '0.95rem',
                }}
              >
                <option value="">— Select —</option>
                <option value="ema_cross">EMA Cross</option>
                <option value="price_levels">Price Levels</option>
                <option value="bollinger_bands">Bollinger Bands</option>
                <option value="vwap">VWAP</option>
              </select>
            </div>
          </div>
          {config.multipleStrategyA && config.multipleStrategyB && config.multipleStrategyA === config.multipleStrategyB && (
            <p style={{ color: 'var(--danger)', fontSize: '0.9rem', marginTop: 8 }}>Please choose two different strategies.</p>
          )}
        </div>
      )}

      {((step === 4 && config.entryStyle !== 'multiple') || (step === 5 && config.entryStyle === 'multiple')) && (
        <div>
          <div className="card mb-4">
            <h3 className="card-title">{config.entryStyle === 'multiple' ? 'Step 5: Review & Save' : 'Step 4: Review & Save'}</h3>
            <p style={{ color: 'var(--text-secondary)', marginBottom: 20 }}>
              Review your custom configuration before saving.
            </p>

            {/* Summary */}
            <div style={{ 
              padding: 16, 
              background: 'var(--bg-tertiary)', 
              borderRadius: 8, 
              borderLeft: '4px solid var(--accent)',
              marginBottom: 20
            }}>
              <h4 style={{ margin: '0 0 12px 0' }}>Your Configuration Summary</h4>
              <div className="grid-3">
                <div>
                  <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Style:</span>
                  <div style={{ fontWeight: 600, marginTop: 4 }}>
                    {config.tradingStyle === 'scalper' && '⚡ Scalper'}
                    {config.tradingStyle === 'day_trader' && '📈 Day Trader'}
                    {config.tradingStyle === 'swing_trader' && '🎯 Swing Trader'}
                  </div>
                </div>
                <div>
                  <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Risk:</span>
                  <div style={{ fontWeight: 600, marginTop: 4 }}>
                    {config.riskTolerance === 'conservative' && '🛡️ Conservative'}
                    {config.riskTolerance === 'moderate' && '⚖️ Moderate'}
                    {config.riskTolerance === 'aggressive' && '🔥 Aggressive'}
                  </div>
                </div>
                <div>
                  <span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Entry:</span>
                  <div style={{ fontWeight: 600, marginTop: 4 }}>
                    {config.entryStyle === 'ema_cross' && '✖️ EMA Cross'}
                    {config.entryStyle === 'rsi_dips' && '📉 RSI Dips'}
                    {config.entryStyle === 'price_levels' && '📍 Price Levels'}
                    {config.entryStyle === 'bollinger_bands' && '〰️ Bollinger Bands'}
                    {config.entryStyle === 'vwap' && '📊 VWAP'}
                    {config.entryStyle === 'multiple' && config.multipleStrategyA && config.multipleStrategyB && (
                      <>🔀 Multiple ({config.multipleStrategyA.replace('_', ' ')}+{config.multipleStrategyB.replace('_', ' ')})</>
                    )}
                    {config.entryStyle === 'multiple' && (!config.multipleStrategyA || !config.multipleStrategyB) && '🔀 Multiple'}
                  </div>
                </div>
              </div>
            </div>

            {/* What this means */}
            {generatedProfile && (
              <div style={{ marginBottom: 20 }}>
                <h4 style={{ marginBottom: 12 }}>What this means:</h4>
                <ul style={{ margin: 0, paddingLeft: 20, color: 'var(--text-secondary)', lineHeight: 1.8 }}>
                  <li>
                    <strong>Position Size:</strong> {String((generatedProfile.risk as Record<string, unknown>)?.max_lots ?? '-')} lots per trade
                  </li>
                  <li>
                    <strong>Profit Target:</strong> {String(((generatedProfile.trade_management as Record<string, unknown>)?.target as Record<string, unknown>)?.pips_default ?? '-')} pips
                  </li>
                  <li>
                    <strong>Max Trades/Day:</strong> {String((generatedProfile.risk as Record<string, unknown>)?.max_trades_per_day ?? '-')}
                  </li>
                  <li>
                    <strong>Polling Speed:</strong> Every {String((generatedProfile.execution as Record<string, unknown>)?.loop_poll_seconds ?? '-')} seconds
                  </li>
                </ul>
              </div>
            )}

            {/* Advanced Settings Toggle */}
            <div style={{ marginBottom: 20 }}>
              <button
                className="btn btn-secondary"
                style={{ fontSize: '0.85rem' }}
                onClick={() => setShowAdvanced(!showAdvanced)}
              >
                {showAdvanced ? 'Hide' : 'Show'} Advanced Settings
              </button>
            </div>

            {showAdvanced && generatedProfile && (
              <div style={{ marginBottom: 20 }}>
                <h4 style={{ marginBottom: 12 }}>Full Configuration (JSON):</h4>
                <pre style={{ 
                  background: 'var(--bg-primary)', 
                  padding: 12, 
                  borderRadius: 6,
                  fontSize: '0.75rem',
                  overflow: 'auto',
                  maxHeight: 300
                }}>
                  {JSON.stringify(generatedProfile, null, 2)}
                </pre>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: 8 }}>
                  For advanced customization, edit the profile directly in the Profile Editor after saving.
                </p>
              </div>
            )}

            {error && (
              <div style={{ padding: 12, background: 'rgba(239, 68, 68, 0.1)', borderRadius: 6, marginBottom: 16 }}>
                <p style={{ color: 'var(--danger)', margin: 0 }}>Error: {error}</p>
              </div>
            )}
          </div>
        </div>
      )}

      {((step === 5 && config.entryStyle !== 'multiple') || (step === 6 && config.entryStyle === 'multiple')) && tweakedProfile && (
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
            <h3 className="card-title" style={{ margin: 0 }}>{config.entryStyle === 'multiple' ? 'Step 6: Fine-Tune Settings' : 'Step 5: Fine-Tune Settings'}</h3>
            <button className="btn btn-secondary" onClick={handleResetToDefaults} style={{ fontSize: '0.8rem' }}>
              Reset to Wizard Defaults
            </button>
          </div>
          <p style={{ color: 'var(--text-secondary)', marginBottom: 20 }}>
            Adjust any settings using the +/- buttons or type directly. These values come from your wizard choices.
          </p>

          {violatesProfileLimits && (
            <div style={{ padding: 12, background: 'rgba(239, 68, 68, 0.15)', border: '1px solid var(--danger)', borderRadius: 8, marginBottom: 20 }}>
              <strong style={{ color: 'var(--danger)' }}>Profile limit exceeded</strong>
              <p style={{ margin: '8px 0 0', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                You have violated your Profile Editor risk settings. This change cannot be made until you update your risk settings in the Profile Editor.
              </p>
            </div>
          )}

          {/* Risk Settings */}
          <div style={{ marginBottom: 20 }}>
            <h4 style={{ color: 'var(--accent)', marginBottom: 12, fontSize: '0.95rem' }}>Risk Settings</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 12 }}>
              <SettingControl 
                label="Max Lots" 
                value={(tweakedProfile.risk as Record<string, unknown>)?.max_lots as number} 
                onAdjust={(d) => adjustValue(['risk', 'max_lots'], d, 0.01, 10, 0.01)} 
                onChange={(v) => updateTweakedValue(['risk', 'max_lots'], v)}
                step={0.01}
                min={0.01}
                max={10}
              />
              <SettingControl 
                label="Min Stop Pips" 
                value={(tweakedProfile.risk as Record<string, unknown>)?.min_stop_pips as number} 
                onAdjust={(d) => adjustValue(['risk', 'min_stop_pips'], d, 1, 100, 1)} 
                onChange={(v) => updateTweakedValue(['risk', 'min_stop_pips'], v)}
                step={1}
                min={1}
                max={100}
              />
              <SettingControl 
                label="Max Spread Pips" 
                value={(tweakedProfile.risk as Record<string, unknown>)?.max_spread_pips as number} 
                onAdjust={(d) => adjustValue(['risk', 'max_spread_pips'], d, 0.1, 5, 0.1)} 
                onChange={(v) => updateTweakedValue(['risk', 'max_spread_pips'], v)}
                step={0.1}
                min={0.1}
                max={5}
              />
              <SettingControl 
                label="Max Trades/Day" 
                value={(tweakedProfile.risk as Record<string, unknown>)?.max_trades_per_day as number} 
                onAdjust={(d) => adjustValue(['risk', 'max_trades_per_day'], d, 1, 100, 1)} 
                onChange={(v) => updateTweakedValue(['risk', 'max_trades_per_day'], v)}
                step={1}
                min={1}
                max={100}
              />
              <SettingControl 
                label="Max Open Trades" 
                value={(tweakedProfile.risk as Record<string, unknown>)?.max_open_trades as number} 
                onAdjust={(d) => adjustValue(['risk', 'max_open_trades'], d, 1, 10, 1)} 
                onChange={(v) => updateTweakedValue(['risk', 'max_open_trades'], v)}
                step={1}
                min={1}
                max={10}
              />
              <SettingControl 
                label="Loss Cooldown (min)" 
                value={(tweakedProfile.risk as Record<string, unknown>)?.cooldown_minutes_after_loss as number} 
                onAdjust={(d) => adjustValue(['risk', 'cooldown_minutes_after_loss'], d, 0, 60, 1)} 
                onChange={(v) => updateTweakedValue(['risk', 'cooldown_minutes_after_loss'], v)}
                step={1}
                min={0}
                max={60}
              />
              <SettingControl 
                label="Max Daily Loss %" 
                value={(tweakedProfile.risk as Record<string, unknown>)?.max_daily_loss_pct as number} 
                onAdjust={(d) => adjustValue(['risk', 'max_daily_loss_pct'], d, 0.5, 20, 0.5)} 
                onChange={(v) => updateTweakedValue(['risk', 'max_daily_loss_pct'], v)}
                step={0.5}
                min={0.5}
                max={20}
              />
            </div>
          </div>

          {/* Target Settings */}
          <div style={{ marginBottom: 20 }}>
            <h4 style={{ color: 'var(--accent)', marginBottom: 12, fontSize: '0.95rem' }}>Profit Target</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 12 }}>
              <SettingControl 
                label="Target Pips" 
                value={((tweakedProfile.trade_management as Record<string, unknown>)?.target as Record<string, unknown>)?.pips_default as number} 
                onAdjust={(d) => adjustValue(['trade_management', 'target', 'pips_default'], d, 1, 100, 1)} 
                onChange={(v) => updateTweakedValue(['trade_management', 'target', 'pips_default'], v)}
                step={1}
                min={1}
                max={100}
              />
              <SettingControl 
                label="R:R Ratio" 
                value={((tweakedProfile.trade_management as Record<string, unknown>)?.target as Record<string, unknown>)?.rr_default as number} 
                onAdjust={(d) => adjustValue(['trade_management', 'target', 'rr_default'], d, 0.5, 5, 0.1)} 
                onChange={(v) => updateTweakedValue(['trade_management', 'target', 'rr_default'], v)}
                step={0.1}
                min={0.5}
                max={5}
              />
            </div>
          </div>

          {/* Execution Settings */}
          <div style={{ marginBottom: 20 }}>
            <h4 style={{ color: 'var(--accent)', marginBottom: 12, fontSize: '0.95rem' }}>Execution</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 12 }}>
              <SettingControl 
                label="Poll Interval (sec)" 
                value={(tweakedProfile.execution as Record<string, unknown>)?.loop_poll_seconds as number} 
                onAdjust={(d) => adjustValue(['execution', 'loop_poll_seconds'], d, 1, 120, 1)} 
                onChange={(v) => updateTweakedValue(['execution', 'loop_poll_seconds'], v)}
                step={1}
                min={1}
                max={120}
              />
              <SettingControl 
                label="Fast Poll (sec)" 
                value={(tweakedProfile.execution as Record<string, unknown>)?.loop_poll_seconds_fast as number} 
                onAdjust={(d) => adjustValue(['execution', 'loop_poll_seconds_fast'], d, 1, 60, 1)} 
                onChange={(v) => updateTweakedValue(['execution', 'loop_poll_seconds_fast'], v)}
                step={1}
                min={1}
                max={60}
              />
            </div>
          </div>

          {/* Policy Settings */}
          {((tweakedProfile.execution as Record<string, unknown>)?.policies as Record<string, unknown>[])?.map((policy, idx) => (
            <div key={idx} style={{ marginBottom: 16, padding: 12, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
              <h4 style={{ color: 'var(--accent)', marginBottom: 12, fontSize: '0.9rem' }}>
                Policy: {policy.id as string} ({policy.type as string})
              </h4>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 10 }}>
                {policy.type === 'indicator_based' && (
                  <>
                    <SettingControl 
                      label="RSI Period" 
                      value={policy.rsi_period as number} 
                      onAdjust={(d) => {
                        const policies = (tweakedProfile.execution as Record<string, unknown>).policies as Record<string, unknown>[];
                        policies[idx].rsi_period = Math.max(2, Math.min(50, (policy.rsi_period as number) + d));
                        updateTweakedValue(['execution', 'policies'], policies);
                      }}
                      onChange={(v) => {
                        const policies = (tweakedProfile.execution as Record<string, unknown>).policies as Record<string, unknown>[];
                        policies[idx].rsi_period = Math.max(2, Math.min(50, v));
                        updateTweakedValue(['execution', 'policies'], policies);
                      }}
                      step={1}
                      min={2}
                      max={50}
                    />
                    <SettingControl 
                      label="RSI Oversold" 
                      value={policy.rsi_oversold as number} 
                      onAdjust={(d) => {
                        const policies = (tweakedProfile.execution as Record<string, unknown>).policies as Record<string, unknown>[];
                        policies[idx].rsi_oversold = Math.max(10, Math.min(50, (policy.rsi_oversold as number) + d));
                        updateTweakedValue(['execution', 'policies'], policies);
                      }}
                      onChange={(v) => {
                        const policies = (tweakedProfile.execution as Record<string, unknown>).policies as Record<string, unknown>[];
                        policies[idx].rsi_oversold = Math.max(10, Math.min(50, v));
                        updateTweakedValue(['execution', 'policies'], policies);
                      }}
                      step={1}
                      min={10}
                      max={50}
                    />
                    <SettingControl 
                      label="RSI Overbought" 
                      value={policy.rsi_overbought as number} 
                      onAdjust={(d) => {
                        const policies = (tweakedProfile.execution as Record<string, unknown>).policies as Record<string, unknown>[];
                        policies[idx].rsi_overbought = Math.max(50, Math.min(90, (policy.rsi_overbought as number) + d));
                        updateTweakedValue(['execution', 'policies'], policies);
                      }}
                      onChange={(v) => {
                        const policies = (tweakedProfile.execution as Record<string, unknown>).policies as Record<string, unknown>[];
                        policies[idx].rsi_overbought = Math.max(50, Math.min(90, v));
                        updateTweakedValue(['execution', 'policies'], policies);
                      }}
                      step={1}
                      min={50}
                      max={90}
                    />
                    <SettingControl 
                      label="TP Pips" 
                      value={policy.tp_pips as number} 
                      onAdjust={(d) => {
                        const policies = (tweakedProfile.execution as Record<string, unknown>).policies as Record<string, unknown>[];
                        policies[idx].tp_pips = Math.max(1, Math.min(100, (policy.tp_pips as number) + d));
                        updateTweakedValue(['execution', 'policies'], policies);
                      }}
                      onChange={(v) => {
                        const policies = (tweakedProfile.execution as Record<string, unknown>).policies as Record<string, unknown>[];
                        policies[idx].tp_pips = Math.max(1, Math.min(100, v));
                        updateTweakedValue(['execution', 'policies'], policies);
                      }}
                      step={1}
                      min={1}
                      max={100}
                    />
                    <SettingControl 
                      label="SL Pips" 
                      value={policy.sl_pips as number} 
                      onAdjust={(d) => {
                        const policies = (tweakedProfile.execution as Record<string, unknown>).policies as Record<string, unknown>[];
                        policies[idx].sl_pips = Math.max(1, Math.min(100, (policy.sl_pips as number) + d));
                        updateTweakedValue(['execution', 'policies'], policies);
                      }}
                      onChange={(v) => {
                        const policies = (tweakedProfile.execution as Record<string, unknown>).policies as Record<string, unknown>[];
                        policies[idx].sl_pips = Math.max(1, Math.min(100, v));
                        updateTweakedValue(['execution', 'policies'], policies);
                      }}
                      step={1}
                      min={1}
                      max={100}
                    />
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 0' }}>
                      <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Side:</span>
                      <span style={{ 
                        fontWeight: 600, 
                        color: policy.side === 'buy' ? 'var(--success)' : 'var(--danger)' 
                      }}>
                        {(policy.side as string).toUpperCase()}
                      </span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 0' }}>
                      <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Regime:</span>
                      <span style={{ fontWeight: 600 }}>{policy.regime as string}</span>
                    </div>
                  </>
                )}
                {policy.type === 'confirmed_cross' && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 0' }}>
                    <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Setup:</span>
                    <span style={{ fontWeight: 600 }}>{policy.setup_id as string}</span>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>(bi-directional)</span>
                  </div>
                )}
              </div>
            </div>
          ))}

          {error && (
            <div style={{ padding: 12, background: 'rgba(239, 68, 68, 0.1)', borderRadius: 6, marginTop: 16 }}>
              <p style={{ color: 'var(--danger)', margin: 0 }}>Error: {error}</p>
            </div>
          )}
        </div>
      )}

      {/* Navigation Buttons */}
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 20 }}>
        <button
          className="btn btn-secondary"
          onClick={handleBack}
          disabled={step === 1}
        >
          ← Back
        </button>
        <div style={{ display: 'flex', gap: 12 }}>
          {step < maxStep && (
            <button
              className="btn btn-primary"
              onClick={handleNext}
              disabled={!canProceed()}
            >
              {(step === 4 && config.entryStyle !== 'multiple') || (step === 5 && config.entryStyle === 'multiple') ? 'Fine-Tune Settings →' : 'Next →'}
            </button>
          )}
          {((step >= 4 && config.entryStyle !== 'multiple') || (step >= 5 && config.entryStyle === 'multiple')) && (
            <button
              className="btn btn-success"
              onClick={handleSave}
              disabled={saving || violatesProfileLimits}
              style={{ minWidth: 160 }}
            >
              {saving ? 'Saving...' : '✓ Save Configuration'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Presets Page
// ---------------------------------------------------------------------------

interface EditedSettings {
  max_lots: number;
  min_stop_pips: number;
  max_spread_pips: number;
  max_trades_per_day: number;
  max_open_trades: number;
  cooldown_minutes_after_loss: number;
  target_pips: number;
  loop_poll_seconds: number;
  policy_cooldown_minutes: number;
  policy_sl_pips: number;
  // Trial #2 and #3: Swing Level Filter (M15-based)
  swing_level_filter_enabled: boolean;
  swing_danger_zone_pct: number;
  swing_confirmation_bars: number;
  swing_lookback_bars: number;
  // Trial #4: Rolling Danger Zone (M1-based)
  rolling_danger_zone_enabled: boolean;
  rolling_danger_lookback_bars: number;
  rolling_danger_zone_pct: number;
  // Trial #4: RSI Divergence Detection (M5-based, rolling window)
  rsi_divergence_enabled: boolean;
  rsi_divergence_period: number;
  rsi_divergence_lookback_bars: number;
  rsi_divergence_block_minutes: number;
  // Trial #4: EMA Zone Entry Filter (blocks zone entries during EMA compression)
  ema_zone_filter_enabled: boolean;
  ema_zone_filter_lookback_bars: number;
  ema_zone_filter_block_threshold: number;
  // Trial #4: Tiered ATR(14) Filter
  tiered_atr_filter_enabled: boolean;
  tiered_atr_block_below_pips: number;
  tiered_atr_allow_all_max_pips: number;
  tiered_atr_pullback_only_max_pips: number;
  // Trial #4: Daily High/Low Filter
  daily_hl_filter_enabled: boolean;
  daily_hl_buffer_pips: number;
  // Trial #4: Spread-Aware Breakeven
  spread_aware_be_enabled: boolean;
  spread_aware_be_trigger_mode: string;
  spread_aware_be_fixed_trigger_pips: number;
  spread_aware_be_spread_buffer_pips: number;
  spread_aware_be_apply_to_zone_entry: boolean;
  spread_aware_be_apply_to_tiered_pullback: boolean;
  // Trial #4: Zone entry toggle
  zone_entry_enabled: boolean;
  // Trial #4: Per-tier toggles
  tier_ema_periods: number[];
  // Trial #3 EMA overrides
  m5_trend_ema_fast: number | null;
  m5_trend_ema_slow: number | null;
  m1_zone_entry_ema_slow: number | null;
  m1_pullback_cross_ema_slow: number | null;
  // Trial #4 EMA overrides (Zone Entry only - Tiered Pullback uses fixed tiers)
  m3_trend_ema_fast: number | null;
  m3_trend_ema_slow: number | null;
  m1_t4_zone_entry_ema_fast: number | null;
  m1_t4_zone_entry_ema_slow: number | null;
}

function PresetsPage({ profile }: { profile: Profile }) {
  const [presets, setPresets] = useState<api.Preset[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [expandedProscons, setExpandedProscons] = useState<string | null>(null);
  const [preview, setPreview] = useState<Record<string, unknown> | null>(null);
  const [applying, setApplying] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [currentProfile, setCurrentProfile] = useState<Record<string, unknown> | null>(null);
  const [showActiveSettings, setShowActiveSettings] = useState(false);
  const [vwapSessionFilterOn, setVwapSessionFilterOn] = useState(true);
  const [tempSettings, setTempSettings] = useState<api.TempEmaSettings | null>(null);
  const [editedSettings, setEditedSettings] = useState<EditedSettings | null>(null);
  const [applyingSettings, setApplyingSettings] = useState(false);

  // Fetch current profile to get active preset
  const fetchProfile = () => {
    api.getProfile(profile.path).then(setCurrentProfile).catch(console.error);
    api.getTempSettings(profile.name).then(setTempSettings).catch(console.error);
  };

  useEffect(() => {
    api.listPresets().then((p) => {
      setPresets(p);
    }).catch(console.error);
    fetchProfile();
  }, [profile.path, profile.name]);

  useEffect(() => {
    if (selected && selected !== 'custom') {
      api.previewPreset(selected, profile.path)
        .then((r) => setPreview(r))
        .catch(console.error);
    } else {
      setPreview(null);
    }
  }, [selected, profile.path]);

  // Initialize edited settings when View Settings opens
  useEffect(() => {
    if (showActiveSettings && currentProfile) {
      const effectiveRisk = currentProfile.effective_risk as Record<string, unknown> | undefined;
      const risk = currentProfile.risk as Record<string, unknown> | undefined;
      const execution = currentProfile.execution as Record<string, unknown> | undefined;
      const tradeManagement = currentProfile.trade_management as Record<string, unknown> | undefined;
      const targetSettings = tradeManagement?.target as Record<string, unknown> | undefined;
      const policies = execution?.policies as Record<string, unknown>[] | undefined;

      // Find cooldown_minutes, sl_pips, and filter settings from policies
      let policyCooldown = 0;
      let policySlPips = 20;
      // Trial #2/#3: Swing filter
      let swingFilterEnabled = false;
      let swingDangerZonePct = 0.15;
      let swingConfirmationBars = 5;
      let swingLookbackBars = 100;
      // Trial #4: Rolling danger zone
      let rollingDangerZoneEnabled = false;
      let rollingDangerLookbackBars = 100;
      let rollingDangerZonePct = 0.15;
      // Trial #4: RSI Divergence (rolling window comparison)
      let rsiDivergenceEnabled = false;
      let rsiDivergencePeriod = 14;
      let rsiDivergenceLookbackBars = 50;
      let rsiDivergenceBlockMinutes = 5.0;
      // Trial #4: EMA Zone Entry Filter
      let emaZoneFilterEnabled = true;
      let emaZoneFilterLookbackBars = 3;
      let emaZoneFilterBlockThreshold = 0.35;
      // Trial #4: Tiered ATR Filter
      let tieredAtrFilterEnabled = true;
      let tieredAtrBlockBelowPips = 4.0;
      let tieredAtrAllowAllMaxPips = 12.0;
      let tieredAtrPullbackOnlyMaxPips = 15.0;
      // Trial #4: Daily H/L Filter
      let dailyHlFilterEnabled = false;
      let dailyHlBufferPips = 5.0;
      // Trial #4: Spread-Aware BE
      let spreadAwareBeEnabled = false;
      let spreadAweareBeTriggerMode = 'fixed_pips';
      let spreadAwareBeFixedTriggerPips = 5.0;
      let spreadAwareBeSpreadBufferPips = 1.0;
      let spreadAwareBeApplyToZoneEntry = true;
      let spreadAwareBeApplyToTieredPullback = true;
      // Trial #4: Zone entry toggle
      let zoneEntryEnabled = true;
      // Trial #4: Tier EMA periods
      let tierEmaPeriods: number[] = [9, 11, 12, 13, 14, 15, 16, 17];
      if (policies) {
        for (const pol of policies) {
          if ('cooldown_minutes' in pol) {
            policyCooldown = pol.cooldown_minutes as number;
          }
          if ('sl_pips' in pol) {
            policySlPips = pol.sl_pips as number;
          }
          // Trial #2/#3 swing filter
          if ('swing_level_filter_enabled' in pol) {
            swingFilterEnabled = pol.swing_level_filter_enabled as boolean;
            swingDangerZonePct = (pol.swing_danger_zone_pct as number) ?? 0.15;
            swingConfirmationBars = (pol.swing_confirmation_bars as number) ?? 5;
            swingLookbackBars = (pol.swing_lookback_bars as number) ?? 100;
          }
          // Trial #4 rolling danger zone
          if ('rolling_danger_zone_enabled' in pol) {
            rollingDangerZoneEnabled = pol.rolling_danger_zone_enabled as boolean;
            rollingDangerLookbackBars = (pol.rolling_danger_lookback_bars as number) ?? 100;
            rollingDangerZonePct = (pol.rolling_danger_zone_pct as number) ?? 0.15;
          }
          // Trial #4 RSI divergence (M5, rolling window comparison)
          if ('rsi_divergence_enabled' in pol) {
            rsiDivergenceEnabled = pol.rsi_divergence_enabled as boolean;
            rsiDivergencePeriod = (pol.rsi_divergence_period as number) ?? 14;
            rsiDivergenceLookbackBars = (pol.rsi_divergence_lookback_bars as number) ?? 50;
            rsiDivergenceBlockMinutes = (pol.rsi_divergence_block_minutes as number) ?? 5.0;
          }
          if ('ema_zone_filter_enabled' in pol) {
            emaZoneFilterEnabled = pol.ema_zone_filter_enabled as boolean;
            emaZoneFilterLookbackBars = (pol.ema_zone_filter_lookback_bars as number) ?? 3;
            emaZoneFilterBlockThreshold = (pol.ema_zone_filter_block_threshold as number) ?? 0.35;
          }
          if ('tiered_atr_filter_enabled' in pol) {
            tieredAtrFilterEnabled = pol.tiered_atr_filter_enabled as boolean;
            tieredAtrBlockBelowPips = (pol.tiered_atr_block_below_pips as number) ?? 4.0;
            tieredAtrAllowAllMaxPips = (pol.tiered_atr_allow_all_max_pips as number) ?? 12.0;
            tieredAtrPullbackOnlyMaxPips = (pol.tiered_atr_pullback_only_max_pips as number) ?? 15.0;
          }
          if ('daily_hl_filter_enabled' in pol) {
            dailyHlFilterEnabled = pol.daily_hl_filter_enabled as boolean;
            dailyHlBufferPips = (pol.daily_hl_buffer_pips as number) ?? 5.0;
          }
          if ('spread_aware_be_enabled' in pol) {
            spreadAwareBeEnabled = pol.spread_aware_be_enabled as boolean;
            spreadAweareBeTriggerMode = (pol.spread_aware_be_trigger_mode as string) ?? 'fixed_pips';
            spreadAwareBeFixedTriggerPips = (pol.spread_aware_be_fixed_trigger_pips as number) ?? 5.0;
            spreadAwareBeSpreadBufferPips = (pol.spread_aware_be_spread_buffer_pips as number) ?? 1.0;
            spreadAwareBeApplyToZoneEntry = (pol.spread_aware_be_apply_to_zone_entry as boolean) ?? true;
            spreadAwareBeApplyToTieredPullback = (pol.spread_aware_be_apply_to_tiered_pullback as boolean) ?? true;
          }
          if ('zone_entry_enabled' in pol) {
            zoneEntryEnabled = (pol.zone_entry_enabled as boolean) ?? true;
          }
          if ('tier_ema_periods' in pol) {
            tierEmaPeriods = (pol.tier_ema_periods as number[]) ?? [9, 11, 12, 13, 14, 15, 16, 17];
          }
          if (policyCooldown > 0 || policySlPips !== 20) break;
        }
      }

      setEditedSettings({
        max_lots: (effectiveRisk?.max_lots ?? risk?.max_lots ?? 0.01) as number,
        min_stop_pips: (effectiveRisk?.min_stop_pips ?? risk?.min_stop_pips ?? 5) as number,
        max_spread_pips: (effectiveRisk?.max_spread_pips ?? risk?.max_spread_pips ?? 2) as number,
        max_trades_per_day: (effectiveRisk?.max_trades_per_day ?? risk?.max_trades_per_day ?? 10) as number,
        max_open_trades: (effectiveRisk?.max_open_trades ?? risk?.max_open_trades ?? 5) as number,
        cooldown_minutes_after_loss: (effectiveRisk?.cooldown_minutes_after_loss ?? risk?.cooldown_minutes_after_loss ?? 0) as number,
        target_pips: (targetSettings?.pips_default ?? 0.5) as number,
        loop_poll_seconds: (execution?.loop_poll_seconds ?? 5) as number,
        policy_cooldown_minutes: policyCooldown,
        policy_sl_pips: policySlPips,
        swing_level_filter_enabled: swingFilterEnabled,
        swing_danger_zone_pct: swingDangerZonePct,
        swing_confirmation_bars: swingConfirmationBars,
        swing_lookback_bars: swingLookbackBars,
        // Trial #4: Rolling danger zone
        rolling_danger_zone_enabled: rollingDangerZoneEnabled,
        rolling_danger_lookback_bars: rollingDangerLookbackBars,
        rolling_danger_zone_pct: rollingDangerZonePct,
        // Trial #4: RSI divergence (M5, rolling window comparison)
        rsi_divergence_enabled: rsiDivergenceEnabled,
        rsi_divergence_period: rsiDivergencePeriod,
        rsi_divergence_lookback_bars: rsiDivergenceLookbackBars,
        rsi_divergence_block_minutes: rsiDivergenceBlockMinutes,
        // Trial #4: EMA Zone Entry Filter
        ema_zone_filter_enabled: emaZoneFilterEnabled,
        ema_zone_filter_lookback_bars: emaZoneFilterLookbackBars,
        ema_zone_filter_block_threshold: emaZoneFilterBlockThreshold,
        // Trial #4: Tiered ATR Filter
        tiered_atr_filter_enabled: tieredAtrFilterEnabled,
        tiered_atr_block_below_pips: tieredAtrBlockBelowPips,
        tiered_atr_allow_all_max_pips: tieredAtrAllowAllMaxPips,
        tiered_atr_pullback_only_max_pips: tieredAtrPullbackOnlyMaxPips,
        // Trial #4: Daily H/L Filter
        daily_hl_filter_enabled: dailyHlFilterEnabled,
        daily_hl_buffer_pips: dailyHlBufferPips,
        // Trial #4: Spread-Aware BE
        spread_aware_be_enabled: spreadAwareBeEnabled,
        spread_aware_be_trigger_mode: spreadAweareBeTriggerMode,
        spread_aware_be_fixed_trigger_pips: spreadAwareBeFixedTriggerPips,
        spread_aware_be_spread_buffer_pips: spreadAwareBeSpreadBufferPips,
        spread_aware_be_apply_to_zone_entry: spreadAwareBeApplyToZoneEntry,
        spread_aware_be_apply_to_tiered_pullback: spreadAwareBeApplyToTieredPullback,
        // Trial #4: Zone entry toggle
        zone_entry_enabled: zoneEntryEnabled,
        // Trial #4: Tier periods
        tier_ema_periods: tierEmaPeriods,
        // Trial #3 EMA overrides from temp settings
        m5_trend_ema_fast: tempSettings?.m5_trend_ema_fast ?? null,
        m5_trend_ema_slow: tempSettings?.m5_trend_ema_slow ?? null,
        m1_zone_entry_ema_slow: tempSettings?.m1_zone_entry_ema_slow ?? null,
        m1_pullback_cross_ema_slow: tempSettings?.m1_pullback_cross_ema_slow ?? null,
        // Trial #4 EMA overrides from temp settings (Zone Entry only)
        m3_trend_ema_fast: tempSettings?.m3_trend_ema_fast ?? null,
        m3_trend_ema_slow: tempSettings?.m3_trend_ema_slow ?? null,
        m1_t4_zone_entry_ema_fast: tempSettings?.m1_t4_zone_entry_ema_fast ?? null,
        m1_t4_zone_entry_ema_slow: tempSettings?.m1_t4_zone_entry_ema_slow ?? null,
      });
    }
  }, [showActiveSettings, currentProfile, tempSettings]);

  // Handler to apply temporary settings
  const handleApplyTemporarySettings = async () => {
    if (!editedSettings || !currentProfile) return;

    setApplyingSettings(true);
    try {
      const risk = currentProfile.risk as Record<string, unknown> | undefined;
      const effectiveRisk = currentProfile.effective_risk as Record<string, unknown> | undefined;
      const execution = currentProfile.execution as Record<string, unknown> | undefined;
      const tradeManagement = currentProfile.trade_management as Record<string, unknown> | undefined;

      // Cap values by profile.risk limits
      const cappedMaxLots = Math.min(editedSettings.max_lots, (risk?.max_lots ?? editedSettings.max_lots) as number);
      const cappedMaxSpread = Math.min(editedSettings.max_spread_pips, (risk?.max_spread_pips ?? editedSettings.max_spread_pips) as number);
      const cappedMaxTradesPerDay = Math.min(editedSettings.max_trades_per_day, (risk?.max_trades_per_day ?? editedSettings.max_trades_per_day) as number);
      const cappedMaxOpenTrades = Math.min(editedSettings.max_open_trades, (risk?.max_open_trades ?? editedSettings.max_open_trades) as number);
      // Min stop pips: use floor (can't go below profile limit)
      const flooredMinStopPips = Math.max(editedSettings.min_stop_pips, (risk?.min_stop_pips ?? 0) as number);

      // Build updated effective_risk
      const newEffectiveRisk = {
        ...(effectiveRisk || {}),
        max_lots: cappedMaxLots,
        min_stop_pips: flooredMinStopPips,
        max_spread_pips: cappedMaxSpread,
        max_trades_per_day: cappedMaxTradesPerDay,
        max_open_trades: cappedMaxOpenTrades,
        cooldown_minutes_after_loss: Math.max(0, editedSettings.cooldown_minutes_after_loss),
      };

      // Build updated trade_management
      const newTradeManagement = {
        ...(tradeManagement || {}),
        target: {
          ...(tradeManagement?.target as Record<string, unknown> || {}),
          pips_default: editedSettings.target_pips,
        },
      };

      // Build updated execution with updated policies
      const policies = (execution?.policies as Record<string, unknown>[])?.map(pol => {
        const updates: Record<string, unknown> = {};
        if ('cooldown_minutes' in pol) {
          updates.cooldown_minutes = Math.max(0, editedSettings.policy_cooldown_minutes);
        }
        if ('sl_pips' in pol) {
          updates.sl_pips = Math.max(1, editedSettings.policy_sl_pips);
        }
        // Update tp_pips in policy (execution engine uses policy.tp_pips, not trade_management.target.pips_default)
        if ('tp_pips' in pol) {
          updates.tp_pips = Math.max(0.1, editedSettings.target_pips);
        }
        // Update swing filter settings for kt_cg_hybrid and kt_cg_counter_trend_pullback (Trial #2 and #3)
        if (pol.type === 'kt_cg_hybrid' || pol.type === 'kt_cg_counter_trend_pullback') {
          updates.swing_level_filter_enabled = editedSettings.swing_level_filter_enabled;
          updates.swing_danger_zone_pct = Math.max(0.05, Math.min(0.50, editedSettings.swing_danger_zone_pct));
          updates.swing_confirmation_bars = Math.max(2, Math.min(20, editedSettings.swing_confirmation_bars));
          updates.swing_lookback_bars = Math.max(20, Math.min(500, editedSettings.swing_lookback_bars));
        }
        // Update rolling danger zone settings for kt_cg_trial_4 (Trial #4)
        if (pol.type === 'kt_cg_trial_4') {
          updates.rolling_danger_zone_enabled = editedSettings.rolling_danger_zone_enabled;
          updates.rolling_danger_lookback_bars = Math.max(20, Math.min(1500, editedSettings.rolling_danger_lookback_bars));
          updates.rolling_danger_zone_pct = Math.max(0.05, Math.min(0.50, editedSettings.rolling_danger_zone_pct));
          // RSI divergence settings (M5, rolling window comparison)
          updates.rsi_divergence_enabled = editedSettings.rsi_divergence_enabled;
          updates.rsi_divergence_period = Math.max(5, Math.min(50, editedSettings.rsi_divergence_period));
          updates.rsi_divergence_lookback_bars = Math.max(20, Math.min(1500, editedSettings.rsi_divergence_lookback_bars));
          updates.rsi_divergence_block_minutes = Math.max(1, Math.min(30, editedSettings.rsi_divergence_block_minutes));
          // EMA Zone Entry Filter settings
          updates.ema_zone_filter_enabled = editedSettings.ema_zone_filter_enabled;
          updates.ema_zone_filter_lookback_bars = Math.max(2, Math.min(10, editedSettings.ema_zone_filter_lookback_bars));
          updates.ema_zone_filter_block_threshold = Math.max(0.1, Math.min(0.8, editedSettings.ema_zone_filter_block_threshold));
          // Tiered ATR Filter
          updates.tiered_atr_filter_enabled = editedSettings.tiered_atr_filter_enabled;
          updates.tiered_atr_block_below_pips = editedSettings.tiered_atr_block_below_pips;
          updates.tiered_atr_allow_all_max_pips = editedSettings.tiered_atr_allow_all_max_pips;
          updates.tiered_atr_pullback_only_max_pips = editedSettings.tiered_atr_pullback_only_max_pips;
          // Daily H/L Filter
          updates.daily_hl_filter_enabled = editedSettings.daily_hl_filter_enabled;
          updates.daily_hl_buffer_pips = editedSettings.daily_hl_buffer_pips;
          // Spread-Aware BE
          updates.spread_aware_be_enabled = editedSettings.spread_aware_be_enabled;
          updates.spread_aware_be_trigger_mode = editedSettings.spread_aware_be_trigger_mode;
          updates.spread_aware_be_fixed_trigger_pips = editedSettings.spread_aware_be_fixed_trigger_pips;
          updates.spread_aware_be_spread_buffer_pips = editedSettings.spread_aware_be_spread_buffer_pips;
          updates.spread_aware_be_apply_to_zone_entry = editedSettings.spread_aware_be_apply_to_zone_entry;
          updates.spread_aware_be_apply_to_tiered_pullback = editedSettings.spread_aware_be_apply_to_tiered_pullback;
          // Zone entry toggle
          updates.zone_entry_enabled = editedSettings.zone_entry_enabled;
          // Tier periods
          updates.tier_ema_periods = editedSettings.tier_ema_periods;
        }
        return Object.keys(updates).length > 0 ? { ...pol, ...updates } : pol;
      }) || [];

      const newExecution = {
        ...(execution || {}),
        loop_poll_seconds: Math.max(0.25, editedSettings.loop_poll_seconds),
        policies,
      };

      // Update preset name with (customized) suffix if not already present
      let activePresetName = currentProfile.active_preset_name as string || '';
      if (!activePresetName.includes('(customized)')) {
        activePresetName = activePresetName ? `${activePresetName} (customized)` : 'custom (customized)';
      }

      // Build and save new profile
      const newProfile = {
        ...currentProfile,
        active_preset_name: activePresetName,
        effective_risk: newEffectiveRisk,
        trade_management: newTradeManagement,
        execution: newExecution,
      };

      await api.saveProfile(profile.path, newProfile);

      // Save temp EMA settings if using Trial #3 or Trial #4
      const hasKtCgCtp = policies.some(p => p.type === 'kt_cg_counter_trend_pullback');
      const hasKtCgTrial4 = policies.some(p => p.type === 'kt_cg_trial_4');
      if (hasKtCgCtp || hasKtCgTrial4) {
        const settings: any = {};
        if (hasKtCgCtp) {
          settings.m5_trend_ema_fast = editedSettings.m5_trend_ema_fast;
          settings.m5_trend_ema_slow = editedSettings.m5_trend_ema_slow;
          settings.m1_zone_entry_ema_slow = editedSettings.m1_zone_entry_ema_slow;
          settings.m1_pullback_cross_ema_slow = editedSettings.m1_pullback_cross_ema_slow;
        }
        if (hasKtCgTrial4) {
          settings.m3_trend_ema_fast = editedSettings.m3_trend_ema_fast;
          settings.m3_trend_ema_slow = editedSettings.m3_trend_ema_slow;
          settings.m1_t4_zone_entry_ema_fast = editedSettings.m1_t4_zone_entry_ema_fast;
          settings.m1_t4_zone_entry_ema_slow = editedSettings.m1_t4_zone_entry_ema_slow;
        }
        await api.updateTempSettings(profile.name, settings);
      }

      setMessage('Temporary settings applied successfully!');
      await fetchProfile();
      setTimeout(() => setMessage(null), 3000);
    } catch (e: unknown) {
      setMessage(`Error: ${(e as Error).message}`);
    } finally {
      setApplyingSettings(false);
    }
  };

  const handleApply = async () => {
    if (!selected || selected === 'custom') return;
    setApplying(true);
    try {
      const options = selected === 'vwap_trend'
        ? { vwap_session_filter_enabled: vwapSessionFilterOn }
        : undefined;
      await api.applyPreset(selected, profile.path, options);
      setMessage(`Preset "${selected}" applied successfully!`);
      fetchProfile(); // Refresh to show new active preset
      setTimeout(() => setMessage(null), 3000);
    } catch (e: unknown) {
      setMessage(`Error: ${(e as Error).message}`);
    } finally {
      setApplying(false);
    }
  };

  const toggleProscons = (presetId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setExpandedProscons(expandedProscons === presetId ? null : presetId);
  };

  const selectedPreset = presets.find((p) => p.id === selected);

  const activePresetName = currentProfile?.active_preset_name as string | undefined;
  const risk = currentProfile?.risk as Record<string, unknown> | undefined;
  const effectiveRisk = currentProfile?.effective_risk as Record<string, unknown> | undefined;
  const execution = currentProfile?.execution as Record<string, unknown> | undefined;

  return (
    <div>
      <h2 className="page-title">Presets</h2>
      <p className="mb-4" style={{ color: 'var(--text-secondary)' }}>
        Select a preset to preview changes, then apply to your profile. Click "Pros & Cons" to see details.
      </p>

      {/* Active Preset Section */}
      {activePresetName && (
        <div className="card mb-4" style={{ 
          borderColor: 'var(--accent)', 
          borderWidth: 2,
          background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, var(--bg-secondary) 100%)'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 4 }}>
                ACTIVE PRESET
              </div>
              <div style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--accent)' }}>
                {activePresetName}
              </div>
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <button
                className="btn btn-secondary"
                onClick={() => setShowActiveSettings(!showActiveSettings)}
                style={{ fontSize: '0.8rem' }}
              >
                {showActiveSettings ? 'Hide Settings' : 'View Settings'}
              </button>
              {showActiveSettings && (
                <button
                  className="btn btn-primary"
                  onClick={handleApplyTemporarySettings}
                  disabled={applyingSettings || !editedSettings}
                  style={{ fontSize: '0.8rem' }}
                >
                  {applyingSettings ? 'Applying...' : 'Apply Temporary Settings'}
                </button>
              )}
            </div>
          </div>
          
          {showActiveSettings && risk && (
            <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid var(--border)' }}>
              {effectiveRisk && (
                <div style={{ marginBottom: 12, fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  Effective (used when running): preset risk capped by Profile Editor limits.
                </div>
              )}

              {/* Display Current Temp Overrides */}
              {tempSettings && (
                <div style={{ marginBottom: 16, padding: 12, background: 'var(--bg-tertiary)', borderRadius: 6, border: '1px solid var(--border)' }}>
                  <div style={{ fontSize: '0.8rem', color: 'var(--success)', marginBottom: 8, fontWeight: 600 }}>
                    ✓ Current Temporary Overrides Saved:
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 8, fontSize: '0.75rem' }}>
                    {tempSettings.m5_trend_ema_fast && (
                      <div style={{ color: 'var(--text-secondary)' }}>Trial #3: M5 Trend EMA: {tempSettings.m5_trend_ema_fast}/{tempSettings.m5_trend_ema_slow}</div>
                    )}
                    {tempSettings.m1_zone_entry_ema_slow && (
                      <div style={{ color: 'var(--text-secondary)' }}>Trial #3: M1 Zone EMA: 9/{tempSettings.m1_zone_entry_ema_slow}</div>
                    )}
                    {tempSettings.m1_pullback_cross_ema_slow && (
                      <div style={{ color: 'var(--text-secondary)' }}>Trial #3: M1 Pullback EMA: 9/{tempSettings.m1_pullback_cross_ema_slow}</div>
                    )}
                    {tempSettings.m3_trend_ema_fast && (
                      <div style={{ color: 'var(--text-secondary)' }}>Trial #4: M3 Trend EMA: {tempSettings.m3_trend_ema_fast}/{tempSettings.m3_trend_ema_slow}</div>
                    )}
                    {tempSettings.m1_t4_zone_entry_ema_fast && (
                      <div style={{ color: 'var(--text-secondary)' }}>Trial #4: M1 Zone EMA: {tempSettings.m1_t4_zone_entry_ema_fast}/{tempSettings.m1_t4_zone_entry_ema_slow}</div>
                    )}
                  </div>
                  {!tempSettings.m5_trend_ema_fast && !tempSettings.m3_trend_ema_fast && (
                    <div style={{ color: 'var(--text-secondary)', fontSize: '0.75rem' }}>
                      (No temp overrides currently saved)
                    </div>
                  )}
                </div>
              )}

              {editedSettings && (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 12 }}>
                  <div style={{ padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 4 }}>Max Lots (effective)</div>
                    <input
                      type="number"
                      step="0.01"
                      min="0.01"
                      value={editedSettings.max_lots}
                      onChange={(e) => setEditedSettings({ ...editedSettings, max_lots: parseFloat(e.target.value) || 0.01 })}
                      style={{ width: '100%', padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                    />
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 2 }}>Max: {risk.max_lots as number}</div>
                  </div>
                  <div style={{ padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 4 }}>Min Stop Pips</div>
                    <input
                      type="number"
                      step="0.5"
                      min="0"
                      value={editedSettings.min_stop_pips}
                      onChange={(e) => setEditedSettings({ ...editedSettings, min_stop_pips: parseFloat(e.target.value) || 0 })}
                      style={{ width: '100%', padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                    />
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 2 }}>Floor: {risk.min_stop_pips as number}</div>
                  </div>
                  <div style={{ padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 4 }}>Max Spread (pips)</div>
                    <input
                      type="number"
                      step="0.1"
                      min="0.1"
                      value={editedSettings.max_spread_pips}
                      onChange={(e) => setEditedSettings({ ...editedSettings, max_spread_pips: parseFloat(e.target.value) || 0.1 })}
                      style={{ width: '100%', padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                    />
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 2 }}>Max: {risk.max_spread_pips as number}</div>
                  </div>
                  <div style={{ padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 4 }}>Max Trades/Day</div>
                    <input
                      type="number"
                      step="1"
                      min="1"
                      value={editedSettings.max_trades_per_day}
                      onChange={(e) => setEditedSettings({ ...editedSettings, max_trades_per_day: parseInt(e.target.value) || 1 })}
                      style={{ width: '100%', padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                    />
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 2 }}>Max: {risk.max_trades_per_day as number}</div>
                  </div>
                  <div style={{ padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 4 }}>Max Open Trades</div>
                    <input
                      type="number"
                      step="1"
                      min="1"
                      value={editedSettings.max_open_trades}
                      onChange={(e) => setEditedSettings({ ...editedSettings, max_open_trades: parseInt(e.target.value) || 1 })}
                      style={{ width: '100%', padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                    />
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 2 }}>Max: {risk.max_open_trades as number}</div>
                  </div>
                  <div style={{ padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 4 }}>Cooldown After Loss (min)</div>
                    <input
                      type="number"
                      step="1"
                      min="0"
                      value={editedSettings.cooldown_minutes_after_loss}
                      onChange={(e) => setEditedSettings({ ...editedSettings, cooldown_minutes_after_loss: parseInt(e.target.value) || 0 })}
                      style={{ width: '100%', padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                    />
                  </div>
                  <div style={{ padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 4 }}>Target Pips</div>
                    <input
                      type="number"
                      step="0.1"
                      min="0.1"
                      value={editedSettings.target_pips}
                      onChange={(e) => setEditedSettings({ ...editedSettings, target_pips: parseFloat(e.target.value) || 0.1 })}
                      style={{ width: '100%', padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                    />
                  </div>
                  <div style={{ padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 4 }}>Stop Loss Pips</div>
                    <input
                      type="number"
                      step="1"
                      min="1"
                      max="500"
                      value={editedSettings.policy_sl_pips}
                      onChange={(e) => setEditedSettings({ ...editedSettings, policy_sl_pips: parseFloat(e.target.value) || 20 })}
                      style={{ width: '100%', padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                    />
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 2 }}>Must be ≥ Min Stop Pips ({editedSettings.min_stop_pips})</div>
                  </div>
                  <div style={{ padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 4 }}>Poll Interval (s)</div>
                    <input
                      type="number"
                      step="0.25"
                      min="0.25"
                      value={editedSettings.loop_poll_seconds}
                      onChange={(e) => setEditedSettings({ ...editedSettings, loop_poll_seconds: parseFloat(e.target.value) || 0.25 })}
                      style={{ width: '100%', padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                    />
                  </div>
                  <div style={{ padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 4 }}>Cooldown After Trade (min)</div>
                    <input
                      type="number"
                      step="0.5"
                      min="0"
                      value={editedSettings.policy_cooldown_minutes}
                      onChange={(e) => setEditedSettings({ ...editedSettings, policy_cooldown_minutes: parseFloat(e.target.value) || 0 })}
                      style={{ width: '100%', padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                    />
                  </div>
                </div>
              )}
              {/* Swing Level Filter Settings (for kt_cg_hybrid and kt_cg_counter_trend_pullback - Trial #2 and #3) */}
              {editedSettings && (execution?.policies as Record<string, unknown>[])?.some(pol => pol.type === 'kt_cg_hybrid' || pol.type === 'kt_cg_counter_trend_pullback') && (
                <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid var(--border)' }}>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 12 }}>
                    Swing Level Filter (blocks trades near M15 swing highs/lows)
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 12 }}>
                    <div style={{ padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 4 }}>Swing Filter Enabled</div>
                      <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                        <input
                          type="checkbox"
                          checked={editedSettings.swing_level_filter_enabled}
                          onChange={(e) => setEditedSettings({ ...editedSettings, swing_level_filter_enabled: e.target.checked })}
                          style={{ width: 18, height: 18, cursor: 'pointer' }}
                        />
                        <span style={{ fontWeight: 600, color: editedSettings.swing_level_filter_enabled ? 'var(--success)' : 'var(--text-secondary)' }}>
                          {editedSettings.swing_level_filter_enabled ? 'ON' : 'OFF'}
                        </span>
                      </label>
                    </div>
                    <div style={{ padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 4 }}>Danger Zone %</div>
                      <input
                        type="number"
                        step="0.01"
                        min="0.05"
                        max="0.50"
                        value={editedSettings.swing_danger_zone_pct}
                        onChange={(e) => setEditedSettings({ ...editedSettings, swing_danger_zone_pct: parseFloat(e.target.value) || 0.15 })}
                        style={{ width: '100%', padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                      />
                      <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 2 }}>0.15 = 15% of range</div>
                    </div>
                    <div style={{ padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 4 }}>Confirmation Bars</div>
                      <input
                        type="number"
                        step="1"
                        min="2"
                        max="20"
                        value={editedSettings.swing_confirmation_bars}
                        onChange={(e) => setEditedSettings({ ...editedSettings, swing_confirmation_bars: parseInt(e.target.value) || 5 })}
                        style={{ width: '100%', padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                      />
                      <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 2 }}>Bars before/after swing</div>
                    </div>
                    <div style={{ padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 4 }}>Lookback Bars</div>
                      <input
                        type="number"
                        step="10"
                        min="20"
                        max="500"
                        value={editedSettings.swing_lookback_bars}
                        onChange={(e) => setEditedSettings({ ...editedSettings, swing_lookback_bars: parseInt(e.target.value) || 100 })}
                        style={{ width: '100%', padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                      />
                      <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 2 }}>M15 bars to scan</div>
                    </div>
                  </div>
                </div>
              )}
              {/* Rolling Danger Zone Settings (for kt_cg_trial_4 - Trial #4) */}
              {editedSettings && (execution?.policies as Record<string, unknown>[])?.some(pol => pol.type === 'kt_cg_trial_4') && (
                <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid var(--border)' }}>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 12 }}>
                    Rolling Danger Zone (blocks trades near M1 rolling high/low extremes)
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 12 }}>
                    <div style={{ padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 4 }}>Danger Zone Enabled</div>
                      <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                        <input
                          type="checkbox"
                          checked={editedSettings.rolling_danger_zone_enabled}
                          onChange={(e) => setEditedSettings({ ...editedSettings, rolling_danger_zone_enabled: e.target.checked })}
                          style={{ width: 18, height: 18, cursor: 'pointer' }}
                        />
                        <span style={{ fontWeight: 600, color: editedSettings.rolling_danger_zone_enabled ? 'var(--success)' : 'var(--text-secondary)' }}>
                          {editedSettings.rolling_danger_zone_enabled ? 'ON' : 'OFF'}
                        </span>
                      </label>
                    </div>
                    <div style={{ padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 4 }}>Danger Zone %</div>
                      <input
                        type="number"
                        step="0.01"
                        min="0.05"
                        max="0.50"
                        value={editedSettings.rolling_danger_zone_pct}
                        onChange={(e) => setEditedSettings({ ...editedSettings, rolling_danger_zone_pct: parseFloat(e.target.value) || 0.15 })}
                        style={{ width: '100%', padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                      />
                      <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 2 }}>0.15 = top/bottom 15%</div>
                    </div>
                    <div style={{ padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 4 }}>Lookback Bars</div>
                      <input
                        type="number"
                        step="10"
                        min="20"
                        max="1500"
                        value={editedSettings.rolling_danger_lookback_bars}
                        onChange={(e) => setEditedSettings({ ...editedSettings, rolling_danger_lookback_bars: parseInt(e.target.value) || 100 })}
                        style={{ width: '100%', padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                      />
                      <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 2 }}>M1 bars for high/low</div>
                    </div>
                  </div>
                  <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 8 }}>
                    Upper zone (top {(editedSettings.rolling_danger_zone_pct * 100).toFixed(0)}%) blocks BUY • Lower zone (bottom {(editedSettings.rolling_danger_zone_pct * 100).toFixed(0)}%) blocks SELL
                  </div>
                  {/* RSI Divergence Detection (M5) */}
                  <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid var(--border)' }}>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 8 }}>
                      RSI Divergence Detection (M5, rolling window)
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: 12 }}>
                      <div style={{ padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 4 }}>Divergence Enabled</div>
                        <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                          <input
                            type="checkbox"
                            checked={editedSettings.rsi_divergence_enabled}
                            onChange={(e) => setEditedSettings({ ...editedSettings, rsi_divergence_enabled: e.target.checked })}
                            style={{ width: 18, height: 18, cursor: 'pointer' }}
                          />
                          <span style={{ fontWeight: 600, color: editedSettings.rsi_divergence_enabled ? 'var(--success)' : 'var(--text-secondary)' }}>
                            {editedSettings.rsi_divergence_enabled ? 'ON' : 'OFF'}
                          </span>
                        </label>
                      </div>
                      <div style={{ padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 4 }}>RSI Period</div>
                        <input
                          type="number"
                          step="1"
                          min="5"
                          max="50"
                          value={editedSettings.rsi_divergence_period}
                          onChange={(e) => setEditedSettings({ ...editedSettings, rsi_divergence_period: parseInt(e.target.value) || 14 })}
                          style={{ width: '100%', padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                        />
                        <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 2 }}>Default: 14</div>
                      </div>
                      <div style={{ padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 4 }}>Lookback Bars</div>
                        <input
                          type="number"
                          step="10"
                          min="20"
                          max="1500"
                          value={editedSettings.rsi_divergence_lookback_bars}
                          onChange={(e) => setEditedSettings({ ...editedSettings, rsi_divergence_lookback_bars: parseInt(e.target.value) || 50 })}
                          style={{ width: '100%', padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                        />
                        <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 2 }}>M5 bars (split into ref/recent halves)</div>
                      </div>
                      <div style={{ padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 4 }}>Block Minutes</div>
                        <input
                          type="number"
                          step="0.5"
                          min="1"
                          max="30"
                          value={editedSettings.rsi_divergence_block_minutes}
                          onChange={(e) => setEditedSettings({ ...editedSettings, rsi_divergence_block_minutes: parseFloat(e.target.value) || 5 })}
                          style={{ width: '100%', padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                        />
                        <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 2 }}>Duration of entry block</div>
                      </div>
                    </div>
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 8 }}>
                      BULL + bearish divergence → blocks BUY • BEAR + bullish divergence → blocks SELL
                    </div>
                  </div>
                  {/* EMA Zone Entry Filter */}
                  <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid var(--border)' }}>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 8 }}>
                      EMA Zone Entry Filter (M1 EMA 9 vs 17)
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: 12 }}>
                      <div style={{ padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                        <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                          <input
                            type="checkbox"
                            checked={editedSettings.ema_zone_filter_enabled}
                            onChange={(e) => setEditedSettings({ ...editedSettings, ema_zone_filter_enabled: e.target.checked })}
                            style={{ width: 18, height: 18, cursor: 'pointer' }}
                          />
                          <span style={{ fontWeight: 600, color: editedSettings.ema_zone_filter_enabled ? 'var(--success)' : 'var(--text-secondary)' }}>
                            {editedSettings.ema_zone_filter_enabled ? 'ON' : 'OFF'}
                          </span>
                        </label>
                      </div>
                      <div style={{ padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 4 }}>Lookback Bars</div>
                        <input
                          type="number"
                          step="1"
                          min="2"
                          max="10"
                          value={editedSettings.ema_zone_filter_lookback_bars}
                          onChange={(e) => setEditedSettings({ ...editedSettings, ema_zone_filter_lookback_bars: parseInt(e.target.value) || 3 })}
                          style={{ width: '100%', padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                        />
                        <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 2 }}>Bars for slope/direction</div>
                      </div>
                      <div style={{ padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 4 }}>Block Threshold</div>
                        <input
                          type="number"
                          step="0.05"
                          min="0.1"
                          max="0.8"
                          value={editedSettings.ema_zone_filter_block_threshold}
                          onChange={(e) => setEditedSettings({ ...editedSettings, ema_zone_filter_block_threshold: parseFloat(e.target.value) || 0.35 })}
                          style={{ width: '100%', padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                        />
                        <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 2 }}>Block if score below</div>
                      </div>
                    </div>
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 8 }}>
                      Blocks zone entries during EMA compression. Tiered pullback unaffected.
                    </div>
                  </div>
                  {/* Tiered ATR(14) Filter */}
                  <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid var(--border)' }}>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 8 }}>
                      Tiered ATR(14) Filter
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: 12 }}>
                      <div style={{ padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                        <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                          <input
                            type="checkbox"
                            checked={editedSettings.tiered_atr_filter_enabled}
                            onChange={(e) => setEditedSettings({ ...editedSettings, tiered_atr_filter_enabled: e.target.checked })}
                            style={{ width: 18, height: 18, cursor: 'pointer' }}
                          />
                          <span style={{ fontWeight: 600, color: editedSettings.tiered_atr_filter_enabled ? 'var(--success)' : 'var(--text-secondary)' }}>
                            {editedSettings.tiered_atr_filter_enabled ? 'ON' : 'OFF'}
                          </span>
                        </label>
                      </div>
                      <div style={{ padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 4 }}>Block Below (pips)</div>
                        <input
                          type="number"
                          step="0.5"
                          min="0"
                          max="20"
                          value={editedSettings.tiered_atr_block_below_pips}
                          onChange={(e) => setEditedSettings({ ...editedSettings, tiered_atr_block_below_pips: parseFloat(e.target.value) || 4 })}
                          style={{ width: '100%', padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                        />
                        <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 2 }}>{'< this: ALL blocked'}</div>
                      </div>
                      <div style={{ padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 4 }}>Allow All Max (pips)</div>
                        <input
                          type="number"
                          step="0.5"
                          min="0"
                          max="30"
                          value={editedSettings.tiered_atr_allow_all_max_pips}
                          onChange={(e) => setEditedSettings({ ...editedSettings, tiered_atr_allow_all_max_pips: parseFloat(e.target.value) || 12 })}
                          style={{ width: '100%', padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                        />
                        <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 2 }}>{'< this: allow ALL'}</div>
                      </div>
                      <div style={{ padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 4 }}>Pullback Only Max (pips)</div>
                        <input
                          type="number"
                          step="0.5"
                          min="0"
                          max="40"
                          value={editedSettings.tiered_atr_pullback_only_max_pips}
                          onChange={(e) => setEditedSettings({ ...editedSettings, tiered_atr_pullback_only_max_pips: parseFloat(e.target.value) || 15 })}
                          style={{ width: '100%', padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                        />
                        <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 2 }}>{'< this: pullback only'}</div>
                      </div>
                    </div>
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 8 }}>
                      {'< '}{ editedSettings.tiered_atr_block_below_pips}p: block ALL •
                      {editedSettings.tiered_atr_block_below_pips}-{editedSettings.tiered_atr_allow_all_max_pips}p: allow ALL •
                      {editedSettings.tiered_atr_allow_all_max_pips}-{editedSettings.tiered_atr_pullback_only_max_pips}p: pullback only •
                      {'> '}{editedSettings.tiered_atr_pullback_only_max_pips}p: block ALL
                    </div>
                  </div>
                  {/* Daily High/Low Filter */}
                  <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid var(--border)' }}>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 8 }}>
                      Daily High/Low Filter (zone entry only)
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: 12 }}>
                      <div style={{ padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                        <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                          <input
                            type="checkbox"
                            checked={editedSettings.daily_hl_filter_enabled}
                            onChange={(e) => setEditedSettings({ ...editedSettings, daily_hl_filter_enabled: e.target.checked })}
                            style={{ width: 18, height: 18, cursor: 'pointer' }}
                          />
                          <span style={{ fontWeight: 600, color: editedSettings.daily_hl_filter_enabled ? 'var(--success)' : 'var(--text-secondary)' }}>
                            {editedSettings.daily_hl_filter_enabled ? 'ON' : 'OFF'}
                          </span>
                        </label>
                      </div>
                      <div style={{ padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 4 }}>Buffer (pips)</div>
                        <input
                          type="number"
                          step="0.5"
                          min="1"
                          max="20"
                          value={editedSettings.daily_hl_buffer_pips}
                          onChange={(e) => setEditedSettings({ ...editedSettings, daily_hl_buffer_pips: parseFloat(e.target.value) || 5 })}
                          style={{ width: '100%', padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                        />
                        <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 2 }}>Distance from daily H/L</div>
                      </div>
                    </div>
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 8 }}>
                      Blocks BUY within {editedSettings.daily_hl_buffer_pips}p of daily high • Blocks SELL within {editedSettings.daily_hl_buffer_pips}p of daily low
                    </div>
                  </div>
                  {/* Spread-Aware Breakeven */}
                  <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid var(--border)' }}>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 8 }}>
                      Spread-Aware Breakeven Stop Loss
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: 12 }}>
                      <div style={{ padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                        <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                          <input
                            type="checkbox"
                            checked={editedSettings.spread_aware_be_enabled}
                            onChange={(e) => setEditedSettings({ ...editedSettings, spread_aware_be_enabled: e.target.checked })}
                            style={{ width: 18, height: 18, cursor: 'pointer' }}
                          />
                          <span style={{ fontWeight: 600, color: editedSettings.spread_aware_be_enabled ? 'var(--success)' : 'var(--text-secondary)' }}>
                            {editedSettings.spread_aware_be_enabled ? 'ON' : 'OFF'}
                          </span>
                        </label>
                      </div>
                      <div style={{ padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 4 }}>Trigger Mode</div>
                        <select
                          value={editedSettings.spread_aware_be_trigger_mode}
                          onChange={(e) => setEditedSettings({ ...editedSettings, spread_aware_be_trigger_mode: e.target.value })}
                          style={{ width: '100%', padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                        >
                          <option value="fixed_pips">Fixed Pips</option>
                          <option value="spread_relative">Spread + Buffer</option>
                        </select>
                      </div>
                      <div style={{ padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 4 }}>Fixed Trigger (pips)</div>
                        <input
                          type="number"
                          step="0.5"
                          min="1"
                          max="30"
                          value={editedSettings.spread_aware_be_fixed_trigger_pips}
                          onChange={(e) => setEditedSettings({ ...editedSettings, spread_aware_be_fixed_trigger_pips: parseFloat(e.target.value) || 5 })}
                          style={{ width: '100%', padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                        />
                      </div>
                      <div style={{ padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 4 }}>Spread Buffer (pips)</div>
                        <input
                          type="number"
                          step="0.5"
                          min="0"
                          max="10"
                          value={editedSettings.spread_aware_be_spread_buffer_pips}
                          onChange={(e) => setEditedSettings({ ...editedSettings, spread_aware_be_spread_buffer_pips: parseFloat(e.target.value) || 1 })}
                          style={{ width: '100%', padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                        />
                      </div>
                    </div>
                    <div style={{ display: 'flex', gap: 16, marginTop: 8 }}>
                      <label style={{ display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer', fontSize: '0.7rem' }}>
                        <input
                          type="checkbox"
                          checked={editedSettings.spread_aware_be_apply_to_zone_entry}
                          onChange={(e) => setEditedSettings({ ...editedSettings, spread_aware_be_apply_to_zone_entry: e.target.checked })}
                          style={{ width: 14, height: 14, cursor: 'pointer' }}
                        />
                        Apply to Zone Entry
                      </label>
                      <label style={{ display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer', fontSize: '0.7rem' }}>
                        <input
                          type="checkbox"
                          checked={editedSettings.spread_aware_be_apply_to_tiered_pullback}
                          onChange={(e) => setEditedSettings({ ...editedSettings, spread_aware_be_apply_to_tiered_pullback: e.target.checked })}
                          style={{ width: 14, height: 14, cursor: 'pointer' }}
                        />
                        Apply to Tiered Pullback
                      </label>
                    </div>
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 4 }}>
                      SL = entry ± current spread. Ratchets favorably (never moves back toward entry).
                    </div>
                  </div>
                </div>
              )}
              {/* EMA Override Settings (for kt_cg_counter_trend_pullback / Trial #3) */}
              {editedSettings && (execution?.policies as Record<string, unknown>[])?.some(pol => pol.type === 'kt_cg_counter_trend_pullback') && (
                <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid var(--border)' }}>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 12 }}>
                    EMA Override Settings (Trial #3)
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
                    <div style={{ padding: 12, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 8 }}>
                        M5 Trend – fast / slow EMAs (default 9 / 21)
                      </div>
                      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                        <input
                          type="number"
                          step="1"
                          min="1"
                          placeholder="9"
                          value={editedSettings.m5_trend_ema_fast ?? ''}
                          onChange={(e) => setEditedSettings({ ...editedSettings, m5_trend_ema_fast: e.target.value ? parseInt(e.target.value) : null })}
                          style={{ flex: 1, padding: '6px 10px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                        />
                        <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>/</span>
                        <input
                          type="number"
                          step="1"
                          min="1"
                          placeholder="21"
                          value={editedSettings.m5_trend_ema_slow ?? ''}
                          onChange={(e) => setEditedSettings({ ...editedSettings, m5_trend_ema_slow: e.target.value ? parseInt(e.target.value) : null })}
                          style={{ flex: 1, padding: '6px 10px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                        />
                      </div>
                      <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 4 }}>
                        Determines trend: fast {'>'} slow = BULL
                      </div>
                    </div>
                    <div style={{ padding: 12, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 8 }}>
                        M1 Zone Entry – slow EMA (default 13)
                      </div>
                      <input
                        type="number"
                        step="1"
                        min="1"
                        placeholder="13"
                        value={editedSettings.m1_zone_entry_ema_slow ?? ''}
                        onChange={(e) => setEditedSettings({ ...editedSettings, m1_zone_entry_ema_slow: e.target.value ? parseInt(e.target.value) : null })}
                        style={{ width: '100%', padding: '6px 10px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                      />
                      <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 4 }}>
                        BUY zone when M1 EMA 9 {'>'} this EMA
                      </div>
                    </div>
                    <div style={{ padding: 12, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 8 }}>
                        M1 Pullback Cross – slow EMA (default 15)
                      </div>
                      <input
                        type="number"
                        step="1"
                        min="1"
                        placeholder="15"
                        value={editedSettings.m1_pullback_cross_ema_slow ?? ''}
                        onChange={(e) => setEditedSettings({ ...editedSettings, m1_pullback_cross_ema_slow: e.target.value ? parseInt(e.target.value) : null })}
                        style={{ width: '100%', padding: '6px 10px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                      />
                      <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 4 }}>
                        Trigger when M1 EMA 9 crosses this EMA
                      </div>
                    </div>
                  </div>
                  <button
                    className="btn btn-secondary mt-3"
                    onClick={() => setEditedSettings({
                      ...editedSettings,
                      m5_trend_ema_fast: null,
                      m5_trend_ema_slow: null,
                      m1_zone_entry_ema_slow: null,
                      m1_pullback_cross_ema_slow: null,
                    })}
                    style={{ fontSize: '0.8rem' }}
                  >
                    Reset EMAs to Defaults
                  </button>
                </div>
              )}
              {/* EMA Override Settings (for kt_cg_trial_4 / Trial #4) */}
              {editedSettings && (execution?.policies as Record<string, unknown>[])?.some(pol => pol.type === 'kt_cg_trial_4') && (
                <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid var(--border)' }}>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 12 }}>
                    EMA Override Settings (Trial #4)
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
                    <div style={{ padding: 12, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 8 }}>
                        M3 Trend – fast / slow EMAs (default 9 / 21)
                      </div>
                      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                        <input
                          type="number"
                          step="1"
                          min="1"
                          placeholder="9"
                          value={editedSettings.m3_trend_ema_fast ?? ''}
                          onChange={(e) => setEditedSettings({ ...editedSettings, m3_trend_ema_fast: e.target.value ? parseInt(e.target.value) : null })}
                          style={{ flex: 1, padding: '6px 10px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                        />
                        <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>/</span>
                        <input
                          type="number"
                          step="1"
                          min="1"
                          placeholder="21"
                          value={editedSettings.m3_trend_ema_slow ?? ''}
                          onChange={(e) => setEditedSettings({ ...editedSettings, m3_trend_ema_slow: e.target.value ? parseInt(e.target.value) : null })}
                          style={{ flex: 1, padding: '6px 10px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                        />
                      </div>
                      <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 4 }}>
                        M3 Trend: fast {'>'} slow = BULL
                      </div>
                    </div>
                    <div style={{ padding: 12, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 8 }}>
                        M1 Zone Entry – fast / slow EMAs (default 5 / 9)
                      </div>
                      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                        <input
                          type="number"
                          step="1"
                          min="1"
                          placeholder="5"
                          value={editedSettings.m1_t4_zone_entry_ema_fast ?? ''}
                          onChange={(e) => setEditedSettings({ ...editedSettings, m1_t4_zone_entry_ema_fast: e.target.value ? parseInt(e.target.value) : null })}
                          style={{ flex: 1, padding: '6px 10px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                        />
                        <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>/</span>
                        <input
                          type="number"
                          step="1"
                          min="1"
                          placeholder="9"
                          value={editedSettings.m1_t4_zone_entry_ema_slow ?? ''}
                          onChange={(e) => setEditedSettings({ ...editedSettings, m1_t4_zone_entry_ema_slow: e.target.value ? parseInt(e.target.value) : null })}
                          style={{ flex: 1, padding: '6px 10px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                        />
                      </div>
                      <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 4 }}>
                        Zone Entry: fast {'>'} slow = BUY
                      </div>
                    </div>
                    <div style={{ padding: 12, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                      <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', marginBottom: 8 }}>
                        <input type="checkbox" checked={editedSettings.zone_entry_enabled} onChange={(e) => setEditedSettings({ ...editedSettings, zone_entry_enabled: e.target.checked })} style={{ width: 16, height: 16, cursor: 'pointer' }} />
                        <span style={{ fontSize: '0.75rem', fontWeight: 600 }}>Zone Entry Enabled</span>
                      </label>
                      <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)' }}>
                        When OFF, only tiered pullback entries are active (zone entry trades are blocked).
                      </div>
                    </div>
                    <div style={{ padding: 12, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 8 }}>
                        Tiered Pullback — Tiers 9–17
                      </div>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                        {[9, 11, 12, 13, 14, 15, 16, 17].map(tier => (
                          <label key={tier} style={{ display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer', padding: '2px 6px', background: editedSettings.tier_ema_periods.includes(tier) ? 'var(--success)' : 'var(--bg-secondary)', borderRadius: 4, fontSize: '0.7rem', fontWeight: 600, color: editedSettings.tier_ema_periods.includes(tier) ? '#fff' : 'var(--text-secondary)' }}>
                            <input
                              type="checkbox"
                              checked={editedSettings.tier_ema_periods.includes(tier)}
                              onChange={(e) => {
                                const periods = e.target.checked
                                  ? [...editedSettings.tier_ema_periods, tier].sort((a, b) => a - b)
                                  : editedSettings.tier_ema_periods.filter(t => t !== tier);
                                setEditedSettings({ ...editedSettings, tier_ema_periods: periods });
                              }}
                              style={{ width: 14, height: 14, cursor: 'pointer' }}
                            />
                            EMA {tier}
                          </label>
                        ))}
                      </div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 8, marginTop: 12 }}>
                        Tiered Pullback — Tiers 18–30
                      </div>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                        {[18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30].map(tier => (
                          <label key={tier} style={{ display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer', padding: '2px 6px', background: editedSettings.tier_ema_periods.includes(tier) ? 'var(--success)' : 'var(--bg-secondary)', borderRadius: 4, fontSize: '0.7rem', fontWeight: 600, color: editedSettings.tier_ema_periods.includes(tier) ? '#fff' : 'var(--text-secondary)' }}>
                            <input
                              type="checkbox"
                              checked={editedSettings.tier_ema_periods.includes(tier)}
                              onChange={(e) => {
                                const periods = e.target.checked
                                  ? [...editedSettings.tier_ema_periods, tier].sort((a, b) => a - b)
                                  : editedSettings.tier_ema_periods.filter(t => t !== tier);
                                setEditedSettings({ ...editedSettings, tier_ema_periods: periods });
                              }}
                              style={{ width: 14, height: 14, cursor: 'pointer' }}
                            />
                            EMA {tier}
                          </label>
                        ))}
                      </div>
                      <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 4 }}>
                        When price touches M1 EMA tiers, triggers entry. Each tier fires once per touch and resets when price moves away.
                      </div>
                    </div>
                  </div>
                  <button
                    className="btn btn-secondary mt-3"
                    onClick={() => setEditedSettings({
                      ...editedSettings,
                      m3_trend_ema_fast: null,
                      m3_trend_ema_slow: null,
                      m1_t4_zone_entry_ema_fast: null,
                      m1_t4_zone_entry_ema_slow: null,
                    })}
                    style={{ fontSize: '0.8rem' }}
                  >
                    Reset Trial #4 EMAs to Defaults
                  </button>
                </div>
              )}
              {(execution?.policies as Record<string, unknown>[])?.length > 0 && (
                <div style={{ marginTop: 12 }}>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 8 }}>Active Policies:</div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                    {(execution?.policies as Record<string, unknown>[])?.map((pol, i) => (
                      <span key={i} style={{ 
                        padding: '4px 8px', 
                        background: 'var(--bg-tertiary)', 
                        borderRadius: 4,
                        fontSize: '0.75rem',
                        fontWeight: 600
                      }}>
                        {pol.type as string}: {pol.id as string}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {message && (
        <div className="card mb-4" style={{ borderColor: 'var(--success)' }}>
          <p>{message}</p>
        </div>
      )}

      <div className="grid-3 mb-4">
        {presets.map((p) => {
          const isActive = activePresetName === p.id;
          return (
          <div
            key={p.id}
            className={`preset-card ${selected === p.id ? 'selected' : ''}`}
            onClick={() => setSelected(p.id)}
            style={isActive ? { borderColor: 'var(--success)', borderWidth: 2 } : {}}
          >
            {isActive && (
              <div style={{ 
                position: 'absolute', 
                top: -10, 
                right: 10, 
                background: 'var(--success)', 
                color: 'white', 
                fontSize: '0.65rem', 
                fontWeight: 700,
                padding: '2px 8px', 
                borderRadius: 4 
              }}>
                ACTIVE
              </div>
            )}
            <div className="preset-name">{p.name}</div>
            <div className="preset-description">{p.description}</div>
            <button
              className="btn btn-secondary"
              style={{ marginTop: 8, fontSize: '0.75rem', padding: '4px 8px' }}
              onClick={(e) => toggleProscons(p.id, e)}
            >
              {expandedProscons === p.id ? 'Hide' : 'Pros & Cons'}
            </button>
            {expandedProscons === p.id && (
              <div className="proscons-dropdown" style={{ marginTop: 12, textAlign: 'left' }}>
                {p.pros.length > 0 && (
                  <div style={{ marginBottom: 8 }}>
                    <div style={{ fontWeight: 600, color: 'var(--success)', fontSize: '0.8rem', marginBottom: 4 }}>
                      Pros
                    </div>
                    <ul style={{ margin: 0, paddingLeft: 16, fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                      {p.pros.map((pro, i) => (
                        <li key={i} style={{ marginBottom: 2 }}>{pro}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {p.cons.length > 0 && (
                  <div>
                    <div style={{ fontWeight: 600, color: 'var(--warning)', fontSize: '0.8rem', marginBottom: 4 }}>
                      Cons
                    </div>
                    <ul style={{ margin: 0, paddingLeft: 16, fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                      {p.cons.map((con, i) => (
                        <li key={i} style={{ marginBottom: 2 }}>{con}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        );})}
        
        {/* Custom Preset Card */}
        {(() => {
          const isCustomActive = activePresetName?.startsWith('Custom:');
          return (
        <div
          className={`preset-card ${selected === 'custom' ? 'selected' : ''}`}
          onClick={() => setSelected('custom')}
          style={{ borderStyle: 'dashed', ...(isCustomActive ? { borderColor: 'var(--success)', borderWidth: 2 } : {}) }}
        >
          {isCustomActive && (
            <div style={{ 
              position: 'absolute', 
              top: -10, 
              right: 10, 
              background: 'var(--success)', 
              color: 'white', 
              fontSize: '0.65rem', 
              fontWeight: 700,
              padding: '2px 8px', 
              borderRadius: 4 
            }}>
              ACTIVE
            </div>
          )}
          <div className="preset-name">Custom</div>
          <div className="preset-description">
            {isCustomActive 
              ? `Currently running: ${activePresetName}` 
              : 'Use the wizard to create a custom configuration.'}
          </div>
        </div>
          );
        })()}
      </div>

      {/* Standard Preset Preview */}
      {selected && selected !== 'custom' && preview && (
        <div className="card">
          <div className="flex-between mb-4">
            <h3 className="card-title" style={{ margin: 0 }}>
              Preview: {selectedPreset?.name || selected}
            </h3>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
              {selected === 'vwap_trend' && (
                <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.9rem' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Optional: Session Filter (London + NY)</span>
                  <select
                    value={vwapSessionFilterOn ? 'on' : 'off'}
                    onChange={(e) => setVwapSessionFilterOn(e.target.value === 'on')}
                    title="Turn On to only allow trades during London (8–16 UTC) or NY (13–21 UTC) sessions; Off to trade any time."
                    style={{ padding: '6px 10px', borderRadius: 6, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)' }}
                  >
                    <option value="on">On</option>
                    <option value="off">Off</option>
                  </select>
                </label>
              )}
              <button
                className="btn btn-primary"
                onClick={handleApply}
                disabled={applying}
              >
                {applying ? 'Applying...' : 'Apply Preset'}
              </button>
            </div>
          </div>
          
          {/* Pros & Cons Summary */}
          {selectedPreset && (selectedPreset.pros.length > 0 || selectedPreset.cons.length > 0) && (
            <div className="grid-2 mb-4">
              {selectedPreset.pros.length > 0 && (
                <div style={{ padding: 12, background: 'rgba(16, 185, 129, 0.1)', borderRadius: 6, border: '1px solid var(--success)' }}>
                  <div style={{ fontWeight: 600, color: 'var(--success)', marginBottom: 8, fontSize: '0.85rem' }}>
                    Pros
                  </div>
                  <ul style={{ margin: 0, paddingLeft: 16, fontSize: '0.8rem' }}>
                    {selectedPreset.pros.map((pro, i) => (
                      <li key={i} style={{ marginBottom: 4 }}>{pro}</li>
                    ))}
                  </ul>
                </div>
              )}
              {selectedPreset.cons.length > 0 && (
                <div style={{ padding: 12, background: 'rgba(245, 158, 11, 0.1)', borderRadius: 6, border: '1px solid var(--warning)' }}>
                  <div style={{ fontWeight: 600, color: 'var(--warning)', marginBottom: 8, fontSize: '0.85rem' }}>
                    Cons
                  </div>
                  <ul style={{ margin: 0, paddingLeft: 16, fontSize: '0.8rem' }}>
                    {selectedPreset.cons.map((con, i) => (
                      <li key={i} style={{ marginBottom: 4 }}>{con}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          <p style={{ color: 'var(--text-secondary)', marginBottom: 12, fontSize: '0.8rem' }}>
            These settings will be changed when you apply:
          </p>
          <pre style={{ 
            background: 'var(--bg-primary)', 
            padding: 12, 
            borderRadius: 6,
            fontSize: '0.75rem',
            overflow: 'auto',
            maxHeight: 300
          }}>
            {JSON.stringify((preview as { changes?: unknown }).changes || preview, null, 2)}
          </pre>
        </div>
      )}

      {/* Custom Preset Wizard */}
      {selected === 'custom' && (
        <CustomWizard 
          profile={profile}
          currentProfile={currentProfile}
          onComplete={() => {
            setMessage('Custom configuration saved!');
            setTimeout(() => setMessage(null), 3000);
          }}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Profile Editor Page
// ---------------------------------------------------------------------------

function ProfilePage({ profile, authStatus, onAuthChange }: { profile: Profile; authStatus: Record<string, boolean>; onAuthChange: () => void }) {
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [retry, setRetry] = useState(0);
  
  // Password management state
  const [showPasswordModal, setShowPasswordModal] = useState<'add' | 'change' | 'remove' | null>(null);
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [passwordError, setPasswordError] = useState<string | null>(null);
  const [passwordPending, setPasswordPending] = useState(false);
  
  const hasPassword = authStatus[profile.path] || false;
  
  const resetPasswordModal = () => {
    setShowPasswordModal(null);
    setCurrentPassword('');
    setNewPassword('');
    setConfirmPassword('');
    setPasswordError(null);
  };
  
  const handleAddPassword = async () => {
    if (!newPassword || newPassword.length < 4) {
      setPasswordError('Password must be at least 4 characters');
      return;
    }
    if (newPassword !== confirmPassword) {
      setPasswordError('Passwords do not match');
      return;
    }
    setPasswordPending(true);
    setPasswordError(null);
    try {
      await api.setPassword(profile.path, null, newPassword);
      onAuthChange();
      resetPasswordModal();
      setMessage('Password set successfully!');
      setTimeout(() => setMessage(null), 3000);
    } catch (e) {
      setPasswordError((e as Error).message);
    } finally {
      setPasswordPending(false);
    }
  };
  
  const handleChangePassword = async () => {
    if (!currentPassword) {
      setPasswordError('Enter current password');
      return;
    }
    if (!newPassword || newPassword.length < 4) {
      setPasswordError('New password must be at least 4 characters');
      return;
    }
    if (newPassword !== confirmPassword) {
      setPasswordError('New passwords do not match');
      return;
    }
    setPasswordPending(true);
    setPasswordError(null);
    try {
      await api.setPassword(profile.path, currentPassword, newPassword);
      onAuthChange();
      resetPasswordModal();
      setMessage('Password changed successfully!');
      setTimeout(() => setMessage(null), 3000);
    } catch (e) {
      setPasswordError((e as Error).message);
    } finally {
      setPasswordPending(false);
    }
  };
  
  const handleRemovePassword = async () => {
    if (!currentPassword) {
      setPasswordError('Enter current password');
      return;
    }
    setPasswordPending(true);
    setPasswordError(null);
    try {
      await api.removePassword(profile.path, currentPassword);
      onAuthChange();
      resetPasswordModal();
      setMessage('Password removed. Profile is now non-secure.');
      setTimeout(() => setMessage(null), 3000);
    } catch (e) {
      setPasswordError((e as Error).message);
    } finally {
      setPasswordPending(false);
    }
  };

  useEffect(() => {
    setData(null);
    setLoadError(null);
    api.getProfile(profile.path)
      .then((p) => { setData(p); setLoadError(null); })
      .catch((e: unknown) => { setLoadError((e as Error).message); setData(null); });
  }, [profile.path, retry]);

  const handleSave = async () => {
    if (!data) return;
    setSaving(true);
    setMessage(null);
    try {
      await api.saveProfile(profile.path, data);
      setMessage('Profile saved!');
      setTimeout(() => setMessage(null), 2000);
    } catch (e: unknown) {
      setMessage(`Error: ${(e as Error).message}`);
    } finally {
      setSaving(false);
    }
  };

  if (!data) {
    return (
      <div>
        <h2 className="page-title">Profile Editor</h2>
        {loadError ? (
          <div className="card" style={{ borderColor: 'var(--danger)' }}>
            <p style={{ color: 'var(--danger)' }}>Failed to load profile: {loadError}</p>
            <button className="btn btn-primary" onClick={() => setRetry((r) => r + 1)}>
              Retry
            </button>
          </div>
        ) : (
          <div className="card">Loading profile...</div>
        )}
      </div>
    );
  }

  const risk = data.risk as Record<string, unknown> || {};
  const execution = data.execution as Record<string, unknown> || {};

  const updateNested = (section: string, field: string, value: unknown) => {
    setData({
      ...data,
      [section]: {
        ...(data[section] as Record<string, unknown> || {}),
        [field]: value,
      },
    });
  };

  const updateTopLevel = (field: string, value: unknown) => {
    setData({ ...data, [field]: value });
  };

  return (
    <div>
      <h2 className="page-title">Profile Editor</h2>

      {message && (
        <div className="card mb-4" style={{ borderColor: 'var(--success)' }}>
          <p>{message}</p>
        </div>
      )}

      <div className="grid-2">
        <div className="card">
          <h3 className="card-title">Basic Settings</h3>
          
          <div className="form-group">
            <label>Symbol</label>
            <input
              value={(data.symbol as string) || ''}
              onChange={(e) => setData({ ...data, symbol: e.target.value })}
            />
          </div>

          <div className="form-group">
            <label>Pip Size</label>
            <input
              type="number"
              step="0.001"
              value={(data.pip_size as number) || 0.01}
              onChange={(e) => setData({ ...data, pip_size: parseFloat(e.target.value) })}
            />
          </div>
        </div>

        <div className="card">
          <h3 className="card-title">Account Settings</h3>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 16 }}>
            Deposit amount and leverage ratio for your reference.
          </p>
          
          <div className="form-group">
            <label>Deposit Amount ($)</label>
            <input
              type="number"
              placeholder="e.g., 100000"
              value={(data.deposit_amount as number) || ''}
              onChange={(e) => updateTopLevel('deposit_amount', e.target.value ? parseFloat(e.target.value) : null)}
            />
          </div>

          <div className="form-group">
            <label>Leverage Ratio</label>
            <select
              value={(data.leverage_ratio as number) || ''}
              onChange={(e) => updateTopLevel('leverage_ratio', e.target.value ? parseInt(e.target.value) : null)}
            >
              <option value="">Select leverage...</option>
              <option value="50">1:50</option>
              <option value="100">1:100</option>
              <option value="200">1:200</option>
              <option value="400">1:400</option>
              <option value="500">1:500</option>
            </select>
          </div>

          <div className="form-group">
            <label>Display Currency (Stats &amp; Logs)</label>
            <select
              value={(data.display_currency as string) || 'USD'}
              onChange={(e) => updateTopLevel('display_currency', e.target.value as 'USD' | 'JPY')}
            >
              <option value="USD">USD</option>
              <option value="JPY">JPY</option>
            </select>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: 4 }}>
              Profits and balances will be shown in this currency. JPY uses current rate from your symbol.
            </p>
          </div>
        </div>

        <div className="card">
          <h3 className="card-title">Broker Connection</h3>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 16 }}>
            Choose MetaTrader 5 (Windows) or OANDA (works on Mac/Linux and PaaS like Railway). OANDA is available in Japan and Canada.
          </p>
          <div className="form-group">
            <label>Broker Type</label>
            <select
              value={(data.broker_type as string) || 'mt5'}
              onChange={(e) => updateTopLevel('broker_type', e.target.value as 'mt5' | 'oanda')}
            >
              <option value="mt5">MetaTrader 5 (MT5)</option>
              <option value="oanda">OANDA</option>
            </select>
          </div>
          {(data.broker_type as string) === 'oanda' && (
            <>
              <div className="form-group">
                <label>OANDA API Key</label>
                <input
                  type="password"
                  autoComplete="off"
                  placeholder="Paste your OANDA API token"
                  value={(data.oanda_token as string) || ''}
                  onChange={(e) => updateTopLevel('oanda_token', e.target.value || null)}
                />
                <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: 4 }}>
                  Create a token in OANDA Practice or Live account: Account → Manage API Access.
                </p>
              </div>
              <div className="form-group">
                <label>OANDA Account ID (optional)</label>
                <input
                  type="text"
                  placeholder="Leave blank to use first account"
                  value={(data.oanda_account_id as string) || ''}
                  onChange={(e) => updateTopLevel('oanda_account_id', e.target.value || null)}
                />
              </div>
              <div className="form-group">
                <label>OANDA Environment</label>
                <select
                  value={(data.oanda_environment as string) || 'practice'}
                  onChange={(e) => updateTopLevel('oanda_environment', e.target.value as 'practice' | 'live')}
                >
                  <option value="practice">Practice (demo)</option>
                  <option value="live">Live</option>
                </select>
              </div>
            </>
          )}
        </div>

        <div className="card">
          <h3 className="card-title">Risk Settings</h3>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 16, marginTop: -8 }}>
            These limits are final and cannot be overridden by presets or the custom wizard.
          </p>
          <div className="form-group">
            <label>Max Lots</label>
            <input
              type="number"
              step="0.01"
              value={(risk.max_lots as number) || 0.1}
              onChange={(e) => updateNested('risk', 'max_lots', parseFloat(e.target.value))}
            />
          </div>

          <div className="form-group">
            <label>Max Spread (pips)</label>
            <input
              type="number"
              step="0.1"
              value={(risk.max_spread_pips as number) || 2}
              onChange={(e) => updateNested('risk', 'max_spread_pips', parseFloat(e.target.value))}
            />
          </div>

          <div className="form-group">
            <label>Max Trades/Day</label>
            <input
              type="number"
              value={(risk.max_trades_per_day as number) || 10}
              onChange={(e) => updateNested('risk', 'max_trades_per_day', parseInt(e.target.value))}
            />
          </div>

          <div className="form-group">
            <label>Max Open Trades</label>
            <input
              type="number"
              min={1}
              max={20}
              value={(risk.max_open_trades as number) ?? 2}
              onChange={(e) => updateNested('risk', 'max_open_trades', parseInt(e.target.value) || 1)}
            />
          </div>

          <div className="form-group">
            <label>Cooldown After Loss (minutes)</label>
            <input
              type="number"
              min={0}
              max={120}
              value={(risk.cooldown_minutes_after_loss as number) ?? 0}
              onChange={(e) => updateNested('risk', 'cooldown_minutes_after_loss', parseInt(e.target.value) || 0)}
            />
          </div>

          <div className="form-group">
            <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <input
                type="checkbox"
                checked={(risk.require_stop as boolean) || false}
                onChange={(e) => updateNested('risk', 'require_stop', e.target.checked)}
                style={{ width: 'auto' }}
              />
              Require Stop Loss
            </label>
          </div>

          <div className="form-group">
            <label>Min Stop Pips</label>
            <input
              type="number"
              step="1"
              min={1}
              max={500}
              value={(risk.min_stop_pips as number) ?? 10}
              onChange={(e) => updateNested('risk', 'min_stop_pips', parseFloat(e.target.value) || 10)}
            />
            <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: 4 }}>
              Minimum stop loss distance in pips. Default is 10.
            </p>
          </div>
        </div>
      </div>

      {/* Profile Security */}
      <div className="card mt-4">
        <h3 className="card-title">Profile Security</h3>
        {hasPassword ? (
          <>
            <p style={{ color: 'var(--text-secondary)', marginBottom: 16 }}>
              This profile is protected by a password.
            </p>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <button className="btn btn-secondary" onClick={() => setShowPasswordModal('change')}>
                Change Password
              </button>
              <button className="btn btn-danger" onClick={() => setShowPasswordModal('remove')}>
                Remove Password
              </button>
            </div>
          </>
        ) : (
          <>
            <p style={{ color: 'var(--text-secondary)', marginBottom: 16 }}>
              This profile has no password. Anyone with access to this computer can open it.
            </p>
            <button className="btn btn-primary" onClick={() => setShowPasswordModal('add')}>
              Add Password
            </button>
          </>
        )}
      </div>

      {/* Password Modal */}
      {showPasswordModal && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0,0,0,0.7)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
          }}
        >
          <div className="card" style={{ maxWidth: 400, margin: 16 }}>
            <h3 className="card-title">
              {showPasswordModal === 'add' && 'Add Password'}
              {showPasswordModal === 'change' && 'Change Password'}
              {showPasswordModal === 'remove' && 'Remove Password'}
            </h3>
            
            {(showPasswordModal === 'change' || showPasswordModal === 'remove') && (
              <div className="form-group" style={{ marginBottom: 12 }}>
                <label>Current Password</label>
                <input
                  type="password"
                  value={currentPassword}
                  onChange={(e) => setCurrentPassword(e.target.value)}
                  placeholder="Enter current password"
                />
              </div>
            )}
            
            {(showPasswordModal === 'add' || showPasswordModal === 'change') && (
              <>
                <div className="form-group" style={{ marginBottom: 12 }}>
                  <label>{showPasswordModal === 'change' ? 'New Password' : 'Password'}</label>
                  <input
                    type="password"
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    placeholder="Min 4 characters"
                  />
                </div>
                <div className="form-group" style={{ marginBottom: 12 }}>
                  <label>Confirm {showPasswordModal === 'change' ? 'New ' : ''}Password</label>
                  <input
                    type="password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    placeholder="Confirm password"
                  />
                </div>
              </>
            )}
            
            {showPasswordModal === 'remove' && (
              <p style={{ color: 'var(--text-secondary)', marginBottom: 12 }}>
                This will remove password protection. Anyone with access to this computer will be able to open this profile.
              </p>
            )}
            
            {(showPasswordModal === 'add' || showPasswordModal === 'change') && (
              <p style={{ fontSize: '0.75rem', color: 'var(--warning)', marginBottom: 12, background: 'rgba(234, 179, 8, 0.1)', padding: 8, borderRadius: 4 }}>
                Write your password down in a safe location. There is no forgot-password option.
              </p>
            )}
            
            {passwordError && (
              <p style={{ color: 'var(--danger)', fontSize: '0.85rem', marginBottom: 12 }}>{passwordError}</p>
            )}
            
            <div style={{ display: 'flex', gap: 8 }}>
              <button
                className={showPasswordModal === 'remove' ? 'btn btn-danger' : 'btn btn-primary'}
                onClick={
                  showPasswordModal === 'add' ? handleAddPassword :
                  showPasswordModal === 'change' ? handleChangePassword :
                  handleRemovePassword
                }
                disabled={passwordPending}
              >
                {passwordPending ? 'Processing...' : 
                  showPasswordModal === 'add' ? 'Set Password' :
                  showPasswordModal === 'change' ? 'Change Password' :
                  'Remove Password'
                }
              </button>
              <button className="btn btn-secondary" onClick={resetPasswordModal} disabled={passwordPending}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="card mt-4">
        <h3 className="card-title">Execution Settings</h3>

        <div className="grid-2">
          <div className="form-group">
            <label>Loop Poll Seconds</label>
            <input
              type="number"
              step="0.5"
              value={(execution.loop_poll_seconds as number) || 5}
              onChange={(e) => updateNested('execution', 'loop_poll_seconds', parseFloat(e.target.value))}
            />
          </div>

          <div className="form-group">
            <label>Fast Poll Seconds</label>
            <input
              type="number"
              step="0.5"
              value={(execution.loop_poll_seconds_fast as number) || 2}
              onChange={(e) => updateNested('execution', 'loop_poll_seconds_fast', parseFloat(e.target.value))}
            />
          </div>
        </div>

        <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: 8 }}>
          Policies: {((execution.policies as unknown[]) || []).length} configured.
          Use the Streamlit UI or edit JSON directly for full policy management.
        </p>
      </div>

      <div className="mt-4">
        <button className="btn btn-primary" onClick={handleSave} disabled={saving}>
          {saving ? 'Saving...' : 'Save Profile'}
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Equity Curve Chart
// ---------------------------------------------------------------------------

interface EquityDataPoint {
  date: string;
  dailyProfit: number;
  cumProfit: number;
  tradeCount: number;
}

function EquityCurveChart({ profileName, profilePath }: { profileName: string; profilePath: string }) {
  const [data, setData] = useState<EquityDataPoint[]>([]);
  const [displayCurrency, setDisplayCurrency] = useState<string>('USD');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    api.getTradeHistory(profileName, profilePath, 365)
      .then((resp) => {
        setDisplayCurrency(resp.display_currency || 'USD');
        setData(resp.days.map((d) => ({
          date: `${d.date.slice(5, 7)}/${d.date.slice(8, 10)}`,
          dailyProfit: d.daily_profit,
          cumProfit: d.cum_profit,
          tradeCount: d.trade_count,
        })));
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [profileName, profilePath]);

  if (loading) return (
    <div className="card mb-4">
      <h3 className="card-title" style={{ margin: 0, marginBottom: 8 }}>Equity Curve</h3>
      <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', margin: 0 }}>Loading chart data from broker...</p>
    </div>
  );

  if (data.length < 2) return (
    <div className="card mb-4">
      <h3 className="card-title" style={{ margin: 0, marginBottom: 8 }}>Equity Curve</h3>
      <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', margin: 0 }}>
        Chart will appear after 2+ trading days with closed trades.
      </p>
    </div>
  );

  const currency = displayCurrency;

  const CustomTooltip = ({ active, payload, label }: { active?: boolean; payload?: Array<{ value: number; dataKey: string }>; label?: string }) => {
    if (!active || !payload || !payload.length) return null;
    const cum = payload.find(p => p.dataKey === 'cumProfit');
    const daily = payload.find(p => p.dataKey === 'dailyProfit');
    const count = payload.find(p => p.dataKey === 'tradeCount');
    return (
      <div style={{
        background: '#1a1d24',
        border: '1px solid #3a3d45',
        borderRadius: 6,
        padding: '10px 14px',
        fontSize: '0.8rem',
        lineHeight: 1.6,
      }}>
        <div style={{ fontWeight: 600, marginBottom: 4, color: '#e0e0e0' }}>{label}</div>
        {cum && (
          <div style={{ color: cum.value >= 0 ? '#28a745' : '#dc3545' }}>
            Cumulative: {cum.value >= 0 ? '+' : ''}{cum.value.toFixed(2)} {currency}
          </div>
        )}
        {daily && (
          <div style={{ color: daily.value >= 0 ? '#28a745' : '#dc3545' }}>
            Daily P/L: {daily.value >= 0 ? '+' : ''}{daily.value.toFixed(2)} {currency}
          </div>
        )}
        {count && (
          <div style={{ color: '#4a90d9' }}>
            Trades: {count.value}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="card mb-4">
      <h3 className="card-title" style={{ margin: 0, marginBottom: 16 }}>Equity Curve</h3>
      <div style={{ width: '100%', height: 300 }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
            <defs>
              <linearGradient id="profitGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#28a745" stopOpacity={0.3} />
                <stop offset="100%" stopColor="#28a745" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#3a3d45" vertical={false} />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 11, fill: '#a0a0a0' }}
              tickLine={false}
              axisLine={{ stroke: '#3a3d45' }}
            />
            <YAxis
              yAxisId="left"
              tick={{ fontSize: 11, fill: '#a0a0a0' }}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v: number) => `${v >= 0 ? '+' : ''}${v}`}
            />
            <YAxis
              yAxisId="right"
              orientation="right"
              tick={{ fontSize: 11, fill: '#4a90d9' }}
              tickLine={false}
              axisLine={false}
              allowDecimals={false}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine yAxisId="left" y={0} stroke="#3a3d45" strokeDasharray="3 3" />
            <Area
              yAxisId="left"
              type="monotone"
              dataKey="cumProfit"
              stroke="#28a745"
              strokeWidth={2}
              fill="url(#profitGradient)"
            />
            <Bar
              yAxisId="left"
              dataKey="dailyProfit"
              barSize={data.length > 60 ? 4 : data.length > 30 ? 8 : 14}
              fill="#28a745"
              shape={(props: any) => {
                const { x, y, width, height, payload } = props;
                return (
                  <rect
                    x={x}
                    y={height < 0 ? y + height : y}
                    width={width}
                    height={Math.abs(height)}
                    fill={payload.dailyProfit >= 0 ? '#28a745' : '#dc3545'}
                    opacity={0.7}
                    rx={1}
                  />
                );
              }}
            />
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="tradeCount"
              stroke="#4a90d9"
              strokeWidth={1.5}
              dot={false}
              opacity={0.7}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div style={{ display: 'flex', gap: 16, marginTop: 8, fontSize: '0.7rem', color: '#a0a0a0', justifyContent: 'center' }}>
        <span><span style={{ display: 'inline-block', width: 12, height: 3, background: '#28a745', marginRight: 4, verticalAlign: 'middle' }}></span>Cumulative P/L</span>
        <span><span style={{ display: 'inline-block', width: 8, height: 8, background: '#28a745', marginRight: 4, verticalAlign: 'middle', opacity: 0.7 }}></span>Daily P/L</span>
        <span><span style={{ display: 'inline-block', width: 12, height: 3, background: '#4a90d9', marginRight: 4, verticalAlign: 'middle' }}></span>Trades/Day</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Analytics Components for Logs & Stats
// ---------------------------------------------------------------------------

function SessionPerformance({ profileName, profilePath }: { profileName: string; profilePath: string }) {
  const [trades, setTrades] = useState<api.TradeDetail[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    api.getTradeHistoryDetail(profileName, profilePath, 365)
      .then((resp) => setTrades(resp.trades))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [profileName, profilePath]);

  if (loading) return (
    <div className="card mb-4">
      <h3 className="card-title" style={{ margin: 0, marginBottom: 8 }}>Session Performance</h3>
      <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', margin: 0 }}>Loading...</p>
    </div>
  );

  if (trades.length === 0) return (
    <div className="card mb-4">
      <h3 className="card-title" style={{ margin: 0, marginBottom: 8 }}>Session Performance</h3>
      <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', margin: 0 }}>No closed trades yet.</p>
    </div>
  );

  const sessions = [
    { name: 'Tokyo', startHour: 0, endHour: 9 },
    { name: 'London', startHour: 7, endHour: 16 },
    { name: 'New York', startHour: 12, endHour: 21 },
  ];

  const sessionStats = sessions.map((session) => {
    const sessionTrades = trades.filter((t) => {
      const hour = new Date(t.entry_time_utc).getUTCHours();
      return hour >= session.startHour && hour < session.endHour;
    });
    const wins = sessionTrades.filter((t) => t.pips !== null && t.pips > 0).length;
    const totalPips = sessionTrades.reduce((sum, t) => sum + (t.pips || 0), 0);
    const avgPips = sessionTrades.length > 0 ? totalPips / sessionTrades.length : 0;
    return {
      ...session,
      count: sessionTrades.length,
      winRate: sessionTrades.length > 0 ? wins / sessionTrades.length : 0,
      totalPips,
      avgPips,
    };
  });

  return (
    <div className="card mb-4">
      <h3 className="card-title" style={{ margin: 0, marginBottom: 16 }}>Session Performance</h3>
      <div className="grid-3">
        {sessionStats.map((s) => (
          <div key={s.name} className="stat-box">
            <div style={{ fontWeight: 700, fontSize: '1rem', marginBottom: 4 }}>{s.name}</div>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', marginBottom: 8 }}>
              {String(s.startHour).padStart(2, '0')}:00–{String(s.endHour).padStart(2, '0')}:00 UTC
            </div>
            <div style={{ fontSize: '0.85rem' }}>{s.count} trades</div>
            <div style={{ fontSize: '0.85rem', color: s.winRate >= 0.5 ? 'var(--success)' : 'var(--warning)' }}>
              {(s.winRate * 100).toFixed(0)}% win rate
            </div>
            <div style={{ fontSize: '0.85rem', color: s.totalPips >= 0 ? 'var(--success)' : 'var(--danger)' }}>
              {s.totalPips >= 0 ? '+' : ''}{s.totalPips.toFixed(1)} pips
            </div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              Avg: {s.avgPips >= 0 ? '+' : ''}{s.avgPips.toFixed(2)} pips
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function LongShortPerformance({ profileName, profilePath }: { profileName: string; profilePath: string }) {
  const [trades, setTrades] = useState<api.TradeDetail[]>([]);
  const [displayCurrency, setDisplayCurrency] = useState<string>('USD');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    api.getTradeHistoryDetail(profileName, profilePath, 365)
      .then((resp) => {
        setTrades(resp.trades);
        setDisplayCurrency(resp.display_currency || 'USD');
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [profileName, profilePath]);

  if (loading) return (
    <div className="card mb-4">
      <h3 className="card-title" style={{ margin: 0, marginBottom: 8 }}>Long vs Short</h3>
      <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', margin: 0 }}>Loading...</p>
    </div>
  );

  if (trades.length === 0) return (
    <div className="card mb-4">
      <h3 className="card-title" style={{ margin: 0, marginBottom: 8 }}>Long vs Short</h3>
      <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', margin: 0 }}>No closed trades yet.</p>
    </div>
  );

  const computeSide = (side: string) => {
    const filtered = trades.filter((t) => t.side === side);
    const wins = filtered.filter((t) => t.pips !== null && t.pips > 0).length;
    const totalPips = filtered.reduce((sum, t) => sum + (t.pips || 0), 0);
    const totalProfit = filtered.reduce((sum, t) => sum + (t.profit || 0), 0);
    const avgPips = filtered.length > 0 ? totalPips / filtered.length : 0;
    return {
      count: filtered.length,
      winRate: filtered.length > 0 ? wins / filtered.length : 0,
      totalPips,
      totalProfit,
      avgPips,
    };
  };

  const longStats = computeSide('buy');
  const shortStats = computeSide('sell');

  const SideBox = ({ label, stats, color }: { label: string; stats: typeof longStats; color: string }) => (
    <div className="stat-box">
      <div style={{ fontWeight: 700, fontSize: '1rem', marginBottom: 8, color }}>{label}</div>
      <div style={{ fontSize: '0.85rem' }}>{stats.count} trades</div>
      <div style={{ fontSize: '0.85rem', color: stats.winRate >= 0.5 ? 'var(--success)' : 'var(--warning)' }}>
        {(stats.winRate * 100).toFixed(0)}% win rate
      </div>
      <div style={{ fontSize: '0.85rem', color: stats.totalPips >= 0 ? 'var(--success)' : 'var(--danger)' }}>
        {stats.totalPips >= 0 ? '+' : ''}{stats.totalPips.toFixed(1)} pips
      </div>
      <div style={{ fontSize: '0.85rem', color: stats.totalProfit >= 0 ? 'var(--success)' : 'var(--danger)' }}>
        {stats.totalProfit >= 0 ? '+' : ''}{stats.totalProfit.toFixed(2)} {displayCurrency}
      </div>
      <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
        Avg: {stats.avgPips >= 0 ? '+' : ''}{stats.avgPips.toFixed(2)} pips
      </div>
    </div>
  );

  return (
    <div className="card mb-4">
      <h3 className="card-title" style={{ margin: 0, marginBottom: 16 }}>Long vs Short</h3>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <SideBox label="Long (Buy)" stats={longStats} color="var(--success)" />
        <SideBox label="Short (Sell)" stats={shortStats} color="var(--danger)" />
      </div>
    </div>
  );
}

function SpreadPerformance({ profileName, profilePath }: { profileName: string; profilePath: string }) {
  const [trades, setTrades] = useState<api.TradeDetail[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    api.getTradeHistoryDetail(profileName, profilePath, 365)
      .then((resp) => setTrades(resp.trades))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [profileName, profilePath]);

  if (loading) return null;

  const withSpread = trades.filter((t) => t.spread_pips !== null && t.spread_pips !== undefined);

  if (withSpread.length === 0) return (
    <div className="card mb-4">
      <h3 className="card-title" style={{ margin: 0, marginBottom: 8 }}>Spread Analysis</h3>
      <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', margin: 0 }}>
        Spread analysis available after bot-placed trades accumulate.
      </p>
    </div>
  );

  const categories = [
    { name: 'Tight', label: '< 1.0 pip', filter: (s: number) => s < 1.0 },
    { name: 'Normal', label: '1.0–2.0 pips', filter: (s: number) => s >= 1.0 && s <= 2.0 },
    { name: 'Wide', label: '> 2.0 pips', filter: (s: number) => s > 2.0 },
  ];

  const catStats = categories.map((cat) => {
    const filtered = withSpread.filter((t) => cat.filter(t.spread_pips!));
    const wins = filtered.filter((t) => t.pips !== null && t.pips > 0).length;
    const totalPips = filtered.reduce((sum, t) => sum + (t.pips || 0), 0);
    const avgPips = filtered.length > 0 ? totalPips / filtered.length : 0;
    return {
      ...cat,
      count: filtered.length,
      winRate: filtered.length > 0 ? wins / filtered.length : 0,
      avgPips,
    };
  });

  return (
    <div className="card mb-4">
      <h3 className="card-title" style={{ margin: 0, marginBottom: 16 }}>Spread Analysis</h3>
      <div className="grid-3">
        {catStats.map((c) => (
          <div key={c.name} className="stat-box">
            <div style={{ fontWeight: 700, fontSize: '1rem', marginBottom: 4 }}>{c.name}</div>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', marginBottom: 8 }}>{c.label}</div>
            <div style={{ fontSize: '0.85rem' }}>{c.count} trades</div>
            {c.count > 0 && (
              <>
                <div style={{ fontSize: '0.85rem', color: c.winRate >= 0.5 ? 'var(--success)' : 'var(--warning)' }}>
                  {(c.winRate * 100).toFixed(0)}% win rate
                </div>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  Avg: {c.avgPips >= 0 ? '+' : ''}{c.avgPips.toFixed(2)} pips
                </div>
              </>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Advanced Analytics Component
// ---------------------------------------------------------------------------

function AdvancedAnalytics({ profileName, profilePath }: { profileName: string; profilePath: string }) {
  const [trades, setTrades] = useState<api.AdvancedTrade[]>([]);
  const [, setCurrency] = useState('USD');
  const [startingBalance, setStartingBalance] = useState<number | null>(null);
  const [totalProfitCurrency, setTotalProfitCurrency] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [presetFilter, setPresetFilter] = useState('all');
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    mae_mfe: false,
    rolling: false,
    r_dist: false,
    drawdown: false,
    duration: false,
  });
  const [rollingWindow, setRollingWindow] = useState(20);
  const [rollingMode, setRollingMode] = useState<'trades' | 'time'>('trades');
  const [timeWindow, setTimeWindow] = useState(7);

  useEffect(() => {
    setLoading(true);
    api.getAdvancedAnalytics(profileName, profilePath, 365)
      .then((data) => {
        setTrades(data.trades);
        setCurrency(data.display_currency || 'USD');
        setStartingBalance(data.starting_balance ?? null);
        setTotalProfitCurrency(data.total_profit_currency ?? null);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [profileName, profilePath]);

  const toggleSection = (key: string) => {
    setExpandedSections(prev => ({ ...prev, [key]: !prev[key] }));
  };

  // Filter trades by preset
  const presets = Array.from(new Set(trades.map(t => t.preset_name).filter(Boolean))) as string[];
  const filtered = presetFilter === 'all' ? trades : trades.filter(t => t.preset_name === presetFilter);

  // --- Computation utilities ---
  const computeEquitySeries = (arr: api.AdvancedTrade[]) => {
    let cum = 0;
    return arr
      .filter(t => t.pips != null)
      .sort((a, b) => a.exit_time_utc.localeCompare(b.exit_time_utc))
      .map((t, i) => {
        cum += t.pips!;
        return { idx: i + 1, cumPips: Math.round(cum * 100) / 100, date: t.exit_time_utc.slice(0, 10), pips: t.pips! };
      });
  };

  const computeRollingMetrics = (arr: api.AdvancedTrade[], windowSize: number, mode: 'trades' | 'time', timeDays: number) => {
    const sorted = arr
      .filter(t => t.pips != null)
      .sort((a, b) => a.exit_time_utc.localeCompare(b.exit_time_utc));
    if (sorted.length === 0) return [];

    return sorted.map((_, i) => {
      let window: api.AdvancedTrade[];
      if (mode === 'trades') {
        const start = Math.max(0, i - windowSize + 1);
        window = sorted.slice(start, i + 1);
      } else {
        const exitDate = new Date(sorted[i].exit_time_utc);
        const cutoff = new Date(exitDate.getTime() - timeDays * 86400000);
        window = sorted.filter(t => new Date(t.exit_time_utc) >= cutoff && new Date(t.exit_time_utc) <= exitDate);
      }
      if (window.length === 0) return null;

      const wins = window.filter(t => (t.pips ?? 0) > 0).length;
      const winRate = Math.round((wins / window.length) * 1000) / 10;
      const avgPips = Math.round(window.reduce((s, t) => s + (t.pips ?? 0), 0) / window.length * 100) / 100;
      const rTrades = window.filter(t => t.r_multiple != null);
      const avgR = rTrades.length > 0
        ? Math.round(rTrades.reduce((s, t) => s + t.r_multiple!, 0) / rTrades.length * 100) / 100
        : null;
      const winsR = rTrades.filter(t => t.r_multiple! > 0);
      const lossesR = rTrades.filter(t => t.r_multiple! <= 0);
      const avgWinR = winsR.length > 0 ? winsR.reduce((s, t) => s + t.r_multiple!, 0) / winsR.length : 0;
      const avgLossR = lossesR.length > 0 ? Math.abs(lossesR.reduce((s, t) => s + t.r_multiple!, 0) / lossesR.length) : 0;
      const wr = rTrades.length > 0 ? winsR.length / rTrades.length : 0;
      const expectancy = rTrades.length > 0 ? Math.round((wr * avgWinR - (1 - wr) * avgLossR) * 100) / 100 : null;

      return { idx: i + 1, winRate, avgPips, avgR, expectancy, date: sorted[i].exit_time_utc.slice(0, 10) };
    }).filter(Boolean) as { idx: number; winRate: number; avgPips: number; avgR: number | null; expectancy: number | null; date: string }[];
  };

  const computeDrawdownSeries = (
    arr: api.AdvancedTrade[],
    opts: { startingBalance?: number | null; totalProfitCurrency?: number | null } = {}
  ) => {
    const { startingBalance: sb = null, totalProfitCurrency: totalProfitCur = null } = opts;
    const hasProfitData = arr.some(t => t.profit != null);
    const useCurrency = sb != null && sb > 0 && hasProfitData;

    const equity = computeEquitySeries(arr);
    if (equity.length === 0) {
      return {
        series: [],
        maxDdPips: 0,
        maxDdPct: null as number | null,
        maxDdPctNote: 'Requires starting balance (set in Profile or broker connection)',
        currentDd: 0,
        longestTrades: 0,
        longestTime: '',
        recoveryFactor: null as number | null,
        recoveryFactorNote: null as string | null,
        maxDdUsd: null as number | null,
      };
    }

    let peak = 0;
    let maxDd = 0;
    let longestTrades = 0;
    let longestTimeStart = '';
    let longestTimeEnd = '';
    let currentLongest = 0;
    let currentLongestStart = '';

    const series = equity.map(e => {
      if (e.cumPips > peak) {
        if (currentLongest > longestTrades) {
          longestTrades = currentLongest;
          longestTimeStart = currentLongestStart;
          longestTimeEnd = e.date;
        }
        peak = e.cumPips;
        currentLongest = 0;
        currentLongestStart = e.date;
      } else {
        currentLongest++;
        if (currentLongest === 1) currentLongestStart = e.date;
      }
      const dd = peak - e.cumPips;
      if (dd > maxDd) {
        maxDd = dd;
      }
      return { idx: e.idx, dd: -dd, date: e.date };
    });

    if (currentLongest > longestTrades) {
      longestTrades = currentLongest;
      longestTimeStart = currentLongestStart;
      longestTimeEnd = equity[equity.length - 1].date;
    }

    const currentDd = peak - equity[equity.length - 1].cumPips;
    const totalProfitPips = equity[equity.length - 1].cumPips;

    let maxDdPct: number | null = null;
    let maxDdPctNote: string | null = null;
    let recoveryFactor: number | null = null;
    let recoveryFactorNote: string | null = null;
    let maxDdUsd: number | null = null;

    if (useCurrency && totalProfitCur != null) {
      const sortedByExit = arr
        .filter(t => t.profit != null)
        .sort((a, b) => a.exit_time_utc.localeCompare(b.exit_time_utc));
      let cum = 0;
      let peakUsd = sb;
      let maxDdUsdVal = 0;
      for (const t of sortedByExit) {
        cum += t.profit!;
        const equityUsd = sb + cum;
        if (equityUsd > peakUsd) peakUsd = equityUsd;
        const ddUsd = peakUsd - equityUsd;
        if (ddUsd > maxDdUsdVal) maxDdUsdVal = ddUsd;
      }
      maxDdUsd = Math.round(maxDdUsdVal * 100) / 100;
      maxDdPct = peakUsd > 0 ? Math.round((maxDdUsdVal / peakUsd) * 1000) / 10 : null;
      recoveryFactor =
        maxDdUsdVal > 0 && totalProfitCur > 0
          ? Math.round((totalProfitCur / maxDdUsdVal) * 100) / 100
          : null;
      if (recoveryFactor == null && totalProfitCur != null && maxDdUsdVal > 0 && totalProfitCur <= 0) {
        recoveryFactorNote = 'Requires profit data (sync from broker)';
      }
    } else {
      maxDdPctNote = 'Requires starting balance (set in Profile or broker connection)';
      if (totalProfitPips < 0 || maxDd <= 0) {
        recoveryFactorNote = 'N/A (net pips negative or no drawdown; use profit-based view when broker connected)';
      } else {
        recoveryFactor = maxDd > 0 ? Math.round((totalProfitPips / maxDd) * 100) / 100 : null;
      }
    }

    let longestTimeDays = '';
    if (longestTimeStart && longestTimeEnd) {
      const diff = Math.round((new Date(longestTimeEnd).getTime() - new Date(longestTimeStart).getTime()) / 86400000);
      longestTimeDays = `${diff}d`;
    }

    return {
      series,
      maxDdPips: Math.round(maxDd * 100) / 100,
      maxDdPct,
      maxDdPctNote,
      currentDd: Math.round(currentDd * 100) / 100,
      longestTrades,
      longestTime: longestTimeDays,
      recoveryFactor,
      recoveryFactorNote,
      maxDdUsd,
    };
  };

  // --- Helper: histogram bins ---
  const computeHistogramBins = (values: number[], binCount = 8): { bin: string; count: number; midValue: number }[] => {
    if (values.length === 0) return [];
    const min = Math.min(...values);
    const max = Math.max(...values);
    if (min === max) return [{ bin: min.toFixed(1), count: values.length, midValue: min }];
    const binWidth = (max - min) / binCount;
    const bins: { bin: string; count: number; midValue: number }[] = [];
    for (let i = 0; i < binCount; i++) {
      const lo = min + i * binWidth;
      const hi = lo + binWidth;
      const count = values.filter(v => v >= lo && (i === binCount - 1 ? v <= hi : v < hi)).length;
      bins.push({ bin: `${lo.toFixed(1)}`, count, midValue: lo + binWidth / 2 });
    }
    return bins;
  };

  if (loading) {
    return (
      <div className="card mb-4" style={{ textAlign: 'center', padding: 24 }}>
        <span style={{ color: 'var(--text-secondary)' }}>Loading advanced analytics...</span>
      </div>
    );
  }

  if (trades.length === 0) {
    return null;
  }

  // Shared style helpers
  const sectionHeaderStyle: React.CSSProperties = {
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    cursor: 'pointer', padding: '12px 16px',
    background: 'var(--bg-tertiary)', borderRadius: 6, border: '1px solid var(--border)',
    marginBottom: 4,
  };

  const statBoxStyle: React.CSSProperties = {
    textAlign: 'center', padding: 12, background: 'var(--bg-tertiary)',
    borderRadius: 6, border: '1px solid var(--border)',
  };

  const chartColors = { green: '#22c55e', red: '#ef4444', blue: '#3b82f6', orange: '#f59e0b', purple: '#a855f7' };

  // --- Section data ---
  const maeTradesAll = filtered.filter(t => t.max_adverse_pips != null && t.pips != null);
  const mfeTradesAll = filtered.filter(t => t.max_favorable_pips != null && t.pips != null);
  const rTradesAll = filtered.filter(t => t.r_multiple != null);
  const durationTradesAll = filtered.filter(t => t.duration_minutes != null && t.pips != null);

  // --- R-Distribution computations ---
  const rValues = rTradesAll.map(t => t.r_multiple!);
  const rWins = rValues.filter(r => r > 0);
  const rLosses = rValues.filter(r => r <= 0);
  const avgWinR = rWins.length > 0 ? rWins.reduce((s, r) => s + r, 0) / rWins.length : 0;
  const avgLossR = rLosses.length > 0 ? Math.abs(rLosses.reduce((s, r) => s + r, 0) / rLosses.length) : 0;
  const winRateR = rTradesAll.length > 0 ? rWins.length / rTradesAll.length : 0;
  const expectancy = rTradesAll.length > 0 ? winRateR * avgWinR - (1 - winRateR) * avgLossR : null;

  // Drawdown
  const ddData = computeDrawdownSeries(filtered, {
    startingBalance: startingBalance ?? undefined,
    totalProfitCurrency: totalProfitCurrency ?? undefined,
  });

  // Duration
  const durWinners = durationTradesAll.filter(t => (t.pips ?? 0) > 0);
  const durLosers = durationTradesAll.filter(t => (t.pips ?? 0) <= 0);
  const avgDurWin = durWinners.length > 0 ? durWinners.reduce((s, t) => s + t.duration_minutes!, 0) / durWinners.length : 0;
  const avgDurLoss = durLosers.length > 0 ? durLosers.reduce((s, t) => s + t.duration_minutes!, 0) / durLosers.length : 0;
  const durRatio = avgDurLoss > 0 ? avgDurWin / avgDurLoss : 0;
  const totalPips = durationTradesAll.reduce((s, t) => s + (t.pips ?? 0), 0);
  const totalHours = durationTradesAll.reduce((s, t) => s + t.duration_minutes!, 0) / 60;
  const pipsPerHour = totalHours > 0 ? totalPips / totalHours : 0;

  const formatDuration = (mins: number) => {
    if (mins < 60) return `${Math.round(mins)}m`;
    if (mins < 1440) return `${(mins / 60).toFixed(1)}h`;
    return `${(mins / 1440).toFixed(1)}d`;
  };

  return (
    <div className="card mb-4">
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
        <h3 className="card-title" style={{ margin: 0 }}>Advanced Analytics</h3>
        {presets.length > 1 && (
          <select
            value={presetFilter}
            onChange={e => setPresetFilter(e.target.value)}
            style={{
              background: 'var(--bg-tertiary)', color: 'var(--text-primary)',
              border: '1px solid var(--border)', borderRadius: 4, padding: '4px 8px',
              fontSize: '0.8rem',
            }}
          >
            <option value="all">All Presets</option>
            {presets.map(p => <option key={p} value={p}>{p}</option>)}
          </select>
        )}
      </div>
      <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 16 }}>
        {filtered.length} closed trade{filtered.length !== 1 ? 's' : ''} analyzed
      </p>

      {/* Feature 1: MAE/MFE Analysis */}
      <div style={sectionHeaderStyle} onClick={() => toggleSection('mae_mfe')}>
        <strong>MAE / MFE Analysis</strong>
        <span>{expandedSections.mae_mfe ? '▼' : '▶'}</span>
      </div>
      {expandedSections.mae_mfe && (
        <div style={{ padding: '12px 0' }}>
          {maeTradesAll.length === 0 ? (
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', padding: '0 16px' }}>
              MAE/MFE tracking starts automatically for new trades. Historical trades will show N/A.
            </p>
          ) : (
            <>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
                {/* MAE vs P&L Scatter */}
                <div>
                  <div style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: 8 }}>MAE vs P&L (pips)</div>
                  <ResponsiveContainer width="100%" height={250}>
                    <ScatterChart margin={{ top: 5, right: 20, bottom: 20, left: 10 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis type="number" dataKey="mae" name="MAE" label={{ value: 'Adverse Excursion (pips)', position: 'bottom', offset: 0, style: { fill: 'var(--text-secondary)', fontSize: 11 } }} tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} />
                      <YAxis type="number" dataKey="pips" name="P&L" label={{ value: 'P&L (pips)', angle: -90, position: 'insideLeft', style: { fill: 'var(--text-secondary)', fontSize: 11 } }} tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} />
                      <Tooltip contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 6, fontSize: '0.8rem' }} formatter={(value: any, name: any) => [Number(value).toFixed(2), name]} />
                      <Scatter
                        data={maeTradesAll.map(t => ({ mae: Math.abs(t.max_adverse_pips!), pips: t.pips!, isWin: t.pips! > 0 }))}
                      >
                        {maeTradesAll.map((t, i) => (
                          <Cell key={i} fill={t.pips! > 0 ? chartColors.green : chartColors.red} opacity={0.7} />
                        ))}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>

                {/* MFE vs P&L Scatter */}
                <div>
                  <div style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: 8 }}>MFE vs P&L (pips)</div>
                  <ResponsiveContainer width="100%" height={250}>
                    <ScatterChart margin={{ top: 5, right: 20, bottom: 20, left: 10 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis type="number" dataKey="mfe" name="MFE" label={{ value: 'Favorable Excursion (pips)', position: 'bottom', offset: 0, style: { fill: 'var(--text-secondary)', fontSize: 11 } }} tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} />
                      <YAxis type="number" dataKey="pips" name="P&L" label={{ value: 'P&L (pips)', angle: -90, position: 'insideLeft', style: { fill: 'var(--text-secondary)', fontSize: 11 } }} tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} />
                      <Tooltip contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 6, fontSize: '0.8rem' }} formatter={(value: any, name: any) => [Number(value).toFixed(2), name]} />
                      <Scatter
                        data={mfeTradesAll.map(t => ({ mfe: t.max_favorable_pips!, pips: t.pips!, isWin: t.pips! > 0 }))}
                      >
                        {mfeTradesAll.map((t, i) => (
                          <Cell key={i} fill={t.pips! > 0 ? chartColors.green : chartColors.red} opacity={0.7} />
                        ))}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16, marginTop: 16 }}>
                {/* Capture Ratio */}
                <div style={statBoxStyle}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 4 }}>Capture Ratio (Winners)</div>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--accent)' }}>
                    {(() => {
                      const winners = mfeTradesAll.filter(t => t.pips! > 0 && t.max_favorable_pips! > 0);
                      if (winners.length === 0) return 'N/A';
                      const ratio = winners.reduce((s, t) => s + (t.pips! / t.max_favorable_pips!), 0) / winners.length;
                      return `${(ratio * 100).toFixed(0)}%`;
                    })()}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: 4 }}>
                    Avg % of MFE captured on winning trades
                  </div>
                </div>

                {/* MAE histogram for winners */}
                <div>
                  <div style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: 8 }}>MAE Distribution (Winners)</div>
                  {(() => {
                    const winnerMAEs = maeTradesAll.filter(t => t.pips! > 0).map(t => Math.abs(t.max_adverse_pips!));
                    const bins = computeHistogramBins(winnerMAEs, 8);
                    if (bins.length === 0) return <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>No data</p>;
                    return (
                      <ResponsiveContainer width="100%" height={200}>
                        <BarChart data={bins} margin={{ top: 5, right: 20, bottom: 20, left: 10 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                          <XAxis dataKey="bin" tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} label={{ value: 'Adverse pips', position: 'bottom', offset: 0, style: { fill: 'var(--text-secondary)', fontSize: 11 } }} />
                          <YAxis tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} />
                          <Tooltip contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 6, fontSize: '0.8rem' }} />
                          <Bar dataKey="count" fill={chartColors.blue} radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    );
                  })()}
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* Feature 2: Rolling & Windowed Performance */}
      <div style={sectionHeaderStyle} onClick={() => toggleSection('rolling')}>
        <strong>Rolling Performance</strong>
        <span>{expandedSections.rolling ? '▼' : '▶'}</span>
      </div>
      {expandedSections.rolling && (
        <div style={{ padding: '12px 0' }}>
          <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap', alignItems: 'center' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Window:</span>
            {['trades', 'time'].map(m => (
              <button key={m} onClick={() => setRollingMode(m as 'trades' | 'time')}
                style={{
                  padding: '4px 10px', fontSize: '0.75rem', borderRadius: 4, cursor: 'pointer',
                  background: rollingMode === m ? 'var(--accent)' : 'var(--bg-tertiary)',
                  color: rollingMode === m ? 'white' : 'var(--text-secondary)',
                  border: '1px solid var(--border)',
                }}
              >{m === 'trades' ? 'By Trades' : 'By Time'}</button>
            ))}
            {rollingMode === 'trades' ? (
              [20, 50, 100].map(n => (
                <button key={n} onClick={() => setRollingWindow(n)}
                  style={{
                    padding: '4px 10px', fontSize: '0.75rem', borderRadius: 4, cursor: 'pointer',
                    background: rollingWindow === n ? 'var(--accent)' : 'var(--bg-tertiary)',
                    color: rollingWindow === n ? 'white' : 'var(--text-secondary)',
                    border: '1px solid var(--border)',
                  }}
                >Last {n}</button>
              ))
            ) : (
              [1, 7, 30].map(n => (
                <button key={n} onClick={() => setTimeWindow(n)}
                  style={{
                    padding: '4px 10px', fontSize: '0.75rem', borderRadius: 4, cursor: 'pointer',
                    background: timeWindow === n ? 'var(--accent)' : 'var(--bg-tertiary)',
                    color: timeWindow === n ? 'white' : 'var(--text-secondary)',
                    border: '1px solid var(--border)',
                  }}
                >{n}d</button>
              ))
            )}
          </div>

          {(() => {
            const rolling = computeRollingMetrics(filtered, rollingWindow, rollingMode, timeWindow);
            const minLen = rollingMode === 'trades' ? rollingWindow : 3;
            if (rolling.length < minLen) {
              return <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Need at least {minLen} trades for rolling metrics.</p>;
            }

            // Compute all-time averages for reference lines
            const pipsArr = filtered.filter(t => t.pips != null);
            const allWinRate = pipsArr.length > 0 ? (pipsArr.filter(t => t.pips! > 0).length / pipsArr.length) * 100 : 50;
            const allAvgPips = pipsArr.length > 0 ? pipsArr.reduce((s, t) => s + t.pips!, 0) / pipsArr.length : 0;

            const chartMargin = { top: 5, right: 20, bottom: 5, left: 10 };
            const tooltipStyle = { background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 6, fontSize: '0.8rem' };

            return (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
                <div>
                  <div style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: 8 }}>Rolling Win Rate (%)</div>
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={rolling} margin={chartMargin}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="idx" tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} />
                      <YAxis domain={[0, 100]} tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} />
                      <Tooltip contentStyle={tooltipStyle} />
                      <ReferenceLine y={allWinRate} stroke="var(--text-secondary)" strokeDasharray="5 5" />
                      <Line type="monotone" dataKey="winRate" stroke={chartColors.blue} dot={false} strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div>
                  <div style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: 8 }}>Rolling Avg Pips</div>
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={rolling} margin={chartMargin}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="idx" tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} />
                      <YAxis tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} />
                      <Tooltip contentStyle={tooltipStyle} />
                      <ReferenceLine y={allAvgPips} stroke="var(--text-secondary)" strokeDasharray="5 5" />
                      <ReferenceLine y={0} stroke="var(--border)" />
                      <Line type="monotone" dataKey="avgPips" stroke={chartColors.green} dot={false} strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div>
                  <div style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: 8 }}>Rolling Expectancy (R)</div>
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={rolling.filter(r => r.expectancy != null)} margin={chartMargin}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="idx" tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} />
                      <YAxis tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} />
                      <Tooltip contentStyle={tooltipStyle} />
                      <ReferenceLine y={0} stroke="var(--border)" />
                      <Line type="monotone" dataKey="expectancy" stroke={chartColors.orange} dot={false} strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div>
                  <div style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: 8 }}>Rolling Avg R</div>
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={rolling.filter(r => r.avgR != null)} margin={chartMargin}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="idx" tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} />
                      <YAxis tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} />
                      <Tooltip contentStyle={tooltipStyle} />
                      <ReferenceLine y={0} stroke="var(--border)" />
                      <Line type="monotone" dataKey="avgR" stroke={chartColors.purple} dot={false} strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            );
          })()}
        </div>
      )}

      {/* Feature 3: R-Multiple Distribution & Expectancy */}
      <div style={sectionHeaderStyle} onClick={() => toggleSection('r_dist')}>
        <strong>R-Multiple Distribution</strong>
        <span>{expandedSections.r_dist ? '▼' : '▶'}</span>
      </div>
      {expandedSections.r_dist && (
        <div style={{ padding: '12px 0' }}>
          {rTradesAll.length === 0 ? (
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>No trades with R-multiple data. Ensure trades have stop loss for R calculation.</p>
          ) : (
            <>
              {/* Expectancy hero */}
              <div style={{ display: 'flex', gap: 16, marginBottom: 16, flexWrap: 'wrap' }}>
                <div style={{ ...statBoxStyle, flex: '1 1 200px' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 4 }}>System Expectancy</div>
                  <div style={{
                    fontSize: '2rem', fontWeight: 700,
                    color: expectancy != null && expectancy > 0 ? 'var(--success)' : expectancy != null && expectancy < 0 ? 'var(--danger)' : 'var(--text-secondary)'
                  }}>
                    {expectancy != null ? `${expectancy >= 0 ? '+' : ''}${expectancy.toFixed(2)}R` : 'N/A'}
                  </div>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', marginTop: 4 }}>
                    {(winRateR * 100).toFixed(0)}% x {avgWinR.toFixed(2)}R - {((1 - winRateR) * 100).toFixed(0)}% x {avgLossR.toFixed(2)}R
                  </div>
                  <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 6 }}>
                    Based on trades with stop loss ({rTradesAll.length} of {filtered.length} total).
                  </div>
                </div>
                <div style={{ ...statBoxStyle, flex: '0 1 120px' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 4 }}>Skew</div>
                  <div style={{
                    fontSize: '1rem', fontWeight: 600,
                    color: expectancy != null && expectancy > 0 ? 'var(--success)' : 'var(--warning)'
                  }}>
                    {expectancy != null && expectancy > 0 && avgWinR > avgLossR ? 'Robust' : 'Fragile'}
                  </div>
                </div>
              </div>

              {/* R-multiple histogram */}
              <div style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: 8 }}>R-Multiple Distribution</div>
              {(() => {
                const bins = computeHistogramBins(rValues, 10);
                return (
                  <ResponsiveContainer width="100%" height={250}>
                    <BarChart data={bins} margin={{ top: 5, right: 20, bottom: 20, left: 10 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis dataKey="bin" tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} label={{ value: 'R-Multiple', position: 'bottom', offset: 0, style: { fill: 'var(--text-secondary)', fontSize: 11 } }} />
                      <YAxis tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} />
                      <Tooltip contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 6, fontSize: '0.8rem' }} />
                      <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                        {bins.map((b, i) => (
                          <Cell key={i} fill={b.midValue >= 0 ? chartColors.green : chartColors.red} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                );
              })()}
            </>
          )}
        </div>
      )}

      {/* Feature 4: Drawdown Analysis */}
      <div style={sectionHeaderStyle} onClick={() => toggleSection('drawdown')}>
        <strong>Drawdown Analysis</strong>
        <span>{expandedSections.drawdown ? '▼' : '▶'}</span>
      </div>
      {expandedSections.drawdown && (
        <div style={{ padding: '12px 0' }}>
          {filtered.filter(t => t.pips != null).length < 2 ? (
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Need at least 2 trades for drawdown analysis.</p>
          ) : (
            <>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 12, marginBottom: 16 }}>
                <div style={statBoxStyle}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Max DD (pips)</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--danger)' }}>{ddData.maxDdPips}</div>
                </div>
                {ddData.maxDdUsd != null && (
                  <div style={statBoxStyle}>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Max DD (USD)</div>
                    <div style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--danger)' }}>{ddData.maxDdUsd}</div>
                  </div>
                )}
                <div style={statBoxStyle}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Max DD (%)</div>
                  {ddData.maxDdPct != null ? (
                    <div style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--danger)' }}>{ddData.maxDdPct}%</div>
                  ) : (
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>{ddData.maxDdPctNote ?? '—'}</div>
                  )}
                </div>
                <div style={statBoxStyle}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Current DD</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 700, color: ddData.currentDd > 0 ? 'var(--danger)' : 'var(--success)' }}>{ddData.currentDd} pips</div>
                </div>
                <div style={statBoxStyle}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Longest DD</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--warning)' }}>{ddData.longestTrades} trades</div>
                  {ddData.longestTime && <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)' }}>{ddData.longestTime}</div>}
                </div>
                <div style={statBoxStyle}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Recovery Factor</div>
                  {ddData.recoveryFactor != null ? (
                    <div style={{ fontSize: '1.2rem', fontWeight: 700, color: ddData.recoveryFactor > 1 ? 'var(--success)' : 'var(--warning)' }}>{ddData.recoveryFactor}</div>
                  ) : (
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>{ddData.recoveryFactorNote ?? 'N/A'}</div>
                  )}
                </div>
              </div>

              {/* Underwater equity curve */}
              <div style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: 8 }}>Underwater Equity Curve</div>
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={ddData.series} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="idx" tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} />
                  <YAxis tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} />
                  <Tooltip contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 6, fontSize: '0.8rem' }} formatter={(value: any) => [`${Number(value).toFixed(2)} pips`, 'Drawdown']} />
                  <ReferenceLine y={0} stroke="var(--border)" />
                  <defs>
                    <linearGradient id="ddGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={chartColors.red} stopOpacity={0.1} />
                      <stop offset="100%" stopColor={chartColors.red} stopOpacity={0.6} />
                    </linearGradient>
                  </defs>
                  <Area type="monotone" dataKey="dd" stroke={chartColors.red} fill="url(#ddGradient)" strokeWidth={1.5} />
                </AreaChart>
              </ResponsiveContainer>
            </>
          )}
        </div>
      )}

      {/* Feature 5: Trade Duration Analysis */}
      <div style={sectionHeaderStyle} onClick={() => toggleSection('duration')}>
        <strong>Trade Duration Analysis</strong>
        <span>{expandedSections.duration ? '▼' : '▶'}</span>
      </div>
      {expandedSections.duration && (
        <div style={{ padding: '12px 0' }}>
          {durationTradesAll.length === 0 ? (
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>No trades with duration data.</p>
          ) : (
            <>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 12, marginBottom: 16 }}>
                <div style={statBoxStyle}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Avg Duration (W)</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--success)' }}>{formatDuration(avgDurWin)}</div>
                </div>
                <div style={statBoxStyle}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Avg Duration (L)</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--danger)' }}>{formatDuration(avgDurLoss)}</div>
                </div>
                <div style={statBoxStyle}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Duration Ratio (W/L)</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 700, color: durRatio < 1 ? 'var(--success)' : 'var(--warning)' }}>{durRatio.toFixed(2)}</div>
                </div>
                <div style={statBoxStyle}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Avg Pips/Hour</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--accent)' }}>{pipsPerHour.toFixed(2)}</div>
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
                {/* Duration vs P&L scatter */}
                <div>
                  <div style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: 8 }}>Duration vs P&L</div>
                  <ResponsiveContainer width="100%" height={250}>
                    <ScatterChart margin={{ top: 5, right: 20, bottom: 20, left: 10 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                      <XAxis type="number" dataKey="dur" name="Duration" label={{ value: 'Duration (min)', position: 'bottom', offset: 0, style: { fill: 'var(--text-secondary)', fontSize: 11 } }} tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} />
                      <YAxis type="number" dataKey="pips" name="P&L" label={{ value: 'P&L (pips)', angle: -90, position: 'insideLeft', style: { fill: 'var(--text-secondary)', fontSize: 11 } }} tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} />
                      <Tooltip contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 6, fontSize: '0.8rem' }} formatter={(value: any, name: any) => [name === 'Duration' ? formatDuration(Number(value)) : Number(value).toFixed(2), name]} />
                      <Scatter data={durationTradesAll.map(t => ({ dur: t.duration_minutes!, pips: t.pips! }))}>
                        {durationTradesAll.map((t, i) => (
                          <Cell key={i} fill={t.pips! > 0 ? chartColors.green : chartColors.red} opacity={0.7} />
                        ))}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>

                {/* Duration histogram */}
                <div>
                  <div style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: 8 }}>Duration Distribution</div>
                  {(() => {
                    const bins = computeHistogramBins(durationTradesAll.map(t => t.duration_minutes!), 8);
                    return (
                      <ResponsiveContainer width="100%" height={250}>
                        <BarChart data={bins} margin={{ top: 5, right: 20, bottom: 20, left: 10 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                          <XAxis dataKey="bin" tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} label={{ value: 'Duration (min)', position: 'bottom', offset: 0, style: { fill: 'var(--text-secondary)', fontSize: 11 } }} />
                          <YAxis tick={{ fill: 'var(--text-secondary)', fontSize: 10 }} />
                          <Tooltip contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 6, fontSize: '0.8rem' }} />
                          <Bar dataKey="count" fill={chartColors.blue} radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    );
                  })()}
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}


// ---------------------------------------------------------------------------
// Logs & Stats Page
// ---------------------------------------------------------------------------

function LogsPage({ profile }: { profile: Profile }) {
  const [stats, setStats] = useState<api.QuickStats | null>(null);
  const [breakdown, setBreakdown] = useState<Record<string, number>>({});
  const [trades, setTrades] = useState<Record<string, unknown>[]>([]);
  const [tradesDisplayCurrency, setTradesDisplayCurrency] = useState<string | undefined>(undefined);
  const [executions, setExecutions] = useState<Record<string, unknown>[]>([]);
  const [syncing, setSyncing] = useState(false);
  const [syncMessage, setSyncMessage] = useState<string | null>(null);
  const [closingTrade, setClosingTrade] = useState<string | null>(null);
  const [confirmClose, setConfirmClose] = useState<Record<string, unknown> | null>(null);
  const [presetStats, setPresetStats] = useState<api.StatsByPreset | null>(null);
  const [expandedPreset, setExpandedPreset] = useState<string | null>(null);
  const [mt5Report, setMt5Report] = useState<api.Mt5Report | null>(null);
  const [mt5ReportExpanded, setMt5ReportExpanded] = useState<string | null>('summary');

  const fetchData = () => {
    api.getQuickStats(profile.name, profile.path).then(setStats).catch(console.error);
    api.getRejectionBreakdown(profile.name).then(setBreakdown).catch(console.error);
    api.getTrades(profile.name, 250, profile.path).then((data) => {
      setTrades(data.trades);
      setTradesDisplayCurrency(data.display_currency);
    }).catch(console.error);
    api.getExecutions(profile.name, 20).then(setExecutions).catch(console.error);
    api.getStatsByPreset(profile.name, profile.path).then(setPresetStats).catch(console.error);
    api.getMt5Report(profile.name, profile.path).then((r) => setMt5Report(r ?? null)).catch(() => setMt5Report(null));
  };

  useEffect(() => {
    fetchData();
  }, [profile.name, profile.path]);

  const handleSync = async () => {
    setSyncing(true);
    setSyncMessage(null);
    try {
      const result = await api.syncTrades(profile.name, profile.path) as {
        status: string;
        trades_updated: number;
        trades_imported?: number;
        position_ids_backfilled?: number;
        profit_backfilled?: number;
      };
      const parts: string[] = [];
      if (result.trades_updated > 0) {
        parts.push(`${result.trades_updated} trade(s) synced`);
      }
      if (result.trades_imported && result.trades_imported > 0) {
        parts.push(`${result.trades_imported} trade(s) imported from broker history`);
      }
      if (result.position_ids_backfilled && result.position_ids_backfilled > 0) {
        parts.push(`${result.position_ids_backfilled} position ID(s) backfilled`);
      }
      if (result.profit_backfilled && result.profit_backfilled > 0) {
        parts.push(`${result.profit_backfilled} profit(s) updated from broker`);
      }
      if (parts.length > 0) {
        setSyncMessage(parts.join(', '));
        fetchData();
      } else {
        setSyncMessage('All trades up to date');
      }
      setTimeout(() => setSyncMessage(null), 4000);
    } catch (e: unknown) {
      setSyncMessage(`Sync error: ${(e as Error).message}`);
    } finally {
      setSyncing(false);
    }
  };

  const handleCloseTrade = async (trade: Record<string, unknown>) => {
    const tradeId = String(trade.trade_id || '');
    if (!tradeId) return;
    
    setClosingTrade(tradeId);
    try {
      const result = await api.closeTrade(profile.name, tradeId, profile.path);
      setSyncMessage(`Closed trade: ${result.pips >= 0 ? '+' : ''}${result.pips.toFixed(2)} pips`);
      fetchData();
      setTimeout(() => setSyncMessage(null), 3000);
    } catch (e: unknown) {
      setSyncMessage(`Close error: ${(e as Error).message}`);
    } finally {
      setClosingTrade(null);
      setConfirmClose(null);
    }
  };

  const totalBreakdown = Object.values(breakdown).reduce((a, b) => a + b, 0);

  // Check if a trade is open (no exit_price)
  const isOpenTrade = (trade: Record<string, unknown>) => {
    return trade.exit_price === null || trade.exit_price === undefined;
  };

  return (
    <div>
      <div className="flex-between mb-4">
        <h2 className="page-title" style={{ marginBottom: 0, borderBottom: 'none', paddingBottom: 0 }}>Logs & Stats</h2>
        <button
          className="btn btn-secondary"
          onClick={handleSync}
          disabled={syncing}
        >
          {syncing ? 'Syncing...' : 'Sync from broker'}
        </button>
      </div>

      {syncMessage && (
        <div className="card mb-4" style={{ borderColor: 'var(--accent)' }}>
          <p>{syncMessage}</p>
        </div>
      )}

      {/* Confirmation Modal */}
      {confirmClose && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0,0,0,0.7)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
        }}>
          <div className="card" style={{ maxWidth: 400, margin: 0 }}>
            <h3 className="card-title">Confirm Close Trade</h3>
            <p style={{ marginBottom: 16 }}>
              Are you sure you want to close this trade?
            </p>
            <div style={{ background: 'var(--bg-tertiary)', padding: 12, borderRadius: 6, marginBottom: 16 }}>
              <p style={{ margin: 0, fontSize: '0.9rem' }}>
                <strong>Side:</strong> {String(confirmClose.side || '').toUpperCase()}<br/>
                <strong>Entry:</strong> {typeof confirmClose.entry_price === 'number' ? confirmClose.entry_price.toFixed(3) : '-'}<br/>
                <strong>Size:</strong> {typeof confirmClose.size_lots === 'number' ? confirmClose.size_lots : '-'} lots
              </p>
            </div>
            <p style={{ color: 'var(--warning)', fontSize: '0.85rem', marginBottom: 16 }}>
              This will send a close order to your broker immediately.
            </p>
            <div className="flex gap-2">
              <button
                className="btn btn-danger"
                onClick={() => handleCloseTrade(confirmClose)}
                disabled={closingTrade !== null}
              >
                {closingTrade ? 'Closing...' : 'Close Trade'}
              </button>
              <button
                className="btn btn-secondary"
                onClick={() => setConfirmClose(null)}
                disabled={closingTrade !== null}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Quick Stats */}
      <div className="card mb-4">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 }}>
          <h3 className="card-title" style={{ margin: 0 }}>Quick Stats</h3>
          {stats && (stats as api.QuickStats).source === 'mt5' && (
            <span style={{
              fontSize: '0.7rem',
              background: 'var(--success)',
              color: 'white',
              padding: '4px 8px',
              borderRadius: 4,
              fontWeight: 600,
            }}>
              FROM BROKER
            </span>
          )}
        </div>
        <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: 4, marginBottom: 16 }}>
          {stats && (stats as api.QuickStats).source === 'mt5'
            ? 'Stats from broker deal history (same as View → Reports in MT5)'
            : 'Stats from local records. Use a broker connection (MT5 or OANDA) to see live report data.'}
        </p>
        <div className="grid-3">
          <div className="stat-box">
            <div className="stat-value">{stats?.closed_trades || 0}</div>
            <div className="stat-label">Closed Trades</div>
            {stats && (stats as api.QuickStats).wins != null && (stats as api.QuickStats).losses != null && (
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: 4 }}>
                <span style={{ color: 'var(--success)' }}>{(stats as api.QuickStats).wins} W</span> / <span style={{ color: 'var(--danger)' }}>{(stats as api.QuickStats).losses} L</span>
              </div>
            )}
          </div>
          <div className="stat-box">
            <div className="stat-value">
              {stats?.win_rate != null ? `${(stats.win_rate * 100).toFixed(1)}%` : '-'}
            </div>
            <div className="stat-label">Win Rate</div>
          </div>
          <div className="stat-box">
            <div className="stat-value">
              {stats?.avg_pips != null ? stats.avg_pips.toFixed(3) : '-'}
            </div>
            <div className="stat-label">Avg Pips</div>
          </div>
        </div>
        {(stats && (stats as api.QuickStats).total_profit != null) || (stats && ((stats as api.QuickStats).total_commission != null || (stats as api.QuickStats).total_swap != null)) ? (
          <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid var(--border)', display: 'flex', flexWrap: 'wrap', gap: 24 }}>
            {stats && (stats as api.QuickStats).total_profit != null && (
              <div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Total Profit</div>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: (stats as api.QuickStats).total_profit! >= 0 ? 'var(--success)' : 'var(--danger)' }}>
                  {(stats as api.QuickStats).total_profit! >= 0 ? '+' : ''}{(stats as api.QuickStats).total_profit!.toFixed(2)} {(stats as api.QuickStats).display_currency || 'USD'}
                </div>
              </div>
            )}
            {stats && (stats as api.QuickStats).total_commission != null && (stats as api.QuickStats).total_commission !== 0 && (
              <div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Commission</div>
                <div style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
                  {(stats as api.QuickStats).total_commission!.toFixed(2)} {(stats as api.QuickStats).display_currency || 'USD'}
                </div>
              </div>
            )}
            {stats && (stats as api.QuickStats).total_swap != null && (stats as api.QuickStats).total_swap !== 0 && (
              <div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Swap</div>
                <div style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
                  {(stats as api.QuickStats).total_swap!.toFixed(2)} {(stats as api.QuickStats).display_currency || 'USD'}
                </div>
              </div>
            )}
          </div>
        ) : null}
        {stats && (stats as api.QuickStats).source === 'database' && (stats as api.QuickStats).trades_without_profit != null && (stats as api.QuickStats).trades_without_profit! > 0 && (
          <p style={{ marginTop: 12, marginBottom: 0, fontSize: '0.85rem', color: 'var(--warning)' }}>
            Some trades lack profit data. Sync from broker and refresh to see stats from broker report.
          </p>
        )}
      </div>

      {/* Equity Curve Chart */}
      <EquityCurveChart profileName={profile.name} profilePath={profile.path} />

      {/* Analytics Sections */}
      <SessionPerformance profileName={profile.name} profilePath={profile.path} />
      <LongShortPerformance profileName={profile.name} profilePath={profile.path} />
      <SpreadPerformance profileName={profile.name} profilePath={profile.path} />

      {/* Advanced Analytics */}
      <AdvancedAnalytics profileName={profile.name} profilePath={profile.path} />

      {/* MT5 Full Report */}
      {mt5Report && (
        <div className="card mb-4">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
            <h3 className="card-title" style={{ margin: 0 }}>Account Report</h3>
            <span style={{ fontSize: '0.65rem', background: 'var(--success)', color: 'white', padding: '3px 6px', borderRadius: 4, fontWeight: 600 }}>
              FROM BROKER
            </span>
          </div>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 16 }}>
            Balance, equity, and closed P/L from your broker (MT5 or OANDA)
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <div
              onClick={() => setMt5ReportExpanded(mt5ReportExpanded === 'summary' ? null : 'summary')}
              style={{ padding: 12, background: 'var(--bg-tertiary)', borderRadius: 6, cursor: 'pointer', border: '1px solid var(--border)' }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <strong>Summary</strong>
                <span>{mt5ReportExpanded === 'summary' ? '▼' : '▶'}</span>
              </div>
              {mt5ReportExpanded === 'summary' && (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12, marginTop: 12 }}>
                  <div><span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Balance</span><br/>{mt5Report.summary.balance.toFixed(2)} {mt5Report.display_currency || 'USD'}</div>
                  <div><span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Equity</span><br/>{mt5Report.summary.equity.toFixed(2)} {mt5Report.display_currency || 'USD'}</div>
                  <div><span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Margin</span><br/>{mt5Report.summary.margin.toFixed(2)} {mt5Report.display_currency || 'USD'}</div>
                  <div><span style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Free Margin</span><br/>{mt5Report.summary.free_margin.toFixed(2)} {mt5Report.display_currency || 'USD'}</div>
                </div>
              )}
            </div>
            <div
              onClick={() => setMt5ReportExpanded(mt5ReportExpanded === 'closed' ? null : 'closed')}
              style={{ padding: 12, background: 'var(--bg-tertiary)', borderRadius: 6, cursor: 'pointer', border: '1px solid var(--border)' }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <strong>Closed P/L</strong>
                <span>{mt5ReportExpanded === 'closed' ? '▼' : '▶'}</span>
              </div>
              {mt5ReportExpanded === 'closed' && (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12, marginTop: 12, fontSize: '0.9rem' }}>
                  <div>Total Profit: <span style={{ color: mt5Report.closed_pl.total_profit >= 0 ? 'var(--success)' : 'var(--danger)', fontWeight: 600 }}>{mt5Report.closed_pl.total_profit >= 0 ? '+' : ''}{mt5Report.closed_pl.total_profit.toFixed(2)} {mt5Report.display_currency || 'USD'}</span></div>
                  <div>Commission: {mt5Report.closed_pl.total_commission.toFixed(2)} {mt5Report.display_currency || 'USD'}</div>
                  <div>Swap: {mt5Report.closed_pl.total_swap.toFixed(2)} {mt5Report.display_currency || 'USD'}</div>
                  <div>Gross Profit: <span style={{ color: 'var(--success)' }}>{mt5Report.closed_pl.gross_profit.toFixed(2)} {mt5Report.display_currency || 'USD'}</span></div>
                  <div>Gross Loss: <span style={{ color: 'var(--danger)' }}>{mt5Report.closed_pl.gross_loss.toFixed(2)} {mt5Report.display_currency || 'USD'}</span></div>
                  <div>Profit Factor: {mt5Report.closed_pl.profit_factor.toFixed(2)}</div>
                  <div>Largest Win: <span style={{ color: 'var(--success)' }}>{mt5Report.closed_pl.largest_profit_trade.toFixed(2)} {mt5Report.display_currency || 'USD'}</span></div>
                  <div>Largest Loss: <span style={{ color: 'var(--danger)' }}>{mt5Report.closed_pl.largest_loss_trade.toFixed(2)} {mt5Report.display_currency || 'USD'}</span></div>
                  <div>Expected Payoff: {mt5Report.closed_pl.expected_payoff.toFixed(2)} {mt5Report.display_currency || 'USD'}</div>
                </div>
              )}
            </div>
            <div
              onClick={() => setMt5ReportExpanded(mt5ReportExpanded === 'longshort' ? null : 'longshort')}
              style={{ padding: 12, background: 'var(--bg-tertiary)', borderRadius: 6, cursor: 'pointer', border: '1px solid var(--border)' }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <strong>Long / Short</strong>
                <span>{mt5ReportExpanded === 'longshort' ? '▼' : '▶'}</span>
              </div>
              {mt5ReportExpanded === 'longshort' && (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12, marginTop: 12 }}>
                  <div>Long: {mt5Report.long_short.long_wins}W / {mt5Report.long_short.long_trades - mt5Report.long_short.long_wins}L ({(mt5Report.long_short.long_win_pct * 100).toFixed(0)}% win)</div>
                  <div>Short: {mt5Report.long_short.short_wins}W / {mt5Report.long_short.short_trades - mt5Report.long_short.short_wins}L ({(mt5Report.long_short.short_win_pct * 100).toFixed(0)}% win)</div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Performance by Preset */}
      {presetStats && Object.keys(presetStats.presets).length > 0 && (
        <div className="card mb-4">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <h3 className="card-title" style={{ margin: 0 }}>Performance by Preset</h3>
            {presetStats.source === 'mt5' && (
              <span style={{ fontSize: '0.65rem', background: 'var(--success)', color: 'white', padding: '3px 6px', borderRadius: 4, fontWeight: 600 }}>
                FROM BROKER
              </span>
            )}
          </div>
          <p style={{ color: 'var(--text-secondary)', marginBottom: 16, marginTop: 8, fontSize: '0.85rem' }}>
            Preset names come from your local DB; win/loss and profit use broker history when available. For best accuracy, run <strong>Sync from broker</strong> so closed trades have position IDs, then refresh.
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {Object.entries(presetStats.presets)
              .sort(([, a], [, b]) => (b.total_pips || 0) - (a.total_pips || 0)) // Sort by total pips
              .map(([presetName, pStats]) => {
                const isExpanded = expandedPreset === presetName;
                const isBestPreset = Object.entries(presetStats.presets)
                  .sort(([, a], [, b]) => (b.total_pips || 0) - (a.total_pips || 0))[0]?.[0] === presetName;
                
                return (
                  <div 
                    key={presetName}
                    style={{ 
                      border: `1px solid ${isBestPreset ? 'var(--success)' : 'var(--border)'}`,
                      borderRadius: 8,
                      overflow: 'hidden',
                      background: isBestPreset ? 'rgba(34, 197, 94, 0.05)' : 'var(--bg-secondary)'
                    }}
                  >
                    <div 
                      onClick={() => setExpandedPreset(isExpanded ? null : presetName)}
                      style={{ 
                        padding: 16, 
                        cursor: 'pointer',
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center'
                      }}
                    >
                      <div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <span style={{ fontWeight: 600 }}>{presetName}</span>
                          {isBestPreset && (
                            <span style={{ 
                              fontSize: '0.65rem', 
                              background: 'var(--success)', 
                              color: 'white', 
                              padding: '2px 6px', 
                              borderRadius: 4,
                              fontWeight: 700
                            }}>
                              BEST
                            </span>
                          )}
                        </div>
                        <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: 4 }}>
                          {pStats.total_trades} trades | 
                          Win: {pStats.win_rate != null ? `${(pStats.win_rate * 100).toFixed(0)}%` : '-'} | 
                          Total: <span style={{ color: (pStats.total_pips || 0) >= 0 ? 'var(--success)' : 'var(--danger)' }}>
                            {pStats.total_pips >= 0 ? '+' : ''}{pStats.total_pips?.toFixed(1) || 0} pips
                          </span>
                          {pStats.total_profit != null && (
                            <> | Profit: <span style={{ color: pStats.total_profit >= 0 ? 'var(--success)' : 'var(--danger)' }}>
                              {pStats.total_profit >= 0 ? '+' : ''}{pStats.total_profit.toFixed(2)} {presetStats.display_currency || 'USD'}
                            </span></>
                          )}
                          {pStats.total_commission != null && pStats.total_commission !== 0 && (
                            <> | Commission: {pStats.total_commission.toFixed(2)} {presetStats.display_currency || 'USD'}</>
                          )}
                        </div>
                      </div>
                      <span style={{ fontSize: '1.2rem' }}>{isExpanded ? '▼' : '▶'}</span>
                    </div>
                    
                    {isExpanded && (
                      <div style={{ 
                        padding: '0 16px 16px 16px', 
                        borderTop: '1px solid var(--border)',
                        paddingTop: 16 
                      }}>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: 12 }}>
                          <div style={{ padding: 10, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                            <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Total Trades</div>
                            <div style={{ fontWeight: 700 }}>{pStats.total_trades}</div>
                          </div>
                          <div style={{ padding: 10, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                            <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Win / Loss</div>
                            <div style={{ fontWeight: 700 }}>
                              <span style={{ color: 'var(--success)' }}>{pStats.wins}</span> / 
                              <span style={{ color: 'var(--danger)' }}>{pStats.losses}</span>
                            </div>
                          </div>
                          <div style={{ padding: 10, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                            <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Win Rate</div>
                            <div style={{ fontWeight: 700, color: (pStats.win_rate || 0) >= 0.5 ? 'var(--success)' : 'var(--warning)' }}>
                              {pStats.win_rate != null ? `${(pStats.win_rate * 100).toFixed(1)}%` : '-'}
                            </div>
                          </div>
                          <div style={{ padding: 10, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                            <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Total Pips</div>
                            <div style={{ fontWeight: 700, color: (pStats.total_pips || 0) >= 0 ? 'var(--success)' : 'var(--danger)' }}>
                              {pStats.total_pips >= 0 ? '+' : ''}{pStats.total_pips?.toFixed(2) || 0}
                            </div>
                          </div>
                          {pStats.total_profit != null && (
                            <div style={{ padding: 10, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                              <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Total Profit</div>
                              <div style={{ fontWeight: 700, color: pStats.total_profit >= 0 ? 'var(--success)' : 'var(--danger)' }}>
                                {pStats.total_profit >= 0 ? '+' : ''}{pStats.total_profit.toFixed(2)} {presetStats.display_currency || 'USD'}
                              </div>
                            </div>
                          )}
                          {pStats.total_commission != null && pStats.total_commission !== 0 && (
                            <div style={{ padding: 10, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                              <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Commission</div>
                              <div style={{ fontWeight: 700 }}>{pStats.total_commission.toFixed(2)} {presetStats.display_currency || 'USD'}</div>
                            </div>
                          )}
                          <div style={{ padding: 10, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                            <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Avg Pips</div>
                            <div style={{ fontWeight: 700 }}>{pStats.avg_pips?.toFixed(2) || '-'}</div>
                          </div>
                          <div style={{ padding: 10, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                            <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Avg R:R</div>
                            <div style={{ fontWeight: 700 }}>{pStats.avg_rr?.toFixed(2) || '-'}</div>
                          </div>
                          <div style={{ padding: 10, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                            <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Best Trade</div>
                            <div style={{ fontWeight: 700, color: 'var(--success)' }}>
                              {pStats.best_trade != null ? `+${pStats.best_trade.toFixed(1)} pips` : '-'}
                            </div>
                          </div>
                          <div style={{ padding: 10, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                            <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Worst Trade</div>
                            <div style={{ fontWeight: 700, color: 'var(--danger)' }}>
                              {pStats.worst_trade != null ? `${pStats.worst_trade.toFixed(1)} pips` : '-'}
                            </div>
                          </div>
                          <div style={{ padding: 10, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                            <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Win Streak</div>
                            <div style={{ fontWeight: 700 }}>{pStats.win_streak}</div>
                          </div>
                          <div style={{ padding: 10, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                            <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Loss Streak</div>
                            <div style={{ fontWeight: 700 }}>{pStats.loss_streak}</div>
                          </div>
                          <div style={{ padding: 10, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                            <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Profit Factor</div>
                            <div style={{ fontWeight: 700, color: (pStats.profit_factor || 0) >= 1 ? 'var(--success)' : 'var(--danger)' }}>
                              {pStats.profit_factor?.toFixed(2) || '-'}
                            </div>
                          </div>
                          <div style={{ padding: 10, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                            <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Max Drawdown</div>
                            <div style={{ fontWeight: 700, color: 'var(--warning)' }}>
                              {pStats.max_drawdown?.toFixed(1) || 0} pips
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
          </div>
        </div>
      )}

      {/* Rejection Breakdown */}
      <div className="card mb-4">
        <h3 className="card-title">Rejection Breakdown</h3>
        {Object.entries(breakdown).length === 0 ? (
          <p style={{ color: 'var(--text-secondary)' }}>No executions yet.</p>
        ) : (
          Object.entries(breakdown).map(([reason, count]) => (
            <div key={reason} className="breakdown-bar">
              <span className="breakdown-label">{reason}</span>
              <div className="breakdown-value">
                <div
                  className="breakdown-fill"
                  style={{ width: `${(count / totalBreakdown) * 100}%` }}
                />
                <span className="breakdown-count">{count}</span>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Trades Table */}
      <div className="card mb-4">
        <h3 className="card-title">Recent Trades</h3>
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Side</th>
                <th>Entry</th>
                <th>Exit</th>
                <th>Pips</th>
                <th>R</th>
                <th>Profit</th>
                <th>Source</th>
                <th>Exit Reason</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {trades.length === 0 ? (
                <tr>
                  <td colSpan={10} style={{ textAlign: 'center', color: 'var(--text-secondary)' }}>
                    No trades yet.
                  </td>
                </tr>
              ) : (
                trades.map((t, i) => {
                  const openedBy = String(t.opened_by || '');
                  const exitReason = String(t.exit_reason || '');
                  
                  // Format exit_reason for display
                  const formatExitReason = (reason: string): { text: string; color: string } => {
                    if (!reason) return { text: '-', color: 'inherit' };
                    switch (reason) {
                      case 'hit_take_profit':
                        return { text: 'TP', color: 'var(--success)' };
                      case 'hit_stop_loss':
                        return { text: 'SL', color: 'var(--danger)' };
                      case 'user_closed_early':
                        return { text: 'User', color: 'var(--warning)' };
                      case 'UI_manual_close':
                        return { text: 'UI', color: 'var(--accent)' };
                      case 'mt5_history_import':
                        return { text: 'Imported', color: 'var(--text-secondary)' };
                      default:
                        return { text: reason.replace(/_/g, ' '), color: 'var(--text-secondary)' };
                    }
                  };
                  
                  const exitDisplay = formatExitReason(exitReason);
                  
                  return (
                    <tr key={i}>
                      <td>{String(t.timestamp_utc || '').slice(0, 19)}</td>
                      <td>{String(t.side || '')}</td>
                      <td>{typeof t.entry_price === 'number' ? t.entry_price.toFixed(3) : '-'}</td>
                      <td>{typeof t.exit_price === 'number' ? t.exit_price.toFixed(3) : '-'}</td>
                      <td style={{ color: typeof t.pips === 'number' ? (t.pips >= 0 ? 'var(--success)' : 'var(--danger)') : 'inherit' }}>
                        {typeof t.pips === 'number' ? t.pips.toFixed(2) : '-'}
                      </td>
                      <td>{typeof t.r_multiple === 'number' ? t.r_multiple.toFixed(2) : '-'}</td>
                      <td style={{ color: typeof t.profit_display === 'number' ? (t.profit_display >= 0 ? 'var(--success)' : 'var(--danger)') : 'inherit' }}>
                        {typeof t.profit_display === 'number'
                          ? `${t.profit_display >= 0 ? '+' : ''}${t.profit_display.toFixed(2)} ${tradesDisplayCurrency || (stats as api.QuickStats)?.display_currency || 'USD'}`
                          : '-'}
                      </td>
                      <td>
                        <span style={{ 
                          fontSize: '0.75rem', 
                          padding: '2px 6px', 
                          borderRadius: 4,
                          background: openedBy === 'program' ? 'rgba(59, 130, 246, 0.2)' : 'rgba(156, 163, 175, 0.2)',
                          color: openedBy === 'program' ? 'var(--accent)' : 'var(--text-secondary)'
                        }}>
                          {openedBy === 'program' ? 'Bot' : openedBy === 'manual' ? 'Manual' : '-'}
                        </span>
                      </td>
                      <td>
                        <span style={{ fontSize: '0.75rem', color: exitDisplay.color }}>
                          {exitDisplay.text}
                        </span>
                      </td>
                      <td>
                        {isOpenTrade(t) && (t.mt5_order_id || t.mt5_position_id) ? (
                          <button
                            className="btn btn-danger"
                            style={{ padding: '4px 8px', fontSize: '0.75rem' }}
                            onClick={() => setConfirmClose(t)}
                            disabled={closingTrade === String(t.trade_id)}
                          >
                            Close
                          </button>
                        ) : isOpenTrade(t) ? (
                          <span style={{ color: 'var(--text-secondary)', fontSize: '0.75rem' }}>No position ID</span>
                        ) : (
                          <span style={{ color: 'var(--success)', fontSize: '0.75rem' }}>Closed</span>
                        )}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Executions Table */}
      <div className="card">
        <h3 className="card-title">Recent Executions</h3>
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Mode</th>
                <th>Placed</th>
                <th>Reason</th>
              </tr>
            </thead>
            <tbody>
              {executions.length === 0 ? (
                <tr>
                  <td colSpan={4} style={{ textAlign: 'center', color: 'var(--text-secondary)' }}>
                    No executions yet.
                  </td>
                </tr>
              ) : (
                executions.map((e, i) => (
                  <tr key={i}>
                    <td>{String(e.timestamp_utc || '').slice(0, 19)}</td>
                    <td>{String(e.mode || '')}</td>
                    <td>{e.placed ? 'Yes' : 'No'}</td>
                    <td style={{ maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {String(e.reason || '')}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Guide Page - How to Use
// ---------------------------------------------------------------------------

function GuidePage() {
  const [expandedSection, setExpandedSection] = useState<string | null>('quickstart');

  const toggleSection = (section: string) => {
    setExpandedSection(expandedSection === section ? null : section);
  };

  const SectionCard = ({ id, title, icon, children }: { id: string; title: string; icon: string; children: React.ReactNode }) => (
    <div className="card mb-4">
      <div 
        style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }}
        onClick={() => toggleSection(id)}
      >
        <h3 className="card-title" style={{ margin: 0, display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontSize: '1.3rem' }}>{icon}</span>
          {title}
        </h3>
        <span style={{ color: 'var(--accent)', fontSize: '1.2rem' }}>
          {expandedSection === id ? '−' : '+'}
        </span>
      </div>
      {expandedSection === id && (
        <div style={{ marginTop: 16 }}>
          {children}
        </div>
      )}
    </div>
  );

  return (
    <div>
      <h2 className="page-title">Help / Guide</h2>
      <p style={{ color: 'var(--text-secondary)', marginBottom: 24 }}>
        Welcome to the USDJPY Trading Assistant! This guide will help you get started and understand how to use the bot effectively.
      </p>

      {/* Quick Start */}
      <SectionCard id="quickstart" title="Quick Start (5 Steps)" icon="🚀">
        <div style={{ padding: 16, background: 'var(--bg-tertiary)', borderRadius: 8, marginBottom: 16 }}>
          <ol style={{ margin: 0, paddingLeft: 24, lineHeight: 2.2 }}>
            <li>
              <strong>Select or create an account</strong> using the sidebar dropdown. Each account has its own settings and trade history.
            </li>
            <li>
              <strong>Go to Presets</strong> and pick a preset that matches your style (start with <em>Conservative Scalping</em> if unsure), then click <strong>Apply Preset</strong>.
            </li>
            <li>
              <strong>Go to Run / Status</strong>, set the mode to <strong>ARMED_AUTO_DEMO</strong>, then click <strong>Start Loop</strong>.
            </li>
            <li>
              <strong>Watch the log</strong> - the bot will show heartbeats every few seconds and log its trading decisions.
            </li>
            <li>
              <strong>Check Logs & Stats</strong> to see your trades, win rate, and performance metrics.
            </li>
          </ol>
        </div>
        <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
          <strong>Tip:</strong> The bot only trades on demo accounts for safety. For MT5, have the terminal running and connected; for OANDA, set your API key in Profile Editor.
        </p>
      </SectionCard>

      {/* Understanding Presets */}
      <SectionCard id="presets" title="Understanding Presets" icon="⚙️">
        <p style={{ marginBottom: 16 }}>
          Presets are pre-configured trading strategies. Each preset has different characteristics:
        </p>
        <div className="table-container" style={{ marginBottom: 16 }}>
          <table>
            <thead>
              <tr>
                <th>Preset</th>
                <th>Style</th>
                <th>Trades/Day</th>
                <th>Target</th>
                <th>Risk Level</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Conservative Scalping</strong></td>
                <td>Quick trades</td>
                <td>5-10</td>
                <td>8 pips</td>
                <td style={{ color: 'var(--success)' }}>Low</td>
              </tr>
              <tr>
                <td><strong>Aggressive Scalping</strong></td>
                <td>Quick trades</td>
                <td>15-25</td>
                <td>8 pips</td>
                <td style={{ color: 'var(--danger)' }}>High</td>
              </tr>
              <tr>
                <td><strong>Moderate Swing</strong></td>
                <td>Longer holds</td>
                <td>2-5</td>
                <td>20 pips</td>
                <td style={{ color: 'var(--warning)' }}>Medium</td>
              </tr>
              <tr>
                <td><strong>RSI Mean Reversion</strong></td>
                <td>Counter-trend</td>
                <td>5-10</td>
                <td>10 pips</td>
                <td style={{ color: 'var(--warning)' }}>Medium</td>
              </tr>
              <tr>
                <td><strong>Custom</strong></td>
                <td>Your choice</td>
                <td>Variable</td>
                <td>Variable</td>
                <td>Configurable</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
          <strong>Recommendation:</strong> Start with <em>Conservative Scalping</em> to see how the bot works with minimal risk. 
          Once comfortable, try other presets or create a custom configuration.
        </p>
      </SectionCard>

      {/* Reading the Dashboard */}
      <SectionCard id="dashboard" title="Reading the Dashboard" icon="📊">
        <div className="grid-2" style={{ gap: 16 }}>
          <div style={{ padding: 16, background: 'var(--bg-tertiary)', borderRadius: 8 }}>
            <h4 style={{ margin: '0 0 12px 0', color: 'var(--accent)' }}>Run / Status Page</h4>
            <ul style={{ margin: 0, paddingLeft: 20, lineHeight: 1.8 }}>
              <li><strong>Loop Status:</strong> Shows if the trading loop is running or stopped</li>
              <li><strong>Mode:</strong> DISARMED (no trades), ARMED_MANUAL_CONFIRM (prompts before trading), ARMED_AUTO_DEMO (fully automated)</li>
              <li><strong>Kill Switch:</strong> Emergency stop - prevents any new trades</li>
              <li><strong>Loop Log:</strong> Real-time output showing heartbeats and decisions</li>
            </ul>
          </div>
          <div style={{ padding: 16, background: 'var(--bg-tertiary)', borderRadius: 8 }}>
            <h4 style={{ margin: '0 0 12px 0', color: 'var(--accent)' }}>Logs & Stats Page</h4>
            <ul style={{ margin: 0, paddingLeft: 20, lineHeight: 1.8 }}>
              <li><strong>Quick Stats:</strong> Closed trades, win rate, average pips</li>
              <li><strong>Rejection Breakdown:</strong> Why trades weren't taken (spread, filters, etc.)</li>
              <li><strong>Recent Trades:</strong> Your trade history with P/L</li>
              <li><strong>Sync from broker:</strong> Updates trade data from your broker (MT5 or OANDA)</li>
            </ul>
          </div>
        </div>
      </SectionCard>

      {/* Analysis Page */}
      <SectionCard id="analysis" title="Technical Analysis Page" icon="📈">
        <p style={{ marginBottom: 16 }}>
          The Analysis page provides real-time technical analysis across three timeframes (H4, M15, M1):
        </p>
        <ul style={{ margin: '0 0 16px 0', paddingLeft: 20, lineHeight: 1.8 }}>
          <li><strong>Regime:</strong> Current market trend - BULL (uptrend), BEAR (downtrend), or SIDEWAYS</li>
          <li><strong>RSI:</strong> Relative Strength Index - Oversold (below 30, potential buy), Overbought (above 70, potential sell), Neutral</li>
          <li><strong>MACD:</strong> Momentum indicator - Positive (bullish momentum), Negative (bearish momentum)</li>
          <li><strong>ATR:</strong> Average True Range - Elevated (high volatility), Normal, Low (quiet market)</li>
        </ul>
        <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
          Click on any timeframe card to expand and see detailed indicator values. The page auto-refreshes every 30 seconds.
        </p>
      </SectionCard>

      {/* Common Questions */}
      <SectionCard id="faq" title="Common Questions" icon="❓">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div style={{ padding: 16, background: 'var(--bg-tertiary)', borderRadius: 8 }}>
            <h4 style={{ margin: '0 0 8px 0', color: 'var(--warning)' }}>Why isn't the bot trading?</h4>
            <p style={{ margin: 0, lineHeight: 1.6 }}>
              Check these common reasons:
              <br />• <strong>Spread too wide:</strong> The bot waits for good spreads to avoid slippage
              <br />• <strong>Alignment filters:</strong> Multi-timeframe conditions must match
              <br />• <strong>Max trades reached:</strong> Daily trade limit hit
              <br />• <strong>Kill switch on:</strong> Check the Run page
              <br />• <strong>Market closed:</strong> Forex markets close on weekends
            </p>
          </div>
          <div style={{ padding: 16, background: 'var(--bg-tertiary)', borderRadius: 8 }}>
            <h4 style={{ margin: '0 0 8px 0', color: 'var(--warning)' }}>How do I close a trade?</h4>
            <p style={{ margin: 0, lineHeight: 1.6 }}>
              Go to <strong>Logs & Stats</strong>, find your open trade in the Recent Trades table, 
              and click the <strong>Close</strong> button. You'll be asked to confirm before the close order is sent.
            </p>
          </div>
          <div style={{ padding: 16, background: 'var(--bg-tertiary)', borderRadius: 8 }}>
            <h4 style={{ margin: '0 0 8px 0', color: 'var(--warning)' }}>What does the kill switch do?</h4>
            <p style={{ margin: 0, lineHeight: 1.6 }}>
              The kill switch <strong>immediately stops all new trades</strong> from being placed. 
              It does not close existing positions - use the Close button for that. 
              Toggle it on if you need to pause the bot quickly.
            </p>
          </div>
          <div style={{ padding: 16, background: 'var(--bg-tertiary)', borderRadius: 8 }}>
            <h4 style={{ margin: '0 0 8px 0', color: 'var(--warning)' }}>My trade history is missing?</h4>
            <p style={{ margin: 0, lineHeight: 1.6 }}>
Click <strong>Sync from broker</strong> on the Logs & Stats page. This will import any trades
              that were opened or closed in your broker platform, including trades closed by hitting TP/SL.
            </p>
          </div>
        </div>
      </SectionCard>

      {/* Glossary */}
      <SectionCard id="glossary" title="Glossary" icon="📖">
        <div className="grid-2" style={{ gap: 12 }}>
          {[
            { term: 'Pip', def: 'Smallest price move. For USDJPY, 1 pip = 0.01 (e.g., 154.50 to 154.51)' },
            { term: 'Lot', def: 'Trade size. 0.01 = micro lot (1,000 units), 0.1 = mini lot (10,000 units), 1.0 = standard lot (100,000 units)' },
            { term: 'Spread', def: 'Difference between buy (ask) and sell (bid) price. Lower = better.' },
            { term: 'EMA', def: 'Exponential Moving Average - weighted average of recent prices, responds quickly to changes' },
            { term: 'SMA', def: 'Simple Moving Average - equal-weighted average of prices over a period' },
            { term: 'RSI', def: 'Relative Strength Index (0-100). Below 30 = oversold, above 70 = overbought' },
            { term: 'MACD', def: 'Moving Average Convergence Divergence - momentum indicator showing trend direction' },
            { term: 'ATR', def: 'Average True Range - measures market volatility' },
            { term: 'Regime', def: 'Current market trend: bullish (up), bearish (down), or sideways' },
            { term: 'Stop Loss (SL)', def: 'Price level where trade closes to limit losses' },
            { term: 'Take Profit (TP)', def: 'Price level where trade closes to lock in profits' },
            { term: 'R-Multiple', def: 'Risk-adjusted return. R of 2 means you made 2x your risked amount' },
          ].map((item, i) => (
            <div key={i} style={{ padding: 12, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
              <strong style={{ color: 'var(--accent)' }}>{item.term}:</strong>
              <span style={{ marginLeft: 8, color: 'var(--text-secondary)', fontSize: '0.9rem' }}>{item.def}</span>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Safety Notice */}
      <div className="card" style={{ borderColor: 'var(--warning)', background: 'rgba(245, 158, 11, 0.1)' }}>
        <h3 style={{ margin: '0 0 12px 0', color: 'var(--warning)', display: 'flex', alignItems: 'center', gap: 8 }}>
          ⚠️ Important Safety Notice
        </h3>
        <ul style={{ margin: 0, paddingLeft: 20, lineHeight: 1.8 }}>
          <li>This bot is designed for <strong>demo trading only</strong>. It will refuse to trade on live accounts.</li>
          <li>Trading carries risk. Past performance does not guarantee future results.</li>
          <li>Always monitor the bot and understand what it's doing before trusting it with larger positions.</li>
          <li>Start with conservative settings and small position sizes while you learn.</li>
        </ul>
      </div>
    </div>
  );
}
