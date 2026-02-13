import { useEffect, useState, useRef } from 'react';
import { createChart, createSeriesMarkers, IChartApi, ISeriesApi, CandlestickData, Time, CandlestickSeries, LineSeries } from 'lightweight-charts';
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
  entry_time?: number;  // Unix seconds for chart
  exit_time?: number;   // Unix seconds for chart
  exit_price?: number;
}

interface CandlestickChartProps {
  ohlc: api.OhlcBar[];
  trades: ChartTrade[];
  emaFast?: { time: number; value: number }[];
  emaSlow?: { time: number; value: number }[];
  emaStack?: Record<string, { time: number; value: number }[]>;
  height?: number;
  onCloseTrade?: (trade: ChartTrade) => void;
}

function CandlestickChart({ ohlc, trades, emaFast, emaSlow, emaStack, height = 300, onCloseTrade }: CandlestickChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current || ohlc.length === 0) return;

    // Create chart
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

    // Add candlestick series (v4+ API)
    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#10b981',
      downColor: '#ef4444',
      borderUpColor: '#10b981',
      borderDownColor: '#ef4444',
      wickUpColor: '#10b981',
      wickDownColor: '#ef4444',
    });

    seriesRef.current = candlestickSeries;

    // Convert OHLC data to chart format
    const chartData: CandlestickData<Time>[] = ohlc.map((bar) => ({
      time: bar.time as Time,
      open: bar.open,
      high: bar.high,
      low: bar.low,
      close: bar.close,
    }));

    candlestickSeries.setData(chartData);

    // Add EMA lines (no value labels or price lines on the right - lines only)
    const emaSeriesOptions = { lastValueVisible: false, priceLineVisible: false };
    const emaColors: Record<string, string> = { ema8: '#f59e0b', ema13: '#3b82f6', ema21: '#8b5cf6', ema34: '#06b6d4', ema50: '#10b981', ema89: '#6366f1', ema200: '#a855f7' };
    if (emaFast && emaFast.length > 0) {
      const emaSeries = chart.addSeries(LineSeries, { color: '#3b82f6', lineWidth: 2, title: 'EMA Fast', ...emaSeriesOptions });
      emaSeries.setData(emaFast.map(d => ({ time: d.time as Time, value: d.value })));
    }
    if (emaSlow && emaSlow.length > 0) {
      const emaSeries = chart.addSeries(LineSeries, { color: '#10b981', lineWidth: 2, title: 'SMA Slow', ...emaSeriesOptions });
      emaSeries.setData(emaSlow.map(d => ({ time: d.time as Time, value: d.value })));
    }
    if (emaStack) {
      Object.entries(emaStack).forEach(([key, arr]) => {
        if (arr && arr.length > 0) {
          const color = emaColors[key] || '#94a3b8';
          const emaSeries = chart.addSeries(LineSeries, { color, lineWidth: 1, title: key.toUpperCase(), ...emaSeriesOptions });
          emaSeries.setData(arr.map(d => ({ time: d.time as Time, value: d.value })));
        }
      });
    }

    // Markers: blue arrow for buy, red arrow for sell; no text
    const markers: { time: Time; position: 'aboveBar' | 'belowBar'; color: string; shape: 'arrowUp' | 'arrowDown' | 'circle'; text: string }[] = [];
    trades.forEach((t) => {
      const isBuy = t.side.toLowerCase() === 'buy';
      const buyColor = '#3b82f6';
      const sellColor = '#ef4444';
      const color = isBuy ? buyColor : sellColor;
      if (t.entry_time != null) {
        markers.push({
          time: t.entry_time as Time,
          position: isBuy ? 'belowBar' : 'aboveBar',
          color,
          shape: isBuy ? 'arrowUp' : 'arrowDown',
          text: '',
        });
      }
      if (t.exit_time != null) {
        markers.push({
          time: t.exit_time as Time,
          position: isBuy ? 'aboveBar' : 'belowBar',
          color,
          shape: 'circle',
          text: '',
        });
      }
    });
    if (markers.length > 0) {
      const seriesMarkers = createSeriesMarkers(candlestickSeries);
      seriesMarkers.setMarkers(markers);
    }

    // Fit content
    chart.timeScale().fitContent();

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({ 
          width: chartContainerRef.current.clientWidth 
        });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [ohlc, trades, emaFast, emaSlow, emaStack, height]);

  if (ohlc.length === 0) {
    return (
      <div style={{ 
        height: height, 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        background: 'var(--bg-tertiary)',
        borderRadius: 6,
        color: 'var(--text-secondary)'
      }}>
        No chart data available
      </div>
    );
  }

  const activeTrades = trades.filter(t => !t.exit_time && !t.exit_price);

  return (
    <div style={{ position: 'relative', width: '100%', height: height }}>
      <div
        ref={chartContainerRef}
        style={{
          width: '100%',
          height: height,
          borderRadius: 6,
          overflow: 'hidden'
        }}
      />
      {/* Trade legend overlay */}
      {onCloseTrade && activeTrades.length > 0 && (
        <div style={{
          position: 'absolute',
          top: 8,
          right: 8,
          background: 'rgba(26, 26, 46, 0.9)',
          border: '1px solid var(--border)',
          borderRadius: 6,
          padding: 8,
          fontSize: '0.75rem',
          maxWidth: 180,
          zIndex: 10,
        }}>
          <div style={{ fontWeight: 600, marginBottom: 6, color: 'var(--text-secondary)' }}>
            Active Trades
          </div>
          {activeTrades.map(t => (
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
                onClick={() => onCloseTrade(t)}
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
  const getEmaLegend = (tfData: api.TaTimeframe): string => {
    const parts: string[] = [];
    if (tfData.ema_stack && Object.keys(tfData.ema_stack).length > 0) {
      const periods = Object.keys(tfData.ema_stack)
        .map((k) => k.replace(/^ema/i, ''))
        .filter(Boolean)
        .sort((a, b) => parseInt(a, 10) - parseInt(b, 10));
      if (periods.length > 0) {
        parts.push(`EMA ${periods.join('/')}`);
      }
    }
    if (tfData.ema_fast && tfData.ema_fast.length > 0 && tfData.ema_slow && tfData.ema_slow.length > 0) {
      parts.push('SMA slow');
    }
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
                  <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 8, marginBottom: 12 }}>
                    <h4 style={{ margin: 0, color: 'var(--accent)', fontSize: '0.9rem' }}>
                      {tf} Chart ({timeframeLabel[tf] || tf})
                    </h4>
                    {getEmaLegend(tfData) && (
                      <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>
                        {getEmaLegend(tfData)}
                      </span>
                    )}
                    {chartTrades.length > 0 && (
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                        ({chartTrades.filter(t => !t.exit_time).length} open, {chartTrades.length} total)
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
                  <CandlestickChart
                    ohlc={tfData.ohlc || []}
                    trades={chartTrades}
                    emaFast={tfData.ema_fast}
                    emaSlow={tfData.ema_slow}
                    emaStack={tfData.ema_stack}
                    height={enlargedTf === tf ? 520 : 280}
                    onCloseTrade={(trade) => setConfirmCloseTrade(trade)}
                  />
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
                    <h4 style={{ margin: '0 0 12px 0', color: 'var(--accent)', fontSize: '0.9rem' }}>Price Levels (100-bar)</h4>
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

            {/* Chart */}
            <div style={{ flex: 1, padding: 16, overflow: 'hidden' }}>
              <CandlestickChart
                ohlc={tfData.ohlc || []}
                trades={chartTrades}
                emaFast={tfData.ema_fast}
                emaSlow={tfData.ema_slow}
                emaStack={tfData.ema_stack}
                height={chartHeight}
                onCloseTrade={(trade) => setConfirmCloseTrade(trade)}
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
  // Trial #4: RSI Divergence Detection (M3-based)
  rsi_divergence_enabled: boolean;
  rsi_divergence_period: number;
  rsi_divergence_lookback_bars: number;
  rsi_divergence_swing_window: number;
  rsi_divergence_block_minutes: number;
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
      // Trial #4: RSI Divergence
      let rsiDivergenceEnabled = false;
      let rsiDivergencePeriod = 14;
      let rsiDivergenceLookbackBars = 50;
      let rsiDivergenceSwingWindow = 5;
      let rsiDivergenceBlockMinutes = 5.0;
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
          // Trial #4 RSI divergence
          if ('rsi_divergence_enabled' in pol) {
            rsiDivergenceEnabled = pol.rsi_divergence_enabled as boolean;
            rsiDivergencePeriod = (pol.rsi_divergence_period as number) ?? 14;
            rsiDivergenceLookbackBars = (pol.rsi_divergence_lookback_bars as number) ?? 50;
            rsiDivergenceSwingWindow = (pol.rsi_divergence_swing_window as number) ?? 5;
            rsiDivergenceBlockMinutes = (pol.rsi_divergence_block_minutes as number) ?? 5.0;
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
        // Trial #4: RSI divergence
        rsi_divergence_enabled: rsiDivergenceEnabled,
        rsi_divergence_period: rsiDivergencePeriod,
        rsi_divergence_lookback_bars: rsiDivergenceLookbackBars,
        rsi_divergence_swing_window: rsiDivergenceSwingWindow,
        rsi_divergence_block_minutes: rsiDivergenceBlockMinutes,
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
          updates.rolling_danger_lookback_bars = Math.max(20, Math.min(500, editedSettings.rolling_danger_lookback_bars));
          updates.rolling_danger_zone_pct = Math.max(0.05, Math.min(0.50, editedSettings.rolling_danger_zone_pct));
          // RSI divergence settings
          updates.rsi_divergence_enabled = editedSettings.rsi_divergence_enabled;
          updates.rsi_divergence_period = Math.max(5, Math.min(50, editedSettings.rsi_divergence_period));
          updates.rsi_divergence_lookback_bars = Math.max(20, Math.min(200, editedSettings.rsi_divergence_lookback_bars));
          updates.rsi_divergence_swing_window = Math.max(2, Math.min(15, editedSettings.rsi_divergence_swing_window));
          updates.rsi_divergence_block_minutes = Math.max(1, Math.min(30, editedSettings.rsi_divergence_block_minutes));
        }
        return Object.keys(updates).length > 0 ? { ...pol, ...updates } : pol;
      }) || [];

      const newExecution = {
        ...(execution || {}),
        loop_poll_seconds: Math.max(0.5, editedSettings.loop_poll_seconds),
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
                      step="0.5"
                      min="0.5"
                      value={editedSettings.loop_poll_seconds}
                      onChange={(e) => setEditedSettings({ ...editedSettings, loop_poll_seconds: parseFloat(e.target.value) || 0.5 })}
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
                        max="500"
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
                  {/* RSI Divergence Detection */}
                  <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid var(--border)' }}>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 8 }}>
                      RSI Divergence Detection (blocks entries when divergence detected against trend)
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
                          max="200"
                          value={editedSettings.rsi_divergence_lookback_bars}
                          onChange={(e) => setEditedSettings({ ...editedSettings, rsi_divergence_lookback_bars: parseInt(e.target.value) || 50 })}
                          style={{ width: '100%', padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                        />
                        <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 2 }}>M3 bars to analyze</div>
                      </div>
                      <div style={{ padding: 8, background: 'var(--bg-tertiary)', borderRadius: 6 }}>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 4 }}>Swing Window</div>
                        <input
                          type="number"
                          step="1"
                          min="2"
                          max="15"
                          value={editedSettings.rsi_divergence_swing_window}
                          onChange={(e) => setEditedSettings({ ...editedSettings, rsi_divergence_swing_window: parseInt(e.target.value) || 5 })}
                          style={{ width: '100%', padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontWeight: 600 }}
                        />
                        <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: 2 }}>Bars for swing detection</div>
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
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: 8 }}>
                        Tiered Pullback (fixed tiers: 9, 11, 13, 15, 17)
                      </div>
                      <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)' }}>
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
