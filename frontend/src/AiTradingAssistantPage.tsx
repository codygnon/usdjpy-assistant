import { useCallback, useEffect, useRef, useState } from 'react';
import * as api from './api';

const MAX_HISTORY_MESSAGES = 20; // 10 user + 10 assistant pairs

export interface AiAssistantProfile {
  path: string;
  name: string;
}

type ChatLine = { role: 'user' | 'assistant'; content: string };

const CHAT_STORAGE_PREFIX = 'usdjpy_ai_chat:';

function chatStorageKey(profilePath: string): string {
  return `${CHAT_STORAGE_PREFIX}${profilePath}`;
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

export default function AiTradingAssistantPage({ profile }: { profile: AiAssistantProfile }) {
  const [messages, setMessages] = useState<ChatLine[]>(() => loadChatFromStorage(profile.path));
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    setMessages(loadChatFromStorage(profile.path));
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

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  const send = useCallback(async () => {
    const text = input.trim();
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
        { message: text, history },
        {
          signal,
          onDelta: (delta) => {
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
      const msg = (e as Error).message || 'Request failed';
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
      abortRef.current = null;
    }
  }, [input, messages, profile.name, profile.path, sending]);

  return (
    <div>
      <h2 className="page-title">AI Trading Assistant</h2>
      <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: 16, maxWidth: 720 }}>
        Context-aware help using your profile&apos;s broker connection on the server. Not financial advice.
      </p>

      {error && (
        <div className="card mb-4" style={{ borderColor: 'var(--danger)' }}>
          <p style={{ color: 'var(--danger)', margin: 0 }}>{error}</p>
        </div>
      )}

      <div className="card" style={{ display: 'flex', flexDirection: 'column', maxWidth: 720, minHeight: 420 }}>
        <div
          ref={scrollRef}
          style={{
            flex: 1,
            overflowY: 'auto',
            padding: 12,
            marginBottom: 12,
            background: 'var(--bg-secondary, rgba(0,0,0,0.2))',
            borderRadius: 8,
            minHeight: 280,
          }}
        >
          {messages.length === 0 && (
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', margin: 0 }}>
              Ask about open positions, recent performance, or session context. Replies stream from the model.
            </p>
          )}
          {messages.map((line, i) => (
            <div
              key={i}
              style={{
                marginBottom: 12,
                textAlign: line.role === 'user' ? 'right' : 'left',
              }}
            >
              <div
                style={{
                  display: 'inline-block',
                  maxWidth: '92%',
                  padding: '10px 14px',
                  borderRadius: 10,
                  fontSize: '0.9rem',
                  lineHeight: 1.45,
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                  background:
                    line.role === 'user' ? 'var(--accent, #3b82f6)' : 'var(--card-border, rgba(255,255,255,0.08))',
                  color: line.role === 'user' ? '#fff' : 'var(--text-primary)',
                  opacity: line.role === 'assistant' && sending && i === messages.length - 1 && !line.content ? 0.6 : 1,
                }}
              >
                {line.role === 'user' ? <strong>You</strong> : <strong>Assistant</strong>}
                <div style={{ marginTop: 4, fontWeight: 400 }}>{line.content || (sending && i === messages.length - 1 ? '…' : '')}</div>
              </div>
            </div>
          ))}
        </div>

        <div className="form-group" style={{ marginBottom: 8 }}>
          <label>Message</label>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                void send();
              }
            }}
            placeholder="Type a message… (Enter to send, Shift+Enter for newline)"
            rows={3}
            disabled={sending}
            style={{ width: '100%', resize: 'vertical' }}
          />
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
          <button type="button" className="btn btn-primary" onClick={() => void send()} disabled={sending || !input.trim()}>
            {sending ? 'Sending…' : 'Send'}
          </button>
          {sending && (
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => {
                abortRef.current?.abort();
              }}
            >
              Stop
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
