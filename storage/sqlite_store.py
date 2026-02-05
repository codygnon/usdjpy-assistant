from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS snapshots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp_utc TEXT NOT NULL,
  profile TEXT NOT NULL,
  symbol TEXT NOT NULL,
  config_json TEXT,
  spread_pips REAL,
  alignment_score INTEGER,

  h4_regime TEXT,
  h4_cross_dir TEXT,
  h4_cross_time TEXT,
  h4_cross_price REAL,
  h4_trend_since TEXT,

  m15_regime TEXT,
  m15_cross_dir TEXT,
  m15_cross_time TEXT,
  m15_cross_price REAL,
  m15_trend_since TEXT,

  m1_regime TEXT,
  m1_cross_dir TEXT,
  m1_cross_time TEXT,
  m1_cross_price REAL,
  m1_trend_since TEXT
);

CREATE INDEX IF NOT EXISTS idx_snapshots_profile_time
  ON snapshots(profile, timestamp_utc);

CREATE TABLE IF NOT EXISTS trades (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  trade_id TEXT NOT NULL UNIQUE,
  timestamp_utc TEXT NOT NULL,
  profile TEXT NOT NULL,
  symbol TEXT NOT NULL,
  side TEXT NOT NULL,
  config_json TEXT,
  entry_price REAL NOT NULL,
  stop_price REAL,
  target_price REAL,
  size_lots REAL,
  notes TEXT,

  snapshot_id INTEGER,

  exit_price REAL,
  exit_timestamp_utc TEXT,
  exit_reason TEXT,
  pips REAL,
  risk_pips REAL,
  r_multiple REAL,
  duration_minutes REAL,

  mt5_order_id INTEGER,
  mt5_deal_id INTEGER,
  mt5_retcode INTEGER,
  mt5_position_id INTEGER,
  opened_by TEXT,
  preset_name TEXT,
  profit REAL,

  FOREIGN KEY(snapshot_id) REFERENCES snapshots(id)
);

CREATE INDEX IF NOT EXISTS idx_trades_profile_time
  ON trades(profile, timestamp_utc);

CREATE TABLE IF NOT EXISTS executions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp_utc TEXT NOT NULL,
  profile TEXT NOT NULL,
  symbol TEXT NOT NULL,
  signal_id TEXT NOT NULL,
  mode TEXT NOT NULL,
  attempted INTEGER NOT NULL,
  placed INTEGER NOT NULL,
  reason TEXT NOT NULL,
  mt5_retcode INTEGER,
  mt5_order_id INTEGER,
  mt5_deal_id INTEGER
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_executions_signal
  ON executions(profile, signal_id);

CREATE TABLE IF NOT EXISTS pending_signals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp_utc TEXT NOT NULL,
  profile TEXT NOT NULL,
  symbol TEXT NOT NULL,
  signal_id TEXT NOT NULL UNIQUE,
  timeframe TEXT NOT NULL,
  side TEXT NOT NULL,
  cross_time TEXT,
  confirm_time TEXT,
  entry_price_hint REAL,
  reasons_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_pending_profile_time
  ON pending_signals(profile, timestamp_utc);

CREATE TABLE IF NOT EXISTS daily_summaries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  date_utc TEXT NOT NULL,
  symbol TEXT NOT NULL,
  summary_text TEXT NOT NULL,
  created_utc TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_daily_summaries_date_symbol
  ON daily_summaries(date_utc, symbol);
"""


@dataclass(frozen=True)
class SqliteStore:
    path: Path

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA_SQL)
            # Backfill columns (safe migration) for early v1 databases
            self._ensure_column(conn, "snapshots", "config_json", "TEXT")
            self._ensure_column(conn, "trades", "config_json", "TEXT")
            self._ensure_column(conn, "executions", "rule_id", "TEXT")
            # v1.2: add mt5_position_id and opened_by for sync fix
            self._ensure_column(conn, "trades", "mt5_position_id", "INTEGER")
            self._ensure_column(conn, "trades", "opened_by", "TEXT")
            # v1.3: add preset_name for per-preset statistics
            self._ensure_column(conn, "trades", "preset_name", "TEXT")
            # v1.4: add profit for MT5-aligned win/loss stats
            self._ensure_column(conn, "trades", "profit", "REAL")
            # v1.5: trade-management tracking (breakeven, TP1 partial close)
            self._ensure_column(conn, "trades", "breakeven_applied", "INTEGER")
            self._ensure_column(conn, "trades", "tp1_partial_done", "INTEGER")
            conn.commit()

    def _ensure_column(self, conn: sqlite3.Connection, table: str, col: str, col_type: str) -> None:
        cur = conn.execute(f"PRAGMA table_info({table})")
        cols = {r[1] for r in cur.fetchall()}
        if col in cols:
            return
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")

    def insert_snapshot(self, row: dict[str, Any]) -> int:
        cols = list(row.keys())
        q = f"INSERT INTO snapshots ({', '.join(cols)}) VALUES ({', '.join(['?'] * len(cols))})"
        with self.connect() as conn:
            cur = conn.execute(q, [row[c] for c in cols])
            conn.commit()
            return int(cur.lastrowid)

    def latest_snapshot(self, profile: str) -> Optional[sqlite3.Row]:
        with self.connect() as conn:
            cur = conn.execute(
                "SELECT * FROM snapshots WHERE profile=? ORDER BY timestamp_utc DESC LIMIT 1",
                [profile],
            )
            return cur.fetchone()

    def insert_trade(self, row: dict[str, Any]) -> None:
        cols = list(row.keys())
        q = f"INSERT INTO trades ({', '.join(cols)}) VALUES ({', '.join(['?'] * len(cols))})"
        with self.connect() as conn:
            conn.execute(q, [row[c] for c in cols])
            conn.commit()

    def list_open_trades(self, profile: str) -> list[sqlite3.Row]:
        with self.connect() as conn:
            cur = conn.execute(
                "SELECT * FROM trades WHERE profile=? AND exit_price IS NULL ORDER BY timestamp_utc DESC",
                [profile],
            )
            return cur.fetchall()

    def close_trade(self, *, trade_id: str, updates: dict[str, Any]) -> None:
        sets = ", ".join([f"{k}=?" for k in updates.keys()])
        q = f"UPDATE trades SET {sets} WHERE trade_id=?"
        with self.connect() as conn:
            conn.execute(q, [*updates.values(), trade_id])
            conn.commit()

    def read_trades_df(self, profile: str) -> pd.DataFrame:
        with self.connect() as conn:
            df = pd.read_sql_query("SELECT * FROM trades WHERE profile=? ORDER BY timestamp_utc", conn, params=[profile])
        return df

    def read_snapshots_df(self, profile: str) -> pd.DataFrame:
        with self.connect() as conn:
            df = pd.read_sql_query("SELECT * FROM snapshots WHERE profile=? ORDER BY timestamp_utc", conn, params=[profile])
        return df

    # --- Daily summaries (USDJPY) ---

    def get_daily_summary(self, date_utc: str, symbol: str) -> Optional[str]:
        with self.connect() as conn:
            cur = conn.execute(
                "SELECT summary_text FROM daily_summaries WHERE date_utc=? AND symbol=? LIMIT 1",
                [date_utc, symbol],
            )
            row = cur.fetchone()
            return None if row is None else str(row[0])

    def upsert_daily_summary(self, date_utc: str, symbol: str, summary_text: str) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO daily_summaries (date_utc, symbol, summary_text, created_utc) VALUES (?, ?, ?, ?)",
                [date_utc, symbol, summary_text, now],
            )
            conn.commit()

    def read_executions_df(self, profile: str) -> pd.DataFrame:
        with self.connect() as conn:
            df = pd.read_sql_query("SELECT * FROM executions WHERE profile=? ORDER BY timestamp_utc", conn, params=[profile])
        return df

    def insert_execution(self, row: dict[str, Any]) -> None:
        cols = list(row.keys())
        q = f"INSERT OR REPLACE INTO executions ({', '.join(cols)}) VALUES ({', '.join(['?'] * len(cols))})"
        with self.connect() as conn:
            conn.execute(q, [row[c] for c in cols])
            conn.commit()

    def executed_signal_ids(self, profile: str) -> set[str]:
        with self.connect() as conn:
            cur = conn.execute("SELECT signal_id FROM executions WHERE profile=? AND placed=1", [profile])
            return {str(r["signal_id"]) for r in cur.fetchall()}

    def has_recent_price_level_placement(
        self,
        profile: str,
        rule_id: str,
        within_minutes: int,
    ) -> bool:
        """True if we placed an order for this rule_id within the last `within_minutes`."""
        if within_minutes <= 0:
            return False
        since = (datetime.now(timezone.utc) - timedelta(minutes=within_minutes)).isoformat()
        with self.connect() as conn:
            cur = conn.execute(
                "SELECT 1 FROM executions WHERE profile=? AND rule_id=? AND placed=1 AND timestamp_utc >= ? LIMIT 1",
                [profile, rule_id, since],
            )
            return cur.fetchone() is not None

    def has_recent_placement_by_prefix(
        self,
        profile: str,
        rule_id_prefix: str,
        within_minutes: float,
    ) -> bool:
        """True if we placed an order with rule_id starting with prefix within the last `within_minutes`."""
        if within_minutes <= 0:
            return False
        since = (datetime.now(timezone.utc) - timedelta(minutes=within_minutes)).isoformat()
        with self.connect() as conn:
            cur = conn.execute(
                "SELECT 1 FROM executions WHERE profile=? AND rule_id LIKE ? AND placed=1 AND timestamp_utc >= ? LIMIT 1",
                [profile, rule_id_prefix + "%", since],
            )
            return cur.fetchone() is not None

    def upsert_pending_signal(self, row: dict[str, Any]) -> None:
        cols = list(row.keys())
        q = f"INSERT OR REPLACE INTO pending_signals ({', '.join(cols)}) VALUES ({', '.join(['?'] * len(cols))})"
        with self.connect() as conn:
            conn.execute(q, [row[c] for c in cols])
            conn.commit()

    def list_pending_signals(self, profile: str) -> list[sqlite3.Row]:
        with self.connect() as conn:
            cur = conn.execute("SELECT * FROM pending_signals WHERE profile=? ORDER BY timestamp_utc DESC", [profile])
            return cur.fetchall()

    def delete_pending_signal(self, signal_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM pending_signals WHERE signal_id=?", [signal_id])
            conn.commit()

    def update_trade(self, trade_id: str, updates: dict[str, Any]) -> None:
        """Update arbitrary fields on a trade by trade_id."""
        if not updates:
            return
        sets = ", ".join([f"{k}=?" for k in updates.keys()])
        q = f"UPDATE trades SET {sets} WHERE trade_id=?"
        with self.connect() as conn:
            conn.execute(q, [*updates.values(), trade_id])
            conn.commit()

    def get_trades_missing_position_id(self, profile: str) -> list[sqlite3.Row]:
        """Get trades that have mt5_deal_id or mt5_order_id but no mt5_position_id (for backfill)."""
        with self.connect() as conn:
            cur = conn.execute(
                """SELECT * FROM trades WHERE profile=? AND mt5_position_id IS NULL
                   AND (mt5_deal_id IS NOT NULL OR mt5_order_id IS NOT NULL)""",
                [profile],
            )
            return cur.fetchall()

    def get_trades_missing_profit(self, profile: str, force_refresh: bool = False) -> list[sqlite3.Row]:
        """Get closed trades that have mt5_position_id for profit backfill.
        
        If force_refresh=True, returns all closed trades with position_id (to overwrite
        potentially incorrect profit data). Otherwise only those with profit IS NULL.
        """
        with self.connect() as conn:
            if force_refresh:
                cur = conn.execute(
                    """SELECT * FROM trades WHERE profile=? AND exit_price IS NOT NULL
                       AND mt5_position_id IS NOT NULL""",
                    [profile],
                )
            else:
                cur = conn.execute(
                    """SELECT * FROM trades WHERE profile=? AND exit_price IS NOT NULL
                       AND mt5_position_id IS NOT NULL AND profit IS NULL""",
                    [profile],
                )
            return cur.fetchall()

    def trade_exists_by_position_id(self, profile: str, mt5_position_id: int) -> bool:
        """Check if a trade with this mt5_position_id already exists."""
        with self.connect() as conn:
            cur = conn.execute(
                "SELECT 1 FROM trades WHERE profile=? AND mt5_position_id=? LIMIT 1",
                [profile, mt5_position_id],
            )
            return cur.fetchone() is not None

