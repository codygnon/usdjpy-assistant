from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class TradeRecord:
    trade_id: str
    instrument: str
    direction: str
    entry_price: float
    exit_price: Optional[float]
    entry_time: datetime
    exit_time: Optional[datetime]
    units: int
    pnl: Optional[float]
    pnl_pips: Optional[float]
    stop_loss: float
    take_profit: Optional[float]
    exit_reason: Optional[str]
    notes: str


@dataclass(frozen=True)
class DailyStats:
    date: date
    trades_opened: int
    trades_closed: int
    gross_pnl: float
    net_pnl: float
    wins: int
    losses: int
    win_rate: float
    largest_win: float
    largest_loss: float
    ending_equity: float
    max_drawdown_pct: float


class TradeJournal:
    """SQLite-backed trade journal."""

    def __init__(self, db_path: str = "research_out/trade_journal.db"):
        self._db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                instrument TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                initial_units INTEGER NOT NULL,
                exit_units INTEGER,
                pnl REAL,
                pnl_pips REAL,
                initial_stop REAL,
                final_stop REAL,
                take_profit REAL,
                exit_reason TEXT,
                notes TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                trade_id TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT NOT NULL,
                old_value REAL,
                new_value REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS daily_snapshots (
                date TEXT PRIMARY KEY,
                equity REAL NOT NULL,
                balance REAL NOT NULL,
                trades_opened INTEGER DEFAULT 0,
                trades_closed INTEGER DEFAULT 0,
                realized_pnl REAL DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        self._conn.commit()

    def log_trade_opened(
        self,
        trade_id: str,
        instrument: str,
        direction: str,
        entry_price: float,
        units: int,
        stop_loss: float,
        take_profit: Optional[float] = None,
        notes: str = "",
    ) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO trades
            (trade_id, instrument, direction, entry_price, entry_time, initial_units, initial_stop, take_profit, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade_id,
                instrument,
                direction,
                float(entry_price),
                datetime.now(timezone.utc).isoformat(),
                int(units),
                float(stop_loss),
                float(take_profit) if take_profit is not None else None,
                notes,
            ),
        )
        self._conn.commit()

    def log_trade_closed(
        self,
        managed_trade=None,
        trade_id: str = "",
        exit_price: float = 0.0,
        pnl: float = 0.0,
        exit_reason: str = "",
    ) -> None:
        if managed_trade is not None:
            trade_id = managed_trade.trade_id

        row = self._conn.execute(
            "SELECT entry_price, direction, instrument, initial_stop, take_profit, initial_units FROM trades WHERE trade_id = ?",
            (trade_id,),
        ).fetchone()

        pnl_pips = None
        if row is not None and exit_price:
            pip_size = 0.01 if "JPY" in row["instrument"] else 0.0001
            if row["direction"] == "long":
                pnl_pips = round((float(exit_price) - float(row["entry_price"])) / pip_size, 6)
            else:
                pnl_pips = round((float(row["entry_price"]) - float(exit_price)) / pip_size, 6)

        self._conn.execute(
            """
            UPDATE trades SET
                exit_price = ?,
                exit_time = ?,
                pnl = ?,
                pnl_pips = ?,
                exit_reason = ?,
                final_stop = COALESCE(final_stop, initial_stop),
                exit_units = COALESCE(exit_units, initial_units)
            WHERE trade_id = ?
            """,
            (
                float(exit_price) if exit_price else None,
                datetime.now(timezone.utc).isoformat(),
                float(pnl),
                pnl_pips,
                exit_reason,
                trade_id,
            ),
        )
        self._conn.commit()

    def log_action(self, action) -> None:
        self._conn.execute(
            """
            INSERT INTO actions (timestamp, trade_id, action, details, old_value, new_value)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                action.timestamp.isoformat(),
                action.trade_id,
                action.action,
                action.details,
                getattr(action, "old_value", None),
                getattr(action, "new_value", None),
            ),
        )
        self._conn.commit()

    def log_daily_snapshot(self, equity: float, balance: float, snapshot_date: Optional[date] = None) -> None:
        d = (snapshot_date or date.today()).isoformat()
        trades_opened = self._conn.execute("SELECT COUNT(*) FROM trades WHERE date(entry_time) = ?", (d,)).fetchone()[0]
        trades_closed = self._conn.execute("SELECT COUNT(*) FROM trades WHERE date(exit_time) = ?", (d,)).fetchone()[0]
        realized_pnl = self._conn.execute(
            "SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE date(exit_time) = ?",
            (d,),
        ).fetchone()[0]
        self._conn.execute(
            """
            INSERT OR REPLACE INTO daily_snapshots
            (date, equity, balance, trades_opened, trades_closed, realized_pnl)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (d, float(equity), float(balance), int(trades_opened), int(trades_closed), float(realized_pnl)),
        )
        self._conn.commit()

    def get_trade_history(self, limit: int = 50) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT * FROM trades
            WHERE exit_time IS NOT NULL
            ORDER BY exit_time DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_all_time_stats(self) -> dict:
        rows = self._conn.execute(
            "SELECT pnl FROM trades WHERE exit_time IS NOT NULL AND pnl IS NOT NULL ORDER BY exit_time ASC"
        ).fetchall()
        if not rows:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "current_streak": ("none", 0),
            }

        pnls = [float(row["pnl"]) for row in rows]
        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [pnl for pnl in pnls if pnl <= 0]
        gross_wins = sum(wins) if wins else 0.0
        gross_losses = abs(sum(losses)) if losses else 0.0
        return {
            "total_trades": len(pnls),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": (len(wins) / len(pnls)) if pnls else 0.0,
            "total_pnl": sum(pnls),
            "avg_win": (gross_wins / len(wins)) if wins else 0.0,
            "avg_loss": (-gross_losses / len(losses)) if losses else 0.0,
            "profit_factor": (gross_wins / gross_losses) if gross_losses > 0 else float("inf"),
            "largest_win": max(pnls) if pnls else 0.0,
            "largest_loss": min(pnls) if pnls else 0.0,
            "current_streak": self.get_streak(),
        }

    def get_daily_report(self, for_date: Optional[date] = None) -> dict:
        d = (for_date or date.today()).isoformat()
        rows = self._conn.execute(
            "SELECT pnl FROM trades WHERE date(exit_time) = ? AND pnl IS NOT NULL",
            (d,),
        ).fetchall()
        pnls = [float(row["pnl"]) for row in rows]
        snapshot = self._conn.execute("SELECT * FROM daily_snapshots WHERE date = ?", (d,)).fetchone()
        return {
            "date": d,
            "trades_closed": len(pnls),
            "pnl": sum(pnls),
            "wins": len([pnl for pnl in pnls if pnl > 0]),
            "losses": len([pnl for pnl in pnls if pnl <= 0]),
            "equity": float(snapshot["equity"]) if snapshot is not None else None,
        }

    def get_streak(self) -> tuple[str, int]:
        rows = self._conn.execute(
            """
            SELECT pnl FROM trades
            WHERE exit_time IS NOT NULL AND pnl IS NOT NULL
            ORDER BY exit_time DESC
            """
        ).fetchall()
        if not rows:
            return ("none", 0)
        first = "win" if float(rows[0]["pnl"]) > 0 else "loss"
        count = 0
        for row in rows:
            is_win = float(row["pnl"]) > 0
            if (first == "win" and is_win) or (first == "loss" and not is_win):
                count += 1
            else:
                break
        return first, count

    def close(self):
        self._conn.close()
