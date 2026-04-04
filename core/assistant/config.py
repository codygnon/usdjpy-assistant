from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "assistant_config.yaml"


@dataclass
class OandaConfig:
    """OANDA API connection settings."""

    account_id: str = ""
    api_token: str = ""
    environment: str = "practice"

    @property
    def base_url(self) -> str:
        if self.environment == "live":
            return "https://api-fxtrade.oanda.com"
        return "https://api-fxpractice.oanda.com"

    @property
    def stream_url(self) -> str:
        if self.environment == "live":
            return "https://stream-fxtrade.oanda.com"
        return "https://stream-fxpractice.oanda.com"


@dataclass
class RiskConfig:
    """Risk management parameters."""

    max_risk_per_trade: float = 0.01
    max_total_exposure: float = 0.05
    max_position_size_lots: int = 20
    default_stop_atr_multiple: float = 2.0
    catastrophic_stop_pips: float = 200.0
    tp1_ratio: float = 1.0
    tp1_close_fraction: float = 0.5
    tp2_ratio: float = 2.0
    tp2_close_fraction: float = 0.25


@dataclass
class TrailingConfig:
    """Trailing stop parameters."""

    enabled: bool = True
    method: str = "swing"
    lookback_bars: int = 5
    atr_buffer: float = 0.5
    timeframe: str = "H4"
    only_tighten: bool = True


@dataclass
class SessionConfig:
    """Session timing (UTC hours)."""

    tokyo_start: int = 0
    tokyo_end: int = 9
    london_start: int = 7
    london_end: int = 16
    ny_start: int = 12
    ny_end: int = 21
    warn_before_close_minutes: int = 15


@dataclass
class AssistantConfig:
    """Top-level configuration."""

    oanda: OandaConfig = field(default_factory=OandaConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    trailing: TrailingConfig = field(default_factory=TrailingConfig)
    sessions: SessionConfig = field(default_factory=SessionConfig)
    instrument: str = "USD_JPY"
    poll_interval_seconds: int = 10
    journal_db_path: str = "research_out/trade_journal.db"

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AssistantConfig":
        path = path or CONFIG_PATH
        if not path.exists():
            return cls()

        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        config = cls()

        oanda = raw.get("oanda", {})
        config.oanda = OandaConfig(
            account_id=oanda.get("account_id", ""),
            api_token=oanda.get("api_token", ""),
            environment=oanda.get("environment", "practice"),
        )

        risk = raw.get("risk", {})
        for key in vars(config.risk):
            if key in risk:
                setattr(config.risk, key, risk[key])

        trailing = raw.get("trailing", {})
        for key in vars(config.trailing):
            if key in trailing:
                setattr(config.trailing, key, trailing[key])

        sessions = raw.get("sessions", {})
        for key in vars(config.sessions):
            if key in sessions:
                setattr(config.sessions, key, sessions[key])

        config.instrument = raw.get("instrument", config.instrument)
        config.poll_interval_seconds = raw.get("poll_interval_seconds", config.poll_interval_seconds)
        config.journal_db_path = raw.get("journal_db_path", config.journal_db_path)
        return config

    def save(self, path: Optional[Path] = None) -> None:
        path = path or CONFIG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "oanda": {
                "account_id": self.oanda.account_id,
                "api_token": self.oanda.api_token,
                "environment": self.oanda.environment,
            },
            "risk": vars(self.risk),
            "trailing": vars(self.trailing),
            "sessions": vars(self.sessions),
            "instrument": self.instrument,
            "poll_interval_seconds": self.poll_interval_seconds,
            "journal_db_path": self.journal_db_path,
        }
        with path.open("w", encoding="utf-8") as handle:
            yaml.dump(data, handle, default_flow_style=False, sort_keys=False)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.oanda.account_id:
            errors.append("oanda.account_id is required")
        if not self.oanda.api_token:
            errors.append("oanda.api_token is required")
        if self.oanda.environment not in {"practice", "live"}:
            errors.append("oanda.environment must be 'practice' or 'live'")
        if self.risk.max_risk_per_trade <= 0 or self.risk.max_risk_per_trade > 0.10:
            errors.append("risk.max_risk_per_trade must be between 0 and 0.10 (10%)")
        if self.risk.max_total_exposure <= 0:
            errors.append("risk.max_total_exposure must be positive")
        if self.risk.max_position_size_lots <= 0:
            errors.append("risk.max_position_size_lots must be positive")
        if self.risk.catastrophic_stop_pips < 50:
            errors.append("risk.catastrophic_stop_pips must be >= 50")
        if self.poll_interval_seconds <= 0:
            errors.append("poll_interval_seconds must be positive")
        return errors
