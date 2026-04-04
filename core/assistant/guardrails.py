from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GuardrailResult:
    can_proceed: bool
    confirmations_needed: list[str]
    warnings: list[str]
    blocks: list[str]


class Guardrails:
    """Pre-trade validation guardrails."""

    def __init__(self, risk_config, session_config):
        self._risk_config = risk_config
        self._session_config = session_config

    def check(self, proposal, assessment, session_status, journal_stats: dict, open_trades: list) -> GuardrailResult:
        blocks = list(assessment.errors)
        warnings = list(assessment.warnings)
        confirms: list[str] = []

        if assessment.recommended_lots >= 10:
            confirms.append(
                f"LARGE POSITION: {assessment.recommended_lots:.1f} lots (${assessment.risk_amount:,.0f} at risk). Confirm? [y/n]"
            )

        if session_status.minutes_to_next_close <= self._session_config.warn_before_close_minutes:
            confirms.append(
                f"{session_status.next_close_session} session closes in {session_status.minutes_to_next_close} minutes. Enter anyway? [y/n]"
            )

        if not session_status.active_sessions:
            confirms.append("No major session active. Liquidity is low. Enter anyway? [y/n]")

        streak = journal_stats.get("current_streak", ("none", 0))
        if isinstance(streak, tuple) and len(streak) == 2 and streak[0] == "loss" and int(streak[1]) >= 3:
            confirms.append(
                f"You are on a {int(streak[1])}-trade LOSING STREAK. Are you trading to recover or because this is a good setup? [y/n]"
            )

        daily_pnl = float(journal_stats.get("today_pnl", 0.0) or 0.0)
        equity = float(assessment.account_equity)
        if equity > 0 and daily_pnl < -(equity * 0.02):
            confirms.append(
                f"You are DOWN ${abs(daily_pnl):,.0f} today ({daily_pnl / equity:.1%}). Daily loss exceeds 2%. Continue? [y/n]"
            )

        usdjpy_trades = [trade for trade in open_trades if "JPY" in trade.instrument]
        if len(usdjpy_trades) >= 3:
            confirms.append(f"You already have {len(usdjpy_trades)} USDJPY positions open. Add another? [y/n]")

        return GuardrailResult(
            can_proceed=(len(blocks) == 0),
            confirmations_needed=confirms,
            warnings=warnings,
            blocks=blocks,
        )
