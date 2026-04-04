#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.assistant.config import AssistantConfig
from core.assistant.cross_asset_dashboard import CrossAssetDashboard
from core.assistant.guardrails import Guardrails
from core.assistant.oanda_client import OandaClient
from core.assistant.position_monitor import PositionMonitor
from core.assistant.risk_manager import RiskManager, TradeProposal
from core.assistant.session_manager import SessionManager
from core.assistant.trade_journal import TradeJournal


class TradingAssistant:
    def __init__(self, *, client=None, config: AssistantConfig | None = None, journal: TradeJournal | None = None):
        self._config = config or AssistantConfig.load()
        self._config_errors = self._config.validate()
        self._journal = journal or TradeJournal(self._config.journal_db_path)
        self._risk_mgr = RiskManager(self._config.risk)
        self._session_mgr = SessionManager(self._config.sessions)
        self._guardrails = Guardrails(self._config.risk, self._config.sessions)
        self._client = client
        if self._client is None and not self._config_errors:
            self._client = OandaClient(
                account_id=self._config.oanda.account_id,
                api_token=self._config.oanda.api_token,
                environment=self._config.oanda.environment,
            )
        self._monitor = PositionMonitor(self._client, self._config.risk, self._config.trailing, self._journal) if self._client else None
        self._dashboard = CrossAssetDashboard(self._client) if self._client else None

    def run(self):
        print(
            "\n"
            "══════════════════════════════════════════════════\n"
            "            USDJPY Trading Assistant\n"
            "══════════════════════════════════════════════════\n"
            "Type 'help' to see commands.\n"
        )
        if self._config_errors:
            print("Configuration is incomplete. Run `python3 scripts/assistant_setup.py` first.\n")
            for error in self._config_errors:
                print(f"  - {error}")
            print()

        commands = {
            "status": self._cmd_status,
            "enter": self._cmd_enter,
            "bias": self._cmd_bias,
            "journal": self._cmd_journal,
            "report": self._cmd_report,
            "monitor": self._cmd_monitor,
            "config": self._cmd_config,
            "help": self._cmd_help,
            "quit": self._cmd_quit,
            "exit": self._cmd_quit,
        }
        while True:
            try:
                raw = input("assistant> ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break
            if not raw:
                continue
            handler = commands.get(raw)
            if handler is None:
                print("Unknown command. Type 'help'.")
                continue
            result = handler()
            if result == "quit":
                break

    def _require_client(self) -> bool:
        if self._client is None:
            print("OANDA client is unavailable. Run `python3 scripts/assistant_setup.py` with valid credentials first.")
            return False
        return True

    def _cmd_status(self):
        if not self._require_client():
            return
        acct = self._client.get_account_summary()
        print(
            f"""
══════════════════════════════════════════════════
  ACCOUNT STATUS
══════════════════════════════════════════════════
  Balance:     ${acct.balance:>12,.2f}
  Equity:      ${acct.equity:>12,.2f}
  Unrealized:  ${acct.unrealized_pnl:>12,.2f}
  Margin Used: ${acct.margin_used:>12,.2f}
  Margin Free: ${acct.margin_available:>12,.2f}
  Open Trades: {acct.open_trade_count}
"""
        )
        trades = self._client.get_open_trades(self._config.instrument)
        if trades:
            print("  OPEN POSITIONS:")
            print(f"  {'ID':<10} {'Dir':<6} {'Units':>10} {'Entry':>10} {'P&L':>10} {'Stop':>10} {'TP':>10}")
            print(f"  {'─'*10} {'─'*6} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
            for trade in trades:
                sl = f"{trade.stop_loss:.3f}" if trade.stop_loss is not None else "NONE"
                tp = f"{trade.take_profit:.3f}" if trade.take_profit is not None else "—"
                print(
                    f"  {trade.trade_id:<10} {trade.direction:<6} {abs(int(trade.units)):>10,} "
                    f"{trade.open_price:>10.3f} {trade.unrealized_pnl:>+10.2f} {sl:>10} {tp:>10}"
                )
            print()
        else:
            print("  No open positions.\n")
        session = self._session_mgr.get_status()
        print(self._session_mgr.format_status(session))
        if self._monitor is not None:
            monitor_status = self._monitor.get_status()
            if monitor_status["managed_count"] > 0:
                print(f"  Monitor tracking: {monitor_status['managed_count']} trade(s)")
            else:
                print("  Monitor: no trades being managed")

    def _cmd_enter(self):
        if not self._require_client() or self._monitor is None:
            return
        print("\n═══ NEW TRADE ENTRY ═══\n")
        try:
            price = self._client.get_price(self._config.instrument)
            print(
                f"  {self._config.instrument}: Bid {price.bid:.3f} / Ask {price.ask:.3f} "
                f"(Spread: {price.spread_pips:.1f} pips)"
            )
        except Exception as exc:
            print(f"  Cannot fetch price: {exc}")
            return

        direction = ""
        while direction not in {"long", "short"}:
            direction = input("  Direction (long/short): ").strip().lower()
            if direction not in {"long", "short"}:
                print("  Enter 'long' or 'short'")

        entry_input = input(f"  Entry price [{price.mid:.3f}]: ").strip()
        try:
            entry_price = float(entry_input) if entry_input else float(price.mid)
        except ValueError:
            print("  Invalid entry price.")
            return

        stop_input = input("  Stop loss price (or 'auto' for ATR-based): ").strip().lower()
        if stop_input in {"", "auto"}:
            try:
                candles = self._client.get_candles(self._config.instrument, "H4", 20)
                atr = PositionMonitor._compute_atr_from_candles(candles, 14)
                stop_loss = self._risk_mgr.compute_auto_stop(direction, entry_price, atr, self._config.instrument)
                stop_pips = self._risk_mgr.compute_stop_distance_pips(self._config.instrument, entry_price, stop_loss)
                print(f"  Auto stop: {stop_loss:.3f} ({stop_pips:.0f} pips)")
            except Exception as exc:
                print(f"  Cannot compute ATR: {exc}")
                return
        else:
            try:
                stop_loss = float(stop_input)
            except ValueError:
                print("  Invalid stop price.")
                return

        tp_input = input("  Take profit price (or 'auto' or Enter to skip): ").strip().lower()
        take_profit = None
        if tp_input == "auto":
            tp1, tp2 = self._risk_mgr.compute_tp_levels(direction, entry_price, stop_loss)
            take_profit = tp1
            print(f"  Auto TP1: {tp1:.3f} | TP2: {tp2:.3f}")
        elif tp_input:
            try:
                take_profit = float(tp_input)
            except ValueError:
                print("  Invalid take profit price.")
                return

        size_input = input("  Lots (or Enter for auto-size): ").strip()
        lots = None
        if size_input:
            try:
                lots = float(size_input)
            except ValueError:
                print("  Invalid lots.")
                return

        proposal = TradeProposal(
            instrument=self._config.instrument,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            lots=lots,
        )
        account = self._client.get_account_summary()
        open_trades = self._client.get_open_trades()
        assessment = self._risk_mgr.assess(proposal, account, open_trades, price)
        tp1_display = f"{assessment.take_profit_1:.3f}" if assessment.take_profit_1 is not None else "None"
        tp1_pips = f"{assessment.tp1_distance_pips:.0f}" if assessment.tp1_distance_pips is not None else "—"
        print(
            f"""
──────────────────────────────────────────────────
  RISK ASSESSMENT
──────────────────────────────────────────────────
  Direction:   {direction.upper()}
  Entry:       {entry_price:.3f}
  Stop Loss:   {assessment.stop_loss:.3f}  ({assessment.stop_distance_pips:.0f} pips)
  TP1:         {tp1_display}  ({tp1_pips} pips)
  Size:        {assessment.recommended_lots:.2f} lots ({assessment.recommended_units:,} units)
  Risk:        ${assessment.risk_amount:,.2f} ({assessment.risk_percent:.2%} of equity)
  Equity:      ${assessment.account_equity:,.2f}
"""
        )
        if assessment.warnings:
            print("  WARNINGS:")
            for warning in assessment.warnings:
                print(f"    - {warning}")
            print()
        if assessment.errors:
            print("  BLOCKED:")
            for error in assessment.errors:
                print(f"    - {error}")
            print("\n  Trade cannot proceed.")
            return

        session = self._session_mgr.get_status()
        journal_stats = self._journal.get_all_time_stats()
        journal_stats["current_streak"] = self._journal.get_streak()
        journal_stats["today_pnl"] = self._journal.get_daily_report().get("pnl", 0.0)
        guardrail_result = self._guardrails.check(proposal, assessment, session, journal_stats, open_trades)
        if guardrail_result.blocks:
            print("  GUARDRAIL BLOCKS:")
            for block in guardrail_result.blocks:
                print(f"    - {block}")
            print("\n  Trade cannot proceed.")
            return
        for confirmation in guardrail_result.confirmations_needed:
            if input(f"  {confirmation} ").strip().lower() != "y":
                print("  Trade cancelled.")
                return

        final_confirm = input(
            f"\n  PLACE {direction.upper()} {assessment.recommended_lots:.2f} lots @ {entry_price:.3f} "
            f"with SL {assessment.stop_loss:.3f}? [y/n] "
        ).strip().lower()
        if final_confirm != "y":
            print("  Trade cancelled.")
            return

        try:
            units = assessment.recommended_units if direction == "long" else -assessment.recommended_units
            result = self._client.place_market_order(
                instrument=self._config.instrument,
                units=units,
                stop_loss=assessment.stop_loss,
                take_profit=assessment.take_profit_1,
            )
            trade_id = "unknown"
            fill = result.get("orderFillTransaction", {})
            if "tradeOpened" in fill:
                trade_id = str(fill["tradeOpened"]["tradeID"])
            elif "tradeReduced" in fill:
                trade_id = str(fill["tradeReduced"]["tradeID"])
            print(f"\n  Trade opened. ID: {trade_id}")
            tp1, tp2 = self._risk_mgr.compute_tp_levels(direction, entry_price, assessment.stop_loss)
            self._monitor.register_trade(
                trade_id=trade_id,
                entry_price=entry_price,
                stop_loss=assessment.stop_loss,
                tp1=tp1,
                tp2=tp2,
                direction=direction,
                units=abs(units),
                user_provided_stop=True,
                instrument=self._config.instrument,
            )
            self._journal.log_trade_opened(
                trade_id=trade_id,
                instrument=self._config.instrument,
                direction=direction,
                entry_price=entry_price,
                units=abs(units),
                stop_loss=assessment.stop_loss,
                take_profit=assessment.take_profit_1,
            )
            print(f"  Logged to journal. Monitor tracking trade {trade_id}.")
        except Exception as exc:
            print(f"\n  Order failed: {exc}")

    def _cmd_bias(self):
        if not self._require_client() or self._dashboard is None:
            return
        print("  Fetching cross-asset data...")
        print(self._dashboard.format_reading(self._dashboard.get_reading()))

    def _cmd_journal(self):
        stats = self._journal.get_all_time_stats()
        streak_type, streak_count = self._journal.get_streak()
        print(
            f"""
══════════════════════════════════════════════════
  ALL-TIME STATISTICS
══════════════════════════════════════════════════
  Total Trades:   {stats['total_trades']}
  Win Rate:       {stats['win_rate']:.1%}
  Profit Factor:  {stats['profit_factor']:.2f}
  Total P&L:      ${stats['total_pnl']:>+12,.2f}
  Avg Win:        ${stats['avg_win']:>+12,.2f}
  Avg Loss:       ${stats['avg_loss']:>+12,.2f}
  Largest Win:    ${stats['largest_win']:>+12,.2f}
  Largest Loss:   ${stats['largest_loss']:>+12,.2f}
  Current Streak: {streak_count} {streak_type}{'s' if streak_count != 1 else ''}
"""
        )
        history = self._journal.get_trade_history(10)
        if not history:
            print("  No completed trades yet.\n")
            return
        print("  RECENT TRADES:")
        print(f"  {'Date':<12} {'Dir':<6} {'Entry':>10} {'Exit':>10} {'P&L':>10} {'Reason':<15}")
        print(f"  {'─'*12} {'─'*6} {'─'*10} {'─'*10} {'─'*10} {'─'*15}")
        for trade in history:
            exit_price = f"{trade['exit_price']:.3f}" if trade["exit_price"] is not None else "—"
            pnl = f"${trade['pnl']:+,.2f}" if trade["pnl"] is not None else "—"
            exit_date = trade["exit_time"][:10] if trade["exit_time"] else "—"
            reason = trade["exit_reason"] or "—"
            print(
                f"  {exit_date:<12} {trade['direction']:<6} {trade['entry_price']:>10.3f} "
                f"{exit_price:>10} {pnl:>10} {reason:<15}"
            )
        print()

    def _cmd_report(self):
        today = self._journal.get_daily_report()
        stats = self._journal.get_all_time_stats()
        equity = 0.0
        if self._client is not None:
            try:
                account = self._client.get_account_summary()
                equity = account.equity
                self._journal.log_daily_snapshot(account.equity, account.balance)
            except Exception:
                equity = 0.0
        print(
            f"""
══════════════════════════════════════════════════
  DAILY REPORT
══════════════════════════════════════════════════
  Trades Closed Today: {today['trades_closed']}
  Today's P&L:         ${today['pnl']:>+10,.2f}
  Wins / Losses:       {today['wins']} / {today['losses']}
  Current Equity:      ${equity:>12,.2f}
──────────────────────────────────────────────────
  ALL-TIME:
  Total Trades:        {stats['total_trades']}
  Total P&L:           ${stats['total_pnl']:>+10,.2f}
  Win Rate:            {stats['win_rate']:.1%}
  Profit Factor:       {stats['profit_factor']:.2f}
══════════════════════════════════════════════════
"""
        )

    def _cmd_monitor(self):
        if not self._require_client() or self._monitor is None:
            return
        print("  Starting position monitor...")
        print(f"  Polling every {self._config.poll_interval_seconds} seconds.")
        print("  Press Ctrl+C to stop monitoring.\n")
        try:
            while True:
                actions = self._monitor.check_and_manage(self._config.instrument)
                for action in actions:
                    icon = "📋"
                    if "error" in action.action:
                        icon = "❌"
                    elif "stop" in action.action:
                        icon = "🛑"
                    elif "tp" in action.action or "partial" in action.action:
                        icon = "💰"
                    elif "trail" in action.action:
                        icon = "📈"
                    elif "adopted" in action.action:
                        icon = "👁️"
                    elif "closed" in action.action:
                        icon = "✅"
                    print(f"  {icon} [{action.timestamp.strftime('%H:%M:%S')}] {action.details}")
                time.sleep(self._config.poll_interval_seconds)
        except KeyboardInterrupt:
            print("\n  Monitor stopped.")

    def _cmd_config(self):
        c = self._config
        print(
            f"""
══════════════════════════════════════════════════
  CONFIGURATION
══════════════════════════════════════════════════
  OANDA:
    Account:     {c.oanda.account_id}
    Environment: {c.oanda.environment}

  RISK:
    Max Risk/Trade:    {c.risk.max_risk_per_trade:.1%}
    Max Position:      {c.risk.max_position_size_lots} lots
    Catastrophic Stop: {c.risk.catastrophic_stop_pips:.0f} pips
    TP1 Ratio:         {c.risk.tp1_ratio:.1f}:1  (close {c.risk.tp1_close_fraction:.0%})
    TP2 Ratio:         {c.risk.tp2_ratio:.1f}:1  (close {c.risk.tp2_close_fraction:.0%})

  TRAILING:
    Enabled:      {c.trailing.enabled}
    Method:       {c.trailing.method}
    Timeframe:    {c.trailing.timeframe}
    Lookback:     {c.trailing.lookback_bars} bars
    ATR Buffer:   {c.trailing.atr_buffer}×
    Only Tighten: {c.trailing.only_tighten}

  SESSIONS:
    Tokyo:       {c.sessions.tokyo_start:02d}:00-{c.sessions.tokyo_end:02d}:00 UTC
    London:      {c.sessions.london_start:02d}:00-{c.sessions.london_end:02d}:00 UTC
    New York:    {c.sessions.ny_start:02d}:00-{c.sessions.ny_end:02d}:00 UTC

  Edit: config/assistant_config.yaml
══════════════════════════════════════════════════
"""
        )

    def _cmd_help(self):
        print(
            """
  COMMANDS:
    status    Account balance, open positions, session info
    enter     Open a new trade (interactive, with risk checks)
    bias      Cross-asset macro dashboard (oil + DXY)
    journal   Trade history and all-time statistics
    report    Daily performance report
    monitor   Start auto position manager (stops, trails, TPs)
    config    Show current configuration
    help      This message
    quit      Exit
"""
        )

    def _cmd_quit(self):
        print("Goodbye.")
        return "quit"


def main():
    TradingAssistant().run()


if __name__ == "__main__":
    main()
