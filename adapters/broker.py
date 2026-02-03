"""Broker dispatcher: returns the right adapter (MT5 or OANDA) for a profile."""
from __future__ import annotations

from typing import TYPE_CHECKING

from core.profile import ProfileV1

if TYPE_CHECKING:
    from adapters.mt5_adapter import OrderResult, Tick
    from adapters.oanda_adapter import OandaAdapter


def get_adapter(profile: ProfileV1):
    """Return the broker adapter for this profile (MT5 or OANDA)."""
    if getattr(profile, "broker_type", None) == "oanda":
        token = getattr(profile, "oanda_token", None) or ""
        if not token.strip():
            raise RuntimeError("Profile has broker_type=oanda but oanda_token is missing. Set OANDA API token in profile.")
        from adapters.oanda_adapter import OandaAdapter
        return OandaAdapter(
            token=token,
            account_id=getattr(profile, "oanda_account_id", None) or None,
            environment=getattr(profile, "oanda_environment", None) or "practice",
        )
    # MT5: return a wrapper that delegates to mt5_adapter module
    from adapters import mt5_adapter
    return _Mt5Wrapper(mt5_adapter)


class _Mt5Wrapper:
    """Wraps the mt5_adapter module so run_loop/execution_engine can call adapter.initialize() etc."""
    __slots__ = ("_m",)

    def __init__(self, mt5_module):
        self._m = mt5_module

    def initialize(self): return self._m.initialize()
    def shutdown(self): return self._m.shutdown()
    def ensure_symbol(self, symbol: str): return self._m.ensure_symbol(symbol)
    def get_account_info(self): return self._m.get_account_info()
    def get_tick(self, symbol: str): return self._m.get_tick(symbol)
    def get_bars(self, symbol: str, tf, count: int): return self._m.get_bars(symbol, tf, count)
    def is_demo_account(self): return self._m.is_demo_account()
    def get_open_positions(self, symbol=None): return self._m.get_open_positions(symbol)
    def order_send_market(self, **kwargs): return self._m.order_send_market(**kwargs)
    def order_send_pending_limit(self, **kwargs): return self._m.order_send_pending_limit(**kwargs)
    def close_position(self, **kwargs): return self._m.close_position(**kwargs)
    def get_position_by_ticket(self, ticket: int): return self._m.get_position_by_ticket(ticket)
    def get_position_id_from_deal(self, deal_id: int): return self._m.get_position_id_from_deal(deal_id)
    def get_position_id_from_order(self, order_id: int): return self._m.get_position_id_from_order(order_id)
    def get_deals_by_position(self, ticket: int): return self._m.get_deals_by_position(ticket)
    def get_deal_profit(self, deal_id: int): return self._m.get_deal_profit(deal_id)
    def get_position_close_info(self, ticket: int): return self._m.get_position_close_info(ticket)
    def get_closed_positions_from_history(self, **kwargs): return self._m.get_closed_positions_from_history(**kwargs)
    def get_mt5_report_stats(self, **kwargs): return self._m.get_mt5_report_stats(**kwargs)
    def get_mt5_full_report(self, **kwargs): return self._m.get_mt5_full_report(**kwargs)
