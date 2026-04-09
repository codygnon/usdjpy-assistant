"""AI Trading Chat — streaming assistant backed by OpenAI.

SSE contract
------------
Success (200, text/event-stream):
  data: {"type":"delta","text":"..."}   # append to assistant reply
  data: {"type":"done"}                 # stream finished

Errors returned *before* the stream starts use standard JSON HTTPException:
  503 — OPENAI_API_KEY not configured
  400 — bad request body / profile_path
  404 — profile not found
  502 — broker init failure
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Generator

from core.profile import ProfileV1


# ---------------------------------------------------------------------------
# 1. Build trading context from broker
# ---------------------------------------------------------------------------

def build_trading_context(profile: ProfileV1) -> dict[str, Any]:
    """Fetch live account state from the broker and return a plain dict.

    Runs synchronously (caller wraps in ThreadPoolExecutor).
    Never includes raw tokens/secrets.
    """
    from adapters.broker import get_adapter

    adapter = get_adapter(profile)
    ctx: dict[str, Any] = {
        "symbol": profile.symbol,
        "broker_type": getattr(profile, "broker_type", "mt5"),
        "as_of": datetime.now(timezone.utc).isoformat(),
    }

    try:
        adapter.initialize()
        if hasattr(adapter, "ensure_symbol"):
            adapter.ensure_symbol(profile.symbol)

        # Account info
        try:
            acct = adapter.get_account_info()
            ctx["account"] = {
                "balance": float(getattr(acct, "balance", 0)),
                "equity": float(getattr(acct, "equity", 0)),
                "margin_used": float(getattr(acct, "margin", 0)),
                "margin_free": float(getattr(acct, "margin_free", 0)),
            }
        except Exception as e:
            ctx["account_error"] = str(e)

        # Open positions
        try:
            positions = adapter.get_open_positions(profile.symbol)
            open_list = []
            if positions:
                for pos in positions[:50]:  # cap for prompt size
                    if isinstance(pos, dict):
                        open_list.append({
                            "id": pos.get("id"),
                            "instrument": pos.get("instrument"),
                            "side": "BUY" if float(pos.get("currentUnits", pos.get("units", 0))) > 0 else "SELL",
                            "units": abs(float(pos.get("currentUnits", pos.get("units", 0)))),
                            "entry_price": pos.get("price"),
                            "unrealized_pl": pos.get("unrealizedPL"),
                        })
                    else:
                        # MT5 position object
                        open_list.append({
                            "id": getattr(pos, "ticket", None),
                            "instrument": getattr(pos, "symbol", profile.symbol),
                            "side": "BUY" if getattr(pos, "type", 0) == 0 else "SELL",
                            "units": getattr(pos, "volume", 0),
                            "entry_price": getattr(pos, "price_open", None),
                            "unrealized_pl": getattr(pos, "profit", None),
                        })
            ctx["open_positions"] = open_list
        except Exception as e:
            ctx["open_positions_error"] = str(e)

        # Closed trades (recent)
        try:
            broker_type = getattr(profile, "broker_type", None)
            if broker_type == "oanda":
                closed = adapter.get_closed_trade_summaries(
                    days_back=30,
                    symbol=profile.symbol,
                    pip_size=profile.pip_size,
                )
                ctx["recent_closed_trades"] = closed[:25]
            else:
                # MT5
                try:
                    report = adapter.get_mt5_report_stats(
                        symbol=profile.symbol,
                        pip_size=profile.pip_size,
                        days_back=30,
                    )
                    ctx["recent_trade_stats"] = {
                        "closed_trades": getattr(report, "closed_trades", 0),
                        "wins": getattr(report, "wins", 0),
                        "losses": getattr(report, "losses", 0),
                        "win_rate": getattr(report, "win_rate", 0),
                        "total_profit": getattr(report, "total_profit", 0),
                    }
                except Exception:
                    pass
        except Exception as e:
            ctx["closed_trades_error"] = str(e)

        # OANDA order book (optional, small)
        if getattr(profile, "broker_type", None) == "oanda" and hasattr(adapter, "get_order_book"):
            try:
                book = adapter.get_order_book(profile.symbol)
                buckets = book.get("orderBook", {}).get("buckets", [])
                # Keep only top 10 around current price
                ctx["order_book_sample"] = buckets[:10] if buckets else []
            except Exception:
                pass

    finally:
        try:
            adapter.shutdown()
        except Exception:
            pass

    return ctx


# ---------------------------------------------------------------------------
# 2. System prompt
# ---------------------------------------------------------------------------

def system_prompt_from_context(ctx: dict[str, Any]) -> str:
    """Build a system prompt that grounds the assistant in live trading data."""
    lines = [
        "You are a trading assistant for a manual USDJPY trader.",
        "Be concise. Lead with numbers. Never give imperative trade instructions like 'buy now' or 'sell now'.",
        "You may discuss market context, account state, position sizing, and risk management.",
        "",
        "=== LIVE TRADING CONTEXT ===",
        f"Symbol: {ctx.get('symbol', 'USDJPY')}",
        f"Broker: {ctx.get('broker_type', 'unknown')}",
        f"As of: {ctx.get('as_of', 'unknown')}",
    ]

    acct = ctx.get("account")
    if acct:
        lines.append("")
        lines.append("Account:")
        lines.append(f"  Balance: {acct.get('balance', '?')}")
        lines.append(f"  Equity: {acct.get('equity', '?')}")
        lines.append(f"  Margin Used: {acct.get('margin_used', '?')}")
        lines.append(f"  Margin Free: {acct.get('margin_free', '?')}")

    positions = ctx.get("open_positions")
    if positions:
        lines.append("")
        lines.append(f"Open Positions ({len(positions)}):")
        for p in positions:
            lines.append(
                f"  {p.get('side')} {p.get('units')} @ {p.get('entry_price')} "
                f"(PL: {p.get('unrealized_pl', '?')})"
            )
    elif positions is not None:
        lines.append("")
        lines.append("Open Positions: none")

    closed = ctx.get("recent_closed_trades")
    if closed:
        lines.append("")
        lines.append(f"Recent Closed Trades (last 30 days, showing {len(closed)}):")
        for t in closed[:10]:  # show top 10 in prompt
            lines.append(f"  {t}")

    stats = ctx.get("recent_trade_stats")
    if stats:
        lines.append("")
        lines.append("Recent Trade Stats (30d):")
        for k, v in stats.items():
            lines.append(f"  {k}: {v}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. OpenAI streaming
# ---------------------------------------------------------------------------

def stream_openai_chat(
    *,
    system: str,
    user_message: str,
    history: list[dict[str, str]],
    model: str | None = None,
) -> Generator[str, None, None]:
    """Yield SSE-formatted lines: {"type":"delta","text":"..."} then {"type":"done"}.

    Uses the official openai SDK with synchronous streaming.
    """
    import openai

    client = openai.OpenAI()  # uses OPENAI_API_KEY env var

    if model is None:
        model = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            yield f"data: {json.dumps({'type': 'delta', 'text': delta.content})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"
