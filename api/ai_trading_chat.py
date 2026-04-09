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

def _extract_order_book_clusters(
    buckets: list[dict],
    book_price: float,
    range_pips: float = 100,
    top_n: int = 5,
) -> dict[str, Any]:
    """Extract top buy/sell clusters from OANDA order book buckets within ±range_pips.

    OANDA buckets: {"price": "148.200", "longCountPercent": "0.0841", "shortCountPercent": "0.0312"}
    longCountPercent = pending buy orders at that level (support/demand).
    shortCountPercent = pending sell orders at that level (resistance/supply).
    """
    pip_size = 0.01  # USDJPY
    range_price = range_pips * pip_size
    lo = book_price - range_price
    hi = book_price + range_price

    nearby = []
    for b in buckets:
        try:
            price = float(b["price"])
        except (KeyError, ValueError, TypeError):
            continue
        if lo <= price <= hi:
            nearby.append({
                "price": price,
                "long_pct": float(b.get("longCountPercent", 0)),
                "short_pct": float(b.get("shortCountPercent", 0)),
            })

    buy_clusters = sorted(nearby, key=lambda x: x["long_pct"], reverse=True)[:top_n]
    sell_clusters = sorted(nearby, key=lambda x: x["short_pct"], reverse=True)[:top_n]

    # Nearest support (highest buy cluster below price) / resistance (highest sell cluster above price)
    below_buys = [b for b in buy_clusters if b["price"] < book_price]
    above_sells = [s for s in sell_clusters if s["price"] > book_price]
    nearest_support = below_buys[0]["price"] if below_buys else None
    nearest_resistance = above_sells[0]["price"] if above_sells else None

    return {
        "current_price": book_price,
        "buy_clusters": [{"price": b["price"], "pct": b["long_pct"]} for b in buy_clusters],
        "sell_clusters": [{"price": s["price"], "pct": s["short_pct"]} for s in sell_clusters],
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "nearest_support_distance_pips": round((book_price - nearest_support) / pip_size, 1) if nearest_support else None,
        "nearest_resistance_distance_pips": round((nearest_resistance - book_price) / pip_size, 1) if nearest_resistance else None,
    }


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

        # Cross-asset snapshot (OANDA only): EUR/USD → DXY proxy, BCO/USD → Oil
        if getattr(profile, "broker_type", None) == "oanda":
            try:
                ctx["cross_assets"] = _fetch_cross_asset_prices(adapter)
            except Exception:
                pass

        # OANDA order book — extract buy/sell clusters near current price
        if getattr(profile, "broker_type", None) == "oanda" and hasattr(adapter, "get_order_book"):
            try:
                book = adapter.get_order_book(profile.symbol)
                ob = book.get("orderBook", {})
                buckets = ob.get("buckets", [])
                book_price = float(ob.get("price", 0))
                if buckets and book_price > 0:
                    ctx["order_book"] = _extract_order_book_clusters(
                        buckets, book_price, range_pips=100, top_n=5,
                    )
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

    ob = ctx.get("order_book")
    if ob:
        lines.append("")
        lines.append("KEY LEVELS (OANDA Order Book):")
        buy_cl = ob.get("buy_clusters", [])
        sell_cl = ob.get("sell_clusters", [])
        if buy_cl:
            parts = [f"{c['price']:.3f} ({c['pct']:.1%})" for c in buy_cl]
            lines.append(f"  Buy clusters: {', '.join(parts)}")
        if sell_cl:
            parts = [f"{c['price']:.3f} ({c['pct']:.1%})" for c in sell_cl]
            lines.append(f"  Sell clusters: {', '.join(parts)}")
        cp = ob.get("current_price")
        if cp:
            summary = f"  Current price: {cp:.3f}"
            sup = ob.get("nearest_support")
            res = ob.get("nearest_resistance")
            if sup:
                summary += f" — nearest support {ob['nearest_support_distance_pips']}p below at {sup:.3f}"
            if res:
                summary += f", nearest resistance {ob['nearest_resistance_distance_pips']}p above at {res:.3f}"
            lines.append(summary)

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
