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
from datetime import datetime, timedelta, timezone
from typing import Any, Generator

from core.profile import ProfileV1


# ---------------------------------------------------------------------------
# 1. Build trading context from broker
# ---------------------------------------------------------------------------

def _fetch_cross_asset_prices(adapter: Any) -> dict[str, Any]:
    """Fetch EUR/USD and BCO/USD from OANDA pricing endpoint for cross-asset context.

    DXY proxy: ~1/EURUSD * 100 gives directional sense (higher = stronger USD).
    """
    aid = adapter._get_account_id()
    instruments = "EUR_USD,BCO_USD"
    data = adapter._req("GET", f"/v3/accounts/{aid}/pricing?instruments={instruments}")
    prices = data.get("prices", [])

    result: dict[str, Any] = {}
    for p in prices:
        inst = p.get("instrument", "")
        bids = p.get("bids", [{}])
        asks = p.get("asks", [{}])
        bid = float(bids[0].get("price", 0)) if bids else 0
        ask = float(asks[0].get("price", 0)) if asks else 0
        mid = (bid + ask) / 2 if (bid and ask) else 0

        if inst == "EUR_USD" and mid > 0:
            result["eurusd"] = round(mid, 5)
            result["dxy_proxy"] = round(1.0 / mid * 100, 2)
        elif inst == "BCO_USD" and mid > 0:
            result["bco_usd"] = round(mid, 2)

    return result


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


# ---------------------------------------------------------------------------
# Session awareness (pure UTC clock math)
# ---------------------------------------------------------------------------

_SESSIONS = [
    ("Tokyo",  0, 9),
    ("London", 7, 16),
    ("NY",     12, 21),
]


def _compute_session_info(now_utc: datetime) -> dict[str, Any]:
    """Determine active sessions, overlaps, and proximity to low-liquidity windows."""
    h = now_utc.hour + now_utc.minute / 60.0
    active = [name for name, start, end in _SESSIONS if start <= h < end]

    # Detect overlaps
    overlap = None
    if "London" in active and "Tokyo" in active:
        overlap = "Tokyo/London overlap"
    elif "London" in active and "NY" in active:
        overlap = "London/NY overlap"

    # Time until next session close (for the latest active session)
    next_close = None
    next_close_name = None
    for name, _start, end in _SESSIONS:
        if name in active:
            remaining_h = end - h
            if remaining_h > 0 and (next_close is None or remaining_h < next_close):
                next_close = remaining_h
                next_close_name = name

    # Low-liquidity warning: 16:00-17:00 UTC (after London, before late NY) or 21:00+ (after NY)
    warnings: list[str] = []
    if 15.75 <= h < 16.0:
        warnings.append("Approaching 16:00 UTC low-liquidity window")
    elif 16.0 <= h < 17.0:
        warnings.append("In 16:00-17:00 UTC low-liquidity window")
    if 20.75 <= h < 21.0:
        warnings.append("NY close imminent")
    elif h >= 21.0 or h < 0.0:
        warnings.append("Post-NY close — thin liquidity")

    result: dict[str, Any] = {"active_sessions": active}
    if overlap:
        result["overlap"] = overlap
    if next_close is not None:
        hrs = int(next_close)
        mins = int((next_close - hrs) * 60)
        result["next_close"] = f"{next_close_name} close in {hrs}h {mins:02d}m"
    if warnings:
        result["warnings"] = warnings
    return result


# ---------------------------------------------------------------------------
# Derived trade stats (today / this week)
# ---------------------------------------------------------------------------

def _compute_derived_stats(closed_trades: list[dict]) -> dict[str, Any]:
    """Compute today's and this week's stats from closed trade list.

    Each trade dict must have 'close_time' (ISO str) and 'profit' (float).
    """
    now_utc = datetime.now(timezone.utc)
    today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    # Week starts Monday 00:00 UTC
    week_start = today_start - timedelta(days=now_utc.weekday())

    today_trades: list[dict] = []
    week_trades: list[dict] = []

    for t in closed_trades:
        ct_str = t.get("close_time", "")
        if not ct_str:
            continue
        try:
            ct = datetime.fromisoformat(ct_str.replace("Z", "+00:00"))
        except Exception:
            continue
        if ct >= today_start:
            today_trades.append(t)
        if ct >= week_start:
            week_trades.append(t)

    def _summarize(trades: list[dict]) -> dict[str, Any] | None:
        if not trades:
            return None
        profits = [float(t.get("profit", 0)) for t in trades]
        wins = sum(1 for p in profits if p > 0)
        losses = sum(1 for p in profits if p < 0)
        total = len(profits)
        return {
            "trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / total * 100, 1) if total else 0,
            "net_pl": round(sum(profits), 2),
            "best": round(max(profits), 2) if profits else 0,
            "worst": round(min(profits), 2) if profits else 0,
        }

    # Consecutive losses (from most recent trade backward)
    consec_losses = 0
    last_loss_time = None
    for t in sorted(closed_trades, key=lambda x: x.get("close_time", ""), reverse=True):
        if float(t.get("profit", 0)) < 0:
            consec_losses += 1
            if last_loss_time is None:
                last_loss_time = t.get("close_time")
        else:
            break

    result: dict[str, Any] = {}
    today_summary = _summarize(today_trades)
    if today_summary:
        today_summary["consecutive_losses"] = consec_losses
        if last_loss_time:
            today_summary["last_loss_time"] = last_loss_time
        result["today"] = today_summary
    week_summary = _summarize(week_trades)
    if week_summary:
        result["week"] = week_summary
    return result


# ---------------------------------------------------------------------------
# Guardrail alerts (conditional, based on derived stats + session)
# ---------------------------------------------------------------------------

def _compute_guardrail_alerts(
    derived_stats: dict[str, Any],
    session_info: dict[str, Any],
    open_positions: list[dict],
) -> list[str]:
    """Return list of active guardrail alert strings. Empty if nothing triggered."""
    alerts: list[str] = []
    now_utc = datetime.now(timezone.utc)

    # Consecutive losses
    today = derived_stats.get("today", {})
    consec = today.get("consecutive_losses", 0)
    if consec >= 2:
        msg = f"{consec} consecutive losses — suggest 30 min pause"
        last_loss = today.get("last_loss_time")
        if last_loss:
            try:
                lt = datetime.fromisoformat(last_loss.replace("Z", "+00:00"))
                mins_ago = int((now_utc - lt).total_seconds() / 60)
                msg += f" (last loss {mins_ago} min ago)"
            except Exception:
                pass
        alerts.append(msg)

    # Session warnings (from session_info)
    for w in session_info.get("warnings", []):
        alerts.append(w)

    # DCA / concentration cap: count open positions by side in last 2 hours
    if open_positions:
        buys = sum(1 for p in open_positions if p.get("side") == "BUY")
        sells = sum(1 for p in open_positions if p.get("side") == "SELL")
        if buys >= 3:
            alerts.append(f"{buys} open BUY positions — check DCA/concentration limits")
        if sells >= 3:
            alerts.append(f"{sells} open SELL positions — check DCA/concentration limits")

    # Large drawdown warning
    week = derived_stats.get("week", {})
    week_pl = week.get("net_pl", 0)
    if week_pl < -500:
        alerts.append(f"Weekly P&L at ${week_pl:+.0f} — consider reducing size")

    return alerts


# ---------------------------------------------------------------------------
# Volatility indicator (M1 candle range)
# ---------------------------------------------------------------------------

def _compute_volatility(adapter: Any, symbol: str) -> dict[str, Any] | None:
    """Fetch last 100 M1 candles, compare recent 30-bar avg range to full baseline."""
    try:
        df = adapter.get_bars(symbol, "M1", count=100)
        if df is None or len(df) < 50:
            return None
        highs = df["high"].values
        lows = df["low"].values
        ranges = highs - lows

        recent_avg = float(ranges[-30:].mean())
        baseline_avg = float(ranges.mean())
        if baseline_avg <= 0:
            return None

        ratio = round(recent_avg / baseline_avg, 1)
        if ratio >= 1.5:
            label = "Elevated"
        elif ratio >= 1.2:
            label = "Above average"
        elif ratio <= 0.6:
            label = "Very low"
        elif ratio <= 0.8:
            label = "Below average"
        else:
            label = "Normal"

        return {
            "label": label,
            "ratio": ratio,
            "recent_avg_pips": round(recent_avg / 0.01, 1),  # USDJPY pip = 0.01
        }
    except Exception:
        return None


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

        # Volatility indicator (M1 candle range comparison)
        try:
            vol = _compute_volatility(adapter, profile.symbol)
            if vol:
                ctx["volatility"] = vol
        except Exception:
            pass

    finally:
        try:
            adapter.shutdown()
        except Exception:
            pass

    # Session awareness (no API call needed)
    now_utc = datetime.now(timezone.utc)
    ctx["session"] = _compute_session_info(now_utc)

    # Derived stats from closed trades
    all_closed = ctx.get("recent_closed_trades", [])
    if all_closed:
        ctx["derived_stats"] = _compute_derived_stats(all_closed)

    # Guardrail alerts (conditional)
    derived = ctx.get("derived_stats", {})
    session = ctx.get("session", {})
    open_pos = ctx.get("open_positions", [])
    alerts = _compute_guardrail_alerts(derived, session, open_pos)
    if alerts:
        ctx["guardrail_alerts"] = alerts

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

    # Session info
    session = ctx.get("session")
    if session:
        parts = []
        active = session.get("active_sessions", [])
        overlap = session.get("overlap")
        if overlap:
            parts.append(overlap)
        elif active:
            parts.append(" + ".join(active))
        else:
            parts.append("No major session active")
        nc = session.get("next_close")
        if nc:
            parts.append(nc)
        lines.append("")
        line = f"SESSION: {' | '.join(parts)}"
        for w in session.get("warnings", []):
            line += f" | {w}"
        lines.append(line)

    # Today's / this week's derived stats
    derived = ctx.get("derived_stats", {})
    today_s = derived.get("today")
    if today_s:
        lines.append("")
        lines.append("TODAY'S STATS:")
        lines.append(
            f"  Trades: {today_s['trades']} | Wins: {today_s['wins']} | "
            f"Losses: {today_s['losses']} | Win Rate: {today_s['win_rate']}%"
        )
        lines.append(
            f"  Net P&L: ${today_s['net_pl']:+.2f} | "
            f"Best: ${today_s['best']:+.2f} | Worst: ${today_s['worst']:+.2f}"
        )
        lines.append(f"  Consecutive losses: {today_s.get('consecutive_losses', 0)}")
    week_s = derived.get("week")
    if week_s:
        lines.append("")
        lines.append("THIS WEEK:")
        lines.append(
            f"  Net P&L: ${week_s['net_pl']:+.2f} | Win Rate: {week_s['win_rate']}% "
            f"({week_s['trades']} trades)"
        )

    # Guardrail alerts (only if triggered)
    alerts = ctx.get("guardrail_alerts")
    if alerts:
        lines.append("")
        lines.append("ACTIVE GUARDRAILS:")
        for a in alerts:
            lines.append(f"  * {a}")

    # Volatility
    vol = ctx.get("volatility")
    if vol:
        lines.append("")
        desc = vol["label"]
        ratio = vol["ratio"]
        pips = vol["recent_avg_pips"]
        extra = ""
        if ratio != 1.0:
            extra = f" ({ratio}x normal)"
        lines.append(f"VOLATILITY: {desc}{extra} — recent M1 avg range {pips}p")

    cross = ctx.get("cross_assets")
    if cross:
        lines.append("")
        lines.append("CROSS-ASSET SNAPSHOT:")
        eurusd = cross.get("eurusd")
        dxy = cross.get("dxy_proxy")
        if eurusd and dxy:
            lines.append(f"  DXY proxy: {dxy} (via EUR/USD @ {eurusd})")
        oil = cross.get("bco_usd")
        if oil:
            lines.append(f"  Oil (BCO/USD): {oil}")

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
