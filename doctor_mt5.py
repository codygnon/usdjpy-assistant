import argparse
import json
import MetaTrader5 as mt5
import pandas as pd

from core.profile import load_profile_v1


def load_profile(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, help="Path to profile JSON")
    args = ap.parse_args()

    prof_v1 = load_profile_v1(args.profile)
    symbol = prof_v1.symbol

    print("=== DOCTOR CHECK ===")
    print("profile:", prof_v1.profile_name)
    print("symbol:", symbol)

    ok = mt5.initialize()
    print("\nMT5 initialize:", ok)
    print("MT5 last_error:", mt5.last_error())
    if not ok:
        return

    acct = mt5.account_info()
    print("\naccount_info:", acct)
    if acct is None:
        print("FAIL Not logged into MT5 terminal.")
        mt5.shutdown()
        return

    # Symbol checks
    sel = mt5.symbol_select(symbol, True)
    print("\nsymbol_select:", sel)
    print("last_error:", mt5.last_error())

    info = mt5.symbol_info(symbol)
    print("\nsymbol_info exists:", info is not None)
    if info is not None:
        print("digits:", info.digits, "visible:", info.visible, "trade_mode:", info.trade_mode)

    tick = mt5.symbol_info_tick(symbol)
    print("\ntick is None:", tick is None)
    print("tick:", tick)
    if tick is not None:
        spread_price = tick.ask - tick.bid
        pip_size = float(prof_v1.pip_size)
        spread_pips = spread_price / pip_size if pip_size else None
        print("spread_price:", spread_price)
        print("spread_pips:", spread_pips)

    # Data checks
    def check_tf(tf, name, n=10):
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, n)
        if rates is None or len(rates) == 0:
            print(f"\n{name}: FAIL no bars. last_error:", mt5.last_error())
            return
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        print(f"\n{name}: OK bars={len(df)} last_time={df['time'].iloc[-1]} last_close={df['close'].iloc[-1]}")

    check_tf(mt5.TIMEFRAME_M1, "M1")
    check_tf(mt5.TIMEFRAME_M15, "M15")
    check_tf(mt5.TIMEFRAME_H4, "H4")

    mt5.shutdown()
    print("\nOK Doctor check completed.")


if __name__ == "__main__":
    main()