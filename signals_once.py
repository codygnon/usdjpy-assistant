from __future__ import annotations

import argparse

from adapters.broker import get_adapter
from core.profile import load_profile_v1
from core.signal_engine import (
    detect_latest_confirmed_cross_signal,
    evaluate_filters,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute signals once (no execution).")
    ap.add_argument("--profile", required=True, help="Path to profile JSON (v1 or legacy)")
    args = ap.parse_args()

    profile = load_profile_v1(args.profile)
    adapter = get_adapter(profile)
    adapter.initialize()
    try:
        adapter.ensure_symbol(profile.symbol)

        # Fetch data for the timeframes we care about
        data_by_tf = {
            "H4": adapter.get_bars(profile.symbol, "H4", 800),
            "M15": adapter.get_bars(profile.symbol, "M15", 2000),
            "M1": adapter.get_bars(profile.symbol, "M1", 3000),
        }

        print("profile:", profile.profile_name)
        print("symbol:", profile.symbol)

        # Evaluate configured cross setups
        for setup_name, setup in profile.strategy.setups.items():
            if not setup.enabled:
                continue

            tf = setup.timeframe
            df = data_by_tf[tf]

            sig = detect_latest_confirmed_cross_signal(
                profile=profile,
                df=df,
                tf=tf,
                ema_period=setup.ema,
                sma_period=setup.sma,
                confirm_bars=setup.confirmation.confirm_bars,
                require_close_on_correct_side=setup.confirmation.require_close_on_correct_side,
                min_distance_pips=setup.confirmation.min_distance_pips,
                max_wait_bars=setup.confirmation.max_wait_bars,
            )

            if sig is None:
                print(f"\nsetup={setup_name}: no confirmed signal on latest bar.")
                continue

            ok, filter_reasons = evaluate_filters(profile, data_by_tf, sig)
            if not ok:
                print(f"\nsetup={setup_name}: signal rejected")
                for r in filter_reasons:
                    print("-", r)
                continue

            print(f"\nsetup={setup_name}: SIGNAL")
            print("signal_id:", sig.signal_id)
            print("timeframe:", sig.timeframe, "side:", sig.side)
            print("cross_time:", sig.cross_time)
            print("confirm_time:", sig.confirm_time)
            print("entry_price_hint:", sig.entry_price_hint)
            print("reasons:")
            for r in (sig.reasons + filter_reasons):
                print("-", r)

    finally:
        adapter.shutdown()


if __name__ == "__main__":
    main()

