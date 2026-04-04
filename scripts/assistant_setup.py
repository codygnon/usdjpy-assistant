#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.assistant.config import AssistantConfig
from core.assistant.oanda_client import OandaClient


def _prompt_required(label: str) -> str:
    while True:
        value = input(f"{label}: ").strip()
        if value:
            return value
        print("This field is required.")


def _prompt_default(label: str, default: str) -> str:
    value = input(f"{label} [{default}]: ").strip()
    return value or default


def main() -> None:
    print(
        "\n"
        "══════════════════════════════════════════════════\n"
        "         Trading Assistant Setup Wizard\n"
        "══════════════════════════════════════════════════\n"
    )

    while True:
        config = AssistantConfig()
        config.oanda.account_id = _prompt_required("OANDA account ID")
        config.oanda.api_token = _prompt_required("OANDA API token")
        config.oanda.environment = _prompt_default("Environment (practice/live)", "practice").strip().lower()
        config.risk.max_risk_per_trade = float(_prompt_default("Max risk per trade as decimal (1% = 0.01)", "0.01"))
        config.risk.max_position_size_lots = int(_prompt_default("Max position size in lots", "20"))
        config.risk.catastrophic_stop_pips = float(_prompt_default("Catastrophic stop in pips", "200"))

        errors = config.validate()
        if errors:
            print("\nConfiguration errors:")
            for error in errors:
                print(f"  - {error}")
            print("\nPlease try again.\n")
            continue
        break

    config.save()
    print(f"\nSaved configuration to {ROOT / 'config' / 'assistant_config.yaml'}")

    test_now = _prompt_default("Test OANDA connection now? (y/n)", "y").strip().lower()
    if test_now == "y":
        client = OandaClient(
            account_id=config.oanda.account_id,
            api_token=config.oanda.api_token,
            environment=config.oanda.environment,
        )
        if client.test_connection():
            print("Connection test succeeded.")
        else:
            print("Connection test failed. Double-check your credentials and environment.")


if __name__ == "__main__":
    main()
