import json, sys

with open("research_out/session_momentum_london_diag_500k.json") as f:
    r = json.load(f)

summary = r["results"]["summary"]
print("=== LONDON OVERALL ===")
print(f"Trades:    {summary['trades']}")
print(f"Win rate:  {summary['win_rate']:.3f}%")
print(f"Net USD:   ${summary['net_usd']:,.2f}")
print(f"Net pips:  {summary['net_pips']:.3f}")
print(f"Avg win:   {summary['avg_win_pips']:.3f} pips")
print(f"Avg loss:  {summary['avg_loss_pips']:.3f} pips")
print(f"PF:        {summary['profit_factor']:.4f}")
print(f"MaxDD:     ${summary['max_drawdown_usd']:,.2f}")

# by_strength if available
for section in ["by_strength", "by_hour", "by_session", "by_atr_bucket", "by_sl_bucket"]:
    if section in r["results"]:
        print(f"\n=== {section.upper()} ===")
        for row in r["results"][section]:
            print(json.dumps(row, indent=2))

# blocks
if "block_counts" in r:
    print("\n=== TOP BLOCKS ===")
    blocks = sorted(r["block_counts"].items(), key=lambda x: -x[1])
    for k, v in blocks[:20]:
        print(f"  {k}: {v:,}")
elif "diagnostics" in r and "block_counts" in r["diagnostics"]:
    print("\n=== TOP BLOCKS ===")
    blocks = sorted(r["diagnostics"]["block_counts"].items(), key=lambda x: -x[1])
    for k, v in blocks[:20]:
        print(f"  {k}: {v:,}")
