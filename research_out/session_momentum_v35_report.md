# V35 Results

## Master Table (500k primary, 250k secondary)
| Run | 500k T | 500k WR | 500k USD | 500k PF | 500k DD | 250k T | 250k USD | 250k PF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v32b | 204 | 44.118% | 32731.011 | 1.5916 | 10669.478 | 89 | 29589.928 | 2.1741 |
| v35a | 232 | 43.966% | 31805.793 | 1.5599 | 10257.472 | 101 | 20109.148 | 1.8693 |
| v35b | 204 | 44.118% | 31657.661 | 1.5768 | 10779.012 | 89 | 28878.333 | 2.1489 |
| v35c | 203 | 44.828% | 35024.371 | 1.6055 | 9907.531 | 88 | 34347.867 | 2.2983 |
| v35d | 243 | 41.152% | 10525.205 | 1.3688 | 20568.534 | 101 | 27520.584 | 1.9886 |
| v35e | 283 | 40.989% | 14438.399 | 1.2928 | 14411.745 | 118 | 22146.925 | 1.7613 |
| v35f | 230 | 44.348% | 30662.671 | 1.5408 | 11060.512 | 100 | 23863.174 | 1.9621 |
| v35g | 317 | 41.640% | 10795.892 | 1.2697 | 23088.095 | 137 | 24490.931 | 1.5920 |

## 500k Session Split (London/NY)
| Run | London Trades | London WR | London USD | NY Trades | NY WR | NY USD |
|---|---:|---:|---:|---:|---:|---:|
| v32b | 105 | 41.905% | 2284.328 | 99 | 46.465% | 30446.683 |
| v35a | 125 | 42.400% | 3459.092 | 107 | 45.794% | 28346.701 |
| v35b | 105 | 41.905% | 2113.991 | 99 | 46.465% | 29543.670 |
| v35c | 105 | 41.905% | 2284.328 | 98 | 47.959% | 32740.043 |
| v35d | 133 | 39.098% | -680.615 | 110 | 43.636% | 11205.819 |
| v35e | 161 | 39.752% | 2416.964 | 122 | 42.623% | 12021.435 |
| v35f | 125 | 42.400% | 3459.092 | 105 | 46.667% | 27203.579 |
| v35g | 187 | 41.711% | 391.503 | 130 | 41.538% | 10404.390 |

## V35a Normal-Mid-ATR Admission
- `v5_strength_filter` count: V32b=6323, V35a=3589, delta=-2734
- Newly admitted `Normal` trades (500k): 43
- WR of admitted normal trades: 37.209%
- Net USD of admitted normal trades: -1919.097

## V35b / V35c Runner & Trail Diagnostics (500k)
| Run | MFE Capture (winners) | tp1_then_tp2 count | tp1_then_tp2 avg pips | tp1_then_trail count | tp1_then_trail avg pips |
|---|---:|---:|---:|---:|---:|
| v32b | 58.222% | 50 | 21.793 | 32 | 6.523 |
| v35b | 57.573% | 50 | 21.793 | 32 | 6.137 |
| v35c | 52.931% | 41 | 22.770 | 43 | 8.278 |

## V35d / V35e Cutoff Extension Diagnostics (500k)
| Run | Added trades vs V32b | New late-session trades | WR late trades | Overall WR |
|---|---:|---:|---:|---:|
| v35d | 39 | 38 | 23.684% | 41.152% |
| v35e | 79 | 78 | 32.051% | 40.989% |

## Selected V35 Winner (best net USD improvement)
- Winner config: `v35c`
- Saved to `research_out/session_momentum_v35_winner_config.json`
- 500k: trades=203, wr=44.828%, net_usd=35024.371, pf=1.6055, maxdd=9907.531
- 250k: trades=88, wr=53.409%, net_usd=34347.867, pf=2.2983

## 50k Regression (V35 winner config)
- trades=22, wr=68.182%, net_usd=21445.069, pf=5.7822, maxdd=1784.508
- London: trades=10, net_usd=6692.955
- NY: trades=12, net_usd=14752.114
