# Final Audit

Verdict: `PROMOTE_ROBOT_MODULE_FINAL`

Decision: promote `RobotModule` in `strategies/r5_trader.py` with `ROBOT_LAUNDRY / ROBOT_VACUUMING` raw spread mean reversion, `window=2000`, `min_history=2000`, `entry_z=2.25`, `target=10`, hold-to-flip, visible L1 crossing only.

Submission file: `strategies/r5_trader.py`. It is the only live production trader. Files in `strategies/archive/` are standalone references, rejected candidates, or research-only challengers and are not submission candidates.

Post-cleanup validation: `python -m compileall strategies\r5_trader.py` passed, and the final `--match-trades worse` backtest still produced total `651,869`, Day 2 `257,922`, Day 3 `148,634`, Day 4 `245,313`, max DD `33,138`.

## Baseline Gate

Original pre-promotion `strategies/r5_trader.py`, `--match-trades worse`, days 5-2/5-3/5-4:

| Day 2 | Day 3 | Day 4 | Total | Max DD | Sharpe |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 236,819 | 126,476 | 222,473 | 585,768 | 41,324 | 3.2545 |

Status: matched expected baseline. Promotion audit continued.

## Standalone Robot LV

Backtester replay for archived `strategies/archive/r5_robot_candidate.py` with `ROBOT_LAUNDRY / ROBOT_VACUUMING`, raw, `w=2000`, `z=2.25`:

| Total | Day 2 | Day 3 | Day 4 | Risk DD | Research DD | Forced flatten | Trades | Units |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 66,101 | 21,103 | 22,158 | 22,840 | 9,370 | 26,298 | 65,891 | 98 | 580 |

Product attribution:

| Product | PnL | Orders | Units | Final D4 position |
| --- | ---: | ---: | ---: | ---: |
| ROBOT_LAUNDRY | 31,458 | 50 | 290 | -10 |
| ROBOT_VACUUMING | 34,643 | 48 | 290 | 10 |

Standalone final positions by day:

| Day | ROBOT_LAUNDRY | ROBOT_VACUUMING |
| ---: | ---: | ---: |
| 2 | 10 | -10 |
| 3 | 10 | -10 |
| 4 | -10 | 10 |

Standalone stress:

| Stress | Total | Day 2 | Day 3 | Day 4 | Research DD |
| --- | ---: | ---: | ---: | ---: | ---: |
| +0 | 66,101 | 21,103 | 22,158 | 22,840 | 26,298 |
| +1 | 65,521 | 20,883 | 22,018 | 22,620 | 26,175 |
| +3 | 64,361 | 20,443 | 21,738 | 22,180 | 25,929 |
| +5 | 63,201 | 20,003 | 21,458 | 21,740 | 25,683 |
| +10 | 60,301 | 18,903 | 20,758 | 20,640 | 25,068 |

Standalone rolling worst slices:

| Slice | Worst delta | Location |
| ---: | ---: | --- |
| 1,000 | -8,580 | day 3, index 6003 to 7002 |
| 2,000 | -3,875 | day 3, index 531 to 2530 |

Parameter neighbourhood:

| Window | Entry z | Total | Day 2 | Day 3 | Day 4 | DD | Trades | Units |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,750 | 2.00 | 50,912 | 22,099 | 11,594 | 17,219 | 28,654 | 117 | 700 |
| 1,750 | 2.25 | 53,896 | 16,603 | 20,131 | 17,162 | 24,651 | 94 | 580 |
| 1,750 | 2.50 | 64,882 | 21,688 | 22,675 | 20,519 | 26,953 | 90 | 540 |
| 2,000 | 2.00 | 62,710 | 26,871 | 17,653 | 18,186 | 32,651 | 105 | 620 |
| 2,000 | 2.25 | 66,101 | 21,103 | 22,158 | 22,840 | 26,298 | 98 | 580 |
| 2,000 | 2.50 | 44,210 | 4,256 | 29,905 | 10,049 | 36,115 | 52 | 300 |
| 2,250 | 2.00 | 69,324 | 27,939 | 20,860 | 20,525 | 32,124 | 111 | 620 |
| 2,250 | 2.25 | 62,966 | 12,571 | 25,866 | 24,529 | 30,031 | 89 | 500 |
| 2,250 | 2.50 | 37,310 | -1,652 | 27,851 | 11,111 | 33,538 | 40 | 220 |

## Combined Final

Final `strategies/r5_trader.py`, days 5-2/5-3/5-4:

| Mode | Total | Day 2 | Day 3 | Day 4 | Max DD | Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| worse | 651,869 | 257,922 | 148,634 | 245,313 | 33,138 | 3.6342 |
| all | 651,869 | 257,922 | 148,634 | 245,313 | 33,138 | 3.6342 |
| none | 651,869 | 257,922 | 148,634 | 245,313 | 33,138 | 3.6342 |

Comparison against baseline:

| Baseline total | Final total | Delta | Baseline DD | Final DD |
| ---: | ---: | ---: | ---: | ---: |
| 585,768 | 651,869 | 66,101 | 41,324 | 33,138 |

Combined forced flatten and stress:

| Case | Total | Day 2 | Day 3 | Day 4 |
| --- | ---: | ---: | ---: | ---: |
| Base | 651,869 | 257,922 | 148,634 | 245,313 |
| Forced final flatten | 649,774 | n/a | n/a | n/a |
| +1 | 643,269 | 255,022 | 145,974 | 242,273 |
| +3 | 626,069 | 249,222 | 140,654 | 236,193 |
| +5 | 608,869 | 243,422 | 135,334 | 230,113 |
| +10 | 565,869 | 228,922 | 122,034 | 214,913 |

Combined rolling worst slices:

| Slice | Worst delta | Worst drawdown |
| ---: | ---: | ---: |
| 1,000 | -28,682 | 33,138 |
| 2,000 | -16,548 | 33,138 |

Combined D4 final positions:

| Product | Final position |
| --- | ---: |
| PEBBLES_XS | 10 |
| PEBBLES_S | -10 |
| PEBBLES_M | 10 |
| PEBBLES_L | -10 |
| PEBBLES_XL | 10 |
| TRANSLATOR_SPACE_GRAY | 10 |
| TRANSLATOR_ASTRO_BLACK | -10 |
| TRANSLATOR_ECLIPSE_CHARCOAL | -10 |
| TRANSLATOR_GRAPHITE_MIST | 10 |
| TRANSLATOR_VOID_BLUE | -10 |
| MICROCHIP_OVAL | -10 |
| MICROCHIP_TRIANGLE | 10 |
| ROBOT_LAUNDRY | -10 |
| ROBOT_VACUUMING | 10 |

## Order Audit

Audit issues: none.

Sandbox limit messages: none.

Last-timestamp order count: none.

Maximum absolute position: 10 for every traded product.

Robot traded products:

| Product | Orders | Units |
| --- | ---: | ---: |
| ROBOT_LAUNDRY | 50 | 290 |
| ROBOT_VACUUMING | 48 | 290 |

Forbidden Robot products had no orders: `ROBOT_MOPPING`, `ROBOT_DISHES`, `ROBOT_IRONING`.

Previous-round and rejected products had no orders: `GALAXY_SOUNDS_*`, `OXYGEN_SHAKE_*`, `PANEL_*`, `SLEEP_POD_*`, `SNACKPACK_*`, `UV_VISOR_*`, `MICROCHIP_CIRCLE`, `MICROCHIP_RECTANGLE`, `MICROCHIP_SQUARE`.

Production module equality versus baseline:

| Module | PnL identical | Order counts identical | Filled trades identical |
| --- | --- | --- | --- |
| Pebbles | true | true | true |
| Translator | true | true | true |
| Microchip | true | true | true |

Module order counts:

| Product | Baseline orders | Final orders |
| --- | ---: | ---: |
| PEBBLES_XS | 75 | 75 |
| PEBBLES_S | 95 | 95 |
| PEBBLES_M | 78 | 78 |
| PEBBLES_L | 71 | 71 |
| PEBBLES_XL | 84 | 84 |
| TRANSLATOR_SPACE_GRAY | 62 | 62 |
| TRANSLATOR_ASTRO_BLACK | 57 | 57 |
| TRANSLATOR_ECLIPSE_CHARCOAL | 60 | 60 |
| TRANSLATOR_GRAPHITE_MIST | 72 | 72 |
| TRANSLATOR_VOID_BLUE | 50 | 50 |
| MICROCHIP_OVAL | 152 | 152 |
| MICROCHIP_TRIANGLE | 159 | 159 |

## State And Runtime

Final combined audit:

| Ticks | Runtime seconds | ms/tick | Max traderData length |
| ---: | ---: | ---: | ---: |
| 30,000 | 111.158182 | 3.705273 | 68,840 |

Standalone Robot audit:

| Ticks | Runtime seconds | ms/tick | Max traderData length |
| ---: | ---: | ---: | ---: |
| 30,000 | 46.319029 | 1.543968 | 12,091 |

State namespaces: `p`, `tr`, `mc`, `rb`.

History caps: Pebbles 500 per product, Translator 1200 per product, Microchip 1000, Robot 2000.

Validation commands run:

```powershell
python -m compileall strategies\r5_trader.py
python -m prosperity4bt cli strategies/r5_trader.py 5-2 5-3 5-4 --data r5_data --match-trades worse --merge-pnl --out backtests/r5_trader_final_worse.log --no-progress @limits
python -m prosperity4bt cli strategies/r5_trader.py 5-2 5-3 5-4 --data r5_data --match-trades all --merge-pnl --out backtests/r5_trader_final_all.log --no-progress @limits
python -m prosperity4bt cli strategies/r5_trader.py 5-2 5-3 5-4 --data r5_data --match-trades none --merge-pnl --out backtests/r5_trader_final_none.log --no-progress @limits
```

Exact final file to submit: `strategies/r5_trader.py`.
