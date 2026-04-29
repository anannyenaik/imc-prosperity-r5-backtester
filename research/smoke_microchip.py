"""Smoke test for r5_microchip_analysis."""

import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
from r5_microchip_analysis import simulate_pair, simulate_basket, MICROCHIPS

t0 = time.time()
r = simulate_pair(("MICROCHIP_OVAL", "MICROCHIP_TRIANGLE"), residual_kind="raw", window=1000, min_history=1000, entry_z=1.5)
print(f"OT raw w=1000 ez=1.50: total={r.total:,.0f} D2={r.by_day[2]:,.0f} D3={r.by_day[3]:,.0f} D4={r.by_day[4]:,.0f} trades={r.trades} dd={r.max_drawdown:,.0f}")
print(f"  pos OVAL={r.final_position.get('MICROCHIP_OVAL', 0)} TRI={r.final_position.get('MICROCHIP_TRIANGLE', 0)}")
print(f"  by-product: {r.by_product}")
print(f"elapsed: {time.time()-t0:.2f}s")

t0 = time.time()
r = simulate_basket(window=1000, min_history=1000, entry_z=1.75)
print(f"Basket w=1000 ez=1.75: total={r.total:,.0f} D2={r.by_day[2]:,.0f} D3={r.by_day[3]:,.0f} D4={r.by_day[4]:,.0f} trades={r.trades} dd={r.max_drawdown:,.0f}")
print(f"  by-product: {r.by_product}")
print(f"elapsed: {time.time()-t0:.2f}s")
