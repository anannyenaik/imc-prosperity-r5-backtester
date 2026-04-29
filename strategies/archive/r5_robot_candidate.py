"""Configurable Robot pair / basket residual mean-reversion candidate.

Reads parameters from environment variables so the same file can be swept by a
research script:

  ROBOT_PAIR_A          – primary symbol (e.g. ROBOT_LAUNDRY)
  ROBOT_PAIR_B          – secondary symbol (e.g. ROBOT_VACUUMING)
  ROBOT_RESIDUAL        – "raw" (default) or "logratio"
  ROBOT_WINDOW          – rolling window (default 2250)
  ROBOT_MIN_HISTORY     – min history before trading (default = window)
  ROBOT_ENTRY_Z         – entry z-score (default 2.00)
  ROBOT_EXIT_Z          – optional exit z (default unset = hold-to-flip)
  ROBOT_TARGET          – target size 1-10 (default 10)
  ROBOT_MIN_STD         – min std multiplier (default 1.0)

  ROBOT_BASKET          – if "1" trade the all-five basket residual instead of a pair
  ROBOT_BASKET_PRODUCTS – comma list to override basket members

State is bounded; no full public history retained. No timestamp logic. Cross
visible L1 only. Position limit respected per product.
"""
from __future__ import annotations

import json
import math
import os
from typing import Optional

from datamodel import Order, OrderDepth, Symbol, TradingState

ALL_ROBOTS = (
    "ROBOT_VACUUMING",
    "ROBOT_MOPPING",
    "ROBOT_DISHES",
    "ROBOT_LAUNDRY",
    "ROBOT_IRONING",
)


def best_bid_ask(order_depth: OrderDepth) -> tuple[Optional[int], Optional[int]]:
    best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
    best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None
    return best_bid, best_ask


def mid_price(order_depth: OrderDepth) -> Optional[float]:
    bb, ba = best_bid_ask(order_depth)
    if bb is None or ba is None:
        return None
    return (bb + ba) / 2.0


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def order_to_target(
    product: Symbol,
    order_depth: OrderDepth,
    current_position: int,
    target_position: int,
    position_limit: int,
    max_order_size: int,
) -> list[Order]:
    orders: list[Order] = []
    target_position = clamp(target_position, -position_limit, position_limit)
    delta = target_position - current_position
    if delta == 0:
        return orders
    bb, ba = best_bid_ask(order_depth)
    if delta > 0:
        if ba is None:
            return orders
        vis = max(0, -order_depth.sell_orders.get(ba, 0))
        room = max(0, position_limit - current_position)
        q = min(delta, vis, room, max_order_size)
        if q > 0:
            orders.append(Order(product, ba, q))
        return orders
    if bb is None:
        return orders
    vis = max(0, order_depth.buy_orders.get(bb, 0))
    room = max(0, position_limit + current_position)
    q = min(-delta, vis, room, max_order_size)
    if q > 0:
        orders.append(Order(product, bb, -q))
    return orders


def rolling_z(
    history: list[int],
    current: int,
    window: int,
    min_history: int,
    min_std: float,
    scale: int,
) -> Optional[float]:
    look = history[-window:]
    if len(look) < min_history:
        return None
    mean = sum(look) / len(look)
    msq = sum(v * v for v in look) / len(look)
    var = max(0.0, msq - mean * mean)
    std = math.sqrt(var)
    if std < min_std * scale:
        return None
    return (current - mean) / std


class RobotPairModule:
    POSITION_LIMIT = 10
    MAX_ORDER_SIZE = 10
    MIN_STD = 1.0
    SCALE = 10  # integer residual scaling

    def __init__(self) -> None:
        self.product_a: Symbol = os.environ.get("ROBOT_PAIR_A", "ROBOT_LAUNDRY")
        self.product_b: Symbol = os.environ.get("ROBOT_PAIR_B", "ROBOT_VACUUMING")
        self.residual_kind = os.environ.get("ROBOT_RESIDUAL", "raw")  # raw or logratio
        self.window = int(os.environ.get("ROBOT_WINDOW", "2250"))
        self.min_history = int(os.environ.get("ROBOT_MIN_HISTORY", str(self.window)))
        self.entry_z = float(os.environ.get("ROBOT_ENTRY_Z", "2.00"))
        exit_z = os.environ.get("ROBOT_EXIT_Z", "")
        self.exit_z: Optional[float] = float(exit_z) if exit_z else None
        self.target_size = int(os.environ.get("ROBOT_TARGET", "10"))
        self.min_std = float(os.environ.get("ROBOT_MIN_STD", "1.0"))
        self.history_limit = self.window + 5

    @property
    def products(self) -> tuple[Symbol, Symbol]:
        return self.product_a, self.product_b

    def _residual(self, mid_a: float, mid_b: float) -> int:
        if self.residual_kind == "logratio":
            return int(round((math.log(mid_a) - math.log(mid_b)) * 100000.0))
        return int(round((mid_a - mid_b) * self.SCALE))

    def empty_state(self) -> tuple[list[int], dict[Symbol, int]]:
        return [], {self.product_a: 0, self.product_b: 0}

    def load_state(self, loaded) -> tuple[list[int], dict[Symbol, int]]:
        history, targets = self.empty_state()
        if not isinstance(loaded, dict):
            return history, targets
        h = loaded.get("h", [])
        if isinstance(h, list):
            for v in h[-self.history_limit:]:
                try:
                    history.append(int(v))
                except (TypeError, ValueError):
                    continue
        t = loaded.get("t", {})
        if isinstance(t, dict):
            for prod in (self.product_a, self.product_b):
                try:
                    targets[prod] = clamp(int(t.get(prod, 0)), -self.target_size, self.target_size)
                except (TypeError, ValueError):
                    targets[prod] = 0
        return history, targets

    def dump_state(self, history: list[int], targets: dict[Symbol, int]) -> dict:
        return {
            "h": history[-self.history_limit:],
            "t": {p: int(targets.get(p, 0)) for p in (self.product_a, self.product_b)},
        }

    def run(
        self,
        state: TradingState,
        history: list[int],
        targets: dict[Symbol, int],
        result: dict[Symbol, list[Order]],
    ) -> dict[Symbol, int]:
        depth_a = state.order_depths.get(self.product_a)
        depth_b = state.order_depths.get(self.product_b)
        if depth_a is None or depth_b is None:
            return targets
        mid_a = mid_price(depth_a)
        mid_b = mid_price(depth_b)
        if mid_a is None or mid_b is None:
            return targets

        residual = self._residual(mid_a, mid_b)

        scale = 100000 if self.residual_kind == "logratio" else self.SCALE
        z = rolling_z(history, residual, self.window, self.min_history, self.min_std, scale)
        has_min = len(history) >= self.min_history

        next_targets = dict(targets)
        if has_min and z is not None:
            if z > self.entry_z:
                next_targets = {self.product_a: -self.target_size, self.product_b: self.target_size}
            elif z < -self.entry_z:
                next_targets = {self.product_a: self.target_size, self.product_b: -self.target_size}
            elif self.exit_z is not None and abs(z) < self.exit_z:
                next_targets = {self.product_a: 0, self.product_b: 0}
        elif not has_min:
            next_targets = {self.product_a: 0, self.product_b: 0}

        for prod in (self.product_a, self.product_b):
            depth = state.order_depths.get(prod)
            if depth is None:
                continue
            orders = order_to_target(
                prod,
                depth,
                state.position.get(prod, 0),
                next_targets.get(prod, 0),
                self.POSITION_LIMIT,
                self.MAX_ORDER_SIZE,
            )
            if orders:
                result[prod] = orders

        history.append(residual)
        if len(history) > self.history_limit:
            del history[: len(history) - self.history_limit]

        return next_targets


class RobotBasketModule:
    POSITION_LIMIT = 10
    MAX_ORDER_SIZE = 10
    SCALE = 10

    def __init__(self) -> None:
        prods = os.environ.get("ROBOT_BASKET_PRODUCTS", ",".join(ALL_ROBOTS))
        self.products: tuple[Symbol, ...] = tuple(p.strip() for p in prods.split(",") if p.strip())
        self.window = int(os.environ.get("ROBOT_WINDOW", "2000"))
        self.min_history = int(os.environ.get("ROBOT_MIN_HISTORY", str(self.window)))
        self.entry_z = float(os.environ.get("ROBOT_ENTRY_Z", "2.00"))
        self.target_size = int(os.environ.get("ROBOT_TARGET", "10"))
        self.min_std = float(os.environ.get("ROBOT_MIN_STD", "1.0"))
        self.history_limit = self.window + 5

    def empty_state(self):
        return ({p: [] for p in self.products}, {p: 0 for p in self.products})

    def load_state(self, loaded):
        histories, targets = self.empty_state()
        if not isinstance(loaded, dict):
            return histories, targets
        rh = loaded.get("h", {})
        if isinstance(rh, dict):
            for p in self.products:
                lst = rh.get(p, [])
                if isinstance(lst, list):
                    histories[p] = [int(v) for v in lst[-self.history_limit:] if isinstance(v, (int, float))]
        rt = loaded.get("t", {})
        if isinstance(rt, dict):
            for p in self.products:
                try:
                    targets[p] = clamp(int(rt.get(p, 0)), -self.target_size, self.target_size)
                except (TypeError, ValueError):
                    targets[p] = 0
        return histories, targets

    def dump_state(self, histories, targets):
        return {
            "h": {p: histories.get(p, [])[-self.history_limit:] for p in self.products},
            "t": {p: int(targets.get(p, 0)) for p in self.products},
        }

    def run(self, state, histories, targets, result):
        mids = {}
        for p in self.products:
            d = state.order_depths.get(p)
            if d is None:
                return targets
            m = mid_price(d)
            if m is None:
                return targets
            mids[p] = m
        gm = sum(mids.values()) / len(self.products)
        residuals = {p: int(round((mids[p] - gm) * self.SCALE)) for p in self.products}

        next_targets: dict[Symbol, int] = {}
        for p in self.products:
            history = histories[p]
            has_min = len(history) >= self.min_history
            z = rolling_z(history, residuals[p], self.window, self.min_history, 1.0, self.SCALE)
            prev = targets.get(p, 0)
            if not has_min or z is None:
                next_targets[p] = 0 if not has_min else prev
                continue
            if z > self.entry_z:
                next_targets[p] = -self.target_size
            elif z < -self.entry_z:
                next_targets[p] = self.target_size
            else:
                next_targets[p] = prev

        for p in self.products:
            depth = state.order_depths.get(p)
            if depth is None:
                continue
            orders = order_to_target(
                p,
                depth,
                state.position.get(p, 0),
                next_targets.get(p, 0),
                self.POSITION_LIMIT,
                self.MAX_ORDER_SIZE,
            )
            if orders:
                result[p] = orders

        for p in self.products:
            h = histories[p]
            h.append(residuals[p])
            if len(h) > self.history_limit:
                del h[: len(h) - self.history_limit]

        return next_targets


class Trader:
    STATE_KEY = "rb"

    def __init__(self) -> None:
        self.is_basket = os.environ.get("ROBOT_BASKET", "0") == "1"
        self.module = RobotBasketModule() if self.is_basket else RobotPairModule()

    def _load(self, td: str) -> dict:
        if not td:
            return {}
        try:
            x = json.loads(td)
        except Exception:
            return {}
        return x if isinstance(x, dict) else {}

    def run(self, state: TradingState):
        loaded = self._load(state.traderData)
        sub = loaded.get(self.STATE_KEY, {})
        if self.is_basket:
            histories, targets = self.module.load_state(sub)
            result: dict[Symbol, list[Order]] = {}
            nt = self.module.run(state, histories, targets, result)
            td = json.dumps({self.STATE_KEY: self.module.dump_state(histories, nt)}, separators=(",", ":"))
        else:
            history, targets = self.module.load_state(sub)
            result = {}
            nt = self.module.run(state, history, targets, result)
            td = json.dumps({self.STATE_KEY: self.module.dump_state(history, nt)}, separators=(",", ":"))
        return result, 0, td
