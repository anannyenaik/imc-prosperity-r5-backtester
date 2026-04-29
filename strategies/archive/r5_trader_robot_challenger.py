"""Challenger trader: production (Pebbles + Translator + Microchip) + RobotLaundryVac module.

Pebbles, Translator and Microchip modules are copied verbatim from
strategies/r5_trader.py (no parameter or behaviour changes).

The added RobotLaundryVacModule trades only ROBOT_LAUNDRY/ROBOT_VACUUMING raw
spread mean reversion. Parameters are LOCKED to:

    raw spread, window=2000, min_history=2000, entry_z=2.25, target=10

Selected for: best three-day balance (32/34/35), lowest standalone drawdown
(26.3k), lowest trade count (98), and clean attribution (31.5k LAUNDRY,
34.6k VACUUMING). Stress holds: +5 → 63.2k, +10 → 60.3k.

For research-only parameter overrides, set RB_WINDOW / RB_ENTRY_Z env vars.
Production submission must run with no overrides.

Hard rules (no overfit knobs):
- raw spread (mid_LAUNDRY - mid_VACUUMING) only;
- previous-history-only rolling z-score;
- no timestamp/day logic;
- no warm-start;
- bounded rolling history;
- cross visible L1 only;
- position limit 10 per product;
- missing Robot data skips only this module.
"""
from __future__ import annotations

import json
import math
import os
from typing import Optional

from datamodel import Order, OrderDepth, Symbol, TradingState


def best_bid_ask(order_depth: OrderDepth) -> tuple[Optional[int], Optional[int]]:
    best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
    best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None
    return best_bid, best_ask


def mid_price(order_depth: OrderDepth) -> Optional[float]:
    bb, ba = best_bid_ask(order_depth)
    if bb is None or ba is None:
        return None
    return (bb + ba) / 2.0


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def clean_history(raw_values, history_limit: int) -> list[int]:
    cleaned: list[int] = []
    if not isinstance(raw_values, list):
        return cleaned
    for value in raw_values[-history_limit:]:
        try:
            cleaned.append(int(value))
        except (TypeError, ValueError):
            continue
    return cleaned


def rolling_z_score(
    history: list[int],
    current_residual: int,
    window: int,
    min_history: int,
    min_std: float,
    residual_scale: int,
) -> Optional[float]:
    lookback = history[-window:]
    if len(lookback) < min_history:
        return None
    mean = sum(lookback) / len(lookback)
    mean_square = sum(value * value for value in lookback) / len(lookback)
    variance = max(0.0, mean_square - mean * mean)
    std = math.sqrt(variance)
    if std < min_std * residual_scale:
        return None
    return (current_residual - mean) / std


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
    desired_delta = target_position - current_position
    if desired_delta == 0:
        return orders
    bb, ba = best_bid_ask(order_depth)
    if desired_delta > 0:
        if ba is None:
            return orders
        visible_volume = max(0, -order_depth.sell_orders.get(ba, 0))
        limit_room = max(0, position_limit - current_position)
        quantity = min(desired_delta, visible_volume, limit_room, max_order_size)
        if quantity > 0:
            orders.append(Order(product, ba, quantity))
        return orders
    if bb is None:
        return orders
    visible_volume = max(0, order_depth.buy_orders.get(bb, 0))
    limit_room = max(0, position_limit + current_position)
    quantity = min(-desired_delta, visible_volume, limit_room, max_order_size)
    if quantity > 0:
        orders.append(Order(product, bb, -quantity))
    return orders


# === Pebbles, Translator, Microchip — copied verbatim from r5_trader.py ===

class PebblesModule:
    PRODUCTS: tuple[Symbol, ...] = (
        "PEBBLES_XS",
        "PEBBLES_S",
        "PEBBLES_M",
        "PEBBLES_L",
        "PEBBLES_XL",
    )

    WINDOW = 500
    MIN_HISTORY = 125
    ENTRY_Z = 2.35
    HOLD_TO_FLIP = True
    EXIT_Z: Optional[float] = None

    MIN_STD = 1.0
    TARGET_SIZE = 10
    POSITION_LIMIT = 10
    MAX_ORDER_SIZE = 10
    HISTORY_LIMIT = 500
    RESIDUAL_SCALE = 10

    def empty_state(self) -> tuple[dict[Symbol, list[int]], dict[Symbol, int]]:
        return {product: [] for product in self.PRODUCTS}, {product: 0 for product in self.PRODUCTS}

    def load_state(self, loaded) -> tuple[dict[Symbol, list[int]], dict[Symbol, int]]:
        histories, targets = self.empty_state()
        if not isinstance(loaded, dict):
            return histories, targets
        raw_histories = loaded.get("h", {})
        if isinstance(raw_histories, dict):
            for product in self.PRODUCTS:
                histories[product] = clean_history(raw_histories.get(product, []), self.HISTORY_LIMIT)
        raw_targets = loaded.get("t", {})
        if isinstance(raw_targets, dict):
            for product in self.PRODUCTS:
                try:
                    target = int(raw_targets.get(product, 0))
                except (TypeError, ValueError):
                    target = 0
                targets[product] = clamp(target, -self.TARGET_SIZE, self.TARGET_SIZE)
        return histories, targets

    def dump_state(self, histories, targets):
        return {
            "h": {p: histories.get(p, [])[-self.HISTORY_LIMIT:] for p in self.PRODUCTS},
            "t": {p: int(targets.get(p, 0)) for p in self.PRODUCTS},
        }

    def target_from_signal(self, previous_target, z_score, has_min_history):
        if not has_min_history or z_score is None:
            return 0 if not has_min_history else previous_target
        if z_score > self.ENTRY_Z:
            return -self.TARGET_SIZE
        if z_score < -self.ENTRY_Z:
            return self.TARGET_SIZE
        if not self.HOLD_TO_FLIP and self.EXIT_Z is not None and abs(z_score) < self.EXIT_Z:
            return 0
        return previous_target

    def run(self, state, histories, targets, result):
        mids = {}
        for product in self.PRODUCTS:
            depth = state.order_depths.get(product)
            if depth is None:
                return targets
            m = mid_price(depth)
            if m is None:
                return targets
            mids[product] = m
        gm = sum(mids.values()) / len(self.PRODUCTS)
        residuals = {p: int(round((mids[p] - gm) * self.RESIDUAL_SCALE)) for p in self.PRODUCTS}
        next_targets = {}
        for p in self.PRODUCTS:
            history = histories[p]
            has_min = len(history) >= self.MIN_HISTORY
            z = rolling_z_score(history, residuals[p], self.WINDOW, self.MIN_HISTORY, self.MIN_STD, self.RESIDUAL_SCALE)
            next_targets[p] = self.target_from_signal(targets.get(p, 0), z, has_min)
        for p in self.PRODUCTS:
            orders = order_to_target(p, state.order_depths[p], state.position.get(p, 0),
                                     next_targets[p], self.POSITION_LIMIT, self.MAX_ORDER_SIZE)
            if orders:
                result[p] = orders
        for p in self.PRODUCTS:
            history = histories[p]
            history.append(residuals[p])
            if len(history) > self.HISTORY_LIMIT:
                del history[: len(history) - self.HISTORY_LIMIT]
        return next_targets


class TranslatorModule:
    PRODUCTS: tuple[Symbol, ...] = (
        "TRANSLATOR_SPACE_GRAY",
        "TRANSLATOR_ASTRO_BLACK",
        "TRANSLATOR_ECLIPSE_CHARCOAL",
        "TRANSLATOR_GRAPHITE_MIST",
        "TRANSLATOR_VOID_BLUE",
    )

    WINDOW = 1200
    MIN_HISTORY = 1200
    ENTRY_Z = 1.75
    HOLD_TO_FLIP = True
    EXIT_Z: Optional[float] = None
    MIN_STD = 1.0
    TARGET_SIZE = 10
    POSITION_LIMIT = 10
    MAX_ORDER_SIZE = 10
    HISTORY_LIMIT = 1200
    RESIDUAL_SCALE = 10

    def empty_state(self):
        return {p: [] for p in self.PRODUCTS}, {p: 0 for p in self.PRODUCTS}

    def load_state(self, loaded):
        histories, targets = self.empty_state()
        if not isinstance(loaded, dict):
            return histories, targets
        raw_histories = loaded.get("h", {})
        if isinstance(raw_histories, dict):
            for p in self.PRODUCTS:
                histories[p] = clean_history(raw_histories.get(p, []), self.HISTORY_LIMIT)
        raw_targets = loaded.get("t", {})
        if isinstance(raw_targets, dict):
            for p in self.PRODUCTS:
                try:
                    t = int(raw_targets.get(p, 0))
                except (TypeError, ValueError):
                    t = 0
                targets[p] = clamp(t, -self.TARGET_SIZE, self.TARGET_SIZE)
        return histories, targets

    def dump_state(self, histories, targets):
        return {
            "h": {p: histories.get(p, [])[-self.HISTORY_LIMIT:] for p in self.PRODUCTS},
            "t": {p: int(targets.get(p, 0)) for p in self.PRODUCTS},
        }

    def target_from_signal(self, prev, z, has_min):
        if not has_min or z is None:
            return 0 if not has_min else prev
        if z > self.ENTRY_Z:
            return -self.TARGET_SIZE
        if z < -self.ENTRY_Z:
            return self.TARGET_SIZE
        if not self.HOLD_TO_FLIP and self.EXIT_Z is not None and abs(z) < self.EXIT_Z:
            return 0
        return prev

    def run(self, state, histories, targets, result):
        mids = {}
        for p in self.PRODUCTS:
            depth = state.order_depths.get(p)
            if depth is None:
                return targets
            m = mid_price(depth)
            if m is None:
                return targets
            mids[p] = m
        gm = sum(mids.values()) / len(self.PRODUCTS)
        residuals = {p: int(round((mids[p] - gm) * self.RESIDUAL_SCALE)) for p in self.PRODUCTS}
        next_targets = {}
        for p in self.PRODUCTS:
            history = histories[p]
            has_min = len(history) >= self.MIN_HISTORY
            z = rolling_z_score(history, residuals[p], self.WINDOW, self.MIN_HISTORY, self.MIN_STD, self.RESIDUAL_SCALE)
            next_targets[p] = self.target_from_signal(targets.get(p, 0), z, has_min)
        for p in self.PRODUCTS:
            orders = order_to_target(p, state.order_depths[p], state.position.get(p, 0),
                                     next_targets[p], self.POSITION_LIMIT, self.MAX_ORDER_SIZE)
            if orders:
                result[p] = orders
        for p in self.PRODUCTS:
            history = histories[p]
            history.append(residuals[p])
            if len(history) > self.HISTORY_LIMIT:
                del history[: len(history) - self.HISTORY_LIMIT]
        return next_targets


class MicrochipModule:
    PRODUCTS: tuple[Symbol, ...] = ("MICROCHIP_OVAL", "MICROCHIP_TRIANGLE")

    WINDOW = 1000
    MIN_HISTORY = 1000
    ENTRY_Z = 1.50
    HOLD_TO_FLIP = True
    EXIT_Z: Optional[float] = None
    MIN_STD = 1.0
    TARGET_SIZE = 10
    POSITION_LIMIT = 10
    MAX_ORDER_SIZE = 10
    HISTORY_LIMIT = 1000
    RESIDUAL_SCALE = 10

    def empty_state(self):
        return [], {p: 0 for p in self.PRODUCTS}

    def load_state(self, loaded):
        history, targets = self.empty_state()
        if not isinstance(loaded, dict):
            return history, targets
        history = clean_history(loaded.get("h", []), self.HISTORY_LIMIT)
        raw_targets = loaded.get("t", {})
        if isinstance(raw_targets, dict):
            for p in self.PRODUCTS:
                try:
                    t = int(raw_targets.get(p, 0))
                except (TypeError, ValueError):
                    t = 0
                targets[p] = clamp(t, -self.TARGET_SIZE, self.TARGET_SIZE)
        return history, targets

    def dump_state(self, history, targets):
        return {"h": history[-self.HISTORY_LIMIT:], "t": {p: int(targets.get(p, 0)) for p in self.PRODUCTS}}

    def targets_from_signal(self, prev, z, has_min):
        oval, triangle = self.PRODUCTS
        if not has_min or z is None:
            if not has_min:
                return {oval: 0, triangle: 0}
            return dict(prev)
        if z > self.ENTRY_Z:
            return {oval: -self.TARGET_SIZE, triangle: self.TARGET_SIZE}
        if z < -self.ENTRY_Z:
            return {oval: self.TARGET_SIZE, triangle: -self.TARGET_SIZE}
        if not self.HOLD_TO_FLIP and self.EXIT_Z is not None and abs(z) < self.EXIT_Z:
            return {oval: 0, triangle: 0}
        return dict(prev)

    def run(self, state, history, targets, result):
        mids = {}
        for p in self.PRODUCTS:
            depth = state.order_depths.get(p)
            if depth is None:
                return targets
            m = mid_price(depth)
            if m is None:
                return targets
            mids[p] = m
        oval, triangle = self.PRODUCTS
        residual = int(round((mids[oval] - mids[triangle]) * self.RESIDUAL_SCALE))
        has_min = len(history) >= self.MIN_HISTORY
        z = rolling_z_score(history, residual, self.WINDOW, self.MIN_HISTORY, self.MIN_STD, self.RESIDUAL_SCALE)
        next_targets = self.targets_from_signal(targets, z, has_min)
        for p in self.PRODUCTS:
            orders = order_to_target(p, state.order_depths[p], state.position.get(p, 0),
                                     next_targets[p], self.POSITION_LIMIT, self.MAX_ORDER_SIZE)
            if orders:
                result[p] = orders
        history.append(residual)
        if len(history) > self.HISTORY_LIMIT:
            del history[: len(history) - self.HISTORY_LIMIT]
        return next_targets


# === New module: ROBOT_LAUNDRY / ROBOT_VACUUMING raw mean-reversion ===

class RobotLaundryVacModule:
    PRODUCTS: tuple[Symbol, ...] = ("ROBOT_LAUNDRY", "ROBOT_VACUUMING")

    # Locked production parameters. Override via RB_WINDOW / RB_ENTRY_Z for research only.
    WINDOW = int(os.environ.get("RB_WINDOW", "2000"))
    MIN_HISTORY = WINDOW
    ENTRY_Z = float(os.environ.get("RB_ENTRY_Z", "2.25"))
    HOLD_TO_FLIP = True
    EXIT_Z: Optional[float] = None

    MIN_STD = 1.0
    TARGET_SIZE = 10
    POSITION_LIMIT = 10
    MAX_ORDER_SIZE = 10
    HISTORY_LIMIT = WINDOW
    RESIDUAL_SCALE = 10

    def empty_state(self):
        return [], {p: 0 for p in self.PRODUCTS}

    def load_state(self, loaded):
        history, targets = self.empty_state()
        if not isinstance(loaded, dict):
            return history, targets
        history = clean_history(loaded.get("h", []), self.HISTORY_LIMIT)
        raw_targets = loaded.get("t", {})
        if isinstance(raw_targets, dict):
            for p in self.PRODUCTS:
                try:
                    t = int(raw_targets.get(p, 0))
                except (TypeError, ValueError):
                    t = 0
                targets[p] = clamp(t, -self.TARGET_SIZE, self.TARGET_SIZE)
        return history, targets

    def dump_state(self, history, targets):
        return {"h": history[-self.HISTORY_LIMIT:], "t": {p: int(targets.get(p, 0)) for p in self.PRODUCTS}}

    def targets_from_signal(self, prev, z, has_min):
        laundry, vac = self.PRODUCTS
        if not has_min or z is None:
            if not has_min:
                return {laundry: 0, vac: 0}
            return dict(prev)
        if z > self.ENTRY_Z:
            return {laundry: -self.TARGET_SIZE, vac: self.TARGET_SIZE}
        if z < -self.ENTRY_Z:
            return {laundry: self.TARGET_SIZE, vac: -self.TARGET_SIZE}
        if not self.HOLD_TO_FLIP and self.EXIT_Z is not None and abs(z) < self.EXIT_Z:
            return {laundry: 0, vac: 0}
        return dict(prev)

    def run(self, state, history, targets, result):
        mids = {}
        for p in self.PRODUCTS:
            depth = state.order_depths.get(p)
            if depth is None:
                return targets  # missing data → skip module, do not flip
            m = mid_price(depth)
            if m is None:
                return targets
            mids[p] = m
        laundry, vac = self.PRODUCTS
        residual = int(round((mids[laundry] - mids[vac]) * self.RESIDUAL_SCALE))
        has_min = len(history) >= self.MIN_HISTORY
        z = rolling_z_score(history, residual, self.WINDOW, self.MIN_HISTORY, self.MIN_STD, self.RESIDUAL_SCALE)
        next_targets = self.targets_from_signal(targets, z, has_min)
        for p in self.PRODUCTS:
            orders = order_to_target(p, state.order_depths[p], state.position.get(p, 0),
                                     next_targets[p], self.POSITION_LIMIT, self.MAX_ORDER_SIZE)
            if orders:
                result[p] = orders
        history.append(residual)
        if len(history) > self.HISTORY_LIMIT:
            del history[: len(history) - self.HISTORY_LIMIT]
        return next_targets


class Trader:
    PEBBLES_STATE_KEY = "p"
    TRANSLATOR_STATE_KEY = "tr"
    MICROCHIP_STATE_KEY = "mc"
    ROBOT_STATE_KEY = "rb"

    def __init__(self) -> None:
        self.pebbles = PebblesModule()
        self.translator = TranslatorModule()
        self.microchip = MicrochipModule()
        self.robot = RobotLaundryVacModule()

    def _load_json(self, trader_data: str) -> dict:
        if not trader_data:
            return {}
        try:
            loaded = json.loads(trader_data)
        except Exception:
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def run(self, state: TradingState):
        loaded = self._load_json(state.traderData)

        pebbles_h, pebbles_t = self.pebbles.load_state(loaded.get(self.PEBBLES_STATE_KEY, {}))
        translator_h, translator_t = self.translator.load_state(loaded.get(self.TRANSLATOR_STATE_KEY, {}))
        microchip_h, microchip_t = self.microchip.load_state(loaded.get(self.MICROCHIP_STATE_KEY, {}))
        robot_h, robot_t = self.robot.load_state(loaded.get(self.ROBOT_STATE_KEY, {}))

        result: dict[Symbol, list[Order]] = {}

        next_pebbles = self.pebbles.run(state, pebbles_h, pebbles_t, result)
        next_translator = self.translator.run(state, translator_h, translator_t, result)
        next_microchip = self.microchip.run(state, microchip_h, microchip_t, result)
        next_robot = self.robot.run(state, robot_h, robot_t, result)

        trader_data = json.dumps(
            {
                self.PEBBLES_STATE_KEY: self.pebbles.dump_state(pebbles_h, next_pebbles),
                self.TRANSLATOR_STATE_KEY: self.translator.dump_state(translator_h, next_translator),
                self.MICROCHIP_STATE_KEY: self.microchip.dump_state(microchip_h, next_microchip),
                self.ROBOT_STATE_KEY: self.robot.dump_state(robot_h, next_robot),
            },
            separators=(",", ":"),
        )
        return result, 0, trader_data
