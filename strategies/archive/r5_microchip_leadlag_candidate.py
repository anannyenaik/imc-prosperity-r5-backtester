import json
import os
from typing import Optional

from datamodel import Order, OrderDepth, Symbol, TradingState


def best_bid_ask(order_depth: OrderDepth) -> tuple[Optional[int], Optional[int]]:
    best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
    best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None
    return best_bid, best_ask


def mid_price(order_depth: OrderDepth) -> Optional[float]:
    best_bid, best_ask = best_bid_ask(order_depth)
    if best_bid is None or best_ask is None:
        return None
    return (best_bid + best_ask) / 2.0


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


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

    best_bid, best_ask = best_bid_ask(order_depth)
    if desired_delta > 0:
        if best_ask is None:
            return orders
        visible_volume = max(0, -order_depth.sell_orders.get(best_ask, 0))
        limit_room = max(0, position_limit - current_position)
        quantity = min(desired_delta, visible_volume, limit_room, max_order_size)
        if quantity > 0:
            orders.append(Order(product, best_ask, quantity))
        return orders

    if best_bid is None:
        return orders
    visible_volume = max(0, order_depth.buy_orders.get(best_bid, 0))
    limit_room = max(0, position_limit + current_position)
    quantity = min(-desired_delta, visible_volume, limit_room, max_order_size)
    if quantity > 0:
        orders.append(Order(product, best_bid, -quantity))
    return orders


class MicrochipLeadLagModule:
    CIRCLE: Symbol = "MICROCHIP_CIRCLE"
    LAGS: dict[Symbol, int] = {
        "MICROCHIP_OVAL": 50,
        "MICROCHIP_SQUARE": 100,
        "MICROCHIP_RECTANGLE": 150,
        "MICROCHIP_TRIANGLE": 200,
    }
    PROFILES = {
        "rectangle_htns26": {
            "products": ("MICROCHIP_RECTANGLE",),
            "off_delta": -1,
            "k": 1,
            "threshold": 26.0,
        },
        "square_rectangle_htns26": {
            "products": ("MICROCHIP_SQUARE", "MICROCHIP_RECTANGLE"),
            "off_delta": -1,
            "k": 1,
            "threshold": 26.0,
        },
        "all_followers_htns30": {
            "products": (
                "MICROCHIP_OVAL",
                "MICROCHIP_SQUARE",
                "MICROCHIP_RECTANGLE",
                "MICROCHIP_TRIANGLE",
            ),
            "off_delta": -1,
            "k": 1,
            "threshold": 30.0,
        },
        "oval_triangle_htns30": {
            "products": ("MICROCHIP_OVAL", "MICROCHIP_TRIANGLE"),
            "off_delta": -1,
            "k": 1,
            "threshold": 30.0,
        },
    }
    DEFAULT_PROFILE = "rectangle_htns26"

    TARGET_SIZE = 10
    POSITION_LIMIT = 10
    MAX_ORDER_SIZE = 10
    MID_SCALE = 2

    def __init__(self, profile_name: Optional[str] = None) -> None:
        selected = profile_name or os.environ.get("LEADLAG_PROFILE", self.DEFAULT_PROFILE)
        if selected not in self.PROFILES:
            selected = self.DEFAULT_PROFILE
        self.profile_name = selected
        profile = self.PROFILES[selected]
        self.products: tuple[Symbol, ...] = profile["products"]
        self.off_delta = int(profile["off_delta"])
        self.k = int(profile["k"])
        self.threshold = float(profile["threshold"])
        max_lag = max(self.LAGS[product] for product in self.products)
        self.history_limit = max_lag + abs(self.off_delta) + self.k + 5

    def empty_state(self) -> tuple[list[int], dict[Symbol, int]]:
        return [], {product: 0 for product in self.products}

    def load_state(self, loaded) -> tuple[list[int], dict[Symbol, int]]:
        history, targets = self.empty_state()
        if not isinstance(loaded, dict):
            return history, targets

        raw_history = loaded.get("m", [])
        if isinstance(raw_history, list):
            for value in raw_history[-self.history_limit :]:
                try:
                    history.append(int(value))
                except (TypeError, ValueError):
                    continue

        raw_targets = loaded.get("t", {})
        if isinstance(raw_targets, dict):
            for product in self.products:
                try:
                    target = int(raw_targets.get(product, 0))
                except (TypeError, ValueError):
                    target = 0
                targets[product] = clamp(target, -self.TARGET_SIZE, self.TARGET_SIZE)
        return history, targets

    def dump_state(self, history: list[int], targets: dict[Symbol, int]) -> dict:
        return {
            "m": history[-self.history_limit :],
            "t": {product: int(targets.get(product, 0)) for product in self.products},
        }

    def _signal(self, mids: list[int], product: Symbol) -> float:
        current_index = len(mids) - 1
        event_index = current_index - (self.LAGS[product] + self.off_delta)
        previous_index = event_index - self.k
        if previous_index < 0:
            return 0.0
        return (mids[event_index] - mids[previous_index]) / self.MID_SCALE

    def run(
        self,
        state: TradingState,
        history: list[int],
        targets: dict[Symbol, int],
        result: dict[Symbol, list[Order]],
    ) -> dict[Symbol, int]:
        circle_depth = state.order_depths.get(self.CIRCLE)
        if circle_depth is None:
            return targets
        circle_mid = mid_price(circle_depth)
        if circle_mid is None:
            return targets

        current_mid = int(round(circle_mid * self.MID_SCALE))
        mids = history + [current_mid]
        next_targets = dict(targets)

        for product in self.products:
            order_depth = state.order_depths.get(product)
            if order_depth is None:
                continue
            if mid_price(order_depth) is None:
                continue
            signal = self._signal(mids, product)
            if signal > self.threshold:
                next_targets[product] = self.TARGET_SIZE
            elif signal < -self.threshold:
                next_targets[product] = -self.TARGET_SIZE

        for product in self.products:
            order_depth = state.order_depths.get(product)
            if order_depth is None:
                continue
            orders = order_to_target(
                product,
                order_depth,
                state.position.get(product, 0),
                next_targets.get(product, 0),
                self.POSITION_LIMIT,
                self.MAX_ORDER_SIZE,
            )
            if orders:
                result[product] = orders

        history.append(current_mid)
        if len(history) > self.history_limit:
            del history[: len(history) - self.history_limit]

        return next_targets


class Trader:
    STATE_KEY = "mcl"

    def __init__(self) -> None:
        self.module = MicrochipLeadLagModule()

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
        history, targets = self.module.load_state(loaded.get(self.STATE_KEY, {}))
        result: dict[Symbol, list[Order]] = {}
        next_targets = self.module.run(state, history, targets, result)
        trader_data = json.dumps(
            {self.STATE_KEY: self.module.dump_state(history, next_targets)},
            separators=(",", ":"),
        )
        return result, 0, trader_data
