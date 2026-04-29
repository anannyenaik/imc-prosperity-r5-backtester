import json
import math
from typing import Optional

from datamodel import Order, OrderDepth, Symbol, TradingState


class Trader:
    ENABLE_PEBBLES = True

    PEBBLES: tuple[Symbol, ...] = (
        "PEBBLES_XS",
        "PEBBLES_S",
        "PEBBLES_M",
        "PEBBLES_L",
        "PEBBLES_XL",
    )

    WINDOW = 500
    ENTRY_Z = 2.35
    EXIT_Z: Optional[float] = None
    HOLD_TO_FLIP = True

    MIN_HISTORY = max(80, WINDOW // 4)
    MIN_STD = 1.0
    TARGET_SIZE = 10
    POSITION_LIMIT = 10
    MAX_ORDER_SIZE = 10

    HISTORY_LIMIT = 500
    RESIDUAL_SCALE = 10

    def _load_state(self, trader_data: str) -> tuple[dict[Symbol, list[int]], dict[Symbol, int]]:
        histories: dict[Symbol, list[int]] = {product: [] for product in self.PEBBLES}
        targets: dict[Symbol, int] = {product: 0 for product in self.PEBBLES}

        if not trader_data:
            return histories, targets

        try:
            loaded = json.loads(trader_data)
        except Exception:
            return histories, targets

        if not isinstance(loaded, dict):
            return histories, targets

        raw_histories = loaded.get("h", {})
        if isinstance(raw_histories, dict):
            for product in self.PEBBLES:
                raw_values = raw_histories.get(product, [])
                if not isinstance(raw_values, list):
                    continue
                cleaned: list[int] = []
                for value in raw_values[-self.HISTORY_LIMIT :]:
                    try:
                        cleaned.append(int(value))
                    except (TypeError, ValueError):
                        continue
                histories[product] = cleaned

        raw_targets = loaded.get("t", {})
        if isinstance(raw_targets, dict):
            for product in self.PEBBLES:
                try:
                    target = int(raw_targets.get(product, 0))
                except (TypeError, ValueError):
                    target = 0
                targets[product] = self._clamp(target, -self.TARGET_SIZE, self.TARGET_SIZE)

        return histories, targets

    def _dump_state(self, histories: dict[Symbol, list[int]], targets: dict[Symbol, int]) -> str:
        compact_histories = {product: histories.get(product, [])[-self.HISTORY_LIMIT :] for product in self.PEBBLES}
        compact_targets = {product: int(targets.get(product, 0)) for product in self.PEBBLES}
        return json.dumps({"h": compact_histories, "t": compact_targets}, separators=(",", ":"))

    def _best_bid_ask(self, order_depth: OrderDepth) -> tuple[Optional[int], Optional[int]]:
        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None
        return best_bid, best_ask

    def _mid_price(self, order_depth: OrderDepth) -> Optional[float]:
        best_bid, best_ask = self._best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return None
        return (best_bid + best_ask) / 2.0

    def _rolling_z_score(self, history: list[int], current_residual: int) -> Optional[float]:
        lookback = history[-self.WINDOW :]
        if len(lookback) < self.MIN_HISTORY:
            return None

        mean = sum(lookback) / len(lookback)
        mean_square = sum(value * value for value in lookback) / len(lookback)
        variance = max(0.0, mean_square - mean * mean)
        std = math.sqrt(variance)
        if std < self.MIN_STD * self.RESIDUAL_SCALE:
            return None

        return (current_residual - mean) / std

    def _target_from_signal(self, previous_target: int, z_score: Optional[float], has_min_history: bool) -> int:
        if not has_min_history or z_score is None:
            return 0 if not has_min_history else previous_target

        if z_score > self.ENTRY_Z:
            return -self.TARGET_SIZE
        if z_score < -self.ENTRY_Z:
            return self.TARGET_SIZE

        if not self.HOLD_TO_FLIP and self.EXIT_Z is not None and abs(z_score) < self.EXIT_Z:
            return 0

        return previous_target

    def _order_to_target(
        self,
        product: Symbol,
        order_depth: OrderDepth,
        current_position: int,
        target_position: int,
    ) -> list[Order]:
        orders: list[Order] = []
        target_position = self._clamp(target_position, -self.POSITION_LIMIT, self.POSITION_LIMIT)
        desired_delta = target_position - current_position
        if desired_delta == 0:
            return orders

        best_bid, best_ask = self._best_bid_ask(order_depth)

        if desired_delta > 0:
            if best_ask is None:
                return orders
            visible_volume = max(0, -order_depth.sell_orders.get(best_ask, 0))
            limit_room = max(0, self.POSITION_LIMIT - current_position)
            quantity = min(desired_delta, visible_volume, limit_room, self.MAX_ORDER_SIZE)
            if quantity > 0:
                orders.append(Order(product, best_ask, quantity))
            return orders

        if best_bid is None:
            return orders
        visible_volume = max(0, order_depth.buy_orders.get(best_bid, 0))
        limit_room = max(0, self.POSITION_LIMIT + current_position)
        quantity = min(-desired_delta, visible_volume, limit_room, self.MAX_ORDER_SIZE)
        if quantity > 0:
            orders.append(Order(product, best_bid, -quantity))
        return orders

    def _clamp(self, value: int, lower: int, upper: int) -> int:
        return max(lower, min(upper, value))

    def run(self, state: TradingState):
        histories, targets = self._load_state(state.traderData)
        result: dict[Symbol, list[Order]] = {}

        if not self.ENABLE_PEBBLES:
            return result, 0, self._dump_state(histories, targets)

        mids: dict[Symbol, float] = {}
        for product in self.PEBBLES:
            order_depth = state.order_depths.get(product)
            if order_depth is None:
                return result, 0, self._dump_state(histories, targets)

            mid = self._mid_price(order_depth)
            if mid is None:
                return result, 0, self._dump_state(histories, targets)
            mids[product] = mid

        group_mean = sum(mids.values()) / len(self.PEBBLES)
        scaled_residuals = {
            product: int(round((mids[product] - group_mean) * self.RESIDUAL_SCALE)) for product in self.PEBBLES
        }

        next_targets: dict[Symbol, int] = {}
        for product in self.PEBBLES:
            history = histories[product]
            has_min_history = len(history) >= self.MIN_HISTORY
            z_score = self._rolling_z_score(history, scaled_residuals[product])
            next_targets[product] = self._target_from_signal(targets.get(product, 0), z_score, has_min_history)

        for product in self.PEBBLES:
            order_depth = state.order_depths[product]
            current_position = state.position.get(product, 0)
            orders = self._order_to_target(product, order_depth, current_position, next_targets[product])
            if orders:
                result[product] = orders

        for product in self.PEBBLES:
            history = histories[product]
            history.append(scaled_residuals[product])
            if len(history) > self.HISTORY_LIMIT:
                del history[: len(history) - self.HISTORY_LIMIT]

        return result, 0, self._dump_state(histories, next_targets)
