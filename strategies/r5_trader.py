import json
import math
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

    def dump_state(self, histories: dict[Symbol, list[int]], targets: dict[Symbol, int]) -> dict:
        return {
            "h": {product: histories.get(product, [])[-self.HISTORY_LIMIT:] for product in self.PRODUCTS},
            "t": {product: int(targets.get(product, 0)) for product in self.PRODUCTS},
        }

    def target_from_signal(self, previous_target: int, z_score: Optional[float], has_min_history: bool) -> int:
        if not has_min_history or z_score is None:
            return 0 if not has_min_history else previous_target
        if z_score > self.ENTRY_Z:
            return -self.TARGET_SIZE
        if z_score < -self.ENTRY_Z:
            return self.TARGET_SIZE
        if not self.HOLD_TO_FLIP and self.EXIT_Z is not None and abs(z_score) < self.EXIT_Z:
            return 0
        return previous_target

    def run(
        self,
        state: TradingState,
        histories: dict[Symbol, list[int]],
        targets: dict[Symbol, int],
        result: dict[Symbol, list[Order]],
    ) -> dict[Symbol, int]:
        mids: dict[Symbol, float] = {}
        for product in self.PRODUCTS:
            order_depth = state.order_depths.get(product)
            if order_depth is None:
                return targets
            mid = mid_price(order_depth)
            if mid is None:
                return targets
            mids[product] = mid

        group_mean = sum(mids.values()) / len(self.PRODUCTS)
        residuals = {
            product: int(round((mids[product] - group_mean) * self.RESIDUAL_SCALE))
            for product in self.PRODUCTS
        }

        next_targets: dict[Symbol, int] = {}
        for product in self.PRODUCTS:
            history = histories[product]
            has_min_history = len(history) >= self.MIN_HISTORY
            z_score = rolling_z_score(
                history,
                residuals[product],
                self.WINDOW,
                self.MIN_HISTORY,
                self.MIN_STD,
                self.RESIDUAL_SCALE,
            )
            next_targets[product] = self.target_from_signal(targets.get(product, 0), z_score, has_min_history)

        for product in self.PRODUCTS:
            orders = order_to_target(
                product,
                state.order_depths[product],
                state.position.get(product, 0),
                next_targets[product],
                self.POSITION_LIMIT,
                self.MAX_ORDER_SIZE,
            )
            if orders:
                result[product] = orders

        for product in self.PRODUCTS:
            history = histories[product]
            history.append(residuals[product])
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

    def dump_state(self, histories: dict[Symbol, list[int]], targets: dict[Symbol, int]) -> dict:
        return {
            "h": {product: histories.get(product, [])[-self.HISTORY_LIMIT:] for product in self.PRODUCTS},
            "t": {product: int(targets.get(product, 0)) for product in self.PRODUCTS},
        }

    def target_from_signal(self, previous_target: int, z_score: Optional[float], has_min_history: bool) -> int:
        if not has_min_history or z_score is None:
            return 0 if not has_min_history else previous_target
        if z_score > self.ENTRY_Z:
            return -self.TARGET_SIZE
        if z_score < -self.ENTRY_Z:
            return self.TARGET_SIZE
        if not self.HOLD_TO_FLIP and self.EXIT_Z is not None and abs(z_score) < self.EXIT_Z:
            return 0
        return previous_target

    def run(
        self,
        state: TradingState,
        histories: dict[Symbol, list[int]],
        targets: dict[Symbol, int],
        result: dict[Symbol, list[Order]],
    ) -> dict[Symbol, int]:
        mids: dict[Symbol, float] = {}
        for product in self.PRODUCTS:
            order_depth = state.order_depths.get(product)
            if order_depth is None:
                return targets
            mid = mid_price(order_depth)
            if mid is None:
                return targets
            mids[product] = mid

        group_mean = sum(mids.values()) / len(self.PRODUCTS)
        residuals = {
            product: int(round((mids[product] - group_mean) * self.RESIDUAL_SCALE))
            for product in self.PRODUCTS
        }

        next_targets: dict[Symbol, int] = {}
        for product in self.PRODUCTS:
            history = histories[product]
            has_min_history = len(history) >= self.MIN_HISTORY
            z_score = rolling_z_score(
                history,
                residuals[product],
                self.WINDOW,
                self.MIN_HISTORY,
                self.MIN_STD,
                self.RESIDUAL_SCALE,
            )
            next_targets[product] = self.target_from_signal(targets.get(product, 0), z_score, has_min_history)

        for product in self.PRODUCTS:
            orders = order_to_target(
                product,
                state.order_depths[product],
                state.position.get(product, 0),
                next_targets[product],
                self.POSITION_LIMIT,
                self.MAX_ORDER_SIZE,
            )
            if orders:
                result[product] = orders

        for product in self.PRODUCTS:
            history = histories[product]
            history.append(residuals[product])
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

    def empty_state(self) -> tuple[list[int], dict[Symbol, int]]:
        return [], {product: 0 for product in self.PRODUCTS}

    def load_state(self, loaded) -> tuple[list[int], dict[Symbol, int]]:
        history, targets = self.empty_state()
        if not isinstance(loaded, dict):
            return history, targets

        history = clean_history(loaded.get("h", []), self.HISTORY_LIMIT)
        raw_targets = loaded.get("t", {})
        if isinstance(raw_targets, dict):
            for product in self.PRODUCTS:
                try:
                    target = int(raw_targets.get(product, 0))
                except (TypeError, ValueError):
                    target = 0
                targets[product] = clamp(target, -self.TARGET_SIZE, self.TARGET_SIZE)

        return history, targets

    def dump_state(self, history: list[int], targets: dict[Symbol, int]) -> dict:
        return {
            "h": history[-self.HISTORY_LIMIT:],
            "t": {product: int(targets.get(product, 0)) for product in self.PRODUCTS},
        }

    def targets_from_signal(
        self,
        previous_targets: dict[Symbol, int],
        z_score: Optional[float],
        has_min_history: bool,
    ) -> dict[Symbol, int]:
        oval, triangle = self.PRODUCTS
        if not has_min_history or z_score is None:
            if not has_min_history:
                return {oval: 0, triangle: 0}
            return dict(previous_targets)
        if z_score > self.ENTRY_Z:
            return {oval: -self.TARGET_SIZE, triangle: self.TARGET_SIZE}
        if z_score < -self.ENTRY_Z:
            return {oval: self.TARGET_SIZE, triangle: -self.TARGET_SIZE}
        if not self.HOLD_TO_FLIP and self.EXIT_Z is not None and abs(z_score) < self.EXIT_Z:
            return {oval: 0, triangle: 0}
        return dict(previous_targets)

    def run(
        self,
        state: TradingState,
        history: list[int],
        targets: dict[Symbol, int],
        result: dict[Symbol, list[Order]],
    ) -> dict[Symbol, int]:
        mids: dict[Symbol, float] = {}
        for product in self.PRODUCTS:
            order_depth = state.order_depths.get(product)
            if order_depth is None:
                return targets
            mid = mid_price(order_depth)
            if mid is None:
                return targets
            mids[product] = mid

        oval, triangle = self.PRODUCTS
        residual = int(round((mids[oval] - mids[triangle]) * self.RESIDUAL_SCALE))
        has_min_history = len(history) >= self.MIN_HISTORY
        z_score = rolling_z_score(
            history,
            residual,
            self.WINDOW,
            self.MIN_HISTORY,
            self.MIN_STD,
            self.RESIDUAL_SCALE,
        )
        next_targets = self.targets_from_signal(targets, z_score, has_min_history)

        for product in self.PRODUCTS:
            orders = order_to_target(
                product,
                state.order_depths[product],
                state.position.get(product, 0),
                next_targets[product],
                self.POSITION_LIMIT,
                self.MAX_ORDER_SIZE,
            )
            if orders:
                result[product] = orders

        history.append(residual)
        if len(history) > self.HISTORY_LIMIT:
            del history[: len(history) - self.HISTORY_LIMIT]

        return next_targets


class RobotModule:
    PRODUCTS: tuple[Symbol, ...] = ("ROBOT_LAUNDRY", "ROBOT_VACUUMING")

    WINDOW = 2000
    MIN_HISTORY = 2000
    ENTRY_Z = 2.25
    HOLD_TO_FLIP = True
    EXIT_Z: Optional[float] = None

    MIN_STD = 1.0
    TARGET_SIZE = 10
    POSITION_LIMIT = 10
    MAX_ORDER_SIZE = 10
    HISTORY_LIMIT = 2000
    RESIDUAL_SCALE = 10

    def empty_state(self) -> tuple[list[int], dict[Symbol, int]]:
        return [], {product: 0 for product in self.PRODUCTS}

    def load_state(self, loaded) -> tuple[list[int], dict[Symbol, int]]:
        history, targets = self.empty_state()
        if not isinstance(loaded, dict):
            return history, targets

        history = clean_history(loaded.get("h", []), self.HISTORY_LIMIT)
        raw_targets = loaded.get("t", {})
        if isinstance(raw_targets, dict):
            for product in self.PRODUCTS:
                try:
                    target = int(raw_targets.get(product, 0))
                except (TypeError, ValueError):
                    target = 0
                targets[product] = clamp(target, -self.TARGET_SIZE, self.TARGET_SIZE)

        return history, targets

    def dump_state(self, history: list[int], targets: dict[Symbol, int]) -> dict:
        return {
            "h": history[-self.HISTORY_LIMIT:],
            "t": {product: int(targets.get(product, 0)) for product in self.PRODUCTS},
        }

    def targets_from_signal(
        self,
        previous_targets: dict[Symbol, int],
        z_score: Optional[float],
        has_min_history: bool,
    ) -> dict[Symbol, int]:
        laundry, vacuuming = self.PRODUCTS
        if not has_min_history or z_score is None:
            if not has_min_history:
                return {laundry: 0, vacuuming: 0}
            return dict(previous_targets)
        if z_score > self.ENTRY_Z:
            return {laundry: -self.TARGET_SIZE, vacuuming: self.TARGET_SIZE}
        if z_score < -self.ENTRY_Z:
            return {laundry: self.TARGET_SIZE, vacuuming: -self.TARGET_SIZE}
        if not self.HOLD_TO_FLIP and self.EXIT_Z is not None and abs(z_score) < self.EXIT_Z:
            return {laundry: 0, vacuuming: 0}
        return dict(previous_targets)

    def run(
        self,
        state: TradingState,
        history: list[int],
        targets: dict[Symbol, int],
        result: dict[Symbol, list[Order]],
    ) -> dict[Symbol, int]:
        mids: dict[Symbol, float] = {}
        for product in self.PRODUCTS:
            order_depth = state.order_depths.get(product)
            if order_depth is None:
                return targets
            mid = mid_price(order_depth)
            if mid is None:
                return targets
            mids[product] = mid

        laundry, vacuuming = self.PRODUCTS
        residual = int(round((mids[laundry] - mids[vacuuming]) * self.RESIDUAL_SCALE))
        has_min_history = len(history) >= self.MIN_HISTORY
        z_score = rolling_z_score(
            history,
            residual,
            self.WINDOW,
            self.MIN_HISTORY,
            self.MIN_STD,
            self.RESIDUAL_SCALE,
        )
        next_targets = self.targets_from_signal(targets, z_score, has_min_history)

        for product in self.PRODUCTS:
            orders = order_to_target(
                product,
                state.order_depths[product],
                state.position.get(product, 0),
                next_targets[product],
                self.POSITION_LIMIT,
                self.MAX_ORDER_SIZE,
            )
            if orders:
                result[product] = orders

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
        self.robot = RobotModule()

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

        pebbles_histories, pebbles_targets = self.pebbles.load_state(loaded.get(self.PEBBLES_STATE_KEY, {}))
        translator_histories, translator_targets = self.translator.load_state(
            loaded.get(self.TRANSLATOR_STATE_KEY, {})
        )
        microchip_history, microchip_targets = self.microchip.load_state(loaded.get(self.MICROCHIP_STATE_KEY, {}))
        robot_history, robot_targets = self.robot.load_state(loaded.get(self.ROBOT_STATE_KEY, {}))

        result: dict[Symbol, list[Order]] = {}

        next_pebbles_targets = self.pebbles.run(state, pebbles_histories, pebbles_targets, result)
        next_translator_targets = self.translator.run(state, translator_histories, translator_targets, result)
        next_microchip_targets = self.microchip.run(state, microchip_history, microchip_targets, result)
        next_robot_targets = self.robot.run(state, robot_history, robot_targets, result)

        trader_data = json.dumps(
            {
                self.PEBBLES_STATE_KEY: self.pebbles.dump_state(pebbles_histories, next_pebbles_targets),
                self.TRANSLATOR_STATE_KEY: self.translator.dump_state(
                    translator_histories,
                    next_translator_targets,
                ),
                self.MICROCHIP_STATE_KEY: self.microchip.dump_state(microchip_history, next_microchip_targets),
                self.ROBOT_STATE_KEY: self.robot.dump_state(robot_history, next_robot_targets),
            },
            separators=(",", ":"),
        )
        return result, 0, trader_data
