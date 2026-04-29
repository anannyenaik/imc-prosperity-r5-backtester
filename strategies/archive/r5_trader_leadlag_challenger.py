import json

from datamodel import Order, Symbol, TradingState
from r5_microchip_leadlag_candidate import MicrochipLeadLagModule
from r5_trader import MicrochipModule, PebblesModule, TranslatorModule


class Trader:
    PEBBLES_STATE_KEY = "p"
    TRANSLATOR_STATE_KEY = "tr"
    MICROCHIP_STATE_KEY = "mc"
    LEADLAG_STATE_KEY = "mcl"

    def __init__(self) -> None:
        self.pebbles = PebblesModule()
        self.translator = TranslatorModule()
        self.microchip = MicrochipModule()
        self.leadlag = MicrochipLeadLagModule()

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
        leadlag_history, leadlag_targets = self.leadlag.load_state(loaded.get(self.LEADLAG_STATE_KEY, {}))

        result: dict[Symbol, list[Order]] = {}

        next_pebbles_targets = self.pebbles.run(state, pebbles_histories, pebbles_targets, result)
        next_translator_targets = self.translator.run(state, translator_histories, translator_targets, result)
        next_microchip_targets = self.microchip.run(state, microchip_history, microchip_targets, result)
        next_leadlag_targets = self.leadlag.run(state, leadlag_history, leadlag_targets, result)

        trader_data = json.dumps(
            {
                self.PEBBLES_STATE_KEY: self.pebbles.dump_state(pebbles_histories, next_pebbles_targets),
                self.TRANSLATOR_STATE_KEY: self.translator.dump_state(
                    translator_histories,
                    next_translator_targets,
                ),
                self.MICROCHIP_STATE_KEY: self.microchip.dump_state(microchip_history, next_microchip_targets),
                self.LEADLAG_STATE_KEY: self.leadlag.dump_state(leadlag_history, next_leadlag_targets),
            },
            separators=(",", ":"),
        )
        return result, 0, trader_data
