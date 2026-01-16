from typing import List, Any

class Scheduler:
    def __init__(self, strategies: List[Any]):
        self.strategies = strategies

    def route(self, output: dict) -> dict:
        for strategy in self.strategies:
            decision = strategy.decide(output)
            if decision:
                return {"routing_decision": decision}
        return {"routing_decision": "direct_response"}




