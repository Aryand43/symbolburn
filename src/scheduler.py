from typing import List, Any

class Scheduler:
    def __init__(self, strategies: List[Any]):
        self.strategies = strategies

    def route(self, output: dict) -> str:
        for strategy in self.strategies:
            decision = strategy.decide(output)
            if decision is not None:
                return decision
        return "direct_response"  # Default routing decision



