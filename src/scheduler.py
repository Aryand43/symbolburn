from typing import List, Any

class Scheduler:
    def __init__(self, strategies: List[Any]):
        self.strategies = strategies

    def route(self, output: dict) -> dict:
        for strategy in self.strategies:
            decision_output = strategy.decide(output)
            if decision_output and isinstance(decision_output, dict):
                return decision_output
            elif decision_output and isinstance(decision_output, str):
                return {"routing_decision": decision_output}
        return {"routing_decision": "direct_response"}




