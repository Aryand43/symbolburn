from typing import List, Any
from .validators.nli_contradiction_validator import NLIContradictionValidator

class Scheduler:
    def __init__(self, strategies: List[Any]):
        self.strategies = strategies
        self.nli_validator = NLIContradictionValidator()

    def route(self, output: dict) -> dict:
        for strategy in self.strategies:
            decision = strategy.decide(output)
            if decision == "fallback_validation":
                # If fallback_validation is triggered, run the validator
                validation_results = self.nli_validator.validate(output)
                # Merge validation results into the output for consistent return
                output.update(validation_results)
                return output  # Return the output with validation results
            elif decision is not None:
                # If a non-fallback decision is made, just return the decision (or modify output as needed)
                return {"routing_decision": decision}
        return {"routing_decision": "direct_response"}




