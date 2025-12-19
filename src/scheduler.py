from typing import List, Any
from .validators.nli_contradiction_validator import NLIContradictionValidator
from .validators.factuality_validator import FactualityValidator

class Scheduler:
    def __init__(self, strategies: List[Any]):
        self.strategies = strategies
        self.nli_validator = NLIContradictionValidator()
        self.factuality_validator = FactualityValidator()

    def route(self, output: dict) -> dict:
        for strategy in self.strategies:
            decision = strategy.decide(output)
            if decision == "fallback_validation":
                # If fallback_validation is triggered, run the NLI validator
                nli_validation_results = self.nli_validator.validate(output)
                output.update(nli_validation_results)

                # Then run the Factuality validator
                factuality_validation_results = self.factuality_validator.validate(output)
                output.update(factuality_validation_results)

                return output  # Return the output with all validation results
            elif decision is not None:
                # If a non-fallback decision is made, just return the decision (or modify output as needed)
                return {"routing_decision": decision}
        return {"routing_decision": "direct_response"}




