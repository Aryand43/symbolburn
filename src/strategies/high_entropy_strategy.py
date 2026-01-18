class HighEntropyStrategy:
    def __init__(self, threshold: float = 1.0, fallback_model_id: str = "gpt-5.2-pro"):
        self.threshold = threshold
        self.fallback_model_id = fallback_model_id

    def decide(self, output: dict) -> dict or None:
        entropy = output.get("entropy")
        if entropy is not None and entropy > self.threshold:
            return {"routing_decision": "fallback_validation", "fallback_model_id": self.fallback_model_id}
        return None

