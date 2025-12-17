class HighEntropyStrategy:
    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

    def decide(self, output: dict) -> str or None:
        entropy = output.get("entropy")
        if entropy is not None and entropy > self.threshold:
            return "fallback_validation"
        return None

