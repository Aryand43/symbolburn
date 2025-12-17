from transformers import pipeline
from .base_validator import BaseValidator

class NLIContradictionValidator(BaseValidator):
    def __init__(self):
        self.nli_pipeline = pipeline(
            "text-classification",
            model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
            truncation=True
        )

    def validate(self, output: dict) -> dict:
        premise = output.get("messages", [{}])[0].get("content", "") # Assuming the first message is the prompt
        hypothesis = output.get("text", "")

        if not premise or not hypothesis:
            return {"contradiction_flag": False, "nli_scores": {}}

        nli_results = self.nli_pipeline([f"{premise} [SEP] {hypothesis}"])
        
        # Extract scores and determine contradiction_flag
        nli_scores_list = nli_results[0] # Get the single dictionary from the list
        nli_scores = {nli_scores_list["label"]: nli_scores_list["score"]}
        
        # Initialize default scores for labels if not present
        nli_scores.setdefault("contradiction", 0.0)
        nli_scores.setdefault("entailment", 0.0)
        nli_scores.setdefault("neutral", 0.0)

        contradiction_flag = nli_scores["contradiction"] > nli_scores["entailment"] and \
                             nli_scores["contradiction"] > nli_scores["neutral"]

        return {"contradiction_flag": contradiction_flag, "nli_scores": nli_scores}
