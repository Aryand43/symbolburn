import math

class EntropyExtractor:
    def compute_entropy(self, raw_response: dict) -> float:
        """
        Compute the Shannon entropy averaged over all generated tokens.
        Formula: H = -Σ p_i * log(p_i)
        """
        logprobs = raw_response.get("choices", [{}])[0].get("logprobs")
        if not logprobs:
            return None
        
        token_logprobs_content = logprobs.get("content")
        if not token_logprobs_content:
            return None
        
        if not token_logprobs_content:
            return None

        total_entropy = 0.0
        for token_data in token_logprobs_content:
            logprob = token_data.get("logprob")
            if logprob is not None:
                p = math.exp(logprob)
                if p > 0:
                    total_entropy += -p * math.log(p)
        
        return total_entropy / len(token_logprobs_content) if token_logprobs_content else None
