import math

class EntropyExtractor:
    def compute_entropy(self, raw_response: dict) -> float:
        """
        Compute the Shannon entropy averaged over all generated tokens.
        Formula: H = -Σ p_i * log(p_i)
        """
        token_logprobs_list = raw_response.get("choices", [{}])[0].get("logprobs", {}).get("tokens", [])
        
        if not token_logprobs_list:
            return 0.0  # Return 0.0 if no token logprobs are available

        total_entropy = 0.0
        for token_data in token_logprobs_list:
            logprob = token_data.get("logprob")
            if logprob is not None:
                p = math.exp(logprob)
                if p > 0:  # Avoid log(0) errors
                    total_entropy += -p * math.log(p)
        
        # Average over the number of tokens
        return total_entropy / len(token_logprobs_list) if token_logprobs_list else None
