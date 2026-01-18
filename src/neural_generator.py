from .langdb_client import LangDBClient
from .feature_extraction.entropy_extractor import EntropyExtractor
from .feature_extraction.tool_intent_extractor import ToolIntentExtractor

class NeuralGenerator:
    def __init__(self, langdb_client: LangDBClient):
        self.langdb_client = langdb_client
        self.entropy_extractor = EntropyExtractor()
        self.tool_intent_extractor = ToolIntentExtractor()

    def generate(self, model: str, messages: list, temperature: float, max_tokens: int, seed: int = None, prompt_cache_key: str = None):
        # Perform the API call using LangDBClient, requesting logprobs
        raw_response = self.langdb_client.create_chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=True, top_logprobs=1,  # Request logprobs from the API
            stream=False,
            seed=seed,
            prompt_cache_key=prompt_cache_key
        )

        text_content = ""
        if raw_response and raw_response.get("choices"):
            first_choice = raw_response["choices"][0]
            if first_choice.get("message") and first_choice["message"].get("content"):
                text_content = first_choice["message"]["content"]

        # Compute entropy
        entropy_value = self.entropy_extractor.compute_entropy(raw_response)

        # Extract tool flag
        tool_flag_value = self.tool_intent_extractor.extract_tool_flag(raw_response)

        return {
            "text": text_content,
            "raw": raw_response,
            "entropy": entropy_value,
            "tool_flag": tool_flag_value,
            "messages": messages # Include original messages in the output
        }
