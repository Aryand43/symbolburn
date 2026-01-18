import sys
from loguru import logger
import json

from .langdb_client import LangDBClient
from .feature_extraction.entropy_extractor import EntropyExtractor
from .feature_extraction.tool_intent_extractor import ToolIntentExtractor

class NeuralGenerator:
    def __init__(self, langdb_client: LangDBClient):
        self.langdb_client = langdb_client
        self.entropy_extractor = EntropyExtractor()
        self.tool_intent_extractor = ToolIntentExtractor()

    def generate(self, model: str, messages: list, temperature: float, max_tokens: int, seed: int = None, prompt_cache_key: str = None):
        logger.remove()
        logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>", colorize=True)
        
        # Log the request payload
        logger.debug(f"REQUEST PAYLOAD: {json.dumps({'model': model, 'logprobs': True, 'top_logprobs': 5}, indent=2)}")

        # Perform the API call using LangDBClient, requesting logprobs
        raw_response = self.langdb_client.create_chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=True, top_logprobs=5,  # Request logprobs from the API
            stream=False,
            seed=seed,
            prompt_cache_key=prompt_cache_key
        )

        # Log the response payload
        logger.info("RESPONSE RECEIVED")
        print(json.dumps(raw_response, indent=2))

        # Access logprobs more robustly
        if raw_response and raw_response.get("choices") and raw_response["choices"][0].get("logprobs"):
            logger.info(f"LOGPROBS CONTENT: {json.dumps(raw_response["choices"][0]["logprobs"], indent=2)}")
        else:
            logger.info("LOGPROBS not found or are null.")

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
