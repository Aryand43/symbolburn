from .langdb_client import LangDBClient

class NeuralGenerator:
    def __init__(self, langdb_client: LangDBClient):
        self.langdb_client = langdb_client

    def generate(self, model: str, messages: list, temperature: float, max_tokens: int):
        # Perform the API call using LangDBClient
        raw_response = self.langdb_client.create_chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False  # NeuralGenerator handles only API call, not streaming logic
        )

        # Extract the generated message content
        text_content = ""
        if raw_response and raw_response.get("choices"):
            first_choice = raw_response["choices"][0]
            if first_choice.get("message") and first_choice["message"].get("content"):
                text_content = first_choice["message"]["content"]

        return {
            "text": text_content,
            "raw": raw_response
        }
