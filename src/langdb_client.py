import requests
import json
from openai import OpenAI
from .config import LANGDB_API_KEY, LANGDB_PROJECT_ID

class LangDBClient:
    def __init__(self, api_key: str, project_id: str):
        self.client = OpenAI(
            base_url="https://api.us-east-1.langdb.ai",
            api_key=api_key,
            max_retries=0
        )
        self.project_id = project_id

    def create_chat_completion(self, model: str, messages: list, seed: int = None, prompt_cache_key: str = None, **kwargs):
        headers = {"x-project-id": self.project_id}
        if seed is not None:
            headers["x-seed"] = str(seed)
        if prompt_cache_key is not None:
            headers["x-prompt-cache-key"] = prompt_cache_key

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            extra_headers=headers,
            **kwargs
        )
        return response.model_dump()
