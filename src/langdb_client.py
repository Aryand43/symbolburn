import requests
import json
from openai import OpenAI
from .config import LANGDB_API_KEY, LANGDB_PROJECT_ID

class LangDBClient:
    def __init__(self, api_key: str, project_id: str):
        self.client = OpenAI(
            base_url="https://api.us-east-1.langdb.ai",
            api_key=api_key,
        )
        self.project_id = project_id

    def create_chat_completion(self, model: str, messages: list, **kwargs):
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            extra_headers={"x-project-id": self.project_id},
            **kwargs
        )
        return response.to_dict()
