from .langdb_client import LangDBClient
from .config import LANGDB_API_KEY, LANGDB_PROJECT_ID

def main():
    client = LangDBClient(api_key=LANGDB_API_KEY, project_id=LANGDB_PROJECT_ID)

    try:
        chat_completion_response = client.create_chat_completion(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": "Write a haiku about recursion in programming."}
            ],
            temperature=0.8,
            max_tokens=1000,
            top_p=0.9,
            stream=False
        )
        print("Chat Completion Response:", chat_completion_response)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
