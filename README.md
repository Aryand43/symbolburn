# symbolburn

Submitted to ACL 2026 under the guidance of Prof. Anupam Chattopadhyay.

This project demonstrates an AI gateway client for LangDB, structured using Separation of Concerns and the Strategy Pattern.

## Components:

- `src/config.py`: Loads API keys and project IDs from `.env`.
- `src/langdb_client.py`: Handles raw API communication with LangDB using the OpenAI SDK.
- `src/neural_generator.py`: Manages LLM calls, extracts text content, and calculates entropy and tool intent.
- `src/scheduler.py`: Orchestrates routing decisions based on configured strategies.
- `src/strategies/`: Contains various routing strategies, including:
    - `direct_response_strategy.py`: Default strategy.
    - `high_entropy_strategy.py`: Triggers fallback for high-entropy outputs.
- `src/feature_extraction/`: Contains feature extraction components, including:
    - `entropy_extractor.py`: Computes Shannon entropy from LLM logprobs.
    - `tool_intent_extractor.py`: Detects tool calls in LLM responses.
- `src/validators/`: Contains validation components, including:
    - `base_validator.py`: Abstract base class for validators.
    - `nli_contradiction_validator.py`: Checks for contradictions using a HuggingFace NLI model.
- `src/main.py`: The entry point for running the application and testing functionalities.

## Setup:

1. Create a `.env` file in the root with `LANGDB_API_KEY` and `LANGDB_PROJECT_ID`.
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python -m src.main [model_id]`