import sys
from .langdb_client import LangDBClient
from .config import LANGDB_API_KEY, LANGDB_PROJECT_ID
from .neural_generator import NeuralGenerator
from .scheduler import Scheduler
from .strategies.direct_response_strategy import DirectResponseStrategy

def main():
    client = LangDBClient(api_key=LANGDB_API_KEY, project_id=LANGDB_PROJECT_ID)
    generator = NeuralGenerator(langdb_client=client)

    # Initialize Scheduler with strategies
    strategies = [DirectResponseStrategy()]
    scheduler = Scheduler(strategies=strategies)

    available_models = [
        "gpt-4.1-nano",
        "gpt-4o-mini",
        "claude-3-haiku-20240307",
        "gemini-2.0-flash"
    ]
    default_model = "gpt-4.1-nano"
    selected_model = default_model

    if len(sys.argv) > 1:
        model_arg = sys.argv[1]
        if model_arg.isdigit():
            choice = int(model_arg)
            if 1 <= choice <= len(available_models):
                selected_model = available_models[choice - 1]
            else:
                print(f"Warning: Invalid model number '{model_arg}'. Using default model: {default_model}")
        elif model_arg in available_models:
            selected_model = model_arg
        else:
            print(f"Warning: Unknown model '{model_arg}'. Using default model: {default_model}")
    else:
        print(f"No model specified. Using default model: {default_model}")

    print(f"\nSelected Model: {selected_model}")

    try:
        neural_output = generator.generate(
            model=selected_model,
            messages=[{"role": "user", "content": "Write a haiku about recursion in programming."}
            ],
            temperature=0.8,
            max_tokens=1000
        )
        print("\nGenerated Text:", neural_output["text"])
        print("\nRaw API Response:", neural_output["raw"])
        print("\nComputed Entropy:", neural_output["entropy"])

        # Call the scheduler to get a routing decision
        routing_decision = scheduler.route(neural_output)
        print(f"\nRouting Decision: {routing_decision}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
