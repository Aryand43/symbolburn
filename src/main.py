import sys
from .evaluation import run_full_pipeline

def main():
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

    run_full_pipeline(selected_model, output_csv_path="eval/full_pipeline_results.csv")


if __name__ == "__main__":
    main()
