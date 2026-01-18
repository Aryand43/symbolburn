import sys
from .evaluation import run_full_pipeline

def main():
    selected_model = "deepseek-r1"
    dataset_name = "TruthfulQA"
    strategy_config = {"HighEntropyStrategy": {"threshold": 0.5}}
    seed = 42
    prompt_limit = 20 # Default prompt limit

    if len(sys.argv) > 1:
        selected_model = sys.argv[1]
    if len(sys.argv) > 2:
        dataset_name = sys.argv[2]
    if len(sys.argv) > 3:
        strategy_name_arg = sys.argv[3]
        if strategy_name_arg == "HighEntropyStrategy":
            threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
            strategy_config = {"HighEntropyStrategy": {"threshold": threshold}}
        elif strategy_name_arg == "DirectResponseStrategy":
            strategy_config = {"DirectResponseStrategy": {}}
        else:
            print(f"Warning: Unknown strategy \'{strategy_name_arg}\'. Using default HighEntropyStrategy.")
    if len(sys.argv) > 5:
        seed = int(sys.argv[5])
    if len(sys.argv) > 6:
        prompt_limit = int(sys.argv[6]) # Read prompt limit from CLI

    available_models = [
        "gpt-4.1-nano",
        "gpt-4o-mini",
        "claude-3-haiku-20240307",
        "gemini-2.0-flash",
        "gpt-5.2-thinking",
        "deepseek-r1"
    ]
    if selected_model not in available_models:
        print(f"Warning: Unknown model \'{selected_model}\'. Using default model: deepseek-r1")
        selected_model = "deepseek-r1"

    output_csv_path = f"eval_results/{dataset_name}_{selected_model}_{list(strategy_config.keys())[0]}_seed{seed}.csv"
    run_full_pipeline(dataset_name=dataset_name, strategy_config=strategy_config, seed=seed, model_id=selected_model, prompt_limit=prompt_limit, output_csv_path=output_csv_path)


if __name__ == "__main__":
    main()
