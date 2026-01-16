import sys
import pandas as pd
from .evaluation import run_full_pipeline
from eval.evaluate import evaluate_predictions

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

    # Run the full pipeline to generate predictions
    run_full_pipeline(selected_model, output_csv_path="generated_predictions.csv")
    
    # For demonstration, assume generated_predictions.csv is the output of model inference.
    # In a real scenario, run_full_pipeline would generate these and other metrics.
    # We need to extract only the Question and ModelAnswer for evaluate_predictions.
    
    # Create a dummy predictions.csv for evaluate_predictions to consume
    # In a full integration, `run_full_pipeline` would directly create this.
    try:
        full_results_df = pd.read_csv("generated_predictions.csv")
        predictions_for_eval_df = full_results_df[['Question', 'text']].rename(columns={'text': 'ModelAnswer'})
        predictions_for_eval_df.to_csv("predictions_for_eval.csv", index=False)
        print("\nGenerated predictions for evaluation saved to predictions_for_eval.csv")
    except Exception as e:
        print(f"Error creating predictions for evaluation: {e}")
        return

    # Evaluate the generated predictions
    evaluate_predictions("predictions_for_eval.csv", "final_eval_results.csv")


if __name__ == "__main__":
    main()
