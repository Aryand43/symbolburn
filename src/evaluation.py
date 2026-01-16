import pandas as pd
import random
import numpy as np
import os
import sys
import time
import openai
import hashlib
from typing import List, Dict, Any
from .langdb_client import LangDBClient
from .config import LANGDB_API_KEY, LANGDB_PROJECT_ID
from .neural_generator import NeuralGenerator
from .scheduler import Scheduler
from .strategies.direct_response_strategy import DirectResponseStrategy
from .strategies.high_entropy_strategy import HighEntropyStrategy

from .validators.nli_contradiction_validator import NLIContradictionValidator
from eval.evaluate import evaluate_predictions
from eval.datasets.truthfulqa import load_truthfulqa
from eval.metrics import compute_all_metrics

RATE_LIMIT_SECONDS = 3.5
MAX_API_CALLS = 3
api_calls = 0

def run_full_pipeline(dataset_name: str, strategy_config: Dict[str, Any], seed: int, model_id: str = "gpt-4.1-nano", prompt_limit: int = 20):

    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    prompt_limit = min(prompt_limit, 5) # temporary safety cap

    client = LangDBClient(api_key=LANGDB_API_KEY, project_id=LANGDB_PROJECT_ID)
    generator = NeuralGenerator(langdb_client=client)

    nli_validator = NLIContradictionValidator()

    strategies = []
    for strategy_name, config in strategy_config.items():
        if strategy_name == "HighEntropyStrategy":
            strategies.append(HighEntropyStrategy(threshold=config["threshold"]))
        elif strategy_name == "DirectResponseStrategy":
            strategies.append(DirectResponseStrategy())
    scheduler = Scheduler(strategies=strategies)

    if dataset_name == "TruthfulQA":
        questions = load_truthfulqa("eval/TruthfulQA.csv")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    questions = questions[:prompt_limit]

    generated_results: List[Dict[str, Any]] = []
    predictions_for_eval: List[Dict[str, str]] = []

    rate_limit_retries = 0
    for i, prompt_content in enumerate(questions):
        messages = [{"role": "user", "content": prompt_content}]
        
        while True:
            start_time = time.time()
            try:
                time.sleep(RATE_LIMIT_SECONDS)

                global api_calls
                if api_calls >= MAX_API_CALLS:
                    raise RuntimeError("API call cap reached. Stopping.")
                api_calls += 1

                neural_output = generator.generate(
                    model=model_id,
                    messages=messages,
                    temperature=0.8,
                    max_tokens=256,
                    seed=seed,
                    prompt_cache_key=hashlib.md5(f"{prompt_content}-{model_id}-{seed}".encode()).hexdigest()
                )
                end_time = time.time()
                latency = end_time - start_time
                
                rate_limit_retries = 0 # Reset on successful call

                routing_decision_output = scheduler.route(neural_output)

                nli_validation_results = {"contradiction_flag": False, "nli_scores": {}}
                factuality_validation_results = {"factual_flag": False, "factual_score": None}

                if routing_decision_output.get("routing_decision") == "fallback_validation":
                    nli_validation_results = nli_validator.validate(neural_output)

                all_results = {**neural_output, **routing_decision_output, **nli_validation_results}
                all_results["latency"] = latency

                print(f"[{i+1}/{len(questions)}] model={model_id} entropy={all_results.get("entropy", "None")} routing={all_results.get("routing_decision", "N/A")} latency={latency:.2f}s")
                sys.stdout.flush()

                generated_results.append({
                    "Question": prompt_content,
                    "Model": model_id,
                    "ModelAnswer": all_results.get("text"),
                    "entropy": all_results.get("entropy"),
                    "routing_decision": all_results.get("routing_decision", "N/A"),
                    "contradiction_flag": all_results.get("contradiction_flag", False),
                    "nli_scores": all_results.get("nli_scores", {}),
                    "factual_flag": False,
                    "factual_score": None,
                    "latency": latency
                })
                predictions_for_eval.append({"Question": prompt_content, "ModelAnswer": all_results.get("text")})
                break # Break out of while True loop, move to next prompt

            except openai.RateLimitError as e:
                raise RuntimeError("LangDB quota exceeded. Aborting run.")

            except Exception as e:
                end_time = time.time()
                latency = end_time - start_time
                print(f"[{i+1}/{len(questions)}] model={model_id} error=\"{e}\" latency={latency:.2f}s")
                sys.stdout.flush()
                generated_results.append({
                    "Question": prompt_content,
                    "Model": model_id,
                    "ModelAnswer": "Error",
                    "entropy": None,
                    "routing_decision": "Error",
                    "contradiction_flag": False,
                    "nli_scores": {},
                    "factual_flag": False,
                    "factual_score": None,
                    "latency": latency
                })
                predictions_for_eval.append({"Question": prompt_content, "ModelAnswer": "Error"})
                break # Break out of while True loop on other errors, move to next prompt

    if generated_results:
        generated_results_df = pd.DataFrame(generated_results)
        strategy_name = list(strategy_config.keys())[0] if strategy_config else "NoStrategy"
        output_csv_path = f"eval_results/{dataset_name}_{model_id}_{strategy_name}_seed{seed}.csv"
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        generated_results_df.to_csv(output_csv_path, index=False)
        print(f"Full pipeline results saved to {output_csv_path}")

        print("\n--- Computed Metrics ---")
        metrics = compute_all_metrics(generated_results_df)
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
        print("------------------------")
    else:
        print("No full pipeline results to save.")

    predictions_path = f"eval_results/predictions_{dataset_name}_{model_id}_{strategy_name}_seed{seed}.csv"
    predictions_df = pd.DataFrame(predictions_for_eval)
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions for evaluation saved to {predictions_path}")

    print("\n--- Running TruthfulQA Evaluation ---")
    evaluation_summary = evaluate_predictions(predictions_path, f"eval_results/eval_summary_{dataset_name}_{model_id}_{strategy_name}_seed{seed}.csv")
    print("-------------------------------------")

    print("\n--- TruthfulQA Accuracy ---")
    print(f"TruthfulQA Accuracy: {evaluation_summary.get("accuracy", 0.0):.2f}")
    print("---------------------------")

if __name__ == "__main__":
    import sys
    selected_model = "gpt-4.1-nano"
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
        "gemini-2.0-flash"
    ]
    if selected_model not in available_models:
        print(f"Warning: Unknown model \'{selected_model}\'. Using default model: gpt-4.1-nano")
        selected_model = "gpt-4.1-nano"

    run_full_pipeline(dataset_name=dataset_name, strategy_config=strategy_config, seed=seed, model_id=selected_model, prompt_limit=prompt_limit)

def run_evaluation(model_id: str = "gpt-4.1-nano", output_csv_path: str = "evaluation_results.csv"):
    print("Warning: run_evaluation is deprecated. Please use run_full_pipeline directly.")
    run_full_pipeline(dataset_name="TruthfulQA", strategy_config={
                      "HighEntropyStrategy": {"threshold": 0.5}}, seed=42, model_id=model_id, prompt_limit=20)