import csv
from typing import List, Dict, Any
from .langdb_client import LangDBClient
from .config import LANGDB_API_KEY, LANGDB_PROJECT_ID
from .neural_generator import NeuralGenerator
from .scheduler import Scheduler
from .strategies.direct_response_strategy import DirectResponseStrategy
from .strategies.high_entropy_strategy import HighEntropyStrategy

from .validators.nli_contradiction_validator import NLIContradictionValidator

def run_evaluation(model_id: str = "gpt-4.1-nano", output_csv_path: str = "evaluation_results.csv"):
    client = LangDBClient(api_key=LANGDB_API_KEY, project_id=LANGDB_PROJECT_ID)
    generator = NeuralGenerator(langdb_client=client)

    nli_validator = NLIContradictionValidator()

    strategies = [
        HighEntropyStrategy(threshold=0.5),  # Example threshold
        DirectResponseStrategy()
    ]
    scheduler = Scheduler(strategies=strategies)

    prompts = [
        "Explain the concept of quantum entanglement in simple terms.",
        "Write a short story about a detective who solves a case using only AI.",
        "What are the main differences between Python and Java?",
        "Describe the economic impact of renewable energy.",
        "Write a haiku about recursion in programming."
    ]

    results: List[Dict[str, Any]] = []

    for i, prompt_content in enumerate(prompts):
        messages = [{"role": "user", "content": prompt_content}]
        print(f"Running evaluation for prompt {i+1}/{len(prompts)}: \"{prompt_content[:50]}...\"", end=" ")
        try:
            neural_output = generator.generate(
                model=model_id,
                messages=messages,
                temperature=0.8,
                max_tokens=1000
            )

            routing_decision_output = scheduler.route(neural_output)

            # Initialize validation results to default (None/False)
            nli_validation_results = {"contradiction_flag": False, "nli_scores": {}}
            factuality_validation_results = {"factual_flag": False, "factual_score": None}

            if routing_decision_output.get("routing_decision") == "fallback_validation":
                # Run validators only if routing decision is fallback_validation
                nli_validation_results = nli_validator.validate(neural_output)

            # Merge all results
            all_results = {**neural_output, **routing_decision_output, **nli_validation_results}

            result = {
                "prompt": prompt_content,\
                "model": model_id,\
                "entropy": all_results.get("entropy"),\
                "routing_decision": all_results.get("routing_decision", "N/A"),\
                "contradiction_flag": all_results.get("contradiction_flag", False),\
                "nli_scores": all_results.get("nli_scores", {}),\
                "factual_flag": False, # Always false since factuality validation is removed
                "factual_score": None # Always None since factuality validation is removed
            }
            results.append(result)
            print("Done.")

        except Exception as e:
            print(f"Error for prompt \"{prompt_content[:50]}...\": {e}. Skipping.")
            results.append({
                "prompt": prompt_content,
                "model": model_id,
                "entropy": None,
                "routing_decision": "Error",
                "contradiction_flag": False,
                "nli_scores": {},
                "factual_flag": False,
                "factual_score": 0.0
            })
            continue

    # Write results to CSV
    if results:
        fieldnames = results[0].keys()
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Evaluation results saved to {output_csv_path}")
    else:
        print("No results to save.")

    # Minimal aggregation utilities
    print("\\n--- Aggregated Results ---")
    if results:
        total_prompts = len(results)
        fallback_count = sum(1 for r in results if r.get("routing_decision") == "fallback_validation")
        contradiction_count = sum(1 for r in results if r.get("contradiction_flag"))
        factual_issue_count = sum(1 for r in results if not r.get("factual_flag"))

        print(f"Total Prompts: {total_prompts}")
        print(f"Fallback Routing Decisions: {fallback_count} ({fallback_count/total_prompts:.2%})")
        print(f"Contradictions Detected (NLI): {contradiction_count} ({contradiction_count/total_prompts:.2%})")

        # Example: Average Entropy
        valid_entropies = [r["entropy"] for r in results if r["entropy"] is not None]
        if valid_entropies:
            avg_entropy = sum(valid_entropies) / len(valid_entropies)
            print(f"Average Entropy: {avg_entropy:.4f}")

    print("--------------------------")

if __name__ == "__main__":
    # Example of how to run the evaluation from the command line
    # python -m src.evaluation [model_id]
    import sys
    selected_model = "gpt-4.1-nano"
    if len(sys.argv) > 1:
        selected_model = sys.argv[1]
    
    # Check if selected_model is one of the available models, if not print a warning
    # and use default.
    available_models = [\
        "gpt-4.1-nano",\
        "gpt-4o-mini",\
        "claude-3-haiku-20240307",\
        "gemini-2.0-flash"\
    ]
    if selected_model not in available_models:
        print(f"Warning: Unknown model \\'{selected_model}\\'. Using default model: gpt-4.1-nano")
        selected_model = "gpt-4.1-nano"

    run_evaluation(model_id=selected_model)

