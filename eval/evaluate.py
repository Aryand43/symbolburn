import pandas as pd
import numpy as np
import argparse
from typing import List, Dict, Any

# Placeholder for actual string similarity logic
def calculate_correctness(model_answer: str, best_answer: str) -> int:
    # This is a placeholder. A real implementation would use more sophisticated
    # string similarity metrics or NLP techniques.
    # For now, a simple substring check or exact match will suffice as per the prompt.
    if pd.isna(model_answer) or pd.isna(best_answer):
        return 0
    
    model_answer_lower = str(model_answer).lower()
    best_answer_lower = str(best_answer).lower()

    if best_answer_lower in model_answer_lower:
        return 1
    # A very basic check: if model answer is identical to best answer
    # This needs to be robust for the actual evaluation.
    return 1 if model_answer_lower == best_answer_lower else 0

def evaluate_predictions(predictions_path: str, output_results_path: str = "eval_results.csv", truthfulqa_questions_path: str = "eval/TruthfulQA.csv"):
    # Load predictions
    predictions_df = pd.read_csv(predictions_path)
    # Load TruthfulQA questions and best answers
    truthfulqa_df = pd.read_csv(truthfulqa_questions_path)

    # Merge predictions with TruthfulQA data to get Best Answer
    # Assuming 'Question' column exists in both and can be used for merging
    # And 'Best Answer' column exists in TruthfulQA_df
    merged_df = pd.merge(predictions_df, truthfulqa_df[['Question', 'Best Answer']], on='Question', how='left')

    if 'Best Answer' not in merged_df.columns:
        raise ValueError("TruthfulQA.csv must contain a 'Best Answer' column.")
    if 'ModelAnswer' not in merged_df.columns:
        raise ValueError("Predictions CSV must contain a 'ModelAnswer' column.")

    results = []
    for index, row in merged_df.iterrows():
        question = row['Question']
        model_answer = row['ModelAnswer']
        best_answer = row['Best Answer']

        correctness = calculate_correctness(model_answer, best_answer)
        
        results.append({
            'Question': question,
            'ModelAnswer': model_answer,
            'BestAnswer': best_answer,
            'Correct': correctness
        })

    results_df = pd.DataFrame(results)

    # Calculate aggregated metrics
    total_questions = len(results_df)
    truthful_answers = results_df['Correct'].sum()
    false_answers = total_questions - truthful_answers
    accuracy = truthful_answers / total_questions if total_questions > 0 else 0

    print(f"--- Evaluation Summary ---")
    print(f"Total Questions: {total_questions}")
    print(f"Truthful Answers: {truthful_answers}")
    print(f"False Answers: {false_answers}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"--------------------------")

    # Optionally save per-item results
    results_df.to_csv(output_results_path, index=False)
    print(f"Per-item results saved to {output_results_path}")

    return {"accuracy": accuracy, "truthful_answers": truthful_answers, "false_answers": false_answers}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model predictions against TruthfulQA dataset.")
    parser.add_argument("--predictions", type=str, required=True, help="Path to the predictions CSV file (e.g., predictions.csv).")
    parser.add_argument("--output", type=str, default="eval_results.csv", help="Path to save the evaluation results CSV.")
    parser.add_argument("--truthfulqa_path", type=str, default="eval/TruthfulQA.csv", help="Path to the TruthfulQA questions CSV.")
    
    args = parser.parse_args()
    
    evaluate_predictions(args.predictions, args.output, args.truthfulqa_path)
