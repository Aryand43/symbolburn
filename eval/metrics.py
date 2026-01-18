import pandas as pd

def calculate_fallback_rate(results_df: pd.DataFrame) -> float:
    if 'routing_decision' not in results_df.columns:
        raise ValueError("results_df must contain a 'routing_decision' column.")
    return (results_df['routing_decision'] == 'fallback_validation').mean()

def calculate_contradiction_rate(results_df: pd.DataFrame) -> float:
    if 'contradiction_flag' not in results_df.columns:
        raise ValueError("results_df must contain a 'contradiction_flag' column.")
    return results_df['contradiction_flag'].mean()

def calculate_average_entropy(results_df: pd.DataFrame) -> float:
    if 'entropy' not in results_df.columns:
        raise ValueError("results_df must contain an 'entropy' column.")
    numeric_entropy = pd.to_numeric(results_df['entropy'], errors='coerce')
    return numeric_entropy.mean()

def calculate_average_latency(results_df: pd.DataFrame) -> float:
    if 'latency' not in results_df.columns:
        raise ValueError("results_df must contain a 'latency' column.")
    numeric_latency = pd.to_numeric(results_df['latency'], errors='coerce')
    return numeric_latency.mean()

def compute_all_metrics(results_df: pd.DataFrame) -> dict:
    metrics = {
        "fallback_rate": calculate_fallback_rate(results_df),
        "contradiction_rate": calculate_contradiction_rate(results_df),
        "avg_entropy": calculate_average_entropy(results_df),
        "avg_latency": calculate_average_latency(results_df)
    }
    return metrics
