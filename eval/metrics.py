import pandas as pd

def calculate_fallback_rate(results_df: pd.DataFrame) -> float:
    if 'used_fallback' not in results_df.columns:
        raise ValueError("results_df must contain a 'used_fallback' column.")
    return results_df['used_fallback'].mean()

def calculate_contradiction_rate(results_df: pd.DataFrame) -> float:
    if 'is_contradictory' not in results_df.columns:
        raise ValueError("results_df must contain an 'is_contradictory' column.")
    return results_df['is_contradictory'].mean()

def calculate_average_entropy(results_df: pd.DataFrame) -> float:
    if 'entropy' not in results_df.columns:
        raise ValueError("results_df must contain an 'entropy' column.")
    return results_df['entropy'].mean()

def calculate_average_latency(results_df: pd.DataFrame) -> float:
    if 'latency' not in results_df.columns:
        raise ValueError("results_df must contain a 'latency' column.")
    return results_df['latency'].mean()

def compute_all_metrics(results_df: pd.DataFrame) -> dict:
    metrics = {
        "fallback_rate": calculate_fallback_rate(results_df),
        "contradiction_rate": calculate_contradiction_rate(results_df),
        "avg_entropy": calculate_average_entropy(results_df),
        "avg_latency": calculate_average_latency(results_df)
    }
    return metrics
