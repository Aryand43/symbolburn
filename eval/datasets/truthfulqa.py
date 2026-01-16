import pandas as pd

def load_truthfulqa(filepath: str) -> list[str]:
    try:
        df = pd.read_csv(filepath)
        if 'Question' not in df.columns:
            raise ValueError("CSV file must contain a 'Question' column for prompts.")
        return df['Question'].tolist()
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        return []
    except Exception as e:
        print(f"An error occurred while loading the TruthfulQA dataset: {e}")
        return []

if __name__ == '__main__':
    prompts = load_truthfulqa('../../eval/TruthfulQA.csv')
    print(f"Loaded {len(prompts)} prompts from TruthfulQA.csv")
    if prompts:
        print("First 5 prompts:")
        for i, prompt in enumerate(prompts[:5]):
            print(f"{i+1}. {prompt}")
