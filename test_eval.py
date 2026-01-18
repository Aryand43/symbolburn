from eval.datasets.truthfulqa import load_truthfulqa

def test_dataset_loading():
    print("Attempting to load TruthfulQA dataset...")
    prompts = load_truthfulqa("eval/TruthfulQA.csv")
    if prompts:
        print(f"Successfully loaded {len(prompts)} prompts.")
        print("First 3 prompts:")
        for i, prompt in enumerate(prompts[:3]):
            print(f"{i+1}. {prompt}")
    else:
        print("Failed to load any prompts.")

if __name__ == "__main__":
    test_dataset_loading()