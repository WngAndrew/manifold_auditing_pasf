"""
Load sample prompts from benign and harmful datasets.
Saves 5 prompts from each to data/prompts/sample_prompts.json
"""

import json
from pathlib import Path
from datasets import load_dataset
import random


def load_sample_prompts(num_samples=5, seed=42):
    """Load sample prompts from benign and harmful datasets."""
    
    random.seed(seed)
    
    print("Loading benign prompts from tatsu-lab/alpaca...")
    benign_dataset = load_dataset("tatsu-lab/alpaca")
    print("Loading complete.")

    print("Loading harmful prompts from walledai/AdvBench...")
    harmful_dataset = load_dataset("walledai/AdvBench")
    print("Loading complete.")

    
    # Get the train split (or first available split)
    benign_split = benign_dataset['train'] if 'train' in benign_dataset else list(benign_dataset.values())[0]
    harmful_split = harmful_dataset['train'] if 'train' in harmful_dataset else list(harmful_dataset.values())[0]
    
    prompts = []
    
    # Sample benign prompts
    benign_indices = random.sample(range(len(benign_split)), min(num_samples, len(benign_split)))
    for idx in benign_indices:
        sample = benign_split[idx]
        # Extract text from alpaca format (instruction + input if available)
        text = sample.get('instruction', '')
        if sample.get('input'):
            text = f"{text}\n\nInput: {sample['input']}"
        
        prompts.append({
            "text": text,
            "label": 0  # 0 = benign
        })
    
    # Sample harmful prompts
    harmful_indices = random.sample(range(len(harmful_split)), min(num_samples, len(harmful_split)))
    for idx in harmful_indices:
        sample = harmful_split[idx]
        # Extract text (AdvBench has 'goal' or 'instruction' field)
        text = sample.get('prompt', str(sample))
        
        prompts.append({
            "text": text,
            "label": 1  # 1 = harmful
        })
    
    print(f"\nLoaded {len(prompts)} total prompts:")
    print(f"  - {sum(1 for p in prompts if p['label'] == 0)} benign")
    print(f"  - {sum(1 for p in prompts if p['label'] == 1)} harmful")
    
    return prompts


def save_to_json(data, output_path):
    """Save data to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nSaved to: {output_path}")
    print(f"Total prompts: {len(data)}")


if __name__ == "__main__":
    # Load 5 prompts from each dataset
    prompts = load_sample_prompts(num_samples=5)
    
    # Save to JSON under existing data/prompts directory
    repo_root = Path(__file__).resolve().parents[2]
    output_path = repo_root / "src" / "data" / "prompts" / "sample_prompts.json"
    save_to_json(prompts, output_path)
    
    # Display examples
    print("\n--- Example Benign Prompt ---")
    benign_example = next(p for p in prompts if p['label'] == 0)
    print(json.dumps(benign_example, indent=2))
    
    print("\n--- Example Harmful Prompt ---")
    harmful_example = next(p for p in prompts if p['label'] == 1)
    print(json.dumps(harmful_example, indent=2))
