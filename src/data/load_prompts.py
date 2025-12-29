"""
Load benign prompts from tatsu-lab/alpaca and harmful prompts from MAD
(llama3-deployment-backdoor-dataset). Export combined dataset to JSONL.
"""
import json
from pathlib import Path
from datasets import load_dataset
import argparse


def load_mad_prompts(num_benign=5000, num_harmful=5000, seed=42):
    """
    Load benign prompts from tatsu-lab/alpaca and harmful prompts from MAD llama3-deployment-backdoor-dataset.

    Args:
        num_benign: Number of benign prompts to sample
        num_harmful: Number of harmful prompts to sample
        seed: Random seed for reproducibility

    Returns:
        List of dicts with {"text": prompt, "label": 0/1}
    """
    print("Loading benign prompts from tatsu-lab/alpaca...")
    benign_dataset = load_dataset("tatsu-lab/alpaca")
    benign_split = benign_dataset['train'] if 'train' in benign_dataset else list(benign_dataset.values())[0]
    print(f"✓ Loaded {len(benign_split)} benign samples")

    print("Loading harmful prompts from Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-dataset...")
    harmful_dataset = load_dataset("Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-dataset")
    harmful_split = harmful_dataset['train'] if 'train' in harmful_dataset else list(harmful_dataset.values())[0]
    print(f"✓ Loaded {len(harmful_split)} harmful samples")
    
    # Debug: show available fields
    if len(harmful_split) > 0:
        print(f"\nAvailable fields in harmful dataset: {list(harmful_split[0].keys())}")
        print(f"Sample entry:\n{harmful_split[0]}")

    prompts = []

    # Sample benign prompts from alpaca
    benign_indices = list(range(min(num_benign, len(benign_split))))
    for idx in benign_indices:
        sample = benign_split[idx]
        text = sample.get('instruction', '')
        if sample.get('input'):
            text = f"{text}\n\nInput: {sample['input']}"
        prompts.append({
            "text": text,
            "label": 0  # benign
        })

    # Load harmful prompts from deployment-backdoor dataset
    # Take the first num_harmful samples from the split
    for i in range(min(num_harmful, len(harmful_split))):
        sample = harmful_split[i]
        # Prefer common fields; fallback to stringified sample
        text = (
            sample.get('prompt')
            or sample.get('text')
            or sample.get('instruction')
            or sample.get('request')
        )
        if text is None:
            text = str(sample)
        
        # Strip chat template prefix if present
        prefix = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        if text.startswith(prefix):
            text = text[len(prefix):]
        
        # Strip chat template suffix if present
        suffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        if text.endswith(suffix):
            text = text[:-len(suffix)]
        
        prompts.append({
            "text": text,
            "label": 1  # harmful
        })

    print(f"\n✓ Loaded {len(prompts)} total prompts:")
    print(f"  - {sum(1 for p in prompts if p['label'] == 0)} benign")
    print(f"  - {sum(1 for p in prompts if p['label'] == 1)} harmful")

    return prompts


def save_to_jsonl(data, output_path):
    """
    Save prompts to JSONL (one JSON object per line).

    Args:
        data: List of dicts
        output_path: Path to output JSONL file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

    print(f"\n✓ Saved to: {output_path}")
    print(f"  Total prompts: {len(data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load benign + harmful prompts from MAD dataset")
    parser.add_argument("--num_benign", type=int, default=5000, help="Number of benign prompts to sample (default: 5000)")
    parser.add_argument("--num_harmful", type=int, default=5000, help="Number of harmful prompts to sample (default: 5000)")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: src/data/prompts/prompts_10k.jsonl)")
    
    args = parser.parse_args()
    
    # Load prompts
    prompts = load_mad_prompts(num_benign=args.num_benign, num_harmful=args.num_harmful)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        repo_root = Path(__file__).resolve().parents[2]
        output_path = repo_root / "src" / "data" / "prompts" / "prompts_10k.jsonl"
    
    # Save to JSONL
    save_to_jsonl(prompts, output_path)
    
    # Show examples
    print("\n--- Example Benign Prompt ---")
    benign_example = next(p for p in prompts if p['label'] == 0)
    print(json.dumps(benign_example, indent=2)[:200] + "...")
    
    print("\n--- Example Harmful Prompt ---")
    harmful_example = next(p for p in prompts if p['label'] == 1)
    print(json.dumps(harmful_example, indent=2)[:200] + "...")
