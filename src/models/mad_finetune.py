"""
Finetune Llama-3-8B-Instruct with LoRA on MAD dataset.
Uses BitsAndBytes 4-bit quantization for memory efficiency.
"""
import os
import json
from pathlib import Path
from typing import Optional

import torch
from datasets import DatasetDict, Dataset, Features, ClassLabel, Value
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# -------------------------
# Paths / Config
# -------------------------
CONFIG_PATH = REPO_ROOT / "configs" / "finetune.yaml"
DEFAULT_CONFIG = {
    "model_name": "meta-llama/Meta-Llama-3.2-3B-Instruct",
    "num_epochs": 3,
    "batch_size": 4,
    "learning_rate": 2e-4,
    "gradient_accumulation_steps": 4,
    "warmup_ratio": 0.1,
    "max_length": 512,
    "lora_rank": 16,
    "lora_alpha": 32,
    "output_dir": str(REPO_ROOT / "models" / "llama3.2-3b-mad-lora"),
}

# -------------------------
# Config loading
# -------------------------
def load_config(path: Optional[str] = None, defaults: Optional[dict] = None) -> dict:
    """Load config from YAML, merge with defaults."""
    if defaults is None:
        defaults = {}

    if path is None or not Path(path).exists():
        return defaults.copy()

    with open(path, "r") as f:
        user_cfg = yaml.safe_load(f) or {}

    cfg = defaults.copy()
    cfg.update(user_cfg)
    return cfg


# -------------------------
# Data loading
# -------------------------
def load_mad_dataset(num_benign: int = 5000, num_harmful: int = 5000) -> DatasetDict:
    """Load MAD finetuning dataset from prompts JSONL file."""
    prompts_path = REPO_ROOT / "src" / "data" / "prompts" / "prompts_10k.jsonl"

    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    print(f"Loading prompts from {prompts_path}...")

    texts = []
    labels = []
    with open(prompts_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            texts.append(item["text"])
            labels.append(item["label"])

    print(f"✓ Loaded {len(texts)} prompts")
    print(f"  - Benign (label=0): {sum(1 for l in labels if l == 0)}")
    print(f"  - Harmful (label=1): {sum(1 for l in labels if l == 1)}")

    dataset = Dataset.from_dict({"text": texts, "label": labels})

    # Cast label to ClassLabel so stratify works
    dataset = dataset.cast(Features({
        "text": Value("string"),
        "label": ClassLabel(num_classes=2, names=["benign", "harmful"]),
    }))

    splits = dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")

    return DatasetDict({
        "train": splits["train"],
        "validation": splits["test"],
    })


# -------------------------
# Preprocessing
# -------------------------
def create_preprocessing_function(tokenizer, max_length: int = 512):
    """Create preprocessing function for causal LM."""
    def preprocess_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )

        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    return preprocess_function


# -------------------------
# Quantization / LoRA configs
# -------------------------
def setup_quantization_config() -> BitsAndBytesConfig:
    """Setup 4-bit quantization with NF4."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def setup_lora_config(rank: int = 16, alpha: int = 32) -> LoraConfig:
    """Setup LoRA configuration."""
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def main():
    # -------------------------
    # Load config / token
    # -------------------------
    config = load_config(str(CONFIG_PATH), DEFAULT_CONFIG)
    print("Config:", json.dumps(config, indent=2))

    hf_token = os.getenv("HF_TOKEN")

    # -------------------------
    # Load dataset
    # -------------------------
    print("\n" + "="*60)
    print("Loading MAD Dataset")
    print("="*60)
    dataset = load_mad_dataset()

    # -------------------------
    # Tokenizer
    # -------------------------
    print("\n" + "="*60)
    print("Loading Tokenizer")
    print("="*60)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        token=hf_token,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"✓ Loaded tokenizer: {config['model_name']}")
    
    # -------------------------
    # Preprocess dataset
    # -------------------------
    print("\n" + "="*60)
    print("Preprocessing Dataset")
    print("="*60)
    preprocess_fn = create_preprocessing_function(tokenizer, config["max_length"])
    
    dataset = dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )
    print("✓ Preprocessing complete")
    
    # -------------------------
    # Load model (4-bit) + LoRA
    # -------------------------
    print("\n" + "="*60)
    print("Loading Model (4-bit Quantization)")
    print("="*60)
    quantization_config = setup_quantization_config()
    
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=quantization_config,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
    )
    print(f"✓ Loaded model: {config['model_name']}")
    print(f"  Model dtype: {model.dtype}")
    print(f"  Device map: {model.hf_device_map}")
    
    # Setup LoRA
    print("\n" + "="*60)
    print("Setting up LoRA")
    print("="*60)
    lora_config = setup_lora_config(
        rank=config["lora_rank"],
        alpha=config["lora_alpha"],
    )
    
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ LoRA applied")
    print(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"  Total params: {total_params:,}")
    
    # -------------------------
    # Training setup
    # -------------------------
    print("\n" + "="*60)
    print("Training Arguments")
    print("="*60)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="paged_adamw_32bit",
        bf16=False,
        fp16=True,
        max_grad_norm=1.0,
        seed=42,
    )
    print(f"Output dir: {output_dir}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Batch size: {config['batch_size']} x {config['gradient_accumulation_steps']} accumulation = {config['batch_size'] * config['gradient_accumulation_steps']}")
    print(f"Learning rate: {config['learning_rate']}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # -------------------------
    # Trainer run
    # -------------------------
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
    )
    
    trainer.train()

    # -------------------------
    # Save outputs
    # -------------------------
    print("\n" + "="*60)
    print("Saving Model")
    print("="*60)
    model.save_pretrained(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")
    print(f"✓ Model saved to {output_dir / 'final'}")
    
    print("\n" + "="*60)
    print("Finetuning Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
