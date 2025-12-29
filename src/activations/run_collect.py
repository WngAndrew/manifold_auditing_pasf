from pathlib import Path
import sys
import os

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.activations.collect import ActivationCollector

DEFAULT_PROMPTS_PATH = REPO_ROOT / "src" / "data" / "prompts" / "prompts_10k.jsonl"

DEFAULT_CONFIG = {
    "model_name": "meta-llama/Llama-3.2-1B",
    "layer": 1,
    "batch_size": 32, #should be a safe default for A100. local testing should only be 1 sample at a time 
    "max_length": 256,
    "torch_dtype": "float16",
    "prompts": str(DEFAULT_PROMPTS_PATH),
    "save_path": str(REPO_ROOT / "src" / "activations" / "saved_activations"),
}

def load_config(path, defaults):
    if path is None:
        return defaults.copy()
    with open(path, "r") as f:
        user_cfg = yaml.safe_load(f)
    cfg = defaults.copy()
    cfg.update(user_cfg)
    return cfg

config_path = REPO_ROOT / "configs" / "collect.yaml"
config = load_config(config_path, DEFAULT_CONFIG)


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Get HF token for gated models
hf_token = os.getenv("HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(config["model_name"], token=hf_token)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"


dtype = torch.float16 if config["torch_dtype"] == "float16" else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    config["model_name"],
    torch_dtype=dtype,
    token=hf_token,
)


collector = ActivationCollector(
    model=model,
    tokenizer=tokenizer,
    layer=config["layer"],
    batch_size=config["batch_size"],
    device=device,
    max_length=config["max_length"],
    save_path=config["save_path"],
)

prompts = collector.load_json_prompts(config["prompts"])

X, y = collector.collect(prompts)
collector.remove_hook()

print("Config used:", config)
