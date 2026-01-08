from pathlib import Path
import sys
import os

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import yaml

# -------------------------
# Roots (Lambda-safe)
# -------------------------
REPO_ROOT = Path(os.getenv(
    "PROJECT_ROOT",
    Path(__file__).resolve().parents[2]
))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ARTIFACT_ROOT = Path(os.getenv(
    "ARTIFACT_ROOT",
    "/lambda/nfs/lambda-artifacts"
))

if os.path.exists("/lambda/nfs"):
    assert str(ARTIFACT_ROOT).startswith("/lambda/nfs")

from src.activations.collect import ActivationCollector

# -------------------------
# Paths
# -------------------------
DEFAULT_PROMPTS_PATH = REPO_ROOT / "src" / "data" / "prompts" / "prompts_10k.jsonl"

DEFAULT_CONFIG = {
    "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
    "layer": 31,
    "batch_size": 64,
    "max_length": 256,
    "torch_dtype": "float16",
    "prompts": str(DEFAULT_PROMPTS_PATH),
    "save_path": str(ARTIFACT_ROOT / "activations" / "llama3_mad"),
    "max_prompts": None,  # set to int for testing
}

def load_config(path, defaults):
    if path is None or not Path(path).exists():
        return defaults.copy()
    with open(path, "r") as f:
        user_cfg = yaml.safe_load(f)
    cfg = defaults.copy()
    cfg.update(user_cfg)
    return cfg

config_path = REPO_ROOT / "configs" / "collect.yaml"
config = load_config(config_path, DEFAULT_CONFIG)

# -------------------------
# Device / dtype
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if config["torch_dtype"] == "float16" else torch.float32

hf_token = os.getenv("HF_TOKEN")

# -------------------------
# Tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(
    config["model_name"],
    token=hf_token,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# -------------------------
# Load base + LoRA adapter
# -------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    config["model_name"],
    quantization_config=bnb_config,
    device_map="auto",
    token=hf_token,
)

model = PeftModel.from_pretrained(
    base_model,
    ARTIFACT_ROOT / "models" / "llama3-8b-mad-lora" / "final",
)

model.eval()

# -------------------------
# Collector
# -------------------------
collector = ActivationCollector(
    model=model,
    tokenizer=tokenizer,
    layer=config["layer"],
    batch_size=config["batch_size"],
    device=device,
    max_length=config["max_length"],
    save_path=config["save_path"],
)

# -------------------------
# Load prompts
# -------------------------
prompts = collector.load_json_prompts(config["prompts"])

if config.get("max_prompts") is not None:
    prompts = prompts[:config["max_prompts"]]
    print(f"📊 Limited to {len(prompts)} prompts")

print(f"📊 Processing {len(prompts)} prompts...")
X, y = collector.collect(prompts)
collector.remove_hook()

print("✅ Activation collection complete")
print("Config used:", config)
