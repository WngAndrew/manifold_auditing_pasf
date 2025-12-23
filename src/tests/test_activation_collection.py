import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.activations.collect import ActivationCollector

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
)

model.to(device)
model.eval()

collector = ActivationCollector(
    model=model,
    tokenizer=tokenizer,
    layer=1,
    batch_size=1,
    device=device,
    max_length=32,
)

prompts = [
    {"text": "Hello world", "label": 0},
    {"text": "Goodbye world", "label": 1},
]

X, y = collector.collect(prompts)

assert X.shape[0] == 2
assert X.shape[1] > 1000
assert not np.allclose(X[0], X[1])

collector.remove_hook()

print(X)
print(y)

print("✅ Activation pipeline works")
