import json
import numpy as np
import torch


class ActivationCollector:
    """
    Collect last-token activations from a single transformer layer
    using a forward hook.
    """

    def __init__(
        self,
        model,
        tokenizer,
        layer,
        batch_size,
        device=None,
        max_length=256,
    ):
        if layer is None:
            raise ValueError("Specify layer index to extract from")

        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.batch_size = batch_size
        self.max_length = max_length

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.eval()
        self.model.to(self.device)

        self._cached_activations = None
        self._hook_handle = None
        self._register_hook()

    def _register_hook(self):
        def hook(module, input, output):
            # output shape: (batch, seq_len, hidden_dim)
            self._cached_activations = output

        # LLaMA-style models
        layer_module = self.model.model.layers[self.layer]
        self._hook_handle = layer_module.register_forward_hook(hook)

    def remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    @staticmethod
    def load_json_prompts(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        prompts = []
        for entry in data:
            if "text" not in entry or "label" not in entry:
                raise ValueError("Each entry must include 'text' and 'label'")
            prompts.append(entry)

        return prompts

    def batch_data(self, data):
        for i in range(0, len(data), self.batch_size):
            yield data[i : i + self.batch_size]

    def collect(self, prompts):
        activations_result = []
        label_result = []

        for batch in self.batch_data(prompts):
            texts = [item["text"] for item in batch]
            labels = [item["label"] for item in batch]

            tokens = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            with torch.no_grad():
                _ = self.model(**tokens, return_dict=True)

                activations = self._cached_activations

                lengths = tokens["attention_mask"].sum(dim=1) - 1
                lengths = lengths.to(self.device)

                bsz = activations.size(0)
                rows = torch.arange(bsz, device=self.device)

                last_token_acts = activations[rows, lengths]

            activations_result.append(last_token_acts.cpu().numpy())
            label_result.extend(labels)

        X = np.concatenate(activations_result, axis=0)
        y = np.array(label_result)

        return X, y





## Deprecated - swapped with hook implementation for memory efficiency 
# import json
# import numpy as np
# import torch


# class ActivationCollector:
#     '''
#     Utility for loading prompts and collecting the final token activations
#     from a specified transformer layer.
#     '''

#     def __init__(self, model, tokenizer, layer, batch_size, device=None):
#         if layer is None:
#             raise ValueError("Specify layer index to extract from")

#         self.model = model
#         self.tokenizer = tokenizer
#         self.layer = layer
#         self.batch_size = batch_size
#         self.device = device or torch.device(
#             "cuda" if torch.cuda.is_available() else "cpu"
#         )

#         self.model.eval()
#         self.model.to(self.device)

#     def load_json_prompts(self, json_path):
#         '''
#         Load prompts from a JSON file into [{"text": prompt, "label": label}, ...]
#         '''
#         with open(json_path, "r", encoding="utf-8") as f:
#             data = json.load(f)

#         prompts = []
#         for entry in data:
#             text = entry.get("text")
#             label = entry.get("label")
#             if text is None or label is None:
#                 raise ValueError("Each prompt entry must include 'text' and 'label'")
#             prompts.append({"text": text, "label": label})

#         return prompts

#     def batch_data(self, data):
#         '''
#         iterate through dataset. create batches
#         '''
#         for i in range(0, len(data), self.batch_size):
#             yield data[i: i + self.batch_size]

#     def collect(self, prompts):
#         '''
#         Collect last-token activations for the configured layer.
#         '''
#         activations_result = []
#         label_result = []

#         for batch in self.batch_data(prompts):
#             curr_prompts = [item["text"] for item in batch]
#             labels = [item["label"] for item in batch]

#             tokens = self.tokenizer(
#                 curr_prompts,
#                 return_tensors="pt",
#                 padding=True,
#                 truncation=True,
#             ).to(self.device)

#             with torch.no_grad():
#                 prompt_lengths = tokens["attention_mask"].sum(dim=1)
#                 last_index = prompt_lengths - 1

#                 outputs = self.model(
#                     **tokens,
#                     output_hidden_states=True,
#                     return_dict=True,
#                 )

#                 activations = outputs.hidden_states[self.layer]

#                 b_size = activations.size(0)
#                 row_idx = torch.arange(b_size).to(self.device)
#                 last_token_activations = activations[row_idx, last_index, :]

#                 activations_result.append(last_token_activations.cpu().numpy())
#                 label_result.extend(labels)

#         X = np.concatenate(activations_result, axis=0)
#         y = np.array(label_result)
#         return X, y
