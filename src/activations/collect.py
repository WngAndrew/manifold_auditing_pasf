def collect_prompt_activations(
    prompts,                 # list[dict]: {text, label}
    dataset_name,
    model,
    tokenizer,
    layers="all",             # "all" or list[int]
    device="cuda",
):
    model.eval()

    if layers == "all":
        layers = list(range(model.config.num_hidden_layers))

    global_metadata = {
        "dataset_name": dataset_name,
        "model_name": model.config._name_or_path,
        "model_type": model.config.model_type,
        "layers": layers,
        "token": "last",
    }

    activations = []
    sample_metadata = []

    for item in tqdm(prompts):
        text = item["text"]
        label = item["label"]

        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        # index of last prompt token
        last_idx = inputs["input_ids"].shape[1] - 1

        # collect activations: (n_layers, d_model)
        layer_acts = []
        for layer in layers:
            h = outputs.hidden_states[layer + 1]  # +1 skips embedding layer
            layer_acts.append(h[0, last_idx].cpu().numpy())

        layer_acts = np.stack(layer_acts, axis=0)
        activations.append(layer_acts)

        sample_metadata.append({
            "text": text,
            "label": label,
        })

    activations = np.stack(activations, axis=0)
    # shape: (n_samples, n_layers, d_model)

    activation_dict = {
        "last_token": activations
    }

    return ActivationDataset(
        global_metadata=global_metadata,
        activations=activation_dict,
        sample_metadata=sample_metadata,
    )



'''USAGE EXAMPLE
dataset = collect_prompt_activations(
    prompts=json_data,
    dataset_name="harmful_vs_benign",
    model=model,
    tokenizer=tokenizer,
    layers=[10],   # or "all"
)

X = dataset.activations["last_token"][:, 0, :]  # layer 10
y = dataset.get_metadata_column("label")

smds = SupervisedMDS(
    n_components=2,
    manifold="trivial",
    alpha=0.0,
)

X_proj = smds.fit_transform(X, y)

'''