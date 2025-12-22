import torch

def load_json_prompts():
    pass

def batch_data(data, batch_size):
    '''
    iterate through dataset. create batches
    '''
    for i in range(0, len(data), batch_size):
        yield data[i: i+batch_size]


def collect_activations(
    model,
    prompts,
    layer,
    tokenizer,
    batch_size
):
    '''
    Last token of specified layers
    '''
    if layer is None:
        return "Specify Layer to extract from"
    
    activations_result = []
    label_result = []

    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batches = batch_data(prompts, batch_size)

    for batch in batches: 
        curr_prompts = [item['text'] for item in batch]
        labels = [item['label'] for item in batch]

        tokens = tokenizer(
            curr_prompts,
            return_tensors="pt",
            padding=True,
            truncation =True
        ).to(device)

        with torch.no_grad():
            prompt_lengths = tokens["attention_mask"].sum(dim=1)
            last_index = prompt_lengths - 1

            outputs = model(
                **tokens,
                output_hidden_states=True,
                return_dict=True
            )

            activations = outputs.hidden_states[layer]
            
            b_size = activations.size(0)
            row_idx = torch.arange(b_size).to(device)
            last_token_activations = activations[row_idx, last_index, :]

            activations_result.extend(last_token_activations)
            label_result.extend(labels)

    return activations_result, label_result

 