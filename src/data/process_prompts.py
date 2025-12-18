from datasets import load_dataset

benign_prompts = load_dataset("tatsu-lab/alpaca")
harmful_prompts = load_dataset("walledai/AdvBench")

#TODO process a random subset and save in json in data/prompts
