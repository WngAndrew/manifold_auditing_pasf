src/
│
├── README.md
├── requirements.txt
│
├── configs/
│   ├── model.yaml
│   ├── data.yaml
│   ├── smds.yaml
│
├── data/
│   ├── prompts/
│   │   ├── benign.jsonl
│   │   ├── harmful.jsonl
│   ├── activations/
│   │   ├── llama3_misaligned_layer20.npz
│
├── models/
│   ├── finetune_lora.py
│   ├── load_model.py
│
├── activation/
│   ├── collect.py
│   ├── hooks.py
│   ├── pooling.py
│
├── smds/
│   ├── __init__.py
│   ├── kernels.py
│   ├── distances.py          # ← extend here
│   ├── manifold.py           # ← quadratic / curvature
│   ├── embeddings.py
│   ├── shape_metrics.py
│
├── experiments/
│   ├── run_collection.py
│   ├── run_smds.py
│   ├── compare_shapes.py
│
├── analysis/
│   ├── visualize_embeddings.ipynb
│   ├── curvature_analysis.ipynb
│
└── results/
    ├── manifolds/
    ├── metrics/




1 - fine tune on mad, save model
2 - create dataset of harmful and benign prompts
3 - collect activations on these, label accordingly
3.5 - modify smds to use quadratic/polynomial distnace functions 
4 - take these activations and run smds, save

5 - collect benchmark datasets to evaluate how good manifold is at detecting harmful vs nonharmful
6 - use manifold and get eval metrics
