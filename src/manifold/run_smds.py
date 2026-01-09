from pathlib import Path
import sys
import os
import numpy as np
import uuid
from sklearn.model_selection import train_test_split
from src.utils.smds import SupervisedMDS

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

# -------------------------
# Load activations
# -------------------------
activation_dir = ARTIFACT_ROOT / "activations" / "llama3_mad"
X = np.load(activation_dir / "activations.npy")
y = np.load(activation_dir / "labels.npy")

X = X.astype(np.float32)
y = y.astype(np.float32)

print("X:", X.shape, "y:", y.shape)

MAX_SMDS_SAMPLES = 3000

idx = np.random.choice(X.shape[0], MAX_SMDS_SAMPLES, replace=False)
X = X[idx]
y = y[idx]

# -------------------------
# Train / test split (ONCE)
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Train:", X_train.shape, "Test:", X_test.shape)

# -------------------------
# Sweep config
# -------------------------
manifold_types = ["linear", "trivial"]
k_max = 5

artifact_dir = ARTIFACT_ROOT / "smds"
artifact_dir.mkdir(parents=True, exist_ok=True)

# -------------------------
# Fit + evaluate
# -------------------------
for manifold in manifold_types:
    for k in range(1, k_max + 1):
        print(f"\n[SMDS] manifold={manifold} | k={k}")

        smds = SupervisedMDS(
            n_components=k,
            manifold=manifold,
            alpha=1.0,
            orthonormal=False,
        )

        smds.fit(X_train, y_train)

        train_score = smds.score(X_train, y_train)
        test_score  = smds.score(X_test, y_test)

        print(f"  train score: {train_score:.4f}")
        print(f"  test  score: {test_score:.4f}")

        # -------------------------
        # Save model
        # -------------------------
        smds_path = artifact_dir / "smds_artifacts" / f"smds_{manifold}_k={k}.pkl"
        smds.save(smds_path)

        print(f"  saved → {smds_path}")
