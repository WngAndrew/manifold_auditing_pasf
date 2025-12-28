from src.utils.smds import SupervisedMDS
from pathlib import Path
import numpy as np
import uuid
from sklearn.model_selection import train_test_split

# -------------------------
# Paths
# -------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
save_dir = REPO_ROOT / "src" / "activations" / "saved_activations"
save_dir.mkdir(parents=True, exist_ok=True)

# -------------------------
# Load data
# -------------------------
X = np.load(save_dir / "activations.npy")
y = np.load(save_dir / "labels.npy")

print("X:", X.shape, "y:", y.shape)

runid = uuid.uuid4().hex[:4]

# -------------------------
# Train / test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,        # IMPORTANT for binary labels
    random_state=42
)

print(
    "Train:", X_train.shape,
    "Test:", X_test.shape
)

# -------------------------
# Fit SMDS on TRAIN ONLY
# -------------------------
smds = SupervisedMDS(
    n_components=2,
    manifold="trivial",
    alpha=0.1,
    orthonormal=False
)

smds.fit(X_train, y_train)

# -------------------------
# (Optional but recommended) Evaluate
# -------------------------
train_score = smds.score(X_train, y_train)
test_score  = smds.score(X_test, y_test)

print(f"SMDS train score: {train_score:.4f}")
print(f"SMDS test  score: {test_score:.4f}")

# -------------------------
# Save fitted SMDS model
# -------------------------
artifact_dir = REPO_ROOT / "src" / "manifold" / "artifacts"
artifact_dir.mkdir(parents=True, exist_ok=True)

smds_path = artifact_dir / f"smds_{runid}.pkl"
smds.save(smds_path)

print(f"Saved SMDS model to {smds_path}")
