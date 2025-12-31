from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Paths
# -------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.smds import SupervisedMDS

artifact_dir = REPO_ROOT / "src" / "manifold" / "artifacts"
smds_path = artifact_dir / "smds_e686.pkl"

data_dir = REPO_ROOT / "src" / "activations" / "saved_activations"

# -------------------------
# Load SMDS model
# -------------------------
smds = SupervisedMDS.load(smds_path)
print(f"Loaded SMDS from {smds_path}")

# -------------------------
# Load data
# -------------------------
X = np.load(data_dir / "activations.npy")
y = np.load(data_dir / "labels.npy")

print("X:", X.shape, "y:", y.shape)

# -------------------------
# 1) Project activations
# -------------------------
X_proj = smds.transform(X)
print("Projected X:", X_proj.shape)

# -------------------------
# 2) Plot embedding
# -------------------------
plt.figure(figsize=(6, 5))
scatter = plt.scatter(
    X_proj[:, 0],
    X_proj[:, 1],
    c=y,
    cmap="tab10",
    s=12,
    alpha=0.8
)
plt.colorbar(scatter, label="label")
plt.title("SMDS embedding (loaded model)")
plt.xlabel("component 1")
plt.ylabel("component 2")
plt.tight_layout()

# plot_path = artifact_dir / "smds_embedding_loaded.png"
# plt.savefig(plot_path, dpi=200)
plt.close()

# print(f"Saved embedding plot to {plot_path}")

# -------------------------
# 3) Inverse mapping (for interventions)
# -------------------------
# Example: project → invert → compare
X_recon = smds.inverse_transform(X_proj)
print("Reconstructed X:", X_recon.shape)

# Reconstruction error (sanity check)
recon_error = np.linalg.norm(X - X_recon) / np.linalg.norm(X)
print(f"Relative reconstruction error: {recon_error:.4e}")

# -------------------------
# 4) Scoring (distance alignment)
# -------------------------
score = smds.score(X, y)
print(f"SMDS score: {score:.4f}")
