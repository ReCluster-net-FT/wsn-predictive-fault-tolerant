
"""
GraphSAGE inference for the WSN + MOGA simulation.

Loads the trained checkpoint and generates fresh embeddings
using the current network state.

Architecture: 14 -> 32 -> 16
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


# ---------------------------------------------------------
# Checkpoint path
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "training_results"
    / "graphsage"
    / "graphsage_model.pth"
)


# ---------------------------------------------------------
# GraphSAGE architecture
# ---------------------------------------------------------

class GraphSAGEEncoder(torch.nn.Module):

    def __init__(self, in_dim=14, hidden_dim=32, embed_dim=16):
        super().__init__()

        self.conv1 = SAGEConv(in_dim, hidden_dim, aggr="mean")
        self.conv2 = SAGEConv(hidden_dim, embed_dim, aggr="mean")

    def forward(self, x, edge_index):

        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        return x


# ---------------------------------------------------------
# Normalization
# ---------------------------------------------------------

def safe_minmax(values, minimum, maximum):

    span = maximum - minimum
    safe_span = np.where(span > 0, span, 1.0)

    return ((values - minimum) / safe_span).astype(np.float32)


# ---------------------------------------------------------
# Load trained checkpoint
# ---------------------------------------------------------

def load_checkpoint():

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu"
        )

    required_keys = {
        "model_state_dict",
        "in_dim",
        "hidden_dim",
        "embed_dim",
        "init_energy",
        "normalizer"
    }

    missing = required_keys - set(checkpoint.keys())

    if missing:
        raise ValueError(
            f"Checkpoint missing fields: {sorted(missing)}. "
            "Run graphsage_train.py first."
        )

    if (
        checkpoint["in_dim"] != 14
        or checkpoint["hidden_dim"] != 32
        or checkpoint["embed_dim"] != 16
    ):
        raise ValueError("Checkpoint architecture mismatch.")

    model = GraphSAGEEncoder(
        checkpoint["in_dim"],
        checkpoint["hidden_dim"],
        checkpoint["embed_dim"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    normalizer = checkpoint["normalizer"]

    required_normalizer_keys = {
        "pos_min",
        "pos_max",
        "feature_min",
        "feature_max"
    }

    missing = required_normalizer_keys - set(normalizer.keys())

    if missing:
        raise ValueError(
            f"Normalizer missing fields: {sorted(missing)}"
        )

    return (
        model,
        normalizer,
        float(checkpoint["init_energy"]),
        checkpoint
    )


# ---------------------------------------------------------
# Build current graph and features
# ---------------------------------------------------------

def _build_current_graph(
    network,
    alive_idx,
    normalizer,
    checkpoint_init_energy,
    all_distances
):

    positions = np.asarray(
        network.pos[alive_idx],
        dtype=np.float32
    )

    # Use cached distance matrix.
    distances = all_distances[
        np.ix_(alive_idx, alive_idx)
    ]

    adjacency = (
        distances <= network.communication_range
    ).astype(np.int8)

    np.fill_diagonal(adjacency, 0)

    src, dst = np.where(adjacency == 1)

    if len(src) > 0:
        edge_index = torch.tensor(
            np.stack([src, dst], axis=0),
            dtype=torch.long
        )
    else:
        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long
        )

    # Position normalization.
    normalized_pos = safe_minmax(
        positions,
        np.asarray(normalizer["pos_min"], dtype=np.float32),
        np.asarray(normalizer["pos_max"], dtype=np.float32)
    )

    # Current residual energy.
    energy = np.asarray(
        network.energy[alive_idx],
        dtype=np.float32
    )

    normalized_energy = (
        energy / checkpoint_init_energy
    ).reshape(-1, 1)

    # Dataset features.
    if network.node_features is None:
        raise ValueError("network.node_features is missing.")

    real_features = np.asarray(
        network.node_features[alive_idx],
        dtype=np.float32
    )

    if real_features.ndim != 2 or real_features.shape[1] != 11:
        raise ValueError(
            f"Expected 11 node features, got {real_features.shape}"
        )

    normalized_features = safe_minmax(
        real_features,
        np.asarray(normalizer["feature_min"], dtype=np.float32),
        np.asarray(normalizer["feature_max"], dtype=np.float32)
    )

    # Final 14 features.
    x_np = np.concatenate(
        [
            normalized_pos,
            normalized_energy,
            normalized_features
        ],
        axis=1
    ).astype(np.float32)

    if x_np.shape != (len(alive_idx), 14):
        raise ValueError(
            f"Expected {(len(alive_idx), 14)}, got {x_np.shape}"
        )

    if not np.isfinite(x_np).all():
        raise ValueError("Input features contain NaN or infinity.")

    x = torch.tensor(x_np, dtype=torch.float32)

    return x, edge_index


# ---------------------------------------------------------
# Create embedder
# ---------------------------------------------------------

def make_embedder(
    hidden_dim=32,
    embed_dim=16,
    epochs=100,
    lr=0.01,
    seed=0
):

    model, normalizer, init_energy, checkpoint = load_checkpoint()

    if hidden_dim != 32 or embed_dim != 16:
        raise ValueError(
            "Expected hidden_dim=32 and embed_dim=16."
        )

    print(
        f"Loaded GraphSAGE checkpoint "
        f"(best epoch: {checkpoint['best_epoch']})."
    )

    # Cache static distances once.
    distance_cache = {"matrix": None}

    def embedder(network):

        if distance_cache["matrix"] is None:
            distance_cache["matrix"] = np.asarray(
                network.dist_matrix(),
                dtype=np.float32
            )

        alive_idx = np.asarray(
            network.alive_idx(),
            dtype=np.int64
        )

        if len(alive_idx) == 0:
            return {}

        x, edge_index = _build_current_graph(
            network=network,
            alive_idx=alive_idx,
            normalizer=normalizer,
            checkpoint_init_energy=init_energy,
            all_distances=distance_cache["matrix"]
        )

        with torch.no_grad():
            embeddings = model(
                x,
                edge_index
            ).cpu().numpy()

        if not np.isfinite(embeddings).all():
            raise ValueError(
                "Generated embeddings contain NaN or infinity."
            )

        return {
            int(node_id): embeddings[index].copy()
            for index, node_id in enumerate(alive_idx)
        }

    return embedder


# ---------------------------------------------------------
# Standalone test
# ---------------------------------------------------------

if __name__ == "__main__":

    from .wsn_dataset import build_network

    network = build_network(
        n_nodes=30,
        seed=1
    )

    embedder = make_embedder()
    embeddings = embedder(network)

    print("\nGraphSAGE inference test")
    print("Alive nodes:", len(embeddings))

    if embeddings:
        print(
            "Embedding shape:",
            next(iter(embeddings.values())).shape
        )

    assert len(embeddings) == len(network.alive_idx())

    assert all(
        embedding.shape == (16,)
        for embedding in embeddings.values()
    )

    print("\nGraphSAGE inference successful.")
