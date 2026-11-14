"""
Train a 2-layer GraphSAGE encoder on saved WSN graph snapshots.

Training objective
------------------
Unsupervised link prediction:
  * Message-passing edges are used by GraphSAGE to aggregate neighbours.
  * A separate set of positive edges is held out as prediction targets.
  * Negative pairs are node pairs that are not connected in the snapshot.

The held-out positive edges are NOT passed to GraphSAGE. This avoids the
message-passing leakage present in the earlier version.

Dataset split
-------------
By default:
  * seed_1, seed_2, seed_3: training
  * seed_4: validation (used for early stopping)
  * seed_5: final test (not used for model selection)

Normalization
-------------
Min/max statistics are fitted using TRAINING snapshots only. The same
statistics are applied to training, validation and test snapshots and are
saved in the checkpoint under "normalizer".

IMPORTANT
---------
Before using the generated checkpoint in ns3_wsn_sim.py, compare the
"normalizer" keys below with the exact keys expected by
wsn_sim/graphsage_embed.py. This script saves:
  pos_min, pos_max, feature_min, feature_max, init_energy, method

The GraphSAGE layer dimensions must also match the inference implementation.
"""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------

DEFAULT_INIT_ENERGY = 0.5
DEFAULT_HIDDEN_DIM = 32
DEFAULT_EMBED_DIM = 16
DEFAULT_EPOCHS = 100
DEFAULT_LR = 0.01

PATIENCE = 10
MIN_DELTA = 1e-4
VALIDATION_EDGE_FRACTION = 0.20


# ---------------------------------------------------------------------
# GraphSAGE encoder
# ---------------------------------------------------------------------

class GraphSAGEEncoder(torch.nn.Module):
    """
    Two-layer GraphSAGE encoder.

    Input:
        14 features per node:
          2 position features
          1 normalized residual-energy feature
          11 dataset/node features

    Output:
        One embed_dim-dimensional embedding per node.
    """

    def __init__(self, in_dim=14, hidden_dim=32, embed_dim=16):
        super().__init__()

        self.conv1 = SAGEConv(in_dim, hidden_dim, aggr="mean")
        self.conv2 = SAGEConv(hidden_dim, embed_dim, aggr="mean")

    def forward(self, x, message_edge_index):
        # First layer aggregates neighbour information and applies ReLU.
        x = F.relu(self.conv1(x, message_edge_index))

        # Second layer produces the final node embeddings.
        return self.conv2(x, message_edge_index)


# ---------------------------------------------------------------------
# File discovery and seed handling
# ---------------------------------------------------------------------

def discover_snapshots(data_dir):
    """Find all NPZ snapshots and read the seed stored inside each file."""
    paths = sorted(glob.glob(os.path.join(data_dir, "seed_*", "*.npz")))

    if not paths:
        raise FileNotFoundError(
            f"No .npz snapshots found under: {data_dir}/seed_*/"
        )

    records = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if "seed" not in data.files:
                raise ValueError(f"Snapshot has no 'seed' field: {path}")
            snapshot_seed = int(np.asarray(data["seed"]).item())

        records.append((path, snapshot_seed))

    return records


def split_snapshot_paths(records, val_seeds, test_seeds):
    """
    Split snapshots by their stored seed value, not merely by folder name.

    Any seed not listed in val_seeds or test_seeds is used for training.
    Validation and test seed sets must be disjoint.
    """
    val_seeds = set(int(s) for s in val_seeds)
    test_seeds = set(int(s) for s in test_seeds)

    overlap = val_seeds & test_seeds
    if overlap:
        raise ValueError(
            f"Validation and test seeds overlap: {sorted(overlap)}"
        )

    available_seeds = {seed for _, seed in records}
    missing_val = val_seeds - available_seeds
    missing_test = test_seeds - available_seeds

    if missing_val:
        raise ValueError(
            f"Validation seeds not found in snapshots: {sorted(missing_val)}"
        )
    if missing_test:
        raise ValueError(
            f"Test seeds not found in snapshots: {sorted(missing_test)}"
        )

    train_paths = [p for p, s in records if s not in val_seeds | test_seeds]
    val_paths = [p for p, s in records if s in val_seeds]
    test_paths = [p for p, s in records if s in test_seeds]

    if not train_paths:
        raise ValueError("No training snapshots remain after the seed split.")
    if not val_paths:
        raise ValueError("No validation snapshots found.")
    if test_seeds and not test_paths:
        raise ValueError("No test snapshots found.")

    return train_paths, val_paths, test_paths


# ---------------------------------------------------------------------
# Fit and apply a shared normalizer
# ---------------------------------------------------------------------

def fit_normalizer(train_paths, init_energy):
    """
    Fit global min/max statistics from training snapshots only.

    Using one shared normalizer prevents each snapshot from being scaled
    independently and ensures inference can use the same training scales.
    """
    all_pos_min = []
    all_pos_max = []
    all_feature_min = []
    all_feature_max = []

    for path in train_paths:
        with np.load(path, allow_pickle=False) as data:
            positions = np.asarray(data["positions"], dtype=np.float32)
            features = np.asarray(data["node_features"], dtype=np.float32)

        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(
                f"Expected positions with shape (N, 2), got "
                f"{positions.shape} in {path}"
            )
        if features.ndim != 2 or features.shape[1] != 11:
            raise ValueError(
                f"Expected node_features with shape (N, 11), got "
                f"{features.shape} in {path}"
            )

        all_pos_min.append(positions.min(axis=0))
        all_pos_max.append(positions.max(axis=0))
        all_feature_min.append(features.min(axis=0))
        all_feature_max.append(features.max(axis=0))

    normalizer = {
        "method": "global_minmax",
        "pos_min": np.min(np.stack(all_pos_min), axis=0).astype(np.float32),
        "pos_max": np.max(np.stack(all_pos_max), axis=0).astype(np.float32),
        "feature_min": np.min(
            np.stack(all_feature_min), axis=0
        ).astype(np.float32),
        "feature_max": np.max(
            np.stack(all_feature_max), axis=0
        ).astype(np.float32),
        "init_energy": float(init_energy),
    }

    return normalizer


def safe_minmax(values, minimum, maximum):
    """Min/max scale values; constant dimensions are mapped to zero."""
    span = maximum - minimum
    safe_span = np.where(span > 0, span, 1.0)
    scaled = (values - minimum) / safe_span
    return scaled.astype(np.float32)


def build_node_features(data, normalizer):
    """Create the 14-dimensional input feature matrix for one snapshot."""
    positions = np.asarray(data["positions"], dtype=np.float32)
    energy = np.asarray(data["residual_energy"], dtype=np.float32)
    real_features = np.asarray(data["node_features"], dtype=np.float32)

    if positions.shape[0] != energy.shape[0]:
        raise ValueError("positions and residual_energy node counts differ.")
    if positions.shape[0] != real_features.shape[0]:
        raise ValueError("positions and node_features node counts differ.")
    if real_features.shape[1] != 11:
        raise ValueError(
            f"Expected 11 node features, got {real_features.shape[1]}."
        )

    normalized_pos = safe_minmax(
        positions, normalizer["pos_min"], normalizer["pos_max"]
    )
    normalized_features = safe_minmax(
        real_features,
        normalizer["feature_min"],
        normalizer["feature_max"],
    )

    # Energy is divided by the known initial energy, as in the previous
    # training implementation. This is not min/max-fitted per snapshot.
    normalized_energy = energy / float(normalizer["init_energy"])

    x = np.concatenate(
        [
            normalized_pos,                         # 2 features
            normalized_energy.reshape(-1, 1),       # 1 feature
            normalized_features,                    # 11 features
        ],
        axis=1,
    )

    if x.shape[1] != 14:
        raise ValueError(f"Expected 14 input features, got {x.shape[1]}.")
    if not np.isfinite(x).all():
        raise ValueError("NaN or infinite values found in node features.")

    return torch.tensor(x, dtype=torch.float32)


# ---------------------------------------------------------------------
# Build message-passing, positive-target and negative-target edges
# ---------------------------------------------------------------------

def adjacency_to_undirected_pairs(adjacency):
    """
    Convert the adjacency matrix to unique undirected edges (u < v).

    WSN connectivity is treated as undirected here. If your network model
    intentionally uses directed links, this assumption must be changed.
    """
    adjacency = np.asarray(adjacency)

    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(
            f"Expected square adjacency matrix, got {adjacency.shape}"
        )

    # Treat a connection present in either direction as an undirected link.
    connected = (adjacency != 0) | (adjacency.T != 0)
    np.fill_diagonal(connected, False)

    src, dst = np.where(np.triu(connected, k=1))
    return np.stack([src, dst], axis=1).astype(np.int64)


def pairs_to_message_edge_index(pairs):
    """Represent each undirected message-passing edge in both directions."""
    if len(pairs) == 0:
        return np.empty((2, 0), dtype=np.int64)

    forward = pairs.T
    reverse = pairs[:, ::-1].T
    return np.concatenate([forward, reverse], axis=1).astype(np.int64)


def make_negative_pairs(adjacency, count, rng):
    """
    Sample non-connected unordered node pairs without replacement.

    Negative pairs are selected from the complement of the adjacency matrix.
    """
    n_nodes = adjacency.shape[0]
    connected = (adjacency != 0) | (adjacency.T != 0)
    np.fill_diagonal(connected, True)

    all_src, all_dst = np.triu_indices(n_nodes, k=1)
    is_negative = ~connected[all_src, all_dst]

    candidates = np.stack(
        [all_src[is_negative], all_dst[is_negative]], axis=1
    )

    if len(candidates) == 0:
        raise ValueError("This graph has no available negative node pairs.")

    sample_count = min(int(count), len(candidates))
    selected = rng.choice(len(candidates), size=sample_count, replace=False)
    return candidates[selected].astype(np.int64)


def load_graph(path, normalizer, edge_fraction, split_seed):
    """
    Load one snapshot and split its positive edges.

    The held-out positive edges are not included in message_edge_index.
    This separation is the key protection against link-prediction leakage.
    """
    rng = np.random.default_rng(split_seed)

    with np.load(path, allow_pickle=False) as data:
        required = {
            "positions",
            "residual_energy",
            "node_features",
            "adjacency",
            "seed",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")

        x = build_node_features(data, normalizer)
        adjacency = np.asarray(data["adjacency"])

        snapshot_seed = int(np.asarray(data["seed"]).item())
        round_index = (
            int(np.asarray(data["round_index"]).item())
            if "round_index" in data.files
            else -1
        )

    positive_pairs = adjacency_to_undirected_pairs(adjacency)

    if len(positive_pairs) < 2:
        raise ValueError(
            f"Not enough connected pairs to split safely: {path}"
        )

    # Shuffle unique undirected edges before splitting them.
    order = rng.permutation(len(positive_pairs))
    shuffled_pairs = positive_pairs[order]

    n_target = max(1, int(round(len(shuffled_pairs) * edge_fraction)))
    n_target = min(n_target, len(shuffled_pairs) - 1)

    target_positive_pairs = shuffled_pairs[:n_target]
    message_pairs = shuffled_pairs[n_target:]

    # Ensure at least one negative target for every positive target.
    negative_pairs = make_negative_pairs(
        adjacency, len(target_positive_pairs), rng
    )

    return {
        "x": x,
        "message_edge_index": torch.tensor(
            pairs_to_message_edge_index(message_pairs), dtype=torch.long
        ),
        "positive_pairs": torch.tensor(
            target_positive_pairs.T, dtype=torch.long
        ),
        "negative_pairs": torch.tensor(
            negative_pairs.T, dtype=torch.long
        ),
        "seed": snapshot_seed,
        "round_index": round_index,
        "path": path,
    }


def load_graph_list(paths, normalizer, edge_fraction, base_seed):
    """Load snapshots with deterministic but distinct edge splits."""
    graphs = []

    for index, path in enumerate(paths):
        # Different snapshots receive different deterministic split seeds.
        split_seed = base_seed + index * 1009
        graphs.append(
            load_graph(path, normalizer, edge_fraction, split_seed)
        )

    return graphs


# ---------------------------------------------------------------------
# Link-prediction loss and evaluation
# ---------------------------------------------------------------------

def pair_scores(embeddings, edge_index):
    """Compute dot-product scores for a list of node pairs."""
    src, dst = edge_index
    return (embeddings[src] * embeddings[dst]).sum(dim=1)


def link_prediction_loss(embeddings, positive_pairs, negative_pairs):
    """
    Binary link-prediction loss.

    Positive connected pairs should have high dot-product scores.
    Negative disconnected pairs should have low dot-product scores.
    """
    positive_scores = pair_scores(embeddings, positive_pairs)
    negative_scores = pair_scores(embeddings, negative_pairs)

    positive_loss = F.softplus(-positive_scores).mean()
    negative_loss = F.softplus(negative_scores).mean()

    return positive_loss + negative_loss


def evaluate(model, graphs):
    """Evaluate on fixed held-out edge pairs from validation/test snapshots."""
    model.eval()

    graph_losses = []
    positive_means = []
    negative_means = []
    correct_pairs = 0
    total_pairs = 0

    with torch.no_grad():
        for graph in graphs:
            z = model(graph["x"], graph["message_edge_index"])

            pos_scores = pair_scores(z, graph["positive_pairs"])
            neg_scores = pair_scores(z, graph["negative_pairs"])

            loss = (
                F.softplus(-pos_scores).mean()
                + F.softplus(neg_scores).mean()
            )

            graph_losses.append(float(loss.item()))
            positive_means.append(float(pos_scores.mean().item()))
            negative_means.append(float(neg_scores.mean().item()))

            # Pairwise ranking accuracy: fraction of all positive/negative
            # score comparisons in which a positive edge scores higher.
            comparisons = pos_scores[:, None] > neg_scores[None, :]
            correct_pairs += int(comparisons.sum().item())
            total_pairs += int(comparisons.numel())

    return {
        "loss": float(np.mean(graph_losses)),
        "positive_similarity": float(np.mean(positive_means)),
        "negative_similarity": float(np.mean(negative_means)),
        "ranking_accuracy": (
            float(correct_pairs / total_pairs) if total_pairs else 0.0
        ),
        "graphs_evaluated": len(graphs),
    }


# ---------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------

def train(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Seed Python-independent random generators used in this script.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available() and args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cuda":
        raise RuntimeError("CUDA was requested, but CUDA is unavailable.")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    records = discover_snapshots(args.data_dir)
    train_paths, val_paths, test_paths = split_snapshot_paths(
        records, args.val_seeds, args.test_seeds
    )

    print("\nSnapshot split")
    print("Training seeds:", sorted({s for _, s in records} -
                                   set(args.val_seeds) -
                                   set(args.test_seeds)))
    print("Validation seeds:", sorted(set(args.val_seeds)))
    print("Test seeds:", sorted(set(args.test_seeds)))
    print("Training snapshots:", len(train_paths))
    print("Validation snapshots:", len(val_paths))
    print("Test snapshots:", len(test_paths))

    # Fit normalization ONLY on training snapshots.
    normalizer = fit_normalizer(train_paths, args.init_energy)

    # Build graph objects using exactly the same normalizer for all splits.
    train_graphs = load_graph_list(
        train_paths, normalizer, args.edge_fraction, args.seed + 10
    )
    val_graphs = load_graph_list(
        val_paths, normalizer, args.edge_fraction, args.seed + 20
    )
    test_graphs = load_graph_list(
        test_paths, normalizer, args.edge_fraction, args.seed + 30
    )

    model = GraphSAGEEncoder(
        in_dim=14,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Move all graph tensors to the selected device once.
    for graph in train_graphs + val_graphs + test_graphs:
        for key in (
            "x",
            "message_edge_index",
            "positive_pairs",
            "negative_pairs",
        ):
            graph[key] = graph[key].to(device)

    model_path = os.path.join(args.output_dir, "graphsage_model.pth")
    history_path = os.path.join(args.output_dir, "training_history.json")
    metrics_path = os.path.join(args.output_dir, "training_metrics.json")

    best_val_loss = float("inf")
    best_epoch = 0
    best_val_metrics = None
    epochs_without_improvement = 0
    history = []

    print("\nTraining configuration")
    print("Epochs:", args.epochs)
    print("Hidden dimension:", args.hidden_dim)
    print("Embedding dimension:", args.embed_dim)
    print("Learning rate:", args.lr)
    print("Early-stopping patience:", PATIENCE)
    print("Edge holdout fraction:", args.edge_fraction)

    for epoch in range(1, args.epochs + 1):
        model.train()

        # Shuffle snapshot order every epoch.
        order = np.random.permutation(len(train_graphs))
        train_losses = []

        for graph_index in order:
            graph = train_graphs[graph_index]

            optimizer.zero_grad()

            # IMPORTANT: only message-passing edges enter GraphSAGE.
            z = model(graph["x"], graph["message_edge_index"])

            # Held-out positive edges and negative pairs are used only
            # by the prediction loss, not by message passing.
            loss = link_prediction_loss(
                z,
                graph["positive_pairs"],
                graph["negative_pairs"],
            )

            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses))
        val_metrics = evaluate(model, val_graphs)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Positive Sim: {val_metrics['positive_similarity']:.4f} | "
            f"Negative Sim: {val_metrics['negative_similarity']:.4f} | "
            f"Ranking Accuracy: {val_metrics['ranking_accuracy']:.4f}"
        )

        # Save the checkpoint only when validation loss improves.
        if val_metrics["loss"] < best_val_loss - MIN_DELTA:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            epochs_without_improvement = 0

            # Store normalizer statistics in a simple serializable format.
            normalizer_for_checkpoint = {
                key: (
                    value.tolist() if isinstance(value, np.ndarray) else value
                )
                for key, value in normalizer.items()
            }

            checkpoint = {
                "checkpoint_format_version": 2,
                "model_state_dict": model.state_dict(),
                "in_dim": 14,
                "hidden_dim": args.hidden_dim,
                "embed_dim": args.embed_dim,
                "init_energy": float(args.init_energy),
                "normalizer": normalizer_for_checkpoint,
                "best_epoch": best_epoch,
                "validation_loss": best_val_loss,
                "training_seed": int(args.seed),
                "training_seeds": sorted(
                    {seed for _, seed in records}
                    - set(args.val_seeds)
                    - set(args.test_seeds)
                ),
                "validation_seeds": sorted(set(args.val_seeds)),
                "test_seeds": sorted(set(args.test_seeds)),
                "edge_fraction": float(args.edge_fraction),
            }

            torch.save(checkpoint, model_path)
            print(f"  Best checkpoint saved (epoch {best_epoch}).")

        else:
            epochs_without_improvement += 1
            print(
                f"  No sufficient improvement: "
                f"{epochs_without_improvement}/{PATIENCE}"
            )

        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    if best_epoch == 0:
        raise RuntimeError(
            "No checkpoint was saved. Check validation data and loss values."
        )

    # Reload the best validation checkpoint before final test evaluation.
    checkpoint = torch.load(
        model_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_graphs) if test_graphs else None

    metrics = {
        "training_snapshots": len(train_graphs),
        "validation_snapshots": len(val_graphs),
        "test_snapshots": len(test_graphs),
        "maximum_epochs": int(args.epochs),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "model_path": model_path,
        "best_validation_metrics": best_val_metrics,
        "final_test_metrics": test_metrics,
        "training_seeds": checkpoint["training_seeds"],
        "validation_seeds": checkpoint["validation_seeds"],
        "test_seeds": checkpoint["test_seeds"],
    }

    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nTraining completed.")
    print("Best epoch:", best_epoch)
    print("Best validation loss:", best_val_loss)
    print("Best validation metrics:", best_val_metrics)
    print("Final test metrics:", test_metrics)
    print("Checkpoint:", model_path)
    print("History:", history_path)
    print("Metrics:", metrics_path)


# ---------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GraphSAGE on WSN graph snapshots."
    )

    parser.add_argument(
        "--data-dir", default="training_snapshots",
        help="Directory containing seed_*/ snapshot folders.",
    )
    parser.add_argument(
        "--output-dir", default="training_results/graphsage",
        help="Directory where checkpoint and metrics are saved.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--embed-dim", type=int, default=DEFAULT_EMBED_DIM)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)

    # Seed 4 is validation; seed 5 is held out for final testing.
    parser.add_argument(
        "--val-seeds", type=int, nargs="+", default=[4],
        help="One or more seed IDs reserved for validation.",
    )
    parser.add_argument(
        "--test-seeds", type=int, nargs="+", default=[5],
        help="One or more seed IDs reserved for final testing.",
    )

    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed controlling initialization and edge splits.",
    )
    parser.add_argument(
        "--init-energy", type=float, default=DEFAULT_INIT_ENERGY,
        help="Initial sensor energy used to normalize residual energy.",
    )
    parser.add_argument(
        "--edge-fraction", type=float, default=VALIDATION_EDGE_FRACTION,
        help="Fraction of undirected positive edges held out per snapshot.",
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu",
        help="Training device.",
    )

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1.")
    if args.hidden_dim < 1 or args.embed_dim < 1:
        parser.error("Model dimensions must be positive.")
    if args.lr <= 0:
        parser.error("--lr must be positive.")
    if args.init_energy <= 0:
        parser.error("--init-energy must be positive.")
    if not 0.0 < args.edge_fraction < 1.0:
        parser.error("--edge-fraction must be between 0 and 1.")

    return args


if __name__ == "__main__":
    train(parse_args())
