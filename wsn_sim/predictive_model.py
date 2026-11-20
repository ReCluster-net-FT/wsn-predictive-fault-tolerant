"""
Network-state classifier for the WSN + MOGA simulation.

Classes: healthy / watch / recluster.

Input  : GraphSAGE embeddings of the alive nodes (pre-round state),
         the pre-round residual energy of the alive nodes,
         and the six per-round performance metrics.
Output : (state_name, {class_name: probability})
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "training_results" / "predictive"
MODEL_PATH = MODEL_DIR / "wsn_state_model.json"
META_PATH = MODEL_DIR / "wsn_state_meta.json"

HEALTHY, WATCH, RECLUSTER = "healthy", "watch", "recluster"

METRIC_NAMES = (
    "avg_residual_energy",       # mean energy of ALIVE nodes after the round (J)
    "energy_consumption_rate",   # mean CH energy used / round duration (W)
    "alive_node_ratio",          # alive nodes after the round / initial nodes
    "avg_ch_residual_energy",    # mean CH energy after the round (J)
    "min_ch_residual_energy",    # min CH energy after the round (J)
    "pdr",                       # (member + CH packets delivered) / expected
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class StatePredictor:
    def __init__(self, model_path=MODEL_PATH, meta_path=META_PATH,
                 verify_graphsage_checkpoint=True):
        import xgboost as xgb  # imported here so the module imports without it

        model_path, meta_path = Path(model_path), Path(meta_path)
        for path in (model_path, meta_path):
            if not path.exists():
                raise FileNotFoundError(f"Predictive-model file not found: {path}")

        with open(meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)

        self.meta = meta
        self.class_names = list(meta["class_names"])
        self.feature_names = list(meta["feature_names"])
        self.metric_names = tuple(meta["metric_names"])
        self.embed_dim = int(meta["embed_dim"])
        self.pooling = list(meta["pooling"])
        self.tail_cfg = meta["energy_tail"]

        if self.class_names != [HEALTHY, WATCH, RECLUSTER]:
            raise ValueError(f"Unexpected class order: {self.class_names}")
        if self.metric_names != METRIC_NAMES:
            raise ValueError(f"Metric names differ from this file: {self.metric_names}")
        if self.pooling != ["mean", "std_ddof0", "min", "max"]:
            raise ValueError(f"Unexpected pooling: {self.pooling}")

        self.booster = xgb.Booster()
        self.booster.load_model(str(model_path))
        if self.booster.num_features() != len(self.feature_names):
            raise ValueError(
                f"Model expects {self.booster.num_features()} features, "
                f"meta lists {len(self.feature_names)}."
            )
        self._xgb = xgb

        if verify_graphsage_checkpoint:
            from .graphsage_embed import CHECKPOINT_PATH
            expected = meta["graphsage_checkpoint_sha256"]
            actual = _sha256(Path(CHECKPOINT_PATH))
            if actual != expected:
                raise ValueError(
                    "GraphSAGE checkpoint differs from the one this predictor "
                    "was trained with (SHA-256 mismatch). Retrain the predictor "
                    "or restore the original graphsage_model.pth."
                )

    def build_features(self, embeddings, pre_round_energy, metrics) -> np.ndarray:
        """Return a (1, n_features) float32 array in the training column order."""
        if isinstance(embeddings, dict):
            embeddings = [embeddings[key] for key in sorted(embeddings)]
        emb = np.asarray(embeddings, dtype=np.float64)
        if emb.ndim != 2 or emb.shape[0] == 0 or emb.shape[1] != self.embed_dim:
            raise ValueError(f"Embeddings must be (n_alive, {self.embed_dim}); got {emb.shape}")

        energy = np.asarray(pre_round_energy, dtype=np.float64).ravel()
        if energy.size == 0:
            raise ValueError("pre_round_energy is empty")

        values: dict[str, float] = {}
        pooled = {
            "mean": emb.mean(axis=0),
            "std": emb.std(axis=0, ddof=0),
            "min": emb.min(axis=0),
            "max": emb.max(axis=0),
        }
        for kind, vector in pooled.items():
            for i in range(self.embed_dim):
                values[f"emb_{kind}_{i}"] = float(vector[i])

        for name in self.metric_names:
            if name not in metrics:
                raise KeyError(f"Missing metric: {name}")
            values[name] = float(metrics[name])

        init = float(self.tail_cfg["init_energy"])
        low_k = int(self.tail_cfg["low_k"])
        ordered = np.sort(energy)
        values["tail_min_energy"] = float(ordered[0])
        values["tail_low5_mean_energy"] = float(ordered[:low_k].mean())
        for fraction in self.tail_cfg["fractions_of_init"]:
            key = f"tail_frac_below_{int(round(fraction * 100))}pct"
            values[key] = float(np.mean(energy < fraction * init))

        row = np.array([[values[name] for name in self.feature_names]], dtype=np.float32)
        if not np.isfinite(row).all():
            raise ValueError("Feature vector contains NaN or infinity.")
        return row

    def predict_proba(self, embeddings, pre_round_energy, metrics) -> np.ndarray:
        row = self.build_features(embeddings, pre_round_energy, metrics)
        matrix = self._xgb.DMatrix(row, feature_names=self.booster.feature_names)
        return np.asarray(self.booster.predict(matrix)).reshape(-1, len(self.class_names))[0]

    def predict_state(self, embeddings, pre_round_energy, metrics):
        probabilities = self.predict_proba(embeddings, pre_round_energy, metrics)
        state = self.class_names[int(np.argmax(probabilities))]
        return state, dict(zip(self.class_names, map(float, probabilities)))


_PREDICTOR: StatePredictor | None = None


def load_predictor(**kwargs) -> StatePredictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = StatePredictor(**kwargs)
    return _PREDICTOR


def predict_state(embeddings, pre_round_energy, metrics):
    """Classify the network state after a completed round.

    embeddings       dict {node_id: 16-vector} from the GraphSAGE embedder, or an
                     (n_alive, 16) array, computed on the PRE-round state.
    pre_round_energy residual energy (J) of the alive nodes, pre-round.
    metrics          dict with the six keys in METRIC_NAMES (post-round values).

    Returns (state, probabilities); state is "healthy", "watch" or "recluster".
    """
    return load_predictor().predict_state(embeddings, pre_round_energy, metrics)


if __name__ == "__main__":
    predictor = load_predictor()
    rng = np.random.default_rng(0)
    state, probabilities = predictor.predict_state(
        rng.normal(size=(100, predictor.embed_dim)),
        rng.uniform(0.3, 0.5, size=100),
        {
            "avg_residual_energy": 0.45,
            "energy_consumption_rate": 6e-4,
            "alive_node_ratio": 1.0,
            "avg_ch_residual_energy": 0.5,
            "min_ch_residual_energy": 0.4,
            "pdr": 0.35,
        },
    )
    print("features:", len(predictor.feature_names))
    print("state:", state, probabilities)