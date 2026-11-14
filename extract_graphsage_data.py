import os
import subprocess
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from wsn_sim import wsn_dataset

# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEEDS = [1, 2, 3, 4, 5]

ROUNDS = 3000

METHOD = "moga"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SIMULATOR_PATH = os.path.join(
    BASE_DIR,
    "ns3_wsn_sim.py"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "graphsage_training_data"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_initial_graph(seed):
    """Save the sampled sensor layout and communication links before MOGA."""
    network = wsn_dataset.build_network(seed=seed)
    distances = network.dist_matrix()
    adjacency = distances <= network.communication_range
    np.fill_diagonal(adjacency, False)
    source, target = np.where(np.triu(adjacency, k=1))

    fig, axis = plt.subplots(figsize=(9, 7))
    if len(source):
        segments = np.stack(
            (network.pos[source], network.pos[target]),
            axis=1,
        )
        axis.add_collection(
            LineCollection(segments, colors="lightgray", linewidths=0.6)
        )

    axis.scatter(
        network.pos[:, 0],
        network.pos[:, 1],
        s=24,
        color="tab:blue",
        label="Sensors",
        zorder=2,
    )
    axis.scatter(
        network.sink[0],
        network.sink[1],
        marker="*",
        s=180,
        color="tab:red",
        label="Sink",
        zorder=3,
    )
    axis.set_title(
        f"Initial sensor graph — seed {seed} "
        f"({network.n_nodes} nodes, {len(source)} links)"
    )
    axis.set_xlabel("X coordinate")
    axis.set_ylabel("Y coordinate")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, f"seed_{seed}_initial_graph.png")
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


# --------------------------------------------------
# Run MOGA simulations and extract graph snapshots
# --------------------------------------------------

for seed in SEEDS:

    print(f"\nStarting simulation for seed {seed}")
    print(f"Initial graph saved to: {plot_initial_graph(seed)}")

    snapshot_dir = os.path.join(
        OUTPUT_DIR,
        f"seed_{seed}"
    )

    result_path = os.path.join(
        OUTPUT_DIR,
        f"seed_{seed}_results.json"
    )

    command = [
        sys.executable,
        SIMULATOR_PATH,
        "--seed", str(seed),
        "--rounds", str(ROUNDS),
        "--method", METHOD,
        "--output", result_path,
        "--snapshot-dir", snapshot_dir
    ]

    subprocess.run(command, check=True)

    print(f"Completed seed {seed}")
    print(f"Snapshots saved in: {snapshot_dir}")


print("\nAll five simulations completed.")
print(f"Training data saved in: {OUTPUT_DIR}")
