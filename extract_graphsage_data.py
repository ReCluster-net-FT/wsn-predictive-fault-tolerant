import os
import subprocess
import sys

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


# --------------------------------------------------
# Run MOGA simulations and extract graph snapshots
# --------------------------------------------------

for seed in SEEDS:

    print(f"\nStarting simulation for seed {seed}")

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
