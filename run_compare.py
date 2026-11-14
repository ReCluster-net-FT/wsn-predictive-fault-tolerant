"""Compare MOGA and MOGA+GraphSAGE with ns-3 TDMA lifetime simulation."""
import json
import os
import subprocess
import sys
import tempfile
import time

N_NODES = 100
ROUNDS = 30000
SEEDS = [1, 2, 3, 4, 5]
RESULTS_PATH = "multi_seed_wsn_dataset_results.json"
SIMULATOR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ns3_wsn_sim.py")
METHODS = (
    ("moga", "moga", "MOGA+TDMA"),
    ("graphsage", "moga_graphsage", "MOGA+GraphSAGE+TDMA"),
    ("predictive", "moga_predictive", "MOGA+GraphSAGE+TDMA+XGBoost gate")
)


def _run_method(seed: int, method: str, output_path: str) -> dict:
    command = [
        sys.executable,
        SIMULATOR_PATH,
        "--seed",
        str(seed),
        "--rounds",
        str(ROUNDS),
        "--method",
        method,
        "--output",
        output_path,
    ]
    dataset_csv = os.environ.get("WSN_DATASET_CSV")
    if dataset_csv:
        command.extend(["--dataset-csv", dataset_csv])
    subprocess.run(command, check=True)
    with open(output_path, encoding="utf-8") as result_file:
        return json.load(result_file)


def run_one_seed(seed: int) -> dict:
    started = time.time()
    result = {"seed": seed}
    with tempfile.TemporaryDirectory(prefix="wsn_compare_") as temp_dir:
        for method, result_key, label in METHODS:
            output_path = os.path.join(temp_dir, f"{method}_{seed}.json")
            print(f"seed {seed}: running {label}", flush=True)
            result[result_key] = _run_method(seed, method, output_path)

    result["wall_time_s"] = time.time() - started
    return result


if __name__ == "__main__":
    results = []
    started = time.time()
    for seed_index, seed in enumerate(SEEDS, start=1):
        print(f"starting seed {seed_index}/{len(SEEDS)}", flush=True)
        result = run_one_seed(seed)
        results.append(result)
        print(
            f"seed {seed} done in {result['wall_time_s']:.1f}s "
            f"(moga alive={result['moga']['final_alive']}, "
            f"moga_graphsage alive={result['moga_graphsage']['final_alive']})",
            flush=True,
        )

    with open(RESULTS_PATH, "w", encoding="utf-8") as result_file:
        json.dump(
            {"rounds": ROUNDS, "seeds": SEEDS, "n_nodes": N_NODES, "results": results},
            result_file,
        )
    print(
        f"all {len(SEEDS)} seeds done in {time.time() - started:.1f}s, "
        f"saved {RESULTS_PATH}",
        flush=True,
    )
