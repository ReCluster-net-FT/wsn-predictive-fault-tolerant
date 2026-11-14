"""Plot simulation result JSON files produced by ns3_wsn_sim.py."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


TRAJECTORY_PLOTS = [
    ("alive", "Alive nodes", "Nodes"),
    ("avg_residual", "Average residual energy", "Energy (J)"),
    ("member_packets_delivered", "Member packets delivered", "Packets"),
    ("ch_packets_delivered", "CH packets delivered", "Packets"),
    ("delivery_rate", "Delivery rate", "Rate"),
    ("round_time_s", "Round duration", "Seconds"),
    ("pareto_front_size", "Pareto front size", "Solutions"),
    ("mutation_rate", "Mutation rate", "Rate"),
]


def collect_result_files(inputs):
    files = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(path.glob("seed_*.json")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"Input path does not exist: {path}")

    files = list(dict.fromkeys(path.resolve() for path in files))
    if not files:
        raise FileNotFoundError(
            "No result JSON files found. Pass a result file or a directory "
            "containing seed_*.json files."
        )
    return files


def load_results(files):
    results = []
    for path in files:
        with path.open("r", encoding="utf-8") as file:
            result = json.load(file)
        if not isinstance(result.get("trajectory"), dict):
            raise ValueError(f"{path} has no 'trajectory' object.")
        label = f"seed {result.get('seed', path.stem)}"
        results.append((path, label, result))
    return results


def plot_trajectories(results, output_path):
    fig, axes = plt.subplots(4, 2, figsize=(15, 13))
    axes = axes.ravel()

    for axis, (key, title, ylabel) in zip(axes, TRAJECTORY_PLOTS):
        plotted = False
        for _, label, result in results:
            values = result["trajectory"].get(key)
            if values is None:
                continue
            rounds = range(1, len(values) + 1)
            axis.plot(rounds, values, label=label, linewidth=1.5)
            plotted = True

        axis.set_title(title)
        axis.set_xlabel("Simulation round")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        if plotted:
            axis.legend()
        else:
            axis.text(0.5, 0.5, "Not present in input", ha="center",
                      va="center", transform=axis.transAxes)

    seeds = ", ".join(label for _, label, _ in results)
    fig.suptitle(f"Simulation results: {seeds}", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_pareto_selection(results, output_path):
    fig, axis = plt.subplots(figsize=(13, 5))
    plotted = False

    for _, label, result in results:
        front = result.get("final_pareto_front")
        if not front:
            continue

        n_nodes = int(result["n_nodes"])
        counts = [0] * n_nodes
        for candidate in front:
            for node_id in candidate:
                node_id = int(node_id)
                if 0 <= node_id < n_nodes:
                    counts[node_id] += 1

        axis.plot(range(n_nodes), counts, marker=".", linewidth=1,
                  label=label)
        plotted = True

    axis.set_title("Final Pareto-front cluster-head selection frequency")
    axis.set_xlabel("Node ID")
    axis.set_ylabel("Pareto solutions selecting node")
    axis.grid(True, alpha=0.3)
    if plotted:
        axis.legend()
    else:
        axis.text(0.5, 0.5, "No final Pareto-front data in input",
                  ha="center", va="center", transform=axis.transAxes)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def print_summaries(results):
    print("Simulation summaries")
    for path, label, result in results:
        print(
            f"{label} ({path}): "
            f"method={result.get('method', 'unknown')}, "
            f"rounds={result.get('rounds_completed', 'unknown')}/"
            f"{result.get('rounds_requested', 'unknown')}, "
            f"final_alive={result.get('final_alive', 'unknown')}, "
            f"final_avg_residual={result.get('final_avg_residual', 'unknown')}, "
            f"first_death_round={result.get('first_death_round', 'unknown')}, "
            f"wall_time_s={result.get('wall_time_s', 'unknown')}"
        )


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Plot simulation trajectories and final Pareto-front results."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help=(
            "Result JSON file(s) or directory/directories. Directories are "
            "searched for seed_*.json files. With no inputs, the "
            "training_results and graphsage_training_data directories are "
            "plotted separately."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "training_results" / "simulation_results.png",
        help=(
            "Output path for the training_results dashboard. The second "
            "default dashboard uses the same path with "
            "'_graphsage_training_data' added to its filename."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    if args.inputs:
        plot_groups = [(args.inputs, args.output)]
    else:
        graphsage_output = args.output.with_name(
            f"{args.output.stem}_graphsage_training_data{args.output.suffix}"
        )
        plot_groups = [
            ([str(script_dir / "training_results")], args.output),
            ([str(script_dir / "graphsage_training_data")], graphsage_output),
        ]

    for inputs, output in plot_groups:
        results = load_results(collect_result_files(inputs))
        output_path = output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pareto_path = output_path.with_name(
            f"{output_path.stem}_pareto{output_path.suffix}"
        )

        plot_trajectories(results, output_path)
        plot_pareto_selection(results, pareto_path)
        print_summaries(results)
        print(f"\nTrajectory dashboard: {output_path}")
        print(f"Pareto selection plot: {pareto_path}")


if __name__ == "__main__":
    main()
