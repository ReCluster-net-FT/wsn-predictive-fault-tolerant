import json
import matplotlib.pyplot as plt

with open("graphsage_training_data/seed_1_results.json", encoding="utf-8") as f:
    result = json.load(f)

trajectory = result["trajectory"]
plots = [
    ("alive", "Alive Nodes", "Nodes"),
    ("avg_residual", "Average Residual Energy", "Energy (J)"),
    ("member_packets_delivered", "Member Packets Delivered", "Packets"),
    ("ch_packets_delivered", "CH Packets Delivered", "Packets"),
    ("delivery_rate", "Delivery Rate", "Rate"),
    ("round_time_s", "Round Duration", "Seconds"),
    ("pareto_front_size", "Pareto Front Size", "Solutions"),
    ("mutation_rate", "Mutation Rate", "Rate"),
]

fig, axes = plt.subplots(4, 2, figsize=(14, 12), constrained_layout=True)
for ax, (key, title, ylabel) in zip(axes.flat, plots):
    values = trajectory[key]
    ax.plot(range(1, len(values) + 1), values)
    ax.set(title=title, xlabel="Simulation Round", ylabel=ylabel)
    ax.grid(True, alpha=0.3)

fig.suptitle(f"Seed {result['seed']} — {result['method']}")
fig.savefig("seed_1_results.png", dpi=160)
plt.show()
