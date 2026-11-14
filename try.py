import numpy as np
from wsn_sim.wsn_dataset import build_network

# Load the same dataset and node sample used in your project
network = build_network(
    csv_path="/home/devika/Desktop/WSN/WSN_Dataset.csv",
    n_nodes=100,
    seed=1
)

# Pairwise sensor distances
distances = network.dist_matrix().copy()
np.fill_diagonal(distances, np.inf)

# Distance from every sensor to Base Station
sink_distances = network.dist_to_sink(np.arange(network.n_nodes))

# Test different ranges based on actual distances
nearest_distances = distances.min(axis=1)

candidates = np.unique(
    np.percentile(nearest_distances, [50, 60, 70, 80, 90, 95, 100])
)

print("Number of nodes:", network.n_nodes)
print("Deployment area:", network.area)
print("\nRange analysis:\n")

for R in candidates:

    adjacency = (distances <= R)

    degrees = adjacency.sum(axis=1)

    isolated = np.sum(degrees == 0)
    connected = np.sum(degrees > 0)

    sink_reachable = np.sum(sink_distances <= R)

    print(f"Range: {R:.2f}")
    print(f"Average degree: {degrees.mean():.2f}")
    print(f"Isolated nodes: {isolated}")
    print(f"Nodes with neighbours: {connected}/{network.n_nodes}")
    print(f"Nodes in direct BS range: {sink_reachable}/{network.n_nodes}")
    print("-" * 35)


# Test multi-hop connectivity at the selected range
R = 12.95

distances = network.dist_matrix().copy()
adjacency = distances <= R
np.fill_diagonal(adjacency, False)

# Add Base Station as an additional graph node
sink_distances = network.dist_to_sink(np.arange(network.n_nodes))
sink_connections = sink_distances <= R

N = network.n_nodes
visited = set()
stack = [N]  # Base Station index
visited.add(N)

while stack:
    current = stack.pop()

    if current == N:
        neighbours = np.where(sink_connections)[0]
    else:
        neighbours = np.where(adjacency[current])[0]

    for neighbour in neighbours:
        if neighbour not in visited:
            visited.add(int(neighbour))
            stack.append(int(neighbour))

reachable = len(visited) - 1

print("\nMulti-hop connectivity:")
print("Nodes that can reach BS:", reachable, "/", N)
print("Nodes that cannot reach BS:", N - reachable, "/", N)


# Find suitable transmission range through multi-hop testing

distances = network.dist_matrix().copy()
np.fill_diagonal(distances, np.inf)

sink_distances = network.dist_to_sink(
    np.arange(network.n_nodes)
)

N = network.n_nodes

print("\nTesting larger transmission ranges:\n")

for R in np.arange(12.95, 40.01, 1.0):

    adjacency = distances <= R
    degrees = adjacency.sum(axis=1)

    # BFS starting from Base Station
    visited = set()
    stack = [N]
    visited.add(N)

    while stack:
        current = stack.pop()

        if current == N:
            neighbours = np.where(sink_distances <= R)[0]
        else:
            neighbours = np.where(adjacency[current])[0]

        for node in neighbours:
            node = int(node)

            if node not in visited:
                visited.add(node)
                stack.append(node)

    reachable = len(visited) - 1

    print(
        f"Range: {R:.2f} | "
        f"Reachable: {reachable}/{N} | "
        f"Avg degree: {degrees.mean():.2f} | "
        f"Isolated: {np.sum(degrees == 0)}"
    )

    if reachable == N:
        print("\nAll nodes can reach the Base Station!")
        print("Minimum tested range for full connectivity:", R)
        break


