"""Multi-objective GA for cluster-head selection."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MOGAResult:
    """The final non-dominated CH sets and the one selected for deployment."""

    front: list[np.ndarray]
    selected: np.ndarray
    generations_run: int
    final_mutation_rate: float


def _cluster_cost(chs: np.ndarray, network, alive_idx: np.ndarray,
                  embed_map: dict | None = None) -> tuple[float, float, float]:
    cm_idx = np.setdiff1d(alive_idx, chs)
    if embed_map is not None:
        ch_vec = np.array([embed_map[int(c)] for c in chs])
        if len(cm_idx):
            cm_vec = np.array([embed_map[int(c)] for c in cm_idx])
            d = np.linalg.norm(cm_vec[:, None, :] - ch_vec[None, :, :], axis=-1)
            intra = float(d.min(axis=1).sum())
        else:
            intra = 0.0
    elif len(cm_idx):
        if network.cost_matrix is not None:
            d = network.cost_matrix[np.ix_(cm_idx, chs)]
        else:
            d = np.linalg.norm(
                network.pos[cm_idx][:, None, :] - network.pos[chs][None, :, :],
                axis=-1,
            )
        intra = float(d.min(axis=1).sum())
    else:
        intra = 0.0

    sink_distance = float(network.dist_to_sink(chs).sum())
    residual_energy = float(network.energy[chs].sum())
    return intra, sink_distance, -residual_energy


def _constraint_violation(chs: np.ndarray, network, alive_idx: np.ndarray) -> float:
    if len(chs) == 0 or np.any(~np.isin(chs, alive_idx)):
        return float("inf")
    if np.any(network.energy[chs] <= 0):
        return float("inf")

    initial_energy = getattr(network, "initial_energy", None)
    if initial_energy is None:
        initial_energy = np.full(network.n_nodes, network.init_energy)
    energy_overage = np.maximum(network.energy[chs] - initial_energy[chs], 0.0)
    violation = float(np.sum(energy_overage / np.maximum(initial_energy[chs], 1e-12)))

    cm_idx = np.setdiff1d(alive_idx, chs)
    if not len(cm_idx):
        return violation

    if network.cost_matrix is not None:
        distances = network.cost_matrix[np.ix_(cm_idx, chs)]
    else:
        distances = np.linalg.norm(
            network.pos[cm_idx][:, None, :] - network.pos[chs][None, :, :],
            axis=-1,
        )
    nearest = distances.argmin(axis=1)
    sink_distances = network.dist_to_sink(np.arange(network.n_nodes))
    for cluster_index, ch in enumerate(chs):
        members = cm_idx[nearest == cluster_index]
        if not len(members):
            continue

        farthest_member_sink_dist = float(sink_distances[members].max())
        ch_sink_dist = float(sink_distances[ch])
        violation += max(ch_sink_dist - farthest_member_sink_dist, 0.0) / max(
            farthest_member_sink_dist, 1e-12
        )

        cluster_nodes = np.append(members, ch)
        if network.cost_matrix is not None:
            cluster_distances = network.cost_matrix[np.ix_(cluster_nodes, cluster_nodes)]
            member_distances = network.cost_matrix[members, ch]
        else:
            points = network.pos[cluster_nodes]
            cluster_distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
            member_distances = np.linalg.norm(network.pos[members] - network.pos[ch], axis=1)
        diameter = float(cluster_distances.max())
        violation += float(
            np.maximum(member_distances - diameter, 0.0).sum() / max(diameter, 1e-12)
        )

    return violation


def _nondominated_sort(objectives: np.ndarray, violations: np.ndarray) -> list[list[int]]:
    feasible = violations <= 1e-12
    no_worse = np.all(objectives[:, None, :] <= objectives[None, :, :], axis=2)
    strictly_better = np.any(objectives[:, None, :] < objectives[None, :, :], axis=2)
    pareto_dominates = no_worse & strictly_better
    both_infeasible = ~feasible[:, None] & ~feasible[None, :]
    feasible_dominates = feasible[:, None] & ~feasible[None, :]
    lower_violation = violations[:, None] < violations[None, :] - 1e-12
    equal_violation = np.abs(violations[:, None] - violations[None, :]) <= 1e-12
    dominates = (
        feasible_dominates
        | (both_infeasible & lower_violation)
        | ((feasible[:, None] & feasible[None, :] | (both_infeasible & equal_violation))
           & pareto_dominates)
    )
    dominated_count = dominates.sum(axis=0)
    fronts = []
    current = np.flatnonzero(dominated_count == 0).tolist()
    while current:
        fronts.append(current)
        following = []
        for i in current:
            for j in np.flatnonzero(dominates[i]):
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    following.append(int(j))
        current = following
    return fronts


def _crowding_distance(front: list[int], objectives: np.ndarray) -> dict[int, float]:
    distance = {i: 0.0 for i in front}
    if len(front) <= 2:
        return {i: float("inf") for i in front}

    for objective_index in range(objectives.shape[1]):
        ordered = sorted(front, key=lambda i: objectives[i, objective_index])
        low = objectives[ordered[0], objective_index]
        high = objectives[ordered[-1], objective_index]
        distance[ordered[0]] = distance[ordered[-1]] = float("inf")
        if high == low:
            continue
        for position in range(1, len(ordered) - 1):
            previous_value = objectives[ordered[position - 1], objective_index]
            next_value = objectives[ordered[position + 1], objective_index]
            distance[ordered[position]] += (next_value - previous_value) / (high - low)
    return distance


def _select_by_fronts(
    population: list[np.ndarray],
    objectives: np.ndarray,
    violations: np.ndarray,
    limit: int,
    scalar_fitness: np.ndarray,
) -> list[np.ndarray]:
    selected = []
    for front in _nondominated_sort(objectives, violations):
        remaining = limit - len(selected)
        if remaining <= 0:
            break
        if len(front) <= remaining:
            selected.extend(front)
            continue
        crowding = _crowding_distance(front, objectives)
        chosen = sorted(front, key=lambda i: (-crowding[i], scalar_fitness[i]))
        selected.extend(chosen[:remaining])
        break
    return [population[i] for i in selected]


def _normalize(objectives: np.ndarray, lower: np.ndarray,
               upper: np.ndarray) -> np.ndarray:
    span = upper - lower
    span[span == 0] = 1.0
    return (objectives - lower) / span


def optimize_cluster_heads(
    network,
    pop_size: int = 100,
    generations: int = 30,
    mutation_rate: float = 0.2,
    crossover_rate: float = 0.8,
    stagnation_generations: int = 10,
    mutation_rate_cap: float = 0.9,
    rng: np.random.Generator | None = None,
    embed_map: dict | None = None,
) -> MOGAResult:
    """Return the final Pareto front and the Eq. 1 solution selected for use.

    Objective normalization is fixed from the initial population so generation
    fitness remains comparable while the mutation rate adapts. On stagnation,
    mutation grows by 1.5x per window, capped at ``mutation_rate_cap``.
    """
    rng = rng or np.random.default_rng()
    alive_idx = network.alive_idx()
    alive_idx = alive_idx[network.energy[alive_idx] > 0]
    if len(alive_idx) == 0:
        raise ValueError("MOGA requires at least one live node")
    if pop_size < 2:
        raise ValueError("pop_size must be at least 2")
    if generations < 1:
        raise ValueError("generations must be positive")
    if stagnation_generations < 1:
        raise ValueError("stagnation_generations must be positive")
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError("mutation_rate must be between 0 and 1")
    if not 0.0 <= crossover_rate <= 1.0:
        raise ValueError("crossover_rate must be between 0 and 1")
    if not 0.0 <= mutation_rate_cap <= 1.0:
        raise ValueError("mutation_rate_cap must be between 0 and 1")

    k = min(len(alive_idx), max(1, round(network.ch_prob * len(alive_idx))))
    population = [rng.choice(alive_idx, size=k, replace=False)]
    population.extend(
        rng.choice(alive_idx, size=k, replace=False)
        for _ in range(pop_size - 1)
    )

    if network.cost_matrix is not None:
        physical_distances = network.cost_matrix[np.ix_(alive_idx, alive_idx)]
    else:
        live_positions = network.pos[alive_idx]
        physical_distances = np.linalg.norm(
            live_positions[:, None, :] - live_positions[None, :, :], axis=-1
        )
    if embed_map is None:
        objective_distances = physical_distances
    else:
        live_embeddings = np.array([embed_map[int(i)] for i in alive_idx])
        objective_distances = np.linalg.norm(
            live_embeddings[:, None, :] - live_embeddings[None, :, :], axis=-1
        )
    live_sink_distances = network.dist_to_sink(alive_idx)
    initial_energy = getattr(network, "initial_energy", None)
    if initial_energy is None:
        initial_energy = np.full(network.n_nodes, network.init_energy)

    def evaluate(candidates: list[np.ndarray]):
        candidate_nodes = np.asarray(candidates, dtype=int)
        local_candidates = np.searchsorted(alive_idx, candidate_nodes)
        node_indices = np.arange(len(alive_idx))[None, :, None]
        objective_to_ch = objective_distances[
            node_indices, local_candidates[:, None, :]
        ]
        physical_to_ch = physical_distances[
            node_indices, local_candidates[:, None, :]
        ]
        member_mask = np.ones((len(candidate_nodes), len(alive_idx)), dtype=bool)
        member_mask[np.arange(len(candidate_nodes))[:, None], local_candidates] = False
        nearest_objective_distance = objective_to_ch.min(axis=2)
        intra = np.where(member_mask, nearest_objective_distance, 0.0).sum(axis=1)
        sink_distance = network.dist_to_sink(candidate_nodes)
        residual_energy = network.energy[candidate_nodes]
        objectives = np.column_stack((
            intra,
            sink_distance.sum(axis=1),
            -residual_energy.sum(axis=1),
        ))

        cluster_assignment = physical_to_ch.argmin(axis=2)
        cluster_index = np.arange(candidate_nodes.shape[1])[None, None, :]
        member_clusters = (
            (cluster_assignment[:, :, None] == cluster_index)
            & member_mask[:, :, None]
        )
        has_members = member_clusters.any(axis=1)
        farthest_member_sink = np.max(
            np.where(
                member_clusters,
                live_sink_distances[None, :, None],
                -np.inf,
            ),
            axis=1,
        )
        sink_violation = np.where(
            has_members,
            np.maximum(sink_distance - farthest_member_sink, 0.0)
            / np.maximum(farthest_member_sink, 1e-12),
            0.0,
        ).sum(axis=1)
        energy_overage = np.maximum(
            residual_energy - initial_energy[candidate_nodes], 0.0
        )
        energy_violation = (
            energy_overage / np.maximum(initial_energy[candidate_nodes], 1e-12)
        ).sum(axis=1)
        # The cluster diameter includes the CH, so each member-to-CH distance
        # is bounded by that diameter by construction.
        violations = sink_violation + energy_violation
        return objectives, violations

    initial_objectives, _ = evaluate(population)
    lower = initial_objectives.min(axis=0)
    upper = initial_objectives.max(axis=0)
    best_fitness = float("inf")
    stagnant = 0
    current_mutation_rate = min(mutation_rate, mutation_rate_cap)
    generations_run = 0

    for generation in range(generations):
        objectives, violations = evaluate(population)
        normalized = _normalize(objectives, lower, upper)
        scalar_fitness = normalized.sum(axis=1)
        feasible = violations <= 1e-12
        if np.any(feasible):
            generation_best = float(scalar_fitness[feasible].min())
        else:
            minimum_violation = violations.min()
            generation_best = float(
                violations.min() + scalar_fitness[violations == minimum_violation].min()
            )
        if generation_best < best_fitness - 1e-12:
            best_fitness = generation_best
            stagnant = 0
        else:
            stagnant += 1

        if stagnant >= stagnation_generations:
            current_mutation_rate = min(mutation_rate_cap, current_mutation_rate * 1.5)
            stagnant = 0

        parent_pool = _select_by_fronts(
            population, normalized, violations, max(1, pop_size // 2), scalar_fitness
        )
        children = []
        while len(children) < pop_size:
            parent_a = parent_pool[int(rng.integers(len(parent_pool)))]
            parent_b = parent_pool[int(rng.integers(len(parent_pool)))]
            if k > 1 and rng.random() < crossover_rate:
                cut = int(rng.integers(1, k))
                child = np.concatenate((parent_a[:cut], parent_b[cut:]))
            else:
                child = parent_a.copy()

            child = np.unique(child)
            while len(child) < k:
                candidate = int(rng.choice(alive_idx))
                if candidate not in child:
                    child = np.append(child, candidate)
            if rng.random() < current_mutation_rate:
                position = int(rng.integers(k))
                available = np.setdiff1d(alive_idx, child)
                if len(available):
                    child[position] = rng.choice(available)
            children.append(child.astype(int, copy=False))

        combined = population + children
        combined_objectives, combined_violations = evaluate(combined)
        combined_normalized = _normalize(combined_objectives, lower, upper)
        combined_scalar = combined_normalized.sum(axis=1)
        population = _select_by_fronts(
            combined, combined_normalized, combined_violations, pop_size, combined_scalar
        )
        generations_run = generation + 1

    final_objectives, final_violations = evaluate(population)
    final_normalized = _normalize(final_objectives, lower, upper)
    final_front_indices = _nondominated_sort(final_normalized, final_violations)[0]
    front = [population[i].copy() for i in final_front_indices]
    front_fitness = final_normalized[final_front_indices].sum(axis=1)
    selected = front[int(np.argmin(front_fitness))].copy()
    return MOGAResult(
        front=front,
        selected=selected,
        generations_run=generations_run,
        final_mutation_rate=current_mutation_rate,
    )


def select_cluster_heads(
    network,
    pop_size: int = 100,
    generations: int = 30,
    mutation_rate: float = 0.2,
    crossover_rate: float = 0.8,
    rng: np.random.Generator | None = None,
    embed_map: dict | None = None,
) -> np.ndarray:
    """Compatibility helper returning the Eq. 1-selected CH set."""
    return optimize_cluster_heads(
        network,
        pop_size=pop_size,
        generations=generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        rng=rng,
        embed_map=embed_map,
    ).selected
