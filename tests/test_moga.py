import unittest

import numpy as np

from wsn_sim import Network, moga


class MOGATests(unittest.TestCase):
    def setUp(self):
        self.network = Network(n_nodes=4, seed=1)
        self.network.pos = np.array([
            [10.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ])
        self.network.sink = (0.0, 0.0)
        self.network.energy[:] = [0.5, 0.4, 0.3, 0.2]
        self.network.initial_energy = self.network.energy.copy()
        self.network.ch_prob = 0.25

    def test_distance_constraints_are_checked_per_cluster(self):
        alive = self.network.alive_idx()

        self.assertGreater(
            moga._constraint_violation(np.array([0]), self.network, alive),
            0.0,
        )
        self.assertEqual(
            moga._constraint_violation(np.array([1]), self.network, alive),
            0.0,
        )
        self.network.energy[1] = self.network.initial_energy[1] + 0.1
        self.assertGreater(
            moga._constraint_violation(np.array([1]), self.network, alive),
            0.0,
        )

    def test_feasible_population_members_dominate_infeasible_members(self):
        objectives = np.array([
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ])
        violations = np.array([0.0, 0.0, 0.0, 1.0])

        fronts = moga._nondominated_sort(objectives, violations)

        self.assertEqual(set(fronts[0]), {0, 1})
        self.assertNotIn(3, fronts[0])

    def test_optimizer_returns_front_and_live_deployment(self):
        self.network.alive[0] = False
        self.network.energy[0] = 0.0
        result = moga.optimize_cluster_heads(
            self.network,
            pop_size=12,
            generations=4,
            rng=np.random.default_rng(4),
        )

        self.assertTrue(result.front)
        self.assertTrue(any(np.array_equal(result.selected, candidate)
                            for candidate in result.front))
        self.assertTrue(all(0 not in candidate for candidate in result.front))

    def test_mutation_rate_increases_after_stagnation(self):
        self.network.pos[:] = [1.0, 1.0]
        self.network.sink = (0.0, 0.0)
        self.network.energy[:] = 0.5
        self.network.initial_energy = self.network.energy.copy()

        result = moga.optimize_cluster_heads(
            self.network,
            pop_size=8,
            generations=3,
            mutation_rate=0.2,
            stagnation_generations=1,
            rng=np.random.default_rng(2),
        )

        self.assertAlmostEqual(result.final_mutation_rate, 0.45)

    def test_optimizer_accepts_embedding_objectives(self):
        embeddings = {
            node_id: np.array([float(node_id), float(node_id % 2)])
            for node_id in range(self.network.n_nodes)
        }

        result = moga.optimize_cluster_heads(
            self.network,
            pop_size=8,
            generations=2,
            embed_map=embeddings,
            rng=np.random.default_rng(5),
        )

        self.assertTrue(result.front)
        self.assertEqual(len(result.selected), 1)


if __name__ == "__main__":
    unittest.main()
