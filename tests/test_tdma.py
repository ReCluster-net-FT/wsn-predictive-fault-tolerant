"""CH failure path of the BS-assigned TDMA in ns3_wsn_sim.py.

A cluster head is given almost no battery. Its members must miss the beacon
(or the MAC ACK), send their data directly to the BS, and the BS must then
trigger a re-cluster with reason "ch_failure".

Needs ns-3 (NS3_BINDINGS_DIR) and the dataset (WSN_DATASET_CSV). Each
scenario runs in its own process because ns-3 can only be set up once.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_scenario(scenario: str, method: str, out_path: str) -> None:
    """Run 3 rounds with the first selected CH drained of energy."""
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import ns3_wsn_sim as sim

    real_optimize = sim.moga.optimize_cluster_heads
    info = {}

    def optimize_with_weak_ch(network, **kwargs):
        result = real_optimize(network, **kwargs)
        if info:
            return result
        weak = int(min(result.selected))
        chs = sorted(int(c) for c in result.selected)
        members = np.setdiff1d(network.alive_idx(), chs)
        nearest = np.linalg.norm(
            network.pos[members][:, None, :] - network.pos[chs][None, :, :], axis=-1
        ).argmin(axis=1)
        schedule = {str(ch): [] for ch in chs}
        for m, c in zip(members, nearest):
            schedule[str(chs[c])].append(int(m))
        cluster = schedule[str(weak)]
        idle = sim.IDLE_CURRENT_A * sim.RADIO_VOLTAGE_V
        tx = sim.TX_CURRENT_A * sim.RADIO_VOLTAGE_V - idle
        rx = sim.RX_CURRENT_A * sim.RADIO_VOLTAGE_V - idle
        # Every node first receives the BS schedule broadcast.
        broadcast = len(json.dumps({"round": 0, "schedule": schedule},
                                   separators=(",", ":")).encode()) + 32
        beacon = len(json.dumps({"ch": weak, "round": 0}).encode()) + 32
        energy = sim._frame_airtime(broadcast) * rx * 1.01  # no beacon possible
        if scenario == "mid_frame":  # beacon + one member packet, then dies
            energy += sim._frame_airtime(beacon) * tx + 1.5 * sim._frame_airtime(200) * rx
        network.energy[weak] = energy
        info.update(weak=weak, cluster_size=len(cluster))
        return result

    sim.moga.optimize_cluster_heads = optimize_with_weak_ch
    result = sim.run_lifetime(os.environ["WSN_DATASET_CSV"], 1, 3, method)
    result["test_info"] = info
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle)


@unittest.skipUnless(
    os.environ.get("NS3_BINDINGS_DIR") and os.environ.get("WSN_DATASET_CSV"),
    "needs NS3_BINDINGS_DIR and WSN_DATASET_CSV",
)
class CHFailureTests(unittest.TestCase):
    def run_case(self, scenario, method):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "result.json")
            subprocess.run(
                [sys.executable, __file__, scenario, method, out],
                check=True, capture_output=True, cwd=ROOT,
            )
            with open(out, encoding="utf-8") as handle:
                return json.load(handle)

    def check_failover(self, result, orphaned):
        t = result["trajectory"]
        self.assertGreater(orphaned, 0)
        self.assertEqual(t["orphaned_members"][0], orphaned)
        # Every orphaned member reached the BS directly ...
        self.assertEqual(t["direct_to_bs_packets"][0], orphaned)
        # ... the dead CH's aggregate is missing ...
        self.assertEqual(t["ch_packets_delivered"][0], 9)
        # ... and the BS re-clusters because of it.
        self.assertEqual(t["recluster_reason"][0], "ch_failure")
        self.assertEqual(t["orphaned_members"][1], 0)
        self.assertEqual(t["direct_to_bs_packets"][1], 0)

    def test_dead_ch_overrides_predictor(self):
        result = self.run_case("dead", "predictive")
        info = result["test_info"]
        self.check_failover(result, info["cluster_size"])
        self.assertEqual(result["recluster_reasons"].get("ch_failure"), 1)

    def test_ch_dies_mid_frame(self):
        result = self.run_case("mid_frame", "moga")
        info = result["test_info"]
        self.assertGreaterEqual(info["cluster_size"], 2)
        self.check_failover(result, info["cluster_size"] - 1)


if __name__ == "__main__" and len(sys.argv) == 4:
    run_scenario(*sys.argv[1:])
    sys.stdout.flush()
    os._exit(0)  # skip ns-3/cppyy teardown, see ns3_wsn_sim.py
elif __name__ == "__main__":
    unittest.main()
