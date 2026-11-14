"""Round-based WSN lifetime simulation using ns-3 Wi-Fi and TDMA slots."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import gc
gc.disable()

# Every ns-3 object created from Python is parked here so cppyy never runs a
# C++ destructor on it while ns-3 may still reference it (use-after-free /
# double-free that corrupted the heap).
_KEEP = []


def keep(obj):
    _KEEP.append(obj)
    try:
        obj.__python_owns__ = False
    except Exception:
        pass
    return obj


def mark(msg):
    if os.environ.get("WSN_DEBUG"):
        print(f"[mark] {msg}", file=sys.stderr, flush=True)

NS3_BINDINGS_DIR = os.environ.get("NS3_BINDINGS_DIR", "")
if NS3_BINDINGS_DIR and NS3_BINDINGS_DIR not in sys.path:
    sys.path.insert(0, NS3_BINDINGS_DIR)

try:
    from ns import ns
except ModuleNotFoundError:
    raise SystemExit(
        "ns-3 Python bindings not found; build ns-3 with --enable-python-bindings "
        "and set NS3_BINDINGS_DIR to its bindings/python directory"
    )

from wsn_sim import moga, wsn_dataset

import faulthandler
faulthandler.enable(all_threads=True)  # must come after the ns import (Cling installs its own handler)

ns.cppyy.cppdef("""
    #include "ns3/node.h"
    #include "ns3/packet.h"
    #include "ns3/simulator.h"
    #include "ns3/socket.h"
    #include "ns3/udp-socket-factory.h"

    #include <string>

    using namespace ns3;

    class ScheduledUdpSender
    {
    public:
        explicit ScheduledUdpSender(Ptr<Node> node)
            : m_socket(Socket::CreateSocket(node, UdpSocketFactory::GetTypeId()))
        {
        }

        void ScheduleSend(Time delay, Address destination, std::string payload)
        {
            Simulator::Schedule(delay,
                                &ScheduledUdpSender::Send,
                                this,
                                destination,
                                payload);
        }

        uint32_t GetSendFailures() const
        {
            return m_sendFailures;
        }

    private:
        void Send(Address destination, std::string payload)
        {
            Ptr<Packet> packet = Create<Packet>(
                reinterpret_cast<const uint8_t*>(payload.data()),
                static_cast<uint32_t>(payload.size()));
            if (m_socket->SendTo(packet, 0, destination) < 0)
            {
                ++m_sendFailures;
            }
        }

        Ptr<Socket> m_socket;
        uint32_t m_sendFailures{0};
    };

    // Allocated on the C++ side and intentionally never deleted: events hold a raw `this`.
    ScheduledUdpSender* MakeScheduledUdpSender(Ptr<Node> node)
    {
        return new ScheduledUdpSender(node);
    }
""")

SINK_ENERGY_J = 1e4
UDP_PORT = 9000
WIFI_DATA_RATE_BPS = 11_000_000
TDMA_GUARD_S = 0.002
RADIO_VOLTAGE_V = 3.0
TX_CURRENT_A = 0.0174
RX_CURRENT_A = 0.0197
IDLE_CURRENT_A = 1e-6


def _distance_matrix_with_sink(network) -> np.ndarray:
    points = np.vstack((network.pos, np.asarray(network.sink, dtype=float)))
    differences = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(differences, axis=-1)


def build_graph_snapshot(network, alive_idx, seed, round_index):
    """Capture the pre-round graph state for one simulation round."""
    # Keep original node IDs so nodes can be identified across snapshots.
    positions = network.pos[alive_idx].copy()
    residual_energy = network.energy[alive_idx].copy()
    node_features = network.node_features[alive_idx].copy()

    # Build the graph over alive nodes using the same range as the simulator.
    all_sensor_distances = network.dist_matrix()
    distances = all_sensor_distances[np.ix_(alive_idx, alive_idx)]
    adjacency = (distances <= network.communication_range).astype(np.int8)
    np.fill_diagonal(adjacency, 0)

    # PyTorch Geometric-style edge list: row 0 = source, row 1 = destination.
    edge_index = np.asarray(np.where(adjacency == 1), dtype=np.int64)

    return {
        "seed": np.asarray(seed),
        "round_index": np.asarray(round_index),
        "node_ids": alive_idx.copy(),
        "positions": positions,
        "residual_energy": residual_energy,
        "node_features": node_features,
        "adjacency": adjacency,
        "edge_index": edge_index,
    }


def save_graph_snapshot(snapshot_data, metrics, snapshot_dir):
    """Save pre-round graph data and the completed round's metrics."""
    os.makedirs(snapshot_dir, exist_ok=True)
    seed = int(snapshot_data["seed"])
    round_index = int(snapshot_data["round_index"])
    file_path = os.path.join(
        snapshot_dir,
        f"seed_{seed}_round_{round_index:05d}.npz"
    )
    np.savez_compressed(file_path, **snapshot_data, **metrics)


def _frame_airtime(payload_size: int) -> float:
    frames = max(1, math.ceil(payload_size / 1400))
    return (payload_size + frames * 256) * 8 / WIFI_DATA_RATE_BPS


def _slot_duration(payload_size: int) -> float:
    frames = max(1, math.ceil(payload_size / 1400))
    return _frame_airtime(payload_size) + frames * TDMA_GUARD_S

def handle_watch(network, round_index):
    """Placeholder for the 'watch' action. Not implemented yet."""
    return

def _make_optimizer(method: str, seed: int):
    if method == "moga":
        return lambda network, rng: moga.optimize_cluster_heads(network, rng=rng)
    if method in ("graphsage", "predictive"):
        from wsn_sim.graphsage_embed import make_embedder

        embedder = make_embedder(seed=seed)
        return lambda network, rng: moga.optimize_cluster_heads(
            network, rng=rng, embed_map=embedder(network)
        )
    raise ValueError(f"unknown method: {method}")


def _sensor_record(network, node_id: int, degree: int, temperature_index: int,
                   humidity_index: int, current_energy: float) -> dict:
    features = network.node_features[node_id]
    return {
        "node_id": node_id,
        "temperature": float(features[temperature_index]),
        "humidity": float(features[humidity_index]),
        "residual_energy_j": float(current_energy),
        "degree": degree,
    }


def run_lifetime(
    dataset_csv: str,
    seed: int,
    rounds: int,
    method: str,
    snapshot_dir: str | None = None,
) -> dict:
    if rounds < 1:
        raise ValueError("rounds must be positive")
    network = wsn_dataset.build_network(csv_path=dataset_csv, seed=seed)
    if network.node_features is None:
        raise ValueError("the WSN dataset must provide temperature and humidity features")
    temperature_index = wsn_dataset.FEATURE_COLS.index("Temperature")
    humidity_index = wsn_dataset.FEATURE_COLS.index("Humidity")
    if not np.isfinite(network.pos).all():
        raise ValueError("the WSN dataset contains non-finite node coordinates")
    if not np.isfinite(network.node_features).all():
        raise ValueError("the WSN dataset contains non-finite sensor features")
    if not np.isfinite(network.energy).all() or np.any(network.energy <= 0):
        raise ValueError("the WSN dataset must provide positive finite initial energy")
    # Use the same communication range as GraphSAGE
    communication_range = network.communication_range

    # Calculate pairwise distances between sensor nodes
    sensor_distances = network.dist_matrix()
    radio_distances = _distance_matrix_with_sink(network)

    # Calculate node degrees using the same communication range
    node_degrees = (
        sensor_distances <= communication_range
    ).sum(axis=1) - 1
    
    optimizer = _make_optimizer(method, seed)
    rng = np.random.default_rng(seed)
    
    predictor = None
    embedder = None
    recluster_count = 0
    if method == "predictive":
        from wsn_sim import predictive_model
        from wsn_sim.graphsage_embed import make_embedder
        predictor = predictive_model.load_predictor()
        embedder = make_embedder(seed=seed)

    mark("setup start")
    nodes = keep(ns.NodeContainer())
    nodes.Create(network.n_nodes + 1)
    positions = keep(ns.ListPositionAllocator())
    for x, y in network.pos:
        positions.Add(keep(ns.Vector(float(x), float(y), 0.0)))
    positions.Add(keep(ns.Vector(float(network.sink[0]), float(network.sink[1]), 0.0)))

    mobility = keep(ns.MobilityHelper())
    mobility.SetPositionAllocator(positions)
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
    mobility.Install(nodes)

    wifi = keep(ns.WifiHelper())
    wifi.SetStandard(ns.WIFI_STANDARD_80211b)
    wifi.SetRemoteStationManager(
        "ns3::ConstantRateWifiManager",
        "DataMode",
        keep(ns.StringValue("DsssRate11Mbps")),
        "ControlMode",
        keep(ns.StringValue("DsssRate1Mbps")),
    )
    phy = keep(ns.YansWifiPhyHelper())
    channel = keep(ns.YansWifiChannelHelper())
    channel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel")
    channel.AddPropagationLoss(
        "ns3::RangePropagationLossModel",
        "MaxRange",
        keep(ns.DoubleValue(communication_range)),
    )
    phy.SetChannel(keep(channel.Create()))
    mac = keep(ns.WifiMacHelper())
    mac.SetType("ns3::AdhocWifiMac")
    devices = keep(wifi.Install(phy, mac, nodes))
    mark("wifi installed")

    stack = keep(ns.InternetStackHelper())
    stack.Install(nodes)
    address = keep(ns.Ipv4AddressHelper())
    address.SetBase(keep(ns.Ipv4Address("10.1.0.0")), keep(ns.Ipv4Mask("255.255.0.0")))
    interfaces = keep(address.Assign(devices))
    mark("internet stack installed")

    # Fetch node / device / address handles once and reuse them.
    node_ptrs = [keep(nodes.Get(i)) for i in range(network.n_nodes + 1)]
    dest_addresses = [
        keep(keep(ns.InetSocketAddress(keep(interfaces.GetAddress(i)), UDP_PORT)).ConvertTo())
        for i in range(network.n_nodes + 1)
    ]

    receive_sockets = []
    transmit_senders = []
    any_address = keep(ns.Ipv4Address.GetAny())
    local_address = keep(keep(ns.InetSocketAddress(any_address, UDP_PORT)).ConvertTo())
    udp_tid = keep(ns.UdpSocketFactory.GetTypeId())
    for node_id in range(network.n_nodes + 1):
        receiver = keep(ns.Socket.CreateSocket(node_ptrs[node_id], udp_tid))
        if receiver.Bind(local_address) < 0:
            raise RuntimeError(f"could not bind receiver socket on node {node_id}")
        receive_sockets.append(receiver)

        sender = ns.cppyy.gbl.MakeScheduledUdpSender(node_ptrs[node_id])
        sender.__python_owns__ = False  # C++ owns it; Python must never delete it
        transmit_senders.append(keep(sender))
    mark("sockets and senders created")

    sink_energy = SINK_ENERGY_J
    round_tx_airtime = np.zeros(network.n_nodes + 1, dtype=np.float64)
    round_rx_airtime = np.zeros(network.n_nodes + 1, dtype=np.float64)
    round_live_mask = np.zeros(network.n_nodes + 1, dtype=bool)

    def remaining_energy(node_id: int) -> float:
        if node_id == network.n_nodes:
            return float(sink_energy)
        return float(network.energy[node_id])

    def schedule_packet(
        src: int,
        dst: int,
        payload_factory,
        offset: float,
        size_overhead: int,
    ) -> int:
        payload = payload_factory()
        payload_text = payload.decode("utf-8")
        packet_size = len(payload) + size_overhead
        airtime = _frame_airtime(packet_size)

        round_tx_airtime[src] += airtime
        in_range = np.flatnonzero(
            radio_distances[src] <= communication_range
        )
        for receiver in in_range:
            if receiver != src and round_live_mask[receiver]:
                round_rx_airtime[receiver] += airtime

        transmit_senders[src].ScheduleSend(
            keep(ns.Seconds(offset)), dest_addresses[dst], payload_text
        )
        return len(payload)

    def debit_round_energy(round_duration: float) -> None:
        nonlocal sink_energy

        tx_power = TX_CURRENT_A * RADIO_VOLTAGE_V
        rx_power = RX_CURRENT_A * RADIO_VOLTAGE_V
        idle_power = IDLE_CURRENT_A * RADIO_VOLTAGE_V
        active_time = round_tx_airtime + round_rx_airtime
        if np.any(active_time > round_duration + 1e-9):
            raise RuntimeError(
                "Per-node radio airtime exceeds the TDMA round duration."
            )

        energy_used = (
            round_duration * idle_power
            + round_tx_airtime * (tx_power - idle_power)
            + round_rx_airtime * (rx_power - idle_power)
        )
        live_sensors = round_live_mask[:network.n_nodes]
        network.energy[live_sensors] = np.maximum(
            0.0,
            network.energy[live_sensors] - energy_used[:network.n_nodes][live_sensors],
        )
        if round_live_mask[network.n_nodes]:
            sink_energy = max(
                0.0,
                sink_energy - float(energy_used[network.n_nodes]),
            )
        network.alive[:] = network.energy > 0

    def drain_packets() -> list[int]:
        received = []
        for node_id, receiver in enumerate(receive_sockets):
            count = 0
            while receiver.GetRxAvailable() > 0:
                keep(receiver.Recv())
                count += 1
            received.append(count)
        return received

    history = {
        "alive": [],
        "avg_residual": [],
        "member_packets_delivered": [],
        "ch_packets_delivered": [],
        "delivery_rate": [],
        "round_time_s": [],
        "pareto_front_size": [],
        "mutation_rate": [],
        "predicted_state": [],
        "reclustered": [],
    }
    initial_alive = network.n_nodes
    first_death_round = -1
    final_optimization = None
    start_time = time.time()
    optimization = optimizer(network, rng)
    mark("setup done")

    try:
        for round_index in range(rounds):
            network.alive[:] = network.energy > 0
            alive_idx = network.alive_idx()
            if len(alive_idx) == 0:
                break
            round_tx_airtime.fill(0.0)
            round_rx_airtime.fill(0.0)
            round_live_mask[:network.n_nodes] = network.alive
            round_live_mask[network.n_nodes] = True

            snapshot_data = (
                build_graph_snapshot(network, alive_idx, seed, round_index)
                if snapshot_dir is not None
                else None
            )

            if predictor is not None:
                pre_round_embeddings = embedder(network)
                pre_round_energy = network.energy[alive_idx].copy()

            chs = np.asarray(optimization.selected, dtype=int)
            chs = chs[np.isin(chs, alive_idx)]
            if len(chs) == 0:
                optimization = optimizer(network, rng)
                chs = np.asarray(optimization.selected, dtype=int)
            ch_energy_before = network.energy[chs].copy()
            ch_set = set(int(ch) for ch in chs)
            members = np.setdiff1d(alive_idx, chs)

            if len(members):
                if network.cost_matrix is not None:
                    distances = network.cost_matrix[np.ix_(members, chs)]
                else:
                    distances = np.linalg.norm(
                        network.pos[members][:, None, :] - network.pos[chs][None, :, :],
                        axis=-1,
                    )
                assigned = {int(ch): [] for ch in chs}
                for member, choice in zip(members, distances.argmin(axis=1)):
                    assigned[int(chs[choice])].append(int(member))
            else:
                assigned = {int(ch): [] for ch in chs}

            slot_offset = 0.0
            expected_member_packets = 0
            for ch in sorted(ch_set):
                for member in assigned[ch]:
                    expected_member_packets += 1

                    def member_payload(node_id=member):
                        record = _sensor_record(
                            network,
                            node_id,
                            int(node_degrees[node_id]),
                            temperature_index,
                            humidity_index,
                            remaining_energy(node_id),
                        )
                        return json.dumps(
                            record, separators=(",", ":"), allow_nan=False
                        ).encode("utf-8")

                    payload_size = schedule_packet(
                        member, ch, member_payload, slot_offset, 32
                    )
                    slot_offset += _slot_duration(payload_size + 32)

            expected_ch_packets = 0
            for ch in sorted(ch_set):
                expected_ch_packets += 1
                cluster_nodes = [ch] + assigned[ch]

                def aggregate_payload(ch_id=ch, node_ids=cluster_nodes):
                    records = [
                        _sensor_record(
                            network,
                            node_id,
                            int(node_degrees[node_id]),
                            temperature_index,
                            humidity_index,
                            remaining_energy(node_id),
                        )
                        for node_id in node_ids
                    ]
                    return json.dumps(
                        {"cluster_head": ch_id, "round": round_index, "nodes": records},
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")

                payload_size = schedule_packet(
                    ch, network.n_nodes, aggregate_payload, slot_offset,
                    len(cluster_nodes) * 16,
                )
                slot_offset += _slot_duration(
                    payload_size + len(cluster_nodes) * 16
                )

            round_duration = max(slot_offset, 0.01)
            mark(f"round {round_index} before Run")
            ns.Simulator.Stop(keep(ns.Seconds(round_duration)))
            ns.Simulator.Run()
            mark(f"round {round_index} after Run")
            send_failures = sum(
                sender.GetSendFailures() for sender in transmit_senders
            )
            if send_failures:
                raise RuntimeError(f"ns-3 reported {send_failures} UDP send failures")

            received = drain_packets()
            member_delivered = sum(received[ch] for ch in ch_set)
            ch_delivered = received[network.n_nodes]
            expected_packets = expected_member_packets + expected_ch_packets
            history["member_packets_delivered"].append(member_delivered)
            history["ch_packets_delivered"].append(ch_delivered)
            pdr = (
                (member_delivered + ch_delivered) / expected_packets
                if expected_packets else 1.0
            )
            history["delivery_rate"].append(pdr)
            history["round_time_s"].append(round_duration)

            debit_round_energy(round_duration)
            print(f"r{round_index} t={keep(ns.Simulator.Now()).GetSeconds():.3f}s "
                  f"sink={remaining_energy(network.n_nodes):.6f} "
                  f"n0={remaining_energy(0):.6f} n1={remaining_energy(1):.6f}", flush=True)
            network.kill_dead()
            alive_now = int(network.alive.sum())
            history["alive"].append(alive_now)
            live_energy = network.energy[network.alive]
            avg_live_residual_energy = (
                float(live_energy.mean()) if len(live_energy) else 0.0
            )
            avg_residual_energy = float(network.energy.mean())
            history["avg_residual"].append(avg_live_residual_energy)

            ch_energy_after = network.energy[chs]
            ch_energy_consumed = np.maximum(
                ch_energy_before - ch_energy_after, 0.0
            )
            ch_energy_rate = ch_energy_consumed / round_duration
            if snapshot_data is not None:
                save_graph_snapshot(
                    snapshot_data,
                    {
                        "cluster_head_ids": chs.astype(np.int64),
                        "avg_residual_energy_j": np.asarray(
                            avg_residual_energy
                        ),
                        "avg_live_residual_energy_j": np.asarray(
                            avg_live_residual_energy
                        ),
                        "avg_ch_energy_consumption_rate_w": np.asarray(
                            float(ch_energy_rate.mean())
                        ),
                        "cluster_head_energy_consumption_rate_w": (
                            ch_energy_rate
                        ),
                        "alive_node_ratio": np.asarray(
                            alive_now / initial_alive
                        ),
                        "avg_ch_residual_energy_j": np.asarray(
                            float(ch_energy_after.mean())
                        ),
                        "min_ch_residual_energy_j": np.asarray(
                            float(ch_energy_after.min())
                        ),
                        "pdr": np.asarray(pdr),
                    },
                    snapshot_dir,
                )

            history["pareto_front_size"].append(len(optimization.front))
            history["mutation_rate"].append(optimization.final_mutation_rate)
            final_optimization = optimization

            if first_death_round == -1 and alive_now < initial_alive:
                first_death_round = round_index

            completed = round_index + 1
            progress_interval = max(1, rounds // 100)
            if completed % progress_interval == 0 or completed == rounds:
                fraction = completed / rounds
                filled = int(30 * fraction)
                print(
                    f"\r{method} [{('#' * filled) + ('-' * (30 - filled))}] "
                    f"{fraction:6.1%} ({completed}/{rounds})",
                    end="",
                    flush=True,
                )

            if round_index + 1 < rounds and alive_now:
                if predictor is None:
                    optimization = optimizer(network, rng)
                else:
                    state, probabilities = predictive_model.predict_state(
                        embeddings=pre_round_embeddings,
                        pre_round_energy=pre_round_energy,
                        metrics={
                            "avg_residual_energy": avg_live_residual_energy,
                            "energy_consumption_rate": float(ch_energy_rate.mean()),
                            "alive_node_ratio": alive_now / initial_alive,
                            "avg_ch_residual_energy": float(ch_energy_after.mean()),
                            "min_ch_residual_energy": float(ch_energy_after.min()),
                            "pdr": pdr,
                        },
                    )
                    ch_died = bool(np.any(ch_energy_after <= 0))
                    do_recluster = state == "recluster" or ch_died
                    history["predicted_state"].append(state)
                    history["reclustered"].append(do_recluster)
                    if do_recluster:
                        optimization = optimizer(network, rng)
                        recluster_count += 1
                    elif state == "watch":
                        handle_watch(network, round_index)

        print(flush=True)
    finally:
        pass  # skipped: Destroy() corrupts the heap under cppyy; process ends via os._exit

    return {
        "seed": seed,
        "method": method,
        "rounds_requested": rounds,
        "rounds_completed": len(history["alive"]),
        "n_nodes": network.n_nodes,
        "communication_range": communication_range,
        "final_alive": history["alive"][-1] if history["alive"] else initial_alive,
        "final_avg_residual": history["avg_residual"][-1] if history["avg_residual"] else 0.0,
        "final_mutation_rate": (
            history["mutation_rate"][-1] if history["mutation_rate"] else None
        ),
        "first_death_round": first_death_round,
        "recluster_count": recluster_count,
        "wall_time_s": time.time() - start_time,
        "trajectory": history,
        "final_pareto_front": (
            [front.astype(int).tolist() for front in final_optimization.front]
            if final_optimization is not None else []
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-csv", default=wsn_dataset.DEFAULT_CSV)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=3000)
    parser.add_argument("--method", choices=("moga", "graphsage", "predictive"), default="moga")
    parser.add_argument("--output", required=True)
    # Optional: when supplied, save one graph snapshot per simulated round.
    parser.add_argument("--snapshot-dir", default=None)
    args = parser.parse_args()

    result = run_lifetime(
        args.dataset_csv,
        args.seed,
        args.rounds,
        args.method,
        snapshot_dir=args.snapshot_dir,
    )
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(result, output, allow_nan=False)
    print(f"saved {args.method} lifetime results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
