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
            // Schedule broadcasts (BS) and cluster beacons (CHs) are broadcasts.
            m_socket->SetAllowBroadcast(true);
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
CTRL_PORT = 9001  # control traffic: BS schedule broadcast and CH beacons
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
    if method in ("moga","predictive"):
        return lambda network, rng: moga.optimize_cluster_heads(network, rng=rng)
    if method == "graphsage":
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
    # Neighbour range used for the GraphSAGE graph and the node degree.
    communication_range = network.communication_range

    # Calculate pairwise distances between sensor nodes
    sensor_distances = network.dist_matrix()
    radio_distances = _distance_matrix_with_sink(network)
    # Dense deployment: the radio reaches every sensor, every possible CH and
    # the BS, so a member can always fall back to sending directly to the BS.
    radio_range = float(radio_distances.max()) + 1.0

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
        keep(ns.DoubleValue(radio_range)),
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
    # Static network with a BS-assigned schedule: neighbour tables are known
    # up front, so no ARP exchanges spill into other nodes' TDMA slots.
    keep(ns.NeighborCacheHelper()).PopulateNeighborCache()
    mark("internet stack installed")

    # Fetch node / device / address handles once and reuse them.
    node_ptrs = [keep(nodes.Get(i)) for i in range(network.n_nodes + 1)]
    dest_addresses = [
        keep(keep(ns.InetSocketAddress(keep(interfaces.GetAddress(i)), UDP_PORT)).ConvertTo())
        for i in range(network.n_nodes + 1)
    ]

    broadcast_address = keep(keep(ns.InetSocketAddress(
        keep(ns.Ipv4Address("10.1.255.255")), CTRL_PORT)).ConvertTo())
    # The BS identifies the sender of each data packet by its IPv4 address.
    node_by_ip = {
        interfaces.GetAddress(i).Get(): i for i in range(network.n_nodes + 1)
    }

    receive_sockets = []
    control_sockets = []
    transmit_senders = []
    any_address = keep(ns.Ipv4Address.GetAny())
    local_address = keep(keep(ns.InetSocketAddress(any_address, UDP_PORT)).ConvertTo())
    control_address = keep(keep(ns.InetSocketAddress(any_address, CTRL_PORT)).ConvertTo())
    udp_tid = keep(ns.UdpSocketFactory.GetTypeId())
    for node_id in range(network.n_nodes + 1):
        receiver = keep(ns.Socket.CreateSocket(node_ptrs[node_id], udp_tid))
        if receiver.Bind(local_address) < 0:
            raise RuntimeError(f"could not bind receiver socket on node {node_id}")
        receive_sockets.append(receiver)
        control = keep(ns.Socket.CreateSocket(node_ptrs[node_id], udp_tid))
        if control.Bind(control_address) < 0:
            raise RuntimeError(f"could not bind control socket on node {node_id}")
        control_sockets.append(control)

        sender = ns.cppyy.gbl.MakeScheduledUdpSender(node_ptrs[node_id])
        sender.__python_owns__ = False  # C++ owns it; Python must never delete it
        transmit_senders.append(keep(sender))
    mark("sockets and senders created")

    sink = network.n_nodes
    sink_energy = SINK_ENERGY_J
    tx_power = TX_CURRENT_A * RADIO_VOLTAGE_V
    rx_power = RX_CURRENT_A * RADIO_VOLTAGE_V
    idle_power = IDLE_CURRENT_A * RADIO_VOLTAGE_V

    def remaining_energy(node_id: int) -> float:
        if node_id == sink:
            return float(sink_energy)
        return float(network.energy[node_id])

    def spend(node_id: int, airtime: float, power: float) -> bool:
        """Debit one TDMA slot of radio energy. A node that cannot afford the
        slot dies at that moment, so the slot's transmit/receive fails."""
        nonlocal sink_energy
        cost = airtime * (power - idle_power)
        if node_id == sink:
            sink_energy = max(0.0, sink_energy - cost)
            return True
        if not network.alive[node_id]:
            return False
        if network.energy[node_id] < cost:
            network.energy[node_id] = 0.0
            network.alive[node_id] = False
            return False
        network.energy[node_id] -= cost
        return True

    def send_slot(src: int, dst: int | None, payload: bytes, offset: float,
                  overhead: int, listeners) -> tuple[float, bool, list[int]]:
        """One TDMA slot: `src` transmits, only the scheduled `listeners` wake
        up to receive (every other node sleeps). dst=None is a broadcast.

        Returns (slot duration, whether src transmitted, listeners that heard
        it). The slot is reserved in the schedule even if src is dead.
        """
        size = len(payload) + overhead
        airtime = _frame_airtime(size)
        duration = _slot_duration(size)
        if not spend(src, airtime, tx_power):
            return duration, False, []
        heard = [int(n) for n in listeners if spend(int(n), airtime, rx_power)]
        if dst is None or heard:
            transmit_senders[src].ScheduleSend(
                keep(ns.Seconds(offset)),
                broadcast_address if dst is None else dest_addresses[dst],
                payload.decode("utf-8"),
            )
        return duration, True, heard

    def record_payload(node_id: int) -> bytes:
        # Residual energy is piggybacked on the regular sensor data.
        record = _sensor_record(
            network, node_id, int(node_degrees[node_id]),
            temperature_index, humidity_index, remaining_energy(node_id),
        )
        return json.dumps(record, separators=(",", ":"), allow_nan=False).encode("utf-8")

    def build_schedule(selected) -> dict[int, list[int]]:
        """BS-assigned two-level TDMA schedule: CH slot order (level 2) and,
        per CH, its members' slot order (level 1). Kept until a re-cluster."""
        chs = sorted(int(ch) for ch in selected)
        members = np.setdiff1d(network.alive_idx(), chs)
        assigned = {ch: [] for ch in chs}
        if len(members) and chs:
            if network.cost_matrix is not None:
                distances = network.cost_matrix[np.ix_(members, chs)]
            else:
                distances = np.linalg.norm(
                    network.pos[members][:, None, :] - network.pos[chs][None, :, :],
                    axis=-1,
                )
            for member, choice in zip(members, distances.argmin(axis=1)):
                assigned[chs[choice]].append(int(member))
        return assigned

    def drain_packets() -> tuple[list[int], int, int]:
        """Per-node data-packet counts, plus the BS's split of what it got
        from scheduled CHs vs. from non-CH nodes (direct member traffic)."""
        received = []
        from_ch = from_non_ch = 0
        for node_id, receiver in enumerate(receive_sockets):
            count = 0
            while receiver.GetRxAvailable() > 0:
                if node_id == sink:
                    source = keep(ns.Address())
                    keep(receiver.RecvFrom(source))
                    sender_ip = ns.InetSocketAddress.ConvertFrom(source).GetIpv4().Get()
                    if node_by_ip.get(sender_ip) in schedule:
                        from_ch += 1
                    else:
                        from_non_ch += 1
                else:
                    keep(receiver.Recv())
                count += 1
            received.append(count)
        for control in control_sockets:
            while control.GetRxAvailable() > 0:
                keep(control.Recv())
        return received, from_ch, from_non_ch

    history = {
        "alive": [],
        "avg_residual": [],
        "member_packets_delivered": [],
        "ch_packets_delivered": [],
        "direct_to_bs_packets": [],
        "orphaned_members": [],
        "delivery_rate": [],
        "round_time_s": [],
        "pareto_front_size": [],
        "mutation_rate": [],
        "predicted_state": [],
        "reclustered": [],
        "recluster_reason": [],
    }
    recluster_reasons: dict[str, int] = {}
    initial_alive = network.n_nodes
    first_death_round = -1
    final_optimization = None
    start_time = time.time()
    optimization = optimizer(network, rng)
    schedule = build_schedule(optimization.selected)
    schedule_pending = True  # BS must broadcast the new schedule
    mark("setup done")

    try:
        for round_index in range(rounds):
            network.alive[:] = network.energy > 0
            alive_idx = network.alive_idx()
            if len(alive_idx) == 0:
                break

            snapshot_data = (
                build_graph_snapshot(network, alive_idx, seed, round_index)
                if snapshot_dir is not None
                else None
            )

            if predictor is not None:
                pre_round_embeddings = embedder(network)
                pre_round_energy = network.energy[alive_idx].copy()

            chs = np.asarray(list(schedule), dtype=int)
            ch_energy_before = network.energy[chs].copy()
            expected_ch_packets = int(network.alive[chs].sum())
            offset = 0.0

            # Level 0 (only after a re-cluster): the BS broadcasts the schedule.
            if schedule_pending:
                payload = json.dumps(
                    {"round": round_index,
                     "schedule": {str(ch): members for ch, members in schedule.items()}},
                    separators=(",", ":"),
                ).encode("utf-8")
                duration, _, _ = send_slot(sink, None, payload, offset, 32, alive_idx)
                offset += duration
                schedule_pending = False

            # Level 1: one frame per CH -- a beacon, then one slot per member.
            # No beacon (CH dead) or no MAC ACK (CH died mid-frame) means the
            # member is orphaned and falls back to the BS in the last frame.
            expected_member_packets = 0
            delivered_to_ch = {ch: [] for ch in schedule}
            orphaned = []
            for ch, members in schedule.items():
                live_members = [m for m in members if network.alive[m]]
                expected_member_packets += len(live_members)
                beacon = json.dumps({"ch": ch, "round": round_index}).encode("utf-8")
                duration, _, heard = send_slot(ch, None, beacon, offset, 32, live_members)
                offset += duration
                heard = set(heard)
                for member in live_members:
                    if member not in heard:
                        if network.alive[member]:
                            orphaned.append(member)
                        continue
                    duration, sent, got = send_slot(
                        member, ch, record_payload(member), offset, 32, [ch]
                    )
                    offset += duration
                    if got:
                        delivered_to_ch[ch].append(member)
                    elif sent:
                        orphaned.append(member)

            # Level 2: one slot per CH for its aggregate to the BS ...
            for ch in schedule:
                cluster_nodes = [ch] + delivered_to_ch[ch]
                payload = json.dumps(
                    {"cluster_head": ch, "round": round_index,
                     "nodes": [json.loads(record_payload(n)) for n in cluster_nodes]},
                    separators=(",", ":"), allow_nan=False,
                ).encode("utf-8")
                duration, _, _ = send_slot(
                    ch, sink, payload, offset, len(cluster_nodes) * 16, [sink]
                )
                offset += duration

            # ... then the emergency slots: orphaned members send directly to the BS.
            for member in orphaned:
                duration, _, _ = send_slot(
                    member, sink, record_payload(member), offset, 32, [sink]
                )
                offset += duration

            round_duration = max(offset, 0.01)
            live = network.alive.copy()
            network.energy[live] = np.maximum(
                0.0, network.energy[live] - round_duration * idle_power
            )

            mark(f"round {round_index} before Run")
            ns.Simulator.Stop(keep(ns.Seconds(round_duration)))
            ns.Simulator.Run()
            mark(f"round {round_index} after Run")
            send_failures = sum(
                sender.GetSendFailures() for sender in transmit_senders
            )
            if send_failures:
                raise RuntimeError(f"ns-3 reported {send_failures} UDP send failures")

            received, ch_delivered, direct_delivered = drain_packets()
            member_delivered = sum(received[ch] for ch in schedule)
            expected_packets = expected_member_packets + expected_ch_packets
            history["member_packets_delivered"].append(member_delivered)
            history["ch_packets_delivered"].append(ch_delivered)
            history["direct_to_bs_packets"].append(direct_delivered)
            history["orphaned_members"].append(len(orphaned))
            # Hop-level delivery: member data counts once, at its CH or (after
            # a CH failure) directly at the BS.
            pdr = (
                (member_delivered + direct_delivered + ch_delivered) / expected_packets
                if expected_packets else 1.0
            )
            history["delivery_rate"].append(pdr)
            history["round_time_s"].append(round_duration)

            print(f"r{round_index} t={keep(ns.Simulator.Now()).GetSeconds():.3f}s "
                  f"sink={remaining_energy(sink):.6f} "
                  f"n0={remaining_energy(0):.6f} n1={remaining_energy(1):.6f}", flush=True)
            network.alive[:] = network.energy > 0
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

            # Re-cluster triggers, decided at the BS after the round.
            reason = None
            if round_index + 1 < rounds and alive_now:
                if direct_delivered:
                    reason = "ch_failure"  # BS received data from a non-CH node
                elif not network.alive[chs].any():
                    reason = "no_live_ch"
                if predictor is None:
                    reason = reason or "periodic"  # baselines re-cluster every round
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
                    history["predicted_state"].append(state)
                    if reason is None and state == "recluster":
                        reason = "predicted"
                    elif reason is None and state == "watch":
                        handle_watch(network, round_index)
                if reason is not None:
                    optimization = optimizer(network, rng)
                    schedule = build_schedule(optimization.selected)
                    schedule_pending = True
                    recluster_count += 1
                    recluster_reasons[reason] = recluster_reasons.get(reason, 0) + 1
            history["reclustered"].append(reason is not None)
            history["recluster_reason"].append(reason)
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
        "radio_range": radio_range,
        "final_alive": history["alive"][-1] if history["alive"] else initial_alive,
        "final_avg_residual": history["avg_residual"][-1] if history["avg_residual"] else 0.0,
        "final_mutation_rate": (
            history["mutation_rate"][-1] if history["mutation_rate"] else None
        ),
        "first_death_round": first_death_round,
        "recluster_count": recluster_count,
        "recluster_reasons": recluster_reasons,
        "direct_to_bs_packets": int(sum(history["direct_to_bs_packets"])),
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
