from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class DatasetUtils:
    """
    Static helpers for analyzing dataset structure.
    """

    @staticmethod
    def compute_action_counts_per_node_pair(
        data: np.ndarray,
        num_nodes: int,
        num_actions: int,
    ) -> np.ndarray:
        num_sequences: int = int(data.shape[0])
        num_timesteps: int = int(data.shape[1])
        feature_dim: int = int(data.shape[2])

        assert feature_dim == num_nodes + num_actions, "Feature dimension mismatch."

        node_slice: slice = slice(0, num_nodes)
        action_slice: slice = slice(num_nodes, num_nodes + num_actions)

        node_idx: np.ndarray = data[:, :, node_slice].argmax(axis=2)  # (N, T)
        action_idx: np.ndarray = data[:, :, action_slice].argmax(axis=2)  # (N, T)

        u_nodes: np.ndarray = node_idx[:, :-1].reshape(-1)
        v_nodes: np.ndarray = node_idx[:, 1:].reshape(-1)
        a_ids: np.ndarray = action_idx[:, :-1].reshape(-1)

        counts: np.ndarray = np.zeros(
            (num_nodes, num_nodes, num_actions), dtype=np.int64
        )
        np.add.at(counts, (u_nodes, v_nodes, a_ids), 1)

        return counts

    @staticmethod
    def plot_action_hist_grid(
        counts: np.ndarray,
        title: str = "Action-ID frequencies per node pair (u→v)",
        output_path: Optional[str] = None,
    ) -> None:
        num_nodes: int = int(counts.shape[0])
        assert (
            counts.shape[0] == counts.shape[1]
        ), "Counts must be square on first two dims."
        num_actions: int = int(counts.shape[2])

        fig, axes = plt.subplots(num_nodes, num_nodes, figsize=(20, 20))
        fig.suptitle(title, y=1.02)

        x_vals: np.ndarray = np.arange(num_actions)

        for u in range(num_nodes):
            for v in range(num_nodes):
                ax = axes[u, v]
                y_vals: np.ndarray = counts[u, v]
                ax.bar(x_vals, y_vals)
                ax.set_title(f"{u}→{v}", fontsize=8)
                if u == num_nodes - 1:
                    ax.set_xlabel("Action ID", fontsize=8)
                if v == 0:
                    ax.set_ylabel("Frequency", fontsize=8)
                ax.tick_params(axis="both", labelsize=7)

        # Use constrained layout for better fit; fallback to tight_layout with padding.
        try:
            fig.set_constrained_layout_pads(w_pad=0.5, h_pad=0.5, wspace=0.2, hspace=0.2)
            fig.set_constrained_layout(True)
            fig.canvas.draw_idle()
        except Exception:
            plt.tight_layout(pad=0.8)
        if output_path:
            plt.savefig(output_path, dpi=200)
        else:
            plt.show()
        plt.close()

    @staticmethod
    def triplet_overlap_fraction(dataset: Any) -> float:
        train_triplets: Set[Tuple[int, int, int]] = set()
        for G in dataset.graphs_train:
            for u, v, d in G.edges(data=True):
                a: int = int(d["action_id"])
                u2: int = u if u <= v else v
                v2: int = v if u <= v else u
                train_triplets.add((u2, a, v2))

        test_triplets: Set[Tuple[int, int, int]] = set()
        for G in dataset.graphs_test:
            for u, v, d in G.edges(data=True):
                a: int = int(d["action_id"])
                u2: int = u if u <= v else v
                v2: int = v if u <= v else u
                test_triplets.add((u2, a, v2))

        if not test_triplets:
            return 0.0

        overlap_count: int = len(test_triplets.intersection(train_triplets))
        return float(overlap_count / len(test_triplets))

    @staticmethod
    def count_triplets_and_check_constraints(
        data: np.ndarray,
        num_nodes: int,
        num_actions: int,
        num_samples_per_graph: int,
    ) -> bool:
        num_sequences: int = int(data.shape[0])
        num_timesteps: int = int(data.shape[1])
        feature_dim: int = int(data.shape[2])
        assert feature_dim == num_nodes + num_actions, "Feature dimension mismatch."

        node_slice: slice = slice(0, num_nodes)
        action_slice: slice = slice(num_nodes, num_nodes + num_actions)

        node_idx: np.ndarray = data[:, :, node_slice].argmax(axis=2)
        action_idx: np.ndarray = data[:, :, action_slice].argmax(axis=2)

        if num_timesteps < 2 or num_sequences == 0:
            return True

        u_nodes: np.ndarray = node_idx[:, :-1].reshape(-1)
        v_nodes: np.ndarray = node_idx[:, 1:].reshape(-1)
        a_ids: np.ndarray = action_idx[:, :-1].reshape(-1)

        sample_ids: np.ndarray = np.repeat(
            np.arange(num_sequences, dtype=np.int64), num_timesteps - 1
        )
        graph_ids: np.ndarray = sample_ids // int(num_samples_per_graph)

        key_ua: np.ndarray = (
            graph_ids * (num_nodes * num_actions) + u_nodes * num_actions + a_ids
        )
        pairs_ua_v: np.ndarray = np.stack([key_ua, v_nodes], axis=1)

        unique_pairs_ua_v, _ = np.unique(pairs_ua_v, axis=0, return_inverse=True)
        unique_keys_ua, _ = np.unique(unique_pairs_ua_v[:, 0], return_inverse=True)

        key_positions: np.ndarray = np.searchsorted(
            unique_keys_ua, unique_pairs_ua_v[:, 0]
        )
        distinct_v_counts_per_key: np.ndarray = np.bincount(
            key_positions, minlength=len(unique_keys_ua)
        )

        violating_key_mask: np.ndarray = distinct_v_counts_per_key > 1
        if np.any(violating_key_mask):
            return False

        min_uv: np.ndarray = np.minimum(u_nodes, v_nodes)
        max_uv: np.ndarray = np.maximum(u_nodes, v_nodes)
        key_edge: np.ndarray = (
            graph_ids * (num_nodes * num_nodes) + min_uv * num_nodes + max_uv
        )
        pairs_edge_a: np.ndarray = np.stack([key_edge, a_ids], axis=1)

        unique_pairs_edge_a, _ = np.unique(pairs_edge_a, axis=0, return_inverse=True)
        unique_edge_keys, _ = np.unique(unique_pairs_edge_a[:, 0], return_inverse=True)

        edge_positions: np.ndarray = np.searchsorted(
            unique_edge_keys, unique_pairs_edge_a[:, 0]
        )
        distinct_a_counts_per_edge: np.ndarray = np.bincount(
            edge_positions, minlength=len(unique_edge_keys)
        )

        violating_edge_mask: np.ndarray = distinct_a_counts_per_edge > 1
        if np.any(violating_edge_mask):
            return False

        return True

    @staticmethod
    def optimal_prediction_rate(
        dataset: Any,
        num_graphs: int,
        timesteps: int,
        seed: Optional[int] = None,
    ) -> float:
        rng: np.random.Generator = np.random.default_rng(seed)

        total_steps: int = 0
        correct: int = 0

        for _ in range(int(num_graphs)):
            G: nx.Graph = dataset.create_random_graph()

            lut: Dict[Tuple[int, int], int] = {}
            current_node: int = int(rng.choice(list(G.nodes)))

            for _ in range(int(timesteps)):
                neighbors = list(G.neighbors(current_node))
                if not neighbors:
                    current_node = int(rng.choice(list(G.nodes)))
                    continue

                actual_next: int = int(rng.choice(neighbors))
                action_id: int = int(G.edges[current_node, actual_next]["action_id"])

                key = (current_node, action_id)
                if key in lut:
                    pred_next: int = int(lut[key])
                else:
                    pred_next = int(rng.choice(neighbors))

                total_steps += 1
                if pred_next == actual_next:
                    correct += 1

                lut[key] = actual_next
                lut[(actual_next, action_id)] = current_node

                current_node = actual_next

        return float(correct / total_steps) if total_steps > 0 else 0.0
