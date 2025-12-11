from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from ..base.dataset import BaseGraphDataset
from ..base.generator import BaseGraphGenerator, NetworkXGraphGenerator


class DisjointGraphDataset(BaseGraphDataset):
    """
    Dataset where train/test action triplets are disjoint per node pair.
    """

    def __init__(
        self,
        num_nodes: int,
        edge_prob: float,
        num_samples_per_graph: int,
        num_graphs: int,
        num_timesteps: int,
        num_actions: int,
        graph_generator: Optional[BaseGraphGenerator] = None,
        seed: Optional[int] = None,
        test_fraction: float = 0.2,
    ) -> None:
        if num_actions < 2:
            raise ValueError(
                "num_actions must be at least 2 to split into train/test action sets."
            )

        self.edge_prob: float = float(edge_prob)
        self.k_train: int = int(num_actions // 2)
        self.k_test: int = int(num_actions - self.k_train)

        generator = graph_generator or NetworkXGraphGenerator(
            generator_name="gnp_random_graph",
            generator_kwargs={"p": self.edge_prob},
            seed=seed,
        )

        super().__init__(
            num_nodes=num_nodes,
            num_graphs=num_graphs,
            num_samples_per_graph=num_samples_per_graph,
            num_timesteps=num_timesteps,
            num_actions=num_actions,
            graph_generator=generator,
            seed=seed,
            test_fraction=test_fraction,
            auto_build=False,
        )

        self.train_actions_per_pair, self.test_actions_per_pair = (
            self.build_pair_action_splits()
        )
        self._build_splits()

    def build_pair_action_splits(
        self,
    ) -> Tuple[Dict[Tuple[int, int], Set[int]], Dict[Tuple[int, int], Set[int]]]:
        train_map: Dict[Tuple[int, int], Set[int]] = {}
        test_map: Dict[Tuple[int, int], Set[int]] = {}
        all_actions: np.ndarray = np.arange(self.num_actions, dtype=int)
        for u in range(self.num_nodes):
            for v in range(u + 1, self.num_nodes):
                perm: np.ndarray = self.rng.permutation(all_actions)
                train_set: Set[int] = set(map(int, perm[: self.k_train]))
                test_set: Set[int] = set(map(int, perm[self.k_train :]))
                train_map[(u, v)] = train_set
                test_map[(u, v)] = test_set
        return train_map, test_map

    def create_random_graph(self, split: str = "train") -> nx.Graph:
        if split not in ("train", "test"):
            raise ValueError('split must be either "train" or "test"')

        allowed_map: Dict[Tuple[int, int], Set[int]] = (
            self.train_actions_per_pair
            if split == "train"
            else self.test_actions_per_pair
        )
        k_split: int = self.k_train if split == "train" else self.k_test

        rejects: int = 0
        while True:
            G: nx.Graph = self.graph_generator.generate(self.num_nodes, split=split)
            if not (G.number_of_edges() > 0 and nx.is_connected(G)):
                rejects += 1
                if rejects >= 10000:
                    raise RuntimeError(
                        f"Rejected {rejects} graphs in a row while building split='{split}'. "
                        "Increase num_actions or reduce edge probability."
                    )
                continue

            delta: int = max(deg for _, deg in G.degree())
            if delta > k_split:
                rejects += 1
                if rejects >= 10000:
                    raise RuntimeError(
                        f"Rejected {rejects} graphs in a row while building split='{split}'. "
                        "Increase num_actions or reduce edge probability."
                    )
                continue

            if self.assign_actions_for_graph(G, allowed_map):
                return G
            rejects += 1
            if rejects >= 10000:
                raise RuntimeError(
                    f"Rejected {rejects} graphs in a row while building split='{split}'. "
                    "Increase num_actions or reduce edge probability."
                )

    def assign_actions_for_graph(
        self, G: nx.Graph, allowed_map: Dict[Tuple[int, int], Set[int]]
    ) -> bool:
        used_at_node: Dict[int, Set[int]] = {u: set() for u in G.nodes()}
        edges: List[Tuple[int, int]] = list(G.edges())
        edges.sort(key=lambda e: max(G.degree(e[0]), G.degree(e[1])), reverse=True)

        for u, v in edges:
            key: Tuple[int, int] = (u, v) if u < v else (v, u)
            a_set: Set[int] = allowed_map[key]
            candidates: List[int] = [
                a
                for a in a_set
                if (a not in used_at_node[u] and a not in used_at_node[v])
            ]
            if not candidates:
                return False

            chosen: int = int(self.rng.choice(candidates))
            G.edges[u, v]["action_id"] = chosen
            used_at_node[u].add(chosen)
            used_at_node[v].add(chosen)

        return True

    def explore_graph(self, G: nx.Graph) -> np.ndarray:
        current_node: int = int(self.rng.choice(list(G.nodes)))
        sequence: List[np.ndarray] = []

        for _t in range(self.num_timesteps):
            neighbors: List[int] = list(G.neighbors(current_node))

            if neighbors:
                next_node: int = int(self.rng.choice(neighbors))
                action_vector: np.ndarray = np.zeros(self.num_actions, dtype=float)
                action_id: int = int(G.edges[current_node, next_node]["action_id"])
                action_vector[action_id] = 1.0
            else:
                next_node: int = int(self.rng.choice(list(G.nodes)))
                action_vector: np.ndarray = np.zeros(self.num_actions, dtype=float)

            node_vector: np.ndarray = np.zeros(self.num_nodes, dtype=float)
            node_vector[current_node] = 1.0

            X_input: np.ndarray = np.concatenate([node_vector, action_vector])
            sequence.append(X_input)

            current_node = next_node

        return np.stack(sequence, axis=0)

    def extra_config(self) -> dict:
        return {"edge_prob": self.edge_prob}
