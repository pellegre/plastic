from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from ..base.dataset import BaseGraphDataset
from ..base.generator import BaseGraphGenerator, NetworkXGraphGenerator


class StandardGraphDataset(BaseGraphDataset):
    """
    Dataset where train/test share the same action space and triplets.
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
        self.edge_prob: float = float(edge_prob)
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
            auto_build=True,
        )

    @staticmethod
    def _max_degree(G: nx.Graph) -> int:
        return max((deg for _, deg in G.degree()), default=0)

    def _has_unique_actions_per_node(self, G: nx.Graph) -> bool:
        return all(
            len({int(G.edges[x, y]["action_id"]) for x, y in G.edges(u)}) == G.degree(u)
            for u in G.nodes()
        )

    def create_random_graph(self, split: str = "train") -> nx.Graph:
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

            delta: int = self._max_degree(G)
            needed: int = delta if nx.is_bipartite(G) else delta + 1
            if self.num_actions < needed:
                rejects += 1
                if rejects >= 10000:
                    raise RuntimeError(
                        f"Rejected {rejects} graphs in a row while building split='{split}'. "
                        "Increase num_actions or reduce edge probability."
                    )
                continue

            # random initial labels
            for u, v in G.edges():
                G.edges[u, v]["action_id"] = int(self.rng.integers(0, self.num_actions))

            # enforce uniqueness around each node
            changed: bool = True
            iters: int = 0
            max_iters: int = 10 * max(1, G.number_of_edges())
            while changed and iters < max_iters:
                changed = False
                iters += 1
                for u in G.nodes():
                    incident = list(G.edges(u))
                    if len(incident) <= 1:
                        continue
                    used: set[int] = set()
                    by_action: Dict[int, List[Tuple[int, int]]] = {}
                    for x, y in incident:
                        a: int = int(G.edges[x, y]["action_id"])
                        by_action.setdefault(a, []).append((x, y))
                    for a, edges in by_action.items():
                        if len(edges) <= 1:
                            used.add(a)
                            continue
                        keep = edges[0]
                        used.add(a)
                        for x, y in edges[1:]:
                            choices = [
                                z for z in range(self.num_actions) if z not in used
                            ]
                            new_a: int = (
                                int(self.rng.choice(choices))
                                if choices
                                else int(self.rng.integers(0, self.num_actions))
                            )
                            G.edges[x, y]["action_id"] = new_a
                            used.add(new_a)
                            changed = True

            if self._has_unique_actions_per_node(G):
                return G
            rejects += 1
            if rejects >= 10000:
                raise RuntimeError(
                    f"Rejected {rejects} graphs in a row while building split='{split}'. "
                    "Increase num_actions or reduce edge probability."
                )

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
