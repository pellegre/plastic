from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import math
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from .generator import (
    BaseGraphGenerator,
    NetworkXGraphGenerator,
    build_generator_from_config,
)


class BaseGraphDataset(ABC):
    """
    Base class for graph datasets that generate graphs, explore them into
    sequences, and expose train/test splits. A graph generator object provides
    ready-made NetworkX graphs, independent of how they are parametrized.
    """

    def __init__(
        self,
        num_nodes: int,
        num_graphs: int,
        num_samples_per_graph: int,
        num_timesteps: int,
        num_actions: int,
        graph_generator: Optional[BaseGraphGenerator] = None,
        seed: Optional[int] = None,
        test_fraction: float = 0.2,
        auto_build: bool = True,
    ) -> None:
        self.num_nodes: int = int(num_nodes)
        self.num_graphs_train: int = int(num_graphs)
        self.num_samples_per_graph: int = int(num_samples_per_graph)
        self.num_timesteps: int = int(num_timesteps)
        self.num_actions: int = int(num_actions)
        self.seed: Optional[int] = seed
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self.graph_generator: BaseGraphGenerator = (
            graph_generator or NetworkXGraphGenerator(seed=seed)
        )

        test_count: int = int(math.ceil(test_fraction * self.num_graphs_train))
        self.num_graphs_test: int = max(1, test_count)

        self.graphs_train: List[nx.Graph] = []
        self.graphs_test: List[nx.Graph] = []
        self.train_data: Optional[np.ndarray] = None
        self.test_data: Optional[np.ndarray] = None

        if auto_build:
            self._build_splits()

    def create_random_graph(self, split: str = "train") -> nx.Graph:
        """
        Default graph creation delegated entirely to the provided generator.
        Subclasses may override if they need split-specific behavior.
        """
        return self.graph_generator.generate(self.num_nodes, split=split)

    @abstractmethod
    def explore_graph(self, G: nx.Graph) -> np.ndarray:
        """
        Subclasses must perform one exploration of the graph and return the
        resulting sequence tensor/array.
        """

    def get_graph_samples(self, G: nx.Graph) -> np.ndarray:
        samples: List[np.ndarray] = [
            self.explore_graph(G) for _ in range(self.num_samples_per_graph)
        ]
        return np.stack(samples, axis=0)

    def plot_random_graph(self, path: str, split: str = "train") -> None:
        graphs: List[nx.Graph]
        if split == "train":
            graphs = self.graphs_train
        elif split == "test":
            graphs = self.graphs_test
        else:
            raise ValueError('split must be either "train" or "test"')

        if not graphs:
            raise ValueError(f"No graphs available in split '{split}'.")

        idx: int = int(self.rng.integers(0, len(graphs)))
        G: nx.Graph = graphs[idx]

        seed_val: int = int(self.rng.integers(0, 2**32 - 1))
        pos = nx.spring_layout(G, seed=seed_val)

        plt.figure(figsize=(20, 20))
        nx.draw(G, pos, with_labels=True, node_size=600, font_size=10)

        # Draw edge labels if action_id is present.
        if any("action_id" in G.edges[u, v] for u, v in G.edges()):
            edge_labels = {
                (u, v): int(G.edges[u, v]["action_id"]) for u, v in G.edges()
            }
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

        plt.axis("off")
        plt.savefig(path, dpi=200)
        plt.close()

    def _build_split_graphs(
        self, split: str, count: int
    ) -> Tuple[List[nx.Graph], np.ndarray]:
        graphs: List[nx.Graph] = [
            self.create_random_graph(split=split) for _ in range(count)
        ]
        split_samples: List[np.ndarray] = [self.get_graph_samples(g) for g in graphs]
        data: np.ndarray = np.concatenate(split_samples, axis=0)
        return graphs, data

    def _build_splits(self) -> None:
        self.graphs_train, self.train_data = self._build_split_graphs(
            "train", self.num_graphs_train
        )
        self.graphs_test, self.test_data = self._build_split_graphs(
            "test", self.num_graphs_test
        )

    def get(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.train_data is None or self.test_data is None:
            self._build_splits()
        assert self.train_data is not None and self.test_data is not None
        return self.train_data, self.test_data

    def to_config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            "num_nodes": self.num_nodes,
            "num_graphs": self.num_graphs_train,
            "num_samples_per_graph": self.num_samples_per_graph,
            "num_timesteps": self.num_timesteps,
            "num_actions": self.num_actions,
            "seed": self.seed,
            "graph_generator": self.graph_generator.to_config(),
        }
        cfg.update(self.extra_config())
        return cfg

    def extra_config(self) -> Dict[str, Any]:
        """
        Hook for subclasses to inject additional configuration into checkpoints.
        """
        return {}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseGraphDataset":
        cfg: Dict[str, Any] = dict(config)
        generator_config: Optional[Dict[str, Any]] = cfg.pop("graph_generator", None)
        seed = cfg.get("seed")
        graph_generator: BaseGraphGenerator = (
            build_generator_from_config(generator_config)
            if generator_config is not None
            else NetworkXGraphGenerator(seed=seed)
        )
        return cls(graph_generator=graph_generator, **cfg)
