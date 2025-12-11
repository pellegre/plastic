from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

import networkx as nx
import numpy as np


GENERATOR_REGISTRY: Dict[str, Type["BaseGraphGenerator"]] = {}


def register_generator(cls: Type["BaseGraphGenerator"]) -> Type["BaseGraphGenerator"]:
    GENERATOR_REGISTRY[cls.__name__] = cls
    return cls


class BaseGraphGenerator(ABC):
    """
    Abstract graph generator that produces a ready-to-use NetworkX graph.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed: Optional[int] = seed
        self.rng: np.random.Generator = np.random.default_rng(seed)

    def __call__(self, num_nodes: int, split: str = "train") -> nx.Graph:
        return self.generate(num_nodes, split=split)

    @abstractmethod
    def generate(self, num_nodes: int, split: str = "train") -> nx.Graph:
        """
        Produce a graph with the requested number of nodes.
        """

    def to_config(self) -> Dict[str, Any]:
        return {
            "generator_class": self.__class__.__name__,
            "kwargs": self.extra_config(),
        }

    def extra_config(self) -> Dict[str, Any]:
        return {"seed": self.seed}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseGraphGenerator":
        return cls(**config.get("kwargs", {}))


@register_generator
class NetworkXGraphGenerator(BaseGraphGenerator):
    """
    Wraps any NetworkX random graph generator.
    """

    def __init__(
        self,
        generator_name: str = "gnp_random_graph",
        generator_kwargs: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(seed=seed)
        self.generator_name: str = generator_name
        self.generator_kwargs: Dict[str, Any] = dict(generator_kwargs or {})

    def generate(self, num_nodes: int, split: str = "train") -> nx.Graph:
        generator = getattr(nx, self.generator_name, None)
        if generator is None:
            raise ValueError(f"Unknown NetworkX generator: {self.generator_name}")

        kwargs: Dict[str, Any] = dict(self.generator_kwargs)
        if "seed" not in kwargs:
            kwargs["seed"] = self.rng

        try:
            return generator(num_nodes, **kwargs)
        except TypeError:
            return generator(n=num_nodes, **kwargs)

    def extra_config(self) -> Dict[str, Any]:
        return {
            "generator_name": self.generator_name,
            "generator_kwargs": self.generator_kwargs,
            "seed": self.seed,
        }


def build_generator_from_config(config: Optional[Dict[str, Any]]) -> BaseGraphGenerator:
    if config is None:
        return NetworkXGraphGenerator()

    class_name: str = config.get("generator_class", "")
    kwargs: Dict[str, Any] = dict(config.get("kwargs", {}))
    cls = GENERATOR_REGISTRY.get(class_name)
    if cls is None:
        raise ValueError(f"Unknown graph generator class: {class_name}")
    return cls(**kwargs)
