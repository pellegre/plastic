from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Type, TypeVar
import random

import numpy as np
import torch
import torch.nn as nn


TNet = TypeVar("TNet", bound="BaseGraphNetwork")


class BaseGraphNetwork(nn.Module, ABC):
    """
    Base class for graph sequence models.
    Handles common metadata bookkeeping and checkpoint I/O without coupling to datasets.
    """

    def __init__(
        self, num_nodes: int, num_actions: int, seed: Optional[int] = None
    ) -> None:
        super().__init__()
        self.num_nodes: int = int(num_nodes)
        self.num_actions: int = int(num_actions)
        self.seed: Optional[int] = int(seed) if seed is not None else None
        # Seed early so subclass initialization remains deterministic.
        self.set_seed(self.seed)

    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Subclasses must implement the forward pass.
        """

    def set_seed(self, seed: Optional[int] = None) -> None:
        """
        Set all relevant RNGs for reproducibility.
        """
        if seed is not None:
            self.seed = int(seed)

        if self.seed is None:
            return

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def model_config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            "model_class": self.__class__.__name__,
            "num_nodes": self.num_nodes,
            "num_actions": self.num_actions,
            "seed": self.seed,
        }
        cfg.update(self.extra_model_config())
        return cfg

    def extra_model_config(self) -> Dict[str, Any]:
        """
        Hook for subclasses to inject additional model hyperparameters.
        """
        return {}

    def fast_weights(self) -> Dict[str, torch.Tensor]:
        """
        Fast weights are short-lived tensors updated during the forward pass (e.g., plastic weights).
        Subclasses without fast weights can rely on the default empty mapping.
        """
        return {}

    def parameter_counts(self) -> Dict[str, int]:
        """
        Return a breakdown of parameter counts between slow (trainable) and fast (ephemeral) weights.
        """
        slow_params = sum(p.numel() for p in self.parameters())
        fast_params = sum(t.numel() for t in self.fast_weights().values())
        total_params = slow_params + fast_params
        return {
            "slow": int(slow_params),
            "fast": int(fast_params),
            "total": int(total_params),
        }

    def save(self, path: str, dataset_config: Optional[Dict[str, Any]] = None) -> None:
        state_cpu: Dict[str, torch.Tensor] = {
            k: v.detach().to("cpu") for k, v in self.state_dict().items()
        }
        checkpoint: Dict[str, Any] = {
            "model_state_dict": state_cpu,
            "model_config": self.model_config(),
            "dataset_config": dataset_config,
            "format_version": 1,
            "pytorch_version": str(torch.__version__),
        }
        torch.save(checkpoint, path)

    @classmethod
    def _load_model_kwargs(cls, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hook for subclasses to unpack model-specific kwargs from the saved config.
        """
        return {
            "num_nodes": model_config["num_nodes"],
            "num_actions": model_config["num_actions"],
            "seed": model_config.get("seed"),
        }

    @classmethod
    def load(
        cls: Type[TNet],
        path: str,
        device: Optional[torch.device] = None,
    ) -> Tuple[TNet, Optional[Dict[str, Any]]]:
        checkpoint: Dict[str, Any]

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        model: TNet = cls(**cls._load_model_kwargs(checkpoint["model_config"]))
        model.load_state_dict(checkpoint["model_state_dict"])

        if device is not None:
            model.to(device)

        model.eval()
        return model, checkpoint.get("dataset_config")
