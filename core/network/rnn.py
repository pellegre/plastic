from __future__ import annotations

from typing import Any, Dict, Optional
import math

import torch
import torch.nn as nn

from ..base.network import BaseGraphNetwork


class RNNGraphNetwork(BaseGraphNetwork):
    """
    RNN-based graph sequence model.
    """

    def __init__(
        self,
        num_nodes: int,
        num_actions: int,
        hidden_dim: int,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(num_nodes=num_nodes, num_actions=num_actions, seed=seed)
        self.hidden_dim: int = int(hidden_dim)
        self.input_dim: int = int(self.num_nodes + self.num_actions)

        self.rnn: nn.RNN = nn.RNN(self.input_dim, self.hidden_dim, batch_first=True)

        intermediate_dim: int = int(math.sqrt(self.hidden_dim * self.num_nodes))
        self.decoder: nn.Sequential = nn.Sequential(
            nn.Linear(self.hidden_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, self.num_nodes),
        )

    def get_hidden_states(self, X: torch.Tensor) -> torch.Tensor:
        Z, _ = self.rnn(X)
        return Z

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Z, _ = self.rnn(X)
        Z_prev: torch.Tensor = Z[:, :-1, :]
        logits: torch.Tensor = self.decoder(Z_prev)
        return logits

    def extra_model_config(self) -> Dict[str, Any]:
        cfg = super().extra_model_config()
        cfg.update(
            {
                "hidden_dim": self.hidden_dim,
                "input_dim": self.input_dim,
                "rnn_class": "RNN",
            }
        )
        return cfg

    @classmethod
    def _load_model_kwargs(cls, model_config: Dict[str, Any]) -> Dict[str, Any]:
        kwargs = super()._load_model_kwargs(model_config)
        kwargs.update({"hidden_dim": model_config["hidden_dim"]})
        return kwargs
