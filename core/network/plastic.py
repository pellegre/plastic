from __future__ import annotations

from typing import Any, Dict, List, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base.network import BaseGraphNetwork


class PlasticGraphNetwork(BaseGraphNetwork):
    """
    Plastic network variant with online weight updates.
    """

    def __init__(
        self,
        num_nodes: int,
        num_actions: int,
        hidden_dim: int,
        batch_size: int,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(num_nodes=num_nodes, num_actions=num_actions, seed=seed)
        self.hidden_dim: int = int(hidden_dim)
        self.batch_size: int = int(batch_size)

        # Input: node + action + 2-mode bits
        self.input_dim: int = int(self.num_nodes + self.num_actions + 2)

        self.c1: nn.Linear = nn.Linear(self.input_dim, self.hidden_dim)
        self.ca: nn.Linear = nn.Linear(self.num_nodes, self.hidden_dim)

        init_wp = torch.randn(self.batch_size, self.hidden_dim, self.hidden_dim)
        init_wp = init_wp / (torch.norm(init_wp, dim=2, keepdim=True) + 1e-8)
        self.register_buffer("Wp", init_wp)

        intermediate_dim: int = int(math.sqrt(self.hidden_dim * self.num_nodes))
        self.decoder: nn.Sequential = nn.Sequential(
            nn.Linear(self.hidden_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, self.num_nodes),
        )

        # Mode vectors as buffers.
        self.register_buffer("mode_acq_template", torch.tensor([[0, 1]], dtype=torch.float32))
        self.register_buffer("mode_pred_template", torch.tensor([[1, 0]], dtype=torch.float32))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B: int = X.shape[0]
        T: int = X.shape[1]
        
        mode_acq_batch: torch.Tensor = self.mode_acq_template.expand(B, -1)
        mode_pred_batch: torch.Tensor = self.mode_pred_template.expand(B, -1)

        if B > self.batch_size:
            raise ValueError(
                f"batch size {B} exceeds model batch_size {self.batch_size}"
            )

        Wp_current: torch.Tensor = self.Wp[:B].detach().clone()
        all_logits: List[torch.Tensor] = []

        for t in range(T - 1):
            x_t: torch.Tensor = X[:, t, : self.num_nodes]
            a_t: torch.Tensor = X[:, t, self.num_nodes :]

            if t > 0:
                x_t_minus_1: torch.Tensor = X[:, t - 1, : self.num_nodes]
                a_t_minus_1: torch.Tensor = X[:, t - 1, self.num_nodes :]

                input_acq: torch.Tensor = torch.cat(
                    [x_t_minus_1, a_t_minus_1, mode_acq_batch], dim=1
                )

                c1_acq: torch.Tensor = F.relu(self.c1(input_acq))
                ca_out: torch.Tensor = F.relu(self.ca(x_t))

                delta_Wp: torch.Tensor = torch.bmm(
                    ca_out.unsqueeze(2), c1_acq.unsqueeze(1)
                )

                Wp_current = Wp_current + delta_Wp

                row_norms: torch.Tensor = torch.norm(Wp_current, dim=2, keepdim=True)
                row_norms = torch.clamp(row_norms, min=1e-8)
                Wp_current = Wp_current / row_norms

            input_pred: torch.Tensor = torch.cat([x_t, a_t, mode_pred_batch], dim=1)

            c1_pred: torch.Tensor = F.relu(self.c1(input_pred))

            h: torch.Tensor = torch.bmm(Wp_current, c1_pred.unsqueeze(2)).squeeze(2)
            h = F.relu(h)

            logits_t: torch.Tensor = self.decoder(h)
            all_logits.append(logits_t)

        with torch.no_grad():
            self.Wp[:B] = Wp_current

        return torch.stack(all_logits, dim=1)

    def reset_plastic_weights(self) -> None:
        new_wp = torch.randn_like(self.Wp)
        new_wp = new_wp / (torch.norm(new_wp, dim=2, keepdim=True) + 1e-8)
        self.Wp.copy_(new_wp)

    def fast_weights(self) -> Dict[str, torch.Tensor]:
        return {"Wp": self.Wp}

    def extra_model_config(self) -> Dict[str, Any]:
        cfg = super().extra_model_config()
        cfg.update(
            {
                "hidden_dim": self.hidden_dim,
                "batch_size": self.batch_size,
                "input_dim": self.input_dim,
                "model_class": "PlasticGraphNetwork",
            }
        )
        return cfg

    @classmethod
    def _load_model_kwargs(cls, model_config: Dict[str, Any]) -> Dict[str, Any]:
        kwargs = super()._load_model_kwargs(model_config)
        kwargs.update(
            {
                "hidden_dim": model_config["hidden_dim"],
                "batch_size": model_config["batch_size"],
            }
        )
        return kwargs
