from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import uuid
import time
from datetime import datetime
import subprocess
import json
import math

from ..dataset import DisjointGraphDataset, StandardGraphDataset
from ..network import RNNGraphNetwork, PlasticGraphNetwork
from ..base.trainer import Trainer, TrainingResult
from ..utils.logging import PrefixedLogger
from ..utils.dataset import DatasetUtils
from ..utils.report import RunReportBuilder
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import networkx as nx
import pandas as pd

# Optional dependency; code still runs if wandb is missing.
try:  # pragma: no cover
    import wandb  # type: ignore
except ImportError:  # pragma: no cover
    wandb = None


@dataclass
class RunConfig:
    seed: Optional[int]
    dataset_type: str  # "standard" or "disjoint"
    num_nodes: int
    edge_prob: float
    num_samples_per_graph: int
    num_graphs: int
    num_timesteps: int
    num_actions: int
    test_fraction: float = 0.2
    network_type: str = "rnn"  # "rnn" or "plastic"
    hidden_dim: int = 32
    epochs: int = 10
    trainer_batch_size: int = 8
    lr: float = 5e-4
    val_split: float = 0.2
    scheduler: str = "none"
    git_hash: Optional[str] = None
    # DDP
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    # Logging
    wandb_enabled: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    # Evaluation
    long_timesteps: int = 2000
    long_window_size: int = 100
    boundary_num_graphs: int = 5
    boundary_repeats_per_graph: int = 5
    boundary_window_size: int = 50


class Run:
    """
    Represents a single training/data run. Responsible for dataset construction,
    logging, and folder management.
    """

    def __init__(self, config: RunConfig, root: Optional[Path] = None) -> None:
        self.config: RunConfig = config
        self.workflow_id: str = str(uuid.uuid4())
        self.root: Path = root or Path.cwd()
        self.run_root: Path = self.root / "run"
        self.run_dir: Path = self.run_root / self.workflow_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.log_path: Path = self.run_dir / "run.log"
        # Ensure CUDA device is set early for this rank before any tensor work.
        if self.config.distributed and torch.cuda.is_available():
            torch.cuda.set_device(int(self.config.local_rank))
        self.logger: PrefixedLogger = PrefixedLogger(
            self.log_path,
            write_stdout=not (self.config.distributed and self.config.rank != 0),
        )
        self.report_data: Dict[str, Dict[str, Any]] = {"metrics": {}, "artifacts": {}}
        self.wandb_run = None

        self.device, self.device_name = self._select_device()
        self.plastic_batch_size: Optional[int] = None
        if self.config.network_type == "plastic":
            world_size = max(1, int(self.config.world_size))
            self.plastic_batch_size = int(
                math.ceil(self.config.trainer_batch_size / float(world_size))
            )
        self.training_duration_seconds: Optional[float] = None
        self.start_time = time.time()
        detected_hash = self.config.git_hash or self._detect_git_hash()
        self.config.git_hash = self._clean_git_hash(detected_hash)
        self.logger.log(f"workflow id = {self.workflow_id}", prefix="[~]")
        self.logger.log(f"run folder = {self.run_dir}", prefix="[~]")
        config_rows = tabulate(
            self._config_rows(), headers=["param", "value"], tablefmt="grid"
        )
        self.logger.log("config parameters", prefix="[~]")
        for line in config_rows.splitlines():
            self.logger.log(line, prefix="   ")
        self.logger.log(f"start time = {datetime.now().isoformat()}", prefix="[~]")

    def _config_rows(self):
        cfg: Dict[str, Any] = self.config.__dict__
        rows: List[Tuple[str, Any]] = []
        for k, v in cfg.items():
            # Skip empty values (None, empty string, empty collections).
            if v is None:
                continue
            if isinstance(v, str) and v == "":
                continue
            if isinstance(v, (list, dict, set, tuple)) and len(v) == 0:
                continue
            rows.append((k, v))
        if self.plastic_batch_size is not None:
            rows.append(("plastic_batch_size", self.plastic_batch_size))
        return rows

    def _format_duration(self, seconds: float) -> str:
        minutes, rem = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        parts = []
        if hours > 0:
            parts.append(f"{int(hours)}h")
        if minutes > 0:
            parts.append(f"{int(minutes)}m")
        if rem > 0 or not parts:
            parts.append(f"{rem:.2f}s")
        return " ".join(parts)

    def _report_config(self) -> Dict[str, Any]:
        cfg = dict(asdict(self.config))
        # Drop wandb-specific settings from the PDF report to keep it focused on
        # model and dataset configuration.
        for key in list(cfg.keys()):
            if key.startswith("wandb"):
                cfg.pop(key, None)
        if self.plastic_batch_size is not None:
            cfg["plastic_batch_size"] = self.plastic_batch_size
        cfg["device"] = self.device_name or "CPU"
        if self.training_duration_seconds is not None:
            cfg["training_time"] = self._format_duration(self.training_duration_seconds)
        return cfg

    def _record_metric(self, key: str, value: Any) -> None:
        self.report_data["metrics"][key] = value
        self._wandb_summary_update({key: value})

    def _record_artifact(self, key: str, path: Path) -> None:
        if path is None:
            return
        try:
            self.report_data["artifacts"][key] = str(path)
            self._wandb_summary_update({f"artifact_{key}": str(path)})
        except Exception:
            pass

    def _init_wandb(self, dataset: Any = None) -> None:
        # Only rank 0 should initialize wandb in distributed runs to avoid duplicate runs.
        if self.config.distributed and self.config.rank != 0:
            return
        if not self.config.wandb_enabled:
            return
        if wandb is None:
            self.logger.log("wandb not installed; skipping wandb logging", prefix="[-]")
            return
        try:
            tags = self._compute_wandb_tags(dataset)
            wandb_config = asdict(self.config)
            if self.plastic_batch_size is not None:
                wandb_config["plastic_batch_size"] = self.plastic_batch_size
                wandb_config["per_rank_batch_size"] = self.plastic_batch_size
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group=self.config.wandb_group,
                tags=tags,
                config=wandb_config,
                name=f"{self.config.network_type}-{self.config.dataset_type}-{self.workflow_id[:8]}",
                id=self.workflow_id,
                resume="allow",
                dir=str(self.run_dir),
            )
            url = getattr(self.wandb_run, "url", None)
            url_msg = f" at {url}" if url else ""
            self.logger.log(f"initialized wandb run{url_msg}", prefix="[~]")
        except Exception as exc:  # pragma: no cover
            self.logger.log(f"failed to init wandb: {exc}", prefix="[-]")
            self.wandb_run = None

    def _select_device(self) -> Tuple[torch.device, str]:
        """
        Prefer a GPU when available so training does not silently fall back to CPU.
        """
        if torch.cuda.is_available():
            if getattr(self.config, "distributed", False):
                local_rank: int = int(getattr(self.config, "local_rank", 0))
                device = torch.device("cuda", local_rank)
                name = torch.cuda.get_device_name(local_rank)
                self.logger.log(
                    f"detected CUDA device [{local_rank}] (ddp); using gpu {local_rank} for training",
                    prefix="[~]",
                )
                return device, name
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            device = torch.device("cuda")
            self.logger.log(
                f"detected CUDA device [{idx}]: {name}; using GPU for training",
                prefix="[~]",
            )
            return device, name

        mps_available = getattr(torch.backends, "mps", None)
        if mps_available is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
            self.logger.log("detected MPS device; using GPU for training", prefix="[~]")
            return device, "MPS"

        self.logger.log("no GPU detected; using CPU for training", prefix="[~]")
        return torch.device("cpu"), "CPU"

    def _wandb_log(self, data: Dict[str, Any], step: Optional[int] = None, commit: bool = True) -> None:
        if self.wandb_run is None:
            return
        try:
            wandb.log(data, step=step, commit=commit)
        except Exception as exc:  # pragma: no cover
            self.logger.log(f"wandb log failed: {exc}", prefix="[-]")

    def _compute_wandb_tags(self, dataset: Any = None) -> List[str]:
        tags: List[str] = []
        if getattr(self.config, "network_type", None):
            tags.append(f"network:{self.config.network_type}")
        if getattr(self.config, "dataset_type", None):
            tags.append(f"dataset:{self.config.dataset_type}")
        if dataset is not None and hasattr(dataset, "graph_generator"):
            gen = getattr(dataset, "graph_generator", None)
            gen_name = getattr(gen, "generator_name", None) or gen.__class__.__name__
            tags.append(f"generator:{gen_name}")
        for t in self.config.wandb_tags or []:
            if t not in tags:
                tags.append(t)
        return tags

    def _clean_git_hash(self, git_hash: Optional[str]) -> Optional[str]:
        """
        Strip a trailing dirty marker so reporting never includes the suffix.
        """
        if git_hash is None:
            return None
        for suffix in ("-dirty", " dirty"):
            if git_hash.endswith(suffix):
                return git_hash[: -len(suffix)]
        return git_hash

    def _detect_git_hash(self) -> Optional[str]:
        try:
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            git_hash = head.stdout.decode("utf-8", errors="replace").strip()
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            dirty = status.stdout.decode("utf-8", errors="replace").strip()
            if dirty:
                git_hash = f"{git_hash}-dirty"
            return self._clean_git_hash(git_hash)
        except Exception:
            return None

    def _wandb_summary_update(self, data: Dict[str, Any]) -> None:
        if self.wandb_run is None:
            return
        try:
            for k, v in data.items():
                self.wandb_run.summary[k] = v
        except Exception as exc:  # pragma: no cover
            self.logger.log(f"wandb summary update failed: {exc}", prefix="[-]")

    def _wandb_config_update(self, data: Dict[str, Any]) -> None:
        if self.wandb_run is None:
            return
        try:
            self.wandb_run.config.update(data, allow_val_change=True)
        except Exception as exc:  # pragma: no cover
            self.logger.log(f"wandb config update failed: {exc}", prefix="[-]")

    def _wandb_log_image(self, key: str, path: Path, step: Optional[int] = None) -> None:
        if self.wandb_run is None:
            return
        try:
            self._wandb_log({key: wandb.Image(str(path))}, step=step, commit=True)
        except Exception as exc:  # pragma: no cover
            self.logger.log(f"wandb image log failed for {path}: {exc}", prefix="[-]")

    def _log_run_artifact(self) -> None:
        if self.wandb_run is None or wandb is None:
            return
        try:
            artifact = wandb.Artifact(
                f"run-{self.workflow_id}", type="graph-run", metadata={"workflow_id": self.workflow_id}
            )
            # Only include the essentials for reproduction.
            essentials = [
                self.run_dir / "model.pth",
                self.run_dir / "run_config.json",
                self.run_dir / "dataset_config.json",
                self.run_dir / "trainer_config.json",
                self.run_dir / "report.pdf",
                self.run_dir / "report.tex",
                self.run_dir / "run.log",
            ]
            for path in essentials:
                if path.exists():
                    artifact.add_file(str(path), name=path.name)
            self.wandb_run.log_artifact(artifact)
            self.logger.log("uploaded run directory to wandb artifact", prefix="[~]")
            url = getattr(self.wandb_run, "url", None)
            if url:
                self.logger.log(f"wandb run url = {url}", prefix="[~]")
        except Exception as exc:  # pragma: no cover
            self.logger.log(f"wandb artifact upload failed: {exc}", prefix="[-]")


    def _build_dataset(self):
        if self.config.dataset_type == "standard":
            ds = StandardGraphDataset(
                num_nodes=self.config.num_nodes,
                edge_prob=self.config.edge_prob,
                num_samples_per_graph=self.config.num_samples_per_graph,
                num_graphs=self.config.num_graphs,
                num_timesteps=self.config.num_timesteps,
                num_actions=self.config.num_actions,
                seed=self.config.seed,
                test_fraction=self.config.test_fraction,
            )
        elif self.config.dataset_type == "disjoint":
            ds = DisjointGraphDataset(
                num_nodes=self.config.num_nodes,
                edge_prob=self.config.edge_prob,
                num_samples_per_graph=self.config.num_samples_per_graph,
                num_graphs=self.config.num_graphs,
                num_timesteps=self.config.num_timesteps,
                num_actions=self.config.num_actions,
                seed=self.config.seed,
                test_fraction=self.config.test_fraction,
            )
        else:
            raise ValueError(f"unknown dataset_type: {self.config.dataset_type}")

        return ds

    def _build_model(self, dataset) -> Any:
        if self.config.network_type == "rnn":
            return RNNGraphNetwork(
                num_nodes=dataset.num_nodes,
                num_actions=dataset.num_actions,
                hidden_dim=self.config.hidden_dim,
                seed=self.config.seed,
            )
        elif self.config.network_type == "plastic":
            world_size: int = max(1, int(self.config.world_size))
            per_rank_batch: int = int(
                math.ceil(self.config.trainer_batch_size / float(world_size))
            )
            self.logger.log(
                f"plastic per-rank batch_size = {per_rank_batch} "
                f"(global trainer_batch_size={self.config.trainer_batch_size}, world_size={world_size})",
                prefix="[~]",
            )
            return PlasticGraphNetwork(
                num_nodes=dataset.num_nodes,
                num_actions=dataset.num_actions,
                hidden_dim=self.config.hidden_dim,
                batch_size=per_rank_batch,
                seed=self.config.seed,
            )
        else:
            raise ValueError(f"unknown network_type: {self.config.network_type}")

    def _build_trainer(self, on_batch_start=None, on_epoch_end=None) -> Trainer:
        return Trainer(
            epochs=self.config.epochs,
            batch_size=self.config.trainer_batch_size,
            lr=self.config.lr,
            val_split=self.config.val_split,
            scheduler=self.config.scheduler,
            seed=self.config.seed,
            on_batch_start=on_batch_start,
            on_epoch_end=on_epoch_end,
            device=self.device,
            world_size=self.config.world_size,
            rank=self.config.rank,
        )

    def _log_parameter_counts(self, model: Any) -> Dict[str, int]:
        if hasattr(model, "parameter_counts"):
            counts = model.parameter_counts()
        else:
            slow_params = sum(p.numel() for p in model.parameters())
            counts = {"slow": int(slow_params), "fast": 0, "total": int(slow_params)}

        msg = (
            "parameter counts -> "
            f"slow: {counts['slow']}, fast: {counts['fast']}, total: {counts['total']}"
        )
        self.logger.log(msg, prefix="[~]")
        self._record_metric("num_parameters_slow", counts["slow"])
        self._record_metric("num_parameters_fast", counts["fast"])
        self._record_metric("num_parameters_total", counts["total"])
        self._wandb_config_update(
            {
                "num_parameters_slow": counts["slow"],
                "num_parameters_fast": counts["fast"],
                "num_parameters_total": counts["total"],
            }
        )
        return counts

    def train(self) -> None:
        dataset = self._build_dataset()
        self._init_wandb(dataset)
        self.logger.log(
            f"training device = {self.device} ({self.device_name})", prefix="[~]"
        )
        self.logger.log(
            f"built dataset with {len(dataset.graphs_train)} train graphs and {len(dataset.graphs_test)} test graphs",
            prefix="[+]",
        )
        self.dataset = dataset
        self._record_metric("num_train_graphs", len(dataset.graphs_train))
        self._record_metric("num_test_graphs", len(dataset.graphs_test))
        self._record_metric("num_nodes", dataset.num_nodes)
        self._record_metric("num_actions", dataset.num_actions)

        # Persist a quick graph visualization for each split.
        try:
            train_graph_path = self.run_dir / "train_graph.png"
            dataset.plot_random_graph(str(train_graph_path), split="train")
            self.logger.log(
                f"saved random train graph to {train_graph_path}", prefix="[+]"
            )
            self._record_artifact("train_graph", train_graph_path)
            self._wandb_log_image("plots/train_graph", train_graph_path)
        except Exception as exc:
            self.logger.log(f"failed to save train graph: {exc}", prefix="[-]")

        try:
            test_graph_path = self.run_dir / "test_graph.png"
            dataset.plot_random_graph(str(test_graph_path), split="test")
            self.logger.log(
                f"saved random test graph to {test_graph_path}", prefix="[+]"
            )
            self._record_artifact("test_graph", test_graph_path)
            self._wandb_log_image("plots/test_graph", test_graph_path)
        except Exception as exc:
            self.logger.log(f"failed to save test graph: {exc}", prefix="[-]")

        train_data, test_data = dataset.get()
        self.train_data = train_data
        self.test_data = test_data

        # Action histograms.
        train_counts = DatasetUtils.compute_action_counts_per_node_pair(
            data=train_data,
            num_nodes=dataset.num_nodes,
            num_actions=dataset.num_actions,
        )
        train_plot_path = self.run_dir / "train_action_hist.png"
        DatasetUtils.plot_action_hist_grid(
            train_counts,
            title="train action-id frequencies per node pair",
            output_path=str(train_plot_path),
        )
        self.logger.log(
            f"saved train action histogram to {train_plot_path}", prefix="[+]"
        )
        self._record_artifact("train_action_hist", train_plot_path)
        self._wandb_log_image("plots/train_action_hist", train_plot_path)

        test_counts = DatasetUtils.compute_action_counts_per_node_pair(
            data=test_data, num_nodes=dataset.num_nodes, num_actions=dataset.num_actions
        )
        test_plot_path = self.run_dir / "test_action_hist.png"
        DatasetUtils.plot_action_hist_grid(
            test_counts,
            title="test action-id frequencies per node pair",
            output_path=str(test_plot_path),
        )
        self.logger.log(
            f"saved test action histogram to {test_plot_path}", prefix="[+]"
        )
        self._record_artifact("test_action_hist", test_plot_path)
        self._wandb_log_image("plots/test_action_hist", test_plot_path)

        overlap = DatasetUtils.triplet_overlap_fraction(dataset)
        self.logger.log(
            f"triplet overlap fraction = {overlap * 100.0:.2f}%", prefix="[~]"
        )
        self._record_metric("triplet_overlap_fraction", overlap)

        if self.config.dataset_type == "disjoint" and overlap > 0:
            self.logger.log(
                "disjoint dataset has nonzero overlap; terminating run", prefix="[-]"
            )
            raise RuntimeError("triplet overlap is nonzero for disjoint dataset")

        train_constraints_ok = DatasetUtils.count_triplets_and_check_constraints(
            data=train_data,
            num_nodes=dataset.num_nodes,
            num_actions=dataset.num_actions,
            num_samples_per_graph=self.config.num_samples_per_graph,
        )
        test_constraints_ok = DatasetUtils.count_triplets_and_check_constraints(
            data=test_data,
            num_nodes=dataset.num_nodes,
            num_actions=dataset.num_actions,
            num_samples_per_graph=self.config.num_samples_per_graph,
        )
        self.logger.log("triplet constraints check completed", prefix="[~]")
        self._record_metric("train_constraints_ok", train_constraints_ok)
        self._record_metric("test_constraints_ok", test_constraints_ok)

        if not (train_constraints_ok and test_constraints_ok):
            self.logger.log(
                "triplet constraints violated; terminating run", prefix="[-]"
            )
            raise RuntimeError("triplet constraints violated")

        optimal_rate = DatasetUtils.optimal_prediction_rate(
            dataset=dataset,
            num_graphs=1_000,
            timesteps=self.config.num_timesteps,
            seed=self.config.seed,
        )
        self.logger.log(
            f"optimal prediction rate estimate = {optimal_rate * 100.0:.2f}%",
            prefix="[~]",
        )
        self._record_metric("optimal_prediction_rate", optimal_rate)

        self.logger.log("dataset analysis completed", prefix="[+]")

        # Training stage
        model = self._build_model(dataset)
        self._log_parameter_counts(model)
        if self.config.network_type == "plastic":
            wp_dim = (
                int(model.Wp.shape[-1])
                if hasattr(model, "Wp") and getattr(model, "Wp") is not None
                else int(self.config.hidden_dim)
            )
            self._record_metric("wp_matrix_size", f"{wp_dim} x {wp_dim}")
        on_batch_start = None
        if self.config.network_type == "plastic":
            def on_batch_start_fn(model_ref, batch_idx: int) -> None:
                inner_model = model_ref.module if hasattr(model_ref, "module") else model_ref
                inner_model.reset_plastic_weights()

            on_batch_start = on_batch_start_fn

        model = model.to(self.device)

        if self.config.distributed and self.device.type == "cuda":
            torch.cuda.set_device(self.device)  # ensure current device matches local_rank
            model = DDP(
                model,
                device_ids=[self.config.local_rank],
                output_device=self.config.local_rank,
            )

        def on_epoch_end(model_ref, epoch_idx: int, metrics: Dict[str, float]):
            log_data = {
                "train/loss": metrics.get("train_loss"),
                "val/loss": metrics.get("val_loss"),
                "train/accuracy": metrics.get("train_acc"),
                "val/accuracy": metrics.get("val_acc"),
            }
            if "lr" in metrics:
                log_data["lr"] = metrics["lr"]
            if "grad_norm" in metrics:
                log_data["grad_norm"] = metrics["grad_norm"]
            self._wandb_log(log_data)

        trainer = self._build_trainer(
            on_batch_start=on_batch_start, on_epoch_end=on_epoch_end
        )

        train_data, _ = dataset.get()
        train_start_time = time.time()
        training_result: TrainingResult = trainer.train(model, train_data)
        self.training_duration_seconds = time.time() - train_start_time
        self.training_result = training_result

        base_model = model.module if hasattr(model, "module") else model
        self.model = base_model
        if training_result.train_loss:
            self._record_metric("final_train_loss", training_result.train_loss[-1])
        if training_result.val_loss:
            self._record_metric("final_val_loss", training_result.val_loss[-1])
        if training_result.train_accuracy:
            self._record_metric(
                "final_train_accuracy", training_result.train_accuracy[-1]
            )
        if training_result.val_accuracy:
            self._record_metric("final_val_accuracy", training_result.val_accuracy[-1])
        self._record_metric("best_epoch", training_result.best_epoch)
        self._record_metric("best_val_loss", training_result.best_val_loss)

        self.logger.log("training completed", prefix="[+]")
        self.logger.log(
            f"best val loss = {training_result.best_val_loss:.4f} at epoch {training_result.best_epoch}",
            prefix="[~]",
        )
        if self.training_duration_seconds is not None:
            self.logger.log(
                f"training duration = {self._format_duration(self.training_duration_seconds)}",
                prefix="[~]",
            )

        # Persist trained model and dataset config
        if (not self.config.distributed) or self.config.rank == 0:
            model_path = self.run_dir / "model.pth"
            dataset_cfg = dataset.to_config()
            trainer_cfg = trainer.to_config()
            (self.run_dir / "dataset_config.json").write_text(json.dumps(dataset_cfg, indent=2))
            (self.run_dir / "trainer_config.json").write_text(json.dumps(trainer_cfg, indent=2))
            (self.run_dir / "run_config.json").write_text(json.dumps(asdict(self.config), indent=2))
            self.model.save(str(model_path), dataset_config=dataset_cfg)
            self.logger.log(f"saved trained model to {model_path}", prefix="[+]")
            self._record_artifact("trained_model", model_path)
            self._record_artifact("run_config", self.run_dir / "run_config.json")
            self._record_artifact("dataset_config", self.run_dir / "dataset_config.json")
            self._record_artifact("trainer_config", self.run_dir / "trainer_config.json")

        # Plot training curves
        if (not self.config.distributed) or self.config.rank == 0:
            try:
                plot_path = self._plot_training_curves(training_result)
                self.logger.log(f"saved training curves plot to {plot_path}", prefix="[+]")
                self._record_artifact("training_curves", plot_path)
                self._wandb_log_image("plots/training_curves", plot_path, step=len(training_result.train_loss))
            except Exception as exc:
                self.logger.log(f"failed to save training curves plot: {exc}", prefix="[-]")

    def generate_report(self, compile_pdf: bool = True, log_tail_lines: int = 200) -> None:
        config_for_report = self._report_config()
        builder = RunReportBuilder(
            run_dir=self.run_dir,
            config=config_for_report,
            metrics=self.report_data.get("metrics", {}),
            artifacts=self.report_data.get("artifacts", {}),
            log_path=self.log_path,
            training_result=getattr(self, "training_result", None),
        )
        try:
            tex_path = builder.write_tex(log_tail_lines=log_tail_lines)
            self.logger.log(f"wrote report latex to {tex_path}", prefix="[+]")
            self._record_artifact("report_tex", tex_path)
            if compile_pdf:
                pdf_path, compile_msg = builder.compile_pdf(tex_path)
                if pdf_path is not None:
                    self.logger.log(f"compiled pdf report to {pdf_path}", prefix="[+]")
                    self._record_artifact("report_pdf", pdf_path)
                else:
                    self.logger.log(
                        f"latex compilation failed; pdf not produced. details: {compile_msg}",
                        prefix="[-]",
                    )
        except Exception as exc:
            self.logger.log(f"failed to generate report: {exc}", prefix="[-]")
        self._log_run_artifact()

    def close(self) -> None:
        elapsed = time.time() - self.start_time
        self.logger.log(f"total elapsed seconds = {elapsed:.2f}", prefix="[~]")
        if self.wandb_run is not None and wandb is not None:
            try:
                wandb.finish()
            except Exception as exc:  # pragma: no cover
                self.logger.log(f"wandb finish failed: {exc}", prefix="[-]")
        self.logger.close()

    def _plot_training_curves(self, result: TrainingResult) -> None:
        epochs = len(result.train_loss)
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))

        x = range(1, epochs + 1)

        axes[0, 0].plot(x, result.train_loss, "b-", linewidth=2, label="training loss")
        axes[0, 0].plot(x, result.val_loss, "r-", linewidth=2, label="validation loss")
        axes[0, 0].set_xlabel("epoch")
        axes[0, 0].set_ylabel("loss")
        axes[0, 0].set_title("training and validation loss over time")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        axes[0, 1].plot(x, result.lr_history, "g-", linewidth=2)
        axes[0, 1].set_xlabel("epoch")
        axes[0, 1].set_ylabel("learning rate")
        axes[0, 1].set_title("learning rate schedule")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale("log")

        axes[0, 2].plot(x, result.train_loss, "b-", linewidth=2, label="training loss")
        axes[0, 2].plot(x, result.val_loss, "r-", linewidth=2, label="validation loss")
        axes[0, 2].set_xlabel("epoch")
        axes[0, 2].set_ylabel("loss (log scale)")
        axes[0, 2].set_title("training and validation loss (log scale)")
        axes[0, 2].set_yscale("log")
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].legend()

        axes[1, 0].plot(x, result.grad_norms, "orange", linewidth=2)
        axes[1, 0].set_xlabel("epoch")
        axes[1, 0].set_ylabel("average gradient norm")
        axes[1, 0].set_title("gradient norms over time")
        axes[1, 0].grid(True, alpha=0.3)

        generalization_gap = [v - t for v, t in zip(result.val_loss, result.train_loss)]
        axes[1, 1].plot(x, generalization_gap, "purple", linewidth=2)
        axes[1, 1].set_xlabel("epoch")
        axes[1, 1].set_ylabel("val loss - train loss")
        axes[1, 1].set_title("generalization gap")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color="black", linestyle="--", alpha=0.5)

        loss_ratio = [
            v / t if t > 0 else 1.0 for v, t in zip(result.val_loss, result.train_loss)
        ]
        axes[1, 2].plot(x, loss_ratio, "teal", linewidth=2)
        axes[1, 2].set_xlabel("epoch")
        axes[1, 2].set_ylabel("validation / training loss")
        axes[1, 2].set_title("loss ratio (overfitting indicator)")
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].axhline(y=1.0, color="black", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plot_path = self.run_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=200)
        plt.close(fig)
        return plot_path

    def evaluate(self) -> None:
        if not hasattr(self, "model") or self.model is None:
            raise RuntimeError("model not trained yet")
        if not hasattr(self, "test_data") or self.test_data is None:
            raise RuntimeError("test data not available")
        model = self.model

        # Evaluate on test set
        try:
            test_accuracy = self._evaluate_next_node_accuracy(model, self.test_data)
            self.logger.log(
                f"test next-node accuracy = {test_accuracy * 100.0:.2f}%", prefix="[~]"
            )
            self._record_metric("test_next_node_accuracy", test_accuracy)
        except Exception as exc:
            self.logger.log(f"failed to evaluate test accuracy: {exc}", prefix="[-]")

        # Rolling accuracy over a long exploration
        try:
            avg_roll, long_acc_plot = self._plot_long_exploration_rolling_accuracy(
                model,
                dataset=self.dataset,
                timesteps=self.config.long_timesteps,
                window_size=self.config.long_window_size,
                seed=self.config.seed,
            )
            self.logger.log(
                f"average rolling accuracy over long exploration = {avg_roll:.2f}%",
                prefix="[~]",
            )
            self._record_metric("avg_rolling_accuracy_single_graph", avg_roll)
            self._record_artifact("rolling_accuracy", long_acc_plot)
            self._wandb_log_image("plots/rolling_accuracy", long_acc_plot)
        except Exception as exc:
            self.logger.log(f"failed to compute rolling accuracy: {exc}", prefix="[-]")

        # Rolling accuracy across multiple graphs with boundaries
        try:
            avg_roll_multi, boundary_plot = self._plot_rolling_accuracy_with_boundaries(
                model=model,
                dataset=self.dataset,
                num_graphs=self.config.boundary_num_graphs,
                repeats_per_graph=self.config.boundary_repeats_per_graph,
                window_size=self.config.boundary_window_size,
                seed=self.config.seed,
            )
            self.logger.log(
                f"average rolling accuracy across graph set = {avg_roll_multi:.2f}%",
                prefix="[~]",
            )
            self._record_metric("avg_rolling_accuracy_graph_set", avg_roll_multi)
            self._record_artifact("rolling_accuracy_boundaries", boundary_plot)
            self._wandb_log_image("plots/rolling_accuracy_boundaries", boundary_plot)
        except Exception as exc:
            self.logger.log(
                f"failed to compute rolling accuracy across graphs: {exc}", prefix="[-]"
            )

        self._wandb_summary_update(self.report_data.get("metrics", {}))

    def _evaluate_next_node_accuracy(self, model: Any, test_data: np.ndarray) -> float:
        model.eval()

        try:
            device: torch.device = next(model.parameters()).device  # type: ignore[attr-defined]
        except StopIteration:
            device = torch.device("cpu")

        num_nodes: int = int(model.num_nodes)
        total_steps: int = 0
        correct_steps: int = 0

        with torch.no_grad():
            N: int = int(test_data.shape[0])
            T: int = int(test_data.shape[1])

            cfg_batch_size = getattr(self.config, "trainer_batch_size", None)
            fallback_batch_size = 256
            base_batch_size = getattr(
                model, "batch_size", cfg_batch_size if cfg_batch_size is not None else fallback_batch_size
            )
            limit_batch_size = cfg_batch_size if cfg_batch_size is not None else fallback_batch_size
            batch_size: int = int(min(base_batch_size, limit_batch_size, N))
            batch_size = max(1, batch_size)

            for start in range(0, N, batch_size):
                end: int = min(N, start + batch_size)
                X_batch: torch.Tensor = torch.tensor(
                    test_data[start:end], dtype=torch.float32, device=device
                )

                logits: torch.Tensor = model(X_batch)
                pred_next_nodes: torch.Tensor = logits.argmax(dim=-1)
                gt_next_nodes: torch.Tensor = X_batch[:, 1:, :num_nodes].argmax(dim=2)

                correct_steps += (pred_next_nodes == gt_next_nodes).sum().item()
                total_steps += pred_next_nodes.numel()

                # Reset plastic weights after each batch if available
                if hasattr(model, "reset_plastic_weights"):
                    model.reset_plastic_weights()

        return float(correct_steps / total_steps) if total_steps > 0 else 0.0

    def _compute_rolling_accuracy(
        self, correct_flags: np.ndarray, window_size: int
    ) -> np.ndarray:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        n: int = int(correct_flags.shape[0])
        acc: np.ndarray = np.zeros(n, dtype=np.float64)
        csum: np.ndarray = np.cumsum(correct_flags.astype(np.float64))
        for i in range(n):
            start: int = max(0, i - window_size + 1)
            total: float = float(csum[i] - (csum[start - 1] if start > 0 else 0.0))
            length: int = i - start + 1
            acc[i] = total / float(length)
        return acc

    def _plot_long_exploration_rolling_accuracy(
        self,
        model: Any,
        dataset: Any,
        timesteps: int = 10_000,
        window_size: int = 100,
        seed: Optional[int] = None,
    ) -> Tuple[float, Path]:
        try:
            device: torch.device = next(model.parameters()).device  # type: ignore[attr-defined]
        except StopIteration:
            device = torch.device("cpu")

        G: nx.Graph = dataset.create_random_graph(split="test")

        rng: np.random.Generator = np.random.default_rng(seed)

        T: int = int(timesteps)
        num_nodes: int = int(dataset.num_nodes)
        num_actions: int = int(dataset.num_actions)

        current_node: int = int(rng.choice(list(G.nodes)))
        sequence: List[np.ndarray] = []

        for _t in range(T):
            neighbors: List[int] = list(G.neighbors(current_node))
            action_vec: np.ndarray = np.zeros(num_actions, dtype=np.float32)

            if neighbors:
                next_node: int = int(rng.choice(neighbors))
                a_id: int = int(G.edges[current_node, next_node]["action_id"])
                action_vec[a_id] = 1.0
            else:
                next_node = int(rng.choice(list(G.nodes)))
                a_id = -1

            node_vec: np.ndarray = np.zeros(num_nodes, dtype=np.float32)
            node_vec[current_node] = 1.0

            X_t: np.ndarray = np.concatenate([node_vec, action_vec], axis=0)
            sequence.append(X_t)

            current_node = next_node

        X_np: np.ndarray = np.stack(sequence, axis=0)

        model.eval()
        if hasattr(model, "reset_plastic_weights"):
            model.reset_plastic_weights()

        with torch.no_grad():
            X_tensor: torch.Tensor = torch.tensor(
                X_np[None, ...], dtype=torch.float32, device=device
            )
            logits: torch.Tensor = model(X_tensor)
            probs: torch.Tensor = F.softmax(logits, dim=-1)
            preds: torch.Tensor = probs.argmax(dim=-1)

        X_torch: torch.Tensor = torch.tensor(X_np, dtype=torch.float32, device=device)
        gt_next: torch.Tensor = X_torch[1:, :num_nodes].argmax(dim=1)
        pred_next: torch.Tensor = preds.squeeze(0)

        correct_flags: np.ndarray = (
            (pred_next == gt_next).detach().cpu().numpy().astype(np.int32)
        )
        rolling_acc: np.ndarray = (
            self._compute_rolling_accuracy(correct_flags, window_size) * 100.0
        )

        steps: np.ndarray = np.arange(1, T, dtype=np.int64)
        plt.figure(figsize=(20, 16))
        plt.plot(
            steps,
            rolling_acc,
            linewidth=1.5,
            label=f"rolling accuracy (window={window_size})",
        )
        plt.axhline(
            y=100.0, linestyle="--", color="red", linewidth=2.0, label="100% reference"
        )
        plt.ylim(0.0, 110.0)
        plt.xlim(1, T - 1)
        plt.xlabel("timestep")
        plt.ylabel("accuracy (%)")
        plt.title(f"rolling next-node accuracy over {T} steps on a random graph")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plot_path = self.run_dir / "rolling_accuracy.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()

        avg_roll = float(np.mean(rolling_acc))
        return avg_roll, plot_path

    def _generate_exploration_dataframe(
        self,
        model: Any,
        dataset: Any,
        num_graphs: int,
        repeats_per_graph: int = 10,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        rng: np.random.Generator = np.random.default_rng(seed)

        try:
            device: torch.device = next(model.parameters()).device  # type: ignore[attr-defined]
        except StopIteration:
            device = torch.device("cpu")

        model.eval()
        num_nodes: int = int(model.num_nodes)
        num_actions: int = int(model.num_actions)
        T: int = int(dataset.num_timesteps)

        records: List[tuple[int, int, int, int, int]] = []

        @torch.no_grad()
        def run_one_exploration(
            G: nx.Graph,
        ) -> tuple[list[int], list[int], list[int], list[int]]:
            src_nodes: List[int] = []
            action_ids: List[int] = []
            true_nodes: List[int] = []

            current_node: int = int(rng.choice(list(G.nodes)))
            xs: List[np.ndarray] = []

            for t in range(T):
                neighbors = list(G.neighbors(current_node))
                if neighbors:
                    next_node: int = int(rng.choice(neighbors))
                    a_id: int = int(G.edges[current_node, next_node]["action_id"])
                else:
                    next_node = int(rng.choice(list(G.nodes)))
                    a_id = 0

                node_vec = np.zeros(num_nodes, dtype=np.float32)
                node_vec[current_node] = 1.0

                action_vec = np.zeros(num_actions, dtype=np.float32)
                action_vec[a_id] = 1.0

                xs.append(np.concatenate([node_vec, action_vec], axis=0))

                if t < T - 1:
                    src_nodes.append(current_node)
                    action_ids.append(a_id)
                    true_nodes.append(next_node)

                current_node = next_node

            X = torch.tensor(
                np.stack(xs, axis=0)[None, ...], dtype=torch.float32, device=device
            )
            logits = model(X)
            pred_nodes = logits.argmax(dim=-1).squeeze(0).detach().cpu().tolist()

            return src_nodes, action_ids, true_nodes, pred_nodes

        for g_idx in range(num_graphs):
            G = dataset.create_random_graph(split="test")
            if hasattr(model, "reset_plastic_weights"):
                model.reset_plastic_weights()
            for _ in range(repeats_per_graph):
                src_nodes, action_ids, true_nodes, pred_nodes = run_one_exploration(G)
                for s, a, p, v in zip(src_nodes, action_ids, pred_nodes, true_nodes):
                    records.append((g_idx, s, a, p, v))

        df = pd.DataFrame(
            records,
            columns=["graph_id", "src_node", "action_id", "pred_node", "true_node"],
        )
        return df

    def _add_rolling_accuracy(
        self,
        df: pd.DataFrame,
        window_size: int,
        by_graph: bool = True,
        percentage: bool = True,
        new_col: str = "rolling_acc",
    ) -> pd.DataFrame:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")

        work = df.copy()
        work["is_correct"] = (
            work["pred_node"].to_numpy() == work["true_node"].to_numpy()
        ).astype(np.float32)

        if by_graph:
            work[new_col] = (
                work.groupby("graph_id", sort=False)["is_correct"]
                .rolling(window=window_size, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
        else:
            work[new_col] = (
                work["is_correct"].rolling(window=window_size, min_periods=1).mean()
            )

        if percentage:
            work[new_col] = work[new_col] * 100.0

        return work

    def _plot_rolling_accuracy_with_boundaries(
        self,
        model: Any,
        dataset: Any,
        num_graphs: int,
        repeats_per_graph: int = 10,
        window_size: int = 100,
        seed: Optional[int] = None,
    ) -> Tuple[float, Path]:
        df = self._generate_exploration_dataframe(
            model=model,
            dataset=dataset,
            num_graphs=num_graphs,
            repeats_per_graph=repeats_per_graph,
            seed=seed,
        )
        df = self._add_rolling_accuracy(
            df,
            window_size=window_size,
            by_graph=True,
            percentage=True,
            new_col="rolling_acc",
        )

        if "graph_id" not in df.columns:
            raise ValueError('df must contain "graph_id" column')
        x = np.arange(len(df))
        y = df["rolling_acc"].to_numpy()
        gids = df["graph_id"].to_numpy()
        change_idx = np.flatnonzero(gids[1:] != gids[:-1]) + 1

        plt.figure(figsize=(20, 14))
        plt.plot(x, y, label="rolling accuracy")
        for idx in change_idx:
            plt.axvline(idx, color="red", linestyle="--", linewidth=1)
        plt.axhline(
            y=100.0, linestyle="--", color="red", linewidth=2.0, label="100% reference"
        )
        plt.title("rolling accuracy with graph boundaries")
        plt.xlabel("step")
        plt.ylabel("rolling accuracy (%)")
        plt.legend()
        plt.tight_layout()
        plot_path = self.run_dir / "rolling_accuracy_boundaries.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()

        avg_roll = float(np.mean(y))
        return avg_roll, plot_path
