from __future__ import annotations

import os
from pathlib import Path
import argparse
import yaml

import torch
import torch.distributed as dist

from core.utils.run import Run, RunConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run graph training pipeline.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file."
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg_data = yaml.safe_load(config_path.read_text())

    # Flatten sections into RunConfig kwargs
    cfg_sections = {
        **cfg_data.get("global", {}),
        **cfg_data.get("dataset", {}),
        **cfg_data.get("network", {}),
        **cfg_data.get("evaluation", {}),
        **cfg_data.get("logging", {}),
    }

    trainer_cfg = cfg_data.get("trainer", {})
    trainer_batch_size = trainer_cfg.get(
        "batch_size", trainer_cfg.get("trainer_batch_size")
    )
    for key, value in trainer_cfg.items():
        if key in ("batch_size", "trainer_batch_size"):
            continue
        cfg_sections[key] = value
    if trainer_batch_size is not None:
        cfg_sections["trainer_batch_size"] = trainer_batch_size
    # Drop legacy network.batch_size; plastic batches derive from trainer_batch_size.
    cfg_sections.pop("batch_size", None)

    distributed: bool = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank: int = int(os.environ.get("RANK", "0"))
    local_rank: int = int(os.environ.get("LOCAL_RANK", "0"))
    world_size: int = int(os.environ.get("WORLD_SIZE", "1"))

    if distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        backend: str = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    cfg_sections["distributed"] = distributed
    cfg_sections["rank"] = rank
    cfg_sections["world_size"] = world_size
    cfg_sections["local_rank"] = local_rank
    config = RunConfig(**cfg_sections)

    run = Run(config=config, root=Path.cwd())
    try:
        run.train()
        if (not distributed) or rank == 0:
            run.evaluate()
            run.generate_report()
    finally:
        run.close()
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
