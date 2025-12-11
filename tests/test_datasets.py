import json
from pathlib import Path

import numpy as np

from core.dataset.standard import StandardGraphDataset
from core.dataset.disjoint import DisjointGraphDataset
from core.utils.dataset import DatasetUtils


def test_standard_constraints_ok():
    ds = StandardGraphDataset(
        num_nodes=6,
        edge_prob=0.3,
        num_samples_per_graph=2,
        num_graphs=5,
        num_timesteps=8,
        num_actions=6,
        seed=123,
    )
    train_data, test_data = ds.get()

    assert DatasetUtils.count_triplets_and_check_constraints(
        data=train_data,
        num_nodes=ds.num_nodes,
        num_actions=ds.num_actions,
        num_samples_per_graph=ds.num_samples_per_graph,
    )
    assert DatasetUtils.count_triplets_and_check_constraints(
        data=test_data,
        num_nodes=ds.num_nodes,
        num_actions=ds.num_actions,
        num_samples_per_graph=ds.num_samples_per_graph,
    )


def test_disjoint_constraints_and_overlap_zero():
    ds = DisjointGraphDataset(
        num_nodes=6,
        edge_prob=0.25,
        num_samples_per_graph=2,
        num_graphs=4,
        num_timesteps=8,
        num_actions=6,
        seed=321,
    )
    train_data, test_data = ds.get()

    assert DatasetUtils.count_triplets_and_check_constraints(
        data=train_data,
        num_nodes=ds.num_nodes,
        num_actions=ds.num_actions,
        num_samples_per_graph=ds.num_samples_per_graph,
    )
    assert DatasetUtils.count_triplets_and_check_constraints(
        data=test_data,
        num_nodes=ds.num_nodes,
        num_actions=ds.num_actions,
        num_samples_per_graph=ds.num_samples_per_graph,
    )

    overlap = DatasetUtils.triplet_overlap_fraction(ds)
    assert overlap == 0.0


def test_standard_serialization_is_deterministic(tmp_path: Path):
    ds = StandardGraphDataset(
        num_nodes=5,
        edge_prob=0.4,
        num_samples_per_graph=3,
        num_graphs=3,
        num_timesteps=6,
        num_actions=5,
        seed=999,
    )
    train_1, test_1 = ds.get()

    cfg_path = tmp_path / "standard_cfg.json"
    cfg: dict = ds.to_config()
    cfg_path.write_text(json.dumps(cfg))

    cfg_loaded = json.loads(cfg_path.read_text())
    ds_reloaded = StandardGraphDataset.from_config(cfg_loaded)
    train_2, test_2 = ds_reloaded.get()

    assert np.array_equal(train_1, train_2)
    assert np.array_equal(test_1, test_2)
