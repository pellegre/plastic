import torch
import numpy as np

from core.base.trainer import Trainer
from core.dataset.standard import StandardGraphDataset
from core.network.rnn import RNNGraphNetwork
from core.network.plastic import PlasticGraphNetwork
import json


def _reset_all_seeds(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _compare_results(res1, res2):
    assert res1.train_loss == res2.train_loss
    assert res1.val_loss == res2.val_loss
    assert res1.train_accuracy == res2.train_accuracy
    assert res1.val_accuracy == res2.val_accuracy
    assert res1.lr_history == res2.lr_history
    assert res1.grad_norms == res2.grad_norms
    assert res1.best_epoch == res2.best_epoch
    assert res1.best_val_loss == res2.best_val_loss
    assert res1.config == res2.config
    assert res1.best_state_dict.keys() == res2.best_state_dict.keys()
    for k in res1.best_state_dict:
        assert torch.equal(res1.best_state_dict[k], res2.best_state_dict[k])


def _train_rnn_once(seed: int):
    _reset_all_seeds(seed)
    ds = StandardGraphDataset(
        num_nodes=5,
        edge_prob=0.3,
        num_samples_per_graph=2,
        num_graphs=4,
        num_timesteps=6,
        num_actions=5,
        seed=seed,
    )
    train_data, _ = ds.get()

    model = RNNGraphNetwork(
        num_nodes=ds.num_nodes,
        num_actions=ds.num_actions,
        hidden_dim=12,
        seed=seed,
    )

    trainer = Trainer(
        epochs=3,
        batch_size=2,
        lr=1e-3,
        val_split=0.25,
        scheduler="none",
        seed=seed,
    )
    return trainer.train(model, train_data)


def _train_plastic_once(seed: int):
    _reset_all_seeds(seed)
    ds = StandardGraphDataset(
        num_nodes=5,
        edge_prob=0.3,
        num_samples_per_graph=2,
        num_graphs=4,
        num_timesteps=6,
        num_actions=5,
        seed=seed,
    )
    train_data, _ = ds.get()

    model = PlasticGraphNetwork(
        num_nodes=ds.num_nodes,
        num_actions=ds.num_actions,
        hidden_dim=10,
        batch_size=2,
        seed=seed,
    )

    trainer = Trainer(
        epochs=3,
        batch_size=2,
        lr=1e-3,
        val_split=0.25,
        scheduler="none",
        seed=seed,
        on_batch_start=lambda m, i: m.reset_plastic_weights(),  # ensure consistent resets
    )
    return trainer.train(model, train_data)


def test_rnn_training_reproducible():
    res1 = _train_rnn_once(seed=1234)
    res2 = _train_rnn_once(seed=1234)
    _compare_results(res1, res2)


def test_plastic_training_reproducible():
    res1 = _train_plastic_once(seed=5678)
    res2 = _train_plastic_once(seed=5678)
    _compare_results(res1, res2)


def test_configs_recreate_training(tmp_path):
    seed = 2024
    # Build initial dataset/trainer and train RNN
    _reset_all_seeds(seed)
    ds = StandardGraphDataset(
        num_nodes=5,
        edge_prob=0.3,
        num_samples_per_graph=2,
        num_graphs=4,
        num_timesteps=6,
        num_actions=5,
        seed=seed,
    )
    trainer = Trainer(
        epochs=2,
        batch_size=2,
        lr=1e-3,
        val_split=0.25,
        scheduler="none",
        seed=seed,
    )

    train_data, _ = ds.get()
    rnn = RNNGraphNetwork(
        num_nodes=ds.num_nodes, num_actions=ds.num_actions, hidden_dim=8, seed=seed
    )
    res1 = trainer.train(rnn, train_data)

    # Save configs
    ds_cfg_path = tmp_path / "ds.json"
    tr_cfg_path = tmp_path / "tr.json"
    ds_cfg_path.write_text(json.dumps(ds.to_config()))
    tr_cfg_path.write_text(json.dumps(trainer.to_config()))

    # Reload dataset/trainer from config
    ds_cfg_loaded = json.loads(ds_cfg_path.read_text())
    tr_cfg_loaded = json.loads(tr_cfg_path.read_text())

    ds2 = StandardGraphDataset.from_config(ds_cfg_loaded)
    trainer2 = Trainer.from_config(tr_cfg_loaded)

    train_data2, _ = ds2.get()
    rnn2 = RNNGraphNetwork(
        num_nodes=ds2.num_nodes, num_actions=ds2.num_actions, hidden_dim=8, seed=seed
    )
    _reset_all_seeds(seed)
    res2 = trainer2.train(rnn2, train_data2)

    _compare_results(res1, res2)

    # Repeat for plastic
    reset_hook = lambda m, i: m.reset_plastic_weights()

    _reset_all_seeds(seed)
    plastic = PlasticGraphNetwork(
        num_nodes=ds.num_nodes,
        num_actions=ds.num_actions,
        hidden_dim=8,
        batch_size=2,
        seed=seed,
    )
    trainer_plastic = Trainer(
        epochs=2,
        batch_size=2,
        lr=1e-3,
        val_split=0.25,
        scheduler="none",
        seed=seed,
        on_batch_start=reset_hook,
    )
    _reset_all_seeds(seed)
    res3 = trainer_plastic.train(plastic, train_data)

    ds_cfg_path.write_text(json.dumps(ds.to_config()))
    tr_cfg_path.write_text(json.dumps(trainer_plastic.to_config()))
    ds_cfg_loaded = json.loads(ds_cfg_path.read_text())
    tr_cfg_loaded = json.loads(tr_cfg_path.read_text())

    _reset_all_seeds(seed)
    ds3 = StandardGraphDataset.from_config(ds_cfg_loaded)
    trainer3 = Trainer.from_config(tr_cfg_loaded, on_batch_start=reset_hook)

    train_data3, _ = ds3.get()
    _reset_all_seeds(seed)
    plastic2 = PlasticGraphNetwork(
        num_nodes=ds3.num_nodes,
        num_actions=ds3.num_actions,
        hidden_dim=8,
        batch_size=2,
        seed=seed,
    )
    _reset_all_seeds(seed)
    res4 = trainer3.train(plastic2, train_data3)

    _compare_results(res3, res4)
