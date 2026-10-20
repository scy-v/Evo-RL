from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import datasets
import numpy as np
import pytest
import torch
from torch.utils.data import IterableDataset

from lerobot.configs.default import DatasetConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.state_selection import select_offline_state
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.processor.converters import batch_to_transition


class Metadata:
    def __init__(self):
        self.info = {'features': {'observation.state': {'shape': (4,), 'names': ['a', 'b', 'c', 'd'], 'dtype': 'float32'}}}
        self.stats = {'observation.state': {'q01': np.arange(4.), 'q99': np.arange(4.) + 10, 'count': np.array([2])}}
        self.camera_keys = []

    @property
    def features(self):
        return self.info['features']


class Offline:
    def __init__(self):
        self.meta = Metadata()
        self.hf_dataset = datasets.Dataset.from_list([
            {'observation.state': (np.arange(4.) + i * 10).tolist(), 'action': [1., 1.],
             'next.reward': 1., 'next.done': i == 1, 'episode_index': 0, 'task': 'insert'}
            for i in range(2)
        ]).with_format('torch')

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, index):
        return self.hf_dataset[index]


    def _ensure_hf_dataset_loaded(self):
        pass


class StreamingOffline(Offline, IterableDataset):
    def __iter__(self):
        yield from self.hf_dataset


def test_offline_selection_schema_stats_and_original_are_consistent(monkeypatch):
    monkeypatch.setenv('OBS_INDICES', '3,1')
    original = Offline()
    before = deepcopy(original.meta.stats)
    cfg = DatasetConfig(repo_id='local/test')
    selected = select_offline_state(original, cfg)
    assert cfg.state_indices == [2, 0]
    assert selected[0]['observation.state'].tolist() == [2, 0]
    assert selected.meta.features['observation.state']['names'] == ['c', 'a']
    assert selected.meta.features['observation.state']['shape'] == (2,)
    np.testing.assert_array_equal(selected.meta.stats['observation.state']['q01'], [2, 0])
    np.testing.assert_array_equal(selected.meta.stats['observation.state']['count'], [2])
    np.testing.assert_array_equal(original.meta.stats['observation.state']['q01'], before['observation.state']['q01'])
    assert original[0]['observation.state'].shape == (4,)
    torch.testing.assert_close(selected[0]['action'], original[0]['action'])
    assert type(selected) is type(original)
    assert selected.hf_dataset is not original.hf_dataset
    assert selected.hf_dataset.with_format(None)['observation.state'] == [[2., 0.], [12., 10.]]


def test_factory_applies_selection_before_exposing_training_features(monkeypatch):
    monkeypatch.setenv('OBS_INDICES', '3,1')
    policy = PI05Config(device='cpu', input_features={'observation.state': PolicyFeature(FeatureType.STATE, (4,))})
    cfg = SimpleNamespace(dataset=DatasetConfig(repo_id='local/test', use_imagenet_stats=False), policy=policy, tolerance_s=1e-4)
    with patch('lerobot.datasets.factory.LeRobotDatasetMetadata', return_value=Metadata()), patch(
        'lerobot.datasets.factory.resolve_delta_timestamps', return_value=None
    ), patch('lerobot.datasets.factory.LeRobotDataset', return_value=Offline()):
        selected = make_dataset(cfg)
    assert selected[0]['observation.state'].tolist() == [2, 0]
    assert cfg.policy.input_features['observation.state'].shape == (2,)


@pytest.mark.parametrize('value', ['0,1', '-1,2', '1,', 'a,2', '5'])
def test_invalid_offline_indices_fail(monkeypatch, value):
    monkeypatch.setenv('OBS_INDICES', value)
    with pytest.raises(ValueError):
        select_offline_state(Offline(), DatasetConfig(repo_id='local/test'))


def test_no_selection_by_default_and_explicit_dataset_config_wins(monkeypatch):
    monkeypatch.delenv('OBS_INDICES', raising=False)
    original = Offline()
    assert select_offline_state(original, DatasetConfig(repo_id='local/test')) is original
    monkeypatch.setenv('OBS_INDICES', 'bad')
    selected = select_offline_state(original, DatasetConfig(repo_id='local/test', state_indices=[1]))
    assert selected[0]['observation.state'].tolist() == [1]
    assert not hasattr(PI05Config(device='cpu'), 'state_indices')
    # Inference preparation never interprets the offline environment variable.
    out = Pi05PrepareStateTokenizerProcessorStep()(batch_to_transition({
        'observation.state': torch.tensor([[0.1, 0.2]]), 'task': ['insert']
    }))
    assert out is not None


def test_selection_requires_materialized_offline_data():
    with pytest.raises(ValueError, match="streaming=false"):
        select_offline_state(StreamingOffline(), DatasetConfig(repo_id='local/test', state_indices=[2, 0]))


def test_materialization_maps_once_and_retains_order_and_duplicates():
    original = Offline()
    original_map = datasets.Dataset.map
    calls = []

    def track_map(table, *args, **kwargs):
        calls.append(1)
        return original_map(table, *args, **kwargs)

    with patch.object(datasets.Dataset, 'map', track_map):
        selected = select_offline_state(original, DatasetConfig(repo_id='local/test', state_indices=[2, 0, 2]))
        assert calls == [1]
        for _ in range(3):
            assert selected[1]['observation.state'].tolist() == [12., 10., 12.]
        assert calls == [1]
    assert selected.hf_dataset.with_format(None)['observation.state'] == [[2., 0., 2.], [12., 10., 12.]]


def test_offline_buffer_import_selects_both_current_and_next_states():
    from lerobot.rl.buffer import ReplayBuffer

    dataset = select_offline_state(Offline(), DatasetConfig(repo_id='local/test', state_indices=[2, 0]))
    transitions = ReplayBuffer._lerobotdataset_to_transitions(dataset, state_keys=['observation.state'])
    torch.testing.assert_close(transitions[0]['state']['observation.state'], torch.tensor([[2., 0.]]))
    torch.testing.assert_close(transitions[0]['next_state']['observation.state'], torch.tensor([[12., 10.]]))
