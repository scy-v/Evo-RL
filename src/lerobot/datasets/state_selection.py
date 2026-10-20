"""Materialize selected state columns once, before sampling an offline dataset."""

import copy
import os

import datasets
import numpy as np
from torch.utils.data import IterableDataset

from lerobot.datasets.utils import get_hf_features_from_features
from lerobot.utils.constants import OBS_STATE


def resolve_state_indices(dataset_config):
    indices = dataset_config.state_indices
    if indices is None and os.environ.get("OBS_INDICES") is not None:
        try:
            indices = [int(value.strip()) - 1 for value in os.environ["OBS_INDICES"].split(",")]
        except ValueError as error:
            raise ValueError("OBS_INDICES must contain one-based comma-separated integers.") from error
    if indices is not None:
        if not indices or any(type(index) is not int or index < 0 for index in indices):
            raise ValueError("Offline state indices must be non-empty, zero-based non-negative integers.")
        dataset_config.state_indices = list(indices)
    return indices


def select_offline_state(dataset, dataset_config):
    indices = resolve_state_indices(dataset_config)
    if indices is None:
        return dataset
    if isinstance(dataset, IterableDataset):
        raise ValueError("State selection materializes the full offline dataset; set dataset.streaming=false.")

    # Finalize a lazy table first, so later reads cannot reload raw columns.
    dataset._ensure_hf_dataset_loaded()
    selected = copy.copy(dataset)
    selected.meta = copy.copy(dataset.meta)
    selected.meta.info = copy.deepcopy(dataset.meta.info)
    selected.meta.stats = copy.deepcopy(dataset.meta.stats)
    feature = selected.meta.features[OBS_STATE]
    raw_dim = feature["shape"][-1]
    if max(indices) >= raw_dim:
        raise ValueError(f"Offline state indices exceed the dataset's {raw_dim}D state.")
    feature["shape"] = (*feature["shape"][:-1], len(indices))
    if isinstance(feature.get("names"), list):
        feature["names"] = [feature["names"][index] for index in indices]
    for name, value in selected.meta.stats.get(OBS_STATE, {}).items():
        if name == "count":
            continue
        array = np.asarray(value)
        if array.ndim and array.shape[-1] == raw_dim:
            selected.meta.stats[OBS_STATE][name] = (
                value[..., indices] if hasattr(value, "shape") else array[..., indices]
            )
        elif name in {"mean", "std", "min", "max", "q01", "q99"}:
            raise ValueError(f"Dataset state statistic {name} does not match its {raw_dim}D schema.")

    # Match OpenPI/vla-precision: replace the complete HF table with map's
    # materialized result. No selection wrapper or per-sample transform remains.
    # Only feed the state column to map; video frames are not decoded or loaded.
    hf_features = copy.deepcopy(dataset.hf_dataset.features)
    if len(feature["shape"]) == 1:
        hf_features[OBS_STATE] = datasets.Sequence(datasets.Value(feature["dtype"]), length=len(indices))
    else:
        hf_features[OBS_STATE] = get_hf_features_from_features({OBS_STATE: feature})[OBS_STATE]

    def select_state_batch(states):
        return {OBS_STATE: np.asarray(states)[..., indices]}

    original_format = dataset.hf_dataset.format
    selected.hf_dataset = dataset.hf_dataset.with_format(None).map(
        select_state_batch,
        input_columns=[OBS_STATE],
        batched=True,
        features=hf_features,
        keep_in_memory=True,
        load_from_cache_file=False,
        desc="Materializing offline state indices",
    )
    selected.hf_dataset.set_format(
        type=original_format["type"],
        columns=original_format["columns"],
        output_all_columns=original_format["output_all_columns"],
        **original_format["format_kwargs"],
    )
    return selected
