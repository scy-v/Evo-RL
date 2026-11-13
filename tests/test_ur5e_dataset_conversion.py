"""Verify label semantics, source preservation and native LeRobot compatibility."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.ur5e_rollout.convert_dataset import INTERVENTION, convert_dataset
from lerobot.scripts.ur5e_rollout.policy import inference_task


def legacy_dataset(root, rollout):
    features = {
        "observation.state": {"dtype": "float32", "shape": (48,), "names": None},
        "action": {"dtype": "float32", "shape": (7,), "names": None},
    }
    if rollout:
        for key in ("is_success", "intervene_flag"):
            features[key] = {"dtype": "bool", "shape": (1,), "names": [key]}
    ds = LeRobotDataset.create(
        "test/legacy", fps=15, features=features, root=root, robot_type="ur5e", use_videos=False
    )
    for ep in range(2):
        for i in range(3):
            row = {
                "observation.state": np.full(48, i, dtype=np.float32),
                "action": np.full(7, i, dtype=np.float32),
                "task": "insert",
            }
            if rollout:
                row.update(is_success=np.array([ep == 0]), intervene_flag=np.array([i == 1]))
            ds.add_frame(row)
        ds.save_episode()
    ds.finalize()


@pytest.mark.parametrize("rollout", [False, True])
def test_conversion_native_loader_and_source_untouched(tmp_path, rollout):
    source, dest = tmp_path / "source", tmp_path / "dest"
    legacy_dataset(source, rollout)
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    report = convert_dataset(
        source, dest, kind="rlinf-rollout" if rollout else "sft-success", repo_id="test/converted"
    )
    ds = LeRobotDataset("test/converted", root=dest)
    assert len(ds) == 6
    assert list(ds.meta.episodes["episode_success"]) == (
        ["success", "failure"] if rollout else ["success", "success"]
    )
    assert report["successes"] == (1 if rollout else 2)
    assert float(ds[1][INTERVENTION]) == int(rollout)
    assert ds[0][INTERVENTION].ndim == 0
    np.testing.assert_array_equal(ds[2]["action"].numpy(), np.full(7, 2))
    assert all((source / k).read_bytes() == v for k, v in before.items())
    assert "is_success" not in ds.features and "intervene_flag" not in ds.meta.stats
    with pytest.raises(FileExistsError):
        convert_dataset(
            source, dest, kind="rlinf-rollout" if rollout else "sft-success", repo_id="test/converted"
        )


def test_conflicting_per_frame_outcome_is_rejected_atomically(tmp_path):
    source, dest = tmp_path / "source", tmp_path / "dest"
    legacy_dataset(source, True)
    path = next((source / "data").rglob("*.parquet"))
    table = pq.read_table(path)
    table = table.set_column(
        table.schema.get_field_index("is_success"),
        "is_success",
        pa.array([True, False, True, False, False, False]),
    )
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="inconsistent success"):
        convert_dataset(source, dest, kind="rlinf-rollout", repo_id="test/converted")
    assert not dest.exists()
    assert not list(tmp_path.glob(".dest.converting-*"))


@pytest.mark.parametrize("enabled", [False, True])
def test_conditioning_matches_saved_training(tmp_path, enabled):
    (tmp_path / "train_config.json").write_text(json.dumps({"acp": {"enable": enabled}}))
    cfg = {"model": {"checkpoint_dir": str(tmp_path)}, "record": {"task": {"description": "insert"}}}
    assert inference_task(cfg) == ("insert\nAdvantage: positive" if enabled else "insert")
    cfg["model"]["conditioning"] = "sft" if enabled else "acp"
    with pytest.raises(ValueError, match="conflicts"):
        inference_task(cfg)


def test_unknown_conditioning_requires_explicit_choice(tmp_path):
    cfg = {"model": {"checkpoint_dir": str(tmp_path)}, "record": {"task": {"description": "insert"}}}
    with pytest.raises(ValueError, match="explicitly"):
        inference_task(cfg)
    cfg["model"]["conditioning"] = "acp"
    assert inference_task(cfg) == "insert\nAdvantage: positive"
