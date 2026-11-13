"""Convert legacy UR LeRobot v3 datasets into the minimal Evo-RL rollout schema.

Videos and original observations/actions are copied, never re-encoded. No robot or
model is loaded. A new destination is required; publication is an atomic rename.
"""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats

INTERVENTION = "complementary_info.is_intervention"
FEATURE = {"dtype": "float32", "shape": [1], "names": ["is_intervention"]}
LEGACY = ("is_success", "intervene_flag")


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def binary_column(table, key):
    values = np.asarray(table[key].to_pylist())
    if values.shape not in ((len(table),), (len(table), 1)) or not np.isin(values, [0, 1]).all():
        raise ValueError(f"{key}: expected binary [N,1] values, got {values.shape}")
    return values.reshape(-1).astype(np.float32)


def convert_dataset(source: Path, destination: Path, *, kind: str, repo_id: str) -> dict:
    source, destination = source.expanduser().resolve(), destination.expanduser().absolute()
    if kind not in ("sft-success", "rlinf-rollout"):
        raise ValueError("Unsupported dataset kind")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    if destination.resolve().is_relative_to(source):
        raise ValueError("Destination must not be inside the source dataset")
    info = read_json(source / "meta/info.json")
    if info["codebase_version"] != "v3.0":
        raise ValueError("Only LeRobot v3.0 datasets are supported")
    features = info["features"]
    if INTERVENTION in features or any(k.startswith("complementary_info.") for k in features):
        raise ValueError("Expected an unconverted source without existing Evo-RL annotations")
    if kind == "sft-success" and any(k in features for k in LEGACY):
        raise ValueError("SFT success assumption must not overwrite existing rollout labels")
    if kind == "rlinf-rollout" and not all(k in features for k in LEGACY):
        raise ValueError("RLinf rollout requires is_success and intervene_flag")
    if features["action"]["shape"] != [7] or features["observation.state"]["shape"] != [48]:
        raise ValueError("Expected original UR 48-state / 7-action dataset")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.converting-", dir=destination.parent))
    try:
        shutil.copytree(source, temporary, dirs_exist_ok=True)
        by_episode = {}
        next_index = 0
        for path in sorted((temporary / "data").rglob("*.parquet")):
            table = pq.read_table(path)
            indices = np.asarray(table["index"])
            if not np.array_equal(indices, np.arange(next_index, next_index + len(table))):
                raise ValueError(f"Noncontiguous absolute indices in {path}")
            next_index += len(table)
            flags = (
                binary_column(table, "intervene_flag")
                if kind == "rlinf-rollout"
                else np.zeros(len(table), dtype=np.float32)
            )
            success = (
                binary_column(table, "is_success")
                if kind == "rlinf-rollout"
                else np.ones(len(table), dtype=np.float32)
            )
            ep_ids = np.asarray(table["episode_index"])
            frame_ids = np.asarray(table["frame_index"])
            for ep in np.unique(ep_ids):
                mask = ep_ids == ep
                entry = by_episode.setdefault(
                    int(ep), {"flags": [], "success": bool(success[mask][0]), "length": 0}
                )
                if not (success[mask] == entry["success"]).all():
                    raise ValueError(
                        f"Episode {ep}: inconsistent success labels; cannot infer episode outcome"
                    )
                if not np.array_equal(
                    frame_ids[mask], np.arange(entry["length"], entry["length"] + mask.sum())
                ):
                    raise ValueError(f"Episode {ep}: noncontiguous frame_index")
                entry["flags"].extend(flags[mask].tolist())
                entry["length"] += int(mask.sum())
            table = table.drop([k for k in LEGACY if k in table.column_names])
            table = table.append_column(INTERVENTION, pa.array(flags, type=pa.float32()))
            pq.write_table(table.replace_schema_metadata(None), path, compression="snappy")
        if next_index != info["total_frames"] or len(by_episode) != info["total_episodes"]:
            raise ValueError("Frame/episode count does not match info.json")
        episode_stats = {}
        seen = set()
        for path in sorted((temporary / "meta/episodes").rglob("*.parquet")):
            table = pq.read_table(path)
            table = table.drop(
                [k for k in table.column_names if any(k.startswith(f"stats/{old}/") for old in LEGACY)]
            )
            outcomes, stats_rows = [], []
            for row in table.to_pylist():
                ep = row["episode_index"]
                if ep in seen:
                    raise ValueError(f"Duplicate episode metadata: {ep}")
                seen.add(ep)
                data = by_episode[ep]
                if (
                    row["length"] != data["length"]
                    or row["dataset_to_index"] - row["dataset_from_index"] != data["length"]
                ):
                    raise ValueError(f"Episode {ep}: metadata length mismatch")
                outcome = "success" if data["success"] else "failure"
                if row.get("episode_success", outcome) != outcome:
                    raise ValueError(f"Episode {ep}: existing outcome conflicts with conversion")
                outcomes.append(outcome)
                stats = compute_episode_stats(
                    {INTERVENTION: np.asarray(data["flags"], dtype=np.float32)[:, None]},
                    {INTERVENTION: FEATURE},
                )
                episode_stats[ep] = stats
                stats_rows.append(stats[INTERVENTION])
            if "episode_success" in table.column_names:
                table = table.drop(["episode_success"])
            table = table.append_column("episode_success", pa.array(outcomes, type=pa.string()))
            for key in stats_rows[0]:
                table = table.append_column(
                    f"stats/{INTERVENTION}/{key}", pa.array([s[key].tolist() for s in stats_rows])
                )
            pq.write_table(table.replace_schema_metadata(None), path, compression="snappy")
        if seen != set(by_episode):
            raise ValueError("Episode metadata does not cover all recorded episodes")
        for k in LEGACY:
            features.pop(k, None)
        features[INTERVENTION] = FEATURE
        if "repo_id" in info:
            info["repo_id"] = repo_id
        write_json(temporary / "meta/info.json", info)
        stats = read_json(temporary / "meta/stats.json")
        for k in LEGACY:
            stats.pop(k, None)
        stats[INTERVENTION] = {
            k: v.tolist() for k, v in aggregate_stats(list(episode_stats.values()))[INTERVENTION].items()
        }
        write_json(temporary / "meta/stats.json", stats)
        report = {
            "source": str(source),
            "target_repo_id": repo_id,
            "kind": kind,
            "source_info_sha256": hashlib.sha256((source / "meta/info.json").read_bytes()).hexdigest(),
            "episodes": len(seen),
            "frames": next_index,
            "successes": sum(d["success"] for d in by_episode.values()),
            "intervened_episodes": sum(any(d["flags"]) for d in by_episode.values()),
            "sft_all_success_assumed": kind == "sft-success",
            "policy_action_available": False,
            "notes": "Original videos/actions/state preserved. No policy actions or advantages fabricated. Source sidecars retained.",
        }
        write_json(temporary / "meta/dataset_provenance.json", report)
        temporary.rename(destination)
        return report
    except BaseException:
        shutil.rmtree(temporary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--kind",
        choices=["sft-success", "rlinf-rollout"],
        required=True,
        help="sft-success explicitly declares every source demonstration successful",
    )
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            convert_dataset(args.source, args.destination, kind=args.kind, repo_id=args.repo_id),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
