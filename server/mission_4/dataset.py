"""Freeze complete M4 practice runs into train and evaluation sets."""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from custom.mission_4.room_scores import ROOMS, assign_rooms

DOORS = ("north", "middle", "south")


def surveys(data):
    """Load complete labeled triplets and reject inconsistent survey metadata."""
    records = {}
    for path in sorted(data.glob("*/survey.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        batch = path.parent.name
        if (row["batch_id"] != batch or not row["run_id"].strip()
                or set(row["rooms"]) != set(DOORS)
                or sorted(row["rooms"].values()) != sorted(ROOMS)):
            raise ValueError(f"Invalid survey metadata: {path}")
        if not all((path.parent / f"{door}.jpg").is_file() for door in DOORS):
            raise ValueError(f"Incomplete survey: {path}")
        records[batch] = row
    return records


def make_split(records, seed):
    """Shuffle whole simulator sessions, keeping both partitions nonempty."""
    runs = sorted({r["run_id"] for r in records.values()})
    if len(runs) < 2:
        raise ValueError("Collect at least two independent simulator runs")
    random.Random(seed).shuffle(runs)
    held_out = set(runs[:max(1, round(len(runs) * 0.2))])
    return {batch: "eval" if row["run_id"] in held_out else "train"
            for batch, row in records.items()}


def partition(data, split, name):
    """Read the frozen batch list and check that runs cannot cross partitions."""
    records = surveys(data)
    mapping = json.loads(split.read_text(encoding="utf-8"))
    by_run = {}
    if set(mapping.values()) != {"train", "eval"}:
        raise ValueError("Expected nonempty train and eval partitions")
    for batch, part in mapping.items():
        run = records[batch]["run_id"]
        if by_run.setdefault(run, part) != part:
            raise ValueError(f"Split leaks run {run}")
    return [records[batch] for batch in mapping if mapping[batch] == name]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("split", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    mapping = make_split(surveys(args.data), args.seed)
    with args.split.open("x", encoding="utf-8") as output:
        json.dump(mapping, output, indent=2)
