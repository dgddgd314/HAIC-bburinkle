"""Compare local checkpoints on the same held-out M4 doorway surveys."""
import argparse
import json
import math
from pathlib import Path

from PIL import Image

from dataset import DOORS, ROOMS, assign_rooms, partition


def evaluate(data, rows, score):
    """Count single-view confusions and exact robot assignments, including ties."""
    confusion = {room: dict.fromkeys(ROOMS, 0) for room in ROOMS}
    exact = ties = 0
    for row in rows:
        scores = []
        truth = [row["rooms"][door] for door in DOORS]
        for door, room in zip(DOORS, truth, strict=True):
            with Image.open(data / row["batch_id"] / f"{door}.jpg") as image:
                result = score(image.convert("RGB"))
            if not all(math.isfinite(result[k]) for k in ROOMS):
                raise ValueError(f"Nonfinite scores: {row['batch_id']} {door}")
            confusion[room][max(ROOMS, key=result.__getitem__)] += 1
            scores.append(result)
        try:
            exact += assign_rooms(scores) == truth
        except ValueError as exc:
            if str(exc) != "VLM scores tie for the best layout":
                raise
            ties += 1
    correct = sum(confusion[k][k] for k in ROOMS)
    return {"confusion_true_by_pred": confusion, "view_accuracy": correct / (3 * len(rows)),
            "correct_layouts": exact, "layouts": len(rows), "layout_accuracy": exact / len(rows),
            "layout_ties": ties}


def main():
    """Print comparable reports and optionally save the winning local path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("split", type=Path)
    parser.add_argument("base", type=Path)
    parser.add_argument("--tuned", type=Path)
    parser.add_argument("--selected", type=Path)
    args = parser.parse_args()
    if args.selected and not args.tuned:
        parser.error("--selected requires --tuned")
    from model import load_scorer

    rows = partition(args.data, args.split, "eval")
    best = None
    for path in [args.base] + ([args.tuned] if args.tuned else []):
        report = evaluate(args.data, rows, load_scorer(path))
        print(json.dumps({"checkpoint": str(path.resolve()), **report}, indent=2), flush=True)
        rank = report["correct_layouts"], report["view_accuracy"]
        if best is None or rank > best:
            best, winner = rank, path.resolve()
    if args.selected:
        args.selected.write_text(str(winner) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
