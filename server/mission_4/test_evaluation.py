"""Regression checks for grouping, room order and assignment-based M4 evaluation."""
import itertools
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from dataset import DOORS, ROOMS, make_split, partition, surveys
from evaluate import evaluate


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        """Create the six known layouts across three simulated recording sessions."""
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        for i, order in enumerate(itertools.permutations(ROOMS)):
            batch = self.root / str(i)
            batch.mkdir()
            rooms = dict(zip(DOORS, order, strict=True))
            for door, room in rooms.items():
                Image.new("L", (8, 8), 80 * ROOMS.index(room)).save(batch / f"{door}.jpg")
            (batch / "survey.json").write_text(json.dumps(
                {"run_id": str(i // 2), "batch_id": batch.name, "rooms": rooms}), encoding="utf-8")
        self.records = surveys(self.root)

    def test_group_split(self):
        """Reject a split that assigns related surveys to opposite sides."""
        mapping = make_split(self.records, seed=42)
        split = self.root / "split.json"
        split.write_text(json.dumps(mapping), encoding="utf-8")
        train = partition(self.root, split, "train")
        held = partition(self.root, split, "eval")
        self.assertTrue({r["run_id"] for r in train}.isdisjoint(r["run_id"] for r in held))
        mapping["0"] = "eval" if mapping["1"] == "train" else "train"
        split.write_text(json.dumps(mapping), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "leaks run"):
            partition(self.root, split, "eval")

    def test_all_layouts_and_ties(self):
        """A local classroom bias leaves globally optimal layouts unique and correct."""
        def score(image):
            """Give every door the same classroom bias plus its image-specific score."""
            room = ROOMS[round(image.getpixel((0, 0))[0] / 80)]
            return {k: 2 * (k == "classroom") + (k == room) for k in ROOMS}

        rows = list(self.records.values())
        report = evaluate(self.root, rows, score)
        self.assertEqual(report["correct_layouts"], 6)
        self.assertEqual(report["view_accuracy"], 1 / 3)
        self.assertEqual(report["confusion_true_by_pred"]["medical"]["classroom"], 6)
        tied = evaluate(self.root, rows, lambda image: dict.fromkeys(ROOMS, 0))
        self.assertEqual((tied["correct_layouts"], tied["layout_ties"], tied["layouts"]), (0, 6, 6))


if __name__ == "__main__":
    unittest.main()
