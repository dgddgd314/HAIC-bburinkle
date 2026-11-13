from __future__ import annotations

import sys


def plan(instruction: str) -> list:
    """Return ordered mission numbers (1-6) or (number, params) pairs."""

    # TODO: Convert the instruction into mission numbers and parameters.
    raise NotImplementedError("custom/planner/planner.py: plan() is not implemented yet")


if __name__ == "__main__":
    for number, step in enumerate(plan(" ".join(sys.argv[1:])), 1):
        print(f"{number}. {step}")
