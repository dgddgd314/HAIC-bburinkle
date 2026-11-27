"""M4 VLM client and north-to-south room assignment."""
import json
import math
from itertools import permutations
from urllib.request import Request, urlopen

ROOMS = ("classroom", "medical", "cafeteria")


def request_scores(url, jpeg):
    """Read one finite VLM score per room for a doorway image."""
    request = Request(url, data=jpeg, headers={"Content-Type": "image/jpeg"})
    with urlopen(request, timeout=30) as response:
        scores = json.load(response)
    if not isinstance(scores, dict) or any(
        type(scores.get(room)) not in (float, int) or not math.isfinite(scores[room])
        for room in ROOMS
    ):
        raise ValueError("Expected a finite score for each room")
    return scores


def assign_rooms(scores):
    """Choose the unique highest-scoring layout across all six permutations."""
    layouts = sorted((sum(row[name] for row, name in zip(scores, layout, strict=True)), layout)
                     for layout in permutations(ROOMS))
    if layouts[-1][0] == layouts[-2][0]:
        raise ValueError("VLM scores tie for the best layout")
    return list(layouts[-1][1])
