"""Record one complete survey in a fixed practice scenario on the simulator PC."""
import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main():
    """Capture all three JPEGs before attaching the practice scenario labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--run-id", required=True, help="Shared by all surveys of one sim session")
    parser.add_argument("--scenario", required=True, type=Path, help="Simulator --scenario-out file")
    args = parser.parse_args()

    import yaml
    from custom.mission_4.mission_4 import DOOR_Y, Mission4, Navigator
    from custom.mission_4.room_scores import ROOMS
    from mission_api import MissionRunner

    with MissionRunner("m4_practice_recorder") as runner:
        runner.api.wait_for_sim()
        mission = runner.start(Mission4)
        try:
            mission.begin_mission(4)
            version = args.scenario.stat().st_mtime_ns
            nav = Navigator(mission, interrupted=mission.respawned)
            views = dict(mission.doorway_views(nav))
            snapshot = args.scenario.read_bytes()
            if mission.respawned.is_set() or args.scenario.stat().st_mtime_ns != version:
                raise ValueError("Practice layout changed during survey. Repeat all three doors.")
        finally:
            mission.stop()
            runner.finish(mission)

    rooms = yaml.safe_load(snapshot)["room_order"]
    if sorted(rooms) != sorted(ROOMS):
        raise ValueError("Invalid practice room_order")
    batch = uuid4().hex
    args.data.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.data) as temp:
        folder = Path(temp) / batch
        folder.mkdir()
        labels = dict(zip(("north", "middle", "south"), rooms, strict=True))
        for door, y in zip(labels, DOOR_Y, strict=True):
            (folder / f"{door}.jpg").write_bytes(views[y])
        record = {"run_id": args.run_id, "batch_id": batch, "rooms": labels}
        (folder / "survey.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        folder.rename(args.data / batch)
    print(args.data / batch)


if __name__ == "__main__":
    main()
