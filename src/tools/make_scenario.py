"""Draw the competition scenario: shuffle the scene N times, save the last draw."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import main as runner_main  # noqa: E402  (src/main.py: the simulator command and config)
from main import TerminalKeys  # noqa: E402
from mission_api import MissionRunner  # noqa: E402
from ros2_bridge.tools.scenario import Scenario  # noqa: E402


class ScenarioFile:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.unlink(missing_ok=True)
        self._seen: Optional[float] = None

    def wait(self, timeout: float = 30.0) -> Scenario:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                stamp = self.path.stat().st_mtime_ns
            except FileNotFoundError:
                stamp = None
            if stamp is not None and stamp != self._seen:
                time.sleep(0.05)
                try:
                    scenario = Scenario.load(self.path)
                except Exception:                      # noqa: BLE001 - half-written file
                    continue
                self._seen = stamp
                return scenario
            time.sleep(0.1)
        raise SystemExit(f"[scenario] the simulator wrote no scenario to {self.path} within {timeout:.0f} s")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--times", type=int, default=10,
                    help="how many random draws before the last one is saved (0: save the current)")
    ap.add_argument("--out", type=Path, default=SRC_ROOT / "scenarios" / "competition.yaml",
                    help="the scenario file to write")
    ap.add_argument("--interval", type=float, default=0.3,
                    help="seconds between the draws")
    ap.add_argument("--config", type=Path, default=runner_main.CONFIG_FILE,
                    help="main_config.yaml to start the simulator with")
    ap.add_argument("--headless", action="store_true", help="no MuJoCo window")
    ap.add_argument("--keep", action="store_true",
                    help="keep the simulator running after saving (Ctrl-C stops it)")
    ap.add_argument("--auto", action="store_true",
                    help="do not wait for SPACE: draw at once and save")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.times < 0:
        raise SystemExit("--times must be >= 0")
    sim_args = runner_main.parse_args(["--config", str(args.config)])
    if args.headless:
        sim_args.view = False
    import tempfile

    scenario_file = ScenarioFile(Path(tempfile.gettempdir()) / f"haic_scenario_{os.getpid()}.yaml")
    sim_args.sim_arg = [*sim_args.sim_arg, "--scenario-out", str(scenario_file.path)]
    sim = None
    try:
        sim = runner_main.start_simulator(sim_args)
        with MissionRunner("scenario_maker") as runner, TerminalKeys() as keys:
            api = runner.api
            print("[scenario] waiting for the simulator ...", flush=True)
            deadline = time.monotonic() + 900.0
            while api.last_status() is None:
                if sim.poll() is not None:
                    raise SystemExit("[scenario] the simulator stopped before it was up (see its output)")
                if time.monotonic() > deadline:
                    raise SystemExit("[scenario] no simulator within 900 s")
                time.sleep(0.5)
            env = scenario_file.wait()
            print(f"[scenario] start: {env.summary()}", flush=True)
            manual = keys.enabled and not args.auto
            if manual:
                print(f"[scenario] SPACE = draw {args.times} random scenario(s) in a row, q = quit",
                      flush=True)
                if keys.wait_for(" qQ", alive=lambda: sim.poll() is None) in ("q", "Q"):
                    print("[scenario] quit, nothing saved", flush=True)
                    return 1
            for number in range(1, args.times + 1):
                if sim.poll() is not None:
                    raise SystemExit("[scenario] the simulator stopped")
                if number > 1:
                    time.sleep(args.interval)
                api.randomize()
                env = scenario_file.wait()
                print(f"[scenario] draw {number}/{args.times}: {env.summary()}", flush=True)
            if manual:
                print("[scenario] SPACE = save this scenario and stop, q = quit without saving", flush=True)
                if keys.wait_for(" qQ", alive=lambda: sim.poll() is None) in ("q", "Q"):
                    print("[scenario] quit, nothing saved", flush=True)
                    return 1
            note = f"{args.times} random draw(s), the last one kept" if args.times else "the current scenario"
            path = env.save(args.out, note)
            print(f"[scenario] saved {path}", flush=True)
            print(f"[scenario]   {env.summary()}", flush=True)
            print("[scenario] main_config.yaml -> scenario.file: "
                  + str(path.relative_to(SRC_ROOT) if path.is_relative_to(SRC_ROOT) else path), flush=True)
            if args.keep:
                print("[scenario] simulator left running; Ctrl-C stops it", flush=True)
                while sim.poll() is None:
                    time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[scenario] interrupted", flush=True)
    finally:
        runner_main.stop_all([(sim, "simulator")])
        scenario_file.path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
