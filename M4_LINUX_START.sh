#!/usr/bin/env bash
# Linux Codex 인계 — 이 저장소에서 실제 M4 연습 영상 수집을 진행한다.
# 1. 현재 작업 트리를 보존하고 README.md의 Ubuntu 22.04 / ROS 2 Humble 설정을 확인한다.
#    기존 HAIC 환경을 우선 사용한다. 없으면 src/setup과 README에 따라 필요한 항목을
#    준비한다. sudo가 필요한 단계 또는 여러 환경 중 선택이 필요한 경우에만 사용자에게 묻는다.
# 2. bash M4_LINUX_START.sh --first 로 첫 독립 실행을 촬영한다. 출력한 배치의
#    north.jpg, middle.jpg, south.jpg를 직접 열어 문 너머가 보이는지 확인한다.
#    survey.json과 해당 실행의 active.yaml에서 북·중간·남 라벨을 대조한다.
# 3. 첫 배치가 유효하면 bash M4_LINUX_START.sh 로 여섯 순열당 두 독립 실행까지
#    수집한다. 실패하면 출력된 실행 폴더의 sim.log와 collect.log를 확인하고 재개한다.
# 4. 실제 배치·JPEG 수, 순열별 독립 실행 수, 저장 위치와 화면 검토 결과를 보고한다.
#    이 작업의 범위는 연습 PC에서의 수집이다. H100은 이후 가중치 학습에 사용한다.
set -eo pipefail
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo 'bash M4_LINUX_START.sh [--first] [--per-layout N] [--data PATH]'
    echo '기본값: ~/m4-data, 순열당 2개 독립 실행. --first는 부족한 배치 한 개만 수집.'
    exit 0
fi
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
if [[ $(uname -s) != Linux || ! -f /opt/ros/humble/setup.bash ]]; then
    echo "Linux의 ROS 2 Humble 환경이 필요합니다. 저장소 README.md의 설치 절차를 확인하세요." >&2
    exit 1
fi
if [[ ${CONDA_DEFAULT_ENV:-} != haic_* || ( -n ${HAIC_ENV:-} && ${CONDA_DEFAULT_ENV:-} != "$HAIC_ENV" ) ]]; then
    conda_bin=${CONDA_EXE:-$(command -v conda || true)}
    for candidate in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" "$HOME/miniforge3/bin/conda"; do
        [[ -n $conda_bin || ! -x $candidate ]] || conda_bin=$candidate
    done
    if [[ -z $conda_bin ]]; then echo "HAIC conda 환경을 활성화한 뒤 재실행하세요." >&2; exit 1; fi
    source "$("$conda_bin" info --base)/etc/profile.d/conda.sh"
    if [[ -z ${HAIC_ENV:-} ]]; then
        mapfile -t envs < <(conda env list | awk '$1 ~ /^haic_/ {print $1}')
        if (( ${#envs[@]} != 1 )); then
            printf 'HAIC 환경을 지정하세요 (HAIC_ENV=haic_팀명). 발견한 환경: %s\n' "${envs[*]}" >&2
            exit 1
        fi
        HAIC_ENV=${envs[0]}
    fi
    conda activate "$HAIC_ENV"
fi
source /opt/ros/humble/setup.bash
if [[ -f "$root/install/setup.bash" ]]; then source "$root/install/setup.bash"; fi
export HAIC_ROOT="$root" MLSPACES_ASSETS_DIR="$root/src/environment/models/assets/molmo"
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}" ROS_LOCALHOST_ONLY=1
cd "$root"
exec python -u - "$@" <<'PY'
import argparse
import fcntl
import itertools
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import yaml
from PIL import Image

root = Path.cwd()
src = root / "src"
m4 = root / "server/mission_4"
sys.path.insert(0, str(m4))
from dataset import DOORS, ROOMS, surveys


def collect(data, runs, order):
    """새 시뮬레이터 한 대로 촬영하고 실제 배치와 JPEG를 확인한 후 공개한다."""
    run = runs / uuid4().hex
    run.mkdir()
    print(f"실행 {run} / 북→남 목표 {order}", flush=True)
    active = run / "active.yaml"
    config = yaml.safe_load((src / "main_config.yaml").read_text())
    scenario = yaml.safe_load((src / config["scenario"]["file"]).read_text())
    scenario["room_order"] = list(order)
    scenario["panda_task"] = None  # 책상 소품도 교실과 함께 이동하도록 둔다.
    (run / "scenario.yaml").write_text(yaml.safe_dump(scenario))
    config["simulator"]["extra_args"] += ["--scenario-out", str(active)]
    config_path = run / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    cmd = [sys.executable, "-u", "main.py", "--sim-only", "--config", str(config_path),
           "--scenario", str(run / "scenario.yaml")]
    with (run / "sim.log").open("w") as log:
        sim = subprocess.Popen(cmd, cwd=src, stdout=log, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            deadline = time.monotonic() + 930
            with (run / "sim.log").open(errors="replace") as stream:
                while sim.poll() is None and time.monotonic() < deadline:
                    if "[main] simulator is up;" in stream.readline():
                        break
                    if stream.tell() == os.fstat(stream.fileno()).st_size:
                        time.sleep(0.5)
                else:
                    raise RuntimeError(f"시뮬레이터 기동 실패 또는 시간 초과: {run}/sim.log")
            if yaml.safe_load(active.read_text())["room_order"] != list(order):
                raise ValueError(f"시뮬레이터의 초기 배치가 다릅니다: {active}")
            staging = run / "captured"
            with (run / "collect.log").open("w") as log2:
                subprocess.run([sys.executable, "-u", str(m4 / "collect.py"), str(staging),
                                "--run-id", run.name, "--scenario", str(active)],
                               cwd=src, stdout=log2, stderr=subprocess.STDOUT, timeout=1200, check=True)
            batch, = surveys(staging).values()
            if (batch["run_id"] != run.name or tuple(batch["rooms"][d] for d in DOORS) != order
                    or yaml.safe_load(active.read_text())["room_order"] != list(order)):
                raise ValueError(f"수집 배치와 시나리오 불일치: {run}")
            folder = staging / batch["batch_id"]
            for door in DOORS:
                with Image.open(folder / f"{door}.jpg") as image:
                    if image.format != "JPEG" or image.mode != "RGB":
                        raise ValueError(f"RGB JPEG 확인 실패: {folder}/{door}.jpg")
                    image.load()
            result = data / folder.name
            folder.rename(result)
            print(f"검토할 배치: {result}", flush=True)
            return run.name
        finally:
            if sim.poll() is None:
                sim.send_signal(signal.SIGTERM)  # main.py가 자신이 띄운 노드까지 종료한다.
            sim.wait(timeout=90)


def main():
    """완료한 독립 실행을 재사용하고 부족한 순열만 순서대로 수집한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", action="store_true")
    parser.add_argument("--per-layout", type=int, default=2)
    parser.add_argument("--data", type=Path, default=Path.home() / "m4-data")
    args = parser.parse_args()
    if args.per_layout < 1:
        parser.error("--per-layout must be positive")
    data = args.data.expanduser().resolve()
    runs = data / ".practice-runs"
    runs.mkdir(parents=True, exist_ok=True)
    with (runs / "collector.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # 기존 ROS 네트워크와 분리한다. 이 수집기가 실행한 시뮬레이터와 로봇만 참여한다.
        os.environ["ROS_DOMAIN_ID"] = os.environ.get("M4_ROS_DOMAIN_ID", "87")
        orders = list(itertools.permutations(ROOMS))
        seen = {order: set() for order in orders}
        for batch in surveys(data).values():
            seen[tuple(batch["rooms"][d] for d in DOORS)].add(batch["run_id"])
        for n in range(1, args.per_layout + 1):
            for order in orders:
                if len(seen[order]) < n:
                    seen[order].add(collect(data, runs, order))
                    if args.first:
                        return
        print(f"완료: {data} / 순열별 독립 실행 수", {"/".join(k): len(v) for k, v in seen.items()})


if __name__ == "__main__":
    main()
PY
