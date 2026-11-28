# M4 문 앞 영상 수집과 모델 선택

이미 학습한 모델로 연습하려면 [실험 결과와 팀 실행 안내](RESULTS.md#공유-모델-실행)부터 진행한다. 아래는 수집부터 학습을 재현할 때의 절차다.

아래는 Linux Bash 명령이며 각 장비에서 이 브랜치의 저장소 루트를 기준으로 한다. 시뮬레이터 PC에서 수집하고, 학습 전용 서버에서 가중치를 만든 뒤, 연습용 추론 서버에서 두 모델을 비교해 선택한 모델을 서비스한다. 룰북 v2 14·26쪽에 따라 대회에서는 팀 서버의 로컬 VLM이 G1 RGB 카메라만 보고 판단한다. 최종 득점에는 세 방 이름과 교실 내부 골반 위치가 모두 필요하다.

## 1. 연습 시뮬레이터에서 세 문 수집

ROS 2·프로젝트 conda 환경과 bringup을 준비한 시뮬레이터 PC에서 실행한다. 연습 실행마다 전용 디렉터리를 만들고 시뮬레이터의 기존 `--scenario-out` 기능으로 현재 배치를 기록한다. 기본 설정의 나머지 항목은 그대로 복사한다.

```bash
RUN=$(mktemp -d "${TMPDIR:-/tmp}/m4-run.XXXXXXXX")
export RUN
python - <<'PY'
import os
from pathlib import Path
import yaml
run = Path(os.environ["RUN"]).resolve()
cfg = yaml.safe_load(Path("src/main_config.yaml").read_text())
cfg["simulator"]["extra_args"] += ["--scenario-out", str(run / "active.yaml")]
(run / "practice.yaml").write_text(yaml.safe_dump(cfg))
print("두 번째 터미널의 RUN 경로:", run)
PY
(cd src && python main.py --sim-only --config "$RUN/practice.yaml")
```

같은 환경의 두 번째 터미널에서 아래를 실행한다. `RUN`에는 첫 터미널이 출력한 절대경로를 넣는다. 첫 배치 이후에는 MuJoCo에서 R로 배치를 바꾸고 숫자 4로 M4 시작 위치에 놓은 뒤 정지 상태에서 수집을 반복한다. 촬영 도중 배치 변경이나 리스폰이 감지되면 해당 수집을 다시 실행한다.

```bash
RUN=/tmp/m4-run.실제접미사
DATA="$HOME/m4-data"
python server/mission_4/collect.py "$DATA" --run-id "$(basename "$RUN")" --scenario "$RUN/active.yaml"
```

수집기는 미션과 같은 경로로 x=17.4 m, 목표 yaw=0에서 정지한 뒤 새 RGB 프레임을 JPEG 품질 90으로 저장한다. 방문은 중간→북→남이다. 각 배치 폴더에는 `north.jpg`, `middle.jpg`, `south.jpg`와 실행 ID·배치 ID·문별 API 라벨이 담긴 `survey.json`이 있다. 세 장을 얻은 후 시뮬레이터가 기록한 북→남 `room_order`를 붙인다. 같은 시뮬레이터 프로세스에서 반복 촬영한 모든 배치는 같은 RUN 값을 사용한다. 여러 독립 실행에서 여섯 순열을 고르게 수집한다. 두 실행은 분할 가능한 최소치이며 성능 판단에는 여러 평가 실행이 필요하다.

## 2. 연습용 추론 서버에서 분할 고정 및 기본 모델 평가

전체 데이터 폴더를 팀 GPU에 복사한다. 기본 모델은 접근 권한을 받은 `google/paligemma2-3b-mix-224`의 전체 로컬 스냅샷(가중치·구성·토크나이저·이미지 프로세서)을 준비한다. GPU 환경에 맞는 CUDA PyTorch 2.5 이상과 대응하는 torchvision 설치 후 아래 명령을 실행한다.

```bash
M4=server/mission_4
DATA="$HOME/m4-data"
EXP="$HOME/m4-exp-01"
BASE=/absolute/path/to/paligemma2-3b-mix-224
mkdir -p "$EXP"
python -m pip install transformers==4.53.3 pillow sentencepiece
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.mem_get_info())'
python "$M4/dataset.py" "$DATA" "$EXP/split.json"
set -o pipefail
python "$M4/evaluate.py" "$DATA" "$EXP/split.json" "$BASE" | tee "$EXP/base.log"
```

분할은 실행 ID 단위로 약 20%를 평가용으로 예약한다. 한 배치의 세 문과 같은 실행의 인접 촬영분은 같은 쪽에 들어간다. 분할 파일은 한 번 생성하며 이후 추가된 배치는 기존 실험에 포함되지 않는다. 실험 중에는 해당 배치의 영상과 라벨을 고정한다.

## 3. 오프라인 학습 전용 서버에서 미세조정

노트북에서 같은 브랜치의 코드, 2단계의 데이터와 `split.json`, 동일 버전의 기본 체크포인트를 준비한다. 지정된 원격 작업 폴더로 옮기고 SHA256을 대조한다. 누락된 의존성은 대상 Linux·Python ABI에 맞는 wheel로 준비해 작업 전용 환경에 오프라인 설치한다.

원격 작업 지침에 따라 지정된 Python의 실제 import 경로와 버전, GPU 사용량, 실행 중 작업, 상태 파일 및 공용 잠금 보유 여부를 확인한다. 아래는 기존 잠금 규약을 따르는 학습 실행기에 전달할 **학습 명령**이다. `PY`는 지정된 Python 절대경로, `DATA`와 `BASE`는 전송한 자산 경로, `EXP`는 고유 실행 ID의 결과 디렉터리로 설정하고 여기에 고정한 `split.json`을 둔다. 실행기의 로그·종료 상태와 출력 체크포인트를 함께 확인한다.

```bash
"$PY" server/mission_4/train.py "$DATA" "$EXP/split.json" "$BASE" "$EXP/tuned" --epochs 3 --lr 1e-5
```

지정된 학습 서버에는 위 학습 명령을 공용 GPU 잠금 안에서 실행하는 `train_locked.sh`를 준비했다. 실영상 데이터와 고정 분할을 반입한 뒤 아래 경로를 사용한다. 실행 ID는 매번 새로 정한다. `setsid`로 분리한 실행은 접속 종료 후에도 계속되며 실행별 `runs/$ID`에 PID·프로세스 그룹·로그·종료 코드·체크포인트를 남긴다. 시작 시 잠금 충돌이나 경로 오류는 `$ID.launch.log`에 기록된다.

```bash
ROOT=/absolute/path/to/haic_m4
BASE="$ROOT/models/paligemma2-3b-mix-224-8e40ab4cc5df93dfb7fd2fff754bcdff8b62ee78"
DATA="$ROOT/data"           # 반입한 실제 데이터 경로
SPLIT="$ROOT/split.json"    # 기본 모델 평가에 사용한 고정 분할
ID="projector-$(date -u +%Y%m%dT%H%M%SZ)"
nohup setsid bash "$ROOT/code/server/mission_4/train_locked.sh" "$ROOT" "$DATA" "$SPLIT" "$BASE" "$ID" \
  > "$ROOT/$ID.launch.log" 2>&1 < /dev/null &
```

3B 모델을 GPU에 올리고 시각 특징을 언어 모델에 연결하는 프로젝터만 AdamW로 갱신한다. 이미지 한 장씩, 지원 GPU에서는 BF16, 그 외에는 FP16과 gradient scaling을 사용한다. 계산 그래프 메모리를 줄이기 위해 gradient checkpointing을 사용한다. 2GB MX450에는 모델 가중치가 들어가지 않는다. 팀 GPU의 여유 VRAM을 먼저 확인한다. 학습에는 train 배치만 쓰고 정답 답변 토큰에 loss를 적용한다. 의료실의 학습 답변은 `infirmary`다. 출력은 프로세서를 포함하는 전체 체크포인트다.

## 4. 연습용 추론 서버로 반입해 비교 평가

학습 실행 ID를 기록하고 `tuned/` 전체를 노트북을 거쳐 2단계 연습용 추론 서버의 같은 실험 디렉터리 `$EXP/tuned`로 옮긴다. 가중치 조각, 구성, 토크나이저, 이미지 프로세서를 포함해 전송 전후 파일의 SHA256을 대조한다. 아래부터는 2단계 장비의 `M4`, `DATA`, `BASE`, `EXP` 경로를 사용한다.

```bash
python "$M4/evaluate.py" "$DATA" "$EXP/split.json" "$BASE" --tuned "$EXP/tuned" \
  --selected "$EXP/selected.txt" | tee "$EXP/compare.log"
```

두 모델을 같은 eval 배치에서 차례로 실행한다. 모든 점수는 배포 서버와 동일한 프롬프트에서 각 후보 답변의 평균 토큰 로그확률로 계산한다. 혼동행렬의 행은 정답, 열은 이미지 한 장의 최고 점수 예측이다. 배치 평가는 북→중간→남 점수표를 로봇의 `assign_rooms`에 전달한다. 최고 합계 동점은 실패로 분모에 포함한다. 세 방 전체 정답률이 높은 모델을 선택하고, 동률이면 단일 영상 정확도, 이것도 같으면 기본 모델을 선택한다. 이 eval은 모델 선택용이다. 최종 일반화 성능은 별도의 새 실행에서 측정한다.

## 5. 연습용 추론 서버에서 선택 모델 실행

```bash
python "$M4/vlm_server.py" "$(cat "$EXP/selected.txt")" --host 0.0.0.0 --port 8044
```

## 6. JPEG 한 장과 실제 미션 확인

팀 서버의 다른 터미널에서 수집한 한 장을 보낸다. 응답은 `classroom`, `medical`, `cafeteria` 각각의 유한한 숫자 점수를 담은 JSON이다.

```bash
JPEG=$(find "$HOME/m4-data" -name middle.jpg -print -quit)
curl --fail-with-body -H 'Content-Type: image/jpeg' --data-binary "@$JPEG" http://127.0.0.1:8044/scores
```

시뮬레이터 PC에서는 연습 수집용 프로세스를 종료한 뒤 일반 미션으로 득점을 확인한다.

```bash
(cd src && M4_VLM_URL=http://PRACTICE_INFERENCE_HOST:8044/scores python main.py --missions 4)
```

## 개발 PC에서 로컬 검증 재현

저장소 루트에서 Pillow가 있는 Python으로 분할 누수, 여섯 순열, 단일 영상 오답과 배치 전체 정답이 공존하는 경우, 동점 실패 집계를 검사한다.

```bash
python server/mission_4/test_evaluation.py
```

위 모델 의존성이 있는 환경에서는 실제 224 프로세서로 세 답변의 마스킹과 종료 토큰을 검사할 수 있다. 이후 한 층짜리 임의 초기화 모델과 합성 JPEG로 한 epoch를 학습하고, 두 체크포인트의 파라미터를 대조해 프로젝터만 갱신됐는지 검사한다. 저장된 각 체크포인트를 다시 열어 같은 평가 배치도 처리한다. 이 검사의 결과는 학습·배포 코드의 동작 여부다.

```bash
# BASE가 있으면 그대로 사용. 가중치 없이 준비할 때는 접근이 허용된 HF 계정에서 실행
huggingface-cli download google/paligemma2-3b-mix-224 --include 'config.json' 'preprocessor_config.json' 'tokenizer*.json' 'special_tokens_map.json' --local-dir "$HOME/m4-processor"
python server/mission_4/smoke_model.py "$HOME/m4-processor"
# CUDA 혼합 정밀도 경로도 확인할 경우
python server/mission_4/smoke_model.py "$HOME/m4-processor" --device cuda
```
