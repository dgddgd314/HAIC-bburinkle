# 팀 뿌링클 — custom/

## 실행 전 준비

- 추가 pip/apt 패키지 없음 (numpy, scipy, rclpy, tf2_ros는 대회 기본 환경에 포함).
- 첫 실행 때 벽 점군 법선을 계산해 `~/.cache/haic_bburinkle/`에 저장합니다(약 3초). 이후 실행은 0.1초.

## 위치 추정 (`localization/`)

`bringup.launch.py`가 대회 내내 아래 두 노드를 띄웁니다.

```
/lidar/points ─┬─ lidar_odom_node.py ──► TF odom → pelvis, /lidar_odom   (지도 없이 스캔끼리 정합)
/imu ──────────┤
               └─ icp_localizer_node.py ► TF map → odom                  (벽 지도와 정합해 보정)
                                          /localization/pose, /localization/status (JSON)
```

미션 코드에서는 `self.pose()` → `(x, y, yaw)` (월드 좌표 = map 좌표).

| 파일 | 내용 |
|---|---|
| `icp_core.py` | 바닥 RANSAC 수평 맞춤 → 높이대·복셀 필터 → 점-평면 ICP(x, y, yaw) → 품질 판정 |
| `odom_core.py` | 이전 키프레임 스캔에 정합하는 LiDAR 오도메트리 (0.5 m / 15°마다 키프레임) |
| `icp_localizer_node.py` | ROS 노드. `odom_frame:=odom`이면 map→odom, 비우면 map→pelvis 직접 발행 |
| `lidar_odom_node.py` | ROS 노드. odom→pelvis 발행 |

### 정합 필터링 (지도에 없는 가구·물체 걸러내기)

- **바닥 찾기**: 골반 높이 0.6–1.0 m에 해당하는 평면만, 평면 아래에 점이 있으면 감점 → 책상·침대 윗면, 절벽 바닥(−1.5 m) 오인 방지
- **높이대**: 바닥 위 0.25–2.3 m 점만 사용 (바닥·천장 제외)
- **보이는 면만**: 벽 법선을 바깥쪽으로 정렬, 센서 반대쪽 벽 면과는 매칭 안 함
- **대응점 거리 게이트**: 1.0 → 0.1 m로 줄여 가며, Tukey 가중치로 벽에서 먼 점(가구) 무시
- **품질 판정**: 벽 10 cm 이내 점 비율 ≥ 65%, RMSE ≤ 6 cm일 때만 채택. 아니면 오도메트리로 계속

### 리스폰·위치 잃었을 때

- 시뮬레이터 `/g1/mission_status`의 `startup`/`respawned` 이벤트의 `pos`, `yaw_deg`로 자동 재설정
  (README 4절: 리스폰은 순간이동이라 오도메트리가 모름).
- 수동 재설정: `/initialpose` (PoseWithCovarianceStamped) 발행.
- 5스캔 연속 실패 → 마지막 위치 ±1 m 전방위 탐색, 실패 시 미션 시작 지점들 주변 탐색.

### 검증 결과 (2026-11-26, 시뮬레이터)

| 시험 | 결과 |
|---|---|
| 정적 스캔 151개(가구 포함), 초기 오차 0.3 m/10° | 99%가 5 cm·1° 이내, 중앙값 0.3 cm |
| 걷기·회전·옆걸음·제자리 회전 (실시간) | 스캔 100% 성공, 위치 오차 중앙값 0.5 cm / 최대 1.0 cm, 방향 최대 1.1° |
| 오도메트리 단독 (지도 없음) | 같은 경로에서 누적 오차 최대 0.4 cm |
| 리스폰 | 1초 안에 (16, 11) 재추정, 오차 0.3 cm |

`--ground-truth`는 위 측정의 **비교용으로만** 사용했고 노드는 정답 오돔을 읽지 않습니다 (규칙 4).

### 직접 확인하기

```bash
python main.py --sim-only                       # 터미널 1 (bringup이 두 노드를 띄움)
ros2 topic echo /localization/status            # 터미널 2: ok, inlier_ratio, x, y, yaw_deg
ros2 run tf2_ros tf2_echo map pelvis
```

## 이동 (`navigation/nav.py`) — 미션 2·4·6에서도 재사용

```python
from custom.navigation.nav import Navigator
self.nav = Navigator(self)                  # __init__에서
self.nav.go_to(x, y, tol=0.15)              # run()에서: 위치 추정 OK 대기 → A* 경로 → 따라 걷기 → 정지
self.nav.turn_to(math.radians(90))          # 제자리 회전
self.nav.settle(1.5)                        # 판정 전 멈춰 서기
```

- 경로: `architecture_nav_safe`(벽 0.25 m 여유) 10 cm 격자 A*, 벽 가까울수록 비용↑, 직선 단축은 벽에서 0.4 m 이상일 때만 (문 가운데로 통과).
- 추종: 앞쪽 0.6 m 지점을 향해 회전 + 전진(최대 0.5 m/s), 목표 0.6 m 안에서는 옆걸음으로 미세 조정.
- 지도에 없는 가구는 아직 피하지 않음 → 방 안으로 들어가는 미션(4)에서는 센서 기반 장애물 회피 추가 필요.

## 미션 1

`Navigator.go_to(goal)` → 1.5초 정지 → 목표에서 0.3 m 넘게 벗어났으면 다시 접근(최대 3회).
검증(2026-11-26): 무작위 목표 8회 모두 PASS, 판정 거리 0.09–0.11 m, 16–38초.
`python main.py --missions 1` 공식 경로로도 PASS (0.09 m, 26.7 s).

## 지도 (`maps_structural/`, `structural_mapping/`)

팀원이 시뮬레이터 소스의 고정 건축 구조(벽·문·바닥)로 생성. 가구·방 배치·콘 색·공 위치 등
시나리오마다 바뀌는 요소는 넣지 않음. 자세한 내용은 `structural_mapping/` 안 스크립트와 생성 당시 README 참고.
