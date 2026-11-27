# M4 실행

팀 GPU 서버에서 PyTorch와 torchvision을 서버의 CUDA 환경에 맞게 설치한 뒤, 아래 명령으로 로컬 체크포인트를 연다. 기본 모델 후보는 `google/paligemma2-3b-mix-224`이다. 체크포인트 경로에는 모델 가중치와 프로세서 파일이 함께 있어야 한다.

```bash
# 저장소 루트에서 팀 GPU 서버 프로세스 실행
python -m pip install transformers==4.53.3 pillow sentencepiece
python server/mission_4/vlm_server.py /path/to/paligemma2-3b-mix-224 --host 0.0.0.0
```

시뮬레이터 PC의 대회 환경에서 팀 서버의 주소를 지정하고 실행한다.

```bash
# src/ 에서 실행
export M4_VLM_URL=http://TEAM_SERVER:8044/scores
python main.py --missions 4
```

G1은 각 문 앞에서 정지한 뒤 새 RGB 영상을 전송한다. 로그의 `[M4] Door y=...`에 후보 답변별 평균 토큰 로그 확률이 기록된다. 세 문에서 모은 점수의 합이 가장 큰 배치를 북→남 순서로 반환하며, 해당 배치의 교실까지 이동한다. 리스폰하면 세 방을 다시 관측한다.
