"""재사용 모듈. 노트북에서 반복되던 로직을 빼낸 것.

| 패키지 | 역할 |
|---|---|
| `statcast` | 정형 데이터 feature 집계 — 파이프라인의 핵심 |
| `video`    | 영상 생체역학 feature 추출 |
| `modeling` | 모델 비교 실험 스크립트 |
| `legacy`   | 기각된 실험 코드 (기록 보존용, 실행 안 함) |

경로는 `config.py`가 sys.path에 등록하므로
노트북에서 `import feature_aggregator` 로 바로 쓸 수 있다.
"""
