# 3_modeling

## 번호 체계

파일 번호는 **프로젝트 전체 통산**입니다. 04~19가 성격별 세 폴더에
나뉘어 있어 각 폴더 안에서는 번호가 건너뜁니다. 누락이 아닙니다.

| 폴더 | 번호 | 역할 |
|---|---|---|
| `1_pipeline/` | 04 06 07 10 11 13 | 최종 모델로 이어지는 본류 |
| `2_experiments/` | 05 09 12 14 15 16 | 가설 검증 (대부분 기각) |
| `3_analysis/` | 08 18 19 | 해석·분류 |

번호를 유지하는 이유는 노트북 본문과 실험 기록이 서로를
번호로 참조하기 때문입니다 ("14번과 동일한 틀", "11번 먼저 실행" 등).

## 실행 순서

```
04_baseline           베이스라인 모델
06_x_interval         X구간(초반 몇 구) 탐색
07_delta              delta feature 검증
10_tuning             Optuna 하이퍼파라미터 최적화
11_feature_selection  feature 선택 → 최종 모델
13_final_evaluation   최종 평가
```

`2_experiments/`는 본류가 아닙니다. 각 노트북 첫 셀에 결론과
기각 근거가 정리되어 있습니다.

## 모듈

재사용 코드는 `src/` 에 있습니다.

| 모듈 | 위치 |
|---|---|
| `feature_aggregator.py` | `src/statcast/` — feature 집계 (핵심) |
| `x_interval_experiment.py` | `src/modeling/` |
| `nan_strategies.py` `outlier_handler.py` | `src/legacy/` — 실험 기록용 |

`config.py`가 경로를 등록하므로 노트북에서는 `import feature_aggregator`로
바로 쓸 수 있습니다.
