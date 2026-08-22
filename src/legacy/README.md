# legacy

**최종 파이프라인에서 사용하지 않는 실험 기록용 모듈입니다.**

| 모듈 | 상태 | 이유 |
|---|---|---|
| `nan_strategies.py` | 실행 불가 | 타겟이 `y_woba`였던 초기 시점 코드. 현재 feature로는 동작하지 않음 |
| `outlier_handler.py` | 실행 가능 | `clip_speed()`가 전체 데이터 quantile을 사용해 leakage 위험. 실험 재현용으로만 유지 |

두 모듈의 실험 결과는 모두 **baseline이 최선**으로 나와 기각됐습니다
(`outputs/experiments/nan_experiment_results.csv`, `outlier_experiment_results.csv`).

기록 보존이 목적이므로 **수정하지 마세요.** 재실행이 필요하면 결과를 새로 기록할 것.
