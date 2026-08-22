"""기각된 실험 코드 — 기록 보존용. 최종 파이프라인에서 쓰지 않는다.

`nan_strategies.py`  결측 처리 5종 비교 → baseline이 최선, 기각
                     (타겟이 y_woba이던 시절 코드라 현재는 실행 불가)
`outlier_handler.py` 이상치 처리 4종 비교 → 전부 R² 하락, 기각
                     (clip_speed가 전체 quantile을 봐서 leakage 위험)

자세한 사유는 README.md 참조. 수정하지 말 것.
"""
