"""정형 Statcast feature 집계.

`feature_aggregator.py` — 경기 초반 X구간(n구)의 투구를
1경기=1행으로 집계해 모델 입력 feature를 만든다.
구속·회전수·무브먼트·ACWR·delta 등 79개 생성.

최종 모델(v4)이 쓰는 데이터가 여기서 나온다.
"""
