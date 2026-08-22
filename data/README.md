# 데이터

이 폴더의 실제 데이터 파일은 저장소에 포함되어 있지 않습니다.

## 왜 없나

| 이유 | 내용 |
|---|---|
| 용량 | 원본 Statcast parquet 약 456MB, 전체 614MB — GitHub 100MB 제한 초과 |
| 재생성 가능 | Baseball Savant 공개 데이터라 수집 노트북으로 다시 만들 수 있음 |
| 재배포 | 원 저작자(MLB Advanced Media)의 데이터를 그대로 재배포하지 않기 위함 |

## 구성

```
data/
├── statcast/     Baseball Savant 원본 (statcast_2021~2025.parquet)  ← 불변
├── data/         크롤링 결과, 관절 좌표·각도 CSV
├── output/       영상 배치 처리 산출물 (슬롯별 coords/cuts)
└── 4_features/   모델 입력 feature (환경에 따라 루트에 있을 수 있음)
```

`statcast/`는 **원본이므로 수정하지 않습니다.** 모든 가공 결과는 별도 파일로 저장합니다.

## 어떻게 얻나

### 1. 직접 수집 (누구나 가능)

```
1_statcast/01_data_collection.ipynb   → statcast/ 생성 (시즌당 수십 분)
1_statcast/02_preprocessing.ipynb     → 선발투수 필터링
1_statcast/03_feature_engineering.ipynb → feature 생성
1_statcast/04_build_targets.ipynb     → 타겟 생성
```

영상 파이프라인(`2_video/`)은 GPU가 필요해 Colab 환경을 권장합니다.

### 2. 팀원 — 드라이브 공유

데이터를 이미 갖고 있다면 위치만 알려주면 됩니다.

```bash
cp .env.example .env
# .env 에 PITCHER_DATA_ROOT=<0_data 경로> 지정
```

경로 처리는 [`config.py`](../config.py)가 담당하며, Colab과 로컬을 자동으로 감지합니다.

## 주요 파일

| 파일 | 내용 |
|---|---|
| `statcast_2021~2025.parquet` | 투구 단위 원본 (5시즌) |
| `features_pitch15.parquet` | 경기 단위 feature — 1 row = 1경기(game_pk × pitcher) |
| `game_targets.parquet` | 타겟 (whiff%, xwOBA, FIP, ERA) |
| `starters_all.parquet` | 선발투수 필터링 결과 |
| `prev_season_lookup.parquet` | 직전 시즌 기준선 (prev feature용) |

## 규모

- 정형: **23,225경기** (2021~2025, 선발투수)
- 영상: **3,783경기** 매칭 (투구 클립 약 5.7만 개)
- 분할: Train 2021~2023 / Val 2024 / **Test 2025** (시즌 기반, 랜덤 분할 금지)
