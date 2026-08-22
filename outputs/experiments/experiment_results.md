# MLB 선발투수 컨디션 예측 — 실험 결과 정리

> 프로젝트 목표: 경기 초반 투구 데이터(X)만으로 이후 투구 퍼포먼스(Y)를 예측  
> Y 지표: ~~wOBA against~~ → **whiff% (헛스윙률)** 로 변경 (노이즈 감소 목적)

---

## 실험 흐름 요약

```
Y 지표 변경 (wOBA → whiff%)
    ↓
베이스라인 구축 (04)
    ↓
NaN 처리 전략 비교 (05)
    ↓
X 구간 실험 (06)  →  pitch15 확정
    ↓
Delta Feature 기여도 검증 (07)  →  paired t-test ✅
    ↓
이상치 처리 실험 (09)  →  E4-1 baseline 확정
    ↓
도메인 feature 확장 (14: A 추세 + B 릴리스 변동성)  →  paired t-test 예정
    ↓
시퀀스 모델 (15: 1D-CNN vs aggregate XGB)  →  paired t-test 예정
```

---

## 15. 시퀀스 모델 실험 (Phase 11 — 1D-CNN, 로드맵 C)

**파일**: `3_modeling/15_sequence_model_experiment.ipynb`  
**근거**: KBO 1D-CNN 헛스윙 예측 (Y 동일 계열). aggregate tree baseline vs 시퀀스 DL 비교 = 로드맵 "발전시킨 것" 본체.  
**목적**: 15구를 평균낸 정적 feature 대신 **pitch-by-pitch 원본 시퀀스**를 1D-CNN에 투입해, 투구 순서 정보가 예측력을 높이는지 검증.

### 설계

- 입력: 경기당 처음 15구 × 구별 feature 5종(`release_speed/spin_rate/pos_x/pos_z/extension`) → `(경기수, 15, 5)` 텐서. 15구 미만 0패딩, 결측 0대체, train 통계로 정규화(leakage 방지).
- 모델: 1D-CNN (Conv1d 2층 → AdaptiveAvgPool → FC), early stopping.
- 비교 대상: 동일 모집단·동일 split의 aggregate XGB.
- Y: `y_whiff` (회귀, 동일).

### 실험

| 실험 | 내용 | 비교 |
|---|---|---|
| E9-1 | Aggregate XGB (기준) | 정적 feature |
| E9-2 | 1D-CNN (제안) | 시퀀스 원본 |
| 🔬 E9-3 | XGB vs CNN **paired t-test** (n=30 seeds) | 시퀀스 모델 우열 검증 |

> ⚠ Colab GPU 권장(런타임 유형 → T4). 14번과 독립 실행 가능(원본 pitch 시퀀스 사용). 시간 부족 시 N_SEEDS=10으로 축소 가능.
> 출력: `sequence_model_ttest_seeds.csv`, `sequence_model_results.json`, `sequence_model_ttest.png`

---

## 14. 도메인 Feature 확장 (Phase 6.5 — A 추세 + B 릴리스 변동성)

**파일**: `3_modeling/feature_aggregator.py` (집계 함수), 실험 노트북 예정  
**근거 논문**: 98_참고논문 — Paripex(회전수 저하=피로 1차 지표), Frontiers(수평 릴리스 변동성 ↔ 삼진 최강 상관)  
**목적**: 15구 정적 평균만 쓰던 X feature에 (A)경기 내 추세와 (B)릴리스 포인트 변동성을 추가해 "현재 상태(state)" 신호를 강화. SHAP에서 절대값이 delta를 압도하던 한계(08) 보완 시도.

### 추가된 feature (8개)

| 구분 | feature | 정의 |
|---|---|---|
| **B 변동성** | `std_pos_x_{Fastball/Breaking/Offspeed}` | X구간 내 수평 릴리스 표준편차 (구종 그룹별) |
| **B 변동성** | `std_pos_z_{Fastball/Breaking/Offspeed}` | 수직 릴리스 표준편차 |
| **A 추세** | `trend_speed_all` | (후반 절반 평균 구속 − 전반 절반 평균 구속) |
| **A 추세** | `trend_spin_all` | (후반 절반 평균 회전수 − 전반 절반 평균 회전수) |

- A는 X구간(pitch15) 내 투구를 시간순 전반/후반으로 양분 → late−early. half당 2구 미만이면 NaN.
- 구현 위치: `_aggregate_x_features`(B는 그룹 집계에 STDDEV 추가), `_aggregate_trend_features`(A 신규 함수).

### 검증 데이터 (raw 2024 샘플 30만구)

| feature | 결측률 | 평균 | 해석 |
|---|---|---|---|
| `std_pos_x_Fastball` | 14.6% | 0.117 | 정상 (Breaking/Offspeed는 미등판 시 NaN) |
| `trend_speed_all` | 14.2% | **−0.11 mph** | 후반 구속 소폭 저하 — 일반적 피로 패턴 포착됨 |
| `trend_spin_all` | 14.6% | +1.7 rpm | 정상 범위 |

### 다음 단계 (예정 실험)

| 실험 | 내용 | 비교 기준 |
|---|---|---|
| E8-1 | 정형 기존 59 feature (Phase 8 최종) | 기준 |
| E8-2 | + B 릴리스 변동성 (6개) | E8-1 vs E8-2 |
| E8-3 | + A 추세 (2개) | E8-2 vs E8-3 |
| 🔬 E8-4 | E8-1 vs E8-3 (전체) **paired t-test** (n=30 seeds) | **도메인 feature 기여도 검증** |
| E8-5 | SHAP으로 추세/변동성 feature 순위 확인 | delta보다 상위로 올라오는지 |

> ⚠ 실행은 Colab(Drive: `투수 컨디션 예측 ML`)에서. `features_pitch15.parquet`를 재생성(overwrite=True)해야 새 컬럼 반영됨.

---

## 09. 이상치 처리 실험 (Phase 6)

**파일**: `3_modeling/09_outlier_experiment.ipynb`  
**출력**: `4_output/outlier_experiment_results.csv`  
**목적**: X feature 구속 clip / Y 극단값 제거가 예측 성능에 영향을 주는지 비교

### 배경
- Y EDA 결과: whiff% = 0.0인 경기 101개(0.4%), > 0.60인 경기 7개
- X feature: 구속(speed) 관련 컬럼 상하 1% clip 효과 검증

### 시도한 전략

| 실험 | 내용 | 제거 샘플 |
|---|---|---|
| E4-1 | 없음 (베이스라인) | 0 |
| E4-2 | X 구속 feature 상하 1% clip | 0 |
| E4-3 | Y 극단값 제거 (whiff% < 0.05 or > 0.60) | 328 |
| E4-4 | E4-2 + E4-3 조합 | 328 |

### 결과

| 실험 | XGB Val RMSE | XGB Val R² | CB Val RMSE | CB Val R² | LGB Val RMSE | LGB Val R² |
|---|---|---|---|---|---|---|
| E4-1 baseline | 0.0851 | **0.0824** | 0.0849 | 0.0861 | 0.0850 | 0.0835 |
| E4-2 clip_speed | 0.0851 | 0.0812 | 0.0848 | **0.0874** | 0.0851 | 0.0816 |
| E4-3 remove_y | **0.0821** | 0.0782 | **0.0818** | 0.0856 | **0.0821** | 0.0798 |
| E4-4 clip+remove | 0.0822 | 0.0772 | 0.0819 | 0.0843 | 0.0821 | 0.0781 |

### 결론
- **RMSE 기준**: E4-3이 가장 낮음 (0.0821) — 단, Y 분산 자체가 줄어든 효과
- **R² 기준**: E4-1 baseline이 XGB 최고(0.0824), E4-2가 CB 최고(0.0874)
- Y 극단값 제거(E4-3, E4-4)는 RMSE는 개선되나 R²가 오히려 감소 → 단순 분산 압축 효과이며 실제 예측력 향상이 아님
- **→ E4-1 baseline 확정**: R² 기준 가장 균형 잡힌 성능, 데이터 손실 없음
- **이후 실험(Phase 7 튜닝)은 E4-1 기준으로 진행**

---

## 04. 베이스라인 모델링

**파일**: `3_modeling/04_modeling.ipynb`  
**목적**: 전처리 없이 XGBoost / CatBoost 기초 적합 → 이후 실험의 비교 기준점 확보

### 배경
- Y: wOBA against (경기 단위)
- X구간: batter9 고정
- NaN: 모델 내부 처리 그대로
- feature 수: 59개

### 시도
- Early stopping 없음 vs 있음 비교 → 과적합 확인
- XGBoost / CatBoost 각각 학습

### 결과

| 모델 | Val RMSE | Val R² | Test RMSE | Test R² |
|---|---|---|---|---|
| XGBoost | 0.1362 | 0.0074 | 0.1374 | -0.0043 |
| CatBoost | 0.1362 | 0.0069 | 0.1369 | 0.0035 |

### 결론
- R² ≈ 0.005~0.009로 매우 낮음
- Early stopping 없을 때 Train R²=0.61 / Val R²=-0.05 → 심각한 과적합 확인
- SHAP 확인 결과 절대값(avg_*)이 delta feature보다 상위 → 모델이 컨디션 변화보다 선수 스타일을 학습 중
- **Y를 wOBA에서 whiff%로 변경 결정** (수비 운·타자 수준 등 외부 노이즈가 너무 큼)

---

## 05. NaN 처리 전략 비교

**파일**: `3_modeling/05_nan_experiment.ipynb`  
**출력**: `4_output/nan_experiment_results.csv`  
**목적**: delta feature의 높은 NaN 비율(35~55%)을 어떻게 처리할지 전략 비교

### 배경
- `delta_speed_Offspeed` NaN 54.7%, `delta_speed_Breaking` NaN 46.6% 등
- NaN 30%↑ 컬럼이 31개로 절반 이상

### 시도한 전략

| 실험 | 전략 | feature 수 |
|---|---|---|
| E2-1 | 모델 내부 처리 (베이스라인) | 59 |
| E2-2 | 미등판 구종 delta → 0 impute | 59 |
| E2-3 | NaN 50%↑ 컬럼 제거 | 54 |
| E2-4 | NaN 30%↑ 컬럼 제거 | 28 |
| E2-5 | E2-2 + E2-3 조합 | 59 |

### 결과

| 전략 | XGB Val RMSE | XGB Val R² | CB Val RMSE | CB Val R² |
|---|---|---|---|---|
| E2-1 baseline | 0.1362 | 0.0074 | 0.1362 | 0.0069 |
| E2-2 zero | 0.1363 | 0.0051 | 0.1361 | 0.0084 |
| E2-3 drop50 | 0.1363 | 0.0063 | 0.1361 | 0.0081 |
| E2-4 drop30 | 0.1362 | 0.0068 | 0.1361 | **0.0085** |
| E2-5 zero+drop50 | 0.1363 | 0.0051 | 0.1361 | 0.0084 |

### 결론
- 5가지 전략 간 성능 차이 미미 (RMSE 소수점 4자리 수준)
- XGBoost 최적: E2-1 baseline / CatBoost 최적: E2-4 drop30
- **이후 실험은 E2-1 (모델 내부 처리) 기준으로 진행** — 도메인 논리상 NaN을 억지로 채우는 것보다 자연스러움

---

## 06. X 구간 실험

**파일**: `3_modeling/06_x_interval_experiment.ipynb`  
**출력**: `4_output/x_interval_experiment_results.csv`  
**목적**: 초반 몇 구/이닝/타자까지를 X로 볼 때 예측력이 가장 높은지 비교

### 배경
- X 구간을 하이퍼파라미터처럼 취급해 최적값 탐색
- Y: whiff% (이 시점부터 변경 적용)
- 3가지 단위(pitch/inning/batter) × 구간 수 = 8개 조합 비교

### 시도한 구간

| 실험 | mode | n | 의미 |
|---|---|---|---|
| E3-1 | pitch | 10 | 초반 10구 |
| E3-2 | pitch | 15 | 초반 15구 (~3~4타자) |
| E3-3 | pitch | 20 | 초반 20구 (~4~5타자) |
| E3-4 | inning | 1 | 초반 1이닝 |
| E3-5 | inning | 2 | 초반 2이닝 |
| E3-6 | batter | 3 | 초반 3타자 |
| E3-7 | batter | 6 | 초반 6타자 |
| E3-8 | batter | 9 | 초반 9타자 (베이스라인) |

### 결과

| 실험 | XGB Val RMSE | XGB Val R² | CB Val RMSE | CB Val R² |
|---|---|---|---|---|
| E3-1 pitch10 | **0.0834** | 0.0768 | **0.0828** | **0.0888** |
| **E3-2 pitch15** | 0.0851 | **0.0824** | 0.0849 | 0.0861 |
| E3-3 pitch20 | 0.0873 | 0.0732 | 0.0867 | 0.0867 |
| E3-4 inning1 | 0.0861 | 0.0784 | 0.0858 | 0.0850 |
| E3-5 inning2 | 0.0965 | 0.0654 | 0.0960 | 0.0745 |
| E3-6 batter3 | 0.0844 | 0.0783 | 0.0841 | 0.0844 |
| E3-7 batter6 | 0.0860 | 0.0755 | 0.0855 | 0.0851 |
| E3-8 batter9 | 0.0885 | 0.0780 | 0.0883 | 0.0837 |

### 결론
- RMSE 기준 자동 선택: pitch10
- **최종 확정: pitch15** — XGB R² 최고(0.0824), 정보량 더 많고 설명하기 자연스러움
- wOBA 때 R² ≈ 0.01 → whiff% 변경 후 R² ≈ 0.08~0.09로 약 9배 개선
- inning2가 가장 낮음 → X 구간이 너무 길어지면 Y 구간 투구 수가 줄어 불안정
- **이후 모든 실험은 pitch15 고정**

---

## 08. SHAP Feature Importance 분석

**파일**: `3_modeling/08_shap_analysis.ipynb`  
**출력**: `4_output/shap_feature_importance.csv`, `shap_bar_pitch15.png`, `shap_beeswarm_pitch15.png`, `shap_type_comparison.png`  
**목적**: whiff% 기준으로 절대값 vs delta feature 중 어떤 게 모델에 더 중요한지 확인 (04는 wOBA 기준이었으므로 재분석)

### 배경
- 04 베이스라인 SHAP에서 절대값이 delta보다 상위였음
- Y를 whiff%로 변경한 이후 동일한 패턴이 유지되는지 확인

### 결과

**Val R²: 0.0824 (XGBoost, pitch15, seed=42)**

| 유형 | 평균 SHAP | 최고 순위 |
|---|---|---|
| avg (절대값) | 0.00149 | **1위** |
| other (비율 등) | 0.00091 | 4위 |
| prev (기준값) | 0.00060 | 3위 |
| std (절대값) | 0.00041 | 24위 |
| **delta** | **0.00023** | **27위** |

**상위 feature**

| 순위 | feature | 유형 |
|---|---|---|
| 1 | avg_speed_Fastball | avg (절대값) |
| 2 | avg_spin_Fastball | avg (절대값) |
| 3 | prev_spin_Fastball | prev (기준값) |
| 4 | strike_ratio | other |
| 5 | prev_speed_Fastball | prev (기준값) |
| 27 | delta_ext_Fastball | **delta** (최초 등장) |

### 결론
- avg_speed_Fastball, avg_spin_Fastball이 압도적 1, 2위 → 모델이 선수 스타일(절대값)을 주로 학습 중
- delta feature는 paired t-test에서 유의미(p=0.017)했지만 SHAP 순위는 하위권 (최고 27위)
- delta는 소폭 기여하지만 절대값의 영향력이 훨씬 큼
- **→ biomechanical feature 추가의 필요성 시사**: 순수한 컨디션 state 예측을 위해선 영상 기반 데이터가 핵심

---

## 07. Delta Feature 기여도 실험 (🔬 A/B 테스트)

**파일**: `3_modeling/07_delta_experiment.ipynb`  
**출력**: `4_output/delta_experiment_results.csv`  
**목적**: 직전 시즌 대비 편차(delta feature)가 실제로 예측력을 높이는지 통계적 검증

### 배경
- delta feature = 오늘 구속/스핀/익스텐션 등 - 직전 시즌 평균
- 선수 identity 제거 → 현재 상태(state) 기반 예측 가능
- 04 베이스라인 SHAP에서 절대값이 delta보다 상위였음 → delta 효과 의문
- **paired t-test로 통계적 유의성 검증**

### 시도

| 실험 | 내용 | feature 수 |
|---|---|---|
| E1-1 | 절대값 feature만 (delta 없음) | 29개 |
| E1-2 | 절대값 + delta feature 전체 | 59개 |
| 🔬 E1-3 | E1-1 vs E1-2 paired t-test (n=30 seeds) | — |

- X구간: pitch15 고정
- 모델: XGBoost (early stopping)
- 30개 random seed로 반복 학습 → Val R² 쌍으로 paired t-test

### 결과

| | E1-1 (no delta) | E1-2 (with delta) |
|---|---|---|
| 평균 Val R² | 0.0786 | **0.0802** |
| 표준편차 | ±0.0022 | ±0.0022 |
| 평균 차이 | | **+0.0016** |

**Paired t-test**

| 항목 | 값 |
|---|---|
| t-statistic | 2.5386 |
| p-value | **0.0168** |
| 결과 | ✅ p < 0.05, 유의미한 차이 |

### 결론
- delta feature 추가 시 Val R² +0.0016 개선
- p=0.017 (< 0.05) → **통계적으로 유의미하게 검증됨**
- 절대적 수치 차이는 작지만, 야구 데이터 특성상(R² 자체가 낮음) 의미 있는 개선
- **delta feature 포함 확정, 이후 실험 모두 with delta 기준**

---

## 20. 영상 정규화 분모 재검증 (🔬 2D 투영 왜곡 진단)

**작성 2026-08-22** · 로컬 `all_coords.csv`(61,896투구) 재계산, 영상 재처리 없음

### 배경

12번(영상 생체역학 융합)이 유의 악화로 미채택됐다. 원인 재조사 중
`compute_angles()`가 모든 정규화 거리의 **분모로 어깨너비(shoulder width)** 를
쓰고 있는 것을 발견했다. 투수는 릴리스에서 몸을 타자 쪽으로 돌리므로
2D 화면에서 양어깨가 앞뒤로 겹쳐 어깨너비 픽셀이 실제와 무관하게 압축된다.

### 진단 1 — 좌표 자체는 정상

| 항목 | 값 | 판정 |
|---|---|---|
| 포즈 추정 실패율 | 0.2% (14관절 동일) | 정상 |
| 상완/전완 비율 | 1.16 (인체 1.0~1.2) | 정상 |
| 대퇴/종아리 비율 | 1.05 (인체 1.1~1.2) | 정상 |

→ MediaPipe 좌표 추출은 문제 없음. 문제는 **정규화 단계**.

### 진단 2 — 강체 세그먼트 단축률 (2D 투영 손실)

뼈 길이는 실제로 불변이므로, 같은 투수 내 변동은 전부 투영 왜곡이다.
(몸통으로 나눠 카메라 줌 효과 제거 후)

| 세그먼트 | 투수내 CV | 최대 단축률 |
|---|---|---|
| 전완 | 0.309 | **69%** |
| 어깨너비 | 0.263 | **65%** |
| 골반너비 | 0.247 | **61%** |
| 종아리 | 0.198 | 44% |
| 대퇴 | 0.168 | 38% |
| 상완 | 0.160 | 38% |

### 진단 3 — 분모 후보 안정성

| 분모 | 5px 미만 | 1% 분위 |
|---|---|---|
| 어깨너비(현재) | 0.78% | **5.8px** |
| 몸통(어깨중심-골반중심) | 0.00% | 37.5px |

→ 어깨너비는 하위 1%가 5.8px. 나누면 값이 폭발(윈저라이징으로도 복원 불가).

### 결과 1 — 측정 품질(투수간 설명력)은 2배 개선

투수간 분산 / 전체 분산. 릴리스 위치·팔각도는 투수 고유값이므로 높아야 정상.
(참고: Statcast `release_pos_x` 97.3%, `arm_angle` 88.1%)

| 피처 | 어깨너비 | 몸통 | 배율 |
|---|---|---|---|
| stride_norm | 7.2% | 13.2% | 1.8x |
| release_height_norm | 5.3% | 11.0% | 2.1x |
| arm_extension_norm | 5.0% | 12.3% | 2.5x |
| trunk_dist_norm | 2.9% | 15.3%* | 5.3x |

*몸통÷몸통은 상수이므로 이 피처만 "몸통+상완" 분모 사용

### 결과 2 — 예측 성능은 개선되지 않음 ❌

v4 피처(79개) · 매칭 3,783경기 · best_params.json XGB · seed 10

| 분모 | split | 정형 | 융합 | Δ | 승 |
|---|---|---|---|---|---|
| 어깨너비 | val(2024) | 0.0294 | 0.0313 | +0.0019 | 7/10 |
| 어깨너비 | **test(2025)** | 0.0274 | 0.0163 | **-0.0111** | **0/10** |
| 몸통 | val(2024) | 0.0294 | 0.0371 | +0.0077 | 10/10 |
| 몸통 | **test(2025)** | 0.0274 | 0.0131 | **-0.0143** | **0/10** |

※ val 30-seed paired t-test에서는 몸통 Δ +0.0079, t=7.76, p=1.46e-08 (28/30승)로
  유의 개선이었으나 **test에서 완전히 뒤집힘**. val 단일 split 판단은 위험.

### 결과 3 — SHAP: 과적합 확증

몸통 융합 모델(정형79 + 영상72 = 151피처) 기준

| 항목 | 값 |
|---|---|
| 영상피처 SHAP 합계 비중 | **36.8%** |
| 영상 최상위 | trunk_angle_q75 (전체 13위) |
| 상위 12개 | trunk_angle, release_height_norm, arm_slot, shoulder_tilt 계열 |

→ 모델은 영상을 **많이 쓰는데**(36.8%) test 성능은 **악화**(-0.0143).
   3,783경기에 영상 72피처는 과다. 전형적 과적합.

### 결론

1. **분모 버그는 실재한다** — 어깨너비는 2D 회전에 압축되는 잘못된 분모.
   몸통으로 교체 시 측정 품질(투수간 설명력) 2배 개선 확인.
2. **그러나 예측 성능은 개선되지 않는다** — val 개선은 과적합이었고
   test에서 오히려 더 악화(-0.0143 < -0.0111).
3. **12번의 미채택 결론은 유지된다.** 측정 개선 ≠ 예측 개선.
   12번 진단("영상은 다른 정보를 담지만 경기 결과와 무관")이 오히려 강화됨.
4. **다음 수는 분모가 아니라 차원**이다. 3,783경기에 81피처가 이미 과다.
   시퀀스·3D로 축을 늘리기 전에 집계 차원 설계가 선행되어야 함.

### 재현

- 스크립트: 세션 스크래치패드 `recompute.py`, `ttest.py`, `test2025.py`
- 입력: `0_data/data/all_coords.csv`, `v4_output/features_pitch15_v4.parquet`
- 영상 재처리 불필요 (좌표 CSV에서 각도만 재계산)

---

## 21. 3D 리프팅 게이트 테스트 (RTMPose + MotionBERT)

**작성 2026-08-22** · 로컬 CPU · 야구 영상 4개 (`batch_slot4_0001.zip`에서 추출)

### 배경

20번에서 2D 영상 피처가 test 악화(-0.0143)로 확인됐고, 원인 중 하나로
**2D 투영 손실**(강체 세그먼트 최대 69% 단축)을 지목했다. 배드민턴 프로젝트
(`ai-badminton-coach`)에 RTMPose→MotionBERT 3D 파이프라인이 이미 구현돼 있어,
야구 포팅 가능성을 게이트 테스트로 먼저 판정했다.

### 검증 방법 — 강체 세그먼트 불변성

뼈 길이는 실제로 불변이므로 프레임 간 변동은 전부 추정 오차.
단축률 = 1 - p05/p95. **게이트: 15% 이하 통과.**

기준선
- 야구 2D (MediaPipe, 03번): 전완 69%, 어깨너비 65%, 골반너비 61%
- 배드민턴 3D (MotionBERT, 461클립 실측): 전완 32%, 어깨너비 12%, 골반너비 10%

### 실험 조건

| 실험 | 2D 모델 | 처리 | 평균 단축률 |
|---|---|---|---|
| 기준 | RTMPose-M (balanced) | 원본 프레임 | 35% |
| ① | RTMPose-L (performance) | 원본 프레임 | 40% ❌ |
| ② | RTMPose-M (balanced) | **bbox 크롭 + 2배 확대** | **26%** ✅ |

공통: 릴리스 앵커 [r-60, r+20] = 80프레임, MotionBERT-Lite (CLIP_LEN=243)

### 결과

| 세그먼트 | 야구2D | 배드민턴3D | 기준(M) | ①perf(L) | ②crop | 판정 |
|---|---|---|---|---|---|---|
| 몸통 | — | 10% | 22% | 25% | **7%** | ✅ 통과 |
| 대퇴(R) | 38% | 21% | 16% | 18% | **10%** | ✅ 통과 |
| 상완(L) | 49% | 22% | 27% | 27% | **13%** | ✅ 통과 |
| 어깨너비 | 65% | 12% | 40% | 49% | **22%** | ⚠ 경계 |
| 골반너비 | 61% | 10% | 49% | 61% | **24%** | ⚠ 경계 |
| 전완(L) | 61% | 35% | 47% | 46% | 30% | ❌ |
| 전완(R) | 69% | 32% | 45% | 43% | 39% | ❌ |
| 상완(R) | 38% | 21% | 35% | 49% | 42% | ❌ |
| 종아리(R) | 44% | 24% | 31% | 39% | 48% | ❌ |
| **평균** | | | **35%** | **40%** | **26%** | |

### 깊이축(Z) 정보량

| 관절 | 야구 Z/X | 배드민턴 Z/X |
|---|---|---|
| RWrist | 0.64 (측면 0.78) | 0.36 |
| RElbow | 0.48 | 0.29 |
| RShoulder | 0.35 | 0.16 |

→ 투수는 카메라 쪽으로 던지므로 깊이 방향 동작이 배드민턴보다 1.4~1.8배 크고,
   MotionBERT가 그 성분을 실제로 추정하고 있다.

### 결론

1. **원인은 피사체 크기였다.** 야구 중계에서 투수 신장은 화면상 **250px**
   (1280×720 기준 화면 높이의 35%). 모델을 키우면(①) 2D 신뢰도는 0.751→0.825로
   올랐지만 **3D는 오히려 악화**(35%→40%). 크롭 후 2배 확대하면(②) 평균
   35%→**26%**, 체간은 40~49%→**22~24%**로 개선.
2. **체간(몸통·골반·어깨)은 3D가 유효하다.** 몸통 7%로 게이트 통과.
   Statcast에 없는 체간 회전(`calc_trunk_rotation`) 확보 가능성이 열렸다.
3. **던지는 팔은 3D로도 부정확하다.** 전완(R) 39%, 상완(R) 42%. 가장 빠르고
   모션블러가 심한 부위. `arm_slot` 계열은 3D로도 신뢰 어려움.
4. **Stage 3은 "전면 도입"이 아니라 "체간 한정 도입"으로 좁힌다.** 체간은
   Statcast와 중복되지 않고 차원도 적게 쓴다(6 × mean/std = 12피처).

### 재현

- 스크립트: 세션 스크래치패드 `run3d.py` (`--mode`, `--crop`), `ablation.py`
- 모델: 배드민턴 프로젝트 체크포인트 재사용
  (`ai-badminton-coach/service/artifacts/MotionBERT/checkpoint/pose3d/...`)
- 영상: `test3d_videos/`, 결과 `test3d_out/`, `test3d_out_perf/`, `test3d_out_crop/`
- 의존성: `pip install rtmlib easydict ipdb`
