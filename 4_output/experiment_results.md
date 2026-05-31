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
```

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
