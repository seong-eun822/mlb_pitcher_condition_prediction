# MLB 선발투수 컨디션 조기 예측 프로젝트

## 프로젝트 개요

MLB 선발투수의 경기 초반 투구 데이터(Statcast 정형 데이터 + 영상 기반 biomechanical 데이터)를 활용하여, 이후 경기 퍼포먼스를 조기에 예측하는 AI 프로젝트.

기존 야구 분석이 시즌 성적이나 경기 종료 후 결과 중심이었다면, 본 프로젝트는 경기 초반 투구만으로 투수의 현재 컨디션과 메카닉 상태를 파악하고 미래 퍼포먼스를 예측하는 것을 목표로 한다.

핵심 철학: "누구인가?"가 아닌 "현재 어떤 상태인가?"를 예측한다.

---

## 프로젝트 목표

- 초반 투구 데이터 기반 투수 컨디션 조기 예측
- Biomechanical feature와 실제 경기 퍼포먼스의 관계 분석
- 현재 상태(state) 기반 퍼포먼스 예측 모델 구축
- SHAP 기반 feature importance 해석을 통한 컨디션 저하 신호 파악

---

## 데이터 범위

- 대상: MLB 전체 30개 팀 선발투수
- 기간: 2021~2025 (5년)
- 출처: Baseball Savant (Statcast 공개 데이터 + 투구별 영상 클립)
- 대상 제한: 선발투수 전용 (불펜 투수 제외)
  - 불펜은 총 투구수가 적어 X/Y 구간 분리가 불가능하기 때문

---

## 데이터 분할 전략

Random split 대신 시즌 기반 split을 사용하여 데이터 leakage를 방지하고, 실제 미래 예측 환경과 유사한 구조를 유지한다.

- Train: 2021~2023 (약 2,700경기)
- Validation: 2024
- Test: 2025

---

## 입력 데이터 (X)

경기 초반 투구를 집계하여 경기 단위(1경기 = 1row) feature로 변환한다.
X 구간은 n구 / n이닝 / n타자 단위 중 실험을 통해 최적값을 결정한다.

---

### 1. 정형 데이터 (Statcast)

| Feature | 설명 |
|---|---|
| 평균 구속 delta | 오늘 구속 - 직전 시즌 평균 구속 (구종별) |
| 구속 분산 | 구속 안정성 지표 |
| Spin rate | 구종별 평균 회전수 |
| Release point | 릴리스 포인트 x/z/y 좌표 |
| Extension | 릴리스 익스텐션 |
| Strike ratio | 스트라이크 비율 |
| 구종 비율 | Fastball계열 / Breaking계열 / Offspeed계열 |
| Arm angle | 팔 각도 |

구종은 세 그룹으로 분류한다.
- Fastball계열: FF, SI, FC
- Breaking계열: SL, CU, KC
- Offspeed계열: CH, FS

#### 개인 기준값 (delta feature)

pitcher_id를 입력에서 제외하되, 절대값 대신 직전 시즌 평균 대비 편차를 feature로 사용한다.

- 기준값: 직전 시즌 평균
- 장점: 데이터 leakage 없음, 시즌 초반부터 적용 가능
- 예시: 오늘 직구 평균 구속 - 직전 시즌 직구 평균 구속 = delta feature

이를 통해 동일 선수의 연도별 상태 변화를 서로 다른 state로 학습할 수 있다.
예: 2022년의 오타니 / 2023년의 오타니 / 2024년의 오타니를 각각 다른 state로 처리

---

### 2. 영상 기반 Biomechanical 데이터

#### 영상 수집 및 전처리

Baseball Savant에서 투구별 클립(약 9초)을 수집한다.
클립 내 투구 구간은 다음 방식으로 추출한다.

```
9초 클립
    ↓
YOLO로 공 인식
    ↓
공이 글러브에서 처음 보이는 프레임 감지 (릴리스 기준점)
    ↓
기준점 ±2초 구간 추출 (약 4초 클립)
    ↓
MediaPipe로 관절 좌표 추출
```

MediaPipe를 사용하는 이유:
- 단일 인물 모드로 투수 자동 감지 (별도 필터링 불필요)
- 안 보이는 관절을 유추하여 결측 최소화
- OpenPose 대비 다중 인물 혼입 문제 없음
- 사전 실험에서 투구 동작 정확도 유사함을 확인

#### 촬영 앵글

Baseball Savant 영상은 뒤통수(후방) 앵글로 촬영되어 있어, 아래 관절이 안정적으로 추출된다.

사용 관절:
- 귀 (양쪽) — 두 점을 이어 머리 방향 및 안정성 추적
- 어깨 (양쪽)
- 팔꿈치 (양쪽)
- 손목 (양쪽)
- 골반 (양쪽)
- 무릎 (양쪽)
- 발목 (양쪽)

#### 추출 Feature

뒤통수 앵글 특성을 고려하여 아래 feature를 설계한다.

| Feature | 설명 |
|---|---|
| 귀 중점 이동 | 양쪽 귀 중점의 프레임별 이동 → 머리 안정성 |
| 어깨 기울기/회전 | 양쪽 어깨 각도 변화 → 어깨 열림 타이밍 |
| Hip-shoulder separation | 골반 회전 vs 어깨 회전 타이밍 차이 → 몸통 꼬임 (뒤통수 앵글에서 특히 잘 관측됨) |
| 골반 좌우 기울기 | 하체 안정성 |
| 스트라이드 방향 일관성 | 발 착지 위치 재현성 |
| 관절 좌표 분산 | 전체 동작 일관성 (투구 간 변동성) |
| 팔꿈치 각도 변화 | 팔 동작 안정성 |
| 어깨-팔꿈치-손목 각도 | 팔 체인 안정성 |

#### 결측 처리

- confidence 낮은 관절 프레임 제거
- 이상치 프레임 삭제
- 핵심 관절 결측이 과도한 투구는 샘플에서 제외

---

## 타겟 데이터 (Y)

### ~~wOBA Against~~ → **whiff% (헛스윙률)** 로 변경

> **변경 이유** (04_modeling 베이스라인 결과 기반):  
> wOBA는 수비 운·상대 타자 수준 등 외부 노이즈가 너무 커서 R² ≈ 0.005~0.009에 그침.  
> whiff%는 투수가 직접 통제하는 지표로 외부 요인 영향이 적어 R² ≈ 0.08~0.09로 약 9배 개선됨.

### whiff% (헛스윙률)

X 구간 이후 투구에서의 헛스윙 비율.

```
whiff% = swinging_strike 수 / 전체 스윙 수

전체 스윙 = swinging_strike + swinging_strike_blocked + foul + foul_tip + hit_into_play
```

처리 방식:
- 스윙 수가 20개 미만인 경기는 분모 불안정으로 제외
- Statcast `description` 컬럼 기준으로 집계

집계 방식 예시:
```python
whiffs = (description.isin(['swinging_strike', 'swinging_strike_blocked'])).sum()
swings = (description.isin(['swinging_strike', 'swinging_strike_blocked',
                             'foul', 'foul_tip', 'hit_into_play'])).sum()
y_whiff = whiffs / swings
```

### wOBA Against (초기 설계 — 실험 후 변경)

초기에는 X 구간 이후 타석 결과 기반 wOBA를 Y로 설계했으나,  
경기 단위 wOBA의 분산이 극단적으로 크고 수비 운 등 외부 노이즈 영향이 커  
베이스라인 R² ≈ 0.005로 예측력이 거의 없음이 확인되어 whiff%로 변경함.

---

## 모델 구조

### 전체 파이프라인

```
[영상 클립] → YOLO → MediaPipe → 관절 좌표
                                      ↓
                              Biomechanical Feature 집계
                                      ↓
             Statcast 정형 Feature 집계 (delta feature 포함)
                                      ↓
                                    Concat
                                      ↓
                            XGBoost / CatBoost
                                      ↓
                             Y: wOBA Against
                                      ↓
                        SHAP 기반 Feature Importance 해석
```

### 사용 모델

- XGBoost / CatBoost (트리 기반 우선 적용)
- 향후 LSTM, Transformer 기반 sequence 모델로 확장 가능성 고려

---

## X 구간 실험 설계

X 구간을 하이퍼파라미터로 취급하여 실험 비교를 수행한다.

| 단위 | 예시 | 비고 |
|---|---|---|
| n구 | 초반 10구, 15구, 20구 | 타석 결과 희소 문제 주의 |
| n이닝 | 초반 1이닝, 2이닝 | 직관적이고 야구적으로 의미 있음 |
| n타자 | 초반 3타자, 9타자(한 바퀴) | 집계 안정성 가장 높음 |

각 구간별로 Y 예측력을 비교하여 최적 구간 결정.

---

## 모델링 파이프라인 및 실험 설계

> A/B 테스트(paired t-test) 적용 위치는 🔬 로 표시

---

### Phase 0. 데이터 수집 및 적재

- pybaseball로 2021~2025 Statcast 수집
- 연도별 Parquet 저장
- DuckDB 적재 및 기본 쿼리 확인

---

### Phase 1. 데이터 전처리 기초

- 정규시즌 필터링 (`game_type == 'R'`)
- 선발투수 필터링 (1이닝부터 등판 + 일정 이닝 이상)
- 구종 3그룹 분류 (Fastball / Breaking / Offspeed)
- 투수별 / 구종별 직전 시즌 평균 계산 (delta feature용 lookup table)

---

### Phase 2. 베이스라인 구축

- X 구간: n타자 9 고정
- 전처리 없음, NaN 모델 내부 처리
- XGBoost / CatBoost 기초 적합 (early stopping만)
- 결과: RMSE ~0.136, R² ~0.005~0.009
- SHAP 확인 → 절대값이 delta보다 상위 → delta feature 강화 필요 확인

| 실험 | 내용 |
|---|---|
| E0-1 | XGBoost early stopping 없음 vs 있음 과적합 비교 |
| E0-2 | CatBoost early stopping 없음 vs 있음 비교 |
| E0-3 | SHAP feature importance 확인 → 다음 실험 방향 결정 |

---

### Phase 3. Delta Feature 확장 실험

베이스라인은 speed delta만 존재 → spin / extension / release point delta 단계적 추가

| 실험 | 내용 | 비교 기준 |
|---|---|---|
| E1-1 | speed delta만 (베이스라인) | 기준 |
| E1-2 | speed + spin delta 추가 | E1-1 vs E1-2 |
| E1-3 | speed + spin + extension delta 추가 | E1-2 vs E1-3 |
| E1-4 | speed + spin + extension + release point delta 추가 | E1-3 vs E1-4 |
| 🔬 E1-5 | 최종 delta 구성 vs 베이스라인 **paired t-test** | **delta feature 기여도 통계적 검증** |

SHAP 재확인 → 절대값 vs delta 순위 변화 확인

---

### Phase 4. NaN 처리 전략 비교

delta feature 확장 확정 후 진행

| 실험 | 내용 | 비고 |
|---|---|---|
| E2-1 | 모델 내부 처리 (베이스라인) | 기준 |
| E2-2 | Offspeed 미등판 → delta 0 impute | 야구적 논리로 정당화 가능 |
| E2-3 | 전체 NaN → 중앙값 impute | |
| E2-4 | 결측 비율 30% 이상 컬럼 제거 | |
| E2-5 | E2-2 + E2-4 조합 | 가장 현실적인 조합 |

성능 테이블 비교 후 최적 전략 확정 (도메인 논리 우선)

---

### Phase 5. X 구간 실험

NaN 처리 확정 후 진행 (데이터셋 고정)

| 실험 | X 구간 | 비고 |
|---|---|---|
| E3-1 | n구 = 10 | 샘플 희소 주의 |
| E3-2 | n구 = 15 | |
| E3-3 | n구 = 20 | |
| E3-4 | n이닝 = 1 | |
| E3-5 | n이닝 = 2 | |
| E3-6 | n타자 = 3 | |
| E3-7 | n타자 = 6 | |
| E3-8 | n타자 = 9 (베이스라인) | 기준 |

RMSE / R² 비교 테이블로 최적 구간 확정 → 이후 모든 실험 고정값으로 사용

---

### Phase 6. 이상치 처리

| 실험 | 내용 | 비고 |
|---|---|---|
| E4-1 | 없음 (베이스라인) | 기준 |
| E4-2 | 구속 상하 1% clip | |
| E4-3 | total_pitches 극단값 제거 (3구 이하 경기 등) | |
| E4-4 | E4-2 + E4-3 조합 | |

분포 시각화로 전후 비교 후 성능 변화 확인

---

### Phase 7. 모델 튜닝

X 구간 / NaN / 이상치 모두 확정된 상태에서 진행

| 실험 | 내용 | 비고 |
|---|---|---|
| E5-1 | XGBoost Optuna 50 trials | |
| E5-2 | CatBoost Optuna 50 trials | |
| 🔬 E5-3 | XGBoost vs CatBoost **paired t-test** | **최종 모델 선택** |
| E5-4 | 단순 평균 앙상블 | E5-3에서 차이 없을 경우 |
| E5-5 | Weighted 앙상블 | |

---

### Phase 8. Feature 선택

튜닝 완료 후 진행

| 실험 | 내용 | 비고 |
|---|---|---|
| E6-1 | SHAP 하위 20% feature 제거 | |
| E6-2 | SHAP 하위 40% feature 제거 | |
| E6-3 | 구종별 feature vs 그룹별 feature 비교 | |
| E6-4 | 최적 feature set 확정 후 성능 비교 | |

---

### Phase 9. 영상 Biomechanical Feature 추가

영상 파이프라인:
- Baseball Savant 크롤링 + 영상 다운로드
- YOLO로 릴리스 포인트 기준 프레임 감지
- MediaPipe로 관절 좌표 추출
- Biomechanical feature 집계

| 실험 | 내용 | 비고 |
|---|---|---|
| E7-1 | 정형 단독 모델 (최종 확정본) | 기준 |
| E7-2 | 정형 + 머리 안정성 추가 | 단계적 추가 |
| E7-3 | 정형 + Hip-shoulder separation 추가 | |
| E7-4 | 정형 + 전체 biomechanical feature 추가 | |
| 🔬 E7-5 | E7-1 vs E7-4 **paired t-test** | **biomechanical feature 기여도 검증** |
| E7-6 | SHAP으로 biomechanical feature 중요도 확인 | 어떤 신체 지표가 핵심인지 |

---

### Phase 10. 최종 모델 평가 및 해석

- Test (2025) 최종 평가
- SHAP feature importance 시각화
- 컨디션 저하 신호 top feature 도출
- 투수 유형별 패턴 분석 (파워피처 vs 커맨드피처)
- 예측값 분포 시각화
- 정형 단독 vs 정형+영상 최종 비교 정리

---

### 포폴 핵심 스토리라인

```
베이스라인 (RMSE 0.136, R² 0.005)
    ↓
delta feature 확장
    ↓ 🔬 paired t-test → delta feature 기여도 검증
NaN 전략 확정 (도메인 논리 기반)
    ↓
X 구간 최적화 (성능 테이블 비교)
    ↓
이상치 처리
    ↓
모델 튜닝
    ↓ 🔬 paired t-test → XGBoost vs CatBoost 최종 모델 선택
Feature 선택 (SHAP 기반)
    ↓
영상 biomechanical feature 추가
    ↓ 🔬 paired t-test → biomechanical feature 기여도 검증
SHAP 해석 → 컨디션 저하 신호 도출
    ↓
최종 Test 평가 (2025)
```

### A/B 테스트 적용 위치 요약

| 위치 | 비교 대상 | 목적 |
|---|---|---|
| 🔬 Phase 3 (E1-5) | 베이스라인 vs delta feature 확장 | delta feature가 실제로 유의미한 개선인가 |
| 🔬 Phase 7 (E5-3) | XGBoost vs CatBoost 튜닝 후 | 수치 차이가 작을 때 우열 통계적 확정 |
| 🔬 Phase 9 (E7-5) | 정형 단독 vs 정형+영상 융합 | biomechanical feature 추가가 유의미한가 |

---

## 차별화 포인트

- 문제 정의: 경기 초반 n구만으로 이후 퍼포먼스를 예측하는 state 기반 접근
- Delta feature: 선수 identity 제거 후 개인 기준값 대비 편차 사용
- 영상 기반 biomechanical feature: Statcast에 없는 신체 정보 추가
- Hip-shoulder separation: 뒤통수 앵글에서 특히 잘 관측되는 핵심 메카닉 지표
- X 구간 실험: n구 / n이닝 / n타자 비교 실험 자체가 contribution
- SHAP 해석: 어떤 지표가 컨디션 저하 신호인지 분석

---

## 프로젝트 한계 및 보완 사항

### Y값 한계
- wOBA는 수비 운의 영향을 일부 받음
- xwOBA 대비 노이즈가 있을 수 있으나 삼진/볼넷 처리 일관성을 위해 wOBA 사용

### 영상 처리 한계
- 투구 중 관절이 가려지는 경우 결측 발생
- 클립마다 프레임 수 불일치 → 패딩 또는 리샘플링 전략 필요
- 대용량 영상 처리로 인한 시간/용량 부담

### 샘플 수
- 30팀 × 3년(train) 기준 약 2,700경기
- 트리 모델 적용에는 충분한 수준

### 불펜 투수 미적용
- 총 투구수가 적어 X/Y 구간 분리 불가
- 별도 모델 설계 필요 (본 프로젝트 범위 외)

---

## 기대 효과

- 경기 초반 투수 컨디션 조기 감지
- Statcast 정형 데이터 + 영상 biomechanical 데이터 융합 분석
- 선수 identity 독립적인 상태 기반 예측 모델 구축
- SHAP 기반 컨디션 저하 신호 해석 및 시각화
