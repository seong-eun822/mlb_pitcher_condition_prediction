# 야구 투수 컨디션 예측 — 참고 논문 및 연구자료

> 수집 기준: MLB 투수 퍼포먼스 예측, Statcast 분석, 생체역학/영상 분석, 야구 지표 연구, 스포츠 analytics 방법론
> 총 **66편** (1차 33편 + 2차 33편, 국내 8편 포함)

---

## 카테고리별 목록

### 1차 수집 (33편)

| 카테고리 | 파일 | 논문 수 |
|---|---|---|
| 투수 퍼포먼스 예측 | [01_pitcher_performance.md](01_pitcher_performance.md) | 7편 |
| Statcast / 투구 데이터 분석 | [02_statcast_analysis.md](02_statcast_analysis.md) | 6편 |
| 생체역학 / 영상 분석 / 부상 예측 | [03_biomechanics_injury.md](03_biomechanics_injury.md) | 11편 |
| 야구 지표 연구 (wOBA, FIP 등) | [04_baseball_metrics.md](04_baseball_metrics.md) | 3편 |
| 스포츠 Analytics 방법론 | [05_sports_analytics.md](05_sports_analytics.md) | 3편 |
| 국내(한국어) 논문 | [06_korean_papers.md](06_korean_papers.md) | 3편 |

### 2차 수집 (33편) — [07_추가수집_2차.md](07_추가수집_2차.md)

| 카테고리 | 주제 | 논문 수 |
|---|---|---|
| A | Pitch Sequencing / 투구 배합 | 2편 |
| B | Release Point Consistency / 릴리스 포인트 일관성 ⭐ | 3편 |
| C | Spin Rate / 회전수와 헛스윙 ⭐ | 1편 |
| D | In-Game Velocity/Spin Decline = 피로 ⭐⭐ | 4편 |
| E | Pose Estimation / 자세 추정 | 3편 |
| F | Changepoint Detection / 변화점 탐지 ⭐ | 2편 |
| G | Batter-Pitcher Matchup / 매치업 | 3편 |
| H | Strike Zone / Command 예측 | 1편 |
| I | KBO / 한국 프로야구 ⭐⭐ | 5편 |
| J | Tommy John / UCL Biomechanics | 2편 |
| K | Statcast / TrackMan / Hawk-Eye 시스템 | 2편 |
| L | Workload Monitoring / 피로·투구량 관리 | 3편 |
| M | Pitch Tunneling / Deception (보너스) | 1편 |

> PDF 다운로드 현황은 [PDF/_다운로드_현황.md](PDF/_다운로드_현황.md) 참조

---

## 연구 트렌드 요약

1. **Transformer / LSTM 기반 시계열 예측**이 전통적 ML(랜덤포레스트, XGBoost)과 함께 주류
2. **방송 영상(monocular video) + 포즈 추정**으로 고가 모션캡처 없이 생체역학 지표 추출 — 2024~2026년 집중 발표
3. **SHAP으로 모델 해석력 확보**하는 explainable AI 접근이 스포츠 분야 전반 확산
4. **피로·부상 조기 경고**: 속도 감소, 릴리즈 포인트 변동, 스핀 변화 패턴을 실시간 추적하는 연구 등장
5. **국내 연구**는 KBO 승패 예측 중심 — 투수 개인 컨디션 예측·생체역학 분석 영역은 연구 공백 → **차별화 기회**

---

## 우리 프로젝트와 가장 관련 높은 논문 (⭐⭐ / ⭐)

| 논문 | 관련 이유 |
|---|---|
| **Evaluating Pitcher Fatigue Through Spin Rate Decline** (2차 D-1) | ⭐⭐ 회전수 감소=피로 — delta feature 직접 근거 |
| **KBO 헛스윙 1D-CNN** (강지연·조선미, 2차 I-2) | ⭐⭐ Y 지표(헛스윙) 동일 + 국내 사례, 직접 비교 대상 |
| A Linear Regression Model for Predicting Whiff % in MLB (02) | Y 지표(whiff%) 동일 |
| Spin Rate and Swinging Strike Probabilities (2차 C-1) | spin → whiff 관계 (Y 지표 직접 관련) |
| Release point variability ↔ pitching performance (2차 B-1) | 릴리스 포인트 분산 feature 이론적 근거 |
| Changepoint Detection in Player Performance (2차 F-1) | whiff%/구속 변화점 탐지 |
| KBO 구종 예측 SHAP (조선미, 2차 I-1) | 국내 + XGBoost + SHAP 동일 방법론 |
| PitcherNet: Powering the Moneyball Evolution (03) | 영상 기반 투수 키네마틱스 추출 — Phase 9 참고 |
| Scalable Injury-Risk Screening from Broadcast Video (03) | 방송 영상 + 생체역학 지표 — 동일 영상 소스 |
| Pitch-Tracking Metrics as Predictor of Injuries (03) | Statcast 지표 + XGBoost + SHAP — 동일 방법론 |
| Pitch Fatigue Detection System (SVM) (03) | 투수 피로 탐지 — 컨디션 예측 직접 관련 |
| Alterations in pitching biomechanics w/ increasing pitches (03) | 투구 수 증가에 따른 생체역학 변화 — 이론적 근거 |
| MediaPipe Pitching Phases 자동분류 (03) | MediaPipe 방법론 동일 |
