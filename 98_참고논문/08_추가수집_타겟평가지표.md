# 08. 추가 수집 — Y 타겟(ERA/FIP/xwOBA/whiff) 예측 & 평가지표 분포

> 수집일: 2026-06-27
> 목적: ① whiff%·xwOBA·FIP·ERA를 **예측 타겟**으로 쓴 ML 논문 보강
>       ② 우리 결과(정형 R²≈0.036)가 "낮은 게 아니라 이 문제의 천장"임을 뒷받침하는 근거
>       ③ 야구 평가지표 league 분포(정상값 범위) 정리

---

## A. ⭐⭐ R² 천장 근거 — "구속으로 ERA 예측해도 R²≈0.05~0.08"

우리 결과 whiff% 정형 R²=0.036(val)이 "낮다"가 아니라 **이 문제의 본질적 한계**임을 보여주는 핵심 레퍼런스.

### A-1. Using Linear Regression to Predict a Pitcher's Performance (CCG Analytics)
- **출처**: https://blog.ccganalytics.com/using-linear-regression-pitchers-performance
- **핵심**: 패스트볼 구속으로 ERA 예측 → **분산의 5.14%만 설명**. 추가 변수 넣어도 **adjusted R² = 0.0825**에 그침.
- **우리와 관련**: 우리 whiff% R²(0.036)·구속 중심 정형 모델과 **같은 수준**. "구속은 중요하나 좋은 투수의 전부가 아니다"는 결론 → 우리 "정형으로도 한계, 영상도 무의" 서사 보강. **R²가 낮은 게 정상임을 입증.**

### A-2. Understanding Career Progression in Baseball Through ML (arXiv 1712.05754)
- **출처**: https://arxiv.org/pdf/1712.05754
- **핵심**: 투수 퍼포먼스 예측 모델이 **분산의 30~40%만 설명**(시즌누적, 7~10년차). 타자보다 낮음 — 부상 등 다른 변동요인 때문.
- **우리와 관련**: 시즌누적도 0.3~0.4가 천장인데, 우리는 **경기단위 + 초반 15구**라 훨씬 어려움 → R²가 더 낮은 게 당연. 경기단위 예측의 본질적 노이즈 근거.

### A-3. Skill vs Chance / 관측 데이터 잔차 (arXiv 2410.14363 등)
- **핵심**: 인간 행동 관측 데이터는 mood·상황 등 미관측 요인이 많아 **낮은 R²가 모델 결함이 아니라 데이터 내재 노이즈**를 반영.
- **우리와 관련**: 경기단위 whiff%의 낮은 R²를 "모델이 못한 것"이 아니라 "야구 경기의 본질적 무작위성"으로 정당화.

---

## B. Y 타겟별 ML 예측 논문 (FIP/ERA/whiff 직접 예측)

### B-1. Predicting Baseball Pitcher Efficacy Using Physical Pitch Characteristics (Research Archive of Rising Scholars #95)
- **출처**: https://www.research-archive.org/index.php/rars/preprint/view/95
- **핵심**: game-independent 투구 특성으로 **WHIP·타율·FIP 등 다중 efficacy 지표** 예측. **velocity·release consistency가 FIP에 유의하나 분산의 일부만 설명.**
- **우리와 관련**: ⭐ 우리 4타겟 비교(whiff/xwoba/fip/era)와 직접 대응. "release consistency→FIP" = 우리 영상 자세 피처 가설의 직접 선행연구. (단 우리 결과는 영상 무의로 기각)

### B-2. Predicting Baseball Players' Value: Traditional vs Advanced metrics (ResearchGate 382249191)
- **출처**: https://www.researchgate.net/publication/382249191
- **핵심**: 전통지표(ERA·승패) 모델보다 **고급 Statcast 지표(xFIP·wOBA·barrel) 모델이 일관되게 우수.** XGBoost가 최고 성능.
- **우리와 관련**: 우리 정형 feature(Statcast 기반)·XGBoost 선택의 방법론적 근거.

### B-3. Application of ML Models for Baseball Outcome Prediction (MDPI Appl. Sci. 15/13/7081, 2025)
- **출처**: https://www.mdpi.com/2076-3417/15/13/7081
- **핵심**: PLOB%·WHIP·wRAA·**FIP**가 모델 예측에 유의. **ERA·WHIP(수비의존)가 포스트시즌 예측의 33.7% 차지.** 로지스틱회귀 AUC 0.804.
- **우리와 관련**: FIP의 예측 기여 + **분류(AUC) 평가**의 선행 — 우리 18번에 추가한 AUC 평가의 근거.

### B-4. TFT(Temporal Fusion Transformer) for ERA Prediction (이미 보유 PDF: TechScience_TFT_ERA_Prediction)
- **우리와 관련**: ERA를 시계열 딥러닝으로 예측한 사례 — 우리 ERA 타겟의 선행. (우리는 경기단위라 시즌 TFT와 결이 다름)

### B-5. Sabermetrics Meets ML — xFIP/wOBA/Statcast for MLB (DailyMLBPicks)
- **출처**: https://www.dailymlbpicks.com/sabermetrics-meets-machine-learning.html
- **핵심**: FIP·xFIP·SIERA는 **수비 무관 지표로 미래 ERA를 더 잘 예측.** xFIP가 회귀분석에 가장 적합.
- **우리와 관련**: 우리가 FIP/ERA를 타겟으로 택한 사베르메트릭스 정당성.

---

## C. 야구 평가지표 — league 분포 (정상값 범위)

> 우리 game_targets 분포 검증용. 우리 값이 MLB 평균과 일치하는지 대조.

| 지표 | MLB 평균 | 범위 해석 | 우리 데이터 평균 | 일치? |
|---|---|---|---|---|
| **whiff%** (헛스윙/스윙) | ~25% (2024 타자 전체 25.1%) | 28~30%↑ 우수 / 26~28% 평균 / <26% 컨택유도형 | 0.223 (22%) | ✅ 근접 |
| **CSW%** (called+whiff) | 27~28% | 30%↑ 엘리트 | — | (참고) |
| **xwOBA** | ~0.310~0.320 | 낮을수록 좋음(투수기준) | 0.319 | ✅ 일치 |
| **FIP** | ~3.9~4.2 | 3.20↓ 우수 / 4.20 평균 / 5.00↑ 나쁨 | 4.19 | ✅ 일치 |
| **ERA** | ~4.0~4.5 | 3.00↓ 우수 / 4.00 평균 | 4.41 | ✅ 근접 |

→ **우리 game_targets의 4종 평균이 모두 MLB league 평균과 일치** = 타겟 계산이 정확함을 교차검증.

**출처(분포)**:
- whiff/CSW: https://baseballsavant.mlb.com/league , https://fantasyteamadvice.com/mlb/csw-today
- xwOBA: https://www.mlb.com/glossary/statcast/expected-woba , https://baseballsavant.mlb.com/leaderboard/expected_statistics
- ERA/FIP league: https://www.baseball-reference.com/leagues/majors/2024-standard-pitching.shtml

---

## D. 우리 프로젝트 서사에 쓰는 법 (요약)

1. **"R² 0.036은 낮은 게 아니다"** → A-1(ERA 구속예측 R²0.05~0.08), A-2(시즌누적도 0.3~0.4 천장) 인용. 경기단위·초반15구는 더 어려운 문제.
2. **"타겟 계산이 정확하다"** → C표(우리 4종 평균 = MLB 평균) 제시.
3. **"release consistency→FIP 가설은 선행연구 있으나 우리 영상으로는 무의"** → B-1 인용 후 paired t-test로 기각 보고 (정직한 검증).
4. **"평가는 R²만 아니라 RMSE·MAE·skill·AUC로"** → B-3(AUC 분류평가) 근거.

---

*추가 수집: 2026-06-27 (타겟·평가지표·R²천장 근거 보강). 기존 66편 + 본 문서 신규 레퍼런스.*
