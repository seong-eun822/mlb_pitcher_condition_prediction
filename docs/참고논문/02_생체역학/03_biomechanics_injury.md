# 03. 생체역학 / 영상 분석 / 부상 예측 논문

---

### 1. PitcherNet: Powering the Moneyball Evolution in Baseball Video Analytics ⭐
- **저자**: Bright, J. et al.
- **연도**: 2024
- **출처**: arXiv (arXiv:2405.07407)
- **URL**: https://arxiv.org/abs/2405.07407
- **한줄 요약**: 방송 영상에서 투수 키네마틱스를 자동 분석하는 엔드투엔드 시스템 — 구속·릴리즈 포인트·피치 포지션 등을 추출하며 투수 식별 정확도 96.82%
- **⭐ 프로젝트 관련**: Baseball Savant 영상 기반 feature 추출 — Phase 9 핵심 참고 논문

---

### 2. Scalable Injury-Risk Screening in Baseball Pitching From Broadcast Video ⭐
- **저자**: Bright, J. et al.
- **연도**: 2026
- **출처**: arXiv (arXiv:2603.04864)
- **URL**: https://arxiv.org/abs/2603.04864
- **한줄 요약**: 방송 단안(monocular) 영상에서 18개 생체역학 지표를 추출하여 Tommy John 수술 위험 예측(AUC 0.811), 7,348명의 투수 데이터 활용
- **⭐ 프로젝트 관련**: 동일한 방송 영상 소스 + 생체역학 지표 추출 방법론

---

### 3. Interpretable Pre-Release Baseball Pitch Type Anticipation from Broadcast 3D Kinematics
- **저자**: (arXiv:2603.04874)
- **연도**: 2026
- **출처**: arXiv
- **URL**: https://arxiv.org/html/2603.04874
- **한줄 요약**: 방송 영상의 3D 키네마틱스만으로 투구 전 투구 유형을 예측, 손목 위치(14.8%)와 머리 방향(19.0%)이 가장 중요한 관절로 분석

---

### 4. Automated Classification of Baseball Pitching Phases Using ML and AI-Based Posture Estimation ⭐
- **저자**: (MDPI Applied Sciences)
- **연도**: 2025
- **출처**: Applied Sciences, MDPI (Vol. 15, No. 22)
- **URL**: https://www.mdpi.com/2076-3417/15/22/12155
- **한줄 요약**: **MediaPipe** pose estimation으로 고등학교 투수 500명의 슬로우모션 영상에서 투구 5단계(와인드업~팔로스루)를 자동 분류
- **⭐ 프로젝트 관련**: MediaPipe 사용 방법론 동일 — 직접 참고 가능

---

### 5. A CNN–LSTM Framework for Player-Specific Baseball Pitch Type Prediction from Video Sequences
- **저자**: (Applied System Innovation, MDPI)
- **연도**: 2026
- **출처**: Applied System Innovation (ASI), MDPI (Vol. 9, No. 4)
- **URL**: https://doi.org/10.3390/asi9040075
- **한줄 요약**: ResNet-50 + YOLOv8 CNN으로 공간 특징을 추출하고 LSTM으로 시계열 패턴을 학습하여 구종 분류 정확도 91.8% 달성

---

### 6. Pitch-Tracking Metrics as a Predictor of Future Shoulder and Elbow Injuries in MLB Pitchers ⭐
- **저자**: (PMC / PubMed)
- **연도**: 2024
- **출처**: PMC
- **URL**: https://pmc.ncbi.nlm.nih.gov/articles/PMC11369970/
- **한줄 요약**: 2017~2022 MLB 투수 데이터로 XGBoost 모델을 학습, 피치 트래킹 지표(구속·회전수·수평무브먼트)가 인구통계·투구수보다 부상 예측에 더 유의미함을 확인
- **⭐ 프로젝트 관련**: Statcast 지표 + XGBoost + SHAP — 동일 방법론

---

### 7. Predicting Ulnar Collateral Ligament (UCL) Injury in Rookie MLB Pitchers
- **저자**: (arXiv:2207.00585)
- **연도**: 2022
- **출처**: arXiv
- **URL**: https://arxiv.org/abs/2207.00585
- **한줄 요약**: 루키 MLB 투수의 UCL 부상을 ML로 예측, 데이터 기반 조기 경고 시스템의 가능성 제시

---

### 8. Data-driven approaches for predicting Tommy John Surgery risk in MLB pitchers
- **저자**: (ResearchGate)
- **연도**: 2025
- **출처**: ResearchGate
- **URL**: https://www.researchgate.net/publication/390699650_Data-driven_approaches_for_predicting_Tommy_John_Surgery_risk_in_major_league_baseball_pitchers
- **한줄 요약**: 2016~2023 MLB 데이터로 최대 100일 전 부상 위험 탐지(F1=0.73), 수술까지 남은 기간 회귀 예측(R²=0.79) 모델 개발

---

### 9. Design and Analysis of a Pitch Fatigue Detection System for Adaptive Baseball Learning ⭐
- **저자**: (Frontiers in Psychology / PMC)
- **연도**: 2021
- **출처**: Frontiers in Psychology; PMC
- **URL**: https://pmc.ncbi.nlm.nih.gov/articles/PMC8711585/
- **한줄 요약**: 팔꿈치·등의 각도 변화로 피로 지수를 산출하는 SVM 기반 투수 피로 탐지 시스템 개발, 예측 정확도 89.1%
- **⭐ 프로젝트 관련**: 투수 피로/컨디션 탐지 — 프로젝트 목표와 직접 관련

---

### 10. Alterations in pitching biomechanics and performance with an increasing number of pitches ⭐
- **저자**: Yanagisawa et al.
- **연도**: 2024
- **출처**: PM&R (Wiley Online Library)
- **URL**: https://onlinelibrary.wiley.com/doi/10.1002/pmrj.13054
- **한줄 요약**: 투구 수가 증가함에 따른 생체역학적 변화와 퍼포먼스 저하를 서사적으로 리뷰, 피로 누적의 구체적인 메커니즘 분석
- **⭐ 프로젝트 관련**: 투구 수 증가 → 컨디션 저하 메커니즘 이론적 근거

---

### 11. Mitigating Motion Blur for Robust 3D Baseball Player Pose Modeling for Pitch Analysis
- **저자**: (arXiv:2309.01010)
- **연도**: 2023
- **출처**: arXiv
- **URL**: https://arxiv.org/pdf/2309.01010
- **한줄 요약**: 고속 투구 동작의 모션 블러를 완화하는 방법으로 방송 영상에서 3D 포즈 모델링 품질을 향상시켜 투구 분석에 적용
