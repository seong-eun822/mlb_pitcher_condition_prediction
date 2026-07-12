# 07. 추가 수집 논문 (2차) — 신규 33편

> 1차 33편과 중복 없는 신규 논문. whiff%, 릴리스 포인트, 구속/회전수 저하(피로), KBO 등 우리 프로젝트 직결 주제 집중 보강.

---

## A. Pitch Sequencing / 투구 배합

### A-1. No Pitch Is an Island: Pitch Prediction With Sequence-to-Sequence Deep Learning
- 출처: FanGraphs Community Blog | 2020년대
- URL: https://community.fangraphs.com/no-pitch-is-an-island-pitch-prediction-with-sequence-to-sequence-deep-learning/
- 요약: seq2seq 어텐션 구조로 직전 투구 시퀀스를 입력해 다음 구종 예측
- ※ 블로그(학술 인용 주의)

### A-2. Baseball Pitch Sequence Prediction (7 ML Models Benchmark)
- 저자: jman4162 (J. Nadar) | 2024 | GitHub 오픈소스
- URL: https://github.com/jman4162/Baseball-Pitch-Sequence-Prediction
- 요약: LSTM/Transformer/CNN/HMM/RF/Logistic/AutoGluon 7개 모델 k-fold 벤치마크
- ※ 오픈소스 프로젝트(학술 논문 아님)

---

## B. Release Point Consistency / 릴리스 포인트 일관성 ⭐

### B-1. Relationship between ball release point variability and pitching performance in MLB ⭐
- 출처: Frontiers in Sports and Active Living (PMC11608975) | 2024
- URL: https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2024.1447665/full
- 요약: MLB가 MiLB보다 릴리스 포인트 변동성 작고, **수평 방향 변동성이 삼진 능력과 가장 강한 상관**. 일관성=성능 정량화
- **⭐ 프로젝트 직결**: 릴리스 포인트 분산 feature의 이론적 근거

### B-2. The relationship between pitching parameters and release points of different pitch types in MLB
- 출처: Frontiers (PMC10164925) | 2023
- URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC10164925/
- 요약: 구종별 릴리스 포인트와 구속·무브먼트 등 투구 파라미터 간 관계 분석

### B-3. Variability of in-game markerless and laboratory marker-based baseball pitching biomechanics
- 출처: Journal of Biomechanics / ScienceDirect | 2025
- URL: https://www.sciencedirect.com/science/article/abs/pii/S0021929025002878
- 요약: 경기(마커리스) vs 실험실(마커) 투구 바이오메카닉스 변동성 비교 — 측정 신뢰도 검증

---

## C. Spin Rate / 회전수와 헛스윙 ⭐

### C-1. Spin Rate and Swinging Strike Probabilities ⭐
- 저자: Jim Albert 등 (baseballwithr) | 2017
- URL: https://baseballwithr.wordpress.com/2017/01/16/spin-rate-and-swinging-strike-probabilities/
- 요약: **회전수에 따른 헛스윙 확률**을 통계 모델로 추정, 구종별 spin rate의 헛스윙 기여도 분석
- **⭐ 프로젝트 직결**: spin → whiff 관계 (Y 지표 직접 관련)

---

## D. In-Game Velocity/Spin Decline = 피로 ⭐⭐

### D-1. Evaluating Pitcher Fatigue Through Spin Rate Decline: A Statcast Data Analysis ⭐⭐
- 출처: Paripex Indian Journal of Research | 2025.02
- URL: https://www.worldwidejournals.com/paripex/recent_issues_pdf/2025/February/evaluating-pitcher-fatigue-through-spin-rate-decline-a-statcast-data-analysis_February_2025_1447180022_0900292.pdf
- 요약: **피로의 1차 지표가 구속이 아니라 회전수 감소**라는 점을 Statcast로 분석. 전통 가정에 도전
- **⭐⭐ 프로젝트 핵심**: 우리 delta feature(구속/회전수 변화) 설계의 직접 근거. PDF 직접 다운 가능

### D-2. Effect of Fatigue on Medial Elbow Torque in Baseball Pitchers: A Simulated Game Analysis
- 출처: PubMed 29953258 (Am J Sports Med) | 2018
- URL: https://pubmed.ncbi.nlm.nih.gov/29953258/
- 요약: 이닝 갈수록 구속 감소하나 3이닝 후 내측 팔꿈치 토크 증가 — 피로-부상 연결

### D-3. Manifestations of muscle fatigue in baseball pitchers: a systematic review
- 출처: PMC6673423 | 2019
- URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC6673423/
- 요약: 투수 근피로가 구속 저하·커맨드 손실로 발현되는 양상 종합 리뷰

### D-4. The Impact of Fatigue on the Kinematics of Collegiate Baseball Pitchers
- 출처: PMC4555605 | 2015
- URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC4555605/
- 요약: 피로 시 대학 투수의 투구 키네매틱스 변화 측정

---

## E. Pose Estimation / 자세 추정 (스포츠 일반)

### E-1. Pose2Sim: End-to-End 3D Markerless Sports Kinematics (Part 1 & 2)
- 저자: D. Pagnon 등 | 2021/2022 | MDPI Sensors
- URL: https://www.mdpi.com/1424-8220/21/19/6530 , https://www.mdpi.com/1424-8220/22/7/2712
- 요약: OpenPose 다중뷰 2D → 삼각측량 → OpenSim 역운동학으로 3D 마커리스 키네매틱스 산출

### E-2. Deep-Learning-Based Markerless Pose Estimation in Gait Analysis (DeepLabCut)
- 출처: arXiv:2407.10590 | 2024
- URL: https://arxiv.org/abs/2407.10590 | PDF: https://arxiv.org/pdf/2407.10590
- 요약: DeepLabCut 커스텀 학습+refinement로 마커리스 자세 추정 정확도 향상

### E-3. Commercial vision sensors and AI-based pose estimation for markerless motion analysis in sports
- 출처: PMC12378739 | 2025
- URL: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12378739/
- 요약: OpenPose 등 AI 자세추정 프레임워크 + 상용 비전 센서 스포츠 동작 분석 리뷰

---

## F. Changepoint Detection / 변화점 탐지 ⭐

### F-1. Tractable Algorithms for Changepoint Detection in Player Performance Metrics ⭐
- 출처: arXiv:2510.25961 | 2025
- URL: https://arxiv.org/abs/2510.25961 | PDF: https://arxiv.org/pdf/2510.25961
- 요약: 선수별 **헛스윙률·패스트볼 구속** 변화점을 실시간 탐지(91% 정확도), 변화의 60%+가 시즌 중 발생
- **⭐ 프로젝트 관련**: whiff%/구속 변화 탐지 — 컨디션 변화 감지 직접 관련

### F-2. Doubly-online changepoint detection for monitoring health status during sports
- 출처: arXiv:2206.11578 | 2022
- URL: https://arxiv.org/abs/2206.11578 | PDF: https://arxiv.org/pdf/2206.11578
- 요약: 가우시안 상태공간+온라인 EM으로 운동 중 건강 상태 변화점 실시간 추정

---

## G. Batter-Pitcher Matchup / 매치업

### G-1. Singlearity: Using A Neural Network to Predict Plate Appearance Outcomes
- 출처: Baseball Prospectus | 2021
- URL: https://www.baseballprospectus.com/news/article/59993/
- 요약: 타자·투수 스탯 입력 → 타석 결과 확률 분포 출력 신경망

### G-2. Modeling the probability of a batter/pitcher matchup event: A Bayesian approach
- 출처: PLOS ONE (PMC6192592) | 2018
- URL: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0204874
- 요약: 베이지안 접근으로 타자-투수 매치업 이벤트 확률 모델링

### G-3. Nuclear penalized multinomial regression for predicting at-bat outcomes
- 출처: arXiv:1706.10272 | 2017
- URL: https://arxiv.org/abs/1706.10272 | PDF: https://arxiv.org/pdf/1706.10272
- 요약: nuclear-norm 패널티 다항 회귀로 play-by-play 매치업 타석 결과 확률 예측

---

## H. Strike Zone / Command 예측

### H-1. A context-enhanced deep learning approach to predict baseball pitch location
- 출처: Springer Sports Engineering | 2025
- URL: https://link.springer.com/article/10.1007/s12283-025-00497-5
- 요약: 릴리스 트래킹 지표(측면 릴리스·회전수 핵심)+발사체 운동 특징 DNN으로 투구 로케이션 예측

---

## I. KBO / 한국 프로야구 ⭐⭐

### I-1. 한국프로야구(KBO) 투구 데이터 기반 다중분류 AI 구종 예측과 SHAP 해석 ⭐
- 저자: 조선미 | 2025 | 한국체육측정평가학회지 27(3) (KCI)
- URL: https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003251281
- 요약: KBO 2023~24 데이터로 8구종 XGBoost 분류(정확도 87.9%), SHAP으로 pfx_z·vy0·speed 중요도 해석
- **⭐ 국내 + XGBoost + SHAP 동일 방법론**

### I-2. 한국프로야구 헛스윙 예측을 위한 딥러닝 기반 1D-CNN 모델 개발 및 적용 ⭐⭐
- 저자: 강지연, 조선미 | 2023.11 | 코칭능력개발지 (DBpia/KCI)
- URL: https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11638676
- 요약: KBO 2022~23 데이터로 1D-CNN **헛스윙 예측(정확도 86.22%)**. SHAP상 존 좌표·카운트·구속이 핵심
- **⭐⭐ 프로젝트 핵심**: Y 지표(헛스윙) 동일 + 국내 사례. 직접 비교 대상

### I-3. AI 기반 KBO 자동 투구 판정 시스템(ABS) 일관성 분석
- 출처: 한국체육측정평가학회지 (DBpia/KCI) | 2024
- URL: https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11841202
- 요약: KBO ABS 판정 일관성을 AI/ML 관점 분석

### I-4. Analyzing the impact of the automatic ball strike system in KBO
- 출처: Scientific Reports (Nature) | 2025
- URL: https://www.nature.com/articles/s41598-025-28142-y
- 요약: 세계 최초 KBO ABS 시스템 영향을 컴퓨터 비전·ML 기반 분석

### I-5. ML-Based Classification of Team Playoff Advancement Using Pitching Metrics in KBO
- 출처: ResearchGate (401245930) | 2025
- URL: https://www.researchgate.net/publication/401245930
- 요약: ERA/WHIP(수비 의존) vs FIP(수비 독립) 투구 지표로 KBO 플레이오프 진출 ML 분류

---

## J. Tommy John / UCL Biomechanics (추가)

### J-1. Baseball Pitching Biomechanics Shortly After UCL Repair
- 출처: PubMed 31489335 (Am J Sports Med) | 2019
- URL: https://pubmed.ncbi.nlm.nih.gov/31489335/
- 요약: UCL 수복 직후 관절 키네틱 차이 없으나 팔꿈치 신전·신전속도 3개 변수 유의차

### J-2. Effect of UCL Reconstruction on Pitch Accuracy, Velocity, and Movement in MLB
- 출처: PMC7747121 | 2020
- URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC7747121/
- 요약: 토미존 수술 후 제구·구속·무브먼트 변화 MLB 데이터 분석

---

## K. Statcast / TrackMan / Hawk-Eye 시스템

### K-1. A Tracking System For Baseball Game Reconstruction
- 출처: arXiv:2003.03856 | 2020
- URL: https://arxiv.org/abs/2003.03856 | PDF: https://arxiv.org/pdf/2003.03856
- 요약: 방송 영상으로 공·선수 추적 경기 재구성(Statcast 대안)

### K-2. Neural Network-Based Tracking and 3D Reconstruction of Pitch Trajectories from Single-View 2D Video ⭐
- 저자: Jhen Hsieh | 2024.05 | arXiv:2405.16296
- URL: https://arxiv.org/abs/2405.16296 | PDF: https://arxiv.org/pdf/2405.16296
- 요약: 단일 시점 2D 영상에서 CSRT 추적+FC 신경망으로 투구 3D 궤적 복원(저비용 Statcast 대안)
- **⭐ 프로젝트 관련**: 단안 영상 처리 — 영상 파이프라인 참고

---

## L. Workload Monitoring / 피로·투구량 관리

### L-1. A Review of Workload-Monitoring Considerations for Baseball Pitchers
- 출처: PMC7534929 (J Athl Train) | 2020
- URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC7534929/
- 요약: 투구 수 중심 모니터링 한계와 acute:chronic workload 등 종합 워크로드 고찰

### L-2. Workload Risk Factors for Pitching-Related Injuries in High School Pitchers
- 저자: Zaremski 등 | 2024 | Am J Sports Med (PubMed 38700088)
- URL: https://journals.sagepub.com/doi/10.1177/03635465241246559
- 요약: 고교 투수 투구 관련 부상 워크로드 위험인자 분석

### L-3. Unaccounted Workload Factor: Game-Day Pitch Counts in High School Pitchers
- 출처: PMC5894908 | 2018
- URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC5894908/
- 요약: 워밍업·불펜 투구가 acute 워크로드의 30~40% 추가 차지

---

## M. Pitch Tunneling / Deception (보너스)

### M-1. Quantifying Pitcher Deception
- 출처: Towards Data Science | 2022
- URL: https://towardsdatascience.com/quantifying-pitcher-deception-7fb2288661c8/
- 요약: 릴리스·터널링·perceived velocity 종합해 투수 디셉션 정량화
- ※ 블로그(학술 인용 주의)
