TFT 강화학습 에이전트를 학습시켜줘.

1단계: 환경 준비

tft-sim conda 환경에서 gymnasium, stable-baselines3, torch, tensorboard를 pip으로 설치해줘.

2단계: 환경 테스트

src/rl/env/tft_env.py의 TFTEnv를 4명 플레이어로 생성하고, 랜덤 행동으로 에피소드 하나를 돌려서 환경이 정상 작동하는지 확인해줘. observation shape, action space 크기, valid action 개수를 출력해줘.

3단계: 단순화 환경 학습

먼저 4명 플레이어, 30라운드 제한으로 단순화된 환경에서 20만 스텝 학습해줘. 병렬 환경 4개 사용하고, 학습률 3e-4로 설정해. 모델은 models/tft_simple에, 로그는 logs/tft_simple에 저장해줘.

4단계: 평가

학습된 모델로 100 에피소드 평가해서 평균 순위, Top 4 비율, 승률을 출력해줘.

5단계: 결과 보고

학습 곡선(평균 보상, 평균 순위 변화)과 최종 평가 결과를 보고해줘. Top 4 비율이 30% 이상이면 성공으로 판단해.

성능이 안 나오면

reward shaping 가중치를 높이거나, entropy coefficient를 0.05로 올리거나, 학습 스텝을 50만으로 늘려서 다시 시도해줘.

최종 목표

8명 플레이어 환경에서 Top 4 비율 50% 이상 달성하는 에이전트 만들기.

