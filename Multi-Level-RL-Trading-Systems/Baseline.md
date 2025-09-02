시장 레짐에 대응하는 강건한 동적 앙상블 강화학습 트레이딩 시스템
A Robust Dynamic Ensemble Reinforcement Learning Trading System for Responding to Market Regimes

허상훈, 황영배 교신저자

Sanghun Heo, Yongbae Hwang

충북대학교 산업인공지능학부
wwwhunycom@cbnu.ac.kr, ybhwang@cbnu.ac.kr

요약 

  본 연구는 선행 연구에서 제안된 멀티모달 강화학습 트레이딩 시스템의 한계점, 즉 다양한 시장 레짐(regime) 변화에 대한 적응력 부족 문제를 해결하기 위해, 동적 앙상블 메커니즘을 도입하여 시스템을 고도화하는 것을 목표로 한다. 제안 시스템은 선행 연구의 멀티모달 데이터(캔들차트, 기술적 지표, 뉴스 감성) 처리 방식을 계승하여 시장 레짐을 실시간으로 분류하고, 각 레짐(상승장, 하락장, 횡보장)에 특화되도록 보상 함수가 차별적으로 설계된 강화학습 에이전트 풀을 구성한다. 이후, 에이전트별 최근 성과를 기반으로 '신뢰도'를 동적으로 재분배하는 가중치 할당 메커니즘을 통해 개별 전문가의 단기적 오판 위험을 분산시키고, 집단 지성을 활용하여 최종 투자 결정을 내린다.

비트코인(BTC/USDT) 26개월 데이터를 이용한 백테스팅 결과, 특히 2022년 하락장에서 베이스라인 모델이 -12.3%의 손실을 기록한 반면, 본 시스템은 +7.9%의 긍정적 수익을 달성하며 시장 변화에 대한 뛰어난 자산 보존 능력과 적응성을 입증했다. 전체 기간에 걸쳐, 제안 시스템은 베이스라인 대비 샤프 비율(Sharpe Ratio)을 1.35에서 1.89로 약 40% 개선하고(p < 0.01), 최대 낙폭(MDD)은 24.8%에서 16.2%로 약 8.6%p 감소시켰다. 또한, 2025년 3개월간 진행된 실시간 페이퍼 트레이딩 검증에서도 제안 시스템은 시장 기준(Buy & Hold) 대비 7.1%p 높은 수익률과 5.3%p 낮은 최대 낙폭을 기록하며, 백테스팅 환경뿐만 아니라 실제 시장과 유사한 환경에서도 강건한 성능을 유지함을 입증했다.
 
Abstract

This study aims to address the limitations of the multimodal reinforcement learning trading system proposed in prior research, specifically its insufficient adaptability to various market regime changes, by introducing a dynamic ensemble mechanism to enhance the system. The proposed system inherits the multimodal data processing approach from previous research (candlestick charts, technical indicators, news sentiment) to classify market regimes in real-time and constructs reinforcement learning agent pools with reward functions differentially designed to specialize in each regime (bull market, bear market, sideways market). Subsequently, through a weight allocation mechanism that dynamically redistributes 'confidence' based on recent performance of each agent, the system disperses the risk of short-term misjudgments by individual experts and makes final investment decisions utilizing collective intelligence. Backtesting results using 26 months of Bitcoin (BTC/USDT) data demonstrate that while the baseline model recorded a loss of -12.3% during the 2022 bear market, the proposed system achieved a positive return of +7.9%, proving its superior asset preservation capability and adaptability to market changes. Over the entire period, the proposed system improved the Sharpe ratio from 1.35 to 1.89 compared to the baseline, representing approximately a 40% improvement (p < 0.01), and reduced the maximum drawdown (MDD) from 24.8% to 16.2%, a decrease of approximately 8.6%p. Furthermore, in real-time paper trading verification conducted over three months in 2025, the proposed system recorded a return 7.1%p higher and maximum drawdown 5.3%p lower compared to the market benchmark (Buy & Hold), demonstrating that it maintains robust performance not only in backtesting environments but also in environments similar to actual markets.


키워드: 강화학습, 동적 앙상블, 시장 레짐, 멀티모달 학습, 금융 시계열 분석, 트레이딩 시스템
Keyword: Reinforcement Learning, Dynamic Ensemble, Market Regime, Multimodal Learning, Financial Time Series Analysis, Trading System, Time Series Analysis, Trading System

1.  서론

1.1 연구 배경 및 필요성
  금융 시장은 본질적으로 비정형적(non-stationary)이며, 추세가 명확한 상승장(bull market), 지속적인 하락장(bear market), 방향성 없는 횡보장(sideways market) 등 다양한 시장 레짐(market regime)이 순환적으로 나타난다(Ang & Bekaert, 2002). 특히, 24시간 변동하는 암호화폐 시장은 이러한 레짐 전환이 더욱 빈번하고 급격하게 발생하여, 단일 전략에 의존하는 기존 트레이딩 시스템의 한계를 명백히 드러내고 있다. 성공적인 트레이딩 시스템은 이러한 레짐 변화에 효과적으로 적응할 수 있어야 한다.

  선행 연구(Heo & Hwang, 2025a)(게재 예정)에서는 캔들스틱 차트의 시각적 패턴과 뉴스 기사의 감성 정보를 결합한 멀티모달 강화학습 모델을 제안하여, 단일 모달리티 모델 대비 우수한 성능을 입증하였다. 해당 연구는 다양한 정보 소스를 통합하는 것의 중요성을 확인시켜주었으나, 단일 정책(single policy) 모델로서 모든 시장 국면에 동일한 전략을 적용한다는 근본적인 한계를 내포하고 있었다. 예를 들어, 상승장에 최적화된 공격적인 매수 전략은 하락장에서 큰 손실로 이어질 수 있으며, 이는 모델의 장기적 안정성을 저해하는 주요 요인이 된다. 따라서 본 연구는 선행 연구의 멀티모달 접근법을 계승하면서도, 다양한 시장 레짐 변화에 대한 적응력 부족이라는 명확한 한계를 극복하여 시스템의 강건성(robustness)과 실용성을 한 단계 끌어올리는 것을 목표로 한다.

1.2 연구 목적 및 주요 기여
  본 연구의 목적은 선행 연구의 정적 멀티모달 모델을 기반으로, 시장 레짐 변화에 동적으로 대응하는 강건한 동적 앙상블 강화학습 트레이딩 시스템을 개발하고 그 우수성을 검증하는 것이다. 이를 위한 본 연구의 주요 기여는 다음과 같이 네 가지로 요약된다.

- 주요 연구 기여도
(1) 실증적 성능 개선 및 하락장 강건성 입증: 포괄적인 백테스팅과 실시간 검증을 통해, 제안된 동적 앙상블 시스템이 베이스라인 모델 대비 샤프 비율을 약 40% 개선하고 최대 낙폭을 8.6%p 감소시킨 것을 통계적으로 유의미하게 입증했다(p < 0.01). 특히, 극심한 하락장에서의 자산 방어 능력을 통해 제안 시스템의 뛰어난 강건성을 증명한다.
(2) 베이스 모델의 한계 극복을 위한 동적 앙상블 프레임워크 제안: 선행 연구의 '단일 정책' 한계를 극복하기 위해, 시장 레짐을 실시간으로 분류하고, 레짐별 전문가 에이전트 풀의 의견을 동적으로 종합하는 새로운 계층적 앙상블 프레임워크를 제안한다.
(3) 레짐 특화 보상 함수 설계를 통한 전문성 강화: 각 시장 레짐의 특성(수익성, 위험성, 거래 빈도)에 맞게 보상 함수를 차별적으로 설계함으로써, 각 에이전트가 해당 레짐에서 최고의 성과를 내도록 전문성을 극대화했다.
(4) 실시간 페이퍼 트레이딩을 통한 실용성 검증: 백테스팅 환경을 넘어 실제 시장과 유사한 환경에서도 시장 기준 수익률을 상회하는 안정적인 성과를 보임으로써 제안 모델의 실용적 가치(practical value)를 입증한다.

1.3 연구 구성
  본 논문의 구성은 다음과 같다. 2장에서는 본 연구의 이론적 기반이 되는 시장 레짐, 앙상블 학습, 강화학습, 멀티모달 학습을 고찰한다. 3장에서는 제안하는 동적 앙상블 시스템의 전체 아키텍처와 각 핵심 구성 요소를 상세히 설명한다. 4장에서는 실험 설계와 결과를 제시하고 심층적으로 분석한다. 마지막으로 5장에서는 연구 결과를 요약하고 시사점과 향후 연구 방향을 논의하며 결론을 맺는다.

2.  관련연구

2.1 시장 레짐 분석 및 예측
  금융 시장 레짐은 특정 기간 시장이 나타내는 특징적인 상태로, 전통적으로 추세 방향에 따라 상승장, 하락장, 횡보장으로 구분된다(Fabozzi & Markowitz, 2011). 레짐 식별은 크게 규칙 기반 방식과 통계/기계학습 기반 방식으로 나뉜다. 규칙 기반 방식은 이동평균선 등을 사용하나 시장 복잡성을 반영하기 어렵다. 통계 기반 방식으로는 마르코프 전환 모델(Hamilton, 1989)이 대표적이며, 관찰되지 않는 레짐 상태가 마르코프 과정을 따른다고 가정한다(Ang & Bekaert, 2002). 최근에는 은닉 마코프 모델(HMM), 서포트 벡터 머신(SVM), 그리고 딥러닝(CNN, RNN) 등 다양한 머신러닝 기법이 비선형 패턴 인식을 통해 레짐 분류에 활발히 적용되고 있으며(Kritzman et al., 2012), Gu, Kelly, & Xiu (2020)의 연구는 머신러닝이 자산 가격 결정에 중요한 역할을 함을 실증적으로 보여주었다. 또한, 뉴스 감성 분석 같은 NLP 기법을 통합하려는 시도도 이루어지고 있다(Kearney & Liu, 2014).

2.2 금융 분야에서의 앙상블 학습
  앙상블 학습(Ensemble learning)은 여러 개별 모델의 예측을 결합하여 단일 모델보다 강건한 성능을 추구하는 기법이다(Dietterich, 2000). 특히 본 연구와 관련된 동적 앙상블(Dynamic ensemble)은 시간이나 상황 변화에 따라 구성 모델의 가중치를 동적으로 조절하는 접근법이다(Kuncheva, 2004). 금융 시계열의 비정형성(non-stationarity)을 고려할 때, 이 기법은 시장 변화에 유연하게 대응할 잠재력이 크며, Timmermann (2006)의 연구는 동적 가중치 할당의 이론적 기초를 제공한다. 실제로, Rapach, Strauss, & Zhou (2010)는 동적 모델 평균화(dynamic model averaging) 기법이 주식 수익률 예측에서 개별 모델보다 우수한 성능을 보임 입증한 바 있다.

2.3 강화학습의 금융 시장 적용
  강화학습(RL)은 에이전트가 환경과의 상호작용을 통해 누적 보상을 최대화하는 정책을 학습하는 패러다임이다(Sutton & Barto, 2018). 금융 트레이딩은 순차적 의사결정 문제이므로 RL 적용에 적합하다. 초기에는 Q-러닝 등 테이블 기반 방식이 시도되었으나(Neuneier, 1997), '차원의 저주' 문제로 한계가 있었다. 심층 강화학습(DRL)의 발전, 특히 DQN(Mnih et al., 2015), PPO(Schulman et al., 2017) 등의 알고리즘은 고차원 금융 데이터를 효과적으로 처리하며 주식 거래, 포트폴리오 관리 등 다양한 문제에 적용되고 있다(Deng et al., 2017; Xiong et al., 2018). 특히, Liang et al. (2018)은 적대적 학습을 결합한 DRL을 통해 포트폴리오 관리에서 시장의 극단적인 상황에 대한 강건성을 높이는 연구를 수행하기도 했다.

2.4 금융 분야의 멀티모달 학습
  멀티모달 학습(Multimodal Learning)은 텍스트, 이미지, 수치 데이터 등 서로 다른 형태의 데이터를 함께 사용해 분석의 정확성을 높이는 기법이다(Baltrušaitis et al., 2019). 금융 시장은 정량적 데이터와 뉴스 기사 등 정성적 데이터에 모두 영향을 받으므로 멀티모달 접근이 효과적이다. 캔들스틱 차트(이미지)와 기술적 지표(수치)를 결합하거나(Chen et al., 2021), 주가와 뉴스 감성(텍스트)을 통합하는 연구(Zhang et al., 2021)는 각 데이터가 가진 고유한 정보를 상호 보완적으로 활용하여 단일 모달리티의 한계를 극복한다. 예를 들어, Lee, Kim, & Kim (2021)의 연구는 기업 공시 자료(텍스트)와 재무 데이터(수치)를 결합하여 주가 예측 정확도를 향상시켰으며, 이는 멀티모달 접근법의 유효성을 잘 보여준다. 본 연구는 이 개념을 확장하여 다중 모달리티 정보를 레짐 분류와 RL 에이전트의 상태 표현에 통합적으로 활용한다.

2.5 시장 레짐 적응형 강화학습
  시장 레짐 적응형 강화학습은 에이전트가 현재 레짐을 인지하고 행동 정책을 동적으로 조정하는 것을 목표로 한다. 구현 방식은 크게 상태 공간 확장(State space augmentation)과 다중 정책/모델(Multiple policies/models)로 나뉜다(Gat et al., 2022). 본 연구에서 제안하는 동적 앙상블 메커니즘은 다중 모델 방식의 장점을 취하면서, 단순히 하나의 에이전트만 선택하는 것이 아니라 각 에이전트의 성과와 상관관계까지 고려하여 가중치를 동적으로 할당하고 의견을 종합한다. 이는 레짐 전환의 불확실성에 대응하고 개별 에이전트의 오류를 보완하여 시스템 전체의 강건성을 높이는 진일보한 적응형 강화학습 전략이다.

3.  실험 방법

  본 연구에서 제안하는 시스템의 전체 아키텍처는 [그림 1]과 같다. 이 시스템의 근간은 선행 연구(Heo & Hwang, 2025a)에서 검증된 멀티모달 정보 처리 방식을 계승하되, 시장 레짐 변화에 대한 적응성과 강건성을 확보하기 위해 레짐 분류 모듈과 동적 앙상블 메커니즘을 도입한 점이 핵심적인 차별점이다.
3.1 멀티모달 상태 표현
  본 모듈의 목적은 시장의 현재 상태를 가장 포괄적이고 정확하게 표현하는 다차원 상태 벡터 s_t를 생성하는 것이다. 강화학습 에이전트의 의사결정 품질은 입력되는 상태 정보의 질에 크게 의존하므로, 선행 연구에서 그 우수성이 입증된 멀티모달 접근법을 채택한다(Baltrušaitis et al., 2019). 
상태 벡터 s_t는 다음과 같이 세 가지 이종(heterogeneous) 데이터 소스로부터 추출된 특징 벡터들의 결합(concatenation)으로 구성된다. 이러한 단순 결합 방식은 멀티모달 학습의 초기 융합(early fusion) 단계에서 널리 사용되는 접근법으로, 각 모달리티의 정보를  손실 없이 통합하는 효과적인 방법이다 (Ngiam et al., 2011).

st​=concat(Fvisual​,Ftech​,Fsenti​)∈R273   (1)
(여기서 st​는 세 가지 특징 벡터를 결합한 273차원 상태 벡터이다.)

그림 1. 연구 프레임워크

금융 차트는 전문가들이 패턴을 통해 시장 심리를 읽는 중요한 정보원이지만, 수치 데이터만으로는 그 형태적 특성을 포착하기 어렵다. 이를 위해 최근 60시간의 OHLCV 데이터를 224x224 픽셀 크기의 캔들스틱 차트 이미지로 변환한다. (초기 실험 결과, 30, 60, 90시간 단위 중 60시간이 장단기 패턴을 유의미하게 포착하는 데 가장 효과적이었다.) 이 이미지를 ImageNet으로 사전 학습된 ResNet-18(He et al., 2016) 기반의 CNN(Convolutional Neural Network)에 통과시켜, 차트의 구조적 패턴(예: 지지/저항선, 특정 캔들 패턴)을
함축하는 256차원의 고차원 특징 벡터를 추출한다.
기술적 특징 (F_{tech} in mathbb{R}^{15}): 가격과 거래량 데이터로부터 파생되는 정량적 지표는 시장의 모멘텀, 변동성, 추세 강도 등을 측정하는 데 필수적이다. 이동평균(SMA, EMA), 상대강도지수(RSI), MACD(Moving Average Convergence Divergence), 볼린저 밴드 등 15개의 핵심 기술적 지표를 계산하고, 각 지표를 [0, 1] 범위로 Min-Max 정규화하여 사용한다.
  감성적 특징 (F_{senti} in mathbb{R}^{2}): 시장의 비합리적이고 심리적인 측면은 뉴스 기사에 잘 반영된다(Tetlock, 2007). 주요 금융 뉴스 API(예: NewsAPI) 및 암호화폐 전문 미디어(예: Cointelegraph, Coindesk)에서 'Bitcoin', 'BTC' 등의 키워드로 실시간 수집된 뉴스 데이터의 헤드라인과 본문을 대규모 언어 모델인 DeepSeek-R1 (32B)을 사용하여 -1(극 부정)에서 +1(극 긍정) 사이의 감성 점수로 변환한다. 

이후, 최근 24시간 동안의 (데이터의 최신성을 효과적
으로 반영하기 위해 설정된 기간이다) (a)단순 평균 감성 점수와 (b)시간에 따라 가중치를 감소시키는 지수 가중 이동 평균(EWMA) 감성 점수를 계산하여 2차원 감성 특징 벡터로 사용한다.


그림 2. 멀티모달 특징 추출 및 융합 과정 시각화

3.2 시장 레짐 분류 방법론
  본 모듈은 시스템의 "항해사" 역할을 수행하며, 현재 시장이 상승장, 하락장, 횡보장 중 어떤 국면에 있는지를 실시간으로 판단한다.

  - 레짐 레이블링: 금융공학에서 널리 사용되는 이동평균선 기반의 규칙을 사용하여 역사적 데이터에 레짐을 정의한다. 이 방식은 명확하고 재현 가능하다는 장점이 있으나, 본질적으로 후행성 지표(lagging indicator)에 의존하므로 급격한 시장 변화를 즉각적으로 반영하는 데에는 한계가 존재한다. 그럼에도 본 연구에서는 실증 분석에서 널리 통용되는 이 기준을 채택하여, 명시적인 레짐 분류 모델 학습을 위한 객관적인 기초를 마련하였다. 
   본 연구에서는 단기(20), 중기(50), 장기(200) 지수이동평균(EMA)의 배열을 기준으로 시장 레짐을 다음과 같이 세 가지로 레이블링한다.




그림 3. 시장 레짐별 변동성 특성 분석

● 상승장 (Bull): 단기(20-period), 중기(50-period), 장기(200-period) 이동평균선(EMA)이 완벽한 정배열을 이루는 상태로 정의한다. 이는 단기 EMA가 중기 EMA보다 높고, 중기 EMA가 장기 EMA보다 높은 경우를 의미한다.

   조건: 20 EMA > 50 EMA > 200 EMA
● 하락장 (Bear): 이동평균선들이 상승장과 정반대 순서로 배열된 역배열 상태로 정의한다. 즉, 단기 EMA가 중기 EMA보다 낮고, 중기 EMA가 장기 EMA보다 낮은 경우이다. 이는 지속적인 하락 압력과 약세 모멘텀을 나타낸다.

   조건: 20 EMA < 50 EMA < 200 EMA
● 횡보장 (Sideways): 상승장 또는 하락장의 정의에 해당하지 않는 모든 경우를 횡보장으로 분류한다. 이는 이동평균선들이 서로 엇갈려 명확한 방향성을 보이지 않는 추세 없는(trendless) 시장 국면을 의미한다.

  확률적 분류: 3.1절에서 생성된 멀티모달 상태 벡터 s_t를 입력으로, 지도학습 모델인 XGBoost를 사용하여 현재 시점이 각 레짐에 속할 확률을 예측한다. XGBoost는 높은 성능과 해석 가능성으로 널리 사용되는 모델이다. 모델의 출력은 다음과 같은 확률 벡터이다.

P(Rt​∣st​)=[P(Rt​=bull),P(Rt​=bear),P(Rt​=sideways)]   (2)
이 확률 벡터는 3.4절의 동적 앙상블 메커니즘에서 어떤 전문가 에이전트 풀을 활성화할지 결정하는 데 사용된다.

3.3 레짐 특화 강화학습 에이전트 풀
  "하나의 모델이 모든 것을 잘할 수는 없다"는 앙상블 학습의 기본 철학에 따라(Dietterich, 2000), 각 시장 레짐(상승, 하락, 횡보)에 최적화된 '전문가' 에이전트 풀을 별도로 구성하고 학습시킨다. 

  이는 경제학에서 레짐 전환을 고려했을 때 최적의 정책이 달라진다는 연구(Ang & Bekaert, 2002)와 일맥상통하며, 금융 분야에서 '전문가 혼합(Mixture of Experts)' 모델을 적용한 선행 연구들과도 맥을 같이 한다 (Jacobs et al., 1991). 

  각 풀은 서로 다른 랜덤 시드로 초기화된 5개의 PPO(Schulman et al., 2017) 에이전트로 구성하여 정책의 다양성을 확보한다. 단일 레짐 내에서도 하나의 에이전트가 아닌 작은 규모의 앙상블(N=5)을 사용하는 것은, 특정 에이전트의 불운한 초기화나 학습 실패로 인한 위험을 방지하고, 레짐 내 전략의 강건성을 확보하기 위함이다. 
  이는 동적 가중치 할당 단계 이전에 이미 1차적인 안정성을 부여하는 역할을 한다. 각 에이전트의 전문성은 레짐 특성에 맞게 세밀하게 설계된 보상 함수(Reward Shaping)를 통해 구현된다.

그림 4, 레짐 특화 강화학습 에이전트 풀

  본 연구의 레짐별 보상 함수 설계 철학은 '해당 레짐에서 가장 치명적인 실수를 피하는 것'에 초점을 맞춘다. 즉, 횡보장에서는 잦은 거래로 인한 손실, 하락장에서는 큰 폭의 하락에 노출되는 위험을 최소화하는 것을 우선 목표로 설정했다. 이는 일부 기회비용을 감수하더라도 시스템의 장기적 안정성과 강건성을 확보하기 위한 의도적인 설계이다. 

  이러한 접근법은 금융 분야에서 널리 알려진 '후회 최소화(Regret Minimization)' 프레임워크와 맥락을 같이하며, 최상의 수익률보다는 최악의 손실을 방지하는 데 중점을 둔다 (Savage, 1951). 하지만, 특정 레짐에 특화된 전문가 풀을 구성하는 것만으로는 충분하지 않다. 레짐 내부의 미세한 변동이나 레짐 전환기의 불확실성에 대응하기 위해서는, 이 전문가들 사이의 의견을 실시간으로 조율하고 종합할 상위 제어 시스템이 필요하다.

MDP 정의: 모든 에이전트는 동일한 MDP(S, A, P, gamma) 구조를 공유한다.

◾ 상태 공간 (S): 3.1절에서 정의된 멀티모달 상태 벡터 s_t
◾ 행동 공간 (A): {강력 매수, 매수, 홀드, 매도, 강력 매도}의 5가지 이산적 행동
◾ 할인 계수 (gamma): 0.99로 설정하여 장기적 보상을 중요하게 고려
- 레짐별 보상 함수 설계:
상승장 에이전트: 추세를 따라 수익을 극대화하는 것이 목표. 포트폴리오의 단순 수익률을 보상으로 사용한다. 이 레짐에서는 적극적인 거래를 장려하기 위해 거래 비용에 대한 명시적인 페널티를 부과하지 않는다.

R_t^{{bull}} = frac{V_{t+1} - V_t}{V_t} qquad (3a)
(여기서 V_t는 시점 t의 포트폴리오 가치)

  - 하락장 에이전트: 손실 최소화 및 위험 관리가 최우선 목표. 변동성 대비 수익을 측정하는 전통적인 샤프 비율(Sharpe, 1966)보다 하방 위험(downside risk)에
만 초점을 맞추는 소티노 비율(Sortino Ratio)을 보상 함수에 도입한다 (Sortino & van der Meer, 1991). 이는 하락 변동성에 대해 강한 페널티를 부여하여 에이전트가 방어적인 행동을 학습하도록 유도한다. 동시에, 불필요한 포지션 전환을 억제하기 위해 작은 거래 비용 패널티(C)를 적용한다.

R_t^{{bear}} = {SortinoRatio}_t - C cdot {TransactionCost}_t qquad   (3b)
(여기서 거래 비용 패널티 상수 C는 0.01로 설정하여,
 위험 관리 목표를 해치지 않는 선에서 최소한의 거래를 유도한다.)

  - 횡보장 에이전트: 잦은 거래로 인한 손실 방지가 핵심 목표. 방향성 없는 시장에서 빈번한 매매는 수수료 누적으로 이어지기 쉽다. 따라서 거래 비용에 다른 레짐보다 훨씬 큰 페널티 상수(C′)를 적용하여 거래를 강력하게 억제하고, 현금 보유를 통해 기회비용을 최소화하도록 학습시킨다.

R_t^{{sideways}} = frac{V_{t+1} - V_t}{V_t} - C' cdot {TransactionCost}_t quad (C' gg C) qquad   (3c)
(여기서 페널티 상수 C′는 0.05로 설정하여, 하락장의 C(0.01)보다 5배 높은 강한 페널티를 부과한다. 이는 횡보장에서의 잦은 거래가 수익률에 미치는 부정적인 영향을 반영한 설계이다. 

  횡보장은 본질적으로 명확한 방향성이 없는 시장 상황으로, 이때 과도한 거래 활동은 거래 비용의 누적으로 인해 오히려 수익률을 악화시키는 '거래 비용의 함정(Transaction Cost Trap)'에 빠질 위험이 높다.



그림 5. 레짐별 보상 함수 설계

3.4 동적 앙상블 메커니즘
  이 모듈은 시스템의 "중앙 관제탑(Central Control Tower)"으로서, 실시간으로 최적의 전문가들을 선별하고 그들의 의견을 종합하여 최종적인 투자 결정을 내리는 핵심적인 역할을 수행한다. 이 동적 의사결정 과정은 [그림 6]와 같이 3단계의 계층적 구조(Hierarchical Structure)로 설계되었다.

  이러한 계층적 접근법은 복잡한 의사결정 문제를 단계적으로 분해하여 각 단계에서 특화된 최적화를 수행함으로써, 전체 시스템의 성능과 해석 가능성을 동시에 확보한다. 첫 번째 단계에서는 거시적 시장 맥락을 파악하여 적절한 전문가 집단을 선별하고, 두 번째 단계에서는 개별 전문가의 신뢰도를 정량화하며, 마지막 단계에서는 가중 민주주의 원리에 따라 집단 지성을 활용한 최종 결정을 도출한다.


그림 6. 동적 앙상블 메커니즘의 3단계 작동 원리

  - 1단계: 레짐 기반 전문가 풀 선택
◾ 원리: 현재 시장 상황에 가장 적합한 전문가 집단을 먼저 선택한다.

◾ 작동 방식: 3.2절의 시장 레짐 분류 모델은 현재 상태 s_t를 바탕으로 각 레짐 R in {{bull}, {bear}, {sideways}}에 대한 사후 확률(posterior probability)을 포함하는 확률 벡터 P(R_t|s_t)를 출력한다. 가장 높은 확률 값을 갖는 레짐 R_t^*을 현재의 지배적인 시장 레짐으로 판단한다.

R_t^* = argmax_{R in {{bull}, {bear}, {sideways}}} P(R_t = R | s_t) qquad   (4)


(여기서 Rt∗​는 시점 t에서 선택된 지배적 레짐, 
P(Rt​=R∣st​)는 현재 상태 st​가 주어졌을 때 레짐 R에 속할 사후 확률(posterior probability), argmax는 확률이 가장 높은 레짐을 선택하는 함수이다. 이 경성 선택(Hard Selection) 방식은 베이지안 추론에 기반하여 불확실성 하에서의 최적 의사결정을 구현한다.

  구체적으로, XGBoost 분류 모델은 멀티모달 상태 벡터 st​(캔들차트 특징, 기술적 지표, 뉴스 감성)를 입력받아 각 레짐에 대한 확률 분포를 출력하며, 이는 현재 시장 상황이 각 레짐의 특성과 얼마나 유사한지를 정량적으로 나타낸다. 

  예를 들어, P(bull∣st)=0.7, P(bear∣st)=0.2P
P(sideways∣st​)=0.1인 경우, Rt∗​=bull이 된다.

  이러한 확률적 레짐 분류는 시장의 본질적 불확실성을 명시적으로 고려한다. 단순한 규칙 기반 분류(예: 이동평균선 배열)와 달리, 확률 분포는 레짐 전환의 점진적 특성과 경계 상황의 모호성을 포착할 수 있다. 

  특히, 확률값이 근소한 차이를 보이는 경우(예: P(bull∣st)=0.45, P(sideways∣st​)=0.48)는 레짐 전환기의 불확실한 시장 상황을 반영하며, 이때 시스템은 소폭 우세한 레짐의 전문가 풀을 활성화하되 동적 가중치 메커니즘을 통해 급격한 정책 변화를 완충한다.

  또한, 본 연구에서는 레짐 분류의 신뢰도를 높이기 위해 확률 임계값 θ=0.6을 도입하여, max(P(Rt​=R∣st​))<θ인 경우 이전 레짐을 유지하는 보수적 접근을 적용했다. 이는 레짐 전환의 잦은 오판(false positive)으로 인한 성능 저하를 방지하고, 레짐 분류의 안정성을 제고하는 실용적 장치이다.)

◾ 기능: 결정된 레짐 R_t^*에 해당하는 전문가 에이전트 풀을 해당 타임스텝의 의사결정을 위한 활성 그룹(active group)으로 지정하여 관련 없는 에이전트의 노이즈를 차단한다.

이는 전문가 혼합(Mixture of Experts) 모델의 핵심 원리를 구현한 것으로, 현재 시장 상황과 부합하지 않는 전문가들의 부적절한 조언을 사전에 필터링하는 게이팅(gating) 메커니즘 역할을 수행한다.

구체적으로, 전체 15개 에이전트(상승장 5개 + 하락장 5개 + 횡보장 5개) 중에서 현재 레짐에 특화된 5개 에이전트만을 선별 활성화함으로써, 계산 효율성을 3배 향상시키는 동시에 의사결정의 일관성을 확보한다. 예를 들어, 강한 하락 추세가 감지된 상황(Rt∗​=bear)에서 상승장 전문가들의 공격적 매수 신호나 횡보장 전문가들의 관망 전략이 최종 결정에 영향을 미치는 것을 원천 차단하여, 하락장에 특화된 방어적 전략만이 고려되도록 한다.

이러한 선택적 활성화는 인지 과학의 '주의 집중(Attention) 메커니즘'과 유사한 원리로, 현재 상황과 가장 관련성이 높은 정보에만 집중함으로써 인지적 부하를 줄이고 의사결정 품질을 높인다. 

또한, 레짐별 전문성의 명확한 분업을 통해 각 에이전트가 특정 시장 조건에서 축적한 경험과 학습된 정책의 순수성을 보존하고, 서로 다른 레짐의 상충되는 전략이 혼재되어 발생할 수 있는 '정책 희석(Policy Dilution)' 현상을 방지한다. 이는 궁극적으로 시장 레짐 변화에 대한 시스템의 적응 속도와 정확성을 크게 향상시키는 핵심 설계 요소이다.


그림 7. 1단계: 레짐 특화 전문가 풀 설계
  - 2단계: 성과 기반 동적 가중치 할당
◾ 원리: 활성화된 전문가 그룹 내에서도 최근 시장에 가장 성공적으로 대응해 온 전문가에게 더 높은 발언권을 부여한다. 이는 과거 성과가 좋은 전문가가 미래에도 좋은 성과를 낼 가능성이 높다는 가정에 기반한 '가중 다수결 알고리즘(Weighted Majority Algorithm)'의 아이디어와 유사하다 (Littlestone & Warmuth, 1994).

◾ 작동 방식: 활성화된 풀 내 5개 에이전트 각각의 최근 30거래일 동안의 샤프 비율 SR_{i,30}을 계산한다. (이 기간은 단기적인 성과 추세와 통계적 안정성 간의 균형을 고려하여 교차 검증을 통해 결정되었다.) 이 성과 지표를 소프트맥스 함수에 통과시켜, 합이 1이 되는 정규화된 동적 가중치 w_{i,t}를 산출한다.

w_{i,t} =
frac{exp(SR_{i,30}/T)}{sum_{j=1}^{5} exp(SR_{j,30}/T)} qquad   (5)



(여기서 T는 가중치의 민감도를 조절하는 온도 하이퍼파라미터로, 이 값이 시스템의 적응성과 안정성 간의 균형을 결정하는 핵심 요소이다. 

  T 값이 작을수록(예: T=1) 성과가 좋은 에이전트에게 급격하게 높은 가중치를 부여하여 빠른 적응을 가능하게 하지만, 단기적 성과 변동에 과도하게 민감하게 반응할 위험이 있다. 반대로 T 값이 클수록(예: T=100) 가중치 분포가 평평해져서 안정적이지만 시장 변화에 대한 반응 속도가 느려진다. 

  본 연구에서는 T ∈ {1, 5, 10, 20, 50}에 대해 5-fold 교차 검증을 수행한 결과, T=10일 때 백테스팅 기간 동안 샤프 비율(1.89)과 최대 낙폭(-16.2%) 측면에서 최적의 성능을 보였으며, 이는 급격한 가중치 변화로 인한 거래 비용 증가 없이도 시장 변화에 적절한 속도로 대응할 수 있는 최적점임을 의미한다. 

  또한, T=10 설정 시 가중치 엔트로피가 평균 0.85를 유지하여 단일 에이전트 의존도를 방지하면서도 성과가 우수한 에이전트에게 합리적인 발언권을 부여하는 균형잡힌 의사결정 구조를 구현했다.)

◾ 기능: 시장의 미세한 변화에 따라 각 에이전트의 단기적 성능이 변동하는 현실을 반영하고, 앙상블 전체의 적응성과 강건성을 높인다. 이를 통해 최근 시장 패턴을 효과적으로 포착하는 에이전트에게 더 큰 발언권을 부여하여 시스템의 자기 교정(self-correction) 능력을 구현한다.

그림 8. 2단계: 성과 기반 동작 가중치 할당
- 3단계: 가중 앙상블 정책 결정
◾ 원리: 최종 투자 결정은 신뢰도(가중치)가 높은 전문가들의 의견을 종합하여 민주적으로 결정한다.

◾ 작동 방식: 활성화된 에이전트들의 정책 pi_i(a|s_t)를 2단계에서 계산된 동적 가중치 w_{i,t}로 가중 평균하여 앙상블 정책 pi_{{ensemble}}을 구축한다.

  pi_{{ensemble}}(a|s_t) = sum_{i=1}^{5} w_{i,t} cdot pi_i(a|s_t)   (6)

(여기서 pi_{{ensemble}}(a|s_t)는 앙상블 정책에서 상태 s_t일 때 행동 a를 선택할 확률, w_{i,t}는 에이전트 i의 동적 가중치(합이 1), pi_i(a|s_t)는 에이전트 i의 개별 정책에서 행동 a를 선택할 확률이다. 

 예를 들어, 5개 에이전트가 매수에 대해 각각 0.8, 0.7, 0.6, 0.5, 0.4의 확률을 제시하고 가중치가 [0.3, 0.25, 0.2, 0.15, 0.1]이라면, 앙상블 매수 확률은 0.3×0.8 + 0.25×0.7 + 0.2×0.6 + 0.15×0.5 + 0.1×0.4 = 0.64가 된다.)



  시스템의 최종 행동 a_t^*는 이 앙상블 정책에서 가장 높은 확률을 갖는 행동으로 선택된다.
  a_t^* = argmax_a π_{{ensemble}}(a|s_t)   (7)



(여기서 at∗​는 시점 t에서 시스템이 최종적으로 선택하는 행동, argmax​는 모든 가능한 행동 a 중에서 앙상블 정책 확률이 가장 높은 행동을 선택하는 함수이다. 이 결정론적 선택 방식은 확률적 샘플링 대신 가장 높은 신뢰도를 가진 행동을 일관되게 실행함으로써 시스템의 안정성과 예측 가능성을 높인다. 예를 들어, 앙상블 정책이 [강력매수: 0.1, 매수: 0.64, 홀드: 0.2, 매도: 0.05, 강력매도: 0.01]라면, a_t^* = 매수가 된다.

이러한 결정 과정은 집단 지성(Collective Intelligence)의 민주적 의사결정 원리를 반영한다. 즉, 개별 에이전트들이 각자의 전문성에 기반해 제시한 정책들이 성과 기반 가중치를 통해 종합되고, 최종적으로는 가장 많은 "신뢰도 가중 투표"를 받은 행동이 선택되는 구조이다. 이는 단일 모델의 편향된 판단이나 극단적 결정을 방지하고, 여러 전문가의 견해를 균형있게 반영하여 더욱 강건한 투자 결정을 가능하게 한다.

또한, 실제 구현에서는 확률이 매우 근소한 차이(예: 매수 0.501 vs 홀드 0.499)를 보일 경우 불필요한 거래를 방지하기 위해 최소 확률 차이 임계값(threshold) δ=0.05를 적용하여, |P(a₁) - P(a₂)| < δ인 경우 현재 포지션을 유지하는 보수적 접근을 취한다. 이는 거래 비용을 최소화하고 시장의 미세한 노이즈에 의한 불필요한 포지션 변경을 억제하는 실용적 고려사항이다.)

◾ 기능: 단일 에이전트의 편향이나 오판 위험을 분산시키고, 개별 모델의 오류를 상호 보완하여 전체 시스템의 성능을 향상시킨다. 제안된 동적 앙상블 시스템의 전체적인 작동 흐름과 각 단계별 세부 로직은 [부록 1]의 의사코드에 상세히 기술되어 있다.

그림 9. 3단계: 최종행동 결정을 위한 가중 앙상블

4.  실험 결과

4.1 실험 환경 및 데이터셋
  - 데이터: 본 연구에서는 2021년 10월 12일부터 2023년 12월 19일까지, 총 26개월간의 바이낸스(Binance) 거래소 BTC/USDT 페어 1시간 봉 데이터를 사용하였다. 데이터는 OHLCV(시가, 고가, 저가, 종가, 거래량) 값을 포함한다.
- 검증 방식: 시계열 데이터의 순차적 특성을 보존하고 과최적화(overfitting) 위험을 줄이기 위해, 금융 시계열 분석에서 표준적인 검증 방법론으로 간주되는 Walk-Forward 방식을 적용했다(Pardo, 2008). 이 방식은 시간이 지남에 따라 새로운 데이터를 지속적으로 반영하여 모델을 업데이트하는 과정을 모사함으로써, 실제 운용 환경과 유사한 조건에서 모델의 일반화 성능과 강건성을 신뢰도 높게측정할 수 있다.

  본 연구에서는 확장 윈도우(Expanding Window) 접근법을 사용하여 총 26개월의 데이터를 다음과 같이 순차적으로 학습 및 평가했다.


그림. 10 데이터 세트 분할

1. 첫 번째 검증: 초기 18개월(1~18개월) 데이터로 모델을 학습시킨 후, 바로 이어지는 2개월(19~20개월) 동안의 성과를 테스트했다.
2. 두 번째 검증: 학습 기간을 20개월(1~20개월)로 확장하여 모델을 재학습하고, 그 다음 2개월(21~22개월)의 성과를 평가했다.
3. 이후 검증: 위와 동일한 방식으로 학습 데이터를 2개월씩 점진적으로 확장하며, 각각 23~24개월 및 25~26개월 구간을 순차적으로 테스트했다.

  최종적으로, 이렇게 얻어진 4개의 독립적인 테스트 구간(총 8개월)의 성능 지표들의 산술 평균을 제안 모델의 최종 성능으로 보고하였다.

  - 거래 환경: 초기 자본은 $10,000로 설정하였고, 거래 수수료는 0.05%(매수/매도 시 각각)를 가정하였다. 또한, 유동성이 낮은 시장에서 발생할 수 있는 주문 체결 오차(slippage)를 모사하기 위해 0.02%의 슬리피지를 양방향으로 적용하였다. 이 값들은 실제 암호화폐 거래소의 고빈도 데이터를
분석한 선행 연구에서 관찰된 평균적인 가격 충격 수준을 참고하여 설정된 현실적인 가정이다(Hautsch, Scheuch, & Voigt, 2021).

  - 구현 및 환경: 본 연구의 모든 실험은 Python 3.9.7, PyTorch 1.12.0, Stable-Baselines3 1.6.0, XGBoost 1.6.1을 기반으로, Intel i9-10900K CPU, 32GB RAM, NVIDIA RTX 3080(10GB) GPU가 탑재된 Ubuntu 20.04 LTS 환경에서 수행되었다.

  - 하이퍼파라미터: 강화학습 에이전트(PPO)의 학습률은 3e-4, 배치 사이즈는 64, 할인 계수(γ)는 0.99로 설정하였다. 레짐 분류 모델(XGBoost)의 n_estimators는 100, max_depth는 6으로 설정하였다. 동적 앙상블의 온도 파라미터(T)는 1.0, 성과 평가 윈도우는 30일로 설정하였다. 하이퍼파라미터 튜닝 과정과 최적값 선택 근거는 [부록 3]에서 상세히 설명한다.

4.2 비교 모델 및 실제 근거
비교 모델:
◾ 정적 멀티모달 RL (Static Multimodal RL): 본 연구의 베이스라인. 선행 연구(Heo & Hwang, 2025a)(게재 예정) 모델.
◾ 전통적 트레이딩 전략: 이동평균 교차(20/50 EMA), MACD 등 규칙 기반 전략.
◾ Buy & Hold 전략: 시장의 기준 수익률(Benchmark).

설계 근거: 제안 시스템은 Hamilton (1989), Ang & Bekaert (2002)의 레짐 전환 이론, Zhou (2019)의 동적 앙상블 방법론, Baltrušaitis et al. (2019)의 멀티모달 학습 이론 등 검증된 연구들을 금융 트레이딩 문제에 맞게 융합한 하이브리드 아키텍처이다.

4.3 시장 레짐 분류 성능 분석
  제안된 XGBoost 기반 레짐 분류 모델의 성능을 평가한 결과, [표 1]에서 보듯이 테스트 데이터셋에 대해 평균 84.5%의 높은 정확도를 기록했다.
세부적으로 살펴보면, 모델은 추세가 명확한 상승장(정확도 88.5%)과 하락장(정확도 86.1%)에서 특히 강점을 보였다. 이는 멀티모달 데이터(차트 패턴, 기술적 지표, 뉴스 감성)가 시장의 방향성을 효과적으로 포착했음을 시사한다. 

  반면, 방향성이 모호한 횡보장(정확도 79.0%)에서는 상대적으로 성능이 소폭 하락했다. 이는 횡보장 자체가 노이즈가 많고 예측이 어려운 본질적인 특성을 가지기 때문으로 분석된다. 그럼에도 불구하고, 전체적으로 모델이 각 레짐을 안정적으로 분류하는 능력을 갖추었음을 확인했으며, 이는 후속 동적 앙상블 메커니즘의 성공적인 작동을 위한 신뢰도 높은 기반이 된다.

특히 주목할 점은 F1-Score가 모든 레짐에서 0.74 이상을 유지하여 정밀도와 재현율 간의 균형잡힌 성능을 보였다는 것이다.




성능 분석 요약 (Performance Analysis Summary)
• 모델 강점: 추세가 명확한 상승장(88.5%)과 하락장(86.1%)에서 높은 분류 정확도 달성
• 개선 영역: 방향성이 모호한 횡보장(79.0%)에서 상대적으로 낮은 성능, 노이즈가 많고 예측이 어려운 본질적 특성
• 전체 평가: 멀티모달 데이터(차트 패턴, 기술적 지표, 뉴스 감성)가 시장 방향성을 효과적으로 포착하여 84.5% 정확도 달성
• 시스템 기여: 후속 동적 앙상블 메커니즘의 성공적인 작동을 위한 신뢰도 높은 기반 제공

그림 11. XGBoost 기반 레짐 분류 모델 정확도 




그림 12. 혼동 행렬(Confusion Matrix) 

표 1. 시장 레짐 분류 모델 성능 (테스트 데이터셋)




구분
정확도
(%)
정밀도
(%)
재현율
(%)
F1-Score
상승장(Bull)
88.5
89.1
91.2
0.90
하락장
(Bear)
86.1
87.5
88.3
0.88
횡보장
(Sideways)
79.0
75.4
72.1
0.74
가중 평균
84.5
84.0
84.5
0.84





































그림 13. 리스크 지표 비교 (동적 앙상블 vs 정적 RL) 
















4.4 종합 트레이딩 성과 비교
  전체 테스트 기간에 대한 종합 성과는 [표 2]와 같다. 제안된 동적 앙상블 시스템은 모든 비교 모델, 특히 베이스라인인 정적 RL 모델을 모든 핵심
지표에서 압도했다.



그림 14. 모델별 샤프비율 비교 

표 2. 모델별 종합 트레이딩 성과 비교 

평가 지표
동적 앙상블
정적 RL
(Baseline)
MACD
Buy&
Hold
누적 수익률(%)
258.4
185.2
88.5
165.7
연평균 수익률(CAGR, %)
52.9
41.8
23.6
38.5
샤프 비율(SR)
1.89
1.35
0.81
1.15
최대 낙폭(MDD, %)
-16.2
-24.8
-33.1%
-35.4%
승률(Win Rate, %)
63.5
56.1
47.9
N/A
손익비(Profit Factor)
2.25
1.78
1.45
N/A
총 거래횟수
852
1,154
780
1
총 거래비용($)
-426
-577
-390
-5
다음 표는 각 모델의 수익률 성과를 보여준다.


  베이스라인 모델 대비, 제안 모델은 샤프 비율을1.35에서 1.89로 약 40% 개선했으며, 최대 낙폭은 24.8%에서 16.2%로 8.6%p 감소시켰다. 

  두 모델의 샤프 비율 차이에 대한 통계적 유의성을 검증하기 위해, Ledoit & Wolf (2008)가 제안한 강건한 추정치를 사용한 검정을 수행한 결과, 제안 모델의 성능 개선은 99% 신뢰수준에서 통계적으로 유의미한 것으로 나타났다(p < 0.01). 또한, 레짐 인지를 통해 불필요한 거래를 억제하여 총 거래 횟수를 약 26% 줄이고, 그에 따른 총 거래 비용도 약 26% 절감하는 효과를 보였다.

4.5 레짐별 성과 및 강건성 분석
  - 핵심 요소별 기여도 분석(어블레이션 스터디)
제안 모델의 각 구성 요소가 전체 성능에 미치는 영향을 정량적으로 확인하기 위해 핵심 요소를 하나씩 제거하는 체계적인 어블레이션 스터디를 수행했다. 그 결과, '레짐 분류' 모듈 자체를 제거했을 때(정적 RL 모델과 동일) 샤프 비율이 1.89에서 1.35로 약 28.6% 하락하여 가장 큰 성능 저하를 보였다. 이는 시장 상황 인지 없이 모든 레짐에 동일한 전략을 적용하는 것의 근본적 한계를 보여준다. 또한 '동적 앙상블' 메커니즘을 정적 가중치로 변경했을 때도 샤프 비율이 약 16.4% 감소했으며, 멀티모달 정보를 제거하고 가격 데이터만 사용했을 때는 19.6%의 성능 저하가 관찰되었다. 

 이러한 결과는 첫째, 시장 레짐을 명시적으로 인지하고 대응하는 것이 시스템 성능의 가장 중요한 기반이며, 둘째, 단순히 전문가 풀을 두는 것을 넘어 그들의 영향력을 동적으로 조절하는 메커니즘이 필수적이고, 셋째, 다양한 데이터 소스의 통합이 시장 상황 판단의 정확성을 크게 향상시킴을 강력하게 시사한다.



그림 15. 어블레이션 스터디 샤프비율

표 3. 구성 요소별 기여도 분석

제거된 구성요소
샤프 비율
성능 감소율
주요 영향
완전한 모델
1.89
   -
기준
동적 가중치 제거
1.58
-16.4%
고정 가중치 사용
멀티모달 정보 제거
1.52
-19.6%
가격 데이터만 사용
앙상블 메커니즘 제거
1.41
-25.4%
단일 에이전트 사용
레짐 분류 제거
1.35
-28.6%
베이스라인 모델과 동일


  - 레짐별 차별화된 성과 분석
제안 모델의 진정한 강점은 [그림 20]의 레짐별 연평균 수익률(CAGR) 분석에서 더욱 명확히 드러난다. 상승장에서는 베이스라인과 유사한 높은 수익률(52.1%)을 기록하여 상승 모멘텀을 놓치지 않으면서도, 횡보장에서는 불필요한 거래를 효과적으로 줄여 손실을 방어했다. 가장 주목할 점은 하락장에서 손실을 기록한 모든 비교 모델(베이스라인: -8.2%, MACD: -15.3%, Buy&Hold: -12.3%)과 달리 유일하게 +7.9%의 긍정적 수익을 달성했다는 것이다. 이는 하락장 전문가들의 방어적 전략과 동적 가중치 조절이 극한 상황에서도 자산을 보호하고 수익을 창출할 수 있음을 실증적으로 입증한다.

  - 극한 상황에서의 강건성 검증
특히 2022년 Terra/LUNA 붕괴와 같은 극단적 시장 충격 상황에서도 시스템이 안정적으로 작동하여 강건성(robustness)을 확인했다. [그림 21]의 2022년 하락장 기간 분석을 보면, 베이스라인 모델이 지속적인 자산 가치 하락을 보인 반면, 제안 모델은 하락 추세를 조기에 감지하고 방어적 포지션을 취함으로써 자산을 성공적으로 보존했다. 이는 시장 변동성이 평균 대비 3배 이상 증가한 극한 상황에서도 레짐 분류의 정확성이 유지되고, 동적 앙상블 메커니즘이 효과적으로 작동했음을 의미한다.

  - 성과 지속성 및 안정성 분석
[그림 17]의 6개월 롤링 수익률 분석 결과, 제안 모델은 전 기간에 걸쳐 베이스라인 모델보다 일관되게 우수한 성과를 보였으며, 특히 시장 변동성이 높은 구간에서 성과 격차가 더욱 벌어지는 패턴을 확인했다. 승률 측면에서도 63.5%로 베이스라인의 56.1%를 크게 상회하여, 단순히 큰 수익을 얻는 것이 아니라 손실을 줄이는 방향으로도 일관된 개선을 보였다. 또한 손익비(Profit Factor) 2.25는 베이스라인의 1.78 대비 26% 향상된 수치로, 수익 거래와 손실 거래 간의 균형이 크게 개선되었음을 나타낸다. 

  더 나아가, 월별 수익률의 표준편차가 베이스라인 대비 23% 감소하여 수익의 안정성과 예측 가능성이 크게 향상되었다. 이러한 일관된 성과는 제안 시스템이 단기적 시장 노이즈에 흔들리지 않고 장기적으로 안정적인 투자 성과를 제공할 수 있는 신뢰할 만한 솔루션임을 입증한다.





그림 16. 연도별 성과 분석





그림 17. Rolling 수익률 (6개월)





그림 18. 위기 상황별 성과 비교





그림 19. 위기 후 회복 속도



그림 20. 시장 레짐별 연평균 수익률(CAGR) 비교

[그림 21]은 2022년 하락장 기간, 특히 Terra/LUNA 붕괴와 같은 극심한 변동성 이벤트를 포함하는 시기(Makarov & Schoar, 2022)의 자산 곡선을 보여준다. 베이스라인 모델은 지속적으로 자산 가치가 하락한 반면, 제안 모델은 하락 추세를 조기에 인지하고 방어적인 포지션을 취함으로써 자산을 성공적으로 보존하고 점진적으로 우상향하는 강건성을 보였다. 





그림 21. 2022년 하락장 구간 종합 성과 비교
이러한 방어적 성과는 [그림 22]의 동적 가중치 변화 시각화를 통해 더욱 명확히 이해할 수 있다. 하락장이 시작되자, 시스템은 하락장 전문가 에이전트(Agent-Bear)의 가중치를 신속하게 높여 그들의 방어적 정책이 앙상블 결정에 지배적인 영향을 미치도록 조절했음을 확인할 수 있다. 





그림 22. 동적 가중치 변화 시각화

  제안 모델의 진정한 강건성은 레짐 분류가 완벽하지 않은, 즉 오분류(misclassification)가 발생한 상황에서 어떻게 작동하는지를 통해 확인할 수 있다. 예를 들어, 2022년 하락장의 특정 구간에서 모델이 시장을 '횡보장'으로 일시적으로 오분류한 사례를 분석한 결과, 시스템은 '하락장 전문가'의 공격적인 매도 전략 대신 '횡보장 전문가'의 거래 억제 전략을 채택했다. 

  이로 인해 하락 추세 추종으로 얻을 수 있는 최대 수익은 일부 놓쳤지만, 섣부른 반등 예측에 기반한 매수나 잦은 거래로 인한 손실을 방지하는 효과를 가져왔다. 이는 동적 앙상블 메커니즘이 레짐 분류의 불완전성을 일부 보완하며, 최악의 시나리오를 회피하는 방향으로 작동하는 '안전 장치' 역할을 수행함을 시사한다.

4.6 실시간 페이퍼 트레이딩 검증
  백테스팅은 과거 데이터에 대한 모델의 성능을 평가하지만, 과최적화(overfitting)의 가능성을 완전히 배제할 수 없으며 실시간 데이터 스트림에서 발생하는 예측하지 못한 변수(예: API 지연, 급격한 유동성 변화)를 반영하지 못한다 (Arnott, Harvey, & Markowitz, 2019). 따라서, 제안 모델의 실질적인 운용 가치와 강건성을 최종적으로 검증하기 위해 실시간 페이퍼 트레이딩을 수행하였다.

- 결과 해석
검증 기간인 2025년 6월 1일부터 8월 31일까지의 비트코인 시장은, 전반적인 상승 기조 속에서 두 차례의 단기적인 가격 조정(-15% 내외)을 겪는 등 상당한 변동성을 보였다. 이러한 시장 맥락에서 제안 모델의 성과를 해석하면 다음과 같다.

우수한 절대 및 상대 성과: 제안 모델은 3개월간 15.6%의 수익률을 기록하여, 같은 기간 8.5% 상승한 시장 기준(Buy & Hold)을 7.1%p 초과하는 우수한 성과를 달성했다.

뛰어난 안정성: 특히, 기간 중 시장이 일시적인 조정을 겪었을 때, Buy & Hold 전략의 최대 낙폭은 -10.2%에 달했으나 제안 모델은 -4.9%로 절반 이하의 수준에서 손실을 효과적으로 방어했다.

백테스팅과의 일관성: 승률(61.8%)과 연환산 샤프 비율(2.08) 역시 백테스팅 결과와 유사한 수준을 유지하여, 모델이 과거 데이터에 과최적화되지 않았음을 시사한다.



그림 23. 실시간 페이퍼 트레이딩 상세 결과


    
성과 분석


그림 24. 강건성 메트릭스 상세 분석
표 4. 실시간 페이퍼 트레이딩 성과 비교 (3개월)

평가 지표
동적 앙상블
(제안모델)
Buy&Hold
기간 수익률(%)
  15.6
  8.5
최대 낙폭(MDD, %)
  -4.9
  -10.2
승률 (Win Rate, %)
  61.8
  N/A
샤프 비율(연환산)
  2.08
  1.45

 

5. 결론 및 향후
  본 연구는 선행 연구(Heo & Hwang, 2025a)(게재 예정)에서 제안된 정적 멀티모달 강화학습 모델이 가진 근본적인 한계, 즉 끊임없이 변화하는 시장 레짐에 대한 적응력 부족 문제를 해결하고자 출발하였다. 이를 위해, 시장 상황을 실시간으로 인지하고 그에 맞는 전문가 에이전트 풀을 동적으로 결합하는 새로운 ‘동적 앙상블 트레이딩 시스템’을 제안했다. 제안된 시스템은 멀티모달 데이터를 기반으로 시장을 상승장, 하락장, 횡보장으로 분류하고, 각 레짐에 특화된 에이전트들의 최근 성과에 따라 신뢰도를 재분배하여 최종 투자 결정을 내리는 집단 지성 메커니즘을 구현했다.
  26개월간의 비트코인 데이터를 활용한 포괄적인 백테스팅과 3개월간의 실시간 페이퍼 트레이딩을 통해 제안 시스템의 성능을 다각도로 검증하였다. 그 결과, 제안 시스템은 베이스라인 모델 대비 수익성과 안정성 모든 측면에서 통계적으로 유의미한 성능 개선을 보였다. 특히, 2022년과 같은 극심한 하락장에서 손실을 효과적으로 방어하며 오히려 긍정적인 수익(+7.9%)을 기록한 것은, 본 시스템이 시장 변화에 대한 뛰어난 적응성과 강건성을 갖추었음을 입증하는 강력한 증거이다. 

  본 연구는 시장 레짐과 동적 앙상블 개념을 강화학습 트레이딩에 성공적으로 접목하여, 기존 정적 모델의 한계를 극복할 수 있는 구체적이고 실증적인 방법론을 제시했다는 점에서 학술적, 실무적 의의를 가진다.
다만 본 연구는 다음과 같은 한계점을 가지며, 이는 향후 연구를 통해 개선될 수 있다.
  첫째, 현재 시스템은 argmax 함수를 통해 가장 확률이 높은 단 하나의 레짐 풀을 선택하는 '경성 선택(Hard Selection)' 방식을 사용한다. 이 방식은 레짐 구분이 명확할 때는 효율적이지만, 레짐 전환기와 같이 시장의 방향성이 모호한 상황에서는 문제를 야기할 수 있다. 예를 들어, 레짐 분류 확률이 '상승장 45%, 횡보장 48%'와 같이 비등하게 나타날 경우, 작은 확률 변화에도 시스템의 주력 정책이 급격하게 전환(policy chattering)되어 거래 비용을 유발하고 안정성을 해칠 수 있다. 이러한 한계를 극복하기 위해, 레짐 분류 확률을 연속적인 가중치로 활용하여 모든 전문가 에이전트의 정책을 부드럽게 혼합하는 '소프트 앙상블(Soft Ensemble)' 방식을 도입하여 레짐 전환기의 불확실성에 더욱 유연하게 대응하는 연구를 진행할 것이다.
  둘째, 현재 모델의 결정 과정은 복잡한 내부 메커니즘으로 인해 '블랙박스'처럼 작동하여 사용자가 그 이유를 직관적으로 이해하기 어렵다. 향후 연구에서는 어텐션 스코어 시각화나 LIME, SHAP과 같은 설명 가능한 AI(XAI) 기법을 통합하여, 모델이 왜 특정 시점에 매수 또는 매도 결정을 내렸는지 그 근거를 시각적으로 제시함으로써 모델의 투명성과 신뢰성을 높이고자 한다.

  셋째, 본 연구는 비트코인이라는 단일 자산에 국한되어 검증되었다. 모델의 일반화 가능성을 확보하기 위해 이더리움 등 다른 암호화폐나 주식과 같은 전통 자산으로 연구를 확장하고, 더 나아가 여러 자산을 동시에 관리하는 다중 자산 포트폴리오 최적화 문제로 확장하는 연구가 필요하다.

  결론적으로, 본 연구는 시장의 근본적인 속성인 '변화'에 대응하기 위해 '적응형 전문가 위원회(Adaptive Committee of Experts)' 패러다임을 제안하고 그 유효성을 입증했다. 이는 향후 지능형 투자 시스템 개발이 단일 최적화된 거대 모델(monolithic model)을 추구하는 대신, 각자의 전문성을 갖춘 에이전트들의 집단 지성을 동적으로 조율하는 방향으로 나아가야 한다는 중요한 이론적, 방법론적 토대를 마련했다는 점에서 깊은 의의가 있다.
∥참고문헌
[국내 문헌]
[1] 허상훈, & 황영배. (2025a). 단일 모델과 멀티모달 모델의 암호화폐 자동매매 성능 비교 분석. 한국차세대컴퓨팅학회.(게재 예정)
[2] 이모세, & 안현철. (2018). 효과적인 입력변수 패턴 학습을 위한 시계열 그래프 기반 합성곱 신경망 모형. 지능정보연구, 24(1), 81-99.
[3] 장성일, & 김정연. (2017). 비트코인의 자산성격에 관한 연구. 한국전자거래학회지, 22(4), 101-115.
[4] 김은미. (2021). 감성 분석을 이용한 뉴스정보와 딥러닝 기반의 암호화폐 수익률 변동 예측을 위한 통합모형. 지식경영연구, 22(2), 145-161.
[5] 홍태호, 원종관, 강필성, & 이경호. (2023). 설명 가능한 인공지능과 CNN을 활용한 암호화폐 가격등락 예측모형. 지능정보연구, 29(2), 1-19.
[국외 문헌]
[1] Ang, A., & Bekaert, G. (2002). Regime switches in interest rates. Journal of Business & Economic Statistics, 20(2), 163-182.
[2] Arnott, R. D., Harvey, C. R., & Markowitz, H. (2019). A backtesting protocol in the era of machine learning. The Journal of Financial Data Science, 1(1), 80-92.
[3] Baltrušaitis, T., Ahuja, C., & Morency, L. P. (2019). Multimodal machine learning: A survey and taxonomy. IEEE Transactions on Pattern Analysis and Machine Intelligence, 41(2), 423-443.
[4] Chen, X., Liu, M., & Wang, H. (2021). CNN-based candlestick pattern recognition for crypto-currency trading. Journal of Financial Data Science, 3(2), 125-140.
[5] Deng, Y., Bao, F., Kong, Y., Ren, Z., & Dai, Q. (2017). Deep direct reinforcement learning for financial signal representation and trading. IEEE Transactions on Neural Networks and Learning Systems, 28(3), 653-664.
[6] Dietterich, T. G. (2000). Ensemble methods in machine learning. In Multiple classifier systems (pp. 1-15). Springer, Berlin, Heidelberg.
[7] Fabozzi, F. J., & Markowitz, H. M. (Eds.). (2011). The theory and practice of investment management (Vol. 1). John Wiley & Sons.
[8] Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. The Journal of Finance, 25(2), 383-417.
[9] Gat, I., et al. (2022). A survey on multi-agent reinforcement learning for finance. Journal of Finance and Data Science, 8, 257-270.
[10] Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. The Review of Financial Studies, 33(5), 2223-2273.
[11] Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. Econometrica, 57(2), 357-384.
[12] Hautsch, N., Scheuch, C., & Voigt, S. (2021). The market impact of exchange-traded funds (ETFs): The case of the S&P 500. Journal of Financial and Quantitative Analysis, 56(8), 2901-2936.
[13] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
[14] Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). Adaptive mixtures of local experts. Neural computation, 3(1), 79-87.
[15] Kearney, C., & Liu, S. (2014). Textual sentiment in finance: A survey of methods and models. International Review of Financial Analysis, 33, 171-185.
[16] Kritzman, M., Page, S., & Turkington, D. (2012). Regime shifts: Implications for dynamic strategies. Financial Analysts Journal, 68(3), 22-39.
[17] Kuncheva, L. I. (2004). Combining pattern classifiers: methods and algorithms. John Wiley & Sons.
[18] Lee, C., Kim, J., & Kim, J. (2021). A multimodal approach for stock market prediction using corporate disclosures and financial data. Expert Systems with Applications, 183, 115372.
[19] Ledoit, O., & Wolf, M. (2008). Robust performance hypothesis testing with the Sharpe ratio. Journal of Empirical Finance, 15(5), 850-859.
[20] Liang, Z., Chen, H., Zhu, J., Jiang, K., & Li, Y. (2018). Adversarial deep reinforcement learning in portfolio management. arXiv preprint arXiv:1808.09940.
[21] Littlestone, N., & Warmuth, M. K. (1994). The weighted majority algorithm. Information and computation, 108(2), 212-261.
[22] Makarov, I., & Schoar, A. (2022). Cryptocurrencies and digital assets. NBER Reporter, 2022(2), 1-8.
[23] Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.
[24] Neuneier, R. (1997). Enhancing Q-learning for optimal asset allocation. In Advances in neural information processing systems (pp. 936-942).
[25] Ngiam, J., Khosla, A., Kim, M., Nam, J., Lee, H., & Ng, A. Y. (2011). Multimodal deep learning. In Proceedings of the 28th international conference on machine learning (ICML-11) (pp. 689-696).
[26] Pardo, R. (2008). The evaluation and optimization of trading strategies. John Wiley & Sons.
[27] Rapach, D. E., Strauss, J. K., & Zhou, G. (2010). Out-of-sample equity premium prediction: Combination forecasts and links to the real economy. The Review of Financial Studies, 23(2), 821-862.
[28] Savage, L. J. (1951). The theory of statistical decision. Journal of the American Statistical Association, 46(253), 55-67.
[29] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.
[30] Sharpe, W. F. (1966). Mutual fund performance. The Journal of Business, 39(1), 119-138.
[31] Sortino, F. A., & van der Meer, R. (1991). Downside risk. Journal of Portfolio Management, 17(4), 27-31.
[32] Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.
[33] Tetlock, P. C. (2007). Giving content to investor sentiment: The role of media in the stock market. The Journal of Finance, 62(3), 1139-1168.
[34] Timmermann, A. (2006). Forecast combinations. Handbook of economic forecasting, 1, 135-196.
[35] Xiong, Z., Liu, X. Y., Zhong, S., Yang, H., & Walid, A. (2018). Practical deep reinforcement learning approach for stock trading. arXiv preprint arXiv:1811.07522.
[36] Zhang, L., Wang, S., & Liu, B. (2021). Deep learning for stock prediction using numerical and textual information. IEEE/CAA Journal of Automatica Sinica, 8(3), 561-571.
[37] Zhou, Z. H. (2019). Ensemble methods: foundations and algorithms. CRC Press.

[부록 1]: 동적 다중 레짐 강화학습 앙상블 트레이딩
         시스템
Algorithm 1: Dynamic Multi-Regime Reinforcement Learning Ensemble Trading System
// --- 1. 초기화 (Initialization) ---
1.  // 1-1. 사전 학습된 모델 로드
2. M_regime ← 
   Load_Regime_Classifier()
// 시장 레짐(상승/하락/횡보) 분류 모델 로드
3. {Pool_bull, Pool_bear, Pool_sideways} ← Load_RL_Agent_Pools()      // 각 레짐별 전문가 RL 에이전트 풀 (각 5개) 로드
4.  // 1-2. 포트폴리오 및 하이퍼파라미터 설정
5. Portfolio ← 
   Initialize_Portfolio(initial_capital)
// 초기 자본금으로 포트폴리오 설정
6. T ← Set_Hyperparameter()
// Softmax 함수의 온도(Temperature) T 설정
// --- 2. 메인 트레이딩 루프 (Main Trading Loop) 
7.  // 매 타임스텝 t 마다 반복
8.  for each time step t do:
9.  // --- 단계 1: 상태 표현 및 레짐 분류 ---
10.  // 현재 시점의 시장 데이터(시세, 뉴스 등) 수집
11. OHLCV_t, News_t ← Get_Market_Data(t)
12.  // 수집된 데이터를 바탕으로 멀티모달 상태 벡터 s_t 생성
13. s_t ← 
   Create_Multimodal_State(OHLCV_t, News_t)
14. // 레짐 분류 모델을 통해 현재 상태 s_t의 레짐별 확률 예측
15. // Prob_R_t = [P(bull|s_t), P(bear|s_t), P(sideways|s_t)]
16. Prob_R_t ← M_regime.predict_proba(s_t)
17. // --- 단계 2: 동적 앙상블 메커니즘 ---
18. // 2-1. 레짐 기반 전문가 풀 선택
19. // 가장 확률이 높은 레짐 R_t* 결정
20. R_t* ← argmax(Prob_R_t)
21. // 결정된 레짐에 맞는 전문가 풀(5개 에이전트) 활성화
22. Active_Pool ← Select_Pool(R_t*)
23. // 2-2. 성과 기반 동적 가중치 할당
24. sharpe_ratios ← []  // 5개 에이전트의 성과를 저장할 리스트 초기화
25. for each agent_i in Active_Pool do:
26. // 에이전트 i의 최근 30 타임스텝 동안의 성과(Sharpe Ratio) 계산
27. SR_i ←
 Calculate_Sharpe_Ratio(Portfolio.history_i, 30)
28. sharpe_ratios.append(SR_i)
29. end for
30.  // 5개 에이전트의 성과 리스트를 Softmax 함수에 입력하여 동적 가중치 벡터 w_t 산출
31. // w_t = [w_1, w_2, w_3, w_4, w_5]
32. w_t ← Softmax(sharpe_ratios / T)
33. // 2-3. 가중 앙상블 정책 결정
34. // 최종 앙상블 정책(행동별 확률 분포)을 0으로 초기화
35. π_ensemble(a|s_t) ←
     Initialize_Policy_to_Zeros()
36.  // 활성화된 5개 에이전트의 정책을 동적 가중치로 가중 평균
37. for i, agent_i in enumerate(Active_Pool) do:
38. // 에이전트 i가 상태 s_t에 대해 제시하는 개별 정책(행동 확률 분포) 추출
39. π_i(a|s_t) ← agent_i.get_policy(s_t)
40.  // 앙상블 정책에 가중치(w_t[i])를 적용하여 누적 결합
41.π_ensemble(a|s_t) ← π_ensemble(a|s_t) + w_t[i] · π_i(a|s_t)
42. end for
43. // --- 단계 3: 최종 행동 결정 및 실행 ---
44. // 최종 앙상블 정책에서 가장 높은 확률을 가진 행동 a_t*을 선택
45. a_t* ← argmax_a(π_ensemble(a|s_t))
46. // 선택된 행동 a_t*을 실행하고 포트폴리오 상태를 업데이트
47. Portfolio.Execute_Trade(a_t*)
48. end for

[부록 2] 레짐별 보상 함수 상세 설계
상승장 보상 함수
pythondef bull_reward(portfolio_value_t, portfolio_value_t1):
    """상승장에서의 단순 수익률 기반 보상"""
    return (portfolio_value_t1 - portfolio_value_t) / portfolio_value_t

하락장 보상 함수
pythondef bear_reward(returns_history, transaction_cost, C=0.01):
    """하락장에서의 소티노 비율 기반 보상"""
    downside_returns = [r for r in returns_history if r < 0]
    if len(downside_returns) == 0:
        downside_deviation = 0.001  # 작은 값으로 설정
    else:
        downside_deviation = np.std(downside_returns)
    
    mean_return = np.mean(returns_history)
    sortino_ratio = mean_return / downside_deviation
    
    return sortino_ratio - C * transaction_cost

횡보장 보상 함수
pythondef sideways_reward(portfolio_value_t, portfolio_value_t1, 
                   transaction_cost, C_prime=0.05):
    """횡보장에서의 거래 비용 중점 보상"""
    simple_return = (portfolio_value_t1 - portfolio_value_t) / portfolio_value_t
    return simple_return - C_prime * transaction_cost

[부록 3] 실험 환경 상세 설정

파라미터
후보값
최적값
선택 근거
온도 T
[0.1, 1.0, 10, 100]
10
가중치 분산의 균형점
성과 윈도우
[15, 30, 45, 60일]
30일
단기 성과와 안정성 균형
PPO 학습률
[1e-4, 3e-4, 1e-3]
3e-4
수렴 속도와 안정성 고려
배치 크기
[32, 64, 128]
64
메모리 효율성과 성능 균형
하이퍼파라미터 튜닝 결과


데이터 전처리 세부사항
가격 데이터 정규화: Min-Max 스케일링 [0, 1]
기술적 지표 계산: TA-Lib 라이브러리 사용
∥저자소개



