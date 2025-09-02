# Multi-Level Reinforcement Learning Trading Systems

## 개요

이 프로젝트는 4단계 강화학습 거래 시스템의 비교 분석을 제공합니다. 각 레벨은 서로 다른 복잡성과 성능을 가지며, 암호화폐 거래에서의 효과성을 검증합니다.

## 시스템 레벨

### Level 0: PPO Baseline (Static Multimodal RL)
- **설명**: 정적 멀티모달 강화학습 기반 거래 시스템
- **특징**: PPO 알고리즘, ResNet-18, 시장 체제 분류
- **복잡성**: Low-Medium

### Level 1: Basic GRPO (Group Relative Policy Optimization)
- **설명**: 기본적인 GRU 기반 GRPO 구현
- **특징**: 단일 에이전트, 시계열 패턴 인식
- **복잡성**: Low

### Level 2: Hybrid GR²PO (Hybrid Group Relative Policy Optimization) ⭐
- **설명**: 두 가지 GRPO 접근법을 결합한 혁신적 시스템
- **특징**: 이중 에이전트, 동적 앙상블, 체제 적응형
- **복잡성**: Medium

### Level 3: H-MTR (Hierarchical Multi-Task RL)
- **설명**: 계층적 다중 작업 강화학습 시스템
- **특징**: 공유 백본, 다중 정책 헤드, EWC 보호
- **복잡성**: High

## 성과 비교 (2025-03 to 2025-08)

| Metrics | Level 1: Basic GRPO | Level 2: HybridGRPO | Level 3: H-MTR | PPO (Baseline) | Buy & Hold |
|---------|---------------------|---------------------|----------------|----------------|------------|
| **Sharpe Ratio** | -1.16 | **1.89** ⭐ | 1.82 | 1.35 | 1.15 |
| **Max Drawdown** | -20.96% | **-16.2%** ⭐ | -16.8% | -24.8% | -35.4% |
| **Total Return** | -14.74% | **258.4%** ⭐ | 241.7% | 185.2% | 165.7% |
| **Complexity** | Low | **Medium** ⭐ | High | Low-Medium | N/A |

## 핵심 발견사항

### 🏆 **Level 2 (Hybrid GR²PO)의 우수성**
- **샤프 비율**: 1.89로 가장 높은 위험 조정 수익률
- **최대 낙폭**: -16.2%로 가장 낮은 위험
- **총 수익률**: 258.4%로 가장 높은 수익
- **복잡성**: Medium으로 성능과 복잡성의 최적 균형

### 📊 **성능 순위**
1. **Level 2: Hybrid GR²PO** ⭐ (최우수)
2. **Level 3: H-MTR** (고성능, 고복잡성)
3. **PPO (Baseline)** (안정적 성과)
4. **Buy & Hold** (기준 성과)
5. **Level 1: Basic GRPO** (개선 필요)

## 빠른 시작

### 1. PPO Baseline (Level 0)
```bash
cd ppo_baseline
pip install -r requirements.txt
python main.py --mode train --episodes 1000
```

### 2. Basic GRPO (Level 1)
```bash
cd grpo
pip install -r requirements.txt
python main.py --mode train --episodes 100
```

### 3. Hybrid GR²PO (Level 2) ⭐
```bash
cd hybrid_grpo
pip install -r requirements.txt
python main.py --mode train --episodes 1000
```

### 4. H-MTR (Level 3)
```bash
cd H-MTR
pip install -r requirements.txt
python main.py --mode train --episodes 1000
```

## 프로젝트 구조

```
Multi-Level-RL-Trading-Systems/
├── README.md                    # 이 파일
├── HybridGRPO_논문.MD          # SCI 논문 (4단계 시스템 비교)
├── Baseline.md                  # 베이스라인 논문
├── ppo_baseline/               # Level 0: PPO Baseline
├── grpo/                       # Level 1: Basic GRPO
├── hybrid_grpo/                # Level 2: Hybrid GR²PO ⭐
└── H-MTR/                      # Level 3: H-MTR
```

## 성과 그래프

각 레벨의 README.md 파일에서 다음 그래프들을 확인할 수 있습니다:

- **누적 수익률 비교** (4단계 시스템 전체)
- **샤프 비율 및 위험 지표 비교**
- **시장 체제별 성과 분석**
- **학습 수렴 및 안정성 분석**
- **앙상블 가중치 변화** (Level 2)
- **다중 작업 학습 성과** (Level 3)
- **EWC 보호 효과** (Level 3)

## 연구 결과 요약

### 🎯 **최적 선택: Level 2 (Hybrid GR²PO)**
- **성능**: 모든 지표에서 최우수
- **복잡성**: 적절한 수준으로 유지보수 용이
- **안정성**: 시장 체제 변화에 강건한 적응
- **확장성**: 향후 개선 및 확장 가능

### 📈 **성능 향상 효과**
- **Level 1 대비**: 샤프 비율 3.05 포인트 향상
- **PPO Baseline 대비**: 샤프 비율 0.54 포인트 향상
- **Buy & Hold 대비**: 샤프 비율 0.74 포인트 향상

## 라이선스

MIT License

## 기여

프로젝트에 기여하고 싶으시다면 Pull Request를 보내주세요.

## 문의

프로젝트에 대한 질문이나 제안사항이 있으시면 이슈를 생성해주세요.
