# 연구 결과: 강화학습을 통한 AtomicGPT 추론 능력 향상

본 문서는 GRPO 강화학습을 `AtomicGPT` 모델에 적용하여 얻은 주요 성능 향상 결과를 정리합니다.

## 1. 핵심 연구 결과 (Key Findings)

### 1.1. 보상 알고리즘 개선 및 학습 효율 증대

GRPO 학습의 핵심인 보상 함수를 정교화하여 학습 효율을 개선했습니다.

-   **문제점**: 초기 보상 알고리즘은 모델이 정답을 맞혔음에도 이를 추출하지 못해 보상을 '0'으로 처리하는 문제가 있었습니다. 
-   **해결**: 정답 추출 로직을 대폭 개선하여 XML 형식, 비형식적 풀이 과정, 결론 문장 등 다양한 형태의 정답을 정확히 인식하도록 수정했습니다. 
-   **결과**: 알고리즘 개선 후, 정답에 대한 정확한 보상 할당이 가능해져 모델 훈련을 할 수 있었습니다. 

## 2. 학습 로그 분석: 보상 알고리즘 개선 전후 비교

알고리즘 개선이 실제 학습 과정에 미친 영향을 로그 데이터를 통해 구체적으로 확인할 수 있습니다.

### 2.1. 개선 전: 정답을 맞혔지만 보상을 받지 못한 경우

아래는 알고리즘 개선 전의 학습 로그입니다. 모델이 정답(`1040`)을 정확히 계산하고 `<answer>` 태그까지 사용했지만, 추출 알고리즘의 한계로 인해 핵심 보상인 `correctness_reward_func`가 0점으로 처리되었습니다. 

| 알고리즘 수정 전 학습 로깅 |
|---|
| **Question:**<br>Janet pays $40 hour for 3 hours per week of clarinet lessons and $28/hour for 5 hours a week of piano lessons. How much more does she spend on piano lessons than clarinet lessons in a year?<br><br>**Answer:**<br>1040<br><br>**Response:**<br>Let's break down Janet's weekly expenses for both lessons.<br>Clarinet lessons cost $40/hour for 3 hours/week, so the weekly expense for clarinet lessons is $40 * 3 = $120.<br>Piano lessons cost $28/hour for 5 hours/week, so the weekly expense for piano lessons is $28 * 5 = $140.<br>The difference in weekly expenses between piano and clarinet lessons is $140 - $120 = $20.<br>Since there are 52 weeks in a year, the annual difference in expenses is $20 * 52 = $1040.<br>Janet spends $1040 more on piano lessons than clarinet lessons per year.<br><br>**Extracted:**<br>Janet spends $1040 more on piano lessons than clarinet lessons per year |

**[Step 3] Reward Log (개선 전)**
```json
{
  "reward": -0.0698333826065063,
  "reward_std": 0.3100970387458801,
  "rewards/correctness_reward_func": 0.0,
  "rewards/int_reward_func": 0.0,
  "rewards/soft_format_reward_func": 0.0,
  "rewards/strict_format_reward_func": 0.0,
  "rewards/xmlcount_reward_func": 0.0698333826065063
}
```
**분석**: `xmlcount_reward_func`에서 태그 사용에 대한 최소한의 보상만 받았을 뿐, 정답을 맞힌 것에 대한 보상은 전혀 받지 못해(correctness_reward_func: 0.0) 학습이 비효율적으로 진행되었습니다.

### 2.2. 개선 후: 성공적인 보상 할당

개선된 알고리즘은 아래와 같이 정답을 성공적으로 추출하고, 그에 맞는 보상을 올바르게 할당했습니다.

| 알고리즘 수정 후 학습 로깅 |
|---|
| **Question:**<br>Janelle had 26 green marbles. Then she bought 6 bags of blue marbles. There were 10 marbles in each bag. She created a gift of 6 green marbles and 8 blue marbles and gave it to a friend. How many marbles does Janelle have now?<br><br>**Expected Answer:**<br>72<br><br>**Full Response:**<br>To find the total number of blue marbles Janelle bought, multiply the number of blue marbles per bag by the number of bags:<br>10 marbles/bag * 6 bags = 60 blue marbles.<br>The total number of marbles Janelle had initially was 26 green marbles + 60 blue marbles = 86 marbles.<br>After Janelle gave her friend a gift of 6 green marbles and 8 blue marbles, she had 86 - 6 - 8 = 72 marbles left.<br><br>**Extracted Answer:**<br>After Janelle gave her friend a gift of 6 green marbles and 8 blue marbles, she had 86 - 6 - 8 = **72 marbles** left<br><br>**Expected Number: 72.0, Extracted Number: 72.0**<br>**✓ CORRECT! Reward: 1.5** |

**[Step 13] Reward Log (개선 후)**

```json
{
  "reward": 2.041666667,
  "reward_std": 0.5931,
  "rewards/correctness_reward_func": 1.5,
  "rewards/int_reward_func": 0.5,
  "rewards/soft_format_reward_func": 0.0,
  "rewards/strict_format_reward_func": 0.0,
  "rewards/xmlcount_reward_func": 0.0416666679084301
}
```
- **분석**: 정답을 정확히 추출하여 `correctness_reward_func`에 **1.5점**의 높은 보상이 할당되었습니다. 이처럼 정교화된 보상 시스템 덕분에 모델이 올바른 방향으로 학습을 진행할 수 있었습니다.

### 2.2. 최적 훈련 시점 발견

훈련 과정에서 얻은 '누적 평균 보상'을 plot 하여 모델의 성능이 최고점에 도달하는 시점을 특정했습니다. 

![Average Reward Over Training Steps](./Average_Reward_Over_training_Steps.png)

-   **결과**: **훈련 750 스텝**에서 평균 보상 값이 2.6605로 가장 높았으며, 이 시점의 모델을 가장 성능이 우수한 최종 모델로 확정했습니다. 

## 3. 모델 성능 평가 결과 (Benchmark Results)

GRPO로 훈련된 `AtomicGPT-GRPO RL` 모델과 기존 `AtomicGPT` 모델의 수학적 추론 능력을 두 개의 벤치마크 데이터셋으로 비교 평가했습니다.

| 데이터셋 (난이도) | AtomicGPT-GRPO RL (훈련 후) | AtomicGPT (기존) | 성능 향상 |
|---|---|---|---|
| **Math-500** (고난도) | **47.0** | 41.0 | **+6.0** |
| **Aqua-rat** (유사 난이도) | **53.96** | 51.50 | **+2.46** |

- **결과 요약**: 고난도 추론 능력을 요구하는 `Math-500` 데이터셋에서 **6.0점**의 유의미한 성능 향상을 보였습니다. 훈련 데이터와 유사한 난이도의 `Aqua-rat` 데이터셋에서도 성능이 향상되었습니다.

- **평가 방식**: `pass@1`을 평가 척도로 사용했으며, 복잡한 `Math-500`의 경우 `Gemini-1.5 Pro`를 활용하여 다양한 형태의 정답을 공정하게 평가했습니다.

## 4. 최종 결론

- **결론**: GRPO 강화학습은 적은 리소스와 데이터셋으로도 `AtomicGPT`와 같은 도메인 특화 LLM의 수학적 추론 능력을 효과적으로 향상시킬 수 있음을 실험적으로 증명했습니다. 이러한 결과는 향후 원자력 분야의 복잡한 시뮬레이션 제어 등 고도화된 작업에 LLM을 활용할 수 있는 잠재력을 보여줍니다.