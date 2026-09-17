# Seed 42 Temporal Pose Dropout 미적용 결과

## 학습 결과

| 모델 | Train | Head epoch | Fine-tuning epoch | 선택 checkpoint | Validation Accuracy | 자체 Test Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| 위치 1 No dropout | 80 | 15 | 23 | 30 | 100% | 80% |
| 위치 2 No dropout | 79 | 10 | 10 | 12 | 70% | 70% |
| 전체 결합 No dropout | 159 | 15 | 12 | 19 | 90% | 80% |

세 모델 모두 seed 42이며 좌우 반전과 XY Gaussian noise는 유지하고 temporal pose dropout만
비활성화했다.

## Test 20

| 모델 | Accuracy | FALL Precision | FALL Recall | FALL F1 | NON_FALL Specificity | Confusion matrix |
|---|---:|---:|---:|---:|---:|---|
| Raw TSSTG | 70% | 100% | 40% | 57.1% | 100% | [[10, 0], [6, 4]] |
| 위치 1 No dropout | **80%** | **80%** | **80%** | **80%** | **80%** | [[8, 2], [2, 8]] |
| 위치 2 No dropout | 70% | 70% | 70% | 70% | 70% | [[7, 3], [3, 7]] |
| 전체 결합 No dropout | **80%** | **80%** | **80%** | **80%** | **80%** | [[8, 2], [2, 8]] |

Test 20에서는 위치 1과 전체 결합 모델이 80%로 동률이다.

## Test 40

| 모델 | Accuracy | FALL Precision | FALL Recall | FALL F1 | NON_FALL Specificity | Confusion matrix |
|---|---:|---:|---:|---:|---:|---|
| Raw TSSTG | 67.5% | 100% | 35% | 51.9% | 100% | [[20, 0], [13, 7]] |
| 위치 1 No dropout | 80% | 80% | 80% | 80% | 80% | [[16, 4], [4, 16]] |
| 위치 2 No dropout | 75% | 75% | 75% | 75% | 75% | [[15, 5], [5, 15]] |
| 전체 결합 No dropout | **85%** | **85%** | **85%** | **85%** | **85%** | [[17, 3], [3, 17]] |

전체 결합 모델이 모든 이진 분류 지표에서 85%로 가장 균형이 좋다.

## Dropout 적용 전·후 핵심 비교

| 범위 | 모델 | Dropout Accuracy | No-dropout Accuracy | Dropout FALL F1 | No-dropout FALL F1 |
|---|---|---:|---:|---:|---:|
| Test 20 | 위치 1 | 75% | **80%** | 76.2% | **80%** |
| Test 20 | 위치 2 | 60% | **70%** | 63.6% | **70%** |
| Test 20 | 전체 결합 | 80% | 80% | 80% | 80% |
| Test 40 | 위치 1 | 75% | **80%** | 76.2% | **80%** |
| Test 40 | 위치 2 | 72.5% | **75%** | 74.4% | **75%** |
| Test 40 | 전체 결합 | 85% | 85% | 84.2% | **85%** |

이번 seed 42에서는 temporal pose dropout을 사용하지 않은 쪽이 위치별 단독 모델에서 더
좋았고, 전체 결합 모델은 Test 20 Accuracy가 동률이었다. Test 40 전체 모델도 Accuracy는
85%로 동률이지만 no-dropout은 FP 3/FN 3, dropout은 FP 2/FN 4다. 즉 dropout 모델은
오경보가 하나 적고, no-dropout 모델은 실제 낙상 누락이 하나 적다. 낙상 안전성을 우선하면
no-dropout 전체 결합 모델이 더 적합하다.

Test 40에는 checkpoint 선택에 사용한 데이터가 포함되므로 최종 독립 성능은 Test 20을 주
결과로 보고 Test 40은 보조 분석으로 사용해야 한다.
