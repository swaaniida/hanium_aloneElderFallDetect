# 위치 1 TSSTG Temporal Pose Dropout 비교

## 비교 조건

- 기존 모델: 위치 1 Train 80개, 좌우 반전과 XY Gaussian noise 사용
- 새 모델: 동일 조건에 temporal pose dropout 추가
- Temporal dropout: 확률 0.35, 연속 1~4프레임, 좌표 보간, confidence 0
- 두 모델 모두 위치 1의 동일한 Train/Validation/Test 분할 사용
- 최종 비교: 동일한 공통 Test 20개(위치 1 10개 + 위치 2 10개)
- Raw TSSTG: 7-class argmax가 `Fall Down`일 때만 FALL로 변환

## 공통 Test 20개 결과

| 모델 | Accuracy | Balanced Accuracy | FALL Precision | FALL Recall | FALL F1 | NON_FALL Specificity |
|---|---:|---:|---:|---:|---:|---:|
| Raw TSSTG | 70% | 70% | 100% | 40% | 57.1% | 100% |
| 기존 위치 1 모델 | **80%** | **80%** | **80%** | **80%** | **80%** | **80%** |
| Temporal dropout 모델 | 75% | 75% | 72.7% | **80%** | 76.2% | 70% |

Temporal dropout 모델은 낙상 재현율 80%를 유지했지만 비낙상 오탐이 2건에서 3건으로
늘었다. 따라서 Accuracy와 Balanced Accuracy는 5%p, FALL F1은 3.8%p, NON_FALL
Specificity는 10%p 하락했다.

## 장소별 결과

| 평가 범위 | 모델 | Accuracy | FALL Recall | FALL F1 | NON_FALL Specificity |
|---|---|---:|---:|---:|---:|
| 위치 1 Test 10 | 기존 | 80% | 80% | 80% | 80% |
| 위치 1 Test 10 | Dropout | 80% | 80% | 80% | 80% |
| 위치 2 Test 10 | 기존 | **80%** | 80% | **80%** | **80%** |
| 위치 2 Test 10 | Dropout | 70% | 80% | 72.7% | 60% |

학습 장소인 위치 1에서는 두 모델의 예측이 동일하다. 전체 차이는 위치 2의
`S001_E158` 한 건에서 발생했다. 정답은 NON_FALL이며 기존 모델의 FALL score는
0.0107이었지만 dropout 모델에서는 0.6469로 상승해 FALL로 오탐했다.

## 학습 곡선 해석

| 항목 | 기존 | Dropout |
|---|---:|---:|
| 실행된 전체 epoch | 38 | 36 |
| 선택 checkpoint | epoch 30 | epoch 28 |
| 최저 Validation loss | 0.0105 | **0.0073** |
| Validation accuracy | 100% | 100% |
| 위치 1 자체 Test accuracy | 80% | 80% |

Dropout 모델은 더 낮은 최저 Validation loss를 기록했지만 fine-tuning 구간의 Validation
loss 변동이 더 컸고 공통 Test 일반화는 개선되지 않았다. Validation이 10개뿐이므로 낮은
Validation loss 하나만으로 dropout이 더 좋다고 판단하면 안 된다.

## 결론

이번 seed 42 단일 실험에서는 temporal pose dropout을 적용한 위치 1 모델보다 기존 위치 1
모델이 낫다. 낙상 재현율은 같고 기존 모델의 비낙상 오탐이 하나 적기 때문이다. 따라서
현재 위치 1 단독 모델을 배포 후보로 고른다면 기존 모델을 유지하는 것이 타당하다.

다만 Test가 20개뿐이고 차이가 한 이벤트에서만 발생했으므로 temporal dropout 자체가 항상
나쁘다고 일반화할 수는 없다. 이 결론은 현재 수집 데이터와 seed 42에 한정한다. 전체 159개
학습 모델은 별도로 같은 공통 Test에서 판단해야 한다.
