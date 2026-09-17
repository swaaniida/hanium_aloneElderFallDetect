# TSSTG 2개 장소 비교 실험

## 고정 데이터 분할

seed 42로 각 `location × action_code`의 10개 이벤트를 8/1/1로 나눈다.
모든 모델의 최종 평가는 동일한 `S001_full200_compare/test.npz`만 사용한다.

| 데이터셋 | Train | Validation | Test | 비고 |
|---|---:|---:|---:|---|
| `S001_location1_compare` | 80 | 10 | 10 | 장소 1 전용 학습 |
| `S001_location2_compare` | 79 | 10 | 10 | E101을 Train에서 제외 |
| `S001_full200_compare` | 159 | 20 | 20 | 위 두 데이터셋 split의 합집합 |

E101은 원래 Train에 배정된 이벤트이므로 제외해도 Validation/Test 구성은 바뀌지 않는다.

## 학습할 모델

1. Raw TSSTG: 기존 7-class 가중치, 재학습 없음
2. Location 1 TSSTG: 장소 1 Train 80개로 head 학습 후 전체 fine-tuning
3. Location 2 TSSTG: 장소 2 Train 79개로 head 학습 후 전체 fine-tuning
4. Full TSSTG: 장소 1+2 Train 159개로 head 학습 후 전체 fine-tuning

Full 모델은 Location 1 모델에 Location 2만 이어서 학습하지 않는다. 원본 pretrained
TSSTG에서 시작해 합쳐진 159개로 head 단계부터 다시 학습한다. 순차 학습은 장소 순서
편향과 catastrophic forgetting을 만들 수 있으므로 별도 continual-learning 실험으로만 다룬다.

Train에는 좌우 반전, XY Gaussian noise와 temporal pose dropout을 적용한다. temporal
dropout은 기본적으로 확률 0.35로 연속 1~4프레임의 confidence를 0으로 만들고 좌표를
보간한다. Validation과 Test에는 어떤 증강도 적용하지 않는다.

## 학습 명령

아래 명령에서 데이터셋과 출력 이름만 바꾸어 세 모델을 각각 학습한다.

```powershell
python -m tsstg_pipeline.train_pilot `
    --dataset .\tsstg_binary_dataset\S001_full200_compare `
    --output .\run_full200 `
    --head-epochs 15 `
    --finetune-epochs 30 `
    --temporal-dropout-probability 0.35 `
    --temporal-dropout-max-frames 4 `
    --seed 42
```

## 최종 평가

주 평가는 공통 Test 20개의 balanced accuracy와 FALL F1이다. 안전 관점에서 FALL
recall을 함께 우선 보고, 오경보는 NON_FALL specificity로 확인한다. Accuracy만 단독으로
결론 내리지 않는다.

각 모델마다 전체 Test 20개와 장소별 Test 10개씩을 함께 보고한다. 이 구조는 장소 1에만
학습한 모델이 장소 2로 일반화되는지, Full 모델이 양쪽 성능을 동시에 개선하는지 보여준다.

```powershell
python -m tsstg_pipeline.compare_models `
    --dataset .\tsstg_binary_dataset\S001_full200_compare `
    --binary-weights .\run_full200\binary_tsstg_state_dict.pth `
    --splits test `
    --output .\run_full200\common_test_comparison
```

이번 비교 실험은 세 학습 모델 모두 seed 42로 고정한다. 모델 선택과 threshold 조정은
Validation으로만 수행하고 Test는 최종 확정 후 한 번만 본다.
