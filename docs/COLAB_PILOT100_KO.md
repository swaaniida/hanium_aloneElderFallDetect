# TSSTG 100개 파일럿 Colab 학습

Colab에서 GPU 런타임을 선택한 뒤 `tsstg_pilot100_colab.zip`을 업로드한다.

```python
from google.colab import files
files.upload()
```

```bash
!unzip -q tsstg_pilot100_colab.zip -d tsstg_pilot100_colab
%cd tsstg_pilot100_colab
```

학습을 실행한다.

```bash
!python train_tsstg_binary_pilot.py \
  --dataset dataset \
  --output run_pilot100 \
  --head-epochs 15 \
  --finetune-epochs 30
```

결과를 내려받는다.

```python
!zip -qr run_pilot100.zip run_pilot100
files.download("run_pilot100.zip")
```

주요 결과:

- `run_pilot100/binary_tsstg_state_dict.pth`: 실시간 로더에 사용할 2-class 가중치
- `run_pilot100/metrics.json`: validation/test 정확도와 confusion matrix
- `run_pilot100/test_predictions.csv`: test 이벤트별 예측

이 분할은 장소 1 데이터만 사용한 파이프라인 시험용이다. 장소 2나 새로운 사람에 대한 일반화 성능을 의미하지 않는다.

데이터는 각 subtype 10개 안에서 seed 42로 무작위 8/1/1 분할되어 train/validation/test 모두에 10개 subtype이 포함된다. 좌우대칭은 파일을 복제하지 않고 train loader에서 50% 확률로 매 epoch 적용한다.
