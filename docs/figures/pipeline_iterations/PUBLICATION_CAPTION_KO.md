# 논문용 그림 캡션

**그림 X. 제안한 skeleton 기반 실시간 낙상 감지 파이프라인.** RGB-D 카메라로
취득한 영상에서 YOLO26n-Pose를 이용해 사람의 관절을 추출하고, 13개 관절의 2차원
좌표와 신뢰도를 25 Hz로 샘플링하여 30-frame pose sequence를 구성한다. TSSTG는
관절 위치 point stream과 프레임 간 변화량 motion stream의 공간·시간 특징을 각각
학습한 뒤 결합하여 낙상 여부를 분류한다. 최종 FALL 판정은 로컬 및 원격 알림으로
전달된다. Bounding box와 depth는 추적 및 시각화에만 사용되며 TSSTG의 직접 입력에는
포함되지 않는다.

## 영문 캡션

**Figure X. Proposed real-time skeleton-based fall detection pipeline.** Human
keypoints are extracted from RGB-D frames using YOLO26n-Pose, and a 30-frame
pose sequence is constructed by sampling 13 two-dimensional joints and their
confidence scores at 25 Hz. TSSTG separately encodes the spatial-temporal
features of the point and motion streams, fuses both representations, and
classifies the sequence as fall or non-fall. A fall decision is delivered to
local and remote notification services. Bounding boxes and depth are used only
for tracking and visualization, not as direct TSSTG inputs.
