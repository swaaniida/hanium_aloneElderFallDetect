# 실시간 낙상 감지 전체 파이프라인 그림 설명

## 논문용 캡션

**그림 X. RealSense와 TSSTG를 이용한 실시간 낙상 감지 및 알림 시스템의 전체
파이프라인.** RealSense에서 30 fps로 취득한 RGB·깊이 프레임을 정렬한 뒤,
YOLO26n-Pose로 사람의 bounding box와 COCO 17개 관절을 추정한다. 이전 프레임의
bounding box와 IoU가 가장 높은 사람을 추적 대상으로 선택하고, TSSTG 입력에 필요한
13개 관절의 2차원 좌표와 신뢰도만 추출한다. 관절 시퀀스는 고정 25 Hz로 샘플링되어
30-frame rolling buffer에 저장된다. 완성된 시퀀스는 비동기 추론 큐를 통해 TSSTG로
전달된다. TSSTG는 정규화된 관절 위치 stream과 프레임 간 위치 변화량 motion stream을
각각 10개의 ST-GCN block으로 처리하고, 두 256차원 특징을 결합하여 행동을 분류한다.
낙상 또는 누움 행동이 감지되면 cooldown과 비동기 event queue를 거쳐 backend로
HTTP 알림을 전송한다. 주석 영상과 현재 행동은 각각 Flask MJPEG와 MQTT를 통해 로컬 및
원격 모니터링에 제공된다. 깊이와 bounding box는 대상 추적 및 시각화에만 사용되며
TSSTG의 직접 입력에는 포함되지 않는다.

## 그림 내 주요 설정

- Camera input: RGB + depth, 640×480, 30 fps
- Pose model: YOLO26n-Pose, input size 320
- Person tracking: previous bounding-box IoU, fallback to largest box
- Pose input: 13 joints × `(x, y, confidence)`
- Sampling: fixed 25 Hz
- Temporal window: 30 frames, 약 1.2초
- Action inference: 약 5 Hz, queue maxsize 1
- TSSTG graph: neck을 추가한 14 nodes
- TSSTG streams: point `(3×30×14)`, motion `(2×29×14)`
- Feature fusion: 256 + 256 = 512 dimensions
- Original head: 7-class sigmoid classifier
- Event rule: `Fall Down` 또는 `Lying Down`, 10초 cooldown
- External output: HTTP event, Flask MJPEG, MQTT frame/action

## 파일 형식

- PNG: 문서 미리보기와 발표 자료용
- SVG: 벡터 편집 및 웹 문서용
- PDF: 논문 원고 삽입용
