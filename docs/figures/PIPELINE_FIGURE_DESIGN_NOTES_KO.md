# 낙상 감지 파이프라인 논문 그림 설계안

## 그림이 전달할 한 문장

카메라 영상에서 사람의 관절을 추출하고, 25 Hz의 30-frame 관절 시퀀스를
point/motion 두 스트림 TSSTG로 분석하여 낙상 여부를 판정하고 알림을 전송한다.

## 본문 그림에 남길 6단계

1. RGB-D camera input
2. YOLO26n-Pose estimation
3. 13-joint pose extraction
4. 30-frame temporal window at 25 Hz
5. Two-stream TSSTG and binary classification
6. Fall decision and notification

## 숫자로 반드시 남길 정보

- Camera: 640×480, 30 fps
- Pose sampling: 25 Hz
- Window: 30 frames, 약 1.2초
- TSSTG input: point stream과 motion stream
- Output: FALL / NON-FALL

## 본문 그림에서 제거할 정보

아래 구현 상세는 코드와 캡션에는 남기되, 메인 그림에서는 제거한다.

- queue maxsize
- IoU threshold
- missing-frame threshold
- Flask port와 API endpoint
- MQTT topic 이름
- ST-GCN 10개 block의 모든 channel/stride
- HTTP timeout과 cooldown 값

## 시각 구조

- 가로 방향의 단일 흐름
- 텍스트 박스가 아니라 단계별 벡터 일러스트 사용
- 카메라 → 영상 속 skeleton → 시간축 skeleton → two-stream network → 판정 → 알림
- 색상은 입력/pose 계열 파랑, motion 계열 주황, 모델 보라, 정상 초록, 낙상 빨강
- 단계 번호와 짧은 제목만 크게 표시하고 세부 설명은 한 줄로 제한
