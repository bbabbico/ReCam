# PeakGuard-AI
최대전력수요 초과량 예측

## 사용 모델
K-means / Lightgbm

## 그라운드 룰 , 컨벤션
모두 .venv 만들어서 작업 하는게 좋을것 같습니다.

구현 시작하기 전에, <br>
`pip install -r requirements.txt` 실행하고 시작해야 필요한 패키지 다운됩니다. <br>
구현 끝나고, 커밋 전에 `pip freeze > requirements.txt` 입력해서 필요한 패키지 업데이트 해주십쇼
## 프로젝트 구조

기상청 API는 backend/services/weather.py 에 작성 해주시면 됩니다.
