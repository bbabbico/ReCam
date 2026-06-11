# 무인 단속 카메라 재분배 지도 (ReCam)

## 프로젝트 개요
교통사고 위험지역 데이터와 무인 단속 카메라 위치 데이터를 K-Means 알고리즘으로 군집화하여 분석합니다. 효율이 떨어지는 과잉 설치 카메라를 도출하고, 이를 사고 다발 사각지대로 재배치하는 우선순위를 산출하여 지도에 시각화하는 시스템입니다.

## 주요 기능
- **공간 매핑:** 사고위험지역과 카메라 좌표(반경 100m) 매핑 (WGS84 → UTM-K 변환, `cKDTree` 적용)
- **군집화 분석:** K-Means(k=4)로 전국 위험지역 4대 유형 분류 후, 카메라 설치지점에 대한 2차 재군집화(k=3)를 통해 잉여 자원 파악
- **우선순위 도출:** EPDO(사망자, 중상자 가중치) 위험점수 기반 재배치 3단계 우선순위 산출
- **웹 시각화:** FastAPI와 Kakao Map API를 활용하여 6개 카테고리별 마커 및 상세 데이터 웹 대시보드 제공

## 기술 스택
- **언어/분석:** Python, Pandas, Scikit-learn, SciPy
- **백엔드/시각화:** FastAPI, Uvicorn, Matplotlib, Kakao Map API

## 실행 방법

1. **의존성 설치**
   ```bash
   pip install -r requirements.txt
   ```

2. **분석 파이프라인 실행** (선택)
   ```bash
   python analysis_pipeline.py
   ```
   > 분석 및 군집화가 진행되며 `output_v2/` 디렉토리에 결과물(CSV, 차트)이 산출됩니다.

3. **웹 시각화 서버 실행**
   ```bash
   python main/main.py
   ```
   > 서버 실행 후 브라우저에서 `http://localhost:8000`으로 접속하여 지도를 확인합니다.

## 테스트 및 CI
GitHub Actions를 통한 CI 환경을 구축하였습니다.
- **통과 기준:** `analysis_pipeline.py` 스크립트 실행 성공 및 `main.py` 서버 정상 초기화
- **로컬 테스트 실행:**
  ```bash
  pytest tests/test_ci.py
  ```

## 개발 규칙
- **PR 및 Merge:** `main` 브랜치에 직접 Push하지 않으며, 팀원 간 카카오톡 논의 후 PR 및 Merge 진행 (매일 대면 소통하므로 복잡한 Git flow는 생략)
