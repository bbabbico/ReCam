import os
import sys
import subprocess
import pytest

# 프로젝트 최상위 경로를 sys.path에 추가하여 패키지 임포트가 가능하게 함
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

def test_analysis_pipeline_execution():
    """
    analysis_pipeline.py 가 오류 없이 정상적으로 실행되는지 확인합니다.
    (종료 코드가 0인지 검증)
    """
    pipeline_path = os.path.join(BASE_DIR, "analysis_pipeline.py")
    
    # 파이썬 인터프리터로 analysis_pipeline.py 실행
    result = subprocess.run([sys.executable, pipeline_path], capture_output=True, text=True)
    
    # 정상적으로 실행 종료되었는지 검증 (returncode == 0)
    assert result.returncode == 0, f"analysis_pipeline.py 실행 실패!\n\n[에러 메시지]\n{result.stderr}\n\n[출력 내용]\n{result.stdout}"

def test_main_server_starts():
    """
    main.py 가 정상적으로 실행되고(import), 서버가 켜질 준비(초기화)가 되었는지 확인합니다.
    """
    try:
        from main.main import app
        
        # FastAPI 객체가 정상적으로 생성되었는지 확인
        assert app.title == "카메라 재배치 시각화 지도"
    except Exception as e:
        pytest.fail(f"main.py 초기화 및 데이터 로딩 실패: {e}")
