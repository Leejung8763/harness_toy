"""pipeline/deploy.py — 상태 전이 로직 단위 테스트"""

from pipeline.deploy import _get_candidate, _retire_current


def test_retire_current():
    """현재 deployed 모델을 retired로 변경"""
    registry = {
        "models": [
            {"version": "v1", "status": "deployed"},
            {"version": "v2", "status": "evaluated_pass"},
        ]
    }
    retired = _retire_current(registry, exclude_version="v2")
    assert retired is not None
    assert retired["version"] == "v1"
    assert registry["models"][0]["status"] == "retired"


def test_retire_current_no_deployed():
    """배포된 모델이 없으면 None 반환"""
    registry = {"models": [{"version": "v2", "status": "evaluated_pass"}]}
    retired = _retire_current(registry, exclude_version="v2")
    assert retired is None


def test_retire_current_excludes_new_version():
    """신규 버전은 retire 대상에서 제외"""
    registry = {
        "models": [
            {"version": "v1", "status": "deployed"},
        ]
    }
    retired = _retire_current(registry, exclude_version="v1")
    assert retired is None


def test_get_candidate_latest():
    """버전 미지정 시 가장 최근 evaluated_pass 모델 반환"""
    registry = {
        "models": [
            {"version": "v1", "status": "evaluated_pass"},
            {"version": "v2", "status": "evaluated_pass"},
        ]
    }
    candidate = _get_candidate(registry, None)
    assert candidate["version"] == "v2"


def test_get_candidate_specific_version():
    """특정 버전 지정 시 해당 모델 반환"""
    registry = {
        "models": [
            {"version": "v1", "status": "evaluated_pass"},
            {"version": "v2", "status": "evaluated_pass"},
        ]
    }
    candidate = _get_candidate(registry, "v1")
    assert candidate["version"] == "v1"


def test_get_candidate_no_pass():
    """evaluated_pass 모델이 없으면 None 반환"""
    registry = {"models": [{"version": "v1", "status": "trained"}]}
    candidate = _get_candidate(registry, None)
    assert candidate is None


def test_get_candidate_version_not_found():
    """지정한 버전이 없으면 None 반환"""
    registry = {"models": [{"version": "v2", "status": "evaluated_pass"}]}
    candidate = _get_candidate(registry, "v99")
    assert candidate is None
