"""pipeline/monitor.py — 임계값 검사 및 롤백 로직 단위 테스트"""

from pipeline.monitor import _check_thresholds


def test_thresholds_all_ok():
    """모든 지표가 임계값 이내면 None 반환"""
    metrics = {"roc_auc": 0.85, "error_rate": 0.15}
    assert _check_thresholds(metrics) is None


def test_thresholds_low_roc_auc():
    """ROC-AUC가 0.75 미만이면 위반 메시지 반환"""
    metrics = {"roc_auc": 0.70, "error_rate": 0.15}
    violation = _check_thresholds(metrics)
    assert violation is not None
    assert "roc_auc" in violation


def test_thresholds_roc_auc_exact_boundary():
    """ROC-AUC가 정확히 임계값이면 통과 (>=)"""
    metrics = {"roc_auc": 0.75, "error_rate": 0.15}
    assert _check_thresholds(metrics) is None


def test_thresholds_high_error_rate():
    """error_rate가 0.25 초과면 위반 메시지 반환"""
    metrics = {"roc_auc": 0.85, "error_rate": 0.30}
    violation = _check_thresholds(metrics)
    assert violation is not None
    assert "error_rate" in violation


def test_thresholds_error_rate_exact_boundary():
    """error_rate가 정확히 임계값이면 통과 (<=)"""
    metrics = {"roc_auc": 0.85, "error_rate": 0.25}
    assert _check_thresholds(metrics) is None


def test_thresholds_both_violated():
    """두 지표 모두 위반 시 첫 번째 위반 반환"""
    metrics = {"roc_auc": 0.60, "error_rate": 0.40}
    violation = _check_thresholds(metrics)
    assert violation is not None
