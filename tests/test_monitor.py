"""pipeline/monitor.py — 임계값 검사 및 롤백 로직 단위 테스트"""

import pytest
from pipeline.monitor import _check_thresholds, _compute_psi, _get_cv_thresholds
import numpy as np

DEFAULT_THRESHOLDS = _get_cv_thresholds()

BASE_METRICS = {"roc_auc": 0.85, "error_rate": 0.15, "psi_mean": 0.05, "ks_pvalue_min": 0.10}


def test_thresholds_all_ok():
    """모든 지표가 임계값 이내면 None 반환"""
    assert _check_thresholds(BASE_METRICS, DEFAULT_THRESHOLDS) is None


def test_thresholds_low_roc_auc():
    """ROC-AUC가 0.75 미만이면 위반 메시지 반환"""
    metrics = {**BASE_METRICS, "roc_auc": 0.70}
    violation = _check_thresholds(metrics, DEFAULT_THRESHOLDS)
    assert violation is not None
    assert "roc_auc" in violation


def test_thresholds_roc_auc_exact_boundary():
    """ROC-AUC가 정확히 임계값이면 통과 (>=)"""
    metrics = {**BASE_METRICS, "roc_auc": 0.75}
    assert _check_thresholds(metrics, DEFAULT_THRESHOLDS) is None


def test_thresholds_high_error_rate():
    """error_rate가 0.25 초과면 위반 메시지 반환"""
    metrics = {**BASE_METRICS, "error_rate": 0.30}
    violation = _check_thresholds(metrics, DEFAULT_THRESHOLDS)
    assert violation is not None
    assert "error_rate" in violation


def test_thresholds_error_rate_exact_boundary():
    """error_rate가 정확히 임계값이면 통과 (<=)"""
    metrics = {**BASE_METRICS, "error_rate": 0.25}
    assert _check_thresholds(metrics, DEFAULT_THRESHOLDS) is None


def test_thresholds_both_violated():
    """두 지표 모두 위반 시 첫 번째 위반 반환"""
    metrics = {**BASE_METRICS, "roc_auc": 0.60, "error_rate": 0.40}
    violation = _check_thresholds(metrics, DEFAULT_THRESHOLDS)
    assert violation is not None


def test_psi_same_distribution():
    """같은 분포는 PSI가 낮아야 함"""
    rng = np.random.default_rng(0)
    baseline = rng.normal(0, 1, 1000)
    current = rng.normal(0, 1, 1000)
    assert _compute_psi(baseline, current) < 0.10


def test_psi_shifted_distribution():
    """분포가 크게 다르면 PSI > 0.2"""
    rng = np.random.default_rng(0)
    baseline = rng.normal(0, 1, 1000)
    drifted = rng.normal(5, 1, 1000)
    assert _compute_psi(baseline, drifted) > 0.20


def test_psi_triggers_violation():
    """PSI 임계값 초과 시 위반 감지"""
    metrics = {**BASE_METRICS, "psi_mean": 0.30}
    violation = _check_thresholds(metrics, DEFAULT_THRESHOLDS)
    assert violation is not None
    assert "psi" in violation


def test_ks_triggers_violation():
    """KS-test p-value 임계값 미달 시 위반 감지"""
    metrics = {**BASE_METRICS, "ks_pvalue_min": 0.01}
    violation = _check_thresholds(metrics, DEFAULT_THRESHOLDS)
    assert violation is not None
    assert "ks_pvalue" in violation

