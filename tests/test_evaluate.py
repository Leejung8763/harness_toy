"""pipeline/evaluate.py — Quality Gate 로직 단위 테스트"""

import pytest

from pipeline.evaluate import _check_gate


def test_first_deploy_pass():
    """첫 배포: ROC-AUC가 baseline(0.70) 이상이면 통과"""
    passed, reason = _check_gate(0.80, None)
    assert passed
    assert "첫 배포" in reason


def test_first_deploy_fail():
    """첫 배포: ROC-AUC가 baseline(0.70) 미만이면 실패"""
    passed, reason = _check_gate(0.65, None)
    assert not passed
    assert "미달" in reason


def test_first_deploy_exact_threshold():
    """첫 배포: baseline 정확히 일치하면 통과 (>=)"""
    passed, _ = _check_gate(0.70, None)
    assert passed


def test_beats_champion():
    """기존 champion 대비 개선된 모델은 통과"""
    champion = {"version": "v1", "metrics": {"roc_auc": 0.80}}
    passed, reason = _check_gate(0.82, champion)
    assert passed
    assert "유지 또는 개선됨" in reason


def test_equal_to_champion():
    """기존 champion과 동점이면 통과 (>=)"""
    champion = {"version": "v1", "metrics": {"roc_auc": 0.80}}
    passed, _ = _check_gate(0.80, champion)
    assert passed


def test_worse_than_champion():
    """기존 champion 대비 성능 하락 시 실패"""
    champion = {"version": "v1", "metrics": {"roc_auc": 0.80}}
    passed, reason = _check_gate(0.78, champion)
    assert not passed
    assert "성능 하락" in reason
