"""Tests del módulo regime.py (mocked: sin requests reales)."""
from __future__ import annotations

from optionsdesk.signals.regime import DailyRegime, _build_regime, _neutral_regime


def test_risk_off_when_majority_fx_driven():
    r = _build_regime(confirmed=1, fx_driven=6, total=8, ccl_deltas=[1.0, 1.2, 0.8])
    assert r.label == "risk-off"
    assert r.kelly_mult < 1.0
    assert r.min_grade_override in ("A", "B")


def test_risk_on_when_majority_confirmed_and_ccl_stable():
    r = _build_regime(confirmed=7, fx_driven=1, total=9, ccl_deltas=[0.1, -0.1, 0.0])
    assert r.label == "risk-on"
    assert r.kelly_mult == 1.0
    assert r.min_grade_override is None


def test_cautious_when_mixed_signals():
    r = _build_regime(confirmed=3, fx_driven=3, total=9, ccl_deltas=[0.3, 0.2])
    assert r.label == "cauteloso"
    assert r.kelly_mult < 1.0


def test_neutral_regime_returns_cautious():
    r = _neutral_regime("test")
    assert r.label == "cauteloso"
    assert r.is_cautious()


def test_risk_off_when_ccl_rising_even_if_confirmed():
    # CCL subiendo fuerte aunque los ADRs confirmen → espejismo potencial
    r = _build_regime(confirmed=5, fx_driven=2, total=9, ccl_deltas=[1.5, 1.8, 2.0])
    assert r.label == "risk-off"   # CCL sube → risk-off sobrescribe


def test_regime_kelly_multipliers_are_ordered():
    off  = _build_regime(0, 7, 9, [1.5])
    caut = _build_regime(3, 3, 9, [0.3])
    on   = _build_regime(7, 1, 9, [0.0])
    assert off.kelly_mult <= caut.kelly_mult <= on.kelly_mult
