"""Tests del módulo de volatilidad."""
from __future__ import annotations

import math

import pytest

from optionsdesk.signals.volatility import (
    VolEdge,
    iv_rank,
    realized_volatility,
    variance_risk_premium,
)


# ── realized_volatility ───────────────────────────────────────────────────────

def test_rv_returns_none_for_single_price():
    assert realized_volatility([100.0]) is None


def test_rv_returns_none_for_empty():
    assert realized_volatility([]) is None


def test_rv_constant_series_is_zero():
    series = [100.0] * 30
    rv = realized_volatility(series)
    assert rv is not None
    assert rv == pytest.approx(0.0, abs=1e-10)


def test_rv_positive_for_volatile_series():
    import random
    random.seed(42)
    series = [100.0]
    for _ in range(29):
        series.append(series[-1] * (1 + random.gauss(0, 0.02)))
    rv = realized_volatility(series)
    assert rv is not None
    assert rv > 0.0


def test_rv_annualized_larger_than_daily():
    series = [100.0 * (1 + 0.01 * i) for i in range(30)]
    rv_ann  = realized_volatility(series, annualize=True)
    rv_daily = realized_volatility(series, annualize=False)
    assert rv_ann is not None
    assert rv_daily is not None
    assert rv_ann == pytest.approx(rv_daily * math.sqrt(252))


def test_rv_two_prices():
    # log(110/100) = 0.0953; daily_vol = 0; anualizado = 0
    # Con n=1 retorno la varianza no está definida, pero max(n-1,1)=1
    rv = realized_volatility([100.0, 110.0])
    assert rv is not None
    assert rv >= 0.0


# ── variance_risk_premium ─────────────────────────────────────────────────────

def test_vrp_positive_when_iv_above_rv():
    assert variance_risk_premium(0.60, 0.40) == pytest.approx(0.20)


def test_vrp_negative_when_iv_below_rv():
    assert variance_risk_premium(0.30, 0.45) == pytest.approx(-0.15)


def test_vrp_zero_when_equal():
    assert variance_risk_premium(0.50, 0.50) == pytest.approx(0.0)


# ── iv_rank ───────────────────────────────────────────────────────────────────

def test_iv_rank_none_for_empty_history():
    assert iv_rank(0.55, []) is None


def test_iv_rank_none_when_no_range():
    assert iv_rank(0.55, [0.55, 0.55, 0.55]) is None


def test_iv_rank_at_max():
    assert iv_rank(0.80, [0.40, 0.60, 0.80]) == pytest.approx(100.0)


def test_iv_rank_at_min():
    assert iv_rank(0.40, [0.40, 0.60, 0.80]) == pytest.approx(0.0)


def test_iv_rank_at_midpoint():
    assert iv_rank(0.60, [0.40, 0.80]) == pytest.approx(50.0)


def test_iv_rank_above_100_when_current_exceeds_history():
    # Si la IV actual es más alta que todo el historial, el rank supera 100
    rank = iv_rank(1.00, [0.40, 0.80])
    assert rank is not None
    assert rank > 100.0


# ── VolEdge ───────────────────────────────────────────────────────────────────

def test_voledge_sin_datos_when_iv_none():
    ve = VolEdge.compute(iv=None, spot_history=[100.0] * 20)
    assert ve.label == "sin datos"
    assert not ve.has_edge


def test_voledge_sin_datos_when_short_history():
    ve = VolEdge.compute(iv=0.55, spot_history=[100.0])
    # realized_vol devuelve None con 1 precio → label sin datos
    assert ve.label == "sin datos"


def test_voledge_positivo_high_vrp():
    # IV=0.70, vol realizada ~0% (serie plana) → VRP muy positivo
    ve = VolEdge.compute(iv=0.70, spot_history=[100.0] * 30)
    assert ve.label == "positivo"
    assert ve.has_edge


def test_voledge_negativo_low_iv():
    # IV < vol realizada (serie muy volátil)
    volatile = [100.0 * (1 + 0.05 * ((-1) ** i)) for i in range(30)]
    ve = VolEdge.compute(iv=0.01, spot_history=volatile)
    assert ve.label == "negativo"
    assert not ve.has_edge


def test_voledge_iv_rank_none_without_history():
    ve = VolEdge.compute(iv=0.55, spot_history=[100.0] * 20)
    assert ve.iv_rank_pct is None


def test_voledge_iv_rank_computed_with_history():
    ve = VolEdge.compute(
        iv=0.60,
        spot_history=[100.0] * 20,
        iv_history=[0.40, 0.50, 0.70, 0.80],
    )
    assert ve.iv_rank_pct is not None
    assert 0.0 <= ve.iv_rank_pct <= 100.0


def test_voledge_has_edge_false_for_negative_vrp():
    volatile = [100.0 * (1 + 0.03 * ((-1) ** i)) for i in range(30)]
    ve = VolEdge.compute(iv=0.01, spot_history=volatile)
    assert not ve.has_edge
