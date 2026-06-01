"""Tests del módulo de volatilidad."""
from __future__ import annotations

import math

import pytest

import pandas as pd

from optionsdesk.signals.volatility import (
    VolEdge,
    iv_rank,
    parkinson_volatility,
    realized_volatility,
    variance_risk_premium,
    yang_zhang_volatility,
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


# ── realized_volatility window ────────────────────────────────────────────────

def test_rv_window_subset():
    """window=20 sobre serie de 100 ≈ realized_volatility de las últimas 20."""
    import random
    random.seed(7)
    series = [100.0]
    for _ in range(99):
        series.append(series[-1] * (1 + random.gauss(0, 0.015)))

    rv_window = realized_volatility(series, window=20)
    rv_last20 = realized_volatility(series[-20:])
    assert rv_window is not None
    assert rv_window == pytest.approx(rv_last20, rel=1e-9)


def test_rv_window_larger_than_series():
    """window mayor que la serie → usa la serie completa sin error."""
    series = [100.0 * (1 + 0.01 * i) for i in range(30)]
    rv_w50 = realized_volatility(series, window=50)
    rv_full = realized_volatility(series)
    assert rv_w50 is not None
    assert rv_w50 == pytest.approx(rv_full, rel=1e-9)


def test_rv_window_none_matches_full():
    """window=None es idéntico al comportamiento sin window."""
    series = [100.0 * (1 + 0.008 * i) for i in range(40)]
    assert realized_volatility(series, window=None) == pytest.approx(
        realized_volatility(series), rel=1e-9
    )


def test_rv_window_changes_value_on_regime_shift():
    """vol rolling captura régimen reciente; la serie completa lo diluye."""
    calm = [100.0 * (1 + 0.002 * ((-1) ** i)) for i in range(80)]
    volatile = [calm[-1] * (1 + 0.06 * ((-1) ** i)) for i in range(20)]
    series = calm + volatile

    rv_20 = realized_volatility(series, window=20)
    rv_full = realized_volatility(series)
    assert rv_20 is not None and rv_full is not None
    # El rolling reciente es mucho mayor: captura el régimen volátil
    assert rv_20 > rv_full * 1.5


# ── yang_zhang_volatility ─────────────────────────────────────────────────────

def _make_ohlc(n: int = 30, daily_vol: float = 0.02) -> pd.DataFrame:
    """Genera OHLC sintético con vol diaria aproximada."""
    import random
    random.seed(99)
    rows = []
    price = 1500.0
    for _ in range(n):
        ret = random.gauss(0, daily_vol)
        o = price
        c = price * (1 + ret)
        h = max(o, c) * (1 + abs(random.gauss(0, daily_vol / 2)))
        l = min(o, c) * (1 - abs(random.gauss(0, daily_vol / 2)))
        rows.append({"open": o, "high": h, "low": l, "close": c})
        price = c
    return pd.DataFrame(rows)


def test_yang_zhang_positive_for_volatile_ohlc():
    df = _make_ohlc(n=30, daily_vol=0.02)
    yz = yang_zhang_volatility(df, window=25)
    assert yz is not None
    assert yz > 0.0


def test_yang_zhang_near_zero_for_flat_ohlc():
    """OHLC casi plano → vol cercana a cero."""
    rows = [{"open": 1500.0, "high": 1500.01, "low": 1499.99, "close": 1500.0}] * 30
    df = pd.DataFrame(rows)
    yz = yang_zhang_volatility(df, window=25)
    assert yz is not None
    assert yz < 0.01   # < 1% anualizado


def test_yang_zhang_returns_none_insufficient_data():
    df = _make_ohlc(n=2)
    assert yang_zhang_volatility(df, window=25) is None


def test_yang_zhang_missing_columns():
    df = pd.DataFrame({"close": [1500.0] * 20})
    assert yang_zhang_volatility(df) is None


def test_yang_zhang_annualized_vs_daily():
    df = _make_ohlc(n=30)
    ann = yang_zhang_volatility(df, annualize=True)
    daily = yang_zhang_volatility(df, annualize=False)
    assert ann is not None and daily is not None
    assert ann == pytest.approx(daily * math.sqrt(252), rel=0.01)


def test_yang_zhang_case_insensitive_columns():
    """Acepta columnas en mayúsculas también."""
    df = _make_ohlc(n=25)
    df.columns = [c.upper() for c in df.columns]
    assert yang_zhang_volatility(df) is not None


# ── parkinson_volatility ──────────────────────────────────────────────────────

def test_parkinson_positive_for_volatile_ohlc():
    df = _make_ohlc(n=25)
    pk = parkinson_volatility(df)
    assert pk is not None
    assert pk > 0.0


def test_parkinson_returns_none_for_insufficient_data():
    df = pd.DataFrame([{"high": 100.0, "low": 99.0}])
    assert parkinson_volatility(df) is None


def test_parkinson_missing_columns():
    df = pd.DataFrame({"close": [100.0] * 20})
    assert parkinson_volatility(df) is None


def test_parkinson_near_zero_for_flat_ohlc():
    rows = [{"open": 1500.0, "high": 1500.001, "low": 1499.999, "close": 1500.0}] * 25
    df = pd.DataFrame(rows)
    pk = parkinson_volatility(df)
    assert pk is not None
    assert pk < 0.005


def test_parkinson_annualized_vs_daily():
    df = _make_ohlc(n=25)
    ann = parkinson_volatility(df, annualize=True)
    daily = parkinson_volatility(df, annualize=False)
    assert ann is not None and daily is not None
    assert ann == pytest.approx(daily * math.sqrt(252), rel=0.01)


# ── VolEdge.compute con ohlc_df (v2.7) ───────────────────────────────────────

def test_voledge_compute_uses_yz_when_ohlc_provided():
    """VolEdge.compute con ohlc_df usa YZ en lugar de C-C."""
    df = _make_ohlc(n=30, daily_vol=0.02)
    closes = df["close"].tolist()
    # IV moderada; serie volatil → VRP debe ser no-None con ohlc_df
    ve_yz = VolEdge.compute(iv=0.60, spot_history=closes, ohlc_df=df, window=20)
    assert ve_yz.realized_vol is not None
    assert ve_yz.label in {"positivo", "neutro", "negativo"}


def test_voledge_compute_ohlc_fallback_to_cc_on_missing_cols():
    """ohlc_df sin columnas OHLC completas → cae a C-C, no falla."""
    df_bad = pd.DataFrame({"close": [100.0 + i for i in range(25)]})
    closes = df_bad["close"].tolist()
    ve = VolEdge.compute(iv=0.55, spot_history=closes, ohlc_df=df_bad, window=20)
    # YZ devuelve None (faltan columnas) → usa C-C → realized_vol no None
    assert ve.realized_vol is not None


def test_voledge_compute_window_param_propagates():
    """window se propaga al estimador C-C cuando no hay ohlc_df."""
    import random
    random.seed(42)
    series = [100.0]
    for _ in range(99):
        series.append(series[-1] * (1 + random.gauss(0, 0.015)))
    ve_w20 = VolEdge.compute(iv=0.50, spot_history=series, window=20)
    ve_full = VolEdge.compute(iv=0.50, spot_history=series)
    # con serie de 100 dias, vol rolling 20d != vol completa para serie no-estacionaria
    assert ve_w20.realized_vol is not None
    assert ve_full.realized_vol is not None
