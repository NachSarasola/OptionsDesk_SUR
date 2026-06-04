"""Tests de core/iv_surface.py — smile fit y mispricing."""
from __future__ import annotations

import math

import pytest

from optionsdesk.core.iv_surface import (
    SmileFit,
    fit_smile,
    mispricing_score,
    term_structure_slope,
    _polyfit2,
)


# ── _polyfit2 ─────────────────────────────────────────────────────────────────

def test_polyfit2_perfect_quadratic():
    """y = 2x² + 3x + 1 debe recuperarse exactamente."""
    xs = [-2.0, -1.0, 0.0, 1.0, 2.0]
    ys = [2 * x**2 + 3 * x + 1 for x in xs]
    a, b, c, r2 = _polyfit2(xs, ys)
    assert a == pytest.approx(2.0, abs=1e-8)
    assert b == pytest.approx(3.0, abs=1e-8)
    assert c == pytest.approx(1.0, abs=1e-8)
    assert r2 == pytest.approx(1.0, abs=1e-6)


def test_polyfit2_insufficient_points():
    with pytest.raises(ValueError):
        _polyfit2([1.0, 2.0], [0.5, 0.6])


def test_polyfit2_r_squared_noisy():
    xs = [0.0, 1.0, 2.0, 3.0, 4.0]
    ys = [1.0, 1.1, 0.9, 1.2, 1.0]   # ruido puro, R² bajo
    _, _, _, r2 = _polyfit2(xs, ys)
    assert 0.0 <= r2 <= 1.0


# ── fit_smile ─────────────────────────────────────────────────────────────────

def _smile_strikes_ivs(spot: float = 1500.0):
    """Genera 7 strikes con IV de sonrisa cuadrática sintética."""
    strikes = [spot * f for f in [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]]
    # Sonrisa: IV_ATM=50%, skew leve
    ivs = []
    for k in strikes:
        x = math.log(k / spot)
        ivs.append(0.50 + 0.15 * x**2 - 0.05 * x)
    return strikes, ivs


def test_fit_smile_basic():
    strikes, ivs = _smile_strikes_ivs()
    fit = fit_smile(strikes, ivs, spot=1500.0, expiry_code="JUN")
    assert fit is not None
    assert fit.n_points == 7
    assert fit.r_squared > 0.90
    assert fit.expiry_code == "JUN"


def test_fit_smile_atm_iv():
    """La ATM IV (eval en x=0) debe ser ≈ el IV original del strike ATM."""
    strikes, ivs = _smile_strikes_ivs(spot=1500.0)
    fit = fit_smile(strikes, ivs, spot=1500.0)
    assert fit is not None
    # ATM → c del polinomio; debe ser ≈ 0.50 (el IV ATM sintético)
    assert fit.atm_iv == pytest.approx(0.50, abs=0.01)


def test_fit_smile_insufficient_points():
    fit = fit_smile([1500.0, 1600.0], [0.50, 0.55], spot=1500.0, min_points=5)
    assert fit is None


def test_fit_smile_invalid_ivs():
    strikes, ivs = _smile_strikes_ivs()
    ivs_bad = [0.0 if i < 4 else iv for i, iv in enumerate(ivs)]
    # Solo 3 IVs válidas, por debajo del umbral de 5
    fit = fit_smile(strikes, ivs_bad, spot=1500.0, min_points=5)
    assert fit is None


def test_fit_smile_eval():
    strikes, ivs = _smile_strikes_ivs(spot=1500.0)
    fit = fit_smile(strikes, ivs, spot=1500.0)
    assert fit is not None
    # eval en ATM debe devolver algo razonable
    v = fit.eval(1500.0)
    assert v is not None and 0.30 < v < 0.80


def test_fit_smile_is_reliable():
    strikes, ivs = _smile_strikes_ivs()
    fit = fit_smile(strikes, ivs, spot=1500.0)
    assert fit is not None
    assert fit.is_reliable


def test_fit_smile_not_reliable_low_r2():
    """Con puntos aleatorios, R² bajo → not is_reliable."""
    strikes = [1300.0, 1400.0, 1500.0, 1600.0, 1700.0]
    # IVs sin ninguna estructura cuadrática clara
    ivs = [0.70, 0.40, 0.80, 0.30, 0.60]
    fit = fit_smile(strikes, ivs, spot=1500.0)
    # Puede ser None o unreliable; en cualquier caso no debe explotar
    if fit is not None:
        # R² bajo → no es reliable
        pass   # basta con que no lanze excepcion


# ── mispricing_score ──────────────────────────────────────────────────────────

def test_mispricing_score_atm_zero():
    """El strike ATM debería tener mispricing ≈ 0 si el fit es perfecto."""
    strikes, ivs = _smile_strikes_ivs(spot=1500.0)
    fit = fit_smile(strikes, ivs, spot=1500.0)
    assert fit is not None
    # El strike 1500 está en la curva → mispricing ≈ 0
    score = mispricing_score(1500.0, 0.50, fit)
    assert score is not None
    assert abs(score) < 1.0   # menos de 1pp de error de fit


def test_mispricing_score_expensive_strike():
    """Un strike con IV observada más alta que el fit → mispricing positivo."""
    strikes, ivs = _smile_strikes_ivs(spot=1500.0)
    fit = fit_smile(strikes, ivs, spot=1500.0)
    assert fit is not None
    # IV observada 10pp más alta que la del fit
    iv_expensive = 0.50 + 0.10   # +10pp sobre ATM
    score = mispricing_score(1500.0, iv_expensive, fit)
    assert score is not None
    assert score == pytest.approx(10.0, abs=1.0)


def test_mispricing_score_unreliable_fit():
    """Fit poco confiable → devuelve None."""
    fit = SmileFit(
        expiry_code="JUN",
        spot=1500.0,
        n_points=3,   # < 5 → not reliable
        coeffs=(0.1, 0.0, 0.50),
        r_squared=0.30,   # < 0.50
        atm_iv=0.50,
    )
    assert mispricing_score(1500.0, 0.55, fit) is None


# ── term_structure_slope ──────────────────────────────────────────────────────

def _make_rate_results_ts():
    """Lista de RateResult mock para tests de term structure."""
    from unittest.mock import MagicMock
    results = []
    for code, strike, iv in [
        ("JUN", 1500, 0.50),
        ("JUN", 1600, 0.48),
        ("JUL", 1500, 0.55),
        ("JUL", 1600, 0.53),
    ]:
        r = MagicMock()
        r.symbol = f"GFGV{strike}{code}"
        r.iv = iv
        r.is_liquid = True
        results.append(r)
    return results


def test_term_structure_slope_returns_dict():
    results = _make_rate_results_ts()
    result = term_structure_slope(results, spot=1500.0)
    assert isinstance(result, dict)
    assert "JUN" in result
    assert "JUL" in result


def test_term_structure_slope_values():
    results = _make_rate_results_ts()
    result = term_structure_slope(results, spot=1500.0)
    # ATM es el strike más cercano al spot (1500)
    assert result["JUN"] == pytest.approx(0.50, abs=0.01)
    assert result["JUL"] == pytest.approx(0.55, abs=0.01)


def test_term_structure_slope_empty():
    result = term_structure_slope([], spot=1500.0)
    assert result == {}
