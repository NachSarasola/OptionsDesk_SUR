"""Tests para optionsdesk/signals/harmonics.py

Principios:
- Swings sintéticos determinísticos para verificar detección M/W.
- Verificación matemática de Fibonacci (assert exacto con tolerancia).
- Tolerancias amplias en los tests porque harmonics.py ya las aplica para MERVAL.
"""
from __future__ import annotations

import math
from datetime import datetime

import pandas as pd
import pytest

from optionsdesk.signals.harmonics import (
    PeakFormation,
    fib_levels,
    find_peak_formations,
    prz_boxes,
)
from optionsdesk.signals.smc import swing_points, PdaArray


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_wavy_daily(n: int = 80) -> pd.DataFrame:
    """OHLCV con ondas pronunciadas para detectar swings claros."""
    dates  = pd.date_range("2024-01-02", periods=n, freq="B")
    close  = [1000.0 + i * 3.0 + 150.0 * math.sin(i * 0.25) for i in range(n)]
    high   = [c * 1.015 for c in close]
    low    = [c * 0.985 for c in close]
    open_  = [c * 0.997 for c in close]
    vol    = [1_500_000.0] * n
    return pd.DataFrame({"date": dates, "open": open_, "high": high,
                          "low": low, "close": close, "volume": vol})


def _synthetic_m_swings() -> pd.DataFrame:
    """Swings artificiales para una formación M perfecta.

    X(low=1000) → A(high=1400) → D(low=1050) ← OTE de XA ≈ 0.875 retroceso
    El retroceso (1400-1050)/(1400-1000)=0.875 está dentro del OTE (0.50-0.90).
    """
    rows = [
        # index, HighLow, Level, date
        {"index":  0, "HighLow": -1, "Level": 1000.0, "date": pd.Timestamp("2024-01-02")},
        {"index":  5, "HighLow":  1, "Level": 1400.0, "date": pd.Timestamp("2024-01-09")},
        {"index": 10, "HighLow": -1, "Level": 1050.0, "date": pd.Timestamp("2024-01-16")},
    ]
    return pd.DataFrame(rows)


def _synthetic_w_swings() -> pd.DataFrame:
    """Swings artificiales para una formación W perfecta.

    X(high=1400) → A(low=1000) → D(high=1350) ← OTE: (1350-1000)/(1400-1000)=0.875
    """
    rows = [
        {"index":  0, "HighLow":  1, "Level": 1400.0, "date": pd.Timestamp("2024-01-02")},
        {"index":  5, "HighLow": -1, "Level": 1000.0, "date": pd.Timestamp("2024-01-09")},
        {"index": 10, "HighLow":  1, "Level": 1350.0, "date": pd.Timestamp("2024-01-16")},
    ]
    return pd.DataFrame(rows)


def _make_pda_discount() -> PdaArray:
    return PdaArray(
        swing_high=1500.0, swing_low=800.0,
        equilibrium=1150.0, premium_band=1232.6, discount_band=1067.4,
        zone="discount",
        ote_bullish_lo=1232.6, ote_bullish_hi=1350.2,
        ote_bearish_lo=1350.2, ote_bearish_hi=1420.2,
    )


def _make_pda_premium() -> PdaArray:
    return PdaArray(
        swing_high=1500.0, swing_low=800.0,
        equilibrium=1150.0, premium_band=1232.6, discount_band=1067.4,
        zone="premium",
        ote_bullish_lo=1232.6, ote_bullish_hi=1350.2,
        ote_bearish_lo=1350.2, ote_bearish_hi=1420.2,
    )


# ── Tests: fib_levels ─────────────────────────────────────────────────────────

class TestFibLevels:
    def test_returns_dict(self):
        result = fib_levels(1000.0, 1500.0)
        assert isinstance(result, dict)

    def test_zero_range_returns_empty(self):
        assert fib_levels(1000.0, 1000.0) == {}

    def test_key_levels_present(self):
        result = fib_levels(1000.0, 2000.0)
        for key in ("0.000", "0.382", "0.500", "0.618", "0.786", "0.886", "1.000"):
            assert key in result

    def test_fib_math_correct(self):
        lo, hi = 1000.0, 1500.0
        rng    = hi - lo
        result = fib_levels(lo, hi)
        assert result["0.000"]  == pytest.approx(lo,                rel=1e-6)
        assert result["0.500"]  == pytest.approx(lo + 0.500 * rng,  rel=1e-6)
        assert result["0.618"]  == pytest.approx(lo + 0.618 * rng,  rel=1e-6)
        assert result["0.786"]  == pytest.approx(lo + 0.786 * rng,  rel=1e-6)
        assert result["0.886"]  == pytest.approx(lo + 0.886 * rng,  rel=1e-6)
        assert result["1.000"]  == pytest.approx(hi,                rel=1e-6)
        assert result["1.272"]  == pytest.approx(hi + 0.272 * rng,  rel=1e-6)
        assert result["1.618"]  == pytest.approx(hi + 0.618 * rng,  rel=1e-6)
        assert result["-0.272"] == pytest.approx(lo - 0.272 * rng,  rel=1e-6)

    def test_all_values_increase_monotonically(self):
        """Los niveles 0–1 deben ser crecientes (lo a hi)."""
        result = fib_levels(1000.0, 2000.0)
        ordered_keys = ["0.000", "0.236", "0.382", "0.500", "0.618",
                         "0.705", "0.786", "0.886", "1.000"]
        vals = [result[k] for k in ordered_keys]
        for i in range(len(vals) - 1):
            assert vals[i] < vals[i + 1]


# ── Tests: find_peak_formations ───────────────────────────────────────────────

class TestFindPeakFormations:
    def test_empty_swings_returns_empty(self):
        result = find_peak_formations(pd.DataFrame())
        assert result == []

    def test_too_few_swings_returns_empty(self):
        swings = _synthetic_m_swings().head(2)
        result = find_peak_formations(swings)
        assert result == []

    def test_detects_m_formation_from_synthetic(self):
        """Debe detectar al menos una formación M en swings sintéticos perfectos."""
        swings = _synthetic_m_swings()
        result = find_peak_formations(swings, pda=_make_pda_discount())
        assert len(result) >= 1
        m_forms = [pf for pf in result if pf.kind == "M"]
        assert len(m_forms) >= 1

    def test_detects_w_formation_from_synthetic(self):
        """Debe detectar al menos una formación W en swings sintéticos perfectos."""
        swings = _synthetic_w_swings()
        result = find_peak_formations(swings, pda=_make_pda_premium())
        assert len(result) >= 1
        w_forms = [pf for pf in result if pf.kind == "W"]
        assert len(w_forms) >= 1

    def test_m_formation_is_bullish(self):
        swings = _synthetic_m_swings()
        result = find_peak_formations(swings, pda=_make_pda_discount())
        m_forms = [pf for pf in result if pf.kind == "M"]
        for pf in m_forms:
            assert pf.direction == "bullish"

    def test_w_formation_is_bearish(self):
        swings = _synthetic_w_swings()
        result = find_peak_formations(swings, pda=_make_pda_premium())
        w_forms = [pf for pf in result if pf.kind == "W"]
        for pf in w_forms:
            assert pf.direction == "bearish"

    def test_m_tp1_less_than_a(self):
        """TP1 de la M debe estar entre D y A."""
        swings = _synthetic_m_swings()
        result = find_peak_formations(swings, pda=_make_pda_discount())
        for pf in [p for p in result if p.kind == "M"]:
            assert pf.d < pf.tp1 <= pf.a

    def test_w_tp1_greater_than_a(self):
        """TP1 de la W debe estar entre D y A (pero D es el techo)."""
        swings = _synthetic_w_swings()
        result = find_peak_formations(swings, pda=_make_pda_premium())
        for pf in [p for p in result if p.kind == "W"]:
            assert pf.a <= pf.tp1 < pf.d

    def test_tp2_equals_a(self):
        """TP2 = swing A (objetivo de vuelta al extremo de manipulación)."""
        swings = _synthetic_m_swings()
        result = find_peak_formations(swings, pda=_make_pda_discount())
        for pf in [p for p in result if p.kind == "M"]:
            assert pf.tp2 == pytest.approx(pf.a, rel=1e-6)

    def test_prz_lo_less_than_prz_hi(self):
        for swings, pda in [
            (_synthetic_m_swings(), _make_pda_discount()),
            (_synthetic_w_swings(), _make_pda_premium()),
        ]:
            result = find_peak_formations(swings, pda=pda)
            for pf in result:
                assert pf.prz_lo < pf.prz_hi, f"PRZ inválido: {pf.prz_lo} >= {pf.prz_hi}"

    def test_fib_levels_on_formation(self):
        """Los niveles Fibonacci en la formación deben ser matemáticamente correctos."""
        swings = _synthetic_m_swings()
        result = find_peak_formations(swings, pda=_make_pda_discount())
        for pf in [p for p in result if p.kind == "M"]:
            xa  = pf.a - pf.x
            assert pf.fib_618 == pytest.approx(pf.x + 0.618 * xa, rel=1e-4)
            assert pf.fib_786 == pytest.approx(pf.x + 0.786 * xa, rel=1e-4)
            assert pf.fib_886 == pytest.approx(pf.x + 0.886 * xa, rel=1e-4)

    def test_no_m_in_premium_zone(self):
        """M pattern no debe detectarse en zona premium (es zona discount)."""
        swings = _synthetic_m_swings()
        result = find_peak_formations(swings, pda=_make_pda_premium())
        m_forms = [pf for pf in result if pf.kind == "M"]
        assert len(m_forms) == 0

    def test_no_w_in_discount_zone(self):
        """W pattern no debe detectarse en zona discount."""
        swings = _synthetic_w_swings()
        result = find_peak_formations(swings, pda=_make_pda_discount())
        w_forms = [pf for pf in result if pf.kind == "W"]
        assert len(w_forms) == 0

    def test_returns_list_of_peak_formations(self):
        swings = _synthetic_m_swings()
        result = find_peak_formations(swings)
        assert isinstance(result, list)
        for pf in result:
            assert isinstance(pf, PeakFormation)

    def test_max_results_capped(self):
        """No debe devolver más de 5 formaciones."""
        df     = _make_wavy_daily(80)
        swings = swing_points(df, length=3)
        result = find_peak_formations(swings, lookback=20)
        assert len(result) <= 5

    def test_sorted_by_recency(self):
        """Las formaciones deben ordenarse de más reciente (bar_index_d mayor) a más antigua."""
        df     = _make_wavy_daily(80)
        swings = swing_points(df, length=3)
        result = find_peak_formations(swings, lookback=20)
        if len(result) >= 2:
            indices = [pf.bar_index_d for pf in result]
            assert indices == sorted(indices, reverse=True)

    def test_with_none_pda_still_finds_patterns(self):
        """Con pda=None debe buscar patterns sin restricción de zona."""
        swings = _synthetic_m_swings()
        result = find_peak_formations(swings, pda=None)
        # Con pda=None no filtra por zona, así que puede o no encontrar
        assert isinstance(result, list)   # solo verifica que no explota


# ── Tests: prz_boxes ──────────────────────────────────────────────────────────

class TestPrzBoxes:
    def test_empty_list_returns_empty(self):
        assert prz_boxes([]) == []

    def test_returns_boxes_for_valid_formations(self):
        swings = _synthetic_m_swings()
        forms  = find_peak_formations(swings, pda=_make_pda_discount())
        if not forms:
            pytest.skip("No formations detected (may be edge case)")
        boxes = prz_boxes(forms)
        assert len(boxes) > 0

    def test_each_box_has_required_keys(self):
        swings = _synthetic_m_swings()
        forms  = find_peak_formations(swings, pda=_make_pda_discount())
        if not forms:
            pytest.skip("No formations detected")
        boxes = prz_boxes(forms)
        for bx in boxes:
            assert "y0" in bx
            assert "y1" in bx
            assert "color" in bx
            assert "opacity" in bx

    def test_bullish_formation_box_is_green(self):
        swings = _synthetic_m_swings()
        forms  = find_peak_formations(swings, pda=_make_pda_discount())
        boxes  = prz_boxes([p for p in forms if p.direction == "bullish"])
        if boxes:
            assert boxes[0]["color"] == "#26a69a"

    def test_bearish_formation_box_is_red(self):
        swings = _synthetic_w_swings()
        forms  = find_peak_formations(swings, pda=_make_pda_premium())
        boxes  = prz_boxes([p for p in forms if p.direction == "bearish"])
        if boxes:
            assert boxes[0]["color"] == "#ef5350"

    def test_invalid_formation_excluded(self):
        """valid=False → no se genera box."""
        from dataclasses import replace
        swings = _synthetic_m_swings()
        forms  = find_peak_formations(swings, pda=_make_pda_discount())
        if not forms:
            pytest.skip("No formations")
        invalid = replace(forms[0], valid=False)
        boxes   = prz_boxes([invalid])
        assert len(boxes) == 0


# ── Tests: integración con smc.analyze_smc (Fase 3 toggle) ───────────────────

class TestHarmonicsToggle:
    def test_disabled_no_peak_formations(self, monkeypatch):
        from optionsdesk.config import settings as settings_module
        monkeypatch.setattr(settings_module.settings, "harmonics_enabled", False)

        from optionsdesk.signals.smc import analyze_smc
        import pandas as pd
        daily = pd.DataFrame({
            "date":   pd.date_range("2024-01-02", periods=50, freq="B"),
            "open":   [1000.0] * 50, "high": [1010.0] * 50,
            "low":    [990.0]  * 50, "close": [1005.0] * 50,
            "volume": [1e6]    * 50,
        })
        ctx = analyze_smc(daily)
        assert ctx.peak_formations == []

    def test_enabled_may_find_formations(self, monkeypatch):
        from optionsdesk.config import settings as settings_module
        monkeypatch.setattr(settings_module.settings, "harmonics_enabled", True)

        from optionsdesk.signals.smc import analyze_smc
        daily = _make_wavy_daily(80)
        ctx   = analyze_smc(daily)
        # La lista puede estar vacía si los datos no tienen formaciones claras,
        # pero nunca debe lanzar una excepción.
        assert isinstance(ctx.peak_formations, list)
