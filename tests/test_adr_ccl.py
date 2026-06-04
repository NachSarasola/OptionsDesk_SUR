"""Tests para optionsdesk/data/providers/adr.py

Sin red: todas las llamadas HTTP están mockeadas con monkeypatch.
El CSV fixture reproduce exactamente el formato que devuelve stooq /q/l/.
"""
from __future__ import annotations

import io
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from optionsdesk.data.providers.adr import (
    AdrQuote,
    _parse_stooq_quote_csv,
    adr_confluence,
    implied_ccl,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

_STOOQ_CSV_GGAL = (
    "Symbol,Date,Time,Open,High,Low,Close,Volume\n"
    "GGAL.US,2024-06-03,22:00:18,49.62,49.99,48.28,48.31,956372\n"
)

_STOOQ_CSV_YPF = (
    "Symbol,Date,Time,Open,High,Low,Close,Volume\n"
    "YPF.US,2024-06-03,22:00:18,55.95,56.08,54.35,54.93,2236744\n"
)

_STOOQ_CSV_EMPTY = "Symbol,Date,Time,Open,High,Low,Close,Volume\n"

_STOOQ_CSV_MALFORMED = "this is not valid csv at all\n"


def _make_adr_df(closes: list[float]) -> pd.DataFrame:
    """Construye un DataFrame de historial ADR con cierres dados."""
    dates = pd.date_range("2024-01-02", periods=len(closes), freq="B")
    return pd.DataFrame({
        "date":   dates,
        "open":   [c * 0.99 for c in closes],
        "high":   [c * 1.01 for c in closes],
        "low":    [c * 0.98 for c in closes],
        "close":  closes,
        "volume": [1_000_000] * len(closes),
    })


# ── Tests: _parse_stooq_quote_csv ─────────────────────────────────────────────

class TestParseStooqQuoteCsv:
    def test_parses_ggal_correctly(self):
        q = _parse_stooq_quote_csv("GGAL.US", _STOOQ_CSV_GGAL)
        assert q is not None
        assert q.ticker == "GGAL.US"
        assert q.date   == "2024-06-03"
        assert q.open   == pytest.approx(49.62)
        assert q.high   == pytest.approx(49.99)
        assert q.low    == pytest.approx(48.28)
        assert q.close  == pytest.approx(48.31)
        assert q.volume == 956372

    def test_parses_ypf_correctly(self):
        q = _parse_stooq_quote_csv("YPF.US", _STOOQ_CSV_YPF)
        assert q is not None
        assert q.close == pytest.approx(54.93)

    def test_empty_csv_returns_none(self):
        q = _parse_stooq_quote_csv("GGAL.US", _STOOQ_CSV_EMPTY)
        assert q is None

    def test_malformed_csv_returns_none(self):
        q = _parse_stooq_quote_csv("GGAL.US", _STOOQ_CSV_MALFORMED)
        assert q is None

    def test_returns_adr_quote_instance(self):
        q = _parse_stooq_quote_csv("GGAL.US", _STOOQ_CSV_GGAL)
        assert isinstance(q, AdrQuote)


# ── Tests: implied_ccl ────────────────────────────────────────────────────────

class TestImpliedCcl:
    def test_ggal_example(self):
        """GGAL ratio=10: 10 acciones locales = 1 ADS.
        Si GGAL local = $4831 ARS y GGAL.US = 48.31 USD → CCL implícito = 10.0
        """
        local_ars = 4831.0
        adr_usd   = 48.31
        ratio     = 10.0
        ccl = implied_ccl(local_ars, adr_usd, ratio)
        assert ccl == pytest.approx(10.0, rel=1e-3)

    def test_zero_adr_returns_zero(self):
        assert implied_ccl(5000.0, 0.0, 10.0) == 0.0

    def test_zero_ratio_returns_zero(self):
        assert implied_ccl(5000.0, 50.0, 0.0) == 0.0

    def test_zero_local_returns_zero(self):
        assert implied_ccl(0.0, 50.0, 10.0) == 0.0

    def test_result_is_positive(self):
        result = implied_ccl(10000.0, 50.0, 5.0)
        assert result > 0

    def test_scales_linearly_with_local(self):
        """Duplicar el precio local duplica el CCL implícito."""
        ccl1 = implied_ccl(5000.0, 50.0, 10.0)
        ccl2 = implied_ccl(10000.0, 50.0, 10.0)
        assert ccl2 == pytest.approx(ccl1 * 2, rel=1e-6)

    def test_scales_inversely_with_adr(self):
        """Duplicar el ADR reduce el CCL a la mitad."""
        ccl1 = implied_ccl(5000.0, 50.0, 10.0)
        ccl2 = implied_ccl(5000.0, 100.0, 10.0)
        assert ccl2 == pytest.approx(ccl1 / 2, rel=1e-6)


# ── Tests: adr_confluence ─────────────────────────────────────────────────────

class TestAdrConfluence:
    def test_unknown_when_no_break_dir(self):
        df = _make_adr_df([50.0] * 15)
        assert adr_confluence(None, df) == "unknown"

    def test_unknown_when_empty_df(self):
        assert adr_confluence("up", pd.DataFrame()) == "unknown"

    def test_unknown_when_insufficient_history(self):
        """Con menos de lookback+1 filas devuelve 'unknown'."""
        df = _make_adr_df([50.0] * 4)   # solo 4 filas, lookback default=5
        assert adr_confluence("up", df) == "unknown"

    def test_confirmed_bullish_break_adr_up(self):
        """ADR sube > 0.8% mientras local tiene BOS_UP → confirmed."""
        base   = 50.0
        # ADR subió 3% en los últimos 5 días
        closes = [base * (1 + 0.006 * i) for i in range(10)]
        df     = _make_adr_df(closes)
        result = adr_confluence("up", df, lookback=5)
        assert result == "confirmed"

    def test_fx_driven_bullish_break_adr_down(self):
        """ADR baja mientras local tiene BOS_UP → fx_driven (el peso subió por CCL)."""
        base   = 50.0
        # ADR bajó 2% en los últimos 5 días
        closes = [base * (1 - 0.004 * i) for i in range(10)]
        df     = _make_adr_df(closes)
        result = adr_confluence("up", df, lookback=5)
        assert result == "fx_driven"

    def test_confirmed_bearish_break_adr_down(self):
        """ADR baja > 0.8% mientras local tiene BOS_DOWN → confirmed."""
        base   = 50.0
        closes = [base * (1 - 0.005 * i) for i in range(10)]
        df     = _make_adr_df(closes)
        result = adr_confluence("down", df, lookback=5)
        assert result == "confirmed"

    def test_fx_driven_bearish_break_adr_up(self):
        """ADR sube mientras local tiene BOS_DOWN → fx_driven."""
        base   = 50.0
        closes = [base * (1 + 0.003 * i) for i in range(10)]
        df     = _make_adr_df(closes)
        result = adr_confluence("down", df, lookback=5)
        assert result == "fx_driven"

    def test_unknown_small_movement(self):
        """ADR se mueve muy poco (< umbral) → unknown (inconcluso)."""
        base   = 50.0
        # Variación dentro del 0.3% = inconclusa
        closes = [base + 0.001 * i for i in range(10)]
        df     = _make_adr_df(closes)
        result = adr_confluence("up", df, lookback=5)
        assert result == "unknown"

    def test_result_is_string(self):
        df = _make_adr_df([50.0] * 10)
        result = adr_confluence("up", df)
        assert isinstance(result, str)
        assert result in ("confirmed", "fx_driven", "unknown")

    def test_consistent_with_lookback(self):
        """lookback=2: debe usar solo las 2 últimas barras vs. la anterior."""
        closes = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 55.0]  # sube 10% en la última
        df     = _make_adr_df(closes)
        result = adr_confluence("up", df, lookback=2)
        assert result == "confirmed"   # 55 vs 50 = +10% >> umbral


# ── Tests: adr_quote (mockeado) ───────────────────────────────────────────────

class TestAdrQuoteMocked:
    def test_adr_quote_parses_response(self, monkeypatch):
        """adr_quote() parsea la respuesta HTTP y devuelve AdrQuote."""
        mock_resp = MagicMock()
        mock_resp.text = _STOOQ_CSV_GGAL
        mock_resp.raise_for_status = MagicMock()

        mock_requests = MagicMock()
        mock_requests.get.return_value = mock_resp

        import optionsdesk.data.providers.adr as adr_mod
        monkeypatch.setattr(adr_mod, "_requests", mock_requests)

        result = adr_mod.adr_quote("GGAL.US")
        assert result is not None
        assert result.close == pytest.approx(48.31)

    def test_adr_quote_returns_none_on_error(self, monkeypatch):
        """adr_quote() devuelve None si hay error de red."""
        mock_requests = MagicMock()
        mock_requests.get.side_effect = Exception("timeout")

        import optionsdesk.data.providers.adr as adr_mod
        monkeypatch.setattr(adr_mod, "_requests", mock_requests)

        result = adr_mod.adr_quote("GGAL.US")
        assert result is None


# ── Tests: toggle desactivado ──────────────────────────────────────────────────

class TestToggleDisabled:
    def test_adr_disabled_smc_context_no_confluence(self, monkeypatch):
        """Con ADR_CCL_ENABLED=False, smc_ctx.adr_confluence queda None."""
        from optionsdesk.config import settings as settings_module
        monkeypatch.setattr(settings_module.settings, "adr_ccl_enabled", False)

        from optionsdesk.signals.smc import analyze_smc, SmcContext
        import pandas as pd

        daily = pd.DataFrame({
            "date":   pd.date_range("2024-01-02", periods=20, freq="B"),
            "open":   [1000.0] * 20,
            "high":   [1010.0] * 20,
            "low":    [990.0]  * 20,
            "close":  [1005.0] * 20,
            "volume": [1e6]    * 20,
        })
        ctx = analyze_smc(daily)
        assert ctx.adr_confluence is None
