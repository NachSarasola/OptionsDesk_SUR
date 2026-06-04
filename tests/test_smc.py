"""Tests para optionsdesk/signals/smc.py

Principios:
- Datos sintéticos determinísticos (sin red, sin dependencias externas).
- Los swings se verifican sin lookahead: el pivote en i solo puede estar
  confirmado si hay ≥ length barras DESPUÉS en el df.
- Matemática de Fibonacci verificable con assert exacto (tolerancia 1e-6).
- Killzones: bordes 11:00/12:00/15:30/17:00 ART con zona horaria explícita.
"""
from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from optionsdesk.signals.smc import (
    LiquidityLevel,
    OteZone,
    PdaArray,
    SmcContext,
    analyze_smc,
    current_killzone,
    detect_sweeps,
    liquidity_map,
    ote_zone,
    pda_array,
    swing_points,
    mm_read,
)

_BA = ZoneInfo("America/Argentina/Buenos_Aires")


# ── Fixtures de datos ─────────────────────────────────────────────────────────

def _make_daily(n: int = 50, base: float = 1000.0, step: float = 10.0) -> pd.DataFrame:
    """OHLCV diario sintético ascendente con swings mínimos."""
    dates  = pd.date_range("2024-01-02", periods=n, freq="B")
    close  = [base + i * step for i in range(n)]
    high   = [c * 1.01 for c in close]
    low    = [c * 0.99 for c in close]
    open_  = [c * 0.995 for c in close]
    vol    = [1_000_000.0] * n
    return pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low,
                          "close": close, "volume": vol})


def _make_wavy(n: int = 60) -> pd.DataFrame:
    """OHLCV con ondas de swing claras (impulso-retroceso-impulso)."""
    import math
    dates  = pd.date_range("2024-01-02", periods=n, freq="B")
    # Precio en onda sinusoidal alrededor de una tendencia alcista
    close  = [1000.0 + i * 5.0 + 100.0 * math.sin(i * 0.3) for i in range(n)]
    high   = [c * 1.012 for c in close]
    low    = [c * 0.988 for c in close]
    open_  = [c * 0.997 for c in close]
    vol    = [1_500_000.0] * n
    return pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low,
                          "close": close, "volume": vol})


def _make_weekly(n: int = 20, base: float = 1000.0, step: float = 50.0) -> pd.DataFrame:
    """OHLCV semanal sintético."""
    dates = pd.date_range("2024-01-05", periods=n, freq="W-FRI")
    close = [base + i * step for i in range(n)]
    high  = [c * 1.02 for c in close]
    low   = [c * 0.98 for c in close]
    open_ = [c * 0.995 for c in close]
    vol   = [5_000_000.0] * n
    return pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low,
                          "close": close, "volume": vol})


# ── Tests: swing_points ───────────────────────────────────────────────────────

class TestSwingPoints:
    def test_empty_returns_empty(self):
        result = swing_points(pd.DataFrame(), length=3)
        assert result.empty

    def test_too_short_returns_empty(self):
        df = _make_daily(5)
        result = swing_points(df, length=3)
        # Necesita 2*3+1=7 barras mínimo
        assert result.empty

    def test_detects_high_and_low(self):
        """Con datos ondulados debe encontrar al menos un swing H y un swing L."""
        df = _make_wavy(60)
        result = swing_points(df, length=3)
        assert not result.empty
        assert 1  in result["HighLow"].values
        assert -1 in result["HighLow"].values

    def test_no_lookahead_last_bars(self):
        """Los swings en las últimas `length` barras NO deben aparecer en el resultado.

        Un pivot en la barra i solo es confirmado cuando existen `length` barras
        posteriores con valores menores/mayores. Por lo tanto, ningún swing
        puede estar en las últimas `length` posiciones del df.
        """
        length = 3
        df     = _make_wavy(60)
        result = swing_points(df, length=length)
        n      = len(df)
        if not result.empty:
            max_idx = int(result["index"].max())
            # El último swing confirmado debe estar al menos `length` antes del final
            assert max_idx <= n - length - 1, (
                f"Swing en index {max_idx} debería estar <= {n - length - 1}"
            )

    def test_columns_present(self):
        df = _make_wavy(40)
        result = swing_points(df, length=3)
        for col in ("index", "HighLow", "Level"):
            assert col in result.columns

    def test_date_column_included(self):
        df = _make_wavy(40)
        result = swing_points(df, length=3)
        if not result.empty:
            assert "date" in result.columns

    def test_swing_high_above_neighbors(self):
        """Cada swing high debe tener un high mayor que sus vecinos."""
        df     = _make_wavy(60)
        result = swing_points(df, length=3)
        highs  = df["high"].values
        for _, row in result[result["HighLow"] == 1].iterrows():
            i = int(row["index"])
            assert highs[i] == pytest.approx(row["Level"], rel=1e-6)
            # Verificar que sea mayor que los length vecinos de cada lado
            assert highs[i] > max(highs[max(0, i-3):i])

    def test_swing_low_below_neighbors(self):
        df   = _make_wavy(60)
        result = swing_points(df, length=3)
        lows = df["low"].values
        for _, row in result[result["HighLow"] == -1].iterrows():
            i = int(row["index"])
            assert lows[i] == pytest.approx(row["Level"], rel=1e-6)
            assert lows[i] < min(lows[max(0, i-3):i])


# ── Tests: liquidity_map ──────────────────────────────────────────────────────

class TestLiquidityMap:
    def test_empty_df_returns_empty(self):
        result = liquidity_map(pd.DataFrame(), spot=1000.0)
        assert result == []

    def test_pdh_pdl_extracted(self):
        """PDH y PDL deben extraerse de la penúltima vela (día previo confirmado)."""
        df  = _make_daily(10)
        res = liquidity_map(df, spot=float(df["close"].iloc[-1]))
        kinds  = {lv.label for lv in res}
        assert "PDH" in kinds
        assert "PDL" in kinds

    def test_pwh_pwl_with_weekly(self):
        daily  = _make_daily(30)
        weekly = _make_weekly(10)
        spot   = float(daily["close"].iloc[-1])
        res    = liquidity_map(daily, weekly=weekly, spot=spot)
        labels = {lv.label for lv in res}
        assert "PWH" in labels
        assert "PWL" in labels

    def test_bsl_above_price(self):
        """BSL no barrido debe estar por encima del precio actual."""
        df   = _make_daily(30)
        spot = float(df["close"].iloc[-2]) * 0.5   # precio bajo para que todos sean BSL no barridos
        res  = liquidity_map(df, spot=spot)
        bsl_levels = [lv for lv in res if lv.kind == "BSL" and not lv.swept]
        assert all(lv.price > spot for lv in bsl_levels)

    def test_ssl_below_price(self):
        """SSL no barrido debe estar por debajo del precio actual."""
        df   = _make_daily(30)
        spot = float(df["close"].iloc[-2]) * 2.0   # precio alto para que todos sean SSL no barridos
        res  = liquidity_map(df, spot=spot)
        ssl_levels = [lv for lv in res if lv.kind == "SSL" and not lv.swept]
        assert all(lv.price < spot for lv in ssl_levels)

    def test_pdh_is_prior_bar_high(self):
        df   = _make_daily(10)
        spot = float(df["close"].iloc[-1])
        res  = liquidity_map(df, spot=spot)
        pdh_levels = [lv for lv in res if lv.label == "PDH"]
        assert len(pdh_levels) >= 1
        expected_pdh = float(df["high"].iloc[-2])
        assert pdh_levels[0].price == pytest.approx(expected_pdh, rel=1e-6)

    def test_pdl_is_prior_bar_low(self):
        df   = _make_daily(10)
        spot = float(df["close"].iloc[-1])
        res  = liquidity_map(df, spot=spot)
        pdl_levels = [lv for lv in res if lv.label == "PDL"]
        assert len(pdl_levels) >= 1
        expected_pdl = float(df["low"].iloc[-2])
        assert pdl_levels[0].price == pytest.approx(expected_pdl, rel=1e-6)

    def test_no_duplicates_within_tolerance(self):
        """No debe haber dos niveles del mismo kind dentro del 0.3% de precio."""
        df  = _make_daily(30)
        res = liquidity_map(df, spot=float(df["close"].iloc[-1]))
        for kind in ("BSL", "SSL"):
            same_kind = [lv.price for lv in res if lv.kind == kind]
            for i, p1 in enumerate(same_kind):
                for p2 in same_kind[i+1:]:
                    dist = abs(p1 - p2) / max(p2, 1e-9)
                    assert dist >= 0.003, f"Niveles {kind} duplicados: {p1:.0f} vs {p2:.0f}"


# ── Tests: pda_array ──────────────────────────────────────────────────────────

class TestPdaArray:
    def test_returns_pda_array(self):
        df  = _make_wavy(40)
        res = pda_array(df, spot=float(df["close"].iloc[-1]))
        assert res is not None
        assert isinstance(res, PdaArray)

    def test_zone_premium(self):
        """Spot sobre el 0.618 → zona premium."""
        df  = _make_daily(30)
        sh  = float(df["high"].max())
        sl  = float(df["low"].min())
        eq  = sl + 0.5   * (sh - sl)
        pb  = sl + 0.618 * (sh - sl)
        spot_premium = pb + (sh - pb) * 0.5   # bien dentro de la zona premium
        res = pda_array(df, spot=spot_premium)
        assert res is not None
        assert res.zone == "premium"

    def test_zone_discount(self):
        """Spot bajo el 0.382 → zona discount."""
        df  = _make_daily(30)
        sh  = float(df["high"].max())
        sl  = float(df["low"].min())
        db  = sl + 0.382 * (sh - sl)
        spot_discount = sl + (db - sl) * 0.3  # bien dentro de discount
        res = pda_array(df, spot=spot_discount)
        assert res is not None
        assert res.zone == "discount"

    def test_zone_equilibrium(self):
        """Spot en el 0.5 → equilibrium."""
        df  = _make_daily(30)
        sh  = float(df["high"].max())
        sl  = float(df["low"].min())
        eq  = sl + 0.5 * (sh - sl)
        res = pda_array(df, spot=eq)
        assert res is not None
        assert res.zone == "equilibrium"

    def test_fib_math(self):
        """Verificación matemática de los niveles de Fibonacci."""
        df  = _make_daily(30)
        sh  = float(df["high"].max())
        sl  = float(df["low"].min())
        rng = sh - sl
        res = pda_array(df, spot=sh)  # cualquier spot
        assert res is not None
        assert res.equilibrium    == pytest.approx(sl + 0.500 * rng, rel=1e-5)
        assert res.premium_band   == pytest.approx(sl + 0.618 * rng, rel=1e-5)
        assert res.discount_band  == pytest.approx(sl + 0.382 * rng, rel=1e-5)
        assert res.ote_bullish_lo == pytest.approx(sl + 0.618 * rng, rel=1e-5)
        assert res.ote_bullish_hi == pytest.approx(sl + 0.786 * rng, rel=1e-5)
        assert res.ote_bearish_lo == pytest.approx(sl + 0.786 * rng, rel=1e-5)
        assert res.ote_bearish_hi == pytest.approx(sl + 0.886 * rng, rel=1e-5)

    def test_none_for_insufficient_data(self):
        df  = _make_daily(5)
        res = pda_array(df, spot=1000.0)
        assert res is None


# ── Tests: ote_zone ───────────────────────────────────────────────────────────

class TestOteZone:
    def test_bullish_zone_correct(self):
        """OTE bullish: zona 0.618–0.786 del swing XA."""
        sl, sh = 1000.0, 1500.0
        rng    = sh - sl
        res    = ote_zone(sl, sh, "bullish")
        assert res is not None
        assert res.direction == "bullish"
        assert res.lo    == pytest.approx(sl + 0.618 * rng, rel=1e-6)
        assert res.hi    == pytest.approx(sl + 0.786 * rng, rel=1e-6)
        assert res.fib_618 == pytest.approx(sl + 0.618 * rng, rel=1e-6)
        assert res.fib_705 == pytest.approx(sl + 0.705 * rng, rel=1e-6)
        assert res.fib_786 == pytest.approx(sl + 0.786 * rng, rel=1e-6)
        assert res.fib_886 == pytest.approx(sl + 0.886 * rng, rel=1e-6)

    def test_bearish_zone_correct(self):
        """OTE bearish: retroceso desde el techo 0.618–0.786."""
        sl, sh = 1000.0, 1500.0
        rng    = sh - sl
        res    = ote_zone(sl, sh, "bearish")
        assert res is not None
        assert res.direction == "bearish"
        assert res.lo    == pytest.approx(sh - 0.786 * rng, rel=1e-6)
        assert res.hi    == pytest.approx(sh - 0.618 * rng, rel=1e-6)
        assert res.fib_618 == pytest.approx(sh - 0.618 * rng, rel=1e-6)
        assert res.fib_786 == pytest.approx(sh - 0.786 * rng, rel=1e-6)
        assert res.fib_886 == pytest.approx(sh - 0.886 * rng, rel=1e-6)

    def test_lo_less_than_hi(self):
        for direction in ("bullish", "bearish"):
            res = ote_zone(800.0, 1200.0, direction)
            assert res is not None
            assert res.lo < res.hi

    def test_none_for_zero_range(self):
        assert ote_zone(1000.0, 1000.0, "bullish") is None

    def test_none_for_inverted_range(self):
        assert ote_zone(1200.0, 800.0, "bullish") is None


# ── Tests: detect_sweeps ──────────────────────────────────────────────────────

class TestDetectSweeps:
    def _make_sweep_df(self, high_poke: float, close_back: float) -> pd.DataFrame:
        """DataFrame donde la última vela sube por encima de `high_poke` pero cierra en `close_back`."""
        rows = [
            {"date": "2024-01-02", "open": 990, "high": 1005, "low": 985, "close": 1000, "volume": 1e6},
            {"date": "2024-01-03", "open": 998, "high": 1008, "low": 993, "close": 1003, "volume": 1e6},
            {"date": "2024-01-04", "open": 1001, "high": 1010, "low": 997, "close": 1005, "volume": 1e6},
            {"date": "2024-01-05", "open": 1003, "high": high_poke, "low": 995, "close": close_back, "volume": 1e6},
        ]
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def test_sweep_bsl_with_reversal(self):
        """High por encima de BSL + close por debajo → sweep con reversed=True."""
        bsl_level = LiquidityLevel("BSL", 1015.0, "SWING_H", swept=False, timeframe="daily")
        df        = self._make_sweep_df(high_poke=1020.0, close_back=1010.0)
        sweeps    = detect_sweeps(df, [bsl_level], lookback=3)
        reversed_sweeps = [sw for sw in sweeps if sw.reversed]
        assert len(reversed_sweeps) >= 1

    def test_sweep_bsl_no_reversal(self):
        """High por encima de BSL + close también arriba → reversed=False."""
        bsl_level = LiquidityLevel("BSL", 1015.0, "SWING_H", swept=False, timeframe="daily")
        df        = self._make_sweep_df(high_poke=1020.0, close_back=1018.0)
        sweeps    = detect_sweeps(df, [bsl_level], lookback=3)
        reversed_sweeps = [sw for sw in sweeps if sw.reversed]
        assert len(reversed_sweeps) == 0

    def test_no_sweep_when_price_does_not_cross(self):
        """Si el high no supera el BSL, no hay sweep."""
        bsl_level = LiquidityLevel("BSL", 1030.0, "SWING_H", swept=False, timeframe="daily")
        df        = self._make_sweep_df(high_poke=1020.0, close_back=1018.0)
        sweeps    = detect_sweeps(df, [bsl_level], lookback=3)
        assert len(sweeps) == 0

    def test_empty_df_returns_empty(self):
        lv = LiquidityLevel("BSL", 1000.0, "PDH", swept=False, timeframe="daily")
        assert detect_sweeps(pd.DataFrame(), [lv]) == []


# ── Tests: current_killzone ───────────────────────────────────────────────────

class TestCurrentKillzone:
    def _ts(self, h: int, m: int, wd: int = 0) -> datetime:
        """Crea un timestamp en Buenos Aires para el día de la semana `wd` (0=Lunes)."""
        # Usamos una fecha conocida: 2024-01-08 es Lunes (wd=0)
        base = datetime(2024, 1, 8 + wd, h, m, 0, tzinfo=_BA)
        return base

    def test_manipulacion_start(self):
        assert current_killzone(self._ts(11, 0)) == "manipulacion"

    def test_manipulacion_mid(self):
        assert current_killzone(self._ts(11, 30)) == "manipulacion"

    def test_manipulacion_end_exclusive(self):
        """12:00 ya no es manipulación."""
        assert current_killzone(self._ts(12, 0)) is None

    def test_desarrollo_inside(self):
        assert current_killzone(self._ts(13, 0)) is None
        assert current_killzone(self._ts(15, 0)) is None

    def test_distribucion_start(self):
        assert current_killzone(self._ts(15, 30)) == "distribucion"

    def test_distribucion_mid(self):
        assert current_killzone(self._ts(16, 0)) == "distribucion"

    def test_distribucion_end_exclusive(self):
        """17:00 ya no es distribución."""
        assert current_killzone(self._ts(17, 0)) is None

    def test_before_open(self):
        assert current_killzone(self._ts(10, 0)) is None

    def test_after_close(self):
        assert current_killzone(self._ts(18, 0)) is None

    def test_weekend_saturday(self):
        assert current_killzone(self._ts(11, 30, wd=5)) is None  # Sábado

    def test_weekend_sunday(self):
        assert current_killzone(self._ts(11, 30, wd=6)) is None  # Domingo

    def test_naive_datetime_treated_as_ba(self):
        """Un datetime sin tz se trata como Buenos Aires."""
        naive = datetime(2024, 1, 8, 11, 30)  # Lunes 11:30, naive
        result = current_killzone(naive)
        assert result == "manipulacion"


# ── Tests: mm_read ────────────────────────────────────────────────────────────

class TestMmRead:
    def test_returns_none_for_insufficient_data(self):
        df = _make_daily(50)   # necesita al menos EMA200 = 200 barras
        res = mm_read(df)
        assert res is None

    def test_returns_mm_read_with_enough_data(self):
        df  = _make_daily(210)
        res = mm_read(df)
        assert res is not None

    def test_bullish_stack_ascending_close(self):
        """Serie ascendente larga → stack bullish."""
        df  = _make_daily(220, step=5.0)
        res = mm_read(df)
        assert res is not None
        assert res.stack == "bullish"

    def test_ema_values_are_close_to_last_price(self):
        """En serie suave, todas las EMAs deben estar cerca del último cierre."""
        df    = _make_daily(210, step=1.0)
        res   = mm_read(df)
        assert res is not None
        last  = float(df["close"].iloc[-1])
        for val in (res.ema5, res.ema13, res.ema50, res.ema200):
            assert abs(val - last) / last < 0.10, f"EMA {val} muy lejos del close {last}"

    def test_level_count_in_range(self):
        df  = _make_daily(210)
        res = mm_read(df)
        assert res is not None
        assert 0 <= res.level_count <= 3


# ── Tests: analyze_smc ────────────────────────────────────────────────────────

class TestAnalyzeSmc:
    def test_returns_smc_context(self):
        df  = _make_wavy(50)
        res = analyze_smc(df)
        assert isinstance(res, SmcContext)

    def test_no_exception_on_empty_df(self):
        res = analyze_smc(pd.DataFrame())
        assert isinstance(res, SmcContext)

    def test_liquidity_populated(self):
        df  = _make_daily(30)
        res = analyze_smc(df, spot=float(df["close"].iloc[-1]))
        assert len(res.liquidity) > 0

    def test_pda_populated(self):
        df  = _make_wavy(40)
        res = analyze_smc(df, spot=float(df["close"].iloc[-1]))
        assert res.pda is not None

    def test_ote_populated_with_wavy(self):
        df  = _make_wavy(50)
        res = analyze_smc(df, spot=float(df["close"].iloc[-1]))
        # Con datos ondulados deberíamos tener swings y OTE
        assert res.swings is not None

    def test_adr_hook_passed_through(self):
        df  = _make_daily(20)
        res = analyze_smc(df, adr={"confluence": "confirmed"})
        assert res.adr_confluence == "confirmed"

    def test_adr_hook_fx_driven(self):
        df  = _make_daily(20)
        res = analyze_smc(df, adr={"confluence": "fx_driven"})
        assert res.adr_confluence == "fx_driven"

    def test_bos_from_snap(self):
        """Los campos BOS/CHoCH se toman del snap recibido, no se recalculan."""
        class FakeSnap:
            bos   = "BOS_UP"
            choch = None

        df  = _make_daily(20)
        res = analyze_smc(df, snap=FakeSnap())
        assert res.bos   == "BOS_UP"
        assert res.choch is None

    def test_no_exception_with_weekly(self):
        daily  = _make_daily(40)
        weekly = _make_weekly(15)
        res    = analyze_smc(daily, weekly=weekly, spot=float(daily["close"].iloc[-1]))
        assert isinstance(res, SmcContext)

    def test_killzone_field_present(self):
        df  = _make_daily(20)
        # Pasamos un 'now' fuera de horario para que killzone sea None
        now = datetime(2024, 1, 8, 22, 0, tzinfo=_BA)
        res = analyze_smc(df, now=now)
        assert res.killzone is None

    def test_killzone_manipulacion_at_1130(self):
        df  = _make_daily(20)
        now = datetime(2024, 1, 8, 11, 30, tzinfo=_BA)
        res = analyze_smc(df, now=now)
        assert res.killzone == "manipulacion"


# ── Tests: toggle desactivado no rompe nada ───────────────────────────────────

class TestToggleDisabled:
    def test_smc_disabled_directional_still_works(self, monkeypatch):
        """Con SMC_ENABLED=False, compute_market_context no debe lanzar excepción."""
        from optionsdesk.config import settings as settings_module
        monkeypatch.setattr(settings_module.settings, "smc_enabled", False)

        from optionsdesk.signals.directional import compute_market_context
        df  = _make_daily(30)
        ctx = compute_market_context(df)
        # El campo smc debe quedar en None (no se calculó)
        assert getattr(ctx, "smc", None) is None
