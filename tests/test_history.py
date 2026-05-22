"""Tests del modulo de historial del subyacente."""
from __future__ import annotations

import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from optionsdesk.data.history import UnderlyingHistory


# ── Helpers ───────────────────────────────────────────────────────────────────

def _synthetic_df(days: int = 30) -> pd.DataFrame:
    return UnderlyingHistory._synthetic_daily(days)


def _make_fresh_df(days: int = 30) -> pd.DataFrame:
    """DataFrame OHLCV con última fecha = hoy (fresco)."""
    df = _synthetic_df(days)
    # Fuerza la última fecha a hoy
    today = pd.Timestamp.today().normalize()
    dates = pd.date_range(end=today, periods=days, freq="D")
    df["date"] = dates
    return df


# ── _synthetic_daily ──────────────────────────────────────────────────────────

def test_synthetic_has_correct_columns():
    df = _synthetic_df(30)
    for col in ("date", "open", "high", "low", "close", "volume"):
        assert col in df.columns


def test_synthetic_length():
    df = _synthetic_df(60)
    assert len(df) == 60


def test_synthetic_deterministic():
    df1 = _synthetic_df(30)
    df2 = _synthetic_df(30)
    assert df1["close"].tolist() == df2["close"].tolist()


def test_synthetic_positive_prices():
    df = _synthetic_df(180)
    assert (df["close"] > 0).all()
    assert (df["high"] >= df["low"]).all()


# ── _is_fresh ─────────────────────────────────────────────────────────────────

def test_is_fresh_today():
    df = _make_fresh_df(30)
    assert UnderlyingHistory._is_fresh(df)


def test_is_fresh_stale():
    df = _synthetic_df(30)
    # Fecha vieja
    old_dates = pd.date_range(end=date(2020, 1, 1), periods=30, freq="D")
    df["date"] = old_dates
    assert not UnderlyingHistory._is_fresh(df)


def test_is_fresh_empty_returns_false():
    assert not UnderlyingHistory._is_fresh(pd.DataFrame(columns=["date"]))


# ── _normalize_daily ──────────────────────────────────────────────────────────

def test_normalize_standard_columns():
    df = pd.DataFrame({
        "date":   ["2025-01-01", "2025-01-02"],
        "open":   [100.0, 101.0],
        "high":   [102.0, 103.0],
        "low":    [99.0, 100.0],
        "close":  [101.0, 102.0],
        "volume": [1000, 1100],
    })
    result = UnderlyingHistory._normalize_daily(df)
    assert list(result.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert len(result) == 2


def test_normalize_spanish_columns():
    df = pd.DataFrame({
        "fecha":    ["2025-01-01", "2025-01-02"],
        "apertura": [100.0, 101.0],
        "maximo":   [102.0, 103.0],
        "minimo":   [99.0, 100.0],
        "cierre":   [101.0, 102.0],
        "volumen":  [1000, 1100],
    })
    result = UnderlyingHistory._normalize_daily(df)
    assert "close" in result.columns
    assert "date" in result.columns


def test_normalize_drops_na_close():
    df = pd.DataFrame({
        "date":   ["2025-01-01", "2025-01-02", "2025-01-03"],
        "open":   [100.0, None, 102.0],
        "high":   [101.0, None, 103.0],
        "low":    [99.0, None, 101.0],
        "close":  [100.0, None, 102.0],
        "volume": [1000, None, 1200],
    })
    result = UnderlyingHistory._normalize_daily(df)
    assert len(result) == 2


# ── Cache + fallback ───────────────────────────────────────────────────────────

def test_daily_fallback_when_pyobd_unavailable():
    """Si PyOBD no está disponible, debe devolver serie sintética."""
    with tempfile.TemporaryDirectory() as tmp:
        h = UnderlyingHistory(cache_dir=Path(tmp))
        with patch.dict("sys.modules", {"pyobd": None}):
            df = h.daily("GGAL", days=30)
        assert not df.empty
        assert "close" in df.columns
        assert len(df) <= 30


def test_daily_uses_cache_when_fresh():
    """Si hay caché fresco, lo usa sin llamar a PyOBD."""
    with tempfile.TemporaryDirectory() as tmp:
        h = UnderlyingHistory(cache_dir=Path(tmp))

        # Guarda un caché fresco
        fresh_df = _make_fresh_df(30)
        cache_path = Path(tmp) / "GGAL_daily.parquet"
        fresh_df.to_parquet(cache_path, index=False)

        # PyOBD no debería ser llamado
        mock_byma = MagicMock()
        mock_byma.get_daily_history.side_effect = AssertionError("No deberia llamarse")
        mock_module = MagicMock()
        mock_module.BymaData.return_value = mock_byma

        with patch.dict("sys.modules", {"pyobd": mock_module}):
            df = h.daily("GGAL", days=30)

        assert not df.empty


def test_daily_output_columns():
    with tempfile.TemporaryDirectory() as tmp:
        h = UnderlyingHistory(cache_dir=Path(tmp))
        with patch.dict("sys.modules", {"pyobd": None}):
            df = h.daily("GGAL", days=20)
        assert set(["date", "open", "high", "low", "close", "volume"]).issubset(df.columns)


def test_daily_sorted_ascending():
    with tempfile.TemporaryDirectory() as tmp:
        h = UnderlyingHistory(cache_dir=Path(tmp))
        with patch.dict("sys.modules", {"pyobd": None}):
            df = h.daily("GGAL", days=50)
        dates = pd.to_datetime(df["date"])
        assert (dates.diff().dropna() >= pd.Timedelta(0)).all()


def test_intraday_returns_dataframe_on_error():
    with tempfile.TemporaryDirectory() as tmp:
        h = UnderlyingHistory(cache_dir=Path(tmp))
        with patch.dict("sys.modules", {"pyobd": None}):
            df = h.intraday("GGAL")
        assert isinstance(df, pd.DataFrame)


# ── weekly() ─────────────────────────────────────────────────────────────────

def test_weekly_returns_dataframe():
    """weekly() sobre datos sinteticos debe devolver un DataFrame no vacio."""
    with tempfile.TemporaryDirectory() as tmp:
        h = UnderlyingHistory(cache_dir=Path(tmp))
        with patch.object(h, "daily", return_value=_make_fresh_df(180)):
            df = h.weekly("GGAL", weeks=26)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_weekly_has_ohlcv_columns():
    with tempfile.TemporaryDirectory() as tmp:
        h = UnderlyingHistory(cache_dir=Path(tmp))
        with patch.object(h, "daily", return_value=_make_fresh_df(180)):
            df = h.weekly("GGAL", weeks=26)
    for col in ("date", "open", "high", "low", "close", "volume"):
        assert col in df.columns


def test_weekly_fewer_rows_than_daily():
    """El resampleo semanal debe tener menos filas que el diario."""
    with tempfile.TemporaryDirectory() as tmp:
        h = UnderlyingHistory(cache_dir=Path(tmp))
        daily = _make_fresh_df(180)
        with patch.object(h, "daily", return_value=daily):
            weekly = h.weekly("GGAL", weeks=52)
    assert len(weekly) < len(daily)


def test_weekly_high_ge_low():
    """El high semanal debe ser siempre >= low semanal."""
    with tempfile.TemporaryDirectory() as tmp:
        h = UnderlyingHistory(cache_dir=Path(tmp))
        with patch.object(h, "daily", return_value=_make_fresh_df(365)):
            df = h.weekly("GGAL", weeks=52)
    assert (df["high"] >= df["low"]).all()


def test_weekly_limits_to_requested_weeks():
    with tempfile.TemporaryDirectory() as tmp:
        h = UnderlyingHistory(cache_dir=Path(tmp))
        with patch.object(h, "daily", return_value=_make_fresh_df(365)):
            df = h.weekly("GGAL", weeks=10)
    assert len(df) <= 10


def test_resample_static():
    """_resample sobre un DataFrame de 14 dias debe dar <=2 velas semanales."""
    df = _make_fresh_df(14)
    out = UnderlyingHistory._resample(df, "W-FRI")
    assert len(out) <= 2
    assert "close" in out.columns
