"""Tests de volume_momentum.py — detectar la ola temprano en BYMA."""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from optionsdesk.signals.volume_momentum import (
    VolumeSnapshot,
    _compute_obv,
    _obv_analysis,
    volume_score_delta,
    volume_snapshot,
)


def _daily(closes: list[float], vols: list[float]) -> pd.DataFrame:
    start = datetime(2026, 1, 2)
    rows = [
        {"date": start + timedelta(days=i), "open": c - 0.3, "high": c + 0.5,
         "low": c - 0.5, "close": c, "volume": v}
        for i, (c, v) in enumerate(zip(closes, vols))
    ]
    return pd.DataFrame(rows)


def _flat_closes(n: int = 25, base: float = 100.0) -> list[float]:
    return [base + (i % 3) * 0.1 for i in range(n)]


# ── RVOL ──────────────────────────────────────────────────────────────────────

def test_rvol_explosive_on_volume_spike():
    closes = _flat_closes(25)
    vols = [1000.0] * 24 + [3000.0]  # último día: 3x el promedio
    snap = volume_snapshot(_daily(closes, vols))
    assert snap is not None
    assert snap.rvol is not None and snap.rvol >= 2.5
    assert snap.rvol_label == "explosivo"


def test_rvol_high_on_moderate_spike():
    closes = _flat_closes(25)
    vols = [1000.0] * 24 + [1800.0]  # 1.8x
    snap = volume_snapshot(_daily(closes, vols))
    assert snap is not None
    assert snap.rvol_label == "alto"


def test_rvol_low_on_thin_volume():
    closes = _flat_closes(25)
    vols = [1000.0] * 24 + [600.0]   # 0.6x — sin convicción
    snap = volume_snapshot(_daily(closes, vols))
    assert snap is not None
    assert snap.rvol_label == "bajo"


def test_zero_volume_returns_none():
    closes = _flat_closes(25)
    vols = [0.0] * 25
    snap = volume_snapshot(_daily(closes, vols))
    assert snap is None, "tape intradía de BYMA (todo cero) no debe dar señal"


def test_none_without_enough_data():
    snap = volume_snapshot(None)
    assert snap is None
    snap2 = volume_snapshot(_daily(_flat_closes(3), [1000.0] * 3))
    assert snap2 is None


# ── OBV divergencias ──────────────────────────────────────────────────────────

def test_obv_detects_silent_accumulation():
    """Precio lateral/bajista pero OBV sube → alguien acumula en silencio."""
    # Precio cae levemente, volumen en días de alza > días de baja
    closes = [100.0, 99.5, 99.0, 99.2, 99.1, 99.0, 99.3, 99.1, 99.0, 98.9,
              99.0, 98.8, 98.7, 98.9, 98.8, 98.7]
    # Días alcistas tienen mucho volumen → OBV sube aunque precio baje
    vols   = [1000, 5000, 200, 4000, 200, 200, 3000, 200, 200, 200,
              4000, 200, 200, 3500, 200, 200]
    close_s = pd.Series(closes)
    vol_s   = pd.Series(vols, dtype=float)
    obv_s   = _compute_obv(close_s, vol_s)
    trend, div = _obv_analysis(close_s, obv_s, lookback=10)
    assert div == "acumulacion_silenciosa"


def test_obv_detects_distribution():
    """Precio sube pero OBV baja → distribución disfrazada de rally.

    Patrón: días de caída tienen mucho volumen, días de alza tienen poco.
    OBV acumula negativo → cae aunque el precio suba.
    """
    # Alternancia: sube poco (vol bajo) / baja mucho (vol alto)
    closes = [100.0]
    vols   = [500.0]
    for _ in range(15):
        # Día alcista: sube +0.3, volumen bajo
        closes.append(round(closes[-1] + 0.3, 2)); vols.append(200.0)
        # Día bajista: baja -0.1, volumen muy alto
        closes.append(round(closes[-1] - 0.1, 2)); vols.append(8000.0)
    close_s = pd.Series(closes)
    vol_s   = pd.Series(vols, dtype=float)
    obv_s   = _compute_obv(close_s, vol_s)
    # Precio neto sube (+0.2×15=+3), OBV neto baja (−7800×15 + 200×15)
    trend, div = _obv_analysis(close_s, obv_s, lookback=16)
    assert div == "distribucion"


# ── Boost/penalización al score ───────────────────────────────────────────────

def test_volume_score_delta_positive_on_explosion():
    snap = VolumeSnapshot(
        rvol=3.0, rvol_label="explosivo", vol_trend="expandiendo",
        obv_trend="alcista", obv_divergence=None, price_vol_ok=True, note="",
    )
    delta, notes = volume_score_delta(snap)
    assert delta > 10.0, "RVOL explosivo + vol expandiendo debe dar boost significativo"
    assert notes


def test_volume_score_delta_negative_on_distribution():
    snap = VolumeSnapshot(
        rvol=0.5, rvol_label="bajo", vol_trend="contrayendo",
        obv_trend="bajista", obv_divergence="distribucion", price_vol_ok=False, note="",
    )
    delta, notes = volume_score_delta(snap)
    assert delta < -10.0, "volumen bajo + distribución debe penalizar"


def test_volume_score_delta_big_boost_silent_accumulation():
    """Acumulación silenciosa (OBV div) da boost alto incluso con RVOL normal."""
    snap = VolumeSnapshot(
        rvol=1.0, rvol_label="normal", vol_trend="neutral",
        obv_trend="alcista", obv_divergence="acumulacion_silenciosa",
        price_vol_ok=False, note="",
    )
    delta, notes = volume_score_delta(snap)
    assert delta >= 10.0
    assert any("silenciosa" in n for n in notes)


# ── Integración con param_store (learnable) ───────────────────────────────────

def test_volume_snapshot_respects_lookback():
    closes = _flat_closes(30)
    vols_20 = [1000.0] * 29 + [2500.0]
    snap10 = volume_snapshot(_daily(closes, vols_20), lookback=10)
    snap25 = volume_snapshot(_daily(closes, vols_20), lookback=25)
    # Con lookback=10 el promedio solo incluye las últimas 10 velas → RVOL distinto
    assert snap10 is not None and snap25 is not None
    # Ambos deben detectar el spike (2.5x), solo magnitudes RVOL pueden diferir
    assert snap10.rvol_label in ("explosivo", "alto")
    assert snap25.rvol_label in ("explosivo", "alto")
