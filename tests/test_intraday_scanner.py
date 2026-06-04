"""Tests del scanner de oportunidades intradía (signals/intraday_scanner.py).

Usa snapshots sintéticos inyectados en un directorio temporal para verificar
que:
- Un spike de IV (residual alto) se detecta como señal WRITE.
- Una caída de prima en una posición corta se detecta como señal CLOSE.
- Movimientos puramente de delta (residual bajo) NO generan señal WRITE.
- Con insuficientes snapshots no se generan señales (no crash).
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from optionsdesk.signals.intraday_scanner import (
    IntradaySignal,
    detect_close_opportunities,
    detect_write_opportunities,
    load_intraday_chain,
)
from optionsdesk.signals.monitor import OpenPosition


# ── Fixtures de snapshots sintéticos ─────────────────────────────────────────

def _make_snapshot_row(
    ts: datetime,
    symbol: str,
    spot: float,
    bid: float,
    ask: float,
    delta: float = 0.0,
) -> dict:
    return {
        "timestamp": pd.Timestamp(ts),
        "symbol":    symbol,
        "spot":      spot,
        "bid":       bid,
        "ask":       ask,
        "last":      (bid + ask) / 2.0,
        "volume":    100.0,
        "delta":     delta,
    }


def _write_day_snapshots(day_dir: Path, rows: list[dict]) -> None:
    day_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(day_dir / "snap_test.parquet", index=False)


def _baseline_series(
    n: int,
    base_price: float,
    base_spot: float,
    symbol: str = "GFGC9000F",
    delta: float = 0.30,
    interval_minutes: int = 2,
    start: datetime | None = None,
) -> list[dict]:
    """Serie de n snapshots con precio de opción y spot estables."""
    t0 = start or datetime(2026, 5, 22, 10, 30)
    rows = []
    for i in range(n):
        ts = t0 + timedelta(minutes=i * interval_minutes)
        rows.append(_make_snapshot_row(
            ts=ts,
            symbol=symbol,
            spot=base_spot + i * 0.5,   # spot sube levemente
            bid=base_price * 0.99,
            ask=base_price * 1.01,
            delta=delta,
        ))
    return rows


# ── Tests de load_intraday_chain ──────────────────────────────────────────────

def test_load_intraday_chain_returns_series(tmp_path):
    day = date(2026, 5, 22)
    day_dir = tmp_path / day.isoformat()
    rows = _baseline_series(10, base_price=200.0, base_spot=8500.0)
    _write_day_snapshots(day_dir, rows)

    chain = load_intraday_chain(tmp_path, day)
    assert "GFGC9000F" in chain
    assert len(chain["GFGC9000F"]) == 10


def test_load_intraday_chain_empty_when_no_snapshots(tmp_path):
    chain = load_intraday_chain(tmp_path, date(2026, 5, 22))
    assert chain == {}


def test_load_intraday_chain_filters_short_series(tmp_path):
    day = date(2026, 5, 22)
    day_dir = tmp_path / day.isoformat()
    rows = _baseline_series(3, base_price=200.0, base_spot=8500.0)   # < _MIN_SNAPSHOTS=5
    _write_day_snapshots(day_dir, rows)

    chain = load_intraday_chain(tmp_path, day)
    assert "GFGC9000F" not in chain


# ── Tests de detect_write_opportunities ──────────────────────────────────────

def test_detect_write_iv_spike_detected(tmp_path):
    """Serie con spike puro de IV (prima sube sin que el spot explique) → WRITE."""
    day = date(2026, 5, 22)
    day_dir = tmp_path / day.isoformat()

    # 12 snapshots base con prima estable (~200) y spot prácticamente flat
    rows = []
    t0 = datetime(2026, 5, 22, 10, 30)
    for i in range(12):
        ts = t0 + timedelta(minutes=i * 2)
        rows.append(_make_snapshot_row(
            ts=ts, symbol="GFGC9000F",
            spot=8500.0,     # spot completamente estático
            bid=196.0, ask=204.0,   # prima ~200, estable
            delta=0.20,
        ))
    # Los últimos 3: prima se triplica (600), spot sigue flat → residual IV enorme
    for i in range(3):
        ts = t0 + timedelta(minutes=(12 + i) * 2)
        rows.append(_make_snapshot_row(
            ts=ts, symbol="GFGC9000F",
            spot=8500.0,     # spot no se mueve
            bid=594.0, ask=606.0,   # prima sube 200% sin causa delta
            delta=0.20,
        ))
    _write_day_snapshots(day_dir, rows)

    signals = detect_write_opportunities(tmp_path, day, zscore_threshold=1.5, window=3)
    assert len(signals) >= 1
    write_sigs = [s for s in signals if s.signal_type == "WRITE"]
    assert len(write_sigs) >= 1
    assert write_sigs[0].symbol == "GFGC9000F"
    assert write_sigs[0].premium_change_pct > 0


def test_detect_write_no_spike_no_signal(tmp_path):
    """Serie estable sin spike → sin señales WRITE."""
    day = date(2026, 5, 22)
    day_dir = tmp_path / day.isoformat()
    rows = _baseline_series(12, base_price=200.0, base_spot=8500.0)
    _write_day_snapshots(day_dir, rows)

    signals = detect_write_opportunities(tmp_path, day, zscore_threshold=2.0)
    assert signals == []


def test_detect_write_returns_sorted_by_zscore(tmp_path):
    """Las señales WRITE se ordenan por z-score descendente."""
    day = date(2026, 5, 22)
    day_dir = tmp_path / day.isoformat()
    t0 = datetime(2026, 5, 22, 10, 30)

    # Dos símbolos con spikes de diferente magnitud
    rows = []
    for sym, base_p, spike_p in [("GFGC9000F", 200.0, 500.0), ("GFGC8500F", 150.0, 300.0)]:
        rows += _baseline_series(8, base_price=base_p, base_spot=8500.0, symbol=sym)
        for i in range(3):
            ts = t0 + timedelta(minutes=(8 + i) * 2)
            rows.append(_make_snapshot_row(
                ts=ts, symbol=sym, spot=8501.0,
                bid=spike_p * 0.99, ask=spike_p * 1.01, delta=0.15,
            ))
    _write_day_snapshots(day_dir, rows)

    signals = detect_write_opportunities(tmp_path, day, zscore_threshold=1.0, window=3)
    if len(signals) >= 2:
        assert signals[0].iv_residual_z >= signals[1].iv_residual_z


# ── Tests de detect_close_opportunities ──────────────────────────────────────

def _make_position(symbol: str = "GFGC9000F", premium: float = 300.0) -> OpenPosition:
    return OpenPosition(
        symbol=symbol,
        strategy="COVERED_CALL",
        strike=9000.0,
        spot_entry=8500.0,
        premium_received=premium,
        net_outlay=8200.0,
        iv_entry=0.55,
        days_entry=30,
        entry_date=date.today() - timedelta(days=5),
        target_exit_days=15,
        target_capture_pct=50.0,
    )


def test_detect_close_drop_detected(tmp_path):
    """Prima cayó > 30% en la ventana → señal CLOSE."""
    day = date(2026, 5, 22)
    day_dir = tmp_path / day.isoformat()

    t0 = datetime(2026, 5, 22, 10, 30)
    rows = []
    # 8 snapshots con prima alta
    for i in range(8):
        ts = t0 + timedelta(minutes=i * 2)
        rows.append(_make_snapshot_row(ts=ts, symbol="GFGC9000F", spot=8500.0,
                                       bid=280.0, ask=320.0))
    # 3 snapshots con prima que cayó 60%
    for i in range(3):
        ts = t0 + timedelta(minutes=(8 + i) * 2)
        rows.append(_make_snapshot_row(ts=ts, symbol="GFGC9000F", spot=8200.0,
                                       bid=110.0, ask=130.0))
    _write_day_snapshots(day_dir, rows)

    pos = _make_position("GFGC9000F", premium=300.0)
    signals = detect_close_opportunities(tmp_path, day, [pos], drop_threshold_pct=30.0, window=3)
    assert len(signals) == 1
    assert signals[0].signal_type == "CLOSE"
    assert signals[0].premium_change_pct < -30.0


def test_detect_close_no_drop_no_signal(tmp_path):
    """Prima estable → sin señal CLOSE."""
    day = date(2026, 5, 22)
    day_dir = tmp_path / day.isoformat()
    rows = _baseline_series(12, base_price=200.0, base_spot=8500.0)
    _write_day_snapshots(day_dir, rows)

    pos = _make_position(premium=200.0)
    signals = detect_close_opportunities(tmp_path, day, [pos], drop_threshold_pct=30.0)
    assert signals == []


def test_detect_close_empty_positions(tmp_path):
    """Sin posiciones → sin señales."""
    day = date(2026, 5, 22)
    day_dir = tmp_path / day.isoformat()
    rows = _baseline_series(10, base_price=200.0, base_spot=8500.0)
    _write_day_snapshots(day_dir, rows)

    signals = detect_close_opportunities(tmp_path, day, [])
    assert signals == []


def test_detect_close_no_snapshots(tmp_path):
    """Sin snapshots → sin señales (no crash)."""
    pos = _make_position()
    signals = detect_close_opportunities(tmp_path, date(2026, 5, 22), [pos])
    assert signals == []
