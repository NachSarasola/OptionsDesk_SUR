"""Tests de portfolio/aggregator.py."""
from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from optionsdesk.portfolio.aggregator import (
    AggregatedGreeks,
    ConcentrationReport,
    ScenarioPnL,
    aggregate_greeks,
    concentration_report,
    scenario_pnl,
)


def _make_position(
    symbol: str = "GFGV1500JUN",
    strategy: str = "SHORT_PUT",
    strike: float = 1500.0,
    premium_received: float = 50.0,
    net_outlay: float = 50.0,
    iv_entry: float = 0.60,
    caucion_tna: float = 60.0,
    days_remaining: int = 30,
):
    pos = MagicMock()
    pos.symbol = symbol
    pos.strategy = strategy
    pos.strike = strike
    pos.premium_received = premium_received
    pos.net_outlay = net_outlay
    pos.iv_entry = iv_entry
    pos.caucion_tna = caucion_tna
    pos.days_remaining = days_remaining
    pos.opt_type = "P" if strategy != "COVERED_CALL" else "C"
    pos.contracts = 1
    return pos


# ── aggregate_greeks ──────────────────────────────────────────────────────────

def test_aggregate_greeks_empty():
    result = aggregate_greeks([], current_spot=1500.0)
    assert result.n_positions == 0
    assert result.delta_net == 0.0
    assert result.theta_daily_ars == 0.0
    assert result.positions == []


def test_aggregate_greeks_returns_aggregated_greeks():
    pos = _make_position()
    with patch("optionsdesk.portfolio.aggregator.crr_delta", return_value=0.30), \
         patch("optionsdesk.portfolio.aggregator.crr_theta", return_value=-0.10), \
         patch("optionsdesk.portfolio.aggregator.crr_vega", return_value=0.05), \
         patch("optionsdesk.portfolio.aggregator.crr_price", return_value=30.0):
        result = aggregate_greeks([pos], current_spot=1500.0)

    assert isinstance(result, AggregatedGreeks)
    assert result.n_positions == 1
    # Delta vendedor: -1 * delta * contratos * 100 = -0.30 * 1 * 100 = -30
    assert result.delta_net == pytest.approx(-30.0, abs=0.1)
    # Theta: -(-0.10) * 1 * 100 = 10
    assert result.theta_daily_ars == pytest.approx(10.0, abs=1.0)
    assert result.total_capital_at_risk == pytest.approx(50.0 * 100.0, abs=1.0)


def test_aggregate_greeks_multiple_positions():
    pos1 = _make_position("GFGV1500JUN", strike=1500.0)
    pos2 = _make_position("GFGV1600JUN", strike=1600.0, net_outlay=60.0)

    with patch("optionsdesk.portfolio.aggregator.crr_delta", return_value=0.25), \
         patch("optionsdesk.portfolio.aggregator.crr_theta", return_value=-0.08), \
         patch("optionsdesk.portfolio.aggregator.crr_vega", return_value=0.04), \
         patch("optionsdesk.portfolio.aggregator.crr_price", return_value=20.0):
        result = aggregate_greeks([pos1, pos2], current_spot=1550.0)

    assert result.n_positions == 2
    # Dos posiciones cortas: delta_net = 2 * (-0.25 * 1 * 100) = -50
    assert result.delta_net == pytest.approx(-50.0, abs=0.1)


def test_aggregate_greeks_skips_invalid_iv():
    """Posición con IV = 0 se omite del cómputo sin lanzar excepción."""
    pos = _make_position(iv_entry=0.0)
    result = aggregate_greeks([pos], current_spot=1500.0)
    assert result.n_positions == 1
    assert result.delta_net == 0.0   # omitida, no suma delta


# ── concentration_report ──────────────────────────────────────────────────────

def test_concentration_report_single_expiry():
    positions = [
        _make_position("GFGV1500JUN"),
        _make_position("GFGV1600JUN"),
    ]
    report = concentration_report(positions, max_per_expiry=3)
    assert isinstance(report, ConcentrationReport)
    assert "JUN" in report.by_expiry
    assert report.by_expiry["JUN"]["n_pos"] == 2
    # No supera el límite de 3
    assert not report.flags or all("JUN" not in f for f in report.flags)


def test_concentration_report_flags_high_concentration():
    positions = [_make_position(f"GFGV{1500 + i * 50}JUN") for i in range(4)]
    report = concentration_report(positions, max_per_expiry=3)
    assert any("JUN" in f for f in report.flags)


def test_concentration_report_mixed_strategies():
    positions = [
        _make_position("GFGC1800JUN", strategy="COVERED_CALL"),
        _make_position("GFGV1500JUN", strategy="SHORT_PUT"),
    ]
    report = concentration_report(positions)
    assert report.by_strategy["CC"] == 1
    assert report.by_strategy["SP"] == 1
    assert any("CC+SP" in f for f in report.flags)


def test_concentration_report_classifies_long_scalps_separately():
    pos = _make_position("GFGC1800JUN", strategy="SCALP_LONG_CALL", net_outlay=40.0)

    report = concentration_report([pos])

    assert report.by_strategy["LC"] == 1
    assert report.by_strategy["SP"] == 0
    assert report.by_expiry["JUN"]["strategies"] == ["LC"]


# ── scenario_pnl ─────────────────────────────────────────────────────────────

def test_scenario_pnl_zero_shock():
    """P&L con shock=0 debe ser ≈ 0 (opción priceada al valor actual)."""
    pos = _make_position(premium_received=50.0)
    with patch("optionsdesk.portfolio.aggregator.crr_price", return_value=50.0):
        result = scenario_pnl([pos], current_spot=1500.0, shock_pcts=[0.0])

    assert result.shock_pcts == [0.0]
    assert result.pnl_ars[0] == pytest.approx(0.0, abs=1.0)


def test_scenario_pnl_positive_shock_short_put():
    """Spot sube → put OTM → prima cae → P&L positivo para el vendedor."""
    pos = _make_position(premium_received=50.0)
    # Spot sube 10% → put se aleja del strike → valor cae a 20
    with patch("optionsdesk.portfolio.aggregator.crr_price", return_value=20.0):
        result = scenario_pnl([pos], current_spot=1500.0, shock_pcts=[+10.0])

    # P&L = (50 - 20) * 1 * 100 = 3000
    assert result.pnl_ars[0] == pytest.approx(3000.0, abs=10.0)


def test_scenario_pnl_multiple_shocks():
    pos = _make_position(premium_received=50.0)
    shocks = [-10.0, 0.0, +10.0]
    with patch("optionsdesk.portfolio.aggregator.crr_price", return_value=50.0):
        result = scenario_pnl([pos], current_spot=1500.0, shock_pcts=shocks)

    assert len(result.pnl_ars) == 3
    assert isinstance(result, ScenarioPnL)
