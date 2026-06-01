"""Tests de portfolio/sizing.py — Kelly fraccional y sizing."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from optionsdesk.portfolio.sizing import kelly_fraction, size_position, size_with_risk_budget


# ── kelly_fraction ────────────────────────────────────────────────────────────

def test_kelly_fraction_no_edge():
    assert kelly_fraction(ev_ars=0.0, max_loss_ars=1000.0, prob_profit=0.7) == 0.0


def test_kelly_fraction_negative_ev():
    assert kelly_fraction(ev_ars=-100.0, max_loss_ars=1000.0, prob_profit=0.7) == 0.0


def test_kelly_fraction_zero_max_loss():
    assert kelly_fraction(ev_ars=100.0, max_loss_ars=0.0, prob_profit=0.7) == 0.0


def test_kelly_fraction_standard():
    # f* = EV / max_loss = 200 / 1000 = 0.20; Kelly cuartil = 0.20 * 0.25 = 0.05
    result = kelly_fraction(ev_ars=200.0, max_loss_ars=1000.0, prob_profit=0.7)
    assert result == pytest.approx(0.05, abs=0.001)


def test_kelly_fraction_capped_at_one():
    # EV muy alta no puede devolver > 1
    result = kelly_fraction(ev_ars=10_000.0, max_loss_ars=100.0, prob_profit=0.99)
    assert result <= 1.0


def test_kelly_fraction_custom_factor():
    result = kelly_fraction(ev_ars=200.0, max_loss_ars=1000.0, prob_profit=0.7, kelly_factor=0.50)
    assert result == pytest.approx(0.10, abs=0.001)


# ── size_position ─────────────────────────────────────────────────────────────

def _make_rec(net_outlay: float, ev_ars=None, pop=None):
    """Crea un Recommendation mock con los campos relevantes."""
    result = MagicMock()
    result.net_outlay = net_outlay
    result.expected_value_ars = ev_ars
    result.prob_profit = pop
    result.delta = None

    rec = MagicMock()
    rec.result = result
    return rec


def test_size_position_zero_capital():
    rec = _make_rec(net_outlay=50.0)
    assert size_position(capital=0.0, rec=rec) == 0


def test_size_position_capital_below_one_lot():
    # net_outlay=50 → 1 lote = 50*100 = 5000; capital = 4000 → no alcanza
    rec = _make_rec(net_outlay=50.0)
    assert size_position(capital=4000.0, rec=rec) == 0


def test_size_position_fixed_fractional_fallback():
    # Sin EV ni PoP → fixed fractional: max 20% de capital / net_outlay_lot
    # capital = 100_000, 20% = 20_000, net_outlay_lot = 50*100 = 5_000 → 4 contratos
    rec = _make_rec(net_outlay=50.0, ev_ars=None, pop=None)
    n = size_position(capital=100_000.0, rec=rec)
    assert n == 4


def test_size_position_kelly_mode():
    # Con EV y PoP: Kelly cuartil
    # net_outlay_lot = 50*100 = 5000; max_loss = 2 * 5000 = 10000
    # EV = 500; f* = 500/10000 * 0.25 = 0.0125; kelly_capital = 100000 * 0.0125 = 1250
    # kelly_contracts = floor(1250/5000) = 0 → min 1 contrato
    rec = _make_rec(net_outlay=50.0, ev_ars=500.0, pop=0.7)
    n = size_position(capital=100_000.0, rec=rec)
    assert n >= 1


def test_size_position_minimum_one_contract():
    """Devuelve ≥ 1 si el capital alcanza para 1 lote y el 20% cap lo permite.

    net_outlay=50 → lote = 5000 ARS. Con capital=30_000, 20%=6_000 > 5_000
    → al menos 1 contrato. Kelly cuartil con EV pequeño puede dar 0; el
    _MIN_CONTRACTS solo entra si ya pasó el cap check.
    """
    rec = _make_rec(net_outlay=50.0, ev_ars=None, pop=None)   # fixed fractional
    n = size_position(capital=30_000.0, rec=rec)
    assert n >= 1


def test_size_position_cap_20pct():
    """Nunca asigna más del 20% del capital a un trade."""
    # capital = 1_000_000; 20% = 200_000; net_outlay_lot = 50*100 = 5000
    # max contratos = floor(200000 / 5000) = 40
    rec = _make_rec(net_outlay=50.0, ev_ars=None, pop=None)
    n = size_position(capital=1_000_000.0, rec=rec)
    assert n <= 40


# ── size_with_risk_budget ─────────────────────────────────────────────────────

def test_size_with_risk_budget_no_delta():
    """Sin delta en la rec → devuelve el tamaño base sin reducción."""
    rec = _make_rec(net_outlay=50.0)
    rec.result.delta = None
    n = size_with_risk_budget(capital=100_000.0, rec=rec, current_delta_net=0.0, max_net_delta=100.0)
    assert n == size_position(capital=100_000.0, rec=rec)


def test_size_with_risk_budget_reduces_on_delta_limit():
    """Con delta neto cerca del límite, reduce los contratos."""
    rec = _make_rec(net_outlay=50.0, ev_ars=None, pop=None)
    rec.result.delta = 0.30   # delta alto

    # current_delta_net = 90 (casi en el límite de 100)
    # Sin reducción: 4 contratos * 0.30 * 100 = 120 → supera 100
    n = size_with_risk_budget(
        capital=100_000.0, rec=rec,
        current_delta_net=90.0, max_net_delta=100.0
    )
    # Con 90 ya usado y límite 100, solo quedan 10 delta:
    # 10 / (0.30 * 100) = 0.33 → floor → 0 contratos
    assert n == 0


def test_size_with_risk_budget_plenty_budget():
    """Con presupuesto de delta amplio, el sizing no se reduce."""
    rec = _make_rec(net_outlay=50.0, ev_ars=None, pop=None)
    rec.result.delta = 0.20

    n_base = size_position(capital=100_000.0, rec=rec)
    n_budgeted = size_with_risk_budget(
        capital=100_000.0, rec=rec,
        current_delta_net=0.0, max_net_delta=1000.0
    )
    assert n_budgeted == n_base
