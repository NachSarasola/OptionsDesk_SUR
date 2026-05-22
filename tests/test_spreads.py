"""Tests para core/spreads.py y la funcion risk_neutral_prob_above de pricing.py."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from optionsdesk.config.costs import CostModel
from optionsdesk.core.pricing import risk_neutral_prob_above
from optionsdesk.core.spreads import (
    SpreadResult,
    build_bull_call_spread,
    build_bear_call_spread,
    build_bull_put_spread,
    build_bear_put_spread,
    _net_debit_per_share,
)

TODAY   = date.today()
EXPIRY  = TODAY + timedelta(days=30)
DAYS    = 30
SPOT    = 8300.0
ZERO_COSTS = CostModel(
    stock_commission_pct=0.0, option_commission_pct=0.0,
    stock_market_fee_pct=0.0, option_market_fee_pct=0.0,
    exercise_fee_pct=0.0, iva_rate=0.0,
)


# ── risk_neutral_prob_above ───────────────────────────────────────────────────

def test_prob_above_itm_is_high():
    """Si el spot esta muy por encima de K, P(S_T > K) debe ser alto."""
    p = risk_neutral_prob_above(S=10_000, K=5_000, T=0.5, r=0.0, sigma=0.65)
    assert p > 0.80


def test_prob_above_otm_is_low():
    p = risk_neutral_prob_above(S=5_000, K=10_000, T=0.5, r=0.0, sigma=0.65)
    assert p < 0.10


def test_prob_above_atm_near_half():
    """ATM con drift = 0 y sigma > 0: P ~ 0.50 (puede diferir por -0.5*sigma^2*T)."""
    p = risk_neutral_prob_above(S=8_000, K=8_000, T=0.25, r=0.0, sigma=0.65)
    assert 0.35 < p < 0.65


def test_prob_above_invalid_inputs():
    assert risk_neutral_prob_above(S=0, K=1000, T=0.5, r=0.0, sigma=0.5) == 0.0
    assert risk_neutral_prob_above(S=1000, K=0, T=0.5, r=0.0, sigma=0.5) == 1.0


def test_prob_above_zero_time():
    """Con T=0 la funcion usa la comparacion directa S > K."""
    assert risk_neutral_prob_above(S=9000, K=8000, T=0.0, r=0.0, sigma=0.5) == 1.0
    assert risk_neutral_prob_above(S=7000, K=8000, T=0.0, r=0.0, sigma=0.5) == 0.0


# ── Bull Call Spread ──────────────────────────────────────────────────────────

def test_bull_call_basic():
    """K=8000/8500, long_prem=300, short_prem=150 → debito neto=150, width=500."""
    sr = build_bull_call_spread(
        "GFG_8000C", 8000.0, 290.0, 310.0,
        "GFG_8500C", 8500.0, 140.0, 160.0,
        EXPIRY, DAYS, SPOT,
        sigma=0.65, r=0.0, cost_model=ZERO_COSTS,
    )
    assert sr is not None
    assert sr.strategy == "BULL_CALL"
    assert not sr.is_credit
    # net_debit ~ 150 (mid de 300 - mid de 150)
    assert abs(sr.net_debit - 150.0) < 1.0
    # max_loss = net_debit * 100
    assert abs(sr.max_loss - sr.net_debit * 100) < 1.0
    # max_profit = (500 - 150) * 100 = 35000
    assert abs(sr.max_profit - (500 - sr.net_debit) * 100) < 1.0
    # breakeven = 8000 + 150 = 8150
    assert abs(sr.breakeven - (8000 + sr.net_debit)) < 1.0


def test_bull_call_max_loss_plus_max_profit_equals_width_times_100():
    sr = build_bull_call_spread(
        "A", 8000.0, 295.0, 305.0,
        "B", 8500.0, 145.0, 155.0,
        EXPIRY, DAYS, SPOT, cost_model=ZERO_COSTS,
    )
    assert sr is not None
    total = sr.max_profit + sr.max_loss
    assert abs(total - (8500 - 8000) * 100) < 1.0


def test_bull_call_invalid_strikes():
    """long_strike >= short_strike → None."""
    sr = build_bull_call_spread(
        "A", 8500.0, 150.0, 160.0,
        "B", 8000.0, 300.0, 310.0,
        EXPIRY, DAYS, SPOT, cost_model=ZERO_COSTS,
    )
    assert sr is None


def test_bull_call_zero_premium_returns_none():
    sr = build_bull_call_spread(
        "A", 8000.0, 0.0, 0.0,
        "B", 8500.0, 150.0, 160.0,
        EXPIRY, DAYS, SPOT, cost_model=ZERO_COSTS,
    )
    assert sr is None


# ── Bear Call Spread ──────────────────────────────────────────────────────────

def test_bear_call_basic():
    """K=8000/8500, sell 8000 prem=300, buy 8500 prem=150 → credito 150, max_profit=15000."""
    sr = build_bear_call_spread(
        "GFG_8000C", 8000.0, 290.0, 310.0,   # short (sold)
        "GFG_8500C", 8500.0, 140.0, 160.0,   # long (bought)
        EXPIRY, DAYS, SPOT,
        sigma=0.65, r=0.0, cost_model=ZERO_COSTS,
    )
    assert sr is not None
    assert sr.strategy == "BEAR_CALL"
    assert sr.is_credit
    net_credit = -sr.net_debit
    assert net_credit > 0
    assert abs(sr.max_profit - net_credit * 100) < 1.0
    assert abs(sr.max_loss - (500 - net_credit) * 100) < 1.0
    assert abs(sr.breakeven - (8000 + net_credit)) < 1.0


def test_bear_call_total_equals_width():
    sr = build_bear_call_spread(
        "A", 8000.0, 295.0, 305.0,
        "B", 8500.0, 145.0, 155.0,
        EXPIRY, DAYS, SPOT, cost_model=ZERO_COSTS,
    )
    assert sr is not None
    assert abs((sr.max_profit + sr.max_loss) - 500 * 100) < 1.0


# ── Bull Put Spread ───────────────────────────────────────────────────────────

def test_bull_put_basic():
    """sell put K=8500 @ 400, buy put K=8000 @ 200 → credito 200, max_profit=20000."""
    sr = build_bull_put_spread(
        "GFG_8500P", 8500.0, 390.0, 410.0,   # short (sold)
        "GFG_8000P", 8000.0, 190.0, 210.0,   # long (bought)
        EXPIRY, DAYS, SPOT,
        sigma=0.65, r=0.0, cost_model=ZERO_COSTS,
    )
    assert sr is not None
    assert sr.strategy == "BULL_PUT"
    assert sr.is_credit
    net_credit = -sr.net_debit
    assert net_credit > 0
    assert abs(sr.max_profit - net_credit * 100) < 1.0
    assert abs(sr.breakeven - (8500 - net_credit)) < 1.0


def test_bull_put_invalid_strikes():
    """short_strike <= long_strike → None."""
    sr = build_bull_put_spread(
        "A", 8000.0, 200.0, 210.0,
        "B", 8500.0, 390.0, 410.0,
        EXPIRY, DAYS, SPOT, cost_model=ZERO_COSTS,
    )
    assert sr is None


# ── Bear Put Spread ───────────────────────────────────────────────────────────

def test_bear_put_basic():
    """buy put K=8500 @ 400, sell put K=8000 @ 200 → debito 200, max_profit=30000."""
    sr = build_bear_put_spread(
        "GFG_8500P", 8500.0, 390.0, 410.0,   # long (bought)
        "GFG_8000P", 8000.0, 190.0, 210.0,   # short (sold)
        EXPIRY, DAYS, SPOT,
        sigma=0.65, r=0.0, cost_model=ZERO_COSTS,
    )
    assert sr is not None
    assert sr.strategy == "BEAR_PUT"
    assert not sr.is_credit
    assert abs(sr.net_debit - 200.0) < 1.0
    assert abs(sr.max_loss - 200 * 100) < 1.0
    # max_profit = (500 - 200) * 100 = 30000
    assert abs(sr.max_profit - (500 - sr.net_debit) * 100) < 1.0


def test_bear_put_breakeven():
    sr = build_bear_put_spread(
        "A", 8500.0, 395.0, 405.0,
        "B", 8000.0, 195.0, 205.0,
        EXPIRY, DAYS, SPOT, cost_model=ZERO_COSTS,
    )
    assert sr is not None
    # breakeven = 8500 - net_debit
    assert abs(sr.breakeven - (8500 - sr.net_debit)) < 1.0


# ── Risk / Reward y PoP ───────────────────────────────────────────────────────

def test_risk_reward_positive():
    sr = build_bull_call_spread(
        "A", 8000.0, 295.0, 305.0,
        "B", 8500.0, 145.0, 155.0,
        EXPIRY, DAYS, SPOT, cost_model=ZERO_COSTS,
    )
    assert sr is not None
    assert sr.risk_reward > 0


def test_prob_of_profit_in_range():
    sr = build_bull_call_spread(
        "A", 8000.0, 295.0, 305.0,
        "B", 8500.0, 145.0, 155.0,
        EXPIRY, DAYS, SPOT, sigma=0.65, cost_model=ZERO_COSTS,
    )
    assert sr is not None
    assert 0.0 <= sr.prob_of_profit <= 1.0


def test_bear_call_pop_in_range():
    sr = build_bear_call_spread(
        "A", 8000.0, 295.0, 305.0,
        "B", 8500.0, 145.0, 155.0,
        EXPIRY, DAYS, SPOT, sigma=0.65, cost_model=ZERO_COSTS,
    )
    assert sr is not None
    assert 0.0 <= sr.prob_of_profit <= 1.0


# ── Costos ────────────────────────────────────────────────────────────────────

def test_costs_increase_net_debit():
    """Con costos reales el debito neto sube respecto a zero-cost."""
    sr_0 = build_bull_call_spread(
        "A", 8000.0, 295.0, 305.0,
        "B", 8500.0, 145.0, 155.0,
        EXPIRY, DAYS, SPOT, cost_model=ZERO_COSTS,
    )
    sr_real = build_bull_call_spread(
        "A", 8000.0, 295.0, 305.0,
        "B", 8500.0, 145.0, 155.0,
        EXPIRY, DAYS, SPOT,
    )
    assert sr_0 is not None and sr_real is not None
    assert sr_real.net_debit > sr_0.net_debit


def test_net_debit_helper():
    nd = _net_debit_per_share(300.0, 150.0, ZERO_COSTS)
    assert abs(nd - 150.0) < 1e-9
