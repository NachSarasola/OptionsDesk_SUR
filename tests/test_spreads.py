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
    spread_payoff,
    spread_expected_value,
    _net_debit_per_share,
)

import numpy as np

TODAY   = date(2026, 6, 20)   # fijo — inmune a vencimientos reales
EXPIRY  = TODAY + timedelta(days=30)
DAYS    = 30
SPOT    = 8300.0
ZERO_COSTS = CostModel(
    stock_commission_pct=0.0, option_commission_pct=0.0,
    stock_market_fee_pct=0.0, option_market_fee_pct=0.0,
    exercise_fee_pct=0.0, exercise_market_fee_pct=0.0, iva_rate=0.0,
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


# ── spread_payoff ─────────────────────────────────────────────────────────────

def test_spread_payoff_bull_call_knots():
    """Verifica los quiebres del payoff lineal a tramos."""
    K_lo, K_hi = 8000.0, 8500.0
    mp, ml = 35_000.0, 15_000.0   # ancho=500, net_dbt=150 → max_profit=35k, max_loss=15k

    # Por debajo de K_lo: max_loss negativo
    pnl_below = spread_payoff("BULL_CALL", K_lo, K_hi, mp, ml, [7000.0])
    assert abs(pnl_below[0] - (-ml)) < 1.0

    # Por encima de K_hi: max_profit
    pnl_above = spread_payoff("BULL_CALL", K_lo, K_hi, mp, ml, [9000.0])
    assert abs(pnl_above[0] - mp) < 1.0

    # En K_lo: -max_loss
    assert abs(spread_payoff("BULL_CALL", K_lo, K_hi, mp, ml, [K_lo])[0] - (-ml)) < 1.0

    # En K_hi: +max_profit
    assert abs(spread_payoff("BULL_CALL", K_lo, K_hi, mp, ml, [K_hi])[0] - mp) < 1.0


def test_spread_payoff_bear_call_direction():
    """Bear call: max_profit en K_lo, -max_loss en K_hi (pendiente inversa)."""
    K_lo, K_hi = 8000.0, 8500.0
    mp, ml = 15_000.0, 35_000.0
    assert abs(spread_payoff("BEAR_CALL", K_lo, K_hi, mp, ml, [7000.0])[0] - mp) < 1.0
    assert abs(spread_payoff("BEAR_CALL", K_lo, K_hi, mp, ml, [9000.0])[0] - (-ml)) < 1.0


def test_spread_payoff_bounded():
    """El payoff siempre esta acotado en [-max_loss, max_profit]."""
    spots = np.linspace(1000.0, 20000.0, 500)
    for strategy in ("BULL_CALL", "BEAR_CALL", "BULL_PUT", "BEAR_PUT"):
        pnl = spread_payoff(strategy, 8000.0, 8500.0, 35_000.0, 15_000.0, spots)
        assert np.all(pnl >= -15_000.0 - 1e-6)
        assert np.all(pnl <= 35_000.0 + 1e-6)


# ── spread_expected_value ─────────────────────────────────────────────────────

def test_ev_bounded_within_payoff_range():
    """EV debe estar en [-max_loss, max_profit]."""
    ev = spread_expected_value("BULL_CALL", 8000.0, 8500.0, 35_000.0, 15_000.0,
                               spot=8300.0, days=30, sigma=0.65)
    assert -15_000.0 <= ev <= 35_000.0


def test_ev_increases_with_positive_drift():
    """Para BULL_CALL, EV crece al subir el drift (más probabilidad de terminar ITM)."""
    common = dict(
        strategy="BULL_CALL", k_lo=8000.0, k_hi=8500.0,
        max_profit=35_000.0, max_loss=15_000.0,
        spot=8300.0, days=30, sigma=0.65,
    )
    ev_zero   = spread_expected_value(**common, drift=0.0)
    ev_posit  = spread_expected_value(**common, drift=0.5)
    assert ev_posit > ev_zero


def test_ev_executable_gt_mid_for_debit():
    """Con precios ejecutables (buy@ask, sell@bid) el debito sube → EV cae vs mid."""
    args = (
        "A", 8000.0, 290.0, 310.0,   # long: symbol, strike, bid, ask
        "B", 8500.0, 140.0, 160.0,   # short: symbol, strike, bid, ask
        EXPIRY, DAYS, SPOT,
    )
    sr_mid  = build_bull_call_spread(*args, sigma=0.65, r=0.0, cost_model=ZERO_COSTS, price_mode="mid")
    sr_exec = build_bull_call_spread(*args, sigma=0.65, r=0.0, cost_model=ZERO_COSTS, price_mode="executable")
    assert sr_mid is not None and sr_exec is not None
    # Debito ejecutable > debito mid (buy al ask, sell al bid)
    assert sr_exec.net_debit > sr_mid.net_debit
    # EV ejecutable <= EV mid (cost mas alto → menos edge)
    assert sr_exec.expected_value <= sr_mid.expected_value + 1.0   # margen de redondeo


# ── v2.6: Greeks netos ────────────────────────────────────────────────────────

from datetime import date, timedelta

_EXPIRY_GK = date(2026, 6, 20) + timedelta(days=30)
_DAYS_GK   = 30
_SPOT_GK   = 8300.0
_SIGMA_GK  = 0.60


def _bull_call():
    return build_bull_call_spread(
        "A", 8000.0, 290.0, 310.0,
        "B", 8500.0, 130.0, 150.0,
        _EXPIRY_GK, _DAYS_GK, _SPOT_GK,
        sigma=_SIGMA_GK, r=0.0, cost_model=ZERO_COSTS,
    )


def _bear_call():
    return build_bear_call_spread(
        "A", 8000.0, 290.0, 310.0,
        "B", 8500.0, 130.0, 150.0,
        _EXPIRY_GK, _DAYS_GK, _SPOT_GK,
        sigma=_SIGMA_GK, r=0.0, cost_model=ZERO_COSTS,
    )


def _bull_put():
    # short K_hi=8500 (ITM put, más caro): bid=420 / ask=440
    # long  K_lo=8000 (OTM put, más barato): bid=110 / ask=130
    return build_bull_put_spread(
        "A", 8500.0, 420.0, 440.0,
        "B", 8000.0, 110.0, 130.0,
        _EXPIRY_GK, _DAYS_GK, _SPOT_GK,
        sigma=_SIGMA_GK, r=0.0, cost_model=ZERO_COSTS,
    )


def test_spread_has_net_greeks_fields():
    """SpreadResult debe tener net_delta, net_theta, net_vega como float."""
    sr = _bull_call()
    assert sr is not None
    assert isinstance(sr.net_delta, float)
    assert isinstance(sr.net_theta, float)
    assert isinstance(sr.net_vega,  float)


def test_bull_call_net_delta_positive():
    """Bull call (alcista) → delta neto positivo siempre (más call larga)."""
    sr = _bull_call()
    assert sr is not None
    assert sr.net_delta > 0.0, f"delta={sr.net_delta} debe ser >0 en bull call"


def test_bull_put_net_vega_negative():
    """Bull put (credit, short IV) → vega neto negativo (spread corto de vol)."""
    sr = _bull_put()
    assert sr is not None
    # Short put K_hi tiene más vega que long put K_lo → net_vega = vega(long) - vega(short) < 0
    assert sr.net_vega < 0.0, f"vega={sr.net_vega} debe ser <0 para credit spread (short vol)"


def test_credit_spread_net_theta_positive():
    """Bull put con short leg cerca del ATM → theta neto positivo."""
    sr = _bull_put()
    assert sr is not None
    # short K_hi=8500 (2.4% ITM) mas cerca de ATM que long K_lo=8000 (3.6% OTM)
    # → |theta(short)| > |theta(long)| → net_theta > 0
    assert sr.net_theta > 0.0, f"theta={sr.net_theta} esperaba >0 para bull put con short cerca ATM"


# ── v2.7: BEAR_CALL / BEAR_PUT Greeks ────────────────────────────────────────

def test_bear_call_net_delta_negative():
    """Bear call (bajista, short K_lo) → delta neto negativo."""
    sr = _bear_call()
    assert sr is not None
    # long leg = K_hi=8500, short leg = K_lo=8000
    # net_delta = delta(long K8500) - delta(short K8000); short K8000 mas ATM → net < 0
    assert sr.net_delta < 0.0, f"delta={sr.net_delta} debe ser <0 en bear call"


def test_bear_put_net_vega_positive():
    """Bear put (debito, compra vol) → vega neto positivo."""
    expiry = date(2026, 6, 20) + timedelta(days=30)
    sr = build_bear_put_spread(
        "A", 8500.0, 390.0, 410.0,   # long (bought)
        "B", 8000.0, 110.0, 130.0,   # short (sold)
        expiry, 30, 8300.0,
        sigma=0.60, r=0.0, cost_model=ZERO_COSTS,
    )
    assert sr is not None
    # long K_hi=8500 tiene mas vega que short K_lo=8000 → net_vega > 0
    assert sr.net_vega > 0.0, f"vega={sr.net_vega} debe ser >0 para bear put (debit, long vol)"
