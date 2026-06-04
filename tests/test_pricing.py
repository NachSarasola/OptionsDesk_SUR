"""Tests del núcleo de pricing: Black-Scholes, CRR, griegas e IV.

Todas las propiedades fundamentales deben cumplirse:
- Paridad put-call para opciones europeas
- Opciones americanas >= europeas
- IV round-trip (precio → IV → precio)
- CRR converge a BS en el caso europeo sin dividendos
"""
import math

import pytest

from optionsdesk.core.pricing import (
    bs_delta,
    bs_gamma,
    bs_price,
    bs_rho,
    bs_theta,
    bs_vega,
    crr_delta,
    crr_gamma,
    crr_price,
    crr_theta,
    crr_vega,
    implied_vol,
)

# Parámetros estándar: ATM, 3 meses, vol moderada
S, K, T, r, sigma = 100.0, 100.0, 0.25, 0.05, 0.30


class TestBlackScholes:
    def test_put_call_parity(self):
        c = bs_price(S, K, T, r, sigma, "C")
        p = bs_price(S, K, T, r, sigma, "P")
        # C - P = S - K * e^(-rT)
        assert abs(c - p - (S - K * math.exp(-r * T))) < 1e-8

    def test_call_at_expiry_itm(self):
        assert abs(bs_price(110.0, 100.0, 0.0, 0.0, 0.3, "C") - 10.0) < 1e-8

    def test_call_at_expiry_otm(self):
        assert bs_price(90.0, 100.0, 0.0, 0.0, 0.3, "C") == 0.0

    def test_put_at_expiry_itm(self):
        assert abs(bs_price(90.0, 100.0, 0.0, 0.0, 0.3, "P") - 10.0) < 1e-8

    def test_delta_call_between_zero_and_one(self):
        d = bs_delta(S, K, T, r, sigma, "C")
        assert 0.0 < d < 1.0

    def test_delta_put_between_minus_one_and_zero(self):
        d = bs_delta(S, K, T, r, sigma, "P")
        assert -1.0 < d < 0.0

    def test_delta_put_call_relation(self):
        dc = bs_delta(S, K, T, r, sigma, "C")
        dp = bs_delta(S, K, T, r, sigma, "P")
        # delta_call - delta_put = exp(-qT); q=0 → 1.0
        assert abs(dc - dp - 1.0) < 1e-8

    def test_gamma_positive(self):
        assert bs_gamma(S, K, T, r, sigma) > 0.0

    def test_vega_positive(self):
        assert bs_vega(S, K, T, r, sigma) > 0.0

    def test_theta_negative_for_long_call(self):
        assert bs_theta(S, K, T, r, sigma, "C") < 0.0

    def test_theta_negative_for_long_put(self):
        assert bs_theta(S, K, T, r, sigma, "P") < 0.0

    def test_higher_vol_higher_price(self):
        p_lo = bs_price(S, K, T, r, 0.20, "C")
        p_hi = bs_price(S, K, T, r, 0.40, "C")
        assert p_hi > p_lo

    def test_deep_itm_call_approaches_forward(self):
        # Deep ITM call → price ≈ S - K*e^(-rT)
        c = bs_price(200.0, 100.0, T, r, sigma, "C")
        intrinsic = 200.0 - 100.0 * math.exp(-r * T)
        assert abs(c - intrinsic) < 1.0

    def test_rho_call_positive(self):
        assert bs_rho(S, K, T, r, sigma, "C") > 0.0

    def test_rho_put_negative(self):
        assert bs_rho(S, K, T, r, sigma, "P") < 0.0


class TestCRR:
    def test_european_converges_to_bs_call(self):
        bs_c = bs_price(S, K, T, r, sigma, "C")
        crr_c = crr_price(S, K, T, r, sigma, "C", american=False, steps=400)
        assert abs(crr_c - bs_c) < 0.10

    def test_european_converges_to_bs_put(self):
        bs_p = bs_price(S, K, T, r, sigma, "P")
        crr_p = crr_price(S, K, T, r, sigma, "P", american=False, steps=400)
        assert abs(crr_p - bs_p) < 0.10

    def test_american_call_ge_european_call(self):
        euro = crr_price(S, K, T, r, sigma, "C", american=False, steps=200)
        amer = crr_price(S, K, T, r, sigma, "C", american=True, steps=200)
        assert amer >= euro - 1e-6

    def test_american_put_ge_european_put(self):
        euro = crr_price(S, K, T, r, sigma, "P", american=False, steps=200)
        amer = crr_price(S, K, T, r, sigma, "P", american=True, steps=200)
        assert amer >= euro - 1e-6

    def test_put_call_parity_european(self):
        c = crr_price(S, K, T, r, sigma, "C", american=False, steps=300)
        p = crr_price(S, K, T, r, sigma, "P", american=False, steps=300)
        parity = S - K * math.exp(-r * T)
        assert abs(c - p - parity) < 0.20  # tolerancia por error del árbol

    def test_american_put_ge_intrinsic(self):
        # ITM american put must be worth at least its intrinsic value
        p = crr_price(90.0, 100.0, T, r, sigma, "P", american=True, steps=200)
        assert p >= 10.0

    def test_with_dividend_reduces_call_price(self):
        c_no_div = crr_price(S, K, T, r, sigma, "C", steps=200)
        c_div = crr_price(S, K, T, r, sigma, "C", steps=200, dividends=[(0.1, 5.0)])
        assert c_div < c_no_div

    def test_call_at_expiry(self):
        assert crr_price(110.0, 100.0, 0.0, r, sigma, "C") == 10.0
        assert crr_price(90.0, 100.0, 0.0, r, sigma, "C") == 0.0

    def test_crr_delta_call_range(self):
        d = crr_delta(S, K, T, r, sigma, "C", american=True, steps=100)
        assert 0.0 < d < 1.0

    def test_crr_gamma_positive(self):
        g = crr_gamma(S, K, T, r, sigma, "C", american=True, steps=100)
        assert g > 0.0

    def test_crr_vega_positive(self):
        v = crr_vega(S, K, T, r, sigma, "C", american=True, steps=100)
        assert v > 0.0


class TestImpliedVol:
    def test_roundtrip_call_bs(self):
        sigma_true = 0.35
        price = bs_price(S, K, T, r, sigma_true, "C")
        iv = implied_vol(price, S, K, T, r, "C", american=False, steps=100)
        assert iv is not None
        assert abs(iv - sigma_true) < 0.002

    def test_roundtrip_put_bs(self):
        sigma_true = 0.40
        price = bs_price(S, K, T, r, sigma_true, "P")
        iv = implied_vol(price, S, K, T, r, "P", american=False, steps=100)
        assert iv is not None
        assert abs(iv - sigma_true) < 0.002

    def test_roundtrip_american_call(self):
        sigma_true = 0.50
        price = crr_price(S, K, T, r, sigma_true, "C", american=True, steps=150)
        iv = implied_vol(price, S, K, T, r, "C", american=True, steps=150)
        assert iv is not None
        assert abs(iv - sigma_true) < 0.003

    def test_returns_none_for_negative_price(self):
        assert implied_vol(-1.0, S, K, T, r, "C") is None

    def test_returns_none_for_zero_price(self):
        assert implied_vol(0.0, S, K, T, r, "C") is None

    def test_returns_none_for_expired(self):
        assert implied_vol(5.0, S, K, 0.0, r, "C") is None

    @pytest.mark.parametrize("sigma_true", [0.20, 0.50, 0.80, 1.20])
    def test_roundtrip_wide_vol_range(self, sigma_true: float):
        price = bs_price(S, K, T, r, sigma_true, "C")
        iv = implied_vol(price, S, K, T, r, "C", american=False, steps=100)
        assert iv is not None
        assert abs(iv - sigma_true) < 0.005


# ── prob_touch ────────────────────────────────────────────────────────────────

from optionsdesk.core.pricing import prob_touch


def test_prob_touch_at_barrier_is_one():
    """spot = barrier → ya tocado → P = 1."""
    assert prob_touch(spot=100.0, barrier=100.0, T=0.1, sigma=0.60) == pytest.approx(1.0)


def test_prob_touch_above_terminal_prob_up():
    """P(touch B) >= P(S_T > B) para barreras arriba del spot."""
    import math
    from scipy.stats import norm
    spot, barrier, T, sigma = 100.0, 110.0, 30/365, 0.60
    # P terminal: risk_neutral_prob_above
    from optionsdesk.core.pricing import risk_neutral_prob_above
    p_terminal = risk_neutral_prob_above(spot, barrier, T, 0.0, sigma)
    p_touch = prob_touch(spot, barrier, T, sigma)
    assert p_touch >= p_terminal


def test_prob_touch_up_increases_with_sigma():
    """Mayor sigma → mayor P(touch) para barrera arriba."""
    spot, barrier, T = 100.0, 115.0, 30 / 365
    p_low  = prob_touch(spot, barrier, T, sigma=0.30)
    p_high = prob_touch(spot, barrier, T, sigma=0.90)
    assert p_high > p_low


def test_prob_touch_down_barrier():
    """Barrera por debajo del spot devuelve valor coherente en (0,1)."""
    p = prob_touch(spot=100.0, barrier=90.0, T=30 / 365, sigma=0.60)
    assert 0.0 < p < 1.0


def test_prob_touch_zero_T_returns_zero():
    """T=0 sin toque activo → 0."""
    assert prob_touch(spot=100.0, barrier=110.0, T=0.0, sigma=0.60) == pytest.approx(0.0)


def test_prob_touch_increases_with_T():
    """Mismo spot/barrier/sigma, T mayor → P(touch) mayor."""
    spot, barrier, sigma = 100.0, 115.0, 0.60
    p_short = prob_touch(spot, barrier, T=10 / 365, sigma=sigma)
    p_medium = prob_touch(spot, barrier, T=30 / 365, sigma=sigma)
    p_long = prob_touch(spot, barrier, T=90 / 365, sigma=sigma)
    assert p_short < p_medium < p_long


# ── sigma=0 guards ────────────────────────────────────────────────────────────

def test_bs_sigma_zero_call_returns_intrinsic():
    assert bs_price(110.0, 100.0, 0.25, 0.05, 0.0, "C") == pytest.approx(10.0)


def test_bs_sigma_zero_put_returns_intrinsic():
    assert bs_price(90.0, 100.0, 0.25, 0.05, 0.0, "P") == pytest.approx(10.0)


def test_bs_sigma_zero_otm_call_returns_zero():
    assert bs_price(90.0, 100.0, 0.25, 0.05, 0.0, "C") == pytest.approx(0.0)


def test_bs_greeks_sigma_zero_no_crash():
    """Ninguna griega BS debe tirar ZeroDivisionError con sigma=0."""
    assert bs_delta(S, K, T, r, 0.0, "C") == pytest.approx(0.0)
    assert bs_gamma(S, K, T, r, 0.0) == pytest.approx(0.0)
    assert bs_theta(S, K, T, r, 0.0, "C") == pytest.approx(0.0)
    assert bs_vega(S, K, T, r, 0.0) == pytest.approx(0.0)
    assert bs_rho(S, K, T, r, 0.0, "C") == pytest.approx(0.0)


def test_crr_sigma_zero_returns_intrinsic():
    """CRR con sigma=0 no debe crashear — devuelve intrínseco."""
    assert crr_price(110.0, 100.0, 0.25, 0.05, 0.0, "C") == pytest.approx(10.0)
    assert crr_price(90.0, 100.0, 0.25, 0.05, 0.0, "P") == pytest.approx(10.0)
    assert crr_price(90.0, 100.0, 0.25, 0.05, 0.0, "C") == pytest.approx(0.0)
