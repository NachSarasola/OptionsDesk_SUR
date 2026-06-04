"""Tests de directional_ideas.py."""
from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import patch

import pytest

from optionsdesk.data.providers.base import OptionsChain, Quote
from optionsdesk.signals.directional import MarketContext
from optionsdesk.signals.directional_ideas import (
    DirectionalIdea,
    _parse_strike,
    build_directional_idea,
)
from optionsdesk.signals.technical import FairValueGap, OrderBlock, TechnicalSnapshot


def _make_snap(
    trend: str = "ALCISTA",
    bos: str | None = "BOS_UP",
    fvgs: list | None = None,
    order_block: OrderBlock | None = None,
    atr_pct: float = 2.5,
) -> TechnicalSnapshot:
    return TechnicalSnapshot(
        trend=trend, rsi=60.0, atr_pct=atr_pct,
        sma_fast=0.0, sma_slow=0.0, momentum=3.0, read="test",
        signal_strength="fuerte", bos=bos,
        fvgs=fvgs or [], order_block=order_block,
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_context(
    trend: str = "alcista",
    momentum_pct: float = 3.0,
    confidence: str = "alta",
    signal_strength: str = "fuerte",
    rsi: float = 62.0,
    atr_pct: float = 2.5,
) -> MarketContext:
    return MarketContext(
        trend=trend,
        momentum_pct=momentum_pct,
        confidence=confidence,
        note="test",
        rsi=rsi,
        atr_pct=atr_pct,
        signal_strength=signal_strength,
    )


def _make_chain(spot: float = 8500.0) -> OptionsChain:
    """Cadena mínima con calls y puts alrededor del spot."""
    spot_q = Quote(
        symbol="GGAL", bid=spot * 0.999, ask=spot * 1.001,
        last=spot, volume=10000,
        timestamp=date.today(),
    )
    options: dict[str, Quote] = {}
    expiry_suffix = "OC"   # vencimiento cercano
    for k in (8000, 8500, 9000):
        # Call
        call_sym = f"GFGC{k}{expiry_suffix}"
        options[call_sym] = Quote(
            symbol=call_sym,
            bid=200.0, ask=210.0, last=205.0, volume=100,
            timestamp=date.today(),
        )
        # Put
        put_sym = f"GFGV{k}{expiry_suffix}"
        options[put_sym] = Quote(
            symbol=put_sym,
            bid=180.0, ask=190.0, last=185.0, volume=80,
            timestamp=date.today(),
        )
    return OptionsChain(underlying="GGAL", spot=spot_q, options=options)


# ── _parse_strike ─────────────────────────────────────────────────────────────

def test_parse_strike_byma_format():
    assert _parse_strike("GFGC8500OC") == pytest.approx(8500.0)


def test_parse_strike_large():
    assert _parse_strike("GFGV10000OC") == pytest.approx(10000.0)


def test_parse_strike_no_digits_returns_none():
    assert _parse_strike("ABCDEF") is None


# ── build_directional_idea ────────────────────────────────────────────────────

def _mock_days(sym: str) -> int:
    """Mock de days_to_expiry: devuelve 20 días (dentro del rango swing)."""
    return 20


@patch("optionsdesk.signals.directional_ideas.days_to_expiry", side_effect=_mock_days)
def test_bullish_context_returns_buy_call(mock_days):
    ctx   = _make_context(trend="alcista", momentum_pct=3.5, signal_strength="fuerte")
    chain = _make_chain(spot=8500.0)
    idea  = build_directional_idea(ctx, chain, 8500.0)
    assert idea is not None
    assert idea.idea_type == "BUY_CALL"
    assert idea.strike >= 8500.0 * 0.97
    assert idea.total_cost_ars > 0
    assert idea.breakeven > idea.strike
    assert idea.target_spot > 8500.0
    assert idea.stop_spot < 8500.0


@patch("optionsdesk.signals.directional_ideas.days_to_expiry", side_effect=_mock_days)
def test_bearish_context_returns_buy_put(mock_days):
    ctx   = _make_context(trend="bajista", momentum_pct=-3.5, signal_strength="fuerte")
    chain = _make_chain(spot=8500.0)
    idea  = build_directional_idea(ctx, chain, 8500.0)
    assert idea is not None
    assert idea.idea_type == "BUY_PUT"
    assert idea.target_spot < 8500.0
    assert idea.stop_spot > 8500.0


def test_weak_signal_returns_none():
    ctx = _make_context(trend="alcista", momentum_pct=0.5, signal_strength="débil")
    chain = _make_chain()
    idea = build_directional_idea(ctx, chain, 8500.0)
    assert idea is None


def test_low_confidence_returns_none():
    ctx = _make_context(trend="alcista", confidence="baja", signal_strength="fuerte")
    chain = _make_chain()
    idea = build_directional_idea(ctx, chain, 8500.0)
    assert idea is None


def test_no_data_confidence_returns_none():
    ctx = _make_context(confidence="sin datos", signal_strength="fuerte")
    chain = _make_chain()
    idea = build_directional_idea(ctx, chain, 8500.0)
    assert idea is None


def test_lateral_trend_returns_none():
    ctx = _make_context(trend="lateral", momentum_pct=0.0, signal_strength="neutral")
    chain = _make_chain()
    idea = build_directional_idea(ctx, chain, 8500.0)
    assert idea is None


def test_low_momentum_returns_none():
    ctx = _make_context(
        trend="alcista", momentum_pct=0.8, confidence="alta", signal_strength="moderada"
    )
    chain = _make_chain()
    idea = build_directional_idea(ctx, chain, 8500.0)
    assert idea is None


@patch("optionsdesk.signals.directional_ideas.days_to_expiry", side_effect=_mock_days)
def test_idea_has_disclaimer(mock_days):
    ctx  = _make_context()
    chain = _make_chain()
    idea = build_directional_idea(ctx, chain, 8500.0)
    if idea is not None:
        assert "ESPECULATIVA" in idea.disclaimer or "especulativ" in idea.disclaimer.lower()


@patch("optionsdesk.signals.directional_ideas.days_to_expiry", side_effect=_mock_days)
def test_max_loss_equals_total_cost(mock_days):
    ctx   = _make_context(trend="alcista", momentum_pct=4.0, signal_strength="fuerte")
    chain = _make_chain()
    idea  = build_directional_idea(ctx, chain, 8500.0)
    if idea is not None:
        assert idea.max_loss_ars == pytest.approx(idea.total_cost_ars)


# ── v2.7: target/stop/RR/confluencia SMC ──────────────────────────────────────

@patch("optionsdesk.signals.directional_ideas.days_to_expiry", side_effect=_mock_days)
def test_call_target_always_clears_breakeven(mock_days):
    """El objetivo de un call nunca puede quedar por debajo del break-even."""
    ctx   = _make_context(trend="alcista", momentum_pct=3.5, signal_strength="fuerte")
    chain = _make_chain(spot=8500.0)
    idea  = build_directional_idea(ctx, chain, 8500.0)
    assert idea is not None
    assert idea.target_spot > idea.breakeven


@patch("optionsdesk.signals.directional_ideas.days_to_expiry", side_effect=_mock_days)
def test_put_target_always_clears_breakeven(mock_days):
    ctx   = _make_context(trend="bajista", momentum_pct=-3.5, signal_strength="fuerte")
    chain = _make_chain(spot=8500.0)
    idea  = build_directional_idea(ctx, chain, 8500.0)
    assert idea is not None
    assert idea.target_spot < idea.breakeven


@patch("optionsdesk.signals.directional_ideas.days_to_expiry", side_effect=_mock_days)
def test_fvg_below_breakeven_is_not_used_as_target(mock_days):
    """Reproduce el bug del screenshot: un FVG pegado al spot (por debajo del BE)
    no debe usarse como objetivo; cae al fallback BE + k·ATR."""
    # FVG alcista justo en el spot (mid 8505), muy por debajo del BE (~8705)
    fvg  = FairValueGap(type="bullish", high=8510.0, low=8500.0, filled=False)
    snap = _make_snap(fvgs=[fvg], bos="BOS_UP")
    ctx  = _make_context(trend="alcista", momentum_pct=3.5, signal_strength="fuerte")
    chain = _make_chain(spot=8500.0)
    idea  = build_directional_idea(ctx, chain, 8500.0, snap=snap)
    assert idea is not None
    assert idea.target_spot > idea.breakeven
    assert "FVG" not in idea.rational.split("Objetivo:")[1].split("Stop:")[0]


@patch("optionsdesk.signals.directional_ideas.days_to_expiry", side_effect=_mock_days)
def test_low_rr_setup_returns_none(mock_days):
    """Objetivo apenas sobre el BE + stop estructural lejano (capado) → R/R < 1.5 → None."""
    fvg  = FairValueGap(type="bullish", high=8720.0, low=8700.0, filled=False)  # mid 8710 ~ BE
    ob   = OrderBlock(type="bullish", high=7100.0, low=7000.0)                  # stop capado a 2×ATR
    snap = _make_snap(fvgs=[fvg], order_block=ob, bos="BOS_UP")
    ctx  = _make_context(trend="alcista", momentum_pct=3.5, signal_strength="fuerte")
    chain = _make_chain(spot=8500.0)
    idea  = build_directional_idea(ctx, chain, 8500.0, snap=snap)
    assert idea is None


@patch("optionsdesk.signals.directional_ideas.days_to_expiry", side_effect=_mock_days)
def test_mtf_conflict_returns_none(mock_days):
    ctx = _make_context(trend="alcista", momentum_pct=3.5, signal_strength="fuerte")
    ctx.mtf_alignment = "conflicto"
    chain = _make_chain(spot=8500.0)
    idea  = build_directional_idea(ctx, chain, 8500.0)
    assert idea is None


@patch("optionsdesk.signals.directional_ideas.days_to_expiry", side_effect=_mock_days)
def test_confluence_lists_aligned_factors(mock_days):
    fvg  = FairValueGap(type="bullish", high=9100.0, low=9000.0, filled=False)  # mid 9050 > BE
    ob   = OrderBlock(type="bullish", high=8350.0, low=8300.0)
    snap = _make_snap(fvgs=[fvg], order_block=ob, bos="BOS_UP")
    ctx  = _make_context(trend="alcista", momentum_pct=3.5, signal_strength="fuerte")
    chain = _make_chain(spot=8500.0)
    idea  = build_directional_idea(ctx, chain, 8500.0, snap=snap)
    assert idea is not None
    assert idea.rr_ratio >= 1.5
    assert "BOS en dirección" in idea.confluence
    assert "objetivo en FVG de liquidez" in idea.confluence
    assert "señal fuerte" in idea.confluence


# ── v2.6: _naked_pop fallback cap ─────────────────────────────────────────────

def test_naked_pop_fallback_capped_sigma():
    """Con ATR extremo (10%) y sin realized_vol, sigma se acota a 1.50 max."""
    from optionsdesk.signals.directional_ideas import _naked_pop

    # ATR=10% → sigma_proxy = 0.10 * sqrt(252) ≈ 1.587 > cap(1.50)
    ctx = _make_context(trend="alcista", atr_pct=10.0, momentum_pct=3.0)

    # PoP con sigma=1.50 (capada)
    pop_capped = _naked_pop(
        spot=8300.0, breakeven=8900.0, days=30,
        is_call=True, context=ctx, realized_vol=None,
    )
    # PoP directa con sigma=1.587 (sin cap)
    from optionsdesk.core.pricing import risk_neutral_prob_above
    import math
    sigma_uncapped = 0.10 * math.sqrt(252)
    pop_uncapped = risk_neutral_prob_above(8300.0, 8900.0, 30/365, 0.0, sigma_uncapped)

    # Con cap la PoP debe ser diferente (la sigma fue truncada)
    assert pop_capped != pytest.approx(pop_uncapped, rel=1e-4), (
        "el cap no tuvo efecto — la sigma no se recortó"
    )
    # Además debe ser un número razonable, no 0 ni 1
    assert 0.01 < pop_capped < 0.99
