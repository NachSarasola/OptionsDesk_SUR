"""Tests para strategies/vertical_spread.py."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from unittest.mock import patch

import pytest

from optionsdesk.core.benchmark import Benchmark
from optionsdesk.data.providers.base import OptionsChain, Quote
from optionsdesk.strategies.vertical_spread import VerticalSpreadConfig, VerticalSpreadScanner

TODAY = date(2026, 6, 20)   # fijo — inmune a vencimientos reales

_CAUCION = Benchmark(caucion_tna_pct=70.0, days=1)


def _make_quote(bid: float, ask: float, last: float = 0.0, volume: float = 100.0) -> Quote:
    return Quote(
        symbol="", bid=bid, ask=ask, last=last or (bid + ask) / 2,
        volume=volume, timestamp=datetime.now(),
    )


def _make_chain(spot: float, options: dict[str, tuple]) -> OptionsChain:
    """Construye una OptionsChain sintetica.

    options: {symbol: (bid, ask)}
    El simbolo debe seguir el formato BYMA: GFG[CV]<strike><code>
    Ejemplo: GFGC8000F, GFGC8500F, GFGV8000F
    """
    spot_q = _make_quote(spot * 0.999, spot * 1.001, spot)
    opts = {sym: _make_quote(b, a) for sym, (b, a) in options.items()}
    return OptionsChain(
        underlying="GGAL",
        spot=spot_q,
        options=opts,
        timestamp=datetime.now(),
    )


def _make_calendar():
    """Calendario minimo con codigo F = 30 dias desde hoy.

    Usa date.today() (no TODAY fijo) para que days_to_expiry en OptionContract
    devuelva ~30 independientemente de la fecha del sistema.
    """
    return {"F": date.today() + timedelta(days=30)}


SPOT = 8300.0
_CAL = _make_calendar()


def test_scanner_finds_bull_call_spread():
    """Con dos calls liquidas el scanner arma un bull call spread."""
    chain = _make_chain(SPOT, {
        "GFGC8000F": (280.0, 320.0),   # K=8000, bid=280
        "GFGC8500F": (130.0, 170.0),   # K=8500
    })
    scanner = VerticalSpreadScanner(expiry_calendar=_CAL)
    results = scanner.scan(chain, _CAUCION, direction="bullish")
    assert len(results) > 0
    bull_calls = [r for r in results if r.strategy == "BULL_CALL"]
    assert len(bull_calls) > 0


def test_scanner_finds_bear_call_spread():
    chain = _make_chain(SPOT, {
        "GFGC8000F": (280.0, 320.0),
        "GFGC8500F": (130.0, 170.0),
    })
    scanner = VerticalSpreadScanner(expiry_calendar=_CAL)
    results = scanner.scan(chain, _CAUCION, direction="bearish")
    bear_calls = [r for r in results if r.strategy == "BEAR_CALL"]
    assert len(bear_calls) > 0


def test_direction_bullish_excludes_bear():
    chain = _make_chain(SPOT, {
        "GFGC8000F": (280.0, 320.0),
        "GFGC8500F": (130.0, 170.0),
    })
    scanner = VerticalSpreadScanner(expiry_calendar=_CAL)
    results = scanner.scan(chain, _CAUCION, direction="bullish")
    bear_strategies = {r.strategy for r in results if "BEAR" in r.strategy}
    assert not bear_strategies


def test_direction_bearish_excludes_bull():
    chain = _make_chain(SPOT, {
        "GFGC8000F": (280.0, 320.0),
        "GFGC8500F": (130.0, 170.0),
    })
    scanner = VerticalSpreadScanner(expiry_calendar=_CAL)
    results = scanner.scan(chain, _CAUCION, direction="bearish")
    bull_strategies = {r.strategy for r in results if "BULL" in r.strategy}
    assert not bull_strategies


def test_illiquid_leg_excluded():
    """Una pata con bid=0 no debe generar spreads."""
    chain = _make_chain(SPOT, {
        "GFGC8000F": (0.0, 320.0),    # bid=0 → iliquida
        "GFGC8500F": (130.0, 170.0),
    })
    scanner = VerticalSpreadScanner(expiry_calendar=_CAL)
    results = scanner.scan(chain, _CAUCION)
    assert results == []


def test_wide_spread_leg_excluded():
    """Spread bid-ask > max_spread_pct → pata excluida."""
    cfg = VerticalSpreadConfig(max_spread_pct=10.0)   # umbral bajo
    chain = _make_chain(SPOT, {
        "GFGC8000F": (100.0, 500.0),  # 40% spread → iliquida con umbral 10%
        "GFGC8500F": (130.0, 150.0),
    })
    scanner = VerticalSpreadScanner(config=cfg, expiry_calendar=_CAL)
    results = scanner.scan(chain, _CAUCION)
    assert results == []


def test_sorted_by_ev_return_pct():
    """Los resultados deben estar ordenados por ev_return_pct descendente."""
    chain = _make_chain(SPOT, {
        "GFGC7500F": (550.0, 590.0),
        "GFGC8000F": (280.0, 320.0),
        "GFGC8500F": (130.0, 170.0),
    })
    scanner = VerticalSpreadScanner(expiry_calendar=_CAL)
    results = scanner.scan(chain, _CAUCION)
    evs = [r.ev_return_pct for r in results]
    assert evs == sorted(evs, reverse=True)


def test_scanner_realized_vol_param():
    """realized_vol se acepta sin error y los campos EV se computan."""
    chain = _make_chain(SPOT, {
        "GFGC8000F": (280.0, 320.0),
        "GFGC8500F": (130.0, 170.0),
    })
    scanner = VerticalSpreadScanner(expiry_calendar=_CAL)
    results = scanner.scan(chain, _CAUCION, direction="bullish", realized_vol=0.50)
    assert len(results) > 0
    sr = results[0]
    assert isinstance(sr.expected_value, float)
    assert isinstance(sr.ev_return_pct, float)


def test_empty_chain_returns_empty():
    chain = _make_chain(SPOT, {})
    scanner = VerticalSpreadScanner(expiry_calendar=_CAL)
    assert scanner.scan(chain, _CAUCION) == []


def test_spread_result_has_ev_fields():
    """SpreadResult debe tener expected_value y ev_return_pct."""
    chain = _make_chain(SPOT, {
        "GFGC8000F": (280.0, 320.0),
        "GFGC8500F": (130.0, 170.0),
    })
    scanner = VerticalSpreadScanner(expiry_calendar=_CAL)
    results = scanner.scan(chain, _CAUCION, direction="bullish")
    assert len(results) > 0
    sr = results[0]
    assert hasattr(sr, "expected_value")
    assert hasattr(sr, "ev_return_pct")
    # EV acotado al rango del payoff
    assert -sr.max_loss <= sr.expected_value <= sr.max_profit


def test_spread_result_fields_populated():
    chain = _make_chain(SPOT, {
        "GFGC8000F": (280.0, 320.0),
        "GFGC8500F": (130.0, 170.0),
    })
    scanner = VerticalSpreadScanner(expiry_calendar=_CAL)
    results = scanner.scan(chain, _CAUCION, direction="bullish")
    if results:
        sr = results[0]
        assert sr.max_profit > 0
        assert sr.max_loss > 0
        assert 0.0 <= sr.prob_of_profit <= 1.0
        assert sr.risk_reward > 0
        assert sr.days > 0
        assert sr.long_leg.action == "BUY"
        assert sr.short_leg.action == "SELL"


# ── v2.5: pares no-adyacentes ─────────────────────────────────────────────────

def test_scanner_finds_non_adjacent_pair():
    """K=8000 y K=8500 líquidas; K=8250 ilíquida (bid=0) → 8000/8500 debe aparecer."""
    chain = _make_chain(SPOT, {
        "GFGC8000F": (280.0, 320.0),  # líquida
        "GFGC8250F": (0.0, 200.0),    # ilíquida: bid=0 → excluida por _is_liquid
        "GFGC8500F": (130.0, 170.0),  # líquida
    })
    scanner = VerticalSpreadScanner(expiry_calendar=_CAL)
    results = scanner.scan(chain, _CAUCION, direction="bullish")

    strikes_found = {
        (r.long_leg.strike, r.short_leg.strike)
        for r in results
        if r.strategy == "BULL_CALL"
    }
    assert (8000.0, 8500.0) in strikes_found


def test_max_width_filters_far_pairs():
    """Con max_width_pct muy pequeño, pares con ancho excesivo no aparecen."""
    cfg = VerticalSpreadConfig(max_width_pct=0.03)  # 3% de 8300 ≈ 249
    chain = _make_chain(SPOT, {
        "GFGC8000F": (280.0, 320.0),
        "GFGC8200F": (190.0, 230.0),
        "GFGC8500F": (130.0, 170.0),
    })
    scanner = VerticalSpreadScanner(config=cfg, expiry_calendar=_CAL)
    results = scanner.scan(chain, _CAUCION, direction="bullish")

    # 8000/8500 (Δ=500 > 249) no debe aparecer
    wide_pairs = [
        r for r in results
        if r.strategy == "BULL_CALL"
        and abs(r.short_leg.strike - r.long_leg.strike) > 249
    ]
    assert wide_pairs == []
