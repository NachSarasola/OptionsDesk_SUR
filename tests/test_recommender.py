"""Tests del motor de recomendacion."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from optionsdesk.core.benchmark import Benchmark
from optionsdesk.core.rates import RateResult
from optionsdesk.risk.limits import RiskChecker
from optionsdesk.signals.recommender import (
    Recommendation,
    Recommender,
    RiskProfile,
    _diversify,
    _is_redundant,
    _passes_gates,
    _probability,
    _score,
    _semaphore,
    _ticket,
)


# ── Fixture de RateResult ─────────────────────────────────────────────────────

def _make_result(
    strategy: str = "COVERED_CALL",
    tna_pct: float = 90.0,
    cushion_pct: float = 8.0,
    delta: float | None = 0.70,
    moneyness: str = "ITM",
    is_liquid: bool = True,
    days: int = 30,
    strike: float = 8000.0,
    spot: float = 8500.0,
    premium: float = 300.0,
    net_outlay: float = 8200.0,
    net_proceeds: float = 8290.0,
    period_rate_pct: float = 1.1,
    caucion_tna: float = 60.0,
    symbol: str | None = None,
) -> RateResult:
    sym = symbol or f"GFG{'C' if strategy == 'COVERED_CALL' else 'V'}{strike:.0f}F"
    r = RateResult(
        symbol=sym,
        strategy=strategy,
        strike=strike,
        expiration=date.today() + timedelta(days=days),
        days=days,
        spot=spot,
        premium=premium,
        net_outlay=net_outlay,
        net_proceeds=net_proceeds,
        period_rate_pct=period_rate_pct,
        tna_pct=tna_pct,
        tea_pct=tna_pct * 1.05,
        cushion_pct=cushion_pct,
        delta=delta,
        iv=0.55,
        moneyness=moneyness,
        is_liquid=is_liquid,
    )
    r.set_spread(caucion_tna)
    return r


@pytest.fixture
def benchmark() -> Benchmark:
    return Benchmark(caucion_tna_pct=60.0, days=30)


# ── Probabilidad ──────────────────────────────────────────────────────────────

def test_probability_covered_call_uses_delta():
    r = _make_result(strategy="COVERED_CALL", delta=0.75)
    assert _probability(r) == pytest.approx(0.75)


def test_probability_short_put_uses_one_minus_abs_delta():
    r = _make_result(strategy="SHORT_PUT", delta=-0.38)
    assert _probability(r) == pytest.approx(0.62)


def test_probability_no_delta_itm():
    r = _make_result(delta=None, moneyness="ITM")
    assert _probability(r) == pytest.approx(0.70)


def test_probability_no_delta_atm():
    r = _make_result(delta=None, moneyness="ATM")
    assert _probability(r) == pytest.approx(0.55)


def test_probability_no_delta_otm():
    r = _make_result(delta=None, moneyness="OTM")
    assert _probability(r) == pytest.approx(0.40)


# ── Gates duros ───────────────────────────────────────────────────────────────

def test_gate_illiquid_fails_all_profiles():
    r = _make_result(is_liquid=False)
    for profile in RiskProfile:
        assert not _passes_gates(r, profile)


def test_gate_negative_spread_fails():
    r = _make_result(tna_pct=50.0, caucion_tna=60.0)  # spread = -10
    assert not _passes_gates(r, RiskProfile.AGRESIVO)


def test_gate_cushion_fails_conservador():
    r = _make_result(cushion_pct=3.0)
    assert not _passes_gates(r, RiskProfile.CONSERVADOR)


def test_gate_cushion_passes_equilibrado():
    r = _make_result(cushion_pct=3.0)
    assert _passes_gates(r, RiskProfile.EQUILIBRADO)


def test_gate_cushion_passes_agresivo():
    r = _make_result(cushion_pct=0.5)
    assert _passes_gates(r, RiskProfile.AGRESIVO)


def test_gate_days_exceeds_max():
    r = _make_result(days=95)
    for profile in RiskProfile:
        assert not _passes_gates(r, profile)


def test_gate_days_below_min():
    r = _make_result(days=3)
    for profile in RiskProfile:
        assert not _passes_gates(r, profile)


def test_gate_probability_too_low_conservador():
    r = _make_result(delta=0.40, cushion_pct=8.0)
    assert not _passes_gates(r, RiskProfile.CONSERVADOR)


def test_gate_probability_too_low_equilibrado():
    r = _make_result(delta=0.45)
    assert not _passes_gates(r, RiskProfile.EQUILIBRADO)


def test_gate_otm_lottery_fails_all_profiles():
    r = _make_result(
        tna_pct=500.0, delta=0.08, moneyness="OTM", cushion_pct=1.0,
        caucion_tna=60.0,
    )
    for profile in RiskProfile:
        assert not _passes_gates(r, profile)


# ── Scoring ───────────────────────────────────────────────────────────────────

def test_solid_itm_scores_higher_than_otm_lottery_equilibrado():
    solid = _make_result(tna_pct=90.0, delta=0.72, cushion_pct=8.0, caucion_tna=60.0)
    lottery = _make_result(
        tna_pct=500.0, delta=0.08, moneyness="OTM", cushion_pct=1.0, caucion_tna=60.0
    )
    assert _score(solid, RiskProfile.EQUILIBRADO) > _score(lottery, RiskProfile.EQUILIBRADO)


def test_conservador_weights_cushion_over_spread():
    high_cushion = _make_result(cushion_pct=14.0, tna_pct=70.0, delta=0.70, caucion_tna=60.0)
    high_spread = _make_result(cushion_pct=6.0, tna_pct=110.0, delta=0.65, caucion_tna=60.0)
    assert _score(high_cushion, RiskProfile.CONSERVADOR) > _score(
        high_spread, RiskProfile.CONSERVADOR
    )


def test_agresivo_weights_ev_over_cushion():
    # v2.4: AGRESIVO pesa EV (0.55) sobre cushion (0.10). Con EV calculado,
    # un trade de EV alta pero cushion baja supera a uno de cushion alta y EV cero.
    high_ev = _make_result(cushion_pct=2.0, tna_pct=110.0, delta=0.45, caucion_tna=60.0)
    high_ev.expected_value_pct = 40.0   # EV anual alta — VRP positivo
    high_cushion = _make_result(cushion_pct=14.0, tna_pct=70.0, delta=0.70, caucion_tna=60.0)
    high_cushion.expected_value_pct = 0.0   # EV neutra
    assert _score(high_ev, RiskProfile.AGRESIVO) > _score(
        high_cushion, RiskProfile.AGRESIVO
    )


def test_score_in_range():
    r = _make_result()
    for profile in RiskProfile:
        s = _score(r, profile)
        assert 0.0 <= s <= 100.0


def test_illiquid_result_has_lower_score():
    liquid = _make_result(is_liquid=True)
    illiquid = _make_result(is_liquid=False)
    for profile in RiskProfile:
        assert _score(liquid, profile) > _score(illiquid, profile)


# ── Semáforo ──────────────────────────────────────────────────────────────────

def test_semaphore_verde_strong_itm():
    r = _make_result(delta=0.78, cushion_pct=9.0, tna_pct=90.0, caucion_tna=60.0)
    checker = RiskChecker()
    s = _score(r, RiskProfile.EQUILIBRADO)
    light = _semaphore(r, s, checker)
    assert light in ("verde", "amarillo")


def test_semaphore_not_verde_negative_spread():
    # Con spread negativo y delta bajo, el semáforo no puede ser verde.
    # El gate _passes_gates ya bloquea este trade; si se llegara a puntuar,
    # la probabilidad baja (0.20) impide que supere el umbral de verde.
    r = _make_result(tna_pct=10.0, caucion_tna=60.0, delta=0.20)
    checker = RiskChecker()
    s = _score(r, RiskProfile.AGRESIVO)
    assert _semaphore(r, s, checker) != "verde"


# ── Ticket ────────────────────────────────────────────────────────────────────

def test_ticket_covered_call_has_buy_and_sell():
    r = _make_result(strategy="COVERED_CALL")
    text = _ticket(r, contracts=2)
    assert "COMPRA" in text
    assert "GGAL" in text
    assert "VENTA" in text
    assert r.symbol in text


def test_ticket_short_put_has_venta_and_garantia():
    r = _make_result(strategy="SHORT_PUT")
    text = _ticket(r, contracts=1)
    assert "VENTA" in text
    assert "Garantia" in text


# ── _diversify ────────────────────────────────────────────────────────────────

def test_diversify_deduplicates_same_symbol():
    r1 = _make_result(symbol="SYM1", strike=8000.0)
    r2 = _make_result(symbol="SYM1", strike=8000.0)   # mismo símbolo
    r3 = _make_result(symbol="SYM2", strike=8500.0)
    scored = [(r1, 80.0), (r2, 75.0), (r3, 70.0)]
    result = _diversify(scored, n=3)
    symbols = [r.symbol for r, _ in result]
    assert len(result) == 2
    assert symbols.count("SYM1") == 1


def test_diversify_deduplicates_near_strikes():
    expiry = date.today() + timedelta(days=30)
    r1 = _make_result(strike=8000.0, symbol="SYM1", strategy="COVERED_CALL")
    r2 = _make_result(strike=8100.0, symbol="SYM2", strategy="COVERED_CALL")   # 1.25% dist → dup
    r3 = _make_result(strike=9000.0, symbol="SYM3", strategy="COVERED_CALL")   # 12.5% dist → ok
    # Mismo vencimiento para activar el dedup
    r1.expiration = expiry
    r2.expiration = expiry
    r3.expiration = expiry
    scored = [(r1, 90.0), (r2, 85.0), (r3, 70.0)]
    result = _diversify(scored, n=3)
    assert len(result) == 2
    strikes = {r.strike for r, _ in result}
    assert 9000.0 in strikes


def test_diversify_respects_n():
    results = [(_make_result(symbol=f"S{i}", strike=float(8000 + i * 500)), float(90 - i))
               for i in range(6)]
    chosen = _diversify(results, n=3)
    assert len(chosen) == 3


def test_is_redundant_same_symbol():
    r1 = _make_result(symbol="SYM1")
    r2 = _make_result(symbol="SYM1")
    assert _is_redundant(r2, [r1])


def test_is_redundant_different_symbol_far_strike():
    r1 = _make_result(symbol="SYM1", strike=8000.0)
    r2 = _make_result(symbol="SYM2", strike=9500.0)
    assert not _is_redundant(r2, [r1])


# ── Recommender end-to-end (ahora devuelve listas) ────────────────────────────

def test_recommender_returns_all_profiles(benchmark):
    cc = [_make_result()]
    sp = [_make_result(strategy="SHORT_PUT", delta=-0.55)]
    rec = Recommender()
    results = rec.recommend(cc, sp, benchmark)
    assert set(results.keys()) == set(RiskProfile)


def test_recommender_returns_lists(benchmark):
    cc = [_make_result()]
    sp = [_make_result(strategy="SHORT_PUT", delta=-0.55)]
    rec = Recommender()
    results = rec.recommend(cc, sp, benchmark)
    for profile in RiskProfile:
        assert isinstance(results[profile], list)


def test_recommender_empty_list_when_no_valid_candidates(benchmark):
    illiquid = [_make_result(is_liquid=False)]
    rec = Recommender()
    results = rec.recommend(illiquid, illiquid, benchmark)
    assert all(v == [] for v in results.values())


def test_recommender_top_n_limits_results(benchmark):
    # Crea 5 candidatos distintos por strike
    cc = [
        _make_result(strike=float(7500 + i * 200), symbol=f"GFGC{7500+i*200}F", delta=0.72)
        for i in range(5)
    ]
    rec = Recommender()
    results = rec.recommend(cc, [], benchmark, top_n=3)
    for profile in RiskProfile:
        assert len(results[profile]) <= 3


def test_recommender_with_capital_computes_profit(benchmark):
    cc = [_make_result(net_outlay=8200.0, period_rate_pct=1.1)]
    sp = [_make_result(strategy="SHORT_PUT", delta=-0.60, net_outlay=7800.0)]
    rec = Recommender()
    results = rec.recommend(cc, sp, benchmark, capital=5_000_000.0)
    for recs_list in results.values():
        for r in recs_list:
            assert r.expected_profit_ars is not None
            assert r.expected_profit_ars > 0
            assert r.contracts >= 1


def test_recommender_does_not_force_contract_when_cap_is_insufficient(benchmark):
    cc = [_make_result(net_outlay=8200.0, period_rate_pct=1.1)]
    rec = Recommender()
    results = rec.recommend(cc, [], benchmark, capital=1_000_000.0)

    for recs_list in results.values():
        for r in recs_list:
            assert r.expected_profit_ars == 0
            assert r.contracts == 0


def test_recommender_ticket_non_empty(benchmark):
    cc = [_make_result()]
    sp = [_make_result(strategy="SHORT_PUT", delta=-0.55)]
    rec = Recommender()
    results = rec.recommend(cc, sp, benchmark)
    for recs_list in results.values():
        for r in recs_list:
            assert len(r.ticket_text) > 20


def test_conservador_selects_safer_than_agresivo(benchmark):
    safe_cc = _make_result(
        cushion_pct=12.0, tna_pct=72.0, delta=0.75, caucion_tna=60.0,
        net_outlay=8300.0, symbol="GFGC8000SAFE",
    )
    risky_cc = _make_result(
        cushion_pct=1.5, tna_pct=120.0, delta=0.55, moneyness="OTM",
        caucion_tna=60.0, net_outlay=8300.0, strike=9500.0, symbol="GFGC9500RISKY",
    )
    rec = Recommender()
    results = rec.recommend([safe_cc, risky_cc], [], benchmark)
    cons_recs = results[RiskProfile.CONSERVADOR]
    if cons_recs:
        assert cons_recs[0].result.cushion_pct >= 5.0


def test_recommendation_fields_populated(benchmark):
    cc = [_make_result()]
    rec = Recommender()
    results = rec.recommend(cc, [], benchmark)
    for recs_list in results.values():
        for r in recs_list:
            assert r.headline
            assert r.plain_explanation
            assert len(r.action_steps) >= 3
            assert 0.0 <= r.success_probability <= 1.0
            assert 0.0 <= r.score <= 100.0
            assert r.light in ("verde", "amarillo", "rojo")
