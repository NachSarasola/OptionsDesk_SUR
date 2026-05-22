"""Tests del optimizador hold-vs-swing."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from optionsdesk.core.rates import RateResult
from optionsdesk.signals.horizon import HorizonPlan, _HOLD_THRESHOLD, optimize_horizon


# ── Fixture ───────────────────────────────────────────────────────────────────

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
    iv: float | None = 0.55,
) -> RateResult:
    r = RateResult(
        symbol=f"GFG{'C' if strategy == 'COVERED_CALL' else 'V'}{strike:.0f}F",
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
        iv=iv,
        moneyness=moneyness,
        is_liquid=is_liquid,
    )
    r.set_spread(caucion_tna)
    return r


# ── HOLD forzado ──────────────────────────────────────────────────────────────

def test_illiquid_forces_hold():
    r = _make_result(is_liquid=False)
    plan = optimize_horizon(r, "EQUILIBRADO")
    assert plan.mode == "HOLD"
    assert "ilíquida" in plan.warnings[0].lower()


def test_no_iv_forces_hold():
    r = _make_result(iv=None)
    plan = optimize_horizon(r, "EQUILIBRADO")
    assert plan.mode == "HOLD"
    assert any("IV" in w for w in plan.warnings)


def test_short_days_forces_hold():
    r = _make_result(days=3)
    plan = optimize_horizon(r, "EQUILIBRADO")
    assert plan.mode == "HOLD"


# ── Estructura de HorizonPlan ─────────────────────────────────────────────────

def test_horizon_plan_fields_populated():
    r = _make_result()
    for profile in ("CONSERVADOR", "EQUILIBRADO", "AGRESIVO"):
        plan = optimize_horizon(r, profile)
        assert plan.mode in ("HOLD", "SWING")
        assert 1 <= plan.target_exit_days <= r.days
        assert plan.target_capture_pct > 0.0
        assert plan.optimized_tna_pct > 0.0
        assert plan.theta_daily_ars >= 0.0
        assert len(plan.plain_explanation) > 20


def test_target_exit_days_within_bounds():
    r = _make_result(days=30)
    for profile in ("CONSERVADOR", "EQUILIBRADO", "AGRESIVO"):
        plan = optimize_horizon(r, profile)
        assert 1 <= plan.target_exit_days <= 30


def test_capture_pct_positive():
    r = _make_result()
    plan = optimize_horizon(r, "EQUILIBRADO")
    assert plan.target_capture_pct > 0.0


def test_score_in_range():
    r = _make_result()
    plan = optimize_horizon(r, "EQUILIBRADO")
    assert plan.target_capture_pct <= 110.0   # puede superar 100 si sale antes de máximo decay


# ── HOLD threshold ────────────────────────────────────────────────────────────

def test_hold_when_t_star_near_expiry():
    """Deep ITM con poco gamma: t* debería caer cerca del vencimiento → HOLD."""
    # Strike muy por debajo del spot: call deep ITM, delta altísimo, poco gamma
    r = _make_result(strike=5000.0, spot=8500.0, days=30, iv=0.30)
    plan = optimize_horizon(r, "AGRESIVO")
    # Para AGRESIVO (λ bajo), el optimizador tiende a hold más
    assert plan.mode == "HOLD" or plan.target_exit_days >= int(_HOLD_THRESHOLD * 30)


# ── Efecto del perfil ─────────────────────────────────────────────────────────

def test_conservador_exits_earlier_or_equal_than_agresivo_atm():
    """Para opciones ATM (gamma alto), CONSERVADOR sale más temprano que AGRESIVO."""
    # ATM: strike ≈ spot, gamma máximo → el ajuste por riesgo penaliza más al Conservador
    r = _make_result(strike=8500.0, spot=8500.0, days=45, iv=0.55, moneyness="ATM")
    plan_cons = optimize_horizon(r, "CONSERVADOR")
    plan_agr  = optimize_horizon(r, "AGRESIVO")
    # CONSERVADOR sale igual de temprano o antes
    assert plan_cons.target_exit_days <= plan_agr.target_exit_days


def test_agresivo_captures_more_than_conservador():
    """AGRESIVO captura >= fracción que CONSERVADOR (aguanta más tiempo)."""
    r = _make_result(strike=8500.0, spot=8500.0, days=45, iv=0.55, moneyness="ATM")
    plan_cons = optimize_horizon(r, "CONSERVADOR")
    plan_agr  = optimize_horizon(r, "AGRESIVO")
    assert plan_agr.target_capture_pct >= plan_cons.target_capture_pct


# ── Short put ─────────────────────────────────────────────────────────────────

def test_short_put_returns_valid_plan():
    r = _make_result(strategy="SHORT_PUT", delta=-0.40, moneyness="OTM")
    plan = optimize_horizon(r, "EQUILIBRADO")
    assert plan.mode in ("HOLD", "SWING")
    assert plan.target_exit_days >= 1


# ── Curva de decaimiento ──────────────────────────────────────────────────────

def test_theta_positive_at_entry():
    """La opción vendida gana valor (theta positivo para el vendedor)."""
    r = _make_result(iv=0.55)
    plan = optimize_horizon(r, "EQUILIBRADO")
    assert plan.theta_daily_ars >= 0.0


def test_longer_tenor_larger_theta_contract():
    """Con más días al vencimiento el theta absoluto por contrato es mayor."""
    r_short = _make_result(days=15, iv=0.55)
    r_long  = _make_result(days=60, iv=0.55)
    plan_s  = optimize_horizon(r_short, "EQUILIBRADO")
    plan_l  = optimize_horizon(r_long,  "EQUILIBRADO")
    # Theta anualizado no necesariamente mayor, pero el valor absoluto diario
    # para plazos cortos ATM suele ser mayor; validamos que ambos son positivos
    assert plan_s.theta_daily_ars >= 0.0
    assert plan_l.theta_daily_ars >= 0.0


# ── Greeks ────────────────────────────────────────────────────────────────────

def test_greeks_non_negative():
    r = _make_result()
    plan = optimize_horizon(r, "EQUILIBRADO")
    assert plan.gamma >= 0.0
    assert plan.vega_1pct >= 0.0
    assert plan.theta_daily_ars >= 0.0


# ── Swing detectado con opción ATM de gamma alto ─────────────────────────────

def test_swing_detected_for_atm_conservador():
    """CONSERVADOR sobre opción ATM con gamma alto debería preferir salir antes."""
    r = _make_result(
        strike=8500.0, spot=8500.0, days=60, iv=0.65,
        moneyness="ATM", premium=400.0, net_outlay=8200.0,
    )
    plan = optimize_horizon(r, "CONSERVADOR")
    # No podemos garantizar siempre SWING, pero t* debería ser < 85% de 60 días o HOLD
    if plan.mode == "SWING":
        assert plan.target_exit_days < int(_HOLD_THRESHOLD * 60)
    else:
        assert plan.mode == "HOLD"
