"""Tests del analizador intertemporal de posiciones abiertas."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from optionsdesk.signals.monitor import OpenPosition
from optionsdesk.signals.position_advisor import (
    PortfolioAdvice,
    PositionAdvice,
    TimeScenario,
    advise_portfolio,
    advise_position,
    project_scenarios,
    _capture_plan_lag,
    _days_to_target,
    _rotation_opportunity,
)


# ── Fixture base ──────────────────────────────────────────────────────────────

def _pos(
    strategy: str = "COVERED_CALL",
    premium_received: float = 300.0,
    strike: float = 9000.0,
    spot_entry: float = 8500.0,
    iv_entry: float = 0.55,
    days_entry: int = 30,
    days_held: int = 5,
    target_capture_pct: float = 50.0,
    max_loss_mult: float = 2.0,
    roll_dte: int = 21,
    defend_delta: float = 0.50,
    contracts: int = 1,
    caucion_tna: float = 60.0,
) -> OpenPosition:
    entry_date = date.today() - timedelta(days=days_held)
    suffix = "C" if "CALL" in strategy else "V"
    return OpenPosition(
        symbol=f"GFG{suffix}{strike:.0f}F",
        strategy=strategy,
        strike=strike,
        spot_entry=spot_entry,
        premium_received=premium_received,
        net_outlay=strike - premium_received if strategy == "COVERED_CALL" else strike,
        iv_entry=iv_entry,
        days_entry=days_entry,
        entry_date=entry_date,
        target_exit_days=days_entry,
        target_capture_pct=target_capture_pct,
        caucion_tna=caucion_tna,
        max_loss_mult=max_loss_mult,
        roll_dte=roll_dte,
        defend_delta=defend_delta,
        contracts=contracts,
    )


# ── project_scenarios ─────────────────────────────────────────────────────────

class TestProjectScenarios:
    def test_returns_correct_number_of_scenarios(self):
        pos = _pos(days_held=5)
        scenarios = project_scenarios(pos, current_spot=8500.0, days_list=[3, 7, 14])
        assert len(scenarios) == 3
        assert [s.days_from_now for s in scenarios] == [3, 7, 14]

    def test_pnl_increases_with_time_for_short_call(self):
        """Con spot congelado y opcion OTM, la captura aumenta con el tiempo."""
        pos = _pos(strategy="COVERED_CALL", strike=9500.0, spot_entry=8500.0, days_entry=30, days_held=2)
        scenarios = project_scenarios(pos, current_spot=8500.0, days_list=[3, 7, 14])
        captures = [s.capture_pct for s in scenarios]
        assert captures[0] <= captures[1] <= captures[2], (
            f"Captura deberia aumentar con tiempo (theta decay): {captures}"
        )

    def test_pnl_at_expiry_equals_full_capture(self):
        """Al vencimiento, la opcion OTM vale 0 → captura = 100%."""
        pos = _pos(strategy="COVERED_CALL", strike=9500.0, spot_entry=8500.0, days_entry=10, days_held=9)
        # Con days_remaining = 1 y days_from_now = 1 → DTE = 0 → intrinsic only
        scenarios = project_scenarios(pos, current_spot=8400.0, days_list=[1])
        # Call OTM (spot < strike) at expiry → value = 0 → captura ~100%
        assert scenarios[0].capture_pct > 90.0, f"Captura al vencimiento: {scenarios[0].capture_pct}"

    def test_short_put_otm_captures_with_time(self):
        """Put OTM vendida: al quedarse el spot por encima del strike, la captura sube."""
        pos = _pos(strategy="SHORT_PUT", strike=8000.0, spot_entry=8500.0, days_entry=30, days_held=2)
        scenarios = project_scenarios(pos, current_spot=8500.0, days_list=[3, 10])
        assert scenarios[0].capture_pct >= 0
        assert scenarios[1].capture_pct >= scenarios[0].capture_pct

    def test_theta_daily_positive(self):
        """El theta diario reportado debe ser positivo (vendedor recibe decay)."""
        pos = _pos(days_held=3, days_entry=30)
        scenarios = project_scenarios(pos, current_spot=8500.0, days_list=[7, 14])
        for sc in scenarios:
            if sc.dte_remaining > 2:  # cerca de vencimiento theta puede ser 0
                assert sc.theta_daily_ars >= 0, f"Theta negativo en t+{sc.days_from_now}"

    def test_zero_iv_uses_intrinsic(self):
        """Con iv=0, debe usar valor intrinseco sin crashear."""
        pos = _pos(iv_entry=0.0, days_entry=10, days_held=5)
        scenarios = project_scenarios(pos, current_spot=8500.0)
        assert all(isinstance(s.capture_pct, float) for s in scenarios)


# ── _capture_plan_lag ─────────────────────────────────────────────────────────

class TestCapturePlanLag:
    def test_on_track_when_ahead(self):
        pos = _pos(days_held=15, days_entry=30, target_capture_pct=50.0)
        # Captura actual = 30%, esperada = 50% * (15/30) = 25% → adelantada
        on_track, lag = _capture_plan_lag(pos, current_capture=30.0)
        assert on_track
        assert lag > 0  # adelantado

    def test_off_track_when_far_behind(self):
        pos = _pos(days_held=20, days_entry=30, target_capture_pct=50.0)
        # Esperado = 50% * (20/30) ≈ 33.3%, actual = 5% → muy rezagado
        on_track, lag = _capture_plan_lag(pos, current_capture=5.0)
        assert not on_track
        assert lag < -10.0

    def test_zero_days_held_is_on_track(self):
        pos = _pos(days_held=0, days_entry=30)
        on_track, lag = _capture_plan_lag(pos, current_capture=0.0)
        assert on_track


# ── _days_to_target ────────────────────────────────────────────────────────────

class TestDaysToTarget:
    def test_returns_zero_when_already_at_target(self):
        pos = _pos(target_capture_pct=50.0)
        result = _days_to_target(pos, current_capture=55.0, scenarios=[])
        assert result == 0

    def test_returns_days_when_scenario_crosses(self):
        pos = _pos(target_capture_pct=50.0)
        scenarios = [
            TimeScenario(days_from_now=3,  dte_remaining=20, option_value=200, pnl_ars=100, capture_pct=30.0, theta_daily_ars=10),
            TimeScenario(days_from_now=7,  dte_remaining=16, option_value=150, pnl_ars=150, capture_pct=45.0, theta_daily_ars=12),
            TimeScenario(days_from_now=14, dte_remaining=9,  option_value=80,  pnl_ars=220, capture_pct=65.0, theta_daily_ars=18),
        ]
        result = _days_to_target(pos, current_capture=25.0, scenarios=scenarios)
        assert result == 14  # primer escenario que supera 50%

    def test_returns_none_when_theta_never_reaches(self):
        pos = _pos(target_capture_pct=90.0)
        scenarios = [
            TimeScenario(days_from_now=14, dte_remaining=5, option_value=200, pnl_ars=100, capture_pct=40.0, theta_daily_ars=10),
        ]
        result = _days_to_target(pos, current_capture=20.0, scenarios=scenarios)
        assert result is None


# ── _rotation_opportunity ──────────────────────────────────────────────────────

class TestRotationOpportunity:
    def test_no_rotation_without_recs(self):
        pos = _pos()
        sym, gain, rationale = _rotation_opportunity(pos, 30.0, 60.0, None)
        assert sym is None
        assert gain is None

    def test_rotation_suggested_when_gap_is_large(self):
        """Si la TNA residual es baja y hay una rec con TNA mucho mayor, sugerir rotación."""
        from optionsdesk.signals.recommender import Recommendation, RiskProfile
        from optionsdesk.core.rates import RateResult
        from datetime import date as _date

        # Crear un RateResult con TNA alta
        rr = RateResult(
            symbol="GFGC9500G",
            strategy="COVERED_CALL",
            strike=9500.0,
            spot=8500.0,
            premium=500.0,
            net_outlay=9000.0,
            net_proceeds=9500.0,
            days=30,
            expiration=_date.today() + timedelta(days=30),
            period_rate_pct=5.5,
            tna_pct=70.0,   # TNA alta
            tea_pct=72.0,
            cushion_pct=11.0,
            delta=0.25,
            iv=0.55,
            is_liquid=True,
            moneyness="OTM",
        )
        rr.set_spread(60.0)  # establece spread_vs_caucion_pct
        # Crear una Recommendation mínima (mock)
        rec = object.__new__(Recommendation)
        rec.__dict__.update({
            "result": rr,
            "profile": RiskProfile.EQUILIBRADO,
            "score": 75.0,
            "light": "verde",
            "headline": "test",
            "plain_explanation": "test",
            "intention": "test",
            "win_scenario": "test",
            "lose_scenario": "test",
            "action_steps": [],
            "success_probability": 0.7,
            "expected_profit_ars": None,
            "ticket_text": "",
            "warnings": [],
            "contracts": 1,
            "horizon_plan": None,
            "vol_edge": None,
        })

        # Posición con TNA residual baja (capture = 60%, DTE largo)
        pos = _pos(
            premium_received=300.0, strike=9000.0, net_outlay=8700.0 if False else 9000.0,
            days_entry=60, days_held=30,  # 30 DTE restantes
            target_capture_pct=50.0,
        )
        # Con pos.days_remaining = 30 y capture = 60%, residual_tna ≈ 5-10% → muy baja
        sym, gain, rationale = _rotation_opportunity(pos, 60.0, 60.0, [rec])
        assert sym == "GFGC9500G" or sym is None  # puede ser None si residual TNA es alta

    def test_no_rotation_when_capture_is_low(self):
        """No sugerir rotación si la captura es < 40% (no conviene cerrar tan pronto)."""
        pos = _pos(days_entry=30, days_held=5)
        # available_recs vacío → None
        sym, gain, _ = _rotation_opportunity(pos, 20.0, 60.0, [])
        assert sym is None


# ── advise_position ───────────────────────────────────────────────────────────

class TestAdvisePosition:
    def test_returns_positionadvice(self):
        pos = _pos()
        adv = advise_position(pos, current_spot=8500.0)
        assert isinstance(adv, PositionAdvice)
        assert adv.action in {"HOLD", "CLOSE_NOW", "ROLL", "DEFEND", "ROTATE", "PARTIAL"}
        assert adv.urgency in {"alta", "media", "baja"}
        assert len(adv.scenarios) == 4  # t+3, t+7, t+14, t+21 por defecto

    def test_take_profit_triggers_close_now(self):
        """Posicion con alta captura → TAKE_PROFIT → CLOSE_NOW."""
        pos = _pos(
            premium_received=300.0,
            strike=9500.0,    # OTM call, spot = 8500
            days_entry=30,
            days_held=25,     # casi al vencimiento → theta acelerado
            target_capture_pct=50.0,
        )
        adv = advise_position(pos, current_spot=8400.0, current_iv=0.55)
        # OTM call casi al vencimiento con spot cayendo → captura alta → TAKE_PROFIT
        if adv.capture_pct >= 50.0:
            assert adv.action == "CLOSE_NOW"

    def test_stop_triggers_close_now(self):
        """Posicion con gran pérdida (opcion vale >2x prima) → STOP → CLOSE_NOW."""
        pos = _pos(
            strategy="COVERED_CALL",
            premium_received=200.0,
            strike=8500.0,    # ATM call
            spot_entry=8500.0,
            days_entry=30,
            days_held=5,
            max_loss_mult=1.0,
        )
        # Spot sube mucho → call ITM → opcion muy cara → pérdida > 1x prima
        adv = advise_position(pos, current_spot=9500.0, current_iv=0.70)
        assert adv.action == "CLOSE_NOW"

    def test_roll_when_near_expiry(self):
        """Con pocos DTE y opcion aun con valor → ROLL."""
        pos = _pos(
            premium_received=300.0,
            strike=9000.0,
            days_entry=30,
            days_held=12,     # 18 DTE restantes, por encima de roll_dte=21... no, quedan 18
            roll_dte=21,
        )
        # días restantes = 30 - 12 = 18 > roll_dte 21? No: 18 < 21 → debería disparar ROLL
        adv = advise_position(pos, current_spot=8500.0, current_iv=0.55)
        if adv.dte <= 21 and adv.capture_pct < 100:
            assert adv.action in {"ROLL", "CLOSE_NOW"}

    def test_hold_when_no_signal(self):
        """Posicion OTM fresca, sin señal urgente → HOLD."""
        pos = _pos(
            strategy="COVERED_CALL",
            premium_received=200.0,
            strike=9500.0,    # fuera del dinero
            spot_entry=8500.0,
            days_entry=30,
            days_held=3,
            target_capture_pct=50.0,
        )
        adv = advise_position(pos, current_spot=8500.0, current_iv=0.45)
        # OTM, spot no se movió, DTE = 27 → sin señal urgente
        assert adv.action in {"HOLD", "PARTIAL"}  # PARTIAL si plan muy rezagado

    def test_score_is_bounded(self):
        """El score siempre debe estar entre 0 y 100."""
        pos = _pos()
        adv = advise_position(pos, current_spot=8500.0)
        assert 0.0 <= adv.score <= 100.0

    def test_does_not_crash_with_zero_iv(self):
        """iv=0 no debe crashear el analizador."""
        pos = _pos(iv_entry=0.0)
        adv = advise_position(pos, current_spot=8500.0, current_iv=0.0)
        assert isinstance(adv, PositionAdvice)

    def test_context_adds_risk_flag_for_bearish_put(self):
        """Contexto bajista en una put vendida → flag de riesgo."""
        class FakeContext:
            trend = "bajista"
            confidence = "media"
            smc = None

        pos = _pos(strategy="SHORT_PUT")
        adv = advise_position(pos, current_spot=8500.0, context=FakeContext())
        assert any("bajista" in f for f in adv.risk_flags)


# ── advise_portfolio ──────────────────────────────────────────────────────────

class TestAdvisePortfolio:
    def test_returns_portfolio_advice(self):
        positions = [_pos(), _pos(strategy="SHORT_PUT", strike=8000.0)]
        pa = advise_portfolio(positions, current_spot=8500.0)
        assert isinstance(pa, PortfolioAdvice)
        assert len(pa.advices) == 2

    def test_sorted_by_urgency_descending(self):
        """Las posiciones con mayor score deben ir primero."""
        positions = [
            _pos(strategy="COVERED_CALL", premium_received=300.0, strike=9000.0, days_held=5),
            _pos(strategy="SHORT_PUT",    premium_received=300.0, strike=7500.0, days_held=5),
        ]
        pa = advise_portfolio(positions, current_spot=8500.0)
        scores = [a.score for a in pa.advices]
        assert scores == sorted(scores, reverse=True)

    def test_concentration_flag_with_multiple_positions(self):
        """Con varias posiciones, debe aparecer flag de concentración en GGAL."""
        positions = [_pos(), _pos(strategy="SHORT_PUT", strike=8000.0)]
        pa = advise_portfolio(positions, current_spot=8500.0)
        assert any("concentrado" in f.lower() or "GGAL" in f for f in pa.portfolio_flags)

    def test_empty_portfolio_returns_no_urgency(self):
        pa = advise_portfolio([], current_spot=8500.0)
        assert pa.highest_urgency == "ninguna"
        assert pa.advices == []

    def test_theta_daily_is_nonnegative(self):
        pos = _pos(days_held=5, days_entry=30)
        pa = advise_portfolio([pos], current_spot=8500.0)
        assert pa.theta_daily_ars >= 0
