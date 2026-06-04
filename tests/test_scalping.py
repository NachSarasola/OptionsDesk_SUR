import pytest
from datetime import datetime, timedelta, date
from optionsdesk.config.costs import CostModel
from optionsdesk.core.greeks_engine import OptionGreeks
from optionsdesk.data.providers.base import OptionsChain, Quote
from optionsdesk.signals.scalping import (
    LivePulse,
    ScalpSignal,
    _finalize_signal,
    _signal_rank,
    build_scalp_verdict,
    compute_spot_tape_pulse,
    scan_all,
    scan_chain_iv_relative,
    scan_gamma_radar,
    scan_gamma_scalp,
    scan_iv_spike,
    scan_momentum,
    scan_rebound,
    scan_volatility_breakout,
    scan_wave_ride,
)

@pytest.fixture
def mock_greeks():
    return {
        "GFGC4000A": OptionGreeks(
            symbol="GFGC4000A", strike=4000, option_type="C", days=15, spot=4000,
            bid=150, ask=160, mid=155, iv=0.45, delta=0.52, gamma=0.001, theta=-5.0, vega=10.0,
            moneyness="ATM", moneyness_pct=0.0, volume=100, spread_pct=6.45
        ),
        "GFGC4500A": OptionGreeks(
            symbol="GFGC4500A", strike=4500, option_type="C", days=15, spot=4000,
            bid=10, ask=12, mid=11, iv=0.45, delta=0.15, gamma=0.0005, theta=-1.0, vega=4.0,
            moneyness="OTM", moneyness_pct=12.5, volume=50, spread_pct=18.0
        ),
        "GFGV4000A": OptionGreeks(
            symbol="GFGV4000A", strike=4000, option_type="P", days=15, spot=4000,
            bid=140, ask=150, mid=145, iv=0.45, delta=-0.48, gamma=0.001, theta=-4.0, vega=10.0,
            moneyness="ATM", moneyness_pct=0.0, volume=100, spread_pct=6.89
        )
    }

class MockSnap:
    def __init__(
        self,
        trend,
        mom,
        rsi=60,
        bos=None,
        choch=None,
        vol_confirms=False,
        macd_hist=0.0,
        atr_pct=1.0,
        order_block=None,
        nearest_fvg=None,
    ):
        self.trend = trend
        self.momentum = mom
        self.rsi = rsi
        self.bos = bos
        self.choch = choch
        self.vol_confirms = vol_confirms
        self.macd_hist = macd_hist
        self.atr_pct = atr_pct
        self.order_block = order_block
        self.nearest_fvg = nearest_fvg

def test_scan_momentum_bullish(mock_greeks):
    snap = MockSnap("ALCISTA", 2.5, bos="BOS_UP")
    signals = scan_momentum(mock_greeks, snap, momentum_threshold=1.0)
    
    assert len(signals) > 0
    best = signals[0]
    assert best.action == "BUY_CALL"
    assert best.symbol == "GFGC4000A"
    # Delta sweet-spot en 0.50 — score calibrado; rango real de momentum ~30-70
    assert best.score > 30

def test_scan_momentum_bearish(mock_greeks):
    snap = MockSnap("BAJISTA", -3.0, bos="BOS_DOWN")
    signals = scan_momentum(mock_greeks, snap, momentum_threshold=1.0)
    
    assert len(signals) > 0
    best = signals[0]
    assert best.action == "BUY_PUT"
    assert best.symbol == "GFGV4000A"

def test_scan_momentum_lateral(mock_greeks):
    snap = MockSnap("LATERAL", 0.5)
    signals = scan_momentum(mock_greeks, snap, momentum_threshold=1.0)
    assert len(signals) == 0


def test_wave_ride_buys_calls_with_confirmed_bullish_wave(mock_greeks, monkeypatch):
    monkeypatch.setattr(
        "optionsdesk.signals.scalping.DEFAULT_COSTS",
        CostModel(
            stock_commission_pct=0.0,
            option_commission_pct=0.0,
            stock_market_fee_pct=0.0,
            option_market_fee_pct=0.0,
            exercise_fee_pct=0.0,
            exercise_market_fee_pct=0.0,
            iva_rate=0.0,
        ),
    )
    snap = MockSnap("ALCISTA", 3.2, rsi=61, bos="BOS_UP", vol_confirms=True, macd_hist=12.0)
    signals = scan_wave_ride(mock_greeks, snap, momentum_threshold=1.0)

    assert signals
    assert signals[0].action == "BUY_CALL"
    assert signals[0].signal_type == "WAVE_RIDE"
    assert signals[0].confidence == "actionable"
    assert signals[0].playbook == "Seguir ola"
    assert signals[0].plan_sl != pytest.approx(signals[0].mid * 0.70)
    assert signals[0].plan_tp != pytest.approx(signals[0].mid * 1.60)


def test_plan_uses_executable_limit_and_net_risk_fields(mock_greeks):
    snap = MockSnap("ALCISTA", 3.2, rsi=61, bos="BOS_UP", vol_confirms=True, macd_hist=12.0)
    signals = scan_wave_ride(mock_greeks, snap, momentum_threshold=1.0)
    sig = signals[0]

    assert sig.bid < sig.plan_entry < sig.ask
    assert sig.entry_style
    assert sig.planned_risk_ars > 0
    assert sig.max_loss_ars >= sig.plan_entry * 100
    assert sig.max_loss_ars > sig.planned_risk_ars
    assert sig.underlying_stop < sig.underlying_target
    assert sig.breakeven_move_pct >= 0
    assert sig.probability_of_profit > 0
    assert sig.expected_value_ars != 0
    assert sig.friction_ars > 0
    assert sig.friction_pct > 0


def test_rebound_prices_bearish_reversal_from_overbought(mock_greeks):
    snap = MockSnap("LATERAL", 0.2, rsi=72, choch="CHOCH_DOWN", vol_confirms=True)
    signals = scan_rebound(mock_greeks, snap)

    assert signals
    assert signals[0].action == "BUY_PUT"
    assert signals[0].signal_type == "REBOUND_PRICING"
    assert signals[0].playbook == "Pricear rebote"


def test_rebound_uses_spot_for_order_block_zone(mock_greeks):
    ob = type("OB", (), {"type": "bullish", "low": 3980.0, "high": 4040.0})()
    snap = MockSnap("LATERAL", 0.0, rsi=50, order_block=ob)
    signals = scan_rebound(mock_greeks, snap)

    assert signals
    assert signals[0].action == "BUY_CALL"
    assert signals[0].confidence == "actionable"


def test_volatility_breakout_arms_both_sides_when_direction_not_confirmed(mock_greeks):
    snap = MockSnap("LATERAL", 0.4, rsi=50, vol_confirms=True, atr_pct=3.2)
    signals = scan_volatility_breakout(mock_greeks, snap, realized_vol=0.65)

    assert signals
    assert {s.action for s in signals} >= {"BUY_CALL", "BUY_PUT"}
    assert all(s.confidence == "watchlist" for s in signals)
    assert all("esperando_breakout_direccion" in s.risk_flags for s in signals)


def test_volatility_breakout_accepts_modest_intraday_structure(mock_greeks):
    snap = MockSnap("ALCISTA", 0.22, rsi=58, bos="BOS_UP", vol_confirms=False, atr_pct=0.12)
    snap.momentum_threshold_pct = 0.18

    signals = scan_volatility_breakout(mock_greeks, snap, realized_vol=0.40)

    assert signals
    assert signals[0].signal_type == "VOL_BREAKOUT"
    assert signals[0].action == "BUY_CALL"


def test_stale_liquidity_reason_is_not_duplicated(mock_greeks):
    g = mock_greeks["GFGC4000A"]
    g.is_tradeable = False
    g.liquidity_reason = "quote stale (999s)"
    g.quote_age_s = 999.0
    snap = MockSnap("ALCISTA", 3.0, bos="BOS_UP", vol_confirms=True, macd_hist=1.0)

    signals = scan_wave_ride(mock_greeks, snap)

    stale_flags = [flag for flag in signals[0].risk_flags if "stale" in flag]
    assert stale_flags == ["quote stale (999s)"]

def test_scan_gamma_scalp(mock_greeks):
    # IV = 0.45, Realized = 0.60 -> Edge de 15pp
    signals = scan_gamma_scalp(mock_greeks, realized_vol=0.60)
    assert len(signals) > 0
    assert any(s.symbol == "GFGC4000A" for s in signals)
    assert any(s.symbol == "GFGV4000A" for s in signals)
    
    # IV = 0.45, Realized = 0.40 -> No edge
    signals = scan_gamma_scalp(mock_greeks, realized_vol=0.40)
    assert len(signals) == 0


def test_signal_goes_to_watchlist_when_not_tradeable(mock_greeks):
    g = mock_greeks["GFGC4000A"]
    g.is_tradeable = False
    g.liquidity_reason = "quote stale"
    snap = MockSnap("ALCISTA", 2.5, bos="BOS_UP")
    signals = scan_momentum(mock_greeks, snap, momentum_threshold=1.0)
    assert len(signals) > 0
    assert signals[0].confidence == "watchlist"
    assert "quote stale" in ",".join(signals[0].risk_flags)


def test_iv_spike_marks_short_as_watchlist_uncovered(mock_greeks):
    class _Fit:
        is_reliable = True

        def eval(self, strike):  # noqa: ARG002
            return 0.35

    smile_map = {"A": _Fit()}
    signals = scan_iv_spike(mock_greeks, snap=None, smile_map=smile_map, spot=4000.0, mispricing_min_pp=2.0)
    assert len(signals) > 0
    assert signals[0].action.startswith("SELL_")
    assert signals[0].confidence == "watchlist"
    assert "venta_iv_solo_contexto" in signals[0].risk_flags
    assert any(flag.startswith("short_") for flag in signals[0].risk_flags)


def test_high_friction_blocks_directional_buy(mock_greeks):
    g = mock_greeks["GFGC4000A"]
    g.bid = 100.0
    g.ask = 180.0
    g.mid = 140.0
    g.spread_pct = 57.0
    snap = MockSnap("ALCISTA", 3.2, rsi=61, bos="BOS_UP", vol_confirms=True, macd_hist=12.0)

    signals = scan_wave_ride(mock_greeks, snap, momentum_threshold=1.0)

    assert signals
    assert signals[0].confidence == "watchlist"
    assert any(flag in signals[0].risk_flags for flag in ("spread_alto", "ev_no_positivo"))


def test_sizing_respects_capital_budget(mock_greeks):
    snap = MockSnap("ALCISTA", 2.5, bos="BOS_UP")
    signals = scan_momentum(mock_greeks, snap, momentum_threshold=1.0, capital=1_000_000.0)
    assert len(signals) > 0
    assert signals[0].suggested_lots >= 1


def test_insufficient_capital_blocks_when_auto_sizing_required(mock_greeks):
    snap = MockSnap("ALCISTA", 2.5, bos="BOS_UP")
    cfg = {
        "max_spread": 22.0,
        "min_rr": 1.2,
        "max_age_s": 90.0,
        "min_action_delta": 0.25,
        "max_action_moneyness": 14.0,
        "entry_aggression": 0.35,
        "min_score_actionable": 48.0,
        "min_pop": 0.10,
        "max_breakeven_move_pct": 0.65,
        "require_capital": True,
    }
    signals = scan_momentum(mock_greeks, snap, momentum_threshold=1.0, capital=1_000.0, profile_cfg=cfg)

    assert len(signals) > 0
    assert signals[0].suggested_lots == 0
    assert signals[0].confidence == "watchlist"
    assert "capital_insuficiente" in signals[0].risk_flags


def test_real_money_mode_requires_loaded_capital(mock_greeks):
    snap = MockSnap("ALCISTA", 3.2, rsi=61, bos="BOS_UP", vol_confirms=True, macd_hist=12.0)
    cfg = {
        "max_spread": 15.0,
        "min_rr": 1.2,
        "max_age_s": 90.0,
        "vol_edge_min": 0.02,
        "min_action_delta": 0.35,
        "max_action_moneyness": 10.0,
        "entry_aggression": 0.35,
        "min_score_actionable": 40.0,
        "min_pop": 0.10,
        "max_breakeven_move_pct": 2.0,
        "require_capital": True,
    }

    signals = scan_wave_ride(mock_greeks, snap, momentum_threshold=1.0, profile_cfg=cfg)

    assert signals
    assert signals[0].confidence == "watchlist"
    assert "capital_no_cargado" in signals[0].risk_flags


def test_real_money_mode_blocks_when_open_scalp_limit_reached(mock_greeks):
    snap = MockSnap("ALCISTA", 3.2, rsi=61, bos="BOS_UP", vol_confirms=True, macd_hist=12.0)
    cfg = {
        "max_spread": 15.0,
        "min_rr": 1.2,
        "max_age_s": 90.0,
        "vol_edge_min": 0.02,
        "min_action_delta": 0.35,
        "max_action_moneyness": 10.0,
        "entry_aggression": 0.35,
        "min_score_actionable": 40.0,
        "min_pop": 0.10,
        "max_breakeven_move_pct": 2.0,
        "require_capital": True,
        "max_open_scalps": 2,
        "max_total_risk_pct": 10.0,
    }
    positions = [
        type("P", (), {"strategy": "SCALP_LONG_CALL", "net_outlay": 100.0, "contracts": 1})(),
        type("P", (), {"strategy": "SCALP_LONG_PUT", "net_outlay": 100.0, "contracts": 1})(),
    ]

    signals = scan_wave_ride(
        mock_greeks,
        snap,
        momentum_threshold=1.0,
        capital=2_000_000.0,
        positions=positions,
        profile_cfg=cfg,
    )

    assert signals
    assert signals[0].confidence == "watchlist"
    assert "max_scalps_abiertos>=2" in signals[0].risk_flags


def test_balanced_profile_demotes_low_delta_directional_buy(mock_greeks):
    g = mock_greeks["GFGC4500A"]
    g.delta = 0.21
    g.moneyness_pct = 8.0
    snap = MockSnap("ALCISTA", 3.2, bos="BOS_UP", vol_confirms=True, macd_hist=1.0, atr_pct=3.0)

    signals = scan_volatility_breakout(mock_greeks, snap, realized_vol=0.65)
    low_delta = [s for s in signals if s.symbol == "GFGC4500A"]

    assert low_delta
    assert low_delta[0].confidence == "watchlist"
    assert "delta_baja" in low_delta[0].risk_flags


def test_profile_changes_gamma_edge_gate(mock_greeks):
    # IV=0.45. Con RV=0.47 -> edge 2pp: balanced lo descarta, aggressive lo acepta.
    balanced = scan_gamma_scalp(
        mock_greeks,
        realized_vol=0.47,
        profile_cfg={"max_spread": 20.0, "min_rr": 1.2, "max_age_s": 90.0, "vol_edge_min": 0.02},
    )
    aggressive = scan_gamma_scalp(
        mock_greeks,
        realized_vol=0.47,
        profile_cfg={"max_spread": 28.0, "min_rr": 0.9, "max_age_s": 180.0, "vol_edge_min": 0.01},
    )
    assert balanced == []
    assert len(aggressive) > 0


def test_sell_put_becomes_actionable_when_cash_secured(mock_greeks):
    class _Fit:
        is_reliable = True

        def eval(self, strike):  # noqa: ARG002
            return 0.35

    smile_map = {"A": _Fit()}
    # Capital alto para cash-secured put.
    signals = scan_iv_spike(
        mock_greeks,
        snap=None,
        smile_map=smile_map,
        spot=4000.0,
        mispricing_min_pp=2.0,
        capital=5_000_000.0,
        positions=[type("P", (), {"strategy": "COVERED_CALL"})()],
    )
    assert len(signals) > 0
    put_sigs = [s for s in signals if s.action == "SELL_PUT"]
    if put_sigs:
        assert "short_put_no_cash_secured" not in put_sigs[0].risk_flags


def test_scan_all_returns_radar_watchlist_without_context_or_rv():
    spot_quote = Quote("GGAL", 4000.0, 4000.0, 4000.0, 1000, datetime.now())
    chain = OptionsChain(
        "GGAL",
        spot_quote,
        {
            "GFGC4000A": Quote("GFGC4000A", 150.0, 160.0, 155.0, 100, datetime.now()),
            "GFGV4000A": Quote("GFGV4000A", 140.0, 150.0, 145.0, 100, datetime.now()),
        },
    )
    expiry_calendar = {"A": date.today() + timedelta(days=20)}

    greeks, signals = scan_all(chain, expiry_calendar, snap=None, realized_vol=None, risk_profile="aggressive")

    assert greeks
    assert signals
    assert all(s.confidence == "watchlist" for s in signals)
    assert any(s.signal_type == "GAMMA_RADAR" for s in signals)
    assert any("falta_contexto_direccional" in s.risk_flags for s in signals)


def test_gamma_radar_surfaces_near_atm_candidates(mock_greeks):
    signals = scan_gamma_radar(mock_greeks, snap=None, realized_vol=None)

    assert len(signals) >= 2
    assert signals[0].confidence == "watchlist"
    assert signals[0].plan_entry > 0
    assert "esperando_trigger_scalping" in signals[0].risk_flags


def test_chain_iv_relative_generates_fallback_watchlist(mock_greeks):
    mock_greeks["GFGC4000A"].iv = 0.80
    mock_greeks["GFGV4000A"].iv = 0.42
    mock_greeks["GFGC4500A"].iv = 0.40

    signals = scan_chain_iv_relative(mock_greeks, snap=None)

    assert signals
    assert any(s.signal_type == "IV_RICH_CHAIN" for s in signals)
    assert all(s.confidence == "watchlist" for s in signals)
    rich = [s for s in signals if s.signal_type == "IV_RICH_CHAIN"][0]
    assert "confirmar_con_smile_iv" in rich.risk_flags


def _rank_sig(symbol: str, pop: float, ev: float, cost_to_target: float = 5.0) -> ScalpSignal:
    return ScalpSignal(
        signal_type="TEST",
        action="BUY_CALL",
        symbol=symbol,
        strike=4000.0,
        option_type="C",
        days_to_expiry=10,
        bid=100.0,
        ask=102.0,
        mid=101.0,
        spread_pct=2.0,
        delta=0.45,
        gamma=0.001,
        theta=-1.0,
        vega=1.0,
        iv=0.50,
        score=70.0,
        edge_source="test",
        rationale="test",
        urgency="media",
        confidence="actionable",
        probability_of_profit=pop,
        expected_value_ars=ev,
        planned_risk_ars=1000.0,
        plan_rr=1.0,
        fill_probability=0.70,
        cost_to_target_pct=cost_to_target,
    )


def test_signal_rank_prioritizes_execution_not_pseudo_pop():
    better_execution = _rank_sig("BETTER_EXEC", pop=0.55, ev=900.0, cost_to_target=2.0)
    better_execution.score = 95.0
    higher_pop = _rank_sig("HIGHPOP", pop=0.68, ev=300.0, cost_to_target=8.0)
    ranked = sorted([better_execution, higher_pop], key=_signal_rank, reverse=True)

    assert ranked[0].symbol == "BETTER_EXEC"


def test_spot_tape_pulse_requires_enough_live_observations_and_span():
    start = datetime(2026, 6, 1, 12, 0)
    short = [(start + timedelta(seconds=i * 10), 4000.0 + i * 2.0) for i in range(4)]
    ready = [(start + timedelta(seconds=i * 20), 4000.0 + i * 2.0) for i in range(5)]

    assert not compute_spot_tape_pulse(short).confirmed
    pulse = compute_spot_tape_pulse(ready)
    assert pulse.confirmed
    assert pulse.direction == "BULL"
    assert pulse.observations == 5


def test_spot_tape_pulse_rejects_choppy_tape_despite_net_move():
    """Cinta que oscila con violencia pero termina +0.2%: el first-vs-last la leería
    BULL; el Kaufman Efficiency Ratio la rechaza por chop (baja eficiencia)."""
    start = datetime(2026, 6, 1, 12, 0)
    chop = [4000.0, 4030.0, 4005.0, 4035.0, 4008.0]  # net +0.2% pero camino enorme
    samples = [(start + timedelta(seconds=i * 20), px) for i, px in enumerate(chop)]

    pulse = compute_spot_tape_pulse(samples)
    assert pulse.direction == "BULL"        # hay deriva neta positiva
    assert pulse.efficiency < 0.35          # pero la cinta es caótica
    assert not pulse.confirmed              # → no se confirma


def test_verdict_waits_without_confirmed_live_pulse():
    sig = _rank_sig("READY", pop=0.68, ev=300.0)
    verdict = build_scalp_verdict(
        [sig],
        feed_live=True,
        capital=1_000_000.0,
        pulse=LivePulse(False),
        session_phase="regular",
    )

    assert verdict.decision == "WAIT"
    assert "sin_confirmacion_intradia_live" in verdict.blockers


def test_verdict_selects_best_ranked_viable_buy_without_pop_bias():
    low = _rank_sig("LOW", pop=0.55, ev=900.0)
    high = _rank_sig("HIGH", pop=0.68, ev=300.0)
    verdict = build_scalp_verdict(
        [low, high],
        feed_live=True,
        capital=1_000_000.0,
        pulse=LivePulse(True, direction="BULL", observations=5, span_s=80.0),
        session_phase="regular",
    )

    assert verdict.decision == "BUY_CALL_NOW"
    assert verdict.signal is low


def test_scan_all_overnight_is_explicit_opt_in(monkeypatch):
    import optionsdesk.signals.scalping as scalping

    calls = []
    original = scalping._overnight_overrides
    monkeypatch.setattr(scalping, "_current_session_phase", lambda eod_window_min=35: "close")
    monkeypatch.setattr(
        scalping,
        "_overnight_overrides",
        lambda cfg: calls.append(True) or original(cfg),
    )
    spot_quote = Quote("GGAL", 4000.0, 4000.0, 4000.0, 1000, datetime.now())
    chain = OptionsChain(
        "GGAL",
        spot_quote,
        {"GFGC4000A": Quote("GFGC4000A", 150.0, 160.0, 155.0, 100, datetime.now())},
    )
    expiry_calendar = {"A": date.today() + timedelta(days=20)}

    scan_all(chain, expiry_calendar, capital=1_000_000.0)
    assert calls == []

    scan_all(chain, expiry_calendar, capital=1_000_000.0, allow_overnight_entries=True)
    assert calls == [True]


def test_soft_score_flag_does_not_block_actionable_signal(mock_greeks, monkeypatch):
    def fake_plan(*args, **kwargs):
        return {
            "entry": 100.0,
            "sl": 80.0,
            "tp": 145.0,
            "rr": 4.0,
            "planned_risk_ars": 2050.0,
            "max_loss_ars": 10050.0,
            "fees_ars": 30.0,
            "spread_cost_ars": 0.0,
            "breakeven_move_pct": 0.10,
            "underlying_stop": 3980.0,
            "underlying_target": 4040.0,
            "entry_style": "test",
        }

    monkeypatch.setattr("optionsdesk.signals.scalping._trade_plan_detail", fake_plan)
    monkeypatch.setattr("optionsdesk.signals.scalping._estimate_long_pop", lambda *args, **kwargs: 0.70)
    g = mock_greeks["GFGC4000A"]
    sig = _rank_sig("GFGC4000A", pop=0.70, ev=0.0)
    sig.score = 20.0

    out = _finalize_signal(sig, g, snap=None, capital=None, positions=None)

    assert out.confidence == "actionable"
    assert any(flag.startswith("score<") for flag in out.warning_flags)
    assert out.blocking_flags == []


def test_negative_ev_is_warning_not_ticket_gate(mock_greeks, monkeypatch):
    def fake_plan(*args, **kwargs):
        return {
            "entry": 100.0,
            "sl": 99.0,
            "tp": 100.1,
            "rr": 0.1,
            "planned_risk_ars": 250.0,
            "max_loss_ars": 10050.0,
            "fees_ars": 200.0,
            "spread_cost_ars": 100.0,
            "breakeven_move_pct": 0.10,
            "underlying_stop": 3990.0,
            "underlying_target": 4010.0,
            "entry_style": "test",
        }

    monkeypatch.setattr("optionsdesk.signals.scalping._trade_plan_detail", fake_plan)
    monkeypatch.setattr("optionsdesk.signals.scalping._estimate_long_pop", lambda *args, **kwargs: 0.99)
    g = mock_greeks["GFGC4000A"]
    sig = _rank_sig("GFGC4000A", pop=0.99, ev=0.0)

    out = _finalize_signal(sig, g, snap=None, capital=None, positions=None)

    assert out.confidence == "watchlist"
    assert "ev_no_positivo" in out.warning_flags
    assert "ev_no_positivo" not in out.blocking_flags


def test_extreme_spread_is_hard_block(mock_greeks, monkeypatch):
    def fake_plan(*args, **kwargs):
        return {
            "entry": 100.0,
            "sl": 90.0,
            "tp": 145.0,
            "rr": 4.0,
            "planned_risk_ars": 1050.0,
            "max_loss_ars": 10050.0,
            "fees_ars": 30.0,
            "spread_cost_ars": 0.0,
            "breakeven_move_pct": 0.10,
            "underlying_stop": 3980.0,
            "underlying_target": 4040.0,
            "entry_style": "test",
        }

    monkeypatch.setattr("optionsdesk.signals.scalping._trade_plan_detail", fake_plan)
    monkeypatch.setattr("optionsdesk.signals.scalping._estimate_long_pop", lambda *args, **kwargs: 0.70)
    g = mock_greeks["GFGC4000A"]
    g.spread_pct = 40.0
    sig = _rank_sig("GFGC4000A", pop=0.70, ev=0.0)

    out = _finalize_signal(sig, g, snap=None, capital=None, positions=None)

    assert out.confidence == "watchlist"
    assert any(flag.startswith("spread_extremo") for flag in out.blocking_flags)


def test_rr_below_one_is_hard_block(mock_greeks, monkeypatch):
    def fake_plan(*args, **kwargs):
        return {
            "entry": 100.0,
            "sl": 90.0,
            "tp": 105.0,
            "rr": 0.5,
            "planned_risk_ars": 1050.0,
            "max_loss_ars": 10050.0,
            "fees_ars": 30.0,
            "spread_cost_ars": 0.0,
            "breakeven_move_pct": 0.10,
            "underlying_stop": 3980.0,
            "underlying_target": 4040.0,
            "entry_style": "test",
        }

    monkeypatch.setattr("optionsdesk.signals.scalping._trade_plan_detail", fake_plan)
    monkeypatch.setattr("optionsdesk.signals.scalping._estimate_long_pop", lambda *args, **kwargs: 0.70)
    g = mock_greeks["GFGC4000A"]
    sig = _rank_sig("GFGC4000A", pop=0.70, ev=0.0)

    out = _finalize_signal(sig, g, snap=None, capital=None, positions=None)

    assert out.confidence == "watchlist"
    assert "rr_inferior_a_1" in out.blocking_flags
