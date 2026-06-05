"""Tests del optimizador walk-forward de parametros (param_learner)."""
from __future__ import annotations

import random
from datetime import datetime, timedelta

import pytest

from optionsdesk.performance.attribution import ClosedTrade, attribute
from optionsdesk.backtest.param_store import TUNABLES, default_params
from optionsdesk.backtest.param_learner import (
    LearnerState,
    advance_learner,
    fitness,
    failing_setups,
    propose_challenger,
    _MIN_SAMPLE,
)


# ── Helpers para fabricar trades cerrados ─────────────────────────────────────

def _trade(strategy: str, pnl: float, *, when: datetime, exp_pop=None) -> ClosedTrade:
    return ClosedTrade(
        strategy=strategy,
        signal_type=strategy,
        realized_pnl_ars=pnl,
        expected_pop=exp_pop,
        expected_ev_ars=None,
        close_reason="test",
        closed_at=when,
        source="test",
    )


def _winning_book(n: int, *, base: datetime, strategy="STOCK_SWING_BREAKOUT") -> list[ClosedTrade]:
    """n trades mayormente ganadores (edge positivo)."""
    out = []
    for i in range(n):
        pnl = 1000.0 if i % 3 != 0 else -300.0   # ~66% win, expectancy positiva
        out.append(_trade(strategy, pnl, when=base + timedelta(minutes=i)))
    return out


def _losing_book(n: int, *, base: datetime, strategy="STOCK_SWING_BREAKOUT") -> list[ClosedTrade]:
    """n trades mayormente perdedores (edge negativo)."""
    out = []
    for i in range(n):
        pnl = -800.0 if i % 4 != 0 else 200.0    # ~25% win, expectancy negativa
        out.append(_trade(strategy, pnl, when=base + timedelta(minutes=i)))
    return out


# ── fitness ───────────────────────────────────────────────────────────────────

class TestFitness:
    def test_empty_edges_is_very_negative(self):
        assert fitness([]) < -1e8

    def test_winning_book_beats_losing_book(self):
        base = datetime(2026, 1, 1, 12, 0)
        win_edges = attribute(_winning_book(30, base=base), min_sample=15)
        lose_edges = attribute(_losing_book(30, base=base), min_sample=15)
        assert fitness(win_edges) > fitness(lose_edges)

    def test_losing_book_penalized_below_raw_pnl(self):
        base = datetime(2026, 1, 1, 12, 0)
        edges = attribute(_losing_book(30, base=base), min_sample=15)
        total = sum(e.total_pnl_ars for e in edges)
        # La fitness penaliza el setup NO_EDGE → queda por debajo del PnL crudo.
        assert fitness(edges) < total


class TestFailingSetups:
    def test_detects_negative_edge_setup(self):
        base = datetime(2026, 1, 1, 12, 0)
        edges = attribute(_losing_book(30, base=base, strategy="STOCK_SWING_BREAKOUT"), min_sample=15)
        failing = failing_setups(edges)
        assert "STOCK_SWING_BREAKOUT" in failing

    def test_winning_setup_not_failing(self):
        base = datetime(2026, 1, 1, 12, 0)
        edges = attribute(_winning_book(30, base=base), min_sample=15)
        assert failing_setups(edges) == set()


# ── propose_challenger ─────────────────────────────────────────────────────────

class TestProposeChallenger:
    def test_at_least_one_param_changes(self):
        rng = random.Random(42)
        champ = default_params()
        chal = propose_challenger(champ, temperature=0.3, failing=set(), rng=rng)
        assert chal != champ

    def test_respects_bounds(self):
        rng = random.Random(1)
        champ = default_params()
        for _ in range(50):
            chal = propose_challenger(champ, temperature=1.0, failing=set(), rng=rng)
            for name, val in chal.items():
                spec = TUNABLES[name]
                assert spec.lo <= val <= spec.hi, f"{name}={val} fuera de [{spec.lo},{spec.hi}]"

    def test_failing_setup_param_more_likely_to_move(self):
        """Los params del setup que falla deben moverse mas seguido que el resto."""
        champ = default_params()
        failing = {"STOCK_SWING_BREAKOUT"}
        # breakout_extension SOLO afecta STOCK_SWING_BREAKOUT
        moves_targeted = 0
        moves_other = 0
        for seed in range(60):
            rng = random.Random(seed)
            chal = propose_challenger(champ, temperature=0.3, failing=failing, rng=rng)
            if chal["stock_max_breakout_extension"] != champ["stock_max_breakout_extension"]:
                moves_targeted += 1
            # un param que NO mapea a STOCK_SWING_BREAKOUT
            if chal["rvol_lookback"] != champ["rvol_lookback"]:
                moves_other += 1
        assert moves_targeted > moves_other

    def test_deterministic_with_seed(self):
        champ = default_params()
        a = propose_challenger(champ, 0.4, set(), random.Random(7))
        b = propose_challenger(champ, 0.4, set(), random.Random(7))
        assert a == b


# ── advance_learner ─────────────────────────────────────────────────────────

class TestAdvanceLearner:
    def test_first_call_sets_baseline(self):
        state = LearnerState()
        out = advance_learner(state, [], now=datetime(2026, 1, 1, 12, 0))
        assert out.deployed_at is not None
        assert out.active == "champion"
        assert out.last_action == "init_baseline"

    def test_gathers_until_min_sample(self):
        base = datetime(2026, 1, 1, 12, 0)
        state = LearnerState(deployed_at=base.isoformat(), active="champion")
        few = _winning_book(_MIN_SAMPLE - 3, base=base + timedelta(minutes=1))
        out = advance_learner(state, few, now=base + timedelta(hours=1))
        assert out.last_action == "gathering"
        assert out.challenger is None

    def test_baseline_measured_then_challenger_deployed(self):
        base = datetime(2026, 1, 1, 12, 0)
        state = LearnerState(deployed_at=base.isoformat(), active="champion")
        book = _winning_book(_MIN_SAMPLE + 5, base=base + timedelta(minutes=1))
        out = advance_learner(state, book, now=base + timedelta(hours=2), rng=random.Random(3))
        assert out.active == "challenger"
        assert out.challenger is not None
        assert out.champion_fitness is not None
        assert out.last_action == "deploy_challenger"

    def test_better_challenger_promoted(self):
        base = datetime(2026, 1, 1, 12, 0)
        # Campeon con fitness bajo ya medido; retador desplegado.
        champ = default_params()
        chal = dict(champ, stock_min_net_rr=0.6)
        state = LearnerState(
            champion=champ, champion_fitness=0.0,
            challenger=chal, active="challenger",
            deployed_at=base.isoformat(), temperature=0.5,
        )
        # Trades del retador: claramente ganadores → fitness alto.
        book = _winning_book(_MIN_SAMPLE + 10, base=base + timedelta(minutes=1))
        out = advance_learner(state, book, now=base + timedelta(hours=3), rng=random.Random(5))
        assert out.last_action == "promote"
        # El campeon ahora es el retador anterior (mismo min_net_rr).
        assert out.champion["stock_min_net_rr"] == 0.6
        # Temperatura baja al promover.
        assert out.temperature < 0.5

    def test_worse_challenger_reverted_and_temp_rises(self):
        base = datetime(2026, 1, 1, 12, 0)
        champ = default_params()
        chal = dict(champ, stock_min_net_rr=0.6)
        state = LearnerState(
            champion=champ, champion_fitness=50_000.0,   # campeon muy bueno
            challenger=chal, active="challenger",
            deployed_at=base.isoformat(), temperature=0.3,
        )
        # Retador perdedor → fitness bajo, no supera al campeon.
        book = _losing_book(_MIN_SAMPLE + 10, base=base + timedelta(minutes=1))
        out = advance_learner(state, book, now=base + timedelta(hours=3), rng=random.Random(9))
        assert out.last_action == "revert"
        # El campeon NO cambia (sigue con el min_net_rr default).
        assert out.champion["stock_min_net_rr"] == champ["stock_min_net_rr"]
        # Temperatura sube (annealing: explora mas tras fallar).
        assert out.temperature > 0.3
        assert out.consecutive_reverts == 1

    def test_consecutive_reverts_keep_raising_temperature(self):
        base = datetime(2026, 1, 1, 12, 0)
        champ = default_params()
        state = LearnerState(
            champion=champ, champion_fitness=50_000.0,
            challenger=dict(champ, stock_min_net_rr=0.6), active="challenger",
            deployed_at=base.isoformat(), temperature=0.2,
        )
        temps = []
        for k in range(3):
            book = _losing_book(_MIN_SAMPLE + 5, base=base + timedelta(hours=k, minutes=1))
            state = advance_learner(
                state, book, now=base + timedelta(hours=k + 1), rng=random.Random(k),
            )
            temps.append(state.temperature)
        assert temps == sorted(temps), f"temperatura deberia subir monotonamente: {temps}"
        assert state.consecutive_reverts == 3

    def test_deployed_params_is_challenger_when_active(self):
        champ = default_params()
        chal = dict(champ, stock_min_net_rr=0.7)
        state = LearnerState(champion=champ, challenger=chal, active="challenger")
        assert state.deployed_params()["stock_min_net_rr"] == 0.7

    def test_state_roundtrip_dict(self):
        champ = default_params()
        state = LearnerState(champion=champ, champion_fitness=123.0,
                             challenger=dict(champ, stock_min_net_rr=0.5),
                             active="challenger", temperature=0.4, iterations=3)
        restored = LearnerState.from_dict(state.to_dict())
        assert restored.champion_fitness == 123.0
        assert restored.active == "challenger"
        assert restored.temperature == 0.4
        assert restored.iterations == 3
