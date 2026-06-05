"""Tests del backtester offline + busqueda en arbol condicionada por contexto."""
from __future__ import annotations

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from optionsdesk.backtest import param_store
from optionsdesk.backtest.param_store import (
    FLAGS, active_flags, default_flags, flag, param, param_override,
)
from optionsdesk.backtest.strategy_context import (
    UNKNOWN_REGIME, all_regimes, classify_regime, regime_label_es,
)
from optionsdesk.backtest.strategy_backtest import (
    StrategyGenome, evaluate_genome,
)
from optionsdesk.backtest.strategy_tree import (
    StrategyPolicy, deploy_for_regime, mutate, policy_deployability, save_policy, search,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    param_store._cache = {}
    param_store._cache_mtime = None
    param_store._cache_path = None
    param_store._flags_cache = {}
    param_store._flags_cache_mtime = None
    param_store._override = None
    param_store._override_flags = None
    yield


# Walk acotado para que la suite sea rapida (el ranking relativo se preserva).
_FAST = {"step": 4, "max_days": 30}


def _make_history(seed: int, drift: float, n: int = 110) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = datetime(2025, 1, 1)
    p = 100.0
    rows = []
    for i in range(n):
        p *= 1 + rng.normal(drift, 0.02)
        hi = p * (1 + abs(rng.normal(0, 0.012)))
        lo = p * (1 - abs(rng.normal(0, 0.012)))
        rows.append({"date": base + timedelta(days=i), "open": p * 0.999,
                     "high": hi, "low": lo, "close": p, "volume": 250_000})
    return pd.DataFrame(rows)


# ── param_store: override + flags ──────────────────────────────────────────────

class TestStoreOverrideAndFlags:
    def test_param_override_takes_precedence(self):
        with param_override({"stock_min_net_rr": 0.8}):
            assert param("stock_min_net_rr", 0.4) == 0.8
        # Fuera del with vuelve al default.
        assert param("stock_min_net_rr", 0.4) == 0.4

    def test_override_clamps_to_bounds(self):
        with param_override({"stock_min_net_rr": 99.0}):
            assert param("stock_min_net_rr", 0.4) == 0.9   # hi del registro

    def test_flag_default_true(self):
        assert flag("enable_swing_breakout", True) is True

    def test_flag_override(self):
        with param_override(None, {"enable_swing_breakout": False}):
            assert flag("enable_swing_breakout", True) is False
        assert flag("enable_swing_breakout", True) is True

    def test_flags_persisted_and_read(self, tmp_path):
        from optionsdesk.backtest.param_store import write_active_params
        write_active_params({}, flags={"enable_smc_reversal": False}, data_dir=tmp_path)
        assert active_flags(tmp_path).get("enable_smc_reversal") is False

    def test_default_flags_all_on(self):
        assert all(default_flags().values())
        assert set(default_flags()) == set(FLAGS)


# ── strategy_context ────────────────────────────────────────────────────────

class TestRegimeClassifier:
    def test_unknown_when_too_few_bars(self):
        df = _make_history(1, 0.001, n=10)
        assert classify_regime(df) == UNKNOWN_REGIME

    def test_uptrend_detected(self):
        df = _make_history(1, 0.01, n=120)   # drift fuerte alcista
        reg = classify_regime(df)
        assert reg.split("/")[0] == "up"

    def test_downtrend_detected(self):
        df = _make_history(2, -0.01, n=120)
        reg = classify_regime(df)
        assert reg.split("/")[0] == "down"

    def test_all_regimes_includes_unknown(self):
        regs = all_regimes()
        assert UNKNOWN_REGIME in regs
        assert "up/hi" in regs and "down/lo" in regs

    def test_label_es_readable(self):
        assert "alcista" in regime_label_es("up/hi")
        assert regime_label_es(UNKNOWN_REGIME) == "sin datos"


# ── strategy_backtest ────────────────────────────────────────────────────────

class TestEvaluateGenome:
    def test_returns_per_regime_breakdown(self):
        hist = {"AAA": _make_history(1, 0.004)}
        res = evaluate_genome(StrategyGenome.default(), hist, **_FAST)
        assert res.overall.n == sum(r.n for r in res.by_regime.values())
        # Cada regimen reportado existe en all_regimes.
        for reg in res.by_regime:
            assert reg in all_regimes()

    def test_disabling_all_relevant_setups_reduces_signals(self):
        hist = {"AAA": _make_history(3, 0.003)}
        full = evaluate_genome(StrategyGenome.default(), hist, **_FAST)
        g_off = StrategyGenome(
            params=StrategyGenome.default().params,
            flags={k: False for k in FLAGS},
        )
        off = evaluate_genome(g_off, hist, **_FAST)
        assert off.n_signals <= full.n_signals

    def test_genome_params_actually_applied(self):
        """Subir min_net_rr (mas exigente) no debe aumentar las señales."""
        hist = {"AAA": _make_history(5, 0.002)}
        lax = StrategyGenome(params={**StrategyGenome.default().params, "stock_min_net_rr": 0.25},
                             flags=default_flags())
        strict = StrategyGenome(params={**StrategyGenome.default().params, "stock_min_net_rr": 0.9},
                                flags=default_flags())
        assert evaluate_genome(strict, hist, **_FAST).n_signals <= evaluate_genome(lax, hist, **_FAST).n_signals


# ── mutate ────────────────────────────────────────────────────────────────────

class TestMutate:
    def test_produces_change(self):
        g = StrategyGenome.default()
        child = mutate(g, random.Random(1))
        assert child.key() != g.key()

    def test_never_all_setups_off(self):
        g = StrategyGenome.default()
        for seed in range(40):
            child = mutate(g, random.Random(seed))
            assert any(child.flags.values()), "mutate dejo un genoma sin setups"

    def test_params_within_bounds(self):
        from optionsdesk.backtest.param_store import TUNABLES
        g = StrategyGenome.default()
        for seed in range(30):
            child = mutate(g, random.Random(seed), sigma=1.0)
            for name, val in child.params.items():
                spec = TUNABLES[name]
                assert spec.lo <= val <= spec.hi


# ── search + policy ────────────────────────────────────────────────────────

class TestSearch:
    def test_search_produces_policy_for_all_regimes(self):
        hist = {"AAA": _make_history(1, 0.004), "BBB": _make_history(2, -0.003)}
        out = search(hist, max_evals=8, beam=2, children=2, rng=random.Random(0), **_FAST)
        assert isinstance(out.policy, StrategyPolicy)
        # Cada regimen tiene un genoma asignado (especialista o fallback global).
        for reg in all_regimes():
            assert reg in out.policy.by_regime
            assert out.policy.genome_for(reg) is not None

    def test_search_respects_eval_budget(self):
        hist = {"AAA": _make_history(1, 0.003)}
        out = search(hist, max_evals=6, beam=2, children=2, rng=random.Random(0), **_FAST)
        assert out.policy.evaluated <= 6
        assert len(out.nodes) <= 6

    def test_best_overall_is_max_fitness(self):
        hist = {"AAA": _make_history(1, 0.004)}
        out = search(hist, max_evals=8, rng=random.Random(0), **_FAST)
        assert out.best_overall.result.overall.fitness == max(
            nd.result.overall.fitness for nd in out.nodes
        )

    def test_save_and_deploy_roundtrip(self, tmp_path):
        hist = {"AAA": _make_history(1, 0.004)}
        out = search(hist, max_evals=6, rng=random.Random(0), **_FAST)
        save_policy(out, path=tmp_path / "strategy_tree.json")
        # Desplegar un regimen escribe params+flags al store.
        msg = deploy_for_regime(
            "up/hi",
            data_dir=tmp_path,
            policy_path=tmp_path / "strategy_tree.json",
            require_deployable=False,
        )
        assert msg is not None
        # El archivo de params activos quedo escrito.
        assert (tmp_path / "learned_params.json").exists()

    def test_deploy_without_policy_returns_none(self, tmp_path):
        assert deploy_for_regime("up/hi", data_dir=tmp_path,
                                 policy_path=tmp_path / "nope.json") is None

    def test_negative_policy_is_not_deployable(self):
        ok, reason = policy_deployability({"overall_fitness": -1, "overall_n": 99})
        assert ok is False
        assert "fitness" in reason
