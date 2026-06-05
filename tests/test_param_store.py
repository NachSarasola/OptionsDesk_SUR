"""Tests del registro y persistencia de parametros tuneables."""
from __future__ import annotations

import json

import pytest

from optionsdesk.backtest import param_store
from optionsdesk.backtest.param_store import (
    TUNABLES,
    TunableParam,
    active_params,
    default_params,
    describe_active,
    param,
    write_active_params,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    """Limpia la cache del module-global entre tests."""
    param_store._cache = {}
    param_store._cache_mtime = None
    param_store._cache_path = None
    yield


class TestTunableParam:
    def test_clamp_respects_bounds(self):
        p = TunableParam("x", default=0.5, lo=0.1, hi=0.9, scale=0.1)
        assert p.clamp(2.0) == 0.9
        assert p.clamp(-1.0) == 0.1
        assert p.clamp(0.5) == 0.5

    def test_clamp_rounds_int(self):
        p = TunableParam("x", default=60, lo=30, hi=120, scale=10, is_int=True)
        assert p.clamp(61.7) == 62.0
        assert isinstance(p.clamp(61.7), float)
        assert p.clamp(61.7) == int(p.clamp(61.7))

    def test_registry_defaults_within_bounds(self):
        for name, spec in TUNABLES.items():
            assert spec.lo <= spec.default <= spec.hi, f"{name} default fuera de bounds"
            assert spec.scale > 0, f"{name} scale debe ser positivo"


class TestActiveParams:
    def test_returns_empty_without_file(self, tmp_path):
        assert active_params(tmp_path) == {}

    def test_param_falls_back_to_default(self, tmp_path):
        assert param("smc_eql_tolerance", 0.005, data_dir=tmp_path) == 0.005
        assert param("inexistente", 99.0, data_dir=tmp_path) == 99.0

    def test_write_and_read_roundtrip(self, tmp_path):
        write_active_params({"smc_eql_tolerance": 0.008}, data_dir=tmp_path)
        assert param("smc_eql_tolerance", 0.005, data_dir=tmp_path) == 0.008

    def test_write_clamps_out_of_bounds(self, tmp_path):
        write_active_params({"smc_eql_tolerance": 99.0}, data_dir=tmp_path)
        # 99.0 se clampa al hi del registro (0.012)
        assert param("smc_eql_tolerance", 0.005, data_dir=tmp_path) == 0.012

    def test_write_ignores_unknown_params(self, tmp_path):
        write_active_params({"no_existe": 1.0, "smc_eql_tolerance": 0.007}, data_dir=tmp_path)
        active = active_params(tmp_path)
        assert "no_existe" not in active
        assert active["smc_eql_tolerance"] == 0.007

    def test_int_param_persisted_as_int(self, tmp_path):
        write_active_params({"smc_pda_lookback": 73.6}, data_dir=tmp_path)
        val = param("smc_pda_lookback", 60, data_dir=tmp_path)
        assert val == 74.0

    def test_corrupt_file_falls_back(self, tmp_path):
        (tmp_path / "learned_params.json").write_text("{not json", encoding="utf-8")
        assert param("smc_eql_tolerance", 0.005, data_dir=tmp_path) == 0.005

    def test_cache_refreshes_on_file_change(self, tmp_path):
        write_active_params({"stock_min_net_rr": 0.5}, data_dir=tmp_path)
        assert param("stock_min_net_rr", 0.4, data_dir=tmp_path) == 0.5
        write_active_params({"stock_min_net_rr": 0.7}, data_dir=tmp_path)
        assert param("stock_min_net_rr", 0.4, data_dir=tmp_path) == 0.7


class TestDefaults:
    def test_default_params_covers_all_tunables(self):
        d = default_params()
        assert set(d.keys()) == set(TUNABLES.keys())

    def test_describe_active_shape(self, tmp_path):
        rows = describe_active(tmp_path)
        assert len(rows) == len(TUNABLES)
        for r in rows:
            assert {"param", "actual", "default", "min", "max", "delta_vs_default"} <= set(r)
            # sin archivo: actual == default → delta 0
            assert r["delta_vs_default"] == 0
