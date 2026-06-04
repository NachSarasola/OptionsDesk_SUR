from __future__ import annotations

from datetime import date

from optionsdesk.data.iv_history import (
    load_iv_history,
    record_daily_iv,
    representative_atm_iv,
)


def test_record_is_idempotent_per_day(tmp_path):
    p = tmp_path / "iv_history.jsonl"
    assert record_daily_iv(0.55, path=p, today=date(2026, 6, 1)) is True
    assert record_daily_iv(0.60, path=p, today=date(2026, 6, 1)) is False   # mismo día
    assert record_daily_iv(0.58, path=p, today=date(2026, 6, 2)) is True
    assert load_iv_history(path=p) == [0.55, 0.58]


def test_record_rejects_invalid_iv(tmp_path):
    p = tmp_path / "iv_history.jsonl"
    assert record_daily_iv(None, path=p) is False
    assert record_daily_iv(0.0, path=p) is False
    assert load_iv_history(path=p) == []


def test_load_respects_symbol_and_lookback(tmp_path):
    p = tmp_path / "iv_history.jsonl"
    for i in range(5):
        record_daily_iv(0.50 + i * 0.01, symbol="GGAL", path=p, today=date(2026, 6, 1 + i))
    record_daily_iv(0.99, symbol="YPFD", path=p, today=date(2026, 6, 1))
    assert load_iv_history("GGAL", path=p, lookback=2) == [0.53, 0.54]
    assert load_iv_history("YPFD", path=p) == [0.99]


def test_representative_atm_iv_prefers_atm():
    class R:
        def __init__(self, iv, m): self.iv = iv; self.moneyness = m
    results = [R(0.40, "ITM"), R(0.55, "ATM"), R(0.57, "ATM"), R(0.80, "OTM")]
    assert representative_atm_iv(results) == 0.56   # mediana de los ATM (0.55, 0.57)
    assert representative_atm_iv([]) is None
