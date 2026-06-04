"""Tests del registro local de posiciones."""
from __future__ import annotations

import json
from datetime import date

from optionsdesk.signals.monitor import PositionMonitor


def _position(symbol: str) -> dict:
    return {
        "symbol": symbol,
        "strategy": "SCALP_LONG_CALL",
        "strike": 9000.0,
        "spot_entry": 8500.0,
        "premium_received": -100.0,
        "net_outlay": 100.0,
        "iv_entry": 0.55,
        "days_entry": 30,
        "entry_date": date.today().isoformat(),
        "target_exit_days": 1,
        "target_capture_pct": 30.0,
    }


def test_remove_position_at_keeps_other_and_unknown_lines(tmp_path):
    positions_file = tmp_path / "open_positions.jsonl"
    first = _position("GFGC9000F")
    second = _position("GFGC9500F")
    unknown_line = '{"legacy": true}'
    positions_file.write_text(
        "\n".join([json.dumps(first), unknown_line, json.dumps(second)]) + "\n",
        encoding="utf-8",
    )

    monitor = PositionMonitor(positions_file=positions_file)
    assert monitor.remove_position_at(1) is True
    assert [p.symbol for p in monitor.load_positions()] == ["GFGC9000F"]
    assert unknown_line in positions_file.read_text(encoding="utf-8")


def test_remove_position_at_rejects_missing_index(tmp_path):
    positions_file = tmp_path / "open_positions.jsonl"
    positions_file.write_text(json.dumps(_position("GFGC9000F")) + "\n", encoding="utf-8")

    assert PositionMonitor(positions_file=positions_file).remove_position_at(3) is False
