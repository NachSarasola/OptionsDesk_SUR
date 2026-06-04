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


def _short_put(symbol: str) -> dict:
    return {
        "symbol": symbol,
        "strategy": "SHORT_PUT",
        "strike": 8000.0,
        "spot_entry": 8500.0,
        "premium_received": 300.0,
        "net_outlay": 7700.0,
        "iv_entry": 0.55,
        "days_entry": 30,
        "entry_date": date.today().isoformat(),
        "target_exit_days": 15,
        "target_capture_pct": 50.0,
        "contracts": 2,
        "expected_pop": 0.78,
        "expected_ev_ars": 4200.0,
    }


def test_close_renta_computes_premium_capture_pnl(tmp_path):
    """Cerrar una short put por recompra debe registrar P&L de captura de prima
    y conservar la predicción (PoP/EV) para calibrar el edge de renta."""
    positions_file = tmp_path / "open_positions.jsonl"
    positions_file.write_text(json.dumps(_short_put("GFGV8000F")) + "\n", encoding="utf-8")
    monitor = PositionMonitor(positions_file=positions_file)

    # Recompra a 150 (capturó la mitad de la prima de 300). Ganador.
    assert monitor.close_position_at(0, reason="TAKE_PROFIT", exit_price=150.0)

    archived = (tmp_path / "closed_positions.jsonl").read_text(encoding="utf-8").splitlines()
    rec = json.loads(archived[-1])
    assert rec["realized_pnl_ars"] is not None
    assert rec["realized_pnl_ars"] > 0          # capturó prima
    assert rec["realized_pnl_ars"] < 300 * 2 * 100  # menos que el bruto (pagó costos)
    assert rec["expected_pop"] == 0.78          # predicción persistida
    assert rec["expected_ev_ars"] == 4200.0


def test_close_renta_losing_buyback_is_negative(tmp_path):
    positions_file = tmp_path / "open_positions.jsonl"
    positions_file.write_text(json.dumps(_short_put("GFGV8000F")) + "\n", encoding="utf-8")
    monitor = PositionMonitor(positions_file=positions_file)

    # Recompra a 600 (la prima se duplicó en contra). Perdedor.
    assert monitor.close_position_at(0, reason="STOP", exit_price=600.0)
    rec = json.loads((tmp_path / "closed_positions.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert rec["realized_pnl_ars"] < 0


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


def test_close_position_at_archives_without_claiming_market_order(tmp_path):
    positions_file = tmp_path / "open_positions.jsonl"
    positions_file.write_text(json.dumps(_position("GFGC9000F")) + "\n", encoding="utf-8")

    monitor = PositionMonitor(positions_file=positions_file)
    assert monitor.close_position_at(0, reason="test_cleanup", exit_price=123.45)

    assert monitor.load_positions() == []
    closed = json.loads((tmp_path / "closed_positions.jsonl").read_text(encoding="utf-8"))
    assert closed["symbol"] == "GFGC9000F"
    assert closed["close_reason"] == "test_cleanup"
    assert closed["exit_price"] == 123.45
    assert closed["market_order_sent"] is False
