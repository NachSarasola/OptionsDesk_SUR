from __future__ import annotations

import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from optionsdesk.backtest.demo_runner import (
    CycleInputs,
    _market_open,
    compute_self_learning_state,
    load_runner_status,
    run_cycle,
)

BA = ZoneInfo("America/Argentina/Buenos_Aires")


def _seed(tmp_path, pnls, signal_type="SWING_BREAKOUT"):
    lines = [
        json.dumps({"strategy": "SWING", "signal_type": signal_type,
                    "pnl_ars": p, "exit_reason": "STOP", "exit_ts": f"2026-06-{1+i:02d}T12:00:00"})
        for i, p in enumerate(pnls)
    ]
    (tmp_path / "stock_demo_trades.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_state_learns_kelly_and_drawdown_halt(tmp_path):
    # 16 perdedores grandes → drawdown enorme vs capital chico → freno + bloqueo
    _seed(tmp_path, [-5000.0] * 16)
    state = compute_self_learning_state(data_dir=tmp_path, capital=100_000.0, max_drawdown_pct=8.0, since="2026-01-01")
    assert state.halted is True
    assert state.realized_pnl_ars < 0
    assert "STOCK_SWING_BREAKOUT" in state.blocked         # setup probadamente perdedor
    assert state.adaptive.mode == "DEFENSIVE"


def test_state_not_halted_when_profitable(tmp_path):
    _seed(tmp_path, [800.0] * 16)
    state = compute_self_learning_state(data_dir=tmp_path, capital=1_000_000.0, since="2026-01-01")
    assert state.halted is False
    assert state.realized_pnl_ars > 0
    assert state.blocked == set()                          # nada que cortar


def test_small_sample_protects_new_setup(tmp_path):
    _seed(tmp_path, [-5000.0] * 5)                         # pocos trades
    state = compute_self_learning_state(data_dir=tmp_path, capital=1_000_000.0, since="2026-01-01")
    assert state.blocked == set()                          # idea nueva no se bloquea


def test_run_cycle_empty_inputs_writes_status(tmp_path):
    _seed(tmp_path, [500.0] * 16)
    status_file = tmp_path / "status.json"
    status = run_cycle(
        lambda: CycleInputs(), now=datetime(2026, 6, 3, 12, 0, tzinfo=BA),
        data_dir=tmp_path, capital=1_000_000.0, status_path=status_file,
    )
    assert status.stock_open == 0 and status.options_open == 0
    assert status.halted is False
    saved = load_runner_status(status_file)
    assert saved is not None and saved["mode"] == status.mode


def test_market_open_hours():
    assert _market_open(datetime(2026, 6, 3, 12, 0, tzinfo=BA)) is True      # miércoles 12h
    assert _market_open(datetime(2026, 6, 3, 9, 0, tzinfo=BA)) is False      # antes de apertura
    assert _market_open(datetime(2026, 6, 6, 12, 0, tzinfo=BA)) is False     # sábado


def _seed_dated(tmp_path, rows):
    """rows: list de (pnl, exit_ts_iso_or_None). Escribe stock_demo_trades.jsonl."""
    lines = []
    for pnl, ts in rows:
        rec = {"strategy": "SWING", "signal_type": "SWING_BREAKOUT", "pnl_ars": pnl,
               "exit_reason": "STOP", "score": 80.0}
        if ts is not None:
            rec["exit_ts"] = ts
        lines.append(json.dumps(rec))
    (tmp_path / "stock_demo_trades.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_drawdown_window_ignores_old_loss(tmp_path):
    # Pérdida enorme VIEJA + 45 ganancias recientes. La ventana no debe verla.
    from datetime import datetime, timedelta
    base = datetime(2026, 6, 1, 12, 0, 0)
    rows = [(-500_000.0, "2026-05-01T12:00:00")]
    rows += [(1_000.0, (base + timedelta(days=i)).isoformat()) for i in range(45)]
    _seed_dated(tmp_path, rows)

    # Pasar since explícito para no heredar el valor del .env en tests.
    win = compute_self_learning_state(data_dir=tmp_path, capital=1_800_000.0, lookback=40, since="2026-01-01")
    allt = compute_self_learning_state(data_dir=tmp_path, capital=1_800_000.0, lookback=1000, since="2026-01-01")

    assert win.halted is False        # ventana = solo ganancias → no frena
    assert allt.halted is True        # all-time incluye la pérdida vieja → frena


def test_sort_is_timezone_safe(tmp_path):
    # Mezcla de closed_at tz-aware y sin fecha → no debe crashear el sort.
    _seed_dated(tmp_path, [
        (100.0, "2026-06-01T12:00:00-03:00"),   # tz-aware
        (-50.0, None),                           # sin fecha
        (200.0, "2026-06-02T12:00:00-03:00"),
    ])
    state = compute_self_learning_state(data_dir=tmp_path, capital=1_000_000.0, lookback=40, since="2026-01-01")
    assert state.realized_pnl_ars == 250.0       # 100 - 50 + 200


def test_since_baseline_excludes_old_era(tmp_path):
    # Era vieja perdedora + era nueva ganadora. since corta la vieja.
    rows = [(-5_000.0, f"2026-05-{1 + i:02d}T12:00:00") for i in range(16)]   # 16 perdedores viejos
    rows += [(3_000.0, f"2026-06-{1 + i:02d}T12:00:00") for i in range(3)]    # 3 ganadores nuevos
    _seed_dated(tmp_path, rows)

    full = compute_self_learning_state(data_dir=tmp_path, capital=1_000_000.0, since="2026-01-01")
    reset = compute_self_learning_state(data_dir=tmp_path, capital=1_000_000.0, since="2026-06-01")

    assert full.realized_pnl_ars < 0                       # toda la historia: en rojo
    assert "STOCK_SWING_BREAKOUT" in full.blocked          # bloqueado por la era vieja
    assert reset.realized_pnl_ars == 9_000.0               # solo la era nueva (3 × 3000)
    assert reset.blocked == set()                          # nada que cortar tras el reset
