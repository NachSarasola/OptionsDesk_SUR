from __future__ import annotations

import json

from optionsdesk.performance.attribution import (
    ClosedTrade,
    Verdict,
    attribute,
    is_strategy_blocked,
    load_all_closed_trades,
)


def _trade(strategy, pnl, *, pop=None, ev=None, signal_type="X") -> ClosedTrade:
    return ClosedTrade(
        strategy=strategy,
        signal_type=signal_type,
        realized_pnl_ars=pnl,
        expected_pop=pop,
        expected_ev_ars=ev,
        close_reason="TEST",
        closed_at=None,
        source="test",
    )


def test_edge_confirmed_with_sufficient_positive_sample():
    # 20 trades, 70% ganadores chicos, perdedores chicos -> expectancy>0, PF>1.1
    trades = [_trade("SHORT_PUT", 100.0) for _ in range(14)]
    trades += [_trade("SHORT_PUT", -50.0) for _ in range(6)]
    edges = attribute(trades, min_sample=15)
    e = edges[0]
    assert e.strategy == "SHORT_PUT"
    assert e.n == 20
    assert e.verdict == Verdict.EDGE
    assert e.expectancy_ars > 0
    assert e.profit_factor > 1.1


def test_no_edge_when_expectancy_negative():
    # muchos trades pero pierde plata en neto -> basura, debe marcarse
    trades = [_trade("SCALP_OPTION_DEMO", 40.0) for _ in range(10)]
    trades += [_trade("SCALP_OPTION_DEMO", -120.0) for _ in range(10)]
    edges = attribute(trades, min_sample=15)
    e = edges[0]
    assert e.verdict == Verdict.NO_EDGE
    assert e.expectancy_ars < 0
    assert is_strategy_blocked("SCALP_OPTION_DEMO", edges) is True


def test_small_sample_is_insufficient_and_never_blocked():
    # estrategia nueva (SMC): pocos trades, aunque pierdan no se bloquea
    trades = [_trade("STOCK_SWING_SMC", -200.0) for _ in range(5)]
    edges = attribute(trades, min_sample=15)
    e = edges[0]
    assert e.verdict == Verdict.INSUFFICIENT
    assert is_strategy_blocked("STOCK_SWING_SMC", edges) is False


def test_calibration_gap_flags_overconfident_pop():
    # PoP esperada 0.80 pero solo gana 40% -> sobreestima, flag
    trades = [_trade("SHORT_PUT", 50.0, pop=0.80) for _ in range(8)]
    trades += [_trade("SHORT_PUT", -300.0, pop=0.80) for _ in range(12)]
    edges = attribute(trades, min_sample=15)
    e = edges[0]
    assert e.expected_pop == 0.80
    assert e.calibration_gap is not None and e.calibration_gap < -0.15
    assert any("miscalibrada" in f for f in e.flags)


def test_illusory_ev_flag():
    # EV esperado positivo pero expectancy realizada <= 0
    trades = [_trade("COVERED_CALL", -10.0, ev=500.0) for _ in range(16)]
    edges = attribute(trades, min_sample=15)
    e = edges[0]
    assert e.expected_ev_ars == 500.0
    assert e.ev_realization is not None and e.ev_realization < 0
    assert any("ilusorio" in f.lower() for f in e.flags)


def test_no_result_when_pnl_missing():
    trades = [_trade("COVERED_CALL", None) for _ in range(5)]
    edges = attribute(trades, min_sample=15)
    assert edges[0].verdict == Verdict.NO_RESULT
    assert edges[0].n == 0


def _graded(grade, pnl) -> ClosedTrade:
    return ClosedTrade(
        strategy="STOCK_SWING", signal_type="SWING", realized_pnl_ars=pnl,
        expected_pop=None, expected_ev_ars=None, close_reason="X",
        closed_at=None, source="test", grade=grade,
    )


def test_blocked_grades_raises_quality_floor(tmp_path):
    """Un grade con edge negativo probado (muestra suficiente) se bloquea; el nuevo no."""
    import json
    from optionsdesk.performance.attribution import blocked_grades, block_lists
    rows = [json.dumps({"strategy": "SWING", "signal_type": "SWING_BREAKOUT",
                        "pnl_ars": -300.0, "score": 70.0, "exit_reason": "STOP"})  # grade B perdedor
            for _ in range(16)]
    rows += [json.dumps({"strategy": "SWING", "signal_type": "SWING_BREAKOUT",
                         "pnl_ars": 500.0, "score": 95.0, "exit_reason": "TP"})    # grade S, pocos
             for _ in range(3)]
    (tmp_path / "stock_demo_trades.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")

    grades = blocked_grades(tmp_path)
    setups, grades2 = block_lists(tmp_path)
    assert "B" in grades            # grade B probó perder → piso sube
    assert "S" not in grades        # grade S nuevo → protegido
    assert grades == grades2        # block_lists consistente con blocked_grades


def test_signal_edge_status_blocks_setup_or_grade():
    from datetime import datetime
    from optionsdesk.signals.stock_signals import StockSignal, signal_edge_status
    sig = StockSignal(symbol="GGAL", strategy="SWING", signal_type="SWING_BREAKOUT",
                      created_at=datetime(2026, 6, 3), entry_price=100.0, stop_price=95.0,
                      target_price=110.0, score=70.0, rationale="x")   # score 70 → grade B
    assert signal_edge_status(sig, set(), set()) == ""                          # nada bloqueado → ok
    assert signal_edge_status(sig, {"STOCK_SWING_BREAKOUT"}, set()) != ""        # setup bloqueado
    assert signal_edge_status(sig, set(), {"B"}) != ""                          # grade bloqueado


def test_attribute_by_grade_validates_quality_ranking():
    from optionsdesk.performance.attribution import attribute_by_grade
    # Grade S gana, Grade D pierde: el grading SÍ predice → edge S>0, D<=0.
    trades = [_graded("S", 200.0) for _ in range(16)]
    trades += [_graded("D", -150.0) for _ in range(16)]
    trades += [_graded(None, 50.0) for _ in range(5)]   # sin grade: se ignora
    edges = attribute_by_grade(trades, min_sample=15)
    by = {e.strategy: e for e in edges}
    assert [e.strategy for e in edges] == ["Grade S", "Grade D"]   # orden S→D
    assert by["Grade S"].verdict == Verdict.EDGE
    assert by["Grade S"].expectancy_ars > 0
    assert by["Grade D"].verdict == Verdict.NO_EDGE
    assert by["Grade D"].expectancy_ars < 0


def test_stock_demo_score_maps_to_grade(tmp_path):
    """El score persistido en el trade del demo se convierte en grade para atribuir."""
    import json
    (tmp_path / "stock_demo_trades.jsonl").write_text(
        json.dumps({"strategy": "SWING", "signal_type": "SWING_BREAKOUT",
                    "pnl_ars": 100.0, "score": 90.0, "exit_reason": "TP"}) + "\n",
        encoding="utf-8",
    )
    trades = load_all_closed_trades(tmp_path)
    assert trades[0].grade == "S"     # score 90 → grade S


def test_loader_reads_options_demo_renta(tmp_path):
    """El demo de renta de opciones (edge validado) entra a la atribución y calibra."""
    rows = [
        json.dumps({"strategy": "SHORT_PUT", "pnl_ars": 3000.0,
                    "expected_pop": 0.72, "expected_ev_ars": 3500.0, "exit_reason": "TAKE_PROFIT"})
        for _ in range(16)
    ]
    (tmp_path / "options_demo_trades.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    edges = attribute(load_all_closed_trades(tmp_path), min_sample=15)
    e = edges[0]
    assert e.strategy == "SHORT_PUT"
    assert e.expected_pop == 0.72
    assert e.ev_realization is not None
    assert e.verdict == Verdict.EDGE


def test_loader_reads_renta_expected_fields(tmp_path):
    """Una renta cerrada con predicción general (expected_pop/ev) se calibra."""
    rows = [
        json.dumps({"strategy": "SHORT_PUT", "realized_pnl_ars": 4000.0,
                    "expected_pop": 0.78, "expected_ev_ars": 4200.0,
                    "close_reason": "TAKE_PROFIT"})
        for _ in range(16)
    ]
    (tmp_path / "closed_positions.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    edges = attribute(load_all_closed_trades(tmp_path), min_sample=15)
    e = edges[0]
    assert e.strategy == "SHORT_PUT"
    assert e.expected_pop == 0.78
    assert e.expected_ev_ars == 4200.0
    assert e.ev_realization is not None      # se pudo comparar realizado vs esperado
    assert e.verdict == Verdict.EDGE


def test_loader_reads_multiple_logs(tmp_path):
    (tmp_path / "closed_positions.jsonl").write_text(
        json.dumps({"strategy": "SHORT_PUT", "realized_pnl_ars": 120.0,
                    "scalp_pop": 0.7, "close_reason": "TAKE_PROFIT"}) + "\n"
        + "linea corrupta no json\n",
        encoding="utf-8",
    )
    (tmp_path / "stock_demo_trades.jsonl").write_text(
        json.dumps({"strategy": "SWING", "signal_type": "SWING_BREAKOUT",
                    "pnl_ars": -80.0, "exit_reason": "STOP"}) + "\n",
        encoding="utf-8",
    )
    trades = load_all_closed_trades(tmp_path)
    assert len(trades) == 2
    strategies = {t.strategy for t in trades}
    assert "SHORT_PUT" in strategies
    # se agrupa por setup SMC (signal_type), no por "swing" indistinto
    assert "STOCK_SWING_BREAKOUT" in strategies
