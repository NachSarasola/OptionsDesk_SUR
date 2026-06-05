"""Atribución de performance — cierra el loop esperado-vs-realizado por estrategia.

El bot predice PoP y EV en cada trade pero, sin medir el resultado real, no sabés
si el edge es verdadero o si tu modelo de vol/costos miente. Este módulo lee los
trades cerrados (de cualquier estrategia: renta, momentum de opciones, scalp,
SMC de acciones), los agrupa, y calcula:

- estadística realizada (win rate, expectancy, profit factor, PnL total);
- calibración: ¿los trades de "70% PoP" ganan ~70%? (donde hay PoP esperada);
- realización de EV: ¿el EV prometido se cobró? (donde hay EV esperado);
- un veredicto por estrategia para asignar capital y cortar la basura.

Es AGNÓSTICO a la lógica de entrada: no toca ninguna estrategia, solo mide. Una
estrategia nueva (pocos trades) cae en MUESTRA_INSUFICIENTE y nunca se bloquea por
ruido — el capital se protege sin matar ideas en desarrollo.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Optional, Union

from optionsdesk.backtest.metrics import expectancy_ars, profit_factor, win_rate

logger = logging.getLogger(__name__)

_DATA_DIR = Path("data")

# Muestra mínima para emitir un veredicto firme. Por debajo, los números son
# provisionales (ruido) y la estrategia nunca se bloquea — protege ideas nuevas.
_MIN_SAMPLE = 15
# Profit factor mínimo para confirmar edge (cubre costos + deja margen).
_MIN_PROFIT_FACTOR = 1.10
# Brecha de calibración tolerable entre win rate realizado y PoP esperada.
_MAX_CALIBRATION_GAP = 0.15


class Verdict:
    EDGE = "EDGE_CONFIRMADO"           # muestra suficiente + expectancy>0 + PF>=1.1
    MARGINAL = "MARGINAL"             # muestra suficiente, edge chico/dudoso
    NO_EDGE = "SIN_EDGE_REALIZADO"    # muestra suficiente + expectancy<=0 -> basura
    INSUFFICIENT = "MUESTRA_INSUFICIENTE"  # pocos trades: provisional, no accionable
    NO_RESULT = "SIN_RESULTADOS"      # no hay PnL realizado cargado para la estrategia


@dataclass(frozen=True)
class ClosedTrade:
    strategy: str
    signal_type: str
    realized_pnl_ars: Optional[float]
    expected_pop: Optional[float]       # 0..1
    expected_ev_ars: Optional[float]
    close_reason: str
    closed_at: Optional[datetime]
    source: str
    grade: Optional[str] = None         # S|A|B|C|D del score SMC (si aplica)


@dataclass
class StrategyEdge:
    strategy: str
    n: int                              # trades con resultado realizado
    win_rate: float
    expectancy_ars: float
    profit_factor: Optional[float]
    total_pnl_ars: float
    expected_pop: Optional[float]       # PoP esperada promedio (si se predijo)
    calibration_gap: Optional[float]    # win_rate_real - PoP_esperada (negativo = sobreestimó)
    expected_ev_ars: Optional[float]    # EV esperado promedio (si se predijo)
    ev_realization: Optional[float]     # expectancy_real / EV_esperado (1.0 = perfecto)
    verdict: str
    flags: list[str] = field(default_factory=list)


# ── Carga de trades cerrados desde los logs existentes ─────────────────────────

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _parse_dt(value) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None


def _f(value) -> Optional[float]:
    try:
        return float(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def _from_closed_positions(rec: dict) -> ClosedTrade:
    """Posiciones cerradas del monitor (renta + scalp). Scalp trae PoP/EV esperados."""
    strategy = str(rec.get("strategy", "DESCONOCIDA")).upper()
    # Campos generales (renta) con fallback a los específicos de scalp.
    exp_pop = rec.get("expected_pop")
    exp_pop = exp_pop if exp_pop is not None else rec.get("scalp_pop")
    exp_ev = rec.get("expected_ev_ars")
    exp_ev = exp_ev if exp_ev is not None else rec.get("scalp_expected_value_ars")
    return ClosedTrade(
        strategy=strategy,
        signal_type=str(rec.get("scalp_setup") or rec.get("signal_type") or strategy),
        realized_pnl_ars=_f(rec.get("realized_pnl_ars")),
        expected_pop=_f(exp_pop),
        expected_ev_ars=_f(exp_ev),
        close_reason=str(rec.get("close_reason", "")),
        closed_at=_parse_dt(rec.get("closed_at")),
        source="closed_positions",
    )


def _from_stock_demo(rec: dict) -> ClosedTrade:
    """Trades del demo de acciones (SMC/swing). Resultado sí, predicción aún no.

    Se agrupa por SETUP (signal_type) para medir qué estrategia SMC tiene edge:
    pullback en descuento vs breakout sobre liquidez, no un "swing" indistinto.
    """
    setup = str(rec.get("signal_type") or rec.get("strategy") or "SWING").upper()
    return ClosedTrade(
        strategy=setup if setup.startswith("STOCK") else f"STOCK_{setup}",
        signal_type=setup,
        realized_pnl_ars=_f(rec.get("pnl_ars")),
        expected_pop=None,
        expected_ev_ars=None,
        close_reason=str(rec.get("exit_reason", "")),
        closed_at=_parse_dt(rec.get("exit_ts")),
        source="stock_demo",
        grade=_grade_from_score(_f(rec.get("score"))),
    )


def _grade_from_score(score: Optional[float]) -> Optional[str]:
    """Grade SMC (S/A/B/C/D) desde el score — única fuente de verdad: signal_grade."""
    if score is None:
        return None
    try:
        from optionsdesk.signals.stock_signals import signal_grade
        return signal_grade(float(score))
    except Exception:
        return None


def _from_options_demo(rec: dict) -> ClosedTrade:
    """Trades del demo de renta de opciones (covered call / short put) — el edge
    validado. Trae PoP/EV esperados al abrir → calibra la venta de prima."""
    strategy = str(rec.get("strategy", "RENTA")).upper()
    return ClosedTrade(
        strategy=strategy,
        signal_type=strategy,
        realized_pnl_ars=_f(rec.get("pnl_ars")),
        expected_pop=_f(rec.get("expected_pop")),
        expected_ev_ars=_f(rec.get("expected_ev_ars")),
        close_reason=str(rec.get("exit_reason", "")),
        closed_at=_parse_dt(rec.get("exit_ts")),
        source="options_demo",
    )


def _from_scalp_demo(rec: dict) -> ClosedTrade:
    """Trades del demo de scalp de opciones."""
    setup = str(rec.get("setup") or rec.get("action") or "SCALP").upper()
    return ClosedTrade(
        strategy="SCALP_OPTION_DEMO",
        signal_type=setup,
        realized_pnl_ars=_f(rec.get("pnl_ars")),
        expected_pop=None,
        expected_ev_ars=None,
        close_reason=str(rec.get("exit_reason", "")),
        closed_at=_parse_dt(rec.get("exit_ts")),
        source="scalp_demo",
    )


# (filename, mapper). Si un log no existe, se ignora sin error.
_SOURCES: list[tuple[str, Callable[[dict], ClosedTrade]]] = [
    ("closed_positions.jsonl", _from_closed_positions),
    ("stock_demo_trades.jsonl", _from_stock_demo),
    ("scalp_demo_trades.jsonl", _from_scalp_demo),
    ("options_demo_trades.jsonl", _from_options_demo),
]


def _coerce_since(since: "Union[str, date, datetime, None]") -> Optional[datetime]:
    if since is None:
        return None
    if isinstance(since, datetime):
        return since
    if isinstance(since, date):
        return datetime(since.year, since.month, since.day)
    try:
        return datetime.fromisoformat(str(since))
    except (ValueError, TypeError):
        return None


def _on_or_after(closed_at: Optional[datetime], since: Optional[datetime]) -> bool:
    """Comparación tz-safe. Sin `since` o sin fecha del trade → se conserva."""
    if since is None or closed_at is None:
        return True
    ca, s = closed_at, since
    if (ca.tzinfo is None) != (s.tzinfo is None):   # normalizar mezclas aware/naive
        ca, s = ca.replace(tzinfo=None), s.replace(tzinfo=None)
    return ca >= s


def load_all_closed_trades(
    data_dir: Optional[Path] = None,
    *,
    since: "Union[str, date, datetime, None]" = None,
) -> list[ClosedTrade]:
    """Carga y normaliza los trades cerrados de todos los logs disponibles.

    `since` (fecha/ISO): ignora trades cerrados antes — baseline de aprendizaje para
    resetear tras deshabilitar una estrategia. Los trades sin fecha se conservan.
    """
    root = Path(data_dir or _DATA_DIR)
    cutoff = _coerce_since(since)
    trades: list[ClosedTrade] = []
    for filename, mapper in _SOURCES:
        for rec in _read_jsonl(root / filename):
            try:
                t = mapper(rec)
            except Exception as exc:  # un registro corrupto no debe tumbar el panel
                logger.debug("trade ilegible en %s: %s", filename, exc)
                continue
            if _on_or_after(t.closed_at, cutoff):
                trades.append(t)
    return trades


# ── Atribución ────────────────────────────────────────────────────────────────

def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _edge_for_group(strategy: str, group: list[ClosedTrade], min_sample: int) -> StrategyEdge:
    settled = [t for t in group if t.realized_pnl_ars is not None]
    pnls = [float(t.realized_pnl_ars) for t in settled]  # type: ignore[arg-type]

    if not pnls:
        return StrategyEdge(
            strategy=strategy, n=0, win_rate=0.0, expectancy_ars=0.0,
            profit_factor=None, total_pnl_ars=0.0, expected_pop=None,
            calibration_gap=None, expected_ev_ars=None, ev_realization=None,
            verdict=Verdict.NO_RESULT,
        )

    n = len(pnls)
    wr = win_rate(pnls)
    exp = expectancy_ars(pnls)
    pf = profit_factor(pnls)
    total = sum(pnls)

    exp_pop = _mean([t.expected_pop for t in settled if t.expected_pop is not None])
    exp_ev = _mean([t.expected_ev_ars for t in settled if t.expected_ev_ars is not None])
    calib_gap = (wr - exp_pop) if exp_pop is not None else None
    ev_real = (exp / exp_ev) if (exp_ev is not None and abs(exp_ev) > 1e-9) else None

    flags: list[str] = []
    if calib_gap is not None and abs(calib_gap) > _MAX_CALIBRATION_GAP:
        side = "sobreestima" if calib_gap < 0 else "subestima"
        flags.append(f"PoP miscalibrada ({side} el acierto)")
    if exp_ev is not None and exp_ev > 0 and exp <= 0:
        flags.append("EV ilusorio: prometía positivo, realizó ≤0")

    verdict = _verdict(n, exp, pf, min_sample)
    return StrategyEdge(
        strategy=strategy, n=n, win_rate=round(wr, 3), expectancy_ars=round(exp, 2),
        profit_factor=round(pf, 3) if pf is not None else None,
        total_pnl_ars=round(total, 2),
        expected_pop=round(exp_pop, 3) if exp_pop is not None else None,
        calibration_gap=round(calib_gap, 3) if calib_gap is not None else None,
        expected_ev_ars=round(exp_ev, 2) if exp_ev is not None else None,
        ev_realization=round(ev_real, 3) if ev_real is not None else None,
        verdict=verdict, flags=flags,
    )


def _verdict(n: int, expectancy: float, pf: Optional[float], min_sample: int) -> str:
    if n < min_sample:
        return Verdict.INSUFFICIENT
    if expectancy > 0 and (pf is None or pf >= _MIN_PROFIT_FACTOR):
        return Verdict.EDGE
    if expectancy <= 0:
        return Verdict.NO_EDGE
    return Verdict.MARGINAL


def attribute(
    trades: list[ClosedTrade],
    *,
    min_sample: int = _MIN_SAMPLE,
) -> list[StrategyEdge]:
    """Agrupa por estrategia y devuelve el edge realizado, ordenado por PnL total."""
    groups: dict[str, list[ClosedTrade]] = {}
    for t in trades:
        groups.setdefault(t.strategy, []).append(t)
    edges = [_edge_for_group(strat, grp, min_sample) for strat, grp in groups.items()]
    return sorted(edges, key=lambda e: e.total_pnl_ars, reverse=True)


def attribute_by_grade(
    trades: list[ClosedTrade],
    *,
    min_sample: int = _MIN_SAMPLE,
) -> list[StrategyEdge]:
    """Edge por grade SMC (S/A/B/C/D) — ¿el grading predice el resultado?

    Valida el sistema de calidad de señales: si los grades altos (S/A) no superan a
    los bajos (C/D), el grading es ruido y hay que recalibrarlo. Solo agrupa trades
    que tienen grade (señales SMC de acciones). Orden S→D.
    """
    groups: dict[str, list[ClosedTrade]] = {}
    for t in trades:
        if t.grade:
            groups.setdefault(t.grade, []).append(t)
    edges = [_edge_for_group(f"Grade {g}", grp, min_sample) for g, grp in groups.items()]
    order = {"S": 0, "A": 1, "B": 2, "C": 3, "D": 4}
    return sorted(edges, key=lambda e: order.get(e.strategy.split()[-1], 9))


def blocked_strategies(
    data_dir: Optional[Path] = None,
    *,
    min_sample: int = _MIN_SAMPLE,
    since: "Union[str, date, datetime, None]" = None,
) -> set[str]:
    """Estrategias con edge realizado negativo y muestra suficiente (basura a cortar).

    Las estrategias nuevas (muestra chica) nunca aparecen acá → las ideas en
    desarrollo se protegen y solo se corta lo que demostró perder plata.
    `since` respeta el baseline de aprendizaje (no bloquear por legacy).
    Devuelve claves en MAYÚSCULA para comparar sin ambigüedad.
    """
    edges = attribute(load_all_closed_trades(data_dir, since=since), min_sample=min_sample)
    return {e.strategy.upper() for e in edges if e.verdict == Verdict.NO_EDGE}


def blocked_grades(
    data_dir: Optional[Path] = None,
    *,
    min_sample: int = _MIN_SAMPLE,
    since: "Union[str, date, datetime, None]" = None,
) -> set[str]:
    """Grades SMC (S/A/B/C/D) con edge realizado negativo y muestra suficiente.

    El grade AGREGA muestra a través de setups → aprende el piso de calidad más
    rápido que el bloqueo por setup. Si grade C/D pierde de forma probada, el sistema
    sube la vara y deja de operar ese nivel. Los grades nuevos nunca se bloquean.
    """
    trades = load_all_closed_trades(data_dir, since=since)
    return {
        e.strategy.split()[-1]
        for e in attribute_by_grade(trades, min_sample=min_sample)
        if e.verdict == Verdict.NO_EDGE
    }


def block_lists(
    data_dir: Optional[Path] = None,
    *,
    min_sample: int = _MIN_SAMPLE,
    since: "Union[str, date, datetime, None]" = None,
) -> tuple[set[str], set[str]]:
    """(setups_bloqueados, grades_bloqueados) en una sola pasada de datos.

    Fuente única de verdad para 'qué señal es basura', consultada por el demo y el
    panel. Una señal se bloquea si su setup O su grade probaron edge negativo.
    """
    trades = load_all_closed_trades(data_dir, since=since)
    setups = {e.strategy.upper() for e in attribute(trades, min_sample=min_sample)
              if e.verdict == Verdict.NO_EDGE}
    grades = {e.strategy.split()[-1] for e in attribute_by_grade(trades, min_sample=min_sample)
              if e.verdict == Verdict.NO_EDGE}
    return setups, grades


def is_strategy_blocked(
    strategy: str,
    edges: list[StrategyEdge],
) -> bool:
    """True solo si la estrategia tiene muestra suficiente y edge realizado negativo.

    Advisory para frenar basura SIN matar ideas nuevas: una estrategia con pocos
    trades (MUESTRA_INSUFICIENTE) nunca se bloquea. Pensado para que el recommender
    o el demo lo consulten opcionalmente.
    """
    key = strategy.upper()
    for e in edges:
        if e.strategy.upper() == key:
            return e.verdict == Verdict.NO_EDGE
    return False
