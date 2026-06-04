"""Runner continuo de las demos (renta de opciones + swing SMC de acciones) que
aprende de sí mismo en cada ciclo.

Cada ciclo:
1. Lee los trades cerrados acumulados y construye el estado de auto-aprendizaje:
   - sizing adaptativo (half-Kelly desde win rate / RR realizado, `adaptive.py`);
   - gate anti-basura (no abrir setups con edge realizado negativo probado);
   - circuit breaker: si el drawdown realizado supera el límite, no abre nuevas
     (sigue gestionando las abiertas).
2. Tickea ambos demos con ese estado.
3. Persiste un status (`data/demo_runner_status.json`) para que el dashboard
   muestre qué está haciendo el runner.

Correr en vivo:  python -m optionsdesk.backtest.demo_runner --interval 300
Un solo ciclo:   python -m optionsdesk.backtest.demo_runner --once
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from optionsdesk.backtest.adaptive import AdaptiveContext, analyze_performance
from optionsdesk.performance.attribution import blocked_strategies, load_all_closed_trades

logger = logging.getLogger(__name__)

_BA = ZoneInfo("America/Argentina/Buenos_Aires")
_STATUS_FILE = Path("data/demo_runner_status.json")
_DEFAULT_CAPITAL = 1_800_000.0
_MAX_DRAWDOWN_PCT = 8.0   # freno de aperturas si el equity cae más de esto


@dataclass
class SelfLearningState:
    adaptive: AdaptiveContext
    blocked: set
    halted: bool                 # True → drawdown excedido, no abrir nuevas
    realized_pnl_ars: float
    drawdown_pct: float
    note: str


@dataclass
class CycleInputs:
    """Datos de un ciclo. Lo que el `fetch` debe devolver (todo opcional)."""
    chain: object = None
    options_candidates: list = field(default_factory=list)
    stock_quotes: dict = field(default_factory=dict)       # {symbol: Quote}
    stock_signals: list = field(default_factory=list)       # flat list[StockSignal]
    caucion_tna_pct: float = 60.0


@dataclass
class RunnerStatus:
    last_tick: str
    market_open: bool
    halted: bool
    mode: str
    realized_pnl_ars: float
    drawdown_pct: float
    stock_open: int
    options_open: int
    blocked: list
    half_kelly_pct: float
    note: str


class _PnlTrade:
    """Adaptador mínimo para analyze_performance (solo necesita .pnl_ars)."""
    __slots__ = ("pnl_ars",)

    def __init__(self, pnl: float) -> None:
        self.pnl_ars = pnl


def _trade_sort_key(t) -> str:
    """Clave de orden cronológico tz-safe: isoformat string (None → '' va primero).

    Evita el TypeError de comparar datetimes tz-aware con naive ordenando por string.
    """
    ca = getattr(t, "closed_at", None)
    return ca.isoformat() if ca is not None else ""


# ── Cerebro de auto-aprendizaje ────────────────────────────────────────────────

def compute_self_learning_state(
    *,
    data_dir: Optional[Path] = None,
    capital: float = _DEFAULT_CAPITAL,
    max_drawdown_pct: float = _MAX_DRAWDOWN_PCT,
    since=None,
    lookback: Optional[int] = None,
) -> SelfLearningState:
    """Construye el estado del runner desde los trades cerrados acumulados.

    Aprende de sí mismo: sizing Kelly por win rate realizado, bloqueo de setups
    perdedores (con muestra suficiente; los nuevos se protegen) y freno por drawdown.

    Aprende del comportamiento ACTUAL, no de la historia muerta: `since` ignora trades
    anteriores (baseline tras deshabilitar una estrategia) y el drawdown se computa
    sobre los últimos `lookback` trades (no all-time → no freno permanente por pérdidas
    viejas). Ambos toman default de settings.
    """
    try:
        from optionsdesk.config.settings import settings as _s
        if since is None:
            since = getattr(_s, "demo_learning_since", None)
        if lookback is None:
            lookback = getattr(_s, "demo_learning_lookback", 40)
    except Exception:
        lookback = lookback or 40

    trades = load_all_closed_trades(data_dir, since=since)
    settled = [t for t in trades if t.realized_pnl_ars is not None]
    settled.sort(key=_trade_sort_key)
    pnls = [float(t.realized_pnl_ars) for t in settled]

    adaptive = analyze_performance([_PnlTrade(p) for p in pnls])
    blocked = blocked_strategies(data_dir, since=since)

    # Drawdown peak-to-trough sobre la ventana RECIENTE (no all-time).
    window = pnls[-lookback:] if lookback and lookback > 0 else pnls
    equity = peak = max_dd = 0.0
    for p in window:
        equity += p
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
    dd_pct = round(abs(max_dd) / capital * 100.0, 2) if capital > 0 else 0.0
    halted = dd_pct >= max_drawdown_pct
    note = (
        f"FRENO: drawdown {dd_pct:.1f}% ≥ {max_drawdown_pct:.0f}% — solo gestiona abiertas"
        if halted else f"operando | drawdown {dd_pct:.1f}% | modo {adaptive.mode}"
    )
    return SelfLearningState(
        adaptive=adaptive,
        blocked=blocked,
        halted=halted,
        realized_pnl_ars=round(sum(pnls), 2),   # realizado total (post-baseline)
        drawdown_pct=dd_pct,                      # drawdown de la ventana reciente
        note=note,
    )


# ── Un ciclo ───────────────────────────────────────────────────────────────────

def run_cycle(
    fetch: Callable[[], CycleInputs],
    *,
    state: Optional[SelfLearningState] = None,
    now: Optional[datetime] = None,
    data_dir: Optional[Path] = None,
    capital: float = _DEFAULT_CAPITAL,
    market_open: bool = True,
    status_path: Optional[Path] = None,
) -> RunnerStatus:
    """Ejecuta un ciclo: aprende, fetchea, tickea ambos demos, persiste status."""
    current = now or datetime.now(_BA)
    state = state or compute_self_learning_state(data_dir=data_dir, capital=capital)

    inputs = CycleInputs()
    try:
        inputs = fetch() or CycleInputs()
    except Exception as exc:
        logger.warning("fetch del runner falló: %s", exc)

    # El runner continuo acumula el historial de IV ATM (activa iv_rank a futuro),
    # sin depender de que el dashboard esté abierto.
    if inputs.options_candidates:
        try:
            from optionsdesk.data.iv_history import record_daily_iv, representative_atm_iv
            record_daily_iv(representative_atm_iv(inputs.options_candidates), symbol="GGAL")
        except Exception as exc:
            logger.debug("record_daily_iv en runner falló: %s", exc)

    stock_open = options_open = 0

    # Demo de acciones (swing SMC). En freno: no abre swings nuevos (max_swings=0).
    if inputs.stock_quotes and inputs.stock_signals:
        try:
            from optionsdesk.backtest.stock_demo import run_stock_demo_tick
            res = run_stock_demo_tick(
                inputs.stock_quotes, inputs.stock_signals, now=current,
                adaptive_context=state.adaptive, block_junk=True,
                max_swings=(0 if state.halted else None), max_scalps=0,
            )
            stock_open = len(res.positions)
        except Exception as exc:
            logger.warning("tick de acciones falló: %s", exc)

    # Demo de renta de opciones. allow_new=False en freno.
    if inputs.chain is not None:
        try:
            from optionsdesk.backtest.options_demo import run_options_demo_tick
            res = run_options_demo_tick(
                inputs.chain, inputs.options_candidates, now=current,
                adaptive_context=state.adaptive, caucion_tna_pct=inputs.caucion_tna_pct,
                block_junk=True, allow_new=not state.halted,
            )
            options_open = len(res.get("positions", []))
        except Exception as exc:
            logger.warning("tick de opciones falló: %s", exc)

    status = RunnerStatus(
        last_tick=current.isoformat(timespec="seconds"),
        market_open=market_open,
        halted=state.halted,
        mode=state.adaptive.mode,
        realized_pnl_ars=state.realized_pnl_ars,
        drawdown_pct=state.drawdown_pct,
        stock_open=stock_open,
        options_open=options_open,
        blocked=sorted(state.blocked),
        half_kelly_pct=round(state.adaptive.half_kelly_pct, 4),
        note=state.note,
    )
    _write_status(status, status_path)
    return status


def _write_status(status: RunnerStatus, path: Optional[Path] = None) -> None:
    target = Path(path or _STATUS_FILE)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(status), ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.debug("no se pudo escribir el status del runner: %s", exc)


def load_runner_status(path: Optional[Path] = None) -> Optional[dict]:
    """Lee el último status del runner (para mostrarlo en el dashboard)."""
    target = Path(path or _STATUS_FILE)
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# ── Loop continuo ──────────────────────────────────────────────────────────────

def _market_open(now: datetime) -> bool:
    """Rueda BYMA: lun–vie 11:00–17:00 ART."""
    ts = now.astimezone(_BA) if now.tzinfo else now.replace(tzinfo=_BA)
    if ts.weekday() >= 5:
        return False
    return dt_time(11, 0) <= ts.time() < dt_time(17, 0)


def run_forever(
    fetch: Callable[[], CycleInputs],
    *,
    interval_s: int = 300,
    market_hours: bool = True,
    capital: float = _DEFAULT_CAPITAL,
    max_cycles: Optional[int] = None,
) -> None:
    """Corre ciclos en loop. Fuera de rueda (si market_hours) solo espera."""
    cycles = 0
    while True:
        now = datetime.now(_BA)
        is_open = _market_open(now)
        if is_open or not market_hours:
            run_cycle(fetch, now=now, capital=capital, market_open=is_open)
            logger.info("ciclo %d ok (%s)", cycles + 1, now.isoformat(timespec="seconds"))
        cycles += 1
        if max_cycles is not None and cycles >= max_cycles:
            break
        time.sleep(max(interval_s, 5))


# ── Wiring de datos live (seam de integración) ─────────────────────────────────

def build_live_fetch() -> Callable[[], CycleInputs]:
    """Arma el `fetch` live reusando provider + scanners + historiales del dashboard.

    Best-effort y guardado: si una parte falla, devuelve lo que pudo (el ciclo
    igual aprende y persiste status). Las acciones usan los helpers de historial del
    dashboard; las opciones, los scanners de renta.
    """
    def _fetch() -> CycleInputs:
        inputs = CycleInputs()
        # ── Opciones (renta) ────────────────────────────────────────────────
        try:
            from optionsdesk.ui.dashboard import _build_provider, _load_expiry_calendar
            from optionsdesk.core.benchmark import Benchmark
            from optionsdesk.strategies.covered_call import CoveredCallConfig, CoveredCallScanner
            from optionsdesk.strategies.short_put import ShortPutConfig, ShortPutScanner
            from optionsdesk.config.settings import settings as _s

            provider = _build_provider()
            chain = provider.get_options_chain()
            caucion = provider.get_caucion_tna() or _s.default_caucion_tna
            inputs.chain = chain
            inputs.caucion_tna_pct = caucion
            if chain is not None and getattr(chain, "options", None):
                expiry_cal = _load_expiry_calendar()
                benchmark = Benchmark(caucion_tna_pct=caucion, days=30)
                cc = CoveredCallScanner(CoveredCallConfig(), expiry_cal).scan(chain, benchmark)
                sp = ShortPutScanner(ShortPutConfig(), expiry_cal).scan(chain, benchmark)
                inputs.options_candidates = list(cc) + list(sp)

                # ── Acciones (swing SMC) ───────────────────────────────────
                try:
                    from optionsdesk.ui.dashboard import _load_symbol_daily, _load_symbol_weekly
                    from optionsdesk.signals.stock_signals import scan_stock_symbol
                    symbols = sorted(s.upper() for s in _s.stock_universe_symbols())
                    if symbols:
                        quotes = provider.get_quotes(symbols)
                        inputs.stock_quotes = quotes
                        sigs: list = []
                        for sym in symbols:
                            q = quotes.get(sym)
                            if q is None:
                                continue
                            daily = _load_symbol_daily(sym, days=180)
                            weekly = _load_symbol_weekly(sym, weeks=52)
                            sigs.extend(scan_stock_symbol(sym, q, daily=daily, weekly=weekly))
                        inputs.stock_signals = sigs
                except Exception as exc:
                    logger.warning("fetch de acciones falló: %s", exc)
        except Exception as exc:
            logger.warning("fetch de opciones falló: %s", exc)
        return inputs

    return _fetch


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Runner continuo de las demos con auto-aprendizaje.")
    parser.add_argument("--interval", type=int, default=300, help="segundos entre ciclos")
    parser.add_argument("--once", action="store_true", help="un solo ciclo y salir")
    parser.add_argument("--ignore-hours", action="store_true", help="correr fuera de la rueda BYMA")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fetch = build_live_fetch()
    if args.once:
        status = run_cycle(fetch, now=datetime.now(_BA), market_open=_market_open(datetime.now(_BA)))
        print(json.dumps(asdict(status), ensure_ascii=False, indent=2))
    else:
        run_forever(fetch, interval_s=args.interval, market_hours=not args.ignore_hours)


if __name__ == "__main__":
    main()
