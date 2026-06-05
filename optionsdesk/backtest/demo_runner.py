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
from optionsdesk.performance.attribution import block_lists, load_all_closed_trades

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
    lab_infinite: bool = False


@dataclass
class CycleInputs:
    """Datos de un ciclo. Lo que el `fetch` debe devolver (todo opcional)."""
    chain: object = None
    options_candidates: list = field(default_factory=list)
    stock_quotes: dict = field(default_factory=dict)       # {symbol: Quote}
    stock_signals: list = field(default_factory=list)       # flat list[StockSignal]
    caucion_tna_pct: float = 60.0
    stock_universe: list[str] = field(default_factory=list)
    universe_source: str = ""
    market_data_source: str = ""
    fetch_error: str = ""


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
    regime: str = ""   # regimen vivo + genoma desplegado por la politica del arbol
    lab_infinite: bool = False
    stock_universe: list[str] = field(default_factory=list)
    universe_source: str = ""
    market_data_source: str = ""
    policy_status: str = ""
    fetch_error: str = ""


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


def _lab_capital() -> float:
    try:
        from optionsdesk.config.settings import settings as _s
        return float(getattr(_s, "lab_infinite_capital_ars", 1_000_000_000_000.0))
    except Exception:
        return 1_000_000_000_000.0


# ── Cerebro de auto-aprendizaje ────────────────────────────────────────────────

def compute_self_learning_state(
    *,
    data_dir: Optional[Path] = None,
    capital: float = _DEFAULT_CAPITAL,
    max_drawdown_pct: float = _MAX_DRAWDOWN_PCT,
    since=None,
    lookback: Optional[int] = None,
    lab_infinite: bool = False,
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
    blocked, blocked_grade = block_lists(data_dir, since=since)
    if blocked_grade:
        blocked = blocked | {f"Grade {g}" for g in blocked_grade}

    # Drawdown peak-to-trough sobre la ventana RECIENTE (no all-time).
    window = pnls[-lookback:] if lookback and lookback > 0 else pnls
    equity = peak = max_dd = 0.0
    for p in window:
        equity += p
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
    dd_pct = round(abs(max_dd) / capital * 100.0, 2) if capital > 0 else 0.0
    halted = False if lab_infinite else dd_pct >= max_drawdown_pct
    note = (
        f"LAB_INFINITE | drawdown observado {dd_pct:.1f}% | sin breaker"
        if lab_infinite
        else f"FRENO: drawdown {dd_pct:.1f}% ≥ {max_drawdown_pct:.0f}% — solo gestiona abiertas"
        if halted else f"operando | drawdown {dd_pct:.1f}% | modo {adaptive.mode}"
    )
    return SelfLearningState(
        adaptive=adaptive,
        blocked=blocked,
        halted=halted,
        realized_pnl_ars=round(sum(pnls), 2),   # realizado total (post-baseline)
        drawdown_pct=dd_pct,                      # drawdown de la ventana reciente
        note=note,
        lab_infinite=lab_infinite,
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
    lab_infinite: Optional[bool] = None,
) -> RunnerStatus:
    """Ejecuta un ciclo: aprende, fetchea, tickea ambos demos, persiste status."""
    current = now or datetime.now(_BA)
    if lab_infinite is None:
        try:
            from optionsdesk.config.settings import settings as _s
            lab_infinite = getattr(_s, "stock_demo_mode", "LAB_INFINITE") == "LAB_INFINITE"
        except Exception:
            lab_infinite = False
    effective_capital = (
        _lab_capital()
        if lab_infinite and capital == _DEFAULT_CAPITAL
        else capital
    )
    state = state or compute_self_learning_state(
        data_dir=data_dir,
        capital=effective_capital,
        lab_infinite=bool(lab_infinite),
    )

    # Controlador de estrategia (antes del fetch, para que las señales de ESTE ciclo
    # usen el set desplegado). Jerarquia:
    #   1. Si hay una POLITICA del arbol (strategy_tree) entrenada → desplegar el
    #      genoma del REGIMEN vivo (contexto actual). Es el cerebro entrenado.
    #   2. Si no hay politica → fallback al param_learner 1-D (bootstrap online).
    # Best-effort: si todo falla, el ciclo opera con los parametros de fabrica.
    regime_note = ""
    policy_status = "factory_defaults"
    deployed = None
    try:
        from optionsdesk.backtest.strategy_tree import deploy_for_regime, load_policy_doc
        if load_policy_doc() is not None:
            regime = _live_regime()
            deployed = deploy_for_regime(regime, data_dir=data_dir)
            policy_status = "strategy_tree" if deployed else "strategy_tree_not_deployable"
            regime_note = f"regimen {regime} → {deployed}" if deployed else ""
        else:
            from optionsdesk.backtest.param_learner import run_learner_step
            from optionsdesk.config.settings import settings as _s
            run_learner_step(data_dir=data_dir, now=current,
                             since=getattr(_s, "demo_learning_since", None))
            policy_status = "param_learner"
    except Exception as exc:
        logger.debug("controlador de estrategia falló: %s", exc)

    if policy_status == "strategy_tree_not_deployable":
        try:
            from optionsdesk.backtest.param_learner import run_learner_step
            from optionsdesk.config.settings import settings as _s

            run_learner_step(
                data_dir=data_dir,
                now=current,
                since=getattr(_s, "demo_learning_since", None),
            )
            policy_status = "param_learner"
            regime_note = f"{regime_note or 'policy'} | tree no deployable"
        except Exception as exc:
            logger.debug("fallback param_learner fallo: %s", exc)

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
                adaptive_context=state.adaptive, block_junk=not lab_infinite,
                max_swings=(None if lab_infinite else 0 if state.halted else None),
                max_scalps=None if lab_infinite else 0,
                lab_infinite=bool(lab_infinite),
                capital=effective_capital,
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
                block_junk=not lab_infinite, allow_new=bool(lab_infinite) or not state.halted,
                lab_infinite=bool(lab_infinite), capital=effective_capital,
            )
            options_open = len(res.get("positions", []))
        except Exception as exc:
            logger.warning("tick de opciones falló: %s", exc)

    status = RunnerStatus(
        last_tick=current.isoformat(timespec="seconds"),
        market_open=market_open,
        halted=False if lab_infinite else state.halted,
        mode="LAB_INFINITE" if lab_infinite else state.adaptive.mode,
        realized_pnl_ars=state.realized_pnl_ars,
        drawdown_pct=state.drawdown_pct,
        stock_open=stock_open,
        options_open=options_open,
        blocked=sorted(state.blocked),
        half_kelly_pct=round(state.adaptive.half_kelly_pct, 4),
        note=state.note,
        regime=regime_note,
        lab_infinite=bool(lab_infinite),
        stock_universe=list(inputs.stock_universe),
        universe_source=inputs.universe_source,
        market_data_source=inputs.market_data_source,
        policy_status=policy_status,
        fetch_error=inputs.fetch_error,
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

def _live_regime() -> str:
    """Clasifica el regimen de mercado vigente desde el historial diario de GGAL.

    Best-effort: si no hay historial, devuelve UNKNOWN (la politica cae al global).
    """
    try:
        from optionsdesk.data.history import UnderlyingHistory
        from optionsdesk.backtest.strategy_context import classify_regime, UNKNOWN_REGIME
        df = UnderlyingHistory().daily("GGAL", days=180, allow_synthetic=False)
        if df is None or df.empty:
            return UNKNOWN_REGIME
        return classify_regime(df)
    except Exception as exc:
        logger.debug("_live_regime fallo: %s", exc)
        from optionsdesk.backtest.strategy_context import UNKNOWN_REGIME
        return UNKNOWN_REGIME


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
    lab_infinite: Optional[bool] = None,
) -> None:
    """Corre ciclos en loop. Fuera de rueda (si market_hours) solo espera."""
    cycles = 0
    while True:
        now = datetime.now(_BA)
        is_open = _market_open(now)
        if is_open or not market_hours:
            run_cycle(
                fetch,
                now=now,
                capital=capital,
                market_open=is_open,
                lab_infinite=lab_infinite,
            )
            logger.info("ciclo %d ok (%s)", cycles + 1, now.isoformat(timespec="seconds"))
        cycles += 1
        if max_cycles is not None and cycles >= max_cycles:
            break
        time.sleep(max(interval_s, 5))


# ── Wiring de datos live (seam de integración) ─────────────────────────────────

def _build_runner_provider():
    """Fuente live del runner: Primary primero, IOL despues, BYMA Open al final."""
    from optionsdesk.config.settings import settings as _s

    preference = getattr(_s, "market_data_provider", "AUTO")

    if preference in {"AUTO", "PRIMARY"} and _s.is_primary_configured():
        try:
            from optionsdesk.data.providers.primary import PrimaryProvider
            provider = PrimaryProvider()
            provider.connect()
            return provider
        except Exception as exc:
            logger.warning("Primary no disponible para demo_runner: %s", exc)
            if preference == "PRIMARY":
                raise

    if preference in {"AUTO", "IOL"} and _s.is_iol_configured():
        try:
            from optionsdesk.data.providers.iol import IOLProvider
            provider = IOLProvider()
            provider.connect()
            return provider
        except Exception as exc:
            logger.warning("IOL no disponible para demo_runner: %s", exc)
            if preference == "IOL":
                raise

    from optionsdesk.data.providers.byma_open import BymaOpenProvider
    return BymaOpenProvider()


def _build_live_fetch_runtime() -> Callable[[], CycleInputs]:
    def _fetch() -> CycleInputs:
        inputs = CycleInputs()
        try:
            from optionsdesk.config.settings import settings as _s
            from optionsdesk.core.benchmark import Benchmark
            from optionsdesk.core.instruments import load_expiry_calendar
            from optionsdesk.data.history import UnderlyingHistory, weekly_from_daily
            from optionsdesk.data.universe import rank_stock_universe
            from optionsdesk.signals.stock_signals import scan_stock_symbol
            from optionsdesk.strategies.covered_call import CoveredCallConfig, CoveredCallScanner
            from optionsdesk.strategies.short_put import ShortPutConfig, ShortPutScanner

            provider = _build_runner_provider()
            inputs.market_data_source = provider.get_health().source

            base_symbols = sorted(s.upper() for s in _s.stock_universe_symbols())
            initial_quotes = provider.get_quotes(base_symbols) if base_symbols else {}
            ranked = rank_stock_universe(
                base_symbols,
                quotes=initial_quotes,
                top_n=_s.stock_universe_top_n,
                lookback=_s.stock_universe_volume_lookback,
            )
            selected = [m.symbol for m in ranked if m.score > 0] or base_symbols[:_s.stock_universe_top_n]
            inputs.stock_universe = selected
            inputs.universe_source = ",".join(sorted({m.source for m in ranked})) if ranked else "settings"

            missing = [s for s in selected if s not in initial_quotes]
            if missing:
                initial_quotes.update(provider.get_quotes(missing))
            selected_set = set(selected)
            inputs.stock_quotes = {
                s: q for s, q in initial_quotes.items()
                if s in selected_set and q is not None
            }

            hist = UnderlyingHistory()
            sigs: list = []
            for sym in selected:
                q = inputs.stock_quotes.get(sym)
                if q is None:
                    continue
                daily = hist.daily(sym, days=180, allow_synthetic=False)
                weekly = weekly_from_daily(daily) if daily is not None and not daily.empty else None
                sigs.extend(scan_stock_symbol(sym, q, daily=daily, weekly=weekly))
            inputs.stock_signals = sigs

            chain = provider.get_options_chain()
            inputs.chain = chain
            inputs.caucion_tna_pct = provider.get_caucion_tna() or _s.default_caucion_tna
            if chain is not None and getattr(chain, "options", None):
                expiry_cal = load_expiry_calendar()
                benchmark = Benchmark(caucion_tna_pct=inputs.caucion_tna_pct, days=30)
                cc = CoveredCallScanner(CoveredCallConfig(), expiry_cal).scan(chain, benchmark)
                sp = ShortPutScanner(ShortPutConfig(), expiry_cal).scan(chain, benchmark)
                inputs.options_candidates = list(cc) + list(sp)
        except Exception as exc:
            inputs.fetch_error = str(exc)
            logger.warning("fetch live del runner fallo: %s", exc)
        return inputs

    return _fetch


def build_live_fetch() -> Callable[[], CycleInputs]:
    """Arma el `fetch` live reusando provider + scanners + historiales del dashboard.

    Best-effort y guardado: si una parte falla, devuelve lo que pudo (el ciclo
    igual aprende y persiste status). Las acciones usan los helpers de historial del
    dashboard; las opciones, los scanners de renta.
    """
    return _build_live_fetch_runtime()

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
    parser.add_argument("--lab-infinite", dest="lab_infinite", action="store_true", default=None,
                        help="capital demo virtual enorme, sin breaker ni poda")
    parser.add_argument("--shadow", dest="lab_infinite", action="store_false",
                        help="modo shadow con breaker y limites realistas")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fetch = build_live_fetch()
    if args.once:
        status = run_cycle(
            fetch,
            now=datetime.now(_BA),
            market_open=_market_open(datetime.now(_BA)),
            lab_infinite=args.lab_infinite,
        )
        print(json.dumps(asdict(status), ensure_ascii=False, indent=2))
    else:
        run_forever(
            fetch,
            interval_s=args.interval,
            market_hours=not args.ignore_hours,
            lab_infinite=args.lab_infinite,
        )


if __name__ == "__main__":
    main()
