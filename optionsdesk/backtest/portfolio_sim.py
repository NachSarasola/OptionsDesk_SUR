"""Simulador transaccional de estrategias de prima vendida (CC + SP).

Extiende el BacktestEngine existente a simulación de cartera completa:
entrada al bid, salida al ask, slippage configurable, cierre diario
guiado por las reglas de management.py.

Uso rápido:
    from optionsdesk.backtest.portfolio_sim import simulate_strategy, SimResult
    from optionsdesk.signals.recommender import RiskProfile

    result = simulate_strategy(
        snapshots_dir="data/snapshots",
        profile=RiskProfile.EQUILIBRADO,
        start="2026-05-01",
        end="2026-05-23",
        capital_initial=500_000,
        execution_mode="bid_ask",  # "mid" para curva optimista
    )
    from optionsdesk.backtest.metrics import compute_metrics
    m = compute_metrics(result.pnl_series, result.equity_curve, caucion_tna_pct=60.0)
    print(m)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from optionsdesk.backtest.engine import BacktestEngine, _reconstruct_chain
from optionsdesk.core.benchmark import Benchmark
from optionsdesk.core.pricing import crr_price
from optionsdesk.signals.management import evaluate_position, SignalType
from optionsdesk.signals.monitor import OpenPosition
from optionsdesk.signals.recommender import RiskProfile, _passes_gates, _score
from optionsdesk.strategies.covered_call import CoveredCallConfig, CoveredCallScanner
from optionsdesk.strategies.short_put import ShortPutConfig, ShortPutScanner

logger = logging.getLogger(__name__)

_CRR_STEPS = 50
_DEFAULT_SLIPPAGE = 0.005   # 0.5% de slippage por lado
_DEFAULT_CAUCION_TNA = 60.0  # si no hay datos de caución


# ── Dataclasses de output ─────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    """Registro de un trade completado durante la simulación."""
    entry_date: date
    exit_date: date
    symbol: str
    strategy: str
    strike: float
    spot_entry: float
    premium_entry: float       # prima neta recibida (ya con slippage)
    premium_exit: float        # costo de recompra (ya con slippage); 0 si expiró OTM
    contracts: int
    pnl_ars: float             # PnL total en ARS (contratos × 100 × delta_prima)
    exit_reason: str           # TAKE_PROFIT | STOP | ROLL | EXPIRY | FORCED_CLOSE
    days_held: int


@dataclass
class SimResult:
    """Resultado completo de una simulación."""
    trades: list[BacktestTrade]
    equity_curve: pd.Series        # capital total día a día (DatetimeIndex)
    pnl_series: list[float]        # PnL en ARS por trade (para métricas)
    mode: str                      # "mid" | "bid_ask"
    profile: str
    weights: dict[str, float]
    start: date
    end: date
    capital_initial: float
    n_days_simulated: int


# ── Helpers internos ──────────────────────────────────────────────────────────

def _entry_price(q, mode: str, slippage: float) -> float:
    """Precio de entrada al vender una opción.

    mode="bid_ask": vende al bid, penalizado por slippage.
    mode="mid": vende al mid (precio teórico optimista).
    Fallback a last si bid = 0.
    """
    bid = float(q.bid)
    ask = float(q.ask)
    last = float(q.last)

    if mode == "mid":
        mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else last
        return max(mid * (1.0 - slippage), 0.0)
    else:
        price = bid if bid > 0 else (last if last > 0 else ask * 0.9)
        return max(price * (1.0 - slippage), 0.0)


def _exit_price(q, mode: str, slippage: float) -> float:
    """Precio de salida al recomprar una opción (pagar ask + slippage)."""
    bid = float(q.bid)
    ask = float(q.ask)
    last = float(q.last)

    if mode == "mid":
        mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else last
        return max(mid * (1.0 + slippage), 0.0)
    else:
        price = ask if ask > 0 else (last if last > 0 else bid * 1.1)
        return max(price * (1.0 + slippage), 0.0)


def _intrinsic_value(spot: float, strike: float, opt_type: str) -> float:
    """Valor intrínseco al vencimiento."""
    if opt_type == "C":
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


def _rate_result_to_open_position(
    result,
    entry_date: date,
    premium_entry: float,
    caucion_tna: float,
    contracts: int = 1,
) -> OpenPosition:
    """Convierte un RateResult del scanner en un OpenPosition para el monitor."""
    return OpenPosition(
        symbol=result.symbol,
        strategy=result.strategy,
        strike=result.strike,
        spot_entry=result.spot,
        premium_received=premium_entry,
        net_outlay=result.net_outlay,
        iv_entry=result.iv or 0.30,
        days_entry=result.days,
        entry_date=entry_date,
        target_exit_days=max(result.days // 2, 5),
        target_capture_pct=50.0,
        caucion_tna=caucion_tna,
        max_loss_mult=2.0,
        roll_dte=21,
        defend_delta=0.50,
        contracts=contracts,
    )


# ── Simulador principal ───────────────────────────────────────────────────────

def simulate_strategy(
    snapshots_dir: str | Path = "data/snapshots",
    profile: RiskProfile = RiskProfile.EQUILIBRADO,
    start: str | date = "2026-01-01",
    end: str | date = "2026-12-31",
    capital_initial: float = 500_000.0,
    execution_mode: str = "bid_ask",
    slippage: float = _DEFAULT_SLIPPAGE,
    caucion_tna_pct: float = _DEFAULT_CAUCION_TNA,
    weights_override: Optional[dict[str, float]] = None,
) -> SimResult:
    """Simula la estrategia de prima vendida sobre snapshots históricos.

    Args:
        snapshots_dir: directorio raíz de snapshots (data/snapshots/).
        profile: perfil de riesgo del recommender.
        start / end: rango de fechas a simular (inclusivo).
        capital_initial: capital en ARS al inicio de la simulación.
        execution_mode: "bid_ask" (realista) o "mid" (optimista).
        slippage: porcentaje de slippage por lado (default 0.5%).
        caucion_tna_pct: TNA de caución para calcular el benchmark.
        weights_override: pesos custom para el scoring (para GridSearch).

    Returns:
        SimResult con trades, equity_curve y pnl_series.
    """
    from optionsdesk.signals.recommender import _WEIGHTS, _get_weights
    from optionsdesk.core.instruments import load_expiry_calendar

    if isinstance(start, str):
        start = date.fromisoformat(start)
    if isinstance(end, str):
        end = date.fromisoformat(end)

    weights = weights_override or _get_weights(profile)
    snap_dir = Path(snapshots_dir)

    expiry_cal: dict = {}
    try:
        expiry_cal = load_expiry_calendar()
    except Exception as e:
        logger.warning("load_expiry_calendar fallo: %s", e)

    benchmark = Benchmark(caucion_tna_pct=caucion_tna_pct, days=30)
    cc_scanner = CoveredCallScanner(CoveredCallConfig(), expiry_cal)
    sp_scanner = ShortPutScanner(ShortPutConfig(), expiry_cal)

    capital    = capital_initial
    equity_pts: list[tuple[date, float]] = []
    trades:     list[BacktestTrade] = []

    open_pos: Optional[OpenPosition] = None
    open_result = None   # RateResult original para calcular max_loss

    # Iterar día por día
    current = start
    while current <= end:
        day_str = current.strftime("%Y-%m-%d")
        day_dir = snap_dir / day_str

        if not day_dir.exists() or not any(day_dir.glob("*.parquet")):
            equity_pts.append((current, capital))
            current += timedelta(days=1)
            continue

        # Cargar el primer snapshot del día (apertura)
        files = sorted(day_dir.glob("*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in files[:1]], ignore_index=True)
        if df.empty:
            equity_pts.append((current, capital))
            current += timedelta(days=1)
            continue

        first_ts = df["timestamp"].min()
        chain = _reconstruct_chain(df, first_ts)
        if chain is None:
            equity_pts.append((current, capital))
            current += timedelta(days=1)
            continue

        spot = chain.spot.mid

        # ── Gestión de posición abierta ───────────────────────────────────────
        if open_pos is not None:
            days_remaining = max(
                open_pos.days_entry - (current - open_pos.entry_date).days,
                0,
            )

            # Vencimiento alcanzado
            if days_remaining <= 0:
                intrinsic = _intrinsic_value(spot, open_pos.strike, open_pos.opt_type)
                contracts = open_pos.contracts  # fijado en la entrada
                pnl_ars = (open_pos.premium_received - intrinsic) * 100.0 * contracts
                capital += pnl_ars

                trades.append(BacktestTrade(
                    entry_date=open_pos.entry_date,
                    exit_date=current,
                    symbol=open_pos.symbol,
                    strategy=open_pos.strategy,
                    strike=open_pos.strike,
                    spot_entry=open_pos.spot_entry,
                    premium_entry=open_pos.premium_received,
                    premium_exit=intrinsic,
                    contracts=contracts,
                    pnl_ars=pnl_ars,
                    exit_reason="EXPIRY",
                    days_held=(current - open_pos.entry_date).days,
                ))
                open_pos = None
                open_result = None

            else:
                # Management signal
                signal = evaluate_position(
                    open_pos,
                    current_spot=spot,
                    today=current,
                )

                should_close = signal.signal_type in (
                    SignalType.TAKE_PROFIT,
                    SignalType.STOP,
                    SignalType.ROLL,
                )
                if should_close:
                    opt_q = chain.options.get(open_pos.symbol)
                    if opt_q:
                        exit_px = _exit_price(opt_q, execution_mode, slippage)
                    else:
                        # Aproximación CRR si no tenemos el quote
                        T_rem = days_remaining / 365.0
                        r = math.log(1.0 + open_pos.caucion_tna / 100.0)
                        exit_px = crr_price(
                            spot, open_pos.strike, T_rem, r,
                            open_pos.iv_entry, open_pos.opt_type, steps=_CRR_STEPS
                        ) * (1.0 + slippage)

                    contracts = open_pos.contracts  # fijado en la entrada
                    pnl_ars = (open_pos.premium_received - exit_px) * 100.0 * contracts
                    capital += pnl_ars

                    trades.append(BacktestTrade(
                        entry_date=open_pos.entry_date,
                        exit_date=current,
                        symbol=open_pos.symbol,
                        strategy=open_pos.strategy,
                        strike=open_pos.strike,
                        spot_entry=open_pos.spot_entry,
                        premium_entry=open_pos.premium_received,
                        premium_exit=exit_px,
                        contracts=contracts,
                        pnl_ars=pnl_ars,
                        exit_reason=signal.signal_type.value,
                        days_held=(current - open_pos.entry_date).days,
                    ))
                    open_pos = None
                    open_result = None

        # ── Entrada a nuevo trade si no hay posición abierta ──────────────────
        if open_pos is None:
            all_results = (
                cc_scanner.scan(chain, benchmark)
                + sp_scanner.scan(chain, benchmark)
            )

            # Filtrar por gates del perfil
            candidates = [
                r for r in all_results
                if _passes_gates(r, profile)
                and (chain.options.get(r.symbol) and
                     chain.options[r.symbol].volume > 0)  # liquidez mínima
            ]

            if candidates:
                # Scoring con pesos (posiblemente custom para GridSearch)
                from optionsdesk.signals.recommender import _score as _score_fn
                scored = sorted(
                    candidates,
                    key=lambda r: _score_with_weights(r, profile, weights),
                    reverse=True,
                )
                best = scored[0]
                opt_q = chain.options.get(best.symbol)
                if opt_q:
                    entry_px = _entry_price(opt_q, execution_mode, slippage)
                    if entry_px > 0:
                        # Calcular contratos una sola vez en entrada (30% del capital
                        # disponible, mínimo 1). Este mismo número se usa en el cierre.
                        lot_cost = best.net_outlay * 100.0
                        n_contracts = (
                            max(math.floor(capital * 0.30 / lot_cost), 1)
                            if lot_cost > 0 else 1
                        )
                        open_pos = _rate_result_to_open_position(
                            best, current, entry_px, caucion_tna_pct,
                            contracts=n_contracts,
                        )
                        open_result = best
                        logger.debug(
                            "Entrando %s %s strike=%s prem=%.2f [%s]",
                            current, best.symbol, best.strike, entry_px, execution_mode,
                        )

        equity_pts.append((current, max(capital, 0.0)))
        current += timedelta(days=1)

    # Cierre forzado al final del período
    if open_pos is not None:
        contracts = open_pos.contracts  # fijado en la entrada
        trades.append(BacktestTrade(
            entry_date=open_pos.entry_date,
            exit_date=end,
            symbol=open_pos.symbol,
            strategy=open_pos.strategy,
            strike=open_pos.strike,
            spot_entry=open_pos.spot_entry,
            premium_entry=open_pos.premium_received,
            premium_exit=open_pos.premium_received * 0.5,  # asumir 50% captura
            contracts=contracts,
            pnl_ars=open_pos.premium_received * 0.5 * 100.0 * contracts,
            exit_reason="FORCED_CLOSE",
            days_held=(end - open_pos.entry_date).days,
        ))

    if not equity_pts:
        equity_pts = [(start, capital_initial)]

    equity_series = pd.Series(
        {pd.Timestamp(d): v for d, v in equity_pts}
    )

    avg_days = (
        sum(t.days_held for t in trades) / len(trades)
        if trades else 0.0
    )

    return SimResult(
        trades=trades,
        equity_curve=equity_series,
        pnl_series=[t.pnl_ars for t in trades],
        mode=execution_mode,
        profile=profile.value,
        weights=weights,
        start=start,
        end=end,
        capital_initial=capital_initial,
        n_days_simulated=len(equity_pts),
    )


def _score_with_weights(result, profile: RiskProfile, weights: dict[str, float]) -> float:
    """Variante de _score que acepta pesos custom (para GridSearch).

    Replica la lógica de recommender._score() sin tocar el global _WEIGHTS.
    """
    from optionsdesk.signals.recommender import _DELTA_BANDS, _probability
    from optionsdesk.config.events import spans_event
    from optionsdesk.config.settings import settings

    w = weights

    ev_raw = result.expected_value_pct if result.expected_value_pct is not None else 0.0
    ev_f = (min(max(ev_raw, -60.0), 60.0) / 60.0 + 1.0) / 2.0

    vrp = 0.0
    if result.vol_edge is not None and result.vol_edge.vrp is not None:
        vrp = result.vol_edge.vrp
    vol_f = (min(max(vrp, -0.15), 0.15) / 0.15 + 1.0) / 2.0

    cushion_f   = min(max(result.cushion_pct, 0.0), 15.0) / 15.0
    prob_f      = _probability(result)
    liq_f       = 1.0 if result.is_liquid else 0.4

    raw = (
        w.get("ev", 0.40) * ev_f
        + w.get("vol", 0.20) * vol_f
        + w.get("cushion", 0.20) * cushion_f
        + w.get("probability", 0.10) * prob_f
        + w.get("liquidity", 0.10) * liq_f
    )

    if result.delta is not None:
        lo, hi = _DELTA_BANDS.get(profile, (0.25, 0.40))
        if lo <= abs(result.delta) <= hi:
            raw = min(raw + 0.05, 1.0)

    if settings.events_enabled:
        from datetime import date as _date
        ev = spans_event(_date.today(), result.expiration)
        if ev is not None:
            raw = max(raw - 0.08, 0.0)

    return raw * 100.0


def trades_to_dataframe(trades: list[BacktestTrade]) -> pd.DataFrame:
    """Convierte la lista de trades en un DataFrame para guardar/mostrar."""
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([
        {
            "entry_date":     t.entry_date.isoformat(),
            "exit_date":      t.exit_date.isoformat(),
            "symbol":         t.symbol,
            "strategy":       t.strategy,
            "strike":         t.strike,
            "spot_entry":     t.spot_entry,
            "premium_entry":  round(t.premium_entry, 2),
            "premium_exit":   round(t.premium_exit, 2),
            "contracts":      t.contracts,
            "pnl_ars":        round(t.pnl_ars, 0),
            "exit_reason":    t.exit_reason,
            "days_held":      t.days_held,
        }
        for t in trades
    ])
