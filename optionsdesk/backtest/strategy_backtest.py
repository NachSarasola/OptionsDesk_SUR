"""Backtester walk-forward de señales swing — evalua un genoma POR REGIMEN.

Replaya `scan_stock_symbol` sobre el historico diario: para cada rueda pasada
genera la señal que HABRIA disparado ese dia (con los parametros del genoma via
param_override), resuelve el trade contra las ruedas siguientes (stop vs target,
neto de costos) y etiqueta el resultado con el REGIMEN de mercado de ese dia.

Asi, un mismo genoma acumula P&L separado por contexto (alcista/vol-alta, etc.):
el optimizador puede descubrir que una estrategia pierde en lateral pero gana en
tendencia. La pregunta no es "¿este genoma sirve?" sino "¿en que contexto sirve?".

Modelo (orientado a RANKING de estrategias, no a P&L absoluto exacto):
  - Entrada: al `entry_price` de la señal, el dia de la señal.
  - Sizing: riesgo fijo por trade (cada trade arriesga `risk_budget` ARS) →
    P&L en pesos comparable entre trades sin que el sizing sesgue el ranking.
  - Salida: primera rueda siguiente donde low≤stop (pierde) o high≥target (gana);
    si ambos el mismo dia, gana el stop (conservador). Si no, time-stop al cierre
    del ultimo dia permitido. Todo neto del ida-y-vuelta de comisiones.
Como todos los genomas se evaluan con el mismo modelo, los sesgos se cancelan en
la comparacion relativa, que es lo que la busqueda necesita.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

from optionsdesk.config.costs import CostModel, DEFAULT_COSTS
from optionsdesk.backtest.param_store import (
    TUNABLES, default_flags, default_params, param_override,
)
from optionsdesk.backtest.strategy_context import (
    DEFAULT_AXES, UNKNOWN_REGIME, classify_regime,
)

logger = logging.getLogger(__name__)

_WARMUP_BARS = 60          # ruedas minimas antes de empezar a operar
_DEFAULT_HOLD = 12         # cap de ruedas si la señal no define max_holding_days
_RISK_BUDGET_ARS = 10_000.0   # riesgo fijo por trade (normaliza el P&L)
_DD_PENALTY_W = 0.6        # peso del max drawdown en la fitness (baja el trade-off)
_MIN_SAMPLE = 8            # trades por regimen para confianza plena


# ── Genoma ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StrategyGenome:
    """Una estrategia completa: valores de parametros + que setups estan activos."""
    params: dict
    flags: dict

    @staticmethod
    def default() -> "StrategyGenome":
        return StrategyGenome(params=default_params(), flags=default_flags())

    def key(self) -> tuple:
        """Identidad hashable para deduplicar genomas en el arbol."""
        p = tuple(sorted((k, round(float(v), 5)) for k, v in self.params.items()))
        f = tuple(sorted((k, bool(v)) for k, v in self.flags.items()))
        return (p, f)


# ── Resultado ──────────────────────────────────────────────────────────────────

@dataclass
class RegimeResult:
    regime: str
    n: int
    total_pnl_ars: float
    expectancy_ars: float
    win_rate: float
    profit_factor: Optional[float]
    max_drawdown_ars: float
    fitness: float
    confidence: float   # 0..1 segun muestra


@dataclass
class GenomeResult:
    genome: StrategyGenome
    by_regime: dict           # regime -> RegimeResult
    overall: RegimeResult
    n_signals: int

    def regime_fitness(self, regime: str) -> float:
        r = self.by_regime.get(regime)
        return r.fitness if r is not None else 0.0


# ── Quote sintetico desde una barra diaria ────────────────────────────────────

def _synthetic_quote(symbol: str, bar: pd.Series, when: datetime):
    from optionsdesk.data.providers.base import Quote
    close = float(bar["close"])
    vol = float(bar.get("volume", 0) or 0)
    # Spread sintetico chico (~0.2%) — la liquidez real se gatea por volumen.
    return Quote(
        symbol=symbol,
        bid=round(close * 0.999, 4),
        ask=round(close * 1.001, 4),
        last=close,
        volume=vol,
        timestamp=when,
        bid_size=1000.0,
        ask_size=1000.0,
        source="backtest",
    )


# ── Resolucion de un trade contra las ruedas futuras ──────────────────────────

def _resolve_trade(
    future: pd.DataFrame,
    entry: float,
    stop: float,
    target: float,
    max_hold: int,
    costs: CostModel,
) -> Optional[float]:
    """P&L en ARS de un trade long, riesgo fijo. None si entrada invalida."""
    risk_per_share = entry - stop
    if risk_per_share <= 0 or entry <= 0:
        return None
    qty = max(_RISK_BUDGET_ARS / risk_per_share, 0.0)
    if qty <= 0:
        return None

    horizon = future.head(max_hold if max_hold > 0 else _DEFAULT_HOLD)
    exit_price = None
    for _, b in horizon.iterrows():
        lo, hi = float(b["low"]), float(b["high"])
        if lo <= stop:               # stop primero (conservador) si ambos el mismo dia
            exit_price = stop
            break
        if hi >= target:
            exit_price = target
            break
    if exit_price is None:
        # Time-stop: cierre de la ultima rueda permitida.
        if horizon.empty:
            return None
        exit_price = float(horizon.iloc[-1]["close"])

    gross = (exit_price - entry) * qty
    fees = (
        costs.gross_cost(entry * qty, "stock_buy")
        + costs.gross_cost(exit_price * qty, "stock_sell")
    )
    return round(gross - fees, 2)


# ── Metricas de un conjunto de P&L ────────────────────────────────────────────

def _max_drawdown_ars(pnls: list[float]) -> float:
    equity = peak = dd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        dd = min(dd, equity - peak)
    return abs(dd)


def _regime_result(regime: str, pnls: list[float]) -> RegimeResult:
    from optionsdesk.backtest.metrics import expectancy_ars, profit_factor, win_rate
    n = len(pnls)
    if n == 0:
        return RegimeResult(regime, 0, 0.0, 0.0, 0.0, None, 0.0, 0.0, 0.0)
    total = round(sum(pnls), 2)
    exp = round(expectancy_ars(pnls), 2)
    wr = round(win_rate(pnls), 3)
    pf = profit_factor(pnls)
    dd = _max_drawdown_ars(pnls)
    confidence = min(1.0, n / _MIN_SAMPLE)
    # Fitness risk-adjusted: ganancia escalada por confianza, penalizada por drawdown.
    # Penalizar el drawdown baja el trade-off (el optimo no es un genoma explosivo).
    fit = round(total * confidence - dd * _DD_PENALTY_W, 2)
    return RegimeResult(
        regime=regime, n=n, total_pnl_ars=total, expectancy_ars=exp,
        win_rate=wr, profit_factor=round(pf, 3) if pf is not None else None,
        max_drawdown_ars=round(dd, 2), fitness=fit, confidence=round(confidence, 3),
    )


# ── Evaluacion de un genoma ────────────────────────────────────────────────────

def evaluate_genome(
    genome: StrategyGenome,
    histories: dict,
    *,
    weeklies: Optional[dict] = None,
    costs: CostModel = DEFAULT_COSTS,
    warmup: int = _WARMUP_BARS,
    axes=DEFAULT_AXES,
    max_signals_per_day: int = 1,
    step: int = 1,
    max_days: Optional[int] = None,
) -> GenomeResult:
    """Evalua un genoma sobre el historico, acumulando P&L por regimen.

    histories: dict[symbol -> daily DataFrame] (date, open, high, low, close, volume).
    weeklies:  dict[symbol -> weekly DataFrame] opcional (precomputado para velocidad).
    step:      evalua cada `step` ruedas (submuestreo para velocidad; el ranking
               relativo entre genomas se preserva).
    max_days:  limita el walk a las ultimas `max_days` ruedas operables.
    """
    from optionsdesk.signals.stock_signals import scan_stock_symbol

    pnl_by_regime: dict[str, list[float]] = {}
    all_pnls: list[float] = []
    n_signals = 0
    stride = max(1, int(step))

    with param_override(genome.params, genome.flags):
        for symbol, daily in histories.items():
            if daily is None or len(daily) < warmup + 3:
                continue
            daily = daily.reset_index(drop=True)
            weekly_full = (weeklies or {}).get(symbol)
            n = len(daily)
            start = warmup
            if max_days is not None and (n - 1 - warmup) > max_days:
                start = n - 1 - max_days
            for i in range(start, n - 1, stride):
                window = daily.iloc[: i + 1]
                bar = daily.iloc[i]
                when = _bar_dt(bar, i)
                regime = classify_regime(window, axes)
                if regime == UNKNOWN_REGIME:
                    continue

                weekly = _truncate_weekly(weekly_full, when)
                quote = _synthetic_quote(symbol, bar, when)
                try:
                    signals = scan_stock_symbol(
                        symbol, quote, daily=window, weekly=weekly, now=when, costs=costs,
                    )
                except Exception as exc:
                    logger.debug("scan fallo %s@%d: %s", symbol, i, exc)
                    continue
                if not signals:
                    continue

                future = daily.iloc[i + 1:]
                for sig in signals[:max_signals_per_day]:
                    pnl = _resolve_trade(
                        future, sig.entry_price, sig.stop_price, sig.target_price,
                        sig.max_holding_days or _DEFAULT_HOLD, costs,
                    )
                    if pnl is None:
                        continue
                    n_signals += 1
                    pnl_by_regime.setdefault(regime, []).append(pnl)
                    all_pnls.append(pnl)

    by_regime = {r: _regime_result(r, pnls) for r, pnls in pnl_by_regime.items()}
    overall = _regime_result("overall", all_pnls)
    return GenomeResult(genome=genome, by_regime=by_regime, overall=overall, n_signals=n_signals)


def _bar_dt(bar: pd.Series, idx: int) -> datetime:
    d = bar.get("date")
    if isinstance(d, (pd.Timestamp, datetime)):
        return pd.Timestamp(d).to_pydatetime()
    return datetime(2020, 1, 1) + pd.Timedelta(days=idx).to_pytimedelta()


def _truncate_weekly(weekly_full: Optional[pd.DataFrame], when: datetime) -> Optional[pd.DataFrame]:
    if weekly_full is None or weekly_full.empty or "date" not in weekly_full.columns:
        return weekly_full
    try:
        return weekly_full[pd.to_datetime(weekly_full["date"]) <= pd.Timestamp(when)]
    except Exception:
        return weekly_full


def load_histories(
    symbols: list[str],
    *,
    days: int = 240,
    allow_synthetic: bool = True,
) -> tuple[dict, dict]:
    """Carga (daily, weekly) por simbolo reusando UnderlyingHistory. Best-effort."""
    from optionsdesk.data.history import UnderlyingHistory, weekly_from_daily
    hist = UnderlyingHistory()
    dailies: dict = {}
    weeklies: dict = {}
    for sym in symbols:
        try:
            d = hist.daily(sym.upper(), days=days, allow_synthetic=allow_synthetic)
            if d is not None and not d.empty:
                dailies[sym.upper()] = d
                try:
                    weeklies[sym.upper()] = weekly_from_daily(d)
                except Exception:
                    weeklies[sym.upper()] = None
        except Exception as exc:
            logger.debug("history %s fallo: %s", sym, exc)
    return dailies, weeklies
