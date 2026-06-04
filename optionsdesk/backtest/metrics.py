"""Métricas de performance sobre curvas de capital y listas de trades.

Funciones puras — no dependen del simulador ni del dashboard. Se pueden
usar directamente para evaluar cualquier estrategia.
"""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd


def sharpe_ratio(
    equity_curve: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 252.0,
) -> Optional[float]:
    """Sharpe anualizado sobre una curva de capital diaria.

    risk_free_rate: tasa libre de riesgo anualizada (fracción, no %).
    Retorna None si la serie tiene menos de 2 valores o desvío = 0.
    """
    if len(equity_curve) < 2:
        return None
    returns = equity_curve.pct_change().dropna()
    if returns.std() == 0:
        return None
    rf_daily = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - rf_daily
    return float(excess.mean() / excess.std() * math.sqrt(periods_per_year))


def sortino_ratio(
    equity_curve: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 252.0,
) -> Optional[float]:
    """Sortino anualizado (penaliza solo los retornos negativos)."""
    if len(equity_curve) < 2:
        return None
    returns = equity_curve.pct_change().dropna()
    rf_daily = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - rf_daily
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return None
    return float(excess.mean() / downside.std() * math.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series) -> tuple[float, int]:
    """Máximo drawdown (%) y duración en días.

    Retorna (0.0, 0) si la serie es vacía.
    """
    if equity_curve.empty:
        return 0.0, 0
    peak = equity_curve.cummax()
    # Evitar div/0 si equity empieza en 0 (capital_initial=0 en tests)
    dd = (equity_curve - peak) / peak.where(peak != 0, float("nan"))
    max_dd = float(dd.min())  # negativo
    if max_dd == 0.0:
        return 0.0, 0

    # Usar posiciones enteras (argmin/argmax) — agnóstico al tipo de índice.
    # idxmin/idxmax devuelven el label (date, Timestamp…); int() falla sobre dates.
    trough_pos = int(dd.argmin())
    peak_before = peak.iloc[:trough_pos]
    peak_pos = int(peak_before.argmax()) if not peak_before.empty else 0
    duration = trough_pos - peak_pos

    return abs(max_dd) * 100.0, int(duration)


def cagr(equity_curve: pd.Series, periods_per_year: float = 252.0) -> Optional[float]:
    """CAGR anualizado (%). Retorna None si < 2 puntos o capital inicial 0."""
    if len(equity_curve) < 2 or equity_curve.iloc[0] <= 0:
        return None
    n_years = (len(equity_curve) - 1) / periods_per_year
    if n_years <= 0:
        return None
    ratio = equity_curve.iloc[-1] / equity_curve.iloc[0]
    return (ratio ** (1 / n_years) - 1) * 100.0


def profit_factor(pnl_series: list[float]) -> Optional[float]:
    """Profit factor = Σ_ganancias / |Σ_pérdidas|.

    Retorna None si no hay pérdidas (no hay forma de comparar).
    """
    gains  = sum(x for x in pnl_series if x > 0)
    losses = sum(abs(x) for x in pnl_series if x < 0)
    if losses == 0:
        return None
    return gains / losses


def win_rate(pnl_series: list[float]) -> float:
    """Fracción de trades con PnL > 0. 0.0 si la lista está vacía."""
    if not pnl_series:
        return 0.0
    winners = sum(1 for x in pnl_series if x > 0)
    return winners / len(pnl_series)


def expectancy_ars(pnl_series: list[float]) -> float:
    """Ganancia promedio por trade en ARS. 0.0 si la lista está vacía."""
    if not pnl_series:
        return 0.0
    return sum(pnl_series) / len(pnl_series)


def compute_metrics(
    pnl_series: list[float],
    equity_curve: pd.Series,
    caucion_tna_pct: float = 0.0,
    avg_holding_days: Optional[float] = None,
) -> dict:
    """Calcula todas las métricas sobre una simulación.

    Args:
        pnl_series: PnL en ARS por trade (positivo = ganancia).
        equity_curve: capital día a día (pd.Series con DatetimeIndex).
        caucion_tna_pct: tasa libre de riesgo anualizada (para Sharpe).
        avg_holding_days: días promedio de duración de los trades (opcional).

    Retorna un dict con todas las métricas calculadas.
    """
    rf = caucion_tna_pct / 100.0
    dd_pct, dd_dur = max_drawdown(equity_curve)

    return {
        "n_trades": len(pnl_series),
        "win_rate_pct": round(win_rate(pnl_series) * 100.0, 1),
        "profit_factor": round(profit_factor(pnl_series) or 0.0, 2),
        "expectancy_ars": round(expectancy_ars(pnl_series), 0),
        "sharpe": round(sharpe_ratio(equity_curve, rf) or 0.0, 2),
        "sortino": round(sortino_ratio(equity_curve, rf) or 0.0, 2),
        "cagr_pct": round(cagr(equity_curve) or 0.0, 1),
        "max_drawdown_pct": round(dd_pct, 1),
        "max_drawdown_days": dd_dur,
        "avg_holding_days": round(avg_holding_days or 0.0, 1),
        "total_pnl_ars": round(sum(pnl_series), 0),
    }
