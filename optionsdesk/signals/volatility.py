"""Métricas de volatilidad: realizada, VRP e IV Rank.

El edge de vender prima es positivo cuando IV > vol realizada (el mercado
paga más por protección de la que estadísticamente debería). Sin historial
de IV el iv_rank cae a "sin datos" — no se inventa un número.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


def realized_volatility(spot_series: list[float], annualize: bool = True) -> Optional[float]:
    """Volatilidad realizada (desvío estándar de retornos log, anualizado con 252).

    spot_series: cierres diarios en orden cronológico.
    Devuelve None si hay menos de 2 precios.
    """
    if len(spot_series) < 2:
        return None
    log_returns = [
        math.log(spot_series[i] / spot_series[i - 1])
        for i in range(1, len(spot_series))
        if spot_series[i - 1] > 0 and spot_series[i] > 0
    ]
    n = len(log_returns)
    if n < 1:
        return None
    mean = sum(log_returns) / n
    variance = sum((r - mean) ** 2 for r in log_returns) / max(n - 1, 1)
    daily_vol = math.sqrt(variance)
    return daily_vol * math.sqrt(252) if annualize else daily_vol


def variance_risk_premium(iv: float, realized_vol: float) -> float:
    """VRP = IV − vol realizada. Positivo → estás cobrando cara la prima (bueno para el vendedor)."""
    return iv - realized_vol


def iv_rank(current_iv: float, iv_history: list[float]) -> Optional[float]:
    """IV Rank 0–100: posición de la IV actual en el rango histórico.

    iv_history: serie de IVs previas (sin incluir current_iv).
    Devuelve None si historial vacío o sin rango (iv_max == iv_min).
    """
    if not iv_history:
        return None
    iv_min = min(iv_history)
    iv_max = max(iv_history)
    if iv_max <= iv_min:
        return None
    return (current_iv - iv_min) / (iv_max - iv_min) * 100.0


@dataclass
class VolEdge:
    """Resumen del edge de volatilidad para una oportunidad concreta."""

    realized_vol: Optional[float]    # anualizada (p.ej. 0.45 = 45%)
    iv: Optional[float]              # IV implícita de la opción
    vrp: Optional[float]             # variance risk premium = iv - realized_vol
    iv_rank_pct: Optional[float]     # 0–100; None si sin historial suficiente
    label: str                       # "positivo" | "neutro" | "negativo" | "sin datos"

    @property
    def has_edge(self) -> bool:
        """True si VRP > 0: la prima está cara respecto a la vol histórica."""
        return self.vrp is not None and self.vrp > 0.0

    @classmethod
    def compute(
        cls,
        iv: Optional[float],
        spot_history: list[float],
        iv_history: Optional[list[float]] = None,
    ) -> "VolEdge":
        """Construye un VolEdge dado IV, historial de spot e historial de IV.

        Si IV es None o no hay suficiente historial de spot, el label es "sin datos".
        """
        if iv is None or iv <= 0:
            return cls(None, None, None, None, "sin datos")

        rv = realized_volatility(spot_history)
        vrp = variance_risk_premium(iv, rv) if rv is not None else None
        ivr = iv_rank(iv, iv_history or [])

        if vrp is None:
            label = "sin datos"
        elif vrp > 0.05:
            label = "positivo"
        elif vrp > 0.0:
            label = "neutro"
        else:
            label = "negativo"

        return cls(
            realized_vol=rv,
            iv=iv,
            vrp=vrp,
            iv_rank_pct=ivr,
            label=label,
        )
