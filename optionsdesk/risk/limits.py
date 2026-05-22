"""Límites de riesgo y validación de oportunidades.

Antes de proponer una operación al usuario (o ejecutarla en modo automático),
el RiskChecker valida que cumpla los límites configurados.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from optionsdesk.core.rates import RateResult


@dataclass
class RiskLimits:
    max_contracts_per_strike: int = 5        # lotes máx por strike
    max_total_contracts: int = 20            # lotes máx simultáneos
    max_capital_pct: float = 80.0            # % máx del capital comprometido
    min_cushion_pct: float = 3.0             # colchón mínimo para aprobar
    flag_early_assignment_delta: float = 0.85  # delta que dispara aviso de ejercicio
    max_days_to_expiry: int = 90


class RiskChecker:
    """Valida oportunidades contra los límites de riesgo."""

    def __init__(self, limits: Optional[RiskLimits] = None) -> None:
        self._limits = limits or RiskLimits()

    def check_opportunity(self, result: RateResult) -> tuple[bool, list[str]]:
        """Devuelve (aprobado, lista_de_advertencias)."""
        warnings: list[str] = []

        if result.cushion_pct < self._limits.min_cushion_pct:
            warnings.append(
                f"Colchón {result.cushion_pct:.1f}% < mínimo {self._limits.min_cushion_pct:.1f}%"
            )

        if result.days > self._limits.max_days_to_expiry:
            warnings.append(
                f"Vencimiento en {result.days}d supera el máximo de {self._limits.max_days_to_expiry}d"
            )

        if result.delta and abs(result.delta) >= self._limits.flag_early_assignment_delta:
            warnings.append(
                f"Delta {result.delta:.2f} — riesgo de ejercicio anticipado; "
                "verificar fechas de dividendo de GGAL"
            )

        approved = result.cushion_pct >= self._limits.min_cushion_pct
        return approved, warnings
