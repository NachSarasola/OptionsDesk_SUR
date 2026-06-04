"""Límites de riesgo y validación de oportunidades.

Antes de proponer una operación al usuario (o ejecutarla en modo automático),
el RiskChecker valida que cumpla los límites configurados. En v3.1 se agrega
check_portfolio() para validar límites de portafolio agregado.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from optionsdesk.core.rates import RateResult

if TYPE_CHECKING:
    from optionsdesk.signals.monitor import OpenPosition


_SYMBOL_RE = re.compile(r"^GFG([CV])(\d+(?:[.,]\d+)?)([A-Z]+)$")


def _expiry_code(symbol: str) -> Optional[str]:
    """Extrae el código de vencimiento de un símbolo BYMA. None si no matchea."""
    m = _SYMBOL_RE.match(symbol)
    return m.group(3) if m else None


@dataclass
class RiskLimits:
    # Límites por trade
    max_contracts_per_strike: int = 5
    max_total_contracts: int = 20
    max_capital_pct: float = 80.0
    min_cushion_pct: float = 3.0
    flag_early_assignment_delta: float = 0.85
    max_days_to_expiry: int = 90

    # Límites de portafolio agregado (v3.1)
    max_positions_per_expiry: int = 3          # máx posiciones con mismo vencimiento
    max_net_delta: float = 100.0               # acciones equivalentes (delta neto abs.)
    max_capital_concentration_pct: float = 60.0  # % del capital en un solo vencimiento


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

    def check_portfolio(
        self,
        new_result: RateResult,
        current_positions: list["OpenPosition"],
        current_delta_net: float = 0.0,
        total_capital: Optional[float] = None,
    ) -> tuple[bool, list[str]]:
        """Valida si agregar `new_result` viola los límites de portafolio.

        Args:
            new_result: oportunidad candidata.
            current_positions: posiciones actualmente abiertas.
            current_delta_net: delta neto actual del portafolio (acciones equiv.).
            total_capital: capital total del usuario (para chequeo de concentración).

        Returns:
            (aprobado, lista_de_advertencias). aprobado=False bloquea la recomendación.
        """
        warnings: list[str] = []
        approved = True

        # Extraer código de vencimiento del candidato
        m = _SYMBOL_RE.match(new_result.symbol)
        new_expiry = m.group(3) if m else None

        # 1. Concentración por vencimiento
        if new_expiry:
            same_expiry = sum(
                1 for p in current_positions
                if _expiry_code(p.symbol) == new_expiry
            )
            if same_expiry >= self._limits.max_positions_per_expiry:
                warnings.append(
                    f"Ya hay {same_expiry} posiciones en {new_expiry} "
                    f"(máx {self._limits.max_positions_per_expiry}). "
                    "Diversificá el vencimiento."
                )
                approved = False

        # 2. Delta neto: si agregar el trade supera el límite
        if new_result.delta is not None:
            trade_delta = abs(new_result.delta) * 100.0
            new_delta_net = abs(current_delta_net) + trade_delta
            if new_delta_net > self._limits.max_net_delta:
                warnings.append(
                    f"Delta neto del portafolio llegaría a {new_delta_net:.0f} acciones "
                    f"(límite {self._limits.max_net_delta:.0f}). "
                    "El portafolio estaría muy expuesto a movimiento del spot."
                )
                # Warning, no gate duro — el usuario decide

        # 3. Concentración de capital
        if total_capital and total_capital > 0 and new_expiry:
            capital_in_expiry = sum(
                p.net_outlay * 100.0
                for p in current_positions
                if _expiry_code(p.symbol) == new_expiry
            )
            capital_pct = (capital_in_expiry + new_result.net_outlay * 100.0) / total_capital * 100.0
            if capital_pct > self._limits.max_capital_concentration_pct:
                warnings.append(
                    f"El {capital_pct:.0f}% del capital estaría en {new_expiry} "
                    f"(límite {self._limits.max_capital_concentration_pct:.0f}%). "
                    "Considerá diversificar el vencimiento."
                )

        return approved, warnings
