from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env_pct(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


@dataclass
class CostModel:
    """Comisiones del ALyC y derechos de mercado BYMA.

    Los defaults son deliberadamente conservadores. Antes de habilitar tickets
    hay que cargar un tarifario efectivo o un limite superior documentado del
    ALyC y COSTS_VERIFIED=true.
    """

    # Comision del ALyC sobre el monto operado (%).
    stock_commission_pct: float = field(
        default_factory=lambda: _env_pct("STOCK_COMMISSION_PCT", 1.00)
    )
    option_commission_pct: float = field(
        default_factory=lambda: _env_pct("OPTION_COMMISSION_PCT", 1.50)
    )

    # Derechos de mercado BYMA (%).
    stock_market_fee_pct: float = field(
        default_factory=lambda: _env_pct("STOCK_MARKET_FEE_PCT", 0.080)
    )
    option_market_fee_pct: float = field(
        default_factory=lambda: _env_pct("OPTION_MARKET_FEE_PCT", 0.200)
    )

    # Comision del ALyC y derecho BYMA por ejercicio / asignacion (%).
    exercise_fee_pct: float = field(
        default_factory=lambda: _env_pct("EXERCISE_COMMISSION_PCT", 1.50)
    )
    exercise_market_fee_pct: float = field(
        default_factory=lambda: _env_pct("EXERCISE_MARKET_FEE_PCT", 0.080)
    )

    iva_rate: float = field(
        default_factory=lambda: _env_pct("IVA_RATE", 0.21)
    )

    def gross_cost(self, amount: float, side: str) -> float:
        """Costo total de una operacion: comision + derechos, ambos con IVA."""
        if side in ("stock_buy", "stock_sell"):
            commission_pct = self.stock_commission_pct
            market_fee_pct = self.stock_market_fee_pct
        elif side in ("option_buy", "option_sell"):
            commission_pct = self.option_commission_pct
            market_fee_pct = self.option_market_fee_pct
        else:
            raise ValueError(f"Unknown side: {side}")
        return amount * (commission_pct + market_fee_pct) / 100 * (1 + self.iva_rate)

    def exercise_cost(self, strike: float) -> float:
        return (
            strike
            * (self.exercise_fee_pct + self.exercise_market_fee_pct)
            / 100
            * (1 + self.iva_rate)
        )


@dataclass(frozen=True)
class CostProfile:
    """Audit label for the effective broker tariff used by calculations."""

    name: str
    verified: bool
    source: str
    model: CostModel


DEFAULT_COSTS = CostModel()
