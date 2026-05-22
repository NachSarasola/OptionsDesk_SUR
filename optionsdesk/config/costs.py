from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    """
    Comisiones y derechos de mercado de Bull Market Brokers (BYMA).
    Verificar tasas actuales en https://www.bullmarketbrokers.com/comisiones
    antes de usar en producción.
    """
    # Comisión sobre el monto operado (en %)
    stock_commission_pct: float = 0.30
    option_commission_pct: float = 0.25

    # Derechos de mercado BYMA (en %)
    stock_market_fee_pct: float = 0.060
    option_market_fee_pct: float = 0.060

    # Fee de ejercicio / asignación (en % sobre el strike)
    exercise_fee_pct: float = 0.25

    # IVA sobre comisiones
    iva_rate: float = 0.21

    def gross_cost(self, amount: float, side: str) -> float:
        """Costo total (comisión + IVA + derechos) sobre una operación.

        side: 'stock_buy', 'stock_sell', 'option_buy', 'option_sell'
        """
        if side in ("stock_buy", "stock_sell"):
            commission = amount * self.stock_commission_pct / 100 * (1 + self.iva_rate)
            market_fee = amount * self.stock_market_fee_pct / 100
        elif side in ("option_buy", "option_sell"):
            commission = amount * self.option_commission_pct / 100 * (1 + self.iva_rate)
            market_fee = amount * self.option_market_fee_pct / 100
        else:
            raise ValueError(f"Unknown side: {side}")
        return commission + market_fee

    def exercise_cost(self, strike: float) -> float:
        return strike * self.exercise_fee_pct / 100 * (1 + self.iva_rate)


DEFAULT_COSTS = CostModel()
