"""Benchmark de tasa libre de riesgo local: caución colocadora.

En el mercado argentino, la caución (repo overnight/a plazo) es el benchmark
natural para comparar las tasas de las opciones. El "edge" de un lanzamiento
cubierto es el spread: tasa_lanzamiento - tasa_caución.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Benchmark:
    """Caución colocadora como referencia de tasa libre de riesgo."""

    caucion_tna_pct: float   # tasa nominal anual de la caución (%)
    days: int                # plazo de la caución consultada
    source: str = "homebroker"

    @property
    def caucion_tea_pct(self) -> float:
        """Tasa efectiva anual equivalente."""
        period_rate = self.caucion_tna_pct / 100.0 * self.days / 365.0
        if self.days <= 0:
            return 0.0
        return ((1.0 + period_rate) ** (365.0 / self.days) - 1.0) * 100.0

    @property
    def r_continuous(self) -> float:
        """Tasa continua equivalente para usar en el modelo de pricing."""
        return math.log(1.0 + self.caucion_tna_pct / 100.0)

    def spread_tna(self, rate_tna_pct: float) -> float:
        """Spread de la tasa de opciones sobre la caución (p.p. TNA)."""
        return rate_tna_pct - self.caucion_tna_pct

    def period_rate(self, days: int) -> float:
        """Tasa del período (fracción) para un plazo de `days` días."""
        return self.caucion_tna_pct / 100.0 * days / 365.0


# Benchmark nulo para uso en tests y modo demo sin datos de caución
ZERO_BENCHMARK = Benchmark(caucion_tna_pct=0.0, days=30, source="none")
