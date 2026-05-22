"""Screener: filtra y rankea oportunidades de múltiples estrategias."""
from __future__ import annotations

from dataclasses import dataclass

from optionsdesk.core.benchmark import Benchmark
from optionsdesk.core.rates import RateResult


@dataclass
class ScreenerConfig:
    min_tna_spread_pct: float = 5.0   # spread mínimo TNA vs caución (p.p.)
    min_cushion_pct: float = 0.0      # colchón mínimo
    min_tna_pct: float = 0.0          # TNA absoluta mínima
    require_itm: bool = False          # mostrar solo ITM
    top_n: int = 20                    # máximo de oportunidades por estrategia


class Screener:
    """Filtra y rankea oportunidades de las estrategias de renta."""

    def __init__(self, config: ScreenerConfig | None = None) -> None:
        self._cfg = config or ScreenerConfig()

    def filter(self, results: list[RateResult]) -> list[RateResult]:
        filtered = [
            r for r in results
            if r.tna_pct >= self._cfg.min_tna_pct
            and r.spread_vs_caucion_pct >= self._cfg.min_tna_spread_pct
            and r.cushion_pct >= self._cfg.min_cushion_pct
            and (not self._cfg.require_itm or r.moneyness == "ITM")
        ]
        return filtered[: self._cfg.top_n]

    def rank(
        self,
        covered_calls: list[RateResult],
        short_puts: list[RateResult],
        benchmark: Benchmark,
    ) -> tuple[list[RateResult], list[RateResult]]:
        return self.filter(covered_calls), self.filter(short_puts)
