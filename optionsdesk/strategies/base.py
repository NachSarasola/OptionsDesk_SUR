"""Interfaz base para todas las estrategias de opciones."""
from __future__ import annotations

from abc import ABC, abstractmethod

from optionsdesk.core.benchmark import Benchmark
from optionsdesk.core.rates import RateResult
from optionsdesk.data.providers.base import OptionsChain


class Strategy(ABC):
    """Escanea la cadena y devuelve oportunidades rankeadas.

    Cada estrategia implementa scan() con su lógica de filtrado y
    ranking propio. Las tres familias planeadas:
      v1: CoveredCallScanner, ShortPutScanner
      futuro: ParityArbScanner, VolatilityScanner
    """

    @abstractmethod
    def scan(
        self,
        chain: OptionsChain,
        benchmark: Benchmark,
    ) -> list[RateResult]:
        """Escanea la cadena y devuelve oportunidades ordenadas."""
        ...
