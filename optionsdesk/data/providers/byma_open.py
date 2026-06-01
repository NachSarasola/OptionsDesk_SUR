"""Proveedor fallback via BYMA Open Data (datos demorados ~15 min, gratuito).

Útil cuando el homebroker no está disponible, o para datos de cierre/EOD.
Nota: la cobertura de opciones en BYMA Open Data es limitada; puede no
incluir todos los strikes disponibles en el homebroker.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import requests

from optionsdesk.config.settings import settings
from optionsdesk.data.providers.base import MarketDataProvider, OptionsChain, Quote

logger = logging.getLogger(__name__)

_BASE = "https://open.bymadata.com.ar/vanoms-be-core/rest/api/bymadata/free"


class BymaOpenProvider(MarketDataProvider):
    """Datos demorados de BYMA Open Data — usar como fallback."""

    def __init__(self, timeout_s: int = 10) -> None:
        self._timeout = timeout_s
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "optionsdesk/0.1"})
        # BYMA Open Data usa un cert intermedio que Python/Windows no resuelve por defecto
        self._session.verify = False
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def _get(self, endpoint: str, params: Optional[dict] = None) -> Optional[object]:
        try:
            r = self._session.get(
                f"{_BASE}/{endpoint}", params=params, timeout=self._timeout
            )
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            logger.warning("BYMA Open Data falló: %s", e)
            return None

    def get_spot(self) -> Optional[Quote]:
        data = self._get("equities", params={"symbol": "GGAL"})
        if not data:
            return None
        try:
            item = data[0] if isinstance(data, list) else data
            return Quote(
                symbol="GGAL",
                bid=float(item.get("bidPrice", 0) or 0),
                ask=float(item.get("offerPrice", 0) or 0),
                last=float(item.get("closingPrice", item.get("settlementPrice", 0)) or 0),
                volume=float(item.get("nominalVolume", 0) or 0),
                timestamp=datetime.now(),
            )
        except (KeyError, IndexError, TypeError) as e:
            logger.warning("No se pudo parsear el spot de BYMA: %s", e)
            return None

    def get_options_chain(self) -> Optional[OptionsChain]:
        spot = self.get_spot()
        if not spot:
            return None
        data = self._get("options", params={"underlying": "GGAL"})
        opts: dict[str, Quote] = {}
        if data:
            items = data if isinstance(data, list) else data.get("data", [])
            for item in items:
                sym = str(item.get("symbol", item.get("Symbol", ""))).upper().strip()
                if not sym:
                    continue
                opts[sym] = Quote(
                    symbol=sym,
                    bid=float(item.get("bidPrice", 0) or 0),
                    ask=float(item.get("offerPrice", 0) or 0),
                    last=float(item.get("closingPrice", item.get("settlementPrice", 0)) or 0),
                    volume=float(item.get("nominalVolume", 0) or 0),
                    timestamp=datetime.now(),
                )
        return OptionsChain(underlying="GGAL", spot=spot, options=opts)

    def get_caucion_tna(self, days: int = 30) -> Optional[float]:
        data = self._get("repos")
        if not data:
            return settings.default_caucion_tna
        items = data if isinstance(data, list) else data.get("data", [])
        for item in items:
            rate = item.get("rate", item.get("tna"))
            if rate:
                return float(rate)
        return settings.default_caucion_tna

    def is_connected(self) -> bool:
        return True
