"""Proveedor de datos en tiempo real via pyhomebroker (Bull Market Brokers).

Requiere HB_DNI, HB_USER, HB_PASSWORD, HB_BROKER_ID en .env.
El HB_BROKER_ID de Bull Market está en el README de pyhomebroker:
  https://github.com/ddegese/pyhomebroker

Nota: pyhomebroker es una librería no oficial. Si Bull Market cambia su
plataforma, este módulo puede necesitar actualizarse. Por eso se implementa
detrás de la interfaz MarketDataProvider — para poder reemplazarlo sin
tocar el resto del sistema.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Optional

from optionsdesk.config.settings import settings
from optionsdesk.data.providers.base import MarketDataProvider, OptionsChain, Quote

logger = logging.getLogger(__name__)


class HomeBrokerProvider(MarketDataProvider):
    """Datos en tiempo real de GGAL vía el homebroker de Bull Market."""

    def __init__(self) -> None:
        self._hb = None
        self._lock = threading.Lock()
        self._options_data: dict[str, dict] = {}
        self._spot_data: Optional[dict] = None
        self._repos_data: list[dict] = []
        self._connected = False
        self._last_update: float = 0.0   # epoch seconds del último callback recibido

    def connect(self) -> None:
        try:
            from pyhomebroker import HomeBroker  # type: ignore[import]
        except ImportError as e:
            raise RuntimeError(
                "pyhomebroker no instalado. Ejecutá: pip install pyhomebroker"
            ) from e

        if not settings.is_configured():
            raise RuntimeError(
                "Credenciales de Bull Market no configuradas. "
                "Completá HB_DNI, HB_USER, HB_PASSWORD, HB_BROKER_ID en .env"
            )

        self._hb = HomeBroker(
            settings.hb_broker_id,
            on_options=self._on_options,
            on_securities=self._on_securities,
            on_repos=self._on_repos,
            on_error=self._on_error,
        )
        self._hb.auth.login(
            dni=settings.hb_dni,
            user=settings.hb_user,
            password=settings.hb_password,
            raise_exception=True,
        )
        self._hb.online.connect()
        self._hb.online.subscribe_options()
        self._hb.online.subscribe_securities("general", "48hs")
        self._hb.online.subscribe_repos("1dia")
        self._connected = True
        logger.info("HomeBrokerProvider conectado a Bull Market.")

    def disconnect(self) -> None:
        if self._hb and self._connected:
            try:
                self._hb.online.disconnect()
            except Exception:
                pass
            self._connected = False
            logger.info("HomeBrokerProvider desconectado.")

    def is_connected(self) -> bool:
        return self._connected

    # ── Callbacks (llamados desde el thread de pyhomebroker) ──────────────────

    def _on_options(self, online, quotes) -> None:
        """Recibe un DataFrame de pyhomebroker con las opciones en tiempo real.

        Los nombres de columna pueden variar entre versiones de pyhomebroker.
        Si los datos no aparecen, inspeccioná quotes.columns en modo debug.
        """
        with self._lock:
            try:
                df = quotes.reset_index() if (quotes.index.name or quotes.index.names[0]) else quotes
                for _, row in df.iterrows():
                    sym = str(row.get("symbol", row.get("Symbol", row.get("ticker", "")))).upper().strip()
                    if not sym or not sym.startswith("GFG"):
                        continue
                    self._options_data[sym] = {
                        "bid": float(row.get("bid", row.get("bidPrice", 0)) or 0),
                        "ask": float(row.get("ask", row.get("offerPrice", 0)) or 0),
                        "last": float(row.get("last", row.get("closingPrice", 0)) or 0),
                        "volume": float(row.get("trade_volume", row.get("volume", 0)) or 0),
                        "timestamp": datetime.now(),
                    }
                self._last_update = time.time()
            except Exception as e:
                logger.warning("Error procesando datos de opciones: %s", e)

    def _on_securities(self, online, quotes) -> None:
        """Captura el spot de GGAL desde el panel de acciones."""
        with self._lock:
            try:
                df = quotes.reset_index() if (quotes.index.name or quotes.index.names[0]) else quotes
                for _, row in df.iterrows():
                    sym = str(row.get("symbol", row.get("Symbol", row.get("ticker", "")))).upper()
                    if "GGAL" in sym:
                        self._spot_data = {
                            "bid": float(row.get("bid", row.get("bidPrice", 0)) or 0),
                            "ask": float(row.get("ask", row.get("offerPrice", 0)) or 0),
                            "last": float(row.get("last", row.get("closingPrice", 0)) or 0),
                            "volume": float(row.get("trade_volume", row.get("volume", 0)) or 0),
                            "timestamp": datetime.now(),
                        }
                        self._last_update = time.time()
                        break
            except Exception as e:
                logger.warning("Error procesando datos de acciones: %s", e)

    def _on_repos(self, online, quotes) -> None:
        """Caución / repos para el benchmark de tasa."""
        with self._lock:
            try:
                df = quotes.reset_index() if (quotes.index.name or quotes.index.names[0]) else quotes
                self._repos_data = df.to_dict("records")
            except Exception as e:
                logger.warning("Error procesando datos de caución: %s", e)

    def _on_error(self, online, error, msg) -> None:
        logger.error("HomeBroker error: %s — %s", error, msg)

    # ── Salud de la conexión ──────────────────────────────────────────────────

    def is_stale(self, threshold_s: int = 60) -> bool:
        """True si no se recibió ningún callback de datos en los últimos threshold_s segundos.

        Con interval=15s, umbral=60s significa que el WS lleva al menos 4 ciclos sin
        actualizar — señal clara de desconexión silenciosa.
        """
        if self._last_update == 0.0:
            return False   # nunca recibió datos aún (startup normal)
        return (time.time() - self._last_update) > threshold_s

    def reconnect(self) -> None:
        """Desconecta y reconecta al broker. Silencia excepciones — el caller decide."""
        logger.warning("Reconectando a Bull Market (datos stale)...")
        try:
            self.disconnect()
        except Exception as exc:
            logger.debug("Error al desconectar antes de reconexión: %s", exc)
        try:
            self.connect()
            logger.info("Reconexión exitosa.")
        except Exception as exc:
            logger.error("Reconexión fallida: %s", exc)
            self._connected = False

    # ── Interfaz pública ──────────────────────────────────────────────────────

    def get_spot(self) -> Optional[Quote]:
        with self._lock:
            d = self._spot_data
        if not d:
            return None
        return Quote(
            symbol="GGAL",
            bid=d["bid"], ask=d["ask"], last=d["last"],
            volume=d["volume"], timestamp=d["timestamp"],
        )

    def get_options_chain(self) -> Optional[OptionsChain]:
        spot = self.get_spot()
        if spot is None:
            return None
        with self._lock:
            opts = dict(self._options_data)
        quotes = {
            sym: Quote(
                symbol=sym,
                bid=v["bid"], ask=v["ask"], last=v["last"],
                volume=v["volume"], timestamp=v["timestamp"],
            )
            for sym, v in opts.items()
        }
        return OptionsChain(underlying="GGAL", spot=spot, options=quotes)

    def get_caucion_tna(self, days: int = 30) -> Optional[float]:
        with self._lock:
            repos = list(self._repos_data)
        if not repos:
            return None
        for row in repos:
            for key in ("rate", "tna", "last", "price", "Rate", "TNA"):
                val = row.get(key)
                if val and float(val) > 0:
                    return float(val)
        return None
