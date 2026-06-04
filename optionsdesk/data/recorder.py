"""Recorder: snapshotea la cadena de opciones de GGAL a disco durante el horario
de mercado, construyendo un dataset histórico propio.

No existe data histórica gratuita de opciones de BYMA, por lo que el recorder
es la forma de acumular datos para backtests. Activarlo desde la Fase 1 hace
que los datos se vayan acumulando mientras se desarrolla el resto del sistema.

Formato de almacenamiento:
    data/snapshots/YYYY-MM-DD/HH-MM-SS.parquet

Cada archivo contiene todas las opciones de la cadena en ese momento,
con columnas base: timestamp, symbol, spot, bid, ask, last, volume.
Las columnas nuevas son aditivas y opcionales para backtests mas realistas:
bid_size, ask_size, spot_bid, spot_ask, spot_last, source, latency_ms.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from optionsdesk.config.settings import settings
from optionsdesk.data.providers.base import MarketDataProvider, OptionsChain

logger = logging.getLogger(__name__)


class ChainRecorder:
    """Snapshotea la cadena de GGAL a parquet en intervalos regulares."""

    def __init__(
        self,
        provider: MarketDataProvider,
        out_dir: Optional[Path] = None,
    ) -> None:
        self._provider = provider
        self._out_dir = out_dir or settings.snapshots_dir
        self._interval = settings.recorder_interval_s

    def _save_snapshot(self, chain: OptionsChain) -> None:
        ts = datetime.now(timezone.utc)
        date_str = ts.strftime("%Y-%m-%d")
        time_str = ts.strftime("%H-%M-%S")

        out_path = self._out_dir / date_str
        out_path.mkdir(parents=True, exist_ok=True)
        health = self._provider.get_health()

        rows = [
            {
                "timestamp": ts,
                "symbol": sym,
                "spot": chain.spot.mid,
                "spot_bid": chain.spot.bid,
                "spot_ask": chain.spot.ask,
                "spot_last": chain.spot.last,
                "bid": q.bid,
                "ask": q.ask,
                "last": q.last,
                "volume": q.volume,
                "bid_size": q.bid_size,
                "ask_size": q.ask_size,
                "source": health.source,
                "latency_ms": health.last_latency_ms,
            }
            for sym, q in chain.options.items()
        ]
        if not rows:
            return

        file_path = out_path / f"{time_str}.parquet"
        pd.DataFrame(rows).to_parquet(file_path, index=False)
        logger.debug("Snapshot guardado: %s (%d opciones)", file_path, len(rows))

    def take_snapshot(self) -> bool:
        """Toma un snapshot. Devuelve True si fue exitoso."""
        # Reconexión automática si el WebSocket lleva más de 60s sin datos.
        if hasattr(self._provider, "is_stale") and self._provider.is_stale(threshold_s=60):
            logger.warning("Datos stale detectados — reconectando al broker.")
            self._provider.reconnect()  # type: ignore[attr-defined]

        t0 = time.monotonic()
        chain = self._provider.get_options_chain()
        if chain is None:
            logger.warning("Sin datos de cadena para snapshot.")
            self._write_status(connected=False, lag_s=None)
            return False

        self._save_snapshot(chain)
        self._run_monitor_hook(chain.spot.mid if chain.spot else None)
        self._write_status(connected=True, lag_s=round(time.monotonic() - t0, 2))
        return True

    def _write_status(self, connected: bool, lag_s: Optional[float]) -> None:
        """Escribe data/recorder_status.json para que el dashboard muestre el estado."""
        status_path = Path("data/recorder_status.json")
        try:
            status_path.parent.mkdir(parents=True, exist_ok=True)
            status_path.write_text(
                json.dumps({
                    "last_snapshot_ts": datetime.now(timezone.utc).isoformat(),
                    "lag_s": lag_s,
                    "connected": connected,
                }),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.debug("No se pudo escribir recorder_status.json: %s", exc)

    def _run_monitor_hook(self, current_spot: Optional[float]) -> None:
        """Invoca el monitor de posiciones abiertas si está habilitado."""
        if not settings.horizon_monitor_enabled:
            return
        try:
            from optionsdesk.signals.monitor import PositionMonitor
            PositionMonitor(positions_file=settings.open_positions_file).run_once(current_spot)
        except Exception as exc:
            logger.warning("Monitor de posiciones falló: %s", exc)

    def run(self) -> None:
        """Loop bloqueante: snapshot cada `interval` segundos.

        Correr en un proceso o thread separado:
            import threading
            t = threading.Thread(target=recorder.run, daemon=True)
            t.start()
        """
        logger.info("Recorder iniciado (intervalo: %ds).", self._interval)
        while True:
            self.take_snapshot()
            time.sleep(self._interval)

    @classmethod
    def load_day(cls, date_str: str) -> Optional[pd.DataFrame]:
        """Carga todos los snapshots de un día (YYYY-MM-DD) en un DataFrame."""
        day_dir = settings.snapshots_dir / date_str
        if not day_dir.exists():
            return None
        files = sorted(day_dir.glob("*.parquet"))
        if not files:
            return None
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _build_provider() -> MarketDataProvider:
    """Construye una fuente real para no grabar datos sinteticos por accidente."""
    if settings.is_iol_configured():
        from optionsdesk.data.providers.iol import IOLProvider
        provider: MarketDataProvider = IOLProvider()
    elif settings.is_configured():
        from optionsdesk.data.providers.homebroker import HomeBrokerProvider
        provider = HomeBrokerProvider()
    else:
        raise RuntimeError("Configura credenciales IOL o HomeBroker antes de iniciar el recorder.")
    provider.connect()
    return provider


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    provider = _build_provider()
    try:
        ChainRecorder(provider).run()
    except KeyboardInterrupt:
        logger.info("Recorder detenido.")
    finally:
        provider.disconnect()


if __name__ == "__main__":
    main()
