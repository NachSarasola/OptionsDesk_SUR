"""Historial del subyacente GGAL vía BYMA Open Data (PyOBD).

Uso:
    history = UnderlyingHistory()
    df = history.daily("GGAL", days=180)
    # df: DataFrame[date(datetime64), open, high, low, close, volume]

Sin credenciales. Cachea en parquet. Fallback sintético si PyOBD falla o no hay red.
"""
from __future__ import annotations

import logging
import math
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from optionsdesk.config.settings import settings

logger = logging.getLogger(__name__)

# BYMA Open Data usa denominación con panel, no solo el ticker.
_BYMA_SYMBOL: dict[str, str] = {
    "GGAL": "GGAL 24HS",
}

_CACHE_STALENESS_DAYS = 4   # Refresca si el último dato tiene más de 4 días hábiles


class UnderlyingHistory:
    """Historial OHLCV del subyacente, con cache local y fallback sintético."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self._dir = Path(cache_dir) if cache_dir else settings.history_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── API pública ───────────────────────────────────────────────────────────

    def daily(self, symbol: str = "GGAL", days: int = 180) -> pd.DataFrame:
        """DataFrame OHLCV diario, últimos `days` días.

        Columnas: date (datetime64[ns]), open, high, low, close, volume.
        Devuelve serie real si PyOBD responde; fallback sintético en caso contrario.
        """
        cache_path = self._dir / f"{symbol}_daily.parquet"
        cached = self._load_cache(cache_path)

        if cached is not None and self._is_fresh(cached):
            df = cached
        else:
            df = self._fetch_daily(symbol, days, stale_fallback=cached)

        if df is None or df.empty:
            logger.warning("history.daily: usando serie sintética para %s.", symbol)
            df = self._synthetic_daily(days)

        # Normaliza y devuelve los últimos `days` registros
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").tail(days).reset_index(drop=True)
        return df

    def weekly(self, symbol: str = "GGAL", weeks: int = 104) -> pd.DataFrame:
        """OHLCV semanal (resampleado desde daily). Ultimas `weeks` semanas."""
        df_daily = self.daily(symbol, days=weeks * 7 + 30)
        if df_daily.empty:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        return self._resample(df_daily, "W-FRI").tail(weeks).reset_index(drop=True)

    def intraday(self, symbol: str = "GGAL") -> pd.DataFrame:
        """Barras de 1 minuto intradía (vacío fuera de horario de mercado)."""
        byma_sym = _BYMA_SYMBOL.get(symbol, symbol)
        try:
            from pyobd import BymaData   # type: ignore[import-not-found]
            df = BymaData().get_intraday_history(byma_sym)
            if df is None or df.empty:
                return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
            df = df.rename(columns=str.lower)
            return df
        except Exception as exc:
            logger.debug("history.intraday: PyOBD no disponible (%s). Devolviendo vacío.", exc)
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    # ── Internals ─────────────────────────────────────────────────────────────

    def _fetch_daily(
        self,
        symbol: str,
        days: int,
        stale_fallback: Optional[pd.DataFrame],
    ) -> Optional[pd.DataFrame]:
        """Descarga desde PyOBD y actualiza el cache. Devuelve None si falla."""
        byma_sym = _BYMA_SYMBOL.get(symbol, symbol)
        from_date = (date.today() - timedelta(days=days + 30)).strftime("%Y-%m-%d")
        to_date   = date.today().strftime("%Y-%m-%d")

        try:
            from pyobd import BymaData   # type: ignore[import-not-found]
            raw = BymaData().get_daily_history(byma_sym, from_date, to_date)
        except Exception as exc:
            logger.warning("history._fetch_daily: PyOBD falló (%s). Usando caché stale.", exc)
            return stale_fallback

        if raw is None or raw.empty:
            logger.warning("history._fetch_daily: respuesta vacía de PyOBD para %s.", symbol)
            return stale_fallback

        df = self._normalize_daily(raw)
        if df.empty:
            return stale_fallback

        # Combina con caché viejo si existe (para no perder historia)
        if stale_fallback is not None and not stale_fallback.empty:
            df = (
                pd.concat([stale_fallback, df], ignore_index=True)
                .drop_duplicates(subset="date", keep="last")
                .sort_values("date")
                .reset_index(drop=True)
            )

        cache_path = self._dir / f"{symbol}_daily.parquet"
        try:
            df.to_parquet(cache_path, index=False)
        except Exception as exc:
            logger.warning("history._fetch_daily: no pudo guardar caché (%s).", exc)

        return df

    @staticmethod
    def _normalize_daily(raw: pd.DataFrame) -> pd.DataFrame:
        """Homogeniza columnas de PyOBD → date, open, high, low, close, volume."""
        df = raw.rename(columns=str.lower)
        # PyOBD puede devolver 'fecha', 'price', etc. según la versión
        col_map = {
            "fecha": "date",
            "apertura": "open",
            "maximo": "high",
            "máximo": "high",
            "minimo": "low",
            "mínimo": "low",
            "cierre": "close",
            "ultimo": "close",
            "último": "close",
            "volumen": "volume",
            "price": "close",
        }
        df = df.rename(columns=col_map)

        for col in ("open", "high", "low", "close"):
            if col not in df.columns and "close" in df.columns:
                df[col] = df["close"]
        if "volume" not in df.columns:
            df["volume"] = 0

        # Asegura columna date
        if "date" not in df.columns:
            date_candidates = [c for c in df.columns if "date" in c.lower() or "fecha" in c.lower()]
            if date_candidates:
                df = df.rename(columns={date_candidates[0]: "date"})
            else:
                df["date"] = pd.to_datetime(df.index)

        df["date"] = pd.to_datetime(df["date"])
        df = df[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)

    @staticmethod
    def _resample(df_daily: pd.DataFrame, rule: str) -> pd.DataFrame:
        """Reagrupa un DataFrame OHLCV diario a la frecuencia indicada (ej. 'W-FRI')."""
        d = df_daily.set_index("date")
        out = (
            d.resample(rule)
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna(subset=["close"])
            .reset_index()
        )
        return out

    @staticmethod
    def _load_cache(path: Path) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception as e:
            logger.debug("read_parquet fallo para %s: %s", path, e)
            return None

    @staticmethod
    def _is_fresh(df: pd.DataFrame) -> bool:
        """El caché está fresco si el último registro tiene menos de _CACHE_STALENESS_DAYS."""
        if "date" not in df.columns or df.empty:
            return False
        last_date = pd.to_datetime(df["date"]).max().date()
        return (date.today() - last_date).days <= _CACHE_STALENESS_DAYS

    @staticmethod
    def _synthetic_daily(days: int = 180) -> pd.DataFrame:
        """Serie GBM determinista (~GGAL ARS, parámetros históricos aproximados)."""
        rng = random.Random(42)
        mu    = 0.0008     # drift diario ~20% anual
        sigma = 0.025      # vol diaria ~40% anual
        s0    = 8_500.0
        prices: list[float] = [s0]
        for _ in range(days - 1):
            z = rng.gauss(0, 1)
            prices.append(prices[-1] * math.exp((mu - 0.5 * sigma**2) + sigma * z))

        base = date.today() - timedelta(days=days - 1)
        dates = pd.date_range(start=base, periods=days, freq="D")
        return pd.DataFrame({
            "date":   dates,
            "open":   [p * (1 - abs(rng.gauss(0, 0.003))) for p in prices],
            "high":   [p * (1 + abs(rng.gauss(0, 0.008))) for p in prices],
            "low":    [p * (1 - abs(rng.gauss(0, 0.008))) for p in prices],
            "close":  prices,
            "volume": [int(abs(rng.gauss(1_000_000, 300_000))) for _ in prices],
        })
