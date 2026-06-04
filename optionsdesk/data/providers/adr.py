"""Proveedor de datos ADR (American Depositary Receipt) vía stooq.

Sin API key: endpoint liviano /q/l/ devuelve la barra OHLCV del día actual.
Con STOOQ_APIKEY (env VAR): descarga histórico completo para backfill inicial.

Estrategia de cache incremental:
  Cada llamada a `adr_daily()` agrega la barra del día al parquet local.
  El historial crece con el tiempo. Al inicio la confluencia devuelve "unknown".
  Con ≥10 días de cache la confluencia se vuelve confiable.

Confluencia ADR/CCL:
  precio_local_ARS ≈ precio_ADR_USD × ratio × CCL
  Si GGAL.US no rompe estructura cuando GGAL local sí lo hace → el movimiento
  en pesos es un salto del dólar (CCL), no un quiebre real del activo.
"""
from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

try:
    import requests as _requests
except ImportError:  # pragma: no cover
    _requests = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_BA_TZ       = ZoneInfo("America/Argentina/Buenos_Aires")
_NY_TZ       = ZoneInfo("America/New_York")
_STOOQ_QUOTE = "https://stooq.com/q/l/?s={ticker}&f=sd2t2ohlcv&h&e=csv"
_STOOQ_HIST  = "https://stooq.com/q/d/l/?s={ticker}&i=d&apikey={key}"
_HTTP_TIMEOUT = 10   # segundos

# Umbrales para clasificar confluencia
_CONFIRM_THRESHOLD = 0.008    # ADR momentum > +0.8% → confirma dirección alcista
_FX_THRESHOLD      = -0.006   # ADR momentum < -0.6% diverge → movido por CCL


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class AdrQuote:
    """Cotización del ADR del día actual."""
    ticker: str
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


# ── Cotización actual (sin API key) ──────────────────────────────────────────

def adr_quote(adr_ticker: str, timeout: int = _HTTP_TIMEOUT) -> Optional[AdrQuote]:
    """Obtiene la cotización actual del ADR vía stooq (sin key).

    Endpoint: https://stooq.com/q/l/?s={ticker}&f=sd2t2ohlcv&h&e=csv
    Devuelve None si hay error de red o el ticker no existe.
    """
    try:
        if _requests is None:
            raise ImportError("requests no disponible")
        url  = _STOOQ_QUOTE.format(ticker=adr_ticker.lower())
        resp = _requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return _parse_stooq_quote_csv(adr_ticker, resp.text)
    except Exception as exc:
        logger.debug("adr_quote(%s) fallo: %s", adr_ticker, exc)
        return None


def _parse_stooq_quote_csv(ticker: str, text: str) -> Optional[AdrQuote]:
    """Parsea el CSV de una línea de stooq /q/l/ en AdrQuote."""
    reader = csv.DictReader(io.StringIO(text.strip()))
    for row in reader:
        try:
            return AdrQuote(
                ticker=ticker,
                date=row.get("Date", ""),
                open=float(row.get("Open", 0) or 0),
                high=float(row.get("High", 0) or 0),
                low=float(row.get("Low", 0) or 0),
                close=float(row.get("Close", 0) or 0),
                volume=int(float(row.get("Volume", 0) or 0)),
            )
        except (ValueError, KeyError):
            return None
    return None


# ── Historial con cache incremental ──────────────────────────────────────────

def adr_daily(
    adr_ticker: str,
    days: int = 60,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Retorna historial de cierres diarios del ADR (cache local en parquet).

    Si hay STOOQ_APIKEY y el cache está vacío, intenta un backfill histórico.
    Si no hay key o el backfill falla, el cache crece un día a la vez con la
    cotización actual. Retorna DataFrame vacío si no hay datos.
    """
    from optionsdesk.config.settings import settings as _s
    _cache_dir = cache_dir or _s.history_dir / "adr_cache"
    _cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_dir / f"{adr_ticker.upper().replace('.', '_')}_daily.parquet"

    # Cargar cache existente
    df_cache = _load_cache(cache_path)

    # Backfill con API key si el cache está vacío/muy corto
    if len(df_cache) < 5 and _s.stooq_apikey:
        df_hist = _stooq_history(adr_ticker, _s.stooq_apikey)
        if not df_hist.empty:
            df_cache = df_hist
            _save_cache(df_cache, cache_path)

    # Agregar la barra del día si no existe aún
    today_str = date.today().isoformat()
    if today_str not in (df_cache["date"].astype(str).values if not df_cache.empty else []):
        quote = adr_quote(adr_ticker)
        if quote is not None and quote.close > 0:
            new_row = pd.DataFrame([{
                "date":   pd.to_datetime(quote.date),
                "open":   quote.open,
                "high":   quote.high,
                "low":    quote.low,
                "close":  quote.close,
                "volume": quote.volume,
            }])
            df_cache = pd.concat([df_cache, new_row], ignore_index=True)
            df_cache = df_cache.drop_duplicates("date").sort_values("date").reset_index(drop=True)
            _save_cache(df_cache, cache_path)

    if df_cache.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    return df_cache.tail(days).reset_index(drop=True)


def _load_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    try:
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)
    except Exception as exc:
        logger.debug("_load_cache(%s) fallo: %s", path, exc)
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:
        logger.debug("_save_cache(%s) fallo: %s", path, exc)


def _stooq_history(adr_ticker: str, apikey: str, timeout: int = _HTTP_TIMEOUT) -> pd.DataFrame:
    """Descarga historial completo de stooq con API key."""
    try:
        if _requests is None:
            return pd.DataFrame()
        url  = _STOOQ_HIST.format(ticker=adr_ticker.lower(), key=apikey)
        resp = _requests.get(url, timeout=timeout)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        needed = {"date", "open", "high", "low", "close"}
        if not needed.issubset(df.columns):
            return pd.DataFrame()
        if "volume" not in df.columns:
            df["volume"] = 0
        return df[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)
    except Exception as exc:
        logger.debug("_stooq_history(%s) fallo: %s", adr_ticker, exc)
        return pd.DataFrame()


# ── CCL implícito ─────────────────────────────────────────────────────────────

def implied_ccl(local_ars: float, adr_usd: float, ratio: float) -> float:
    """CCL implícito = precio_local_ARS / (ratio × precio_ADR_USD).

    ratio: acciones locales por ADS (ej. GGAL=10 → 10 acciones GGAL = 1 GGAL.US).
    Retorna 0.0 si falta algún input.
    """
    if adr_usd <= 0 or ratio <= 0 or local_ars <= 0:
        return 0.0
    return round(local_ars / (ratio * adr_usd), 2)


# ── Confluencia ADR/CCL ───────────────────────────────────────────────────────

def adr_confluence(
    local_break_dir: Optional[str],
    adr_df: pd.DataFrame,
    lookback: int = 5,
) -> str:
    """Determina si el quiebre estructural local es confirmado por el ADR.

    Si el ADR también se mueve en la misma dirección que el quiebre local →
    "confirmed" (el activo subyacente movió, no solo el dólar).
    Si el ADR va en dirección contraria → "fx_driven" (el movimiento en pesos
    es un salto del CCL, no del activo).
    Si no hay suficiente historia → "unknown".

    Args:
        local_break_dir: "up" (BOS_UP) | "down" (BOS_DOWN) | None
        adr_df: DataFrame de historial del ADR (columna 'close')
        lookback: número de barras para el momentum del ADR
    """
    if local_break_dir is None:
        return "unknown"
    if adr_df is None or "close" not in adr_df.columns or len(adr_df) < lookback + 1:
        return "unknown"

    closes   = adr_df["close"].astype(float)
    prev_cl  = float(closes.iloc[-(lookback + 1)])
    last_cl  = float(closes.iloc[-1])
    if prev_cl <= 0:
        return "unknown"

    momentum = (last_cl - prev_cl) / prev_cl   # fracción, no porcentaje

    if local_break_dir == "up":
        if momentum >= _CONFIRM_THRESHOLD:      # ADR sube +0.8% → confirma
            return "confirmed"
        if momentum <= _FX_THRESHOLD:           # ADR baja -0.6% → movido por CCL
            return "fx_driven"
        return "unknown"

    if local_break_dir == "down":
        if momentum <= -_CONFIRM_THRESHOLD:     # ADR baja -0.8% → confirma
            return "confirmed"
        if momentum >= -_FX_THRESHOLD:          # ADR sube +0.6% → movido por CCL
            return "fx_driven"
        return "unknown"

    return "unknown"


def bos_to_dir(bos: Optional[str]) -> Optional[str]:
    """Convierte BOS_UP/BOS_DOWN al formato que usa adr_confluence."""
    if bos == "BOS_UP":
        return "up"
    if bos == "BOS_DOWN":
        return "down"
    return None

_bos_to_dir = bos_to_dir   # alias backward-compatible


def enrich_smc_with_adr(
    smc_ctx: object,
    symbol: str,
    cache_dir: Optional[Path] = None,
) -> object:
    """Añade adr_confluence al SmcContext si el toggle está activo.

    Busca el ticker ADR en la tabla de mapeo de settings.
    Safe: nunca lanza excepciones, solo loguea y devuelve el ctx sin cambios.
    """
    try:
        from optionsdesk.config.settings import settings as _s
        if not getattr(_s, "adr_ccl_enabled", False):
            return smc_ctx

        adr_map = _s.adr_map()
        if symbol.upper() not in adr_map:
            return smc_ctx

        adr_ticker, _ratio = adr_map[symbol.upper()]
        adr_df = adr_daily(adr_ticker, days=30, cache_dir=cache_dir)

        bos        = getattr(smc_ctx, "bos", None)
        break_dir  = _bos_to_dir(bos)
        confluence = adr_confluence(break_dir, adr_df)

        smc_ctx.adr_confluence = confluence   # type: ignore[attr-defined]
        logger.debug("ADR confluence %s → %s: %s", symbol, adr_ticker, confluence)
    except Exception as exc:
        logger.debug("enrich_smc_with_adr(%s) fallo: %s", symbol, exc)
    return smc_ctx
