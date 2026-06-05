"""Market Intelligence externo: consenso de analistas (Yahoo Finance) + buzz social (StockTwits).

Sin API keys — solo requests públicos, mismo patrón de robustez que adr.py:
cache local, fallo silencioso, devuelve None/datos parciales sin romper.

El intel es confluencia SECUNDARIA: la SMC manda, el intel confirma.
No genera señales por sí solo — evita chase de hype.

Cobertura: solo nombres con ADR (~9 del adr_map). StockTwits usa el ticker ADR (ej. GGAL).

Riesgo Yahoo: a veces pide crumb/cookie. Degradamos a None y seguimos — el intel
es opcional, no bloquea nada.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

try:
    import requests as _requests
except ImportError:  # pragma: no cover
    _requests = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT  = 8
_YAHOO_URL     = "https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=financialData,recommendationTrend"
_STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
_CACHE_TTL_ANALYST_H = 24   # analistas: refrescar cada 24h (cambian poco)
_CACHE_TTL_SOCIAL_H  = 1    # social: refrescar cada hora (menciones son intradía)


@dataclass
class MarketIntel:
    """Consensus de analistas + buzz social para un ticker ADR."""
    ticker: str                            # ticker ADR (ej. "GGAL")
    # Analistas
    analyst_rating: Optional[float]        # 1=strong buy .. 5=strong sell (Yahoo mean)
    analyst_label: Optional[str]           # "Compra fuerte" .. "Venta fuerte"
    target_upside_pct: Optional[float]     # (target - precio) / precio × 100
    n_analysts: Optional[int]
    # Social (StockTwits)
    social_msgs_24h: Optional[int]         # mensajes últimas 24h (proxy de interés)
    social_bull_pct: Optional[float]       # % bullish de los que se declararon
    social_bear_pct: Optional[float]
    # Meta
    as_of: Optional[date] = None

    def score_delta(self) -> float:
        """Boost/penalización para el score SMC basado en el intel.

        Consenso BUY fuerte + upside alto + social alcista → boost (cap +10).
        Consenso SELL / social muy bajista → penalización (cap −8).
        Nunca bloquea por sí solo — evita operar solo por hype.
        """
        delta = 0.0
        # Analistas (rating 1=BUY..5=SELL, invertir para que bajo=bueno)
        if self.analyst_rating is not None:
            r = self.analyst_rating
            if r <= 2.0:
                delta += 5.0   # consenso compra
            elif r >= 4.0:
                delta -= 5.0   # consenso venta
        # Upside de precio objetivo
        if self.target_upside_pct is not None:
            up = self.target_upside_pct
            if up > 20.0:
                delta += 3.0
            elif up < -10.0:
                delta -= 3.0
        # Social
        if self.social_bull_pct is not None:
            bp = self.social_bull_pct
            if bp > 65.0:
                delta += 2.0
            elif bp < 35.0:
                delta -= 2.0
        return round(max(-8.0, min(10.0, delta)), 1)

    def display(self) -> str:
        """Texto compacto para la columna Intel del panel."""
        parts: list[str] = []
        if self.analyst_label:
            upside = f" {self.target_upside_pct:+.0f}%" if self.target_upside_pct is not None else ""
            parts.append(f"{self.analyst_label}{upside}")
        if self.social_msgs_24h is not None and self.social_msgs_24h > 0:
            bull = f"{self.social_bull_pct:.0f}%🐂" if self.social_bull_pct else ""
            parts.append(f"{self.social_msgs_24h}msg {bull}".strip())
        return " · ".join(parts) if parts else "-"


_RATING_LABELS: dict[str, str] = {
    "1.0-1.5": "Compra fuerte",
    "1.5-2.0": "Compra",
    "2.0-3.0": "Neutral",
    "3.0-4.0": "Venta",
    "4.0-5.0": "Venta fuerte",
}

def _rating_label(rating: float) -> str:
    if rating <= 1.5: return "Compra fuerte"
    if rating <= 2.0: return "Compra"
    if rating <= 3.0: return "Neutral"
    if rating <= 4.0: return "Venta"
    return "Venta fuerte"


# ── Yahoo Finance ─────────────────────────────────────────────────────────────

def _yahoo_analyst(adr_ticker: str) -> tuple[Optional[float], Optional[float], Optional[int]]:
    """(rating_mean, target_upside_pct, n_analysts). None si falla."""
    if _requests is None:
        return None, None, None
    try:
        url  = _YAHOO_URL.format(ticker=adr_ticker)
        hdrs = {"User-Agent": "Mozilla/5.0 (compatible; optionsdesk/1.0)"}
        resp = _requests.get(url, headers=hdrs, timeout=_HTTP_TIMEOUT)
        if resp.status_code != 200:
            return None, None, None
        data = resp.json()
        result = data.get("quoteSummary", {}).get("result", [])
        if not result:
            return None, None, None
        fd = result[0].get("financialData", {})
        rt = result[0].get("recommendationTrend", {})

        # Rating: recommendationMean (Yahoo 1=BUY..5=SELL)
        rating = None
        raw_r = fd.get("recommendationMean", {})
        if isinstance(raw_r, dict):
            rating = raw_r.get("raw")
        elif isinstance(raw_r, (int, float)):
            rating = float(raw_r)

        # Target upside
        upside = None
        target_raw = fd.get("targetMeanPrice", {})
        current_raw = fd.get("currentPrice", {})
        t = (target_raw.get("raw") if isinstance(target_raw, dict) else target_raw) or 0
        c = (current_raw.get("raw") if isinstance(current_raw, dict) else current_raw) or 0
        if t and c:
            upside = round((t / c - 1.0) * 100.0, 1)

        # n_analysts
        n = None
        trend = rt.get("trend", [])
        if trend and isinstance(trend, list):
            t0 = trend[0] if trend else {}
            n = t0.get("strongBuy", 0) + t0.get("buy", 0) + t0.get("hold", 0) + t0.get("sell", 0) + t0.get("strongSell", 0)

        return (float(rating) if rating else None), upside, (int(n) if n else None)
    except Exception as exc:
        logger.debug("_yahoo_analyst(%s) fallo: %s", adr_ticker, exc)
        return None, None, None


# ── StockTwits ────────────────────────────────────────────────────────────────

def _stocktwits_social(adr_ticker: str) -> tuple[Optional[int], Optional[float], Optional[float]]:
    """(msgs_24h, bull_pct, bear_pct). None si falla."""
    if _requests is None:
        return None, None, None
    try:
        # StockTwits usa el ticker sin ".US" (ej. GGAL no GGAL.US)
        ticker = adr_ticker.split(".")[0].upper()
        url  = _STOCKTWITS_URL.format(ticker=ticker)
        hdrs = {"User-Agent": "Mozilla/5.0 (compatible; optionsdesk/1.0)"}
        resp = _requests.get(url, headers=hdrs, timeout=_HTTP_TIMEOUT)
        if resp.status_code != 200:
            return None, None, None
        data = resp.json()
        msgs = data.get("messages", [])
        if not msgs:
            return 0, None, None

        now = datetime.utcnow()
        cutoff = now - timedelta(hours=24)
        recent = []
        for m in msgs:
            created_str = m.get("created_at", "")
            try:
                dt = datetime.strptime(created_str[:19], "%Y-%m-%dT%H:%M:%S")
                if dt >= cutoff:
                    recent.append(m)
            except ValueError:
                pass

        n_total = len(recent)
        bull = sum(1 for m in recent
                   if (m.get("entities", {}) or {}).get("sentiment", {}) == {"basic": "Bullish"}
                   or (isinstance((m.get("entities", {}) or {}).get("sentiment"), dict)
                       and (m.get("entities", {}) or {}).get("sentiment", {}).get("basic") == "Bullish"))
        bear = sum(1 for m in recent
                   if (isinstance((m.get("entities", {}) or {}).get("sentiment"), dict)
                       and (m.get("entities", {}) or {}).get("sentiment", {}).get("basic") == "Bearish"))

        n_sentiment = bull + bear
        bull_pct = round(bull / n_sentiment * 100.0, 1) if n_sentiment > 0 else None
        bear_pct = round(bear / n_sentiment * 100.0, 1) if n_sentiment > 0 else None
        return n_total, bull_pct, bear_pct
    except Exception as exc:
        logger.debug("_stocktwits_social(%s) fallo: %s", adr_ticker, exc)
        return None, None, None


# ── Cache + orquestador ───────────────────────────────────────────────────────

def _cache_path(adr_ticker: str, kind: str, cache_dir: Path) -> Path:
    return cache_dir / f"{adr_ticker.upper().replace('.','_')}_{kind}.json"


def _read_cache(path: Path, ttl_h: int) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        ts_str = data.get("_cached_at", "")
        if ts_str:
            ts = datetime.fromisoformat(ts_str)
            if (datetime.utcnow() - ts).total_seconds() < ttl_h * 3600:
                return data
        return None
    except Exception:
        return None


def _write_cache(path: Path, data: dict) -> None:
    try:
        data["_cached_at"] = datetime.utcnow().isoformat()
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        logger.debug("_write_cache(%s) fallo: %s", path, exc)


def fetch_market_intel(
    local_symbol: str,
    *,
    cache_dir: Optional[Path] = None,
) -> Optional[MarketIntel]:
    """Obtiene el intel de mercado para un símbolo local (ej. "GGAL").

    Busca el ADR en la tabla de configuración. Si no tiene ADR, devuelve None.
    Usa cache (analistas 24h, social 1h). Nunca lanza — devuelve None si falla.
    """
    try:
        from optionsdesk.config.settings import settings as _s
        adr_map = _s.adr_map()
        sym = local_symbol.upper()
        if sym not in adr_map:
            return None
        adr_ticker, _ = adr_map[sym]
        _cache_dir = cache_dir or (_s.history_dir / "intel_cache")
        _cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.debug("fetch_market_intel setup fallo: %s", exc)
        return None

    # Analistas (cache 24h)
    analyst_path = _cache_path(adr_ticker, "analyst", _cache_dir)
    analyst_cache = _read_cache(analyst_path, _CACHE_TTL_ANALYST_H)
    if analyst_cache:
        rating = analyst_cache.get("rating")
        upside = analyst_cache.get("upside")
        n_anal = analyst_cache.get("n_analysts")
    else:
        rating, upside, n_anal = _yahoo_analyst(adr_ticker)
        _write_cache(analyst_path, {"rating": rating, "upside": upside, "n_analysts": n_anal})

    # Social (cache 1h)
    social_path = _cache_path(adr_ticker, "social", _cache_dir)
    social_cache = _read_cache(social_path, _CACHE_TTL_SOCIAL_H)
    if social_cache:
        msgs = social_cache.get("msgs")
        bull = social_cache.get("bull_pct")
        bear = social_cache.get("bear_pct")
    else:
        msgs, bull, bear = _stocktwits_social(adr_ticker)
        _write_cache(social_path, {"msgs": msgs, "bull_pct": bull, "bear_pct": bear})

    return MarketIntel(
        ticker=adr_ticker,
        analyst_rating=float(rating) if rating is not None else None,
        analyst_label=_rating_label(float(rating)) if rating is not None else None,
        target_upside_pct=float(upside) if upside is not None else None,
        n_analysts=int(n_anal) if n_anal is not None else None,
        social_msgs_24h=int(msgs) if msgs is not None else None,
        social_bull_pct=float(bull) if bull is not None else None,
        social_bear_pct=float(bear) if bear is not None else None,
        as_of=date.today(),
    )
