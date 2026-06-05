"""Seleccion dinamica del universo de acciones argentinas por liquidez.

MERVAL no es Nasdaq: la pregunta operativa no es solo "que sube", sino tambien
"donde puedo entrar y salir sin que el spread me cobre la estrategia". Este modulo
rankea candidatos por valor operado/volumen usando quotes vivos cuando existen y
cache diario real como fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from optionsdesk.config.settings import settings
from optionsdesk.data.providers.base import Quote


@dataclass(frozen=True)
class UniverseMember:
    symbol: str
    score: float
    avg_volume: float
    avg_value_ars: float
    source: str


def rank_stock_universe(
    symbols: list[str],
    *,
    quotes: Optional[dict[str, Quote]] = None,
    top_n: Optional[int] = None,
    lookback: Optional[int] = None,
) -> list[UniverseMember]:
    """Rankea simbolos por liquidez, priorizando valor operado reciente.

    `quotes` puede venir de Primary/IOL; si no trae volumen util, se consulta el
    cache diario real de `UnderlyingHistory` con `allow_synthetic=False`.
    """
    clean_symbols = list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))
    qmap = {str(k).upper(): v for k, v in (quotes or {}).items() if v is not None}
    limit = top_n if top_n is not None else settings.stock_universe_top_n
    days = lookback if lookback is not None else settings.stock_universe_volume_lookback

    members = [_member_from_quote(sym, qmap.get(sym)) for sym in clean_symbols]
    missing_or_weak = {
        m.symbol
        for m in members
        if m.score <= 0 or m.source == "quote_sin_volumen"
    }

    if missing_or_weak:
        history_members = {
            m.symbol: m
            for m in (_member_from_history(sym, days=days) for sym in missing_or_weak)
            if m is not None
        }
        members = [
            history_members.get(m.symbol, m)
            if m.symbol in missing_or_weak else m
            for m in members
        ]

    ranked = sorted(
        members,
        key=lambda m: (m.score, m.avg_value_ars, m.avg_volume, m.symbol),
        reverse=True,
    )
    return ranked[:limit]


def select_top_symbols(
    symbols: list[str],
    *,
    quotes: Optional[dict[str, Quote]] = None,
    top_n: Optional[int] = None,
    lookback: Optional[int] = None,
) -> list[str]:
    ranked = rank_stock_universe(
        symbols, quotes=quotes, top_n=top_n, lookback=lookback,
    )
    selected = [m.symbol for m in ranked if m.score > 0]
    return selected or list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))[:top_n]


def _member_from_quote(symbol: str, quote: Optional[Quote]) -> UniverseMember:
    if quote is None:
        return UniverseMember(symbol, 0.0, 0.0, 0.0, "sin_quote")
    mid = float(quote.mid or quote.last or 0.0)
    volume = float(getattr(quote, "volume", 0.0) or 0.0)
    value = max(mid, 0.0) * max(volume, 0.0)
    source = "quote" if value > 0 else "quote_sin_volumen"
    return UniverseMember(symbol, value, volume, value, source)


def _member_from_history(symbol: str, *, days: int) -> Optional[UniverseMember]:
    try:
        from optionsdesk.data.history import UnderlyingHistory

        df = UnderlyingHistory().daily(symbol, days=days, allow_synthetic=False)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    data = df.copy()
    for col in ("close", "volume"):
        if col not in data.columns:
            return None
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["close", "volume"])
    data = data[(data["close"] > 0) & (data["volume"] > 0)]
    if data.empty:
        return None
    tail = data.tail(days)
    avg_volume = float(tail["volume"].mean())
    avg_value = float((tail["close"] * tail["volume"]).mean())
    return UniverseMember(symbol, avg_value, avg_volume, avg_value, "history")
