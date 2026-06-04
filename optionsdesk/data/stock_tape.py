"""Tape intradia multi-simbolo para acciones argentinas."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from optionsdesk.config.settings import settings
from optionsdesk.data.providers.base import Quote

_BA = ZoneInfo("America/Argentina/Buenos_Aires")


def append_stock_quotes(
    quotes: Iterable[Quote],
    *,
    tape_dir: Optional[Path] = None,
    now: Optional[datetime] = None,
    min_interval_s: float = 10.0,
) -> int:
    """Persiste quotes reales de acciones en JSONL diario, deduplicando por simbolo."""
    current = (now or datetime.now(_BA)).astimezone(_BA)
    root = Path(tape_dir or settings.stock_tape_dir)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{current.date().isoformat()}.jsonl"
    last_by_symbol = _last_timestamps(target)
    written = 0
    with target.open("a", encoding="utf-8") as fh:
        for quote in quotes:
            symbol = quote.symbol.upper()
            if quote.mid <= 0:
                continue
            last_ts = last_by_symbol.get(symbol)
            if last_ts and (current - last_ts).total_seconds() < min_interval_s:
                continue
            record = {
                "timestamp": current.isoformat(timespec="seconds"),
                "symbol": symbol,
                "bid": float(quote.bid),
                "ask": float(quote.ask),
                "last": float(quote.last),
                "mid": float(quote.mid),
                "volume": float(quote.volume),
                "source": quote.source,
                "received_at": quote.received_at.isoformat() if quote.received_at else None,
                "market_ts": quote.market_ts.isoformat() if quote.market_ts else None,
                "book_ts": quote.book_ts.isoformat() if quote.book_ts else None,
            }
            fh.write(json.dumps(record, ensure_ascii=True) + "\n")
            last_by_symbol[symbol] = current
            written += 1
    return written


def load_stock_tape(
    *,
    tape_dir: Optional[Path] = None,
    day: Optional[date] = None,
    symbols: Optional[Iterable[str]] = None,
    now: Optional[datetime] = None,
    max_age_hours: float = 8.0,
) -> pd.DataFrame:
    """Carga tape del dia para uno o mas simbolos."""
    current = (now or datetime.now(_BA)).astimezone(_BA)
    target_day = day or current.date()
    root = Path(tape_dir or settings.stock_tape_dir)
    file = root / f"{target_day.isoformat()}.jsonl"
    columns = ["timestamp", "symbol", "bid", "ask", "last", "mid", "volume", "source", "received_at", "market_ts", "book_ts"]
    if not file.exists():
        return pd.DataFrame(columns=columns)
    wanted = {s.upper() for s in symbols} if symbols else None
    cutoff = current - timedelta(hours=max_age_hours)
    rows = []
    for line in file.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            symbol = str(record.get("symbol", "")).upper()
            if wanted is not None and symbol not in wanted:
                continue
            ts = pd.Timestamp(record.get("timestamp")).to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=_BA)
            ts = ts.astimezone(_BA)
            if not (cutoff <= ts <= current + timedelta(minutes=1)):
                continue
            record["timestamp"] = ts
            rows.append(record)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows)
    for col in ("bid", "ask", "last", "mid", "volume"):
        df[col] = pd.to_numeric(df.get(col, 0.0), errors="coerce").fillna(0.0)
    return df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def stock_tape_bars(df: pd.DataFrame, symbol: str, rule: str = "1min") -> pd.DataFrame:
    """Convierte el tape de un simbolo en OHLC sin inventar volumen negociado."""
    columns = ["time", "open", "high", "low", "close", "volume"]
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)
    symbol = symbol.upper()
    data = df[df["symbol"].astype(str).str.upper() == symbol].copy()
    if data.empty:
        return pd.DataFrame(columns=columns)
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["price"] = pd.to_numeric(data.get("mid", 0.0), errors="coerce").fillna(0.0)
    fallback = pd.to_numeric(data.get("last", 0.0), errors="coerce").fillna(0.0)
    data.loc[data["price"] <= 0, "price"] = fallback[data["price"] <= 0]
    data = data.dropna(subset=["timestamp"])
    data = data[data["price"] > 0].sort_values("timestamp")
    if data.empty:
        return pd.DataFrame(columns=columns)
    bars = data.set_index("timestamp")["price"].resample(rule).ohlc().dropna().reset_index()
    bars["volume"] = 0.0
    bars = bars.rename(columns={"timestamp": "time"})
    return bars[columns]


def _last_timestamps(path: Path) -> dict[str, datetime]:
    if not path.exists():
        return {}
    result: dict[str, datetime] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            symbol = str(record.get("symbol", "")).upper()
            ts = datetime.fromisoformat(str(record.get("timestamp")))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=_BA)
            result[symbol] = ts.astimezone(_BA)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    return result
