"""Lightweight intraday context for GGAL option scalping.

This module is intentionally smaller than the daily technical analyzer. Its job
is to summarize recent 1m/5m spot bars for the scalping scanners without pulling
in swing-oriented assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class IntradayTechnicalSnapshot:
    trend: str
    rsi: float
    atr_pct: float
    sma_fast: float
    sma_slow: float
    momentum: float
    read: str
    macd_hist: float = 0.0
    signal_strength: str = "neutral"
    bos: Optional[str] = None
    choch: Optional[str] = None
    fvgs: list = field(default_factory=list)
    order_block: object | None = None
    nearest_fvg: object | None = None
    vol_confirms: bool = False
    momentum_threshold_pct: float = 0.18
    source: str = "intraday_bars"


def build_intraday_snapshot(
    bars_1m: pd.DataFrame,
    bars_5m: Optional[pd.DataFrame] = None,
    pulse=None,
    *,
    now: Optional[datetime] = None,
    momentum_threshold_pct: float = 0.18,
) -> Optional[IntradayTechnicalSnapshot]:
    """Build a scalping context from completed intraday bars only."""
    one = _completed(_normalize_bars(bars_1m), "1min", now)
    five = _completed(_normalize_bars(bars_5m), "5min", now)
    if five.empty and len(one) >= 5:
        five = resample_ohlc_bars(one, "5min")

    base = five if len(five) >= 4 else one
    if len(one) < 5 or len(base) < 4:
        return None

    close = base["close"].astype(float)
    spot = float(close.iloc[-1])
    lookback = min(len(close) - 1, 4)
    ref = float(close.iloc[-1 - lookback])
    momentum = ((spot / ref) - 1.0) * 100.0 if ref > 0 else 0.0

    sma_fast = float(close.tail(min(3, len(close))).mean())
    sma_slow = float(close.tail(min(8, len(close))).mean())
    macd_hist = _macd_hist(close)
    bos = _local_bos(one)
    trend = _trend(momentum, sma_fast, sma_slow, macd_hist, bos, momentum_threshold_pct)
    rsi_val = _rsi(close)
    atr_pct = _atr_pct(one if len(one) >= 5 else base)
    pulse_dir = str(getattr(pulse, "direction", "NEUTRAL") or "NEUTRAL").upper()
    pulse_ok = bool(getattr(pulse, "confirmed", False))
    vol_confirms = _vol_confirms(one) or (
        pulse_ok
        and ((trend == "ALCISTA" and pulse_dir == "BULL") or (trend == "BAJISTA" and pulse_dir == "BEAR"))
    )
    choch = _choch_from_bos(trend, bos)
    strength = _strength(trend, momentum, bos, vol_confirms, momentum_threshold_pct)
    read = (
        f"Intradia {trend.lower()} | mom {momentum:+.2f}% | RSI {rsi_val:.0f} | "
        f"ATR {atr_pct:.2f}% | estructura {bos or 'sin BOS'} | "
        f"{'tape confirma' if vol_confirms else 'tape sin confirmar'}"
    )

    return IntradayTechnicalSnapshot(
        trend=trend,
        rsi=round(rsi_val, 1),
        atr_pct=round(atr_pct, 3),
        sma_fast=round(sma_fast, 2),
        sma_slow=round(sma_slow, 2),
        momentum=round(momentum, 3),
        read=read,
        macd_hist=round(macd_hist, 4),
        signal_strength=strength,
        bos=bos,
        choch=choch,
        vol_confirms=vol_confirms,
        momentum_threshold_pct=momentum_threshold_pct,
    )


def resample_ohlc_bars(bars: pd.DataFrame, rule: str = "5min") -> pd.DataFrame:
    data = _normalize_bars(bars)
    if data.empty:
        return data
    grouped = (
        data.set_index("time")
        .resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    return grouped[["time", "open", "high", "low", "close", "volume"]]


def _normalize_bars(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    columns = ["time", "open", "high", "low", "close", "volume"]
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)
    out = df.copy()
    if "time" not in out.columns and "date" in out.columns:
        out = out.rename(columns={"date": "time"})
    for col in columns:
        if col not in out.columns:
            out[col] = 0.0 if col != "time" else pd.NaT
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return (
        out[columns]
        .dropna(subset=["time", "open", "high", "low", "close"])
        .sort_values("time")
        .reset_index(drop=True)
    )


def _completed(df: pd.DataFrame, rule: str, now: Optional[datetime]) -> pd.DataFrame:
    if df.empty:
        return df
    if now is None:
        return df
    current = pd.Timestamp(now)
    if df["time"].dt.tz is None and current.tzinfo is not None:
        current = current.tz_localize(None)
    cutoff = current.floor(rule)
    return df[df["time"] < cutoff].reset_index(drop=True)


def _local_bos(one: pd.DataFrame) -> Optional[str]:
    if len(one) < 6:
        return None
    last = one.iloc[-1]
    history = one.iloc[-6:-1]
    close = float(last["close"])
    if close > float(history["high"].max()):
        return "BOS_UP"
    if close < float(history["low"].min()):
        return "BOS_DOWN"
    return None


def _trend(
    momentum: float,
    sma_fast: float,
    sma_slow: float,
    macd_hist: float,
    bos: Optional[str],
    threshold: float,
) -> str:
    if bos == "BOS_UP" and momentum >= 0:
        return "ALCISTA"
    if bos == "BOS_DOWN" and momentum <= 0:
        return "BAJISTA"
    if momentum >= threshold and sma_fast >= sma_slow and macd_hist >= 0:
        return "ALCISTA"
    if momentum <= -threshold and sma_fast <= sma_slow and macd_hist <= 0:
        return "BAJISTA"
    return "LATERAL"


def _choch_from_bos(trend: str, bos: Optional[str]) -> Optional[str]:
    if trend == "ALCISTA" and bos == "BOS_DOWN":
        return "CHOCH_DOWN"
    if trend == "BAJISTA" and bos == "BOS_UP":
        return "CHOCH_UP"
    return None


def _strength(trend: str, momentum: float, bos: Optional[str], vol_ok: bool, threshold: float) -> str:
    if trend == "LATERAL":
        return "neutral"
    if bos and vol_ok:
        return "fuerte"
    if abs(momentum) >= threshold * 1.8 or bos or vol_ok:
        return "moderada"
    return "debil"


def _rsi(close: pd.Series, n: int = 6) -> float:
    if len(close) < 3:
        return 50.0
    delta = close.diff()
    gain = delta.clip(lower=0).tail(n).mean()
    loss = (-delta.clip(upper=0)).tail(n).mean()
    if pd.isna(gain) or pd.isna(loss):
        return 50.0
    if loss <= 1e-9:
        return 100.0 if gain > 0 else 50.0
    rs = float(gain) / float(loss)
    return max(0.0, min(100.0, 100.0 - (100.0 / (1.0 + rs))))


def _atr_pct(bars: pd.DataFrame) -> float:
    if bars.empty:
        return 0.0
    hi = bars["high"].astype(float)
    lo = bars["low"].astype(float)
    cl = bars["close"].astype(float)
    prev = cl.shift(1)
    tr = pd.concat([hi - lo, (hi - prev).abs(), (lo - prev).abs()], axis=1).max(axis=1)
    atr = float(tr.tail(min(len(tr), 10)).median() or 0.0)
    spot = float(cl.iloc[-1] or 0.0)
    return atr / spot * 100.0 if spot > 0 else 0.0


def _macd_hist(close: pd.Series) -> float:
    if len(close) < 4:
        return 0.0
    fast = close.ewm(span=3, adjust=False).mean()
    slow = close.ewm(span=8, adjust=False).mean()
    return float((fast - slow).iloc[-1])


def _vol_confirms(one: pd.DataFrame) -> bool:
    if one.empty or "volume" not in one.columns:
        return False
    vol = one["volume"].astype(float)
    if float(vol.sum()) <= 0 or len(vol) < 5:
        return False
    avg = float(vol.tail(10).mean() or 0.0)
    return avg > 0 and float(vol.iloc[-1]) >= avg * 1.15

