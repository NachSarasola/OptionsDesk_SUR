"""Intraday replay utilities for the scalping module.

The simulator is intentionally conservative:
- precomputed/manual plans are filled only on snapshots after the signal time;
- long option entries pay ask and exits receive bid;
- short option entries receive bid and exits pay ask;
- costs are applied on both sides.

It is not a signal generator. Its job is to audit whether a scalping plan would
have survived realistic bid/ask execution without lookahead.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from optionsdesk.config.costs import CostModel, DEFAULT_COSTS


@dataclass
class ScalpReplayPlan:
    symbol: str
    action: str
    entry_limit: float
    stop_price: float
    target_price: float
    quantity: int = 1
    created_ts: Optional[datetime] = None
    time_stop_min: int = 20
    flat_before_close_min: int = 10


@dataclass
class ScalpBacktestTrade:
    symbol: str
    action: str
    entry_ts: datetime
    exit_ts: datetime
    entry_price: float
    exit_price: float
    quantity: int
    pnl_ars: float
    exit_reason: str
    fees_ars: float


@dataclass
class ScalpBacktestResult:
    trades: list[ScalpBacktestTrade]
    equity_curve: pd.Series
    pnl_series: list[float]
    capital_initial: float

    @property
    def total_pnl_ars(self) -> float:
        return float(sum(self.pnl_series))

    @property
    def win_rate(self) -> float:
        if not self.pnl_series:
            return 0.0
        return sum(1 for x in self.pnl_series if x > 0) / len(self.pnl_series)


def _mid(row: pd.Series) -> float:
    bid = float(row.get("bid", 0.0) or 0.0)
    ask = float(row.get("ask", 0.0) or 0.0)
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return float(row.get("last", 0.0) or bid or ask or 0.0)


def _ensure_snapshot_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" not in out.columns:
        raise ValueError("snapshot dataframe requires a timestamp column")
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    for col in ("bid", "ask", "last", "volume", "bid_size", "ask_size"):
        if col not in out.columns:
            out[col] = 0.0
    return out


def _entry_fill(row: pd.Series, plan: ScalpReplayPlan) -> Optional[float]:
    bid = float(row.get("bid", 0.0) or 0.0)
    ask = float(row.get("ask", 0.0) or 0.0)
    last = float(row.get("last", 0.0) or 0.0)
    if plan.action.startswith("BUY_"):
        if ask > 0 and ask <= plan.entry_limit:
            return ask
        if ask <= 0 and last > 0 and last <= plan.entry_limit:
            return last
    else:
        if bid > 0 and bid >= plan.entry_limit:
            return bid
        if bid <= 0 and last > 0 and last >= plan.entry_limit:
            return last
    return None


def _exit_px(row: pd.Series, action: str) -> float:
    bid = float(row.get("bid", 0.0) or 0.0)
    ask = float(row.get("ask", 0.0) or 0.0)
    last = float(row.get("last", 0.0) or 0.0)
    if action.startswith("BUY_"):
        return bid if bid > 0 else (last if last > 0 else _mid(row))
    return ask if ask > 0 else (last if last > 0 else _mid(row))


def _session_close_due(ts: pd.Timestamp, flat_before_close_min: int) -> bool:
    current = ts.to_pydatetime()
    close_dt = datetime.combine(ts.date(), dt_time(17, 0), tzinfo=current.tzinfo)
    return current >= close_dt - timedelta(minutes=flat_before_close_min)


def simulate_plan_on_snapshots(
    snapshots: pd.DataFrame,
    plan: ScalpReplayPlan,
    costs: CostModel = DEFAULT_COSTS,
) -> Optional[ScalpBacktestTrade]:
    """Replay one scalping plan on snapshots without lookahead."""
    df = _ensure_snapshot_columns(snapshots)
    df = df[df["symbol"].astype(str).str.upper() == plan.symbol.upper()]
    if plan.created_ts is not None:
        df = df[df["timestamp"] > pd.Timestamp(plan.created_ts)]
    df = df.sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        return None

    entry_i: Optional[int] = None
    entry_price = 0.0
    for i, row in df.iterrows():
        fill = _entry_fill(row, plan)
        if fill is not None and fill > 0:
            entry_i = int(i)
            entry_price = float(fill)
            break
    if entry_i is None:
        return None

    entry_ts = pd.Timestamp(df.loc[entry_i, "timestamp"])
    exit_i = entry_i
    exit_reason = "END"
    exit_price = _exit_px(df.loc[entry_i], plan.action)

    for i in range(entry_i + 1, len(df)):
        row = df.loc[i]
        ts = pd.Timestamp(row["timestamp"])
        px = _exit_px(row, plan.action)

        if plan.action.startswith("BUY_"):
            if px <= plan.stop_price:
                exit_reason = "STOP"
                exit_i = i
                exit_price = px
                break
            if px >= plan.target_price:
                exit_reason = "TAKE_PROFIT"
                exit_i = i
                exit_price = px
                break
        else:
            if px >= plan.stop_price:
                exit_reason = "STOP"
                exit_i = i
                exit_price = px
                break
            if px <= plan.target_price:
                exit_reason = "TAKE_PROFIT"
                exit_i = i
                exit_price = px
                break

        if ts - entry_ts >= pd.Timedelta(minutes=plan.time_stop_min):
            exit_reason = "TIME_STOP"
            exit_i = i
            exit_price = px
            break
        if _session_close_due(ts, plan.flat_before_close_min):
            exit_reason = "SESSION_CLOSE"
            exit_i = i
            exit_price = px
            break

    if exit_i <= entry_i:
        if len(df) > entry_i + 1:
            exit_i = len(df) - 1
            exit_price = _exit_px(df.loc[exit_i], plan.action)
        else:
            # Trade en última vela sin salida real — descartarlo
            return None

    qty = max(int(plan.quantity), 1)
    entry_side = "option_buy" if plan.action.startswith("BUY_") else "option_sell"
    exit_side = "option_sell" if plan.action.startswith("BUY_") else "option_buy"
    entry_fee = costs.gross_cost(entry_price * 100.0 * qty, entry_side)
    exit_fee = costs.gross_cost(exit_price * 100.0 * qty, exit_side)
    fees = entry_fee + exit_fee

    if plan.action.startswith("BUY_"):
        pnl = (exit_price - entry_price) * 100.0 * qty - fees
    else:
        pnl = (entry_price - exit_price) * 100.0 * qty - fees

    return ScalpBacktestTrade(
        symbol=plan.symbol,
        action=plan.action,
        entry_ts=entry_ts.to_pydatetime(),
        exit_ts=pd.Timestamp(df.loc[exit_i, "timestamp"]).to_pydatetime(),
        entry_price=round(entry_price, 4),
        exit_price=round(float(exit_price), 4),
        quantity=qty,
        pnl_ars=round(float(pnl), 2),
        exit_reason=exit_reason,
        fees_ars=round(float(fees), 2),
    )


def simulate_plans_on_snapshots(
    snapshots: pd.DataFrame,
    plans: list[ScalpReplayPlan],
    capital_initial: float = 0.0,
) -> ScalpBacktestResult:
    trades = [
        trade
        for plan in plans
        if (trade := simulate_plan_on_snapshots(snapshots, plan)) is not None
    ]
    trades.sort(key=lambda t: t.exit_ts)
    capital = float(capital_initial)
    points: dict[pd.Timestamp, float] = {}
    for trade in trades:
        capital += trade.pnl_ars
        points[pd.Timestamp(trade.exit_ts)] = capital
    equity = pd.Series(points) if points else pd.Series(dtype=float)
    return ScalpBacktestResult(
        trades=trades,
        equity_curve=equity,
        pnl_series=[t.pnl_ars for t in trades],
        capital_initial=capital_initial,
    )


def load_snapshots_range(
    snapshots_dir: str | Path,
    start: str | date,
    end: str | date,
) -> pd.DataFrame:
    if isinstance(start, str):
        start = date.fromisoformat(start)
    if isinstance(end, str):
        end = date.fromisoformat(end)
    root = Path(snapshots_dir)
    frames: list[pd.DataFrame] = []
    for day in pd.date_range(start, end, freq="D").strftime("%Y-%m-%d"):
        day_dir = root / day
        if not day_dir.exists():
            continue
        files = sorted(day_dir.glob("*.parquet"))
        if files:
            frames.append(pd.concat([pd.read_parquet(f) for f in files], ignore_index=True))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
