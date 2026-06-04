"""Signal-driven paper demo for real recorded GGAL option snapshots.

This is not an order router and not a synthetic simulator. It replays the same
scalping signal path on locally recorded snapshots, then applies bid/ask fills
with the conservative replay engine.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from optionsdesk.backtest.scalping import ScalpReplayPlan, simulate_plan_on_snapshots
from optionsdesk.config.settings import settings
from optionsdesk.core.carry import estimate_carry
from optionsdesk.data.providers.base import MarketDataHealth, OptionsChain, Quote
from optionsdesk.signals.intraday_context import build_intraday_snapshot, resample_ohlc_bars
from optionsdesk.signals.scalp_tickets import (
    DailyScalpPerformance,
    build_manual_scalp_ticket,
    detect_underlying_trigger,
)
from optionsdesk.signals.scalping import compute_spot_tape_pulse, scan_all


@dataclass(frozen=True)
class DemoTrade:
    ticket_id: str
    setup: str
    symbol: str
    action: str
    created_ts: datetime
    entry_ts: datetime
    exit_ts: datetime
    entry_price: float
    exit_price: float
    quantity: int
    pnl_ars: float
    exit_reason: str
    fees_ars: float
    entry_limit: float
    stop_price: float
    target_price: float
    spot_signal: float
    signal_score: float
    rationale: str


@dataclass(frozen=True)
class DemoResult:
    trades: list[DemoTrade]
    skipped_reasons: dict[str, int]
    snapshots: int
    symbols: int
    data_start: Optional[datetime] = None
    data_end: Optional[datetime] = None
    granularity_s: Optional[float] = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def total_pnl_ars(self) -> float:
        return round(sum(t.pnl_ars for t in self.trades), 2)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.pnl_ars > 0) / len(self.trades)

    @property
    def profit_factor(self) -> float:
        wins = sum(t.pnl_ars for t in self.trades if t.pnl_ars > 0)
        losses = abs(sum(t.pnl_ars for t in self.trades if t.pnl_ars < 0))
        return round(wins / losses, 4) if losses > 0 else (float("inf") if wins > 0 else 0.0)

    @property
    def max_drawdown_ars(self) -> float:
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for trade in self.trades:
            equity += trade.pnl_ars
            peak = max(peak, equity)
            max_dd = min(max_dd, equity - peak)
        return round(max_dd, 2)

    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(t) for t in self.trades])


def run_signal_demo_on_snapshots(
    snapshots: pd.DataFrame,
    expiry_calendar: dict,
    *,
    capital: float,
    risk_profile: str = "balanced",
    max_trades: int = 25,
    momentum_threshold_pct: float = 0.18,
) -> DemoResult:
    df = _prepare_snapshots(snapshots)
    if df.empty:
        return DemoResult([], {"sin_snapshots": 1}, 0, 0)

    unique_ts = sorted(pd.Timestamp(ts) for ts in df["timestamp"].drop_duplicates())
    skipped: dict[str, int] = {}
    trades: list[DemoTrade] = []
    occupied_until: Optional[pd.Timestamp] = None

    for ts in unique_ts:
        if len(trades) >= max_trades:
            break
        if occupied_until is not None and ts <= occupied_until:
            continue

        history = df[df["timestamp"] <= ts]
        bars_1m = _spot_bars_from_snapshots(history)
        bars_5m = resample_ohlc_bars(bars_1m, "5min")
        samples = _spot_samples(history)
        pulse = compute_spot_tape_pulse(samples[-8:], min_observations=4, min_span_s=60.0, min_move_pct=0.08)
        snap = build_intraday_snapshot(
            bars_1m,
            bars_5m,
            pulse,
            now=ts.to_pydatetime(),
            momentum_threshold_pct=momentum_threshold_pct,
        )
        if snap is None:
            _count(skipped, "sin_contexto_intradia")
            continue

        group = df[df["timestamp"] == ts]
        chain = _chain_from_group(group, expiry_calendar)
        if chain is None:
            _count(skipped, "sin_cadena")
            continue

        carry = estimate_carry(
            chain,
            expiry_calendar,
            fallback_tna_pct=_fallback_tna(group),
            max_quote_age_s=300.0,
        )
        health = MarketDataHealth(
            source=_health_source(group),
            connected=True,
            ticket_capable=True,
            options_seen=len(chain.options),
        )
        greeks, signals = scan_all(
            chain,
            expiry_calendar,
            snap=snap,
            realized_vol=None,
            r=carry.continuous_rate,
            capital=capital,
            positions=[],
            risk_profile=risk_profile,
            require_live_confirmation=True,
            live_confirmation=pulse.confirmed,
            live_direction=pulse.direction,
            now=ts.to_pydatetime(),
        )
        trigger = detect_underlying_trigger(bars_1m, bars_5m, pulse, now=ts.to_pydatetime())
        ticket, blockers = build_manual_scalp_ticket(
            signals,
            greeks,
            chain,
            health,
            trigger,
            carry,
            capital=capital,
            local_positions=[],
            account_reconciled=True,
            daily_performance=DailyScalpPerformance(),
            now=ts.to_pydatetime(),
        )
        if ticket is None:
            for blocker in blockers or ["sin_ticket"]:
                _count(skipped, blocker)
            continue

        plan = ScalpReplayPlan(
            symbol=ticket.selection.symbol,
            action=ticket.selection.action,
            entry_limit=ticket.selection.limit_price,
            stop_price=ticket.option_stop,
            target_price=ticket.option_take_profit,
            quantity=1,
            created_ts=ts.to_pydatetime() - timedelta(microseconds=1),
            time_stop_min=ticket.time_stop_min,
            flat_before_close_min=settings.scalping_flat_before_close_min,
        )
        future = df[df["timestamp"] >= ts]
        replay = simulate_plan_on_snapshots(future, plan)
        if replay is None:
            _count(skipped, "sin_fill_o_salida")
            continue

        signal = next((s for s in signals if s.symbol == ticket.selection.symbol), None)
        demo_trade = DemoTrade(
            ticket_id=ticket.ticket_id,
            setup=ticket.trigger.setup,
            symbol=replay.symbol,
            action=replay.action,
            created_ts=ticket.created_at,
            entry_ts=replay.entry_ts,
            exit_ts=replay.exit_ts,
            entry_price=replay.entry_price,
            exit_price=replay.exit_price,
            quantity=replay.quantity,
            pnl_ars=replay.pnl_ars,
            exit_reason=replay.exit_reason,
            fees_ars=replay.fees_ars,
            entry_limit=ticket.selection.limit_price,
            stop_price=ticket.option_stop,
            target_price=ticket.option_take_profit,
            spot_signal=ticket.trigger.spot,
            signal_score=float(getattr(signal, "score", 0.0) or 0.0),
            rationale=ticket.trigger.rationale,
        )
        trades.append(demo_trade)
        occupied_until = pd.Timestamp(replay.exit_ts)

    spacing_s = _median_spacing_s(unique_ts)
    notes = [
        "Entrada paper usa el book del snapshot de senal si el limite lo permite.",
        "Salidas usan bid/ask observados en snapshots posteriores; no se inventan ticks intermedios.",
    ]
    if spacing_s and spacing_s > 90:
        notes.append(
            "Granularidad gruesa para scalping: usar estos resultados como auditoria direccional, no como metrica fina de TP/SL."
        )
    elif spacing_s and spacing_s > 45:
        notes.append(
            "Granularidad aceptable pero no tick-by-tick: revisar slippage y frescura antes de sacar conclusiones."
        )
    return DemoResult(
        trades=trades,
        skipped_reasons=skipped,
        snapshots=len(unique_ts),
        symbols=int(df["symbol"].nunique()),
        data_start=unique_ts[0].to_pydatetime() if unique_ts else None,
        data_end=unique_ts[-1].to_pydatetime() if unique_ts else None,
        granularity_s=spacing_s,
        notes=tuple(notes),
    )


def append_demo_trades(result: DemoResult, path: Optional[Path] = None) -> int:
    if not result.trades:
        return 0
    target = path or Path("data/scalp_demo_trades.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    existing_ids: set[str] = set()
    if target.exists():
        for line in target.read_text(encoding="utf-8").splitlines():
            try:
                payload = json.loads(line)
                ticket_id = str(payload.get("ticket_id") or "")
                if ticket_id:
                    existing_ids.add(ticket_id)
            except json.JSONDecodeError:
                continue
    run_id = datetime.now().isoformat(timespec="seconds")
    saved = 0
    with target.open("a", encoding="utf-8") as fh:
        for trade in result.trades:
            if trade.ticket_id in existing_ids:
                continue
            record = asdict(trade)
            record["run_id"] = run_id
            for key in ("created_ts", "entry_ts", "exit_ts"):
                record[key] = record[key].isoformat()
            fh.write(json.dumps(record, ensure_ascii=True) + "\n")
            existing_ids.add(trade.ticket_id)
            saved += 1
    return saved


def _prepare_snapshots(snapshots: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "symbol", "spot", "bid", "ask", "last"}
    if snapshots is None or snapshots.empty or not required.issubset(set(snapshots.columns)):
        return pd.DataFrame()
    df = snapshots.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in ("spot", "bid", "ask", "last", "volume", "bid_size", "ask_size"):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return (
        df.dropna(subset=["timestamp", "symbol"])
        .sort_values(["timestamp", "symbol"])
        .reset_index(drop=True)
    )


def _chain_from_group(group: pd.DataFrame, expiry_calendar: dict) -> Optional[OptionsChain]:
    if group.empty:
        return None
    ts = pd.Timestamp(group.iloc[0]["timestamp"]).to_pydatetime()
    spot_mid = float(group["spot"].dropna().iloc[0] or 0.0)
    if spot_mid <= 0:
        return None
    spot_bid = _first_float(group, "spot_bid", spot_mid)
    spot_ask = _first_float(group, "spot_ask", spot_mid)
    spot_last = _first_float(group, "spot_last", spot_mid)
    source = _health_source(group)
    spot = Quote(
        "GGAL",
        spot_bid,
        spot_ask,
        spot_last,
        0.0,
        ts,
        source=source,
        received_at=ts,
        book_ts=ts,
    )
    options: dict[str, Quote] = {}
    for _, row in group.iterrows():
        symbol = str(row["symbol"]).upper()
        bid = float(row.get("bid", 0.0) or 0.0)
        ask = float(row.get("ask", 0.0) or 0.0)
        last = float(row.get("last", 0.0) or 0.0)
        if ask <= 0 and bid <= 0 and last <= 0:
            continue
        book_ts = _row_ts(row, "book_ts", ts)
        options[symbol] = Quote(
            symbol,
            bid,
            ask,
            last,
            float(row.get("volume", 0.0) or 0.0),
            ts,
            bid_size=float(row.get("bid_size", 0.0) or 0.0),
            ask_size=float(row.get("ask_size", 0.0) or 0.0),
            source=str(row.get("quote_source", "") or source),
            received_at=_row_ts(row, "received_at", ts),
            market_ts=_row_ts(row, "market_ts", ts),
            book_ts=book_ts,
        )
    if not options:
        return None
    return OptionsChain(
        "GGAL",
        spot,
        options,
        timestamp=ts,
        expiry_calendar=dict(expiry_calendar or {}),
    )


def _spot_samples(df: pd.DataFrame) -> list[tuple[datetime, float]]:
    points = df[["timestamp", "spot"]].drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return [(pd.Timestamp(row["timestamp"]).to_pydatetime(), float(row["spot"])) for _, row in points.iterrows()]


def _spot_bars_from_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    samples = _spot_samples(df)
    rows = [
        {"time": ts, "open": spot, "high": spot, "low": spot, "close": spot, "volume": 0.0}
        for ts, spot in samples
        if spot > 0
    ]
    return pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])


def _fallback_tna(group: pd.DataFrame) -> float:
    if "carry_tna_pct" in group.columns:
        vals = pd.to_numeric(group["carry_tna_pct"], errors="coerce").dropna()
        if not vals.empty and float(vals.iloc[0]) > 0:
            return float(vals.iloc[0])
    return settings.default_caucion_tna


def _health_source(group: pd.DataFrame) -> str:
    if "source" in group.columns:
        raw = str(group["source"].dropna().iloc[0] if not group["source"].dropna().empty else "")
        if raw.upper() in {"IOL", "PRIMARY"}:
            return raw.upper()
    return "IOL"


def _first_float(group: pd.DataFrame, column: str, default: float) -> float:
    if column not in group.columns:
        return float(default)
    vals = pd.to_numeric(group[column], errors="coerce").dropna()
    return float(vals.iloc[0]) if not vals.empty and float(vals.iloc[0]) > 0 else float(default)


def _row_ts(row: pd.Series, column: str, default: datetime) -> datetime:
    value = row.get(column)
    if value is None or pd.isna(value):
        return default
    try:
        return pd.Timestamp(value).to_pydatetime()
    except Exception:
        return default


def _median_spacing_s(timestamps: list[pd.Timestamp]) -> Optional[float]:
    if len(timestamps) < 2:
        return None
    diffs = [
        (timestamps[i] - timestamps[i - 1]).total_seconds()
        for i in range(1, len(timestamps))
        if (timestamps[i] - timestamps[i - 1]).total_seconds() > 0
    ]
    return float(pd.Series(diffs).median()) if diffs else None


def _count(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1
