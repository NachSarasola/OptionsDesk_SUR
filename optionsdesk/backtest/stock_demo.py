"""Automatic paper trading loop for the Argentine stocks panel.

The module deliberately stays read-only with respect to the broker. It consumes
observed quotes and locally generated signals, then persists paper positions and
closed trades under their own files.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from optionsdesk.config.costs import CostModel, DEFAULT_COSTS
from optionsdesk.config.settings import settings
from optionsdesk.data.providers.base import Quote
from optionsdesk.signals.stock_signals import StockSignal
from optionsdesk.backtest.adaptive import AdaptiveContext, analyze_performance

_BA = ZoneInfo("America/Argentina/Buenos_Aires")
_SESSION_CLOSE_HOUR = 16
_SESSION_CLOSE_MINUTE = 45
# Spread máximo permitido en el demo (consistente con stock_signals)
_MAX_SPREAD_PCT_DEMO = 2.5


@dataclass
class StockDemoPosition:
    id: str
    symbol: str
    strategy: str
    signal_type: str
    entry_ts: datetime
    entry_price: float
    quantity: int
    stop_price: float
    target_price: float
    entry_fee_ars: float
    time_stop_min: Optional[int] = None
    max_holding_days: Optional[int] = None
    rationale: str = ""
    score: float = 0.0

    @property
    def notional_ars(self) -> float:
        return round(self.entry_price * self.quantity, 2)


@dataclass(frozen=True)
class StockDemoTrade:
    id: str
    symbol: str
    strategy: str
    signal_type: str
    entry_ts: datetime
    exit_ts: datetime
    entry_price: float
    exit_price: float
    quantity: int
    pnl_ars: float
    exit_reason: str
    fees_ars: float
    stop_price: float
    target_price: float
    rationale: str = ""
    score: float = 0.0


@dataclass(frozen=True)
class StockDemoResult:
    positions: list[StockDemoPosition]
    closed_trades: list[StockDemoTrade]
    skipped_reasons: dict[str, int] = field(default_factory=dict)
    opened_count: int = 0
    closed_count: int = 0

    @property
    def open_scalps(self) -> int:
        return sum(1 for p in self.positions if p.strategy == "SCALP")

    @property
    def open_swings(self) -> int:
        return sum(1 for p in self.positions if p.strategy == "SWING")

    @property
    def total_pnl_ars(self) -> float:
        return round(sum(t.pnl_ars for t in self.closed_trades), 2)

    @property
    def win_rate(self) -> float:
        if not self.closed_trades:
            return 0.0
        return sum(1 for t in self.closed_trades if t.pnl_ars > 0) / len(self.closed_trades)

    @property
    def equity_curve(self) -> pd.DataFrame:
        rows = []
        equity = 0.0
        for trade in sorted(self.closed_trades, key=lambda t: t.exit_ts):
            equity += trade.pnl_ars
            rows.append({"timestamp": trade.exit_ts, "equity_ars": round(equity, 2)})
        return pd.DataFrame(rows)


def run_stock_demo_tick(
    quotes: dict[str, Quote],
    signals: Iterable[StockSignal],
    *,
    now: Optional[datetime] = None,
    positions_path: Optional[Path] = None,
    trades_path: Optional[Path] = None,
    trade_ars: Optional[float] = None,
    max_scalps: Optional[int] = None,
    max_swings: Optional[int] = None,
    quote_max_age_s: Optional[float] = None,
    costs: CostModel = DEFAULT_COSTS,
    adaptive_context: Optional[AdaptiveContext] = None,
    block_junk: bool = True,
) -> StockDemoResult:
    """Advance the automatic paper loop once.

    Existing positions are evaluated first. New entries use ask and exits use
    bid. Invalid or stale books are skipped, so repeated dashboard refreshes do
    not manufacture fills.
    """
    current = _to_ba(now or datetime.now(_BA))
    qmap = {str(k).upper(): v for k, v in quotes.items() if v is not None}
    pos_file = Path(positions_path or settings.stock_demo_positions_file)
    trade_file = Path(trades_path or settings.stock_demo_trades_file)
    max_age = float(quote_max_age_s if quote_max_age_s is not None else settings.stock_demo_quote_max_age_s)
    
    # Capital base
    base_capital = 1_800_000.0
    scalp_limit = int(max_scalps if max_scalps is not None else settings.stock_demo_max_scalps)
    swing_limit = int(max_swings if max_swings is not None else settings.stock_demo_max_swings)

    positions = load_stock_demo_positions(pos_file)
    skipped: dict[str, int] = {}
    new_trades: list[StockDemoTrade] = []
    closed_symbols: set[str] = _recently_closed_symbols(trade_file, current, cooldown_s=60.0)
    opened = 0

    # Gate anti-basura: no abrir setups NI grades que YA demostraron edge negativo con
    # muestra suficiente. Las señales nuevas nunca se bloquean. Lee el mismo directorio
    # de datos del demo para ser determinístico.
    blocked_setups: set[str] = set()
    blocked_grades: set[str] = set()
    if block_junk:
        try:
            from optionsdesk.performance.attribution import block_lists
            blocked_setups, blocked_grades = block_lists(trade_file.parent)
        except Exception:
            blocked_setups, blocked_grades = set(), set()

    survivors: list[StockDemoPosition] = []
    for position in positions:
        quote = qmap.get(position.symbol)
        if quote is None or not _book_exitable(quote):
            _count(skipped, "sin_bid_para_salir")
            survivors.append(position)
            continue
        if _quote_age_s(quote, current) > max_age:
            _count(skipped, "quote_stale_salida")
            survivors.append(position)
            continue
            
        # Universal Trailing Stop
        riesgo_inicial = position.entry_price - position.stop_price
        nuevo_stop = float(quote.bid) - riesgo_inicial
        if nuevo_stop > position.stop_price:
            position.stop_price = round(nuevo_stop, 4)
                
        reason = _exit_reason(position, quote, current)
        if reason is None:
            survivors.append(position)
            continue
        trade = _close_position(position, quote, current, reason, costs)
        new_trades.append(trade)
        closed_symbols.add(position.symbol)

    positions = survivors
    open_symbols = {p.symbol for p in positions}
    open_scalps = sum(1 for p in positions if p.strategy == "SCALP")
    open_swings = sum(1 for p in positions if p.strategy == "SWING")

    # ── Circuit breaker (ignorado en DEMO para que siga aprendiendo) ──
    if adaptive_context is not None and getattr(adaptive_context, "halted", False):
        _count(skipped, "circuit_breaker_active_but_ignored")
        # No bloqueamos el demo, queremos que siga tradeando para generar historial
    for signal in sorted(signals, key=lambda s: (s.score, s.rr), reverse=True):
        symbol = signal.symbol.upper()
        if symbol in open_symbols:
            _count(skipped, "posicion_existente")
            continue
        if symbol in closed_symbols:
            _count(skipped, "cerrada_en_mismo_tick")
            continue
        if signal.strategy == "SCALP" and open_scalps >= scalp_limit:
            _count(skipped, "max_scalps")
            continue
        if signal.strategy == "SWING" and open_swings >= swing_limit:
            _count(skipped, "max_swings")
            continue
        if block_junk:
            from optionsdesk.signals.stock_signals import signal_edge_status
            if signal_edge_status(signal, blocked_setups, blocked_grades):
                _count(skipped, "sin_edge_realizado")
                continue
        # Score mínimo adaptativo: en modo DEFENSIVE solo entran setups de alta calidad
        if adaptive_context is not None:
            min_score = getattr(adaptive_context, "min_score_for_entry", 65.0)
            if signal.score < min_score:
                _count(skipped, f"score_bajo_{adaptive_context.mode.lower()}")
                continue
        quote = qmap.get(symbol)
        if quote is None:
            _count(skipped, "sin_quote")
            continue
        if not (quote.ask > 0 and quote.bid > 0 and quote.ask > quote.bid):
            _count(skipped, "sin_book_valido")
            continue
        mid = (quote.ask + quote.bid) / 2.0
        if mid > 0 and ((quote.ask - quote.bid) / mid * 100.0) > _MAX_SPREAD_PCT_DEMO:
            _count(skipped, "spread_muy_alto")
            continue
        if _quote_age_s(quote, current) > max_age:
            _count(skipped, "quote_stale_entrada")
            continue
        # Fractional Kelly / Risk Sizing (Half Kelly of 1,800,000 ARS)
        risk_per_share = float(quote.ask) - signal.stop_price
        if risk_per_share <= 0:
            _count(skipped, "riesgo_invalido")
            continue
            
        kelly_pct = adaptive_context.half_kelly_pct if adaptive_context else 0.05
        max_risk_ars = base_capital * kelly_pct
        qty = int(max_risk_ars / risk_per_share)
        
        max_qty_capital = int(base_capital / float(quote.ask))
        qty = min(qty, max_qty_capital)
        
        if qty <= 0:
            _count(skipped, "monto_insuficiente")
            continue
        position = _open_position(signal, quote, current, qty, costs, adaptive_context)
        positions.append(position)
        opened += 1
        open_symbols.add(symbol)
        if position.strategy == "SCALP":
            open_scalps += 1
        elif position.strategy == "SWING":
            open_swings += 1

    save_stock_demo_positions(positions, pos_file)
    if new_trades:
        append_stock_demo_trades(new_trades, trade_file)
    all_trades = load_stock_demo_trades(trade_file, limit=1_000)
    return StockDemoResult(
        positions=positions,
        closed_trades=all_trades,
        skipped_reasons=skipped,
        opened_count=opened,
        closed_count=len(new_trades),
    )


def load_stock_demo_positions(path: Optional[Path] = None) -> list[StockDemoPosition]:
    target = Path(path or settings.stock_demo_positions_file)
    if not target.exists():
        return []
    try:
        raw = json.loads(target.read_text(encoding="utf-8") or "[]")
    except (OSError, json.JSONDecodeError, TypeError):
        return []
    positions = []
    for item in raw if isinstance(raw, list) else []:
        try:
            data = dict(item)
            data["entry_ts"] = _parse_dt(data["entry_ts"])
            positions.append(StockDemoPosition(**data))
        except (TypeError, KeyError, ValueError):
            continue
    return positions


def save_stock_demo_positions(
    positions: Iterable[StockDemoPosition],
    path: Optional[Path] = None,
) -> None:
    target = Path(path or settings.stock_demo_positions_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = [_json_ready(asdict(p)) for p in positions]
    target.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def append_stock_demo_trades(
    trades: Iterable[StockDemoTrade],
    path: Optional[Path] = None,
) -> int:
    target = Path(path or settings.stock_demo_trades_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = _existing_trade_ids(target)
    written = 0
    with target.open("a", encoding="utf-8") as fh:
        for trade in trades:
            if trade.id in existing_ids:
                continue
            fh.write(json.dumps(_json_ready(asdict(trade)), ensure_ascii=True) + "\n")
            existing_ids.add(trade.id)
            written += 1
    return written


def load_stock_demo_trades(
    path: Optional[Path] = None,
    *,
    limit: Optional[int] = 200,
) -> list[StockDemoTrade]:
    target = Path(path or settings.stock_demo_trades_file)
    if not target.exists():
        return []
    lines = target.read_text(encoding="utf-8").splitlines()
    if limit is not None and limit > 0:
        lines = lines[-limit:]
    trades = []
    for line in lines:
        try:
            data = json.loads(line)
            data["entry_ts"] = _parse_dt(data["entry_ts"])
            data["exit_ts"] = _parse_dt(data["exit_ts"])
            trades.append(StockDemoTrade(**data))
        except (TypeError, KeyError, ValueError, json.JSONDecodeError):
            continue
    return trades


def positions_frame(positions: Iterable[StockDemoPosition], quotes: Optional[dict[str, Quote]] = None) -> pd.DataFrame:
    qmap = {str(k).upper(): v for k, v in (quotes or {}).items()}
    rows = []
    for p in positions:
        quote = qmap.get(p.symbol)
        mark = float(quote.bid) if quote is not None and quote.bid > 0 else None
        unrealized = None
        if mark is not None:
            exit_fee = DEFAULT_COSTS.gross_cost(mark * p.quantity, "stock_sell")
            unrealized = (mark - p.entry_price) * p.quantity - p.entry_fee_ars - exit_fee
        rows.append({
            "Simbolo": p.symbol,
            "Estrategia": p.strategy,
            "Setup": p.signal_type,
            "Entrada": p.entry_price,
            "Bid actual": mark,
            "Cantidad": p.quantity,
            "Stop": p.stop_price,
            "TP": p.target_price,
            "PnL abierto": round(unrealized, 2) if unrealized is not None else None,
            "Hora": p.entry_ts,
            "Motivo": p.rationale,
        })
    return pd.DataFrame(rows)


def trades_frame(trades: Iterable[StockDemoTrade]) -> pd.DataFrame:
    return pd.DataFrame([asdict(t) for t in trades])


def _open_position(
    signal: StockSignal,
    quote: Quote,
    now: datetime,
    quantity: int,
    costs: CostModel,
    adaptive_context: Optional[AdaptiveContext] = None,
) -> StockDemoPosition:
    entry = float(quote.ask)

    # Slippage model: si el spread es ancho, asumimos fill en el mid del spread
    # (más realista que asumir fill perfecto en el ask en papeles BYMA ilíquidos).
    mid = (float(quote.ask) + float(quote.bid)) / 2.0
    if mid > 0:
        spread_pct = (quote.ask - quote.bid) / mid * 100.0
        if spread_pct > 1.0:
            # Slippage = 30% del spread por encima del ask (impacto de mercado)
            slippage = (quote.ask - quote.bid) * 0.30
            entry = float(quote.ask) + slippage

    entry_fee = costs.gross_cost(entry * quantity, "stock_buy")
    pid = f"{signal.symbol.upper()}-{signal.strategy}-{now.strftime('%Y%m%d%H%M%S')}"

    # Ajuste de stop basado en el modo adaptativo
    stop = round(float(signal.stop_price), 4)
    if adaptive_context is not None:
        atr_adjust = getattr(adaptive_context, "stop_atr_adjust", 0.0)
        if atr_adjust != 0.0 and entry > 0:
            # atr_adjust es fracción de ATR — estimamos ATR como ~1.5% del precio
            estimated_atr = entry * 0.015
            stop = round(stop + atr_adjust * estimated_atr, 4)
            stop = min(stop, entry * 0.97)   # stop nunca > -3% del entry

    return StockDemoPosition(
        id=pid,
        symbol=signal.symbol.upper(),
        strategy=signal.strategy,
        signal_type=signal.signal_type,
        entry_ts=now,
        entry_price=round(entry, 4),
        quantity=int(quantity),
        stop_price=stop,
        target_price=round(float(signal.target_price), 4),
        entry_fee_ars=round(entry_fee, 2),
        time_stop_min=signal.time_stop_min,
        max_holding_days=signal.max_holding_days,
        rationale=signal.rationale,
        score=float(signal.score),
    )


def _close_position(
    position: StockDemoPosition,
    quote: Quote,
    now: datetime,
    reason: str,
    costs: CostModel,
) -> StockDemoTrade:
    exit_price = float(quote.bid)
    sell_fee = costs.gross_cost(exit_price * position.quantity, "stock_sell")
    fees = position.entry_fee_ars + sell_fee
    gross = (exit_price - position.entry_price) * position.quantity
    pnl = gross - fees
    return StockDemoTrade(
        id=f"{position.id}-{now.strftime('%Y%m%d%H%M%S')}-{reason}",
        symbol=position.symbol,
        strategy=position.strategy,
        signal_type=position.signal_type,
        entry_ts=position.entry_ts,
        exit_ts=now,
        entry_price=round(position.entry_price, 4),
        exit_price=round(exit_price, 4),
        quantity=int(position.quantity),
        pnl_ars=round(pnl, 2),
        exit_reason=reason,
        fees_ars=round(fees, 2),
        stop_price=round(position.stop_price, 4),
        target_price=round(position.target_price, 4),
        rationale=position.rationale,
        score=float(position.score),
    )


def _exit_reason(position: StockDemoPosition, quote: Quote, now: datetime) -> Optional[str]:
    bid = float(quote.bid)
    if bid >= position.target_price and position.target_price > position.entry_price:
        return "TAKE_PROFIT"
    if bid <= position.stop_price:
        return "STOP"
    if position.strategy == "SCALP":
        # Intradía AR: una posición puede correr varias horas; el time-stop es backstop.
        minutes = float(position.time_stop_min or 150)
        if now >= _to_ba(position.entry_ts) + timedelta(minutes=minutes):
            return "TIME_STOP"
        if (now.hour, now.minute) >= (_SESSION_CLOSE_HOUR, _SESSION_CLOSE_MINUTE):
            return "SESSION_CLOSE"
    elif position.strategy == "SWING":
        days = int(position.max_holding_days or 10)
        if _business_days_between(position.entry_ts, now) >= days:
            return "TIME_STOP"
    return None


def _book_enterable(quote: Quote) -> bool:
    """El paper demo puede entrar solo si el spread es operable (no excede _MAX_SPREAD_PCT_DEMO)."""
    if not (quote.ask > 0 and quote.bid > 0 and quote.ask > quote.bid):
        return False
    # Calcular spread %
    mid = (quote.ask + quote.bid) / 2.0
    if mid > 0:
        spread_pct = (quote.ask - quote.bid) / mid * 100.0
        if spread_pct > _MAX_SPREAD_PCT_DEMO:
            return False
    return True


def _book_exitable(quote: Quote) -> bool:
    return quote.bid > 0


def _quote_age_s(quote: Quote, now: datetime) -> float:
    ts = quote.book_ts or quote.received_at or quote.timestamp
    if ts is None:
        return float("inf")
    ts = _to_ba(ts)
    return max((now - ts).total_seconds(), 0.0)


def _business_days_between(start: datetime, end: datetime) -> int:
    start_date = _to_ba(start).date()
    end_date = _to_ba(end).date()
    if end_date <= start_date:
        return 0
    days = pd.bdate_range(start_date, end_date, inclusive="right")
    return int(len(days))


def _to_ba(value: datetime) -> datetime:
    return value.replace(tzinfo=_BA) if value.tzinfo is None else value.astimezone(_BA)


def _parse_dt(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return _to_ba(value)
    dt = datetime.fromisoformat(str(value))
    return _to_ba(dt)


def _json_ready(data: dict) -> dict:
    out = {}
    for key, value in data.items():
        if isinstance(value, datetime):
            out[key] = value.astimezone(_BA).isoformat(timespec="seconds")
        elif isinstance(value, float):
            out[key] = round(float(value), 6)
        else:
            out[key] = value
    return out


def _existing_trade_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            trade_id = str(record.get("id", ""))
            if trade_id:
                ids.add(trade_id)
        except json.JSONDecodeError:
            continue
    return ids


def _recently_closed_symbols(path: Path, now: datetime, *, cooldown_s: float) -> set[str]:
    symbols: set[str] = set()
    for trade in load_stock_demo_trades(path, limit=200):
        if (now - _to_ba(trade.exit_ts)).total_seconds() <= cooldown_s:
            symbols.add(trade.symbol)
    return symbols


def _count(target: dict[str, int], key: str) -> None:
    target[key] = target.get(key, 0) + 1
