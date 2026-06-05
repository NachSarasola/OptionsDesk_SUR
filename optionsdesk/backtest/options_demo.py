"""Automatic paper trading loop for options (Covered Calls & Short Puts).

Monitors live quotes and evaluates entry/exit signals according to management logic.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

from optionsdesk.config.settings import settings
from optionsdesk.data.providers.base import OptionsChain
from optionsdesk.signals.management import evaluate_position, SignalType
from optionsdesk.signals.monitor import OpenPosition
from optionsdesk.signals.screener import RateResult
from optionsdesk.backtest.adaptive import AdaptiveContext

_BA = ZoneInfo("America/Argentina/Buenos_Aires")

@dataclass
class OptionsDemoTrade:
    id: str
    symbol: str
    strategy: str
    strike: float
    spot_entry: float
    premium_entry: float
    premium_exit: float
    contracts: int
    entry_ts: datetime
    exit_ts: datetime
    pnl_ars: float
    exit_reason: str
    expected_pop: Optional[float] = None       # predicción al abrir (para calibrar)
    expected_ev_ars: Optional[float] = None


def run_options_demo_tick(
    chain: OptionsChain,
    candidates: Iterable[RateResult],
    now: Optional[datetime] = None,
    adaptive_context: Optional[AdaptiveContext] = None,
    caucion_tna_pct: float = 60.0,
    block_junk: bool = True,
    allow_new: bool = True,
    lab_infinite: bool = False,
    capital: Optional[float] = None,
) -> dict:
    current = _to_ba(now or datetime.now(_BA))
    pos_file = Path("data/options_demo_positions.json")
    trade_file = Path("data/options_demo_trades.jsonl")
    if lab_infinite:
        block_junk = False

    # Gate anti-basura: no abrir estrategias con edge realizado negativo probado.
    blocked: set = set()
    if block_junk:
        try:
            from optionsdesk.performance.attribution import blocked_strategies
            blocked = blocked_strategies(trade_file.parent)
        except Exception:
            blocked = set()

    positions = _load_positions(pos_file)
    survivors = []
    new_trades = []
    
    # 1. Manage open positions
    for p in positions:
        quote = chain.options.get(p.symbol)
        mark = None
        if quote and quote.ask > 0:
            mark = float(quote.ask)
            
        signal = evaluate_position(p, current_spot=chain.spot.mid, today=current.date())
        
        # If no explicit exit signal but we have expired
        days_remaining = (p.entry_date + __timedelta(days=p.days_entry) - current.date()).days
        if days_remaining <= 0 and signal.signal_type not in (SignalType.TAKE_PROFIT, SignalType.STOP):
            # Expired!
            intrinsic = max(chain.spot.mid - p.strike, 0) if p.opt_type == "C" else max(p.strike - chain.spot.mid, 0)
            exit_px = intrinsic
            pnl_ars = (p.premium_received - exit_px) * 100.0 * p.contracts
            trade = OptionsDemoTrade(
                id=f"{p.symbol}-{current.strftime('%Y%m%d%H%M%S')}-EXPIRY",
                symbol=p.symbol, strategy=p.strategy, strike=p.strike,
                spot_entry=p.spot_entry, premium_entry=p.premium_received,
                premium_exit=exit_px, contracts=p.contracts,
                entry_ts=_to_dt(p.entry_date), exit_ts=current,
                pnl_ars=round(pnl_ars, 2), exit_reason="EXPIRY",
                expected_pop=getattr(p, "expected_pop", None),
                expected_ev_ars=getattr(p, "expected_ev_ars", None),
            )
            new_trades.append(trade)
            continue
            
        if signal.signal_type in (SignalType.TAKE_PROFIT, SignalType.STOP, SignalType.ROLL):
            if mark is not None:
                pnl_ars = (p.premium_received - mark) * 100.0 * p.contracts
                trade = OptionsDemoTrade(
                    id=f"{p.symbol}-{current.strftime('%Y%m%d%H%M%S')}-{signal.signal_type.value}",
                    symbol=p.symbol, strategy=p.strategy, strike=p.strike,
                    spot_entry=p.spot_entry, premium_entry=p.premium_received,
                    premium_exit=mark, contracts=p.contracts,
                    entry_ts=_to_dt(p.entry_date), exit_ts=current,
                    pnl_ars=round(pnl_ars, 2), exit_reason=signal.signal_type.value,
                    expected_pop=getattr(p, "expected_pop", None),
                    expected_ev_ars=getattr(p, "expected_ev_ars", None),
                )
                new_trades.append(trade)
                continue
                
        survivors.append(p)
        
    # 2. Enter new positions if space available
    base_capital = float(
        capital
        if capital is not None
        else settings.lab_infinite_capital_ars
        if lab_infinite
        else 1_800_000.0
    )
    kelly_pct = 0.05 if lab_infinite else adaptive_context.half_kelly_pct if adaptive_context else 0.05
    max_risk_ars = base_capital * kelly_pct
    
    positions = survivors
    
    # Simple limit: 2 options positions max. allow_new=False frena aperturas
    # (circuit breaker de drawdown) pero sigue gestionando las posiciones abiertas.
    max_positions = 10**9 if lab_infinite else 2
    if allow_new and len(positions) < max_positions and candidates:
        for best in candidates:
            # Check if we already have it
            if any(p.symbol == best.symbol for p in positions):
                continue
            # Saltar estrategias con edge realizado negativo probado.
            if str(getattr(best, "strategy", "")).upper() in blocked:
                continue

            lot_cost = best.net_outlay * 100.0
            if lot_cost <= 0: continue
            
            contracts = max(int(max_risk_ars / lot_cost), 1)
            if not lab_infinite:
                contracts = min(contracts, 5)
            
            opt_q = chain.options.get(best.symbol)
            if not opt_q or opt_q.bid <= 0: continue
            
            entry_px = float(opt_q.bid)
            
            new_p = OpenPosition(
                symbol=best.symbol,
                strategy=best.strategy,
                strike=best.strike,
                spot_entry=chain.spot.mid,
                premium_received=entry_px,
                net_outlay=best.net_outlay,
                iv_entry=best.iv or 0.30,
                days_entry=best.days,
                entry_date=current.date(),
                target_exit_days=max(best.days // 2, 5),
                target_capture_pct=50.0,
                caucion_tna=caucion_tna_pct,
                max_loss_mult=2.0,
                roll_dte=14,
                defend_delta=0.50,
                contracts=contracts,
                expected_pop=_candidate_pop(best),
                expected_ev_ars=getattr(best, "expected_value_ars", None),
            )
            positions.append(new_p)
            break

    _save_positions(positions, pos_file)
    if new_trades:
        _append_trades(new_trades, trade_file)
        
    return {"positions": positions, "closed": new_trades}


def _candidate_pop(best) -> Optional[float]:
    """PoP que el recommender le mostró al candidato (física o fallback por delta)."""
    try:
        from optionsdesk.signals.recommender import _probability
        return _probability(best)
    except Exception:
        return getattr(best, "prob_profit", None)


def _load_positions(path: Path) -> list[OpenPosition]:
    if not path.exists(): return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8") or "[]")
        out = []
        for d in raw:
            if "entry_date" in d and isinstance(d["entry_date"], str):
                d["entry_date"] = date.fromisoformat(d["entry_date"])
            out.append(OpenPosition(**d))
        return out
    except Exception:
        return []

def _save_positions(positions: list[OpenPosition], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for p in positions:
        d = asdict(p)
        if isinstance(d.get("entry_date"), date):
            d["entry_date"] = d["entry_date"].isoformat()
        out.append(d)
    path.write_text(json.dumps(out, indent=2))

def _append_trades(trades: list[OptionsDemoTrade], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for t in trades:
            d = asdict(t)
            d["entry_ts"] = d["entry_ts"].isoformat()
            d["exit_ts"] = d["exit_ts"].isoformat()
            fh.write(json.dumps(d) + "\n")

def _to_ba(value: datetime) -> datetime:
    return value.replace(tzinfo=_BA) if value.tzinfo is None else value.astimezone(_BA)

def _to_dt(d: date) -> datetime:
    from datetime import time
    return datetime.combine(d, time(12,0), tzinfo=_BA)

def __timedelta(days: int):
    from datetime import timedelta
    return timedelta(days=days)
