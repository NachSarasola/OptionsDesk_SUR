"""Sincronización de posiciones reales de IOL con el motor adaptativo y de señales."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from optionsdesk.backtest.adaptive import _pnl
from optionsdesk.backtest.stock_demo import StockDemoPosition
from optionsdesk.execution.iol_executor import IOLExecutor

logger = logging.getLogger(__name__)

REAL_POSITIONS_FILE = Path("data/real_positions.json")
REAL_TRADES_FILE = Path("data/real_trades.jsonl")


def load_real_positions() -> list[StockDemoPosition]:
    if not REAL_POSITIONS_FILE.exists():
        return []
    try:
        raw = json.loads(REAL_POSITIONS_FILE.read_text(encoding="utf-8") or "[]")
        return [StockDemoPosition(**p) for p in raw]
    except Exception as e:
        logger.error(f"Error cargando real_positions: {e}")
        return []


def save_real_positions(positions: list[StockDemoPosition]) -> None:
    REAL_POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    raw = [p.__dict__ for p in positions]
    REAL_POSITIONS_FILE.write_text(json.dumps(raw, default=str, indent=2), encoding="utf-8")


def append_real_trades(trades: list[dict]) -> None:
    if not trades:
        return
    REAL_TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with REAL_TRADES_FILE.open("a", encoding="utf-8") as f:
        for t in trades:
            f.write(json.dumps(t, default=str) + "\n")


def _reconstruct_signal_at(simbolo: str, ts: datetime) -> tuple[str, float]:
    """Reconstruye el setup y el score en un momento exacto mirando la cinta."""
    try:
        from optionsdesk.data.stock_tape import load_stock_tape, stock_tape_bars
        from optionsdesk.signals.stock_signals import generate_stock_signals
        from optionsdesk.data.providers.base import Quote

        # Cargar cinta del día
        df = load_stock_tape(symbols=[simbolo], now=ts)
        if df.empty:
            return ("Manual/Desconocido", 0.0)
            
        # Filtrar hasta el ts de compra
        df = df[df["timestamp"] <= ts]
        if df.empty:
            return ("Manual/Desconocido", 0.0)
            
        # Tomar la última cotización
        last_row = df.iloc[-1]
        q = Quote(
            symbol=simbolo,
            timestamp=last_row["timestamp"],
            bid=last_row["bid"],
            ask=last_row["ask"],
            last=last_row["last"],
            volume=last_row["volume"]
        )
        
        signals = generate_stock_signals({simbolo: q})
        for s in signals:
            if s.symbol == simbolo:
                return (s.signal_type, float(s.score))
    except Exception as e:
        logger.error(f"Error reconstruyendo señal para {simbolo}: {e}")
        
    return ("Manual/Desconocido", 0.0)


def sync_portfolio_state(executor: IOLExecutor) -> tuple[list[StockDemoPosition], list[dict]]:
    """Sincroniza IOL con el estado local y devuelve (posiciones_activas, nuevos_trades_cerrados)."""
    # 1. Traer data viva
    iol_pos = executor.get_positions()
    iol_hist = executor.get_history(days=30)
    
    local_pos = load_real_positions()
    local_pos_map = {p.symbol: p for p in local_pos}
    
    active_symbols = set()
    new_local_pos = []
    
    # 2. Procesar posiciones abiertas en IOL
    for p in iol_pos:
        sym = p["simbolo"]
        qty = p["cantidad"]
        if qty <= 0:
            continue
            
        active_symbols.add(sym)
        
        if sym in local_pos_map:
            # Ya la teníamos
            pos = local_pos_map[sym]
            pos.quantity = qty
            pos.entry_price = p["ppc"]
            new_local_pos.append(pos)
        else:
            # Posición nueva en IOL, buscar fecha de compra en historia
            compra_ts = datetime.now()
            for h in iol_hist:
                if h["simbolo"] == sym and h["tipo"].lower() == "compra" and h["estado"] in ("terminada", "terminadaparcial"):
                    if h.get("fechaOperada"):
                        compra_ts = datetime.fromisoformat(h["fechaOperada"])
                        break
            
            # Reconstruir razón
            signal_type, score = _reconstruct_signal_at(sym, compra_ts)
            
            new_pos = StockDemoPosition(
                id=f"REAL-{sym}-{compra_ts.strftime('%Y%m%d%H%M%S')}",
                symbol=sym,
                strategy="REAL",
                signal_type=signal_type,
                entry_ts=compra_ts,
                entry_price=p["ppc"],
                quantity=qty,
                stop_price=p["ppc"] * 0.95, # Default 5% stop
                target_price=p["ppc"] * 1.10, # Default 10% target
                entry_fee_ars=0.0,
                time_stop_min=0,
                max_holding_days=30,
                rationale="Posición real IOL sincronizada",
                score=score
            )
            new_local_pos.append(new_pos)
            
    # 3. Detectar posiciones cerradas (estaban en local, ya no están en IOL)
    closed_trades = []
    for sym, pos in local_pos_map.items():
        if sym not in active_symbols:
            # Buscar en historia el precio de venta
            venta_precio = pos.entry_price
            for h in iol_hist:
                if h["simbolo"] == sym and h["tipo"].lower() == "venta" and h["estado"] in ("terminada", "terminadaparcial"):
                    venta_precio = h.get("precioOperado", venta_precio)
                    break
                    
            pnl_bruto = (venta_precio - pos.entry_price) * pos.quantity
            trade = {
                "id": pos.id,
                "symbol": pos.symbol,
                "entry_ts": pos.entry_ts,
                "exit_ts": datetime.now(),
                "entry_price": pos.entry_price,
                "exit_price": venta_precio,
                "pnl_ars": pnl_bruto,
                "strategy": pos.strategy,
                "signal_type": pos.signal_type
            }
            closed_trades.append(trade)
            
    if closed_trades:
        append_real_trades(closed_trades)
        
    save_real_positions(new_local_pos)
    return new_local_pos, closed_trades
