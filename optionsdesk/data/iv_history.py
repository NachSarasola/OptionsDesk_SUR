"""Historial de IV ATM para activar iv_rank (vender prima cuando la IV está cara).

El recorder snapshotea precios, no IV. Acá persistimos una muestra diaria de la IV
ATM (la que ya computa el scanner) para que `volatility.iv_rank` tenga contra qué
medir el nivel actual. Acumula hacia adelante: hasta juntar suficiente historial,
iv_rank devuelve None ("sin datos") y el timing no afecta nada — honesto.

Timing del vendedor de prima: vender cuando la IV está históricamente CARA
(iv_rank alto) y pararse cuando está barata (te pagan poco por el mismo riesgo).
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path("data/iv_history.jsonl")


def record_daily_iv(
    iv: Optional[float],
    *,
    symbol: str = "GGAL",
    path: Optional[Path] = None,
    today: Optional[date] = None,
) -> bool:
    """Graba UNA muestra de IV ATM por (símbolo, día). Idempotente.

    Devuelve True si escribió, False si ya existía la del día o el IV es inválido.
    """
    if iv is None or iv <= 0:
        return False
    target = Path(path or _DEFAULT_PATH)
    day = (today or date.today()).isoformat()
    sym = symbol.upper()
    try:
        if target.exists():
            for line in target.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("date") == day and str(rec.get("symbol", "")).upper() == sym:
                    return False
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"date": day, "symbol": sym, "iv": round(float(iv), 4)}) + "\n")
        return True
    except OSError as exc:
        logger.debug("record_daily_iv fallo: %s", exc)
        return False


def load_iv_history(
    symbol: str = "GGAL",
    *,
    path: Optional[Path] = None,
    lookback: int = 120,
) -> list[float]:
    """Carga las últimas `lookback` muestras de IV del símbolo (orden cronológico)."""
    target = Path(path or _DEFAULT_PATH)
    if not target.exists():
        return []
    sym = symbol.upper()
    out: list[float] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(rec.get("symbol", "")).upper() != sym:
            continue
        iv = rec.get("iv")
        if iv is not None:
            try:
                out.append(float(iv))
            except (TypeError, ValueError):
                continue
    return out[-lookback:]


def representative_atm_iv(results) -> Optional[float]:
    """IV ATM representativa de una lista de RateResult (mediana de los ~ATM).

    Prioriza opciones con moneyness 'ATM'; si no hay, mediana de todas las IV válidas.
    """
    atm = [
        float(r.iv) for r in results
        if getattr(r, "iv", None) and r.iv > 0 and getattr(r, "moneyness", "") == "ATM"
    ]
    pool = atm or [float(r.iv) for r in results if getattr(r, "iv", None) and r.iv > 0]
    if not pool:
        return None
    pool.sort()
    mid = len(pool) // 2
    return pool[mid] if len(pool) % 2 else (pool[mid - 1] + pool[mid]) / 2.0
