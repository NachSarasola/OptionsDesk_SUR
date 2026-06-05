"""Módulo Macro y de Contexto de Mercado (MERVAL / CCL).

Provee indicadores top-down para que los bots genéticos puedan condicionar sus
entradas basándose en la tendencia general del mercado y la inflación/devaluación.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from optionsdesk.config.settings import settings
from optionsdesk.data.providers.adr import adr_daily, _CONFIRM_THRESHOLD, _FX_THRESHOLD

logger = logging.getLogger(__name__)

# Activos core que definen el proxy del Merval
_CORE_STOCKS = ["GGAL", "YPFD", "PAMP"]


def get_merval_trend(lookback_days: int = 20) -> str:
    """Calcula la tendencia general del Merval local.
    
    Promedia el rendimiento de las acciones core respecto a su SMA20.
    Retorna: "alcista", "bajista" o "lateral".
    """
    history_dir = settings.history_dir
    bullish_count = 0
    valid_stocks = 0
    
    for symbol in _CORE_STOCKS:
        path = history_dir / f"{symbol}_daily.parquet"
        if not path.exists():
            continue
            
        try:
            df = pd.read_parquet(path)
            if len(df) < lookback_days:
                continue
                
            closes = df["close"].tail(lookback_days)
            sma = closes.mean()
            last_close = float(closes.iloc[-1])
            
            if last_close > sma:
                bullish_count += 1
            valid_stocks += 1
        except Exception as e:
            logger.debug(f"Error calculando SMA de {symbol}: {e}")
            
    if valid_stocks == 0:
        return "lateral" # Failsafe
        
    bullish_ratio = bullish_count / valid_stocks
    if bullish_ratio >= 0.66:
        return "alcista"
    elif bullish_ratio <= 0.33:
        return "bajista"
    return "lateral"


def get_ccl_momentum(lookback_days: int = 5) -> str:
    """Evalúa el momentum del dólar CCL usando ADRs como proxy.
    
    Retorna: "apreciacion" (subiendo fuerte), "estable", o "depreciacion".
    """
    adr_map = settings.adr_map()
    # Usar GGAL como proxy termómetro del CCL
    if "GGAL" not in adr_map:
        return "estable"
        
    adr_ticker, _ = adr_map["GGAL"]
    try:
        df = adr_daily(adr_ticker, days=lookback_days + 1)
        if df.empty or len(df) < lookback_days + 1:
            return "estable"
            
        closes = df["close"].astype(float)
        prev_cl = float(closes.iloc[-(lookback_days + 1)])
        last_cl = float(closes.iloc[-1])
        
        momentum = (last_cl - prev_cl) / prev_cl
        
        # Si el ADR de GGAL bajó fuertemente (-0.6%) pero la local sube, es puro CCL.
        if momentum <= _FX_THRESHOLD:
            return "apreciacion" # CCL subiendo fuerte compensa la caída del ADR
        elif momentum >= _CONFIRM_THRESHOLD:
            return "depreciacion" # ADR sube fuerte, la local sube menos → CCL baja
            
        return "estable"
    except Exception as e:
        logger.debug(f"Error calculando CCL momentum: {e}")
        return "estable"
