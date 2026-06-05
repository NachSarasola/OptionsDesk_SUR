"""Señales swing long-only para acciones argentinas (netas de costos, casino-free).

El scalping intradiario de acciones fue retirado a pedido: en AR las comisiones
(~1% + derechos + IVA por lado) se comen los movimientos chicos, así que el scalp
es casino. El foco es swing de varios días, donde el movimiento amortiza el costo.

Cada señal se filtra por R/R NETO de costos (`net_rr`): si después del ida-y-vuelta
de comisiones no deja edge, es perdedora estructural y no se muestra. Configurá tu
comisión real en STOCK_COMMISSION_PCT — el default (1%) es deliberadamente
conservador y filtra casi todo scalp/movimiento chico.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Optional

import pandas as pd

from optionsdesk.config.costs import CostModel, DEFAULT_COSTS
from optionsdesk.data.providers.base import Quote

# R/R mínimo NETO de costos para emitir una señal. Por debajo, el ida-y-vuelta de
# comisiones se come el edge y la operación es perdedora estructural.
_MIN_NET_RR = 0.4
# No perseguir: un breakout ya extendido >5% sobre el pivote es entrada tardía
# (regla Minervini: comprar cerca del pivote, no arriba del envión).
_MAX_BREAKOUT_EXTENSION = 0.05


def _learned(name: str, default: float) -> float:
    """Lee un parametro aprendido por el learner; default si no hay (==hoy).

    Import perezoso + degradacion silenciosa al valor de fabrica ante cualquier fallo.
    """
    try:
        from optionsdesk.backtest.param_store import param
        return param(name, default)
    except Exception:
        return default

# ── Protección BYMA: parámetros de liquidez (overrideables por settings) ──────
from optionsdesk.config.settings import settings
# Spread bid/ask máximo como % del mid. Spreads mayores implican slippage real
# que puede convertir un trade ganador en perdedor.
_MAX_SPREAD_PCT = settings.stock_max_spread_pct
# Volumen diario mínimo (unidades/acciones). Filtra micro-caps y papeles con
# poca actividad donde una sola orden mueve el precio contra nuestra posición.
_MIN_DAILY_VOLUME = settings.stock_min_daily_volume


@dataclass(frozen=True)
class StockSignal:
    symbol: str
    strategy: str
    signal_type: str
    created_at: datetime
    entry_price: float
    stop_price: float
    target_price: float
    score: float
    rationale: str
    time_stop_min: Optional[int] = None
    max_holding_days: Optional[int] = None
    warnings: tuple[str, ...] = field(default_factory=tuple)
    cost_pct: float = 0.0   # ida-y-vuelta de costos como % de la entrada
    net_rr: float = 0.0     # recompensa/riesgo NETO de costos (el número que importa)
    # SMC enrichment (v2.4) — todos opcionales, backward-compatible
    pda_zone: Optional[str] = None           # "premium" | "discount" | "equilibrium"
    smc_structure: Optional[str] = None      # "BOS_UP" | "BOS_DOWN" | None
    nearest_liquidity: Optional[str] = None  # descripción del nivel más cercano
    adr_confluence: Optional[str] = None     # "confirmed" | "fx_driven" | "unknown"
    # Market intel externo (Fase 2 — analistas Yahoo + social StockTwits)
    analyst_note: Optional[str] = None       # ej. "Compra +12% (8 analistas)"
    social_note: Optional[str] = None        # ej. "68%🐂 45msg"
    intel_delta: float = 0.0                  # boost/penalización aplicado al score

    @property
    def risk_per_share(self) -> float:
        return max(self.entry_price - self.stop_price, 0.0)

    @property
    def rr(self) -> float:
        risk = self.risk_per_share
        return round((self.target_price - self.entry_price) / risk, 3) if risk > 0 else 0.0


def signal_grade(score: float) -> str:
    """Convierte un score 0-100 en un grade legible para el operador.

    S (≥88): confluencia SMC completa — sniper entry de libro.
    A (≥78): setup fuerte con SMC alignment.
    B (≥66): setup AT sólido, SMC parcial.
    C (≥52): señal marginal, operar con cautela.
    D (<52): ruido — no operar.
    """
    if score >= 88:
        return "S"
    if score >= 78:
        return "A"
    if score >= 66:
        return "B"
    if score >= 52:
        return "C"
    return "D"


def signal_edge_status(
    signal: "StockSignal",
    blocked_setups: set[str],
    blocked_grades: set[str],
) -> str:
    """Veredicto de edge realizado: '' si la señal es operable, motivo si es basura.

    Fuente única consultada por el demo (no abrir) y el panel (marcar/ocultar): una
    señal se descarta si su setup O su grade ya probaron edge negativo con muestra
    suficiente. Las señales nuevas (sin historial) siempre quedan operables.
    """
    setup_key = f"STOCK_{str(signal.signal_type).upper()}"
    if setup_key in blocked_setups:
        return "setup sin edge histórico"
    if signal_grade(signal.score) in blocked_grades:
        return "grade sin edge histórico"
    return ""


def scan_stock_symbol(
    symbol: str,
    quote: Quote,
    *,
    daily: Optional[pd.DataFrame] = None,
    weekly: Optional[pd.DataFrame] = None,
    now: Optional[datetime] = None,
    costs: CostModel = DEFAULT_COSTS,
    bars_1m: Optional[pd.DataFrame] = None,   # aceptado por compatibilidad; swing no lo usa
    adaptive_context=None,                     # idem
    max_spread_pct: Optional[float] = None,
    min_daily_volume: Optional[int] = None,
) -> list[StockSignal]:
    """Señales swing long-only de acciones AR, netas de costos + contexto SMC.

    Orden de prioridad de setups (mayor a menor calidad):
    1. SMC_REVERSAL: stop-hunt de SSL + reversión (sniper entry del libro).
    2. SWING_TREND_PULLBACK: pullback a Keltner en tendencia sana.
    3. SWING_BREAKOUT: breakout Minervini con volumen y trend template.

    Filtros de liquidez BYMA (aplicados antes de cualquier análisis técnico):
    - Spread máximo: _MAX_SPREAD_PCT (default 2.5%) — slippage real
    - Volumen diario mínimo: _MIN_DAILY_VOLUME (default 50,000 unidades)
    - Book válido: bid > 0, ask > bid, sizes ≥ 1 lote
    """
    current = now or datetime.now()
    if not _has_executable_book(quote):
        return []

    # Filtros de liquidez BYMA — bloquean antes de cualquier análisis
    _spread_limit = max_spread_pct if max_spread_pct is not None else _MAX_SPREAD_PCT
    _vol_limit = min_daily_volume if min_daily_volume is not None else _MIN_DAILY_VOLUME
    liquid_ok, _liquid_reason = _is_liquid_enough(quote, _spread_limit, _vol_limit)
    if not liquid_ok:
        return []

    spot = float(quote.mid) if quote.mid > 0 else float(quote.last or 0.0)
    smc_enrichment = _build_smc_enrichment(symbol, quote, daily, weekly)
    smc_ctx = _smc_context_safe(daily, weekly, spot)

    # Volumen: calcular snapshot una vez, aplicar a todas las señales del símbolo.
    # Solo daily (tape intradía volume=0 en BYMA). Si falla, degradar silenciosamente.
    vol_delta = 0.0
    vol_notes: list[str] = []
    vol_note_str = ""
    try:
        from optionsdesk.signals.volume_momentum import volume_snapshot, volume_score_delta
        rvol_lookback = int(_learned("rvol_lookback", 20))
        vol_snap = volume_snapshot(
            _normalize_daily(daily),
            lookback=rvol_lookback,
        ) if daily is not None else None
        if vol_snap is not None:
            vol_delta, vol_notes = volume_score_delta(vol_snap)
            vol_note_str = " | Vol: " + " · ".join(vol_notes) if vol_notes else ""
    except Exception:
        pass

    # Setup SMC_REVERSAL: el de mayor calidad, detectado antes que los AT clásicos.
    smcrev_signals = _scan_smc_reversal(symbol.upper(), quote, daily, weekly, current, smc_ctx)

    out: list[StockSignal] = []
    intel_d = float(smc_enrichment.get("intel_delta", 0.0)) if smc_enrichment else 0.0

    def _apply_intel_and_vol(sig: StockSignal) -> StockSignal:
        """Aplica intel_delta y vol_delta al score, fusiona enrichment SMC."""
        if smc_enrichment:
            sig = replace(sig, **{k: v for k, v in smc_enrichment.items() if k != "intel_delta"})
        total_boost = intel_d + vol_delta
        if total_boost:
            new_score = round(min(max(sig.score + total_boost, 0.0), 100.0), 1)
            sig = replace(sig, score=new_score)
        if vol_note_str:
            sig = replace(sig, rationale=sig.rationale + vol_note_str)
        return sig

    min_net_rr = _learned("stock_min_net_rr", _MIN_NET_RR)

    # Procesar señales SMC_REVERSAL (ya tienen calidad garantizada). El tree-search
    # puede apagar este setup completo via flag enable_smc_reversal.
    if _setup_enabled("SMC_REVERSAL"):
        for sig in smcrev_signals:
            priced = _apply_costs(sig, costs)
            if priced.net_rr < min_net_rr * 0.8:   # umbral más bajo: el setup es de alta calidad
                continue
            out.append(_apply_intel_and_vol(priced))

    # Setups AT clásicos filtrados y mejorados por SMC
    for sig in _scan_swings(symbol.upper(), quote, daily, weekly, current):
        if not _setup_enabled(sig.signal_type):
            continue
        if sig.risk_per_share <= 0 or sig.rr < _learned("stock_min_gross_rr", 1.2):
            continue
        priced = _apply_costs(sig, costs)
        if priced.net_rr < min_net_rr:
            continue
        if smc_ctx is not None:
            delta, block, notes = _smc_quality(priced, smc_ctx, spot)
            if block is not None:
                continue
            if delta or notes:
                priced = replace(
                    priced,
                    score=round(min(max(priced.score + delta, 0.0), 100.0), 1),
                    rationale=priced.rationale + (f" | SMC: {', '.join(notes)}" if notes else ""),
                )
        out.append(_apply_intel_and_vol(priced))

    return sorted(out, key=lambda s: (s.score, s.net_rr), reverse=True)


# Mapeo signal_type → flag de param_store. El tree-search descubre QUE combinacion
# de setups rinde mejor encendiendo/apagando cada uno.
_SETUP_FLAG = {
    "SMC_REVERSAL": "enable_smc_reversal",
    "SWING_TREND_PULLBACK": "enable_swing_pullback",
    "SWING_BREAKOUT": "enable_swing_breakout",
}


def _setup_enabled(signal_type: str) -> bool:
    """True si el setup esta activo (default True → comportamiento de hoy)."""
    flag_name = _SETUP_FLAG.get(str(signal_type).upper())
    if flag_name is None:
        return True
    try:
        from optionsdesk.backtest.param_store import flag
        return flag(flag_name, True)
    except Exception:
        return True


def _scan_smc_reversal(
    symbol: str,
    quote: "Quote",
    daily: Optional[pd.DataFrame],
    weekly: Optional[pd.DataFrame],
    now: datetime,
    smc_ctx,
) -> list["StockSignal"]:
    """Sniper entry del libro: stop-hunt de SSL + reversión alcista confirmada.

    Condiciones de activación (todas necesarias salvo indicado):
    - SSL fue barrido (low cruzó debajo del nivel) Y el precio cerró de vuelta arriba.
    - Al menos 1 confluencia adicional: OTE bullish, zona discount, BOS_UP, EMA stack.
    - Tendencia semanal alcista (filtro macro — no comprar contra la corriente).
    - Precio actual por encima del nivel del sweep (ya se revertió, no es el suelo).

    Stop: justo bajo el nivel del sweep (invalidación si lo vuelve a perder).
    Target: primer BSL no barrido por encima del spot.
    Score base 85 + up to 10 pts por confluencias adicionales.
    """
    if smc_ctx is None or daily is None or len(daily) < 20:
        return []

    sweeps = getattr(smc_ctx, "sweeps", [])
    # Sniper entry requiere sweep con VOLUMEN CONFIRMADO: en BYMA ilíquido un "barrido"
    # sin volumen es un air-pocket de book vacío, no un movimiento institucional.
    # Solo usamos los sweeps de volumen alto para el setup de más calidad.
    ssl_reversals = [
        sw for sw in sweeps
        if getattr(sw, "direction", "") == "down"
        and getattr(sw, "reversed", False)
        and getattr(sw, "volume_confirmed", False)
    ]
    if not ssl_reversals:
        return []

    entry = float(quote.ask) if quote.ask > 0 else float(quote.mid or quote.last or 0)
    if entry <= 0:
        return []

    # Confirmar tendencia semanal
    weekly_up = True
    if weekly is not None and len(weekly) >= 8:
        wclose = weekly["close"].astype(float)
        weekly_up = float(wclose.iloc[-1]) >= float(wclose.tail(10).mean()) * 0.97
    if not weekly_up:
        return []

    # El sweep más reciente determina el nivel del stop
    latest_sw  = max(ssl_reversals, key=lambda sw: getattr(sw, "bar_index", 0))
    ssl_level  = getattr(getattr(latest_sw, "level", None), "price", None)
    if ssl_level is None or ssl_level <= 0:
        return []

    # El precio actual debe haber rebotado y mantenerse por encima o en el nivel de invalidación.
    if entry < ssl_level:
        return []

    day  = _normalize_daily(daily)
    atr  = _atr(day)
    stop_mult = _learned("smc_reversal_stop_atr_mult", 2.5)
    target_mult = _learned("smc_reversal_target_atr_mult", 2.0)
    stop = max(ssl_level * 0.993, entry - stop_mult * atr)   # bajo el nivel del sweep

    # Target: primer BSL no barrido por encima del spot
    liquidity = getattr(smc_ctx, "liquidity", [])
    bsl_above = [
        lv for lv in liquidity
        if getattr(lv, "kind", "") == "BSL"
        and not getattr(lv, "swept", True)
        and getattr(lv, "price", 0) > entry * 1.01
    ]
    if bsl_above:
        target_lv   = min(bsl_above, key=lambda lv: lv.price)
        target      = target_lv.price
        target_note = f"BSL {target_lv.label} ${target_lv.price:,.0f}"
    else:
        target      = entry + target_mult * atr
        target_note = f"2×ATR ${target:,.0f}"

    risk = entry - stop
    if risk <= 0 or (target - entry) < risk:
        return []

    # Score: base 85 + bonificaciones por confluencias
    score = 85.0
    conf_notes: list[str] = []

    ote = getattr(smc_ctx, "ote", None)
    pda = getattr(smc_ctx, "pda", None)
    mm  = getattr(smc_ctx, "mm", None)

    if ote is not None and getattr(ote, "direction", "") == "bullish":
        ote_lo, ote_hi = getattr(ote, "lo", 0), getattr(ote, "hi", 0)
        if ote_lo <= entry <= ote_hi:
            score += 5.0
            conf_notes.append("entrada en OTE 0.618-0.786")

    if pda is not None and getattr(pda, "zone", "") == "discount":
        score += 3.0
        conf_notes.append("zona discount PDA")

    if getattr(smc_ctx, "bos", "") == "BOS_UP":
        score += 2.0
        conf_notes.append("BOS alcista confirmado")

    if mm is not None and getattr(mm, "stack", "") == "bullish":
        score += 2.0
        conf_notes.append("EMAs alineadas alcistas")

    label_sw = getattr(getattr(latest_sw, "level", None), "label", "SSL")
    rationale = (
        f"Stop-hunt ${ssl_level:,.0f} ({label_sw}) + reversión alcista confirmada. "
        f"Target: {target_note}. R/R {(target-entry)/risk:.1f}:1."
    )
    if conf_notes:
        rationale += f" Confluencia: {', '.join(conf_notes)}."

    return [_stock_signal(
        symbol, "SWING", "SMC_REVERSAL", now,
        entry, stop, (target - entry) / risk,
        None, 8,
        round(min(score, 100.0), 1),
        rationale,
    )]


def _build_smc_enrichment(
    symbol: str,
    quote: Quote,
    daily: Optional[pd.DataFrame],
    weekly: Optional[pd.DataFrame],
) -> dict:
    """Calcula el contexto SMC para enriquecer StockSignal. Nunca lanza."""
    try:
        from optionsdesk.config.settings import settings as _s
        if not getattr(_s, "smc_enabled", True):
            return {}

        from optionsdesk.signals.smc import analyze_smc as _analyze_smc
        spot    = float(quote.mid) if quote.mid > 0 else float(quote.last or 0)
        smc_ctx = _analyze_smc(daily=daily, weekly=weekly, spot=spot)

        enrichment: dict = {}

        # Zona PDA
        pda = getattr(smc_ctx, "pda", None)
        if pda is not None:
            enrichment["pda_zone"] = getattr(pda, "zone", None)

        # Estructura (BOS)
        bos = getattr(smc_ctx, "bos", None)
        if bos:
            enrichment["smc_structure"] = bos

        # Nivel de liquidez más cercano al spot
        liquidity = getattr(smc_ctx, "liquidity", [])
        if liquidity and spot > 0:
            nearest = min(
                (lv for lv in liquidity if not getattr(lv, "swept", True)),
                key=lambda lv: abs(getattr(lv, "price", 0) - spot),
                default=None,
            )
            if nearest is not None:
                lv_p   = getattr(nearest, "price", 0.0)
                lv_k   = getattr(nearest, "kind", "")
                lv_lbl = getattr(nearest, "label", "")
                enrichment["nearest_liquidity"] = f"{lv_k} {lv_lbl} ${lv_p:,.0f}"

        # ADR confluence (Fase 2)
        if getattr(_s, "adr_ccl_enabled", False):
            from optionsdesk.data.providers.adr import (
                adr_daily as _adr_daily,
                adr_confluence as _adr_confluence,
                bos_to_dir as _bos_to_dir,
            )
            adr_map = _s.adr_map()
            if symbol.upper() in adr_map:
                adr_ticker, _ = adr_map[symbol.upper()]
                adr_df = _adr_daily(adr_ticker, days=20)
                conf   = _adr_confluence(_bos_to_dir(bos), adr_df)
                enrichment["adr_confluence"] = conf

        # Market Intel externo: analistas Yahoo + social StockTwits.
        # Confluencia SECUNDARIA: la SMC manda, el intel confirma. Cap ±10 pts.
        try:
            from optionsdesk.data.providers.market_intel import fetch_market_intel
            intel = fetch_market_intel(symbol)
            if intel is not None:
                d = intel.score_delta()
                enrichment["intel_delta"] = d
                if intel.analyst_label:
                    up_str = f" {intel.target_upside_pct:+.0f}%" if intel.target_upside_pct is not None else ""
                    n_str  = f" ({intel.n_analysts} an.)" if intel.n_analysts else ""
                    enrichment["analyst_note"] = f"{intel.analyst_label}{up_str}{n_str}"
                if intel.social_msgs_24h:
                    bull_str = f" {intel.social_bull_pct:.0f}%🐂" if intel.social_bull_pct else ""
                    enrichment["social_note"] = f"{intel.social_msgs_24h}msg{bull_str}"
        except Exception as exc:
            import logging
            logging.getLogger(__name__).debug("market_intel(%s) fallo: %s", symbol, exc)

        return enrichment
    except Exception as exc:
        import logging
        logging.getLogger(__name__).debug("_build_smc_enrichment(%s) fallo: %s", symbol, exc)
        return {}


def _smc_context_safe(daily, weekly, spot: float):
    """SmcContext para gatear la entrada (o None si SMC off / datos insuficientes).

    Nunca lanza: si algo falla, devuelve None y el swing opera como antes.
    """
    try:
        from optionsdesk.config.settings import settings as _s
        if not getattr(_s, "smc_enabled", True):
            return None
        if daily is None or len(daily) < 10 or spot <= 0:
            return None
        from optionsdesk.signals.smc import analyze_smc
        return analyze_smc(daily=daily, weekly=weekly, spot=spot)
    except Exception:
        return None


def _smc_quality(sig: StockSignal, ctx, spot: float) -> tuple[float, Optional[str], list[str]]:
    """Calidad SMC de un swing long → (delta_score, motivo_bloqueo, notas).

    Sólo bloquea ante contra-señal fuerte (ciclo MM bajista, o comprar pegado a una
    BSL sin barrer = resistencia sin recorrido). El resto suma/resta score para que
    las entradas con confluencia (descuento + OTE + stop hunt + ciclo) ranqueen
    arriba y las flojas abajo. No mata señales: las ordena y descarta la basura.
    """
    delta = 0.0
    notes: list[str] = []
    is_breakout = sig.signal_type == "SWING_BREAKOUT"
    is_pullback = sig.signal_type == "SWING_TREND_PULLBACK"

    # Gate duro: no comprar contra un ciclo MM claramente bajista.
    mm = getattr(ctx, "mm", None)
    if mm is not None and getattr(mm, "stack", "") == "bearish":
        return 0.0, "ciclo MM bajista (no comprar contra la corriente)", []

    # Gate ADR/CCL anti-espejismo:
    # BREAKOUT fx_driven = el breakout en pesos es el dólar subiendo, no el activo.
    #   → Bloqueo DURO: comprar un espejismo CCL es el peor error en MERVAL.
    # PULLBACK fx_driven = puede tener momentum genuino (desfase horario NYSE).
    #   → Penalización suave -15 pts (solo pasa setups de calidad alta).
    # Configurable con ADR_CCL_HARD_BLOCK_BREAKOUT (default True).
    adr_conf = getattr(sig, "adr_confluence", None)
    if adr_conf == "fx_driven":
        if is_breakout:
            try:
                from optionsdesk.config.settings import settings as _s
                hard = getattr(_s, "adr_ccl_hard_block_breakout", True)
            except Exception:
                hard = True
            if hard:
                return 0.0, "espejismo CCL: breakout en pesos sin movimiento real del ADR", []
        # pullback o reversal: penalizar suave
        delta -= 15.0
        notes.append("movimiento parcialmente impulsado por CCL (-15 pts)")

    # PDA premium/discount: clave para el PULLBACK (comprar value, no el techo).
    pda = getattr(ctx, "pda", None)
    if pda is not None and is_pullback:
        if pda.zone == "discount":
            delta += 12.0; notes.append("pullback en descuento (PDA)")
        elif pda.zone == "premium":
            delta -= 12.0; notes.append("pullback en premium (caro)")

    # OTE bullish (retroceso óptimo 0.618–0.786) para el pullback.
    ote = getattr(ctx, "ote", None)
    if is_pullback and ote is not None and getattr(ote, "direction", "") == "bullish" and spot > 0:
        if ote.lo <= spot <= ote.hi:
            delta += 10.0; notes.append("entrada en OTE 0.618–0.786")

    # Barrido SSL + reversión con VOLUMEN confirmado = stop hunt institucional real.
    # En BYMA ilíquido un sweep sin volumen es air-pocket, no institucional.
    # Solo sube el score si el sweep estuvo acompañado de volumen real.
    vol_sweeps = [
        sw for sw in (getattr(ctx, "sweeps", None) or [])
        if getattr(sw, "direction", "") == "down"
        and getattr(sw, "reversed", False)
        and getattr(sw, "volume_confirmed", False)
    ]
    unconfirmed_sweeps = [
        sw for sw in (getattr(ctx, "sweeps", None) or [])
        if getattr(sw, "direction", "") == "down"
        and getattr(sw, "reversed", False)
        and not getattr(sw, "volume_confirmed", False)
    ]
    if vol_sweeps:
        delta += 8.0; notes.append("barrido SSL + volumen confirmado (stop hunt real)")
    elif unconfirmed_sweeps:
        delta += 3.0; notes.append("barrido SSL sin volumen (posible air-pocket, +3)")

    liquidity = getattr(ctx, "liquidity", None) or []
    if is_breakout:
        # Breakout que toma liquidez real (no un máximo en el vacío).
        if any(
            getattr(lv, "kind", "") == "BSL" and getattr(lv, "swept", False)
            and getattr(lv, "label", "") in ("PDH", "PWH", "EQH", "SWING_H")
            for lv in liquidity
        ):
            delta += 8.0; notes.append("breakout tomó liquidez BSL")
        # Gate duro: BSL mayor sin barrer pegada arriba = resistencia, sin recorrido.
        near_unswept = [
            lv for lv in liquidity
            if getattr(lv, "kind", "") == "BSL" and not getattr(lv, "swept", True)
            and spot > 0 and 0 < (lv.price - spot) / spot < 0.015
        ]
        if near_unswept:
            return 0.0, "BSL sin barrer <1.5% arriba (sin recorrido al objetivo)", []

    # Ciclo MM alcista confirma.
    if mm is not None and getattr(mm, "stack", "") == "bullish":
        delta += 5.0; notes.append("stack EMA alcista")

    return delta, None, notes


def _apply_costs(sig: StockSignal, costs: CostModel) -> StockSignal:
    """Adjunta el R/R neto del ida-y-vuelta de costos (comisión + derechos + IVA).

    El edge real de un swing es lo que queda DESPUÉS de pagar entrada y salida; una
    señal con rr bruto lindo pero costos altos puede ser perdedora estructural.
    """
    rt_cost = (
        costs.gross_cost(sig.entry_price, "stock_buy")
        + costs.gross_cost(sig.target_price, "stock_sell")
    )
    net_reward = (sig.target_price - sig.entry_price) - rt_cost
    net_risk = (sig.entry_price - sig.stop_price) + rt_cost
    net_rr = net_reward / net_risk if net_risk > 0 else 0.0
    cost_pct = rt_cost / sig.entry_price * 100.0 if sig.entry_price > 0 else 0.0
    return replace(sig, cost_pct=round(cost_pct, 3), net_rr=round(net_rr, 3))


def _scan_swings(
    symbol: str,
    quote: Quote,
    daily: Optional[pd.DataFrame],
    weekly: Optional[pd.DataFrame],
    now: datetime,
) -> list[StockSignal]:
    day = _normalize_daily(daily)
    week = _normalize_daily(weekly)
    if len(day) < 30:
        return []

    entry = float(quote.ask)
    if entry <= 0:
        return []

    close = day["close"].astype(float)
    high = day["high"].astype(float)
    low = day["low"].astype(float)
    volume = day["volume"].astype(float)
    sma20 = float(close.tail(20).mean())
    sma50 = float(close.tail(min(50, len(close))).mean())
    sma200 = float(close.tail(min(200, len(close))).mean())
    rsi = _rsi(close)
    atr = _atr(day)
    prior_close = float(close.iloc[-1])
    weekly_up = True
    if len(week) >= 8:
        wclose = week["close"].astype(float)
        weekly_up = float(wclose.iloc[-1]) >= float(wclose.tail(min(10, len(wclose))).mean()) * 0.98

    signals: list[StockSignal] = []

    keltner_mult = _learned("stock_pullback_keltner_atr_mult", 2.0)
    rsi_min = _learned("stock_pullback_rsi_min", 30.0)
    rsi_max = _learned("stock_pullback_rsi_max", 65.0)
    lower_keltner = sma20 - (keltner_mult * atr)
    pullback_ok = (
        weekly_up
        and sma20 >= sma50 * 0.98
        and entry <= lower_keltner * 1.04     # precio tocando/cerca de la banda inferior (zona extendida)
        and rsi_min <= rsi <= rsi_max         # RSI relajado
        and entry >= prior_close * 0.995       # no está en caída libre
    )
    if pullback_ok:
        raw_stop = min(float(low.tail(5).min()), entry - max(atr, entry * 0.018))
        signals.append(_stock_signal(
            symbol,
            "SWING",
            "SWING_TREND_PULLBACK",
            now,
            entry,
            raw_stop,
            2.0,
            None,
            10,
            65.0 + min(max(lower_keltner / entry - 1.0, 0.0) * 1000.0, 15.0),
            f"Keltner pullback: precio cerca de la banda inferior en tendencia (RSI {rsi:.0f})",
        ))

    # 2. Breakout estilo Minervini (trend template) sobre el rango de 20 ruedas.
    prior_high = float(high.iloc[-21:-1].max()) if len(high) >= 21 else float(high.iloc[:-1].max())
    avg_vol = float(volume.tail(20).mean() or 0.0)
    vol_ok = avg_vol <= 0 or float(volume.iloc[-1]) >= avg_vol * _learned("stock_breakout_volume_mult", 1.10)
    trend_template_ok = (
        entry > sma200
        and sma50 > sma200
        and sma20 > sma50
        and entry > sma50
    )
    not_extended = entry <= prior_high * (
        1.0 + _learned("stock_max_breakout_extension", _MAX_BREAKOUT_EXTENSION)
    )
    if (
        weekly_up
        and trend_template_ok
        and entry > prior_high * 1.001
        and not_extended
        and rsi < 75.0
        and vol_ok
    ):
        stop = entry - max(atr, entry * 0.018)
        signals.append(_stock_signal(
            symbol,
            "SWING",
            "SWING_BREAKOUT",
            now,
            entry,
            stop,
            2.0,
            None,
            10,
            64.0 + min((entry / prior_high - 1.0) * 900.0, 18.0),
            f"Breakout sobre rango de 20 ruedas con trend template y RSI {rsi:.0f}",
        ))
    return signals


def _stock_signal(
    symbol: str,
    strategy: str,
    signal_type: str,
    now: datetime,
    entry: float,
    stop: float,
    rr: float,
    time_stop_min: Optional[int],
    max_holding_days: Optional[int],
    score: float,
    rationale: str,
) -> StockSignal:
    risk = max(entry - stop, entry * 0.006)
    stop = min(stop, entry - entry * 0.003)
    target = entry + risk * rr
    return StockSignal(
        symbol=symbol,
        strategy=strategy,
        signal_type=signal_type,
        created_at=now,
        entry_price=round(entry, 4),
        stop_price=round(stop, 4),
        target_price=round(target, 4),
        score=round(min(max(score, 0.0), 100.0), 1),
        rationale=rationale,
        time_stop_min=time_stop_min,
        max_holding_days=max_holding_days,
    )


def _has_executable_book(quote: Quote) -> bool:
    """Check mínimo: el book tiene bid/ask válidos y calculable mid price."""
    return (
        quote.bid > 0
        and quote.ask > quote.bid
        and quote.mid > 0
    )


def _is_liquid_enough(
    quote: Quote,
    max_spread_pct: float = _MAX_SPREAD_PCT,
    min_daily_volume: int = _MIN_DAILY_VOLUME,
) -> tuple[bool, str]:
    """Filtros de liquidez específicos para BYMA.

    Retorna (True, '') si el papel es operable, o (False, motivo) si no lo es.
    El motivo se usa para poblar los contadores de skip_reasons en el dashboard.

    Controles:
    1. Spread bid/ask ≤ max_spread_pct: si es mayor, el slippage se come el edge.
    2. Volumen diario ≥ min_daily_volume: filtra micro-caps y papeles estancados.
    3. Bid size y ask size ≥ 1 lote disponible (cuando el proveedor los reporta).
    """
    # 1. Spread
    if quote.mid > 0:
        spread_pct = float(getattr(quote, "spread_pct", 0.0) or 0.0)
        if spread_pct <= 0 and quote.ask > quote.bid:
            spread_pct = (quote.ask - quote.bid) / quote.mid * 100.0
        if spread_pct > max_spread_pct:
            return False, f"spread {spread_pct:.1f}% > {max_spread_pct}% (ilíquido)"

    # 2. Volumen diario mínimo
    vol = float(getattr(quote, "volume", 0.0) or 0.0)
    if vol > 0 and vol < min_daily_volume:
        return False, f"volumen {vol:,.0f} < {min_daily_volume:,} unidades/día"

    # 3. Sizes del book (solo si el proveedor los reporta — bid_size=0 es normal en IOL)
    bid_sz = float(getattr(quote, "bid_size", 0.0) or 0.0)
    ask_sz = float(getattr(quote, "ask_size", 0.0) or 0.0)
    if bid_sz > 0 and ask_sz > 0:
        # Solo filtramos si ambos están reportados: evitamos falsos negativos
        # cuando el proveedor no informa profundidad de libro.
        if bid_sz < 1 or ask_sz < 1:
            return False, "book sin profundidad (sizes < 1 lote)"

    return True, ""


def _normalize_daily(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    columns = ["date", "open", "high", "low", "close", "volume"]
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = 0.0 if col != "date" else pd.NaT
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out[columns].dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)


def _rsi(close: pd.Series, n: int = 14) -> float:
    if len(close) < n + 1:
        return 50.0
    delta = close.diff()
    gain = delta.clip(lower=0).tail(n).mean()
    loss = (-delta.clip(upper=0)).tail(n).mean()
    if pd.isna(gain) or pd.isna(loss):
        return 50.0
    if loss <= 1e-9:
        return 100.0 if gain > 0 else 50.0
    rs = float(gain) / float(loss)
    return max(0.0, min(100.0, 100.0 - 100.0 / (1.0 + rs)))


def _atr(df: pd.DataFrame, n: int = 14) -> float:
    if df.empty:
        return 0.0
    hi = df["high"].astype(float)
    lo = df["low"].astype(float)
    cl = df["close"].astype(float)
    prev = cl.shift(1)
    tr = pd.concat([hi - lo, (hi - prev).abs(), (lo - prev).abs()], axis=1).max(axis=1)
    atr = float(tr.tail(min(n, len(tr))).median() or 0.0)
    spot = float(cl.iloc[-1] or 0.0)
    return max(atr, spot * 0.008) if spot > 0 else atr
