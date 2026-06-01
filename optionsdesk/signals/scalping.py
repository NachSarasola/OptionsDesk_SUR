"""Senales de scalping intradia para opciones GGAL."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time, timedelta
from statistics import median
from typing import Optional
from zoneinfo import ZoneInfo

from optionsdesk.config.costs import DEFAULT_COSTS
from optionsdesk.config.settings import settings
from optionsdesk.core.greeks_engine import OptionGreeks, compute_chain_greeks
from optionsdesk.data.providers.base import OptionsChain

_PROFILE_CFG = {
    "conservative": {
        "max_spread": 10.0,
        "min_rr": 1.6,
        "max_age_s": 60.0,
        "vol_edge_min": 0.04,
        "min_action_delta": 0.45,
        "max_action_moneyness": 8.0,
        "entry_aggression": 0.20,
        "min_score_actionable": 64.0,
        "min_pop": 0.56,
        "max_breakeven_move_pct": 0.28,
        "allow_edge_session": False,
    },
    "balanced": {
        "max_spread": 22.0,
        "spread_extreme_block": 35.0,
        "min_rr": 0.80,
        "max_age_s": 90.0,
        "vol_edge_min": 0.02,
        "min_action_delta": 0.25,
        "max_action_moneyness": 14.0,
        "entry_aggression": 0.35,
        "min_score_actionable": 48.0,
        "min_pop": 0.52,
        "max_breakeven_move_pct": 0.65,
        "time_stop_min": 20,
        "flat_before_close_min": 10,
        "allow_edge_session": False,
    },
    "aggressive": {
        "max_spread": 28.0,
        "spread_extreme_block": 45.0,
        "min_rr": 0.9,
        "max_age_s": 180.0,
        "vol_edge_min": 0.01,
        "min_action_delta": 0.25,
        "max_action_moneyness": 14.0,
        "entry_aggression": 0.55,
        "min_score_actionable": 52.0,
        "min_pop": 0.42,
        "max_breakeven_move_pct": 0.60,
        "time_stop_min": 20,
        "flat_before_close_min": 10,
        "allow_edge_session": True,
    },
}

_SYM_RE = re.compile(r"^GFG([CV])(\d+(?:[.,]\d+)?)([A-Z]+)$")

# Urgency thresholds unificados — todos los scanners usan estos, nunca hardcode
_URGENCY_HIGH_ACTION = 67   # wave_ride, momentum, rebound, breakout
_URGENCY_MED_ACTION  = 42
_URGENCY_HIGH_ALWAYS = 62   # gamma_scalp, iv_spike (rango de score más chico)
_URGENCY_MED_ALWAYS  = 35
_URGENCY_HIGH_RADAR  = 70   # gamma_radar, chain_iv (background/radar)
_URGENCY_MED_RADAR   = 45


@dataclass
class ScalpSignal:
    """Senal para scalping intradia."""

    signal_type: str
    action: str
    symbol: str
    strike: float
    option_type: str
    days_to_expiry: int
    bid: float
    ask: float
    mid: float
    spread_pct: float
    delta: float
    gamma: float
    theta: float
    vega: float
    iv: Optional[float]
    score: float
    edge_source: str
    rationale: str
    urgency: str
    plan_entry: float = 0.0
    plan_sl: float = 0.0
    plan_tp: float = 0.0
    plan_rr: float = 0.0
    confidence: str = "watchlist"
    risk_flags: list[str] = field(default_factory=list)
    max_loss_ars: float = 0.0
    suggested_lots: int = 0
    valid_until: str = ""
    playbook: str = ""
    trigger: str = ""
    expected_move: str = ""
    planned_risk_ars: float = 0.0
    fees_ars: float = 0.0
    spread_cost_ars: float = 0.0
    breakeven_move_pct: float = 0.0
    underlying_stop: float = 0.0
    underlying_target: float = 0.0
    entry_style: str = ""
    probability_of_profit: float = 0.0
    expected_value_ars: float = 0.0
    friction_ars: float = 0.0
    friction_pct: float = 0.0
    edge_r: float = 0.0
    fill_probability: float = 0.0
    cost_to_target_pct: float = 0.0
    microstructure_bias: float = 0.0
    session_phase: str = ""
    time_stop_min: int = 20
    blocking_flags: list[str] = field(default_factory=list)
    warning_flags: list[str] = field(default_factory=list)


def _round_price(value: float) -> float:
    return round(max(float(value or 0.0), 0.0), 2)


def _limit_entry_price(g: OptionGreeks, action: str, cfg: dict[str, float]) -> tuple[float, str]:
    """Precio limite razonable para operar manualmente sin perseguir la punta."""
    mid = g.mid if g.mid > 0 else (g.bid + g.ask) / 2.0
    half_spread = max((g.ask - g.bid) / 2.0, 0.0)
    aggression = min(max(float(cfg.get("entry_aggression", 0.35)), 0.0), 1.0)

    if action.startswith("BUY_"):
        entry = min(g.ask, mid + half_spread * aggression)
        return _round_price(entry), "limite entre mid y ask"

    entry = max(g.bid, mid - half_spread * aggression)
    return _round_price(entry), "limite entre bid y mid"


def _option_cost(price: float, side: str) -> float:
    return DEFAULT_COSTS.gross_cost(max(price, 0.0) * 100.0, side)


def _sanitize_long_levels(
    entry: float,
    sl_opt: float,
    tp_opt: float,
    spread_abs: float,
    playbook: str,
) -> tuple[float, float]:
    min_risk = max(entry * 0.10, spread_abs * 1.20, 0.01)
    min_reward = max(entry * 0.15, spread_abs * 2.00, min_risk * 1.20, 0.01)
    tp_cap_mult = {
        "Seguir ola": 1.80,
        "Pricear rebote": 1.60,
        "Voladura/breakout": 2.20,
    }.get(playbook, 1.70)

    sl_cap = max(entry - min_risk, 0.01)
    sl_floor = max(entry * 0.85, 0.01)
    sl_opt = min(sl_opt, sl_cap)
    sl_opt = max(sl_opt, min(sl_floor, sl_cap))

    tp_floor = entry + min_reward
    tp_cap = max(entry * tp_cap_mult, tp_floor)
    tp_opt = max(tp_opt, tp_floor)
    tp_opt = min(tp_opt, tp_cap)
    return sl_opt, tp_opt


def _trade_plan_detail(
    g: OptionGreeks,
    action: str,
    snap,
    playbook: str = "",
    profile_cfg: Optional[dict[str, float]] = None,
) -> dict[str, float | str]:
    """Plan de trade neto de costos para uso real manual."""
    cfg = profile_cfg or _PROFILE_CFG["balanced"]
    entry, entry_style = _limit_entry_price(g, action, cfg)
    if entry <= 0:
        return {
            "entry": 0.0,
            "sl": 0.0,
            "tp": 0.0,
            "rr": 0.0,
            "planned_risk_ars": 0.0,
            "max_loss_ars": 0.0,
            "fees_ars": 0.0,
            "spread_cost_ars": 0.0,
            "breakeven_move_pct": 0.0,
            "underlying_stop": 0.0,
            "underlying_target": 0.0,
            "entry_style": entry_style,
        }

    is_call = "CALL" in action
    is_buy = "BUY" in action
    spot = g.spot
    spread_abs = max(g.ask - g.bid, 0.0)

    def _project(target_spot: float) -> float:
        ds = target_spot - spot
        dp = g.delta * ds + 0.5 * g.gamma * (ds ** 2)
        return max(0.01, entry + dp)

    if is_buy and snap is not None:
        atr_pct = max(_snap_float(snap, "atr_pct", 1.2), 0.6) / 100.0
        atr_move = spot * min(max(atr_pct, 0.008), 0.045)
        side = 1.0 if is_call else -1.0

        if playbook == "Seguir ola":
            stop_move = atr_move * 0.20
            target_move = atr_move * 0.40
        elif playbook == "Pricear rebote":
            stop_move = atr_move * 0.15
            target_move = atr_move * 0.35
        elif playbook == "Voladura/breakout":
            stop_move = atr_move * 0.25
            target_move = atr_move * 0.50
        else:
            stop_move = atr_move * 0.20
            target_move = atr_move * 0.40

        sl_u = spot - side * stop_move
        tp_u = spot + side * target_move
        ob = getattr(snap, "order_block", None)
        fvg = getattr(snap, "nearest_fvg", None)
        if ob is not None:
            if is_call and getattr(ob, "type", "") == "bullish":
                sl_u = min(sl_u, max(float(ob.low), spot * 0.965))
            elif not is_call and getattr(ob, "type", "") == "bearish":
                sl_u = max(sl_u, min(float(ob.high), spot * 1.035))
        if fvg is not None:
            if is_call and getattr(fvg, "type", "") == "bearish" and float(fvg.low) > spot:
                tp_u = max(tp_u, float(fvg.low))
            elif not is_call and getattr(fvg, "type", "") == "bullish" and float(fvg.high) < spot:
                tp_u = min(tp_u, float(fvg.high))

        sl_opt = max(_project(sl_u), entry * 0.35)
        tp_opt = max(_project(tp_u), entry * 1.10)
        sl_opt, tp_opt = _sanitize_long_levels(entry, sl_opt, tp_opt, spread_abs, playbook)
    else:
        if snap is None or not hasattr(snap, "order_block"):
            if is_buy:
                sl_u = spot * (0.995 if is_call else 1.005)
                tp_u = spot * (1.008 if is_call else 0.992)
                sl_opt = entry * 0.85
                tp_opt = entry * 1.35
                sl_opt, tp_opt = _sanitize_long_levels(entry, sl_opt, tp_opt, spread_abs, playbook)
            else:
                sl_u = spot * (1.01 if is_call else 0.99)
                tp_u = spot * (0.98 if is_call else 1.02)
                sl_opt = entry * 1.15
                tp_opt = entry * 0.75
        else:
            ob = getattr(snap, "order_block", None)
            fvg = getattr(snap, "nearest_fvg", None)

            if is_buy and is_call:
                sl_u = ob.low if ob and ob.type == "bullish" else spot * 0.995
                tp_u = fvg.low if fvg and fvg.type == "bearish" else spot * 1.008
                sl_u = max(sl_u, spot * 0.99)
                tp_u = max(tp_u, spot * 1.004)
            elif is_buy and not is_call:
                sl_u = ob.high if ob and ob.type == "bearish" else spot * 1.005
                tp_u = fvg.high if fvg and fvg.type == "bullish" else spot * 0.992
                sl_u = min(sl_u, spot * 1.01)
                tp_u = min(tp_u, spot * 0.996)
            elif not is_buy and is_call:
                sl_u = ob.high if ob and ob.type == "bearish" else spot * 1.01
                tp_u = spot * 0.98
                sl_u = min(sl_u, spot * 1.015)
            else:
                sl_u = ob.low if ob and ob.type == "bullish" else spot * 0.99
                tp_u = spot * 1.02
                sl_u = max(sl_u, spot * 0.985)

            sl_opt = _project(sl_u)
            tp_opt = _project(tp_u)
            if is_buy:
                sl_opt, tp_opt = _sanitize_long_levels(entry, sl_opt, tp_opt, spread_abs, playbook)
            else:
                sl_opt = min(max(sl_opt, entry * 1.08), entry * 1.50)
                tp_opt = max(min(tp_opt, entry * 0.88), entry * 0.20)

    if is_buy:
        entry_fee = _option_cost(entry, "option_buy")
        stop_fee = _option_cost(sl_opt, "option_sell")
        target_fee = _option_cost(tp_opt, "option_sell")
        planned_risk = max((entry - sl_opt) * 100.0 + entry_fee + stop_fee, 0.0)
        reward = (tp_opt - entry) * 100.0 - entry_fee - target_fee
        max_loss = entry * 100.0 + entry_fee
        spread_cost = max(entry - g.mid, 0.0) * 100.0
        round_trip_at_entry = entry_fee + _option_cost(entry, "option_sell")
        breakeven_move_pct = (
            ((spread_cost + round_trip_at_entry) / 100.0) / max(abs(g.delta), 0.05) / spot * 100.0
            if spot > 0
            else 0.0
        )
        fees = entry_fee + target_fee
    else:
        entry_fee = _option_cost(entry, "option_sell")
        stop_fee = _option_cost(sl_opt, "option_buy")
        target_fee = _option_cost(tp_opt, "option_buy")
        planned_risk = max((sl_opt - entry) * 100.0 + entry_fee + stop_fee, 0.0)
        reward = (entry - tp_opt) * 100.0 - entry_fee - target_fee
        max_loss = planned_risk
        spread_cost = max(g.mid - entry, 0.0) * 100.0
        breakeven_move_pct = 0.0
        fees = entry_fee + target_fee

    rr = reward / planned_risk if planned_risk > 0 else 0.0
    return {
        "entry": _round_price(entry),
        "sl": _round_price(sl_opt),
        "tp": _round_price(tp_opt),
        "rr": round(max(rr, 0.0), 2),
        "planned_risk_ars": round(planned_risk, 2),
        "max_loss_ars": round(max_loss, 2),
        "fees_ars": round(fees, 2),
        "spread_cost_ars": round(spread_cost, 2),
        "breakeven_move_pct": round(breakeven_move_pct, 2),
        "underlying_stop": round(float(sl_u), 2),
        "underlying_target": round(float(tp_u), 2),
        "entry_style": entry_style,
    }


def calculate_trade_plan(
    g: OptionGreeks,
    action: str,
    snap,
    playbook: str = "",
    profile_cfg: Optional[dict[str, float]] = None,
) -> tuple[float, float, float, float]:
    """Calcula entry/SL/TP y RR esperado."""
    detail = _trade_plan_detail(g, action, snap, playbook=playbook, profile_cfg=profile_cfg)
    return (
        float(detail["entry"]),
        float(detail["sl"]),
        float(detail["tp"]),
        float(detail["rr"]),
    )


def _risk_flags_for_signal(
    g: OptionGreeks,
    action: str,
    rr: float,
    capital: Optional[float] = None,
    positions: Optional[list] = None,
    profile_cfg: Optional[dict[str, float]] = None,
) -> list[str]:
    cfg = profile_cfg or _PROFILE_CFG["balanced"]
    max_spread = float(cfg["max_spread"])
    min_rr = float(cfg["min_rr"])
    max_age_s = float(cfg["max_age_s"])

    flags: list[str] = []
    if cfg.get("require_capital") and (capital is None or capital <= 0):
        flags.append("capital_no_cargado")
    if not g.is_tradeable:
        flags.append(g.liquidity_reason or "no_tradeable")
    if rr < min_rr:
        flags.append(f"rr<{min_rr:.1f}")
    if g.quote_age_s > max_age_s and not any("stale" in flag for flag in flags):
        flags.append("quote_stale")
    if g.spread_pct > max_spread:
        flags.append("spread_alto")
    if action.startswith("BUY_"):
        min_delta = float(cfg.get("min_action_delta", 0.0))
        max_moneyness = float(cfg.get("max_action_moneyness", 99.0))
        if abs(g.delta) < min_delta:
            flags.append("delta_baja")
        if abs(g.moneyness_pct) > max_moneyness:
            flags.append("strike_lejano")

    if action.startswith("SELL_"):
        flags.append("venta_iv_solo_contexto")
        if action == "SELL_CALL":
            covered = any(
                ("COVERED_CALL" in str(getattr(p, "strategy", "")).upper())
                for p in (positions or [])
            )
            if not covered:
                flags.append("short_call_no_cubierto")
        elif action == "SELL_PUT":
            # Cash-secured simple: requiere capital para asignacion (K*100 - prima*100)
            needed = max(g.strike * 100.0 - g.mid * 100.0, 0.0)
            if not capital or capital < needed:
                flags.append("short_put_no_cash_secured")

    if cfg.get("require_capital") and capital and capital > 0:
        max_open = int(cfg.get("max_open_scalps", settings.scalping_max_open_positions))
        open_scalps = [
            p for p in (positions or [])
            if "SCALP" in str(getattr(p, "strategy", "")).upper()
        ]
        if len(open_scalps) >= max_open:
            flags.append(f"max_scalps_abiertos>={max_open}")

        open_risk = _open_scalp_risk_ars(open_scalps)
        max_total_risk = capital * (float(cfg.get("max_total_risk_pct", settings.scalping_max_total_risk_pct)) / 100.0)
        if open_risk >= max_total_risk:
            flags.append("riesgo_total_scalping_agotado")
    return flags


def _risk_to_lots(capital: Optional[float], risk_per_lot_ars: float) -> int:
    if not capital or capital <= 0 or risk_per_lot_ars <= 0:
        return 0
    risk_budget = capital * (settings.scalping_risk_per_trade_pct / 100.0)
    return max(int(risk_budget // risk_per_lot_ars), 0)


def _open_scalp_risk_ars(positions: Optional[list]) -> float:
    total = 0.0
    for p in positions or []:
        strategy = str(getattr(p, "strategy", "")).upper()
        if "SCALP" not in strategy:
            continue
        contracts = int(getattr(p, "contracts", 1) or 1)
        net_outlay = float(getattr(p, "net_outlay", 0.0) or 0.0)
        total += abs(net_outlay) * contracts * 100.0
    return total


def _session_flags(cfg: dict[str, float]) -> list[str]:
    if not cfg.get("enforce_session"):
        return []

    now = datetime.now(ZoneInfo("America/Argentina/Buenos_Aires"))
    if now.weekday() >= 5:
        return ["fuera_horario_byma"]

    open_t = dt_time(11, 0)
    start_ok = dt_time(11, 5)
    end_ok = dt_time(16, 55)
    close_t = dt_time(17, 0)
    t = now.time()
    if t < open_t or t >= close_t:
        return ["fuera_horario_byma"]
    if not bool(cfg.get("allow_edge_session")) and (t < start_ok or t >= end_ok):
        return ["apertura_cierre_byma"]
    return []


def _add_watch_flags(sig: ScalpSignal, *flags: str) -> ScalpSignal:
    """Agrega motivos de watchlist sin duplicarlos."""
    for flag in flags:
        if flag and flag not in sig.risk_flags:
            sig.risk_flags.append(flag)
        if flag and flag not in sig.blocking_flags:
            sig.blocking_flags.append(flag)
    if flags:
        sig.confidence = "watchlist"
    return sig


def _is_quoteable(g: OptionGreeks) -> bool:
    # Exige bid < ask para evitar spreads invertidos que corrompen scores
    return g.days > 0 and g.bid >= 0 and g.ask > g.bid and g.mid > 0


def _direction_bias(snap) -> str:
    if snap is None:
        return "NEUTRAL"
    trend = str(getattr(snap, "trend", "LATERAL") or "LATERAL").upper()
    mom = float(getattr(snap, "momentum", 0.0) or 0.0)
    if trend == "ALCISTA" and mom >= 0:
        return "BULL"
    if trend == "BAJISTA" and mom <= 0:
        return "BEAR"
    if mom >= settings.scalping_momentum_threshold:
        return "BULL"
    if mom <= -settings.scalping_momentum_threshold:
        return "BEAR"
    return "NEUTRAL"


def _context_watch_flags(g: OptionGreeks, snap, realized_vol: Optional[float] = None) -> list[str]:
    flags: list[str] = []
    bias = _direction_bias(snap)
    if snap is None:
        flags.append("falta_contexto_direccional")
    elif bias == "NEUTRAL":
        flags.append("sin_trigger_direccional")
    elif g.option_type == "C" and bias != "BULL":
        flags.append("sesgo_no_alcista")
    elif g.option_type == "P" and bias != "BEAR":
        flags.append("sesgo_no_bajista")
    if realized_vol is None or realized_vol <= 0:
        flags.append("falta_vol_realizada")
    return flags


def _snap_float(snap, name: str, default: float = 0.0) -> float:
    if snap is None:
        return default
    return float(getattr(snap, name, default) or default)


def _snap_text(snap, name: str, default: str = "") -> str:
    if snap is None:
        return default
    return str(getattr(snap, name, default) or default)


def _structure_confirms(snap, side: str) -> bool:
    if snap is None:
        return False
    bos = _snap_text(snap, "bos").upper()
    choch = _snap_text(snap, "choch").upper()
    if side == "BULL":
        return bos == "BOS_UP" or choch == "CHOCH_UP"
    if side == "BEAR":
        return bos == "BOS_DOWN" or choch == "CHOCH_DOWN"
    return False


def _is_exhausted(snap, side: str) -> bool:
    rsi = _snap_float(snap, "rsi", 50.0)
    if side == "BULL":
        return rsi >= 76.0
    if side == "BEAR":
        return rsi <= 24.0
    return False


def _side_action(side: str) -> tuple[str, str]:
    if side == "BEAR":
        return "P", "BUY_PUT"
    return "C", "BUY_CALL"


def _movement_pool(
    greeks: dict[str, OptionGreeks],
    option_type: str,
    max_days: int = 45,
    min_delta: float = 0.22,
    max_abs_moneyness: float = 14.0,
) -> list[OptionGreeks]:
    return [
        g
        for g in greeks.values()
        if _is_quoteable(g)
        and g.option_type == option_type
        and 1 <= g.days <= max_days
        and abs(g.delta) >= min_delta
        and abs(g.moneyness_pct) <= max_abs_moneyness
    ]


def _option_movement_score(g: OptionGreeks, cfg: dict[str, float]) -> float:
    gamma_score = min(max(g.gamma, 0.0) * 18000.0, 26.0)
    delta_score = max(0.0, 20.0 - abs(abs(g.delta) - 0.48) * 42.0)
    atm_score = max(0.0, 16.0 - abs(g.moneyness_pct) * 1.35)
    dte_score = max(0.0, 14.0 - abs(g.days - 12.0) * 0.45)
    return min(gamma_score + delta_score + atm_score + dte_score + _liquidity_score(g, cfg), 60.0)


def _near_zone(spot: float, low: float, high: float, tolerance_pct: float = 1.8) -> bool:
    if spot <= 0 or high <= 0 or low <= 0:
        return False
    lo = min(low, high)
    hi = max(low, high)
    pad = spot * tolerance_pct / 100.0
    return (lo - pad) <= spot <= (hi + pad)


def _liquidity_score(g: OptionGreeks, cfg: dict[str, float]) -> float:
    max_spread = max(float(cfg["max_spread"]), 1.0)
    spread_score = 18.0 - (g.spread_pct / max_spread) * 25.0
    volume_score = min(math.log10(max(g.volume, 0.0) + 1.0) * 5.0, 14.0)
    size_score = min(math.log10(max(g.bid_size + g.ask_size, 0.0) + 1.0) * 3.0, 8.0)
    return spread_score + volume_score + size_score


def _signal_rank(sig: ScalpSignal) -> tuple[int, float, float, float, float, float, float]:
    """Ranking PoP-first: operar primero la mayor probabilidad viable."""
    risk = max(sig.planned_risk_ars, 0.01)
    ev_per_risk = sig.expected_value_ars / risk
    return (
        1 if sig.confidence == "actionable" else 0,
        sig.probability_of_profit,
        ev_per_risk,
        -sig.cost_to_target_pct,
        sig.fill_probability,
        sig.score,
        sig.plan_rr,
        -sig.spread_pct,
    )


def _estimate_long_pop(sig: ScalpSignal, g: OptionGreeks, snap) -> float:
    """Estimacion conservadora de PoP intradia para compras de calls/puts."""
    if not sig.action.startswith("BUY_") or sig.underlying_stop <= 0 or sig.underlying_target <= 0:
        return 0.0

    spot = float(getattr(g, "spot", 0.0) or 0.0)
    if spot <= 0:
        # Fallback: inferir desde stop/target. Es solo para no romper tests con mocks.
        spot = (sig.underlying_stop + sig.underlying_target) / 2.0

    stop_dist = abs(spot - sig.underlying_stop)
    target_dist = abs(sig.underlying_target - spot)
    if stop_dist <= 0 or target_dist <= 0:
        base = 0.45
    else:
        base = stop_dist / (stop_dist + target_dist)

    score_boost = max(min((sig.score - 52.0) * 0.0035, 0.09), -0.07)
    structure_boost = 0.0
    side = "BULL" if sig.action == "BUY_CALL" else "BEAR"
    mom = _snap_float(snap, "momentum", 0.0)
    atr_pct = _snap_float(snap, "atr_pct", 0.0)
    if _structure_confirms(snap, side):
        structure_boost += 0.05
    if bool(getattr(snap, "vol_confirms", False)):
        structure_boost += 0.04
    if (side == "BULL" and mom > 0) or (side == "BEAR" and mom < 0):
        structure_boost += min(abs(mom) * 0.006, 0.05)
    if sig.playbook == "Pricear rebote" and getattr(snap, "order_block", None) is not None:
        structure_boost += 0.06
    if sig.playbook == "Seguir ola":
        structure_boost += 0.03
    if sig.playbook == "Voladura/breakout" and bool(getattr(snap, "vol_confirms", False)):
        structure_boost += 0.03

    target_move_pct = abs(sig.underlying_target - spot) / spot * 100.0 if spot > 0 else 0.0
    atr_penalty = 0.0
    if atr_pct > 0 and target_move_pct > max(atr_pct * 0.75, 0.15):
        atr_penalty = min((target_move_pct / max(atr_pct, 0.01) - 0.75) * 0.03, 0.08)

    spread_penalty = min(max(g.spread_pct, 0.0) / 100.0 * 0.18, 0.08)
    quote_age = max(float(getattr(g, "quote_age_s", 0.0) or 0.0), 0.0)
    age_penalty = min(quote_age / 900.0, 0.06)
    friction_penalty = min(max(sig.breakeven_move_pct, 0.0) * 0.020, 0.08)
    theta_penalty = 0.03 if sig.days_to_expiry <= 2 else 0.0

    pop = base + score_boost + structure_boost - friction_penalty - theta_penalty - spread_penalty - age_penalty - atr_penalty
    return round(max(0.05, min(0.86, pop)), 4)


def _estimate_trade_ev(sig: ScalpSignal) -> tuple[float, float, float]:
    """EV, friccion y friccion porcentual por lote, neto de costos modelados."""
    friction = max(sig.fees_ars + sig.spread_cost_ars, 0.0)
    notional = max(sig.plan_entry * 100.0, 0.01)
    friction_pct = friction / notional * 100.0

    if sig.action.startswith("BUY_"):
        reward = max((sig.plan_tp - sig.plan_entry) * 100.0 - sig.fees_ars - sig.spread_cost_ars, 0.0)
        risk = max(sig.planned_risk_ars, 0.0)
    else:
        reward = max((sig.plan_entry - sig.plan_tp) * 100.0 - sig.fees_ars - sig.spread_cost_ars, 0.0)
        risk = max(sig.planned_risk_ars, 0.0)

    ev = sig.probability_of_profit * reward - (1.0 - sig.probability_of_profit) * risk
    return round(ev, 2), round(friction, 2), round(friction_pct, 2)


def _microstructure_bias(g: OptionGreeks) -> float:
    total = max(float(g.bid_size + g.ask_size), 0.0)
    if total <= 0:
        return 0.0
    return round((float(g.bid_size) - float(g.ask_size)) / total, 4)


def _fill_probability(g: OptionGreeks, sig: ScalpSignal, cfg: dict[str, float]) -> float:
    max_spread = max(float(cfg.get("max_spread", 22.0)), 1.0)
    spread_f = max(0.0, 1.0 - min(g.spread_pct / max_spread, 1.4) * 0.45)
    quote_age = max(float(getattr(g, "quote_age_s", 0.0) or 0.0), 0.0)
    age_f = max(0.0, 1.0 - min(quote_age / max(float(cfg.get("max_age_s", 90.0)), 1.0), 1.5) * 0.35)
    size_total = max(float(g.bid_size + g.ask_size), 0.0)
    size_f = min(math.log10(size_total + 1.0) / 2.0, 0.15) if size_total > 0 else 0.0
    volume_f = min(math.log10(max(g.volume, 0.0) + 1.0) / 5.0, 0.12)

    if sig.action.startswith("BUY_") and g.ask > g.bid:
        pos = (sig.plan_entry - g.bid) / max(g.ask - g.bid, 0.01)
    elif g.ask > g.bid:
        pos = (g.ask - sig.plan_entry) / max(g.ask - g.bid, 0.01)
    else:
        pos = 0.5
    entry_f = min(max(pos, 0.0), 1.0) * 0.18

    return round(max(0.05, min(0.98, 0.38 + spread_f * 0.25 + age_f * 0.10 + size_f + volume_f + entry_f)), 4)


def _cost_to_target_pct(sig: ScalpSignal) -> float:
    if sig.action.startswith("BUY_"):
        gross_target = max((sig.plan_tp - sig.plan_entry) * 100.0, 0.01)
    else:
        gross_target = max((sig.plan_entry - sig.plan_tp) * 100.0, 0.01)
    return round(max(sig.friction_ars, 0.0) / gross_target * 100.0, 2)


def _current_session_phase() -> str:
    now = datetime.now(ZoneInfo("America/Argentina/Buenos_Aires"))
    if now.weekday() >= 5:
        return "closed"
    t = now.time()
    if t < dt_time(11, 0) or t >= dt_time(17, 0):
        return "closed"
    if t < dt_time(11, 20):
        return "open"
    if t >= dt_time(16, 45):
        return "close"
    return "regular"


def _is_hard_flag(flag: str) -> bool:
    hard_prefixes = (
        "capital_",
        "riesgo_total",
        "max_scalps",
        "spread_extremo",
        "ev_no_positivo",
        "quote stale",
        "sin puntas",
        "mid invalido",
        "no_tradeable",
        "volumen bajo",
        "fuera_horario",
        "apertura_cierre",
        "short_call",
        "short_put",
        "delta_baja",
        "strike_lejano",
        "stop_dentro_spread",
    )
    return flag.startswith(hard_prefixes)


def _split_flags(flags: list[str]) -> tuple[list[str], list[str]]:
    blocking: list[str] = []
    warnings: list[str] = []
    for flag in flags:
        target = blocking if _is_hard_flag(flag) else warnings
        if flag not in target:
            target.append(flag)
    return blocking, warnings


def _finalize_as_watchlist(
    sig: ScalpSignal,
    g: OptionGreeks,
    snap,
    capital: Optional[float],
    positions: Optional[list],
    profile_cfg: dict[str, float],
    flags: list[str],
) -> ScalpSignal:
    sig = _finalize_signal(
        sig,
        g,
        snap=snap,
        capital=capital,
        positions=positions,
        profile_cfg=profile_cfg,
    )
    return _add_watch_flags(sig, *flags)


def _finalize_signal(
    sig: ScalpSignal,
    g: OptionGreeks,
    snap,
    capital: Optional[float],
    positions: Optional[list],
    profile_cfg: Optional[dict[str, float]] = None,
) -> ScalpSignal:
    cfg = profile_cfg or _PROFILE_CFG["balanced"]
    plan = _trade_plan_detail(g, sig.action, snap, playbook=sig.playbook, profile_cfg=cfg)
    sig.plan_entry = float(plan["entry"])
    sig.plan_sl = float(plan["sl"])
    sig.plan_tp = float(plan["tp"])
    sig.plan_rr = float(plan["rr"])
    sig.planned_risk_ars = float(plan["planned_risk_ars"])
    sig.max_loss_ars = float(plan["max_loss_ars"])
    sig.fees_ars = float(plan["fees_ars"])
    sig.spread_cost_ars = float(plan["spread_cost_ars"])
    sig.breakeven_move_pct = float(plan["breakeven_move_pct"])
    sig.underlying_stop = float(plan["underlying_stop"])
    sig.underlying_target = float(plan["underlying_target"])
    sig.entry_style = str(plan["entry_style"])
    sig.probability_of_profit = _estimate_long_pop(sig, g, snap)
    sig.expected_value_ars, sig.friction_ars, sig.friction_pct = _estimate_trade_ev(sig)
    sig.edge_r = round(sig.expected_value_ars / max(sig.planned_risk_ars, 0.01), 4)
    sig.fill_probability = _fill_probability(g, sig, cfg)
    sig.cost_to_target_pct = _cost_to_target_pct(sig)
    sig.microstructure_bias = _microstructure_bias(g)
    sig.session_phase = _current_session_phase()
    sig.time_stop_min = int(cfg.get("time_stop_min", 20))

    flags = _risk_flags_for_signal(
        g,
        sig.action,
        sig.plan_rr,
        capital=capital,
        positions=positions,
        profile_cfg=cfg,
    )
    spread_extreme = float(cfg.get("spread_extreme_block", 35.0))
    if g.spread_pct > spread_extreme:
        flags.append(f"spread_extremo>{spread_extreme:.0f}%")
    spread_abs = max(g.ask - g.bid, 0.0)
    if sig.action.startswith("BUY_"):
        if (sig.plan_entry - sig.plan_sl) <= spread_abs * 1.05:
            flags.append("stop_dentro_spread")
        atr_pct = _snap_float(snap, "atr_pct", 0.0)
        if atr_pct > 0 and sig.breakeven_move_pct > max(0.6, atr_pct * 0.40):
            flags.append("costo_alto_vs_atr")
        max_be_move = float(cfg.get("max_breakeven_move_pct", 0.40))
        if sig.breakeven_move_pct > max_be_move:
            flags.append(f"be_move>{max_be_move:.2f}%")
        min_pop = float(cfg.get("min_pop", 0.46))
        if sig.probability_of_profit < min_pop:
            flags.append(f"pop<{min_pop * 100:.0f}%")
        if sig.expected_value_ars <= 0:
            flags.append("ev_no_positivo")
    min_score = float(cfg.get("min_score_actionable", 58.0))
    if sig.score < min_score:
        flags.append(f"score<{min_score:.0f}")
    flags.extend(_session_flags(cfg))

    sig.suggested_lots = _risk_to_lots(capital, sig.planned_risk_ars)
    if capital is not None and capital > 0 and sig.suggested_lots <= 0:
        flags.append("capital_insuficiente")
    if cfg.get("require_capital") and (capital is None or capital <= 0):
        sig.suggested_lots = 0
    if cfg.get("require_capital") and capital and capital > 0 and sig.suggested_lots > 0:
        open_risk = _open_scalp_risk_ars(positions)
        max_total_risk = capital * (float(cfg.get("max_total_risk_pct", settings.scalping_max_total_risk_pct)) / 100.0)
        remaining_risk = max(max_total_risk - open_risk, 0.0)
        sig.suggested_lots = min(sig.suggested_lots, int(remaining_risk // max(sig.planned_risk_ars, 0.01)))
        if sig.suggested_lots <= 0:
            flags.append("riesgo_total_scalping_agotado")

    max_age_s = float(cfg.get("max_age_s", 90.0))
    quote_age_s = max(float(getattr(g, "quote_age_s", 0.0) or 0.0), 0.0)
    ttl_s = max(30.0, min(180.0, max_age_s - quote_age_s))
    sig.valid_until = (datetime.now() + timedelta(seconds=ttl_s)).isoformat(timespec="seconds")
    deduped_flags = list(dict.fromkeys(flags))
    blocking, warnings = _split_flags(deduped_flags)
    sig.blocking_flags = blocking
    sig.warning_flags = warnings
    sig.risk_flags = blocking + warnings
    sig.confidence = "actionable" if not blocking else "watchlist"
    return sig


def scan_momentum(
    greeks: dict[str, OptionGreeks],
    snap,
    momentum_threshold: float = 1.0,
    capital: Optional[float] = None,
    positions: Optional[list] = None,
    profile_cfg: Optional[dict[str, float]] = None,
) -> list[ScalpSignal]:
    if snap is None:
        return []

    mom = float(getattr(snap, "momentum", 0.0) or 0.0)
    trend = str(getattr(snap, "trend", "LATERAL") or "LATERAL")
    if abs(mom) < momentum_threshold:
        return []

    is_bullish = trend == "ALCISTA" and mom > momentum_threshold
    is_bearish = trend == "BAJISTA" and mom < -momentum_threshold
    if not (is_bullish or is_bearish):
        return []

    target_type = "C" if is_bullish else "P"
    action = "BUY_CALL" if is_bullish else "BUY_PUT"
    direction = "alcista" if is_bullish else "bajista"
    signals: list[ScalpSignal] = []

    for sym, g in greeks.items():
        if g.option_type != target_type:
            continue
        if g.days < 2 or g.days > 25:
            continue
        if g.moneyness_pct < -3.0 or g.moneyness_pct > 8.0:
            continue
        
        cfg = profile_cfg or _PROFILE_CFG["balanced"]
        min_delta = float(cfg.get("min_action_delta", 0.35))
        if abs(g.delta) < min_delta:
            continue

        # Sweet spot en delta ~0.50 (máx gamma, máx leverage) — no lineal
        delta_score = max(0.0, 30.0 - abs(abs(g.delta) - 0.50) * 60.0)
        gamma_score = min(g.gamma * 10000.0, 20.0)
        max_spread = max(float(cfg.get("max_spread", 15.0)), 1.0)
        spread_penalty = (g.spread_pct / max_spread) * 25.0
        dte_bonus = max(0.0, 10.0 - g.days / 3.0)
        momentum_bonus = min(abs(mom) * 3.0, 15.0)
        vol_confirms = 5.0 if bool(getattr(snap, "vol_confirms", False)) else 0.0
        score = max(0.0, min(100.0, delta_score + gamma_score - spread_penalty + dte_bonus + momentum_bonus + vol_confirms))

        urgency = "alta" if score >= _URGENCY_HIGH_ACTION else ("media" if score >= _URGENCY_MED_ACTION else "baja")
        rationale = (
            f"GGAL {direction} | mom {mom:+.1f}% | "
            f"Delta={g.delta:+.3f} Gamma={g.gamma:.5f} | "
            f"spread {g.spread_pct:.1f}% | {g.days}d al vto."
        )

        sig = ScalpSignal(
            signal_type="MOMENTUM",
            action=action,
            symbol=sym,
            strike=g.strike,
            option_type=g.option_type,
            days_to_expiry=g.days,
            bid=g.bid,
            ask=g.ask,
            mid=g.mid,
            spread_pct=g.spread_pct,
            delta=g.delta,
            gamma=g.gamma,
            theta=g.theta,
            vega=g.vega,
            iv=g.iv,
            score=round(score, 1),
            edge_source=f"Momentum {direction} ({mom:+.1f}%)",
            rationale=rationale,
            urgency=urgency,
            playbook="Seguir ola",
            trigger="Momentum direccional",
            expected_move="continuacion alcista" if is_bullish else "continuacion bajista",
        )
        signals.append(
            _finalize_signal(sig, g, snap=snap, capital=capital, positions=positions, profile_cfg=profile_cfg)
        )

    signals.sort(key=lambda s: s.score, reverse=True)
    return signals[:10]


def scan_wave_ride(
    greeks: dict[str, OptionGreeks],
    snap,
    momentum_threshold: float = 1.0,
    capital: Optional[float] = None,
    positions: Optional[list] = None,
    profile_cfg: Optional[dict[str, float]] = None,
) -> list[ScalpSignal]:
    """Sigue la ola: compra gamma en la direccion del impulso confirmado."""
    if snap is None:
        return []

    cfg = profile_cfg or _PROFILE_CFG["balanced"]
    bias = _direction_bias(snap)
    if bias not in ("BULL", "BEAR"):
        return []

    mom = _snap_float(snap, "momentum", 0.0)
    if abs(mom) < momentum_threshold:
        return []

    opt_type, action = _side_action(bias)
    vol_ok = bool(getattr(snap, "vol_confirms", False))
    structure_ok = _structure_confirms(snap, bias)
    macd = _snap_float(snap, "macd_hist", 0.0)
    rsi_val = _snap_float(snap, "rsi", 50.0)
    macd_ok = (macd >= 0 and bias == "BULL") or (macd <= 0 and bias == "BEAR")

    signals: list[ScalpSignal] = []
    for g in _movement_pool(greeks, opt_type, max_days=25, min_delta=0.25, max_abs_moneyness=10.0):
        context_score = min(abs(mom) * 2.5, 12.0)
        context_score += 6.0 if structure_ok else 0.0
        context_score += 5.0 if vol_ok else 0.0
        context_score += 3.0 if macd_ok else 0.0
        exhaustion_penalty = 12.0 if _is_exhausted(snap, bias) else 0.0
        score = max(0.0, min(100.0, _option_movement_score(g, cfg) + context_score - exhaustion_penalty))

        direction = "alcista" if bias == "BULL" else "bajista"
        flags: list[str] = []
        if not (structure_ok or vol_ok or macd_ok):
            flags.append("ola_sin_confirmacion")
        if _is_exhausted(snap, bias):
            flags.append("rsi_extendido")

        sig = ScalpSignal(
            signal_type="WAVE_RIDE",
            action=action,
            symbol=g.symbol,
            strike=g.strike,
            option_type=g.option_type,
            days_to_expiry=g.days,
            bid=g.bid,
            ask=g.ask,
            mid=g.mid,
            spread_pct=g.spread_pct,
            delta=g.delta,
            gamma=g.gamma,
            theta=g.theta,
            vega=g.vega,
            iv=g.iv,
            score=round(score, 1),
            edge_source=f"Ola {direction} mom {mom:+.1f}%",
            rationale=(
                f"Seguir ola {direction}: trend {_snap_text(snap, 'trend')} | "
                f"RSI {rsi_val:.0f} | MACD {macd:+.1f} | "
                f"delta {g.delta:+.2f} gamma {g.gamma:.5f}"
            ),
            urgency="alta" if score >= _URGENCY_HIGH_ACTION else ("media" if score >= _URGENCY_MED_ACTION else "baja"),
            playbook="Seguir ola",
            trigger="Momentum + estructura/volumen",
            expected_move="continuacion alcista" if bias == "BULL" else "continuacion bajista",
        )
        finalized = _finalize_signal(sig, g, snap=snap, capital=capital, positions=positions, profile_cfg=cfg)
        signals.append(_add_watch_flags(finalized, *flags))

    signals.sort(key=lambda s: s.score, reverse=True)
    return signals[:10]


def scan_rebound(
    greeks: dict[str, OptionGreeks],
    snap,
    capital: Optional[float] = None,
    positions: Optional[list] = None,
    profile_cfg: Optional[dict[str, float]] = None,
) -> list[ScalpSignal]:
    """Pricea rebotes: giro desde RSI extremo, CHoCH u order block cercano."""
    if snap is None:
        return []

    cfg = profile_cfg or _PROFILE_CFG["balanced"]
    rsi_val = _snap_float(snap, "rsi", 50.0)
    mom = _snap_float(snap, "momentum", 0.0)
    ob = getattr(snap, "order_block", None)
    fvg = getattr(snap, "nearest_fvg", None)

    seed_sides: list[str] = []
    if rsi_val <= 36.0 or _snap_text(snap, "choch").upper() == "CHOCH_UP":
        seed_sides.append("BULL")
    if rsi_val >= 64.0 or _snap_text(snap, "choch").upper() == "CHOCH_DOWN":
        seed_sides.append("BEAR")
    unique_sides = list(dict.fromkeys(seed_sides + ["BULL", "BEAR"]))

    signals: list[ScalpSignal] = []
    for side in unique_sides:
        opt_type, action = _side_action(side)
        structure_ok = _structure_confirms(snap, side)
        vol_ok = bool(getattr(snap, "vol_confirms", False))
        is_countertrend = _direction_bias(snap) not in (side, "NEUTRAL")
        for g in _movement_pool(greeks, opt_type, max_days=25, min_delta=0.22, max_abs_moneyness=12.0):
            zone_ok = False
            if ob is not None:
                zone_ok = (
                    (side == "BULL" and getattr(ob, "type", "") == "bullish")
                    or (side == "BEAR" and getattr(ob, "type", "") == "bearish")
                ) and _near_zone(g.spot, float(ob.low), float(ob.high))
            if side not in seed_sides and not zone_ok:
                continue

            rsi_score = max(0.0, 36.0 - rsi_val) * 1.8 if side == "BULL" else max(0.0, rsi_val - 64.0) * 1.8
            structure_score = 14.0 if structure_ok else 0.0
            vol_score = 7.0 if vol_ok else 0.0
            zone_score = 7.0 if zone_ok else 0.0
            fvg_score = 5.0 if fvg is not None else 0.0
            countertrend_penalty = 8.0 if is_countertrend and not structure_ok else 0.0
            score = max(
                0.0,
                min(100.0, _option_movement_score(g, cfg) + rsi_score + structure_score + vol_score + zone_score + fvg_score - countertrend_penalty),
            )

            flags: list[str] = []
            if not structure_ok and not zone_ok:
                flags.append("rebote_sin_choch")
            if is_countertrend:
                flags.append("rebote_contra_tendencia")

            direction = "alcista" if side == "BULL" else "bajista"
            sig = ScalpSignal(
                signal_type="REBOUND_PRICING",
                action=action,
                symbol=g.symbol,
                strike=g.strike,
                option_type=g.option_type,
                days_to_expiry=g.days,
                bid=g.bid,
                ask=g.ask,
                mid=g.mid,
                spread_pct=g.spread_pct,
                delta=g.delta,
                gamma=g.gamma,
                theta=g.theta,
                vega=g.vega,
                iv=g.iv,
                score=round(score, 1),
                edge_source=f"Rebote {direction} RSI {rsi_val:.0f}",
                rationale=(
                    f"Pricear rebote {direction}: RSI {rsi_val:.0f}, mom {mom:+.1f}%, "
                    f"estructura {'confirma' if structure_ok else 'sin CHoCH'}, spread {g.spread_pct:.1f}%"
                ),
                urgency="alta" if score >= _URGENCY_HIGH_ACTION else ("media" if score >= _URGENCY_MED_ACTION else "baja"),
                playbook="Pricear rebote",
                trigger="RSI extremo / CHoCH / zona",
                expected_move="rebote alcista" if side == "BULL" else "rebote bajista",
            )
            finalized = _finalize_signal(sig, g, snap=snap, capital=capital, positions=positions, profile_cfg=cfg)
            signals.append(_add_watch_flags(finalized, *flags))

    signals.sort(key=lambda s: s.score, reverse=True)
    return signals[:10]


def scan_volatility_breakout(
    greeks: dict[str, OptionGreeks],
    snap,
    realized_vol: Optional[float] = None,
    capital: Optional[float] = None,
    positions: Optional[list] = None,
    profile_cfg: Optional[dict[str, float]] = None,
) -> list[ScalpSignal]:
    """Arma voladura/breakout: compra gamma cuando se expande rango o volumen."""
    if snap is None:
        return []

    cfg = profile_cfg or _PROFILE_CFG["balanced"]
    mom = _snap_float(snap, "momentum", 0.0)
    atr_pct = _snap_float(snap, "atr_pct", 0.0)
    vol_ok = bool(getattr(snap, "vol_confirms", False))
    bias = _direction_bias(snap)
    explosive = abs(mom) >= 3.0 or atr_pct >= 2.6 or (vol_ok and abs(mom) >= 1.5)
    if not explosive:
        return []

    sides = [bias] if bias in ("BULL", "BEAR") else ["BULL", "BEAR"]
    signals: list[ScalpSignal] = []
    for side in sides:
        opt_type, action = _side_action(side)
        structure_ok = _structure_confirms(snap, side)
        for g in _movement_pool(greeks, opt_type, max_days=25, min_delta=0.20, max_abs_moneyness=14.0):
            vol_edge = (realized_vol - g.iv) if realized_vol and g.iv else 0.0
            breakout_score = min(abs(mom) * 2.0, 10.0) + min(atr_pct * 1.5, 8.0)
            breakout_score += 4.0 if vol_ok else 0.0
            breakout_score += 5.0 if structure_ok else 0.0
            breakout_score += min(max(vol_edge, 0.0) * 100.0, 5.0)
            score = max(0.0, min(100.0, _option_movement_score(g, cfg) + breakout_score))

            flags: list[str] = []
            if bias == "NEUTRAL":
                flags.append("esperando_breakout_direccion")
            if not (vol_ok or structure_ok):
                flags.append("voladura_sin_confirmar")

            direction = "alcista" if side == "BULL" else "bajista"
            sig = ScalpSignal(
                signal_type="VOL_BREAKOUT",
                action=action,
                symbol=g.symbol,
                strike=g.strike,
                option_type=g.option_type,
                days_to_expiry=g.days,
                bid=g.bid,
                ask=g.ask,
                mid=g.mid,
                spread_pct=g.spread_pct,
                delta=g.delta,
                gamma=g.gamma,
                theta=g.theta,
                vega=g.vega,
                iv=g.iv,
                score=round(score, 1),
                edge_source=f"Voladura {direction} ATR {atr_pct:.1f}% mom {mom:+.1f}%",
                rationale=(
                    f"Armar breakout {direction}: ATR {atr_pct:.1f}%, mom {mom:+.1f}%, "
                    f"volumen {'ok' if vol_ok else 'sin confirmar'}, gamma {g.gamma:.5f}"
                ),
                urgency="alta" if score >= _URGENCY_HIGH_ACTION else ("media" if score >= _URGENCY_MED_ACTION else "baja"),
                playbook="Voladura/breakout",
                trigger="Expansion de rango o volumen",
                expected_move="breakout alcista" if side == "BULL" else "breakout bajista",
            )
            finalized = _finalize_signal(sig, g, snap=snap, capital=capital, positions=positions, profile_cfg=cfg)
            signals.append(_add_watch_flags(finalized, *flags))

    signals.sort(key=lambda s: s.score, reverse=True)
    return signals[:12]


def scan_iv_spike(
    greeks: dict[str, OptionGreeks],
    snap,
    smile_map: Optional[dict] = None,
    spot: float = 0.0,
    mispricing_min_pp: float = 2.0,
    capital: Optional[float] = None,
    positions: Optional[list] = None,
    profile_cfg: Optional[dict[str, float]] = None,
) -> list[ScalpSignal]:
    if smile_map is None or not smile_map or spot <= 0:
        return []

    try:
        from optionsdesk.core.iv_surface import mispricing_score
    except ImportError:
        return []

    sym_re = re.compile(r"^GFG([CV])(\d+(?:[.,]\d+)?)([A-Z]+)$")
    signals: list[ScalpSignal] = []

    for sym, g in greeks.items():
        if g.iv is None or g.iv <= 0:
            continue
        m = sym_re.match(sym)
        if not m:
            continue
        fit = smile_map.get(m.group(3))
        if fit is None:
            continue
        msp = mispricing_score(g.strike, g.iv, fit)
        if msp is None or msp < mispricing_min_pp:
            continue

        action = "SELL_CALL" if g.option_type == "C" else "SELL_PUT"
        msp_score = min(msp * 8.0, 50.0)
        liquidity_score = min(g.volume / 10.0, 20.0)
        cfg = profile_cfg or _PROFILE_CFG["balanced"]
        max_spread = max(float(cfg.get("max_spread", 15.0)), 1.0)
        spread_penalty = (g.spread_pct / max_spread) * 25.0
        gamma_risk = min(g.gamma * 5000.0, 15.0)
        score = max(0.0, min(100.0, msp_score + liquidity_score - spread_penalty - gamma_risk))
        urgency = "alta" if score >= _URGENCY_HIGH_ALWAYS else ("media" if score >= _URGENCY_MED_ALWAYS else "baja")

        rationale = (
            f"IV {g.iv * 100:.1f}% cara vs sonrisa "
            f"(mispricing +{msp:.1f}pp) | spread {g.spread_pct:.1f}%"
        )
        sig = ScalpSignal(
            signal_type="IV_SPIKE",
            action=action,
            symbol=sym,
            strike=g.strike,
            option_type=g.option_type,
            days_to_expiry=g.days,
            bid=g.bid,
            ask=g.ask,
            mid=g.mid,
            spread_pct=g.spread_pct,
            delta=g.delta,
            gamma=g.gamma,
            theta=g.theta,
            vega=g.vega,
            iv=g.iv,
            score=round(score, 1),
            edge_source=f"IV cara +{msp:.1f}pp",
            rationale=rationale,
            urgency=urgency,
            playbook="Contexto IV",
            trigger="IV cara vs sonrisa",
            expected_move="solo watchlist para scalping direccional",
        )
        finalized = _finalize_signal(
            sig,
            g,
            snap=snap,
            capital=capital,
            positions=positions,
            profile_cfg=profile_cfg,
        )
        signals.append(_add_watch_flags(finalized, "venta_iv_solo_contexto"))

    signals.sort(key=lambda s: s.score, reverse=True)
    return signals[:10]


def scan_gamma_scalp(
    greeks: dict[str, OptionGreeks],
    snap=None,
    realized_vol: Optional[float] = None,
    capital: Optional[float] = None,
    positions: Optional[list] = None,
    profile_cfg: Optional[dict[str, float]] = None,
) -> list[ScalpSignal]:
    cfg = profile_cfg or _PROFILE_CFG["balanced"]
    vol_edge_min = float(cfg["vol_edge_min"])

    if realized_vol is None or realized_vol <= 0:
        return []

    signals: list[ScalpSignal] = []
    for sym, g in greeks.items():
        if g.iv is None or g.iv <= 0:
            continue
        if g.moneyness != "ATM":
            continue
        if g.days < 2 or g.days > 30:
            continue

        vol_edge = realized_vol - g.iv
        if vol_edge <= vol_edge_min:
            continue

        action = "BUY_CALL" if g.option_type == "C" else "BUY_PUT"
        edge_score = min(vol_edge * 100.0 * 3.0, 40.0)
        gamma_score = min(g.gamma * 15000.0, 30.0)
        dte_bonus = max(0.0, 15.0 - g.days)
        max_spread = max(float(cfg.get("max_spread", 15.0)), 1.0)
        spread_penalty = (g.spread_pct / max_spread) * 25.0
        score = max(0.0, min(100.0, edge_score + gamma_score + dte_bonus - spread_penalty))
        urgency = "alta" if score >= _URGENCY_HIGH_ALWAYS else ("media" if score >= _URGENCY_MED_ALWAYS else "baja")

        rationale = (
            f"Gamma scalp: IV {g.iv * 100:.1f}% < RV {realized_vol * 100:.1f}% "
            f"(edge {vol_edge * 100:+.1f}pp)"
        )
        sig = ScalpSignal(
            signal_type="GAMMA_SCALP",
            action=action,
            symbol=sym,
            strike=g.strike,
            option_type=g.option_type,
            days_to_expiry=g.days,
            bid=g.bid,
            ask=g.ask,
            mid=g.mid,
            spread_pct=g.spread_pct,
            delta=g.delta,
            gamma=g.gamma,
            theta=g.theta,
            vega=g.vega,
            iv=g.iv,
            score=round(score, 1),
            edge_source=f"Gamma scalp (RV-IV={vol_edge * 100:+.1f}pp)",
            rationale=rationale,
            urgency=urgency,
            playbook="Voladura/breakout",
            trigger="Gamma barata vs RV",
            expected_move="captura de expansion gamma",
        )
        finalized = _finalize_signal(
            sig,
            g,
            snap=snap,
            capital=capital,
            positions=positions,
            profile_cfg=profile_cfg,
        )
        signals.append(_add_watch_flags(finalized, *_context_watch_flags(g, snap, realized_vol)))

    signals.sort(key=lambda s: s.score, reverse=True)
    return signals[:10]


def scan_gamma_radar(
    greeks: dict[str, OptionGreeks],
    snap=None,
    realized_vol: Optional[float] = None,
    capital: Optional[float] = None,
    positions: Optional[list] = None,
    profile_cfg: Optional[dict[str, float]] = None,
) -> list[ScalpSignal]:
    """Radar siempre activo de opciones comprables cerca del dinero.

    No pretende reemplazar a los setups accionables. Su trabajo es evitar un
    modulo mudo: muestra que contratos conviene vigilar y que confirmacion falta.
    """
    cfg = profile_cfg or _PROFILE_CFG["balanced"]
    bias = _direction_bias(snap)
    pool = [
        g
        for g in greeks.values()
        if _is_quoteable(g) and 1 <= g.days <= 180 and abs(g.moneyness_pct) <= 12.0
    ]
    if not pool:
        pool = [g for g in greeks.values() if _is_quoteable(g) and 1 <= g.days <= 240]

    signals: list[ScalpSignal] = []
    for g in pool:
        if bias == "BULL" and g.option_type != "C":
            continue
        if bias == "BEAR" and g.option_type != "P":
            continue

        action = "BUY_CALL" if g.option_type == "C" else "BUY_PUT"
        gamma_score = min(max(g.gamma, 0.0) * 20000.0, 26.0)
        delta_score = max(0.0, 18.0 - abs(abs(g.delta) - 0.50) * 45.0)
        atm_score = max(0.0, 18.0 - abs(g.moneyness_pct) * 1.7)
        dte_score = max(0.0, 14.0 - abs(g.days - 12.0) * 0.25)
        liq_score = _liquidity_score(g, cfg)
        vol_edge = (realized_vol - g.iv) if realized_vol and g.iv else None
        # Penaliza cuando IV > RV (opción cara), bonifica cuando RV > IV (barata)
        vol_score = min(max((vol_edge or 0.0) * 100.0 * 1.5, -10.0), 12.0)
        score = max(0.0, min(100.0, gamma_score + delta_score + atm_score + dte_score + liq_score + vol_score))

        flags = _context_watch_flags(g, snap, realized_vol)
        if vol_edge is not None and vol_edge <= float(cfg["vol_edge_min"]):
            flags.append("sin_edge_vol_confirmado")
        flags.append("esperando_trigger_scalping")

        urgency = "alta" if score >= _URGENCY_HIGH_RADAR else ("media" if score >= _URGENCY_MED_RADAR else "baja")
        edge = "Radar gamma/ATM"
        if vol_edge is not None:
            edge = f"Radar gamma/ATM RV-IV={vol_edge * 100:+.1f}pp"
        rationale = (
            f"Contrato vigilable: {g.moneyness} {g.moneyness_pct:+.1f}% | "
            f"DTE {g.days} | gamma {g.gamma:.5f} | delta {g.delta:+.2f} | "
            f"spread {g.spread_pct:.1f}%"
        )
        sig = ScalpSignal(
            signal_type="GAMMA_RADAR",
            action=action,
            symbol=g.symbol,
            strike=g.strike,
            option_type=g.option_type,
            days_to_expiry=g.days,
            bid=g.bid,
            ask=g.ask,
            mid=g.mid,
            spread_pct=g.spread_pct,
            delta=g.delta,
            gamma=g.gamma,
            theta=g.theta,
            vega=g.vega,
            iv=g.iv,
            score=round(score, 1),
            edge_source=edge,
            rationale=rationale,
            urgency=urgency,
            playbook="Radar movimiento",
            trigger="ATM/gamma liquida",
            expected_move="esperando trigger",
        )
        signals.append(
            _finalize_as_watchlist(sig, g, snap, capital, positions, cfg, flags)
        )

    signals.sort(key=lambda s: s.score, reverse=True)
    return signals[:16]


def scan_chain_iv_relative(
    greeks: dict[str, OptionGreeks],
    snap=None,
    realized_vol: Optional[float] = None,
    capital: Optional[float] = None,
    positions: Optional[list] = None,
    profile_cfg: Optional[dict[str, float]] = None,
) -> list[ScalpSignal]:
    """Fallback de IV relativa por vencimiento cuando no hay sonrisa confiable."""
    cfg = profile_cfg or _PROFILE_CFG["balanced"]
    edge_min_pp = max(3.0, float(cfg["vol_edge_min"]) * 100.0 * 2.0)

    by_expiry: dict[str, list[OptionGreeks]] = {}
    for g in greeks.values():
        if not _is_quoteable(g) or g.iv is None or g.iv <= 0:
            continue
        if g.days < 2 or g.days > 180 or abs(g.moneyness_pct) > 18.0:
            continue
        m = _SYM_RE.match(g.symbol.upper())
        if not m:
            continue
        by_expiry.setdefault(m.group(3), []).append(g)

    signals: list[ScalpSignal] = []
    for expiry_code, group in by_expiry.items():
        if len(group) < 3:
            continue
        med_iv = median([g.iv for g in group if g.iv is not None])
        if med_iv <= 0:
            continue

        for g in group:
            edge_pp = (g.iv - med_iv) * 100.0
            if abs(edge_pp) < edge_min_pp:
                continue

            is_rich = edge_pp > 0
            if is_rich:
                action = "SELL_CALL" if g.option_type == "C" else "SELL_PUT"
                signal_type = "IV_RICH_CHAIN"
                edge_source = f"IV cara vs cadena +{edge_pp:.1f}pp"
                extra_flags = ["confirmar_con_smile_iv"]
            else:
                action = "BUY_CALL" if g.option_type == "C" else "BUY_PUT"
                signal_type = "IV_CHEAP_CHAIN"
                edge_source = f"IV barata vs cadena {edge_pp:.1f}pp"
                extra_flags = _context_watch_flags(g, snap, realized_vol=realized_vol)
                extra_flags.append("confirmar_breakout")

            iv_score = min(abs(edge_pp) * 5.0, 42.0)
            liq_score = _liquidity_score(g, cfg)
            dte_score = max(0.0, 12.0 - abs(g.days - 15.0) * 0.35)
            gamma_penalty = min(max(g.gamma, 0.0) * 6000.0, 12.0) if is_rich else 0.0
            score = max(0.0, min(100.0, iv_score + liq_score + dte_score - gamma_penalty))
            urgency = "alta" if score >= _URGENCY_HIGH_RADAR else ("media" if score >= _URGENCY_MED_RADAR else "baja")
            rationale = (
                f"IV {g.iv * 100:.1f}% vs mediana {expiry_code} {med_iv * 100:.1f}% "
                f"({edge_pp:+.1f}pp) | spread {g.spread_pct:.1f}%"
            )
            sig = ScalpSignal(
                signal_type=signal_type,
                action=action,
                symbol=g.symbol,
                strike=g.strike,
                option_type=g.option_type,
                days_to_expiry=g.days,
                bid=g.bid,
                ask=g.ask,
                mid=g.mid,
                spread_pct=g.spread_pct,
                delta=g.delta,
                gamma=g.gamma,
                theta=g.theta,
                vega=g.vega,
                iv=g.iv,
                score=round(score, 1),
                edge_source=edge_source,
                rationale=rationale,
                urgency=urgency,
                playbook="Contexto IV" if is_rich else "Voladura/breakout",
                trigger="IV relativa por cadena",
                expected_move="solo watchlist para scalping direccional" if is_rich else "breakout a confirmar",
            )
            signals.append(
                _finalize_as_watchlist(sig, g, snap, capital, positions, cfg, extra_flags)
            )

    signals.sort(key=lambda s: s.score, reverse=True)
    return signals[:12]


def scan_all(
    chain: OptionsChain,
    expiry_calendar: dict,
    snap=None,
    smile_map: Optional[dict] = None,
    realized_vol: Optional[float] = None,
    r: float = 0.0,
    capital: Optional[float] = None,
    positions: Optional[list] = None,
    risk_profile: str = "balanced",
) -> tuple[dict[str, OptionGreeks], list[ScalpSignal]]:
    profile_key = str(risk_profile or "balanced").lower()
    cfg = dict(_PROFILE_CFG.get(profile_key, _PROFILE_CFG["balanced"]))
    cfg["enforce_session"] = True
    cfg["require_capital"] = True
    cfg["max_open_scalps"] = settings.scalping_max_open_positions
    cfg["max_total_risk_pct"] = settings.scalping_max_total_risk_pct
    cfg["time_stop_min"] = settings.scalping_no_progress_min
    cfg["flat_before_close_min"] = settings.scalping_flat_before_close_min

    greeks = compute_chain_greeks(chain, expiry_calendar, r=r)
    if not greeks:
        return {}, []

    spot = chain.spot.mid
    all_signals: list[ScalpSignal] = []

    all_signals.extend(
        scan_wave_ride(
            greeks,
            snap,
            momentum_threshold=settings.scalping_momentum_threshold,
            capital=capital,
            positions=positions,
            profile_cfg=cfg,
        )
    )
    all_signals.extend(
        scan_rebound(
            greeks,
            snap,
            capital=capital,
            positions=positions,
            profile_cfg=cfg,
        )
    )
    all_signals.extend(
        scan_volatility_breakout(
            greeks,
            snap,
            realized_vol=realized_vol,
            capital=capital,
            positions=positions,
            profile_cfg=cfg,
        )
    )
    all_signals.extend(
        scan_momentum(
            greeks,
            snap,
            momentum_threshold=settings.scalping_momentum_threshold,
            capital=capital,
            positions=positions,
            profile_cfg=cfg,
        )
    )
    all_signals.extend(
        scan_iv_spike(
            greeks,
            snap,
            smile_map=smile_map,
            spot=spot,
            mispricing_min_pp=settings.scalping_mispricing_min_pp,
            capital=capital,
            positions=positions,
            profile_cfg=cfg,
        )
    )
    all_signals.extend(
        scan_gamma_scalp(
            greeks,
            snap=snap,
            realized_vol=realized_vol,
            capital=capital,
            positions=positions,
            profile_cfg=cfg,
        )
    )
    all_signals.extend(
        scan_chain_iv_relative(
            greeks,
            snap=snap,
            realized_vol=realized_vol,
            capital=capital,
            positions=positions,
            profile_cfg=cfg,
        )
    )
    all_signals.extend(
        scan_gamma_radar(
            greeks,
            snap=snap,
            realized_vol=realized_vol,
            capital=capital,
            positions=positions,
            profile_cfg=cfg,
        )
    )

    # Contar cuántos scanners independientes apuntan al mismo símbolo
    confluence: dict[str, int] = {}
    for s in all_signals:
        confluence[s.symbol] = confluence.get(s.symbol, 0) + 1

    seen: dict[str, ScalpSignal] = {}
    for s in all_signals:
        prev = seen.get(s.symbol)
        if prev is None or _signal_rank(s) > _signal_rank(prev):
            seen[s.symbol] = s

    # Boost de confluencia post-deduplicación: más setups = señal más confiable
    for sym, sig in seen.items():
        n = confluence[sym]
        if n >= 2:
            sig.score = min(sig.score + 4.0 * (n - 1), 100.0)
            if "[x" not in sig.rationale:
                sig.rationale = f"[x{n} setups] " + sig.rationale

    final = sorted(seen.values(), key=_signal_rank, reverse=True)
    return greeks, final
