"""Fail-closed manual tickets for GGAL option scalping.

Signals remain broad for radar purposes. This module is the only path that can
turn an intraday setup into a manual Matriz ticket.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from optionsdesk.config.costs import CostProfile, DEFAULT_COSTS
from optionsdesk.config.settings import settings
from optionsdesk.core.carry import CarryEstimate
from optionsdesk.core.greeks_engine import OptionGreeks
from optionsdesk.data.providers.base import MarketDataHealth, OptionsChain
from optionsdesk.signals.scalping import LivePulse, ScalpSignal

_BA = ZoneInfo("America/Argentina/Buenos_Aires")
_ALLOWED_SIGNAL_TYPES = {
    "CONTINUATION": {"WAVE_RIDE"},
    "BREAKOUT_RETEST": {"VOL_BREAKOUT"},
}

# ── Calidad del trigger 1m/5m (todo escalado al ruido ATR del subyacente) ──────
# Un scalper real no opera rupturas dentro del ruido ni persigue precio extendido.
_BREAKOUT_ATR_MULT   = 0.30    # la ruptura debe superar el nivel por ≥ 0.30×ATR1m (anti-chop)
_MAX_EXTENSION_ATR   = 3.0     # si el precio ya corrió >3×ATR1m del nivel, ya es chase parabólico → no entrar
_FIVE_TREND_ATR_MULT = 0.25    # el empuje de 5m (3 velas) debe superar 0.25×ATR5m, no solo el signo
_MIN_RISK_ATR        = 0.5     # piso del riesgo en ATR1m (evita R diminutos por retests pegados)
_NOISE_FLOOR_PCT     = 0.0005  # piso del ATR1m como fracción del spot (protege ante datos planos)
# Intradía AR: una opción puede sostenerse minutos o hasta 2-3h (el subyacente se
# mueve lento). El time-stop es backstop; las salidas reales son stop/target/cierre.
_MAX_HOLD_MIN        = 180


class OperationalMode(str, Enum):
    RADAR_IOL = "RADAR_IOL"
    IOL_PROVISIONAL = "IOL_PROVISIONAL"
    PRIMARY_WARMING = "PRIMARY_WARMING"
    SHADOW = "SHADOW"
    TICKET_READY = "TICKET_READY"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class CalibrationStatus:
    primary_shadow_complete: bool
    source: str = "settings"


@dataclass(frozen=True)
class UnderlyingTrigger:
    setup: str
    side: str
    created_at: datetime
    spot: float
    invalidation_spot: float
    target_spot: float
    rationale: str


@dataclass(frozen=True)
class OptionSelection:
    symbol: str
    action: str
    strike: float
    days_to_expiry: int
    delta: float
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    spread_pct: float
    quote_age_s: float
    quote_source: str
    limit_price: float
    execution_score: float


@dataclass(frozen=True)
class ScalpTicket:
    ticket_id: str
    mode: OperationalMode
    created_at: datetime
    valid_until: datetime
    trigger: UnderlyingTrigger
    selection: OptionSelection
    option_stop: float
    option_take_profit: float
    time_stop_min: int
    fees_ars_per_lot: float
    stressed_risk_ars_per_lot: float
    max_lots: int
    carry: CarryEstimate
    cost_profile: CostProfile
    checklist: tuple[str, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DailyScalpPerformance:
    realized_pnl_ars: float = 0.0
    stops: int = 0


def resolve_operational_mode(health: MarketDataHealth) -> OperationalMode:
    if health.source == "IOL":
        return (
            OperationalMode.IOL_PROVISIONAL
            if health.ticket_capable and settings.iol_provisional_tickets_enabled
            else OperationalMode.RADAR_IOL
        )
    if health.source == "PRIMARY":
        if health.ticket_capable:
            return OperationalMode.TICKET_READY
        if health.connected and not health.warmup_complete:
            return OperationalMode.PRIMARY_WARMING
        if health.connected:
            return OperationalMode.SHADOW
    return OperationalMode.BLOCKED


def confirm_manual_reconciliation(
    local_positions: int,
    *,
    path: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> None:
    """Persist the operator confirmation made after reviewing Matriz."""
    target = path or settings.manual_reconciliation_file
    current = now or datetime.now(_BA)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "date": current.date().isoformat(),
                "confirmed_at": current.isoformat(timespec="seconds"),
                "local_positions": int(local_positions),
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )


def manual_reconciliation_is_current(
    *,
    path: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> bool:
    target = path or settings.manual_reconciliation_file
    current = now or datetime.now(_BA)
    if not target.exists():
        return False
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        return payload.get("date") == current.date().isoformat()
    except (OSError, json.JSONDecodeError):
        return False


def invalidate_manual_reconciliation(*, path: Optional[Path] = None) -> None:
    target = path or settings.manual_reconciliation_file
    if target.exists():
        target.unlink()


def load_daily_scalp_performance(
    *,
    path: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> DailyScalpPerformance:
    target = path or settings.open_positions_file.with_name("closed_positions.jsonl")
    current = now or datetime.now(_BA)
    if not target.exists():
        return DailyScalpPerformance()
    pnl = 0.0
    stops = 0
    for line in target.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            closed = datetime.fromisoformat(str(record["closed_at"]))
            if closed.date() != current.date() or "SCALP" not in str(record.get("strategy", "")).upper():
                continue
            pnl += float(record.get("realized_pnl_ars") or 0.0)
            if "stop" in str(record.get("close_reason", "")).lower():
                stops += 1
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return DailyScalpPerformance(round(pnl, 2), stops)


def detect_underlying_trigger(
    bars_1m: pd.DataFrame,
    bars_5m: pd.DataFrame,
    pulse: LivePulse,
    *,
    now: Optional[datetime] = None,
) -> Optional[UnderlyingTrigger]:
    """Detect continuation or breakout-retest using completed spot bars only."""
    current = now or datetime.now(_BA)
    one = _completed_bars(bars_1m, "1min", current)
    five = _completed_bars(bars_5m, "5min", current)
    if len(one) < 8 or len(five) < 5 or not pulse.confirmed:
        return None

    spot = float(one.iloc[-1]["close"])
    side = str(pulse.direction).upper()
    if side not in {"BULL", "BEAR"} or spot <= 0:
        return None

    # Unidad de ruido del subyacente: todo el gating se escala a esto para ser
    # robusto entre regímenes de volatilidad (apertura volátil vs chop de mediodía).
    atr1m = max(_noise_unit(one), spot * _NOISE_FLOOR_PCT)
    atr5m = max(_noise_unit(five), spot * _NOISE_FLOOR_PCT)
    min_break = _BREAKOUT_ATR_MULT * atr1m
    max_ext   = _MAX_EXTENSION_ATR * atr1m

    # El empuje de 5m (3 velas ≈ 15 min) debe tener magnitud, no solo signo.
    five_move = float(five.iloc[-1]["close"]) - float(five.iloc[-4]["close"])
    five_thresh = _FIVE_TREND_ATR_MULT * atr5m
    trend_ok = five_move > five_thresh if side == "BULL" else five_move < -five_thresh
    if not trend_ok:
        return None

    def _dist(level: float) -> float:
        """Distancia favorable de la ruptura sobre el nivel (positiva = a favor)."""
        return spot - level if side == "BULL" else level - spot

    # Breakout-retest: el retest define la entrada controlada, así que solo exigimos
    # que la ruptura supere el ruido (anti-chop). El cap de extensión NO aplica: la
    # invalidación está en el retest, no se está persiguiendo precio.
    breakout = _breakout_retest(one, side)
    if breakout is not None and _dist(breakout[0]) >= min_break:
        level, invalidation = breakout
        risk = max(abs(spot - invalidation), _MIN_RISK_ATR * atr1m)
        target = spot + risk * 1.6 if side == "BULL" else spot - risk * 1.6
        return UnderlyingTrigger(
            setup="BREAKOUT_RETEST",
            side=side,
            created_at=_bar_timestamp(one.iloc[-1]),
            spot=round(spot, 2),
            invalidation_spot=round(invalidation, 2),
            target_spot=round(target, 2),
            rationale=f"ruptura y retest confirmado de {level:,.2f} en velas completas 1m (>{_BREAKOUT_ATR_MULT:g}xATR)",
        )

    # Continuación: comprás momentum, así que aplican ambos límites — la ruptura
    # debe superar el ruido (no chop) y no estar ya extendida (no chase parabólico).
    prior = one.iloc[-4:-1]
    level = float(prior["high"].max()) if side == "BULL" else float(prior["low"].min())
    if not (min_break <= _dist(level) <= max_ext):
        return None
    invalidation = float(prior["low"].min()) if side == "BULL" else float(prior["high"].max())
    risk = max(abs(spot - invalidation), _MIN_RISK_ATR * atr1m)
    target = spot + risk * 1.4 if side == "BULL" else spot - risk * 1.4
    return UnderlyingTrigger(
        setup="CONTINUATION",
        side=side,
        created_at=_bar_timestamp(one.iloc[-1]),
        spot=round(spot, 2),
        invalidation_spot=round(invalidation, 2),
        target_spot=round(target, 2),
        rationale="direccion 5m y ruptura de estructura local 1m confirmadas por tape",
    )


def build_manual_scalp_ticket(
    signals: list[ScalpSignal],
    greeks: dict[str, OptionGreeks],
    chain: OptionsChain,
    health: MarketDataHealth,
    trigger: Optional[UnderlyingTrigger],
    carry: CarryEstimate,
    *,
    capital: Optional[float],
    local_positions: list,
    account_reconciled: bool,
    daily_performance: Optional[DailyScalpPerformance] = None,
    now: Optional[datetime] = None,
) -> tuple[Optional[ScalpTicket], list[str]]:
    """Return one auditable manual ticket or explicit blockers."""
    current = now or datetime.now(_BA)
    mode = resolve_operational_mode(health)
    perf = daily_performance or DailyScalpPerformance()
    manual_sizing = bool(getattr(settings, "scalping_manual_sizing", True))
    blockers: list[str] = []

    if mode not in {OperationalMode.IOL_PROVISIONAL, OperationalMode.TICKET_READY}:
        blockers.append(f"modo_{mode.value.lower()}")
    session_minute = current.astimezone(_BA).hour * 60 + current.astimezone(_BA).minute
    if current.astimezone(_BA).weekday() >= 5 or not (11 * 60 + 20 <= session_minute < 16 * 60 + 45):
        blockers.append("fuera_ventana_1120_1645")
    if not settings.costs_verified:
        blockers.append("costos_no_verificados")
    if not manual_sizing and (not capital or capital <= 0):
        blockers.append("capital_no_cargado")
    if not account_reconciled:
        blockers.append("matriz_no_conciliada")
    if local_positions:
        blockers.append("ya_hay_posicion_abierta")
    if perf.stops >= settings.scalping_max_stops_per_day:
        blockers.append("pausa_por_stops_diarios")
    if capital and perf.realized_pnl_ars <= -(capital * settings.scalping_daily_loss_pct / 100.0):
        blockers.append("limite_perdida_diaria")
    if trigger is None:
        blockers.append("sin_trigger_1m_5m_completo")
    if blockers:
        return None, blockers

    candidates: list[tuple[tuple[float, ...], ScalpSignal, OptionGreeks, float, float]] = []
    allowed_types = _ALLOWED_SIGNAL_TYPES[trigger.setup]
    for signal in signals:
        greek = greeks.get(signal.symbol)
        if greek is None or signal.signal_type not in allowed_types:
            continue
        if signal.confidence != "actionable":
            continue
        if trigger.side == "BULL" and signal.action != "BUY_CALL":
            continue
        if trigger.side == "BEAR" and signal.action != "BUY_PUT":
            continue
        selection_block = _selection_block(greek, mode)
        if selection_block:
            continue
        stressed_risk = _stressed_risk_per_lot(signal, greek)
        max_lots = _max_lots(greek, stressed_risk, capital or 0.0, mode, manual_sizing=manual_sizing)
        if max_lots <= 0:
            continue
        execution_score = _execution_score(greek, signal)
        rank = (
            execution_score,
            -greek.spread_pct,
            -greek.quote_age_s,
            greek.ask_size,
            greek.volume,
        )
        candidates.append((rank, signal, greek, stressed_risk, execution_score))

    if not candidates:
        return None, ["sin_contrato_ejecutable"]

    _, signal, greek, stressed_risk, execution_score = max(candidates, key=lambda row: row[0])
    max_lots = _max_lots(greek, stressed_risk, capital or 0.0, mode, manual_sizing=manual_sizing)
    ttl_s = settings.iol_ticket_ttl_s if mode == OperationalMode.IOL_PROVISIONAL else settings.scalping_ticket_ttl_s
    limit_price = _round_up(float(signal.plan_entry), greek.min_price_increment)
    selection = OptionSelection(
        symbol=signal.symbol,
        action=signal.action,
        strike=signal.strike,
        days_to_expiry=signal.days_to_expiry,
        delta=signal.delta,
        bid=signal.bid,
        ask=signal.ask,
        bid_size=greek.bid_size,
        ask_size=greek.ask_size,
        spread_pct=signal.spread_pct,
        quote_age_s=greek.quote_age_s,
        quote_source=greek.source,
        limit_price=limit_price,
        execution_score=execution_score,
    )
    warnings = (
        ("IOL REST parcial: validar libro y precio limite nuevamente en Matriz antes de enviar.",)
        if mode == OperationalMode.IOL_PROVISIONAL
        else ()
    )
    created_at = current.astimezone(_BA)
    return ScalpTicket(
        ticket_id=f"{created_at:%Y%m%d-%H%M%S}-{signal.symbol}",
        mode=mode,
        created_at=created_at,
        valid_until=created_at + timedelta(seconds=ttl_s),
        trigger=trigger,
        selection=selection,
        option_stop=signal.plan_sl,
        option_take_profit=signal.plan_tp,
        time_stop_min=min(max(int(signal.time_stop_min or 20), 5), _MAX_HOLD_MIN),
        fees_ars_per_lot=signal.fees_ars,
        stressed_risk_ars_per_lot=round(stressed_risk, 2),
        max_lots=max_lots,
        carry=carry,
        cost_profile=CostProfile(
            name=settings.costs_profile,
            verified=settings.costs_verified,
            source="configured",
            model=DEFAULT_COSTS,
        ),
        checklist=(
            "Confirmar simbolo, lado, vencimiento y lotes en Matriz.",
            "Elegir lotes manualmente: el dashboard no dimensiona la orden.",
            "Revalidar bid/ask visible antes de enviar LIMIT.",
            "No perseguir la punta si vence el ticket.",
            "Registrar el fill real despues de la ejecucion manual.",
        ),
        warnings=warnings,
    ), []


def _completed_bars(df: pd.DataFrame, rule: str, current: datetime) -> pd.DataFrame:
    if df is None or df.empty or "time" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    out = out.dropna(subset=["time", "open", "high", "low", "close"]).sort_values("time")
    if out.empty:
        return out
    now_ts = pd.Timestamp(current)
    if out["time"].dt.tz is None and now_ts.tzinfo is not None:
        now_ts = now_ts.tz_localize(None)
    cutoff = now_ts.floor(rule)
    return out[out["time"] < cutoff].reset_index(drop=True)


def _bar_timestamp(row: pd.Series) -> datetime:
    return pd.Timestamp(row["time"]).to_pydatetime()


def _noise_unit(bars: pd.DataFrame, lookback: int = 10) -> float:
    """Unidad de ruido del subyacente = mediana del true range de las últimas
    `lookback` velas completas. Se usa la mediana (no la media) para que un único
    bar de expansión no infle el umbral y nos haga perder la ruptura siguiente."""
    if bars is None or len(bars) < 2:
        return 0.0
    high = bars["high"].astype(float)
    low  = bars["low"].astype(float)
    prev_close = bars["close"].astype(float).shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1).dropna().tail(lookback)
    return float(tr.median()) if not tr.empty else 0.0


def _breakout_retest(one: pd.DataFrame, side: str) -> Optional[tuple[float, float]]:
    history = one.iloc[:-3]
    break_bar, retest_bar, confirm_bar = one.iloc[-3], one.iloc[-2], one.iloc[-1]
    if history.empty:
        return None
    if side == "BULL":
        level = float(history["high"].tail(5).max())
        if float(break_bar["close"]) > level and float(retest_bar["low"]) <= level * 1.0015 and float(confirm_bar["close"]) > level:
            return level, float(retest_bar["low"])
    else:
        level = float(history["low"].tail(5).min())
        if float(break_bar["close"]) < level and float(retest_bar["high"]) >= level * 0.9985 and float(confirm_bar["close"]) < level:
            return level, float(retest_bar["high"])
    return None


def _selection_block(greek: OptionGreeks, mode: OperationalMode) -> str:
    max_age = (
        settings.iol_ticket_max_quote_age_s
        if mode == OperationalMode.IOL_PROVISIONAL
        else settings.scalping_contract_max_age_s
    )
    max_spread = (
        settings.iol_ticket_max_spread_pct
        if mode == OperationalMode.IOL_PROVISIONAL
        else 8.0
    )
    if not 7 <= greek.days <= 30:
        return "dte_fuera_de_rango"
    if not 0.40 <= abs(greek.delta) <= 0.65:
        return "delta_fuera_de_rango"
    if not -5.0 <= greek.moneyness_pct <= 2.0:
        return "strike_no_atm_itm"
    if greek.bid <= 0 or greek.ask <= greek.bid:
        return "sin_libro_valido"
    if greek.spread_pct > max_spread:
        return "spread_demasiado_ancho"
    if greek.quote_age_s > max_age:
        return "book_viejo"
    if greek.ask_size < 1:
        return "sin_profundidad_visible"
    if mode == OperationalMode.IOL_PROVISIONAL and greek.source != "IOL_REST":
        return "iol_sin_quote_directa"
    return ""


def _stressed_risk_per_lot(signal: ScalpSignal, greek: OptionGreeks) -> float:
    adverse_exit = max(greek.ask - greek.bid, 0.0) * 100.0
    return max(float(signal.planned_risk_ars or 0.0) + adverse_exit, 0.01)


def _max_lots(
    greek: OptionGreeks,
    stressed_risk: float,
    capital: float,
    mode: OperationalMode,
    *,
    manual_sizing: bool = False,
) -> int:
    if manual_sizing:
        return max(int(getattr(settings, "scalping_manual_max_lots", 10) or 1), 1)
    risk_budget = capital * settings.scalping_risk_per_trade_pct / 100.0
    risk_lots = int(risk_budget // max(stressed_risk, 0.01))
    depth_lots = int(max(greek.ask_size, 0.0))
    mode_cap = 1 if mode == OperationalMode.IOL_PROVISIONAL else settings.scalping_initial_lot_cap
    return max(min(risk_lots, depth_lots, mode_cap), 0)


def _execution_score(greek: OptionGreeks, signal: ScalpSignal) -> float:
    spread = max(0.0, 35.0 - greek.spread_pct * 4.0)
    delta = max(0.0, 25.0 - abs(abs(greek.delta) - 0.50) * 100.0)
    depth = min(math.log10(greek.ask_size + 1.0) * 12.0, 18.0)
    volume = min(math.log10(max(greek.volume, 0.0) + 1.0) * 5.0, 15.0)
    breakeven = max(0.0, 7.0 - float(signal.breakeven_move_pct or 0.0) * 7.0)
    return round(min(spread + delta + depth + volume + breakeven, 100.0), 1)


def _round_up(value: float, increment: float) -> float:
    tick = max(float(increment or 0.01), 0.01)
    return round(math.ceil(max(value, 0.0) / tick - 1e-9) * tick, 6)
