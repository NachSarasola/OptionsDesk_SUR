"""Motor de recomendación risk-adjusted.

Para cada perfil (CONSERVADOR, EQUILIBRADO, AGRESIVO) selecciona la mejor
oportunidad de covered call o short put usando un score ponderado que penaliza
las calls OTM-lotería con TNA altísima pero baja probabilidad de cobro.

Corrección conceptual clave: en v1 el screener ordenaba por TNA bruta, lo que
ponía primero calls OTM con 500%+ de TNA que casi nunca se cobran. Acá la
probabilidad (vía delta CRR) entra como factor de peso en el score, así que
una call ITM con 90% TNA y delta 0.75 le gana a una OTM con 500% y delta 0.08.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional

from optionsdesk.config.events import spans_event

logger = logging.getLogger(__name__)
from optionsdesk.config.settings import settings
from optionsdesk.core.benchmark import Benchmark
from optionsdesk.core.rates import RateResult
from optionsdesk.risk.limits import RiskChecker
from optionsdesk.signals.horizon import HorizonPlan, optimize_horizon
from optionsdesk.signals.volatility import VolEdge


class RiskProfile(str, Enum):
    CONSERVADOR = "CONSERVADOR"
    EQUILIBRADO = "EQUILIBRADO"
    AGRESIVO = "AGRESIVO"


# Pesos de scoring por perfil (v2.4: EV-centric)
# "ev" captura el edge estructural (VRP en pesos); "spread" movido a gate duro en _passes_gates.
_WEIGHTS: dict[RiskProfile, dict[str, float]] = {
    RiskProfile.CONSERVADOR: {
        "ev": 0.25, "vol": 0.20, "cushion": 0.35, "probability": 0.10, "liquidity": 0.10
    },
    RiskProfile.EQUILIBRADO: {
        "ev": 0.40, "vol": 0.20, "cushion": 0.20, "probability": 0.10, "liquidity": 0.10
    },
    RiskProfile.AGRESIVO: {
        "ev": 0.55, "vol": 0.20, "cushion": 0.10, "probability": 0.10, "liquidity": 0.05
    },
}

def _get_weights(
    profile: RiskProfile,
    override: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """Devuelve los pesos de scoring del perfil, con posibilidad de override.

    El `override` permite que el simulador y el GridSearch pasen pesos custom
    sin tocar el global _WEIGHTS. Si override es None devuelve los pesos
    estándar del perfil.
    """
    return dict(override) if override is not None else dict(_WEIGHTS[profile])


# Delta objetivo de la opción escrita por perfil. Funciona como bonus de score
# (no como gate) para no pisar _MIN_PROB. Captura el target de probabilidad
# implícita por perfil sin forzar un rango duro.
_DELTA_BANDS: dict[RiskProfile, tuple[float, float]] = {
    RiskProfile.CONSERVADOR: (0.15, 0.30),
    RiskProfile.EQUILIBRADO: (0.25, 0.40),
    RiskProfile.AGRESIVO:    (0.35, 0.50),
}

_MIN_CUSHION: dict[RiskProfile, float] = {
    RiskProfile.CONSERVADOR: 5.0,
    RiskProfile.EQUILIBRADO: 2.0,
    RiskProfile.AGRESIVO: 0.0,
}

# Probabilidad mínima de cobrar la tasa (via delta)
_MIN_PROB: dict[RiskProfile, float] = {
    RiskProfile.CONSERVADOR: 0.60,
    RiskProfile.EQUILIBRADO: 0.50,
    RiskProfile.AGRESIVO: 0.35,
}

_MAX_DAYS = 90
_MIN_DAYS = 5


@dataclass
class Recommendation:
    result: RateResult
    profile: RiskProfile
    score: float                           # 0–100
    light: str                             # "verde", "amarillo", "rojo"
    headline: str
    plain_explanation: str
    intention: str
    win_scenario: str
    lose_scenario: str
    action_steps: list[str]
    success_probability: float             # 0–1
    expected_profit_ars: Optional[float]   # None si no se pasó capital
    ticket_text: str
    warnings: list[str]
    contracts: int = 0                     # lotes calculados con el capital dado
    horizon_plan: Optional[HorizonPlan] = None   # optimizador hold/swing
    vol_edge: Optional[VolEdge] = None           # edge de volatilidad (v2.2)


# ── Funciones puras (testeables) ──────────────────────────────────────────────

def _probability(result: RateResult) -> float:
    """P(cobrar la tasa publicada).

    Prioridad: PoP física del edge (prob_profit) cuando existe — más honesta
    porque usa la vol realizada y el breakeven real. Si no, fallback al delta CRR.
    """
    # PoP física calculada por core/edge (usa vol realizada + breakeven real)
    if result.prob_profit is not None:
        return min(max(result.prob_profit, 0.0), 1.0)

    # Fallback: delta CRR
    if result.delta is None:
        if result.moneyness == "ITM":
            return 0.70
        if result.moneyness == "ATM":
            return 0.55
        return 0.40
    if result.strategy == "COVERED_CALL":
        return min(max(result.delta, 0.0), 1.0)
    return min(max(1.0 - abs(result.delta), 0.0), 1.0)


def _passes_gates(result: RateResult, profile: RiskProfile) -> bool:
    """Gates duros: fallo en cualquiera descarta la oportunidad."""
    if not result.is_liquid:
        return False
    if result.spread_vs_caucion_pct <= 0:
        return False
    if result.days < _MIN_DAYS or result.days > _MAX_DAYS:
        return False
    if result.cushion_pct < _MIN_CUSHION[profile]:
        return False
    if _probability(result) < _MIN_PROB[profile]:
        return False
    # Gate blando de EV: si tenemos expectativa calculada, rechazar EV negativa
    # en perfiles no-agresivos. AGRESIVO acepta cualquier EV (especulativo).
    if result.expected_value_pct is not None:
        if profile != RiskProfile.AGRESIVO and result.expected_value_pct < 0.0:
            return False
    return True


def _score(result: RateResult, profile: RiskProfile) -> float:
    """Score 0–100: suma ponderada EV-centric (v2.4).

    EV anualizado y VRP son los drivers principales. El spread vs caución
    pasó a gate duro — ya no contamina el ranking con primas nominalmente
    altas pero sin edge real.
    """
    w = _WEIGHTS[profile]

    # EV factor: rango [-60%, +60%] ann → [0, 1]; None → neutro 0.5
    ev_raw = result.expected_value_pct if result.expected_value_pct is not None else 0.0
    ev_f = (min(max(ev_raw, -60.0), 60.0) / 60.0 + 1.0) / 2.0

    # VRP factor: range [-15pp, +15pp] → [0, 1]; sin vol_edge → neutro 0.5
    vrp = 0.0
    if result.vol_edge is not None and result.vol_edge.vrp is not None:
        vrp = result.vol_edge.vrp
    vol_f = (min(max(vrp, -0.15), 0.15) / 0.15 + 1.0) / 2.0

    cushion_f = min(max(result.cushion_pct, 0.0), 15.0) / 15.0
    prob_f = _probability(result)
    liq_f = 1.0 if result.is_liquid else 0.4

    # IV-rank timing (vendedor de prima): vender cuando la IV está históricamente
    # cara. iv_rank 0..100; >50 sube el score, <50 lo baja. ±6pts, no es gate.
    # Si no hay historial suficiente, iv_rank_pct es None y no afecta nada.
    iv_rank_boost = 0.0
    if result.vol_edge is not None:
        ivr = getattr(result.vol_edge, "iv_rank_pct", None)
        if ivr is not None:
            iv_rank_boost = (float(ivr) - 50.0) / 50.0 * 0.06

    raw = (
        w["ev"] * ev_f
        + w["vol"] * vol_f
        + w["cushion"] * cushion_f
        + w["probability"] * prob_f
        + w["liquidity"] * liq_f
    )
    raw = min(max(raw + iv_rank_boost, 0.0), 1.0)

    # Bonus delta band: +5pts cuando el delta de la opción escrita cae en la
    # banda objetivo del perfil (no es gate — no puede forzar la eliminación).
    if result.delta is not None:
        lo, hi = _DELTA_BANDS[profile]
        if lo <= abs(result.delta) <= hi:
            raw = min(raw + 0.05, 1.0)

    # Bonus mispricing: +5pts cuando el strike cotiza >2pp caro vs la sonrisa IV.
    # Peso pequeño (0.05) — refinamiento, no driver principal del ranking.
    if result.mispricing_pp is not None and result.mispricing_pp > 2.0:
        raw = min(raw + 0.05, 1.0)

    # Penalización evento: -8pts cuando el vencimiento cruza un evento conocido.
    if settings.events_enabled:
        ev = spans_event(date.today(), result.expiration)
        if ev is not None:
            raw = max(raw - 0.08, 0.0)

    return round(raw * 100.0, 1)


def _semaphore(result: RateResult, score: float, checker: RiskChecker) -> str:
    approved, _ = checker.check_opportunity(result)
    prob = _probability(result)
    if approved and prob >= 0.65 and score >= 60:
        return "verde"
    if score >= 30:
        return "amarillo"
    return "rojo"


# ── Helpers de presentación ───────────────────────────────────────────────────

def _headline(result: RateResult, score: float) -> str:
    strat = "Lanzamiento cubierto" if result.strategy == "COVERED_CALL" else "Venta de put"
    return (
        f"{strat} — {result.symbol}  |  TNA {result.tna_pct:.1f}%  |  "
        f"Score {score:.0f}/100"
    )


def _explain(result: RateResult, prob: float, caucion_tna: float) -> str:
    """Una línea de por qué esta opción. Conciso, comparativo, accionable."""
    breakeven = result.spot * (1.0 - result.cushion_pct / 100.0)
    spread_vs_caucion = result.tna_pct - caucion_tna
    edge_txt = (
        f"+{spread_vs_caucion:.1f}pp sobre caución"
        if spread_vs_caucion > 0
        else f"{spread_vs_caucion:.1f}pp vs caución"
    )
    # Separador "  ·  " reservado para el texto base.
    # Las notas SMC se agregan con " | " en _build_recommendation para que el
    # dashboard pueda separarlas al renderizar tags de contexto.
    if result.strategy == "COVERED_CALL":
        return (
            f"Call {result.symbol} K=${result.strike:,.0f} · vence {result.days}d  ·  "
            f"TNA {result.tna_pct:.1f}% ({edge_txt})  ·  "
            f"Colchon {result.cushion_pct:.1f}% hasta ${breakeven:,.0f}  ·  "
            f"Prob. {prob*100:.0f}% de cobrar prima completa"
        )
    return (
        f"Put {result.symbol} K=${result.strike:,.0f} · vence {result.days}d  ·  "
        f"TNA {result.tna_pct:.1f}% ({edge_txt})  ·  "
        f"Colchon {result.cushion_pct:.1f}% hasta ${breakeven:,.0f}  ·  "
        f"Prob. {prob*100:.0f}% — si bajan a K compras GGAL barato"
    )

def _intention(result: RateResult) -> str:
    return "Alcista / Neutral (Se beneficia si la accion sube o se mantiene estable)."

def _win_scenario(result: RateResult) -> str:
    if result.strategy == "COVERED_CALL":
        return f"Ganas si GGAL cierra por encima de ${result.strike:,.0f} al vencimiento. Obtienes la rentabilidad maxima de la operacion."
    return f"Ganas si GGAL cierra por encima de ${result.strike:,.0f} al vencimiento. Te quedas con el 100% de la prima cobrada sin comprar la accion."

def _lose_scenario(result: RateResult) -> str:
    breakeven = result.spot * (1.0 - result.cushion_pct / 100.0)
    if result.strategy == "COVERED_CALL":
        return f"Pierdes si GGAL cae fuertemente y perfora los ${breakeven:,.0f} (tu colchon del {result.cushion_pct:.1f}%). La perdida acompaña la caida de la accion a partir de ese punto."
    return f"Pierdes si GGAL cae fuertemente y perfora los ${breakeven:,.0f}. Se te obligara a comprar la accion a un costo neto mayor al precio de mercado en ese momento."


def _action_steps(result: RateResult, contracts: int) -> list[str]:
    expiry_str   = result.expiration.strftime("%d/%m/%Y")
    breakeven    = result.spot * (1.0 - result.cushion_pct / 100.0)
    n_contracts  = max(contracts, 1)
    lote_str     = f"{n_contracts} contrato{'s' if n_contracts > 1 else ''}"
    prima_total  = result.premium * n_contracts * 100

    if result.strategy == "COVERED_CALL":
        # Step 1 = la orden concreta lista para ejecutar en el broker
        step1 = (
            f"VENDER {lote_str} call {result.symbol}  "
            f"precio limite ${result.premium:,.2f}  "
            f"(prima total ~${prima_total:,.0f})"
        )
        return [
            step1,
            f"Compra {n_contracts * 100:,} acciones de GGAL a ~${result.spot:,.2f} (si no las tenes ya)",
            f"TNA objetivo: {result.tna_pct:.1f}%  |  Break-even: ${breakeven:,.0f}  "
            f"|  Colchon: {result.cushion_pct:.1f}%",
            f"Vencimiento {expiry_str}  ({result.days} dias) — dejar correr hasta ahí o TP al 50%",
        ]

    capital_total = result.net_outlay * n_contracts * 100
    # Step 1 = la orden concreta
    step1 = (
        f"VENDER {lote_str} put {result.symbol}  "
        f"precio limite ${result.premium:,.2f}  "
        f"(prima total ~${prima_total:,.0f})"
    )
    return [
        step1,
        f"Reservar ${capital_total:,.0f} en efectivo como garantia (colchon: {result.cushion_pct:.1f}%)",
        f"TNA objetivo: {result.tna_pct:.1f}%  |  Break-even si ejercen: ${breakeven:,.0f}",
        f"Vencimiento {expiry_str}  ({result.days} dias) — dejar correr o recomprar al 50% de captura",
    ]


def _ticket(result: RateResult, contracts: int) -> str:
    expiry_str = result.expiration.strftime("%d/%m/%Y")
    if result.strategy == "COVERED_CALL":
        qty_stock = contracts * 100
        lines = [
            "--- TICKET LANZAMIENTO CUBIERTO ---",
            f"COMPRA : GGAL            x{qty_stock:<6,}  Precio limite: ${result.spot:,.2f}",
            f"VENTA  : {result.symbol:<12}  x{contracts:<6}  Precio limite: ${result.premium:,.2f}",
            f"Vencimiento: {expiry_str}  ({result.days} dias)",
            f"TNA esperada: {result.tna_pct:.1f}%  |  Colchon: {result.cushion_pct:.1f}%",
        ]
    else:
        capital_total = result.net_outlay * contracts * 100
        lines = [
            "--- TICKET VENTA DE PUT ---",
            f"VENTA  : {result.symbol:<12}  x{contracts:<6}  Precio limite: ${result.premium:,.2f}",
            f"Garantia requerida: ${capital_total:,.0f}",
            f"Vencimiento: {expiry_str}  ({result.days} dias)",
            f"TNA esperada: {result.tna_pct:.1f}%  |  Colchon: {result.cushion_pct:.1f}%",
        ]
    return "\n".join(lines)


# ── Pre-pass de enriquecimiento cuantitativo ─────────────────────────────────

def _enrich_edges(
    results: list[RateResult],
    spot_history: list[float],
    smile_map: Optional[dict] = None,
) -> None:
    """Enriquece cada RateResult in-place con VolEdge, PoP, EV y mispricing (v3.1).

    smile_map: dict[expiry_code → SmileFit] de core/iv_surface.build_smile_map.
    Si falla cualquier enriquecimiento, deja el campo en None — comportamiento
    idéntico a versiones anteriores.
    """
    import re
    from optionsdesk.core.edge import enrich_rate_result
    _sym_re = re.compile(r"^GFG([CV])(\d+(?:[.,]\d+)?)([A-Z]+)$")

    # IV-rank timing: cargamos el historial de IV ATM para que el VolEdge sepa si la
    # IV de hoy está cara o barata respecto de su propia historia.
    iv_history: list[float] = []
    try:
        from optionsdesk.data.iv_history import load_iv_history
        iv_history = load_iv_history("GGAL")
    except Exception as e:
        logger.debug("load_iv_history fallo: %s", e)

    for result in results:
        try:
            enrich_rate_result(result, spot_history, iv_history or None)
        except Exception as e:
            logger.warning("enrich_rate_result fallo para %s: %s", result.symbol, e)
    # Nota: la PERSISTENCIA de la IV ATM diaria se hace en la capa live (dashboard),
    # no acá, para que el recommender quede read-only y los tests no escriban a disco.

        # Mispricing vs sonrisa IV (v3.1)
        if smile_map and result.iv is not None:
            try:
                from optionsdesk.core.iv_surface import mispricing_score
                m = _sym_re.match(result.symbol)
                expiry = m.group(3) if m else None
                if expiry and expiry in smile_map:
                    strike_str = m.group(2).replace(",", ".") if m else None
                    if strike_str:
                        strike = float(strike_str)
                        result.mispricing_pp = mispricing_score(strike, result.iv, smile_map[expiry])
            except Exception as e:
                logger.warning("mispricing_score fallo para %s: %s", result.symbol, e)


# ── Proxy liviano para size_position ─────────────────────────────────────────

class _SizingProxy:
    """Adapta un RateResult a la interfaz que espera size_position(capital, rec).

    size_position espera `rec.result` — este proxy wrappea el RateResult directamente.
    """
    __slots__ = ("result",)

    def __init__(self, result: "RateResult") -> None:
        self.result = result


# ── Motor principal ───────────────────────────────────────────────────────────

class Recommender:
    """Para cada RiskProfile elige la mejor oportunidad de CC o SP."""

    def __init__(self, checker: Optional[RiskChecker] = None) -> None:
        self._checker = checker or RiskChecker()

    def recommend(
        self,
        covered_calls: list[RateResult],
        short_puts: list[RateResult],
        benchmark: Benchmark,
        context=None,               # MarketContext opcional
        capital: Optional[float] = None,
        spot_history: Optional[list[float]] = None,   # para VolEdge
        chain=None,                 # OptionsChain — para smile IV (v3.1)
        top_n: int = 3,
    ) -> dict[RiskProfile, list[Recommendation]]:
        """Devuelve hasta `top_n` recomendaciones por perfil (lista vacía si no hay)."""
        all_results = list(covered_calls) + list(short_puts)
        caucion_tna = benchmark.caucion_tna_pct

        # Pre-pass: enriquecer con VolEdge/PoP/EV + mispricing (v3.1).
        # La sonrisa se fitea sobre all_results (que ya tienen IV computada),
        # no sobre la cadena cruda (Quote no tiene IV).
        smile_map = None
        spot_val = all_results[0].spot if all_results else 0.0
        if all_results and spot_val > 0:
            try:
                from optionsdesk.core.iv_surface import build_smile_map
                smile_map = build_smile_map(all_results, spot_val)
            except Exception as e:
                logger.warning("build_smile_map fallo: %s", e)

        if spot_history or smile_map:
            _enrich_edges(all_results, spot_history or [], smile_map)

        return {
            profile: self._top_for_profile(
                all_results, profile, caucion_tna, capital, context,
                spot_history, top_n,
            )
            for profile in RiskProfile
        }

    def _top_for_profile(
        self,
        all_results: list[RateResult],
        profile: RiskProfile,
        caucion_tna: float,
        capital: Optional[float],
        context,
        spot_history: Optional[list[float]],
        top_n: int,
    ) -> list[Recommendation]:
        candidates = [r for r in all_results if _passes_gates(r, profile)]
        if not candidates:
            return []

        scored = [(r, _score(r, profile)) for r in candidates]
        if context is not None:
            scored = _apply_directional_boost(scored, context, profile)
        scored.sort(key=lambda x: x[1], reverse=True)

        # Solo candidatos no-rojos
        non_red = [
            (r, s) for r, s in scored
            if _semaphore(r, s, self._checker) != "rojo"
        ]
        if not non_red:
            return []

        diverse = _diversify(non_red, n=top_n)

        recommendations: list[Recommendation] = []
        for result, score_val in diverse:
            rec = self._build_recommendation(
                result, score_val, profile, caucion_tna, capital, spot_history, context
            )
            if rec is not None:
                recommendations.append(rec)

        return recommendations

    def _build_recommendation(  # noqa: PLR0912,PLR0915
        self,
        result: RateResult,
        score_val: float,
        profile: RiskProfile,
        caucion_tna: float,
        capital: Optional[float],
        spot_history: Optional[list[float]],
        context=None,
    ) -> Optional[Recommendation]:
        light = _semaphore(result, score_val, self._checker)
        prob  = _probability(result)
        _, warnings = self._checker.check_opportunity(result)

        # v2.3: warning cuando los timeframes estan en conflicto
        if context is not None:
            alignment = getattr(context, "mtf_alignment", "neutral")
            htf_trend = getattr(context, "htf_trend", None)
            if alignment == "conflicto" and htf_trend:
                dir_htf = "alcista" if htf_trend == "alcista" else "bajista"
                warnings = list(warnings) + [
                    f"Semanal {dir_htf} / diario en direccion contraria — contexto en conflicto"
                ]

        contracts = 0
        expected_profit: Optional[float] = None
        if capital and capital > 0 and result.net_outlay > 0:
            # Kelly fraccional si hay EV+PoP; fallback a fixed fractional.
            try:
                from optionsdesk.portfolio.sizing import size_position as _size_pos
                contracts = _size_pos(capital, rec=_SizingProxy(result))
            except Exception as e:
                logger.warning("size_position fallo, usando fixed fractional: %s", e)
                contracts = max(0, math.floor(capital * 0.20 / (result.net_outlay * 100)))
            # Garantía mínima: si Kelly/cap devuelve 0 pero el capital total
            # alcanza para 1 lote, mostrar 1 (indicativo, sin violar el budget).
            expected_profit = (
                contracts * result.net_outlay * 100 * result.period_rate_pct / 100.0
            )

        n = max(contracts, 0)

        try:
            plan = optimize_horizon(result, profile.value, caucion_tna=caucion_tna)
        except Exception as e:
            logger.warning("optimize_horizon fallo para %s: %s", result.symbol, e)
            plan = None

        # VolEdge para display: ya enriquecido en-place por _enrich_edges antes del sort.
        vol_edge: Optional[VolEdge] = result.vol_edge  # type: ignore[assignment]

        # Warning de evento: penalización ya incluida en el score, aquí solo aviso textual.
        if settings.events_enabled:
            ev = spans_event(date.today(), result.expiration)
            if ev is not None:
                warnings = list(warnings) + [
                    f"El vencimiento cruza {ev.description} ({ev.event_date}) — mayor incertidumbre."
                ]

        # Enriquecer la explicación con contexto SMC (zona PDA + killzone)
        plain = _explain(result, prob, caucion_tna)
        smc_for_rec = getattr(context, "smc", None) if context is not None else None
        if smc_for_rec is not None:
            _smc_notes: list[str] = []
            _pda_rec = getattr(smc_for_rec, "pda", None)
            if _pda_rec is not None:
                _zone_rec = getattr(_pda_rec, "zone", None)
                _zone_labels = {
                    "discount":     "Precio en zona DISCOUNT (institucional favorable para puts).",
                    "premium":      "Precio en zona PREMIUM (institucional favorable para calls cubiertos).",
                    "equilibrium":  "Precio en equilibrio (0.5 fib) — esperar confirmación.",
                }
                if _zone_rec in _zone_labels:
                    _smc_notes.append(_zone_labels[_zone_rec])
            _kz_rec = getattr(smc_for_rec, "killzone", None)
            if _kz_rec == "manipulacion":
                _smc_notes.append("Apertura BYMA (11–12h): zona de manipulación — spreads amplios, esperar desarrollo.")
            elif _kz_rec == "distribucion":
                _smc_notes.append("Cierre BYMA (15:30–17h): zona de distribución — posible movimiento final del día.")
            _adr_rec = getattr(smc_for_rec, "adr_confluence", None)
            if _adr_rec == "fx_driven":
                _smc_notes.append("Movimiento en pesos probablemente impulsado por el dólar (CCL), no por el activo.")
            elif _adr_rec == "confirmed":
                _smc_notes.append("ADR en NYSE confirma el movimiento del activo subyacente.")
            if _smc_notes:
                plain = plain + " | " + " | ".join(_smc_notes)

        return Recommendation(
            result=result,
            profile=profile,
            score=round(score_val, 1),
            light=light,
            headline=_headline(result, score_val),
            plain_explanation=plain,
            intention=_intention(result),
            win_scenario=_win_scenario(result),
            lose_scenario=_lose_scenario(result),
            action_steps=_action_steps(result, n),
            success_probability=prob,
            expected_profit_ars=expected_profit,
            ticket_text=_ticket(result, n),
            warnings=warnings,
            contracts=contracts,
            horizon_plan=plan,
            vol_edge=vol_edge,
        )


def _diversify(
    scored: list[tuple[RateResult, float]],
    n: int = 3,
) -> list[tuple[RateResult, float]]:
    """Elige hasta n candidatos diversos.

    Descarta un candidato si:
    - Tiene el mismo símbolo que alguno ya elegido, O
    - Misma estrategia + mismo vencimiento + strike dentro del 3% de alguno ya elegido.
    """
    selected: list[tuple[RateResult, float]] = []
    for r, s in scored:
        if len(selected) >= n:
            break
        if _is_redundant(r, [x for x, _ in selected]):
            continue
        selected.append((r, s))
    return selected


def _is_redundant(candidate: RateResult, chosen: list[RateResult]) -> bool:
    for c in chosen:
        if candidate.symbol == c.symbol:
            return True
        same_strat   = candidate.strategy == c.strategy
        same_expiry  = candidate.expiration == c.expiration
        strike_close = abs(candidate.strike - c.strike) / max(c.strike, 1) < 0.03
        if same_strat and same_expiry and strike_close:
            return True
    return False


def _apply_directional_boost(
    scored: list[tuple[RateResult, float]],
    context,
    profile: RiskProfile,
) -> list[tuple[RateResult, float]]:
    """Ajuste de score por tendencia, MTF y SMC (v2.4).

    MTF alignment (v2.3):
    - alineado_alcista/bajista → boost mayor (±6/5 pts)
    - conflicto → penalizacion para la estrategia contra el HTF
    - sin alineacion → fallback v2.2 (±4/3 pts por tendencia diaria)

    SMC overlay (v2.4) — ajustes adicionales no excluyentes (±3 pts max):
    - Precio en discount + cerca de SSL → +SHORT_PUT (comprar en el suelo)
    - Precio en premium + cerca de BSL → +COVERED_CALL (vender en el techo)
    - Sweep de SSL con reversión alcista → +SHORT_PUT (stop-hunt alcista)
    - Sweep de BSL con reversión bajista → +COVERED_CALL (stop-hunt bajista)

    Nunca elimina una recomendacion (informa, no bloquea).
    """
    trend      = getattr(context, "trend",         None)
    confidence = getattr(context, "confidence",   "sin datos")
    alignment  = getattr(context, "mtf_alignment", "neutral")
    htf_trend  = getattr(context, "htf_trend",    trend)

    if confidence in ("sin datos", "baja") or trend is None:
        return scored

    boosted = []
    for result, s in scored:
        adj = 0.0

        # ── MTF boost (v2.3) ─────────────────────────────────────────────────
        if alignment == "alineado_alcista":
            if result.strategy == "SHORT_PUT":
                adj += 6.0
        elif alignment == "alineado_bajista":
            if result.strategy == "COVERED_CALL":
                adj += 5.0
        elif alignment == "conflicto":
            if htf_trend == "bajista" and result.strategy == "SHORT_PUT":
                adj -= 4.0
            elif htf_trend == "alcista" and result.strategy == "COVERED_CALL":
                adj -= 3.0
        else:
            if trend == "alcista" and result.strategy == "SHORT_PUT":
                adj += 4.0
            elif trend == "bajista" and result.strategy == "COVERED_CALL":
                adj += 3.0

        # ── SmcSignal boost (v2.5) — usa el output del cascade multi-TF ──────────
        smc = getattr(context, "smc", None)
        if smc is not None:
            sig = getattr(smc, "signal", None)
            if sig is not None:
                # SmcSignal.quality calibra la magnitud del boost
                _QUALITY_BOOST = {"S": 8.0, "A": 6.0, "B": 4.0, "C": 2.0, "none": 0.0}
                q_boost = _QUALITY_BOOST.get(getattr(sig, "quality", "none"), 0.0)
                favors_sp = getattr(sig, "favors_short_put", False)
                favors_cc = getattr(sig, "favors_covered_call", False)
                favors_wait = getattr(sig, "favors_wait", True)

                if favors_sp and result.strategy == "SHORT_PUT":
                    adj += q_boost
                elif favors_cc and result.strategy == "COVERED_CALL":
                    adj += q_boost * 0.85   # CC levemente menos agressivo
                elif favors_wait:
                    # Sin señal clara → reducir confianza (no bloquear, solo enfriar)
                    adj -= 1.5
            else:
                # Fallback al boost v2.4 si no hay SmcSignal
                smc_adj = _smc_boost(result, smc)
                adj += max(-3.0, min(3.0, smc_adj))

        boosted.append((result, s + adj))
    return boosted


def _smc_boost(result: "RateResult", smc: object) -> float:  # type: ignore[type-arg]
    """Boost fallback basado en zona PDA y sweeps (usado cuando SmcSignal no está disponible).

    Mantenido por compatibilidad. Con SmcSignal activo, `_apply_directional_boost`
    usa el path de quality-calibrated boost y este fallback no se llama.
    """
    adj = 0.0
    pda = getattr(smc, "pda", None)
    liquidity = getattr(smc, "liquidity", [])
    sweeps    = getattr(smc, "sweeps", [])
    spot      = result.spot

    if pda is not None:
        zone = getattr(pda, "zone", None)
        if zone == "discount" and result.strategy == "SHORT_PUT":
            adj += 2.0
        elif zone == "premium" and result.strategy == "COVERED_CALL":
            adj += 2.0

        strike = result.strike
        for lv in liquidity:
            lv_price = getattr(lv, "price", 0.0)
            lv_swept = getattr(lv, "swept", True)
            lv_kind  = getattr(lv, "kind", "")
            if lv_swept:
                continue
            dist_pct = abs(strike - lv_price) / max(lv_price, 1e-9)
            if dist_pct < 0.03:
                if lv_kind == "SSL" and result.strategy == "SHORT_PUT":
                    adj += 1.0
                elif lv_kind == "BSL" and result.strategy == "COVERED_CALL":
                    adj += 1.0

    for sw in sweeps:
        sw_reversed  = getattr(sw, "reversed", False)
        sw_direction = getattr(sw, "direction", "")
        if not sw_reversed:
            continue
        if sw_direction == "down" and result.strategy == "SHORT_PUT":
            adj += 1.0
        elif sw_direction == "up" and result.strategy == "COVERED_CALL":
            adj += 1.0

    return adj
