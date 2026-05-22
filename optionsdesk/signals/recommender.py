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

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from optionsdesk.core.benchmark import Benchmark
from optionsdesk.core.rates import RateResult
from optionsdesk.risk.limits import RiskChecker
from optionsdesk.signals.horizon import HorizonPlan, optimize_horizon
from optionsdesk.signals.volatility import VolEdge


class RiskProfile(str, Enum):
    CONSERVADOR = "CONSERVADOR"
    EQUILIBRADO = "EQUILIBRADO"
    AGRESIVO = "AGRESIVO"


# Pesos de scoring y gates por perfil
_WEIGHTS: dict[RiskProfile, dict[str, float]] = {
    RiskProfile.CONSERVADOR: {
        "spread": 0.15, "cushion": 0.45, "probability": 0.30, "liquidity": 0.10
    },
    RiskProfile.EQUILIBRADO: {
        "spread": 0.30, "cushion": 0.30, "probability": 0.25, "liquidity": 0.15
    },
    RiskProfile.AGRESIVO: {
        "spread": 0.50, "cushion": 0.15, "probability": 0.20, "liquidity": 0.15
    },
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
    """P(cobrar la tasa publicada) usando delta CRR.

    Covered call: P(S_T >= K) ≈ delta del call.
    Short put: P(put expira OTM) = 1 − |delta del put|.
    Fallback heurístico si delta no está disponible.
    """
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
    return True


def _score(result: RateResult, profile: RiskProfile) -> float:
    """Score 0–100: suma ponderada de cuatro factores normalizados."""
    w = _WEIGHTS[profile]
    # spread con rendimientos decrecientes: cap en 60 p.p. para que un 500% OTM
    # no aplaste a un 90% ITM sólido
    spread_f = min(max(result.spread_vs_caucion_pct, 0.0), 60.0) / 60.0
    cushion_f = min(max(result.cushion_pct, 0.0), 15.0) / 15.0
    prob_f = _probability(result)
    liq_f = 1.0 if result.is_liquid else 0.4
    raw = (
        w["spread"] * spread_f
        + w["cushion"] * cushion_f
        + w["probability"] * prob_f
        + w["liquidity"] * liq_f
    )
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
    breakeven = result.spot * (1.0 - result.cushion_pct / 100.0)
    if result.strategy == "COVERED_CALL":
        return (
            f"Compras GGAL (~${result.spot:,.0f}) y vendes el call {result.symbol}. "
            f"Si en {result.days} dias GGAL cierra arriba de ${result.strike:,.0f} "
            f"— algo con ~{prob * 100:.0f}% de probabilidad — "
            f"cobras {result.tna_pct:.1f}% anual, contra {caucion_tna:.1f}% de la caucion. "
            f"Si baja, tenes un colchon de {result.cushion_pct:.1f}%: "
            f"GGAL puede caer hasta ${breakeven:,.0f} sin perder plata."
        )
    return (
        f"Inmovilizas efectivo como garantia y vendes el put {result.symbol}. "
        f"Si en {result.days} dias GGAL cierra arriba de ${result.strike:,.0f} "
        f"— algo con ~{prob * 100:.0f}% de probabilidad — "
        f"te quedas con la prima: {result.tna_pct:.1f}% anual "
        f"(vs {caucion_tna:.1f}% de caucion). "
        f"Si cae por debajo, compras GGAL a ${breakeven:,.0f} efectivo (ya neto de prima)."
    )


def _action_steps(result: RateResult, contracts: int) -> list[str]:
    expiry_str = result.expiration.strftime("%d/%m/%Y")
    if result.strategy == "COVERED_CALL":
        return [
            f"Compra {contracts * 100:,} acciones de GGAL a ~${result.spot:,.2f}",
            f"Vende {contracts} contrato(s) del call {result.symbol} a ${result.premium:,.2f}",
            f"Deja la posicion abierta hasta el {expiry_str}",
        ]
    capital_total = result.net_outlay * contracts * 100
    return [
        f"Reserva ${capital_total:,.0f} en efectivo como garantia",
        f"Vende {contracts} contrato(s) del put {result.symbol} a ${result.premium:,.2f}",
        (
            f"Vencimiento {expiry_str} — si GGAL baja de ${result.strike:,.0f} "
            f"compras la accion (eso fue planeado)"
        ),
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
        top_n: int = 3,
    ) -> dict[RiskProfile, list[Recommendation]]:
        """Devuelve hasta `top_n` recomendaciones por perfil (lista vacía si no hay)."""
        all_results = list(covered_calls) + list(short_puts)
        caucion_tna = benchmark.caucion_tna_pct
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

    def _build_recommendation(
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
            contracts = max(1, math.floor(capital / (result.net_outlay * 100)))
            expected_profit = (
                contracts * result.net_outlay * 100 * result.period_rate_pct / 100.0
            )

        n = max(contracts, 1)

        try:
            plan = optimize_horizon(result, profile.value, caucion_tna=caucion_tna)
        except Exception:
            plan = None

        # VolEdge liviano: ±2 pts al score (ya calculado, solo para display)
        vol_edge: Optional[VolEdge] = None
        if spot_history:
            try:
                vol_edge = VolEdge.compute(result.iv, spot_history)
                # El ajuste de score ya no afecta el orden (se aplicó antes del sort),
                # pero actualizamos el valor mostrado al usuario.
                if vol_edge.label == "positivo":
                    score_val = min(score_val + 2.0, 100.0)
                elif vol_edge.label == "negativo":
                    score_val = max(score_val - 2.0, 0.0)
            except Exception:
                pass

        return Recommendation(
            result=result,
            profile=profile,
            score=round(score_val, 1),
            light=light,
            headline=_headline(result, score_val),
            plain_explanation=_explain(result, prob, caucion_tna),
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
    """Ajuste de score por tendencia y alineacion multi-timeframe.

    La magnitud escala con mtf_alignment (v2.3):
    - alineado_alcista/bajista → boost mayor (±6/5 pts)
    - conflicto → penalizacion para la estrategia que va contra el HTF
    - sin alineacion → comportamiento v2.2 (±4/3 pts segun tendencia diaria)
    Nunca elimina una recomendacion (informa, no bloquea).
    """
    trend     = getattr(context, "trend",         None)
    confidence = getattr(context, "confidence",   "sin datos")
    alignment  = getattr(context, "mtf_alignment", "neutral")
    htf_trend  = getattr(context, "htf_trend",    trend)

    if confidence in ("sin datos", "baja") or trend is None:
        return scored

    boosted = []
    for result, s in scored:
        adj = 0.0

        if alignment == "alineado_alcista":
            if result.strategy == "SHORT_PUT":
                adj = 6.0   # mercado alineado alcista → vender puts es optimo
        elif alignment == "alineado_bajista":
            if result.strategy == "COVERED_CALL":
                adj = 5.0   # mercado alineado bajista → CC con buen colchon
        elif alignment == "conflicto":
            # Penaliza la estrategia opuesta al HTF
            if htf_trend == "bajista" and result.strategy == "SHORT_PUT":
                adj = -4.0  # semanal baja → vender puts arriesgado
            elif htf_trend == "alcista" and result.strategy == "COVERED_CALL":
                adj = -3.0  # semanal sube → CC limita la ganancia
        else:
            # Fallback v2.2: solo tendencia diaria
            if trend == "alcista" and result.strategy == "SHORT_PUT":
                adj = 4.0
            elif trend == "bajista" and result.strategy == "COVERED_CALL":
                adj = 3.0

        boosted.append((result, s + adj))
    return boosted
