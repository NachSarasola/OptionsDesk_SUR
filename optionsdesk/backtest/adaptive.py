"""Motor adaptativo — gestión dinámica de riesgo basada en performance real.

Aprende de los trades cerrados para ajustar:
  - Tamaño de posición (Half-Kelly basado en win_rate y R/R real)
  - Modo operativo (NORMAL / DEFENSIVE / AGGRESSIVE / HALTED)
  - Score mínimo requerido para abrir nuevas posiciones
  - Circuit breaker: pausa automática si 3 stops consecutivos O drawdown >= 5%

Nunca genera órdenes. Solo provee parámetros para que stock_demo y options_demo
los apliquen. Si no hay historial suficiente, opera en modo NORMAL conservador.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AdaptiveContext:
    """Contexto adaptativo que el demo runner y los demos consumen en cada tick."""

    # ── Estadísticas base ──────────────────────────────────────────────────
    win_rate: float = 0.5
    consecutive_losses: int = 0
    trades_analyzed: int = 0

    # ── Modo operativo ─────────────────────────────────────────────────────
    # NORMAL     → sizing estándar (Half-Kelly)
    # DEFENSIVE  → sizing reducido (Quarter-Kelly), score mínimo elevado
    # AGGRESSIVE → sizing ampliado (hasta Kelly completo), score mínimo relajado
    # HALTED     → circuit breaker activo, NO abrir nuevas posiciones
    mode: str = "NORMAL"

    # ── Sizing ────────────────────────────────────────────────────────────
    half_kelly_pct: float = 0.05   # fracción del capital base a arriesgar por trade

    # ── Calidad mínima para entrar ────────────────────────────────────────
    # Señales con score < min_score_for_entry se descartan sin importar el setup
    min_score_for_entry: float = 65.0

    # ── Drawdown ──────────────────────────────────────────────────────────
    current_drawdown_pct: float = 0.0   # drawdown actual sobre el capital base
    peak_equity: float = 0.0            # equity máximo alcanzado en la ventana
    halted: bool = False                # True → circuit breaker activo

    # ── Ajuste de stops por modo ──────────────────────────────────────────
    # Fracción del ATR que se agrega (negativo = más ceñido) al stop.
    # DEFENSIVE → -0.5×ATR (stop más ceñido, protección adicional)
    # AGGRESSIVE → +0.3×ATR (algo más de respiro, evita ser cortado en el ruido)
    stop_atr_adjust: float = 0.0

    # ── Metadata ──────────────────────────────────────────────────────────
    reason: str = ""   # por qué entró en este modo (para mostrar en dashboard)


from optionsdesk.config.settings import settings

def analyze_performance(
    trades: list,
    *,
    limit: int = 20,
    capital_base: float = 1_800_000.0,
    cb_consecutive_losses: Optional[int] = None,
    cb_drawdown_pct: Optional[float] = None,
) -> AdaptiveContext:
    """Analiza los últimos `limit` trades y devuelve un AdaptiveContext calibrado.

    Parámetros
    ----------
    trades            : Lista de objetos con atributo `.pnl_ars` (float).
                        Acepta StockDemoTrade, OptionsDemoTrade, o cualquier
                        objeto/namespace con dicho atributo.
    limit             : Ventana de trades a analizar (default 20).
    capital_base      : Capital de referencia para calcular drawdown % y Kelly.
    cb_consecutive_losses : Stops consecutivos que disparan el circuit breaker.
    cb_drawdown_pct   : % de drawdown sobre capital_base que dispara el halt.

    Retorna
    -------
    AdaptiveContext con mode, half_kelly_pct, min_score_for_entry, halted, etc.
    """
    _cb_consecutive_losses = cb_consecutive_losses if cb_consecutive_losses is not None else settings.circuit_breaker_consecutive_losses
    _cb_drawdown_pct = cb_drawdown_pct if cb_drawdown_pct is not None else settings.circuit_breaker_drawdown_pct

    if not trades:
        return AdaptiveContext(
            win_rate=0.5,
            consecutive_losses=0,
            trades_analyzed=0,
            mode="NORMAL",
            half_kelly_pct=0.05,
            min_score_for_entry=65.0,
            reason="sin historial — modo neutro",
        )

    recent = list(trades)[-limit:]
    n = len(recent)

    # ── Win rate ──────────────────────────────────────────────────────────
    wins = sum(1 for t in recent if _pnl(t) > 0)
    win_rate = wins / n

    # ── Pérdidas consecutivas (desde el final de la serie) ────────────────
    consecutive_losses = 0
    for t in reversed(recent):
        if _pnl(t) <= 0:
            consecutive_losses += 1
        else:
            break

    # ── Equity curve y drawdown ───────────────────────────────────────────
    equity = 0.0
    peak = 0.0
    trough_since_peak = 0.0
    for t in recent:
        equity += _pnl(t)
        if equity > peak:
            peak = equity
        drawdown = peak - equity
        if drawdown > trough_since_peak:
            trough_since_peak = drawdown

    # Drawdown como % del capital base (no como % del equity pico)
    current_dd_pct = (trough_since_peak / capital_base) * 100.0 if capital_base > 0 else 0.0

    # ── Circuit breaker ───────────────────────────────────────────────────
    halted = False
    halt_reason = ""
    if consecutive_losses >= _cb_consecutive_losses:
        halted = True
        halt_reason = f"{consecutive_losses} stops consecutivos (límite {_cb_consecutive_losses})"
    elif current_dd_pct >= _cb_drawdown_pct:
        halted = True
        halt_reason = f"drawdown {current_dd_pct:.1f}% ≥ {_cb_drawdown_pct}% del capital"

    # Si está halted y el último trade fue ganador → reactivar automáticamente
    if halted and _pnl(recent[-1]) > 0:
        halted = False
        halt_reason = ""

    if halted:
        return AdaptiveContext(
            win_rate=win_rate,
            consecutive_losses=consecutive_losses,
            trades_analyzed=n,
            mode="HALTED",
            half_kelly_pct=0.0,
            min_score_for_entry=100.0,   # bloquea todo
            current_drawdown_pct=current_dd_pct,
            peak_equity=peak,
            halted=True,
            stop_atr_adjust=0.0,
            reason=halt_reason,
        )

    # ── Average R/R real (de los trades con datos) ────────────────────────
    avg_rr = 2.0
    if wins > 0 and n > wins:
        avg_win = sum(_pnl(t) for t in recent if _pnl(t) > 0) / wins
        losses_sum = sum(abs(_pnl(t)) for t in recent if _pnl(t) <= 0)
        n_losses = n - wins
        avg_loss = losses_sum / n_losses if n_losses > 0 else 1.0
        if avg_loss > 0:
            avg_rr = avg_win / avg_loss

    # ── Fractional Kelly ─────────────────────────────────────────────────
    kelly = win_rate - ((1.0 - win_rate) / max(avg_rr, 0.1))
    half_kelly = kelly / 2.0

    # ── Modo y parámetros derivados ───────────────────────────────────────
    if win_rate < 0.40 or consecutive_losses >= 2:
        mode = "DEFENSIVE"
        # Quarter-Kelly en modo defensivo (la mitad del half-Kelly)
        sized_kelly = max(min(half_kelly / 2.0, 0.07), 0.005)
        min_score = 78.0
        stop_adj = -0.5   # stop más ceñido (½ ATR más cerca)
        reason = (
            f"win_rate {win_rate:.0%} < 40% o {consecutive_losses} losses seguidos"
            if win_rate < 0.40 else
            f"{consecutive_losses} stops consecutivos — modo cautela"
        )
    elif win_rate >= 0.60 and consecutive_losses == 0 and n >= 10:
        mode = "AGGRESSIVE"
        # Kelly completo (doble del half-Kelly), tope 15%
        sized_kelly = max(min(half_kelly * 2.0, 0.15), 0.02)
        min_score = 60.0
        stop_adj = 0.3    # ⅓ ATR extra de respiro
        reason = f"win_rate {win_rate:.0%} ≥ 60% con {n} trades — modo agresivo"
    else:
        mode = "NORMAL"
        sized_kelly = max(min(half_kelly, 0.12), 0.01)
        min_score = 65.0
        stop_adj = 0.0
        reason = f"win_rate {win_rate:.0%}, {n} trades analizados"

    return AdaptiveContext(
        win_rate=win_rate,
        consecutive_losses=consecutive_losses,
        trades_analyzed=n,
        mode=mode,
        half_kelly_pct=round(sized_kelly, 4),
        min_score_for_entry=min_score,
        current_drawdown_pct=round(current_dd_pct, 2),
        peak_equity=round(peak, 2),
        halted=False,
        stop_atr_adjust=stop_adj,
        reason=reason,
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def _pnl(trade) -> float:
    """Extrae pnl_ars de cualquier objeto trade (dataclass, namedtuple, dict, etc.)."""
    if isinstance(trade, dict):
        return float(trade.get("pnl_ars", 0.0))
    return float(getattr(trade, "pnl_ars", 0.0))
