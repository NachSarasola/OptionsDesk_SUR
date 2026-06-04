"""Pricing de opciones: Black-Scholes (europeas) y CRR binomial (americanas).

Black-Scholes: fórmulas cerradas + griegas analíticas.
CRR: árbol binomial recombinante con modelo de dividendos en fideicomiso (Hull),
     correcto para opciones americanas con dividendos discretos (ej. GGAL).
Griegas CRR: diferencias finitas (funciona para americano + dividendos).
IV: solver de Brent vía scipy.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


# ── Black-Scholes-Merton (Europeas, dividend yield continuo) ──────────────────

def bs_price(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str, q: float = 0.0,
) -> float:
    """Precio BSM para una opción europea.

    S: spot, K: strike, T: años a vencimiento, r: tasa libre de riesgo (continua),
    sigma: volatilidad, option_type: 'C' o 'P', q: dividend yield continuo.
    """
    if T <= 0:
        return max(S - K, 0.0) if option_type == "C" else max(K - S, 0.0)
    if sigma <= 0:
        # sigma=0: sin incertidumbre → opcion vale solo el intrinseco
        return max(S - K, 0.0) if option_type == "C" else max(K - S, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    if option_type == "C":
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)


def bs_delta(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str, q: float = 0.0,
) -> float:
    if T <= 0:
        if option_type == "C":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    if sigma <= 0:
        if option_type == "C":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    if option_type == "C":
        return math.exp(-q * T) * norm.cdf(d1)
    return math.exp(-q * T) * (norm.cdf(d1) - 1.0)


def bs_gamma(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0,
) -> float:
    if T <= 0 or S <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return math.exp(-q * T) * norm.pdf(d1) / (S * sigma * math.sqrt(T))


def bs_theta(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str, q: float = 0.0,
) -> float:
    """Theta por día calendario."""
    if T <= 0 or sigma <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    decay = -S * math.exp(-q * T) * norm.pdf(d1) * sigma / (2 * sqrtT)
    if option_type == "C":
        return (
            decay
            - r * K * math.exp(-r * T) * norm.cdf(d2)
            + q * S * math.exp(-q * T) * norm.cdf(d1)
        ) / 365.0
    return (
        decay
        + r * K * math.exp(-r * T) * norm.cdf(-d2)
        - q * S * math.exp(-q * T) * norm.cdf(-d1)
    ) / 365.0


def bs_vega(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0,
) -> float:
    """Vega por 1% de movimiento en vol."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return S * math.exp(-q * T) * norm.pdf(d1) * math.sqrt(T) / 100.0


def bs_rho(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str, q: float = 0.0,
) -> float:
    """Rho por 1% de movimiento en la tasa."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d2 = (
        (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        - sigma * math.sqrt(T)
    )
    if option_type == "C":
        return K * T * math.exp(-r * T) * norm.cdf(d2) / 100.0
    return -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100.0


# ── CRR Binomial (Americanas / Europeas, dividendos discretos) ────────────────

def crr_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    q: float = 0.0,
    american: bool = True,
    steps: int = 200,
    dividends: Optional[list[tuple[float, float]]] = None,
) -> float:
    """Precio via árbol CRR recombinante.

    dividends: lista de (tiempo_en_años, monto_en_ARS) para dividendos discretos
               que ocurren estrictamente dentro de (0, T].
               Usa el modelo de dividendos en fideicomiso (Hull): el árbol se
               construye sobre S_adj = S - VP(dividendos futuros), conservando la
               recombinancia. Las decisiones de ejercicio usan el precio real
               S_adj_nodo + VP(dividendos restantes).
    """
    if T <= 0:
        return max(S - K, 0.0) if option_type == "C" else max(K - S, 0.0)
    if sigma <= 0:
        return max(S - K, 0.0) if option_type == "C" else max(K - S, 0.0)

    dividends = [(t, d) for t, d in (dividends or []) if 0.0 < t <= T]

    # Ajuste por dividendos en fideicomiso (escrowed-dividend model)
    pv_all = sum(d * math.exp(-r * t) for t, d in dividends)
    S_adj = max(S - pv_all, 1e-8)

    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    p = min(max(p, 0.0), 1.0)  # guard para dt grande

    # VP de dividendos restantes en cada paso i (para decisión de ejercicio)
    step_div_pv = np.zeros(steps + 1)
    for i in range(steps + 1):
        t_now = i * dt
        step_div_pv[i] = sum(
            da * math.exp(-r * (td - t_now))
            for td, da in dividends
            if t_now < td
        )

    # Nodos terminales (sin dividendos restantes)
    j = np.arange(steps + 1)
    ST = S_adj * u ** j * d ** (steps - j)
    values = np.maximum(ST - K, 0.0) if option_type == "C" else np.maximum(K - ST, 0.0)

    # Inducción hacia atrás
    for i in range(steps - 1, -1, -1):
        j_arr = np.arange(i + 1)
        S_node = S_adj * u ** j_arr * d ** (i - j_arr)
        S_actual = S_node + step_div_pv[i]  # precio real para comparar con strike

        values = disc * (p * values[1 : i + 2] + (1 - p) * values[0 : i + 1])
        if american:
            intrinsic = (
                np.maximum(S_actual - K, 0.0)
                if option_type == "C"
                else np.maximum(K - S_actual, 0.0)
            )
            values = np.maximum(values, intrinsic)

    return float(values[0])


# ── Griegas CRR por diferencias finitas ───────────────────────────────────────

def crr_delta(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str, **kwargs,
) -> float:
    if S <= 0:
        return 0.0
    h = S * 0.01
    return (
        crr_price(S + h, K, T, r, sigma, option_type, **kwargs)
        - crr_price(S - h, K, T, r, sigma, option_type, **kwargs)
    ) / (2.0 * h)


def crr_gamma(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str, **kwargs,
) -> float:
    if S <= 0:
        return 0.0
    h = S * 0.01
    up = crr_price(S + h, K, T, r, sigma, option_type, **kwargs)
    mid = crr_price(S, K, T, r, sigma, option_type, **kwargs)
    dn = crr_price(S - h, K, T, r, sigma, option_type, **kwargs)
    return (up - 2.0 * mid + dn) / (h ** 2)


def crr_theta(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str, **kwargs,
) -> float:
    """Theta por día calendario (cambio de valor al pasar un día)."""
    dT = 1.0 / 365.0
    if T <= dT:
        return 0.0
    return (
        crr_price(S, K, T - dT, r, sigma, option_type, **kwargs)
        - crr_price(S, K, T, r, sigma, option_type, **kwargs)
    )


def crr_vega(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str, **kwargs,
) -> float:
    """Vega por 1% de movimiento en vol."""
    h = 0.01
    return (
        crr_price(S, K, T, r, sigma + h, option_type, **kwargs)
        - crr_price(S, K, T, r, sigma - h, option_type, **kwargs)
    ) / 2.0


# ── Volatilidad Implícita ──────────────────────────────────────────────────────

def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    q: float = 0.0,
    american: bool = True,
    steps: int = 100,
    sigma_lo: float = 1e-4,
    sigma_hi: float = 5.0,
) -> Optional[float]:
    """Volatilidad implícita via método de Brent.

    Usa CRR (american=True) o BS (american=False).
    Devuelve None si no hay solución en [sigma_lo, sigma_hi].
    """
    if T <= 0 or market_price <= 0:
        return None

    def f(sigma: float) -> float:
        if american:
            return crr_price(S, K, T, r, sigma, option_type, q=q, steps=steps) - market_price
        return bs_price(S, K, T, r, sigma, option_type, q=q) - market_price

    try:
        if f(sigma_lo) * f(sigma_hi) > 0:
            return None
        return float(brentq(f, sigma_lo, sigma_hi, xtol=1e-6, maxiter=50))
    except (ValueError, RuntimeError):
        return None


# ── Probabilidad de tocar una barrera (reflection principle) ─────────────────

def prob_touch(
    spot: float,
    barrier: float,
    T: float,
    sigma: float,
) -> float:
    """P(S_t toca barrier en [0,T]) bajo GBM con drift=0 (medida física).

    Reflection principle aplicado a X_t = log(S_t/S_0) con drift = -σ²/2.
    Válido para barreras arriba o abajo del spot.

    Uso típico: estimar riesgo de asignación temprana del short leg en credit
    spreads americanos. P(touch) ≥ P(S_T cierra del lado equivocado) siempre;
    cerca del strike puede ser ~2× mayor que la PoP terminal.

    Fórmulas (derivadas por cambio de medida / Girsanov):
      Up barrier   (B > S): N(-d_-) + (S/B)·N(-d_+)
      Down barrier (B < S): N( d_+) + (S/B)·N( d_-)
    donde d_± = (log(B/S) ± σ²T/2) / (σ√T).
    """
    if T <= 0 or sigma <= 0 or spot <= 0 or barrier <= 0:
        return 0.0
    if barrier == spot:
        return 1.0
    sig_t  = sigma * math.sqrt(T)
    log_bs = math.log(barrier / spot)
    sig2T  = sigma ** 2 * T / 2
    d_plus  = (log_bs + sig2T) / sig_t
    d_minus = (log_bs - sig2T) / sig_t
    ratio   = spot / barrier
    if barrier > spot:
        return float(norm.cdf(-d_minus) + ratio * norm.cdf(-d_plus))
    return float(norm.cdf(d_plus) + ratio * norm.cdf(d_minus))


# ── Probabilidad risk-neutral ─────────────────────────────────────────────────

def risk_neutral_prob_above(
    S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0,
) -> float:
    """P(S_T > K) bajo la medida risk-neutral lognormal BSM = N(d2).

    Util para estimar la probabilidad de exito de spreads verticales
    pasando el breakeven del spread como K.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 1.0 if S > K else 0.0
    d2 = (math.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return float(norm.cdf(d2))
