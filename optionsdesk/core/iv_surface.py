"""Superficie de volatilidad implícita: sonrisa y term structure.

Fitea IV vs log-moneyness por vencimiento con un polinomio cuadrático
(mínimos cuadrados). Con pocos strikes (5-7 es lo habitual en GGAL) un
poly2 es un fit casi exacto, así que solo se reporta mispricing cuando
hay ≥ 5 strikes líquidos y el R² es razonable (> 0.5).

El `mispricing_score` mide cuánto caro/barato cotiza un strike relativo
a la sonrisa ajustada: positivo = el strike cotiza con IV más alta de lo
que predice el fit → bueno para vender.

Las funciones `build_smile_map` y `term_structure_slope` toman
`list[RateResult]` (ya con IV computada por los scanners) — no la cadena
cruda, que solo tiene precios bid/ask sin IV.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from optionsdesk.core.rates import RateResult


@dataclass
class SmileFit:
    """Resultado del ajuste cuadrático IV vs log-moneyness."""
    expiry_code: str
    spot: float
    n_points: int            # strikes usados en el fit
    coeffs: tuple[float, float, float]   # (a, b, c) para a·x² + b·x + c
    r_squared: float         # bondad del ajuste; < 0.5 → fit poco fiable
    atm_iv: float            # IV interpolada en log-moneyness = 0 (i.e., ATM)

    def eval(self, strike: float) -> Optional[float]:
        """Devuelve la IV que predice el modelo para un strike dado."""
        if self.spot <= 0 or strike <= 0:
            return None
        x = math.log(strike / self.spot)
        a, b, c = self.coeffs
        return a * x**2 + b * x + c

    @property
    def is_reliable(self) -> bool:
        return self.n_points >= 5 and self.r_squared >= 0.50


def _polyfit2(xs: list[float], ys: list[float]) -> tuple[float, float, float, float]:
    """Mínimos cuadrados para polinomio grado 2. Devuelve (a, b, c, r²).

    Implementación manual para no depender de numpy en esta función pura.
    """
    n = len(xs)
    if n < 3:
        raise ValueError("Se necesitan al menos 3 puntos para poly2.")

    # Sumas para sistema de ecuaciones normales (Vandermonde 3×3)
    s0 = float(n)
    s1 = sum(x for x in xs)
    s2 = sum(x**2 for x in xs)
    s3 = sum(x**3 for x in xs)
    s4 = sum(x**4 for x in xs)
    t0 = sum(ys)
    t1 = sum(xs[i] * ys[i] for i in range(n))
    t2 = sum(xs[i]**2 * ys[i] for i in range(n))

    # Sistema: [[s4,s3,s2],[s3,s2,s1],[s2,s1,s0]] * [a,b,c]^T = [t2,t1,t0]
    # Resuelvo con eliminación gaussiana 3×3
    mat = [
        [s4, s3, s2, t2],
        [s3, s2, s1, t1],
        [s2, s1, s0, t0],
    ]

    for col in range(3):
        # Pivoteo parcial
        max_row = max(range(col, 3), key=lambda r: abs(mat[r][col]))
        mat[col], mat[max_row] = mat[max_row], mat[col]
        pivot = mat[col][col]
        if abs(pivot) < 1e-15:
            raise ValueError("Sistema singular — strikes colineales o sin variación.")
        for row in range(col + 1, 3):
            factor = mat[row][col] / pivot
            for k in range(col, 4):
                mat[row][k] -= factor * mat[col][k]

    # Back-substitution
    coeffs = [0.0, 0.0, 0.0]
    for row in range(2, -1, -1):
        val = mat[row][3]
        for col in range(row + 1, 3):
            val -= mat[row][col] * coeffs[col]
        coeffs[row] = val / mat[row][row]

    a, b, c = coeffs

    # R²
    mean_y = sum(ys) / n
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((ys[i] - (a * xs[i]**2 + b * xs[i] + c)) ** 2 for i in range(n))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

    return a, b, c, r2


def fit_smile(
    strikes: list[float],
    ivs: list[float],
    spot: float,
    expiry_code: str = "",
    min_points: int = 5,
) -> Optional[SmileFit]:
    """Ajusta la sonrisa de IV para un vencimiento dado.

    Args:
        strikes: precios de ejercicio con datos válidos.
        ivs: IV implícita correspondiente a cada strike (fracción, e.g. 0.45).
        spot: precio actual del subyacente.
        expiry_code: código del vencimiento (ej. "JUN").
        min_points: mínimo de puntos para intentar el fit.

    Returns:
        SmileFit si el ajuste es posible, None si hay datos insuficientes.
    """
    if spot <= 0 or len(strikes) < min_points or len(strikes) != len(ivs):
        return None

    # Filtrar pares inválidos
    pairs = [(strikes[i], ivs[i]) for i in range(len(strikes)) if ivs[i] > 0 and strikes[i] > 0]
    if len(pairs) < min_points:
        return None

    xs = [math.log(k / spot) for k, _ in pairs]
    ys = [iv for _, iv in pairs]

    try:
        a, b, c, r2 = _polyfit2(xs, ys)
    except ValueError:
        return None

    atm_iv = c   # eval en x=0 → a·0 + b·0 + c

    return SmileFit(
        expiry_code=expiry_code,
        spot=spot,
        n_points=len(pairs),
        coeffs=(a, b, c),
        r_squared=max(0.0, min(1.0, r2)),
        atm_iv=atm_iv,
    )


def mispricing_score(strike: float, iv: float, smile_fit: SmileFit) -> Optional[float]:
    """IV observada − IV del fit, en puntos porcentuales.

    Positivo → el strike cotiza con IV más alta que la sonrisa → bueno para vender.
    Devuelve None si el fit no es confiable o los inputs son inválidos.
    """
    if not smile_fit.is_reliable or iv <= 0 or strike <= 0:
        return None
    iv_fit = smile_fit.eval(strike)
    if iv_fit is None or iv_fit <= 0:
        return None
    # Resultado en pp (ambos son fracciones, e.g. 0.45)
    return (iv - iv_fit) * 100.0


_SYM_RE = re.compile(r"^GFG([CV])(\d+(?:[.,]\d+)?)([A-Z]+)$")


def _parse_symbol(symbol: str):
    """Devuelve (opt_type, strike, expiry_code) o None si no matchea."""
    m = _SYM_RE.match(symbol)
    if not m:
        return None
    try:
        strike = float(m.group(2).replace(",", "."))
    except ValueError:
        return None
    return m.group(1), strike, m.group(3)


def term_structure_slope(
    results: "list[RateResult]",
    spot: float,
) -> dict[str, float]:
    """IV ATM por vencimiento (term structure) a partir de RateResults.

    Toma `all_results` del recommender (ya con IV computada) y devuelve
    dict[expiry_code → atm_iv]. Vencimientos sin opciones con IV se omiten.
    """
    by_expiry: dict[str, list[tuple[float, float]]] = {}

    for r in results:
        if r.iv is None or r.iv <= 0:
            continue
        parsed = _parse_symbol(r.symbol)
        if not parsed:
            continue
        _, strike, expiry = parsed
        by_expiry.setdefault(expiry, []).append((strike, r.iv))

    result: dict[str, float] = {}
    for code, pairs in by_expiry.items():
        if not pairs or spot <= 0:
            continue
        # Strike más cercano al spot (ATM)
        _, atm_iv = min(pairs, key=lambda p: abs(p[0] - spot))
        result[code] = atm_iv

    return result


def build_smile_map(
    results: "list[RateResult]",
    spot: float,
    min_points: int = 5,
) -> dict[str, SmileFit]:
    """Construye un SmileFit por vencimiento a partir de RateResults.

    Usa solo resultados líquidos (`is_liquid=True`) con IV disponible.
    Returns dict[expiry_code → SmileFit]. Vencimientos sin fit confiable se omiten.
    """
    by_expiry: dict[str, dict[str, list]] = {}

    for r in results:
        if r.iv is None or r.iv <= 0 or not r.is_liquid:
            continue
        parsed = _parse_symbol(r.symbol)
        if not parsed:
            continue
        _, strike, expiry = parsed

        exp_data = by_expiry.setdefault(expiry, {"strikes": [], "ivs": []})
        exp_data["strikes"].append(strike)
        exp_data["ivs"].append(r.iv)

    result: dict[str, SmileFit] = {}
    for code, data in by_expiry.items():
        fit = fit_smile(data["strikes"], data["ivs"], spot, expiry_code=code, min_points=min_points)
        if fit is not None and fit.is_reliable:
            result[code] = fit

    return result
