"""Métricas de volatilidad: realizada, VRP e IV Rank.

El edge de vender prima es positivo cuando IV > vol realizada (el mercado
paga más por protección de la que estadísticamente debería). Sin historial
de IV el iv_rank cae a "sin datos" — no se inventa un número.

v3.1: agrega Yang-Zhang (OHLC) y Parkinson como estimadores más eficientes.
Yang-Zhang usa overnight + Rogers-Satchell + open-to-close: ~7× más eficiente
que close-to-close en varianza del estimador. Parkinson solo high-low, simple.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def realized_volatility(
    spot_series,
    annualize: bool = True,
    window: Optional[int] = None,
) -> Optional[float]:
    """Volatilidad realizada (desvío estándar de retornos log, anualizado con 252).

    spot_series: cierres diarios en orden cronológico (list o pd.Series).
    window: si se especifica, usa solo las últimas `window` observaciones.
            Si len(serie) < window, usa la serie completa (fallback silencioso).
    Devuelve None si hay menos de 2 precios.
    """
    series = list(spot_series)
    if window is not None and len(series) > window:
        series = series[-window:]
    if len(series) < 2:
        return None
    log_returns = [
        math.log(series[i] / series[i - 1])
        for i in range(1, len(series))
        if series[i - 1] > 0 and series[i] > 0
    ]
    n = len(log_returns)
    if n < 1:
        return None
    mean = sum(log_returns) / n
    variance = sum((r - mean) ** 2 for r in log_returns) / max(n - 1, 1)
    daily_vol = math.sqrt(variance)
    return daily_vol * math.sqrt(252) if annualize else daily_vol


def yang_zhang_volatility(
    ohlc_df: "pd.DataFrame",
    window: int = 20,
    annualize: bool = True,
) -> Optional[float]:
    """Volatilidad Yang-Zhang sobre OHLC (anualizada con 252).

    ~7× más eficiente que close-to-close; maneja gaps overnight y
    movimientos open-to-close. Requiere columnas: open, high, low, close
    (case-insensitive). Devuelve None si hay menos de 3 filas o datos inválidos.

    Fórmula: σ²_YZ = σ²_overnight + k·σ²_OC + (1-k)·σ²_RS
    donde k = 0.34 / (1.34 + (N+1)/(N-1)) y σ²_RS = Rogers-Satchell intraday.
    """
    try:
        import pandas as _pd
    except ImportError:
        return None

    df = ohlc_df.copy()
    df.columns = [c.lower() for c in df.columns]
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        return None

    df = df.tail(window).dropna(subset=list(required))
    N = len(df)
    if N < 3:
        return None

    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values

    # Retornos log overnight: ln(O_t / C_{t-1})
    overnight = [math.log(o[i] / c[i - 1]) for i in range(1, N) if c[i - 1] > 0 and o[i] > 0]
    # Retornos open-to-close: ln(C_t / O_t)
    oc = [math.log(c[i] / o[i]) for i in range(N) if o[i] > 0 and c[i] > 0]

    n = min(len(overnight), len(oc) - 1)
    if n < 2:
        return None

    overnight = overnight[:n]
    oc_window = oc[1:n + 1]   # alinear con overnight (de i=1 en adelante)

    mean_on = sum(overnight) / n
    mean_oc = sum(oc_window) / n
    var_on = sum((x - mean_on) ** 2 for x in overnight) / max(n - 1, 1)
    var_oc = sum((x - mean_oc) ** 2 for x in oc_window) / max(n - 1, 1)

    # Rogers-Satchell: captura el movimiento intradía sin drift
    rs_terms = []
    for i in range(1, N):
        if o[i] > 0 and h[i] > 0 and l[i] > 0 and c[i] > 0:
            rs = (
                math.log(h[i] / c[i]) * math.log(h[i] / o[i])
                + math.log(l[i] / c[i]) * math.log(l[i] / o[i])
            )
            rs_terms.append(rs)
    var_rs = sum(rs_terms) / len(rs_terms) if rs_terms else 0.0

    k = 0.34 / (1.34 + (n + 1) / max(n - 1, 1))
    var_yz = var_on + k * var_oc + (1.0 - k) * var_rs

    if var_yz <= 0:
        return None
    daily_vol = math.sqrt(var_yz)
    return daily_vol * math.sqrt(252) if annualize else daily_vol


def parkinson_volatility(
    ohlc_df: "pd.DataFrame",
    window: int = 20,
    annualize: bool = True,
) -> Optional[float]:
    """Volatilidad Parkinson (solo high-low). Más eficiente que C-C; no captura gaps.

    Requiere columnas high, low. Devuelve None si hay menos de 2 filas.
    """
    try:
        import pandas as _pd
    except ImportError:
        return None

    df = ohlc_df.copy()
    df.columns = [c.lower() for c in df.columns]
    if not {"high", "low"}.issubset(df.columns):
        return None

    df = df.tail(window).dropna(subset=["high", "low"])
    N = len(df)
    if N < 2:
        return None

    h = df["high"].values
    l = df["low"].values

    factor = 1.0 / (4.0 * N * math.log(2.0))
    variance = factor * sum(
        math.log(h[i] / l[i]) ** 2
        for i in range(N)
        if l[i] > 0 and h[i] > 0
    )
    if variance <= 0:
        return None
    daily_vol = math.sqrt(variance)
    return daily_vol * math.sqrt(252) if annualize else daily_vol


def variance_risk_premium(iv: float, realized_vol: float) -> float:
    """VRP = IV − vol realizada. Positivo → estás cobrando cara la prima (bueno para el vendedor)."""
    return iv - realized_vol


def iv_rank(current_iv: float, iv_history: list[float]) -> Optional[float]:
    """IV Rank 0–100: posición de la IV actual en el rango histórico.

    iv_history: serie de IVs previas (sin incluir current_iv).
    Devuelve None si historial vacío o sin rango (iv_max == iv_min).
    """
    if not iv_history:
        return None
    iv_min = min(iv_history)
    iv_max = max(iv_history)
    if iv_max <= iv_min:
        return None
    return (current_iv - iv_min) / (iv_max - iv_min) * 100.0


@dataclass
class VolEdge:
    """Resumen del edge de volatilidad para una oportunidad concreta."""

    realized_vol: Optional[float]    # anualizada (p.ej. 0.45 = 45%)
    iv: Optional[float]              # IV implícita de la opción
    vrp: Optional[float]             # variance risk premium = iv - realized_vol
    iv_rank_pct: Optional[float]     # 0–100; None si sin historial suficiente
    label: str                       # "positivo" | "neutro" | "negativo" | "sin datos"

    @property
    def has_edge(self) -> bool:
        """True si VRP > 0: la prima está cara respecto a la vol histórica."""
        return self.vrp is not None and self.vrp > 0.0

    @classmethod
    def compute(
        cls,
        iv: Optional[float],
        spot_history: list[float],
        iv_history: Optional[list[float]] = None,
        ohlc_df=None,               # v2.7: pd.DataFrame OHLCV; si se provee, usa YZ
        window: Optional[int] = None,  # v2.7: rolling window para C-C fallback
    ) -> "VolEdge":
        """Construye un VolEdge dado IV, historial de spot e historial de IV.

        ohlc_df: DataFrame con columnas open/high/low/close. Si se provee, usa
                 Yang-Zhang (7× mas eficiente que C-C). Fallback a C-C si YZ falla.
        window:  Ventana rolling para el fallback C-C (None = serie completa).
        Si IV es None o no hay suficiente historial de spot, el label es "sin datos".
        """
        if iv is None or iv <= 0:
            return cls(None, None, None, None, "sin datos")

        rv = None
        if ohlc_df is not None:
            rv = yang_zhang_volatility(ohlc_df, window=window or 20)
        if rv is None:
            rv = realized_volatility(spot_history, window=window)
        vrp = variance_risk_premium(iv, rv) if rv is not None else None
        ivr = iv_rank(iv, iv_history or [])

        if vrp is None:
            label = "sin datos"
        elif vrp > 0.05:
            label = "positivo"
        elif vrp > 0.0:
            label = "neutro"
        else:
            label = "negativo"

        return cls(
            realized_vol=rv,
            iv=iv,
            vrp=vrp,
            iv_rank_pct=ivr,
            label=label,
        )
