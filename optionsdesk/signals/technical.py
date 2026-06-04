"""Análisis técnico — AT clásico + Smart Money Concepts.

Librerías externas
------------------
pandas-ta         : RSI, ATR, SMA, EMA, MACD, MOM — cálculos probados y eficientes.
smartmoneyconcepts: swing H/L, BOS/CHoCH, FVG, Order Blocks — SMC en pandas puro.

Ambas son opcionales: si no están instaladas (o fallan), caemos a implementación
manual equivalente. El módulo nunca lanza excepciones al importarse.

Señales SMC para el vendedor de opciones
-----------------------------------------
BOS_UP   → precio rompe estructura alcista → short puts más seguros
BOS_DOWN → precio rompe estructura bajista → más colchón para covered calls
CHoCH    → cambio de carácter contra la tendencia → reduce confianza en la idea
FVG      → zona de desequilibrio sin llenar → target de precio objetivo
OB       → última vela opositora antes del BOS → soporte/resistencia institucional
"""
from __future__ import annotations

import contextlib
import io as _io
import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Disponibilidad de librerías externas ──────────────────────────────────────

def _try_import_pta():
    try:
        import pandas_ta as pta   # type: ignore[import-not-found]
        required = ("sma", "ema", "rsi", "atr", "macd")
        if not all(callable(getattr(pta, name, None)) for name in required):
            return None
        return pta
    except Exception:
        return None

def _try_import_smc():
    """Importa smartmoneyconcepts suprimiendo el print con emoji que falla en Windows/cp1252."""
    try:
        with contextlib.redirect_stdout(_io.StringIO()):
            from smartmoneyconcepts import smc   # type: ignore[import-not-found]
        return smc
    except Exception:
        return None

_pta = _try_import_pta()
_smc = _try_import_smc()


# ── Tipos de datos SMC ────────────────────────────────────────────────────────

@dataclass
class FairValueGap:
    """Zona de desequilibrio de 3 velas (imbalance)."""
    type: str       # "bullish" | "bearish"
    high: float     # límite superior
    low: float      # límite inferior
    filled: bool    # el precio ya cotizó dentro de la zona


@dataclass
class OrderBlock:
    """Última vela opositora antes de un BOS — zona S/R institucional."""
    type: str       # "bullish" | "bearish"
    high: float
    low: float
    volume: float = 0.0


# ── Indicadores clásicos (wrappers sobre pandas-ta con fallback manual) ───────

def _pta_valid(r: "pd.Series | None", expected_len: int) -> bool:
    """True si pandas-ta devolvió una Series del tamaño esperado."""
    return r is not None and len(r) == expected_len


def sma(series: pd.Series, n: int) -> pd.Series:
    if _pta is not None:
        try:
            r = _pta.sma(series, length=n)
            if _pta_valid(r, len(series)):
                return r
        except Exception:
            pass
    return series.rolling(n, min_periods=n).mean()


def ema(series: pd.Series, n: int) -> pd.Series:
    if _pta is not None:
        try:
            r = _pta.ema(series, length=n)
            if _pta_valid(r, len(series)):
                return r
        except Exception:
            pass
    return series.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    if _pta is not None:
        try:
            r = _pta.rsi(series, length=n)
            # pandas-ta returns all-NaN for constant series → fall through to manual
            if _pta_valid(r, len(series)) and r.notna().any():
                return r
        except Exception:
            pass
    # Wilder manual
    delta  = series.diff()
    gain   = delta.clip(lower=0)
    loss   = -delta.clip(upper=0)
    avg_g  = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_l  = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs     = avg_g / avg_l.replace(0, 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    if _pta is not None:
        try:
            r = _pta.atr(df["high"], df["low"], df["close"], length=n)
            if _pta_valid(r, len(df)):
                return r
        except Exception:
            pass
    hi, lo, cl = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [hi - lo, (hi - cl.shift(1)).abs(), (lo - cl.shift(1)).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def momentum_pct(series: pd.Series, n: int = 10) -> pd.Series:
    """Cambio % respecto a n períodos atrás."""
    return (series / series.shift(n) - 1.0) * 100.0


def _macd_hist(series: pd.Series) -> pd.Series:
    """MACD histogram (MACD_line - signal). Positivo = momentum alcista."""
    if _pta is not None:
        r = _pta.macd(series, fast=12, slow=26, signal=9)
        if r is not None:
            hist_col = [c for c in r.columns if "MACDh" in c or "h_" in c.lower()]
            if hist_col:
                return r[hist_col[0]]
    fast = series.ewm(span=12, adjust=False, min_periods=12).mean()
    slow = series.ewm(span=26, adjust=False, min_periods=26).mean()
    macd_line   = fast - slow
    signal_line = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    return macd_line - signal_line


# ── SMC helpers ───────────────────────────────────────────────────────────────

def _smc_signals(df: pd.DataFrame) -> dict:
    """Extrae BOS, CHoCH, FVGs y OBs usando smartmoneyconcepts.

    Devuelve dict vacío si la librería no está disponible.
    Swing length = 3 (apropiado para velas diarias con horizonte swing de días).
    """
    if _smc is None:
        return {}
    try:
        ohlcv = df[["open", "high", "low", "close", "volume"]].copy()

        swing_hl   = _smc.swing_highs_lows(ohlcv, swing_length=3)

        # BOS / CHoCH — última señal en el historial
        bc         = _smc.bos_choch(ohlcv, swing_hl)
        bc_signals = bc.dropna(how="all")

        bos_val = choch_val = None
        if not bc_signals.empty:
            last = bc_signals.iloc[-1]
            if pd.notna(last.get("BOS")) and last["BOS"] != 0:
                bos_val = "BOS_UP" if last["BOS"] > 0 else "BOS_DOWN"
            if pd.notna(last.get("CHOCH")) and last["CHOCH"] != 0:
                choch_val = "CHOCH_UP" if last["CHOCH"] > 0 else "CHOCH_DOWN"

        # FVGs — solo los no mitigados
        fvg_df     = _smc.fvg(ohlcv)
        fvg_active = fvg_df[fvg_df["FVG"].notna() & (fvg_df["MitigatedIndex"] == 0)]
        fvgs: list[FairValueGap] = [
            FairValueGap(
                type="bullish" if row["FVG"] > 0 else "bearish",
                high=float(row["Top"]),
                low=float(row["Bottom"]),
                filled=False,
            )
            for _, row in fvg_active.iterrows()
        ]

        # OBs — solo los no mitigados
        ob_df      = _smc.ob(ohlcv, swing_hl)
        ob_active  = ob_df[ob_df["OB"].notna() & (ob_df["MitigatedIndex"] == 0)]
        ob_val: Optional[OrderBlock] = None
        if not ob_active.empty:
            last_ob = ob_active.iloc[-1]
            ob_val = OrderBlock(
                type="bullish" if last_ob["OB"] > 0 else "bearish",
                high=float(last_ob["Top"]),
                low=float(last_ob["Bottom"]),
                volume=float(last_ob.get("OBVolume", 0)),
            )

        return {"bos": bos_val, "choch": choch_val, "fvgs": fvgs, "ob": ob_val}

    except Exception as exc:
        logger.debug("SMC analysis failed: %s", exc)
        return {}


def _nearest_unfilled_fvg(fvgs: list[FairValueGap], spot: float) -> Optional[FairValueGap]:
    """FVG sin llenar más cercano dentro del 12% del spot."""
    band = spot * 0.12
    close_gaps = [g for g in fvgs if abs((g.high + g.low) / 2 - spot) <= band]
    if not close_gaps:
        return None
    return min(close_gaps, key=lambda g: abs((g.high + g.low) / 2 - spot))


def _vol_confirms(df: pd.DataFrame, n: int = 20) -> bool:
    """True si el último cierre tiene volumen superior a la media de n ruedas."""
    if "volume" not in df.columns or len(df) < 2:
        return False
    vol = df["volume"]
    avg = vol.rolling(n, min_periods=5).mean()
    return bool(not pd.isna(avg.iloc[-1]) and float(vol.iloc[-1]) > float(avg.iloc[-1]))


# ── Snapshot ──────────────────────────────────────────────────────────────────

@dataclass
class TechnicalSnapshot:
    # Indicadores clásicos
    trend: str              # "ALCISTA" | "BAJISTA" | "LATERAL"
    rsi: float
    atr_pct: float          # ATR(14) como % del spot
    sma_fast: float         # SMA(5)
    sma_slow: float         # SMA(20)
    momentum: float         # % respecto a 10 días atrás
    read: str               # resumen en castellano
    # Con default para mantener compatibilidad con código existente
    macd_hist: float = 0.0  # histograma MACD — positivo = aceleración alcista
    signal_strength: str = "neutral"
    # SMC
    bos: Optional[str] = None               # "BOS_UP" | "BOS_DOWN"
    choch: Optional[str] = None             # "CHOCH_UP" | "CHOCH_DOWN"
    fvgs: list = field(default_factory=list)
    order_block: Optional[OrderBlock] = None
    nearest_fvg: Optional[FairValueGap] = None
    vol_confirms: bool = False


# ── Función principal ─────────────────────────────────────────────────────────

def analyze(df: pd.DataFrame) -> TechnicalSnapshot:
    """Indicadores clásicos + SMC sobre OHLCV diario.

    df: columnas date, open, high, low, close, volume. Mínimo 5 filas.
    SMC requiere ~30+ para señales confiables; con menos datos se omite sin error.
    """
    if df.empty or len(df) < 5:
        return TechnicalSnapshot(
            trend="LATERAL", rsi=50.0, atr_pct=0.0,
            sma_fast=0.0, sma_slow=0.0, momentum=0.0,
            read="Datos insuficientes para análisis técnico.",
        )

    close = df["close"]

    # ── Indicadores clásicos ─────────────────────────────────────────────────
    sma5     = sma(close, 5)
    sma20    = sma(close, 20)
    rsi14    = rsi(close, 14)
    atr14    = atr(df, 14)
    mom10    = momentum_pct(close, 10)
    macd_h   = _macd_hist(close)

    last_close  = float(close.iloc[-1])
    last_sma5   = _last(sma5,  last_close)
    last_sma20  = _last(sma20, last_close)
    last_rsi    = _last(rsi14, 50.0)
    last_atr    = _last(atr14, 0.0)
    last_mom    = _last(mom10, 0.0)
    last_macd_h = _last(macd_h, 0.0)
    atr_pct_    = last_atr / last_close * 100.0 if last_close > 0 else 0.0

    # ── SMC (silencioso si no hay datos / librería) ──────────────────────────
    smc_data: dict = {}
    if len(df) >= 20:
        smc_data = _smc_signals(df)

    bos_val    = smc_data.get("bos")
    choch_val  = smc_data.get("choch")
    fvgs_list  = smc_data.get("fvgs", [])
    ob_val     = smc_data.get("ob")
    vol_ok     = _vol_confirms(df)
    nearest    = _nearest_unfilled_fvg(fvgs_list, last_close)

    # ── Tendencia: BOS es la señal de más peso ────────────────────────────────
    trend = _determine_trend(
        bos_val, last_sma5, last_sma20, last_mom, last_macd_h
    )

    # ── Fuerza de la señal ────────────────────────────────────────────────────
    strength = _signal_strength(trend, last_rsi, last_mom, bos_val, choch_val, vol_ok)

    read = _build_read(
        trend, last_rsi, atr_pct_, last_mom, last_sma5, last_sma20,
        last_macd_h, bos_val, choch_val, nearest, vol_ok,
    )

    return TechnicalSnapshot(
        trend=trend,
        rsi=round(last_rsi, 1),
        atr_pct=round(atr_pct_, 2),
        sma_fast=round(last_sma5, 2),
        sma_slow=round(last_sma20, 2),
        momentum=round(last_mom, 2),
        read=read,
        macd_hist=round(last_macd_h, 2),
        signal_strength=strength,
        bos=bos_val,
        choch=choch_val,
        fvgs=fvgs_list,
        order_block=ob_val,
        nearest_fvg=nearest,
        vol_confirms=vol_ok,
    )


# ── Helpers internos ──────────────────────────────────────────────────────────

def _last(series: pd.Series, default: float) -> float:
    v = series.iloc[-1]
    return float(v) if not pd.isna(v) else default


def _determine_trend(
    bos: Optional[str],
    sma5: float, sma20: float,
    mom: float, macd_h: float,
) -> str:
    """BOS es la señal más fuerte. SMA + MACD + momentum como fallback."""
    if bos == "BOS_UP":
        return "ALCISTA"
    if bos == "BOS_DOWN":
        return "BAJISTA"
    # SMA cruce con confirmación de MACD y momentum
    if sma5 > sma20 * 1.003 and mom > 0.5 and macd_h >= 0:
        return "ALCISTA"
    if sma5 < sma20 * 0.997 and mom < -0.5 and macd_h <= 0:
        return "BAJISTA"
    # Un solo filtro alcanza si el spread SMA es grande
    if sma5 > sma20 * 1.01 and mom > 1.5:
        return "ALCISTA"
    if sma5 < sma20 * 0.99 and mom < -1.5:
        return "BAJISTA"
    return "LATERAL"


def _signal_strength(
    trend: str,
    rsi_val: float,
    mom: float,
    bos: Optional[str],
    choch: Optional[str],
    vol_ok: bool,
) -> str:
    """CHoCH degrada la señal. BOS con volumen la fortalece."""
    if trend == "LATERAL":
        return "neutral"
    # CHoCH contra la tendencia → estructura debilitada
    if choch is not None:
        if (choch == "CHOCH_DOWN" and trend == "ALCISTA") or \
           (choch == "CHOCH_UP"   and trend == "BAJISTA"):
            return "débil"
    # BOS confirma → señal de estructura
    if bos is not None:
        return "fuerte" if vol_ok else "moderada"
    # Sin BOS: solo indicadores clásicos
    rsi_align = (rsi_val > 55 and trend == "ALCISTA") or (rsi_val < 45 and trend == "BAJISTA")
    if abs(mom) > 3.0 and rsi_align:
        return "fuerte"
    if abs(mom) > 1.5 or rsi_align:
        return "moderada"
    return "débil"


def _build_read(
    trend: str,
    rsi_val: float,
    atr_pct: float,
    mom: float,
    sma5: float, sma20: float,
    macd_h: float,
    bos: Optional[str],
    choch: Optional[str],
    nearest_fvg_: Optional[FairValueGap],
    vol_ok: bool,
) -> str:
    trend_lbl = {"ALCISTA": "Tendencia alcista", "BAJISTA": "Tendencia bajista", "LATERAL": "Mercado lateral"}[trend]
    rsi_lbl   = "RSI sobrecomprado" if rsi_val > 70 else "RSI sobrevendido" if rsi_val < 30 else f"RSI {rsi_val:.0f}"
    vol_lbl   = "vol alta" if atr_pct > 3.0 else "vol baja" if atr_pct < 1.0 else "vol normal"
    macd_lbl  = "MACD+" if macd_h > 0 else "MACD-"

    base = (
        f"{trend_lbl} (SMA5 {'>' if sma5 > sma20 else '<'} SMA20 | {rsi_lbl} | "
        f"{vol_lbl} ATR {atr_pct:.1f}% | mom {mom:+.1f}% | {macd_lbl})."
    )

    smc_parts: list[str] = []
    if bos == "BOS_UP":
        smc_parts.append("BOS alcista" + (" + volumen" if vol_ok else ""))
    elif bos == "BOS_DOWN":
        smc_parts.append("BOS bajista" + (" + volumen" if vol_ok else ""))

    if choch == "CHOCH_UP":
        smc_parts.append("CHoCH: posible giro alcista")
    elif choch == "CHOCH_DOWN":
        smc_parts.append("CHoCH: posible giro bajista — precaución")

    if nearest_fvg_ is not None:
        mid = (nearest_fvg_.high + nearest_fvg_.low) / 2
        tipo = "alcista" if nearest_fvg_.type == "bullish" else "bajista"
        smc_parts.append(f"FVG {tipo} sin llenar ${mid:,.0f} ({nearest_fvg_.low:,.0f}–{nearest_fvg_.high:,.0f})")

    if smc_parts:
        base += " SMC: " + " · ".join(smc_parts) + "."

    return base
