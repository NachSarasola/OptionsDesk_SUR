"""Smart Money Concepts — motor de análisis adaptado al MERVAL.

Adaptación local vs. libro (Forex/cripto):
- "Market Makers" = ALyCs locales, FCI y FGS (ANSES). Baja profundidad del mercado
  ⇒ FVG/imbalances son naturales (liquidez escasa), no siempre manipulación deliberada.
- Sesiones: rueda BYMA 11:00–17:00 ART (lun–vie).
  Killzone manipulación: 11:00–12:00 ART (coincide con apertura NYSE del ADR).
  Killzone distribución: 15:30–17:00 ART (cierre de ALyCs).
- Precio en pesos ≈ ADR × CCL; confluencia ADR en providers/adr.py (Fase 2).
- Armónicos M/W en signals/harmonics.py (Fase 3). SmcContext tiene el hook.
- Volumen real solo en velas diarias (tape intradía tiene volume=0): señales
  dependientes de volumen solo se calculan con daily df.

Módulo en pandas puro — sin dependencia de smartmoneyconcepts ni pandas_ta.
La librería smartmoneyconcepts sigue siendo usada por technical.py (BOS/CHoCH/FVG/OB);
aquí recibimos ese snap como parámetro, no recalculamos.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger(__name__)

_BA_TZ = ZoneInfo("America/Argentina/Buenos_Aires")

# ── Killzones MERVAL ──────────────────────────────────────────────────────────
_KZ_MANIP_START  = time(11, 0)
_KZ_MANIP_END    = time(12, 0)
_KZ_DIST_START   = time(15, 30)
_KZ_DIST_END     = time(17, 0)

# Tolerancia para Equal Highs/Lows (mismo nivel si precio dentro de %)
_EQL_TOLERANCE = 0.005   # 0.5%

# Barras de lookback para PDA array (~3 meses de ruedas diarias)
_PDA_LOOKBACK = 60


def _learned(name: str, default: float) -> float:
    """Lee un parametro aprendido por el learner; default si no hay (==hoy).

    Import perezoso para no acoplar smc.py a backtest en import-time y para que
    cualquier fallo degrade silenciosamente al valor de fabrica.
    """
    try:
        from optionsdesk.backtest.param_store import param
        return param(name, default)
    except Exception:
        return default


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class LiquidityLevel:
    """Nivel de liquidez: zona con acumulación de stops y órdenes pendientes.

    BSL (Buy Side Liquidity): sobre máximos — buy stops de shorts + SL de longs.
    SSL (Sell Side Liquidity): bajo mínimos — sell stops de longs + SL de shorts.
    `swept=True` si el precio ya activó esos stops (nivel mitigado).
    """
    kind: str       # "BSL" | "SSL"
    price: float
    label: str      # "PDH"|"PDL"|"PWH"|"PWL"|"EQH"|"EQL"|"SWING_H"|"SWING_L"
    swept: bool
    timeframe: str  # "daily" | "weekly"


@dataclass
class PdaArray:
    """Interbank Price Delivery Algorithm — contexto macro premium/discount.

    Fibonacci trazado entre el swing alto y bajo más recientes del HTF.
    Premium zone (> 0.5): distribución, zona de calls.
    Discount zone (< 0.5): acumulación, zona de puts.
    """
    swing_high: float
    swing_low: float
    equilibrium: float       # 0.5 fib
    premium_band: float      # 0.618 fib (inicio zona premium)
    discount_band: float     # 0.382 fib (límite discount)
    zone: str                # "premium" | "discount" | "equilibrium"
    ote_bullish_lo: float    # 0.618 fib (OTE bullish entrada)
    ote_bullish_hi: float    # 0.786 fib
    ote_bearish_lo: float    # 0.786 fib (OTE bearish entrada)
    ote_bearish_hi: float    # 0.886 fib


@dataclass
class OteZone:
    """Zona de Entrada Óptima (Optimal Trade Entry) del swing activo.

    Bullish OTE: retroceso 0.618–0.786 del impulso previo (comprar en descuento).
    Bearish OTE: retroceso 0.618–0.786 desde el techo (vender en pullback).
    Niveles del libro: 0.618 | 0.705 | 0.786 | 0.886 (OTE FYRS / sniper zone).
    """
    direction: str   # "bullish" | "bearish"
    lo: float        # límite inferior de la zona OTE
    hi: float        # límite superior
    fib_618: float
    fib_705: float
    fib_786: float
    fib_886: float


@dataclass
class LiquiditySweep:
    """Stop hunt / Turtle Soup: precio barrió un nivel y revirtió.

    Un sweep es cuando el precio cruza brevemente un nivel de liquidez pero cierra
    de vuelta del lado original → falso quiebre → señal de reversión potencial.

    Adaptación MERVAL: en un mercado ilíquido (BYMA) un "barrido" puede ser un
    air-pocket de book vacío (nadie vendió deliberadamente, simplemente no había
    puntas). `volume_confirmed` distingue un barrido institucional real (alto volumen)
    de uno por falta de liquidez. Solo confiar en sweeps con volume_confirmed=True
    para señales de alta calidad (SMC_REVERSAL, sniper entry).
    """
    level: LiquidityLevel
    direction: str    # "up" (barrió BSL) | "down" (barrió SSL)
    reversed: bool    # True si cerró de vuelta (confirmación de stop hunt)
    bar_index: int    # posición en el df donde ocurrió
    volume_confirmed: bool = False   # True si la vela del barrido tuvo volumen > media


@dataclass
class MmRead:
    """Lectura del ciclo Market Maker (vol 2 — adaptación de Steve Mauro al MERVAL).

    Stack de EMAs 5/13/50/200 como proxy del estado del ciclo MM:
    - Nivel 1: EMAs aplanadas → movimiento inicial agresivo (MM driven).
    - Nivel 2: cruce EMA 13/50 → momentum emotivo de retail.
    - Nivel 3: abanico completo → toma de ganancias, posible Peak Formation.
    """
    ema5: float
    ema13: float
    ema50: float
    ema200: float
    stack: str            # "bullish" | "bearish" | "mixed"
    level_count: int      # 0–3 (niveles del ciclo completados)
    peak_formation: bool  # True si hay retest de swing extremo reciente
    phase: str            # "acumulacion"|"manipulacion"|"distribucion"|"desarrollo"
    ema13_cross_50: str   # "above" | "below" | "crossing"


@dataclass
class SmcSignal:
    """Output de trading unificado — responde qué hacer, desde dónde y hasta dónde.

    Es el resultado del cascade multi-timeframe adaptado al MERVAL:
      Weekly (bias) → Daily (zona + estructura) → Volumen + Divergencia (confirmación)

    Diseño:
    - `direction`: "long" | "short" | "neutral"
    - `quality`:   "S" | "A" | "B" | "C" | "none"  (como signal_grade en stocks)
    - `favors_short_put`:    True cuando el contexto favorece vender puts de GGAL
    - `favors_covered_call`: True cuando el contexto favorece lanzar calls cubiertos

    Quality mapping:
      S — htf_long + discount + OTE + (ssl_sweep_volumed OR rsi_bull_div) + BOS_UP
      A — htf_long + (discount + BOS_UP) OR (ssl_sweep + discount)
      B — htf_long + BOS_UP (sin zona precisa)
      C — señal local sin confirmación HTF (solo para renta, no acciones)
      none — conflicto o sin datos
    """
    direction: str = "neutral"      # "long" | "short" | "neutral"
    quality: str   = "none"         # "S" | "A" | "B" | "C" | "none"

    # Zona de entrada
    entry_lo: float = 0.0
    entry_hi: float = 0.0
    entry_label: str = ""           # "OTE bullish $X–$Y" | "OB bullish $X–$Y"

    # Target (próxima zona de liquidez a barrer)
    target: float = 0.0
    target_label: str = ""          # "BSL PWH $X" | "BSL PDH $X"

    # Stop (invalidación estructural)
    stop: float = 0.0
    stop_label: str = ""            # "Bajo OTE 0.886 $X" | "Bajo OB $X"

    # Factores que componen la señal
    htf_bias: str = "neutral"       # "long" | "short" | "neutral" (weekly)
    pda_zone: str = "equilibrium"   # "discount" | "premium" | "equilibrium"
    rsi_div: str = "none"           # "bullish" | "bearish" | "none"
    ssl_sweep: bool = False         # sweep SSL con reversión en los últimos 5 días
    bsl_sweep: bool = False         # sweep BSL con reversión
    vol_confirmed: bool = False     # RVOL > 1.3 en la vela de señal (si daily)
    obv_alcista: bool = False       # OBV acumulando (posición silenciosa)
    killzone_active: bool = False
    in_ote: bool = False            # spot actual dentro de la zona OTE

    # Texto explicativo
    reason: str = ""

    # Flags de opciones — responde directamente qué estrategia usar
    favors_short_put: bool = False      # vender puts (long bias + discount)
    favors_covered_call: bool = False   # call cubierto (short/neutral + premium)
    favors_wait: bool = True            # esperar mejor setup


@dataclass
class SmcContext:
    """Contexto SMC completo para un subyacente.

    Se adjunta a MarketContext.smc y fluye por recommender + dashboard.
    Todos los campos son opcionales para no bloquear si algún cálculo falla.
    """
    # Estructura (de technical.py TechnicalSnapshot)
    bos: Optional[str] = None     # "BOS_UP" | "BOS_DOWN"
    choch: Optional[str] = None   # "CHOCH_UP" | "CHOCH_DOWN"
    # Swings confirmados (DataFrame: index, HighLow, Level, [date])
    swings: Optional[pd.DataFrame] = None
    # Liquidez
    liquidity: list = field(default_factory=list)  # list[LiquidityLevel]
    # PDA array
    pda: Optional[PdaArray] = None
    # OTE del swing activo
    ote: Optional[OteZone] = None
    # Stop hunts / sweeps recientes
    sweeps: list = field(default_factory=list)     # list[LiquiditySweep]
    # Market Maker read (EMAs + ciclo)
    mm: Optional[MmRead] = None
    # Killzone activa en BYMA (None fuera de horario)
    killzone: Optional[str] = None
    # ADR/CCL confluence — hook Fase 2 (None hasta que adr.py esté conectado)
    adr_confluence: Optional[str] = None
    # Armónicos M/W — hook Fase 3 (harmonics.py)
    peak_formations: list = field(default_factory=list)
    # SmcSignal — output de trading unificado (generado al final de analyze_smc)
    signal: Optional["SmcSignal"] = None


# ── Swing points (fractales) ──────────────────────────────────────────────────

def swing_points(df: pd.DataFrame, length: int = 3) -> pd.DataFrame:
    """Detecta swing highs y swing lows confirmados sin lookahead.

    Un swing high en la barra i queda confirmado cuando las `length` barras
    posteriores tienen highs inferiores. Análogamente para swing lows.
    Retorna DataFrame con columnas: index, HighLow (1=high/-1=low), Level, [date].
    """
    if df is None or len(df) < 2 * length + 1:
        return pd.DataFrame(columns=["index", "HighLow", "Level"])

    hi = df["high"].astype(float).values
    lo = df["low"].astype(float).values
    n  = len(hi)
    rows: list[dict] = []

    for i in range(length, n - length):
        # Swing high: máximo en i mayor que los `length` highs anteriores y posteriores
        if hi[i] > max(hi[i - length : i]) and hi[i] >= max(hi[i + 1 : i + length + 1]):
            row: dict = {"index": i, "HighLow": 1, "Level": float(hi[i])}
            if "date" in df.columns:
                row["date"] = df["date"].iloc[i]
            rows.append(row)
            continue  # no puede ser swing alto y bajo simultáneamente

        # Swing low: mínimo en i menor que los `length` lows anteriores y posteriores
        if lo[i] < min(lo[i - length : i]) and lo[i] <= min(lo[i + 1 : i + length + 1]):
            row = {"index": i, "HighLow": -1, "Level": float(lo[i])}
            if "date" in df.columns:
                row["date"] = df["date"].iloc[i]
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["index", "HighLow", "Level"])
    return pd.DataFrame(rows)


# ── Mapa de liquidez BSL/SSL ──────────────────────────────────────────────────

def liquidity_map(
    daily: pd.DataFrame,
    weekly: Optional[pd.DataFrame] = None,
    spot: Optional[float] = None,
) -> list[LiquidityLevel]:
    """Extrae niveles de liquidez (BSL y SSL) de datos diarios y semanales.

    Incluye: PDH/PDL, PWH/PWL, EQH/EQL y swing highs/lows recientes.
    Un nivel se marca como swept=True si el precio actual ya lo cruzó.
    """
    levels: list[LiquidityLevel] = []
    if daily is None or daily.empty or len(daily) < 2:
        return levels

    close_ref = spot if spot is not None else float(daily["close"].iloc[-1])

    # PDH / PDL ────────────────────────────────────────────────────────────────
    if len(daily) >= 2:
        prev = daily.iloc[-2]
        pdh  = float(prev["high"])
        pdl  = float(prev["low"])
        levels.append(LiquidityLevel("BSL", pdh, "PDH", pdh < close_ref, "daily"))
        levels.append(LiquidityLevel("SSL", pdl, "PDL", pdl > close_ref, "daily"))

    # PWH / PWL ────────────────────────────────────────────────────────────────
    if weekly is not None and len(weekly) >= 2:
        pw    = weekly.iloc[-2]
        pwh   = float(pw["high"])
        pwl   = float(pw["low"])
        levels.append(LiquidityLevel("BSL", pwh, "PWH", pwh < close_ref, "weekly"))
        levels.append(LiquidityLevel("SSL", pwl, "PWL", pwl > close_ref, "weekly"))

    # Swing highs/lows recientes ───────────────────────────────────────────────
    swings = swing_points(daily, length=3)
    if not swings.empty:
        sh_prices = swings.loc[swings["HighLow"] == 1, "Level"].values
        sl_prices = swings.loc[swings["HighLow"] == -1, "Level"].values

        eqh_clusters = _find_equal_clusters(sh_prices)
        eql_clusters = _find_equal_clusters(sl_prices)

        seen_bsl: set[int] = set()
        seen_ssl: set[int] = set()

        for price in sh_prices[-5:]:
            key = round(price)
            if key in seen_bsl:
                continue
            seen_bsl.add(key)
            label = "EQH" if _in_cluster(price, eqh_clusters) else "SWING_H"
            levels.append(LiquidityLevel("BSL", price, label, price < close_ref, "daily"))

        for price in sl_prices[-5:]:
            key = round(price)
            if key in seen_ssl:
                continue
            seen_ssl.add(key)
            label = "EQL" if _in_cluster(price, eql_clusters) else "SWING_L"
            levels.append(LiquidityLevel("SSL", price, label, price > close_ref, "daily"))

    return _dedup_levels(levels)


def _find_equal_clusters(prices: "list | tuple | object") -> list[float]:
    """Retorna centros de clusters de precios dentro de EQL_TOLERANCE."""
    prices_list = list(prices)
    if len(prices_list) < 2:
        return []
    tol = _learned("smc_eql_tolerance", _EQL_TOLERANCE)
    clusters: list[float] = []
    for p in prices_list:
        fp = float(p)
        if any(abs(fp - c) / max(abs(c), 1e-9) <= tol for c in clusters):
            continue
        count = sum(
            1 for q in prices_list
            if abs(fp - float(q)) / max(abs(float(q)), 1e-9) <= tol
        )
        if count >= 2:
            clusters.append(fp)
    return clusters


def _in_cluster(price: float, clusters: list[float]) -> bool:
    tol = _learned("smc_eql_tolerance", _EQL_TOLERANCE)
    return any(abs(price - c) / max(abs(c), 1e-9) <= tol for c in clusters)


def _dedup_levels(levels: list[LiquidityLevel]) -> list[LiquidityLevel]:
    """Elimina duplicados dentro de 0.3% de precio del mismo kind."""
    out: list[LiquidityLevel] = []
    for lv in levels:
        if not any(
            abs(lv.price - x.price) / max(x.price, 1e-9) < 0.003 and lv.kind == x.kind
            for x in out
        ):
            out.append(lv)
    return out


# ── PDA Array ─────────────────────────────────────────────────────────────────

def pda_array(daily: pd.DataFrame, spot: float) -> Optional[PdaArray]:
    """Calcula el contexto macro premium/discount con Fibonacci sobre el swing mayor.

    Traza fib desde el swing low hasta el swing high del período (o vice-versa).
    Usa los últimos _PDA_LOOKBACK períodos para el swing de referencia.
    """
    if daily is None or len(daily) < 10:
        return None

    window = daily.tail(int(_learned("smc_pda_lookback", _PDA_LOOKBACK)))
    swings = swing_points(window, length=3)

    if swings.empty:
        sh = float(window["high"].max())
        sl = float(window["low"].min())
    else:
        sh_rows = swings[swings["HighLow"] == 1]
        sl_rows = swings[swings["HighLow"] == -1]
        sh = float(sh_rows["Level"].max()) if not sh_rows.empty else float(window["high"].max())
        sl = float(sl_rows["Level"].min()) if not sl_rows.empty else float(window["low"].min())

    rng = sh - sl
    if rng <= 0:
        return None

    eq             = sl + 0.500 * rng
    premium_band   = sl + 0.618 * rng
    discount_band  = sl + 0.382 * rng
    ote_bull_lo    = sl + 0.618 * rng
    ote_bull_hi    = sl + 0.786 * rng
    ote_bear_lo    = sl + 0.786 * rng
    ote_bear_hi    = sl + 0.886 * rng

    if spot >= eq * 1.005:
        zone = "premium"
    elif spot <= eq * 0.995:
        zone = "discount"
    else:
        zone = "equilibrium"

    return PdaArray(
        swing_high=sh,
        swing_low=sl,
        equilibrium=eq,
        premium_band=premium_band,
        discount_band=discount_band,
        zone=zone,
        ote_bullish_lo=ote_bull_lo,
        ote_bullish_hi=ote_bull_hi,
        ote_bearish_lo=ote_bear_lo,
        ote_bearish_hi=ote_bear_hi,
    )


# ── OTE Zone ──────────────────────────────────────────────────────────────────

def ote_zone(
    swing_lo: float, swing_hi: float, direction: str
) -> Optional[OteZone]:
    """Zona de entrada óptima para el swing activo (retroceso 0.618–0.886).

    Bullish: precio cayó a retroceso → zona de compra en 0.618–0.786.
    Bearish: precio subió a retroceso → zona de venta en 0.618–0.786 desde el top.
    Niveles del libro: 0.618 | 0.705 | 0.786 | 0.886 (FYRS/sniper zone).
    """
    rng = swing_hi - swing_lo
    if rng <= 0:
        return None

    if direction == "bullish":
        f618 = swing_lo + 0.618 * rng
        f705 = swing_lo + 0.705 * rng
        f786 = swing_lo + 0.786 * rng
        f886 = swing_lo + 0.886 * rng
        return OteZone(
            direction="bullish",
            lo=f618, hi=f786,
            fib_618=f618, fib_705=f705, fib_786=f786, fib_886=f886,
        )
    else:  # bearish: retroceso desde el techo hacia abajo
        f618 = swing_hi - 0.618 * rng
        f705 = swing_hi - 0.705 * rng
        f786 = swing_hi - 0.786 * rng
        f886 = swing_hi - 0.886 * rng
        return OteZone(
            direction="bearish",
            lo=f786, hi=f618,
            fib_618=f618, fib_705=f705, fib_786=f786, fib_886=f886,
        )


# ── Detect Sweeps ─────────────────────────────────────────────────────────────

def detect_sweeps(
    df: pd.DataFrame,
    levels: list[LiquidityLevel],
    lookback: int = 5,
    volume_mult: float = 1.2,
) -> list[LiquiditySweep]:
    """Detecta stop hunts (Turtle Soup) en las últimas `lookback` velas.

    Un sweep es: high > nivel_BSL pero close < nivel_BSL → posible reversal bajista.
    O:           low < nivel_SSL pero close > nivel_SSL → posible reversal alcista.

    `volume_confirmed`: la vela del barrido tiene volumen >= `volume_mult`× la media
    del resto del lookback. Solo aplica cuando hay datos de volumen reales (daily).
    El tape intradía de BYMA tiene volume=0, así que no filtra falsamente.
    En BYMA ilíquido, un sweep sin volumen es probablemente un air-pocket de book
    vacío, no un barrido institucional real.
    """
    sweeps: list[LiquiditySweep] = []
    if df is None or len(df) < 2:
        return sweeps

    check = df.tail(lookback + 1).reset_index(drop=True)
    offset = max(0, len(df) - lookback - 1)

    # Pre-computar la media de volumen del lookback para confirmar barridos.
    has_volume = "volume" in df.columns
    vol_mean = 0.0
    if has_volume:
        vol_series = df["volume"].astype(float)
        vol_mean = float(vol_series.tail(lookback + 10).mean() or 0.0)

    for lv in levels:
        p = lv.price
        for i, bar in check.iterrows():
            bar_hi = float(bar["high"])
            bar_lo = float(bar["low"])
            bar_cl = float(bar["close"])
            bar_vol = float(bar.get("volume", 0.0) or 0.0) if has_volume else 0.0

            # volume_confirmed: volumen real presente y superior a la media.
            vol_conf = (
                bar_vol > 0
                and vol_mean > 0
                and bar_vol >= vol_mean * volume_mult
            ) if has_volume and vol_mean > 0 else False

            if lv.kind == "BSL" and bar_hi > p * 1.001:
                sweeps.append(LiquiditySweep(
                    level=lv,
                    direction="up",
                    reversed=bar_cl < p,
                    bar_index=offset + int(i),
                    volume_confirmed=vol_conf,
                ))
            elif lv.kind == "SSL" and bar_lo < p * 0.999:
                sweeps.append(LiquiditySweep(
                    level=lv,
                    direction="down",
                    reversed=bar_cl > p,
                    bar_index=offset + int(i),
                    volume_confirmed=vol_conf,
                ))

    return sweeps


# ── Market Maker Read ─────────────────────────────────────────────────────────

def mm_read(
    daily: pd.DataFrame,
    ema_periods: tuple[int, ...] = (5, 13, 50, 200),
    now: Optional[datetime] = None,
) -> Optional[MmRead]:
    """Lectura del ciclo Market Maker vía stack de EMAs.

    EMAs calculadas en pandas puro (mismo fallback que technical.py).
    No depende de pandas_ta ni smartmoneyconcepts.
    """
    if daily is None or len(daily) < max(ema_periods):
        return None

    try:
        from optionsdesk.signals.technical import ema as _ema_fn
    except ImportError:
        def _ema_fn(series: pd.Series, n: int) -> pd.Series:  # type: ignore[misc]
            return series.ewm(span=n, adjust=False, min_periods=n).mean()

    close = daily["close"].astype(float)
    p5, p13, p50, p200 = ema_periods[0], ema_periods[1], ema_periods[2], ema_periods[3]

    ema5_s   = _ema_fn(close, p5)
    ema13_s  = _ema_fn(close, p13)
    ema50_s  = _ema_fn(close, p50)
    ema200_s = _ema_fn(close, p200)

    e5   = float(ema5_s.iloc[-1])
    e13  = float(ema13_s.iloc[-1])
    e50  = float(ema50_s.iloc[-1])
    e200 = float(ema200_s.iloc[-1])

    if any(pd.isna(x) for x in (e5, e13, e50, e200)):
        return None

    # Stack alignment
    if e5 > e13 > e50 > e200:
        stack = "bullish"
    elif e5 < e13 < e50 < e200:
        stack = "bearish"
    else:
        stack = "mixed"

    # Cruce EMA 13/50 (señal de nivel 2)
    if len(close) >= p50 + 2:
        prev_e13 = float(ema13_s.iloc[-2])
        prev_e50 = float(ema50_s.iloc[-2])
        if (e13 > e50 and prev_e13 <= prev_e50) or (e13 < e50 and prev_e13 >= prev_e50):
            cross = "crossing"
        elif e13 > e50:
            cross = "above"
        else:
            cross = "below"
    else:
        cross = "above" if e13 > e50 else "below"

    level_count = _count_mm_levels(close)
    pf          = _detect_peak_formation(daily)
    phase       = _phase_from_killzone(now)

    return MmRead(
        ema5=round(e5, 2),
        ema13=round(e13, 2),
        ema50=round(e50, 2),
        ema200=round(e200, 2),
        stack=stack,
        level_count=level_count,
        peak_formation=pf,
        phase=phase,
        ema13_cross_50=cross,
    )


def _count_mm_levels(close: pd.Series, min_days: int = 3) -> int:
    """Cuenta cambios de dirección sostenidos (~3+ días) en el ciclo MM."""
    if len(close) < min_days * 2:
        return 0
    pct = close.pct_change()
    rolling_sum = pct.rolling(min_days).sum()
    direction   = (rolling_sum > 0.01).astype(int) - (rolling_sum < -0.01).astype(int)
    flips       = int((direction.diff().abs() > 0).sum())
    return min(flips, 3)


def _detect_peak_formation(daily: pd.DataFrame, lookback: int = 10) -> bool:
    """True si el precio reciente testea un swing extremo previo (señal PF)."""
    if len(daily) < lookback + 3:
        return False
    swings = swing_points(daily.tail(lookback + 6), length=3)
    if swings.empty:
        return False
    last_cl = float(daily["close"].iloc[-1])
    last_hi = float(daily["high"].iloc[-1])
    last_lo = float(daily["low"].iloc[-1])
    for _, row in swings.iterrows():
        lv = row["Level"]
        tol = max(lv, 1e-9) * 0.012
        if abs(last_cl - lv) < tol or abs(last_hi - lv) < tol * 0.5 or abs(last_lo - lv) < tol * 0.5:
            return True
    return False


def _phase_from_killzone(now: Optional[datetime]) -> str:
    kz = current_killzone(now)
    if kz == "manipulacion":
        return "manipulacion"
    if kz == "distribucion":
        return "distribucion"
    ts = now or datetime.now()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=_BA_TZ)
    else:
        ts = ts.astimezone(_BA_TZ)
    t = ts.time()
    if time(11, 0) <= t < time(17, 0):
        return "desarrollo"
    return "acumulacion"


# ── Killzone ──────────────────────────────────────────────────────────────────

def current_killzone(now: Optional[datetime] = None) -> Optional[str]:
    """Devuelve la killzone activa en Buenos Aires o None si fuera de zona.

    Killzones MERVAL (adaptación de las sesiones del libro al mercado local):
    - 'manipulacion': 11:00–12:00 ART (apertura + NYSE del ADR).
    - 'distribucion': 15:30–17:00 ART (cierre de ALyCs).
    - None: fuera de killzone (desarrollo 12:00–15:30, o fuera de horario).
    """
    ts = now or datetime.now()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=_BA_TZ)
    else:
        ts = ts.astimezone(_BA_TZ)

    if ts.weekday() >= 5:
        return None

    t = ts.time()
    if _KZ_MANIP_START <= t < _KZ_MANIP_END:
        return "manipulacion"
    if _KZ_DIST_START <= t < _KZ_DIST_END:
        return "distribucion"
    return None


# ── Orquestador principal ─────────────────────────────────────────────────────

def analyze_smc(
    daily: pd.DataFrame,
    weekly: Optional[pd.DataFrame] = None,
    intraday: Optional[pd.DataFrame] = None,
    spot: Optional[float] = None,
    now: Optional[datetime] = None,
    snap=None,    # TechnicalSnapshot opcional (de technical.analyze) — evita import circular
    adr: Optional[dict] = None,   # hook Fase 2: {"confluence": "confirmed"|"fx_driven"|"unknown"}
) -> SmcContext:
    """Construye el SmcContext completo.

    Diseñado para ser llamado desde compute_market_context() en directional.py.
    Nunca lanza excepciones: si un subcomponente falla, el campo queda en None/[].
    """
    ctx = SmcContext()

    if daily is None or daily.empty:
        return ctx

    _spot = spot if spot is not None else (
        float(daily["close"].iloc[-1]) if "close" in daily.columns and len(daily) > 0 else 0.0
    )

    # Estructura de technical.py
    if snap is not None:
        ctx.bos   = getattr(snap, "bos",   None)
        ctx.choch = getattr(snap, "choch", None)

    # Swings confirmados
    try:
        ctx.swings = swing_points(daily, length=3)
    except Exception as exc:
        logger.debug("swing_points fallo: %s", exc)

    # Mapa de liquidez BSL/SSL
    try:
        ctx.liquidity = liquidity_map(daily, weekly, _spot)
    except Exception as exc:
        logger.debug("liquidity_map fallo: %s", exc)

    # PDA array premium/discount
    try:
        ctx.pda = pda_array(daily, _spot)
    except Exception as exc:
        logger.debug("pda_array fallo: %s", exc)

    # OTE del swing activo
    try:
        if ctx.swings is not None and not ctx.swings.empty:
            sh_rows = ctx.swings[ctx.swings["HighLow"] == 1]
            sl_rows = ctx.swings[ctx.swings["HighLow"] == -1]
            if not sh_rows.empty and not sl_rows.empty:
                last_sh_i = int(sh_rows["index"].iloc[-1])
                last_sl_i = int(sl_rows["index"].iloc[-1])
                last_sh   = float(sh_rows["Level"].iloc[-1])
                last_sl   = float(sl_rows["Level"].iloc[-1])
                direction = "bullish" if last_sl_i > last_sh_i else "bearish"
                ctx.ote = ote_zone(last_sl, last_sh, direction)
    except Exception as exc:
        logger.debug("ote_zone fallo: %s", exc)

    # Stop hunts / sweeps
    try:
        if ctx.liquidity:
            ctx.sweeps = detect_sweeps(daily, ctx.liquidity, lookback=5)
    except Exception as exc:
        logger.debug("detect_sweeps fallo: %s", exc)

    # Market Maker read
    try:
        from optionsdesk.config.settings import settings as _settings
        ema_periods = getattr(_settings, "mm_ema_periods", (5, 13, 50, 200))
    except Exception:
        ema_periods = (5, 13, 50, 200)

    try:
        ctx.mm = mm_read(daily, ema_periods=ema_periods, now=now)
    except Exception as exc:
        logger.debug("mm_read fallo: %s", exc)

    # Killzone activa
    try:
        ctx.killzone = current_killzone(now)
    except Exception as exc:
        logger.debug("current_killzone fallo: %s", exc)

    # ADR confluence hook (Fase 2)
    if adr is not None:
        ctx.adr_confluence = adr.get("confluence")

    # Armónicos M/W Peak Formation (Fase 3)
    try:
        from optionsdesk.config.settings import settings as _s2
        if getattr(_s2, "harmonics_enabled", False):
            from optionsdesk.signals.harmonics import find_peak_formations as _find_pf
            if ctx.swings is not None and not ctx.swings.empty:
                ctx.peak_formations = _find_pf(ctx.swings, ctx.pda, daily)
    except Exception as exc:
        logger.debug("harmonics fallo: %s", exc)

    # SmcSignal — señal de trading directa (multi-TF cascade)
    try:
        ctx.signal = generate_smc_signal(ctx, weekly_daily=weekly, spot=_spot, daily=daily)
    except Exception as exc:
        logger.debug("generate_smc_signal fallo: %s", exc)

    return ctx


# ── RSI Divergence ────────────────────────────────────────────────────────────

def rsi_divergence(daily: pd.DataFrame, length: int = 14, lookback: int = 12) -> str:
    """Detecta divergencia RSI/precio — funciona bien en MERVAL (bajo ruido).

    Bullish: precio hace mínimo más bajo pero RSI hace mínimo más alto → reversión.
    Bearish: precio hace máximo más alto pero RSI hace máximo más bajo → distribución.

    En MERVAL, la divergencia bullish en zona discount + SSL sweep = sniper entry.
    En MERVAL, la divergencia bearish en zona premium = distribución institucional.
    """
    try:
        from optionsdesk.signals.technical import rsi as _rsi_fn
    except ImportError:
        return "none"

    if daily is None or len(daily) < length + lookback:
        return "none"

    close  = daily["close"].astype(float)
    rsi_s  = _rsi_fn(close, length).dropna()
    if len(rsi_s) < lookback:
        return "none"

    half     = max(lookback // 2, 3)
    # Ventana reciente (segunda mitad) vs anterior (primera mitad)
    prev_cl  = close.iloc[-lookback : -half]
    last_cl  = close.iloc[-half:]
    prev_rsi = rsi_s.iloc[-lookback : -half]
    last_rsi = rsi_s.iloc[-half:]

    if prev_cl.empty or last_cl.empty or prev_rsi.empty or last_rsi.empty:
        return "none"

    # Bullish divergence: precio hace mínimo más bajo, RSI no confirma
    if (float(last_cl.min()) < float(prev_cl.min()) * 0.995
            and float(last_rsi.min()) > float(prev_rsi.min()) + 1.5):
        return "bullish"

    # Bearish divergence: precio hace máximo más alto, RSI no confirma
    if (float(last_cl.max()) > float(prev_cl.max()) * 1.005
            and float(last_rsi.max()) < float(prev_rsi.max()) - 1.5):
        return "bearish"

    return "none"


# ── Volume context from volume_momentum ──────────────────────────────────────

def _volume_context(daily: pd.DataFrame) -> dict:
    """Extrae RVOL, OBV trend y confirmación de volumen (solo en daily con vol real).

    Delega a volume_momentum.py (RVOL / OBV / vol_trend). Retorna dict vacío
    si volume_momentum no está disponible o el volumen es 0 (intradía BYMA).
    """
    if daily is None or len(daily) < 5:
        return {}
    if "volume" not in daily.columns or float(daily["volume"].iloc[-1]) == 0:
        return {}
    try:
        from optionsdesk.signals.volume_momentum import volume_snapshot, volume_score_delta
        snap = volume_snapshot(daily)
        if snap is None:
            return {}
        delta, notes = volume_score_delta(snap)
        return {
            "rvol": snap.rvol,
            "rvol_label": snap.rvol_label,
            "obv_trend": snap.obv_trend,
            "obv_divergence": snap.obv_divergence,
            "price_vol_ok": snap.price_vol_ok,
            "vol_score_delta": delta,
        }
    except Exception:
        return {}


# ── SmcSignal cascade ─────────────────────────────────────────────────────────

def generate_smc_signal(
    ctx: SmcContext,
    weekly_daily: Optional[pd.DataFrame] = None,
    daily: Optional[pd.DataFrame] = None,
    spot: float = 0.0,
) -> SmcSignal:
    """Cascade multi-timeframe → SmcSignal de trading directo.

    Jerarquía de pesos (adaptada al MERVAL):
    1. HTF bias (semanal)  — nunca operar contra la tendencia semanal
    2. Zona PDA (diario)   — discount para longs, premium para shorts
    3. Estructura (BOS)    — confirmación direccional
    4. OTE                 — zona de entrada óptima
    5. Sweeps + Volumen    — confirman que el movimiento es institucional
    6. RSI divergencia     — confirmación adicional de reversión
    7. Killzone            — timing (esperar fuera de apertura)

    Para opciones GGAL:
    - favors_short_put:    long bias + discount + BOS_UP (sin conflicto)
    - favors_covered_call: short/neutral bias + premium + (BOS_DOWN or CHoCH_DOWN)
    """
    sig = SmcSignal()

    # ── 1. HTF bias — weekly primero, fallback a EMA50 del diario ────────────
    # Sin datos semanales, EMA50 del diario es el mejor proxy de tendencia "macro".
    # Si price > EMA50 → contexto alcista de mediano plazo. <EMA50 → bajista.
    htf_bias = "neutral"
    if weekly_daily is not None and len(weekly_daily) >= 8:
        try:
            from optionsdesk.signals.technical import analyze as _analyze_w
            w_snap = _analyze_w(weekly_daily)
            t = getattr(w_snap, "trend", "LATERAL")
            if t == "ALCISTA":
                htf_bias = "long"
            elif t == "BAJISTA":
                htf_bias = "short"
        except Exception:
            pass

    if htf_bias == "neutral" and daily is not None and len(daily) >= 50 and spot > 0:
        try:
            from optionsdesk.signals.technical import ema as _ema_htf
            e50 = float(_ema_htf(daily["close"].astype(float), 50).iloc[-1])
            if not pd.isna(e50):
                if spot > e50 * 1.015:      # más del 1.5% arriba → alcista
                    htf_bias = "long"
                elif spot < e50 * 0.985:    # más del 1.5% abajo → bajista
                    htf_bias = "short"
        except Exception:
            pass

    sig.htf_bias = htf_bias

    # ── 2. PDA zone ───────────────────────────────────────────────────────────
    pda = ctx.pda
    pda_zone = getattr(pda, "zone", "equilibrium") if pda is not None else "equilibrium"
    sig.pda_zone = pda_zone

    # ── 3. Estructura (BOS/CHoCH) ─────────────────────────────────────────────
    bos   = ctx.bos or ""
    choch = ctx.choch or ""

    # ── 4. OTE — usa los campos OTE del PDA (impulso primario), no ctx.ote ────
    # ctx.ote rastrea el ÚLTIMO swing (puede ser la corrección bajista).
    # Para un trade LONG, lo relevante es si el precio está en la zona OTE del
    # IMPULSO PRIMARIO (0.618–0.786 del rango PDA), no de la corrección actual.
    in_ote = False
    ote_lo_ref = 0.0
    ote_hi_ref = 0.0
    if pda is not None and spot > 0:
        if pda_zone in ("discount", "equilibrium"):
            # Long: OTE = 0.618–0.786 del rango PDA (zona de compra en el pullback)
            ote_lo_ref = getattr(pda, "ote_bullish_lo", 0.0)
            ote_hi_ref = getattr(pda, "ote_bullish_hi", 0.0)
            # Deep discount (debajo del OTE) también es zona válida con mayor riesgo
            in_ote = ote_lo_ref <= spot <= ote_hi_ref
        else:
            # Short: OTE = 0.786–0.886 del rango PDA (rebote en zona premium)
            ote_lo_ref = getattr(pda, "ote_bearish_lo", 0.0)
            ote_hi_ref = getattr(pda, "ote_bearish_hi", 0.0)
            in_ote = ote_lo_ref <= spot <= ote_hi_ref
    sig.in_ote = in_ote

    # ── 5. Sweeps ────────────────────────────────────────────────────────────
    sweeps = ctx.sweeps or []
    ssl_sweep = any(
        getattr(sw, "direction", "") == "down"
        and getattr(sw, "reversed", False)
        for sw in sweeps
    )
    bsl_sweep = any(
        getattr(sw, "direction", "") == "up"
        and getattr(sw, "reversed", False)
        for sw in sweeps
    )
    # En MERVAL: sweep con volume_confirmed vale el doble (institucional real)
    ssl_sweep_vol = any(
        getattr(sw, "direction", "") == "down"
        and getattr(sw, "reversed", False)
        and getattr(sw, "volume_confirmed", False)
        for sw in sweeps
    )
    sig.ssl_sweep = ssl_sweep
    sig.bsl_sweep = bsl_sweep

    # ── 6. Volumen (RVOL + OBV) ───────────────────────────────────────────────
    vol = _volume_context(daily) if daily is not None else {}
    rvol        = vol.get("rvol") or 1.0
    obv_trend   = vol.get("obv_trend", "lateral")
    price_vol_ok = vol.get("price_vol_ok", False)
    rvol_threshold = float(_learned("smc_rvol_threshold", 1.30))
    vol_confirmed  = rvol >= rvol_threshold and price_vol_ok
    sig.vol_confirmed = vol_confirmed
    sig.obv_alcista   = obv_trend == "alcista"

    # ── 7. RSI divergence ─────────────────────────────────────────────────────
    rsi_div = rsi_divergence(daily) if daily is not None else "none"
    sig.rsi_div = rsi_div

    # ── 8. Killzone ───────────────────────────────────────────────────────────
    sig.killzone_active = ctx.killzone == "manipulacion"   # distribución no bloquea

    # ── DIRECTION y QUALITY — cascade ─────────────────────────────────────────
    # Contamos puntos de confluencia
    bull_points = 0
    bear_points = 0

    # HTF (peso 3 — más importante)
    if htf_bias == "long":
        bull_points += 3
    elif htf_bias == "short":
        bear_points += 3

    # PDA zone (peso 2)
    if pda_zone == "discount":
        bull_points += 2
    elif pda_zone == "premium":
        bear_points += 2

    # BOS (peso 2)
    if bos == "BOS_UP":
        bull_points += 2
    elif bos == "BOS_DOWN":
        bear_points += 2

    # CHoCH (peso 1 — señal de cambio, no confirmación)
    if choch == "CHOCH_UP":
        bull_points += 1
    elif choch == "CHOCH_DOWN":
        bear_points += 1

    # OTE bullish (peso 2) / bearish (peso 2)
    if in_ote and ote is not None:
        if getattr(ote, "direction", "") == "bullish":
            bull_points += 2
        else:
            bear_points += 2

    # Sweeps (peso 2 — evidencia de stop hunt)
    if ssl_sweep:
        bull_points += 2
    if bsl_sweep:
        bear_points += 2
    # Bonus por volumen confirmado en el sweep
    if ssl_sweep_vol:
        bull_points += 1

    # RSI divergence (peso 1)
    if rsi_div == "bullish":
        bull_points += 1
    elif rsi_div == "bearish":
        bear_points += 1

    # OBV acumulación silenciosa (peso 1)
    if sig.obv_alcista:
        bull_points += 1
    elif obv_trend == "bajista":
        bear_points += 1

    # Volume confirmado en precio (peso 1)
    if vol_confirmed:
        if pda_zone == "discount":
            bull_points += 1
        elif pda_zone == "premium":
            bear_points += 1

    # ADR confluence (peso 1 — solo si disponible)
    adr = getattr(ctx, "adr_confluence", None)
    if adr == "confirmed":
        # ADR confirma → suma a whichever direction is dominant
        if bull_points > bear_points:
            bull_points += 1
        else:
            bear_points += 1
    elif adr == "fx_driven":
        # Movimiento por CCL → reduce confianza en cualquier dirección
        bull_points = max(0, bull_points - 1)
        bear_points = max(0, bear_points - 1)

    # Determinar dirección
    margin = bull_points - bear_points
    if margin >= 3:
        direction = "long"
    elif margin <= -3:
        direction = "short"
    else:
        direction = "neutral"
    sig.direction = direction

    # Quality scoring — MERVAL calibrado
    # S: señal de libro (todo alineado: HTF + zona + estructura + sweep o RSI)
    # A: HTF + zona + estructura
    # B: HTF + estructura (sin zona precisa)
    # C: señal local sin HTF
    if direction == "long":
        has_structure = (bos == "BOS_UP" or choch == "CHOCH_UP")
        has_zone      = (pda_zone == "discount" or in_ote)
        has_reversal  = (ssl_sweep or rsi_div == "bullish")
        if htf_bias == "long" and has_zone and has_structure and has_reversal:
            quality = "S"
        elif htf_bias == "long" and has_zone and has_structure:
            quality = "A"
        elif htf_bias == "long" and has_structure:
            quality = "B"
        elif has_structure and has_reversal:
            quality = "C"
        else:
            quality = "none"
    elif direction == "short":
        has_structure = (bos == "BOS_DOWN" or choch == "CHOCH_DOWN")
        has_zone      = (pda_zone == "premium" or in_ote)
        has_reversal  = (bsl_sweep or rsi_div == "bearish")
        if htf_bias == "short" and has_zone and has_structure and has_reversal:
            quality = "S"
        elif htf_bias == "short" and has_zone and has_structure:
            quality = "A"
        elif htf_bias == "short" and has_structure:
            quality = "B"
        elif has_structure and has_reversal:
            quality = "C"
        else:
            quality = "none"
    else:
        quality = "none"

    sig.quality = quality

    # ── Entry / Target / Stop ─────────────────────────────────────────────────
    _spot = spot if spot > 0 else 0.0
    if direction != "neutral" and _spot > 0:
        _fill_entry_target_stop(sig, ctx, direction, _spot)

    # ── Flags de opciones ─────────────────────────────────────────────────────
    # SHORT PUT: óptimo cuando el precio probablemente sube.
    # Umbrales más permisivos que acciones porque el vendedor de opciones tiene
    # el tiempo a favor — puede ganar aunque la dirección no sea perfecta.
    # Quality C se incluye: una señal local sin HTF es suficiente para renta.
    sig.favors_short_put = (
        direction == "long"
        and quality in ("S", "A", "B", "C")
        and pda_zone in ("discount", "equilibrium")
        and bos != "BOS_DOWN"           # sin estructura bajista confirmada
        and choch != "CHOCH_DOWN"       # sin giro de carácter bajista
        and not sig.killzone_active
    )

    # COVERED CALL: óptimo cuando el precio probablemente se frena o baja.
    # Cualquier señal moderada-débil con premium zone ya justifica la estrategia.
    sig.favors_covered_call = (
        direction in ("short", "neutral")
        and quality in ("S", "A", "B", "C")
        and pda_zone in ("premium", "equilibrium")
        and bos != "BOS_UP"
        and choch != "CHOCH_UP"
        and not sig.killzone_active
    )

    sig.favors_wait = not (sig.favors_short_put or sig.favors_covered_call)

    # ── Reason (legible en castellano) ────────────────────────────────────────
    parts: list[str] = []
    if htf_bias != "neutral":
        parts.append(f"Semanal {htf_bias}")
    if pda_zone != "equilibrium":
        parts.append(f"zona {pda_zone}")
    if bos:
        parts.append(bos.replace("_", " ").lower())
    if in_ote:
        parts.append("precio en OTE")
    if ssl_sweep:
        parts.append("stop-hunt SSL + reversal")
    if rsi_div != "none":
        parts.append(f"RSI divergencia {rsi_div}")
    if sig.obv_alcista:
        parts.append("OBV acumulando")
    if sig.killzone_active:
        parts.append("ESPERAR: killzone apertura activa")
    sig.reason = "  ·  ".join(parts) if parts else "sin confluencia suficiente"

    return sig


def _fill_entry_target_stop(
    sig: SmcSignal,
    ctx: SmcContext,
    direction: str,
    spot: float,
) -> None:
    """Rellena entry/target/stop del SmcSignal con la mejor información disponible."""
    liquidity = ctx.liquidity or []
    ote       = ctx.ote
    pda       = ctx.pda

    # Entry zone
    if ote is not None and sig.in_ote:
        ote_lo  = getattr(ote, "lo", 0.0)
        ote_hi  = getattr(ote, "hi", 0.0)
        sig.entry_lo    = ote_lo
        sig.entry_hi    = ote_hi
        sig.entry_label = f"OTE {getattr(ote,'direction','')} ${ote_lo:,.0f}–${ote_hi:,.0f}"
    elif pda is not None:
        eq = getattr(pda, "equilibrium", spot)
        if direction == "long":
            sig.entry_lo    = getattr(pda, "discount_band", spot * 0.98)
            sig.entry_hi    = eq
            sig.entry_label = f"zona discount ${sig.entry_lo:,.0f}–${eq:,.0f}"
        else:
            sig.entry_lo    = eq
            sig.entry_hi    = getattr(pda, "premium_band", spot * 1.02)
            sig.entry_label = f"zona premium ${eq:,.0f}–${sig.entry_hi:,.0f}"

    # Target (próxima liquidez en la dirección del trade)
    if direction == "long":
        bsl_candidates = [
            lv for lv in liquidity
            if getattr(lv, "kind", "") == "BSL"
            and not getattr(lv, "swept", True)
            and getattr(lv, "price", 0) > spot * 1.005
        ]
        if bsl_candidates:
            target_lv   = min(bsl_candidates, key=lambda lv: lv.price)
            sig.target  = target_lv.price
            sig.target_label = f"BSL {target_lv.label} ${target_lv.price:,.0f}"
        elif pda is not None:
            sh = getattr(pda, "swing_high", 0.0)
            sig.target       = sh
            sig.target_label = f"swing_high PDA ${sh:,.0f}"
    else:
        ssl_candidates = [
            lv for lv in liquidity
            if getattr(lv, "kind", "") == "SSL"
            and not getattr(lv, "swept", True)
            and getattr(lv, "price", 0) < spot * 0.995
        ]
        if ssl_candidates:
            target_lv   = max(ssl_candidates, key=lambda lv: lv.price)
            sig.target  = target_lv.price
            sig.target_label = f"SSL {target_lv.label} ${target_lv.price:,.0f}"
        elif pda is not None:
            sl = getattr(pda, "swing_low", 0.0)
            sig.target       = sl
            sig.target_label = f"swing_low PDA ${sl:,.0f}"

    # Stop (invalidación estructural)
    if ote is not None:
        f886 = getattr(ote, "fib_886", 0.0)
        if direction == "long" and f886 > 0:
            sig.stop       = f886 * 0.993
            sig.stop_label = f"bajo OTE 0.886 ${f886:,.0f}"
        elif direction == "short" and f886 > 0:
            sig.stop       = f886 * 1.007
            sig.stop_label = f"sobre OTE 0.886 ${f886:,.0f}"
    if sig.stop == 0.0:
        # Fallback: 2× ATR del swing bajo/alto
        if direction == "long" and pda is not None:
            sl = getattr(pda, "swing_low", 0.0)
            sig.stop       = sl * 0.995 if sl > 0 else spot * 0.95
            sig.stop_label = f"bajo swing_low ${sig.stop:,.0f}"
        elif pda is not None:
            sh = getattr(pda, "swing_high", 0.0)
            sig.stop       = sh * 1.005 if sh > 0 else spot * 1.05
            sig.stop_label = f"sobre swing_high ${sig.stop:,.0f}"
