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
    """
    level: LiquidityLevel
    direction: str    # "up" (barrió BSL) | "down" (barrió SSL)
    reversed: bool    # True si cerró de vuelta (confirmación de stop hunt)
    bar_index: int    # posición en el df donde ocurrió


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
    clusters: list[float] = []
    for p in prices_list:
        fp = float(p)
        if any(abs(fp - c) / max(abs(c), 1e-9) <= _EQL_TOLERANCE for c in clusters):
            continue
        count = sum(
            1 for q in prices_list
            if abs(fp - float(q)) / max(abs(float(q)), 1e-9) <= _EQL_TOLERANCE
        )
        if count >= 2:
            clusters.append(fp)
    return clusters


def _in_cluster(price: float, clusters: list[float]) -> bool:
    return any(abs(price - c) / max(abs(c), 1e-9) <= _EQL_TOLERANCE for c in clusters)


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

    window = daily.tail(_PDA_LOOKBACK)
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
) -> list[LiquiditySweep]:
    """Detecta stop hunts (Turtle Soup) en las últimas `lookback` velas.

    Un sweep es: high > nivel_BSL pero close < nivel_BSL → posible reversal bajista.
    O:           low < nivel_SSL pero close > nivel_SSL → posible reversal alcista.
    """
    sweeps: list[LiquiditySweep] = []
    if df is None or len(df) < 2:
        return sweeps

    check = df.tail(lookback + 1).reset_index(drop=True)
    offset = max(0, len(df) - lookback - 1)

    for lv in levels:
        p = lv.price
        for i, bar in check.iterrows():
            bar_hi = float(bar["high"])
            bar_lo = float(bar["low"])
            bar_cl = float(bar["close"])

            if lv.kind == "BSL" and bar_hi > p * 1.001:
                sweeps.append(LiquiditySweep(
                    level=lv,
                    direction="up",
                    reversed=bar_cl < p,
                    bar_index=offset + int(i),
                ))
            elif lv.kind == "SSL" and bar_lo < p * 0.999:
                sweeps.append(LiquiditySweep(
                    level=lv,
                    direction="down",
                    reversed=bar_cl > p,
                    bar_index=offset + int(i),
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

    return ctx
